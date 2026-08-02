import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from spf.capture_schema import (
    normalize_capture_config,
    validate_transport_schema,
)
from spf.capture_status import CaptureStatusWriter
from spf.capture_failure import terminate_capture_process
from spf.gps.boundaries import boundaries  # crissy_boundary_convex
from spf.gps.boundaries import find_closest_boundary
from spf.mavlink.mavlink_controller import (
    DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
    DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
    Drone,
    connect_with_heartbeat,
    drone_get_planner,
    mavlink_connection_factory,
    resolve_ardupilot_serial,
    tones,
)
from spf.utils import (
    DataVersionNotImplemented,
    filenames_from_time_in_seconds,
    is_pi,
    load_config,
)


READINESS_TONE_INTERVAL_SECONDS = 15.0
ANNOYING_TONES_DISABLE_PATH = Path.home() / "disable_annoying_tones"


def maybe_play_readiness_wait_tone(
    drone,
    *,
    now: float,
    next_tone_at: float,
    disable_path: Path = ANNOYING_TONES_DISABLE_PATH,
) -> float:
    """Play one low-duty readiness chirp unless the operator disabled it."""
    if now < next_tone_at:
        return next_tone_at
    while next_tone_at <= now:
        next_tone_at += READINESS_TONE_INTERVAL_SECONDS
    if not disable_path.exists():
        drone.buzzer(tones["readiness-wait"])
    return next_tone_at


def yaml_defaults(yaml_config, device_mapping_fn):
    # open device mapping and figure out URIs
    with open(device_mapping_fn, "r") as device_mapping:
        port_to_uri = {}
        for line in device_mapping:
            mapping = line.strip().split()
            if len(mapping) == 2:
                port_to_uri[int(mapping[0])] = f"pluto://usb:1.{mapping[1]}.5"
            elif len(mapping) == 3:
                port_to_uri[
                    int(mapping[0])
                ] = f"pluto://usb:{mapping[1]}.{mapping[2]}.5"
            else:
                raise ValueError("port mapping invalid")

    for receiver in yaml_config["receivers"] + [yaml_config["emitter"]]:
        if "receiver-port" in receiver:
            receiver["receiver-uri"] = port_to_uri[receiver["receiver-port"]]
    if "emitter-port" in yaml_config["emitter"]:
        yaml_config["emitter"]["emitter-uri"] = port_to_uri[
            yaml_config["emitter"]["emitter-port"]
        ]

    if args.records_per_receiver is not None:
        yaml_config["n-records-per-receiver"] = args.records_per_receiver

    # add in our current config
    if args.routine is not None:
        yaml_config["routine"] = args.routine

    if args.tx_gain is not None:
        assert yaml_config["emitter"]["type"] == "sdr"
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    if args.drone_uri is not None:
        yaml_config["drone-uri"] = args.drone_uri

    if args.inference:
        yaml_config["inference"] = True
    return yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--yaml-config",
        type=str,
        help="YAML config file",
        required=True,
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="tag files", required=False, default=""
    )
    parser.add_argument(
        "--tx-gain", type=int, help="tag files", required=False, default=None
    )
    parser.add_argument(
        "-l",
        "--logging-level",
        type=str,
        help="Logging level",
        default="INFO",
        required=False,
    )
    parser.add_argument(
        "-r", "--routine", type=str, help="GRBL routine", required=False, default=None
    )
    parser.add_argument(
        "--temp", type=str, help="temp dirname", required=False, default="./temp"
    )
    parser.add_argument(
        "--status-file",
        type=str,
        help="atomically updated durable capture status JSON",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--run-for-seconds",
        type=int,
        help="run for this long and exit",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-m",
        "--device-mapping",
        type=str,
        help="Device mapping file",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--records-per-receiver",
        type=int,
        help="how many records to get per receiver",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--drone-uri",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--nthetas",
        type=int,
        help="nthetas",
        default=None,
    )
    parser.add_argument(
        "--ultrasonic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ignore-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--inference", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--write-to-disk", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--fake-drone", action=argparse.BooleanOptionalAction)
    parser.add_argument("--exit", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_started_at = datetime.now().timestamp()  #

    # read YAML
    yaml_config = yaml_defaults(
        normalize_capture_config(load_config(args.yaml_config)),
        args.device_mapping,
    )
    validate_transport_schema(yaml_config)

    temp_filenames, final_filenames = filenames_from_time_in_seconds(
        run_started_at,
        args.temp,
        yaml_config,
        data_version=yaml_config["data-version"],
        craft="rover",
        tag=args.tag,
    )
    status_writer = None
    if args.status_file is not None:
        status_writer = CaptureStatusWriter(
            args.status_file,
            capture_name=Path(temp_filenames["data"]).name,
            expected_records_per_receiver=yaml_config["n-records-per-receiver"],
            receiver_count=len(yaml_config["receivers"]),
        )
        status_writer.publish(
            "starting",
            [0] * len(yaml_config["receivers"]),
            force=True,
        )

    logger = logging.getLogger(__name__)

    # setup logging
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(temp_filenames["log"]),
    ]
    logging.basicConfig(
        handlers=handlers,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, args.logging_level.upper(), None),
    )

    # make a copy of the YAML
    with open(temp_filenames["yaml"], "w") as outfile:
        yaml.dump(yaml_config, outfile, default_flow_style=False)

    distance_finder = None
    # A fake-drone run must be hardware-independent.  In particular, do not
    # initialize RPi.GPIO merely because the tests happen to run on a Pi.
    if is_pi() and args.ultrasonic and not args.fake_drone:
        from spf.distance_finder.distance_finder_controller import (
            DistanceFinderController,
        )

        distance_finder = DistanceFinderController(
            trigger=yaml_config["distance-finder"]["trigger"],
            echo=yaml_config["distance-finder"]["echo"],
        )

    logging.info("MavRadioCollection: Starting data collector...")
    if not args.fake_drone:
        if yaml_config["drone-uri"] == "serial":
            endpoint = resolve_ardupilot_serial()
            yaml_config["drone-uri"] = endpoint
        else:
            endpoint = yaml_config["drone-uri"]
        connection_factory = mavlink_connection_factory(endpoint)
        connection, initial_heartbeat = connect_with_heartbeat(
            connection_factory,
            attempts=DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
            heartbeat_timeout=DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
            retry_backoff=DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
        )
        drone = Drone(
            connection,
            distance_finder=distance_finder,
            ignore_mode=args.ignore_mode,
            connection_factory=connection_factory,
            reconnect_attempts=DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
            reconnect_backoff=DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
            reconnect_heartbeat_timeout=DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
        )
        drone.process_message(initial_heartbeat)
        drone.start()
    else:
        drone = Drone(
            None,
            distance_finder=distance_finder,
            fake=True,
            ignore_mode=args.ignore_mode,
        )

    next_readiness_tone_at = time.monotonic() + READINESS_TONE_INTERVAL_SECONDS
    while not args.fake_drone and not drone.drone_ready:
        drone.raise_if_connection_failed()
        logging.info(
            f"Drone startup wait for drone ready: gps:{str(drone.gps)} , ekf:{str(drone.ekf_healthy)}"
        )
        next_readiness_tone_at = maybe_play_readiness_wait_tone(
            drone,
            now=time.monotonic(),
            next_tone_at=next_readiness_tone_at,
        )
        time.sleep(2)

    # The collector imports NumPy/Torch, Zarr, and the SDR stack. Keep them off
    # the preflight critical path so MAVLink status and readiness monitoring
    # begin promptly after boot. Collector construction remains in the same
    # place below, after navigation readiness and planner setup.
    from spf.data_collector import (
        DroneDataCollectorRaw,
        DroneDataCollectorRawV6,
        DroneDataCollectorRawV7,
        capture_signal_handlers,
    )

    boundary_name = yaml_config.get("boundary", "franklin_safe")
    if boundary_name == "auto":
        # find out which one is closest
        boundary_name = find_closest_boundary(drone.gps)
        print(f"Closest boundary is {boundary_name}")
    elif boundary_name not in boundaries:
        logging.error(f"Failed to find boundary {boundary_name} in valid boundaries")
        sys.exit(1)
    # Per-rover resting offset (east_m, north_m) away from the boundary centroid,
    # so co-located rovers do not converge on the same point. Absent -> centroid.
    drone.set_and_start_planner(
        drone_get_planner(
            yaml_config["routine"],
            boundary=boundaries[boundary_name],
            rest_offset_m=yaml_config.get("rest-offset-m"),
        )
    )

    if args.inference:
        pass

    if args.checkpoint:
        # load model config and use that theta
        from spf.scripts.train_utils import load_config_from_fn

        config = load_config_from_fn(args.checkpoint_config)
        assert args.nthetas is None, "nthetas cannot be set when loading checkpoint"
        args.nthetas = config["global"]["nthetas"]
    elif args.nthetas is None:
        logging.warning("Setting nthetas to 65 as default")
        args.nthetas = 65

    if args.realtime:
        from spf.dataset.spf_dataset import training_only_keys, v5inferencedataset
        from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper

        v5inf = v5inferencedataset(
            yaml_fn=temp_filenames["yaml"],
            nthetas=args.nthetas,
            gpu=False,
            paired=True,
            model_config_fn="",
            skip_fields=["signal_matrix"] + training_only_keys,
            vehicle_type="rover",
            skip_segmentation=True,
            skip_detrend=False,
            max_store_size=3,  # needs to process fast enough otherwise delayed
        )
        nn_ds = v5spfdataset_nn_wrapper(
            v5inf,
            args.checkpoint_config,
            args.checkpoint,
            inference_cache=None,
            device="cpu",
            v4=False,
            absolute=True,
        )

    if yaml_config["data-version"] == 4:
        data_collector = DroneDataCollectorRaw(
            realtime_v5inf=v5inf if args.realtime else None,
            data_filename=temp_filenames["data"] if args.write_to_disk else None,
            yaml_config=yaml_config,
            position_controller=drone,
            status_writer=status_writer,
        )
    elif yaml_config["data-version"] == 6:
        data_collector = DroneDataCollectorRawV6(
            realtime_v5inf=v5inf if args.realtime else None,
            data_filename=temp_filenames["data"] if args.write_to_disk else None,
            yaml_config=yaml_config,
            position_controller=drone,
            status_writer=status_writer,
        )
    elif yaml_config["data-version"] == 7:
        data_collector = DroneDataCollectorRawV7(
            realtime_v5inf=v5inf if args.realtime else None,
            data_filename=temp_filenames["data"] if args.write_to_disk else None,
            yaml_config=yaml_config,
            position_controller=drone,
            status_writer=status_writer,
        )
    else:
        raise DataVersionNotImplemented

    logging.info("MavRadioCollection: Radios online...")
    data_collector.radios_to_online()  # blocking

    def check_exit():
        if not args.fake_drone:
            drone.raise_if_connection_failed()
        if args.run_for_seconds > 0 and time.time() - start_time > args.run_for_seconds:
            sys.exit(0)

    start_time = time.time()
    while not args.fake_drone and not drone.is_planner_in_control():
        logging.info(
            f"MavRadioCollection: Waiting for drone to start moving {time.time()}"
        )
        check_exit()
        time.sleep(5)  # easy poll this

    logging.info("MavRadioCollection: Planner has started controling the drone...")

    system_time = datetime.fromtimestamp(datetime.now().timestamp()).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    gps_time = datetime.fromtimestamp(drone.gps_time).strftime("%Y_%m_%d_%H_%M_%S")

    logging.info(
        f"MavRadioCollection: Current system time: {system_time} current gps time {gps_time}"
    )

    capture_failure = None
    try:
        with capture_signal_handlers(data_collector):
            data_collector.start()
            try:
                while data_collector.is_collecting():
                    # if args.realtime:
                    #     for x in nn_ds: # x[1]['paired'].shape  / torch.Size([1, 65])
                    #         pass
                    check_exit()
                    time.sleep(0.5)
            except BaseException as error:
                # Connection-health failures and explicit run limits occur in
                # the main thread. Route them through the same writer/radio
                # cleanup path as receiver-thread failures.
                data_collector.request_stop(error)
            data_collector.done()
    except BaseException as error:
        # DataCollector has stopped RX, muted TX, persisted the exact incomplete
        # state and closed LMDB before done() re-raises the primary incident.
        capture_failure = error
    finally:
        if capture_failure is None:
            drone.planner_should_move = False
        else:
            # This also interrupts an already-issued waypoint and requests
            # HOLD. If MAVLink is temporarily unavailable, Drone retries HOLD
            # after the next fresh heartbeat.
            drone.request_motion_stop(
                f"capture incident {data_collector.capture_incident_id or 'unassigned'}"
            )
    if capture_failure is not None:
        drone.wait_for_abort_hold(timeout_seconds=2.0)
        terminate_capture_process(
            capture_failure,
            incident_id=data_collector.capture_incident_id,
            error_source=data_collector.capture_error_source,
        )

    if args.realtime:
        v5inf.close()  # make sure to close this outside of context manager!

    # we finished lets move files out to final positions

    logging.info("MavRadioCollection: Moving files to final location ...")
    for k in temp_filenames:
        os.rename(temp_filenames[k], final_filenames[k])
    data_collector.publish_operator_status(
        "complete",
        artifact=final_filenames["data"],
        force=True,
    )

    # wait for it to release control back, that happens when this goes false
    seconds_to_wait = 60
    while seconds_to_wait > 0 and drone.is_planner_in_control():
        time.sleep(2)
        seconds_to_wait -= 2

    # Post-capture navigation is operationally important, but the Zarr above
    # is already finalized, renamed and durably reported complete. Keep those
    # two outcomes separate if MAVLink fails while parking the rover.
    try:
        if not args.fake_drone and not drone.move_to_home():
            raise RuntimeError("return-home operation did not reach home")
    except BaseException as error:
        incident_id = uuid.uuid4().hex
        if status_writer is not None:
            status_writer.publish(
                "complete",
                data_collector.records_written_by_receiver,
                error=error,
                incident_id=incident_id,
                error_source="post_capture_navigation",
                artifact=final_filenames["data"],
                force=True,
            )
        drone.request_motion_stop(f"post-capture navigation incident {incident_id}")
        drone.wait_for_abort_hold(timeout_seconds=2.0)
        terminate_capture_process(
            error,
            incident_id=incident_id,
            error_source="post_capture_navigation",
        )

    if is_pi() and not args.fake_drone:
        time.sleep(5)
        # subprocess.getoutput("sudo halt")
        subprocess.getoutput("sudo sync")
