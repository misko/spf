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
from spf.capture_log import configure_capture_logging
from spf.capture_status import CaptureStatusWriter
from spf.capture_failure import terminate_capture_process
from spf.gps.boundaries import boundaries  # crissy_boundary_convex
from spf.gps.boundaries import find_closest_boundary
from spf.mavlink.mavlink_controller import (
    DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
    DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
    STALL_DETECT_SECONDS,
    STALL_MANUAL_SECONDS,
    STALL_PROGRESS_RADIUS_M,
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


# Losing planner control does NOT raise. A takeover is a designed interlock --
# the stall handover depends on it -- and CAPTURE_RESTART_ATTEMPTS defaults to
# 1, so raising would end a session after two of them. The capture keeps
# running and declares how much of itself was recorded degraded.
LOST = "lost"
RECOVERED = "recovered"


def planner_control_lost(drone) -> bool:
    """True when the vehicle is no longer under planner control.

    `is_planner_in_control()` used to be read exactly once, to decide when to
    START recording, and never again. An operator taking MANUAL is a designed
    interlock -- the stall handover relies on it -- but the rover then sits
    still while the collector keeps writing snapshots. On 2026-08-07 rover 4
    advanced 320 -> 578 of 3000 records with the mode column reading MANUAL,
    and gps_lat/long are exactly the fields rx_pos/tx_pos ground truth is
    derived from, so those records describe a vehicle that was not moving.

    `drone is None` is the --fake-drone bench path: there is no vehicle to lose
    control of, and those captures must still run.
    """
    if drone is None:
        return False
    return not drone.is_planner_in_control()


def navigation_unhealthy(drone) -> str:
    """Why the vehicle's position/heading is untrustworthy right now, or "".

    Separate from planner control on purpose. A takeover means nobody was
    driving; a GPS or compass failure means we WERE driving and did not know
    where we were. Both make records unusable, for different reasons and with
    different fixes, so they are counted separately.
    """
    if drone is None:
        return ""
    return drone.navigation_health_warning() or ""


class LostIntervalTracker:
    """How much of a capture was recorded in a degraded state, and how often.

    Used twice: once for "the planner was not driving" (an operator in MANUAL,
    a stall handover, a dropped link) and once for "the vehicle did not know
    where it was" (GPS or compass unhealthy, EKF unhappy).

    Aborting the capture instead would be the wrong trade: a takeover is a
    designed interlock (the stall handover depends on it), and
    CAPTURE_RESTART_ATTEMPTS defaults to 1, so two takeovers would end a
    session. So the capture keeps running and declares how much of itself was
    recorded degraded, which is a thing analysis can filter on. Without this
    the only trace is a flat bearing track someone puzzles over months later.
    """

    def __init__(self):
        self.lost_seconds = 0.0
        self.episodes = 0
        self._lost_since = None

    def update(self, lost: bool, now: float) -> str:
        """Feed one observation. Returns the edge crossed: "", LOST or RECOVERED.

        RECOVERED is reported, not just recorded, because a degraded capture
        that never comes back and one that recovers in four seconds want very
        different responses from whoever is standing in the field.
        """
        if lost and self._lost_since is None:
            self._lost_since = now
            self.episodes += 1
            return LOST
        if not lost and self._lost_since is not None:
            self.lost_seconds += now - self._lost_since
            self._lost_since = None
            return RECOVERED
        return ""

    def finish(self, now: float) -> None:
        """Close an interval still open when the capture ended.

        Rover 4 on 2026-08-07 was still in MANUAL as the capture kept
        advancing, so the final takeover has no closing edge to observe.
        """
        if self._lost_since is not None:
            self.lost_seconds += now - self._lost_since
            self._lost_since = None


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

    # Stall handling. Detection defaults on everywhere -- its worst outcome is
    # handing a working rover to a human. Recovery (autonomous reversing) is
    # opt-in per rover; drone_run.sh resolves the fleet default.
    parser.add_argument(
        "--crash-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--crash-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--stall-detect-seconds",
        type=float,
        default=STALL_DETECT_SECONDS,
        help="no progress for this long counts as a stall",
    )
    parser.add_argument(
        "--stall-manual-seconds",
        type=float,
        default=STALL_MANUAL_SECONDS,
        help="with --crash-recovery, give up and hand to MANUAL after this long",
    )
    parser.add_argument(
        "--stall-progress-radius-m",
        type=float,
        default=STALL_PROGRESS_RADIUS_M,
        help=(
            "displacement that counts as progress; scale it WITH "
            "--stall-detect-seconds so the implied speed floor stays "
            "what production uses (3m/10s = 0.3 m/s)"
        ),
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

    # Setup logging into the capture's own .log sidecar. This MUST go through
    # configure_capture_logging: importing spf.mavlink.mavlink_controller above
    # already called logging.basicConfig(filename="logs.log"), so a plain
    # basicConfig here is a no-op and leaves the sidecar zero bytes while the
    # run's log lands in a relative logs.log. configure_capture_logging forces
    # the handlers, makes the path absolute, and writes the provenance header
    # (argv, config path and its source, rover id, tag, host, git commit, and
    # both local and UTC timestamps) ahead of the first log line.
    configure_capture_logging(
        temp_filenames["log"],
        level=args.logging_level,
        argv=sys.argv,
        config_path=args.yaml_config,
        tag=args.tag,
        run_started_at=run_started_at,
        data_filename=final_filenames["data"],
        # The header is written to the .log.tmp; name the file it becomes so a
        # reader of the finalized sidecar sees its own path.
        extra={
            "log_sidecar_final": str(Path(final_filenames["log"]).absolute()),
        },
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
            crash_detect=args.crash_detect,
            crash_recovery=args.crash_recovery,
            stall_detect_seconds=args.stall_detect_seconds,
            stall_manual_seconds=args.stall_manual_seconds,
            stall_progress_radius_m=args.stall_progress_radius_m,
        )
        drone.process_message(initial_heartbeat)
        drone.start()
    else:
        drone = Drone(
            None,
            distance_finder=distance_finder,
            fake=True,
            ignore_mode=args.ignore_mode,
            crash_detect=args.crash_detect,
            crash_recovery=args.crash_recovery,
            stall_detect_seconds=args.stall_detect_seconds,
            stall_manual_seconds=args.stall_manual_seconds,
            stall_progress_radius_m=args.stall_progress_radius_m,
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
    planner_control_tracker = LostIntervalTracker()
    navigation_tracker = LostIntervalTracker()
    data_collector.planner_control_lost_seconds = 0.0
    data_collector.navigation_unhealthy_seconds = 0.0
    try:
        with capture_signal_handlers(data_collector):
            data_collector.start()
            try:
                while data_collector.is_collecting():
                    # if args.realtime:
                    #     for x in nn_ds: # x[1]['paired'].shape  / torch.Size([1, 65])
                    #         pass
                    check_exit()
                    # is_planner_in_control() used to be read once, to decide
                    # when to START, and never again -- so a capture ran on
                    # through an operator takeover writing snapshots of a
                    # stationary rover. Watch it for the whole capture instead.
                    now = time.time()
                    vehicle = None if args.fake_drone else drone
                    edge = planner_control_tracker.update(
                        planner_control_lost(vehicle), now
                    )
                    if edge == LOST:
                        logging.error(
                            "PLANNER CONTROL LOST: %s, but this capture is still "
                            "recording. These records describe a STATIONARY rover; "
                            "filter them on planner_control_lost_seconds.",
                            vehicle.planner_control_loss_reason(),
                        )
                    elif edge == RECOVERED:
                        logging.warning(
                            "PLANNER CONTROL RESUMED after %.0fs; the capture "
                            "continues.",
                            planner_control_tracker.lost_seconds,
                        )
                    data_collector.planner_control_lost_seconds = (
                        planner_control_tracker.lost_seconds
                    )
                    # Separately: were we driving but lost track of where we
                    # were? gps_lat/gps_long/heading are the ground truth, so a
                    # record written with an unhealthy GPS or compass is wrong
                    # rather than merely uninformative.
                    warning = navigation_unhealthy(vehicle)
                    edge = navigation_tracker.update(bool(warning), now)
                    if edge == LOST:
                        logging.error(
                            "NAVIGATION UNHEALTHY: %s, but this capture is still "
                            "recording. The gps/heading ground truth in these "
                            "records may be WRONG; filter them on "
                            "navigation_unhealthy_seconds.",
                            warning,
                        )
                    elif edge == RECOVERED:
                        logging.warning(
                            "NAVIGATION RECOVERED after %.0fs; the capture "
                            "continues.",
                            navigation_tracker.lost_seconds,
                        )
                    data_collector.navigation_unhealthy_seconds = (
                        navigation_tracker.lost_seconds
                    )
                    time.sleep(0.5)
            except BaseException as error:
                # Connection-health failures and explicit run limits occur in
                # the main thread. Route them through the same writer/radio
                # cleanup path as receiver-thread failures.
                data_collector.request_stop(error)
            # Close a takeover still open at the end: rover 4 was still in
            # MANUAL as its capture kept advancing, so the final interval has no
            # closing edge to observe.
            closed_at = time.time()
            planner_control_tracker.finish(now=closed_at)
            navigation_tracker.finish(now=closed_at)
            data_collector.planner_control_lost_seconds = (
                planner_control_tracker.lost_seconds
            )
            data_collector.navigation_unhealthy_seconds = (
                navigation_tracker.lost_seconds
            )
            if planner_control_tracker.episodes:
                logging.error(
                    "Capture recorded %.0fs across %d takeover(s) with no planner "
                    "control; those records describe a stationary rover.",
                    planner_control_tracker.lost_seconds,
                    planner_control_tracker.episodes,
                )
            if navigation_tracker.episodes:
                logging.error(
                    "Capture recorded %.0fs across %d episode(s) of unhealthy "
                    "navigation; the gps/heading ground truth in those records "
                    "may be wrong.",
                    navigation_tracker.lost_seconds,
                    navigation_tracker.episodes,
                )
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

    if not args.fake_drone:
        # wait for it to release control back, that happens when this goes false
        #
        # The latch, not is_planner_in_control(): this waits for the planner
        # THREAD to leave its loop before the parking below starts commanding
        # the vehicle. The live signal goes false the moment an operator takes
        # MANUAL, which would end this wait while run_planner was still issuing
        # repositions -- and then park and planner would fight over the rover.
        seconds_to_wait = 60
        while seconds_to_wait > 0 and drone.planner_is_still_driving():
            time.sleep(2)
            seconds_to_wait -= 2

        # Post-capture navigation is operationally important, but the Zarr above
        # is already finalized, renamed and durably reported complete. Keep those
        # two outcomes separate if MAVLink fails while parking the rover. A
        # fake-drone capture has no MAVLink vehicle or home and deliberately
        # stops after validating the capture artifact.
        try:
            if not drone.move_to_home():
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
