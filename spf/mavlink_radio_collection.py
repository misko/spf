import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml
from pymavlink import mavutil

from spf.data_collector import DroneDataCollectorRaw
from spf.dataset.spf_dataset import v5inferencedataset,training_only_keys
from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper
from spf.distance_finder.distance_finder_controller import DistanceFinderController
from spf.gps.boundaries import boundaries  # crissy_boundary_convex
from spf.gps.boundaries import find_closest_boundary
from spf.mavlink.mavlink_controller import (
    Drone,
    drone_get_planner,
    get_ardupilot_serial,
)
from spf.scripts.train_utils import load_config_from_fn
from spf.utils import (
    DataVersionNotImplemented,
    filenames_from_time_in_seconds,
    is_pi,
    load_config,
)


def yaml_defaults(yaml_config, device_mapping_fn):
    # open device mapping and figure out URIs
    with open(device_mapping_fn, "r") as device_mapping:
        port_to_uri = {}
        for line in device_mapping:
            mapping = line.strip().split()
            if len(mapping) == 2:
                port_to_uri[int(mapping[0])] = f"pluto://usb:1.{mapping[1]}.5"
            elif len(mapping) == 3:
                port_to_uri[int(mapping[0])] = (
                    f"pluto://usb:{mapping[1]}.{mapping[2]}.5"
                )
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
    yaml_config = yaml_defaults(load_config(args.yaml_config), args.device_mapping)

    temp_filenames, final_filenames = filenames_from_time_in_seconds(
        run_started_at,
        args.temp,
        yaml_config,
        data_version=yaml_config["data-version"],
        craft="rover",
        tag=args.tag,
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
    if is_pi() and args.ultrasonic:
        distance_finder = DistanceFinderController(
            trigger=yaml_config["distance-finder"]["trigger"],
            echo=yaml_config["distance-finder"]["echo"],
        )

    logging.info("MavRadioCollection: Starting data collector...")
    if not args.fake_drone:
        if yaml_config["drone-uri"] == "serial":
            serial = get_ardupilot_serial()
            if serial is None:
                print("Failed to get serial")
                sys.exit(1)
            yaml_config["drone-uri"] = serial
            connection = mavutil.mavlink_connection(serial, baud=115200)
        else:
            connection = mavutil.mavlink_connection(yaml_config["drone-uri"])
        drone = Drone(
            connection, distance_finder=distance_finder, ignore_mode=args.ignore_mode
        )
        drone.start()
    else:
        drone = Drone(
            None,
            distance_finder=distance_finder,
            fake=True,
            ignore_mode=args.ignore_mode,
        )

    while not args.fake_drone and not drone.drone_ready:
        logging.info(
            f"Drone startup wait for drone ready: gps:{str(drone.gps)} , ekf:{str(drone.ekf_healthy)}"
        )
        time.sleep(10)

    boundary_name = yaml_config.get("boundary", "franklin_safe")
    if boundary_name == "auto":
        # find out which one is closest
        boundary_name = find_closest_boundary(drone.gps)
        print(f"Closest boundary is {boundary_name}")
    elif boundary_name not in boundaries:
        logging.error(f"Failed to find boundary {boundary_name} in valid boundaries")
        sys.exit(1)
    drone.set_and_start_planner(
        drone_get_planner(yaml_config["routine"], boundary=boundaries[boundary_name])
    )

    if args.inference:
        pass


    if args.checkpoint:
        # load model config and use that theta
        config = load_config_from_fn(args.checkpoint_config)
        assert args.nthetas is None, "nthetas cannot be set when loading checkpoint"
        args.nthetas = config['global']['nthetas']
    elif args.nthetas is None:
        logging.warning("Setting nthetas to 65 as default")
        args.nthetas=65

    if args.realtime:
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
            max_store_size=2
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
        )
    else:
        raise DataVersionNotImplemented

    logging.info("MavRadioCollection: Radios online...")
    data_collector.radios_to_online()  # blocking

    def check_exit():
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

    data_collector.start()
    while data_collector.is_collecting():
        check_exit()
        time.sleep(5)

    data_collector.done()

    drone.planner_should_move = False
    # we finished lets move files out to final positions

    logging.info("MavRadioCollection: Moving files to final location ...")
    for k in temp_filenames:
        os.rename(temp_filenames[k], final_filenames[k])

    # wait for it to release control back, that happens when this goes false
    seconds_to_wait = 60
    while seconds_to_wait > 0 and drone.is_planner_in_control():
        time.sleep(2)
        seconds_to_wait -= 2

    drone.move_to_home()

    if is_pi() and not args.fake_drone:
        time.sleep(5)
        # subprocess.getoutput("sudo halt")
        subprocess.getoutput("sudo sync")
