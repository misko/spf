import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml
from pymavlink import mavutil

from spf.data_collector import DroneDataCollector, DroneDataCollectorRaw
from spf.distance_finder.distance_finder_controller import DistanceFinderController
from spf.gps.boundaries import franklin_safe  # crissy_boundary_convex
from spf.mavlink.mavlink_controller import (
    Drone,
    drone_get_planner,
    get_ardupilot_serial,
)
from spf.utils import DataVersionNotImplemented, filenames_from_time_in_seconds, is_pi

if __name__ == "__main__":
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
        "--ultrasonic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--fake-drone", action=argparse.BooleanOptionalAction)
    parser.add_argument("--exit", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    run_started_at = datetime.now().timestamp()  #
    # read YAML
    with open(args.yaml_config, "r") as stream:
        yaml_config = yaml.safe_load(stream)

    # open device mapping and figure out URIs
    with open(args.device_mapping, "r") as device_mapping:
        port_to_uri = {
            int(mapping[0]): f"usb:1.{mapping[1]}.5"
            for mapping in [line.strip().split() for line in device_mapping]
        }

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
    # setup filename
    # tmpdir = tempfile.TemporaryDirectory()
    # temp_dir_name = tmpdir.name

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

    boundary = franklin_safe
    planner = drone_get_planner(yaml_config["routine"], boundary=boundary)

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
            connection,
            planner=planner,
            distance_finder=distance_finder,
        )

        drone.start()
    else:
        drone = Drone(
            None,
            planner=planner,
            distance_finder=distance_finder,
            fake=True,
        )

    if yaml_config["data-version"] == 3:
        data_collector = DroneDataCollector(
            data_filename=temp_filenames["data"],
            yaml_config=yaml_config,
            position_controller=drone,
        )
    elif yaml_config["data-version"] == 4:
        data_collector = DroneDataCollectorRaw(
            data_filename=temp_filenames["data"],
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
    while not args.fake_drone and not drone.has_planner_started_moving():
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

    # we finished lets move files out to final positions

    logging.info("MavRadioCollection: Moving files to final location ...")
    for k in temp_filenames:
        os.rename(temp_filenames[k], final_filenames[k])

    if is_pi() and not args.fake_drone:
        time.sleep(5)
        subprocess.getoutput("sudo halt")
