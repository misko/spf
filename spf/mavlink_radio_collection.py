import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime

import yaml
from pymavlink import mavutil

from spf.data_collector import DroneDataCollector
from spf.distance_finder.distance_finder_controller import DistanceFinderController
from spf.gps.boundaries import franklin_safe  # crissy_boundary_convex
from spf.grbl.grbl_interactive import BouncePlanner, Dynamics
from spf.mavlink.mavlink_controller import Drone, get_adrupilot_serial
from spf.utils import is_pi

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
        "-m",
        "--device-mapping",
        type=str,
        help="Device mapping file",
        default=None,
        required=True,
    )
    args = parser.parse_args()

    run_started_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # read YAML
    with open(args.yaml_config, "r") as stream:
        yaml_config = yaml.safe_load(stream)

    # open device mapping and figure out URIs
    with open(args.device_mapping, "r") as device_mapping:
        port_to_uri = {
            int(mapping[0]): f"usb:1.{mapping[1]}.5"
            for mapping in [line.strip().split() for line in device_mapping]
        }

    for receiver in yaml_config["receivers"]:
        if "receiver-port" in receiver:
            receiver["receiver-uri"] = port_to_uri[receiver["receiver-port"]]

    # add in our current config
    if args.routine is not None:
        yaml_config["routine"] = args.routine

    if args.tx_gain is not None:
        assert yaml_config["emitter"]["type"] == "sdr"
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    output_files_prefix = f"rover_{run_started_at}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}"
    if args.tag != "":
        output_files_prefix += f"_tag_{args.tag}"

    # setup filename
    tmpdir = tempfile.TemporaryDirectory()
    temp_dir_name = tmpdir.name
    filename_log = f"{temp_dir_name}/{output_files_prefix}.log.tmp"
    filename_yaml = f"{temp_dir_name}/{output_files_prefix}.yaml.tmp"
    filename_npy = f"{temp_dir_name}/{output_files_prefix}.npy.tmp"
    temp_filenames = [filename_log, filename_yaml, filename_npy]
    final_filenames = [x.replace(".tmp", "") for x in temp_filenames]

    logger = logging.getLogger(__name__)

    # setup logging
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(filename_log),
    ]
    logging.basicConfig(
        handlers=handlers,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, args.logging_level.upper(), None),
    )

    # make a copy of the YAML
    with open(filename_yaml, "w") as outfile:
        yaml.dump(yaml_config, outfile, default_flow_style=False)

    logging.info(json.dumps(yaml_config, sort_keys=True, indent=4))

    boundary = franklin_safe

    logging.info("Connecting to drone...")

    if yaml_config["drone-uri"] == "serial":
        serial = get_adrupilot_serial()
        if serial is None:
            print("Failed to get serial")
            sys.exit(1)
        yaml_config["drone-uri"] = serial
        connection = mavutil.mavlink_connection(serial, baud=115200)
    else:
        connection = mavutil.mavlink_connection(yaml_config["drone-uri"])

    distance_finder = None
    if is_pi():
        distance_finder = DistanceFinderController(
            trigger=yaml_config["distance-finder"]["trigger"],
            echo=yaml_config["distance-finder"]["echo"],
        )
    drone = Drone(
        connection,
        planner=BouncePlanner(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000000001,
            ),
            start_point=boundary.mean(axis=0),
            epsilon=0.0000001,
            step_size=0.1,
        ),
        boundary=boundary,
        distance_finder=distance_finder,
    )
    drone.start()

    logging.info("Starting data collector...")
    data_collector = DroneDataCollector(
        filename_npy=filename_npy, yaml_config=yaml_config, position_controller=drone
    )
    data_collector.radios_to_online()  # blocking

    while not drone.has_planner_started_moving():
        logging.info(f"waiting for drone to start moving {time.time()}")
        time.sleep(5)  # easy poll this

    data_collector.start()
    logging.info("DRONE IS READY!!! LETS GOOO!!!")

    while data_collector.is_collecting():
        time.sleep(5)

    # we finished lets move files out to final positions
    for idx in range(len(temp_filenames)):
        os.rename(temp_filenames[idx], final_filenames[idx])
