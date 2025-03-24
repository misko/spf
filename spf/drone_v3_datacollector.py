import argparse
import json
import logging
import os
import time
from datetime import datetime

import yaml

from spf.data_collector import FakeDroneDataCollector
from spf.gps.boundaries import franklin_safe

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
            int(mapping[0]): f"pluto://usb:1.{mapping[1]}.5"
            for mapping in [line.strip().split() for line in device_mapping]
        }

    for receiver in yaml_config["receivers"] + [yaml_config["emitter"]]:
        if "receiver-port" in receiver:
            receiver["receiver-uri"] = port_to_uri[receiver["receiver-port"]]
    if "emitter-port" in yaml_config["emitter"]:
        yaml_config["emitter"]["emitter-uri"] = port_to_uri[
            yaml_config["emitter"]["emitter-port"]
        ]

    if args.tx_gain is not None:
        assert yaml_config["emitter"]["type"] == "sdr"
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    output_files_prefix = f"rover_{run_started_at}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}"
    if args.tag != "":
        output_files_prefix += f"_tag_{args.tag}"

    # setup filename
    # tmpdir = tempfile.TemporaryDirectory()
    # temp_dir_name = tmpdir.name
    temp_dir_name = "./"
    filename_log = f"{temp_dir_name}/{output_files_prefix}.log.tmp"
    filename_yaml = f"{temp_dir_name}/{output_files_prefix}.yaml.tmp"
    filename_data = f"{temp_dir_name}/{output_files_prefix}.npy.tmp"
    temp_filenames = [filename_log, filename_yaml, filename_data]
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

    logging.info("Starting data collector...")
    data_collector = FakeDroneDataCollector(
        filename_npy=filename_data, yaml_config=yaml_config, position_controller=None
    )
    data_collector.radios_to_online()  # blocking

    if len(yaml_config["receivers"]) == 0:
        logging.info("EMITTER ONLINE!")
        while True:
            time.sleep(5)
    else:
        data_collector.start()
        while data_collector.is_collecting():
            time.sleep(5)

    # we finished lets move files out to final positions
    for idx in range(len(temp_filenames)):
        os.rename(temp_filenames[idx], final_filenames[idx])
