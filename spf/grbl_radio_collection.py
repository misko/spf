import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime

import yaml

from spf.data_collector import GrblDataCollector
from spf.grbl.grbl_interactive import get_default_gm

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
        "-s", "--serial", type=str, help="serial", required=False, default=None
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="tag files", required=False, default=""
    )
    parser.add_argument("-n", "--dry-run", action=argparse.BooleanOptionalAction)
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
    args = parser.parse_args()

    run_started_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # read YAML
    with open(args.yaml_config, "r") as stream:
        yaml_config = yaml.safe_load(stream)
    if args.serial is not None or "grbl-serial" not in yaml_config:
        yaml_config["grbl-serial"] = args.serial
    if args.routine is not None:
        yaml_config["routine"] = args.routine
    assert yaml_config["routine"] is not None

    if args.tx_gain is not None:
        assert yaml_config["emitter"]["type"] == "sdr"
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    output_files_prefix = f"rover_{run_started_at}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}"
    if args.tag != "":
        output_files_prefix += f"_tag_{args.tag}"

    if "dry-run" not in yaml_config or args.dry_run:
        yaml_config["dry-run"] = args.dry_run

    # setup filename
    tmpdir = tempfile.TemporaryDirectory()
    temp_dir_name = tmpdir.name
    filename_log = f"{temp_dir_name}/{output_files_prefix}.log.tmp"
    filename_yaml = f"{temp_dir_name}/{output_files_prefix}.yaml.tmp"
    filename_npy = f"{temp_dir_name}/{output_files_prefix}.npy.tmp"
    temp_filenames = [filename_log, filename_yaml, filename_npy]
    final_filenames = [os.path.basename(x.replace(".tmp", "")) for x in temp_filenames]

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

    # setup GRBL
    gm = get_default_gm(yaml_config["grbl-serial"], yaml_config["routine"])
    if yaml_config["grbl-serial"] is not None:
        gm.start()

    logging.info("Starting data collector...")
    data_collector = GrblDataCollector(
        filename_npy=filename_npy, yaml_config=yaml_config, position_controller=gm
    )
    data_collector.radios_to_online()  # blocking

    while not gm.has_planner_started_moving():
        logging.info(f"waiting for grbl to start moving {time.time()}")
        time.sleep(5)  # easy poll this
    logging.info("DRONE IS READY!!! LETS GOOO!!!")

    data_collector.start()
    while data_collector.is_collecting():
        time.sleep(5)

    # we finished lets move files out to final positions
    for idx in range(len(temp_filenames)):
        os.rename(temp_filenames[idx], final_filenames[idx])

    data_collector.done()
