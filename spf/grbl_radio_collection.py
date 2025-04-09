import argparse
import json
import logging
import os
import time
from datetime import datetime

import yaml

from spf.data_collector import GrblDataCollector, GrblDataCollectorRaw
from spf.grbl.grbl_interactive import get_default_gm
from spf.utils import (
    DataVersionNotImplemented,
    filenames_from_time_in_seconds,
    load_config,
)


def grbl_radio_parser():
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
    parser.add_argument(
        "-n", "--dry-run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--tx-gain", type=int, help="tag files", required=False, default=None
    )
    parser.add_argument(
        "--device-mapping", type=str, help="device mapping", required=False, default=None
    )
    parser.add_argument(
        "--n-records", type=int, help="nrecords", required=False, default=-1
    )
    parser.add_argument(
        "--seconds-per-sample",
        type=float,
        help="seconds per sample",
        required=False,
        default=-1.0,
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
        "-o",
        "--output-dir",
        type=str,
        help="output dir",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--temp", type=str, help="temp dirname", required=False, default="./temp"
    )
    return parser


def grbl_radio_main(args):
    run_started_at = datetime.now().timestamp()  #
    # read YAML
    # with open(args.yaml_config, "r") as stream:
    yaml_config = load_config(args.yaml_config)  # yaml.safe_load(stream)
    if args.serial is not None or "grbl-serial" not in yaml_config:
        yaml_config["grbl-serial"] = args.serial
    if args.routine is not None:
        yaml_config["routine"] = args.routine
    assert yaml_config["routine"] is not None

    # open device mapping and figure out URIs
    if args.device_mapping is not None:
        with open(args.device_mapping, "r") as device_mapping:
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

    if args.tx_gain is not None:
        assert yaml_config["emitter"]["type"] == "sdr"
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    if 'craft' not in yaml_config:
        yaml_config['craft']="wallarrayv3"

    if "dry-run" not in yaml_config or args.dry_run:
        yaml_config["dry-run"] = args.dry_run

    if "n-records-per-receiver" not in yaml_config or args.n_records >= 0:
        yaml_config["n-records-per-receiver"] = args.n_records

    if "seconds-per-sample" not in yaml_config or args.seconds_per_sample >= 0:
        yaml_config["seconds-per-sample"] = args.n_records

    for receiver in yaml_config["receivers"]:
        assert 'motor_channel' in receiver

    temp_filenames, final_filenames = filenames_from_time_in_seconds(
        run_started_at,
        args.temp,
        yaml_config,
        data_version=yaml_config["data-version"],
        craft=yaml_config['craft'],
        tag=args.tag,
    )

    logger = logging.getLogger(__name__)

    # setup logging
    handlers = [
        logging.StreamHandler(),
    ]
    if not yaml_config["dry-run"]:
        handlers.append(logging.FileHandler(temp_filenames["log"]))
    logging.basicConfig(
        handlers=handlers,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, args.logging_level.upper(), None),
    )

    # make a copy of the YAML
    if not yaml_config["dry-run"]:
        with open(temp_filenames["yaml"], "w") as outfile:
            yaml.dump(yaml_config, outfile, default_flow_style=False)

    logging.info(json.dumps(yaml_config, sort_keys=True, indent=4))

    # setup GRBL
    gm = get_default_gm(yaml_config["grbl-serial"], yaml_config["routine"])
    gm.start()

    logging.info("Starting data collector...")
    if yaml_config["data-version"] == 2:
        data_collector = GrblDataCollector(
            data_filename=temp_filenames["data"],
            yaml_config=yaml_config,
            position_controller=gm,
        )
    elif yaml_config["data-version"] == 5:
        data_collector = GrblDataCollectorRaw(
            data_filename=temp_filenames["data"],
            yaml_config=yaml_config,
            position_controller=gm,
        )
    else:
        raise DataVersionNotImplemented

    data_collector.radios_to_online()  # blocking

    while not gm.has_planner_started_moving():
        logging.info(f"waiting for grbl to start moving {time.time()}")
        time.sleep(5)  # easy poll this
    logging.info("GRBL IS READY!!! LETS GOOO!!!")

    data_collector.start()
    while data_collector.is_collecting():
        time.sleep(5)

    data_collector.done()

    # we finished lets move files out to final positions
    if not yaml_config["dry-run"]:
        logging.info("GRBLRadioCollection: Moving files to final location ...")
        for k in temp_filenames:
            logging.info(
                f" > GRBLRadioCollection: Moving files to final location ... {temp_filenames[k]}, {final_filenames[k]}"
            )
            os.rename(temp_filenames[k], final_filenames[k])
    logging.info("GRBLRadioCollection: sleep before exit")
    time.sleep(10)


if __name__ == "__main__":
    parser = grbl_radio_parser()
    args = parser.parse_args()
    grbl_radio_main(args)
