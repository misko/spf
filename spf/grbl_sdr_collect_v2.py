import argparse
import faulthandler
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import yaml
from grbl.grbl_interactive import get_default_gm, stop_grbl
from matplotlib import pyplot as plt
from tqdm import tqdm

from spf.rf import beamformer
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    ReceiverConfig,
    get_avg_phase,
    get_pplus,
    plot_recv_signal,
    setup_rx,
    setup_rxtx,
    setup_rxtx_and_phase_calibration,
    shutdown_radios,
)
from spf.wall_array_v2 import v2_column_names

faulthandler.enable()

# keep running radio data collection and GRBL
run_collection = True


def shutdown():
    global run_collection
    run_collection = False
    shutdown_radios()
    stop_grbl()


def signal_handler(sig, frame):
    logging.info("Ctrl-c issued -> SHUT IT DOWN!")
    shutdown()


signal.signal(signal.SIGINT, signal_handler)


@dataclass
class DataSnapshot:
    timestamp: float
    rx_theta_in_pis: float
    rx_center_pos: np.array
    rx_spacing: float
    avg_phase_diff: float
    beam_sds: np.array
    signal_matrix: Optional[np.array]
    rssis: np.array
    gains: np.array


def prepare_record_entry(ds: DataSnapshot, rx_pos: np.array, tx_pos: np.array):
    # t,rx,ry,rtheta,rspacing,avgphase,sds
    return np.hstack(
        [
            ds.timestamp,  # 1
            tx_pos,  # 2
            rx_pos,  # 2
            ds.rx_theta_in_pis * np.pi,  # 1
            ds.rx_spacing,  # 1
            ds.avg_phase_diff,  # 2
            ds.rssis,  # 2
            ds.gains,  # 2
            ds.beam_sds,  # 65
        ]
    )


class ThreadedRX:
    def __init__(self, pplus, time_offset, nthetas):
        self.pplus = pplus
        self.read_lock = threading.Lock()
        self.ready_lock = threading.Lock()
        self.ready_lock.acquire()
        self.run = False
        self.time_offset = time_offset
        self.nthetas = nthetas
        assert self.pplus.rx_config.rx_pos is not None

    def start_read_thread(self):
        self.t = threading.Thread(target=self.read_forever)
        self.run = True
        self.t.start()

    def read_forever(self):
        logging.info(f"{str(self.pplus.rx_config.uri)} PPlus read_forever()")
        while self.run:
            if self.read_lock.acquire(blocking=True, timeout=0.5):
                # got the semaphore, read some data!
                tries = 0
                try:
                    signal_matrix = self.pplus.sdr.rx()
                    rssis = self.pplus.rssis()
                    gains = self.pplus.gains()
                except Exception as e:
                    logging.error(
                        f"Failed to receive RX data! removing file : retry {tries}",
                        e,
                    )
                    time.sleep(0.1)
                    tries += 1
                    if tries > 15:
                        logging.error("GIVE UP")
                        shutdown()
                        return

                # process the data
                signal_matrix[1] *= np.exp(1j * self.pplus.phase_calibration)
                current_time = time.time() - self.time_offset  # timestamp
                _, beam_sds, _ = beamformer(
                    self.pplus.rx_config.rx_pos,
                    signal_matrix,
                    self.pplus.rx_config.lo,
                    spacing=self.nthetas,
                )

                avg_phase_diff = get_avg_phase(signal_matrix)

                self.data = DataSnapshot(
                    timestamp=current_time,
                    rx_center_pos=self.pplus.rx_config.rx_spacing,
                    rx_theta_in_pis=self.pplus.rx_config.rx_theta_in_pis,
                    rx_spacing=self.pplus.rx_config.rx_spacing,
                    beam_sds=beam_sds,
                    avg_phase_diff=avg_phase_diff,
                    signal_matrix=signal_matrix if args.plot else None,
                    rssis=rssis,
                    gains=gains,
                )

                try:
                    self.ready_lock.release()  # tell the parent we are ready to provide
                except Exception as e:
                    logging.error(f"Thread encountered an issue exiting {str(e)}")
                    self.run = False
                # logging.info(f"{self.pplus.rx_config.uri} READY")

        logging.info(f"{str(self.pplus.rx_config.uri)} PPlus read_forever() exit!")


def grbl_thread_runner(gm, routine):
    global run_collection
    while run_collection:
        logging.info("GRBL thread runner")
        if routine is None:
            logging.info("No routine to run, just spining")
            while run_collection:
                time.sleep(0.5)
        else:
            try:
                if routine in gm.routines:
                    logging.info(f"RUNNING ROUTINE {routine}")
                    gm.routines[routine]()
                    gm.get_ready()
                    gm.run()
                else:
                    raise ValueError(f"Unknown grbl routine f{routine}")
            except Exception as e:
                logging.error(e)
        if not run_collection:
            break
        logging.info("GRBL thread runner loop")
        time.sleep(10)  # cool off the motor
    logging.info("Exiting GRBL thread")


temp_filenames = []
final_filenames = []

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
        "--tx-gain", type=int, help="tag files", required=False, default=None
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="tag files", required=False, default=""
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
        "-s",
        "--grbl-serial",
        type=str,
        help="GRBL serial dev",
        default=None,
        required=False,
    )
    parser.add_argument("-p", "--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument("-n", "--dry-run", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--skip-phase-calibration", action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    run_started_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # read YAML
    with open(args.yaml_config, "r") as stream:
        yaml_config = yaml.safe_load(stream)

    # add in our current config
    if args.routine is not None:
        yaml_config["routine"] = args.routine

    if args.tx_gain is not None:
        assert "emitter" in yaml_config
        yaml_config["emitter"]["tx-gain"] = args.tx_gain

    output_files_prefix = f"wallarrayv2_{run_started_at}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}"
    if args.tag != "":
        output_files_prefix += f"_tag_{args.tag}"

    # setup filename
    filename_log = f"{output_files_prefix}.log.tmp"
    filename_yaml = f"{output_files_prefix}.yaml.tmp"
    filename_npy = f"{output_files_prefix}.npy.tmp"
    temp_filenames = [filename_log, filename_yaml, filename_npy]
    final_filenames = [x.replace(".tmp", "") for x in temp_filenames]

    # setup logging
    handlers = [logging.StreamHandler()]
    if not args.dry_run:
        handlers += [
            logging.FileHandler(filename_log),
        ]
    logging.basicConfig(
        handlers=handlers,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, args.logging_level.upper(), None),
    )
    if args.grbl_serial is None:
        logging.info("Running without GRBL SERIAL!!")
        for x in range(50):
            if not run_collection:
                break
            time.sleep(0.1)

    if run_collection:
        if not args.dry_run:
            with open(filename_yaml, "w") as outfile:
                yaml.dump(yaml_config, outfile, default_flow_style=False)

        # record matrix
        column_names = v2_column_names(nthetas=yaml_config["n-thetas"])
        if args.dry_run:
            record_matrix = None
        else:
            record_matrix = np.memmap(
                filename_npy,
                dtype="float32",
                mode="w+",
                shape=(
                    2,  # TODO should be nreceivers
                    yaml_config["n-records-per-receiver"],
                    len(column_names),
                ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
            )

        logging.info(json.dumps(yaml_config, sort_keys=True, indent=4))

        # lets open all the radios
        radio_uris = []
        if "emitter" in yaml_config:
            radio_uris.append(["ip:%s" % yaml_config["emitter"]["receiver-ip"]])
        for receiver in yaml_config["receivers"]:
            radio_uris.append("ip:%s" % receiver["receiver-ip"])
        for radio_uri in radio_uris:
            get_pplus(uri=radio_uri)

        time.sleep(0.1)

    # get radios online
    receiver_pplus = {}
    pplus_rx, pplus_tx = (None, None)
    for receiver in yaml_config["receivers"]:
        if run_collection:
            rx_config = ReceiverConfig(
                lo=receiver["f-carrier"],
                rf_bandwidth=receiver["bandwidth"],
                sample_rate=receiver["f-sampling"],
                gains=[receiver["rx-gain"], receiver["rx-gain"]],
                gain_control_modes=[receiver["rx-gain-mode"], receiver["rx-gain-mode"]],
                enabled_channels=[0, 1],
                buffer_size=receiver["buffer-size"],
                intermediate=receiver["f-intermediate"],
                uri="ip:%s" % receiver["receiver-ip"],
                rx_spacing=receiver["antenna-spacing-m"],
                rx_theta_in_pis=receiver["theta-in-pis"],
                motor_channel=receiver["motor_channel"],
                rx_buffers=receiver["rx-buffers"],
            )
            if "emitter-ip" in receiver:
                tx_config = EmitterConfig(
                    lo=receiver["f-carrier"],
                    rf_bandwidth=receiver["bandwidth"],
                    sample_rate=receiver["f-sampling"],
                    intermediate=receiver["f-intermediate"],
                    gains=[-30, -80],
                    enabled_channels=[0],
                    cyclic=True,
                    uri="ip:%s" % receiver["emitter-ip"],
                )
                if args.skip_phase_calibration or (
                    "skip_phase_calibration" in yaml_config
                    and yaml_config["skip_phase_calibration"]
                ):
                    pplus_rx, pplus_tx = setup_rxtx(
                        rx_config=rx_config, tx_config=tx_config
                    )
                else:
                    pplus_rx, pplus_tx = setup_rxtx_and_phase_calibration(
                        rx_config=rx_config,
                        tx_config=tx_config,
                        n_calibration_frames=yaml_config["calibration-frames"],
                    )
            else:
                assert args.skip_phase_calibration or (
                    "skip_phase_calibration" in yaml_config
                    and yaml_config["skip_phase_calibration"]
                )
                # there is no emitter to setup, its already blasting
                pplus_rx = setup_rx(rx_config=rx_config)

            if pplus_rx is None:
                logging.info("Failed to bring RXTX online, shuttingdown")
                run_collection = False
                break
            else:
                logging.debug("RX online!")
                receiver_pplus[pplus_rx.uri] = pplus_rx
                assert pplus_rx.rx_config.rx_pos is not None

    if run_collection and "emitter" in yaml_config:
        # setup the emitter
        target_yaml_config = yaml_config["emitter"]
        target_rx_config = ReceiverConfig(
            lo=target_yaml_config["f-carrier"],
            rf_bandwidth=target_yaml_config["bandwidth"],
            sample_rate=target_yaml_config["f-sampling"],
            gains=[target_yaml_config["rx-gain"], target_yaml_config["rx-gain"]],
            gain_control_modes=[
                target_yaml_config["rx-gain-mode"],
                target_yaml_config["rx-gain-mode"],
            ],
            enabled_channels=[0, 1],
            buffer_size=target_yaml_config["buffer-size"],
            intermediate=target_yaml_config["f-intermediate"],
            uri="ip:%s" % target_yaml_config["receiver-ip"],
        )
        target_tx_config = EmitterConfig(
            lo=target_yaml_config["f-carrier"],
            rf_bandwidth=target_yaml_config["bandwidth"],
            sample_rate=target_yaml_config["f-sampling"],
            intermediate=target_yaml_config["f-intermediate"],
            gains=[target_yaml_config["tx-gain"], -80],
            enabled_channels=[0],
            cyclic=True,
            uri="ip:%s" % target_yaml_config["emitter-ip"],
            motor_channel=target_yaml_config["motor_channel"],
        )

        if target_rx_config.uri not in receiver_pplus:
            setup_rxtx(
                rx_config=target_rx_config, tx_config=target_tx_config, leave_tx_on=True
            )
        else:
            logging.info(f"Re-using {target_rx_config.uri} as RX for TX")
            setup_rxtx(
                rx_config=target_rx_config,
                tx_config=target_tx_config,
                leave_tx_on=True,
                provided_pplus_rx=receiver_pplus[target_rx_config.uri],
            )

    # setup GRBL
    gm = None
    gm_thread = None
    if run_collection:
        if args.grbl_serial is not None:
            gm = get_default_gm(args.grbl_serial)
            gm.routines[yaml_config["routine"]]()  # setup run
            logging.info("Waiting for GRBL to get into position")
            gm.get_ready()  # move into position
            gm_thread = threading.Thread(
                target=grbl_thread_runner, args=(gm, yaml_config["routine"])
            )
            gm_thread.start()

    time.sleep(0.2)
    # setup read threads

    read_threads = []
    time_offset = time.time()
    if run_collection:
        for _, pplus_rx in receiver_pplus.items():
            if pplus_rx is None:
                continue
            read_thread = ThreadedRX(
                pplus_rx, time_offset, nthetas=yaml_config["n-thetas"]
            )
            read_thread.start_read_thread()
            read_threads.append(read_thread)

    plot_figures_and_axs = None
    if args.plot is not None:
        plot_figures_and_axs = [
            plt.subplots(2, 4, figsize=(16, 6), layout="constrained")
            for _ in range(len(receiver_pplus))
        ]

    if run_collection:
        record_index = 0
        for record_index in tqdm(range(yaml_config["n-records-per-receiver"])):
            if not run_collection:
                logging.info("Breaking man loop early")
                break
            for read_thread_idx, read_thread in enumerate(read_threads):
                while run_collection and not read_thread.ready_lock.acquire(
                    timeout=0.5
                ):
                    pass
                ###
                # copy the data out

                rx_pos = np.array([0, 0])
                tx_pos = np.array([0, 0])
                if gm is not None:
                    tx_pos = gm.controller.position["xy"][
                        target_tx_config.motor_channel
                    ]
                    rx_pos = gm.controller.position["xy"][
                        read_thread.pplus.rx_config.motor_channel
                    ]
                if not args.dry_run:
                    record_matrix[read_thread_idx, record_index] = prepare_record_entry(
                        ds=read_thread.data, rx_pos=rx_pos, tx_pos=tx_pos
                    )

                if plot_figures_and_axs is not None and pplus_rx is not None:
                    fig, axs = plot_figures_and_axs[read_thread_idx]
                    plot_recv_signal(
                        read_thread.pplus,
                        frames=1,
                        fig=fig,
                        axs=axs,
                        title=read_thread.pplus.uri,
                        signal_matrixs=[read_thread.data.signal_matrix],
                    )
                ###

                read_thread.read_lock.release()

    if run_collection:  # keep files and move to final
        for idx in range(len(temp_filenames)):
            os.rename(temp_filenames[idx], final_filenames[idx])

    shutdown()
    logging.info("Shuttingdown: sending false to threads")
    for read_thread in read_threads:
        read_thread.run = False
    logging.info("Shuttingdown: start thread join!")
    for read_thread in read_threads:
        read_thread.t.join()
    if gm_thread is not None:
        logging.info("Grab grbl thread")
        gm_thread.join()

    logging.info("Shuttingdown: done")
