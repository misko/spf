import argparse
import faulthandler
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import yaml
from grbl.grbl_interactive import get_default_gm, stop_grbl
from tqdm import tqdm

from spf.rf import beamformer
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    ReceiverConfig,
    get_avg_phase,
    get_pplus,
    setup_rxtx,
    setup_rxtx_and_phase_calibration,
    shutdown_radios,
)

faulthandler.enable()

run_collection = True


@dataclass
class DataSnapshot:
    timestamp: float
    rx_theta_in_pis: float
    rx_center_pos: np.array
    rx_spacing: float
    avg_phase_diff: float
    beam_sds: np.array


def prepare_record_entry(ds: DataSnapshot, rx_pos: np.array, tx_pos: np.array):
    # t,rx,ry,rtheta,rspacing,avgphase,sds
    return np.hstack(
        [
            ds.timestamp,  # 1
            tx_pos,  # 2
            rx_pos,  # 2
            ds.rx_theta_in_pis,  # 1
            ds.rx_spacing,  # 1
            ds.avg_phase_diff,  # 2
            ds.beam_sds,  # 65
        ]
    )


def shutdown():
    global run_collection
    run_collection = False
    shutdown_radios()
    stop_grbl()


def signal_handler(sig, frame):
    logging.info("Ctrl-c issued -> SHUT IT DOWN!")
    shutdown()


signal.signal(signal.SIGINT, signal_handler)


class ThreadedRX:
    def __init__(self, pplus, time_offset):
        self.pplus = pplus
        self.read_lock = threading.Lock()
        self.ready_lock = threading.Lock()
        self.ready_lock.acquire()
        self.run = False
        self.time_offset = time_offset

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
                except Exception as e:
                    logging.error(
                        f"Failed to receive RX data! removing file : retry {tries}",
                        e,
                    )
                    time.sleep(0.1)
                    tries += 1
                    if tries > 10:
                        logging.error("GIVE UP")
                        return

                # process the data
                signal_matrix[1] *= np.exp(1j * self.pplus.phase_calibration)
                current_time = time.time() - self.time_offset  # timestamp
                _, beam_sds, _ = beamformer(
                    self.pplus.rx_config.rx_pos,
                    signal_matrix,
                    self.pplus.rx_config.intermediate,
                )

                avg_phase_diff = get_avg_phase(signal_matrix)

                self.data = DataSnapshot(
                    timestamp=current_time,
                    rx_center_pos=self.pplus.rx_config.rx_spacing,
                    rx_theta_in_pis=self.pplus.rx_config.rx_theta_in_pis,
                    rx_spacing=self.pplus.rx_config.rx_spacing,
                    beam_sds=beam_sds,
                    avg_phase_diff=avg_phase_diff,
                )

                try:
                    self.ready_lock.release()  # tell the parent we are ready to provide
                except Exception as e:
                    logging.error(f"Thread encountered an issue exiting {str(e)}")
                    self.run = False
                # logging.info(f"{self.pplus.rx_config.uri} READY")

        logging.info(f"{str(self.pplus.rx_config.uri)} PPlus read_forever() exit!")


def bounce_grbl(gm):
    direction = None
    global run_collection
    while run_collection:
        logging.info("TRY TO BOUNCE")
        try:
            direction = gm.bounce(100, direction=direction)
        except Exception as e:
            logging.error(e)
        if not run_collection:
            break
        logging.info("TRY TO BOUNCE RET")
        time.sleep(10)  # cool off the motor
    logging.info("Exiting GRBL thread")


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
        "-l",
        "--logging-level",
        type=str,
        help="Logging level",
        default="INFO",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--grbl-serial",
        type=str,
        help="GRBL serial dev",
        default=None,
        required=False,
    )
    args = parser.parse_args()

    # setup logging
    start_logging_at = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"{start_logging_at}.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=getattr(logging, args.logging_level.upper(), None),
    )

    # read YAML
    with open(args.yaml_config, "r") as stream:
        yaml_config = yaml.safe_load(stream)

    logging.info(json.dumps(yaml_config, sort_keys=True, indent=4))

    # lets open all the radios
    radio_uris = ["ip:%s" % yaml_config["emitter"]["receiver-ip"]]
    for receiver in yaml_config["receivers"]:
        radio_uris.append("ip:%s" % receiver["receiver-ip"])
    for radio_uri in radio_uris:
        get_pplus(uri=radio_uri)

    time.sleep(0.1)

    # get radios online
    receiver_pplus = []
    pplus_rx, pplus_tx = (None, None)
    for receiver in yaml_config["receivers"]:
        rx_config = ReceiverConfig(
            lo=receiver["f-carrier"],
            rf_bandwidth=receiver["bandwidth"],
            sample_rate=receiver["f-sampling"],
            gains=[receiver["rx-gain"], receiver["rx-gain"]],
            gain_control_mode=receiver["rx-gain-mode"],
            enabled_channels=[0, 1],
            buffer_size=receiver["buffer-size"],
            intermediate=receiver["f-intermediate"],
            uri="ip:%s" % receiver["receiver-ip"],
            rx_spacing=receiver["antenna-spacing-m"],
            rx_theta_in_pis=receiver["theta-in-pis"],
            motor_channel=receiver["motor_channel"],
        )
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
        pplus_rx, pplus_tx = setup_rxtx_and_phase_calibration(
            rx_config=rx_config,
            tx_config=tx_config,
            n_calibration_frames=800,
            # leave_tx_on=False,
            # using_tx_already_on=None,
        )
        pplus_rx.record_matrix = np.memmap(
            receiver["output-file"],
            dtype="float32",
            mode="w+",
            shape=(
                yaml_config["n-records-per-receiver"],
                7 + 2 + 65,
            ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
        )
        logging.info("RX online!")
        receiver_pplus.append(pplus_rx)

    # setup the emitter
    target_yaml_config = yaml_config["emitter"]
    target_rx_config = ReceiverConfig(
        lo=target_yaml_config["f-carrier"],
        rf_bandwidth=target_yaml_config["bandwidth"],
        sample_rate=target_yaml_config["f-sampling"],
        gains=[target_yaml_config["rx-gain"], target_yaml_config["rx-gain"]],
        gain_control_mode=target_yaml_config["rx-gain-mode"],
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
        gains=[-30, -80],
        enabled_channels=[0],
        cyclic=True,
        uri="ip:%s" % target_yaml_config["emitter-ip"],
        motor_channel=target_yaml_config["motor_channel"],
    )

    setup_rxtx(rx_config=target_rx_config, tx_config=target_tx_config)

    # threadA semaphore to produce fresh data
    # threadB semaphore to produce fresh data
    # thread to bounce
    #

    # setup GRBL
    gm = None
    gm_thread = None
    if args.grbl_serial is not None:
        gm = get_default_gm(args.grbl_serial)
        gm_thread = threading.Thread(target=bounce_grbl, args=(gm,))
        gm_thread.start()

    # setup read threads

    time_offset = time.time()
    read_threads = []
    for pplus_rx in receiver_pplus:
        read_thread = ThreadedRX(pplus_rx, time_offset)
        read_thread.start_read_thread()
        read_threads.append(read_thread)

    record_index = 0
    for record_index in tqdm(range(yaml_config["n-records-per-receiver"])):
        if not run_collection:
            logging.info("Breaking man loop early")
            break
        for read_thread in read_threads:
            while run_collection and not read_thread.ready_lock.acquire(timeout=0.5):
                pass
            ###
            # copy the data out

            rx_pos = np.array([0, 0])
            tx_pos = np.array([0, 0])
            if gm is not None:
                tx_pos = gm.controller.position["xy"][target_tx_config.motor_channel]
                rx_pos = gm.controller.position["xy"][
                    read_thread.pplus.rx_config.motor_channel
                ]

            read_thread.pplus.record_matrix[record_index] = prepare_record_entry(
                ds=read_thread.data, rx_pos=rx_pos, tx_pos=tx_pos
            )
            ###
            read_thread.read_lock.release()
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
