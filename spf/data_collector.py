import logging
import sys
import threading
import time
from typing import Optional

import numpy as np
from attr import dataclass
from tqdm import tqdm

from spf.rf import beamformer_given_steering, precompute_steering_vectors
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    PPlus,
    ReceiverConfig,
    get_avg_phase,
    get_pplus,
    setup_rx,
    setup_rxtx,
)
from spf.wall_array_v2 import v2_column_names, v3_column_names


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


def prepare_record_entry_v3(ds: DataSnapshot, current_pos_heading_and_time):
    # t,rx,ry,rtheta,rspacing,avgphase,sds
    return np.hstack(
        [
            ds.timestamp,
            current_pos_heading_and_time["gps_time"],  # 1
            current_pos_heading_and_time["gps"],  # 2
            current_pos_heading_and_time["heading"],  # 1
            ds.rx_theta_in_pis * np.pi,  # 1
            ds.rx_spacing,  # 1
            ds.avg_phase_diff,  # 2
            ds.rssis,  # 2
            ds.gains,  # 2
            ds.beam_sds,  # 65
        ]
    )


def prepare_record_entry_v2(ds: DataSnapshot, rx_pos: np.array, tx_pos: np.array):
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
    def __init__(self, pplus: PPlus, time_offset, nthetas):
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
        steering_vectors = precompute_steering_vectors(
            receiver_positions=self.pplus.rx_config.rx_pos,
            carrier_frequency=self.pplus.rx_config.lo,
            spacing=self.nthetas,
        )

        while self.run:
            if self.read_lock.acquire(blocking=True, timeout=0.5):
                # got the semaphore, read some data!
                tries = 0
                try:
                    signal_matrix = self.pplus.sdr.rx()
                    rssis = self.pplus.rssis()
                    gains = self.pplus.gains()
                    # rssi_and_gain = self.pplus.get_rssi_and_gain()
                except Exception as e:
                    logging.error(
                        f"Failed to receive RX data! removing file : retry {tries} {e}",
                    )
                    time.sleep(0.1)
                    tries += 1
                    if tries > 15:
                        logging.error("GIVE UP")
                        sys.exit(1)

                # process the data
                current_time = time.time() - self.time_offset  # timestamp
                # _, beam_sds, _ = beamformer(
                #     self.pplus.rx_config.rx_pos,
                #     signal_matrix,
                #     self.pplus.rx_config.lo,
                #     spacing=self.nthetas,
                # )
                signal_matrix = np.vstack(signal_matrix)
                beam_sds = beamformer_given_steering(
                    steering_vectors=steering_vectors, signal_matrix=signal_matrix
                )
                # assert np.isclose(beam_sds, beam_sds2).all()

                avg_phase_diff = get_avg_phase(signal_matrix)

                self.data = DataSnapshot(
                    timestamp=current_time,
                    rx_center_pos=self.pplus.rx_config.rx_spacing,
                    rx_theta_in_pis=self.pplus.rx_config.rx_theta_in_pis,
                    rx_spacing=self.pplus.rx_config.rx_spacing,
                    beam_sds=beam_sds,
                    avg_phase_diff=avg_phase_diff,
                    signal_matrix=None,
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


class DataCollector:
    def __init__(self, yaml_config, filename_npy, position_controller, tag=""):
        self.yaml_config = yaml_config
        self.filename_npy = filename_npy
        self.record_matrix = None
        self.position_controller = position_controller
        self.finished_collecting = False

    def radios_to_online(self):
        # record matrix
        if not self.yaml_config["dry-run"]:
            self.record_matrix = np.memmap(
                self.filename_npy,
                dtype="float32",
                mode="w+",
                shape=(
                    2,  # TODO should be nreceivers
                    self.yaml_config["n-records-per-receiver"],
                    len(self.column_names),
                ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
            )

        # lets open all the radios
        radio_uris = []
        if self.yaml_config["emitter"]["type"] == "sdr":
            radio_uris.append(self.yaml_config["emitter"]["receiver-uri"])
        for receiver in self.yaml_config["receivers"]:
            radio_uris.append(receiver["receiver-uri"])
        for radio_uri in radio_uris:
            get_pplus(uri=radio_uri)

        time.sleep(0.1)

        target_yaml_config = self.yaml_config["emitter"]
        if target_yaml_config["type"] == "sdr":  # this  wont be the case for mavlink
            # setup the emitter
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
                uri=target_yaml_config["receiver-uri"],
            )
            target_tx_config = EmitterConfig(
                lo=target_yaml_config["f-carrier"],
                rf_bandwidth=target_yaml_config["bandwidth"],
                sample_rate=target_yaml_config["f-sampling"],
                intermediate=target_yaml_config["f-intermediate"],
                gains=[target_yaml_config["tx-gain"], -80],
                enabled_channels=[0],
                cyclic=True,
                uri=target_yaml_config["emitter-uri"],
                motor_channel=(
                    target_yaml_config["motor_channel"]
                    if "motor_channel" in target_yaml_config
                    else None
                ),
            )

            pplus_rx, _ = setup_rxtx(
                rx_config=target_rx_config, tx_config=target_tx_config, leave_tx_on=True
            )
            pplus_rx.close_rx()

        # get radios online
        self.receiver_pplus = {}
        for receiver in self.yaml_config["receivers"]:
            rx_config = ReceiverConfig(
                lo=receiver["f-carrier"],
                rf_bandwidth=receiver["bandwidth"],
                sample_rate=receiver["f-sampling"],
                gains=[receiver["rx-gain"], receiver["rx-gain"]],
                gain_control_modes=[
                    receiver["rx-gain-mode"],
                    receiver["rx-gain-mode"],
                ],
                enabled_channels=[0, 1],
                buffer_size=receiver["buffer-size"],
                intermediate=receiver["f-intermediate"],
                uri=receiver["receiver-uri"],
                rx_spacing=receiver["antenna-spacing-m"],
                rx_theta_in_pis=receiver["theta-in-pis"],
                motor_channel=(
                    receiver["motor_channel"] if "motor_channel" in receiver else None
                ),
                rx_buffers=receiver["rx-buffers"],
            )
            assert "emitter-uri" not in receiver
            assert (
                "skip_phase_calibration" not in self.yaml_config
                or self.yaml_config["skip_phase_calibration"]
            )
            # there is no emitter to setup, its already blasting
            pplus_rx = setup_rx(rx_config=rx_config)

            if pplus_rx is None:
                logging.info("Failed to bring RXTX online, shuttingdown")
                sys.exit(1)
            else:
                logging.debug("RX online!")
                self.receiver_pplus[pplus_rx.uri] = pplus_rx
                assert pplus_rx.rx_config.rx_pos is not None

        self.read_threads = []
        time_offset = time.time()
        for _, pplus_rx in self.receiver_pplus.items():
            if pplus_rx is None:
                continue
            read_thread = ThreadedRX(
                pplus_rx, time_offset, nthetas=self.yaml_config["n-thetas"]
            )
            read_thread.start_read_thread()
            self.read_threads.append(read_thread)

        self.collector_thread = threading.Thread(
            target=self.run_collector_thread, daemon=True
        )

    def start(self):
        self.collector_thread.start()

    def run_collector_thread(self):
        raise NotImplementedError

    def is_collecting(self):
        return not self.finished_collecting


class DroneDataCollector(DataCollector):
    def __init__(self, *args, **kwargs):
        super(DroneDataCollector, self).__init__(*args, **kwargs)
        self.column_names = v3_column_names(nthetas=self.yaml_config["n-thetas"])

    def run_collector_thread(self):
        record_index = 0
        for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):
            for read_thread_idx, read_thread in enumerate(self.read_threads):
                while not read_thread.ready_lock.acquire(timeout=0.5):
                    pass
                ###
                # copy the data out
                current_pos_heading_and_time = (
                    self.position_controller.get_position_bearing_and_time()
                )

                self.record_matrix[read_thread_idx, record_index] = (
                    prepare_record_entry_v3(
                        ds=read_thread.data,
                        current_pos_heading_and_time=current_pos_heading_and_time,
                    )
                )

                read_thread.read_lock.release()

        self.finished_collecting = True


class GrblDataCollector(DataCollector):
    def __init__(self, *args, **kwargs):
        super(GrblDataCollector, self).__init__(*args, **kwargs)
        self.column_names = v2_column_names(nthetas=self.yaml_config["n-thetas"])

    def run_collector_thread(self):
        record_index = 0
        for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):
            for read_thread_idx, read_thread in enumerate(self.read_threads):
                while not read_thread.ready_lock.acquire(timeout=0.5):
                    pass
                ###
                # copy the data out

                tx_pos = self.position_controller.controller.position["xy"][
                    self.yaml_config["emitter"]["motor_channel"]
                ]
                rx_pos = self.position_controller.controller.position["xy"][
                    read_thread.pplus.rx_config.motor_channel
                ]

                self.record_matrix[read_thread_idx, record_index] = (
                    prepare_record_entry_v2(
                        ds=read_thread.data, rx_pos=rx_pos, tx_pos=tx_pos
                    )
                )

                read_thread.read_lock.release()
        self.finished_collecting = True
