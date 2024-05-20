import logging
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from attr import dataclass
from tqdm import tqdm

from spf.dataset.rover_idxs import v3rx_column_names
from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys, v4rx_new_dataset
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.dataset.wall_array_v2_idxs import v2_column_names
from spf.rf import beamformer_given_steering, get_avg_phase, precompute_steering_vectors
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    PPlus,
    ReceiverConfig,
    get_pplus,
    rx_config_from_receiver_yaml,
    setup_rx,
    setup_rxtx,
)
from spf.utils import zarr_shrink


@dataclass
class DataSnapshotRaw:
    signal_matrix: np.array
    system_timestamp: float
    rssis: np.array
    gains: np.array
    rx_theta_in_pis: float
    rx_spacing: float
    rx_lo: float
    rx_bandwidth: float
    avg_phase_diff: float


@dataclass
class DataSnapshotV4(DataSnapshotRaw):
    gps_timestamp: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_long: Optional[float] = None
    heading: Optional[float] = None


@dataclass
class DataSnapshotV5(DataSnapshotRaw):
    tx_pos_x_mm: Optional[float] = None
    tx_pos_y_mm: Optional[float] = None
    rx_pos_x_mm: Optional[float] = None
    rx_pos_y_mm: Optional[float] = None


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

    gps_time_1, gps_time_2 = struct.unpack(
        "ff", struct.pack("d", current_pos_heading_and_time["gps_time"])
    )
    # _z = struct.unpack("d", struct.pack("ff", a, b))[0]
    return np.hstack(
        [
            ds.timestamp,  # 1
            gps_time_1,  # 1
            gps_time_2,  # 1
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


def data_to_snapshot(
    current_time, signal_matrix, steering_vectors, rssis, gains, rx_config
):

    beam_sds = beamformer_given_steering(
        steering_vectors=steering_vectors, signal_matrix=signal_matrix
    )

    avg_phase_diff = get_avg_phase(signal_matrix)

    return DataSnapshot(
        timestamp=current_time,
        rx_center_pos=rx_config.rx_spacing,
        rx_theta_in_pis=rx_config.rx_theta_in_pis,
        rx_spacing=rx_config.rx_spacing,
        beam_sds=beam_sds,
        avg_phase_diff=avg_phase_diff,
        signal_matrix=None,
        rssis=rssis,
        gains=gains,
    )


class ThreadedRX:
    def __init__(self, pplus: PPlus, time_offset, nthetas, seconds_per_sample=0):
        self.pplus = pplus
        self.read_lock = threading.Lock()
        self.ready_lock = threading.Lock()
        self.ready_lock.acquire()
        self.run = False
        self.time_offset = time_offset
        self.nthetas = nthetas
        self.rx_config = self.pplus.rx_config
        self.seconds_per_sample = seconds_per_sample
        assert self.pplus.rx_config.rx_pos is not None

    def read_forever(self):
        logging.info(f"{str(self.rx_config.uri)} PPlus read_forever()")
        self.steering_vectors = precompute_steering_vectors(
            receiver_positions=self.rx_config.rx_pos,
            carrier_frequency=self.rx_config.lo,
            spacing=self.nthetas,
        )

        average_time_per_loop = -1
        alpha = 0.9
        idx = 0
        while self.run:
            start_time = time.time()
            if self.read_lock.acquire(blocking=True, timeout=0.5):
                # got the semaphore, read some data!
                self.get_data()
                try:
                    self.ready_lock.release()  # tell the parent we are ready to provide
                except Exception as e:
                    logging.error(f"Thread encountered an issue exiting {str(e)}")
                    self.run = False
                # logging.info(f"{self.pplus.rx_config.uri} READY")
            finish_time = time.time()
            elapsed_time = finish_time - start_time
            if idx > 20:  # skip first 20 for timing
                if average_time_per_loop < 0:
                    average_time_per_loop = elapsed_time
                else:
                    average_time_per_loop = (
                        average_time_per_loop * alpha + (1 - alpha) * elapsed_time
                    )
                if (
                    self.seconds_per_sample >= 0
                    and average_time_per_loop < self.seconds_per_sample
                ):
                    time.sleep(self.seconds_per_sample - average_time_per_loop)
            idx += 1

        logging.info(f"{str(self.rx_config.uri)} PPlus read_forever() exit!")

    def join(self):
        self.t.join()

    def start_read_thread(self):
        self.t = threading.Thread(target=self.read_forever, daemon=True)
        self.run = True
        self.t.start()

    def get_rx(self, max_retries=15) -> Dict[str, Any]:
        tries = 0
        while tries < max_retries:
            try:
                signal_matrix = self.pplus.sdr.rx()
                rssis = self.pplus.rssis()
                gains = self.pplus.gains()
                return {"signal_matrix": signal_matrix, "rssis": rssis, "gains": gains}
            except Exception as e:
                logging.error(
                    f"Failed to receive RX data! removing file : retry {tries} {e}",
                )
                time.sleep(0.1)
                tries += 1
                if tries > max_retries:
                    logging.error("GIVE UP")
                    sys.exit(1)
        return None

    def get_data(self):
        sdr_rx = self.get_rx()

        # process the data
        signal_matrix = np.vstack(sdr_rx["signal_matrix"])
        current_time = time.time() - self.time_offset  # timestamp

        self.data = data_to_snapshot(
            current_time=current_time,
            signal_matrix=signal_matrix,
            steering_vectors=self.steering_vectors,
            rssis=sdr_rx["rssis"],
            gains=sdr_rx["gains"],
            rx_config=self.pplus.rx_config,
        )


class ThreadedRXRaw(ThreadedRX):
    def get_data(self):
        self.data = None

        sdr_rx = self.get_rx()

        # process the data
        signal_matrix = np.vstack(sdr_rx["signal_matrix"])
        current_time = time.time() - self.time_offset  # timestamp

        avg_phase_diff = get_avg_phase(signal_matrix)

        self.data = self.snapshot_class(
            signal_matrix=signal_matrix,
            system_timestamp=current_time,
            rssis=sdr_rx["rssis"],
            gains=sdr_rx["gains"],
            rx_theta_in_pis=self.pplus.rx_config.rx_theta_in_pis,
            rx_spacing=self.pplus.rx_config.rx_spacing,
            avg_phase_diff=avg_phase_diff,
            rx_lo=self.pplus.rx_config.lo,
            rx_bandwidth=self.pplus.rx_config.rf_bandwidth,
        )


class ThreadedRXRawV4(ThreadedRXRaw):
    def __init__(self, **kwargs):
        self.snapshot_class = DataSnapshotV4
        super(ThreadedRXRawV4, self).__init__(
            **kwargs,
        )


class ThreadedRXRawV5(ThreadedRXRaw):
    def __init__(self, **kwargs):
        self.snapshot_class = DataSnapshotV5
        super(ThreadedRXRawV5, self).__init__(
            **kwargs,
        )


class DataCollector:
    def __init__(
        self,
        yaml_config,
        data_filename,
        position_controller,
        thread_class,
        tag="",
    ):
        self.yaml_config = yaml_config
        self.data_filename = data_filename
        #
        self.record_matrix = None
        self.position_controller = position_controller
        self.finished_collecting = False
        self.thread_class = thread_class
        self.setup_record_matrix()

    def radios_to_online(self):
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
                rx_config=target_rx_config,
                tx_config=target_tx_config,
                leave_tx_on=True,
            )
            pplus_rx.close_rx()

        # get radios online
        self.receiver_pplus = {}
        self.rx_configs = []
        for receiver in self.yaml_config["receivers"]:
            rx_config = rx_config_from_receiver_yaml(receiver)
            self.rx_configs.append(rx_config)
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
        self.prepare_threads()

    def prepare_threads(self):
        self.read_threads = []
        time_offset = time.time()
        for _, pplus_rx in self.receiver_pplus.items():
            if pplus_rx is None:
                continue
            seconds_per_sample = -1
            if "seconds-per-sample" in self.yaml_config:
                seconds_per_sample = self.yaml_config["seconds-per-sample"]
            read_thread = self.thread_class(
                pplus=pplus_rx,
                time_offset=time_offset,
                nthetas=self.yaml_config["n-thetas"],
                seconds_per_sample=seconds_per_sample,
            )
            read_thread.start_read_thread()
            self.read_threads.append(read_thread)

        self.collector_thread = threading.Thread(
            target=self.run_collector_thread, daemon=True
        )

    def start(self):
        self.collector_thread.start()

    def done(self):
        self.collector_thread.join()

    def is_collecting(self):
        return not self.finished_collecting

    def setup_record_matrix(self):
        raise NotImplementedError

    def write_to_record_matrix(self, thread_idx, record_idx, read_thread: ThreadedRX):
        raise NotImplementedError

    def run_collector_thread(self):
        for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):
            for read_thread_idx, read_thread in enumerate(self.read_threads):
                while not read_thread.ready_lock.acquire(timeout=0.5):
                    pass
                ###
                # copy the data out
                data = read_thread.data

                self.write_to_record_matrix(
                    read_thread_idx,
                    record_idx=record_index,
                    data=data,
                )

                read_thread.read_lock.release()
        logging.info("Collector thread is exiting!")
        self.finished_collecting = True

        self.close()

        # clean up lost threads
        for read_thread_idx, read_thread in enumerate(self.read_threads):
            read_thread.run = False
        for read_thread_idx, read_thread in enumerate(self.read_threads):
            read_thread.join()

    def close(self):
        pass


# V4 data format
class DroneDataCollectorRaw(DataCollector):
    def __init__(self, *args, **kwargs):
        super(DroneDataCollectorRaw, self).__init__(
            *args,
            thread_class=ThreadedRXRawV4,
            **kwargs,
        )

    def setup_record_matrix(self):
        # make sure all receivers are sharing a common buffer size
        buffer_size = None
        for receiver in self.yaml_config["receivers"]:
            assert "buffer-size" in receiver
            if buffer_size is None:
                buffer_size = receiver["buffer-size"]
            else:
                assert buffer_size == receiver["buffer-size"]
        # record matrix
        self.zarr = v4rx_new_dataset(
            filename=self.data_filename,
            timesteps=self.yaml_config["n-records-per-receiver"],
            buffer_size=buffer_size,
            n_receivers=len(self.yaml_config["receivers"]),
            chunk_size=512,
            compressor=None,
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        current_pos_heading_and_time = (
            self.position_controller.get_position_bearing_and_time()
        )
        data.heading = current_pos_heading_and_time["heading"]
        data.gps_long = current_pos_heading_and_time["gps"][0]
        data.gps_lat = current_pos_heading_and_time["gps"][1]

        z = self.zarr[f"receivers/r{thread_idx}"]
        z.signal_matrix[record_idx] = data.signal_matrix
        for k in v4rx_f64_keys + v4rx_2xf64_keys:
            z[k][record_idx] = getattr(data, k)  # getattr(data, k)

    def close(self):
        self.zarr.store.close()
        self.zarr = None
        zarr_shrink(self.data_filename)


# V3 data format
class DroneDataCollector(DataCollector):
    def __init__(self, *args, **kwargs):
        super(DroneDataCollector, self).__init__(
            *args,
            thread_class=ThreadedRX,
            **kwargs,
        )

    def setup_record_matrix(self):
        # record matrix
        self.record_matrix = np.memmap(
            self.data_filename,
            dtype="float32",
            mode="w+",
            shape=(
                2,  # TODO should be nreceivers
                self.yaml_config["n-records-per-receiver"],
                len(v3rx_column_names(nthetas=self.yaml_config["n-thetas"])),
            ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        current_pos_heading_and_time = (
            self.position_controller.get_position_bearing_and_time()
        )

        self.record_matrix[thread_idx, record_idx] = prepare_record_entry_v3(
            ds=data,
            current_pos_heading_and_time=current_pos_heading_and_time,
        )


# V5 data format
class GrblDataCollectorRaw(DataCollector):
    def __init__(self, *args, **kwargs):
        super(GrblDataCollectorRaw, self).__init__(
            *args,
            thread_class=ThreadedRXRawV5,
            **kwargs,
        )

    def setup_record_matrix(self):
        # make sure all receivers are sharing a common buffer size
        buffer_size = None
        for receiver in self.yaml_config["receivers"]:
            assert "buffer-size" in receiver
            if buffer_size is None:
                buffer_size = receiver["buffer-size"]
            else:
                assert buffer_size == receiver["buffer-size"]
        # record matrix
        self.zarr = v5rx_new_dataset(
            filename=self.data_filename,
            timesteps=self.yaml_config["n-records-per-receiver"],
            buffer_size=buffer_size,
            n_receivers=len(self.yaml_config["receivers"]),
            chunk_size=512,
            compressor=None,
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        tx_pos = self.position_controller.controller.position["xy"][
            self.yaml_config["emitter"]["motor_channel"]
        ]
        rx_pos = self.position_controller.controller.position["xy"][
            self.rx_configs[0].motor_channel
        ]

        data.tx_pos_x_mm = tx_pos[0]
        data.tx_pos_y_mm = tx_pos[1]
        data.rx_pos_x_mm = rx_pos[0]
        data.rx_pos_y_mm = rx_pos[1]

        z = self.zarr[f"receivers/r{thread_idx}"]
        z.signal_matrix[record_idx] = data.signal_matrix
        for k in v5rx_f64_keys + v5rx_2xf64_keys:
            z[k][record_idx] = getattr(data, k)

    def close(self):
        self.zarr.store.close()
        self.zarr = None
        zarr_shrink(self.data_filename)


# V2 data format
class GrblDataCollector(DataCollector):
    def __init__(self, *args, **kwargs):
        super(GrblDataCollector, self).__init__(
            *args,
            thread_class=ThreadedRX,
            **kwargs,
        )

    def setup_record_matrix(self):
        # record matrix
        self.record_matrix = np.memmap(
            self.data_filename,
            dtype="float32",
            mode="w+",
            shape=(
                2,  # TODO should be nreceivers
                self.yaml_config["n-records-per-receiver"],
                len(v2_column_names(nthetas=self.yaml_config["n-thetas"])),
            ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        tx_pos = self.position_controller.controller.position["xy"][
            self.yaml_config["emitter"]["motor_channel"]
        ]
        rx_pos = self.position_controller.controller.position["xy"][
            self.rx_configs[0].motor_channel
        ]

        self.record_matrix[thread_idx, record_idx] = prepare_record_entry_v2(
            ds=data, rx_pos=rx_pos, tx_pos=tx_pos
        )
