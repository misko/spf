import logging
import multiprocessing
import queue
import struct
import sys
import threading
import time
import traceback
from concurrent import futures
from typing import Any, Dict, Optional

import numpy as np
#from attr import dataclass
from dataclasses import dataclass, asdict

from tqdm import tqdm

from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys, v4rx_new_dataset
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.dataset.wall_array_v2_idxs import v2_column_names
from spf.rf import beamformer_given_steering, get_avg_phase, precompute_steering_vectors
from spf.scripts.zarr_utils import zarr_shrink
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    PPlus,
    ReceiverConfig,
    get_pplus,
    rx_config_from_receiver_yaml,
    setup_rx,
    setup_rxtx,
)


class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=50, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)


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
    rx_heading_in_pis: float = 0.0  # wall array did not originally have this


@dataclass
class DataSnapshotV4(DataSnapshotRaw):
    gps_timestamp: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_long: Optional[float] = None


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
        self.read_q = queue.Queue(maxsize=1)
        # self.read_q = multiprocessing.Queue(maxsize=1)
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
            try:
                data = self.get_data()
            except Exception as e:
                logging.error(f"Failed to read data , aborting {e}")
                self.read_q.put(None, timeout=5)
                self.run = False
                continue
            put_on_queue = False
            while self.run and not put_on_queue:
                try:
                    self.read_q.put(data, timeout=0.5)
                    put_on_queue = True
                except queue.Full:
                    pass
            finish_time = time.time()
            elapsed_time = finish_time - start_time
            if idx % 100 == 0:
                self.pplus.soft_reset_radio()  # try to calibrate
            elif idx > 20:  # skip first 20 for timing
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

    def start_read_thread(self, thread=True):
        if thread:
            self.t = threading.Thread(target=self.read_forever, daemon=True)
            self.run = True
            self.t.start()
        else:
            self.t = multiprocessing.Process(target=self.read_forever, daemon=True)
            self.run = True
            self.t.start()

    def get_rx(self, max_retries=15) -> Dict[str, Any]:
        tries = 0
        while tries < max_retries:
            try:
                signal_matrix = self.pplus.sdr.rx() # complex128 for pluto, eventhough its 12bit TODO
                rssis = self.pplus.rssis()
                gains = self.pplus.gains()
                return {"signal_matrix": signal_matrix, "rssis": rssis, "gains": gains}
            except Exception as e:
                logging.error(
                    f"Failed to receive RX data! : retry {tries} {e}",
                )
                time.sleep(0.1)
                tries += 1
                if tries > max_retries:
                    logging.error("GIVE UP")
                    sys.exit(1)
        return None

    def get_data(self):
        sdr_rx = self.get_rx()
        if sdr_rx is None:
            raise ValueError("SDR RX is None, aborting.")
        # process the data
        signal_matrix = np.vstack(sdr_rx["signal_matrix"])
        current_time = time.time() - self.time_offset  # timestamp

        return data_to_snapshot(
            current_time=current_time,
            signal_matrix=signal_matrix,
            steering_vectors=self.steering_vectors,
            rssis=sdr_rx["rssis"],
            gains=sdr_rx["gains"],
            rx_config=self.pplus.rx_config,
        )


class ThreadedRXRaw(ThreadedRX):
    def get_data(self):

        sdr_rx = self.get_rx()

        # process the data
        signal_matrix = np.vstack(sdr_rx["signal_matrix"])
        current_time = time.time() - self.time_offset  # timestamp after sample arrives

        avg_phase_diff = get_avg_phase(signal_matrix)
        assert self.pplus.rx_config.rx_spacing > 0.001
        return self.snapshot_class(
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

    def run_inner_collector_thread(self):
        futures = []
        with ThreadPoolExecutorWithQueueSizeLimit(
            max_workers=1, maxsize=1
        ) as executor:
            for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):
                for read_thread_idx, read_thread in enumerate(self.read_threads):
                    data = read_thread.read_q.get()
                    if data is None:
                        return
                    futures.append(
                        executor.submit(
                            self.write_to_record_matrix,
                            read_thread_idx,
                            record_idx=record_index,
                            data=data,
                        )
                    )
                    while len(futures) > 4:
                        future_exception = futures.pop(0).exception()
                        if future_exception:
                            logging.error(
                                "".join(
                                    traceback.format_exception(
                                        type(future_exception),
                                        future_exception,
                                        future_exception.__traceback__,
                                    )
                                )
                            )
        return

    def run_collector_thread(self):
        logging.info("Collector thread is running!")
        # https://stackoverflow.com/questions/48263704/threadpoolexecutor-how-to-limit-the-queue-maxsize
        self.run_inner_collector_thread()
        # read_thread.read_q.shutdown() # py 3.13
        logging.info("Collector thread is exiting!")
        self.finished_collecting = True

        self.close()

        # clean up lost threads
        logging.info("Collector tell threads to quit!")
        for read_thread_idx, read_thread in enumerate(self.read_threads):
            read_thread.run = False
        logging.info("Collector wait for join!")
        for read_thread_idx, read_thread in enumerate(self.read_threads):
            read_thread.join()
        logging.info("Collector clean exit!")

    def close(self):
        pass


# V4 data format
class DroneDataCollectorRaw(DataCollector):
    def __init__(self, realtime_v5inf, *args, **kwargs):
        super(DroneDataCollectorRaw, self).__init__(
            *args,
            thread_class=ThreadedRXRawV4,
            **kwargs,
        )
        self.realtime_v5inf=realtime_v5inf

    def setup_record_matrix(self):
        if self.data_filename is not None:
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
                config=self.yaml_config,
            )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        current_pos_heading_and_time = (
            self.position_controller.get_position_bearing_and_time()
        )
        data.heading = current_pos_heading_and_time["heading"]
        data.gps_long = current_pos_heading_and_time["gps"][0]
        data.gps_lat = current_pos_heading_and_time["gps"][1]
        data.gps_timestamp = current_pos_heading_and_time["gps_time"]

        if self.realtime_v5inf is not None:
            data_dict=asdict(data)
            data_dict['signal_matrix']=data_dict['signal_matrix'].reshape(1,1,*data_dict['signal_matrix'].shape)
            self.realtime_v5inf.write_to_idx(record_idx, thread_idx, data_dict)
        if self.data_filename is not None:
            z = self.zarr[f"receivers/r{thread_idx}"]
            z.signal_matrix[record_idx] = data.signal_matrix
            for k in v4rx_f64_keys + v4rx_2xf64_keys:
                z[k][record_idx] = getattr(data, k)  # getattr(data, k)


    def close(self):
        self.zarr.store.close()
        self.zarr = None
        logging.info(f"Trying to shrink... {self.data_filename}")
        zarr_shrink(self.data_filename)


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
        if not self.yaml_config["dry-run"]:
            # record matrix
            self.zarr = v5rx_new_dataset(
                filename=self.data_filename,
                timesteps=self.yaml_config["n-records-per-receiver"],
                buffer_size=buffer_size,
                n_receivers=len(self.yaml_config["receivers"]),
                chunk_size=512,
                compressor=None,
                config=self.yaml_config,
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
        assert data.rx_lo > 1

        if not self.yaml_config["dry-run"]:
            z = self.zarr[f"receivers/r{thread_idx}"]
            z.signal_matrix[record_idx] = data.signal_matrix
            for k in v5rx_f64_keys + v5rx_2xf64_keys:
                z[k][record_idx] = getattr(data, k)

    def close(self):
        if not self.yaml_config["dry-run"]:
            logging.info(f"Trying to close LMDB... {self.data_filename}")
            self.zarr.store.close()
            logging.info(f"Trying to close 2 LMDB... {self.data_filename}")
            self.zarr = None
            logging.info(f"Trying to shrink... {self.data_filename}")
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
