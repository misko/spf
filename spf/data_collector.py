import logging
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import struct
import sys
import threading
import time
from concurrent import futures
from typing import Any, Dict, Optional
import concurrent.futures
import numpy as np

# from attr import dataclass
from dataclasses import dataclass, asdict

from tqdm import tqdm

from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys, v4rx_new_dataset
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.dataset.v6_data import v6rx_2x_keys, v6rx_scalar_keys, v6rx_new_dataset
from spf.dataset.v7_data import v7rx_2x_keys, v7rx_scalar_keys, v7rx_new_dataset
from spf.dataset.wall_array_v2_idxs import v2_column_names
from spf.rf import (
    beamformer_given_steering,
    get_avg_phase,
    get_avg_phase_fast,
    get_avg_phase_fast2,
    precompute_steering_vectors,
)
from spf.scripts.zarr_utils import zarr_shrink
from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    PPlus,
    ReceiverConfig,
    SdrDeviceIdentity,
    get_pplus,
    rx_config_from_receiver_yaml,
    setup_rx,
    setup_rxtx,
)

SDR_IDENTITY_VERSION = 1


def _identity_zarr_attrs(
    identity: SdrDeviceIdentity,
    firmware_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = {
        "sdr_identity_version": SDR_IDENTITY_VERSION,
        "sdr_family": identity.sdr_family,
        "iio_uri_at_capture": identity.receiver_uri,
        "rx_transport": identity.rx_transport,
    }
    optional_attrs = {
        "sdr_serial": identity.serial,
        "usb_vendor_id": identity.usb_vendor_id,
        "usb_product_id": identity.usb_product_id,
        "usb_bus_at_capture": identity.usb_bus,
        "usb_address_at_capture": identity.usb_address,
        "usb_port_path": (
            list(identity.usb_port_path) if identity.usb_port_path is not None else None
        ),
    }
    attrs.update(
        {key: value for key, value in optional_attrs.items() if value is not None}
    )

    if identity.rx_transport == "direct_usb":
        direct_attrs = {
            # Preserve the original v6 attribute names while also exposing the
            # transport-independent identity above.
            "direct_usb_serial": identity.serial,
            "direct_usb_bus": identity.usb_bus,
            "direct_usb_port_path": (
                list(identity.usb_port_path)
                if identity.usb_port_path is not None
                else None
            ),
            "direct_usb_interface": identity.direct_usb_interface,
            "direct_usb_bulk_in_endpoint": identity.direct_usb_bulk_in_endpoint,
            "direct_usb_bulk_out_endpoint": identity.direct_usb_bulk_out_endpoint,
            "gain_metadata_protocol_version": (identity.direct_usb_protocol_version),
            "direct_usb_protocol_min": identity.direct_usb_protocol_min,
            "direct_usb_protocol_max": identity.direct_usb_protocol_max,
            "direct_usb_supported_features": identity.direct_usb_supported_features,
            "gain_metadata_capability_flags": (identity.direct_usb_capability_flags),
        }
        attrs.update(
            {key: value for key, value in direct_attrs.items() if value is not None}
        )
        if firmware_provenance is not None:
            attrs.update(firmware_provenance)
    return attrs


def _capture_firmware_provenance(
    yaml_config: dict[str, Any],
    identity: SdrDeviceIdentity,
) -> dict[str, Any] | None:
    if identity.rx_transport != "direct_usb":
        return None
    firmware = yaml_config.get("pluto-firmware")
    if not isinstance(firmware, dict):
        return None

    provenance = {
        "firmware_release_tag": firmware.get("release-tag"),
        "firmware_image_sha256": firmware.get("image-sha256"),
        "firmware_git_sha": firmware.get("firmware-git-sha"),
        "firmware_gadget_git_sha": firmware.get("gadget-git-sha"),
        "firmware_boot_mode": firmware.get("boot-mode"),
        "firmware_verified": False,
    }
    ready_path = Path(
        os.environ.get(
            "SPF_DIRECT_USB_READY_FILE",
            "/run/spf/direct_usb_ready.json",
        )
    )
    if ready_path.is_file() and identity.serial:
        from spf.scripts.pluto_ready_manifest import (
            fingerprint_for_serial,
            firmware_for_serial,
            load_manifest,
        )

        manifest = load_manifest(ready_path)
        actual = firmware_for_serial(manifest, identity.serial)
        if actual is not None:
            expected_matches = (
                actual.get("release_tag") == firmware.get("release-tag")
                and actual.get("image_sha256") == firmware.get("image-sha256")
                and actual.get("firmware_git_sha") == firmware.get("firmware-git-sha")
                and actual.get("gadget_git_sha") == firmware.get("gadget-git-sha")
                and actual.get("boot_mode") == firmware.get("boot-mode")
            )
            provenance["firmware_verified"] = bool(
                expected_matches and actual.get("firmware_verified")
            )
            provenance["firmware_ready_manifest_version"] = manifest.get(
                "ready_manifest_version"
            )
            fingerprint = fingerprint_for_serial(manifest, identity.serial)
            if fingerprint is not None:
                attachment = fingerprint.get("attachment", {})
                stable_identity = fingerprint.get("stable_identity", {})
                expected_port_path = (
                    ".".join(str(part) for part in identity.usb_port_path)
                    if identity.usb_port_path is not None
                    else None
                )
                identity_matches = (
                    stable_identity.get("pluto_serial") == identity.serial
                    and attachment.get("usb_bus") == identity.usb_bus
                    and attachment.get("usb_address") == identity.usb_address
                    and attachment.get("usb_port_path") == expected_port_path
                )
                if identity_matches and provenance["firmware_verified"]:
                    provenance["hardware_fingerprint_schema_version"] = fingerprint.get(
                        "schema_version"
                    )
                    provenance["hardware_fingerprint_v1"] = fingerprint
    return {key: value for key, value in provenance.items() if value is not None}


class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=50, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)


class ProcessPoolExecutorWithQueueSizeLimit:
    def __init__(self, max_workers=None, maxsize=0):
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        if maxsize > 0:
            self._semaphore = threading.BoundedSemaphore(maxsize)
        else:
            self._semaphore = None

    def submit(self, fn, *args, **kwargs):
        if self._semaphore:
            self._semaphore.acquire()
        future = self._executor.submit(fn, *args, **kwargs)
        if self._semaphore:
            future.add_done_callback(lambda f: self._semaphore.release())
        return future

    def shutdown(self, wait=True):
        self._executor.shutdown(wait=wait)

        # Add context manager methods:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


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
class DataSnapshotV6(DataSnapshotV4):
    gain_index_start: Optional[np.ndarray] = None
    gain_index_end: Optional[np.ndarray] = None
    gain_metadata_valid: bool = False
    gain_endpoints_equal: Optional[np.ndarray] = None
    gain_metadata_flags: int = 0
    stream_id: int = 0
    buffer_sequence: int = 0
    sample_sequence: int = 0
    gain_start_read_duration_ns: int = 0
    gain_end_read_duration_ns: int = 0
    first_gain_change_sample: Optional[np.ndarray] = None
    iq_power_dbfs: Optional[np.ndarray] = None


@dataclass
class DataSnapshotV7(DataSnapshotV6):
    gain_db_start: Optional[np.ndarray] = None
    gain_db_end: Optional[np.ndarray] = None
    rssi_db_start: Optional[np.ndarray] = None
    rssi_db_end: Optional[np.ndarray] = None
    rssi_metadata_valid: bool = False
    rssi_start_read_duration_ns: int = 0
    rssi_end_read_duration_ns: int = 0


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
        self.error = None
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
                self.error = e
                logging.exception("Failed to read data, aborting")
                self.run = False
                # The collector polls ``error`` directly. The sentinel is only
                # a wake-up hint: never block trying to enqueue it behind a
                # frame the collector intentionally stopped consuming.
                try:
                    self.read_q.put_nowait(None)
                except queue.Full:
                    pass
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
        if self.pplus.rx_config.rx_transport == "direct_usb":
            # A direct-USB retry creates a new stream generation and would
            # conceal the failed frame. Fail the collection instead.
            max_retries = 1
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        last_error = None
        for tries in range(max_retries):
            try:
                signal_matrix = self.pplus.rx()
                return {
                    "signal_matrix": signal_matrix,
                    "rssis": self.pplus.rssis(),
                    "gains": self.pplus.gains(),
                }
            except Exception as e:
                logging.error(
                    f"Failed to receive RX data! : retry {tries} {e}",
                )
                last_error = e
                if tries + 1 < max_retries:
                    time.sleep(0.1)
        assert last_error is not None
        raise last_error

    def get_data(self):
        sdr_rx = self.get_rx()
        if sdr_rx is None:
            raise ValueError("SDR RX is None, aborting.")
        # process the data
        signal_matrix = np.vstack(sdr_rx["signal_matrix"], dtype=np.complex64)
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
        signal_matrix = np.vstack(sdr_rx["signal_matrix"]).astype(np.complex64)
        current_time = time.time() - self.time_offset  # timestamp after sample arrives

        avg_phase_diff = get_avg_phase_fast2(signal_matrix)
        assert self.pplus.rx_config.rx_spacing > 0.001
        snapshot_kwargs = dict(
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
        if issubclass(self.snapshot_class, DataSnapshotV6):
            snapshot_kwargs.update(
                gain_index_start=sdr_rx["gain_index_start"],
                gain_index_end=sdr_rx["gain_index_end"],
                gain_metadata_valid=sdr_rx["gain_metadata_valid"],
                gain_endpoints_equal=sdr_rx["gain_endpoints_equal"],
                gain_metadata_flags=sdr_rx["gain_metadata_flags"],
                stream_id=sdr_rx["stream_id"],
                buffer_sequence=sdr_rx["buffer_sequence"],
                sample_sequence=sdr_rx["sample_sequence"],
                gain_start_read_duration_ns=sdr_rx["gain_start_read_duration_ns"],
                gain_end_read_duration_ns=sdr_rx["gain_end_read_duration_ns"],
                first_gain_change_sample=sdr_rx["first_gain_change_sample"],
                iq_power_dbfs=sdr_rx["iq_power_dbfs"],
            )
        if issubclass(self.snapshot_class, DataSnapshotV7):
            snapshot_kwargs.update(
                gain_db_start=sdr_rx["gain_db_start"],
                gain_db_end=sdr_rx["gain_db_end"],
                rssi_db_start=sdr_rx["rssi_db_start"],
                rssi_db_end=sdr_rx["rssi_db_end"],
                rssi_metadata_valid=sdr_rx["rssi_metadata_valid"],
                rssi_start_read_duration_ns=sdr_rx["rssi_start_read_duration_ns"],
                rssi_end_read_duration_ns=sdr_rx["rssi_end_read_duration_ns"],
            )
        return self.snapshot_class(**snapshot_kwargs)


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


class ThreadedRXRawV6(ThreadedRXRaw):
    def __init__(self, **kwargs):
        self.snapshot_class = DataSnapshotV6
        super(ThreadedRXRawV6, self).__init__(**kwargs)

    def get_rx(self, max_retries=15) -> Dict[str, Any]:
        """Retain the explicit v1 metadata schema for v6 datasets."""

        if self.pplus.rx_config.rx_transport == "direct_usb":
            max_retries = 1
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        last_error = None
        for tries in range(max_retries):
            try:
                return asdict(self.pplus.rx_with_metadata())
            except Exception as e:
                logging.error("Failed to receive RX data! : retry %s %s", tries, e)
                last_error = e
                if tries + 1 < max_retries:
                    time.sleep(0.1)
        assert last_error is not None
        raise last_error


class ThreadedRXRawV7(ThreadedRXRawV6):
    def __init__(self, **kwargs):
        self.snapshot_class = DataSnapshotV7
        ThreadedRXRaw.__init__(self, **kwargs)


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
        self.collection_error = None
        self.cleanup_errors = []
        self.records_written_by_receiver = [0] * len(yaml_config["receivers"])
        self._records_written_lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.setup_record_matrix()
        self._mark_capture_state("in_progress")

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
        self._record_receiver_identities()
        self.prepare_threads()

    def _record_receiver_identities(self):
        pplus_receivers = list(self.receiver_pplus.values())
        expected_receivers = len(self.yaml_config["receivers"])
        if len(pplus_receivers) != expected_receivers:
            raise RuntimeError(
                "receiver identity is ambiguous: "
                f"{expected_receivers} configured receivers resolved to "
                f"{len(pplus_receivers)} radio objects"
            )

        identities = [
            pplus_receiver.receiver_identity() for pplus_receiver in pplus_receivers
        ]
        pluto_identities = [
            identity for identity in identities if identity.sdr_family == "pluto"
        ]
        pluto_serials = [identity.serial for identity in pluto_identities]
        if any(serial is None or not serial for serial in pluto_serials):
            raise RuntimeError("every Pluto receiver must expose a non-empty serial")
        if len(pluto_serials) != len(set(pluto_serials)):
            raise RuntimeError("multiple receiver entries resolved to one Pluto serial")

        local_pluto_paths = [
            identity.usb_port_path
            for identity in pluto_identities
            if identity.usb_port_path is not None
        ]
        if len(local_pluto_paths) != len(set(local_pluto_paths)):
            raise RuntimeError(
                "multiple receiver entries resolved to one Pluto USB physical path"
            )

        self.receiver_identities = identities
        if self.data_filename is None:
            return

        self.zarr.attrs["sdr_identity_version"] = SDR_IDENTITY_VERSION
        for receiver_idx, identity in enumerate(identities):
            receiver_z = self.zarr[f"receivers/r{receiver_idx}"]
            receiver_z.attrs.update(
                _identity_zarr_attrs(
                    identity,
                    _capture_firmware_provenance(self.yaml_config, identity),
                )
            )
            if self.yaml_config.get("data-version") == 7:
                if receiver_z.attrs.get("firmware_verified") is not True:
                    raise RuntimeError(
                        f"{identity.serial}: V7 capture requires boot-verified firmware"
                    )
                if not isinstance(
                    receiver_z.attrs.get("hardware_fingerprint_v1"), dict
                ):
                    raise RuntimeError(
                        f"{identity.serial}: V7 capture requires a matching "
                        "post-firmware hardware fingerprint"
                    )

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
        if self.collection_error is not None:
            raise self.collection_error

    def request_stop(self, error):
        """Request cooperative cleanup without replacing an earlier failure."""

        if self.collection_error is None:
            latent_read_error = next(
                (
                    read_thread.error
                    for read_thread in getattr(self, "read_threads", ())
                    if read_thread.error is not None
                ),
                None,
            )
            self.collection_error = latent_read_error or error
        self.stop_requested.set()
        for read_thread in getattr(self, "read_threads", ()):
            read_thread.run = False

    def is_collecting(self):
        return not self.finished_collecting

    def setup_record_matrix(self):
        raise NotImplementedError

    def write_to_record_matrix(self, thread_idx, record_idx, read_thread: ThreadedRX):
        raise NotImplementedError

    def _write_record_and_track(self, thread_idx, record_idx, data):
        self.write_to_record_matrix(thread_idx, record_idx, data)
        with self._records_written_lock:
            self.records_written_by_receiver[thread_idx] += 1
            # Commit progress only after the full record. An abrupt death may
            # under-count the record currently being committed, but it can
            # never claim that an unwritten record is safe to consume.
            zarr = getattr(self, "zarr", None)
            if self.data_filename is not None and zarr is not None:
                zarr.attrs["capture_records_written_by_receiver"] = list(
                    self.records_written_by_receiver
                )

    @staticmethod
    def _error_summary(error):
        return f"{type(error).__name__}: {error}"

    def _mark_capture_state(self, status):
        """Persist lifecycle state while the temporary Zarr is still writable."""

        zarr = getattr(self, "zarr", None)
        if self.data_filename is None or zarr is None:
            return
        zarr.attrs["capture_status"] = status
        zarr.attrs["capture_records_written_by_receiver"] = list(
            self.records_written_by_receiver
        )
        if self.collection_error is not None:
            zarr.attrs["capture_error_type"] = type(self.collection_error).__name__
            zarr.attrs["capture_error_message"] = str(self.collection_error)
            error_number = getattr(self.collection_error, "errno", None)
            if error_number is not None:
                zarr.attrs["capture_error_errno"] = int(error_number)
        if self.cleanup_errors:
            zarr.attrs["capture_cleanup_errors"] = [
                self._error_summary(error) for error in self.cleanup_errors
            ]

    def run_inner_collector_thread(self):
        futures = []
        with ThreadPoolExecutorWithQueueSizeLimit(max_workers=2, maxsize=1) as executor:
            for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):
                for read_thread_idx, read_thread in enumerate(self.read_threads):
                    while True:
                        if read_thread.error is not None:
                            raise read_thread.error
                        if self.stop_requested.is_set():
                            assert self.collection_error is not None
                            raise self.collection_error
                        try:
                            data = read_thread.read_q.get(timeout=0.5)
                            break
                        except queue.Empty:
                            if read_thread.error is not None:
                                raise read_thread.error
                    if data is None:
                        if read_thread.error is not None:
                            raise read_thread.error
                        raise RuntimeError(
                            f"receiver {read_thread_idx} stopped without an error"
                        )
                    futures.append(
                        executor.submit(
                            self._write_record_and_track,
                            read_thread_idx,
                            record_idx=record_index,
                            data=data,
                        )
                    )
                    while len(futures) > 4:
                        futures.pop(0).result()
            for future in futures:
                future.result()
        return

    def run_collector_thread(self):
        logging.info("Collector thread is running!")
        try:
            # https://stackoverflow.com/questions/48263704/threadpoolexecutor-how-to-limit-the-queue-maxsize
            self.run_inner_collector_thread()
        except Exception as error:
            if self.collection_error is None:
                self.collection_error = error
            logging.exception("Collector failed")
        finally:
            logging.info("Collector tell threads to quit!")
            for read_thread in self.read_threads:
                read_thread.run = False

            logging.info("Collector wait for join!")
            for read_thread in self.read_threads:
                try:
                    read_thread.join()
                except Exception as error:
                    self.cleanup_errors.append(error)
                    logging.exception("Receiver thread cleanup failed")

            for uri, pplus in self.receiver_pplus.items():
                try:
                    pplus.close()
                except Exception as error:
                    self.cleanup_errors.append(error)
                    logging.exception("%s: radio cleanup/TX mute failed", uri)

            if self.collection_error is None and self.cleanup_errors:
                self.collection_error = self.cleanup_errors[0]

            self._mark_capture_state(
                "complete" if self.collection_error is None else "incomplete"
            )
            try:
                self.close()
            except Exception as error:
                self.cleanup_errors.append(error)
                logging.exception("Capture store finalization failed")
                if self.collection_error is None:
                    self.collection_error = error

            self.finished_collecting = True
            if self.collection_error is None:
                logging.info("Collector clean exit!")
            else:
                logging.error(
                    "Collector exit with primary error: %s",
                    self._error_summary(self.collection_error),
                )

    def close(self):
        pass


class CaptureInterrupted(RuntimeError):
    """A collection was stopped by a process-control signal."""

    def __init__(self, signal_number):
        self.signal_number = int(signal_number)
        try:
            signal_name = signal.Signals(self.signal_number).name
        except ValueError:
            signal_name = str(self.signal_number)
        super().__init__(f"capture interrupted by {signal_name}")


class capture_signal_handlers:
    """Route SIGINT/SIGTERM through collector and Zarr cleanup."""

    def __init__(self, collector):
        self.collector = collector
        self.previous = {}

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("capture signal handlers require the main thread")

        def request_stop(signal_number, _frame):
            logging.error(
                "Capture interruption requested by %s",
                signal.Signals(signal_number).name,
            )
            self.collector.request_stop(CaptureInterrupted(signal_number))

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            self.previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_stop)
        return self

    def __exit__(self, _error_type, _error, _traceback):
        for signal_number, previous_handler in self.previous.items():
            signal.signal(signal_number, previous_handler)
        self.previous.clear()


# V4 data format
class DroneDataCollectorRaw(DataCollector):
    def __init__(self, realtime_v5inf, *args, **kwargs):
        super(DroneDataCollectorRaw, self).__init__(
            *args,
            thread_class=ThreadedRXRawV4,
            **kwargs,
        )
        self.realtime_v5inf = realtime_v5inf

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
        # rx_heading_in_pis is a declared dataclass field (asdict() only carries
        # declared fields, so setting only .heading would silently drop it from
        # the realtime path); heading is degrees, in_pis = deg/180.
        data.rx_heading_in_pis = current_pos_heading_and_time["heading"] / 180.0
        data.gps_long = current_pos_heading_and_time["gps"][0]
        data.gps_lat = current_pos_heading_and_time["gps"][1]
        data.gps_timestamp = current_pos_heading_and_time["gps_time"]
        if self.realtime_v5inf is not None:
            data_dict = asdict(data)
            data_dict["signal_matrix"] = (
                data_dict["signal_matrix"]
                .reshape(1, 1, *data_dict["signal_matrix"].shape)
                .astype(np.complex64)
            )
            self.realtime_v5inf.write_to_idx(record_idx, thread_idx, data_dict)
        if self.data_filename is not None:
            z = self.zarr[f"receivers/r{thread_idx}"]
            z.signal_matrix[record_idx] = data.signal_matrix
            for k in v4rx_f64_keys + v4rx_2xf64_keys:
                z[k][record_idx] = getattr(data, k)  # getattr(data, k)

    def close(self):
        if self.data_filename is not None:
            zarr = getattr(self, "zarr", None)
            if zarr is None:
                return
            self.zarr = None
            zarr.store.close()
            logging.info(f"Trying to shrink... {self.data_filename}")
            zarr_shrink(self.data_filename)


class DroneDataCollectorRawV6(DroneDataCollectorRaw):
    """Rover collector preserving v4 fields plus direct-USB metadata."""

    def __init__(self, realtime_v5inf, *args, **kwargs):
        DataCollector.__init__(
            self,
            *args,
            thread_class=ThreadedRXRawV6,
            **kwargs,
        )
        self.realtime_v5inf = realtime_v5inf

    def setup_record_matrix(self):
        if self.data_filename is None:
            return
        buffer_sizes = {
            receiver["buffer-size"] for receiver in self.yaml_config["receivers"]
        }
        if len(buffer_sizes) != 1:
            raise ValueError("all receivers must use one buffer size")
        self.zarr = v6rx_new_dataset(
            filename=self.data_filename,
            timesteps=self.yaml_config["n-records-per-receiver"],
            buffer_size=buffer_sizes.pop(),
            n_receivers=len(self.yaml_config["receivers"]),
            chunk_size=512,
            compressor=None,
            config=self.yaml_config,
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        super().write_to_record_matrix(thread_idx, record_idx, data)
        if self.data_filename is None:
            return
        receiver_z = self.zarr[f"receivers/r{thread_idx}"]
        for key in v6rx_scalar_keys:
            receiver_z[key][record_idx] = getattr(data, key)
        for key in v6rx_2x_keys:
            receiver_z[key][record_idx] = getattr(data, key)


class DroneDataCollectorRawV7(DroneDataCollectorRaw):
    """Rover collector for protocol v2 gain/RSSI and stream metadata."""

    def __init__(self, realtime_v5inf, *args, **kwargs):
        DataCollector.__init__(
            self,
            *args,
            thread_class=ThreadedRXRawV7,
            **kwargs,
        )
        self.realtime_v5inf = realtime_v5inf

    def setup_record_matrix(self):
        if self.data_filename is None:
            return
        buffer_sizes = {
            receiver["buffer-size"] for receiver in self.yaml_config["receivers"]
        }
        if len(buffer_sizes) != 1:
            raise ValueError("all receivers must use one buffer size")
        self.zarr = v7rx_new_dataset(
            filename=self.data_filename,
            timesteps=self.yaml_config["n-records-per-receiver"],
            buffer_size=buffer_sizes.pop(),
            n_receivers=len(self.yaml_config["receivers"]),
            chunk_size=512,
            compressor=None,
            config=self.yaml_config,
        )

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        super().write_to_record_matrix(thread_idx, record_idx, data)
        if self.data_filename is None:
            return
        receiver_z = self.zarr[f"receivers/r{thread_idx}"]
        for key in v7rx_scalar_keys:
            receiver_z[key][record_idx] = getattr(data, key)
        for key in v7rx_2x_keys:
            receiver_z[key][record_idx] = getattr(data, key)


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
