"""V7-compatible Zarr storage for gain-by-frequency phase calibration."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    ScheduleEntry,
)
from spf.data_collector import (
    SDR_IDENTITY_VERSION,
    _capture_firmware_provenance,
    _identity_zarr_attrs,
)
from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys
from spf.dataset.v7_data import (
    V7_GAIN_EVENT_CAPACITY,
    V7_GAIN_OBSERVATION_CAPACITY,
    v7rx_2x_keys,
    v7rx_gain_series_scalar_keys,
    v7rx_sample_time_scalar_keys,
    v7rx_scalar_keys,
    v7rx_new_dataset,
)
from spf.rf import get_avg_phase_fast2
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sdr_controller import PlutoRxBuffer, SdrDeviceIdentity


CALIBRATION_SCHEMA = "spf.calibration.dual_rx_gain_frequency"
CALIBRATION_SCHEMA_VERSION = 1
PHASE_CONVENTION = "angle(rx1) - angle(rx2)"

QUALITY_REASON_BITS = {
    "rx1_tone_snr_low": 1 << 0,
    "rx2_tone_snr_low": 1 << 1,
    "rx1_tone_too_weak": 1 << 2,
    "rx2_tone_too_weak": 1 << 3,
    "rx1_tone_too_strong": 1 << 4,
    "rx2_tone_too_strong": 1 << 5,
    "rx1_clipping": 1 << 6,
    "rx2_clipping": 1 << 7,
    "cross_channel_coherence_low": 1 << 8,
    "within_capture_phase_unstable": 1 << 9,
    "tone_peak_at_search_edge": 1 << 10,
}

SWEEP_SCALAR_FIELDS = {
    "sweep_completed": np.bool_,
    "sweep_epoch": np.uint16,
    "sweep_frequency_index": np.uint16,
    "sweep_schedule_index": np.uint64,
    "sweep_attempts": np.uint16,
    "sweep_quality_valid": np.bool_,
    "sweep_quality_reason_mask": np.uint32,
    "sweep_lo_frequency_hz": np.uint64,
    "sweep_rf_tone_frequency_hz": np.float64,
    "sweep_tx_gain_db": np.float32,
    "sweep_tx_digital_amplitude": np.float32,
    "tone_frequency_hz": np.float64,
    "tone_frequency_error_hz": np.float64,
    "phase_difference_rad": np.float64,
    "within_capture_phase_std_rad": np.float64,
    "within_capture_resultant_length": np.float32,
    "coherence": np.float32,
    "amplitude_ratio_db_rx1_over_rx2": np.float32,
}

SWEEP_2X_FIELDS = {
    "sweep_requested_gain_db": np.int16,
    "tone_dbfs": np.float32,
    "tone_snr_db": np.float32,
    "dc_dbfs": np.float32,
    "clipping_fraction": np.float32,
}


def quality_reason_mask(reasons: Iterable[str]) -> int:
    mask = 0
    for reason in reasons:
        try:
            mask |= QUALITY_REASON_BITS[reason]
        except KeyError as error:
            raise ValueError(f"unknown tone quality reason: {reason}") from error
    return mask


def _create_calibration_arrays(receiver, timesteps: int) -> None:
    for key, dtype in SWEEP_SCALAR_FIELDS.items():
        receiver.create_dataset(
            key,
            shape=(timesteps,),
            chunks=(timesteps,),
            dtype=dtype,
            compressor=None,
        )
    for key, dtype in SWEEP_2X_FIELDS.items():
        receiver.create_dataset(
            key,
            shape=(timesteps, 2),
            chunks=(timesteps, 2),
            dtype=dtype,
            compressor=None,
        )


def _run_signature(
    config: CalibrationConfig,
    identity: SdrDeviceIdentity,
    yaml_config: dict[str, Any],
) -> str:
    payload = {
        "calibration": config.as_json(),
        "serial": identity.serial,
        "usb_port_path": identity.usb_port_path,
        "firmware": yaml_config.get("pluto-firmware"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _software_provenance() -> dict[str, Any]:
    """Capture the source revision used to write the calibration dataset."""

    repository = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = True
    return {
        "calibration_software_git_sha": sha,
        "calibration_software_git_dirty": dirty,
    }


class CalibrationV7Writer:
    """One serial-specific, resumable V7 calibration dataset."""

    def __init__(
        self,
        path: Path,
        *,
        config: CalibrationConfig,
        schedule: list[ScheduleEntry],
        identity: SdrDeviceIdentity,
        yaml_config: dict[str, Any],
        resume: bool = True,
        rf_dc_tracking_observed: dict[str, str | None] | None = None,
    ):
        self.path = Path(path)
        self.config = config
        self.schedule = schedule
        self.identity = identity
        self.yaml_config = yaml_config
        self.rf_dc_tracking_observed = rf_dc_tracking_observed
        self.signature = _run_signature(config, identity, yaml_config)
        data_path = self.path / "data.mdb"
        if data_path.exists():
            if not resume:
                raise FileExistsError(f"calibration dataset exists: {self.path}")
            # A resumed exhaustive capture must retain the same LMDB write
            # mode as initial creation. The generic synchronous ``rw`` mode
            # makes every per-frame Zarr field assignment durable separately
            # and is several times slower than the normal capture path.
            self.zarr = zarr_open_from_lmdb_store(
                str(self.path), mode="rw", map_async=True
            )
            actual = self.zarr.attrs.get("calibration_run_signature")
            if actual != self.signature:
                self.zarr.store.close()
                raise ValueError(
                    "existing calibration dataset has a different run signature"
                )
            self.receiver = self.zarr["receivers/r0"]
            self._validate_existing_shape()
            return

        self.path.mkdir(parents=True, exist_ok=True)
        config_document = dict(yaml_config)
        config_document["data-version"] = 7
        config_document["calibration"] = {
            "schema": CALIBRATION_SCHEMA,
            "schema-version": CALIBRATION_SCHEMA_VERSION,
            **config.as_json(),
        }
        self.zarr = v7rx_new_dataset(
            filename=str(self.path),
            timesteps=len(schedule),
            buffer_size=config.buffer_size,
            n_receivers=1,
            config=config_document,
            chunk_size=1,
            compressor=None,
        )
        self.zarr.attrs.update(
            {
                "sdr_identity_version": SDR_IDENTITY_VERSION,
                "calibration_schema": CALIBRATION_SCHEMA,
                "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
                "calibration_run_signature": self.signature,
                "lmdb_write_policy": "map_async_block_sync",
                "phase_convention": PHASE_CONVENTION,
                # Request and confirmation are separate fields, as with
                # ``firmware_verified``: a claim that the RF-DC tracking loop
                # was pinned is only worth as much as the chip's readback.
                "rf_dc_calibration_policy": self.config.rf_dc_calibration_policy,
                "rf_dc_offset_tracking_en_requested": (
                    "unset"
                    if self.config.rf_dc_offset_tracking_en is None
                    else str(int(self.config.rf_dc_offset_tracking_en))
                ),
                "rf_dc_offset_tracking_en_observed": json.dumps(
                    self.rf_dc_tracking_observed, sort_keys=True
                ),
                **_software_provenance(),
            }
        )
        self.receiver = self.zarr["receivers/r0"]
        provenance = _capture_firmware_provenance(yaml_config, identity)
        self.receiver.attrs.update(_identity_zarr_attrs(identity, provenance))
        if identity.rx_transport == "iio":
            self.receiver.attrs.update(
                {
                    "gain_metadata_protocol_version": 3,
                    "iio_metadata_buffer_enabled": True,
                    "iio_backend": identity.receiver_uri.split(":", 1)[0],
                }
            )
        if self.receiver.attrs.get("firmware_verified") is not True:
            self.zarr.store.close()
            raise RuntimeError(
                f"{identity.serial}: V7 calibration requires boot-verified firmware"
            )
        if not isinstance(self.receiver.attrs.get("hardware_fingerprint_v1"), dict):
            self.zarr.store.close()
            raise RuntimeError(
                f"{identity.serial}: V7 calibration requires a matching "
                "post-firmware hardware fingerprint"
            )
        _create_calibration_arrays(self.receiver, len(schedule))
        self._initialize_schedule()

    def _validate_existing_shape(self) -> None:
        expected = (len(self.schedule), 2, self.config.buffer_size)
        if self.receiver.signal_matrix.shape != expected:
            raise ValueError(
                f"existing signal shape {self.receiver.signal_matrix.shape} "
                f"does not match {expected}"
            )
        for name in SWEEP_SCALAR_FIELDS:
            if name not in self.receiver:
                raise ValueError(f"existing dataset is missing {name}")

    def _initialize_schedule(self) -> None:
        receiver = self.receiver
        # These arrays are each one full-length chunk. Populate them in bulk;
        # element-wise Zarr assignment turns an exhaustive sweep into hundreds
        # of thousands of LMDB transactions before RF capture can even start.
        receiver.sweep_epoch[:] = np.fromiter(
            (entry.epoch for entry in self.schedule),
            dtype=np.uint16,
            count=len(self.schedule),
        )
        receiver.sweep_frequency_index[:] = np.fromiter(
            (entry.frequency_index for entry in self.schedule),
            dtype=np.uint16,
            count=len(self.schedule),
        )
        receiver.sweep_schedule_index[:] = np.arange(
            len(self.schedule), dtype=np.uint64
        )
        lo_frequency = np.fromiter(
            (entry.lo_frequency_hz for entry in self.schedule),
            dtype=np.uint64,
            count=len(self.schedule),
        )
        receiver.sweep_lo_frequency_hz[:] = lo_frequency
        receiver.sweep_rf_tone_frequency_hz[:] = (
            lo_frequency.astype(np.float64) + self.config.tone_offset_hz
        )
        receiver.sweep_requested_gain_db[:] = np.asarray(
            [(entry.gain_rx1_db, entry.gain_rx2_db) for entry in self.schedule],
            dtype=np.int16,
        )
        receiver.sweep_tx_gain_db[:] = np.fromiter(
            (
                self.config.tx_gain_for(entry.gain_rx1_db, entry.gain_rx2_db)
                for entry in self.schedule
            ),
            dtype=np.float32,
            count=len(self.schedule),
        )
        receiver.sweep_tx_digital_amplitude[:] = self.config.tx_digital_amplitude

    def is_complete(self, entry: ScheduleEntry) -> bool:
        return bool(self.receiver.sweep_completed[entry.record_index])

    def increment_attempts(self, entry: ScheduleEntry) -> int:
        index = entry.record_index
        attempts = int(self.receiver.sweep_attempts[index]) + 1
        self.receiver.sweep_attempts[index] = attempts
        return attempts

    def write(
        self,
        entry: ScheduleEntry,
        frame: PlutoRxBuffer,
        analysis: dict[str, Any],
        *,
        system_timestamp: float,
        tx_gain_db: float | None = None,
    ) -> None:
        index = entry.record_index
        signal = np.asarray(frame.signal_matrix, dtype=np.complex64)
        if signal.shape != (2, self.config.buffer_size):
            raise ValueError(f"unexpected signal shape: {signal.shape}")
        receiver = self.receiver
        expected_tx_gain = self.config.tx_gain_for(entry.gain_rx1_db, entry.gain_rx2_db)
        if tx_gain_db is None:
            tx_gain_db = expected_tx_gain
        if not math.isclose(tx_gain_db, expected_tx_gain, abs_tol=1e-6):
            raise ValueError(
                f"TX gain {tx_gain_db} does not match policy {expected_tx_gain}"
            )
        receiver.sweep_tx_gain_db[index] = tx_gain_db

        # IQ and all standard V7 fields are committed before the completion bit.
        receiver.signal_matrix[index] = signal
        scalar_values = {
            "system_timestamp": system_timestamp,
            "gps_timestamp": math.nan,
            "gps_lat": math.nan,
            "gps_long": math.nan,
            "heading": math.nan,
            "rx_theta_in_pis": math.nan,
            "rx_spacing": math.nan,
            "rx_lo": entry.lo_frequency_hz,
            "rx_bandwidth": self.config.bandwidth_hz,
        }
        for key in v4rx_f64_keys:
            receiver[key][index] = scalar_values[key]
        receiver.avg_phase_diff[index] = get_avg_phase_fast2(signal)
        receiver.rssis[index] = frame.rssis
        receiver.gains[index] = frame.gains

        for key in v7rx_scalar_keys:
            receiver[key][index] = getattr(frame, key)
        for key in v7rx_2x_keys:
            receiver[key][index] = getattr(frame, key)
        for key in v7rx_sample_time_scalar_keys:
            receiver[key][index] = getattr(frame, key)

        observation_count = len(frame.gain_observation_valid)
        event_count = len(frame.gain_event_flags)
        if observation_count > V7_GAIN_OBSERVATION_CAPACITY:
            raise ValueError("gain observation count exceeds V7 capacity")
        if event_count > V7_GAIN_EVENT_CAPACITY:
            raise ValueError("gain event count exceeds V7 capacity")
        gain_series_scalars = {
            "gain_observation_count": observation_count,
            "gain_observation_interval_samples": (
                frame.gain_observation_interval_samples
            ),
            "gain_observation_overflow_count": (frame.gain_observation_overflow_count),
            "gain_event_count": event_count,
            "gain_event_overflow_count": frame.gain_event_overflow_count,
        }
        for key in v7rx_gain_series_scalar_keys:
            receiver[key][index] = gain_series_scalars[key]

        bounds = np.full(
            (V7_GAIN_OBSERVATION_CAPACITY, 2), np.iinfo(np.uint64).max, np.uint64
        )
        indices = np.full((V7_GAIN_OBSERVATION_CAPACITY, 2), 0xFF, np.uint8)
        gains_db = np.full((V7_GAIN_OBSERVATION_CAPACITY, 2), np.nan, np.float32)
        valid = np.zeros(V7_GAIN_OBSERVATION_CAPACITY, np.bool_)
        durations = np.zeros(V7_GAIN_OBSERVATION_CAPACITY, np.uint32)
        if observation_count:
            bounds[:observation_count] = frame.gain_observation_sample_bounds
            indices[:observation_count] = frame.gain_observation_index
            gains_db[:observation_count] = frame.gain_observation_db
            valid[:observation_count] = frame.gain_observation_valid
            durations[:observation_count] = frame.gain_observation_read_duration_ns
        receiver.gain_observation_sample_bounds[index] = bounds
        receiver.gain_observation_index[index] = indices
        receiver.gain_observation_db[index] = gains_db
        receiver.gain_observation_valid[index] = valid
        receiver.gain_observation_read_duration_ns[index] = durations

        event_sequences = np.full(
            V7_GAIN_EVENT_CAPACITY, np.iinfo(np.uint64).max, np.uint64
        )
        event_flags = np.zeros(V7_GAIN_EVENT_CAPACITY, np.uint16)
        if event_count:
            event_sequences[:event_count] = frame.gain_event_sample_sequence
            event_flags[:event_count] = frame.gain_event_flags
        receiver.gain_event_sample_sequence[index] = event_sequences
        receiver.gain_event_flags[index] = event_flags

        receiver.tone_frequency_hz[index] = analysis["tone_frequency_hz"]
        receiver.tone_frequency_error_hz[index] = analysis["tone_frequency_error_hz"]
        receiver.phase_difference_rad[index] = analysis["phase_difference_rad"]
        receiver.within_capture_phase_std_rad[index] = analysis[
            "within_capture_phase_std_rad"
        ]
        receiver.within_capture_resultant_length[index] = analysis[
            "within_capture_resultant_length"
        ]
        receiver.coherence[index] = analysis["coherence"]
        receiver.amplitude_ratio_db_rx1_over_rx2[index] = analysis[
            "amplitude_ratio_db_rx1_over_rx2"
        ]
        for name in ("tone_dbfs", "tone_snr_db", "dc_dbfs", "clipping_fraction"):
            receiver[name][index] = analysis[name]
        receiver.sweep_quality_valid[index] = analysis["quality_valid"]
        receiver.sweep_quality_reason_mask[index] = quality_reason_mask(
            analysis["quality_reasons"]
        )
        receiver.sweep_completed[index] = True

    def sync(self) -> None:
        """Durably checkpoint all asynchronous LMDB writes."""

        if getattr(self, "zarr", None) is None:
            return
        store = self.zarr.store
        database = getattr(store, "db", None)
        if database is not None:
            # The pinned python-lmdb 1.4 API accepts ``force`` positionally
            # but not as a keyword.
            database.sync(True)
        else:
            store.flush()

    def close(self) -> None:
        if getattr(self, "zarr", None) is not None:
            try:
                self.sync()
            finally:
                self.zarr.store.close()
                self.zarr = None
                self.receiver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
