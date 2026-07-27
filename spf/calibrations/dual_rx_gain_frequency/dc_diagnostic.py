"""Matched TX2-off/TX2-on diagnosis of high-RX2-gain DC failures."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spf.calibrations.dual_rx_gain_frequency.dc_offset import (
    read_rf_dc_registers,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import (
    DirectUsbLoopbackRadio,
)
from spf.calibrations.dual_rx_gain_frequency.runner import (
    _analyze,
    _open_preflight_radio,
    load_calibration_document,
)


DIAGNOSTIC_SCHEMA = "spf.calibration.dual_rx_gain_frequency.rx2_dc_diagnostic"
DIAGNOSTIC_SCHEMA_VERSION = 1
RECOVERY_SCHEMA = "spf.calibration.dual_rx_gain_frequency.rf_dc_recovery"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _default_dc_reader(radio: Any) -> dict[str, Any]:
    return {
        bank: read_rf_dc_registers(radio.sdr._ctrl, input_port=bank)
        for bank in ("A", "B_C")
    }


def _frame_metadata(frame: Any) -> dict[str, Any]:
    return {
        "gain_metadata_valid": bool(frame.gain_metadata_valid),
        "rssi_metadata_valid": bool(frame.rssi_metadata_valid),
        "gain_db_start": np.asarray(frame.gain_db_start),
        "gain_db_end": np.asarray(frame.gain_db_end),
        "gain_endpoints_equal": np.asarray(frame.gain_endpoints_equal),
        "gain_metadata_flags": int(frame.gain_metadata_flags),
        "rssi_db_start": np.asarray(frame.rssi_db_start),
        "rssi_db_end": np.asarray(frame.rssi_db_end),
        "stream_id": int(frame.stream_id),
        "buffer_sequence": int(frame.buffer_sequence),
        "sample_sequence": int(frame.sample_sequence),
        "iq_power_dbfs": np.asarray(frame.iq_power_dbfs),
    }


def _validate_gain_metadata(frame: Any, *, gain_rx1_db: int, gain_rx2_db: int) -> None:
    expected = np.asarray([gain_rx1_db, gain_rx2_db], dtype=np.float32)
    if not frame.gain_metadata_valid or not frame.rssi_metadata_valid:
        raise RuntimeError("diagnostic frame metadata is invalid")
    if not np.array_equal(frame.gain_db_start, expected):
        raise RuntimeError(
            f"gain start {frame.gain_db_start.tolist()} != {expected.tolist()}"
        )
    if not np.array_equal(frame.gain_db_end, expected):
        raise RuntimeError(
            f"gain end {frame.gain_db_end.tolist()} != {expected.tolist()}"
        )
    if not np.asarray(frame.gain_endpoints_equal).all():
        raise RuntimeError("diagnostic gain endpoints differ")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[bool, int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            bool(record["tx2_enabled"]),
            int(record["gain_rx1_db"]),
            int(record["gain_rx2_db"]),
        )
        grouped.setdefault(key, []).append(record)

    states = []
    by_pair: dict[tuple[int, int], dict[bool, dict[str, Any]]] = {}
    for (tx2_enabled, gain1, gain2), rows in sorted(grouped.items()):
        tone_dbfs = np.asarray([row["analysis"]["tone_dbfs"] for row in rows])
        dc_dbfs = np.asarray([row["analysis"]["dc_dbfs"] for row in rows])
        clipping = np.asarray([row["analysis"]["clipping_fraction"] for row in rows])
        state = {
            "tx2_enabled": tx2_enabled,
            "gain_rx1_db": gain1,
            "gain_rx2_db": gain2,
            "frames": len(rows),
            "quality_valid_frames": sum(
                bool(row["analysis"]["quality_valid"]) for row in rows
            ),
            "median_tone_dbfs": np.median(tone_dbfs, axis=0).tolist(),
            "median_dc_dbfs": np.median(dc_dbfs, axis=0).tolist(),
            "median_clipping_fraction": np.median(clipping, axis=0).tolist(),
            "maximum_clipping_fraction": np.max(clipping, axis=0).tolist(),
        }
        states.append(state)
        by_pair.setdefault((gain1, gain2), {})[tx2_enabled] = state

    comparisons = []
    for (gain1, gain2), pair in sorted(by_pair.items()):
        off = pair.get(False)
        on = pair.get(True)
        if off is None or on is None:
            continue
        off_dc = float(off["median_dc_dbfs"][1])
        on_dc = float(on["median_dc_dbfs"][1])
        off_clip = float(off["maximum_clipping_fraction"][1])
        on_clip = float(on["maximum_clipping_fraction"][1])
        if off_dc <= -30 and off_clip == 0 and (on_dc >= -20 or on_clip > 0):
            interpretation = "supports_tx2_coupled_rx2_dc_failure"
        elif off_dc >= -20 or off_clip > 0:
            interpretation = "supports_tx_independent_or_persistent_rx2_dc_failure"
        else:
            interpretation = "no_large_rx2_dc_failure_observed"
        comparisons.append(
            {
                "gain_rx1_db": gain1,
                "gain_rx2_db": gain2,
                "rx2_dc_on_minus_off_db": on_dc - off_dc,
                "rx2_max_clipping_off": off_clip,
                "rx2_max_clipping_on": on_clip,
                "interpretation": interpretation,
            }
        )
    return {
        "state_summaries": states,
        "on_off_comparisons": comparisons,
        "interpretation_policy": {
            "healthy_off_maximum_dc_dbfs": -30,
            "failed_on_minimum_dc_dbfs": -20,
            "any_clipping_is_failure": True,
            "warning": (
                "These thresholds classify the diagnostic symptom only. They "
                "do not prove the coupling path or authorize clipped IQ."
            ),
        },
    }


def run_rx2_dc_diagnostic(
    *,
    config_path: Path,
    serial: str,
    output_dir: Path,
    frequency_hz: int,
    gain_rx1_db: int,
    gain_rx2_values_db: tuple[int, ...],
    frames_per_state: int = 3,
    radio_factory: Callable[..., Any] = DirectUsbLoopbackRadio,
    dc_reader: Callable[[Any], dict[str, Any]] = _default_dc_reader,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Capture matched fresh-context TX2-off/on frames for one serial.

    Every gain/state point opens a new radio context. TX-off points never arm
    the DDS. TX-on points negotiate the existing direct-RX-safe handoff before
    the requested gains are applied. This prevents a failed TX-on correction
    state from being mistaken for a fresh TX-off baseline.
    """

    document, config = load_calibration_document(config_path)
    if frequency_hz not in config.frequencies_hz:
        raise ValueError("diagnostic frequency is outside the calibration config")
    if gain_rx1_db not in config.gains_db:
        raise ValueError("diagnostic RX1 gain is outside the calibration config")
    if not gain_rx2_values_db:
        raise ValueError("at least one RX2 gain is required")
    if len(set(gain_rx2_values_db)) != len(gain_rx2_values_db):
        raise ValueError("diagnostic RX2 gains must be unique")
    if any(gain not in config.gains_db for gain in gain_rx2_values_db):
        raise ValueError("diagnostic RX2 gain is outside the calibration config")
    if frames_per_state <= 0:
        raise ValueError("frames per state must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    records_path = output_dir / "records.jsonl"
    failure_log = output_dir / "handoff_failures.jsonl"
    records: list[dict[str, Any]] = []

    manifest = {
        "schema": DIAGNOSTIC_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "serial": serial,
        "frequency_hz": int(frequency_hz),
        "gain_rx1_db": int(gain_rx1_db),
        "gain_rx2_values_db": list(gain_rx2_values_db),
        "frames_per_state": frames_per_state,
        "state_order": ["fresh_tx2_off", "fresh_tx2_on"],
        "config_path": str(config_path),
        "firmware": document["pluto-firmware"],
        "warning": (
            "This is a diagnostic artifact, not a calibration V7 dataset. "
            "The paused exhaustive V7 artifact is not opened or modified."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)

    for tx2_enabled in (False, True):
        for gain_rx2_db in gain_rx2_values_db:
            radio = None
            preflight = None
            handoff = None
            try:
                if tx2_enabled:
                    radio, preflight, handoff = _open_preflight_radio(
                        serial=serial,
                        frequency_hz=frequency_hz,
                        config=config,
                        radio_factory=radio_factory,
                        failure_log=failure_log,
                        prepare_rf_dc=False,
                    )
                else:
                    radio = radio_factory(serial, config)
                    radio.configure_frequency(frequency_hz, start_tone=False)
                available = set(radio.available_gains())
                if gain_rx1_db not in available or gain_rx2_db not in available:
                    raise ValueError("requested diagnostic gain is unavailable")
                radio.set_gains(gain_rx1_db, gain_rx2_db)
                tx_gain_db = None
                if tx2_enabled:
                    tx_gain_db = config.tx_gain_for(gain_rx1_db, gain_rx2_db)
                    radio.set_tx_gain(tx_gain_db)
                sleep(config.settle_seconds)
                radio.discard(config.discard_frames_after_gain)
                correction_before = dc_reader(radio)
                for repeat in range(frames_per_state):
                    captured_at = time.time()
                    frame = radio.capture()
                    _validate_gain_metadata(
                        frame,
                        gain_rx1_db=gain_rx1_db,
                        gain_rx2_db=gain_rx2_db,
                    )
                    analysis = _analyze(frame, config)
                    correction_after = dc_reader(radio)
                    state_name = "tx2_on" if tx2_enabled else "tx2_off"
                    frame_name = (
                        f"{state_name}_g1_{gain_rx1_db:+03d}_"
                        f"g2_{gain_rx2_db:+03d}_r{repeat:02d}.npy"
                    )
                    np.save(
                        frames_dir / frame_name,
                        np.asarray(frame.signal_matrix, dtype=np.complex64),
                        allow_pickle=False,
                    )
                    record = {
                        "schema": DIAGNOSTIC_SCHEMA + ".frame",
                        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                        "serial": serial,
                        "frequency_hz": int(frequency_hz),
                        "gain_rx1_db": int(gain_rx1_db),
                        "gain_rx2_db": int(gain_rx2_db),
                        "tx2_enabled": tx2_enabled,
                        "tx_gain_db": tx_gain_db,
                        "repeat": repeat,
                        "captured_at_unix_seconds": captured_at,
                        "iq_file": str(Path("frames") / frame_name),
                        "frame_metadata": _frame_metadata(frame),
                        "analysis": analysis,
                        "rf_dc_correction_before_point": correction_before,
                        "rf_dc_correction_after_frame": correction_after,
                        "preflight": preflight,
                        "handoff": handoff,
                    }
                    records.append(_jsonable(record))
                    _append_jsonl(records_path, record)
            finally:
                if radio is not None:
                    radio.close()
                sleep(0.25)

    summary = {
        "schema": DIAGNOSTIC_SCHEMA + ".summary",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "complete",
        "serial": serial,
        "frequency_hz": int(frequency_hz),
        "record_count": len(records),
        "expected_record_count": (2 * len(gain_rx2_values_db) * frames_per_state),
        **_summarize_records(records),
    }
    if summary["record_count"] != summary["expected_record_count"]:
        raise RuntimeError("diagnostic frame count is incomplete")
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_rf_dc_recovery(
    *,
    config_path: Path,
    serial: str,
    output_path: Path,
    frequency_hz: int,
    gain_rx1_db: int,
    gain_rx2_values_db: tuple[int, ...],
    radio_factory: Callable[..., Any] = DirectUsbLoopbackRadio,
    dc_reader: Callable[[Any], dict[str, Any]] = _default_dc_reader,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Snapshot selected LUT entries, run RF-DC calibration, and snapshot again.

    TX is explicitly never armed. This invokes only the Linux driver's
    supported ``calib_mode=rf_dc_offs`` operation. It does not claim to rerun
    the separate BB-DC initialization calibration.
    """

    document, config = load_calibration_document(config_path)
    if frequency_hz not in config.frequencies_hz:
        raise ValueError("recovery frequency is outside the calibration config")
    if gain_rx1_db not in config.gains_db:
        raise ValueError("recovery RX1 gain is outside the calibration config")
    if not gain_rx2_values_db or any(
        gain not in config.gains_db for gain in gain_rx2_values_db
    ):
        raise ValueError("recovery RX2 gain is outside the calibration config")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"recovery output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def scan(radio: Any) -> list[dict[str, Any]]:
        snapshots = []
        for gain_rx2_db in gain_rx2_values_db:
            radio.set_gains(gain_rx1_db, gain_rx2_db)
            sleep(config.settle_seconds)
            snapshots.append(
                {
                    "gain_rx1_db": gain_rx1_db,
                    "gain_rx2_db": gain_rx2_db,
                    "correction_banks": dc_reader(radio),
                }
            )
        return snapshots

    with radio_factory(serial, config) as radio:
        radio.configure_frequency(frequency_hz, start_tone=False)
        before = scan(radio)
        radio.set_gains(gain_rx1_db, gain_rx1_db)
        sleep(config.settle_seconds)
        started = time.time()
        radio.run_rf_dc_calibration()
        completed = time.time()
        after = scan(radio)

    result = {
        "schema": RECOVERY_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "complete",
        "serial": serial,
        "frequency_hz": frequency_hz,
        "gain_rx1_db": gain_rx1_db,
        "gain_rx2_values_db": list(gain_rx2_values_db),
        "tx2_enabled": False,
        "operation": "Linux IIO calib_mode=rf_dc_offs",
        "operation_started_unix_seconds": started,
        "operation_completed_unix_seconds": completed,
        "before": before,
        "after": after,
        "firmware": document["pluto-firmware"],
        "warning": (
            "RF-DC initialization only. ADI's complete recovery guidance also "
            "calls for BB-DC initialization with the input isolated; that is "
            "not exposed by this Linux calib_mode operation."
        ),
    }
    _write_json(output_path, result)
    return result
