"""Two-radio orchestrator for direct-USB V7 gain/frequency calibration."""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    ScheduleEntry,
    build_schedule,
    group_schedule_by_frequency,
)
from spf.calibrations.dual_rx_gain_frequency.dataset import CalibrationV7Writer
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.scripts.pluto_ready_manifest import load_manifest


HANDOFF_PRIME_SEQUENCE = (False, True, False, True)


def _dataclass_keys(cls) -> set[str]:
    return {field.name for field in fields(cls)}


def load_calibration_document(path: Path) -> tuple[dict[str, Any], CalibrationConfig]:
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("calibration config must be a YAML mapping")
    if document.get("data-version") != 7:
        raise ValueError("calibration config requires data-version: 7")
    firmware = document.get("pluto-firmware")
    if not isinstance(firmware, dict):
        raise ValueError("calibration config requires pluto-firmware provenance")
    raw = document.get("calibration")
    if not isinstance(raw, dict):
        raise ValueError("calibration config requires a calibration mapping")
    normalized = {key.replace("-", "_"): value for key, value in raw.items()}
    quality_raw = normalized.pop("quality", {})
    if not isinstance(quality_raw, dict):
        raise ValueError("calibration.quality must be a mapping")
    quality = {key.replace("-", "_"): value for key, value in quality_raw.items()}
    unknown_quality = set(quality) - _dataclass_keys(ToneQualityThresholds)
    if unknown_quality:
        raise ValueError(f"unknown quality settings: {sorted(unknown_quality)}")
    normalized["quality"] = ToneQualityThresholds(**quality)
    normalized["frequencies_hz"] = tuple(
        int(value) for value in normalized["frequencies_hz"]
    )
    if "gains_db" in normalized:
        normalized["gains_db"] = tuple(int(value) for value in normalized["gains_db"])
    unknown = set(normalized) - _dataclass_keys(CalibrationConfig)
    if unknown:
        raise ValueError(f"unknown calibration settings: {sorted(unknown)}")
    config = CalibrationConfig(**normalized)
    config.validate()
    return document, config


def serials_from_ready_manifest(path: Path) -> tuple[str, ...]:
    manifest = load_manifest(path)
    radios = manifest.get("radios", [])
    serials = tuple(str(radio.get("serial", "")) for radio in radios)
    if not serials or any(not serial for serial in serials):
        raise ValueError("ready manifest does not contain valid radio serials")
    if len(serials) != len(set(serials)):
        raise ValueError("ready manifest contains duplicate serials")
    if any(not radio.get("firmware_verified") for radio in radios):
        raise ValueError("ready manifest contains an unverified radio")
    return serials


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _validate_frame_gain(frame, entry: ScheduleEntry) -> None:
    requested = np.asarray([entry.gain_rx1_db, entry.gain_rx2_db], dtype=np.float32)
    if not frame.gain_metadata_valid:
        raise RuntimeError("gain metadata is invalid")
    if not frame.rssi_metadata_valid:
        raise RuntimeError("RSSI metadata is invalid")
    if not np.array_equal(frame.gain_db_start, requested):
        raise RuntimeError(
            f"gain start {frame.gain_db_start.tolist()} != {requested.tolist()}"
        )
    if not np.array_equal(frame.gain_db_end, requested):
        raise RuntimeError(
            f"gain end {frame.gain_db_end.tolist()} != {requested.tolist()}"
        )
    if not np.asarray(frame.gain_endpoints_equal).all():
        raise RuntimeError("raw gain endpoints differ within the captured frame")


def _analyze(frame, config: CalibrationConfig) -> dict[str, Any]:
    return analyze_common_tone(
        frame.signal_matrix,
        sample_rate_hz=config.sample_rate_hz,
        expected_tone_offset_hz=config.tone_offset_hz,
        tone_search_width_hz=config.tone_search_width_hz,
        transient_samples=config.transient_samples,
        phase_segments=config.phase_segments,
        thresholds=config.quality,
    )


def _preflight_tone(radio, config: CalibrationConfig) -> dict[str, Any]:
    """Require a clean reference-gain direct frame before recording a block."""

    reference_gain = config.tx_reference_rx_gain_db
    radio.set_gains(reference_gain, reference_gain)
    radio.set_tx_gain(config.tx_gain_for(reference_gain, reference_gain))
    time.sleep(config.settle_seconds)
    radio.discard(config.discard_frames_after_gain)
    frame = radio.capture()
    requested = np.asarray([reference_gain, reference_gain], dtype=np.float32)
    if not frame.gain_metadata_valid or not frame.rssi_metadata_valid:
        raise RuntimeError("preflight metadata is invalid")
    if not np.array_equal(frame.gain_db_start, requested) or not np.array_equal(
        frame.gain_db_end, requested
    ):
        raise RuntimeError("preflight gain metadata does not match reference gain")
    analysis = _analyze(frame, config)
    if not analysis["quality_valid"]:
        raise RuntimeError(
            "direct-USB tone preflight failed: "
            + ", ".join(analysis["quality_reasons"])
        )
    return analysis


def _open_preflight_radio(
    *,
    serial: str,
    frequency_hz: int,
    config: CalibrationConfig,
    radio_factory: Callable[..., Any],
    failure_log: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Open one radio and negotiate a direct-RX-safe FPGA DDS handoff."""

    failures = []
    for attempt, prime_after_arm in enumerate(HANDOFF_PRIME_SEQUENCE, start=1):
        radio = None
        try:
            radio = radio_factory(serial, config)
            radio.configure_frequency(frequency_hz, start_tone=False)
            radio.start_tone(
                tx_channel=1,
                prime_after_arm=prime_after_arm,
            )
            preflight = _preflight_tone(radio, config)
            return (
                radio,
                preflight,
                {
                    "attempt": attempt,
                    "tx_source": config.tx_source,
                    "prime_after_arm": prime_after_arm,
                },
            )
        except Exception as error:
            failure = {
                "status": "fail",
                "serial": serial,
                "frequency_hz": frequency_hz,
                "attempt": attempt,
                "tx_source": config.tx_source,
                "prime_after_arm": prime_after_arm,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            _append_jsonl(failure_log, failure)
            if radio is not None:
                radio.close()
            # Give FunctionFS/libusb teardown a bounded interval before a new
            # pyadi context claims the same USB-IIO interface.
            time.sleep(0.5)
    summary = "; ".join(
        f"attempt {failure['attempt']} prime_after_arm="
        f"{failure['prime_after_arm']}: {failure['error']}"
        for failure in failures
    )
    raise RuntimeError(f"{serial}: no direct-safe TX2 DDS handoff: {summary}")


def run_calibration(
    *,
    config_path: Path,
    output_dir: Path,
    ready_manifest_path: Path = Path("/run/spf/direct_usb_ready.json"),
    serials: tuple[str, ...] | None = None,
    radio_factory: Callable[..., Any] = DirectUsbLoopbackRadio,
    progress: Callable[[str, ScheduleEntry, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run or resume a serial-specific V7 dataset for every selected radio."""

    yaml_document, config = load_calibration_document(config_path)
    if serials is None:
        serials = serials_from_ready_manifest(ready_manifest_path)
    if not serials:
        raise ValueError("at least one radio serial is required")
    schedule = build_schedule(config)
    blocks = group_schedule_by_frequency(schedule)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    identities = {}
    for serial in serials:
        with radio_factory(serial, config) as radio:
            available = set(radio.available_gains())
            unavailable = sorted(set(config.gains_db) - available)
            if unavailable:
                raise RuntimeError(
                    f"{serial}: requested unavailable gains {unavailable}"
                )
            identities[serial] = radio.identity()

    with ExitStack() as stack:
        writers = {
            serial: stack.enter_context(
                CalibrationV7Writer(
                    output_dir / serial / "calibration.v7.zarr",
                    config=config,
                    schedule=schedule,
                    identity=identities[serial],
                    yaml_config=yaml_document,
                    resume=True,
                )
            )
            for serial in serials
        }

        total = len(schedule) * len(serials)
        completed = sum(
            writer.is_complete(entry)
            for writer in writers.values()
            for entry in schedule
        )
        for block_index, block in enumerate(blocks):
            radio_order = (
                tuple(reversed(serials)) if block_index % 2 else tuple(serials)
            )
            for serial in radio_order:
                writer = writers[serial]
                pending = [entry for entry in block if not writer.is_complete(entry)]
                if not pending:
                    continue
                radio = None
                try:
                    radio, preflight, handoff = _open_preflight_radio(
                        serial=serial,
                        frequency_hz=block[0].lo_frequency_hz,
                        config=config,
                        radio_factory=radio_factory,
                        failure_log=output_dir / serial / "preflight_failures.jsonl",
                    )
                    _append_jsonl(
                        output_dir / serial / "preflight.jsonl",
                        {
                            "status": "pass",
                            "epoch": block[0].epoch,
                            "frequency_index": block[0].frequency_index,
                            "frequency_hz": block[0].lo_frequency_hz,
                            "reference_gain_db": config.tx_reference_rx_gain_db,
                            "tone_dbfs": preflight["tone_dbfs"],
                            "tone_snr_db": preflight["tone_snr_db"],
                            "coherence": preflight["coherence"],
                            "within_capture_phase_std_deg": preflight[
                                "within_capture_phase_std_deg"
                            ],
                            **handoff,
                        },
                    )
                    for entry in pending:
                        if progress:
                            progress(serial, entry, completed, total)
                        for retry in range(config.max_retries + 1):
                            attempt = writer.increment_attempts(entry)
                            started = time.monotonic()
                            try:
                                radio.set_gains(entry.gain_rx1_db, entry.gain_rx2_db)
                                tx_gain_db = config.tx_gain_for(
                                    entry.gain_rx1_db, entry.gain_rx2_db
                                )
                                radio.set_tx_gain(tx_gain_db)
                                time.sleep(config.settle_seconds)
                                radio.discard(config.discard_frames_after_gain)
                                frame = radio.capture()
                                _validate_frame_gain(frame, entry)
                                analysis = _analyze(frame, config)
                                writer.write(
                                    entry,
                                    frame,
                                    analysis,
                                    system_timestamp=time.time(),
                                    tx_gain_db=tx_gain_db,
                                )
                                _append_jsonl(
                                    output_dir / serial / "observations.jsonl",
                                    {
                                        "status": "ok",
                                        "measurement_key": entry.key,
                                        "record_index": entry.record_index,
                                        "attempt": attempt,
                                        "elapsed_seconds": time.monotonic() - started,
                                        "quality_valid": analysis["quality_valid"],
                                        "quality_reasons": analysis["quality_reasons"],
                                        "phase_difference_rad": analysis[
                                            "phase_difference_rad"
                                        ],
                                        "tone_frequency_hz": analysis[
                                            "tone_frequency_hz"
                                        ],
                                    },
                                )
                                completed += 1
                                break
                            except Exception as error:
                                _append_jsonl(
                                    output_dir / serial / "observations.jsonl",
                                    {
                                        "status": "error",
                                        "measurement_key": entry.key,
                                        "record_index": entry.record_index,
                                        "attempt": attempt,
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                        "elapsed_seconds": time.monotonic() - started,
                                    },
                                )
                                if retry >= config.max_retries:
                                    break
                finally:
                    try:
                        if radio is not None:
                            radio.close()
                    finally:
                        # Per-frame LMDB writes are asynchronous for capture
                        # throughput. Make every radio/frequency block a
                        # durable restart boundary.
                        writer.sync()

        result = {
            "status": "complete" if completed == total else "partial",
            "radio_serials": list(serials),
            "measurements_per_radio": len(schedule),
            "expected_measurements": total,
            "completed_measurements": completed,
            "output_dir": str(output_dir),
        }
        (output_dir / "run_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result


def probe_loopback(
    *,
    config_path: Path,
    serial: str,
    frequency_hz: int | None = None,
    gain_db: int | None = None,
    tx_channel: int = 1,
    minimum_on_off_delta_db: float = 20.0,
    radio_factory: Callable[..., Any] = DirectUsbLoopbackRadio,
) -> dict[str, Any]:
    """Measure TX2-off and TX2-on tone levels without writing a dataset."""

    _, config = load_calibration_document(config_path)
    frequency_hz = (
        config.frequencies_hz[0] if frequency_hz is None else int(frequency_hz)
    )
    gain_db = config.gains_db[len(config.gains_db) // 2] if gain_db is None else gain_db
    last_result = None
    for attempt, prime_after_arm in enumerate(HANDOFF_PRIME_SEQUENCE, start=1):
        with radio_factory(serial, config) as radio:
            if gain_db not in radio.available_gains():
                raise ValueError(f"gain {gain_db} is unavailable")
            radio.configure_frequency(frequency_hz, start_tone=False)
            radio.set_gains(gain_db, gain_db)
            time.sleep(config.settle_seconds)
            radio.discard(config.discard_frames_after_gain)
            off = radio.capture()
            off_analysis = _analyze(off, config)

            radio.start_tone(
                tx_channel=tx_channel,
                prime_after_arm=prime_after_arm,
            )
            if tx_channel == 1:
                radio.set_tx_gain(config.tx_gain_for(gain_db, gain_db))
            time.sleep(config.frequency_settle_seconds)
            radio.discard(config.discard_frames_after_gain)
            on = radio.capture()
            on_analysis = _analyze(on, config)
            delta = np.asarray(on_analysis["tone_dbfs"]) - np.asarray(
                off_analysis["tone_dbfs"]
            )
            passed = bool(
                on_analysis["quality_valid"]
                and np.all(delta >= minimum_on_off_delta_db)
            )
            last_result = {
                "status": "pass" if passed else "fail",
                "serial": serial,
                "frequency_hz": frequency_hz,
                "gain_db": gain_db,
                "tx_channel": tx_channel,
                "tx_source": config.tx_source,
                "handoff_attempt": attempt,
                "prime_after_arm": prime_after_arm,
                "minimum_on_off_delta_db": minimum_on_off_delta_db,
                "tone_off_dbfs": off_analysis["tone_dbfs"],
                "tone_on_dbfs": on_analysis["tone_dbfs"],
                "tone_on_off_delta_db": delta.tolist(),
                "tone_on_snr_db": on_analysis["tone_snr_db"],
                "tone_on_frequency_hz": on_analysis["tone_frequency_hz"],
                "tone_on_phase_difference_rad": on_analysis[
                    "phase_difference_rad"
                ],
                "tone_on_quality_valid": on_analysis["quality_valid"],
                "tone_on_quality_reasons": on_analysis["quality_reasons"],
            }
            if passed:
                return last_result
        time.sleep(0.5)
    assert last_result is not None
    return last_result
