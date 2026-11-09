"""Strict validation and reporting for V7 gain/frequency calibration datasets."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import analyze_common_tone, circular_stats, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    build_schedule,
)
from spf.calibrations.dual_rx_gain_frequency.dataset import (
    CALIBRATION_SCHEMA,
    CALIBRATION_SCHEMA_VERSION,
    QUALITY_REASON_BITS,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import UNSAFE_FLAGS
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def _decode_reasons(mask: int) -> list[str]:
    known_mask = 0
    reasons = []
    for reason, bit in QUALITY_REASON_BITS.items():
        known_mask |= bit
        if mask & bit:
            reasons.append(reason)
    if mask & ~known_mask:
        raise ValueError(f"unknown quality reason bits: 0x{mask & ~known_mask:x}")
    return reasons


def _phase_error(a: float, b: float) -> float:
    return abs(float(wrap_phase(a - b)))


def validate_dataset(
    path: Path,
    *,
    config: CalibrationConfig,
    expected_serial: str | None = None,
    recompute_iq: bool = True,
    expected_rx_transport: str = "direct_usb",
    expected_iio_backend: str | None = None,
) -> dict[str, Any]:
    """Validate one serial-specific dataset and summarize phase coverage."""

    config.validate()
    schedule = build_schedule(config)
    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        if zarr.attrs.get("radio_metadata_schema_version") != 2:
            raise ValueError("dataset is not a V7 radio-metadata dataset")
        if zarr.attrs.get("calibration_schema") != CALIBRATION_SCHEMA:
            raise ValueError("dataset has the wrong calibration schema")
        if zarr.attrs.get("calibration_schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("dataset has an unsupported calibration version")
        receiver = zarr["receivers/r0"]
        serial = receiver.attrs.get("sdr_serial")
        if expected_serial is not None and serial != expected_serial:
            raise ValueError(f"serial {serial!r} != expected {expected_serial!r}")
        if receiver.attrs.get("rx_transport") != expected_rx_transport:
            raise ValueError(
                f"calibration transport {receiver.attrs.get('rx_transport')!r} "
                f"!= expected {expected_rx_transport!r}"
            )
        expected_protocol = 3 if expected_rx_transport == "iio" else 2
        if receiver.attrs.get("gain_metadata_protocol_version") != expected_protocol:
            raise ValueError(
                f"calibration did not use metadata protocol v{expected_protocol}"
            )
        if expected_iio_backend is not None:
            if receiver.attrs.get("iio_backend") != expected_iio_backend:
                raise ValueError(
                    f"IIO backend {receiver.attrs.get('iio_backend')!r} != "
                    f"expected {expected_iio_backend!r}"
                )
        if receiver.attrs.get("firmware_verified") is not True:
            raise ValueError("calibration firmware is not boot-verified")
        expected_shape = (len(schedule), 2, config.buffer_size)
        if receiver.signal_matrix.shape != expected_shape:
            raise ValueError(
                f"signal shape {receiver.signal_matrix.shape} != {expected_shape}"
            )

        completed = np.asarray(receiver.sweep_completed[:], dtype=bool)
        quality_valid = np.asarray(receiver.sweep_quality_valid[:], dtype=bool)
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        quality_reasons = Counter()
        capture_sequence_modes = set()
        seen_iio_sequences = set()
        for entry in schedule:
            index = entry.record_index
            expected_gain = np.asarray(
                [entry.gain_rx1_db, entry.gain_rx2_db], dtype=np.float32
            )
            if int(receiver.sweep_schedule_index[index]) != index:
                raise ValueError(f"schedule index mismatch at record {index}")
            if int(receiver.sweep_epoch[index]) != entry.epoch:
                raise ValueError(f"epoch mismatch at record {index}")
            if int(receiver.sweep_frequency_index[index]) != entry.frequency_index:
                raise ValueError(f"frequency index mismatch at record {index}")
            if int(receiver.sweep_lo_frequency_hz[index]) != entry.lo_frequency_hz:
                raise ValueError(f"LO coordinate mismatch at record {index}")
            if not np.array_equal(
                receiver.sweep_requested_gain_db[index], expected_gain
            ):
                raise ValueError(f"requested gain coordinate mismatch at {index}")
            if not completed[index]:
                continue
            if not bool(receiver.gain_metadata_valid[index]):
                raise ValueError(f"invalid gain metadata at record {index}")
            if not bool(receiver.rssi_metadata_valid[index]):
                raise ValueError(f"invalid RSSI metadata at record {index}")
            flags = int(receiver.gain_metadata_flags[index])
            if flags & int(UNSAFE_FLAGS):
                raise ValueError(f"unsafe metadata flags at record {index}")
            if not np.asarray(receiver.gain_endpoints_equal[index]).all():
                raise ValueError(f"gain endpoints differ at record {index}")
            if not np.array_equal(receiver.gain_db_start[index], expected_gain):
                raise ValueError(f"gain start mismatch at record {index}")
            if not np.array_equal(receiver.gain_db_end[index], expected_gain):
                raise ValueError(f"gain end mismatch at record {index}")
            if not np.isfinite(receiver.rssi_db_start[index]).all():
                raise ValueError(f"invalid RSSI start at record {index}")
            if not np.isfinite(receiver.rssi_db_end[index]).all():
                raise ValueError(f"invalid RSSI end at record {index}")
            actual_sequence = (
                int(receiver.buffer_sequence[index]),
                int(receiver.sample_sequence[index]),
            )
            if expected_rx_transport == "iio":
                if actual_sequence[1] <= 0:
                    raise ValueError(f"invalid IIO sample sequence at record {index}")
                if actual_sequence in seen_iio_sequences:
                    raise ValueError(
                        f"duplicate IIO capture sequence {actual_sequence}"
                    )
                seen_iio_sequences.add(actual_sequence)
                expected_end = actual_sequence[1] + config.buffer_size
                if int(receiver.sample_counter_end_exclusive[index]) != expected_end:
                    raise ValueError(f"IIO sample end mismatch at record {index}")
                if not bool(receiver.sample_time_valid[index]):
                    raise ValueError(f"IIO sample time is invalid at record {index}")
                capture_sequence_modes.add("iio_request_driven")
            else:
                legacy_sequence = (0, 0)
                batched_sequence = (
                    config.discard_frames_after_gain,
                    config.discard_frames_after_gain * config.buffer_size,
                )
                if actual_sequence == legacy_sequence:
                    capture_sequence_modes.add("separate_discard_and_capture")
                elif actual_sequence == batched_sequence:
                    capture_sequence_modes.add("batched_discard_and_capture")
                else:
                    raise ValueError(
                        f"unexpected finite stream sequence {actual_sequence} "
                        f"at record {index}"
                    )
            if int(receiver.rx_lo[index]) != entry.lo_frequency_hz:
                raise ValueError(f"V7 RX LO mismatch at record {index}")

            stored_reasons = _decode_reasons(
                int(receiver.sweep_quality_reason_mask[index])
            )
            quality_reasons.update(stored_reasons)
            stored_phase = float(receiver.phase_difference_rad[index])
            if recompute_iq:
                signal = receiver.signal_matrix[index]
                if not np.isfinite(signal).all():
                    raise ValueError(f"non-finite IQ at record {index}")
                if not np.any(signal[0]) or not np.any(signal[1]):
                    raise ValueError(f"zero IQ channel at record {index}")
                analysis = analyze_common_tone(
                    signal,
                    sample_rate_hz=config.sample_rate_hz,
                    expected_tone_offset_hz=config.tone_offset_hz,
                    tone_search_width_hz=config.tone_search_width_hz,
                    transient_samples=config.transient_samples,
                    phase_segments=config.phase_segments,
                    thresholds=config.quality,
                )
                if _phase_error(analysis["phase_difference_rad"], stored_phase) > 1e-5:
                    raise ValueError(f"stored phase does not match IQ at {index}")
                if (
                    abs(
                        analysis["tone_frequency_hz"]
                        - float(receiver.tone_frequency_hz[index])
                    )
                    > 1e-3
                ):
                    raise ValueError(f"stored tone frequency mismatch at {index}")
                if bool(analysis["quality_valid"]) != bool(quality_valid[index]):
                    raise ValueError(f"stored quality flag mismatch at {index}")
                if set(analysis["quality_reasons"]) != set(stored_reasons):
                    raise ValueError(f"stored quality reasons mismatch at {index}")

            grouped[
                (entry.frequency_index, entry.gain_rx1_db, entry.gain_rx2_db)
            ].append(
                {
                    "epoch": entry.epoch,
                    "quality_valid": bool(quality_valid[index]),
                    "phase": stored_phase,
                }
            )

        expected_cells = len(config.frequencies_hz) * len(config.gain_pairs)
        cell_rows = []
        for frequency_index, frequency in enumerate(config.frequencies_hz):
            for gain1, gain2 in config.gain_pairs:
                observations = grouped.get((frequency_index, gain1, gain2), [])
                valid = [
                    observation
                    for observation in observations
                    if observation["quality_valid"]
                ]
                stats = circular_stats(observation["phase"] for observation in valid)
                circular_std_deg = (
                    math.degrees(stats["circular_std_rad"])
                    if stats["circular_std_rad"] is not None
                    else None
                )
                passing = bool(
                    len(valid) >= config.min_quality_valid_per_cell
                    and circular_std_deg is not None
                    and circular_std_deg <= config.max_across_repeat_phase_std_deg
                )
                cell_rows.append(
                    {
                        "frequency_hz": frequency,
                        "gain_rx1_db": gain1,
                        "gain_rx2_db": gain2,
                        "role": (
                            "held_out"
                            if config.is_held_out_pair(gain1, gain2)
                            else "training"
                        ),
                        "n_complete": len(observations),
                        "n_quality_valid": len(valid),
                        "phase_mean_rad": stats["mean_rad"],
                        "phase_circular_std_deg": circular_std_deg,
                        "pass": passing,
                    }
                )
        passing_cells = sum(row["pass"] for row in cell_rows)
        training_cells = [row for row in cell_rows if row["role"] == "training"]
        held_out_cells = [row for row in cell_rows if row["role"] == "held_out"]
        passing_training_cells = sum(row["pass"] for row in training_cells)
        passing_held_out_cells = sum(row["pass"] for row in held_out_cells)
        complete_count = int(np.count_nonzero(completed))
        if len(capture_sequence_modes) > 1:
            raise ValueError(
                "dataset mixes separate and batched discard/capture sequencing"
            )
        if complete_count != len(schedule):
            status = "partial"
        elif passing_cells != expected_cells:
            status = "fail_quality"
        else:
            status = "pass"
        return {
            "schema": f"{CALIBRATION_SCHEMA}.validation",
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": status,
            "serial": serial,
            "expected_frames": len(schedule),
            "completed_frames": complete_count,
            "quality_valid_frames": int(np.count_nonzero(completed & quality_valid)),
            "expected_cells": expected_cells,
            "passing_cells": passing_cells,
            "expected_training_cells": len(training_cells),
            "passing_training_cells": passing_training_cells,
            "expected_held_out_cells": len(held_out_cells),
            "passing_held_out_cells": passing_held_out_cells,
            "capture_sequence_mode": (
                next(iter(capture_sequence_modes)) if capture_sequence_modes else None
            ),
            "quality_reason_counts": dict(quality_reasons),
            "cells": cell_rows,
        }
    finally:
        zarr.store.close()


def write_validation_report(path: Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
