"""Reproducible model comparison for dual-RX gain/frequency calibrations.

This module deliberately reads only scalar coordinates, quality decisions, and
stored phase estimates from an existing V7 calibration. It never opens a radio
or mutates a dataset. Full-IQ recomputation remains the responsibility of the
strict ``validate`` command.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from spf.bench.dual_rx_phase import circular_stats, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.dual_rx_gain_frequency.runner import (
    load_calibration_document,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


COMPARISON_SCHEMA = "spf.calibration.dual_rx_gain_frequency.comparison"
COMPARISON_SCHEMA_VERSION = 2
CALIBRATION_SCHEMA = "spf.calibration.dual_rx_gain_frequency.preliminary_model"
CALIBRATION_SCHEMA_VERSION = 1

# Exact source used by firmware dd6b1f4. The first gain-table byte contains the
# LNA index in bits 6:5 and mixer-Gm index in bits 4:0. Values below are mapped
# from requested hardwaregain -10..62 dB through the driver's high-band
# absolute-gain table to its first matching full-table row.
GAIN_TABLE_LINUX_GIT_SHA = "d798b0d821b85ebd51ecffbfa68d8e4d69b77132"
GAIN_TABLE_SOURCE_PATH = "drivers/iio/adc/ad9361.c"
HIGH_BAND_GAIN_MIN_DB = -10
HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN = (
    0x00,
    0x00,
    0x00,
    0x00,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x01,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x02,
    0x04,
    0x04,
    0x04,
    0x04,
    0x04,
    0x04,
    0x04,
    0x24,
    0x24,
    0x24,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x44,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x6B,
    0x6C,
    0x6D,
    0x6E,
    0x6F,
)


def _derive_stage_boundaries(
    control_bytes: tuple[int, ...],
    *,
    minimum_gain_db: int,
    minimum_plateau_length: int = 3,
) -> tuple[int, ...]:
    """Return starts of multi-gain LNA/mixer plateaus.

    The high-band table finishes with a one-index-per-dB mixer ramp from
    52..62 dB. The linear term represents that ramp; introducing eleven
    one-point step parameters would defeat the purpose of a compact model.
    """

    boundaries = []
    start = 0
    while start < len(control_bytes):
        end = start + 1
        while end < len(control_bytes) and control_bytes[end] == control_bytes[start]:
            end += 1
        if start and end - start >= minimum_plateau_length:
            boundaries.append(minimum_gain_db + start)
        start = end
    return tuple(boundaries)


DEFAULT_STAGE_BOUNDARIES_DB = _derive_stage_boundaries(
    HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN,
    minimum_gain_db=HIGH_BAND_GAIN_MIN_DB,
)
DEFAULT_ANCHOR_GAINS_DB = (0, 16, 26, 41, 45)

MODEL_ORDER = (
    "constant",
    "linear_difference",
    "linear_ordered",
    "stage_shared",
    "stage_ordered",
    "categorical_shared",
    "categorical_ordered",
)
MODEL_LABELS = {
    "constant": "Constant baseline",
    "linear_difference": "Linear gain difference",
    "linear_ordered": "Separate linear RX1/RX2",
    "stage_shared": "Shared signed stage curve",
    "stage_ordered": "Ordered stage-boundary model",
    "categorical_shared": "Shared signed categorical curve",
    "categorical_ordered": "Ordered categorical additive",
}

ANALYSIS_ARRAYS = (
    "sweep_completed",
    "sweep_quality_valid",
    "sweep_epoch",
    "sweep_frequency_index",
    "sweep_lo_frequency_hz",
    "sweep_requested_gain_db",
    "phase_difference_rad",
    "system_timestamp",
)


def _portable_path(path: Path) -> str:
    """Render artifact paths relative to the repository when recognizable."""

    parts = path.parts
    if "artifacts" in parts:
        return str(Path(*parts[parts.index("artifacts") :]))
    return str(path)


def _wrap(value: np.ndarray | float) -> np.ndarray:
    return np.asarray(wrap_phase(np.asarray(value)), dtype=np.float64)


def _metrics(residual: np.ndarray) -> dict[str, float | int]:
    degrees = np.abs(np.degrees(_wrap(residual)))
    if not degrees.size:
        raise ValueError("cannot summarize empty residuals")
    return {
        "n": int(degrees.size),
        "circular_mae_deg": float(np.mean(degrees)),
        "circular_rmse_deg": float(np.sqrt(np.mean(degrees**2))),
        "circular_p95_deg": float(np.percentile(degrees, 95)),
        "circular_max_deg": float(np.max(degrees)),
    }


def _stage_boundaries(config: CalibrationConfig) -> tuple[int, ...]:
    lower = min(config.gains_db)
    upper = max(config.gains_db)
    return tuple(
        boundary
        for boundary in DEFAULT_STAGE_BOUNDARIES_DB
        if lower < boundary <= upper
    )


def _reference_gain(config: CalibrationConfig) -> int:
    return min(
        config.gains_db,
        key=lambda gain: abs(gain - config.tx_reference_rx_gain_db),
    )


def _design_matrix(
    model: str,
    gain1: np.ndarray,
    gain2: np.ndarray,
    *,
    config: CalibrationConfig,
) -> np.ndarray:
    gain1 = np.asarray(gain1, dtype=np.int64)
    gain2 = np.asarray(gain2, dtype=np.int64)
    if gain1.shape != gain2.shape or gain1.ndim != 1:
        raise ValueError("gain vectors must be equal-length one-dimensional arrays")
    columns: list[np.ndarray] = [np.ones(gain1.size)]
    boundaries = _stage_boundaries(config)
    if model == "constant":
        pass
    elif model == "linear_difference":
        columns.append((gain1 - gain2) / 20.0)
    elif model == "linear_ordered":
        columns.extend([gain1 / 20.0, gain2 / 20.0])
    elif model == "stage_shared":
        columns.append((gain1 - gain2) / 20.0)
        columns.extend(
            [
                (gain1 >= boundary).astype(float) - (gain2 >= boundary).astype(float)
                for boundary in boundaries
            ]
        )
    elif model == "stage_ordered":
        columns.extend([gain1 / 20.0, gain2 / 20.0])
        columns.extend([(gain1 >= boundary).astype(float) for boundary in boundaries])
        columns.extend([(gain2 >= boundary).astype(float) for boundary in boundaries])
    elif model == "categorical_shared":
        reference = _reference_gain(config)
        columns.extend(
            [
                (gain1 == gain).astype(float) - (gain2 == gain).astype(float)
                for gain in config.gains_db
                if gain != reference
            ]
        )
    elif model == "categorical_ordered":
        reference = _reference_gain(config)
        columns.extend(
            [
                (gain1 == gain).astype(float)
                for gain in config.gains_db
                if gain != reference
            ]
        )
        columns.extend(
            [
                (gain2 == gain).astype(float)
                for gain in config.gains_db
                if gain != reference
            ]
        )
    else:
        raise ValueError(f"unknown comparison model: {model}")
    return np.column_stack(columns)


def _fit_circular(
    model: str,
    gain1: np.ndarray,
    gain2: np.ndarray,
    phase: np.ndarray,
    *,
    config: CalibrationConfig,
) -> np.ndarray:
    matrix = _design_matrix(model, gain1, gain2, config=config)
    phase = np.asarray(phase, dtype=np.float64)
    if phase.shape != gain1.shape:
        raise ValueError("phase and gain vectors must have equal shapes")
    initial, *_ = np.linalg.lstsq(matrix, np.unwrap(phase), rcond=None)
    result = least_squares(
        lambda beta: _wrap(phase - matrix @ beta),
        initial,
        max_nfev=300,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    if not result.success:
        raise RuntimeError(f"{model} did not converge: {result.message}")
    return np.asarray(result.x)


def _predict(
    model: str,
    parameters: np.ndarray,
    gain1: np.ndarray,
    gain2: np.ndarray,
    *,
    config: CalibrationConfig,
) -> np.ndarray:
    matrix = _design_matrix(model, gain1, gain2, config=config)
    return _wrap(matrix @ parameters)


def _nominal_parameter_count(model: str, config: CalibrationConfig) -> int:
    return int(
        _design_matrix(
            model,
            np.asarray([_reference_gain(config)]),
            np.asarray([_reference_gain(config)]),
            config=config,
        ).shape[1]
    )


def _hash_arrays(arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(attrs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in ANALYSIS_ARRAYS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_radio(dataset_path: Path, *, config: CalibrationConfig) -> dict[str, Any]:
    zarr = zarr_open_from_lmdb_store(str(dataset_path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        arrays = {name: np.asarray(receiver[name][:]) for name in ANALYSIS_ARRAYS}
        attrs = {
            "calibration_run_signature": zarr.attrs.get("calibration_run_signature"),
            "calibration_software_git_sha": zarr.attrs.get(
                "calibration_software_git_sha"
            ),
            "phase_convention": zarr.attrs.get("phase_convention"),
            "sdr_serial": receiver.attrs.get("sdr_serial"),
            "firmware_release_tag": receiver.attrs.get("firmware_release_tag"),
            "firmware_git_sha": receiver.attrs.get("firmware_git_sha"),
            "firmware_gadget_git_sha": receiver.attrs.get("firmware_gadget_git_sha"),
            "firmware_image_sha256": receiver.attrs.get("firmware_image_sha256"),
            "gain_metadata_protocol_version": receiver.attrs.get(
                "gain_metadata_protocol_version"
            ),
        }
    finally:
        zarr.store.close()

    serial = attrs["sdr_serial"]
    if not serial:
        raise ValueError(f"{dataset_path}: missing Pluto serial")
    completed = np.asarray(arrays["sweep_completed"], dtype=bool)
    epoch = np.asarray(arrays["sweep_epoch"], dtype=np.int64)
    frequency_index = np.asarray(arrays["sweep_frequency_index"], dtype=np.int64)
    # A complete block is the configured schedule, which may be either the
    # full Cartesian gain grid or the much smaller additive-cross design.
    # Treating every schedule as Cartesian silently excluded every observation
    # from additive-cross datasets.
    expected_per_block = len(config.gain_pairs)
    full_blocks: list[dict[str, int]] = []
    analysis_mask = np.zeros(completed.shape, dtype=bool)
    for epoch_value in range(config.repetitions):
        for index, frequency_hz in enumerate(config.frequencies_hz):
            coordinate = (epoch == epoch_value) & (frequency_index == index)
            complete_count = int(np.count_nonzero(completed & coordinate))
            if complete_count == expected_per_block:
                analysis_mask |= completed & coordinate
                full_blocks.append(
                    {
                        "epoch": epoch_value,
                        "frequency_index": index,
                        "frequency_hz": int(frequency_hz),
                        "completed_frames": complete_count,
                    }
                )
            elif complete_count:
                full_blocks.append(
                    {
                        "epoch": epoch_value,
                        "frequency_index": index,
                        "frequency_hz": int(frequency_hz),
                        "completed_frames": complete_count,
                        "excluded_as_partial": True,
                    }
                )

    requested = np.asarray(arrays["sweep_requested_gain_db"], dtype=np.int64)
    quality = np.asarray(arrays["sweep_quality_valid"], dtype=bool)
    return {
        "serial": str(serial),
        "dataset_path": _portable_path(dataset_path),
        "attrs": attrs,
        "arrays": arrays,
        "analysis_input_sha256": _hash_arrays(arrays, attrs),
        "completed_frames": int(np.count_nonzero(completed)),
        "analysis_mask": analysis_mask,
        "quality_mask": analysis_mask & quality,
        "epoch": epoch,
        "frequency_index": frequency_index,
        "frequency_hz": np.asarray(arrays["sweep_lo_frequency_hz"], dtype=np.int64),
        "gain1": requested[:, 0],
        "gain2": requested[:, 1],
        "phase": np.asarray(arrays["phase_difference_rad"], dtype=np.float64),
        "timestamp": np.asarray(arrays["system_timestamp"], dtype=np.float64),
        "full_blocks": full_blocks,
    }


def _select_group(radio: dict[str, Any], frequency_hz: int) -> dict[str, np.ndarray]:
    selected = radio["quality_mask"] & (radio["frequency_hz"] == frequency_hz)
    return {
        "gain1": radio["gain1"][selected],
        "gain2": radio["gain2"][selected],
        "phase": radio["phase"][selected],
        "timestamp": radio["timestamp"][selected],
        "epoch": radio["epoch"][selected],
    }


def _supported_test(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> np.ndarray:
    return np.isin(test["gain1"], np.unique(train["gain1"])) & np.isin(
        test["gain2"], np.unique(train["gain2"])
    )


def _subset(
    group: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: value[selected] for key, value in group.items()}


def _random_cell_cv(
    group: dict[str, np.ndarray], *, config: CalibrationConfig
) -> dict[str, Any]:
    gain1 = group["gain1"]
    gain2 = group["gain2"]
    gain_lookup = {gain: index for index, gain in enumerate(config.gains_db)}
    gain1_index = np.asarray([gain_lookup[int(value)] for value in gain1])
    gain2_index = np.asarray([gain_lookup[int(value)] for value in gain2])
    folds = (gain1_index * 73 + gain2_index * 37 + gain1_index * gain2_index * 19) % 5
    residuals: dict[str, list[float]] = defaultdict(list)
    for held_fold in range(5):
        train = _subset(group, folds != held_fold)
        test = _subset(group, folds == held_fold)
        supported = _supported_test(train, test)
        test = _subset(test, supported)
        if not train["phase"].size or not test["phase"].size:
            continue
        for model in MODEL_ORDER:
            fitted = _fit_circular(
                model,
                train["gain1"],
                train["gain2"],
                train["phase"],
                config=config,
            )
            prediction = _predict(
                model,
                fitted,
                test["gain1"],
                test["gain2"],
                config=config,
            )
            residuals[model].extend(_wrap(test["phase"] - prediction))
    constant_mae = _metrics(np.asarray(residuals["constant"]))["circular_mae_deg"]
    return {
        "evaluation": (
            "Five deterministic folds hold out complete ordered gain-pair "
            "cells; all repeats of a cell remain in the same fold."
        ),
        "models": {
            model: {
                "label": MODEL_LABELS[model],
                "nominal_parameters": _nominal_parameter_count(model, config),
                **_metrics(np.asarray(residuals[model])),
                "mae_reduction_vs_constant_fraction": float(
                    1
                    - _metrics(np.asarray(residuals[model]))["circular_mae_deg"]
                    / constant_mae
                ),
            }
            for model in MODEL_ORDER
        },
    }


def _quadrant_cv(
    group: dict[str, np.ndarray], *, config: CalibrationConfig
) -> dict[str, Any]:
    lower = max(min(config.gains_db), -10)
    upper = min(max(config.gains_db), 45)
    split = 18
    selected = (
        (group["gain1"] >= lower)
        & (group["gain1"] <= upper)
        & (group["gain2"] >= lower)
        & (group["gain2"] <= upper)
        & (np.abs(group["gain1"] - group["gain2"]) <= 30)
    )
    restricted = _subset(group, selected)
    residuals: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = {}
    models = ("stage_shared", "stage_ordered", "categorical_ordered")
    for high1, high2 in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        held = ((restricted["gain1"] >= split) == high1) & (
            (restricted["gain2"] >= split) == high2
        )
        train = _subset(restricted, ~held)
        test = _subset(restricted, held)
        supported = _supported_test(train, test)
        test = _subset(test, supported)
        counts[f"{int(high1)}{int(high2)}"] = int(test["phase"].size)
        if not train["phase"].size or not test["phase"].size:
            continue
        for model in models:
            fitted = _fit_circular(
                model,
                train["gain1"],
                train["gain2"],
                train["phase"],
                config=config,
            )
            prediction = _predict(
                model,
                fitted,
                test["gain1"],
                test["gain2"],
                config=config,
            )
            residuals[model].extend(_wrap(test["phase"] - prediction))
    return {
        "evaluation": (
            "Hold out each low/high RX1-by-RX2 quadrant in turn while retaining "
            "every tested gain state elsewhere in training."
        ),
        "restriction": {
            "gain_range_db": [lower, upper],
            "maximum_absolute_gain_mismatch_db": 30,
            "low_high_split_db": split,
            "supported_test_observations_by_quadrant": counts,
        },
        "models": {
            model: (
                {
                    "label": MODEL_LABELS[model],
                    **_metrics(np.asarray(residuals[model])),
                }
                if residuals[model]
                else None
            )
            for model in models
        },
    }


def _time_diagnostics(
    group: dict[str, np.ndarray], *, config: CalibrationConfig
) -> dict[str, Any]:
    fitted = _fit_circular(
        "categorical_ordered",
        group["gain1"],
        group["gain2"],
        group["phase"],
        config=config,
    )
    residual = _wrap(
        group["phase"]
        - _predict(
            "categorical_ordered",
            fitted,
            group["gain1"],
            group["gain2"],
            config=config,
        )
    )
    hours = (group["timestamp"] - np.mean(group["timestamp"])) / 3600.0
    slope = np.linalg.lstsq(
        np.column_stack([np.ones(hours.size), hours]),
        residual,
        rcond=None,
    )[0][1]
    elapsed_correlation = spearmanr(group["timestamp"], residual)
    order = np.argsort(group["timestamp"])
    jump = np.zeros(group["phase"].size)
    jump[order[1:]] = np.abs(np.diff(group["gain1"][order])) + np.abs(
        np.diff(group["gain2"][order])
    )
    jump_correlation = spearmanr(jump, np.abs(residual))
    return {
        "training_residual": _metrics(residual),
        "linear_residual_drift_deg_per_hour": float(np.degrees(slope)),
        "timestamp_vs_signed_residual_spearman": {
            "rho": float(elapsed_correlation.statistic),
            "p_value": float(elapsed_correlation.pvalue),
        },
        "prior_total_gain_jump_vs_absolute_residual_spearman": {
            "rho": float(jump_correlation.statistic),
            "p_value": float(jump_correlation.pvalue),
        },
    }


def _fit_source(
    source: dict[str, np.ndarray], *, config: CalibrationConfig
) -> tuple[np.ndarray, set[int], set[int]]:
    fitted = _fit_circular(
        "categorical_ordered",
        source["gain1"],
        source["gain2"],
        source["phase"],
        config=config,
    )
    return fitted, set(source["gain1"]), set(source["gain2"])


def _transfer(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    config: CalibrationConfig,
) -> dict[str, Any]:
    fitted, source_rx1, source_rx2 = _fit_source(source, config=config)
    supported = np.isin(target["gain1"], list(source_rx1)) & np.isin(
        target["gain2"], list(source_rx2)
    )
    target = _subset(target, supported)
    prediction = _predict(
        "categorical_ordered",
        fitted,
        target["gain1"],
        target["gain2"],
        config=config,
    )
    raw_residual = _wrap(target["phase"] - prediction)
    optimal_shift = float(np.angle(np.mean(np.exp(1j * raw_residual))))
    result: dict[str, Any] = {
        "supported_target_observations": int(target["phase"].size),
        "unanchored": _metrics(raw_residual),
        "optimal_intercept_lower_bound": {
            **_metrics(_wrap(raw_residual - optimal_shift)),
            "intercept_shift_deg": float(np.degrees(optimal_shift)),
            "warning": (
                "Uses every target observation to align the intercept and is "
                "therefore a descriptive lower bound, not an operational policy."
            ),
        },
    }
    anchor_policies = {
        "single_26_db_equal_gain": (26,),
        "five_equal_gain_anchors": DEFAULT_ANCHOR_GAINS_DB,
    }
    for name, anchor_gains in anchor_policies.items():
        anchors = np.isin(target["gain1"], anchor_gains) & (
            target["gain1"] == target["gain2"]
        )
        if not np.any(anchors):
            result[name] = None
            continue
        anchor_shift = float(np.angle(np.mean(np.exp(1j * raw_residual[anchors]))))
        scored = ~anchors
        result[name] = {
            "anchor_gains_db": sorted(
                set(int(value) for value in target["gain1"][anchors])
            ),
            "anchor_observations": int(np.count_nonzero(anchors)),
            "scored_observations": int(np.count_nonzero(scored)),
            "source_shape_plus_anchor": _metrics(
                _wrap(raw_residual[scored] - anchor_shift)
            ),
            "constant_anchor_only": _metrics(
                _wrap(
                    target["phase"][scored]
                    - float(np.angle(np.mean(np.exp(1j * target["phase"][anchors]))))
                )
            ),
            "intercept_shift_deg": float(np.degrees(anchor_shift)),
        }

    epoch_residuals = []
    anchor_observations = 0
    anchored_epochs = []
    for epoch in sorted(set(int(value) for value in target["epoch"])):
        in_epoch = target["epoch"] == epoch
        anchors = in_epoch & (target["gain1"] == 26) & (target["gain2"] == 26)
        scored = in_epoch & ~anchors
        if not np.any(anchors) or not np.any(scored):
            continue
        anchor_shift = float(np.angle(np.mean(np.exp(1j * raw_residual[anchors]))))
        epoch_residuals.append(_wrap(raw_residual[scored] - anchor_shift))
        anchor_observations += int(np.count_nonzero(anchors))
        anchored_epochs.append(epoch)
    result["one_26_db_anchor_per_epoch"] = (
        {
            "anchor_gain_db": 26,
            "anchored_epochs": anchored_epochs,
            "anchor_observations": anchor_observations,
            "scored_observations": int(
                sum(residual.size for residual in epoch_residuals)
            ),
            "source_shape_plus_epoch_anchor": _metrics(np.concatenate(epoch_residuals)),
            "policy": (
                "Fit one intercept from the target radio's quality-valid 26/26 "
                "frame in each epoch and exclude every anchor from scoring."
            ),
        }
        if epoch_residuals
        else None
    )
    return result


def _weighted_metric_summary(
    rows: list[tuple[int, dict[str, float]]],
) -> dict[str, Any] | None:
    """Pool count-weighted MAE/RMSE summaries without inventing a pooled p95."""

    observations = sum(count for count, _ in rows)
    if not observations:
        return None
    return {
        "n_observations": observations,
        "circular_mae_deg": float(
            sum(count * metrics["circular_mae_deg"] for count, metrics in rows)
            / observations
        ),
        "circular_rmse_deg": float(
            math.sqrt(
                sum(
                    count * metrics["circular_rmse_deg"] ** 2 for count, metrics in rows
                )
                / observations
            )
        ),
        "circular_max_deg": float(
            max(metrics["circular_max_deg"] for _, metrics in rows)
        ),
        "p95_available": False,
    }


def _cross_radio_transfer_summary(
    transfers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize same-frequency transfer by direction and gain-table region."""

    regions = {
        "all_frequencies": lambda frequency: True,
        "low_full_gain_table": lambda frequency: frequency <= 1_300_000_000,
        "middle_full_gain_table": lambda frequency: (
            1_300_000_000 < frequency <= 4_000_000_000
        ),
        "high_full_gain_table": lambda frequency: frequency > 4_000_000_000,
        "vtx_5_7_to_5_9_ghz": lambda frequency: (
            5_700_000_000 <= frequency <= 5_900_000_000
        ),
    }
    directions = sorted(key.rsplit(":", 1)[0] for key in transfers)
    result = {}
    for direction in sorted(set(directions)):
        direction_rows = [
            (int(key.rsplit(":", 1)[1]), value)
            for key, value in transfers.items()
            if key.rsplit(":", 1)[0] == direction
        ]
        region_results = {}
        for region, includes in regions.items():
            selected = [
                value for frequency, value in direction_rows if includes(frequency)
            ]
            if not selected:
                continue
            whole_run_rows = [
                (
                    row["single_26_db_equal_gain"]["scored_observations"],
                    row["single_26_db_equal_gain"]["source_shape_plus_anchor"],
                )
                for row in selected
                if row["single_26_db_equal_gain"] is not None
            ]
            per_epoch_rows = [
                (
                    row["one_26_db_anchor_per_epoch"]["scored_observations"],
                    row["one_26_db_anchor_per_epoch"]["source_shape_plus_epoch_anchor"],
                )
                for row in selected
                if row["one_26_db_anchor_per_epoch"] is not None
            ]
            region_results[region] = {
                "frequency_count": len(selected),
                "one_26_db_anchor_over_whole_run": _weighted_metric_summary(
                    whole_run_rows
                ),
                "one_26_db_anchor_per_epoch": _weighted_metric_summary(per_epoch_rows),
            }
        result[direction] = region_results
    return result


def _decode_stage_parameters(
    parameters: np.ndarray, *, config: CalibrationConfig
) -> dict[str, Any]:
    boundaries = _stage_boundaries(config)
    expected = 3 + 2 * len(boundaries)
    if parameters.size != expected:
        raise ValueError(f"stage model has {parameters.size} != {expected} parameters")
    boundary_count = len(boundaries)
    return {
        "intercept_rad": float(_wrap(parameters[0])),
        "rx1_linear_rad_per_db": float(parameters[1] / 20.0),
        "rx2_linear_rad_per_db": float(parameters[2] / 20.0),
        "rx1_step_rad_by_gain_boundary": {
            str(boundary): float(parameters[3 + index])
            for index, boundary in enumerate(boundaries)
        },
        "rx2_step_rad_by_gain_boundary": {
            str(boundary): float(parameters[3 + boundary_count + index])
            for index, boundary in enumerate(boundaries)
        },
    }


def _decode_categorical_parameters(
    parameters: np.ndarray, *, config: CalibrationConfig
) -> dict[str, Any]:
    reference = _reference_gain(config)
    non_reference = [gain for gain in config.gains_db if gain != reference]
    expected = 1 + 2 * len(non_reference)
    if parameters.size != expected:
        raise ValueError(
            f"categorical model has {parameters.size} != {expected} parameters"
        )
    rx1 = {str(reference): 0.0}
    rx2 = {str(reference): 0.0}
    for index, gain in enumerate(non_reference):
        rx1[str(gain)] = float(parameters[1 + index])
        rx2[str(gain)] = float(parameters[1 + len(non_reference) + index])
    return {
        "reference_gain_db": reference,
        "intercept_rad": float(_wrap(parameters[0])),
        "rx1_effect_rad_by_gain_db": dict(
            sorted(rx1.items(), key=lambda item: int(item[0]))
        ),
        "rx2_effect_rad_by_gain_db": dict(
            sorted(rx2.items(), key=lambda item: int(item[0]))
        ),
    }


def _pair_support(
    group: dict[str, np.ndarray], *, config: CalibrationConfig
) -> dict[str, Any]:
    count = np.zeros((len(config.gains_db), len(config.gains_db)), dtype=np.uint8)
    repeat_std = np.full(count.shape, np.nan, dtype=np.float64)
    gain_lookup = {gain: index for index, gain in enumerate(config.gains_db)}
    for gain1 in config.gains_db:
        for gain2 in config.gains_db:
            selected = (group["gain1"] == gain1) & (group["gain2"] == gain2)
            phases = group["phase"][selected]
            i = gain_lookup[gain1]
            j = gain_lookup[gain2]
            count[i, j] = phases.size
            if phases.size >= 2:
                stats = circular_stats(phases)
                if stats["circular_std_rad"] is not None:
                    repeat_std[i, j] = math.degrees(stats["circular_std_rad"])
    supported = (
        (count >= config.min_quality_valid_per_cell)
        & np.isfinite(repeat_std)
        & (repeat_std <= config.max_across_repeat_phase_std_deg)
    )
    supported_pairs = [
        [int(gain1), int(gain2)]
        for gain1 in config.gains_db
        for gain2 in config.gains_db
        if supported[gain_lookup[gain1], gain_lookup[gain2]]
    ]
    return {
        "observed_ordered_gain_pair_count": int(np.count_nonzero(count)),
        "production_supported_ordered_gain_pairs": supported_pairs,
        "production_supported_pair_count": len(supported_pairs),
    }


def _calibration_for_radio(
    radio: dict[str, Any],
    group_results: dict[int, dict[str, Any]],
    *,
    config: CalibrationConfig,
) -> dict[str, Any]:
    models = []
    full_frequencies = sorted(group_results)
    for frequency_hz in full_frequencies:
        group = _select_group(radio, frequency_hz)
        stage = _fit_circular(
            "stage_ordered",
            group["gain1"],
            group["gain2"],
            group["phase"],
            config=config,
        )
        categorical = _fit_circular(
            "categorical_ordered",
            group["gain1"],
            group["gain2"],
            group["phase"],
            config=config,
        )
        support = _pair_support(group, config=config)
        epochs = sorted(set(int(value) for value in group["epoch"]))
        models.append(
            {
                "frequency_hz": frequency_hz,
                "complete_epochs_used": epochs,
                "quality_valid_observations": int(group["phase"].size),
                "deployable": bool(support["production_supported_pair_count"]),
                "ordered_stage_model": {
                    "formula": (
                        "phase = intercept + rx1_linear*gain1 + "
                        "rx2_linear*gain2 + sum(rx1_step[gain1>=boundary]) + "
                        "sum(rx2_step[gain2>=boundary])"
                    ),
                    "parameters": _decode_stage_parameters(stage, config=config),
                    "training_metrics": _metrics(
                        _wrap(
                            group["phase"]
                            - _predict(
                                "stage_ordered",
                                stage,
                                group["gain1"],
                                group["gain2"],
                                config=config,
                            )
                        )
                    ),
                    "held_out_cell_metrics": group_results[frequency_hz][
                        "random_cell_five_fold"
                    ]["models"]["stage_ordered"],
                    "held_out_quadrant_metrics": group_results[frequency_hz][
                        "quadrant_holdout"
                    ]["models"]["stage_ordered"],
                },
                "ordered_categorical_additive_model": {
                    "formula": (
                        "phase = intercept + RX1_effect[gain1] + " "RX2_effect[gain2]"
                    ),
                    "parameters": _decode_categorical_parameters(
                        categorical, config=config
                    ),
                    "training_metrics": _metrics(
                        _wrap(
                            group["phase"]
                            - _predict(
                                "categorical_ordered",
                                categorical,
                                group["gain1"],
                                group["gain2"],
                                config=config,
                            )
                        )
                    ),
                    "held_out_cell_metrics": group_results[frequency_hz][
                        "random_cell_five_fold"
                    ]["models"]["categorical_ordered"],
                    "held_out_quadrant_metrics": group_results[frequency_hz][
                        "quadrant_holdout"
                    ]["models"]["categorical_ordered"],
                },
                **support,
            }
        )
    all_supported = sum(model["production_supported_pair_count"] for model in models)
    complete_epoch_counts = [len(model["complete_epochs_used"]) for model in models]
    minimum_complete_epochs = min(complete_epoch_counts) if complete_epoch_counts else 0
    if all_supported:
        status = "candidate_with_repeat_supported_cells"
        warning = (
            "Only ordered gain pairs marked production-supported satisfy the "
            "repeatability gate. Full-IQ validation and live metadata/quality "
            "checks remain mandatory."
        )
    elif minimum_complete_epochs <= 1:
        status = "preliminary_single_epoch"
        warning = (
            "Only one complete epoch is available at one or more fitted "
            "frequencies. No ordered gain pair yet satisfies the configured "
            "repeat criterion; coefficients are engineering candidates, not "
            "production corrections."
        )
    else:
        status = "insufficient_repeat_support"
        warning = (
            "Multiple epochs were analyzed, but no ordered gain pair satisfies "
            "the configured repeatability gate. Do not deploy these coefficients."
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": status,
        "deployable": bool(all_supported),
        "serial": radio["serial"],
        "phase_convention": radio["attrs"]["phase_convention"],
        "gain_values_db": list(config.gains_db),
        "input": {
            "dataset_path": radio["dataset_path"],
            "analysis_input_sha256": radio["analysis_input_sha256"],
            **radio["attrs"],
        },
        "correction_rule": (
            "corrected_phase = wrap(measured_RX1_minus_RX2 - predicted_phase)"
        ),
        "support_policy": {
            "minimum_quality_valid_epochs": config.min_quality_valid_per_cell,
            "maximum_repeat_circular_std_deg": (config.max_across_repeat_phase_std_deg),
            "exact_frequency_and_ordered_gain_pair_required": True,
            "interpolation_or_extrapolation_allowed": False,
        },
        "warning": warning,
        "frequency_models": models,
    }


def analyze_artifact(
    *,
    config_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Analyze all serial-specific V7 datasets below one run root."""

    _, config = load_calibration_document(config_path)
    radio_paths = sorted(artifact_root.glob("*/calibration.v7.zarr"))
    if not radio_paths:
        raise FileNotFoundError(f"no serial-specific V7 datasets under {artifact_root}")
    radios = [_load_radio(path, config=config) for path in radio_paths]
    serials = [radio["serial"] for radio in radios]
    if len(serials) != len(set(serials)):
        raise ValueError("duplicate Pluto serials in artifact root")

    group_results_by_radio: dict[str, dict[int, dict[str, Any]]] = {}
    for radio in radios:
        completed_frequencies = sorted(
            {
                int(block["frequency_hz"])
                for block in radio["full_blocks"]
                if not block.get("excluded_as_partial")
            }
        )
        frequency_results: dict[int, dict[str, Any]] = {}
        for frequency_hz in completed_frequencies:
            group = _select_group(radio, frequency_hz)
            if not group["phase"].size:
                continue
            frequency_results[frequency_hz] = {
                "quality_valid_observations": int(group["phase"].size),
                "complete_epochs": sorted(set(int(value) for value in group["epoch"])),
                "random_cell_five_fold": _random_cell_cv(group, config=config),
                "quadrant_holdout": _quadrant_cv(group, config=config),
                "time_and_order": _time_diagnostics(group, config=config),
            }
        group_results_by_radio[radio["serial"]] = frequency_results

    cross_frequency: dict[str, Any] = {}
    for radio in radios:
        frequencies = sorted(group_results_by_radio[radio["serial"]])
        for source_frequency in frequencies:
            for target_frequency in frequencies:
                if source_frequency == target_frequency:
                    continue
                key = f"{radio['serial']}:{source_frequency}->{target_frequency}"
                cross_frequency[key] = _transfer(
                    _select_group(radio, source_frequency),
                    _select_group(radio, target_frequency),
                    config=config,
                )

    cross_radio: dict[str, Any] = {}
    for source_radio in radios:
        for target_radio in radios:
            if source_radio["serial"] == target_radio["serial"]:
                continue
            shared_frequencies = sorted(
                set(group_results_by_radio[source_radio["serial"]])
                & set(group_results_by_radio[target_radio["serial"]])
            )
            for frequency_hz in shared_frequencies:
                key = (
                    f"{source_radio['serial']}->{target_radio['serial']}:"
                    f"{frequency_hz}"
                )
                cross_radio[key] = _transfer(
                    _select_group(source_radio, frequency_hz),
                    _select_group(target_radio, frequency_hz),
                    config=config,
                )

    complete_capture = all(
        sum(not block.get("excluded_as_partial") for block in radio["full_blocks"])
        == config.repetitions * len(config.frequencies_hz)
        for radio in radios
    )
    complete_epoch_counts = [
        len(result["complete_epochs"])
        for frequency_results in group_results_by_radio.values()
        for result in frequency_results.values()
    ]
    minimum_complete_epochs = min(complete_epoch_counts, default=0)
    limitations = [
        "Only completely captured epoch/frequency blocks are analyzed.",
        "Model comparison trusts stored quality decisions and phase values; "
        "run strict validation with IQ recomputation before deployment.",
        "Weak-signal and rail/DC failures remain explicit unsupported cells.",
    ]
    if minimum_complete_epochs < config.repetitions:
        limitations.insert(
            1,
            f"The least-covered fitted radio/frequency currently has "
            f"{minimum_complete_epochs}/{config.repetitions} complete epochs; "
            "remaining epochs are required for final promotion.",
        )
    limitations.insert(
        2,
        "The compact stage basis was developed during epoch-0 exploration; "
        "repeat epochs must be treated as confirmatory evidence.",
    )
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "analysis_status": (
            "complete_capture_candidate"
            if complete_capture
            else "partial_capture_engineering_evidence"
        ),
        "configured_repetitions": config.repetitions,
        "config_path": str(config_path),
        "artifact_root": _portable_path(artifact_root),
        "stage_boundaries_db": list(_stage_boundaries(config)),
        "stage_boundary_provenance": {
            "linux_git_sha": GAIN_TABLE_LINUX_GIT_SHA,
            "source_path": GAIN_TABLE_SOURCE_PATH,
            "high_band_requested_gain_range_db": [
                HIGH_BAND_GAIN_MIN_DB,
                HIGH_BAND_GAIN_MIN_DB + len(HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN) - 1,
            ],
            "derivation": (
                "Starts of LNA/mixer-byte plateaus lasting at least three "
                "requested gain states. The 52..62 dB one-index-per-dB mixer "
                "ramp is represented by the linear term."
            ),
            "selection_warning": (
                "This exploratory basis was designed after epoch-0 inspection. "
                "Its coefficients and model choice require confirmation on "
                "the untouched repeat epochs."
            ),
        },
        "radios": {
            radio["serial"]: {
                "dataset_path": radio["dataset_path"],
                "analysis_input_sha256": radio["analysis_input_sha256"],
                "completed_frames": radio["completed_frames"],
                "full_and_partial_blocks": radio["full_blocks"],
                "provenance": radio["attrs"],
                "frequency_results": {
                    str(frequency): value
                    for frequency, value in group_results_by_radio[
                        radio["serial"]
                    ].items()
                },
            }
            for radio in radios
        },
        "cross_frequency_transfer": cross_frequency,
        "cross_radio_transfer": cross_radio,
        "cross_radio_transfer_summary": _cross_radio_transfer_summary(cross_radio),
        "minimum_complete_epochs_per_fitted_radio_frequency": (minimum_complete_epochs),
        "limitations": limitations,
    }
    calibrations = {
        radio["serial"]: _calibration_for_radio(
            radio,
            group_results_by_radio[radio["serial"]],
            config=config,
        )
        for radio in radios
    }
    return comparison, calibrations


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _short_serial(serial: str) -> str:
    return f"…{serial[-12:]}"


def render_comparative_report(
    comparison: dict[str, Any],
    calibrations: dict[str, dict[str, Any]],
    *,
    reproduce_command: str,
) -> str:
    """Render a concise engineering report from machine-readable results."""

    minimum_epochs = comparison["minimum_complete_epochs_per_fitted_radio_frequency"]
    configured_epochs = comparison["configured_repetitions"]
    lines = [
        "# Dual-RX gain/frequency phase model comparison",
        "",
        f"> Status: `{comparison['analysis_status']}`. The least-covered fitted "
        f"radio/frequency has {minimum_epochs}/{configured_epochs} complete "
        "epochs. Coefficients remain engineering candidates until the configured "
        "capture and strict full-IQ validation are complete.",
        "",
        "## Reproduce",
        "",
        "From the SPF repository root, with the source V7 artifact present:",
        "",
        "```bash",
        reproduce_command,
        "```",
        "",
        "The command reads no IQ into the optimiser and makes no radio calls. It "
        "hashes every V7 scalar array used by the analysis. Full-IQ verification "
        "is a separate mandatory `validate` step.",
        "",
        "## Input checkpoint",
        "",
        "| Pluto serial | Completed frames | Scalar-input SHA-256 | Complete blocks used |",
        "|---|---:|---|---|",
    ]
    for serial, radio in comparison["radios"].items():
        complete = [
            block
            for block in radio["full_and_partial_blocks"]
            if not block.get("excluded_as_partial")
        ]
        blocks = ", ".join(
            f"epoch {block['epoch']} @ {block['frequency_hz']/1e6:.0f} MHz"
            for block in complete
        )
        lines.append(
            f"| `{serial}` | {radio['completed_frames']} | "
            f"`{radio['analysis_input_sha256']}` | {blocks} |"
        )

    lines.extend(
        [
            "",
            "## Errors explained by competing models",
            "",
            "Every table below uses five deterministic folds that hold out complete "
            "ordered gain-pair cells. A repeated cell is never split across train "
            "and test. “Reduction” is the held-out MAE reduction relative to one "
            "constant phase at the same radio and frequency.",
        ]
    )
    for serial, radio in comparison["radios"].items():
        for frequency_text, result in radio["frequency_results"].items():
            lines.extend(
                [
                    "",
                    f"### {_short_serial(serial)} at {int(frequency_text)/1e6:.0f} MHz",
                    "",
                    "| Model | Parameters | Held-out MAE | p95 | MAE reduction |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            models = result["random_cell_five_fold"]["models"]
            for model in MODEL_ORDER:
                row = models[model]
                lines.append(
                    f"| {row['label']} | {row['nominal_parameters']} | "
                    f"{_fmt(row['circular_mae_deg'])}° | "
                    f"{_fmt(row['circular_p95_deg'])}° | "
                    f"{row['mae_reduction_vs_constant_fraction']:.1%} |"
                )
            quadrant = result["quadrant_holdout"]["models"]
            lines.extend(
                [
                    "",
                    "A harder test removes an entire low/high RX1-by-RX2 quadrant "
                    "while leaving each gain state represented elsewhere:",
                    "",
                    "| Model | Held-out quadrant MAE | p95 |",
                    "|---|---:|---:|",
                ]
            )
            for model in ("stage_shared", "stage_ordered", "categorical_ordered"):
                row = quadrant[model]
                if row:
                    lines.append(
                        f"| {row['label']} | "
                        f"{_fmt(row['circular_mae_deg'])}° | "
                        f"{_fmt(row['circular_p95_deg'])}° |"
                    )
                else:
                    lines.append(f"| {MODEL_LABELS[model]} | n/a | n/a |")
            drift = result["time_and_order"]
            lines.append(
                f"\nResidual linear drift was "
                f"{drift['linear_residual_drift_deg_per_hour']:+.2f}°/hour; "
                "the report does not interpret the small rank correlations as "
                "temperature because no temperature channel was recorded."
            )

    first_models = next(
        result["random_cell_five_fold"]["models"]
        for radio in comparison["radios"].values()
        for result in radio["frequency_results"].values()
    )
    stage_parameters = first_models["stage_ordered"]["nominal_parameters"]
    categorical_parameters = first_models["categorical_ordered"]["nominal_parameters"]
    gain_count = len(next(iter(calibrations.values()))["gain_values_db"])
    boundaries = comparison["stage_boundaries_db"]
    boundary_text = ", ".join(str(value) for value in boundaries) or "none"
    lines.extend(
        [
            "",
            "## Parsimonious interpretation",
            "",
            f"The {stage_parameters}-parameter ordered stage-boundary model is "
            "the compact candidate evaluated against the exact categorical "
            "reference:",
            "",
            "```text",
            "phase(f,g1,g2) = intercept(f)",
            "                 + linear_RX1(f)*g1 + stage_steps_RX1(f,g1)",
            "                 + linear_RX2(f)*g2 + stage_steps_RX2(f,g2)",
            "                 + residual",
            "```",
            "",
            f"Its configured in-range boundaries (`{boundary_text} dB`) are "
            "selected from "
            "starts of LNA/mixer-byte plateaus lasting at least three requested "
            "gain states in `drivers/iio/adc/ad9361.c` at Linux commit "
            f"`{GAIN_TABLE_LINUX_GIT_SHA}`. The final 52–62 dB one-index-per-dB "
            "mixer ramp is represented by the linear term when it is present in "
            "the configured grid. RX1 and RX2 retain separate effects. The "
            f"{categorical_parameters}-parameter categorical model remains the "
            "exact-grid reference; held-out tables above, rather than parameter "
            "count alone, decide whether the compact approximation is adequate.",
            "",
            "Important: this compact basis was developed during epoch-0 "
            "exploration, so its selection is post-hoc. Completed repeat blocks "
            "add evidence, but the remaining epoch and final dense grid must "
            "provide confirmatory model-selection evidence.",
            "",
            "## Preliminary calibration artifacts",
            "",
            "One machine-readable JSON calibration is emitted per serial. Each "
            "contains both the compact stage model and exact categorical effects, "
            "plus a fail-closed list of production-supported ordered pairs.",
            "",
            "| Pluto serial | Frequencies fitted | Production-supported pairs | Status |",
            "|---|---|---:|---|",
        ]
    )
    for serial, calibration in calibrations.items():
        frequencies = ", ".join(
            f"{row['frequency_hz']/1e6:.0f} MHz"
            for row in calibration["frequency_models"]
        )
        supported = sum(
            row["production_supported_pair_count"]
            for row in calibration["frequency_models"]
        )
        lines.append(
            f"| `{serial}` | {frequencies} | {supported} | "
            f"`{calibration['status']}` |"
        )

    lines.extend(
        [
            "",
            "### Compact coefficients by radio and frequency",
            "",
            "Slopes and steps are shown in degrees. The exact-radian values and "
            f"the {gain_count}-state categorical curves are in the linked JSON "
            "artifacts.",
            "",
            "| Radio / calibration | Frequency | Intercept | RX1 / RX2 slope | RX1 stage steps (boundary:value) | RX2 stage steps |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for serial, calibration in calibrations.items():
        for row in calibration["frequency_models"]:
            parameters = row["ordered_stage_model"]["parameters"]
            rx1_steps = parameters["rx1_step_rad_by_gain_boundary"]
            rx2_steps = parameters["rx2_step_rad_by_gain_boundary"]
            boundary_order = sorted(rx1_steps, key=int)
            rx1_text = ", ".join(
                f"{boundary}:{math.degrees(rx1_steps[boundary]):+.2f}°"
                for boundary in boundary_order
            )
            rx2_text = ", ".join(
                f"{boundary}:{math.degrees(rx2_steps[boundary]):+.2f}°"
                for boundary in boundary_order
            )
            lines.append(
                f"| [{_short_serial(serial)}](calibrations/{serial}.json) | "
                f"{row['frequency_hz']/1e6:.0f} MHz | "
                f"{math.degrees(parameters['intercept_rad']):+.2f}° | "
                f"{math.degrees(parameters['rx1_linear_rad_per_db']):+.3f} / "
                f"{math.degrees(parameters['rx2_linear_rad_per_db']):+.3f}°/dB | "
                f"{rx1_text} | {rx2_text} |"
            )

    total_supported = sum(
        row["production_supported_pair_count"]
        for calibration in calibrations.values()
        for row in calibration["frequency_models"]
    )
    support_text = (
        f"The current complete blocks provide {total_supported} "
        "radio/frequency/ordered-pair entries that pass the configured repeat "
        "support gate. They remain conditional on strict full-IQ validation and "
        "live signal/metadata quality."
        if total_supported
        else (
            "No current ordered pair passes the configured repeat-support gate. "
            f"The least-covered fitted radio/frequency has "
            f"{minimum_epochs}/{configured_epochs} complete epochs."
        )
    )
    lines.extend(
        [
            "",
            support_text,
            "",
            "## Cross-radio transfer summary",
            "",
            "These are count-weighted summaries of same-frequency categorical "
            "gain-shape transfer. A pooled p95 cannot be reconstructed from "
            "per-frequency summaries, so MAE, RMSE, and maximum are reported.",
            "",
            "| Direction | Region | Anchor policy | Frames | MAE | RMSE | Max |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for direction, regions in comparison["cross_radio_transfer_summary"].items():
        for region, policies in regions.items():
            for policy_name, policy_label in (
                ("one_26_db_anchor_over_whole_run", "one 26/26 over whole run"),
                ("one_26_db_anchor_per_epoch", "one 26/26 per epoch"),
            ):
                metrics = policies[policy_name]
                if metrics is None:
                    continue
                lines.append(
                    f"| `{direction}` | {region.replace('_', ' ')} | "
                    f"{policy_label} | {metrics['n_observations']} | "
                    f"{_fmt(metrics['circular_mae_deg'])}° | "
                    f"{_fmt(metrics['circular_rmse_deg'])}° | "
                    f"{_fmt(metrics['circular_max_deg'])}° |"
                )

    lines.extend(
        [
            "",
            "## Transfer to another frequency or radio",
            "",
            "The following figures transfer a complete categorical gain shape. "
            "“Optimal” uses all target observations to align the intercept and is "
            "only a descriptive lower bound. The per-epoch column uses one "
            "quality-valid 26/26 frame from the target radio in each epoch, excludes "
            "those anchors from scoring, and represents a same-session policy. "
            "Equal-gain anchors do not replace validation.",
            "",
            "| Transfer | Unanchored MAE | One whole-run anchor MAE | One anchor/epoch MAE | Five whole-run anchors MAE | Optimal-intercept MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, result in {
        **comparison["cross_frequency_transfer"],
        **comparison["cross_radio_transfer"],
    }.items():
        single = result.get("single_26_db_equal_gain")
        per_epoch = result.get("one_26_db_anchor_per_epoch")
        five = result.get("five_equal_gain_anchors")
        lines.append(
            f"| `{key}` | {_fmt(result['unanchored']['circular_mae_deg'])}° | "
            f"{_fmt(single['source_shape_plus_anchor']['circular_mae_deg']) if single else 'n/a'}° | "
            f"{_fmt(per_epoch['source_shape_plus_epoch_anchor']['circular_mae_deg']) if per_epoch else 'n/a'}° | "
            f"{_fmt(five['source_shape_plus_anchor']['circular_mae_deg']) if five else 'n/a'}° | "
            f"{_fmt(result['optimal_intercept_lower_bound']['circular_mae_deg'])}° |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Previously calibrated (“seen”) radio",
            "",
            "After all three epochs and full-IQ validation pass, use that serial’s "
            "exact-frequency, ordered RX1/RX2 calibration. Apply "
            "`wrap(measured_RX1_minus_RX2 - predicted_phase)`. Fail closed for a "
            "weak/clipped live frame, invalid gain metadata, an unvalidated gain "
            "pair, a different frequency, or a gain-change event. Keep the compact "
            "stage model as the preferred explanation and smoothing diagnostic; "
            "the exact-grid additive model is the conservative operational table.",
            "",
            "A reboot or materially different temperature is not yet a “seen” "
            "condition. Begin a session with several equal-gain anchors spanning "
            "the stage boundaries and reject the stored table if their residuals "
            "are inconsistent. The planned repeated epochs and reboot/temperature "
            "checks must quantify the threshold.",
            "",
            "### New (“unseen”) radio with anchor measurements",
            "",
            "Do not silently label a transferred model as calibrated. The current "
            "two-radio transfer shows that a source-radio gain shape plus anchors "
            "is useful but leaves several degrees of error unless the target "
            "intercept is refreshed in the same session. A per-session 26/26 dB "
            "anchor addresses intercept drift but cannot reveal gain-state-specific "
            "disagreement; distributed equal-gain anchors are preferable when time "
            "allows because they also exercise multiple gain states. "
            "Use the transferred result only with an explicit lower-confidence "
            "flag and collect that serial’s full calibration when precision matters.",
            "",
            "### New radio without any anchor",
            "",
            "Do not apply another serial’s absolute phase correction. Radio-to-radio "
            "baseline shifts are large enough to dominate the residual. At most, "
            "use the population/stage shape as a prior for experiment design; report "
            "the phase as uncalibrated until at least an intercept anchor is measured.",
            "",
            "### New RF frequency",
            "",
            "The nearby-frequency transfer is better than a constant phase but still "
            "roughly two to three times worse than a same-frequency model. Measure "
            "the requested frequency. A linear effective-delay baseline may reduce "
            "the number of frequency anchors only after the full four-frequency, "
            "three-epoch dataset validates it.",
            "",
            "## Required next evidence",
            "",
            "1. Diagnose the RX2 high-gain DC/rail condition with matched TX2-on and "
            "TX2-off captures before spending the remaining exhaustive epochs.",
            "2. Resume this checkpoint only if preparation semantics remain "
            "unchanged; otherwise start a clean V7 artifact.",
            "3. Complete three epochs at all four frequencies for both serials.",
            "4. Recompute every stored phase and quality decision from full IQ.",
            "5. Run leave-one-epoch-out, reboot, and temperature/anchor validation.",
            "6. Promote only exact serial/frequency/gain cells that pass the support "
            "policy into a production correction artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparative_bundle(
    *,
    config_path: Path,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write comparison JSON, per-radio calibrations, and Markdown report."""

    comparison, calibrations = analyze_artifact(
        config_path=config_path,
        artifact_root=artifact_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir = output_dir / "calibrations"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparative_analysis.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    calibration_files = {}
    for serial, calibration in calibrations.items():
        path = calibration_dir / f"{serial}.json"
        path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
        calibration_files[serial] = str(path)

    reproduce_command = (
        "python -m spf.calibrations.dual_rx_gain_frequency compare-models \\\n"
        f"  --config {config_path} \\\n"
        f"  --artifact-root {_portable_path(artifact_root)} \\\n"
        f"  --output-dir {output_dir}"
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        render_comparative_report(
            comparison,
            calibrations,
            reproduce_command=reproduce_command,
        )
    )
    return {
        "comparison": str(comparison_path),
        "report": str(report_path),
        "calibrations": calibration_files,
    }
