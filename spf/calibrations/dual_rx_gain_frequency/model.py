"""Circular additive phase model for validated gain/frequency calibration data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import circular_stats, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def _circular_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * values))))


def fit_additive_surface(
    phase: np.ndarray,
    gain1_index: np.ndarray,
    gain2_index: np.ndarray,
    gain_count: int,
    *,
    iterations: int = 50,
) -> dict[str, Any]:
    """Fit phase ~= intercept + RX1(g1) + RX2(g2) on the unit circle."""

    phase = np.asarray(phase, dtype=np.float64)
    gain1_index = np.asarray(gain1_index, dtype=np.int64)
    gain2_index = np.asarray(gain2_index, dtype=np.int64)
    if not (
        phase.ndim == gain1_index.ndim == gain2_index.ndim == 1
        and phase.size == gain1_index.size == gain2_index.size
    ):
        raise ValueError("phase and gain indices must be equal-length vectors")
    if phase.size == 0:
        raise ValueError("cannot fit an empty phase surface")
    intercept = _circular_mean(phase)
    rx1 = np.zeros(gain_count, dtype=np.float64)
    rx2 = np.zeros(gain_count, dtype=np.float64)
    for _ in range(iterations):
        for index in range(gain_count):
            selected = gain1_index == index
            if np.any(selected):
                rx1[index] = _circular_mean(
                    wrap_phase(phase[selected] - intercept - rx2[gain2_index[selected]])
                )
        for index in range(gain_count):
            selected = gain2_index == index
            if np.any(selected):
                rx2[index] = _circular_mean(
                    wrap_phase(phase[selected] - intercept - rx1[gain1_index[selected]])
                )
        residual_without_intercept = wrap_phase(
            phase - rx1[gain1_index] - rx2[gain2_index]
        )
        intercept = _circular_mean(residual_without_intercept)

    common_indices = np.intersect1d(np.unique(gain1_index), np.unique(gain2_index))
    if common_indices.size == 0:
        raise ValueError(
            "additive surface is not identifiable: no gain appears in both roles"
        )
    reference = int(common_indices[np.argmin(np.abs(common_indices - gain_count // 2))])
    intercept = float(wrap_phase(intercept + rx1[reference] + rx2[reference]))
    rx1 = wrap_phase(rx1 - rx1[reference])
    rx2 = wrap_phase(rx2 - rx2[reference])
    prediction = wrap_phase(intercept + rx1[gain1_index] + rx2[gain2_index])
    residual = wrap_phase(phase - prediction)
    return {
        "intercept_rad": intercept,
        "reference_gain_index": reference,
        "rx1_effect_rad": rx1,
        "rx2_effect_rad": rx2,
        "prediction_rad": prediction,
        "residual_rad": residual,
    }


def _residual_metrics(residual: np.ndarray) -> dict[str, float]:
    degrees = np.abs(np.degrees(np.asarray(residual)))
    return {
        "circular_mae_deg": float(np.mean(degrees)),
        "circular_rmse_deg": float(np.sqrt(np.mean(degrees**2))),
        "circular_p95_deg": float(np.percentile(degrees, 95)),
        "circular_max_deg": float(np.max(degrees)),
    }


def fit_dataset(path: Path, *, config: CalibrationConfig) -> dict[str, Any]:
    """Fit one additive gain model per frequency and leave-one-epoch-out CV."""

    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        completed = np.asarray(receiver.sweep_completed[:], dtype=bool)
        valid = completed & np.asarray(receiver.sweep_quality_valid[:], dtype=bool)
        phases = np.asarray(receiver.phase_difference_rad[:], dtype=np.float64)
        epochs = np.asarray(receiver.sweep_epoch[:], dtype=np.int64)
        frequency_indices = np.asarray(
            receiver.sweep_frequency_index[:], dtype=np.int64
        )
        requested = np.asarray(receiver.sweep_requested_gain_db[:], dtype=np.int64)
        gain_lookup = {gain: index for index, gain in enumerate(config.gains_db)}
        gain1_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 0]]
        )
        gain2_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 1]]
        )
        if np.any(valid & ((gain1_indices < 0) | (gain2_indices < 0))):
            raise ValueError("dataset contains gains outside the configured set")

        frequency_models = []
        cv_residuals = []
        for frequency_index, frequency_hz in enumerate(config.frequencies_hz):
            selected = valid & (frequency_indices == frequency_index)
            if not np.any(selected):
                frequency_models.append(
                    {
                        "frequency_hz": frequency_hz,
                        "status": "no_quality_valid_data",
                    }
                )
                continue
            fitted = fit_additive_surface(
                phases[selected],
                gain1_indices[selected],
                gain2_indices[selected],
                len(config.gains_db),
            )
            rx1_counts = np.bincount(
                gain1_indices[selected], minlength=len(config.gains_db)
            )
            rx2_counts = np.bincount(
                gain2_indices[selected], minlength=len(config.gains_db)
            )
            rx1_effect = [
                float(value) if rx1_counts[index] else None
                for index, value in enumerate(fitted["rx1_effect_rad"])
            ]
            rx2_effect = [
                float(value) if rx2_counts[index] else None
                for index, value in enumerate(fitted["rx2_effect_rad"])
            ]
            cell_residuals: dict[tuple[int, int], list[float]] = {}
            selected_indices = np.flatnonzero(selected)
            for source_index, residual in zip(selected_indices, fitted["residual_rad"]):
                pair = (
                    int(requested[source_index, 0]),
                    int(requested[source_index, 1]),
                )
                cell_residuals.setdefault(pair, []).append(float(residual))
            interaction = np.full((len(config.gains_db), len(config.gains_db)), np.nan)
            for (gain1, gain2), values in cell_residuals.items():
                interaction[gain_lookup[gain1], gain_lookup[gain2]] = circular_stats(
                    values
                )["mean_rad"]
            frequency_models.append(
                {
                    "frequency_hz": frequency_hz,
                    "status": "fit",
                    "n_observations": int(np.count_nonzero(selected)),
                    "reference_gain_db": config.gains_db[
                        fitted["reference_gain_index"]
                    ],
                    "rx1_observation_count_by_gain": rx1_counts.tolist(),
                    "rx2_observation_count_by_gain": rx2_counts.tolist(),
                    "intercept_rad": fitted["intercept_rad"],
                    "rx1_effect_rad": rx1_effect,
                    "rx2_effect_rad": rx2_effect,
                    "interaction_residual_rad": interaction.tolist(),
                    "training_metrics": _residual_metrics(fitted["residual_rad"]),
                }
            )

            for held_epoch in range(config.repetitions):
                train = selected & (epochs != held_epoch)
                test = selected & (epochs == held_epoch)
                if not np.any(train) or not np.any(test):
                    continue
                train_rx1 = np.bincount(
                    gain1_indices[train], minlength=len(config.gains_db)
                )
                train_rx2 = np.bincount(
                    gain2_indices[train], minlength=len(config.gains_db)
                )
                test &= (train_rx1[gain1_indices] > 0) & (train_rx2[gain2_indices] > 0)
                if not np.any(test):
                    continue
                fold = fit_additive_surface(
                    phases[train],
                    gain1_indices[train],
                    gain2_indices[train],
                    len(config.gains_db),
                )
                prediction = wrap_phase(
                    fold["intercept_rad"]
                    + fold["rx1_effect_rad"][gain1_indices[test]]
                    + fold["rx2_effect_rad"][gain2_indices[test]]
                )
                cv_residuals.extend(wrap_phase(phases[test] - prediction))

        fitted_intercepts = [
            (model["frequency_hz"], model["intercept_rad"])
            for model in frequency_models
            if model.get("status") == "fit"
        ]
        delay_model = None
        if len(fitted_intercepts) >= 2:
            fitted_intercepts.sort()
            frequency = np.asarray([item[0] for item in fitted_intercepts])
            unwrapped = np.unwrap([item[1] for item in fitted_intercepts])
            slope, intercept = np.polyfit(frequency, unwrapped, 1)
            delay_model = {
                "slope_rad_per_hz": float(slope),
                "intercept_rad": float(intercept),
                "descriptive_delay_seconds": float(slope / (2 * np.pi)),
                "warning": (
                    "Descriptive only: LO retunes and calibration state can add "
                    "phase offsets that are not physical cable delay."
                ),
            }
        return {
            "schema": "spf.calibration.dual_rx_gain_frequency.model",
            "schema_version": 1,
            "serial": receiver.attrs.get("sdr_serial"),
            "phase_convention": zarr.attrs.get("phase_convention"),
            "gain_values_db": list(config.gains_db),
            "quality_valid_observations": int(np.count_nonzero(valid)),
            "frequency_models": frequency_models,
            "cross_validation_metrics": (
                _residual_metrics(np.asarray(cv_residuals)) if cv_residuals else None
            ),
            "frequency_intercept_delay_model": delay_model,
        }
    finally:
        zarr.store.close()


def write_model(path: Path, model: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
