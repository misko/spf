"""Fit and evaluate the compact additive-cross gain calibration design."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import circular_stats, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.dual_rx_gain_frequency.model import fit_additive_surface
from spf.calibrations.models.phase import (
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    SUPPORT_SCHEMA,
    SUPPORT_SCHEMA_VERSION,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


LEGACY_STAGE_GRID_GAINS_DB = (
    -1,
    0,
    3,
    5,
    6,
    15,
    16,
    17,
    23,
    25,
    26,
    27,
    33,
    34,
    41,
    52,
    62,
)

COMPLETE_2P4_MODEL_NAME = "complete_2p4_shared_gain_lut_per_radio"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _portable_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _residual_metrics(residual_rad: np.ndarray) -> dict[str, float] | None:
    residual_rad = np.asarray(residual_rad, dtype=np.float64)
    if residual_rad.size == 0:
        return None
    absolute_deg = np.abs(np.degrees(residual_rad))
    return {
        "n_observations": int(residual_rad.size),
        "circular_mae_deg": float(np.mean(absolute_deg)),
        "circular_rmse_deg": float(np.sqrt(np.mean(absolute_deg**2))),
        "circular_p95_deg": float(np.percentile(absolute_deg, 95)),
        "circular_max_deg": float(np.max(absolute_deg)),
    }


def _curve_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(np.unwrap(first), np.unwrap(second))[0, 1])


def _evaluate_sparse_grid(
    *,
    gains_db: tuple[int, ...],
    shared_effect_rad: np.ndarray,
    intercept_rad: float,
    held_out_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score legacy sparse-grid interpolation against dense held-out cells."""

    gains = np.asarray(gains_db, dtype=np.int64)
    sparse = np.asarray(
        [gain for gain in LEGACY_STAGE_GRID_GAINS_DB if gain in gains_db],
        dtype=np.int64,
    )
    if sparse.size < 2 or sparse[0] != gains[0] or sparse[-1] != gains[-1]:
        return {"status": "unavailable"}
    dense_unwrapped = np.unwrap(np.asarray(shared_effect_rad, dtype=np.float64))
    sparse_indices = np.searchsorted(gains, sparse)
    sparse_effect = dense_unwrapped[sparse_indices]
    linear_effect = np.interp(gains, sparse, sparse_effect)
    nearest_sparse_index = np.abs(gains[:, None] - sparse[None, :]).argmin(axis=1)
    nearest_effect = sparse_effect[nearest_sparse_index]
    residuals: dict[str, list[float]] = {
        "dense_integer_grid": [],
        "sparse_linear_interpolation": [],
        "sparse_nearest": [],
    }
    nearest_by_maximum_gain_error: dict[int, list[float]] = {}
    for cell in held_out_cells:
        observed = cell["observed_mean_rad"]
        if observed is None:
            continue
        gain1 = int(cell["gain_rx1_db"])
        gain2 = int(cell["gain_rx2_db"])
        gain1_index = int(np.searchsorted(gains, gain1))
        gain2_index = int(np.searchsorted(gains, gain2))
        predictions = {
            "dense_integer_grid": (
                intercept_rad
                + dense_unwrapped[gain1_index]
                - dense_unwrapped[gain2_index]
            ),
            "sparse_linear_interpolation": (
                intercept_rad + linear_effect[gain1_index] - linear_effect[gain2_index]
            ),
            "sparse_nearest": (
                intercept_rad
                + nearest_effect[gain1_index]
                - nearest_effect[gain2_index]
            ),
        }
        for name, prediction in predictions.items():
            residuals[name].append(float(wrap_phase(observed - prediction)))
        maximum_gain_error = max(
            abs(gain1 - int(sparse[np.abs(sparse - gain1).argmin()])),
            abs(gain2 - int(sparse[np.abs(sparse - gain2).argmin()])),
        )
        nearest_by_maximum_gain_error.setdefault(maximum_gain_error, []).append(
            residuals["sparse_nearest"][-1]
        )
    return {
        "status": "fit",
        "gain_values_db": sparse.tolist(),
        "evaluation_unit": "held-out ordered-pair circular cell mean",
        "metrics": {
            name: _residual_metrics(np.asarray(values))
            for name, values in residuals.items()
        },
        "nearest_metrics_by_maximum_gain_error_db": [
            {
                "maximum_gain_error_db": maximum_gain_error,
                **(_residual_metrics(np.asarray(values)) or {}),
            }
            for maximum_gain_error, values in sorted(
                nearest_by_maximum_gain_error.items()
            )
        ],
    }


def analyze_additive_cross_dataset(
    path: Path,
    *,
    config: CalibrationConfig,
) -> dict[str, Any]:
    """Fit only the axis cross and score predictions on off-axis held-out pairs."""

    config.validate()
    if config.schedule_design != "additive_cross":
        raise ValueError("additive-cross analysis requires an additive_cross schedule")
    if not config.held_out_gain_pairs:
        raise ValueError("additive-cross analysis requires held-out gain pairs")
    reference_gain = config.schedule_reference_gain_db
    assert reference_gain is not None
    gain_lookup = {gain: index for index, gain in enumerate(config.gains_db)}
    reference_index = gain_lookup[reference_gain]
    training_pairs = set(config.training_gain_pairs)
    held_out_pairs = set(config.held_out_gain_pairs)

    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        completed = np.asarray(receiver.sweep_completed[:], dtype=bool)
        quality_valid = np.asarray(receiver.sweep_quality_valid[:], dtype=bool)
        eligible = completed & quality_valid
        phases = np.asarray(receiver.phase_difference_rad[:], dtype=np.float64)
        frequency_indices = np.asarray(
            receiver.sweep_frequency_index[:], dtype=np.int64
        )
        requested = np.asarray(receiver.sweep_requested_gain_db[:], dtype=np.int64)
        gain1_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 0]],
            dtype=np.int64,
        )
        gain2_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 1]],
            dtype=np.int64,
        )
        if np.any(eligible & ((gain1_indices < 0) | (gain2_indices < 0))):
            raise ValueError("dataset contains gains outside the configured set")
        pair_roles = np.asarray(
            [
                (
                    "training"
                    if (int(pair[0]), int(pair[1])) in training_pairs
                    else (
                        "held_out"
                        if (int(pair[0]), int(pair[1])) in held_out_pairs
                        else "unexpected"
                    )
                )
                for pair in requested
            ]
        )
        if np.any(completed & (pair_roles == "unexpected")):
            raise ValueError("dataset contains an unexpected ordered gain pair")

        frequency_results = []
        all_independent_residuals: list[float] = []
        all_shared_residuals: list[float] = []
        for frequency_index, frequency_hz in enumerate(config.frequencies_hz):
            at_frequency = eligible & (frequency_indices == frequency_index)
            training = at_frequency & (pair_roles == "training")
            held_out = at_frequency & (pair_roles == "held_out")
            if not np.any(training):
                frequency_results.append(
                    {
                        "frequency_hz": frequency_hz,
                        "status": "no_quality_valid_training_data",
                    }
                )
                continue
            fit = fit_additive_surface(
                phases[training],
                gain1_indices[training],
                gain2_indices[training],
                len(config.gains_db),
            )
            rx1_effect = wrap_phase(
                fit["rx1_effect_rad"] - fit["rx1_effect_rad"][reference_index]
            )
            rx2_effect = wrap_phase(
                fit["rx2_effect_rad"] - fit["rx2_effect_rad"][reference_index]
            )
            intercept = float(
                wrap_phase(
                    fit["intercept_rad"]
                    + fit["rx1_effect_rad"][reference_index]
                    + fit["rx2_effect_rad"][reference_index]
                )
            )
            shared_effect = np.angle(np.exp(1j * rx1_effect) + np.exp(-1j * rx2_effect))
            symmetry_error = wrap_phase(rx1_effect + rx2_effect)

            independent_prediction = wrap_phase(
                intercept
                + rx1_effect[gain1_indices[held_out]]
                + rx2_effect[gain2_indices[held_out]]
            )
            shared_prediction = wrap_phase(
                intercept
                + shared_effect[gain1_indices[held_out]]
                - shared_effect[gain2_indices[held_out]]
            )
            independent_residual = wrap_phase(phases[held_out] - independent_prediction)
            shared_residual = wrap_phase(phases[held_out] - shared_prediction)
            all_independent_residuals.extend(independent_residual.tolist())
            all_shared_residuals.extend(shared_residual.tolist())

            held_out_cells = []
            for gain1, gain2 in config.held_out_gain_pairs:
                cell = (
                    held_out & (requested[:, 0] == gain1) & (requested[:, 1] == gain2)
                )
                cell_phase = phases[cell]
                stats = circular_stats(cell_phase)
                prediction = float(
                    wrap_phase(
                        intercept
                        + rx1_effect[gain_lookup[gain1]]
                        + rx2_effect[gain_lookup[gain2]]
                    )
                )
                held_out_cells.append(
                    {
                        "gain_rx1_db": gain1,
                        "gain_rx2_db": gain2,
                        "n_quality_valid": int(np.count_nonzero(cell)),
                        "observed_mean_rad": stats["mean_rad"],
                        "prediction_rad": prediction,
                        "mean_residual_deg": (
                            math.degrees(
                                float(wrap_phase(stats["mean_rad"] - prediction))
                            )
                            if stats["mean_rad"] is not None
                            else None
                        ),
                    }
                )
            adjacent_steps = [
                {
                    "gain_from_db": gain1,
                    "gain_to_db": gain2,
                    "shared_effect_step_deg": math.degrees(
                        float(
                            wrap_phase(
                                shared_effect[gain_lookup[gain2]]
                                - shared_effect[gain_lookup[gain1]]
                            )
                        )
                    ),
                }
                for gain1, gain2 in zip(config.gains_db[:-1], config.gains_db[1:])
            ]
            sparse_grid_evaluation = _evaluate_sparse_grid(
                gains_db=config.gains_db,
                shared_effect_rad=shared_effect,
                intercept_rad=intercept,
                held_out_cells=held_out_cells,
            )
            frequency_results.append(
                {
                    "frequency_hz": frequency_hz,
                    "status": "fit",
                    "reference_gain_db": reference_gain,
                    "quality_valid_training_observations": int(
                        np.count_nonzero(training)
                    ),
                    "quality_valid_held_out_observations": int(
                        np.count_nonzero(held_out)
                    ),
                    "training_metrics": _residual_metrics(fit["residual_rad"]),
                    "held_out_independent_rx_metrics": _residual_metrics(
                        independent_residual
                    ),
                    "held_out_shared_gain_curve_metrics": _residual_metrics(
                        shared_residual
                    ),
                    "rx1_vs_negative_rx2_curve_correlation": _curve_correlation(
                        rx1_effect, -rx2_effect
                    ),
                    "rx1_vs_negative_rx2_rms_disagreement_deg": float(
                        np.sqrt(np.mean(np.degrees(symmetry_error) ** 2))
                    ),
                    "intercept_rad": intercept,
                    "rx1_effect_rad": rx1_effect.tolist(),
                    "rx2_effect_rad": rx2_effect.tolist(),
                    "shared_gain_effect_rad": shared_effect.tolist(),
                    "adjacent_shared_gain_steps": adjacent_steps,
                    "held_out_cells": held_out_cells,
                    "legacy_sparse_grid_evaluation": sparse_grid_evaluation,
                }
            )

        return {
            "schema": "spf.calibration.dual_rx_gain_frequency.additive_cross",
            "schema_version": 1,
            "serial": receiver.attrs.get("sdr_serial"),
            "phase_convention": zarr.attrs.get("phase_convention"),
            "schedule_design": config.schedule_design,
            "reference_gain_db": reference_gain,
            "gain_values_db": list(config.gains_db),
            "training_pairs_per_frequency": len(config.training_gain_pairs),
            "held_out_pairs_per_frequency": len(config.held_out_gain_pairs),
            "overall_held_out_independent_rx_metrics": _residual_metrics(
                np.asarray(all_independent_residuals)
            ),
            "overall_held_out_shared_gain_curve_metrics": _residual_metrics(
                np.asarray(all_shared_residuals)
            ),
            "frequency_results": frequency_results,
        }
    finally:
        zarr.store.close()


def render_additive_cross_markdown(result: dict[str, Any]) -> str:
    """Render the decisive held-out results and gain-curve diagnostics."""

    lines = [
        "# Integer-gain additive-cross analysis",
        "",
        f"- Pluto serial: `{result['serial']}`",
        f"- Phase convention: `{result['phase_convention']}`",
        f"- Reference gain: {result['reference_gain_db']} dB",
        (
            f"- Design per frequency: {result['training_pairs_per_frequency']} "
            f"training-axis pairs + {result['held_out_pairs_per_frequency']} "
            "off-axis held-out pairs"
        ),
        "",
        "The fit uses only `(gain, reference)` and `(reference, gain)` frames. "
        "Off-axis pairs are never used to estimate the gain curves.",
        "",
        "## Held-out prediction",
        "",
        "| Frequency | Valid held-out frames | Independent RX curves MAE / p95 | Shared H(g) MAE / p95 | RX1 vs -RX2 correlation | Curve disagreement RMS |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["frequency_results"]:
        if row["status"] != "fit":
            lines.append(
                f"| {row['frequency_hz'] / 1e6:.3f} MHz | 0 | n/a | n/a | n/a | n/a |"
            )
            continue
        independent = row["held_out_independent_rx_metrics"]
        shared = row["held_out_shared_gain_curve_metrics"]
        correlation = row["rx1_vs_negative_rx2_curve_correlation"]
        lines.append(
            f"| {row['frequency_hz'] / 1e6:.3f} MHz | "
            f"{row['quality_valid_held_out_observations']} | "
            f"{independent['circular_mae_deg']:.2f}° / "
            f"{independent['circular_p95_deg']:.2f}° | "
            f"{shared['circular_mae_deg']:.2f}° / "
            f"{shared['circular_p95_deg']:.2f}° | "
            f"{correlation:.4f} | "
            f"{row['rx1_vs_negative_rx2_rms_disagreement_deg']:.2f}° |"
        )
    overall_independent = result["overall_held_out_independent_rx_metrics"]
    overall_shared = result["overall_held_out_shared_gain_curve_metrics"]
    lines.extend(
        [
            "",
            "## Overall result",
            "",
            (
                "- Independent RX1/RX2 curves: "
                f"{overall_independent['circular_mae_deg']:.2f}° MAE, "
                f"{overall_independent['circular_p95_deg']:.2f}° p95."
                if overall_independent
                else "- Independent RX1/RX2 curves: no valid held-out observations."
            ),
            (
                "- Shared antisymmetric H(g) curve: "
                f"{overall_shared['circular_mae_deg']:.2f}° MAE, "
                f"{overall_shared['circular_p95_deg']:.2f}° p95."
                if overall_shared
                else "- Shared antisymmetric H(g) curve: no valid held-out observations."
            ),
            "",
            "## Legacy 17-gain grid on the same held-out cells",
            "",
            "| Frequency | Dense integer MAE / p95 | Sparse linear MAE / p95 | Sparse nearest MAE / p95 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["frequency_results"]:
        sparse = row.get("legacy_sparse_grid_evaluation") or {}
        if sparse.get("status") != "fit":
            continue
        metrics = sparse["metrics"]
        dense = metrics["dense_integer_grid"]
        linear = metrics["sparse_linear_interpolation"]
        nearest = metrics["sparse_nearest"]
        lines.append(
            f"| {row['frequency_hz'] / 1e6:.3f} MHz | "
            f"{dense['circular_mae_deg']:.2f}° / "
            f"{dense['circular_p95_deg']:.2f}° | "
            f"{linear['circular_mae_deg']:.2f}° / "
            f"{linear['circular_p95_deg']:.2f}° | "
            f"{nearest['circular_mae_deg']:.2f}° / "
            f"{nearest['circular_p95_deg']:.2f}° |"
        )
    lines.extend(
        [
            "",
            "Linear interpolation and nearest-neighbour use only the previously "
            "published 17 stage-focused gains. Their errors are scored against "
            "the same off-axis cells as the dense integer curve.",
            "",
            "## Largest adjacent integer-gain steps",
            "",
            "| Frequency | Gain transition | Absolute phase step | Signed phase step |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["frequency_results"]:
        if row["status"] != "fit":
            continue
        largest = sorted(
            row["adjacent_shared_gain_steps"],
            key=lambda step: abs(step["shared_effect_step_deg"]),
            reverse=True,
        )[:10]
        for step in largest:
            lines.append(
                f"| {row['frequency_hz'] / 1e6:.3f} MHz | "
                f"{step['gain_from_db']}→{step['gain_to_db']} dB | "
                f"{abs(step['shared_effect_step_deg']):.2f}° | "
                f"{step['shared_effect_step_deg']:+.2f}° |"
            )
    lines.extend(
        [
            "",
            "Equal manual and AGC-reported dB states are treated as the same gain "
            "state for this experiment. The result tests gain-table structure; it "
            "does not replay historical capture timing.",
            "",
        ]
    )
    return "\n".join(lines)


def write_additive_cross_bundle(
    *,
    dataset_path: Path,
    config: CalibrationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    result = analyze_additive_cross_dataset(dataset_path, config=config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(render_additive_cross_markdown(result))
    return result


def compare_additive_cross_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare dense gain curves and test a radio-shared gain shape."""

    if len(results) < 2:
        raise ValueError("at least two additive-cross results are required")
    serials = [str(result["serial"]) for result in results]
    if len(set(serials)) != len(serials):
        raise ValueError("additive-cross results contain duplicate serials")
    gains = tuple(int(gain) for gain in results[0]["gain_values_db"])
    reference_gain = int(results[0]["reference_gain_db"])
    frequencies = tuple(
        int(row["frequency_hz"]) for row in results[0]["frequency_results"]
    )
    for result in results:
        if tuple(int(gain) for gain in result["gain_values_db"]) != gains:
            raise ValueError("additive-cross gain axes differ")
        if int(result["reference_gain_db"]) != reference_gain:
            raise ValueError("additive-cross reference gains differ")
        if (
            tuple(int(row["frequency_hz"]) for row in result["frequency_results"])
            != frequencies
        ):
            raise ValueError("additive-cross frequency axes differ")

    gain_lookup = {gain: index for index, gain in enumerate(gains)}
    rows_by_serial_frequency = {
        (str(result["serial"]), int(row["frequency_hz"])): row
        for result in results
        for row in result["frequency_results"]
    }
    frequency_comparisons = []
    all_frequency_shared_residuals: list[float] = []
    all_curves = []
    for frequency_hz in frequencies:
        curves = np.asarray(
            [
                rows_by_serial_frequency[(serial, frequency_hz)][
                    "shared_gain_effect_rad"
                ]
                for serial in serials
            ],
            dtype=np.float64,
        )
        shared_curve = np.angle(np.mean(np.exp(1j * curves), axis=0))
        all_curves.extend(curves)
        pairwise = []
        for first_index, first_serial in enumerate(serials):
            for second_index in range(first_index + 1, len(serials)):
                second_serial = serials[second_index]
                difference = wrap_phase(curves[first_index] - curves[second_index])
                pairwise.append(
                    {
                        "first_serial": first_serial,
                        "second_serial": second_serial,
                        "curve_correlation": _curve_correlation(
                            curves[first_index], curves[second_index]
                        ),
                        "curve_rms_difference_deg": float(
                            np.sqrt(np.mean(np.degrees(difference) ** 2))
                        ),
                        "curve_maximum_difference_deg": float(
                            np.max(np.abs(np.degrees(difference)))
                        ),
                    }
                )
        residuals = []
        for serial in serials:
            row = rows_by_serial_frequency[(serial, frequency_hz)]
            for cell in row["held_out_cells"]:
                if cell["observed_mean_rad"] is None:
                    continue
                prediction = (
                    float(row["intercept_rad"])
                    + shared_curve[gain_lookup[int(cell["gain_rx1_db"])]]
                    - shared_curve[gain_lookup[int(cell["gain_rx2_db"])]]
                )
                residuals.append(
                    float(wrap_phase(cell["observed_mean_rad"] - prediction))
                )
        all_frequency_shared_residuals.extend(residuals)
        frequency_comparisons.append(
            {
                "frequency_hz": frequency_hz,
                "radio_shared_gain_effect_rad": shared_curve.tolist(),
                "pairwise_radio_curve_comparisons": pairwise,
                "held_out_radio_shared_curve_metrics": _residual_metrics(
                    np.asarray(residuals)
                ),
            }
        )

    one_curve_for_band = np.angle(np.mean(np.exp(1j * np.asarray(all_curves)), axis=0))
    band_shared_residuals = []
    for serial in serials:
        for frequency_hz in frequencies:
            row = rows_by_serial_frequency[(serial, frequency_hz)]
            for cell in row["held_out_cells"]:
                if cell["observed_mean_rad"] is None:
                    continue
                prediction = (
                    float(row["intercept_rad"])
                    + one_curve_for_band[gain_lookup[int(cell["gain_rx1_db"])]]
                    - one_curve_for_band[gain_lookup[int(cell["gain_rx2_db"])]]
                )
                band_shared_residuals.append(
                    float(wrap_phase(cell["observed_mean_rad"] - prediction))
                )

    within_radio_frequency = []
    if len(frequencies) >= 2:
        for serial in serials:
            first = np.asarray(
                rows_by_serial_frequency[(serial, frequencies[0])][
                    "shared_gain_effect_rad"
                ]
            )
            second = np.asarray(
                rows_by_serial_frequency[(serial, frequencies[1])][
                    "shared_gain_effect_rad"
                ]
            )
            difference = wrap_phase(first - second)
            within_radio_frequency.append(
                {
                    "serial": serial,
                    "first_frequency_hz": frequencies[0],
                    "second_frequency_hz": frequencies[1],
                    "curve_correlation": _curve_correlation(first, second),
                    "curve_rms_difference_deg": float(
                        np.sqrt(np.mean(np.degrees(difference) ** 2))
                    ),
                    "curve_maximum_difference_deg": float(
                        np.max(np.abs(np.degrees(difference)))
                    ),
                }
            )

    return {
        "schema": "spf.calibration.dual_rx_gain_frequency.additive_cross_comparison",
        "schema_version": 1,
        "serials": serials,
        "frequencies_hz": list(frequencies),
        "gain_values_db": list(gains),
        "reference_gain_db": reference_gain,
        "frequency_comparisons": frequency_comparisons,
        "within_radio_frequency_comparisons": within_radio_frequency,
        "held_out_frequency_specific_radio_shared_curve_metrics": (
            _residual_metrics(np.asarray(all_frequency_shared_residuals))
        ),
        "held_out_one_curve_for_2p4_band_metrics": _residual_metrics(
            np.asarray(band_shared_residuals)
        ),
        "one_curve_for_2p4_band_effect_rad": one_curve_for_band.tolist(),
    }


def render_additive_cross_comparison_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Two-radio integer-gain curve comparison",
        "",
        f"- Radios: {', '.join(f'`{serial}`' for serial in result['serials'])}",
        (
            "- Frequencies: "
            + ", ".join(
                f"{frequency / 1e6:.3f} MHz" for frequency in result["frequencies_hz"]
            )
        ),
        f"- Reference gain: {result['reference_gain_db']} dB",
        "",
        "## Does the gain curve transfer between radios?",
        "",
        "| Frequency | Curve correlation | RMS difference | Maximum difference | Shared-curve held-out MAE / p95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["frequency_comparisons"]:
        comparison = row["pairwise_radio_curve_comparisons"][0]
        metrics = row["held_out_radio_shared_curve_metrics"]
        lines.append(
            f"| {row['frequency_hz'] / 1e6:.3f} MHz | "
            f"{comparison['curve_correlation']:.4f} | "
            f"{comparison['curve_rms_difference_deg']:.2f}° | "
            f"{comparison['curve_maximum_difference_deg']:.2f}° | "
            f"{metrics['circular_mae_deg']:.2f}° / "
            f"{metrics['circular_p95_deg']:.2f}° |"
        )
    per_frequency = result["held_out_frequency_specific_radio_shared_curve_metrics"]
    one_band = result["held_out_one_curve_for_2p4_band_metrics"]
    lines.extend(
        [
            "",
            (
                "Using one radio-shared curve per exact frequency gives "
                f"{per_frequency['circular_mae_deg']:.2f}° MAE and "
                f"{per_frequency['circular_p95_deg']:.2f}° p95 on held-out "
                "cell means."
            ),
            (
                "Using one curve for both 2.4 GHz frequencies gives "
                f"{one_band['circular_mae_deg']:.2f}° MAE and "
                f"{one_band['circular_p95_deg']:.2f}° p95."
            ),
            "",
            "Each radio/frequency retains its own intercept in these transfer "
            "tests. Only the gain-dependent curve is shared.",
            "",
            "## Frequency sensitivity of the gain curve",
            "",
            "| Radio | Curve correlation | RMS difference | Maximum difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["within_radio_frequency_comparisons"]:
        lines.append(
            f"| `{row['serial']}` | {row['curve_correlation']:.4f} | "
            f"{row['curve_rms_difference_deg']:.2f}° | "
            f"{row['curve_maximum_difference_deg']:.2f}° |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The radio-specific intercept remains necessary. The comparison above "
            "tests whether the much cheaper gain-dependent term can be shared. "
            "Held-out cells, not training-axis cells, determine the reported "
            "transfer error.",
            "",
        ]
    )
    return "\n".join(lines)


def write_additive_cross_comparison_bundle(
    *,
    analysis_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    results = [json.loads(Path(path).read_text()) for path in analysis_paths]
    comparison = compare_additive_cross_results(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(
        render_additive_cross_comparison_markdown(comparison)
    )
    return comparison


def export_complete_2p4_model(
    *,
    analysis_path: Path | list[Path],
    validation_path: Path | list[Path],
    output_root: Path,
    maximum_held_out_p95_deg: float = 5.0,
) -> dict[str, Any]:
    """Export complete integer-gain models from validated axis-cross fits.

    All inputs must belong to one physical radio and use the same gain axis.
    Every individual gain must pass on both fitted axes. Off-axis held-out
    observations validate the additive factorization. The resulting support
    domain is the full ordered cartesian product at every exact fitted LO.
    """

    analysis_paths = (
        [Path(path).resolve() for path in analysis_path]
        if isinstance(analysis_path, list)
        else [Path(analysis_path).resolve()]
    )
    validation_paths = (
        [Path(path).resolve() for path in validation_path]
        if isinstance(validation_path, list)
        else [Path(validation_path).resolve()]
    )
    if len(analysis_paths) != len(validation_paths):
        raise ValueError("each analysis must have one matching validation")
    if not analysis_paths:
        raise ValueError("at least one complete calibration source is required")
    output_root = Path(output_root).resolve()
    serial = None
    gains_db = None
    reference_gain_db = None
    phase_convention = None
    frequency_data: dict[int, dict[str, Any]] = {}
    failed_held_out_cells = []
    measured_training_cell_count = 0
    passing_training_cell_count = 0
    measured_held_out_cell_count = 0
    passing_held_out_cell_count = 0
    source_rows = []
    overall_metrics_by_source = []
    for current_analysis_path, current_validation_path in zip(
        analysis_paths, validation_paths
    ):
        analysis = json.loads(current_analysis_path.read_text())
        validation = json.loads(current_validation_path.read_text())
        current_serial = str(analysis["serial"])
        if validation.get("serial") != current_serial:
            raise ValueError("analysis and validation belong to different radios")
        if serial is None:
            serial = current_serial
        elif current_serial != serial:
            raise ValueError("complete-model sources belong to different radios")
        if analysis.get("schedule_design") != "additive_cross":
            raise ValueError("complete model requires an additive-cross analysis")
        if validation.get("status") == "partial":
            raise ValueError("partial capture cannot define a complete model")

        current_gains = [int(value) for value in analysis["gain_values_db"]]
        if current_gains != list(range(current_gains[0], current_gains[-1] + 1)):
            raise ValueError("complete model requires a contiguous integer gain axis")
        current_reference_gain = int(analysis["reference_gain_db"])
        current_phase_convention = str(analysis["phase_convention"])
        if gains_db is None:
            gains_db = current_gains
            reference_gain_db = current_reference_gain
            phase_convention = current_phase_convention
        elif (
            current_gains != gains_db
            or current_reference_gain != reference_gain_db
            or current_phase_convention != phase_convention
        ):
            raise ValueError("complete-model source axes or conventions differ")
        assert gains_db is not None
        assert reference_gain_db is not None
        reference_index = gains_db.index(reference_gain_db)

        frequency_rows = [
            row for row in analysis["frequency_results"] if row.get("status") == "fit"
        ]
        if len(frequency_rows) != len(analysis["frequency_results"]):
            raise ValueError("every requested frequency must have a fitted model")
        source_frequencies = {int(row["frequency_hz"]) for row in frequency_rows}
        validation_frequencies = {
            int(row["frequency_hz"]) for row in validation["cells"]
        }
        if validation_frequencies != source_frequencies:
            raise ValueError("analysis and validation frequency sets do not match")
        training_cells = [
            row for row in validation["cells"] if row.get("role") == "training"
        ]
        failed_training = [row for row in training_cells if not row.get("pass")]
        if failed_training:
            raise ValueError(
                f"{len(failed_training)} fitted-axis cells failed the quality gate"
            )
        expected_axis_pairs = 2 * len(gains_db) - 1
        if int(analysis["training_pairs_per_frequency"]) != expected_axis_pairs:
            raise ValueError("training schedule does not cover both complete gain axes")

        held_out_cells = [
            row for row in validation["cells"] if row.get("role") == "held_out"
        ]
        measured_training_cell_count += len(training_cells)
        passing_training_cell_count += sum(bool(row["pass"]) for row in training_cells)
        measured_held_out_cell_count += len(held_out_cells)
        passing_held_out_cell_count += sum(bool(row["pass"]) for row in held_out_cells)
        failed_held_out_cells.extend(
            [
                int(row["frequency_hz"]),
                int(row["gain_rx1_db"]),
                int(row["gain_rx2_db"]),
            ]
            for row in held_out_cells
            if not row["pass"]
        )

        for row in frequency_rows:
            frequency_hz = int(row["frequency_hz"])
            if frequency_hz in frequency_data:
                raise ValueError(f"duplicate complete-model frequency {frequency_hz}")
            shared_effect = np.asarray(row["shared_gain_effect_rad"], dtype=np.float64)
            if shared_effect.shape != (len(gains_db),):
                raise ValueError("gain-effect length does not match complete gain axis")
            shared_effect = wrap_phase(shared_effect - shared_effect[reference_index])
            metrics = row["held_out_shared_gain_curve_metrics"]
            if (
                metrics is None
                or not metrics["n_observations"]
                or metrics["circular_p95_deg"] > maximum_held_out_p95_deg
            ):
                raise ValueError(
                    f"{frequency_hz} Hz held-out p95 does not pass "
                    f"{maximum_held_out_p95_deg:.1f} degrees"
                )
            frequency_data[frequency_hz] = {
                "intercept_rad": float(row["intercept_rad"]),
                "shared_effect_rad": shared_effect,
                "evaluation": {
                    "frequency_hz": frequency_hz,
                    "held_out_metrics": metrics,
                    "quality_valid_training_observations": int(
                        row["quality_valid_training_observations"]
                    ),
                    "quality_valid_held_out_observations": int(
                        row["quality_valid_held_out_observations"]
                    ),
                },
            }

        source_row = {
            "analysis_path": _portable_path(current_analysis_path),
            "analysis_sha256": _sha256(current_analysis_path),
            "validation_path": _portable_path(current_validation_path),
            "validation_sha256": _sha256(current_validation_path),
            "validation_status": validation["status"],
            "frequencies_hz": sorted(
                int(row["frequency_hz"]) for row in frequency_rows
            ),
        }
        source_rows.append(source_row)
        overall_metrics_by_source.append(
            {
                **source_row,
                "held_out_metrics": analysis[
                    "overall_held_out_shared_gain_curve_metrics"
                ],
            }
        )

    assert serial is not None
    assert gains_db is not None
    assert reference_gain_db is not None
    assert phase_convention is not None
    reference_index = gains_db.index(reference_gain_db)
    frequencies_hz = sorted(frequency_data)
    coefficients: dict[str, float] = {}
    frequency_evaluation = []
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        row = frequency_data[frequency_hz]
        coefficients[f"frequency[{frequency_index}].intercept"] = row["intercept_rad"]
        for gain_index, effect in enumerate(row["shared_effect_rad"]):
            if gain_index == reference_index:
                continue
            coefficients[
                f"frequency[{frequency_index}].rx1_phase[{gain_index}]"
            ] = float(effect)
            coefficients[
                f"frequency[{frequency_index}].rx2_phase[{gain_index}]"
            ] = float(-effect)
        frequency_evaluation.append(row["evaluation"])

    full_cell_count = len(frequencies_hz) * len(gains_db) * len(gains_db)
    support_path = output_root / f"{COMPLETE_2P4_MODEL_NAME}_support" / f"{serial}.json"
    support = {
        "schema": SUPPORT_SCHEMA,
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "radio_serial": serial,
        "phase_convention": phase_convention,
        "frequencies_hz": frequencies_hz,
        "gains_db": gains_db,
        "support_kind": "cartesian_product",
        "supported_cell_count": full_cell_count,
        "validation_basis": (
            "complete RX1 and RX2 integer-gain axes with off-axis held-out "
            "validation of the additive factorization"
        ),
        "measured_training_cell_count": measured_training_cell_count,
        "passing_training_cell_count": passing_training_cell_count,
        "measured_held_out_cell_count": measured_held_out_cell_count,
        "passing_held_out_cell_count": passing_held_out_cell_count,
        "failed_held_out_cells": sorted(failed_held_out_cells),
        "sources": source_rows,
    }
    _write_json(support_path, support)

    model_path = output_root / COMPLETE_2P4_MODEL_NAME / f"{serial}.json"
    model = {
        "schema": MODEL_SCHEMA,
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_name": COMPLETE_2P4_MODEL_NAME,
        "label": "Complete 2.4 GHz shared gain LUT with per-radio intercept",
        "scope": "per_radio",
        "kind": "frequency_specific_gain_additive",
        "formula": (
            "offset(f,g1,g2) = intercept[radio,f] + H[radio,f,g1] " "- H[radio,f,g2]"
        ),
        "phase_convention": phase_convention,
        "radio_serial": serial,
        "reference_frequency_hz": float(frequencies_hz[0]),
        "reference_gain_db": reference_gain_db,
        "frequencies_hz": frequencies_hz,
        "gains_db": gains_db,
        "can_predict_unseen_frequency": False,
        "coefficients_rad": coefficients,
        "parameter_count": len(frequencies_hz) * len(gains_db),
        "support_profile": {
            "path": str(
                Path("..") / f"{COMPLETE_2P4_MODEL_NAME}_support" / f"{serial}.json"
            ),
            "sha256": _sha256(support_path),
            "strict_prediction_default": True,
        },
        "evaluation": {
            "unit": "held-out off-axis frame",
            "maximum_accepted_held_out_p95_deg": maximum_held_out_p95_deg,
            "overall_held_out_metrics_by_source": overall_metrics_by_source,
            "by_frequency": frequency_evaluation,
        },
        "sources": source_rows,
    }
    _write_json(model_path, model)
    registry_path = output_root / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        family = registry.setdefault("models", {}).setdefault(
            COMPLETE_2P4_MODEL_NAME,
            {
                "can_predict_unseen_frequency": False,
                "configs_by_serial": {},
                "formula": model["formula"],
                "kind": model["kind"],
                "label": model["label"],
            },
        )
        family["configs_by_serial"][serial] = str(model_path.relative_to(output_root))
        registry["complete_2p4_model"] = COMPLETE_2P4_MODEL_NAME
        _write_json(registry_path, registry)
    return {
        "model_path": str(model_path),
        "support_path": str(support_path),
        "serial": serial,
        "frequencies_hz": frequencies_hz,
        "gains_db": gains_db,
        "supported_cell_count": full_cell_count,
        "overall_held_out_metrics_by_source": overall_metrics_by_source,
    }
