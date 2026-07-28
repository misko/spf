"""Evaluate universal gain LUTs adapted by one or two target-radio values."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.model_matrix import (
    MODEL_SPECS,
    _fit,
    _metrics,
    _predict,
    _radio_arrays_from_paths,
    _subset,
)

SCHEMA = "spf.calibration.dual_rx_gain_frequency.low_cost_radio_calibration"
SCHEMA_VERSION = 1
BASE_MODEL_NAMES = (
    "frequency_lut_additive_gain_universal",
    "frequency_specific_additive_gain_universal",
    "full_cell_lut_universal",
)
DEFAULT_ONE_ANCHOR_HZ = 2_412_000_000
DEFAULT_TWO_ANCHORS_HZ = (868_000_000, 5_866_000_000)
DEFAULT_SECOND_GAIN_PAIR_DB = (62, 26)


def _circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError("cannot average an empty circular vector")
    return float(np.angle(np.mean(np.exp(1j * values))))


def _adaptation_delta(
    frequency_hz: np.ndarray,
    *,
    anchor_frequencies_hz: tuple[int, ...],
    anchor_residuals_rad: tuple[float, ...],
) -> np.ndarray:
    """Return a constant or frequency-linear circular phase adjustment."""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    if len(anchor_frequencies_hz) == 1:
        return np.full(frequency_hz.shape, anchor_residuals_rad[0])
    if len(anchor_frequencies_hz) != 2:
        raise ValueError("only one- and two-value adaptation is supported")
    first_frequency, second_frequency = anchor_frequencies_hz
    if first_frequency == second_frequency:
        raise ValueError("two calibration frequencies must differ")
    first_delta = float(anchor_residuals_rad[0])
    second_delta = first_delta + float(
        wrap_phase(anchor_residuals_rad[1] - first_delta)
    )
    fraction = (frequency_hz - first_frequency) / (second_frequency - first_frequency)
    return first_delta + fraction * (second_delta - first_delta)


def _prepare_leave_radio_out(
    *,
    spec,
    data: dict[str, np.ndarray],
    provenance: list[dict[str, Any]],
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> list[dict[str, Any]]:
    prepared = []
    for radio_index, radio in enumerate(provenance):
        train = _subset(data, data["radio"] != radio_index)
        target = _subset(data, data["radio"] == radio_index)
        fit = _fit(
            spec,
            train,
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
            reference_frequency_hz=reference_frequency_hz,
        )
        prediction, supported = _predict(
            fit,
            target,
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
        )
        full_prediction = np.full(target["phase"].shape, np.nan)
        full_prediction[supported] = prediction
        prepared.append(
            {
                "radio_index": radio_index,
                "serial": radio["serial"],
                "target": target,
                "supported": supported,
                "prediction": full_prediction,
                "base_residual": wrap_phase(target["phase"] - full_prediction),
            }
        )
    return prepared


def _evaluate_strategy(
    *,
    prepared: list[dict[str, Any]],
    reference_gain_db: int,
    anchor_frequencies_hz: tuple[int, ...],
) -> dict[str, Any]:
    residuals: list[float] = []
    per_radio: dict[str, Any] = {}
    expected = 0
    for prepared_radio in prepared:
        target = prepared_radio["target"]
        supported = prepared_radio["supported"]
        full_prediction = prepared_radio["prediction"]
        base_residual = prepared_radio["base_residual"]
        anchor_residuals = []
        evaluation = supported.copy()
        anchor_counts = []
        for frequency in anchor_frequencies_hz:
            anchor = (
                supported
                & (target["frequency_hz"] == frequency)
                & (target["gain1_db"] == reference_gain_db)
                & (target["gain2_db"] == reference_gain_db)
            )
            if not np.any(anchor):
                raise ValueError(
                    f"{prepared_radio['serial']}: no valid reference-gain anchor at "
                    f"{frequency} Hz"
                )
            anchor_counts.append(int(np.count_nonzero(anchor)))
            anchor_residuals.append(_circular_mean(base_residual[anchor]))
            evaluation &= ~anchor
        adjustment = _adaptation_delta(
            target["frequency_hz"],
            anchor_frequencies_hz=anchor_frequencies_hz,
            anchor_residuals_rad=tuple(anchor_residuals),
        )
        local_residual = wrap_phase(
            target["phase"][evaluation]
            - full_prediction[evaluation]
            - adjustment[evaluation]
        )
        residuals.extend(local_residual.tolist())
        expected += int(np.count_nonzero(evaluation))
        per_radio[str(prepared_radio["radio_index"])] = {
            "serial": prepared_radio["serial"],
            "anchor_observations": anchor_counts,
            "anchor_values_rad": anchor_residuals,
            **_metrics(local_residual, expected=int(np.count_nonzero(evaluation))),
        }
    return {
        "anchor_frequencies_hz": list(anchor_frequencies_hz),
        "calibration_values_per_radio": len(anchor_frequencies_hz),
        "per_radio": per_radio,
        **_metrics(np.asarray(residuals), expected=expected),
    }


def _unadapted_leave_radio_out(
    *,
    prepared: list[dict[str, Any]],
) -> dict[str, Any]:
    residuals = []
    per_radio = {}
    expected = 0
    for prepared_radio in prepared:
        target = prepared_radio["target"]
        supported = prepared_radio["supported"]
        local = prepared_radio["base_residual"][supported]
        residuals.extend(local.tolist())
        expected += int(target["phase"].size)
        per_radio[str(prepared_radio["radio_index"])] = {
            "serial": prepared_radio["serial"],
            **_metrics(local, expected=int(target["phase"].size)),
        }
    return {
        "calibration_values_per_radio": 0,
        "per_radio": per_radio,
        **_metrics(np.asarray(residuals), expected=expected),
    }


def _evaluate_per_frequency_strategy(
    *,
    prepared: list[dict[str, Any]],
    frequencies_hz: tuple[int, ...],
    reference_gain_db: int,
    second_gain_pair_db: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    residuals = []
    per_radio = {}
    expected = 0
    for prepared_radio in prepared:
        target = prepared_radio["target"]
        supported = prepared_radio["supported"]
        prediction = prepared_radio["prediction"]
        base_residual = prepared_radio["base_residual"]
        evaluation = supported.copy()
        correction = np.zeros(target["phase"].shape, dtype=np.float64)
        anchor_counts = []
        for frequency in frequencies_hz:
            reference_anchor = (
                supported
                & (target["frequency_hz"] == frequency)
                & (target["gain1_db"] == reference_gain_db)
                & (target["gain2_db"] == reference_gain_db)
            )
            if not np.any(reference_anchor):
                return None
            reference_delta = _circular_mean(base_residual[reference_anchor])
            selected_frequency = target["frequency_hz"] == frequency
            local_correction = np.full(
                int(np.count_nonzero(selected_frequency)),
                reference_delta,
                dtype=np.float64,
            )
            anchor_counts.append(int(np.count_nonzero(reference_anchor)))
            evaluation &= ~reference_anchor
            if second_gain_pair_db is not None:
                gain1, gain2 = second_gain_pair_db
                gain_difference = gain1 - gain2
                if not gain_difference:
                    return None
                second_anchor = (
                    supported
                    & (target["frequency_hz"] == frequency)
                    & (target["gain1_db"] == gain1)
                    & (target["gain2_db"] == gain2)
                )
                if not np.any(second_anchor):
                    return None
                second_delta = _circular_mean(base_residual[second_anchor])
                slope = float(wrap_phase(second_delta - reference_delta)) / (
                    gain_difference
                )
                local_correction += slope * (
                    target["gain1_db"][selected_frequency]
                    - target["gain2_db"][selected_frequency]
                )
                anchor_counts.append(int(np.count_nonzero(second_anchor)))
                evaluation &= ~second_anchor
            correction[selected_frequency] = local_correction
        local_residual = wrap_phase(
            target["phase"][evaluation]
            - prediction[evaluation]
            - correction[evaluation]
        )
        residuals.extend(local_residual.tolist())
        local_expected = int(np.count_nonzero(evaluation))
        expected += local_expected
        per_radio[str(prepared_radio["radio_index"])] = {
            "serial": prepared_radio["serial"],
            "anchor_observations": anchor_counts,
            **_metrics(local_residual, expected=local_expected),
        }
    return {
        "calibration_values_per_radio": len(frequencies_hz)
        * (2 if second_gain_pair_db is not None else 1),
        "calibration_values_per_operating_frequency": (
            2 if second_gain_pair_db is not None else 1
        ),
        "second_gain_pair_db": (
            list(second_gain_pair_db) if second_gain_pair_db is not None else None
        ),
        "per_radio": per_radio,
        **_metrics(np.asarray(residuals), expected=expected),
    }


def _best(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return min(results, key=lambda row: row["circular_mae_deg"])


def _cell_means(data: dict[str, np.ndarray]) -> dict[tuple[int, int, int], float]:
    groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for frequency, gain1, gain2, phase in zip(
        data["frequency_hz"],
        data["gain1_db"],
        data["gain2_db"],
        data["phase"],
    ):
        groups[(int(frequency), int(gain1), int(gain2))].append(float(phase))
    return {
        coordinate: _circular_mean(np.asarray(values))
        for coordinate, values in groups.items()
    }


def _repeatability(
    *,
    config_path: Path,
    primary_paths: tuple[Path, ...],
    repeat_paths: tuple[Path, ...],
) -> dict[str, Any]:
    primary_by_serial = {}
    for path in primary_paths:
        _, data, provenance = _radio_arrays_from_paths(
            config_path=config_path, dataset_paths=(path,)
        )
        primary_by_serial[provenance[0]["serial"]] = (data, provenance[0])
    repeat_by_serial = {}
    for path in repeat_paths:
        _, data, provenance = _radio_arrays_from_paths(
            config_path=config_path, dataset_paths=(path,)
        )
        repeat_by_serial[provenance[0]["serial"]] = (data, provenance[0])
    common_serials = sorted(set(primary_by_serial) & set(repeat_by_serial))
    rows = []
    all_residuals = []
    for serial in common_serials:
        primary_data, primary_provenance = primary_by_serial[serial]
        repeat_data, repeat_provenance = repeat_by_serial[serial]
        first = _cell_means(primary_data)
        second = _cell_means(repeat_data)
        coordinates = sorted(set(first) & set(second))
        residual = np.asarray(
            [wrap_phase(second[key] - first[key]) for key in coordinates]
        )
        all_residuals.extend(residual.tolist())
        rows.append(
            {
                "serial": serial,
                "common_quality_valid_cells": len(coordinates),
                "primary_analysis_input_sha256": primary_provenance[
                    "analysis_input_sha256"
                ],
                "repeat_analysis_input_sha256": repeat_provenance[
                    "analysis_input_sha256"
                ],
                **_metrics(residual, expected=len(coordinates)),
            }
        )
    return {
        "per_radio": rows,
        **_metrics(np.asarray(all_residuals), expected=len(all_residuals)),
    }


def analyze_low_cost_calibration(
    *,
    config_path: Path,
    dataset_paths: Iterable[Path],
    repeat_dataset_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    primary_paths = tuple(Path(path) for path in dataset_paths)
    repeat_paths = tuple(Path(path) for path in repeat_dataset_paths)
    config, data, provenance = _radio_arrays_from_paths(
        config_path=config_path,
        dataset_paths=primary_paths,
    )
    gain_count = len(config.gains_db)
    frequency_count = len(config.frequencies_hz)
    reference_gain_db = min(
        config.gains_db,
        key=lambda value: abs(value - config.tx_reference_rx_gain_db),
    )
    reference_gain = config.gains_db.index(reference_gain_db)
    reference_frequency_hz = float(np.mean(config.frequencies_hz))
    frequencies = tuple(int(value) for value in config.frequencies_hz)
    if DEFAULT_ONE_ANCHOR_HZ not in frequencies:
        raise ValueError("default one-value anchor is absent from the config")
    if not set(DEFAULT_TWO_ANCHORS_HZ).issubset(frequencies):
        raise ValueError("default two-value anchors are absent from the config")
    spec_by_name = {spec.name: spec for spec in MODEL_SPECS}
    models = {}
    for model_name in BASE_MODEL_NAMES:
        spec = spec_by_name[model_name]
        common = {
            "spec": spec,
            "data": data,
            "provenance": provenance,
            "gain_count": gain_count,
            "frequency_count": frequency_count,
            "reference_gain": reference_gain,
            "reference_frequency_hz": reference_frequency_hz,
        }
        prepared = _prepare_leave_radio_out(**common)
        unadapted = _unadapted_leave_radio_out(prepared=prepared)
        one_anchor = [
            _evaluate_strategy(
                prepared=prepared,
                reference_gain_db=reference_gain_db,
                anchor_frequencies_hz=(frequency,),
            )
            for frequency in frequencies
        ]
        two_anchor = [
            _evaluate_strategy(
                prepared=prepared,
                reference_gain_db=reference_gain_db,
                anchor_frequencies_hz=(first, second),
            )
            for index, first in enumerate(frequencies)
            for second in frequencies[index + 1 :]
        ]
        one_per_frequency = _evaluate_per_frequency_strategy(
            prepared=prepared,
            frequencies_hz=frequencies,
            reference_gain_db=reference_gain_db,
        )
        two_per_frequency_candidates = [
            result
            for gain1 in config.gains_db
            for gain2 in config.gains_db
            if gain1 != gain2
            for result in (
                _evaluate_per_frequency_strategy(
                    prepared=prepared,
                    frequencies_hz=frequencies,
                    reference_gain_db=reference_gain_db,
                    second_gain_pair_db=(gain1, gain2),
                ),
            )
            if result is not None
        ]
        models[model_name] = {
            "label": spec.label,
            "formula": spec.formula,
            "unadapted": unadapted,
            "fixed_one_value": next(
                row
                for row in one_anchor
                if row["anchor_frequencies_hz"] == [DEFAULT_ONE_ANCHOR_HZ]
            ),
            "fixed_two_values": next(
                row
                for row in two_anchor
                if row["anchor_frequencies_hz"] == list(DEFAULT_TWO_ANCHORS_HZ)
            ),
            "exploratory_best_one_value": _best(one_anchor),
            "exploratory_best_two_values": _best(two_anchor),
            "one_value_per_frequency": one_per_frequency,
            "fixed_two_values_per_frequency": next(
                row
                for row in two_per_frequency_candidates
                if row["second_gain_pair_db"] == list(DEFAULT_SECOND_GAIN_PAIR_DB)
            ),
            "exploratory_best_two_values_per_frequency": _best(
                two_per_frequency_candidates
            ),
            "one_value_candidates": one_anchor,
            "two_value_candidates": two_anchor,
        }
    repeatability = (
        _repeatability(
            config_path=config_path,
            primary_paths=primary_paths,
            repeat_paths=repeat_paths,
        )
        if repeat_paths
        else None
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "phase_convention": "RX1 minus RX2",
        "dataset_paths": [radio["dataset_path"] for radio in provenance],
        "repeat_dataset_paths": [str(path) for path in repeat_paths],
        "provenance": provenance,
        "reference_gain_db": int(reference_gain_db),
        "frequencies_hz": list(frequencies),
        "anchor_semantics": (
            "One calibration value is the circular mean phase residual from "
            "the reference-gain cell at one frequency. Two values are measured "
            "at two frequencies and linearly interpolate/extrapolate the "
            "target-radio residual versus frequency."
        ),
        "selection_warning": (
            "Fixed strategies are predeclared. Exploratory best anchors were "
            "selected on these same four radios and require validation on a "
            "fifth unseen radio."
        ),
        "models": models,
        "repeatability": repeatability,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_low_cost_report(result: dict[str, Any]) -> str:
    lines = [
        "# Four-radio low-cost calibration transfer",
        "",
        "## Question",
        "",
        "Can a gain LUT learned from other radios be transferred to a new radio "
        "after measuring only one or two scalar phase-calibration values, instead "
        "of collecting the full 10,404-frame dense calibration?",
        "",
        "The evaluation is leave-one-physical-radio-out. The target radio never "
        "contributes to its universal gain LUT. Its only allowed adaptation data "
        "are the declared reference-gain anchor values; anchor frames are excluded "
        "from scoring.",
        "",
        f"- Phase convention: `{result['phase_convention']}`.",
        f"- Reference gain: {result['reference_gain_db']} dB on RX1 and RX2.",
        f"- {result['anchor_semantics']}",
        f"- {result['selection_warning']}",
        "",
        "## Input radios",
        "",
        "| Serial | Completed | Quality-valid | Scalar-input SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for radio in result["provenance"]:
        lines.append(
            f"| `{radio['serial']}` | {radio['completed_frames']} | "
            f"{radio['quality_valid_observations']} | "
            f"`{radio['analysis_input_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Universal LUT plus target-radio anchors",
            "",
            "| Universal base | Target values | Anchors | MAE ° | RMSE ° | P95 ° | Coverage |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    strategy_order = (
        ("unadapted", "0", "none"),
        ("fixed_one_value", "1", "fixed"),
        ("fixed_two_values", "2", "fixed"),
        ("exploratory_best_one_value", "1", "exploratory best"),
        ("exploratory_best_two_values", "2", "exploratory best"),
    )
    for model in result["models"].values():
        for key, values, label in strategy_order:
            row = model[key]
            anchors = row.get("anchor_frequencies_hz", [])
            anchor_text = (
                "none"
                if not anchors
                else ", ".join(f"{value / 1e6:.0f} MHz" for value in anchors)
            )
            lines.append(
                f"| {model['label']} ({label}) | {values} | {anchor_text} | "
                f"{_fmt(row['circular_mae_deg'])} | "
                f"{_fmt(row['circular_rmse_deg'])} | "
                f"{_fmt(row['circular_p95_deg'])} | "
                f"{100 * row['coverage_fraction']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "![Low-cost strategy comparison](low_cost_strategy_comparison.png)",
            "",
            "## Calibration at each operating frequency",
            "",
            "If a deployment uses one RF channel, the following strategies require "
            "only one or two values for that channel. Evaluating all 12 frequencies "
            "uses 12 or 24 values, still far below a dense gain sweep.",
            "",
            "| Universal base | Values per operating frequency | Second gain pair | MAE ° | RMSE ° | P95 ° |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for model in result["models"].values():
        for key, label in (
            ("one_value_per_frequency", "reference gain only"),
            ("fixed_two_values_per_frequency", "fixed"),
            ("exploratory_best_two_values_per_frequency", "exploratory best"),
        ):
            row = model[key]
            second = row["second_gain_pair_db"]
            second_text = (
                "none"
                if second is None
                else f"RX1 {second[0]} / RX2 {second[1]} dB ({label})"
            )
            lines.append(
                f"| {model['label']} | "
                f"{row['calibration_values_per_operating_frequency']} | "
                f"{second_text} | "
                f"{_fmt(row['circular_mae_deg'])} | "
                f"{_fmt(row['circular_rmse_deg'])} | "
                f"{_fmt(row['circular_p95_deg'])} |"
            )
    lines.extend(
        [
            "",
            "## Per-radio result for the recommended base",
            "",
        ]
    )
    recommended = result["models"]["frequency_specific_additive_gain_universal"]
    lines.extend(
        [
            "| Serial | No calibration MAE ° | One global value MAE ° | Two global values MAE ° | One value at each frequency MAE ° |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for index, radio in enumerate(result["provenance"]):
        key = str(index)
        lines.append(
            f"| `{radio['serial']}` | "
            f"{_fmt(recommended['unadapted']['per_radio'][key]['circular_mae_deg'])} | "
            f"{_fmt(recommended['fixed_one_value']['per_radio'][key]['circular_mae_deg'])} | "
            f"{_fmt(recommended['fixed_two_values']['per_radio'][key]['circular_mae_deg'])} | "
            f"{_fmt(recommended['one_value_per_frequency']['per_radio'][key]['circular_mae_deg'])} |"
        )
    if result["repeatability"] is not None:
        lines.extend(
            [
                "",
                "## Independent dense-run repeatability",
                "",
                "| Serial | Common quality-valid cells | Cell-mean drift MAE ° | RMSE ° | P95 ° |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["repeatability"]["per_radio"]:
            lines.append(
                f"| `{row['serial']}` | {row['common_quality_valid_cells']} | "
                f"{_fmt(row['circular_mae_deg'])} | "
                f"{_fmt(row['circular_rmse_deg'])} | "
                f"{_fmt(row['circular_p95_deg'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The full model ladder is reported separately in `MODEL_MATRIX_REPORT.md`. "
            "This report focuses on deployment cost. A one-value strategy can only "
            "remove a board-wide phase offset. A two-value strategy additionally "
            "removes one frequency-linear differential-delay term. Neither can "
            "represent arbitrary band-specific retune offsets.",
            "",
            "Use the fixed-anchor results for engineering decisions. The exploratory "
            "best-anchor rows are an upper bound and a proposal for the next-board "
            "test, not an unbiased estimate for future hardware.",
            "",
            "A robust field calibration should acquire several frames at each anchor "
            "and store their circular mean as one scalar value. With three frames per "
            "anchor, one- and two-value calibration require 3 or 6 frames instead of "
            "10,404, reductions of 3,468× and 1,734× respectively.",
            "",
            "For multi-frequency operation, one value at each of the 12 measured "
            "frequencies uses 36 robust calibration frames (289× fewer than dense); "
            "two values per frequency use 72 frames (144.5× fewer).",
            "",
            "## Reproduction",
            "",
            "The exact command, dataset list, scalar hashes, JSON results, CSV table, "
            "and plot are stored beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_plot(result: dict[str, Any], output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = (
        ("unadapted", "0 values", "#7f8c8d"),
        ("fixed_one_value", "1 global value", "#4472c4"),
        ("fixed_two_values", "2 global values", "#70ad47"),
        ("one_value_per_frequency", "1 value / frequency", "#5b9bd5"),
        ("fixed_two_values_per_frequency", "2 values / frequency", "#a5a5a5"),
    )
    model_labels = (
        "Frequency + gain LUT",
        "Per-frequency additive LUT",
        "Full cell LUT",
    )
    models = list(result["models"].values())
    positions = np.arange(len(models))
    width = 0.16
    figure, axis = plt.subplots(figsize=(12, 6))
    for strategy_index, (key, label, color) in enumerate(strategies):
        offset = (strategy_index - (len(strategies) - 1) / 2) * width
        axis.bar(
            positions + offset,
            [model[key]["circular_mae_deg"] for model in models],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, model_labels)
    axis.set_ylabel("Leave-one-radio-out circular MAE (degrees)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    path = output_dir / "low_cost_strategy_comparison.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def write_low_cost_bundle(
    *,
    config_path: Path,
    dataset_paths: Iterable[Path],
    repeat_dataset_paths: Iterable[Path],
    output_dir: Path,
    command: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_low_cost_calibration(
        config_path=config_path,
        dataset_paths=dataset_paths,
        repeat_dataset_paths=repeat_dataset_paths,
    )
    result["plot"] = _write_plot(result, output_dir)
    result["reproduction_command"] = command
    (output_dir / "low_cost_calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "LOW_COST_CALIBRATION_REPORT.md").write_text(
        render_low_cost_report(result)
    )
    with (output_dir / "low_cost_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "model",
                "strategy",
                "calibration_values_per_radio",
                "anchor_frequencies_hz",
                "mae_deg",
                "rmse_deg",
                "p95_deg",
                "coverage_fraction",
            )
        )
        for model_name, model in result["models"].items():
            for strategy in (
                "unadapted",
                "fixed_one_value",
                "fixed_two_values",
                "exploratory_best_one_value",
                "exploratory_best_two_values",
                "one_value_per_frequency",
                "fixed_two_values_per_frequency",
                "exploratory_best_two_values_per_frequency",
            ):
                row = model[strategy]
                writer.writerow(
                    (
                        model_name,
                        strategy,
                        row["calibration_values_per_radio"],
                        ";".join(
                            str(value) for value in row.get("anchor_frequencies_hz", [])
                        ),
                        row["circular_mae_deg"],
                        row["circular_rmse_deg"],
                        row["circular_p95_deg"],
                        row["coverage_fraction"],
                    )
                )
    return result
