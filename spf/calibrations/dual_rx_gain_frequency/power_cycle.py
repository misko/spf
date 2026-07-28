"""Compare compact before/after power-cycle phase-calibration surveys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase


SCHEMA = "spf.calibration.dual_rx_gain_frequency.power_cycle_comparison"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PowerCycleThresholds:
    """Acceptance thresholds for drift remaining after each strategy."""

    maximum_mae_deg: float = 2.0
    maximum_p95_deg: float = 5.0
    minimum_common_cell_fraction: float = 0.8

    def validate(self) -> None:
        if self.maximum_mae_deg <= 0:
            raise ValueError("maximum MAE must be positive")
        if self.maximum_p95_deg <= 0:
            raise ValueError("maximum P95 must be positive")
        if not 0 < self.minimum_common_cell_fraction <= 1:
            raise ValueError("minimum common-cell fraction must be in (0, 1]")


def _circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError("cannot average an empty circular vector")
    return float(np.angle(np.mean(np.exp(1j * values))))


def _metrics(values_rad: np.ndarray, *, expected: int) -> dict[str, Any]:
    values = np.asarray(values_rad, dtype=np.float64)
    if not values.size:
        return {
            "evaluated_cells": 0,
            "expected_cells": int(expected),
            "coverage_fraction": 0.0,
            "circular_mae_deg": None,
            "circular_rmse_deg": None,
            "circular_p95_deg": None,
            "circular_max_deg": None,
            "circular_bias_deg": None,
        }
    absolute_deg = np.abs(np.degrees(values))
    return {
        "evaluated_cells": int(values.size),
        "expected_cells": int(expected),
        "coverage_fraction": float(values.size / expected) if expected else 0.0,
        "circular_mae_deg": float(np.mean(absolute_deg)),
        "circular_rmse_deg": float(np.sqrt(np.mean(absolute_deg**2))),
        "circular_p95_deg": float(np.percentile(absolute_deg, 95)),
        "circular_max_deg": float(np.max(absolute_deg)),
        "circular_bias_deg": float(np.degrees(_circular_mean(values))),
    }


def _passes(metrics: dict[str, Any], thresholds: PowerCycleThresholds) -> bool:
    return bool(
        metrics["circular_mae_deg"] is not None
        and metrics["circular_p95_deg"] is not None
        and metrics["circular_mae_deg"] <= thresholds.maximum_mae_deg
        and metrics["circular_p95_deg"] <= thresholds.maximum_p95_deg
    )


def analyze_cell_maps(
    before: dict[tuple[int, int, int], float],
    after: dict[tuple[int, int, int], float],
    *,
    expected_cells: int,
    reference_gain_db: int = 26,
    global_anchor_frequency_hz: int = 2_412_000_000,
    thresholds: PowerCycleThresholds = PowerCycleThresholds(),
    expected_frequencies_hz: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Classify whether a power cycle changes baseline or gain-LUT shape.

    Keys are ``(frequency_hz, gain_rx1_db, gain_rx2_db)``. Values are
    circular-mean RX1-minus-RX2 phases in radians.
    """

    thresholds.validate()
    if expected_cells <= 0:
        raise ValueError("expected cells must be positive")
    coordinates = sorted(set(before) & set(after))
    raw = {
        coordinate: float(wrap_phase(after[coordinate] - before[coordinate]))
        for coordinate in coordinates
    }
    raw_values = np.asarray([raw[key] for key in coordinates], dtype=np.float64)
    raw_metrics = _metrics(raw_values, expected=expected_cells)
    sufficient_coverage = (
        raw_metrics["coverage_fraction"] >= thresholds.minimum_common_cell_fraction
    )

    global_anchor_key = (
        int(global_anchor_frequency_hz),
        int(reference_gain_db),
        int(reference_gain_db),
    )
    global_anchor_available = global_anchor_key in raw
    if global_anchor_available:
        global_anchor_delta = raw[global_anchor_key]
        global_adjusted = wrap_phase(raw_values - global_anchor_delta)
        global_metrics = _metrics(global_adjusted, expected=expected_cells)
    else:
        global_anchor_delta = None
        global_metrics = _metrics(np.asarray([]), expected=expected_cells)

    frequencies = sorted(
        {int(value) for value in expected_frequencies_hz}
        if expected_frequencies_hz is not None
        else {key[0] for key in coordinates}
    )
    if not frequencies:
        raise ValueError("at least one expected frequency is required")
    per_frequency_anchor_delta = {}
    missing_frequency_anchors = []
    for frequency_hz in frequencies:
        anchor_key = (frequency_hz, reference_gain_db, reference_gain_db)
        if anchor_key not in raw:
            missing_frequency_anchors.append(frequency_hz)
        else:
            per_frequency_anchor_delta[frequency_hz] = raw[anchor_key]

    if not missing_frequency_anchors:
        per_frequency_adjusted = np.asarray(
            [
                wrap_phase(raw[key] - per_frequency_anchor_delta[key[0]])
                for key in coordinates
            ],
            dtype=np.float64,
        )
        per_frequency_metrics = _metrics(
            per_frequency_adjusted, expected=expected_cells
        )
    else:
        per_frequency_metrics = _metrics(np.asarray([]), expected=expected_cells)

    best_global_offset = _circular_mean(raw_values) if raw_values.size else None
    best_global_metrics = (
        _metrics(
            wrap_phase(raw_values - best_global_offset),
            expected=expected_cells,
        )
        if best_global_offset is not None
        else _metrics(np.asarray([]), expected=expected_cells)
    )

    best_frequency_offsets = {}
    best_frequency_adjusted = []
    for frequency_hz in frequencies:
        values = np.asarray(
            [raw[key] for key in coordinates if key[0] == frequency_hz],
            dtype=np.float64,
        )
        if not values.size:
            continue
        offset = _circular_mean(values)
        best_frequency_offsets[frequency_hz] = offset
        best_frequency_adjusted.extend(wrap_phase(values - offset).tolist())
    best_frequency_metrics = _metrics(
        np.asarray(best_frequency_adjusted), expected=expected_cells
    )

    if not sufficient_coverage:
        verdict = "inconclusive_insufficient_common_cells"
    elif missing_frequency_anchors:
        verdict = "inconclusive_missing_reference_anchors"
    elif _passes(raw_metrics, thresholds):
        verdict = "reusable_without_session_calibration"
    elif global_anchor_available and _passes(global_metrics, thresholds):
        verdict = "one_global_session_anchor_required"
    elif not missing_frequency_anchors and _passes(per_frequency_metrics, thresholds):
        verdict = "one_anchor_per_frequency_required"
    else:
        verdict = "gain_dependent_recalibration_required"

    return {
        "verdict": verdict,
        "power_cycle_reproducible_without_calibration": (
            verdict == "reusable_without_session_calibration"
        ),
        "expected_cells": int(expected_cells),
        "common_passing_cells": len(coordinates),
        "thresholds": asdict(thresholds),
        "reference_gain_db": int(reference_gain_db),
        "global_anchor_frequency_hz": int(global_anchor_frequency_hz),
        "raw_drift": raw_metrics,
        "one_global_anchor_adjusted": global_metrics,
        "one_anchor_per_frequency_adjusted": per_frequency_metrics,
        "diagnostic_best_global_offset_adjusted": best_global_metrics,
        "diagnostic_best_frequency_offsets_adjusted": best_frequency_metrics,
        "global_anchor_delta_deg": (
            float(np.degrees(global_anchor_delta))
            if global_anchor_delta is not None
            else None
        ),
        "per_frequency_anchor_delta_deg": {
            str(frequency_hz): float(np.degrees(value))
            for frequency_hz, value in per_frequency_anchor_delta.items()
        },
        "diagnostic_best_frequency_offset_deg": {
            str(frequency_hz): float(np.degrees(value))
            for frequency_hz, value in best_frequency_offsets.items()
        },
        "missing_frequency_anchors_hz": missing_frequency_anchors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"required power-cycle input is missing: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _passing_cell_map(
    validation: dict[str, Any],
) -> dict[tuple[int, int, int], float]:
    result = {}
    for row in validation.get("cells", []):
        if not row.get("pass") or row.get("phase_mean_rad") is None:
            continue
        key = (
            int(row["frequency_hz"]),
            int(row["gain_rx1_db"]),
            int(row["gain_rx2_db"]),
        )
        if key in result:
            raise ValueError(f"duplicate validation cell: {key}")
        result[key] = float(row["phase_mean_rad"])
    return result


def _validation_coordinates(
    validation: dict[str, Any],
) -> set[tuple[int, int, int]]:
    result = set()
    for row in validation.get("cells", []):
        key = (
            int(row["frequency_hz"]),
            int(row["gain_rx1_db"]),
            int(row["gain_rx2_db"]),
        )
        if key in result:
            raise ValueError(f"duplicate validation cell: {key}")
        result.add(key)
    expected = int(validation.get("expected_cells", 0))
    if expected <= 0 or len(result) != expected:
        raise ValueError(
            f"validation contains {len(result)} coordinates, expected {expected}"
        )
    return result


def _load_run(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    plan_path = root / "automation_plan.json"
    result_path = root / "automation_result.json"
    plan = _read_json(plan_path)
    result = _read_json(result_path)
    if result.get("status") != "complete":
        raise ValueError(f"power-cycle input run is not complete: {root}")
    serials = tuple(str(value) for value in plan.get("radio_serials", []))
    if not serials or len(serials) != len(set(serials)):
        raise ValueError(f"invalid radio serial list in {plan_path}")
    validations = {}
    validation_sha256 = {}
    for serial in serials:
        path = root / serial / "validation.json"
        validation = _read_json(path)
        if validation.get("serial") != serial:
            raise ValueError(f"validation serial mismatch: {path}")
        if validation.get("status") == "partial":
            raise ValueError(f"partial dataset cannot be compared: {path}")
        _validation_coordinates(validation)
        validations[serial] = validation
        validation_sha256[serial] = _sha256(path)
    return {
        "root": str(root),
        "plan": plan,
        "plan_sha256": _sha256(plan_path),
        "result_sha256": _sha256(result_path),
        "validation_sha256": validation_sha256,
        "serials": serials,
        "validations": validations,
    }


def compare_power_cycle_runs(
    *,
    before_root: Path,
    after_root: Path,
    reference_gain_db: int = 26,
    global_anchor_frequency_hz: int = 2_412_000_000,
    thresholds: PowerCycleThresholds = PowerCycleThresholds(),
) -> dict[str, Any]:
    """Compare two complete automated subsample roots by physical serial."""

    before = _load_run(before_root)
    after = _load_run(after_root)
    if set(before["serials"]) != set(after["serials"]):
        raise ValueError("before/after runs contain different physical serials")
    before_plan = before["plan"]
    after_plan = after["plan"]
    if before_plan.get("calibration_config_sha256") != after_plan.get(
        "calibration_config_sha256"
    ):
        raise ValueError("before/after runs used different calibration configs")
    if before_plan.get("firmware") != after_plan.get("firmware"):
        raise ValueError("before/after runs used different firmware provenance")

    per_radio = []
    for serial in sorted(before["serials"]):
        before_validation = before["validations"][serial]
        after_validation = after["validations"][serial]
        before_expected = int(before_validation.get("expected_cells", 0))
        after_expected = int(after_validation.get("expected_cells", 0))
        if before_expected <= 0 or before_expected != after_expected:
            raise ValueError(f"{serial}: expected cell counts do not match")
        before_coordinates = _validation_coordinates(before_validation)
        after_coordinates = _validation_coordinates(after_validation)
        if before_coordinates != after_coordinates:
            raise ValueError(f"{serial}: before/after requested cells do not match")
        frequencies_hz = sorted({key[0] for key in before_coordinates})
        comparison = analyze_cell_maps(
            _passing_cell_map(before_validation),
            _passing_cell_map(after_validation),
            expected_cells=before_expected,
            reference_gain_db=reference_gain_db,
            global_anchor_frequency_hz=global_anchor_frequency_hz,
            thresholds=thresholds,
            expected_frequencies_hz=frequencies_hz,
        )
        per_radio.append(
            {
                "serial": serial,
                "before_validation_status": before_validation.get("status"),
                "after_validation_status": after_validation.get("status"),
                **comparison,
            }
        )

    verdict_priority = {
        "reusable_without_session_calibration": 0,
        "one_global_session_anchor_required": 1,
        "one_anchor_per_frequency_required": 2,
        "gain_dependent_recalibration_required": 3,
        "inconclusive_insufficient_common_cells": 4,
        "inconclusive_missing_reference_anchors": 4,
    }
    overall = max(
        (row["verdict"] for row in per_radio),
        key=lambda value: verdict_priority[value],
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "overall_verdict": overall,
        "all_radios_reproducible_without_calibration": all(
            row["power_cycle_reproducible_without_calibration"] for row in per_radio
        ),
        "before": {
            key: before[key]
            for key in (
                "root",
                "plan_sha256",
                "result_sha256",
                "validation_sha256",
            )
        },
        "after": {
            key: after[key]
            for key in (
                "root",
                "plan_sha256",
                "result_sha256",
                "validation_sha256",
            )
        },
        "per_radio": per_radio,
    }


def _metric_text(metrics: dict[str, Any]) -> str:
    if metrics["circular_mae_deg"] is None:
        return "unavailable"
    return (
        f"{metrics['circular_mae_deg']:.3f}° MAE, "
        f"{metrics['circular_p95_deg']:.3f}° P95"
    )


def render_power_cycle_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Pluto+ power-cycle calibration reproducibility",
        "",
        f"Overall verdict: `{result['overall_verdict']}`.",
        "",
        "The comparison uses only cells that passed the normal three-epoch",
        "quality and repeatability gates in both runs. Phase convention is",
        "RX1 minus RX2.",
        "",
        "| Radio | Common cells | Raw drift | One global anchor | "
        "One anchor/frequency | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for radio in result["per_radio"]:
        lines.append(
            f"| `{radio['serial']}` | {radio['common_passing_cells']}/"
            f"{radio['expected_cells']} | {_metric_text(radio['raw_drift'])} | "
            f"{_metric_text(radio['one_global_anchor_adjusted'])} | "
            f"{_metric_text(radio['one_anchor_per_frequency_adjusted'])} | "
            f"`{radio['verdict']}` |"
        )
    thresholds = result["per_radio"][0]["thresholds"]
    lines.extend(
        [
            "",
            "## Pass/fail policy",
            "",
            f"A strategy passes when MAE is at most "
            f"{thresholds['maximum_mae_deg']:.3f}° and P95 is at most "
            f"{thresholds['maximum_p95_deg']:.3f}°, with at least "
            f"{100 * thresholds['minimum_common_cell_fraction']:.1f}% common "
            "passing-cell coverage.",
            "",
            "- Raw drift passes: reuse the stored calibration after power cycle.",
            "- One-global-anchor residual passes: refresh one session-wide offset.",
            f"- Per-frequency-anchor residual passes: refresh "
            f"{result['per_radio'][0]['reference_gain_db']}/"
            f"{result['per_radio'][0]['reference_gain_db']} dB at each operating "
            "frequency.",
            "- Per-frequency-anchor residual fails: gain-dependent recalibration "
            "is required.",
            "",
            "## Provenance",
            "",
            f"- Before: `{result['before']['root']}`",
            f"- After: `{result['after']['root']}`",
            "",
            "The comparator requires identical radio serial ordering, calibration",
            "configuration hash, and firmware provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_power_cycle_bundle(
    *,
    before_root: Path,
    after_root: Path,
    output_dir: Path,
    reference_gain_db: int = 26,
    global_anchor_frequency_hz: int = 2_412_000_000,
    thresholds: PowerCycleThresholds = PowerCycleThresholds(),
) -> dict[str, Any]:
    result = compare_power_cycle_runs(
        before_root=before_root,
        after_root=after_root,
        reference_gain_db=reference_gain_db,
        global_anchor_frequency_hz=global_anchor_frequency_hz,
        thresholds=thresholds,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "power_cycle_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(render_power_cycle_markdown(result))
    return result
