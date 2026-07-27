"""Cross-radio interpretation of fitted dual-RX phase baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase


CROSS_RADIO_SCHEMA = "spf.calibration.dual_rx_gain_frequency.cross_radio"
CROSS_RADIO_SCHEMA_VERSION = 1


def _residual_metrics(residual_rad: np.ndarray) -> dict[str, float]:
    absolute_deg = np.abs(np.degrees(wrap_phase(residual_rad)))
    return {
        "circular_mae_deg": float(np.mean(absolute_deg)),
        "circular_rmse_deg": float(np.sqrt(np.mean(absolute_deg**2))),
        "circular_p95_deg": float(np.percentile(absolute_deg, 95)),
        "circular_max_deg": float(np.max(absolute_deg)),
    }


def _linear_delay_fit(
    frequency_hz: np.ndarray,
    phase_delta_unwrapped_rad: np.ndarray,
) -> dict[str, Any]:
    if frequency_hz.size < 2:
        raise ValueError("cross-radio delay fit needs at least two frequencies")
    reference_hz = float(np.mean(frequency_hz))
    slope, phase_at_reference = np.polyfit(
        frequency_hz - reference_hz,
        phase_delta_unwrapped_rad,
        1,
    )
    fitted = phase_at_reference + slope * (frequency_hz - reference_hz)
    residual = wrap_phase(phase_delta_unwrapped_rad - fitted)
    delay_seconds = -float(slope) / (2 * np.pi)
    return {
        "frequency_count": int(frequency_hz.size),
        "minimum_frequency_hz": int(np.min(frequency_hz)),
        "maximum_frequency_hz": int(np.max(frequency_hz)),
        "reference_frequency_hz": reference_hz,
        "phase_at_reference_rad_unwrapped": float(phase_at_reference),
        "slope_rad_per_hz": float(slope),
        "effective_delay_seconds": delay_seconds,
        "free_space_equivalent_path_m": delay_seconds * 299_792_458.0,
        "residual_metrics": _residual_metrics(residual),
    }


def _intercepts(model: dict[str, Any]) -> dict[int, float]:
    rows = model.get("frequency_models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("model has no fitted frequency models")
    result = {}
    for row in rows:
        if row.get("status") != "fit":
            continue
        result[int(row["frequency_hz"])] = float(row["intercept_rad"])
    if not result:
        raise ValueError("model has no usable fitted frequency intercepts")
    return result


def compare_radio_models(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> dict[str, Any]:
    """Fit descriptive A-minus-B differential delay globally and by band."""

    serial_a = model_a.get("serial")
    serial_b = model_b.get("serial")
    if not serial_a or not serial_b or serial_a == serial_b:
        raise ValueError("cross-radio comparison requires two distinct serials")
    intercept_a = _intercepts(model_a)
    intercept_b = _intercepts(model_b)
    frequencies = np.asarray(
        sorted(set(intercept_a).intersection(intercept_b)),
        dtype=np.float64,
    )
    if frequencies.size < 2:
        raise ValueError("models do not share at least two fitted frequencies")
    phase_delta = np.unwrap(
        np.asarray(
            [
                wrap_phase(intercept_a[int(frequency)] - intercept_b[int(frequency)])
                for frequency in frequencies
            ],
            dtype=np.float64,
        )
    )

    regions = (
        ("low_full_gain_table", frequencies <= 1_300_000_000),
        (
            "middle_full_gain_table",
            (frequencies > 1_300_000_000) & (frequencies <= 4_000_000_000),
        ),
        ("high_full_gain_table", frequencies > 4_000_000_000),
        (
            "vtx_5_7_to_5_9_ghz",
            (frequencies >= 5_700_000_000) & (frequencies <= 5_900_000_000),
        ),
    )
    region_fits = {
        name: _linear_delay_fit(frequencies[selected], phase_delta[selected])
        for name, selected in regions
        if np.count_nonzero(selected) >= 2
    }
    individual = {}
    for label, model in (("radio_a", model_a), ("radio_b", model_b)):
        delay = model.get("frequency_intercept_delay_model")
        individual[label] = (
            {
                "descriptive_delay_seconds": delay["descriptive_delay_seconds"],
                "free_space_equivalent_path_m": delay["equivalent_free_space_path_m"],
                "residual_metrics": delay["fit_residual_metrics"],
            }
            if delay
            else None
        )

    return {
        "schema": CROSS_RADIO_SCHEMA,
        "schema_version": CROSS_RADIO_SCHEMA_VERSION,
        "phase_delta_convention": "radio_a intercept minus radio_b intercept",
        "radio_a_serial": serial_a,
        "radio_b_serial": serial_b,
        "common_frequency_count": int(frequencies.size),
        "individual_descriptive_delay": individual,
        "global_difference_fit": _linear_delay_fit(frequencies, phase_delta),
        "region_difference_fits": region_fits,
        "interpretation": (
            "A stable physical differential-path mismatch would produce a "
            "consistent delay across regions. Band-dependent sign or magnitude "
            "changes indicate that gain tables, analogue filtering, LO retunes, "
            "calibration state, or external cables also contribute."
        ),
        "warning": (
            "Effective delays and free-space-equivalent lengths are descriptive "
            "slopes, not measurements of literal PCB trace length."
        ),
    }


def render_cross_radio_markdown(result: dict[str, Any]) -> str:
    """Render a compact engineering interpretation of a comparison."""

    lines = [
        "# Cross-radio differential-phase interpretation",
        "",
        f"- Radio A: `{result['radio_a_serial']}`",
        f"- Radio B: `{result['radio_b_serial']}`",
        f"- Common frequencies: {result['common_frequency_count']}",
        f"- Convention: {result['phase_delta_convention']}",
        "",
        "## Descriptive linear-delay fits",
        "",
        "| Region | Frequencies | Effective delay | Free-space equivalent | Residual MAE / p95 / max |",
        "|---|---:|---:|---:|---:|",
    ]
    fits = [("All frequencies", result["global_difference_fit"])]
    fits.extend(
        (name.replace("_", " "), fit)
        for name, fit in result["region_difference_fits"].items()
    )
    for label, fit in fits:
        residual = fit["residual_metrics"]
        lines.append(
            f"| {label} | {fit['frequency_count']} | "
            f"{fit['effective_delay_seconds'] * 1e12:.2f} ps | "
            f"{fit['free_space_equivalent_path_m'] * 1e3:.2f} mm | "
            f"{residual['circular_mae_deg']:.2f}° / "
            f"{residual['circular_p95_deg']:.2f}° / "
            f"{residual['circular_max_deg']:.2f}° |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
            result["warning"],
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_radio_bundle(
    *,
    model_a_path: Path,
    model_b_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write deterministic JSON and Markdown from two fitted model files."""

    result = compare_radio_models(
        json.loads(Path(model_a_path).read_text()),
        json.loads(Path(model_b_path).read_text()),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_radio_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "CROSS_RADIO_REPORT.md").write_text(
        render_cross_radio_markdown(result)
    )
    return result
