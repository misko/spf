"""Human-readable analysis artifacts for completed calibration datasets."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MISMATCH_BANDS_DB = (
    (0, 5),
    (6, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 72),
)


def build_analysis_summary(
    validation: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Combine cell validation and model diagnostics into one stable summary."""

    if validation.get("serial") != model.get("serial"):
        raise ValueError("validation and model serials differ")
    cells = validation.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("validation report does not contain cells")

    frequency_rows = []
    by_frequency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_frequency[int(cell["frequency_hz"])].append(cell)
    model_by_frequency = {
        int(row["frequency_hz"]): row for row in model.get("frequency_models", [])
    }
    for frequency_hz, rows in sorted(by_frequency.items()):
        passing = [row for row in rows if row["pass"]]
        repeat_std = [
            float(row["phase_circular_std_deg"])
            for row in passing
            if row["phase_circular_std_deg"] is not None
        ]
        fitted = model_by_frequency.get(frequency_hz, {})
        frequency_rows.append(
            {
                "frequency_hz": frequency_hz,
                "passing_cells": len(passing),
                "total_cells": len(rows),
                "passing_fraction": len(passing) / len(rows),
                "median_repeat_phase_std_deg": (
                    float(np.median(repeat_std)) if repeat_std else None
                ),
                "model_status": fitted.get("status", "missing"),
                "model_observations": fitted.get("n_observations", 0),
                "training_metrics": fitted.get("training_metrics"),
            }
        )

    mismatch_rows = []
    for lower, upper in MISMATCH_BANDS_DB:
        selected = [
            cell
            for cell in cells
            if lower
            <= abs(int(cell["gain_rx1_db"]) - int(cell["gain_rx2_db"]))
            <= upper
        ]
        passing = sum(bool(cell["pass"]) for cell in selected)
        quality_valid = sum(int(cell["n_quality_valid"]) for cell in selected)
        complete = sum(int(cell["n_complete"]) for cell in selected)
        mismatch_rows.append(
            {
                "minimum_mismatch_db": lower,
                "maximum_mismatch_db": upper,
                "passing_cells": passing,
                "total_cells": len(selected),
                "passing_cell_fraction": (
                    passing / len(selected) if selected else None
                ),
                "quality_valid_frames": quality_valid,
                "complete_frames": complete,
                "quality_valid_frame_fraction": (
                    quality_valid / complete if complete else None
                ),
            }
        )

    return {
        "schema": "spf.calibration.dual_rx_gain_frequency.analysis",
        "schema_version": 1,
        "serial": validation["serial"],
        "validation_status": validation["status"],
        "completed_frames": validation["completed_frames"],
        "expected_frames": validation["expected_frames"],
        "quality_valid_frames": validation["quality_valid_frames"],
        "passing_cells": validation["passing_cells"],
        "expected_cells": validation["expected_cells"],
        "quality_reason_counts": validation["quality_reason_counts"],
        "frequency_summary": frequency_rows,
        "gain_mismatch_summary": mismatch_rows,
        "cross_validation_metrics": model.get("cross_validation_metrics"),
        "frequency_intercept_delay_model": model.get("frequency_intercept_delay_model"),
    }


def _format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the stable summary as a concise engineering report."""

    valid_fraction = (
        summary["quality_valid_frames"] / summary["completed_frames"]
        if summary["completed_frames"]
        else 0.0
    )
    lines = [
        "# Dual-RX gain/frequency calibration report",
        "",
        f"- Pluto serial: `{summary['serial']}`",
        (
            f"- Frames: {summary['completed_frames']}/"
            f"{summary['expected_frames']} complete; "
            f"{summary['quality_valid_frames']} phase-valid "
            f"({valid_fraction:.1%})"
        ),
        (
            f"- Cells: {summary['passing_cells']}/"
            f"{summary['expected_cells']} pass the three-epoch criterion"
        ),
        f"- Validation status: `{summary['validation_status']}`",
        "",
        "## Frequency coverage",
        "",
        "| Frequency (MHz) | Passing cells | Coverage | Median repeat std | Model observations | Train RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["frequency_summary"]:
        training = row.get("training_metrics") or {}
        lines.append(
            "| "
            f"{row['frequency_hz'] / 1e6:.3f} | "
            f"{row['passing_cells']}/{row['total_cells']} | "
            f"{row['passing_fraction']:.1%} | "
            f"{_format_number(row['median_repeat_phase_std_deg'])}° | "
            f"{row['model_observations']} | "
            f"{_format_number(training.get('circular_rmse_deg'))}° |"
        )

    lines.extend(
        [
            "",
            "## Gain-mismatch coverage",
            "",
            "| Absolute RX gain mismatch | Passing cells | Cell coverage | Valid frames | Frame coverage |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["gain_mismatch_summary"]:
        cell_fraction = row["passing_cell_fraction"]
        frame_fraction = row["quality_valid_frame_fraction"]
        lines.append(
            "| "
            f"{row['minimum_mismatch_db']}–{row['maximum_mismatch_db']} dB | "
            f"{row['passing_cells']}/{row['total_cells']} | "
            f"{_format_percent(cell_fraction)} | "
            f"{row['quality_valid_frames']}/{row['complete_frames']} | "
            f"{_format_percent(frame_fraction)} |"
        )

    cross_validation = summary.get("cross_validation_metrics")
    lines.extend(["", "## Model diagnostics", ""])
    if cross_validation:
        lines.extend(
            [
                (
                    "- Leave-one-epoch-out circular MAE: "
                    f"{cross_validation['circular_mae_deg']:.2f}°"
                ),
                (
                    "- Leave-one-epoch-out circular RMSE: "
                    f"{cross_validation['circular_rmse_deg']:.2f}°"
                ),
                (
                    "- Leave-one-epoch-out circular p95: "
                    f"{cross_validation['circular_p95_deg']:.2f}°"
                ),
                (
                    "- Leave-one-epoch-out maximum error: "
                    f"{cross_validation['circular_max_deg']:.2f}°"
                ),
            ]
        )
    else:
        lines.append("- Cross-validation was unavailable.")
    lines.extend(
        [
            "",
            (
                "The frequency-intercept slope is descriptive only. LO retunes "
                "can introduce phase state changes, so it is not asserted to be "
                "physical cable delay."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _plot_heatmaps(
    validation: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = validation["cells"]
    gains = sorted(
        {int(cell["gain_rx1_db"]) for cell in cells}
        | {int(cell["gain_rx2_db"]) for cell in cells}
    )
    gain_index = {gain: index for index, gain in enumerate(gains)}
    filenames = []
    by_frequency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_frequency[int(cell["frequency_hz"])].append(cell)

    for frequency_hz, rows in sorted(by_frequency.items()):
        valid_count = np.zeros((len(gains), len(gains)), dtype=np.float64)
        phase_deg = np.full((len(gains), len(gains)), np.nan)
        repeat_std = np.full((len(gains), len(gains)), np.nan)
        for row in rows:
            y = gain_index[int(row["gain_rx2_db"])]
            x = gain_index[int(row["gain_rx1_db"])]
            valid_count[y, x] = int(row["n_quality_valid"])
            if row["pass"]:
                phase_deg[y, x] = math.degrees(float(row["phase_mean_rad"]))
                repeat_std[y, x] = float(row["phase_circular_std_deg"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        panels = (
            (valid_count, "Valid epochs (of 3)", "viridis", 0, 3),
            (phase_deg, "Passing-cell phase (degrees)", "twilight", -180, 180),
            (repeat_std, "Repeat circular std (degrees)", "magma", 0, 5),
        )
        extent = (gains[0] - 0.5, gains[-1] + 0.5, gains[0] - 0.5, gains[-1] + 0.5)
        for axis, (matrix, title, cmap, minimum, maximum) in zip(axes, panels):
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
                extent=extent,
                cmap=cmap,
                vmin=minimum,
                vmax=maximum,
            )
            axis.set_title(title)
            axis.set_xlabel("RX1 manual gain (dB)")
            axis.set_ylabel("RX2 manual gain (dB)")
            fig.colorbar(image, ax=axis, shrink=0.8)
        fig.suptitle(
            f"{validation['serial']} — {frequency_hz / 1e6:.3f} MHz",
            fontsize=12,
        )
        filename = f"phase_surface_{frequency_hz}.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def write_analysis_bundle(
    *,
    validation_path: Path,
    model_path: Path,
    output_dir: Path,
    plots: bool = True,
) -> dict[str, Any]:
    validation = json.loads(Path(validation_path).read_text())
    model = json.loads(Path(model_path).read_text())
    summary = build_analysis_summary(validation, model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["plot_files"] = _plot_heatmaps(validation, output_dir) if plots else []
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(render_markdown(summary))
    return summary
