"""Human-readable analysis artifacts for completed calibration datasets."""

from __future__ import annotations

import gc
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
                "cross_validation_metrics": fitted.get("cross_validation_metrics"),
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
        "schema_version": 2,
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
        "model_comparisons": model.get("model_comparisons"),
        "frequency_intercept_delay_model": model.get("frequency_intercept_delay_model"),
    }


def _format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _close_figure(plt: Any, fig: Any) -> None:
    """Release renderer arrays promptly during large multi-frequency reports."""

    fig.clear()
    plt.close(fig)
    gc.collect()


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
        "| Frequency (MHz) | Passing cells | Coverage | Median repeat std | Model observations | Train RMSE | CV MAE / p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["frequency_summary"]:
        training = row.get("training_metrics") or {}
        cross_validation = row.get("cross_validation_metrics") or {}
        lines.append(
            "| "
            f"{row['frequency_hz'] / 1e6:.3f} | "
            f"{row['passing_cells']}/{row['total_cells']} | "
            f"{row['passing_fraction']:.1%} | "
            f"{_format_number(row['median_repeat_phase_std_deg'])}° | "
            f"{row['model_observations']} | "
            f"{_format_number(training.get('circular_rmse_deg'))}° | "
            f"{_format_number(cross_validation.get('circular_mae_deg'))}° / "
            f"{_format_number(cross_validation.get('circular_p95_deg'))}° |"
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

    model_comparisons = summary.get("model_comparisons") or {}
    comparison_specs = (
        (
            "additive_vs_gain_difference_only",
            "Ordered additive vs gain difference",
            "additive",
            "gain_difference_only",
        ),
        (
            "additive_vs_cell_interaction",
            "Additive vs cell interaction",
            "additive",
            "additive_plus_cell_interaction",
        ),
        (
            "frequency_specific_vs_shared_gain_curves",
            "Frequency-specific vs shared curves",
            "frequency_specific_gain_curves",
            "frequency_shared_gain_curves",
        ),
        (
            "frequency_specific_vs_gain_table_shared_gain_curves",
            "Frequency-specific vs gain-table-shared curves",
            "frequency_specific_gain_curves",
            "gain_table_shared_gain_curves",
        ),
        (
            "frequency_specific_vs_linear_delay_intercept",
            "Per-frequency vs constant-plus-delay baseline",
            "frequency_specific_intercepts",
            "linear_delay_intercept",
        ),
        (
            "unanchored_vs_one_frame_anchor",
            "Unanchored vs one-frame anchor",
            "unanchored",
            "one_frame_anchored",
        ),
    )
    if any(model_comparisons.get(spec[0]) for spec in comparison_specs):
        lines.extend(
            [
                "",
                "### Paired model comparisons",
                "",
                "| Comparison | Held-out frames | First MAE / p95 | Second MAE / p95 | Recommended |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for key, label, first_name, second_name in comparison_specs:
            comparison = model_comparisons.get(key)
            if not comparison:
                continue
            first = comparison[first_name]
            second = comparison[second_name]
            lines.append(
                f"| {label} | {comparison['n_observations']} | "
                f"{first['circular_mae_deg']:.2f}° / "
                f"{first['circular_p95_deg']:.2f}° | "
                f"{second['circular_mae_deg']:.2f}° / "
                f"{second['circular_p95_deg']:.2f}° | "
                f"`{comparison['recommended_model']}` |"
            )
        lines.extend(
            [
                "",
                (
                    "Every row uses an identical held-out observation mask. "
                    "Differences within the declared 0.1° MAE equivalence margin "
                    "select the predeclared simpler operational model."
                ),
            ]
        )

    delay_model = summary.get("frequency_intercept_delay_model")
    if delay_model:
        residual = delay_model["fit_residual_metrics"]
        lines.extend(
            [
                "",
                "### Effective differential-delay description",
                "",
                (
                    "- Common reference gain: "
                    f"{delay_model['reference_gain_db']} dB on RX1 and RX2"
                ),
                (
                    "- Phase slope: "
                    f"{math.degrees(delay_model['slope_rad_per_hz']) * 1e8:.2f}° "
                    "per 100 MHz"
                ),
                (
                    "- Effective differential delay: "
                    f"{delay_model['descriptive_delay_seconds'] * 1e9:.3f} ns"
                ),
                (
                    "- Free-space-equivalent signed path difference: "
                    f"{delay_model['equivalent_free_space_path_m'] * 100:.2f} cm"
                ),
                (
                    "- Linear-fit residual RMSE / maximum: "
                    f"{residual['circular_rmse_deg']:.2f}° / "
                    f"{residual['circular_max_deg']:.2f}°"
                ),
                "",
                delay_model["warning"],
            ]
        )

    confidence_tiers = model_comparisons.get("confidence_tiers") or []
    if confidence_tiers:
        lines.extend(
            [
                "",
                "### Signal-confidence tiers",
                "",
                "| Minimum SNR in both channels | Eligible frames | Held-out frames | MAE | RMSE | p95 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tier in confidence_tiers:
            additive = tier.get("additive")
            if not additive:
                continue
            lines.append(
                f"| {tier['minimum_both_channel_tone_snr_db']:.0f} dB | "
                f"{tier['eligible_observations']} | "
                f"{additive['n_observations']} | "
                f"{additive['circular_mae_deg']:.2f}° | "
                f"{additive['circular_rmse_deg']:.2f}° | "
                f"{additive['circular_p95_deg']:.2f}° |"
            )
    if not delay_model:
        lines.extend(
            [
                "",
                (
                    "An effective differential-delay fit requires at least two "
                    "frequencies with a common observed reference gain."
                ),
            ]
        )

    plot_files = set(summary.get("plot_files") or [])
    if plot_files:
        lines.extend(
            [
                "",
                "## Model fit plots",
                "",
                (
                    "In the per-frequency diagnostics, solid lines are additive-model "
                    "predictions and circular markers are passing three-epoch cell "
                    "means. Error bars show repeat circular standard deviation. "
                    "Failed or unsupported cells are not drawn. Phase is placed on "
                    "the branch nearest the fitted frequency intercept so wrap-around "
                    "does not create false jumps."
                ),
                "",
            ]
        )
        gain_effect_plots = sorted(
            filename
            for filename in plot_files
            if filename == "fitted_gain_effects.png"
            or filename.startswith("fitted_gain_effects_")
        )
        if gain_effect_plots:
            lines.extend(
                [
                    "### Fitted gain effects across frequency",
                    "",
                ]
            )
            for page_index, filename in enumerate(gain_effect_plots, start=1):
                suffix = (
                    ""
                    if len(gain_effect_plots) == 1
                    else f", overview page {page_index}"
                )
                lines.extend(
                    [
                        (f"![Fitted RX1 and RX2 gain effects{suffix}]" f"({filename})"),
                        "",
                    ]
                )
        if "frequency_intercept_delay.png" in plot_files:
            lines.extend(
                [
                    "### Frequency baseline and delay description",
                    "",
                    (
                        "![Per-frequency baseline and constant-plus-delay "
                        "description](frequency_intercept_delay.png)"
                    ),
                    "",
                ]
            )
        lines.extend(["### Per-frequency data versus fit", ""])
        for row in summary["frequency_summary"]:
            frequency_hz = int(row["frequency_hz"])
            fit_plot = f"model_fit_{frequency_hz}.png"
            if fit_plot not in plot_files:
                continue
            surface_plot = f"phase_surface_{frequency_hz}.png"
            residual_plot = f"additive_residual_{frequency_hz}.png"
            links = []
            if surface_plot in plot_files:
                links.append(f"[coverage and phase surface]({surface_plot})")
            if residual_plot in plot_files:
                links.append(f"[residual heatmap]({residual_plot})")
            lines.extend(
                [
                    f"#### {frequency_hz / 1e6:.3f} MHz",
                    "",
                    (
                        f"![Gain sweeps and observed-versus-fitted phase at "
                        f"{frequency_hz / 1e6:.3f} MHz]({fit_plot})"
                    ),
                    "",
                ]
            )
            if links:
                lines.extend(["Additional views: " + " · ".join(links), ""])
    lines.append("")
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
        _close_figure(plt, fig)
        filenames.append(filename)
    return filenames


def _plot_model_diagnostics(
    model: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Plot fitted gain curves and the remaining ordered-pair residuals."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = np.asarray(model.get("gain_values_db", []), dtype=np.float64)
    fitted = [
        row for row in model.get("frequency_models", []) if row.get("status") == "fit"
    ]
    if gains.size == 0 or not fitted:
        return []

    filenames = []
    frequencies_per_overview = 12
    overview_pages = [
        fitted[index : index + frequencies_per_overview]
        for index in range(0, len(fitted), frequencies_per_overview)
    ]
    for page_index, page_rows in enumerate(overview_pages, start=1):
        fig, axes = plt.subplots(
            len(page_rows),
            1,
            figsize=(11, max(3.5, 3.2 * len(page_rows))),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for axis, row in zip(axes[:, 0], page_rows):
            rx1 = np.asarray(
                [np.nan if value is None else value for value in row["rx1_effect_rad"]],
                dtype=np.float64,
            )
            rx2 = np.asarray(
                [np.nan if value is None else value for value in row["rx2_effect_rad"]],
                dtype=np.float64,
            )
            axis.plot(gains, np.degrees(rx1), label="RX1 effect", linewidth=1.4)
            axis.plot(gains, np.degrees(rx2), label="RX2 effect", linewidth=1.4)
            axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel("Phase effect (°)")
            axis.set_title(f"{int(row['frequency_hz']) / 1e6:.3f} MHz")
            axis.legend(loc="best")
        axes[-1, 0].set_xlabel("Manual gain (dB)")
        page_suffix = (
            ""
            if len(overview_pages) == 1
            else f" — page {page_index}/{len(overview_pages)}"
        )
        fig.suptitle(f"{model['serial']} — fitted ordered gain effects{page_suffix}")
        filename = (
            "fitted_gain_effects.png"
            if len(overview_pages) == 1
            else f"fitted_gain_effects_{page_index:02d}.png"
        )
        fig.savefig(output_dir / filename, dpi=160)
        _close_figure(plt, fig)
        filenames.append(filename)

    delay_model = model.get("frequency_intercept_delay_model")
    if delay_model and len(delay_model.get("frequency_points", [])) >= 2:
        points = delay_model["frequency_points"]
        frequency_mhz = np.asarray(
            [point["frequency_hz"] / 1e6 for point in points],
            dtype=np.float64,
        )
        observed_deg = np.degrees([point["phase_rad_unwrapped"] for point in points])
        fitted_deg = np.degrees(
            [point["fitted_phase_rad_unwrapped"] for point in points]
        )
        fig, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        axis.plot(
            frequency_mhz,
            observed_deg,
            "o",
            label="Fitted per-frequency baseline",
        )
        axis.plot(
            frequency_mhz,
            fitted_deg,
            "-",
            label="Constant-plus-delay fit",
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("RF frequency (MHz)")
        axis.set_ylabel("Unwrapped RX1−RX2 baseline phase (°)")
        axis.set_title(
            f"{model['serial']} — effective differential delay "
            f"{delay_model['descriptive_delay_seconds'] * 1e9:.3f} ns"
        )
        axis.legend(loc="best")
        filename = "frequency_intercept_delay.png"
        fig.savefig(output_dir / filename, dpi=160)
        _close_figure(plt, fig)
        filenames.append(filename)

    for row in fitted:
        interaction = np.degrees(
            np.asarray(row["interaction_residual_rad"], dtype=np.float64)
        )
        finite = np.abs(interaction[np.isfinite(interaction)])
        if finite.size == 0:
            continue
        limit = max(1.0, float(np.percentile(finite, 95)))
        fig, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
        extent = (
            gains[0] - 0.5,
            gains[-1] + 0.5,
            gains[0] - 0.5,
            gains[-1] + 0.5,
        )
        image = axis.imshow(
            interaction.T,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xlabel("RX1 manual gain (dB)")
        axis.set_ylabel("RX2 manual gain (dB)")
        axis.set_title(
            f"…{str(model['serial'])[-12:]} — additive residual, "
            f"{int(row['frequency_hz']) / 1e6:.3f} MHz"
        )
        fig.colorbar(image, ax=axis, label="Circular residual (°)")
        filename = f"additive_residual_{int(row['frequency_hz'])}.png"
        fig.savefig(output_dir / filename, dpi=160)
        _close_figure(plt, fig)
        filenames.append(filename)
    return filenames


def _representative_gains(
    gains: np.ndarray, reference_gain_db: float | None
) -> list[float]:
    """Choose up to three fixed-gain traces: low, reference, and high."""

    if gains.size <= 3:
        return [float(value) for value in gains]
    reference = (
        float(reference_gain_db)
        if reference_gain_db is not None
        else float(gains[len(gains) // 2])
    )
    middle = float(gains[int(np.argmin(np.abs(gains - reference)))])
    selected = [float(gains[0]), middle, float(gains[-1])]
    if len(set(selected)) == 3:
        return selected

    for candidate in gains:
        value = float(candidate)
        if value not in selected:
            selected.insert(-1, value)
        if len(set(selected)) >= 3:
            break
    return list(dict.fromkeys(selected))[:3]


def _phase_near_reference(value: float, reference: float) -> float:
    """Return ``value`` on the circular branch nearest ``reference``."""

    return reference + float(np.angle(np.exp(1j * (value - reference))))


def _plot_observed_vs_model(
    validation: dict[str, Any],
    model: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Plot passing cell means against each fitted per-frequency model."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = np.asarray(model.get("gain_values_db", []), dtype=np.float64)
    if gains.size == 0:
        return []
    gain_index = {float(gain): index for index, gain in enumerate(gains)}
    cells_by_frequency: dict[
        int, dict[tuple[float, float], dict[str, Any]]
    ] = defaultdict(dict)
    for cell in validation.get("cells", []):
        frequency_hz = int(cell["frequency_hz"])
        pair = (float(cell["gain_rx1_db"]), float(cell["gain_rx2_db"]))
        cells_by_frequency[frequency_hz][pair] = cell

    filenames = []
    colors = plt.get_cmap("tab10").colors
    for row in model.get("frequency_models", []):
        if row.get("status") != "fit":
            continue
        frequency_hz = int(row["frequency_hz"])
        cells = cells_by_frequency.get(frequency_hz, {})
        intercept = float(row["intercept_rad"])
        rx1_effect = np.asarray(
            [np.nan if value is None else value for value in row["rx1_effect_rad"]],
            dtype=np.float64,
        )
        rx2_effect = np.asarray(
            [np.nan if value is None else value for value in row["rx2_effect_rad"]],
            dtype=np.float64,
        )
        supported = np.asarray(row["supported_gain_pair"], dtype=bool)
        fixed_gains = _representative_gains(gains, row.get("reference_gain_db"))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        rx2_axis, rx1_axis, identity_axis, residual_axis = axes.ravel()

        for color_index, fixed_rx1 in enumerate(fixed_gains):
            i = gain_index[fixed_rx1]
            fitted_x = []
            fitted_y = []
            observed_x = []
            observed_y = []
            observed_error = []
            for j, variable_rx2 in enumerate(gains):
                cell = cells.get((fixed_rx1, float(variable_rx2)))
                if (
                    not supported[i, j]
                    or not np.isfinite(rx1_effect[i])
                    or not np.isfinite(rx2_effect[j])
                    or not cell
                    or not cell.get("pass")
                    or cell.get("phase_mean_rad") is None
                ):
                    fitted_x.append(float(variable_rx2))
                    fitted_y.append(np.nan)
                    continue
                prediction = _phase_near_reference(
                    intercept + rx1_effect[i] + rx2_effect[j], intercept
                )
                observed = _phase_near_reference(
                    float(cell["phase_mean_rad"]), prediction
                )
                fitted_x.append(float(variable_rx2))
                fitted_y.append(math.degrees(prediction))
                observed_x.append(float(variable_rx2))
                observed_y.append(math.degrees(observed))
                observed_error.append(float(cell["phase_circular_std_deg"]))
            color = colors[color_index]
            rx2_axis.plot(
                fitted_x,
                fitted_y,
                color=color,
                linewidth=1.5,
                label=f"RX1 = {fixed_rx1:g} dB",
            )
            if observed_x:
                rx2_axis.errorbar(
                    observed_x,
                    observed_y,
                    yerr=observed_error,
                    fmt="o",
                    color=color,
                    markersize=4,
                    capsize=2,
                    linestyle="none",
                )

        for color_index, fixed_rx2 in enumerate(fixed_gains):
            j = gain_index[fixed_rx2]
            fitted_x = []
            fitted_y = []
            observed_x = []
            observed_y = []
            observed_error = []
            for i, variable_rx1 in enumerate(gains):
                cell = cells.get((float(variable_rx1), fixed_rx2))
                if (
                    not supported[i, j]
                    or not np.isfinite(rx1_effect[i])
                    or not np.isfinite(rx2_effect[j])
                    or not cell
                    or not cell.get("pass")
                    or cell.get("phase_mean_rad") is None
                ):
                    fitted_x.append(float(variable_rx1))
                    fitted_y.append(np.nan)
                    continue
                prediction = _phase_near_reference(
                    intercept + rx1_effect[i] + rx2_effect[j], intercept
                )
                observed = _phase_near_reference(
                    float(cell["phase_mean_rad"]), prediction
                )
                fitted_x.append(float(variable_rx1))
                fitted_y.append(math.degrees(prediction))
                observed_x.append(float(variable_rx1))
                observed_y.append(math.degrees(observed))
                observed_error.append(float(cell["phase_circular_std_deg"]))
            color = colors[color_index]
            rx1_axis.plot(
                fitted_x,
                fitted_y,
                color=color,
                linewidth=1.5,
                label=f"RX2 = {fixed_rx2:g} dB",
            )
            if observed_x:
                rx1_axis.errorbar(
                    observed_x,
                    observed_y,
                    yerr=observed_error,
                    fmt="o",
                    color=color,
                    markersize=4,
                    capsize=2,
                    linestyle="none",
                )

        observed_phase = []
        predicted_phase = []
        residual_deg = []
        mismatch_db = []
        residual_rx1_gain = []
        for (gain_rx1, gain_rx2), cell in sorted(cells.items()):
            i = gain_index.get(gain_rx1)
            j = gain_index.get(gain_rx2)
            if (
                i is None
                or j is None
                or not supported[i, j]
                or not cell.get("pass")
                or cell.get("phase_mean_rad") is None
                or not np.isfinite(rx1_effect[i])
                or not np.isfinite(rx2_effect[j])
            ):
                continue
            prediction = _phase_near_reference(
                intercept + rx1_effect[i] + rx2_effect[j], intercept
            )
            observed = _phase_near_reference(float(cell["phase_mean_rad"]), prediction)
            predicted_phase.append(math.degrees(prediction))
            observed_phase.append(math.degrees(observed))
            residual_deg.append(math.degrees(observed - prediction))
            mismatch_db.append(gain_rx2 - gain_rx1)
            residual_rx1_gain.append(gain_rx1)

        if predicted_phase:
            limits = [
                min(predicted_phase + observed_phase),
                max(predicted_phase + observed_phase),
            ]
            padding = max(1.0, 0.05 * (limits[1] - limits[0]))
            identity_axis.plot(
                [limits[0] - padding, limits[1] + padding],
                [limits[0] - padding, limits[1] + padding],
                color="black",
                linewidth=0.8,
                linestyle="--",
            )
            identity_axis.scatter(
                predicted_phase,
                observed_phase,
                c=residual_rx1_gain,
                cmap="viridis",
                s=35,
                edgecolor="white",
                linewidth=0.4,
            )
            absolute_residual = np.abs(np.asarray(residual_deg))
            identity_axis.text(
                0.03,
                0.97,
                (
                    f"cell-mean MAE {np.mean(absolute_residual):.2f}°\n"
                    f"p95 {np.percentile(absolute_residual, 95):.2f}°"
                ),
                transform=identity_axis.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            residual_scatter = residual_axis.scatter(
                mismatch_db,
                residual_deg,
                c=residual_rx1_gain,
                cmap="viridis",
                s=35,
                edgecolor="white",
                linewidth=0.4,
            )
            fig.colorbar(
                residual_scatter,
                ax=residual_axis,
                label="RX1 manual gain (dB)",
            )

        for axis in (rx2_axis, rx1_axis, identity_axis, residual_axis):
            axis.grid(True, alpha=0.25)
        rx2_axis.set_title(
            f"RX2 sweep at {len(fixed_gains)} representative fixed RX1 gains"
        )
        rx2_axis.set_xlabel("RX2 manual gain (dB)")
        rx2_axis.set_ylabel("RX1−RX2 phase near fitted branch (°)")
        rx2_axis.legend(loc="best")
        rx1_axis.set_title(
            f"RX1 sweep at {len(fixed_gains)} representative fixed RX2 gains"
        )
        rx1_axis.set_xlabel("RX1 manual gain (dB)")
        rx1_axis.set_ylabel("RX1−RX2 phase near fitted branch (°)")
        rx1_axis.legend(loc="best")
        identity_axis.set_title("Passing cell means versus final additive fit")
        identity_axis.set_xlabel("Fitted phase (°)")
        identity_axis.set_ylabel("Observed cell-mean phase (°)")
        residual_axis.axhline(0, color="black", linewidth=0.8)
        residual_axis.set_title("Final-fit cell-mean residual by gain mismatch")
        residual_axis.set_xlabel("RX2 gain − RX1 gain (dB)")
        residual_axis.set_ylabel("Observed − fitted phase (°)")
        fig.suptitle(
            f"{model['serial']} — {frequency_hz / 1e6:.3f} MHz\n"
            "lines = additive fit; circles = passing three-epoch cell means"
        )
        filename = f"model_fit_{frequency_hz}.png"
        fig.savefig(output_dir / filename, dpi=160)
        _close_figure(plt, fig)
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
    summary["plot_files"] = (
        _plot_heatmaps(validation, output_dir)
        + _plot_model_diagnostics(model, output_dir)
        + _plot_observed_vs_model(validation, model, output_dir)
        if plots
        else []
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(render_markdown(summary))
    return summary
