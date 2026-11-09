#!/usr/bin/env python3
"""Generate static Matplotlib walkthroughs for empirical phase calibration."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def find_repo_root():
    for candidate in [Path.cwd(), Path(__file__).resolve().parent]:
        for parent in [candidate, *candidate.parents]:
            if (parent / "spf").is_dir() and (parent / "empirical_dists").is_dir():
                return parent
    raise RuntimeError("run from an SPF repository checkout")


ROOT = find_repo_root()
SCRIPT = Path(__file__).resolve()
REPORT_DATA = (
    SCRIPT.parent.parent
    if SCRIPT.parent.name == "analysis" and SCRIPT.parent.parent.name == "2026_08_25_empirical_dist_analysis"
    else ROOT / "reports/data/2026_08_25_empirical_dist_analysis"
)
FIGURE_DIR = ROOT / "reports/figures/2026_08_25_empirical_dist_analysis"
ABLATION = REPORT_DATA / "calibration_heldout_ablation.csv"
PROVENANCE = ROOT / "spf/calibrations/empirical_p_dist/reports/empirical_rebuild_20260809_v1/provenance.json"

ROVER = "#2878B5"
WALL = "#E07A2D"
TEXT = "#20252B"
MUTED = "#59616A"
GRID = "#D5D9DE"
IDEAL = "#555B63"
GOOD = "#2A9D6F"
BAD = "#B43E3E"
PARAM_COLORS = {"c": "#8E5BA6", "g": "#2878B5", "delta": "#2A9D6F", "tau": "#737B84"}
PLATFORM_COLORS = {"rover": ROVER, "wall": WALL}
REGIMES = ["<0.25", "0.25–0.50", "0.50–1.00", ">1.00"]


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def add_bar_labels(ax, bars, fmt="{:.1f}", suffix="", pad_fraction=0.025):
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * pad_fraction
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        va = "bottom" if value >= 0 else "top"
        offset = pad if value >= 0 else -pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value) + suffix,
            ha="center",
            va=va,
            fontsize=8.5,
            color=TEXT,
        )


def load_inputs():
    sys.path.insert(0, str(REPORT_DATA / "analysis"))
    import calibration_parameter_distributions as cpd

    keys, provenance = cpd.load_key_fits(REPORT_DATA / "metrics_all_keys.csv", PROVENANCE)
    _, receiver_fits, diagnostics = cpd.load_dataset_fits(
        REPORT_DATA / "calibration_quality_scan_inputs.csv", provenance, keys
    )
    config = pd.read_csv(REPORT_DATA / "calibration_configuration_summary.csv")
    systematics = pd.read_csv(REPORT_DATA / "calibration_systematics_summary.csv")
    heldout = pd.read_csv(ABLATION)
    heldout["rho_regime"] = pd.cut(
        heldout["rho"],
        [-np.inf, 0.25, 0.50, 1.00, np.inf],
        labels=REGIMES,
        right=False,
    )
    return keys, receiver_fits, config, systematics, heldout, diagnostics


PARAMS = {
    "c": {
        "title": "Phase offset c: a receiver-path bias that should be removed",
        "subtitle": "c moves the phase-vs-bearing curve vertically; it does not change the curve's shape",
        "sample": "phase_offset_deg",
        "config_r0": "r0_phase_deg_median_or_circular_mean",
        "config_r1": "r1_phase_deg_median_or_circular_mean",
        "systematics": "phase",
        "symbol": "c",
        "unit": "degrees",
        "ideal": 0.0,
        "hist_range": (-180.0, 180.0),
        "hist_bins": np.arange(-180, 181, 15),
        "scatter_ylim": (-180, 180),
        "mad": "4.70°",
        "note": "Exact configuration explains 89% of variation; fit c separately for r0 and r1.",
    },
    "g": {
        "title": "Geometry gain g: an effective phase-slope correction",
        "subtitle": "g rescales d/λ in the forward model; g=1 is the nominal antenna geometry",
        "sample": "g",
        "config_r0": "r0_g_median_or_circular_mean",
        "config_r1": "r1_g_median_or_circular_mean",
        "systematics": "g",
        "symbol": "g",
        "unit": "× nominal",
        "ideal": 1.0,
        "hist_range": (0.65, 3.05),
        "hist_bins": np.arange(0.65, 3.051, 0.10),
        "scatter_ylim": (0.65, 3.05),
        "mad": "0.020",
        "note": "Repeatable by configuration, but low-d/λ fits are weakly identifiable; shrink g toward 1.",
    },
    "delta": {
        "title": "Bearing / mount shift δ: a small angular-origin correction",
        "subtitle": "δ shifts the bearing entering sin(θ−δ); rover values can also absorb heading-label bias",
        "sample": "theta_shift_deg",
        "config_r0": "r0_theta_deg_median_or_circular_mean",
        "config_r1": "r1_theta_deg_median_or_circular_mean",
        "systematics": "theta",
        "symbol": "δ",
        "unit": "degrees",
        "ideal": 0.0,
        "hist_range": (-55.0, 55.0),
        "hist_bins": np.arange(-55, 56, 5),
        "scatter_ylim": (-55, 55),
        "mad": "1.15°",
        "note": "Only 43% is configuration-repeatable; use a tight zero-centered prior and require holdout gain.",
    },
}


def parameter_histogram(ax, receiver_fits, meta):
    for platform, color, label in (("rover", ROVER, "Rover"), ("wall", WALL, "Wall")):
        values = receiver_fits.loc[receiver_fits.platform == platform, meta["sample"]].to_numpy(float)
        values = values[np.isfinite(values)]
        weights = np.full(len(values), 100.0 / len(values))
        ax.hist(
            values,
            bins=meta["hist_bins"],
            weights=weights,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=f"{label} (n={len(values):,} receiver fits)",
        )
        abs_q = np.quantile(np.abs(values - meta["ideal"]), [0.5, 0.9])
        digits = 2 if meta["symbol"] == "g" else 1
        quantity = "σφ" if meta["symbol"] == "σφ" else "|Δ|"
        ax.text(
            0.03,
            0.93 if platform == "rover" else 0.83,
            f"{label}: median {quantity} {abs_q[0]:.{digits}f}, 90th {abs_q[1]:.{digits}f}",
            transform=ax.transAxes,
            fontsize=8.8,
            color=color,
            va="top",
        )
    ax.axvline(meta["ideal"], color=IDEAL, linestyle="--", linewidth=1.2)
    ax.set_xlim(*meta["hist_range"])
    ax.set_xlabel(f"Fitted {meta['symbol']} ({meta['unit']})")
    ax.set_ylabel("Within-platform files (%)")
    ax.set_title("1. How large are the fitted values?", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    style_axis(ax)


def configuration_scatter(ax, config, meta):
    for _, row in config.iterrows():
        x = row["d_lambda"]
        y0, y1 = row[meta["config_r0"]], row[meta["config_r1"]]
        color = PLATFORM_COLORS[row["platform"]]
        ax.plot([x, x], [y0, y1], color=color, alpha=0.22, linewidth=0.8)
        ax.scatter(x, y0, s=25, color=color, marker="o", alpha=0.78, edgecolor="white", linewidth=0.35)
        ax.scatter(x, y1, s=29, color=color, marker="x", alpha=0.85, linewidth=1.0)
    ax.axhline(meta["ideal"], color=IDEAL, linestyle="--", linewidth=1.2)
    ax.axvline(0.5, color="#A5ABB2", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.08, 1.60)
    ax.set_ylim(*meta["scatter_ylim"])
    ax.set_xlabel("Electrical spacing d/λ (combines frequency and physical spacing)")
    ax.set_ylabel(f"Exact-config center ({meta['unit']})")
    ax.set_title("2. Where is it systematic?", fontsize=12, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROVER, markeredgecolor=ROVER, label="Rover"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WALL, markeredgecolor=WALL, label="Wall"),
        Line2D([0], [0], marker="o", color=IDEAL, linestyle="none", label="r0"),
        Line2D([0], [0], marker="x", color=IDEAL, linestyle="none", label="r1"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.3, loc="best", ncol=2)
    style_axis(ax)


def heldout_regime_bars(ax, heldout, param):
    x = np.arange(len(REGIMES), dtype=float)
    width = 0.36
    all_values = []
    bars_by_platform = []
    for offset, platform, color, label in ((-width / 2, "rover", ROVER, "Rover"), (width / 2, "wall", WALL, "Wall")):
        medians, low, high, counts = [], [], [], []
        for regime in REGIMES:
            values = heldout.loc[
                (heldout.platform == platform) & (heldout.rho_regime.astype(str) == regime), param
            ].to_numpy(float)
            counts.append(len(values))
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                medians.append(median)
                low.append(median - q25)
                high.append(q75 - median)
                all_values.extend([q25, q75])
            else:
                medians.append(np.nan)
                low.append(0.0)
                high.append(0.0)
        bars = ax.bar(x + offset, np.nan_to_num(medians), width, color=color, alpha=0.92, label=label)
        valid = np.isfinite(medians)
        ax.errorbar(
            x[valid] + offset,
            np.asarray(medians)[valid],
            yerr=np.vstack([np.asarray(low)[valid], np.asarray(high)[valid]]),
            fmt="none",
            ecolor=TEXT,
            elinewidth=0.9,
            capsize=2.2,
        )
        for bar, value, count in zip(bars, medians, counts):
            if count:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.1f}\nn={count}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7.5,
                    color=TEXT,
                )
            else:
                bar.set_alpha(0.0)
        bars_by_platform.append(bars)
    ax.axhline(0, color=IDEAL, linewidth=1.0)
    ax.set_xticks(x, ["<0.25", "0.25–0.50", "0.50–1.00", ">1.00"])
    ax.set_ylabel("Held-out phase-error reduction (degrees)")
    ax.set_title("3. How much does it help?", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.3, loc="best")
    style_axis(ax)
    return all_values


def repeatability_bars(ax, systematics, receiver_fits, meta):
    labels = ["Receiver", "Device +\nband", "Band + spacing", "Exact config"]
    grouping = [
        "receiver",
        "device + band + receiver",
        "device + band + physical spacing + receiver",
        "exact configuration + receiver",
    ]
    frame = systematics[systematics.parameter == meta["systematics"]].set_index("grouping")
    values = [float(frame.loc[name, "r2_in_sample"]) for name in grouping]
    bars = ax.bar(np.arange(len(values)), values, color=["#AAB2BB", "#7EA7C7", "#4B8CBA", "#236A9A"])
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("In-sample variance explained, R²")
    ax.set_title("4. Can it be reused by configuration?", fontsize=12, fontweight="bold")
    style_axis(ax)
    add_bar_labels(ax, bars, fmt="{:.2f}", pad_fraction=0.012)

    if meta["systematics"] == "g":
        rover_values = receiver_fits.loc[receiver_fits.platform == "rover", "g"].to_numpy(float)
        wall_values = receiver_fits.loc[receiver_fits.platform == "wall", "g"].to_numpy(float)
        bound_rover = 100 * np.mean(np.isclose(rover_values, 0.9) | np.isclose(rover_values, 1.1))
        bound_wall = 100 * np.mean(np.isclose(wall_values, 0.7) | np.isclose(wall_values, 3.0))
        detail = f"Exact-config within MAD: {meta['mad']}\nGrid-bound g: rover {bound_rover:.1f}%, wall {bound_wall:.1f}%"
    elif meta["systematics"] == "theta":
        rover_values = receiver_fits.loc[receiver_fits.platform == "rover", "theta_shift_rad"].to_numpy(float)
        wall_values = receiver_fits.loc[receiver_fits.platform == "wall", "theta_shift_rad"].to_numpy(float)
        bound_rover = 100 * np.mean(np.isclose(np.abs(rover_values), 0.9))
        bound_wall = 100 * np.mean(np.isclose(np.abs(wall_values), 0.35))
        detail = f"Exact-config within MAD: {meta['mad']}\nGrid-bound δ: rover {bound_rover:.1f}%, wall {bound_wall:.1f}%"
    elif meta["systematics"] == "phase":
        detail = f"Exact-config within MAD: {meta['mad']}\nNo hard search bound on analytic c"
    else:
        detail = f"Exact-config within MAD: {meta['mad']}\nσφ is derived from corrected residuals"
    ax.text(0.03, 0.94, detail, transform=ax.transAxes, va="top", fontsize=8.8, color=MUTED)


def make_parameter_figure(name, receiver_fits, config, systematics, heldout):
    meta = PARAMS[name]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.115, top=0.84, hspace=0.43, wspace=0.25)
    fig.patch.set_facecolor("white")
    fig.suptitle(meta["title"], fontsize=17, fontweight="bold", color=TEXT, y=0.965)
    fig.text(0.5, 0.915, meta["subtitle"], ha="center", fontsize=10.5, color=MUTED)

    parameter_histogram(axes[0, 0], receiver_fits, meta)
    configuration_scatter(axes[0, 1], config, meta)
    heldout_regime_bars(axes[1, 0], heldout, name)
    repeatability_bars(axes[1, 1], systematics, receiver_fits, meta)

    helps = {
        platform: 100 * np.mean(heldout.loc[heldout.platform == platform, name] > 0)
        for platform in ("rover", "wall")
    }
    fig.text(
        0.5,
        0.045,
        f"Held-out contribution is a Shapley attribution over all c/g/δ fit orders: {helps['rover']:.1f}% of rover and {helps['wall']:.1f}% of wall files benefit.  {meta['note']}",
        ha="center",
        fontsize=9.1,
        color=MUTED,
    )
    path = FIGURE_DIR / {
        "c": "phase_offset_calibration_walkthrough.png",
        "g": "geometry_gain_calibration_walkthrough.png",
        "delta": "bearing_shift_calibration_walkthrough.png",
    }[name]
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_sigma_figure(keys, receiver_fits, config, systematics):
    meta = {
        "sample": "phase_sigma_deg",
        "config_r0": "r0_sigma_deg_median_or_circular_mean",
        "config_r1": "r1_sigma_deg_median_or_circular_mean",
        "systematics": "sigma",
        "symbol": "σφ",
        "unit": "degrees",
        "ideal": 0.0,
        "hist_range": (0.0, 105.0),
        "hist_bins": np.arange(0, 106, 5),
        "scatter_ylim": (0, 105),
        "mad": "2.53°",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.115, top=0.84, hspace=0.43, wspace=0.25)
    fig.patch.set_facecolor("white")
    fig.suptitle("Residual spread σφ: likelihood width, not a phase correction", fontsize=17, fontweight="bold", color=TEXT, y=0.965)
    fig.text(0.5, 0.915, "σφ says how uncertain the calibrated phase model is; smaller means a sharper likelihood", ha="center", fontsize=10.5, color=MUTED)
    parameter_histogram(axes[0, 0], receiver_fits, meta)
    configuration_scatter(axes[0, 1], config, meta)

    ax = axes[1, 0]
    order = ["Pluto, 0.9 GHz", "Pluto, 2.4 GHz", "Pluto, 5.8 GHz", "bladeRF, 2.4 GHz", "bladeRF, 5.8 GHz"]
    labels, medians, low, high, colors = [], [], [], [], []
    for label in order:
        values = keys.loc[keys.device_band == label, "cal_phase_sigma_deg"].to_numpy(float)
        if not len(values):
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        labels.append(label.replace(", ", "\n"))
        medians.append(median)
        low.append(median - q25)
        high.append(q75 - median)
        colors.append("#2878B5" if "2.4" in label else "#E07A2D" if "5.8" in label else "#2A9D6F")
    bars = ax.bar(np.arange(len(labels)), medians, color=colors)
    ax.errorbar(np.arange(len(labels)), medians, yerr=np.vstack([low, high]), fmt="none", ecolor=TEXT, capsize=3, linewidth=1)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Key-level σφ (degrees); bar=median, whisker=IQR")
    ax.set_title("3. How wide should the likelihood be?", fontsize=12, fontweight="bold")
    style_axis(ax)
    add_bar_labels(ax, bars, fmt="{:.1f}", suffix="°")

    repeatability_bars(axes[1, 1], systematics, receiver_fits, meta)
    fig.text(
        0.5,
        0.045,
        "σφ does not move the predicted phase, so it cannot reduce phase MAE by itself.  It prevents a noisy configuration from being treated as overconfident; exact-config R²=0.828.",
        ha="center",
        fontsize=9.1,
        color=MUTED,
    )
    path = FIGURE_DIR / "residual_spread_calibration_walkthrough.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_benefit_summary(heldout):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.80, wspace=0.28)
    fig.patch.set_facecolor("white")
    fig.suptitle("How much does each calibration parameter help on held-out time blocks?", fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.895, "81 timing-clean rover files and 393 timing-clean Pluto wall files; each file has equal weight", ha="center", fontsize=10.5, color=MUTED)

    width = 0.34
    # Panel 1: median error path.
    ax = axes[0]
    stages = ["Nominal", "+ c, g, δ", "+ time τ"]
    x = np.arange(3)
    for offset, platform, color, label in ((-width / 2, "rover", ROVER, "Rover"), (width / 2, "wall", WALL, "Wall")):
        d = heldout[heldout.platform == platform]
        values = [d.baseline_mae_deg.median(), d.static_mae_deg.median(), d.tau_mae_deg.median()]
        bars = ax.bar(x + offset, values, width, color=color, label=f"{label} (n={len(d)})")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+1.0, f"{value:.1f}°", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, stages)
    ax.set_ylabel("Median held-out absolute phase error")
    ax.set_title("Total prediction error", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    style_axis(ax)

    # Panel 2: mean additive attribution so c+g+delta equals mean static gain.
    ax = axes[1]
    params = ["c", "g", "delta", "tau"]
    labels = ["Phase\noffset c", "Geometry\ngain g", "Bearing\nshift δ", "Time\noffset τ"]
    x = np.arange(4)
    for offset, platform, color in ((-width / 2, "rover", ROVER), (width / 2, "wall", WALL)):
        d = heldout[heldout.platform == platform]
        values = [d[p].mean() for p in params]
        bars = ax.bar(x + offset, values, width, color=color)
        add_bar_labels(ax, bars, fmt="{:.2f}", suffix="°", pad_fraction=0.018)
    ax.axhline(0, color=IDEAL, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean held-out error reduction")
    ax.set_title("Average contribution", fontsize=12.5, fontweight="bold")
    style_axis(ax)

    # Panel 3: fraction with positive heldout contribution.
    ax = axes[2]
    for offset, platform, color in ((-width / 2, "rover", ROVER), (width / 2, "wall", WALL)):
        d = heldout[heldout.platform == platform]
        values = [100 * np.mean(d[p] > 0) for p in params]
        bars = ax.bar(x + offset, values, width, color=color)
        add_bar_labels(ax, bars, fmt="{:.1f}", suffix="%", pad_fraction=0.014)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Files with lower held-out error")
    ax.set_title("How often it helps", fontsize=12.5, fontweight="bold")
    style_axis(ax)

    fig.text(
        0.5,
        0.055,
        "c/g/δ contributions are Shapley values over all fit orders (they sum to the static-calibration gain). τ is fitted after c/g/δ. Training blocks select parameters; alternating blocks score them.",
        ha="center",
        fontsize=9.0,
        color=MUTED,
    )
    path = FIGURE_DIR / "calibration_parameter_heldout_help.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    keys, receiver_fits, config, systematics, heldout, diagnostics = load_inputs()
    paths = [
        make_benefit_summary(heldout),
        make_parameter_figure("c", receiver_fits, config, systematics, heldout),
        make_parameter_figure("g", receiver_fits, config, systematics, heldout),
        make_parameter_figure("delta", receiver_fits, config, systematics, heldout),
        make_sigma_figure(keys, receiver_fits, config, systematics),
    ]
    print(json.dumps({"figures": [str(path) for path in paths], "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
