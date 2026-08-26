#!/usr/bin/env python3
"""Summarize fitted calibration parameters and their configuration repeatability.

This analysis is read-only with respect to empirical PKLs, provenance, and quality-scan
inputs.  It writes only CSV and PNG products beneath the selected report directories.

Two deliberately different estimates are shown:

* 48 key-level parameters fitted to pooled ``r/nosym`` posterior matrices by
  ``compare_theory.py``; and
* per-dataset/per-receiver diagnostic fits from the 2026-07-12 quality scan.

The latter used bounded search grids (wall ``g`` 0.70--3.00, rover ``g`` 0.90--1.10),
so its histograms are diagnostics of repeatability, not continuous parameter posteriors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


C_M_S = 299_792_458.0
CONFIG_MIN_DATASETS = 5
BAND_ORDER = ["0.9 GHz", "2.4 GHz", "5.8 GHz"]
BAND_COLORS = {
    "0.9 GHz": "#009E73",
    "2.4 GHz": "#0072B2",
    "5.8 GHz": "#D55E00",
}
DEVICE_MARKERS = {"PLUTO": "o", "BLADERF2": "s"}
DEVICE_NAMES = {"PLUTO": "Pluto", "BLADERF2": "bladeRF"}
PARAMETERS = {
    "g": {
        "key": "cal_effective_spacing_gain",
        "sample": "g",
        "label": r"effective phase-slope gain $g$",
        "short": r"$g$",
        "ideal": 1.0,
        "key_range": (0.5, 2.85),
        "sample_range": (0.65, 3.05),
    },
    "theta": {
        "key": "cal_theta_shift_deg",
        "sample": "theta_shift_deg",
        "label": r"bearing / mount shift $\delta$ (deg)",
        "short": r"$\delta$ (deg)",
        "ideal": 0.0,
        "key_range": (-15.0, 15.0),
        "sample_range": (-55.0, 55.0),
    },
    "phase": {
        "key": "cal_phase_offset_deg",
        "sample": "phase_offset_deg",
        "label": r"constant phase offset $c$ (deg)",
        "short": r"$c$ (deg)",
        "ideal": 0.0,
        "key_range": (-30.0, 90.0),
        "sample_range": (-180.0, 180.0),
    },
    "sigma": {
        "key": "cal_phase_sigma_deg",
        "sample": "phase_sigma_deg",
        "label": r"corrected circular phase spread $\sigma_\phi$ (deg)",
        "short": r"$\sigma_\phi$ (deg)",
        "ideal": 0.0,
        "key_range": (15.0, 90.0),
        "sample_range": (0.0, 105.0),
    },
}


def find_repo_root(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / "spf").is_dir() and (parent / "data_quality_reports").is_dir():
            return parent
    raise RuntimeError(f"could not locate repository root above {path}")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo = find_repo_root(script.parent)
    report = script.parent.parent
    if report.parent.name == "data":
        figure_dir = report.parents[1] / "figures" / report.name
    else:
        figure_dir = report / "figures"
    quality_scan = repo / "data_quality_reports/scan_2026_07_12_v2/metrics.csv"
    packaged_scan = report / "calibration_quality_scan_inputs.csv"
    if not quality_scan.exists() and packaged_scan.exists():
        quality_scan = packaged_scan

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=report / "metrics_all_keys.csv")
    p.add_argument(
        "--provenance",
        type=Path,
        default=repo
        / "spf/calibrations/empirical_p_dist/reports/empirical_rebuild_20260809_v1/provenance.json",
    )
    p.add_argument(
        "--quality-scan",
        type=Path,
        default=quality_scan,
    )
    p.add_argument("--output-dir", type=Path, default=report)
    p.add_argument("--figure-dir", type=Path, default=figure_dir)
    return p.parse_args()


def wrap_rad(x: np.ndarray | pd.Series | float) -> np.ndarray:
    return (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_mean_rad(x: pd.Series | np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.arctan2(np.sin(values).mean(), np.cos(values).mean()))


def median_abs_deviation(x: pd.Series | np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def circular_mad_rad(x: pd.Series | np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    center = circular_mean_rad(values)
    return float(np.median(np.abs(wrap_rad(values - center))))


def band_for_hz(hz: float) -> str:
    if hz < 1.5e9:
        return "0.9 GHz"
    if hz < 4e9:
        return "2.4 GHz"
    return "5.8 GHz"


def parse_provenance_configuration(entry: dict) -> tuple[float, float]:
    groups = entry["by_lo_and_spacing"]
    if len(groups) != 1:
        raise ValueError(f"expected one LO/spacing group, got {groups}")
    label = next(iter(groups))
    match = re.fullmatch(r"sp([0-9.]+)\.rxlo([0-9.eE+\-]+)", label)
    if match is None:
        raise ValueError(f"unrecognized provenance configuration {label!r}")
    return float(match.group(1)), float(match.group(2))


def load_key_fits(metrics_path: Path, provenance_path: Path) -> tuple[pd.DataFrame, dict]:
    metrics = pd.read_csv(metrics_path)
    with provenance_path.open() as f:
        provenance = json.load(f)

    config_rows = []
    for key, entry in provenance["keys"].items():
        spacing_m, rx_lo_hz = parse_provenance_configuration(entry)
        config_rows.append(
            {
                "key": key,
                "physical_spacing_mm": spacing_m * 1000.0,
                "rx_lo_hz": rx_lo_hz,
                "rx_lo_ghz": rx_lo_hz / 1e9,
                "band": band_for_hz(rx_lo_hz),
                "provenance_n_datasets": int(entry["n_datasets"]),
            }
        )
    config = pd.DataFrame(config_rows)
    keys = metrics.merge(config, on="key", validate="one_to_one")
    if not np.array_equal(keys["n_datasets"], keys["provenance_n_datasets"]):
        raise AssertionError("key source counts disagree with provenance")
    keys["device_name"] = keys["device"].map(DEVICE_NAMES)
    keys["device_band"] = keys["device_name"] + ", " + keys["band"]
    keys["rho_regime"] = pd.cut(
        keys["d_lambda"],
        [-np.inf, 0.25, 0.50, 1.00, np.inf],
        labels=["<0.25", "0.25–0.50", "0.50–1.00", ">1.00"],
        right=False,
    )
    return keys.sort_values(["device", "d_lambda"]).reset_index(drop=True), provenance


def strip_dataset_suffix(path_or_name: str) -> str:
    name = Path(path_or_name).name
    for suffix in (".zarr.zip", ".zarr", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_dataset_fits(
    scan_path: Path, provenance: dict, keys: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    current_names = {
        strip_dataset_suffix(record["path"])
        for record in provenance["datasets"]["records"]
    }
    scan = pd.read_csv(scan_path)
    joined = scan[scan["dataset"].map(strip_dataset_suffix).isin(current_names)].copy()

    key_config = keys.set_index("key")
    joined["key"] = joined.apply(
        lambda r: f"SDRDEVICE.{r['device']}_{float(r['wavelength_spacing']):.5f}",
        axis=1,
    )
    missing_keys = sorted(set(joined["key"]) - set(key_config.index))
    if missing_keys:
        raise AssertionError(f"quality scan contains unmapped current keys: {missing_keys}")

    joined["config_rx_lo_hz"] = joined["key"].map(key_config["rx_lo_hz"])
    joined["config_spacing_mm"] = joined["key"].map(
        key_config["physical_spacing_mm"]
    )
    # A small number of scan rows recorded rx_lo=0.  The current table's one-to-one
    # provenance mapping supplies the known configuration without inventing a frequency.
    joined["rx_lo_hz_filled"] = joined["rx_lo"].where(
        joined["rx_lo"] > 0, joined["config_rx_lo_hz"]
    )
    joined["rx_lo_mhz"] = np.rint(joined["rx_lo_hz_filled"] / 1e6).astype(int)
    joined["band"] = joined["rx_lo_hz_filled"].map(band_for_hz)
    joined["physical_spacing_mm"] = joined["config_spacing_mm"]
    joined["d_lambda"] = joined["wavelength_spacing"].round(5)
    joined["exact_config"] = (
        joined["device"].astype(str)
        + "|"
        + joined["platform"].astype(str)
        + "|"
        + joined["rx_lo_mhz"].astype(str)
        + "|"
        + joined["d_lambda"].map(lambda x: f"{x:.5f}")
    )

    good = joined[
        ~joined["r0_low_coverage"].astype(bool)
        & ~joined["r1_low_coverage"].astype(bool)
    ].copy()
    counts = good.groupby("exact_config")["dataset"].transform("size")
    configured = good[counts >= CONFIG_MIN_DATASETS].copy()

    long_rows = []
    for receiver in ("r0", "r1"):
        part = configured[
            [
                "dataset",
                "device",
                "platform",
                "band",
                "rx_lo_mhz",
                "physical_spacing_mm",
                "d_lambda",
                "exact_config",
                "status",
                f"{receiver}_g",
                f"{receiver}_dtheta",
                f"{receiver}_offset_c",
                f"{receiver}_circstd_corr",
                f"{receiver}_g_at_bound",
            ]
        ].copy()
        part = part.rename(
            columns={
                f"{receiver}_g": "g",
                f"{receiver}_dtheta": "theta_shift_rad",
                f"{receiver}_offset_c": "phase_offset_rad",
                f"{receiver}_circstd_corr": "phase_sigma_rad",
                f"{receiver}_g_at_bound": "g_at_bound",
            }
        )
        part["receiver"] = receiver
        long_rows.append(part)
    receiver_fits = pd.concat(long_rows, ignore_index=True)
    receiver_fits["theta_shift_deg"] = np.degrees(receiver_fits["theta_shift_rad"])
    receiver_fits["phase_offset_deg"] = np.degrees(receiver_fits["phase_offset_rad"])
    receiver_fits["phase_sigma_deg"] = np.degrees(receiver_fits["phase_sigma_rad"])
    receiver_fits["device_name"] = receiver_fits["device"].map(DEVICE_NAMES)

    diagnostics = {
        "quality_scan_rows": int(len(scan)),
        "joined_current_rows": int(len(joined)),
        "good_both_receivers_rows": int(len(good)),
        "configured_rows": int(len(configured)),
        "receiver_observations": int(len(receiver_fits)),
        "exact_configurations": int(configured["exact_config"].nunique()),
        "scan_rows_not_current_or_failed": int(len(scan) - len(joined)),
        "current_provenance_rows_not_in_scan": int(
            provenance["datasets"]["loaded"] - len(joined)
        ),
        "grid_bound_flag_receiver_observations": int(
            receiver_fits["g_at_bound"].sum()
        ),
    }
    return joined, receiver_fits, diagnostics


def grouped_r2(
    data: pd.DataFrame, value: str, group_cols: list[str], circular: bool = False
) -> float:
    values = data[value].to_numpy(dtype=float)
    finite = np.isfinite(values)
    d = data.loc[finite].copy()
    values = values[finite]
    if circular:
        center = circular_mean_rad(values)
        total = np.square(wrap_rad(values - center)).sum()
        predicted = d.groupby(group_cols, dropna=False)[value].transform(
            lambda x: circular_mean_rad(x)
        )
        within = np.square(wrap_rad(values - predicted.to_numpy(dtype=float))).sum()
    else:
        center = values.mean()
        total = np.square(values - center).sum()
        predicted = d.groupby(group_cols, dropna=False)[value].transform("mean")
        within = np.square(values - predicted.to_numpy(dtype=float)).sum()
    return float(1.0 - within / total) if total > 0 else math.nan


def systematics_tables(
    configured: pd.DataFrame, receiver_fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = [
        ("receiver", ["receiver"]),
        ("device + receiver", ["device", "receiver"]),
        ("device + band + receiver", ["device", "band", "receiver"]),
        (
            "device + band + physical spacing + receiver",
            ["device", "band", "physical_spacing_mm", "receiver"],
        ),
        ("exact configuration + receiver", ["exact_config", "receiver"]),
    ]
    rows = []
    for param, meta in PARAMETERS.items():
        circular = param == "phase"
        value = "phase_offset_rad" if circular else meta["sample"]
        for level, columns in levels:
            rows.append(
                {
                    "parameter": param,
                    "parameter_label": meta["short"],
                    "grouping": level,
                    "r2_in_sample": grouped_r2(
                        receiver_fits, value, columns, circular=circular
                    ),
                }
            )

        exact_cols = ["exact_config", "receiver"]
        if circular:
            mads = receiver_fits.groupby(exact_cols)[value].apply(circular_mad_rad)
            median_mad = math.degrees(float(mads.median()))
        else:
            mads = receiver_fits.groupby(exact_cols)[value].apply(median_abs_deviation)
            median_mad = float(mads.median())
        rows[-1]["median_exact_config_within_mad"] = median_mad
        rows[-1]["within_mad_unit"] = "unitless" if param == "g" else "degrees"
    configured = configured.copy()
    configured["phase_delta_rad"] = wrap_rad(
        configured["r0_offset_c"] - configured["r1_offset_c"]
    )
    delta_groups = configured.groupby("exact_config")["phase_delta_rad"]
    rows.append(
        {
            "parameter": "receiver_phase_delta",
            "parameter_label": r"$c_{r0}-c_{r1}$",
            "grouping": "exact configuration",
            "r2_in_sample": grouped_r2(
                configured,
                "phase_delta_rad",
                ["exact_config"],
                circular=True,
            ),
            "median_exact_config_within_mad": math.degrees(
                float(delta_groups.apply(circular_mad_rad).median())
            ),
            "within_mad_unit": "degrees",
        }
    )
    systematics = pd.DataFrame(rows)
    config_rows = []
    for config, group in configured.groupby("exact_config", sort=True):
        row = {
            "exact_config": config,
            "device": group["device"].iloc[0],
            "platform": group["platform"].iloc[0],
            "rx_lo_mhz": int(group["rx_lo_mhz"].iloc[0]),
            "physical_spacing_mm": float(group["physical_spacing_mm"].iloc[0]),
            "d_lambda": float(group["d_lambda"].iloc[0]),
            "n_datasets": int(len(group)),
            "r0_minus_r1_phase_mean_deg": math.degrees(
                circular_mean_rad(group["phase_delta_rad"])
            ),
            "r0_minus_r1_phase_mad_deg": math.degrees(
                circular_mad_rad(group["phase_delta_rad"])
            ),
        }
        for receiver in ("r0", "r1"):
            for out_name, source, circular in (
                ("g", f"{receiver}_g", False),
                ("theta_deg", f"{receiver}_dtheta", False),
                ("phase_deg", f"{receiver}_offset_c", True),
                ("sigma_deg", f"{receiver}_circstd_corr", False),
            ):
                values = group[source].astype(float)
                if circular:
                    center = math.degrees(circular_mean_rad(values))
                    mad = math.degrees(circular_mad_rad(values))
                elif out_name.endswith("_deg"):
                    center = math.degrees(float(values.median()))
                    mad = math.degrees(median_abs_deviation(values))
                else:
                    center = float(values.median())
                    mad = median_abs_deviation(values)
                row[f"{receiver}_{out_name}_median_or_circular_mean"] = center
                row[f"{receiver}_{out_name}_mad"] = mad
        config_rows.append(row)
    config_summary = pd.DataFrame(config_rows).sort_values(
        ["device", "platform", "rx_lo_mhz", "d_lambda"]
    )
    return systematics, config_summary


def key_parameter_summary(keys: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(level: str, group: str, frame: pd.DataFrame) -> None:
        for param, meta in PARAMETERS.items():
            values = frame[meta["key"]].astype(float)
            q = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "parameter": param,
                    "n_keys": int(len(frame)),
                    "n_source_datasets": int(frame["n_datasets"].sum()),
                    "min": float(values.min()),
                    "p05": float(q.loc[0.05]),
                    "p25": float(q.loc[0.25]),
                    "median": float(q.loc[0.50]),
                    "p75": float(q.loc[0.75]),
                    "p95": float(q.loc[0.95]),
                    "max": float(values.max()),
                }
            )

    add("overall", "all keys", keys)
    for group, frame in keys.groupby("device_band", sort=False):
        add("device_band", group, frame)
    for group, frame in keys.groupby("rho_regime", observed=True, sort=False):
        add("d_lambda_regime", str(group), frame)
    add("identified_subset", "d/lambda >= 0.5", keys[keys["d_lambda"] >= 0.5])
    add(
        "identified_subset",
        "d/lambda >= 0.5 and n >= 10",
        keys[(keys["d_lambda"] >= 0.5) & (keys["n_datasets"] >= 10)],
    )
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_key_histograms(keys: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel()
    ordered_groups = [
        ("Pluto, 0.9 GHz", "PLUTO", "0.9 GHz"),
        ("Pluto, 2.4 GHz", "PLUTO", "2.4 GHz"),
        ("Pluto, 5.8 GHz", "PLUTO", "5.8 GHz"),
        ("bladeRF, 2.4 GHz", "BLADERF2", "2.4 GHz"),
        ("bladeRF, 5.8 GHz", "BLADERF2", "5.8 GHz"),
    ]
    line_styles = {"PLUTO": "-", "BLADERF2": "--"}
    legend_handles = []
    for label, device, band in ordered_groups:
        n = len(keys[(keys["device"] == device) & (keys["band"] == band)])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=BAND_COLORS[band],
                linestyle=line_styles[device],
                linewidth=2.0,
                label=f"{label} (n={n})",
            )
        )

    for ax, (param, meta) in zip(axes, PARAMETERS.items()):
        lo, hi = meta["key_range"]
        bins = np.linspace(lo, hi, 17)
        for _, device, band in ordered_groups:
            values = keys.loc[
                (keys["device"] == device) & (keys["band"] == band), meta["key"]
            ].to_numpy(float)
            if len(values) == 0:
                continue
            weights = np.full(len(values), 100.0 / len(values))
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2.0,
                color=BAND_COLORS[band],
                linestyle=line_styles[device],
            )
        ax.axvline(meta["ideal"], color="#222222", linestyle=":", linewidth=1.3)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(meta["label"])
        ax.set_ylabel("within-family keys (%)")
        ax.set_title(meta["short"])
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Key-level calibration fits: distributions by device and frequency band",
        fontsize=15,
        y=1.035,
    )
    fig.text(
        0.5,
        -0.01,
        "Each key has equal weight. Dotted lines mark ideal g=1 or zero offset; "
        "small families (especially bladeRF 5.8 GHz, n=3) are visibly discrete.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    save_figure(fig, path)


def plot_key_frequency_spacing(keys: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.4), sharey="col")
    x_specs = [
        ("rx_lo_ghz", "carrier / LO (GHz)"),
        ("physical_spacing_mm", "physical spacing label (mm)"),
    ]
    for row, (x_col, x_label) in enumerate(x_specs):
        for col, (param, meta) in enumerate(PARAMETERS.items()):
            ax = axes[row, col]
            for device, marker in DEVICE_MARKERS.items():
                for band in BAND_ORDER:
                    frame = keys[(keys["device"] == device) & (keys["band"] == band)]
                    if frame.empty:
                        continue
                    sizes = 24.0 + 18.0 * np.log10(frame["n_datasets"].to_numpy() + 1)
                    ax.scatter(
                        frame[x_col],
                        frame[meta["key"]],
                        s=sizes,
                        marker=marker,
                        color=BAND_COLORS[band],
                        alpha=0.82,
                        edgecolor="white",
                        linewidth=0.55,
                    )
            ax.axhline(meta["ideal"], color="#222222", linestyle=":", linewidth=1.1)
            ax.set_xlabel(x_label)
            if col == 0:
                ax.set_ylabel(meta["short"])
            if row == 0:
                ax.set_title(meta["label"])
            ax.set_ylim(meta["key_range"])
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BAND_COLORS[b],
               markeredgecolor="white", markersize=8, label=b)
        for b in BAND_ORDER
    ] + [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#777777",
               markersize=8, label=DEVICE_NAMES[d])
        for d, m in DEVICE_MARKERS.items()
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Key-level calibration parameters versus frequency and physical spacing",
        fontsize=15,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.01,
        "Point area increases with source-dataset count. Frequency, spacing, device, "
        "and corpus are unbalanced, so visible slopes are descriptive rather than causal.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    save_figure(fig, path)


def plot_dataset_histograms(receiver_fits: pd.DataFrame, path: Path) -> None:
    row_specs = [
        ("Pluto wall", "PLUTO", "wall"),
        ("Pluto rover", "PLUTO", "rover"),
        ("bladeRF wall", "BLADERF2", "wall"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.2), sharex="col", sharey="col")
    for row, (row_label, device, platform) in enumerate(row_specs):
        family = receiver_fits[
            (receiver_fits["device"] == device)
            & (receiver_fits["platform"] == platform)
        ]
        for col, (param, meta) in enumerate(PARAMETERS.items()):
            ax = axes[row, col]
            lo, hi = meta["sample_range"]
            bins = np.linspace(lo, hi, 29)
            for band in BAND_ORDER:
                values = family.loc[family["band"] == band, meta["sample"]].dropna()
                if values.empty:
                    continue
                weights = np.full(len(values), 100.0 / len(values))
                ax.hist(
                    values,
                    bins=bins,
                    weights=weights,
                    histtype="step",
                    linewidth=1.8,
                    color=BAND_COLORS[band],
                    label=f"{band} (n={len(values) // 2} datasets)",
                )
            ax.axvline(meta["ideal"], color="#222222", linestyle=":", linewidth=1.1)
            ax.set_xlim(lo, hi)
            if row == 0:
                ax.set_title(meta["short"])
            if row == 2:
                ax.set_xlabel(meta["label"])
            if col == 0:
                ax.set_ylabel(f"{row_label}\nwithin-band fits (%)")
            if col == 3:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    axes[0, 0].text(
        0.02,
        0.96,
        "wall g grid: 0.70–3.00",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )
    axes[1, 0].text(
        0.02,
        0.96,
        "rover g grid: 0.90–1.10",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )
    fig.suptitle(
        "Per-dataset, per-receiver diagnostic fits (good angular coverage; exact configs n≥5)",
        fontsize=15,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "r0 and r1 are pooled within each panel. Histogram spikes partly reflect bounded, "
        "quantized scan grids; they are not posterior certainty or true physical discreteness.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    save_figure(fig, path)


def plot_configuration_systematics(
    systematics: pd.DataFrame, config_summary: pd.DataFrame, path: Path
) -> None:
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(15.5, 8.6), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )
    grouping_order = [
        "receiver",
        "device + receiver",
        "device + band + receiver",
        "device + band + physical spacing + receiver",
        "exact configuration + receiver",
    ]
    grouping_labels = [
        "receiver",
        "device",
        "device + band",
        "device + band + spacing",
        "exact configuration",
    ]
    x = np.arange(len(grouping_order))
    width = 0.19
    param_colors = {
        "g": "#0072B2",
        "theta": "#E69F00",
        "phase": "#CC79A7",
        "sigma": "#009E73",
    }
    for offset, (param, meta) in enumerate(PARAMETERS.items()):
        frame = systematics[systematics["parameter"] == param].set_index("grouping")
        values = frame.loc[grouping_order, "r2_in_sample"].to_numpy()
        ax0.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            color=param_colors[param],
            label=meta["short"],
        )
    ax0.set_xticks(x, grouping_labels, rotation=26, ha="right")
    ax0.set_ylim(0, 1.0)
    ax0.set_ylabel(r"in-sample variance explained $R^2$")
    ax0.set_title("How much fitted variation is configuration-systematic?")
    ax0.legend(frameon=False, ncol=2, loc="upper left")
    ax0.text(
        0.02,
        0.02,
        "High exact-config R² indicates repeatability,\nnot a causal decomposition.",
        transform=ax0.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )

    shown = config_summary[config_summary["n_datasets"] >= 10].copy()
    shown["abs_mean"] = shown["r0_minus_r1_phase_mean_deg"].abs()
    shown = shown.nlargest(16, "abs_mean").sort_values("r0_minus_r1_phase_mean_deg")
    labels = shown.apply(
        lambda r: (
            f"{DEVICE_NAMES[r['device']]} {r['platform']}  "
            f"{r['rx_lo_mhz'] / 1000:.3f} GHz  ρ={r['d_lambda']:.5f}  n={int(r['n_datasets'])}"
        ),
        axis=1,
    )
    y = np.arange(len(shown))
    colors = [BAND_COLORS[band_for_hz(v * 1e6)] for v in shown["rx_lo_mhz"]]
    ax1.errorbar(
        shown["r0_minus_r1_phase_mean_deg"],
        y,
        xerr=shown["r0_minus_r1_phase_mad_deg"],
        fmt="none",
        ecolor="#777777",
        elinewidth=1.2,
        capsize=2.5,
        zorder=1,
    )
    ax1.scatter(
        shown["r0_minus_r1_phase_mean_deg"],
        y,
        c=colors,
        s=48,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax1.axvline(0, color="#222222", linestyle=":", linewidth=1.1)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(-130, 130)
    ax1.set_xlabel(r"circular mean of $c_{r0}-c_{r1}$ (deg); whisker = within-config MAD")
    ax1.set_title("Largest repeatable receiver-path phase differences")
    fig.suptitle(
        "Configuration structure is strong for phase offset, gain, and residual spread",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    keys, provenance = load_key_fits(args.metrics, args.provenance)
    joined, receiver_fits, diagnostics = load_dataset_fits(
        args.quality_scan, provenance, keys
    )
    configured_names = set(receiver_fits["dataset"])
    configured = joined[joined["dataset"].isin(configured_names)].copy()
    systematics, config_summary = systematics_tables(configured, receiver_fits)
    parameter_summary = key_parameter_summary(keys)

    keys.to_csv(args.output_dir / "calibration_key_parameters.csv", index=False)
    parameter_summary.to_csv(
        args.output_dir / "calibration_parameter_summary.csv", index=False
    )
    systematics.to_csv(
        args.output_dir / "calibration_systematics_summary.csv", index=False
    )
    config_summary.to_csv(
        args.output_dir / "calibration_configuration_summary.csv", index=False
    )
    packaged_scan = args.output_dir / "calibration_quality_scan_inputs.csv"
    if args.quality_scan.resolve() != packaged_scan.resolve():
        # Normalize line endings so the frozen CSV is reviewable on every platform
        # without changing field content.
        with args.quality_scan.open("r", newline=None) as source, packaged_scan.open(
            "w", newline="\n"
        ) as destination:
            shutil.copyfileobj(source, destination)
    with (args.output_dir / "calibration_distribution_metadata.json").open("w") as f:
        json.dump(
            {
                **diagnostics,
                "configuration_min_datasets": CONFIG_MIN_DATASETS,
                "inputs": {
                    "key_metrics": {
                        "path": str(args.metrics),
                        "sha256": sha256_file(args.metrics),
                    },
                    "rebuild_provenance": {
                        "path": str(args.provenance),
                        "sha256": sha256_file(args.provenance),
                    },
                    "quality_scan": {
                        "path": str(args.quality_scan),
                        "sha256": sha256_file(args.quality_scan),
                    },
                },
                "parameter_model": (
                    "phi_measured = wrap(c - 2*pi*g*(d/lambda)*sin(theta-delta) + noise)"
                ),
                "quality_scan_bounds": {
                    "wall_g": [0.70, 3.00],
                    "rover_g": [0.90, 1.10],
                    "wall_delta_rad": [-0.35, 0.35],
                    "rover_delta_rad": [-0.90, 0.90],
                },
                "notes": [
                    "R2 is descriptive and in-sample.",
                    "Phase-offset R2 uses squared wrapped-angle distances.",
                    "Within-MAD is the median of exact-configuration/receiver group MADs.",
                    "Configuration analysis requires both receivers to have usable angular coverage and exact groups of at least five datasets.",
                    "rx_lo=0 quality-scan rows are filled from the key's one-to-one rebuild provenance configuration.",
                ],
            },
            f,
            indent=2,
        )

    plot_key_histograms(
        keys, args.figure_dir / "calibration_parameter_histograms.png"
    )
    plot_key_frequency_spacing(
        keys, args.figure_dir / "calibration_parameters_by_frequency_spacing.png"
    )
    plot_dataset_histograms(
        receiver_fits, args.figure_dir / "per_dataset_calibration_histograms.png"
    )
    plot_configuration_systematics(
        systematics,
        config_summary,
        args.figure_dir / "calibration_configuration_systematics.png",
    )

    print(json.dumps(diagnostics, indent=2))
    exact = systematics[systematics["grouping"] == "exact configuration + receiver"]
    print(
        exact[
            ["parameter", "r2_in_sample", "median_exact_config_within_mad"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
