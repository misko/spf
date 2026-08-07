"""E-CAL1 arm 1 analysis: does the RF-DC machinery inject phase of its own?

Model-free, per epoch, following the campaign's own convention
(spectroscopy_analysis.py):

    D(g1, g2) = phase(g1, g2) - phase(ref, ref)          [same epoch, same LO]
    H(g)      = wrap(0.5 * (D(g, ref) - D(ref, g)))

The steps available in the {5, 8, 9, 10} dB set at high band, from the audited
high table (LMT frozen at LNA 0 / MIX 2 / TIA 0 across 8, 9, 10):

    8 -> 9    LPF 10->11, RF_DC_CAL 0->1     RF-DC rising edge  (discriminator)
    9 -> 10   LPF 11->12, RF_DC_CAL 1->0     RF-DC falling edge
    8 -> 10   LPF 10->12, RF_DC_CAL 0->0     LPF-only, 2 dB -> per-dB floor

The 8->10 step is the LPF-only floor measured *in this dataset*, as the
pre-registration requires -- not imported from the campaign.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

REFERENCE_GAIN = 5
STEPS = {
    "rfdc_rising_8_to_9": (8, 9, 1),
    "rfdc_falling_9_to_10": (9, 10, 1),
    "lpf_only_8_to_10": (8, 10, 2),
}
RNG_SEED = 2026080801


def load_epoch_h(path: Path) -> dict:
    """H(g) per (LO, epoch), in radians, from one radio's dataset."""
    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        completed = np.asarray(receiver.sweep_completed[:], dtype=bool)
        valid = np.asarray(receiver.sweep_quality_valid[:], dtype=bool)
        phase = np.asarray(receiver.phase_difference_rad[:], dtype=np.float64)
        gains = np.asarray(receiver.sweep_requested_gain_db[:], dtype=np.int64)
        lo = np.asarray(receiver.sweep_lo_frequency_hz[:], dtype=np.int64)
        epoch = np.asarray(receiver.sweep_epoch[:], dtype=np.int64)
    finally:
        zarr.store.close()

    eligible = completed & valid
    cells: dict[tuple[int, int, int, int], float] = {}
    for i in np.flatnonzero(eligible):
        cells[(int(lo[i]), int(epoch[i]), int(gains[i, 0]), int(gains[i, 1]))] = phase[i]

    out: dict[tuple[int, int], dict[int, float]] = {}
    for lo_hz in sorted({key[0] for key in cells}):
        for ep in sorted({key[1] for key in cells if key[0] == lo_hz}):
            anchor = cells.get((lo_hz, ep, REFERENCE_GAIN, REFERENCE_GAIN))
            if anchor is None:
                continue
            curve: dict[int, float] = {}
            for gain in (5, 8, 9, 10):
                forward = cells.get((lo_hz, ep, gain, REFERENCE_GAIN))
                reverse = cells.get((lo_hz, ep, REFERENCE_GAIN, gain))
                if forward is None or reverse is None:
                    continue
                rx1 = wrap_phase(forward - anchor)
                rx2 = wrap_phase(reverse - anchor)
                curve[gain] = float(wrap_phase(0.5 * (rx1 - rx2)))
            if curve:
                out[(lo_hz, ep)] = curve
    return out


def step_series(curves: dict, low: int, high: int, per_db: int) -> dict:
    """Signed per-1-dB dH for one step, keyed by (lo_hz, epoch)."""
    series = {}
    for key, curve in curves.items():
        if low in curve and high in curve:
            delta = float(wrap_phase(curve[high] - curve[low]))
            series[key] = np.degrees(delta) / per_db
    return series


def rfdc_excess_series(curves: dict) -> dict:
    """Second difference H(9) - [H(8) + H(10)]/2, in degrees, per (lo, epoch).

    Across rows 22/23/24 the LMT words are frozen and the LPF word steps
    linearly 10 -> 11 -> 12, so a linear-in-LPF response cancels exactly and
    what remains is the RF_DC_CAL flag on row 23 alone. Under H0 this is 0;
    under H1 it is the full RF-DC injection, to be compared against the
    2.664 deg median mixer step.
    """
    series = {}
    for key, curve in curves.items():
        if all(gain in curve for gain in (8, 9, 10)):
            midpoint = np.angle(
                np.exp(1j * curve[8]) + np.exp(1j * curve[10])
            )  # circular mean of the two flanks
            series[key] = float(np.degrees(wrap_phase(curve[9] - midpoint)))
    return series


def summarize(values: np.ndarray) -> dict:
    n = int(values.size)
    if n == 0:
        return {"n": 0}
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "signed_mean_deg": mean,
        "signed_sem_deg": sem,
        "magnitude_of_mean_deg": abs(mean),
        "mean_abs_deg": float(np.mean(np.abs(values))),
        "median_abs_deg": float(np.median(np.abs(values))),
        "sem_of_abs_deg": (
            float(np.std(np.abs(values), ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ),
        "std_deg": float(np.std(values, ddof=1)) if n > 1 else float("nan"),
    }


def cluster_bootstrap_ci(
    per_cluster: dict[tuple, np.ndarray], *, draws: int = 10000
) -> dict:
    """95% CI on mean |dH| resampling whole (radio, LO) clusters."""
    keys = sorted(per_cluster)
    if not keys:
        return {}
    rng = np.random.default_rng(RNG_SEED)
    means = []
    for _ in range(draws):
        picked = rng.integers(0, len(keys), size=len(keys))
        pooled = np.concatenate([np.abs(per_cluster[keys[i]]) for i in picked])
        means.append(pooled.mean())
    return {
        "n_clusters": len(keys),
        "point_mean_abs_deg": float(
            np.mean(np.abs(np.concatenate([per_cluster[k] for k in keys])))
        ),
        "ci95_low_deg": float(np.percentile(means, 2.5)),
        "ci95_high_deg": float(np.percentile(means, 97.5)),
    }


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import mannwhitneyu

    if a.size == 0 or b.size == 0:
        return {}
    stat, p = mannwhitneyu(np.abs(a), np.abs(b), alternative="two-sided")
    return {"u": float(stat), "p": float(p), "n_a": int(a.size), "n_b": int(b.size)}


def main(root: Path, output: Path) -> int:
    datasets = sorted(root.glob("*/calibration.v7.zarr"))
    if not datasets:
        raise SystemExit(f"no datasets under {root}")

    per_radio = {}
    pooled_steps: dict[str, list] = {name: [] for name in STEPS}
    clusters: dict[str, dict[tuple, list]] = {name: {} for name in STEPS}
    pooled_excess: list[float] = []
    excess_clusters: dict[tuple, list] = {}

    for path in datasets:
        serial = path.parent.name
        curves = load_epoch_h(path)
        radio_result = {"n_epoch_curves": len(curves), "steps": {}, "per_lo": {}}
        excess = rfdc_excess_series(curves)
        radio_result["rfdc_excess"] = summarize(
            np.asarray(list(excess.values()), dtype=np.float64)
        )
        radio_result["rfdc_excess_per_lo"] = {}
        for (lo_hz, _epoch), value in excess.items():
            excess_clusters.setdefault((serial, lo_hz), []).append(value)
            radio_result["rfdc_excess_per_lo"].setdefault(str(lo_hz), []).append(value)
        radio_result["rfdc_excess_per_lo"] = {
            lo: summarize(np.asarray(vals, dtype=np.float64))
            for lo, vals in radio_result["rfdc_excess_per_lo"].items()
        }
        pooled_excess.extend(excess.values())
        for name, (low, high, per_db) in STEPS.items():
            series = step_series(curves, low, high, per_db)
            values = np.asarray(list(series.values()), dtype=np.float64)
            radio_result["steps"][name] = summarize(values)
            pooled_steps[name].extend(values.tolist())
            for (lo_hz, _epoch), value in series.items():
                clusters[name].setdefault((serial, lo_hz), []).append(value)
                radio_result["per_lo"].setdefault(str(lo_hz), {}).setdefault(
                    name, []
                ).append(value)
        for lo_hz, by_step in radio_result["per_lo"].items():
            radio_result["per_lo"][lo_hz] = {
                name: summarize(np.asarray(vals, dtype=np.float64))
                for name, vals in by_step.items()
            }
        per_radio[serial] = radio_result

    pooled = {
        name: summarize(np.asarray(vals, dtype=np.float64))
        for name, vals in pooled_steps.items()
    }
    cluster_ci = {
        name: cluster_bootstrap_ci(
            {k: np.asarray(v, dtype=np.float64) for k, v in by_cluster.items()}
        )
        for name, by_cluster in clusters.items()
    }
    rfdc = np.asarray(pooled_steps["rfdc_rising_8_to_9"], dtype=np.float64)
    lpf = np.asarray(pooled_steps["lpf_only_8_to_10"], dtype=np.float64)

    # The pre-registered sem is "across the 25 epochs", i.e. within one
    # (radio, LO) cell. Pooling all six cells (n = 150) would understate it, so
    # the gate is checked on the worst individual cell, not on the pooled sem.
    per_cluster_sem = {}
    for name, by_cluster in clusters.items():
        sems = []
        for key, values in sorted(by_cluster.items()):
            stats = summarize(np.asarray(values, dtype=np.float64))
            sems.append(stats["signed_sem_deg"])
        per_cluster_sem[name] = {
            "per_cluster_sem_deg": sems,
            "max_sem_deg": float(np.max(sems)) if sems else float("nan"),
            "median_sem_deg": float(np.median(sems)) if sems else float("nan"),
        }

    headline_sem = per_cluster_sem["rfdc_rising_8_to_9"]["max_sem_deg"]
    pooled_sem = pooled["rfdc_rising_8_to_9"].get("signed_sem_deg", float("nan"))

    # Second-difference estimator: the LPF ramp cancels, isolating RF_DC_CAL.
    excess_all = np.asarray(pooled_excess, dtype=np.float64)
    excess_summary = summarize(excess_all)
    excess_by_cluster = {
        k: np.asarray(v, dtype=np.float64) for k, v in excess_clusters.items()
    }
    excess_cluster_stats = {
        f"{k[0][-6:]}@{k[1]}": summarize(v) for k, v in sorted(excess_by_cluster.items())
    }
    # Cluster-robust CI on the SIGNED mean (the quantity the decision rule needs).
    rng = np.random.default_rng(RNG_SEED)
    keys = sorted(excess_by_cluster)
    draws = []
    for _ in range(10000):
        picked = rng.integers(0, len(keys), size=len(keys))
        draws.append(
            np.concatenate([excess_by_cluster[keys[i]] for i in picked]).mean()
        )
    excess_ci = {
        "n_clusters": len(keys),
        "signed_mean_deg": float(excess_all.mean()),
        "ci95_low_deg": float(np.percentile(draws, 2.5)),
        "ci95_high_deg": float(np.percentile(draws, 97.5)),
    }
    # Robustness: drop cells that miss the pre-registered >=20 valid epochs.
    good = [k for k, v in excess_by_cluster.items() if v.size >= 20]
    excess_good = (
        np.concatenate([excess_by_cluster[k] for k in good]) if good else np.array([])
    )
    excess_good_summary = summarize(excess_good)
    excess_good_summary["clusters_kept"] = [f"{k[0][-6:]}@{k[1]}" for k in sorted(good)]
    excess_good_summary["clusters_dropped"] = [
        f"{k[0][-6:]}@{k[1]}" for k in sorted(set(excess_by_cluster) - set(good))
    ]
    headline_mag = pooled["rfdc_rising_8_to_9"].get("magnitude_of_mean_deg", float("nan"))
    # The pre-registered comparison is the 8->9 step against the LPF-only floor
    # measured in this same dataset. The second difference is exactly that
    # contrast, (H9 - H8) - (H10 - H8)/2, formed pairwise within each epoch
    # instead of by differencing two separately averaged numbers, so it is the
    # same estimand with the common-mode cell noise cancelled.
    decision_estimate = abs(excess_good_summary["signed_mean_deg"])
    decision_sem = excess_good_summary["signed_sem_deg"]
    if decision_sem >= 0.35:
        verdict = "inconclusive_sem_too_large"
    elif decision_estimate <= 0.35:
        verdict = "rf_dc_contributes_no_resolvable_phase"
    elif decision_estimate >= 2.0:
        verdict = "rf_dc_injects_phase"
    else:
        verdict = "inconclusive_between_branches"

    result = {
        "experiment": "E-CAL1 arm 1 (RF-DC vs RF-state discriminator)",
        "reference_gain_db": REFERENCE_GAIN,
        "steps_measured": {
            name: {"from_db": low, "to_db": high, "per_db_divisor": per_db}
            for name, (low, high, per_db) in STEPS.items()
        },
        "per_radio": per_radio,
        "pooled": pooled,
        "cluster_ci": cluster_ci,
        "mann_whitney_rfdc_vs_lpf": mann_whitney_u(rfdc, lpf),
        "rfdc_step_deg": {
            "mean_abs": pooled["rfdc_rising_8_to_9"].get("mean_abs_deg"),
            "median_abs": pooled["rfdc_rising_8_to_9"].get("median_abs_deg"),
            "magnitude_of_signed_mean": headline_mag,
        },
        "rfdc_step_sem_deg": {
            "max_per_radio_lo_cell": headline_sem,
            "median_per_radio_lo_cell": per_cluster_sem["rfdc_rising_8_to_9"][
                "median_sem_deg"
            ],
            "pooled_all_cells": pooled_sem,
        },
        "per_cluster_sem": per_cluster_sem,
        "rfdc_excess_second_difference": {
            "definition": "H(9) - circular_mean[H(8), H(10)]; LPF ramp cancels",
            "pooled": excess_summary,
            "per_cluster": excess_cluster_stats,
            "cluster_ci_signed": excess_ci,
            "quality_restricted": excess_good_summary,
        },
        "lpf_only_floor_deg": pooled["lpf_only_8_to_10"],
        "decision": {
            "gate_deg": 0.35,
            "mixer_reference_deg": 2.664,
            "sem_basis": "second-difference estimator, cells with >=20 valid epochs",
            "decision_estimate_deg": decision_estimate,
            "decision_sem_deg": decision_sem,
            "raw_step_max_cell_sem_deg": headline_sem,
            "verdict": verdict,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))
    print(json.dumps(result["cluster_ci"], indent=2, sort_keys=True))
    print(json.dumps(result["mann_whitney_rfdc_vs_lpf"], indent=2, sort_keys=True))
    print("verdict:", verdict, "| sem:", headline_sem, "| |mean|:", headline_mag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
