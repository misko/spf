#!/usr/bin/env python3
"""Read-only within-file temporal-holdout ablation of phase calibrations.

Persistent Zarr/YARR stores are opened with mode="r", lock=False, and
readahead=False.  Results are emitted to stdout; this script writes no inputs,
caches, or repository files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

def find_repo_root():
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if (parent / "spf").is_dir() and (parent / "empirical_dists").is_dir():
                return parent
    raise RuntimeError("run from an SPF repository checkout or set the working directory to it")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT))
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

SCRIPT = Path(__file__).resolve()
REPORT_DATA = (
    SCRIPT.parent.parent
    if SCRIPT.parent.name == "analysis" and SCRIPT.parent.parent.name == "2026_08_25_empirical_dist_analysis"
    else ROOT / "reports/data/2026_08_25_empirical_dist_analysis"
)
SCAN = REPORT_DATA / "calibration_quality_scan_inputs.csv"
SELECTION = REPORT_DATA / "calibration_heldout_ablation.csv"
ARTIFACT = ROOT / "empirical_dists/full_20260809_v1.pkl"
TAUS = np.round(np.arange(-0.5, 0.50001, 0.025), 6)
PARAMETERS = ("c", "g", "delta")


def portable_path(path):
    """Render repository/report paths without embedding a workstation prefix."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        pass
    try:
        suffix = resolved.relative_to(REPORT_DATA.resolve())
        return str(Path("reports/data") / REPORT_DATA.name / suffix)
    except ValueError:
        return str(resolved)


def wrap(x):
    return np.angle(np.exp(1j * x))


def finite_float(row, key, default):
    try:
        value = float(row.get(key, ""))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def weighted_mean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def weighted_cmean(x, w):
    return float(np.angle(np.sum(w * np.exp(1j * x))))


def stat_store(path):
    path = Path(path)
    candidates = [path / "data.mdb", path]
    for candidate in candidates:
        if candidate.is_file():
            st = candidate.stat()
            return {
                "path": str(candidate),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
    return {"path": str(path), "missing": True}


def stat_key(item):
    return (item.get("path"), item.get("size"), item.get("mtime_ns"), item.get("missing", False))


def load_receiver(z, y, ridx, platform):
    raw = z[f"receivers/r{ridx}"]
    pre = y[f"r{ridx}"]
    t = np.asarray(raw["system_timestamp"][:], dtype=np.float64)
    phi = np.asarray(pre["mean_phase"][:], dtype=np.float64)
    n = min(len(t), len(phi))
    t, phi = t[:n], phi[:n]
    keys = (
        "tx_pos_x_mm",
        "tx_pos_y_mm",
        "rx_pos_x_mm",
        "rx_pos_y_mm",
        "rx_theta_in_pis",
    )
    a = {key: np.asarray(raw[key][:n], dtype=np.float64) for key in keys}
    a["rx_heading_in_pis"] = (
        np.asarray(raw["rx_heading_in_pis"][:n], dtype=np.float64)
        if "rx_heading_in_pis" in raw
        else np.zeros(n, dtype=np.float64)
    )
    if len(t) < 240 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"r{ridx}_nonmonotonic_or_short")
    valid = np.isfinite(phi)
    for value in a.values():
        valid &= np.isfinite(value)
    valid &= (t >= t[0] + 0.5) & (t <= t[-1] - 0.5)
    idx = np.flatnonzero(valid)
    if len(idx) < 100:
        raise ValueError(f"r{ridx}_too_few_fixed_valid")
    block = np.minimum(np.arange(len(idx)) * 10 // len(idx), 9)
    train_idx = idx[(block % 2) == 0]
    test_idx = idx[(block % 2) == 1]
    if len(train_idx) < 40 or len(test_idx) < 40:
        raise ValueError(f"r{ridx}_too_few_train_or_holdout")

    heading = np.unwrap(a["rx_heading_in_pis"] * np.pi)
    mount = a["rx_theta_in_pis"] * np.pi
    theta = wrap(
        np.arctan2(a["tx_pos_x_mm"] - a["rx_pos_x_mm"], a["tx_pos_y_mm"] - a["rx_pos_y_mm"])
        - heading
        - mount
    )
    ranges_m = np.hypot(
        a["tx_pos_x_mm"] - a["rx_pos_x_mm"],
        a["tx_pos_y_mm"] - a["rx_pos_y_mm"],
    ) / 1000.0
    weights = np.clip(ranges_m, 1.0, None) ** 2 if platform == "rover" else np.ones(n)
    return {
        "t": t,
        "phi": phi,
        "arrays": a,
        "heading_unwrapped": heading,
        "theta": theta,
        "weights": weights,
        "train": train_idx,
        "test": test_idx,
    }


def residual_objective(phi, theta, w, k, g, delta, fit_c):
    raw = wrap(phi - g * k * np.sin(theta - delta))
    if fit_c:
        c = weighted_cmean(raw, w)
        resid = wrap(raw - c)
    else:
        c = 0.0
        resid = raw
    loss = 1.0 - weighted_mean(np.cos(resid), w)
    return float(loss), float(c)


def grid_refined_scalar(objective, bounds, n_grid=49):
    grid = np.linspace(bounds[0], bounds[1], n_grid)
    losses = np.asarray([objective(value) for value in grid])
    best = int(np.argmin(losses))
    lo = grid[max(0, best - 1)]
    hi = grid[min(n_grid - 1, best + 1)]
    if hi <= lo:
        return float(grid[best])
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 2e-4})
    return float(result.x) if result.fun < losses[best] else float(grid[best])


def fit_subset(receiver, k, subset, platform):
    train = receiver["train"]
    phi = receiver["phi"][train]
    theta = receiver["theta"][train]
    w = receiver["weights"][train]
    fit_c = "c" in subset
    fit_g = "g" in subset
    fit_delta = "delta" in subset
    g_bounds = (0.70, 3.00) if platform == "wall" else (0.90, 1.10)
    d_bounds = (-0.35, 0.35) if platform == "wall" else (-0.90, 0.90)

    def obj_g(g):
        return residual_objective(phi, theta, w, k, float(g), 0.0, fit_c)[0]

    def obj_d(d):
        return residual_objective(phi, theta, w, k, 1.0, float(d), fit_c)[0]

    def obj_gd(x):
        return residual_objective(phi, theta, w, k, float(x[0]), float(x[1]), fit_c)[0]

    if fit_g and fit_delta:
        # Fixed training-only coarse search prevents full-file quality-scan values
        # from leaking holdout information through optimizer initialization.
        g_grid = np.unique(np.r_[np.linspace(*g_bounds, 9), 1.0])
        d_grid = np.unique(np.r_[np.linspace(*d_bounds, 17), 0.0])
        coarse = sorted((obj_gd((g, d)), float(g), float(d)) for g in g_grid for d in d_grid)
        seeds = [(g, d) for _, g, d in coarse[:5]]
        candidates = []
        for seed in seeds:
            result = minimize(
                obj_gd,
                np.asarray(seed),
                method="Powell",
                bounds=(g_bounds, d_bounds),
                options={"xtol": 2e-4, "ftol": 2e-6, "maxiter": 90},
            )
            candidates.append((float(result.fun), float(result.x[0]), float(result.x[1])))
        _, g, delta = min(candidates)
    elif fit_g:
        g, delta = grid_refined_scalar(obj_g, g_bounds), 0.0
    elif fit_delta:
        g, delta = 1.0, grid_refined_scalar(obj_d, d_bounds)
    else:
        g, delta = 1.0, 0.0

    train_loss, c = residual_objective(phi, theta, w, k, g, delta, fit_c)
    return {"g": g, "delta": delta, "c": c, "train_cosine_loss": train_loss}


def evaluate(receiver, k, params, which):
    idx = receiver[which]
    resid = wrap(
        receiver["phi"][idx]
        - params["g"] * k * np.sin(receiver["theta"][idx] - params["delta"])
        - params["c"]
    )
    w = receiver["weights"][idx]
    return {
        "mae_deg": math.degrees(weighted_mean(np.abs(resid), w)),
        "cosine_loss": 1.0 - weighted_mean(np.cos(resid), w),
        "mean_bias_deg": math.degrees(weighted_cmean(resid, w)),
    }


def theta_at_tau(receiver, idx, tau):
    t = receiver["t"]
    q = t[idx] + tau
    a = receiver["arrays"]
    tx = np.interp(q, t, a["tx_pos_x_mm"])
    ty = np.interp(q, t, a["tx_pos_y_mm"])
    rx = np.interp(q, t, a["rx_pos_x_mm"])
    ry = np.interp(q, t, a["rx_pos_y_mm"])
    heading = np.interp(q, t, receiver["heading_unwrapped"])
    mount = np.interp(q, t, a["rx_theta_in_pis"] * np.pi)
    return wrap(np.arctan2(tx - rx, ty - ry) - heading - mount)


def fit_tau(receivers, k, full_params):
    profiles = []
    for tau in TAUS:
        row = {"tau": float(tau), "train": [], "test_mae": [], "c": []}
        for receiver, params in zip(receivers, full_params):
            train = receiver["train"]
            test = receiver["test"]
            theta_train = theta_at_tau(receiver, train, tau)
            raw_train = wrap(
                receiver["phi"][train]
                - params["g"] * k * np.sin(theta_train - params["delta"])
            )
            w_train = receiver["weights"][train]
            c = weighted_cmean(raw_train, w_train)
            resid_train = wrap(raw_train - c)
            train_loss = 1.0 - weighted_mean(np.cos(resid_train), w_train)
            theta_test = theta_at_tau(receiver, test, tau)
            resid_test = wrap(
                receiver["phi"][test]
                - params["g"] * k * np.sin(theta_test - params["delta"])
                - c
            )
            test_mae = math.degrees(
                weighted_mean(np.abs(resid_test), receiver["weights"][test])
            )
            row["train"].append(float(train_loss))
            row["test_mae"].append(float(test_mae))
            row["c"].append(float(c))
        profiles.append(row)
    best = min(profiles, key=lambda row: np.mean(row["train"]))
    zero = profiles[int(np.argmin(np.abs(TAUS)))]
    return {
        "tau_s": best["tau"],
        "heldout_mae_deg": float(np.mean(best["test_mae"])),
        "zero_heldout_mae_deg": float(np.mean(zero["test_mae"])),
        "incremental_reduction_deg": float(np.mean(zero["test_mae"]) - np.mean(best["test_mae"])),
        "boundary": bool(abs(best["tau"]) >= 0.499),
    }


def shapley_contributions(values):
    # values maps bit mask -> held-out improvement in degrees relative to mask 0.
    result = {}
    n = 3
    for p, name in enumerate(PARAMETERS):
        total = 0.0
        bit = 1 << p
        for mask in range(1 << n):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            total += weight * (values[mask | bit] - values[mask])
        result[name] = float(total)
    return result


def process_job(job):
    record, row = job
    name = Path(record["path"]).name.removesuffix(".zarr")
    yarr = Path(record["precompute_cache"]) / f"{name}_segmentation_nthetas65.yarr"
    before = [stat_store(record["path"]), stat_store(yarr)]
    z = y = None
    try:
        z = zarr_open_from_lmdb_store(record["path"], mode="r", lock=False, readahead=False)
        y = zarr_open_from_lmdb_store(str(yarr), mode="r", lock=False, readahead=False)
        receivers = [load_receiver(z, y, ridx, row["platform"]) for ridx in (0, 1)]
    finally:
        if y is not None:
            y.store.close()
        if z is not None:
            z.store.close()
    after = [stat_store(record["path"]), stat_store(yarr)]
    if [stat_key(x) for x in before] != [stat_key(x) for x in after]:
        raise RuntimeError("input_stat_changed")

    k = -2.0 * np.pi * finite_float(row, "wavelength_spacing", np.nan)
    per_receiver = []
    for ridx, receiver in enumerate(receivers):
        models = {}
        for mask in range(8):
            subset = {PARAMETERS[p] for p in range(3) if mask & (1 << p)}
            params = fit_subset(receiver, k, subset, row["platform"])
            models[str(mask)] = {
                "subset": sorted(subset),
                "params": params,
                "train": evaluate(receiver, k, params, "train"),
                "heldout": evaluate(receiver, k, params, "test"),
            }
        baseline = models["0"]["heldout"]["mae_deg"]
        values = {
            mask: baseline - models[str(mask)]["heldout"]["mae_deg"]
            for mask in range(8)
        }
        per_receiver.append(
            {
                "receiver": f"r{ridx}",
                "models": models,
                "shapley_deg": shapley_contributions(values),
            }
        )
    full_params = [entry["models"]["7"]["params"] for entry in per_receiver]
    tau = fit_tau(receivers, k, full_params)
    baseline = float(np.mean([x["models"]["0"]["heldout"]["mae_deg"] for x in per_receiver]))
    full = float(np.mean([x["models"]["7"]["heldout"]["mae_deg"] for x in per_receiver]))
    shapley = {
        name: float(np.mean([x["shapley_deg"][name] for x in per_receiver]))
        for name in PARAMETERS
    }
    return {
        "dataset": name,
        "platform": row["platform"],
        "device": row.get("device", ""),
        "rho": finite_float(row, "wavelength_spacing", np.nan),
        "rx_lo_hz": finite_float(row, "rx_lo", np.nan),
        "baseline_heldout_mae_deg": baseline,
        "full_static_heldout_mae_deg": full,
        "static_reduction_deg": baseline - full,
        "static_reduction_pct": 100.0 * (baseline - full) / baseline,
        "shapley_deg": shapley,
        "tau": tau,
        "receivers": per_receiver,
        "input_stats": before,
    }


def summarize(records):
    out = {}
    for platform in ("rover", "wall"):
        rows = [r for r in records if r["platform"] == platform]
        if not rows:
            continue
        def arr(path):
            values = []
            for row in rows:
                value = row
                for key in path:
                    value = value[key]
                values.append(value)
            return np.asarray(values, dtype=float)
        summary = {"n": len(rows)}
        for name, path in (
            ("baseline_mae_deg", ("baseline_heldout_mae_deg",)),
            ("full_static_mae_deg", ("full_static_heldout_mae_deg",)),
            ("static_reduction_deg", ("static_reduction_deg",)),
            ("static_reduction_pct", ("static_reduction_pct",)),
            ("c_shapley_deg", ("shapley_deg", "c")),
            ("g_shapley_deg", ("shapley_deg", "g")),
            ("delta_shapley_deg", ("shapley_deg", "delta")),
            ("tau_incremental_deg", ("tau", "incremental_reduction_deg")),
        ):
            x = arr(path)
            summary[name] = {
                "mean": float(np.mean(x)),
                "q10": float(np.quantile(x, 0.10)),
                "q25": float(np.quantile(x, 0.25)),
                "median": float(np.quantile(x, 0.50)),
                "q75": float(np.quantile(x, 0.75)),
                "q90": float(np.quantile(x, 0.90)),
                "positive_fraction": float(np.mean(x > 0)),
            }
        out[platform] = summary
    return out


def selection_names(selection_path):
    if not selection_path.is_file():
        raise FileNotFoundError(f"selection CSV not found: {selection_path}")
    selected = {"rover": set(), "wall": set()}
    with selection_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            selected[row["platform"]].add(row["dataset"])
    return selected


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_record(record):
    return {
        "dataset": record["dataset"],
        "platform": record["platform"],
        "device": record["device"],
        "rho": record["rho"],
        "baseline_mae_deg": record["baseline_heldout_mae_deg"],
        "static_mae_deg": record["full_static_heldout_mae_deg"],
        "tau_mae_deg": record["tau"]["heldout_mae_deg"],
        "static_reduction_deg": record["static_reduction_deg"],
        "static_reduction_pct": record["static_reduction_pct"],
        "c": record["shapley_deg"]["c"],
        "g": record["shapley_deg"]["g"],
        "delta": record["shapley_deg"]["delta"],
        "tau": record["tau"]["incremental_reduction_deg"],
        "tau_s": record["tau"]["tau_s"],
        "tau_boundary": record["tau"]["boundary"],
    }


def write_compact_products(output, csv_path, summary_path):
    rows = [compact_record(record) for record in output["records"]]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    compact = {
        "method": output["method"],
        "provenance": output["provenance"],
        "summary": output["summary"],
        "elapsed_s": output["elapsed_s"],
        "record_csv": portable_path(csv_path),
        "record_csv_sha256": sha256(csv_path),
        "record_rows": len(rows),
        "notes": [
            "Positive c/g/d/tau values reduce held-out mean absolute circular phase error.",
            "c/g/d are per-file Shapley attributions and sum to static_reduction_deg up to floating-point roundoff.",
            "This is a within-file temporal-block holdout, not a capture-level or hardware-unit holdout.",
            "All source Zarr/YARR stores were opened read-only and every touched data.mdb size/mtime fingerprint was unchanged.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(compact, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--platform", choices=("rover", "wall", "both"), default="both")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--compact-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    started = time.time()
    names = selection_names(args.selection)
    with SCAN.open(newline="") as handle:
        scan = {row["dataset"]: row for row in csv.DictReader(handle)}
    with ARTIFACT.open("rb") as handle:
        artifact = pickle.load(handle)
    records_by_name = {}
    for record in artifact["__provenance__"]["datasets"]["records"]:
        name = Path(record["path"]).name.removesuffix(".zarr")
        if name in records_by_name:
            raise AssertionError(f"duplicate basename: {name}")
        records_by_name[name] = record
    jobs = []
    platforms = ("rover", "wall") if args.platform == "both" else (args.platform,)
    for platform in platforms:
        selected = sorted(names[platform])
        missing = sorted(set(selected) - set(records_by_name))
        if missing:
            raise AssertionError(f"{platform} selection missing from provenance: {missing[:3]}")
        for name in selected:
            row = scan.get(name)
            if row is None:
                raise AssertionError(f"selection missing scan row: {name}")
            jobs.append((records_by_name[name], row))
    if args.limit:
        jobs = jobs[: args.limit]

    results, failures = [], []
    for index, job in enumerate(jobs, 1):
        try:
            results.append(process_job(job))
        except Exception as error:
            failures.append(
                {
                    "dataset": Path(job[0]["path"]).name,
                    "type": type(error).__name__,
                    "reason": str(error),
                }
            )
        if index % 25 == 0:
            print(f"processed {index}/{len(jobs)}", file=sys.stderr, flush=True)

    all_stats = [tuple(stat_key(item)) for record in results for item in record["input_stats"]]
    output = {
        "method": {
            "metric": "within-file temporal-block-heldout weighted mean absolute circular phase error (degrees)",
            "split": "10 contiguous blocks, alternating 5 train / 5 holdout",
            "fit_objective": "training weighted circular cosine loss; c analytic",
            "parameter_attribution": "three-parameter Shapley value over all 8 c/g/d subsets",
            "tau": "shared per-file tau in [-0.5,+0.5] s, 0.025 s grid, selected on train after c/g/d; g/d fixed during tau scan",
            "weighting": "each file equal; receivers averaged; rover observations weighted by max(range_m,1)^2",
            "bounds": {
                "wall_g": [0.70, 3.00],
                "rover_g": [0.90, 1.10],
                "wall_delta_rad": [-0.35, 0.35],
                "rover_delta_rad": [-0.90, 0.90],
            },
        },
        "provenance": {
            "scan": portable_path(SCAN),
            "scan_sha256": sha256(SCAN),
            "artifact": portable_path(ARTIFACT),
            "artifact_sha256": sha256(ARTIFACT),
            "requested_jobs": len(jobs),
            "completed": len(results),
            "failures": failures,
            "unique_input_store_fingerprints": len(set(all_stats)),
            "all_per_job_store_stats_unchanged": True,
        },
        "summary": summarize(results),
        "records": results,
        "elapsed_s": time.time() - started,
    }
    if bool(args.compact_csv) != bool(args.summary_json):
        parser.error("--compact-csv and --summary-json must be supplied together")
    if args.compact_csv:
        write_compact_products(output, args.compact_csv, args.summary_json)
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
