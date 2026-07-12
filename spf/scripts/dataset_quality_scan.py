"""
Read-only fleet scanner for dataset reliability/correctness metrics.

Computes ground-truth-based metrics (claude_docs/03_datasets/data_quality_plan.md)
for every dataset in the train/val splits, branching per platform (wall array vs
rover, plan section 1b): platform-specific validity gates, correction menus, and
distance weighting. Fits the low-dimensional systematic model per receiver:

    phi_meas ~ c + g * (-2*pi*(d/lambda)) * sin(theta_gt - dtheta)

Outputs (never modifies any dataset): a CSV of per-dataset rows plus two ranked
markdown reports (wall / rover), stratified by device x spacing.

Usage:
  python spf/scripts/dataset_quality_scan.py \
      --splits /mnt/md2/splits/apr17_train_nosig_noroverbounce_noblade.txt \
               /mnt/md2/splits/apr17_val_nosig_noroverbounce.txt \
      --precompute-cache /mnt/md2/cache/precompute_cache_3p7 \
      --output-dir data_quality_reports/scan1 [--limit N] [--parallel 12]
"""

import argparse
import csv
import logging
import os
import traceback
from multiprocessing import Pool

import numpy as np
import torch

from spf.dataset.spf_dataset import v5spfdataset

# ---------------- circular statistics ----------------


def circ_mean_std(d, w=None):
    z = np.exp(1j * d)
    zb = z.mean() if w is None else (z * w).sum() / w.sum()
    R = min(max(np.abs(zb), 1e-9), 1.0)
    return float(np.angle(zb)), float(np.sqrt(-2 * np.log(R)))


def wrap(x):
    return np.angle(np.exp(1j * x))


# ---------------- per-receiver metric core ----------------


def fit_systematics(phi_m, theta, k, g_grid, d_grid, w=None):
    """Grid-fit gain g and mount shift dtheta; offset c is analytic (circ mean).
    Returns (g, dtheta, c, circstd_corrected)."""
    best = (1.0, 0.0, 0.0, np.inf)
    sin_t = {}
    for dth in d_grid:
        sin_t[dth] = np.sin(theta - dth)
    for g in g_grid:
        for dth in d_grid:
            resid = wrap(phi_m - g * k * sin_t[dth])
            c, s = circ_mean_std(resid, w)
            if s < best[3]:
                best = (float(g), float(dth), c, s)
    return best


def receiver_metrics(phi_m, theta, k, platform, ranges_m, max_fit_points=4000):
    out = {}
    ok = np.isfinite(phi_m) & np.isfinite(theta)
    out["nan_frac"] = float(1.0 - ok.mean()) if len(phi_m) else 1.0
    out["n_valid"] = int(ok.sum())
    if out["n_valid"] < 50:
        return out
    pm, th = phi_m[ok], theta[ok]
    rng = ranges_m[ok]

    # distance weighting (rover only): bearing-noise variance ~ 1/range^2
    w = None
    if platform == "rover":
        w = np.clip(rng, 1.0, None) ** 2
        w = w / w.sum()

    resid_raw = wrap(pm - k * np.sin(th))
    out["bias"], out["circstd_raw"] = circ_mean_std(resid_raw, w)
    out["outlier_frac"] = float((np.abs(wrap(resid_raw - out["bias"])) > 1.0).mean())

    # subsample for the grid fit
    if len(pm) > max_fit_points:
        idx = np.linspace(0, len(pm) - 1, max_fit_points).astype(int)
        pm_f, th_f = pm[idx], th[idx]
        w_f = None if w is None else w[idx]
    else:
        pm_f, th_f, w_f = pm, th, w

    # platform correction menus (plan section 1b; grids widened in v2)
    if platform == "wall":
        g_grid = np.arange(0.70, 3.01, 0.02)
        d_grid = np.arange(-0.35, 0.351, 0.02)
    else:  # rover: g is a diagnostic, not a fit target
        g_grid = np.arange(0.90, 1.11, 0.02)
        d_grid = np.arange(-0.90, 0.901, 0.02)
    # v2: coverage-aware identifiability guard — with narrow angular coverage,
    # g and dtheta are degenerate; fit offset only.
    occ = np.histogram(th_f, bins=np.linspace(-np.pi, np.pi, 13))[0]
    out["coverage_bins"] = int((occ >= 20).sum())
    out["low_coverage"] = bool(out["coverage_bins"] < 4)
    if out["low_coverage"]:
        c, s_corr = circ_mean_std(wrap(pm_f - k * np.sin(th_f)), w_f)
        out.update({"g": 1.0, "dtheta": 0.0, "offset_c": c,
                    "circstd_corr": s_corr, "g_at_bound": False})
    else:
        g, dth, c, s_corr = fit_systematics(pm_f, th_f, k, g_grid, d_grid, w_f)
        out.update({"g": g, "dtheta": dth, "offset_c": c, "circstd_corr": s_corr})
        out["g_at_bound"] = bool(
            np.isclose(g, g_grid[0]) or np.isclose(g, g_grid[-1])
            or np.isclose(abs(dth), d_grid[-1])
        )

    # M11 drift proxy: residual circ-mean per time quarter (post-fit)
    resid_fit = wrap(pm - out["g"] * k * np.sin(th - out["dtheta"]) - out["offset_c"])
    quarters = np.array_split(resid_fit, 4)
    qmeans = [circ_mean_std(q)[0] for q in quarters if len(q) > 10]
    out["drift_span"] = float(np.max(qmeans) - np.min(qmeans)) if len(qmeans) > 1 else 0.0

    # M12 structure: dispersion of binned residual means across theta
    bins = np.linspace(-np.pi, np.pi, 13)
    bmeans = []
    bi = np.digitize(th, bins) - 1
    for b in range(12):
        sel = bi == b
        if sel.sum() > 20:
            bmeans.append(circ_mean_std(resid_fit[sel])[0])
    out["structure"] = (
        float(circ_mean_std(np.array(bmeans))[1]) if len(bmeans) >= 4 else np.nan
    )
    return out


# ---------------- per-dataset worker ----------------


def scan_one(job):
    zarr_fn, precompute_cache = job
    torch.set_num_threads(1)
    row = {"dataset": os.path.basename(zarr_fn).replace(".zarr", "")}
    try:
        ds = v5spfdataset(
            zarr_fn,
            nthetas=65,
            ignore_qc=True,
            skip_fields=set(["signal_matrix"]),
            precompute_cache=precompute_cache,
            paired=False,
        )
        name = row["dataset"].lower()
        platform = "wall" if ("wall" in name or "wall" in str(ds.vehicle_type).lower()) else "rover"
        row.update(
            platform=platform,
            device=str(ds.sdr_device_type).replace("SDRDEVICE.", ""),
            wavelength_spacing=round(float(ds.rx_wavelength_spacing), 5),
            rx_lo=float(ds.cached_keys[0]["rx_lo"][0]),
            n_snapshots=int(ds.n_snapshots),
        )

        # shared geometry
        tx = np.stack(
            [ds.cached_keys[0]["tx_pos_x_mm"].numpy(), ds.cached_keys[0]["tx_pos_y_mm"].numpy()], 1
        )
        rx = np.stack(
            [ds.cached_keys[0]["rx_pos_x_mm"].numpy(), ds.cached_keys[0]["rx_pos_y_mm"].numpy()], 1
        )
        ranges_m = np.linalg.norm(tx - rx, axis=1) / 1000.0
        row["range_med_m"] = float(np.median(ranges_m))

        # M3 position liveness + M4 timestamps
        ts = ds.cached_keys[0]["system_timestamp"].numpy()
        dt = np.diff(ts)
        row["ts_nonmono_frac"] = float((dt <= 0).mean()) if len(dt) else np.nan
        row["ts_med_dt"] = float(np.median(dt)) if len(dt) else np.nan
        moved = (np.abs(np.diff(rx, axis=0)).sum(1) + np.abs(np.diff(tx, axis=0)).sum(1)) > 0
        if len(moved):
            frozen = ~moved
            # longest frozen run + does it end the dataset (the #42 signature)
            runs, cur = [], 0
            for f in frozen:
                cur = cur + 1 if f else 0
                runs.append(cur)
            row["frozen_max_frac"] = float(np.max(runs) / len(frozen))
            row["frozen_tail"] = bool(runs[-1] > max(20, 0.05 * len(frozen)))
            if row["frozen_tail"]:
                # v2: snapshot index where the terminal freeze begins (salvage prefix)
                row["frozen_tail_start"] = int(len(frozen) - runs[-1] + 1)
        # speed plausibility (mm/interval -> m/s)
        if len(dt) and np.median(dt) > 0:
            spd = np.linalg.norm(np.diff(rx, axis=0), axis=1) / 1000.0 / np.maximum(dt, 1e-3)
            row["rx_speed_p99"] = float(np.percentile(spd, 99))

        # per-receiver residual metrics
        k = -2 * np.pi * float(ds.rx_wavelength_spacing)
        for r in range(2):
            phi_m = ds.mean_phase[f"r{r}"].numpy()
            theta = ds.ground_truth_thetas[r].numpy()
            m = receiver_metrics(phi_m, theta, k, platform, ranges_m)
            row.update({f"r{r}_{kk}": vv for kk, vv in m.items()})
        ds.close()

        # common/differential mount decomposition (plan M9/M13; wall too in v2)
        if "r0_dtheta" in row and "r1_dtheta" in row:
            row["dtheta_common"] = round((row["r0_dtheta"] + row["r1_dtheta"]) / 2, 4)
            row["dtheta_diff"] = round((row["r0_dtheta"] - row["r1_dtheta"]) / 2, 4)
            if platform == "rover":
                row["heading_common"] = row["dtheta_common"]
                row["mount_diff"] = row["dtheta_diff"]
        row["scan_v"] = 2

        row["status"], row["reasons"] = gate(row)
    except Exception as e:  # noqa: BLE001 - robust fleet scan, record and continue
        row["status"] = "ERROR"
        row["reasons"] = f"{type(e).__name__}: {e}"
        logging.debug(traceback.format_exc())
    return row


def gate(row):
    """Tier-1/2 gates, thresholds per platform (plan section 1b)."""
    reasons = []
    plat = row.get("platform", "?")
    nan0, nan1 = row.get("r0_nan_frac", 1.0), row.get("r1_nan_frac", 1.0)
    if plat == "wall":
        mx = max(nan0, nan1)
        if mx > 0.20:
            reasons.append("QUAR:nan>20%")
        elif mx > 0.05:
            reasons.append("FLAG:nan5-20%")
        if row.get("frozen_tail", False):
            reasons.append("QUAR:frozen_tail(#42)")
        for r in (0, 1):
            if abs(row.get(f"r{r}_g", 1.0) - 1.0) > 0.25:
                reasons.append(f"FLAG:r{r}_gain={row.get(f'r{r}_g'):.2f}")
            if row.get(f"r{r}_circstd_corr", 0.0) > 0.8:
                reasons.append(f"FLAG:r{r}_noisy")
    else:
        if max(nan0, nan1) > 0.90 or min(
            row.get("r0_n_valid", 0), row.get("r1_n_valid", 0)
        ) < 100:
            reasons.append("QUAR:no_signal")
        if abs(row.get("heading_common", 0.0)) > 0.25:
            reasons.append(f"FLAG:heading={row.get('heading_common'):.2f}")
        for r in (0, 1):
            if row.get(f"r{r}_circstd_corr", 0.0) > 0.7:
                reasons.append(f"FLAG:r{r}_noisy")
    if row.get("ts_nonmono_frac", 0.0) and row["ts_nonmono_frac"] > 0.01:
        reasons.append("FLAG:ts_nonmonotonic")
    if any(row.get(f"r{r}_g_at_bound", False) for r in (0, 1)):
        reasons.append("FLAG:fit_at_bound")
    if any(row.get(f"r{r}_low_coverage", False) for r in (0, 1)):
        reasons.append("INFO:low_coverage")
    status = (
        "QUARANTINE"
        if any(x.startswith("QUAR") for x in reasons)
        else ("FLAG" if any(x.startswith("FLAG") for x in reasons) else "OK")
    )
    return status, ";".join(reasons)


# ---------------- reporting ----------------


def write_report(rows, platform, out_fn):
    rs = [r for r in rows if r.get("platform") == platform and r.get("status") != "ERROR"]
    errs = [r for r in rows if r.get("status") == "ERROR"]
    with open(out_fn, "w") as f:
        f.write(f"# Dataset quality scan — {platform} ({len(rs)} datasets)\n\n")
        # group summary: device x spacing
        groups = {}
        for r in rs:
            key = (r.get("device", "?"), r.get("wavelength_spacing", "?"))
            groups.setdefault(key, []).append(r)
        f.write("## Group summary (device x d/lambda)\n\n")
        f.write("| device | d/lam | n | med circstd raw r0 | med corr r0 | med g r0 | med dtheta r0 |"
                " med heading | QUAR | FLAG |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for (dev, sp), g in sorted(groups.items()):
            med = lambda kk: (  # noqa: E731
                f"{np.nanmedian([x.get(kk, np.nan) for x in g]):.3f}"
                if any(np.isfinite(x.get(kk, np.nan)) for x in g)
                else "-"
            )
            nq = sum(1 for x in g if x["status"] == "QUARANTINE")
            nf = sum(1 for x in g if x["status"] == "FLAG")
            f.write(
                f"| {dev} | {sp} | {len(g)} | {med('r0_circstd_raw')} | {med('r0_circstd_corr')} |"
                f" {med('r0_g')} | {med('r0_dtheta')} | {med('heading_common')} | {nq} | {nf} |\n"
            )
        # worst first
        f.write("\n## Worst 25 (by corrected circstd r0)\n\n")
        f.write("| dataset | status | reasons | nan% r0/r1 | circstd raw->corr r0 | g r0 | dtheta r0 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        key_fn = lambda r: -(r.get("r0_circstd_corr") or 0)  # noqa: E731
        for r in sorted(rs, key=key_fn)[:25]:
            f.write(
                f"| {r['dataset'][:48]} | {r['status']} | {r.get('reasons','')[:40]} |"
                f" {100*r.get('r0_nan_frac',1):.0f}/{100*r.get('r1_nan_frac',1):.0f} |"
                f" {r.get('r0_circstd_raw',np.nan):.2f}->{r.get('r0_circstd_corr',np.nan):.2f} |"
                f" {r.get('r0_g',np.nan):.2f} | {r.get('r0_dtheta',np.nan):+.2f} |\n"
            )
        f.write(f"\n## Errors ({len(errs)})\n\n")
        for r in errs[:20]:
            f.write(f"- {r['dataset']}: {r.get('reasons','')[:120]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--precompute-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    files = []
    for s in args.splits:
        with open(s) as f:
            files += [line.strip() for line in f if line.strip()]
    files = sorted(set(files))
    if args.limit:
        files = files[: args.limit]
    logging.info(f"scanning {len(files)} datasets with {args.parallel} workers")

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [(f, args.precompute_cache) for f in files]
    with Pool(args.parallel) as p:
        rows = []
        for i, row in enumerate(p.imap_unordered(scan_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                logging.info(f"{i+1}/{len(files)} done")

    # v2: serial re-check of ERRORs (parallel LMDB opens can fake failures)
    path_of = {os.path.basename(f_).replace(".zarr", ""): f_ for f_ in files}
    for i, row in enumerate(rows):
        if row.get("status") == "ERROR" and row["dataset"] in path_of:
            retry = scan_one((path_of[row["dataset"]], args.precompute_cache))
            if retry.get("status") != "ERROR":
                retry["reasons"] = (retry.get("reasons", "") + ";recovered_on_serial_retry").strip(";")
                rows[i] = retry
            else:
                rows[i]["reasons"] = (row.get("reasons", "") or "") + " (persistent)"

    # CSV (union of keys)
    keys = sorted({k for r in rows for k in r})
    csv_fn = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for plat in ("wall", "rover"):
        write_report(rows, plat, os.path.join(args.output_dir, f"report_{plat}.md"))
    n_by = {}
    for r in rows:
        n_by[r.get("status", "?")] = n_by.get(r.get("status", "?"), 0) + 1
    logging.info(f"wrote {csv_fn}; status counts: {n_by}")


if __name__ == "__main__":
    main()
