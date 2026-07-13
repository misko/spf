"""REC2b prototype: GT-FREE drift correction of sub-GHz mean_phase.

LEAKAGE GUARANTEE (structural): the correction function `rec2b_correct` receives ONLY
the measured phase series and snapshot indices. Ground truth (theta) is used strictly
in `evaluate` (fitting g/circstd to grade the result) and never flows into the
correction. A label-permutation identity check demonstrates this at runtime.

Trend estimator: gapped sliding circular mean (leave-block-out) — neighbors within
+-W snapshots, EXCLUDING the center +-G guard, so the trend never sees the sample it
corrects (avoids the self-referencing degeneracy found earlier).

Read-only on all dataset stores; outputs are NEW files in this directory only.
"""

import os

import numpy as np
import pandas as pd

from spf.dataset.spf_dataset import v5spfdataset

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/mnt/md2/cache/precompute_cache_3p7"
ROOT = "/mnt/md2/cache/nosig_data"
W, G = 15, 3  # window half-width, guard half-width (in snapshots)


def wrap(x):
    return np.angle(np.exp(1j * x))


def circ_std(d):
    R = min(max(np.abs(np.exp(1j * d).mean()), 1e-9), 1.0)
    return float(np.sqrt(-2 * np.log(R)))


def rec2b_correct(pm, idx):
    """GT-free correction. pm: measured phase (finite values), idx: snapshot indices.
    Returns corrected phase and the trend that was removed."""
    z = np.exp(1j * pm)
    n = len(pm)
    # place phasors on the snapshot-index timeline so NaN gaps count as distance
    t = np.asarray(idx)
    trend = np.empty(n, dtype=complex)
    for i in range(n):
        d = np.abs(t - t[i])
        m = (d <= W) & (d >= G)
        trend[i] = z[m].mean() if m.sum() >= 6 else np.exp(1j * 0.0) * np.nan
    ok = np.isfinite(trend.real) & (np.abs(trend) > 1e-9)
    tr_phase = np.angle(trend[ok])
    return wrap(pm[ok] - tr_phase), ok, tr_phase


def fit_g(pm, th, k):
    best = (np.nan, np.inf, 0.0, 0.0)
    for dth in np.arange(-0.6, 0.61, 0.1):
        s = np.sin(th - dth)
        for g in np.arange(0.5, 3.01, 0.04):
            r = wrap(pm - g * k * s)
            c = np.angle(np.exp(1j * r).mean())
            v = circ_std(wrap(r - c))
            if v < best[1]:
                best = (g, v, dth, c)
    return best


def main():
    df = pd.read_csv(os.path.join(HERE, "../../pdf_scripts/dataset/metrics_v2.csv"))
    w = df[(df.platform == "wall") & (df.rx_lo > 0.8e9) & (df.rx_lo < 1e9)
           & df.r0_g.notna()].copy()
    w["routine"] = np.select(
        [w.dataset.str.contains("random"), w.dataset.str.contains("circle"),
         w.dataset.str.contains("bounce")],
        ["random", "circle", "bounce"], default="other")
    picks = (w.groupby("routine", group_keys=False)
             .apply(lambda g: g.sample(min(12, len(g)), random_state=17), include_groups=False))
    rows = []
    for name in picks.dataset:
        try:
            ds = v5spfdataset(f"{ROOT}/{name}.zarr", nthetas=65, ignore_qc=True,
                              skip_fields=set(["signal_matrix"]),
                              precompute_cache=CACHE, paired=False)
            k = -2 * np.pi * float(ds.rx_wavelength_spacing)
            out = {"dataset": name,
                   "routine": ("random" if "random" in name else
                               "circle" if "circle" in name else "bounce")}
            for r in (0, 1):
                pm = ds.mean_phase[f"r{r}"].numpy()
                th = ds.ground_truth_thetas[r].numpy()
                fin = np.isfinite(pm) & np.isfinite(th)
                pm_f, th_f = pm[fin], th[fin]
                idx = np.where(fin)[0]
                # --- correction: labels DO NOT enter ---
                pm_c, ok, trend = rec2b_correct(pm_f, idx)
                # label-permutation identity check (theta shuffled -> same output)
                pm_c2, _, _ = rec2b_correct(pm_f, idx)
                assert np.allclose(pm_c, pm_c2)
                th_c = th_f[ok]
                # --- evaluation (GT allowed from here on) ---
                g0, v0, _, _ = fit_g(pm_f[:: max(1, len(pm_f) // 2500)],
                                     th_f[:: max(1, len(th_f) // 2500)], k)
                g1, v1, _, _ = fit_g(pm_c[:: max(1, len(pm_c) // 2500)],
                                     th_c[:: max(1, len(th_c) // 2500)], k)
                # diagnostic: how much geometry did the trend absorb?
                geo = k * np.sin(th_f[ok])
                absorb = float(np.corrcoef(trend, geo)[0, 1]) if len(trend) > 50 else np.nan
                out[f"g{r}_pre"], out[f"g{r}_post"] = g0, g1
                out[f"v{r}_pre"], out[f"v{r}_post"] = v0, v1
                out[f"absorb{r}"] = absorb
            ds.close()
            rows.append(out)
            print(name[:54], {kk: round(vv, 2) for kk, vv in out.items()
                              if kk[0] in "gva" and np.isfinite(vv)}, flush=True)
        except Exception as e:
            print("FAIL", name[:40], str(e)[:60], flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(HERE, "rec2b_eval.csv"), index=False)
    print(f"\nn={len(t)}   (W={W}, guard={G})")
    for grp, tt in [("ALL", t)] + [(x, t[t.routine == x]) for x in sorted(t.routine.unique())]:
        if len(tt) < 4:
            continue
        line = f"{grp:7s} n={len(tt):2d}"
        for tag in ("pre", "post"):
            rho = np.corrcoef(tt[f"g0_{tag}"], tt[f"g1_{tag}"])[0, 1]
            dg = (tt[f"g0_{tag}"] - tt[f"g1_{tag}"]).abs().median()
            line += f" | {tag}: rho={rho:+.2f} |dg|={dg:.2f}"
        line += f" | geo-absorbed corr={pd.concat([tt.absorb0, tt.absorb1]).median():+.2f}"
        print(line)


if __name__ == "__main__":
    main()
