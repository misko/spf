"""Paired per-capture comparison of the four phase-correction arms.

Never a corpus mean: E-INF1 already produced one false ranking that way (a 0.039
difference that paired testing showed was p = 0.780 with 92% of it from a single
capture). Every arm is compared to `none` on the SAME capture at the SAME seed,
then aggregated with a Wilcoxon signed-rank test over captures.

Reports MSE and calibration std(z). std(z) is arguably the primary endpoint here:
the correctable phase error is ~0.3-0.7 deg against filter RMSEs of 41-56 deg, so
a systematic removal is far more likely to show up in the honesty of the reported
variance than in accuracy.
"""

from __future__ import annotations

import collections
import glob
import math
import pickle
import sys

import numpy as np
from scipy.stats import wilcoxon

ARMS = ("none", "constant", "arm_lut", "shuffled")
FOLD = math.pi ** 2 / 6      # uniform floor, folded half-circle
FULL = math.pi ** 2 / 3      # uniform floor, full circle


def load(workdir):
    rows = []
    for f in glob.glob(f"{workdir}/**/*.pkl", recursive=True):
        try:
            r = pickle.load(open(f, "rb"))
        except Exception:
            continue
        rows += r if isinstance(r, list) else [r]
    return rows


def main(workdir):
    rows = load(workdir)
    print(f"{len(rows)} result rows from {workdir}\n")

    # (family, capture, seed, rx) -> {arm: (mse, std_z, cov1)}
    cell = collections.defaultdict(dict)
    for r in rows:
        m = r["metrics"]
        v = m.get("mse_craft_theta", m.get("mse_single_radio_theta"))
        if v is None:
            continue
        key = (r["type"], r["ds_fn"], r["seed"], r.get("rx_idx", 0))
        cell[key][r["phase_correction"]] = (v, m.get("calib_std_z"), m.get("calib_cov1"))

    for fam in sorted({k[0] for k in cell}):
        base = FOLD if "single_radio" in fam and "dual" not in fam else FULL
        keys = [k for k in cell if k[0] == fam and set(ARMS) <= set(cell[k])]
        if not keys:
            continue
        # average seeds within a capture first -- a capture is the unit of pairing
        by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
        for k in keys:
            for a in ARMS:
                by_cap[k[1]][a].append(cell[k][a])
        caps = sorted(by_cap)
        print(f"=== {fam} ===  {len(caps)} captures, uniform floor {base:.3f} rad²")
        mse = {a: np.array([np.mean([x[0] for x in by_cap[c][a]]) for c in caps]) for a in ARMS}
        sz = {a: np.array([np.nanmean([x[1] for x in by_cap[c][a]]) for c in caps]) for a in ARMS}
        cv = {a: np.array([np.nanmean([x[2] for x in by_cap[c][a]]) for c in caps]) for a in ARMS}
        print(f"{'arm':<10}{'MSE':>9}{'RMSE':>8}{'vs rand':>9}{'std(z)':>9}{'cov68':>8}"
              f"{'ΔMSE vs none':>14}{'better on':>11}{'Wilcoxon p':>12}")
        for a in ARMS:
            d = mse[a] - mse["none"]
            if a == "none":
                p, better, dtxt = float("nan"), "", ""
            else:
                try:
                    p = wilcoxon(mse[a], mse["none"]).pvalue
                except ValueError:
                    p = float("nan")
                better = f"{int((d < 0).sum())}/{len(caps)}"
                dtxt = f"{d.mean():+.4f}"
            print(f"{a:<10}{mse[a].mean():>9.4f}{math.degrees(math.sqrt(mse[a].mean())):>7.1f}°"
                  f"{base/mse[a].mean():>8.2f}x{np.nanmean(sz[a]):>9.2f}{np.nanmean(cv[a]):>8.3f}"
                  f"{dtxt:>14}{better:>11}{'' if a=='none' else f'{p:>12.4f}'}")
        # calibration as a separate paired test
        print(f"{'':<10}calibration std(z) paired vs none:", end="")
        for a in ARMS[1:]:
            try:
                p = wilcoxon(sz[a], sz["none"]).pvalue
            except ValueError:
                p = float("nan")
            print(f"   {a}: Δ={np.nanmean(sz[a]-sz['none']):+.3f} p={p:.4f}", end="")
        print("\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/mnt/qnap01/mouse9911/rovers_2026/filter_runs/phasecorr")
