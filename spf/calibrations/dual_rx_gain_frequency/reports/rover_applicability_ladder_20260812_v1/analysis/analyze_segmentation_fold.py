"""Does a post-hoc phase correction commute with what segmentation stored?

Measured on the real committed rover precompute caches, read-only. No file under
/mnt is opened for writing and spf/dataset/segmentation.py is not touched.

Two stored quantities behave differently:
  precompute r{i}/mean_phase          weighted CIRCULAR mean, no fold
  precompute r{i}/weighted_windows_stats[0]  trimmed mean of FOLDED per-window phase
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

PRE = Path("/mnt/qnap01/mouse9911/rovers_2026/precompute")
N_CAPTURES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CORRECTIONS_DEG = [2.0, 5.0, 10.0, 20.0]


def fold(t):
    t = np.array(t, dtype=np.float64, copy=True)
    m = np.abs(t) > np.pi / 2
    t[m] = np.sign(t[m]) * np.pi - t[m]
    return t


paths = sorted(PRE.glob("*.yarr"))[:N_CAPTURES]
res = {"n_captures": len(paths), "captures": [p.name for p in paths]}

folded_err = {c: [] for c in CORRECTIONS_DEG}
circ_err = {c: [] for c in CORRECTIONS_DEG}
mp_outside = []
wws_outside = []
n_frames = 0
reproduces = []

for p in paths:
    z = zarr_open_from_lmdb_store(str(p), mode="r")
    for r in ("r0", "r1"):
        if r not in z:
            continue
        aws = np.asarray(z[r]["all_windows_stats"][:, 0, :], dtype=np.float64)
        mask = np.asarray(z[r]["downsampled_segmentation_mask"][:])
        wws = np.asarray(z[r]["weighted_windows_stats"][:, 0])
        mp = np.asarray(z[r]["mean_phase"][:])
        mp_outside.append(np.abs(mp[np.isfinite(mp)]) > np.pi / 2)
        wws_outside.append(np.abs(wws[np.isfinite(wws)]) > np.pi / 2)
        for i in range(aws.shape[0]):
            m = mask[i]
            if m.sum() < 4:
                continue
            w = aws[i][m]
            n_frames += 1
            base_folded = trim_mean(fold(w), 0.1)
            reproduces.append(abs(base_folded - wws[i]))
            # circular reference (what r{i}/mean_phase does, unweighted here)
            base_circ = np.angle(np.exp(1j * w).mean())
            for cdeg in CORRECTIONS_DEG:
                c = np.deg2rad(cdeg)
                # correct per window, THEN fold and trim  (the right way)
                right = trim_mean(fold(w - c), 0.1)
                # subtract from the stored folded scalar     (the post-hoc way)
                wrong = base_folded - c
                folded_err[cdeg].append(np.rad2deg(abs(right - wrong)))
                right_c = np.angle(np.exp(1j * (w - c)).mean())
                wrong_c = base_circ - c
                d = np.rad2deg(abs(np.angle(np.exp(1j * (right_c - wrong_c)))))
                circ_err[cdeg].append(d)

mp_outside = np.concatenate(mp_outside)
wws_outside = np.concatenate(wws_outside)
res["n_frames_scored"] = n_frames
res["stored_mean_phase_outside_half_pi_fraction"] = float(mp_outside.mean())
res["stored_weighted_windows_stats0_outside_half_pi_fraction"] = float(
    wws_outside.mean()
)
res["trimmed_fold_reproduces_weighted_windows_stats_rad"] = dict(
    median=float(np.median(reproduces)), p95=float(np.percentile(reproduces, 95)),
    max=float(np.max(reproduces)),
)
res["post_hoc_error_on_folded_trimmed_mean_deg"] = {}
res["post_hoc_error_on_circular_mean_deg"] = {}
for cdeg in CORRECTIONS_DEG:
    e = np.array(folded_err[cdeg])
    res["post_hoc_error_on_folded_trimmed_mean_deg"][f"{cdeg:g}"] = dict(
        mean=float(e.mean()), median=float(np.median(e)),
        p95=float(np.percentile(e, 95)), max=float(e.max()),
        fraction_above_half_the_correction=float((e > 0.5 * cdeg).mean()),
        fraction_at_least_1p9x_correction=float((e >= 1.9 * cdeg).mean()),
    )
    ec = np.array(circ_err[cdeg])
    res["post_hoc_error_on_circular_mean_deg"][f"{cdeg:g}"] = dict(
        mean=float(ec.mean()), median=float(np.median(ec)),
        p95=float(np.percentile(ec, 95)), max=float(ec.max()),
    )

Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
