"""WITHDRAWN 2026-08-14 -- THIS ANALYSIS IS WRONG. Superseded by
geometry_conditioned.py. Retained rather than deleted because its numbers were
published.

The bug: this script bins on ``rx_theta_in_pis`` and calls the bins ground-truth
bearing. ``rx_theta_in_pis`` is the ARRAY MOUNT ORIENTATION and is CONSTANT per
receiver per capture (measured: 1.0 on r0, 0.5 on r1). Every frame of a receiver
fell into ONE bin, so "the phase residual at fixed bearing" was actually the phase
variation across the entire trajectory -- overwhelmingly the real geometric signal.

Everything it reported is withdrawn: the 81.98 deg residual, r = -0.0245,
r^2 = 0.060%, the 2.1% phase share, and the "other 98% is multipath, geometry and
segmentation" interpretation. Correctly conditioned on ground_truth_phis the
residual is 36.73 deg and the correlation is +0.0138 (r^2 = 0.019%) -- the sign of
the correlation was an artifact too.

Original docstring follows.
"""

"""Why the correction changes nothing: it explains 0.06% of rover phase variance.

The paired sweep showed every correction arm slightly WORSE than baseline, and the
negative control degraded by a similar amount. That says the perturbation's
magnitude was being measured, not its content -- but it does not say WHY the
content is absent. This does.

Method: within narrow (2 degree) ground-truth bearing bins, remove the bin mean
from both the measured phase and the model's predicted correction, then regress
one on the other across all bins and captures. Bearing is thereby held ~fixed, so
what remains in phi should be the gain-dependent term the model claims to predict.

    slope beta = +1  the model exactly explains the gain-dependent phase
    slope beta =  0  no relationship on these radios
    slope beta <  0  subtracting the correction ADDS error

Read-only. Opens rover stores through the standard read-only path and writes
nothing.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from spf.dataset.phase_corrected_dataset import PhaseCorrectedDataset  # noqa: E402
from spf.dataset.spf_dataset import v5spfdataset  # noqa: E402

MODEL = ("spf/calibrations/models/gsc9_arm_lut_per_radio/"
         "1040007c4a94000211000b009186843ef2.json")
KW = dict(
    nthetas=65, ignore_qc=True,
    precompute_cache="/mnt/qnap01/mouse9911/rovers_2026/precompute",
    paired=True, snapshots_per_session=1, readahead=True,
    skip_fields=set(["windowed_beamformer", "weighted_beamformer", "all_windows_stats",
                     "downsampled_segmentation_mask", "signal_matrix",
                     "simple_segmentations"]),
    empirical_data_fn="empirical_dists/full_20260809_v1.pkl", segmentation_version=3.7,
)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def main(list_fn, n_captures=8, bin_deg=2.0):
    fns = [l.strip() for l in open(list_fn) if l.strip()][:n_captures]
    X, Y = [], []
    used = 0
    for fn in fns:
        try:
            ds = v5spfdataset(prefix=fn, **KW)
        except Exception:
            continue
        used += 1
        w = PhaseCorrectedDataset(ds, "arm_lut", MODEL)
        for r in range(ds.n_receivers):
            phi = np.asarray(ds.mean_phase[f"r{r}"], dtype=float)
            corr = np.asarray(w._corr[r], dtype=float)
            th = np.asarray(ds.cached_keys[r]["rx_theta_in_pis"], dtype=float) * np.pi
            ok = np.isfinite(phi) & np.isfinite(th) & np.isfinite(corr)
            phi, corr, th = phi[ok], corr[ok], th[ok]
            b = np.round(np.degrees(wrap(th)) / bin_deg).astype(int)
            for bb in np.unique(b):
                m = b == bb
                if m.sum() < 12:
                    continue
                pm = np.angle(np.mean(np.exp(1j * phi[m])))
                X.append(corr[m] - corr[m].mean())
                Y.append(wrap(phi[m] - pm))
    X, Y = np.concatenate(X), np.concatenate(Y)
    beta = float(np.polyfit(X, Y, 1)[0])
    r = float(np.corrcoef(X, Y)[0, 1])
    print(f"{len(X)} frames, {used} captures, {bin_deg:g}-degree bearing bins\n")
    print(f"  slope beta                    = {beta:+.4f}   (+1 would be a perfect model)")
    print(f"  correlation r                 = {r:+.4f}   (SE ~ {1/np.sqrt(len(X)):.4f})")
    print(f"  variance of phi explained     = {100*r**2:.3f}%")
    print(f"  sd of predicted correction    = {np.degrees(X.std()):.3f} deg")
    print(f"  sd of phi residual in a bin   = {np.degrees(Y.std()):.3f} deg")
    print(f"  correction as a share of it   = {100*X.std()/Y.std():.1f}%")
    print("\nbeta is negative but r is ~0, so the slope is noise amplified by the "
          "\nvariance ratio, not a sign error. The correction simply carries almost "
          "\nno information about these radios' phase.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
