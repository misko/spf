"""Geometry-conditioned test of the donor correction on rover data.

REPLACES why_null.py, WHICH WAS WRONG. That script binned on
``rx_theta_in_pis`` and called the bins "ground-truth bearing". They are not:
``rx_theta_in_pis`` is the ARRAY MOUNT ORIENTATION and is CONSTANT per receiver
per capture (measured: 1.0 on r0 and 0.5 on r1). Every frame of a receiver
therefore landed in one bin, so the "residual at fixed bearing" was the phase
variation over the whole trajectory -- overwhelmingly the real geometric signal
the array exists to measure. Its 81.98 deg denominator, its r = -0.0245, its
0.060% and the "2.1% of the phase budget" that followed are all withdrawn.

This conditions on geometry exactly instead of approximately, using the quantity
the dataset already computes:

    e_none = wrap(mean_phase - ground_truth_phi)
    e_corr = wrap(mean_phase - correction - ground_truth_phi)

both centred circularly per stream. It also deduplicates on the underlying RX
capture, because the 48 merged stores reuse RX recordings across TX partners.

WHAT THIS CAN AND CANNOT SHOW. It bounds the DONOR model -- R18's table applied
to rover radios that are explicitly held out from it. A noisy or mismatched
predictor is attenuated toward zero correlation even when the underlying physical
term is real, so a null here does NOT bound a same-radio or sample-weighted
correction. That distinction was elided in the withdrawn analysis.
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


def centre(x):
    return wrap(np.asarray(x) - np.angle(np.mean(np.exp(1j * np.asarray(x)))))


def main(list_fn):
    fns = [l.strip() for l in open(list_fn) if l.strip()]
    seen, use = set(), []
    for f in fns:                      # dedup on the underlying RX capture
        rx = f.split("/")[-1].split(".")[0]
        if rx not in seen:
            seen.add(rx)
            use.append(f)
    print(f"{len(fns)} merged stores -> {len(use)} unique RX captures")

    per_stream, X, Y = [], [], []
    for f in use:
        try:
            ds = v5spfdataset(prefix=f, **KW)
        except Exception:
            continue
        w = PhaseCorrectedDataset(ds, "arm_lut", MODEL)
        for r in range(ds.n_receivers):
            phi = np.asarray(ds.mean_phase[f"r{r}"], dtype=float)
            gtp = np.asarray(ds.ground_truth_phis[r], dtype=float)
            c = np.asarray(w._corr[r], dtype=float)
            n = min(len(phi), len(gtp), len(c))
            phi, gtp, c = phi[:n], gtp[:n], c[:n]
            ok = np.isfinite(phi) & np.isfinite(gtp) & np.isfinite(c)
            if ok.sum() < 50:
                continue
            e0 = centre(wrap(phi[ok] - gtp[ok]))
            e1 = centre(wrap(phi[ok] - c[ok] - gtp[ok]))
            per_stream.append((np.degrees(np.abs(e0)).mean(),
                               np.degrees(np.abs(e1)).mean(), int(ok.sum())))
            X.append(c[ok] - c[ok].mean())
            Y.append(e0)
    a = np.array(per_stream)
    X, Y = np.concatenate(X), np.concatenate(Y)
    d = a[:, 1] - a[:, 0]
    # per-stream bootstrap: streams are the independent unit, not frames
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    r = float(np.corrcoef(X, Y)[0, 1])
    print(f"{len(a)} receiver-streams, {int(a[:,2].sum())} frames\n")
    print(f"  mean |e| without correction : {a[:,0].mean():7.3f} deg")
    print(f"  mean |e| with correction    : {a[:,1].mean():7.3f} deg")
    print(f"  change                      : {d.mean():+7.3f} deg  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  better on {(d<0).sum()}/{len(d)} streams")
    print(f"  corr(correction, residual)  : {r:+.4f}   r^2 = {100*r**2:.3f}%")
    print(f"  sd correction {np.degrees(X.std()):.2f} deg vs sd residual "
          f"{np.degrees(Y.std()):.2f} deg")
    print("\nBounds the DONOR only. Attenuation from predictor error means this does not\n"
          "bound a same-radio or sample-weighted correction.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
