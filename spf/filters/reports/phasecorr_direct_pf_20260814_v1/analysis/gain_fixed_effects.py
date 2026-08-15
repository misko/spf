"""Does gain state carry repeatable phase information in ROVER data, with no model?

!! THE CONCLUSION THIS SCRIPT WAS USED TO SUPPORT IS WITHDRAWN (2026-08-14). !!

The code is correct as far as it goes and its numbers reproduce exactly. What was
wrong was the INFERENCE: a small positive held-out change was read as evidence that
the gain effect is absent. This statistic cannot support that, for two reasons.

1. NO POWER. The reported metric, mean|wrap(e)|, responds QUADRATICALLY to a small
   phase offset buried in a 49.24 deg residual: measured on the real 124,950
   residuals, Delta = 0.0101 * A^2 deg for an rms term of A deg. The term at issue
   is 1.0-1.9 deg, so a PERFECT correction could move the headline by +0.010 to
   +0.036 deg -- against a fold-seed sd of 0.033 deg, and a min_n sensitivity that
   flips the sign (+0.042 at 8, -0.013 at 25). Break-even amplitude for a free fit
   is 4.83 deg (cell) / 2.86 (arm) / 1.87 (rfblock). A one-parameter projection onto
   a hypothesised LUT would break even at 0.27 deg; that is the statistic to use.
   See power_calibration.py.

2. THE NESTING CLAIM WAS FALSE. The original docstring said "nothing imported from a
   bench can beat it", because a same-radio model would be a constrained version of
   this one. It would not: state_keys() below keys on (g1, g2, LO) and NOTHING else,
   while this corpus pools 6 distinct physical Plutos across 3 rover units. Radio
   identity (sdr_serial, present in all 84 streams, one lookup away at line 114) is
   a dimension this fit DISCARDS, so a per-radio model is a different model, not a
   sub-model. Same for a within-buffer gain trajectory.

Two further defects, real but not the cause of the result: 'arm' and 'rfblock' below
sum MARGINAL conditional means rather than fitting jointly (exact 2x overcount when
g1 == g2; measured inflation 1.02x here), and load()'s dedup silently drops 6 merged
stores -- 9,274 frames, 12 streams -- that are DISJOINT IN TIME, not duplicates.

Retained unmodified because its numbers were published. Do not cite it as a bound.

Original docstring follows.
------------------------------------------------------------------------------------

This is the upper bound the donor analyses could not give. Every earlier test asked
"does R18's table predict rover phase?", which conflates two things: a negligible
physical term, and a real term with the wrong predictor. Here the gain effect is
estimated FROM THE ROVER DATA ITSELF, so nothing imported from a bench can beat it.

    e = wrap(mean_phase - ground_truth_phi)        geometry removed exactly
    e ~ alpha_stream + delta(gain state)           delta learned, not imported

alpha_stream is handled by centring each stream circularly, in train and test
alike -- the same "a per-session constant is absorbed downstream" assumption the
rest of the pipeline makes, and the assumption the empirical table was shown to
satisfy.

delta is a CIRCULAR MEAN per gain state, not a least-squares coefficient: e spans
+-180 deg with a ~49 deg sd, and ordinary regression on wrapped angles is invalid
at that spread.

CROSS-VALIDATION IS THE POINT. With ~500 distinct gain cells and ~125k frames,
free parameters always reduce in-sample error. Only prediction on HELD-OUT
PHYSICAL CAPTURES shows whether the effect is repeatable. Folds are split on the
underlying RX capture, because the 48 merged stores are only 42 unique recordings.

Three parameterisations are compared, cheapest first:
    cell        delta[(g1,g2)]                    most flexible, sparsest
    arm         d1[g1] - d2[g2]                   the additive form
    rfblock     mixer/LNA words, arm-specific     the physical form

and two strata: frames whose gain is stable within the buffer, and frames whose
gain changes mid-buffer (69% at 5766 MHz) where the state is ill-defined.

Read-only. Nothing under /mnt is written.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from spf.dataset.spf_dataset import v5spfdataset  # noqa: E402

KW = dict(
    nthetas=65, ignore_qc=True,
    precompute_cache="/mnt/qnap01/mouse9911/rovers_2026/precompute",
    paired=True, snapshots_per_session=1, readahead=True,
    skip_fields=set(["windowed_beamformer", "weighted_beamformer", "all_windows_stats",
                     "downsampled_segmentation_mask", "signal_matrix",
                     "simple_segmentations"]),
    empirical_data_fn="empirical_dists/full_20260809_v1.pkl", segmentation_version=3.7,
)
HS = None


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def cmean(x):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(x))))) if len(x) else 0.0


def state_keys(kind, g1, g2, lo):
    """Feature key(s) per frame for each parameterisation."""
    if kind == "cell":
        return [(f"c{int(a)}_{int(b)}_{int(round(f/1e6))}", 1.0)
                for a, b, f in zip(g1, g2, lo)]
    global HS
    if HS is None:
        sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/"
                           "dual_rx_gain_frequency/reports/"
                           "gain_state_phase_model_20260802_v1/analysis")
        import features
        HS = features.HardwareStates()
    out = []
    for a, b, f in zip(g1, g2, lo):
        fm = int(round(f / 1e6))
        if kind == "arm":
            out.append([(f"a1_{int(a)}_{fm}", 1.0), (f"a2_{int(b)}_{fm}", -1.0)])
        else:
            s1, s2 = HS.state(2, int(a)), HS.state(2, int(b))
            out.append([(f"m1_{s1[1]}_{fm}", 1.0), (f"l1_{s1[0]}_{fm}", 1.0),
                        (f"m2_{s2[1]}_{fm}", -1.0), (f"l2_{s2[0]}_{fm}", -1.0)])
    return out


def load(list_fn):
    fns = [l.strip() for l in open(list_fn) if l.strip()]
    seen, use = set(), []
    for f in fns:
        rx = f.split("/")[-1].split(".")[0]
        if rx not in seen:
            seen.add(rx)
            use.append((rx, f))
    data = []
    for rx, f in use:
        try:
            ds = v5spfdataset(prefix=f, **KW)
        except Exception:
            continue
        for r in range(ds.n_receivers):
            phi = np.asarray(ds.mean_phase[f"r{r}"], dtype=float)
            gtp = np.asarray(ds.ground_truth_phis[r], dtype=float)
            k = ds.cached_keys[r]
            g = np.asarray(k["gains"], dtype=float)
            lo = np.asarray(k["rx_lo"], dtype=float)
            # gain_endpoints_equal lives in the zarr, NOT in cached_keys; reading it
            # from cached_keys silently yields all-True and empties the unstable
            # stratum, which is how the first run of this script was wrong.
            zr = ds.z["receivers"][f"r{r}"]
            stab = np.asarray(zr["gain_endpoints_equal"][:]).all(axis=1)
            n = min(len(phi), len(gtp), len(g), len(lo), len(stab))
            phi, gtp, g, lo, stab = phi[:n], gtp[:n], g[:n], lo[:n], stab[:n]
            ok = (np.isfinite(phi) & np.isfinite(gtp)
                  & np.isfinite(g).all(axis=1) & np.isfinite(lo))
            if ok.sum() < 50:
                continue
            e = wrap(phi[ok] - gtp[ok])
            e = wrap(e - cmean(e))                 # per-stream centring
            data.append(dict(rx=rx, e=e, g1=g[ok, 0], g2=g[ok, 1], lo=lo[ok],
                             stable=np.asarray(stab)[ok]))
    return data


def cv(data, kind, folds=6, min_n=8, mask_name="all", seed=0):
    rx = sorted({d["rx"] for d in data})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rx))
    assign = {rx[i]: int(order[j] % folds) for j, i in enumerate(range(len(rx)))}
    before, after, n_tot = [], [], 0
    for f in range(folds):
        acc = defaultdict(list)
        for d in data:                              # fit on training captures
            if assign[d["rx"]] == f:
                continue
            m = d["stable"] if mask_name == "stable" else (
                ~d["stable"] if mask_name == "unstable" else np.ones(len(d["e"]), bool))
            if not m.any():
                continue
            for feats, ev in zip(state_keys(kind, d["g1"][m], d["g2"][m], d["lo"][m]),
                                 d["e"][m]):
                fs = feats if isinstance(feats, list) else [feats]
                for nm, sgn in fs:
                    acc[nm].append(sgn * ev)
        delta = {k: cmean(v) for k, v in acc.items() if len(v) >= min_n}
        for d in data:                              # evaluate on held-out captures
            if assign[d["rx"]] != f:
                continue
            m = d["stable"] if mask_name == "stable" else (
                ~d["stable"] if mask_name == "unstable" else np.ones(len(d["e"]), bool))
            if not m.any():
                continue
            pred = []
            for feats in state_keys(kind, d["g1"][m], d["g2"][m], d["lo"][m]):
                fs = feats if isinstance(feats, list) else [feats]
                pred.append(sum(sgn * delta.get(nm, 0.0) for nm, sgn in fs))
            e = d["e"][m]
            p = np.asarray(pred)
            before.append(np.abs(e).sum())
            after.append(np.abs(wrap(e - wrap(p - cmean(p)))).sum())
            n_tot += len(e)
    if n_tot == 0:
        return float('nan'), float('nan'), 0
    b, a = np.degrees(sum(before) / n_tot), np.degrees(sum(after) / n_tot)
    return b, a, n_tot


def main(list_fn):
    data = load(list_fn)
    print(f"{len(data)} receiver-streams from {len({d['rx'] for d in data})} unique "
          f"RX captures, {sum(len(d['e']) for d in data)} frames\n")
    print(f"{'stratum':<10}{'model':<10}{'frames':>9}{'mean|e| before':>16}"
          f"{'after':>10}{'change':>10}")
    for mask in ("all", "stable", "unstable"):
        for kind in ("cell", "arm", "rfblock"):
            b, a, n = cv(data, kind, mask_name=mask)
            print(f"{mask:<10}{kind:<10}{n:>9}{b:>15.3f}°{a:>9.3f}°{a-b:>+9.3f}°")
        print()
    print("Held-out captures throughout. A negative change means gain state carries")
    print("repeatable information; ~0 or positive means it does not.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
