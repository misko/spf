"""How large an effect could gain_fixed_effects.py have detected? Answer: not this one.

THIS IS THE ANALYSIS THAT SHOULD HAVE RUN FIRST. Addendum 3 reported that gain-state
fixed effects fitted on rover data are WORSE on held-out captures by +0.018 to
+0.087 deg, and read that as an upper bound proving the effect absent. Nobody asked
what the statistic would have read if the effect were REAL. This asks.

Two calibrations, both on the same 124,950 real residuals the addendum used:

1. SENSITIVITY. mean|wrap(e)| responds quadratically to a small phase term buried in
   a 49.24 deg residual, because d2/dx2 |x| is a delta at the origin: adding an
   independent offset of rms A shifts the mean absolute value by ~A^2 * f_e(0).
   Fitting that on the real residual distribution gives

       Delta(mean|e|) = 0.0101 * A^2   deg, for A in deg

   AMENDED 2026-08-14 -- THE AMPLITUDE WAS THE WRONG MOMENT. This originally read
   "the physical term at issue is 1.0-1.9 deg (bench-measured sd)". It is a MEAN
   ABSOLUTE DEVIATION, correctly labelled as such where it is defined
   (rover_model_gsc9_20260814_v1/REPORT.md:313) and mislabelled everywhere it was
   consumed. This law takes a SECOND moment, so the error is squared. The rms about
   the same weighted circular mean is 1.78 / 2.70 / 4.18 / 5.65 deg; observed
   rms/MAD is 1.81-3.02, so even a Gaussian 1.253 conversion would be wrong.
   Corrected, a PERFECT correction is worth +0.032 to +0.322 deg (+0.032 to +0.074
   for R18, the only donor deployed) -- which must be compared against:

2. NOISE. The published number came from ONE fold seed at ONE min_n, with no
   dispersion reported. Re-running the addendum's own per-cell estimator across seeds
   and min_n shows a seed sd of 0.033 deg and a SIGN FLIP between min_n 8 and 25.

AMENDED: with the corrected amplitude the ceiling is 1.3-1.4x the fold-seed sd for
the deployed donor (6.3x for the damaged unit), NOT below it. The experiment was
UNDER-POWERED, not blind. Addendum 3's withdrawal stands on its other two legs, which
are untouched: the ceiling still sits well inside the statistic's observed run-to-run
range (+0.002 to +0.111 deg), and the sign still flips with min_n.

TWO WARNINGS about the methods below, both measured after this file was written:

  - breakeven() COUNTS PARAMETERS, and that is the wrong instrument. It predicted
    4.83/2.86/1.87 deg for cell/arm/rfblock -- a 2.58x spread with the smallest model
    most sensitive. Injecting the real effect along the actual gain sequence measures
    3.08/3.41/2.91 -- a 1.17x spread with the ordering INVERTED, and cost per free
    parameter ANTI-correlated with k. Use lut_injection_power.py and
    detection_threshold.py instead. The values printed here are retained only because
    they were published.

  - sensitivity() injects rng.normal. An i.i.d. Gaussian is ORTHOGONAL to a structured
    estimator's basis: recovery is 35-153% for a gain-shaped term and 0% for a Gaussian
    of identical rms. It is adequate for calibrating the METRIC's response, which is all
    it is used for here (it agrees with a real-shape injection to 3% at ~2 deg), and
    useless for calibrating an ESTIMATOR's power.

Read-only. Opens rover stores through the standard read-only path, writes nothing.
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

# The bench-measured gain-phase term, in degrees RMS, occupancy-weighted over the
# rover's real gain cells. AUDITED 2026-08-14 (true_amplitude.py): the previously used
# 1.0-1.9 was a MAD, not an rms. Full set, R18/R17 x 5766/5840: 1.78 / 2.70 / 4.18 /
# 5.65 deg. R18 is the only donor actually deployed, so its range is used here.
TERM_LO, TERM_HI = 1.78, 2.70   # R18 rms, the deployed donor (NOT the 1.0-1.9 MAD)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def cmean(x):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(x))))) if len(x) else 0.0


def load(list_fn):
    """Same residual construction as gain_fixed_effects.py, same dedup, for comparability.

    The dedup is known to drop 6 stores / 9,274 frames that are disjoint in time rather
    than duplicated (see REPORT.md Retraction 2). It is reproduced here deliberately so
    the calibration describes the statistic AS PUBLISHED.
    """
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
            n = min(len(phi), len(gtp), len(g), len(lo))
            phi, gtp, g, lo = phi[:n], gtp[:n], g[:n], lo[:n]
            ok = (np.isfinite(phi) & np.isfinite(gtp)
                  & np.isfinite(g).all(axis=1) & np.isfinite(lo))
            if ok.sum() < 50:
                continue
            e = wrap(phi[ok] - gtp[ok])
            data.append(dict(rx=rx, e=wrap(e - cmean(e)),
                             g1=g[ok, 0], g2=g[ok, 1], lo=lo[ok]))
    return data


def sensitivity(E, seed=1, draws=12):
    """Ceiling on Delta(mean|e|) from perfectly removing an rms-A term.

    Equivalently the PENALTY of leaving it in: E|wrap(e + d)| - E|wrap(e)| for d
    independent of e.

    AMENDED 2026-08-14: this docstring used to add "a d correlated with e would only
    make the ceiling smaller". That is WRONG IN SIGN. For a term aligned with sign(e)
    the response is LINEAR, not quadratic. Measured for the real donor shape,
    Delta = 0.00692*A + 0.00761*A^2 -- the linear coefficient equals the measured
    alignment E[sign(e)*u] = +0.00699 and DOMINATES below A = 0.91 deg, making the
    true response 1.93x LARGER at A = 0.5. At the ~2 deg that matters here the two
    laws agree to 5%, so the numbers above are unaffected. On this data the
    independence assumption is defensible anyway: corr(LUT, e) = +0.0125, capture-
    bootstrap 95% CI [-0.0025, +0.0302] (1.4 sigma).
    """
    rng = np.random.default_rng(seed)
    base = np.degrees(np.abs(E)).mean()
    out = []
    for A_deg in (0.5, 1.0, 1.4, 1.9, 3.0, 5.0, 10.0):
        A = np.radians(A_deg)
        d = [np.degrees(np.abs(wrap(E + rng.normal(0, A, len(E))))).mean() - base
             for _ in range(draws)]
        out.append((A_deg, float(np.mean(d)), float(np.std(d))))
    return base, out


def cv_cell(data, min_n=8, folds=6, seed=0):
    """The addendum's per-cell estimator, reproduced so its noise can be measured."""
    rx = sorted({d["rx"] for d in data})
    order = np.random.default_rng(seed).permutation(len(rx))
    assign = {rx[i]: int(order[j] % folds) for j, i in enumerate(range(len(rx)))}

    def key(a, b, l):
        return f"c{int(a)}_{int(b)}_{int(round(l/1e6))}"

    before = after = 0.0
    n = 0
    for f in range(folds):
        acc = defaultdict(list)
        for d in data:
            if assign[d["rx"]] == f:
                continue
            for a_, b_, l_, ev in zip(d["g1"], d["g2"], d["lo"], d["e"]):
                acc[key(a_, b_, l_)].append(ev)
        delta = {k: cmean(v) for k, v in acc.items() if len(v) >= min_n}
        for d in data:
            if assign[d["rx"]] != f:
                continue
            p = np.array([delta.get(key(a_, b_, l_), 0.0)
                          for a_, b_, l_ in zip(d["g1"], d["g2"], d["lo"])])
            before += np.abs(d["e"]).sum()
            after += np.abs(wrap(d["e"] - wrap(p - cmean(p)))).sum()
            n += len(d["e"])
    return np.degrees((after - before) / n)


def breakeven(E, k_params, n_eff):
    """rms amplitude at which a free k-parameter fit's signal covers its parameter cost.

    Signal scales as 0.0101*A^2; the cost of k free circular means estimated from
    n_eff effectively-independent frames scales as 0.0101 * k * var(e)/n_eff. Solve.
    """
    sd = np.degrees(E.std())
    return float(np.sqrt(k_params * sd ** 2 / n_eff))


def main(list_fn):
    data = load(list_fn)
    E = np.concatenate([d["e"] for d in data])
    print(f"{len(data)} receiver-streams, {len({d['rx'] for d in data})} unique RX captures, "
          f"{len(E)} frames")
    print(f"residual sd {np.degrees(E.std()):.2f} deg, mean|e| {np.degrees(np.abs(E)).mean():.3f} deg\n")

    base, rows = sensitivity(E)
    print("=== 1. SENSITIVITY: ceiling on Delta(mean|e|) from a PERFECT correction ===")
    for A, d, s in rows:
        print(f"  rms term {A:5.1f} deg  ->  {d:+.4f} deg  (sd {s:.4f})")
    a = np.array([r[0] for r in rows])
    dv = np.array([r[1] for r in rows])
    k = float((dv[:4] / a[:4] ** 2).mean())
    print(f"\n  quadratic law:  Delta = {k:.4f} * A^2  deg")
    print(f"  Gaussian check: f_e(0) = 1/(sd*sqrt(2pi)) = "
          f"{1/(np.degrees(E.std())*np.sqrt(2*np.pi)):.4f} (real residual is peakier)")
    print(f"  AT THE TERM ACTUALLY AT STAKE ({TERM_LO}-{TERM_HI} deg): "
          f"{k*TERM_LO**2:+.4f} to {k*TERM_HI**2:+.4f} deg\n")

    print("=== 2. NOISE: dispersion of the published statistic (cell, all frames) ===")
    ch = [cv_cell(data, seed=s) for s in range(8)]
    print(f"  8 fold seeds: {' '.join(f'{c:+.3f}' for c in ch)}")
    print(f"  mean {np.mean(ch):+.4f}  sd {np.std(ch):.4f}   "
          f"(published value used seed 0 only, with no dispersion reported)")
    print("  min_n sweep (seed 0):")
    for mn in (8, 15, 25, 50, 100):
        print(f"    min_n {mn:4d} -> {cv_cell(data, min_n=mn):+.4f} deg"
              + ("   <-- as published" if mn == 8 else ""))

    print("\n=== 3. BREAK-EVEN AMPLITUDE per parameterisation ===")
    # lag-1 autocorrelation of the residual deflates the effective sample size
    r1 = float(np.mean([np.corrcoef(d["e"][:-1], d["e"][1:])[0, 1]
                        for d in data if len(d["e"]) > 2]))
    n_eff = int(len(E) * (1 - r1) / (1 + r1))
    print(f"  lag-1 autocorr {r1:.3f} -> n_eff {n_eff} of {len(E)} frames")
    for nm, kp in (("cell", 326), ("arm", 114), ("rfblock", 49), ("1-param projection", 1)):
        print(f"  {nm:<20} k={kp:4d}  break-even A = {breakeven(E, kp, n_eff):5.2f} deg")

    print(f"\nCONCLUSION: the term at stake is {TERM_LO}-{TERM_HI} deg. The per-cell fit")
    print("Addendum 3 leans on needs a 2.5-4.8x larger effect before its signal covers its")
    print("own parameter cost; 'arm' needs 1.5-2.9x; only 'rfblock' is anywhere near the")
    print("boundary, and it sits right at the top of the plausible range. Meanwhile the")
    print("signal ceiling (+0.010 to +0.036 deg) is BELOW the statistic's own seed sd")
    print("(0.033 deg) and its sign flips on min_n. The addendum could not have detected")
    print("the effect it claimed to bound. A 1-parameter projection breaks even at 0.27")
    print("deg -- 4-7x below the term -- which is why that is the instrument to use next.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
