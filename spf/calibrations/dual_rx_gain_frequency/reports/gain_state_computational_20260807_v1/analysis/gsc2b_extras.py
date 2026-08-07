"""E-GSC2 supplements.

1. The N = 113 asymptote (fit on the whole dense stage A), so the identifiability
   curve has its right-hand endpoint.
2. An exact reproduction of the E-CAL3 ten-LO refit reported in REPORT.md 8.1:
   train on the prospective campaign's OWN ten pre-registered LOs, test on its
   other 103.
3. The aliasing diagnostic: the conditioning of the two-ripple frequency basis
   evaluated at the training comb, and the magnitude of the fitted ripple
   amplitudes. This is what distinguishes the pre-registered uniform comb from a
   random comb of the same size.

ANCHORED convention throughout; fail-closed; degrees.
"""

from __future__ import annotations

import json

import numpy as np

import gsc_common as G
import spflib as S
from models import build_design

TAU_POOLED = (2.54e-9, 0.94e-9)


def ripple_conditioning(f_hz, taus):
    """Condition number of [cos,sin] at tau1,tau2 over the training comb.

    A comb that aliases the ripple makes these four columns near-collinear, so
    the fitted amplitudes are large and arbitrary between the samples.
    """
    cols = []
    for t in taus:
        cols.append(np.cos(2 * np.pi * f_hz * t))
        cols.append(np.sin(2 * np.pi * f_hz * t))
    M = np.column_stack(cols)
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


def ripple_amplitude(design, theta_full, names_active, taus):
    """RMS magnitude of the fitted ripple amplitude columns, in degrees."""
    amps = [
        abs(v) for n, v in zip(names_active, theta_full)
        if "|cos" in n or "|sin" in n
    ]
    if not amps:
        return 0.0
    return float(np.degrees(np.sqrt(np.mean(np.square(amps)))))


def fit_report(f, model, tr, te, label, uneq):
    d = build_design(f, model.terms)
    tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
    p, sp, taus, ncol = model.fit_eval(d, f.D, tri, tei)
    pred = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    pred[tei] = p
    sup[tei] = sp
    # refit explicitly so the coefficients can be inspected
    active = np.any(np.abs(d.S[tri]) > 0, axis=0)
    X = d.matrix(taus)
    theta = model._solve(X[tri][:, active], f.D[tri])
    names_active = [n for n, a in zip(d.names, active) if a]
    st = G.score(f.D, pred, sup, te, uneq)
    st.update(
        {
            "label": label,
            "tau_ns": (np.asarray(taus) * 1e9).round(4).tolist(),
            "n_columns": int(ncol),
            "n_train_rows": int(tr.sum()),
            "ripple_rms_amp_deg": ripple_amplitude(d, theta, names_active, taus),
            "ripple_max_amp_deg": float(
                np.degrees(
                    max(
                        [abs(v) for n, v in zip(names_active, theta)
                         if "|cos" in n or "|sin" in n] or [0.0]
                    )
                )
            ),
        }
    )
    return st


def main(out_path="gsc2b_extras.json"):
    fa = G.load_anchored(["A"])
    fp = G.load_anchored(["P_dense"])
    los = np.unique(fa.lo_hz)
    prereg = np.array(
        [los[int(np.argmin(np.abs(los / 1e6 - m)))] for m in G.PREREG_10_MHZ]
    )

    out = {"convention": "ANCHORED, fail-closed, degrees"}

    # ---------------- 1. N = 113 asymptote: fit all of stage A ---------------
    cols = {k: np.concatenate([fa.cols[k], fp.cols[k]]) for k in fa.cols}
    f = S.Frames(cols)
    is_A = np.concatenate([np.ones(len(fa), bool), np.zeros(len(fp), bool)])
    uneq = f.g1 != f.g2
    asym = []
    for variant, taus in (("free", None), ("frozen", G.TAU_FLEET)):
        st = fit_report(
            f, G.rung_model("L26", taus), is_A, ~is_A,
            f"N=113 stage-A fit, {variant} delays -> prospective", uneq,
        )
        asym.append(st)
        print(
            f"asymptote {variant:6s} tau={st['tau_ns']} prospective uneq "
            f"{st['uneq_mae_deg']:.4f} (base {st['baseline_uneq_mae_deg']:.4f}) "
            f"cov {st['coverage']:.4f}"
        )
    out["asymptote_N113"] = asym

    # ---------------- 2. exact E-CAL3 ten-LO refit reproduction --------------
    in_prereg = np.isin(fp.lo_hz, prereg)
    uneq_p = fp.g1 != fp.g2
    repro = []
    for variant, taus in (
        ("free", None),
        ("frozen stage-A 2.56/0.92 ns", G.TAU_FLEET),
        ("frozen pooled 2.54/0.94 ns", TAU_POOLED),
    ):
        st = fit_report(
            fp, G.rung_model("L26", taus), in_prereg, ~in_prereg,
            f"E-CAL3 repro: train on prospective's own 10 LOs, {variant}", uneq_p,
        )
        repro.append(st)
        print(
            f"E-CAL3 repro {variant:28s} tau={st['tau_ns']} "
            f"uneq {st['uneq_mae_deg']:8.4f} P95 {st['uneq_p95_deg']:7.3f} "
            f"(base {st['baseline_uneq_mae_deg']:.4f}) "
            f"ripple_rms {st['ripple_rms_amp_deg']:8.2f} deg"
        )
    out["ecal3_ten_lo_reproduction"] = repro

    # ---------------- 3. aliasing diagnostic ---------------------------------
    rng = np.random.default_rng(20260807)
    cond_rand = []
    for _ in range(2000):
        idx = rng.choice(len(los), 10, replace=False)
        cond_rand.append(ripple_conditioning(los[idx], G.TAU_FLEET))
    cond_rand = np.array(cond_rand)
    cond_prereg = ripple_conditioning(prereg, G.TAU_FLEET)
    cond_all = ripple_conditioning(los, G.TAU_FLEET)
    pct = float((cond_rand > cond_prereg).mean())
    out["aliasing"] = {
        "tau_ns": [t * 1e9 for t in G.TAU_FLEET],
        "ripple_period_1_over_tau_mhz": [1e-6 / t for t in G.TAU_FLEET],
        "prereg_comb_mhz": list(G.PREREG_10_MHZ),
        "prereg_spacing_mhz": float(np.median(np.diff(prereg)) / 1e6),
        "prereg_spacing_in_ripple_periods": float(
            np.median(np.diff(prereg)) * G.TAU_FLEET[0]
        ),
        "cond_prereg_10": cond_prereg,
        "cond_random_10_median": float(np.median(cond_rand)),
        "cond_random_10_q90": float(np.percentile(cond_rand, 90)),
        "cond_random_10_max": float(cond_rand.max()),
        "frac_random_10_worse_than_prereg": pct,
        "cond_all_113": cond_all,
    }
    print(
        f"\naliasing: prereg-10 cond {cond_prereg:.2f}; random-10 median "
        f"{np.median(cond_rand):.2f} (q90 {np.percentile(cond_rand,90):.2f}); "
        f"{pct*100:.1f}% of random combs are worse; all-113 cond {cond_all:.3f}"
    )

    # the same, but as the actual sparse-fit outcome on stage A
    fits = []
    for label, sel in (("prereg10", prereg),):
        in_tr = np.isin(fa.lo_hz, sel)
        for variant, taus in (("free", None), ("frozen", G.TAU_FLEET)):
            st = fit_report(
                fa, G.rung_model("L26", taus), in_tr, ~in_tr,
                f"stage-A {label} {variant}", fa.g1 != fa.g2,
            )
            fits.append(st)
    rng2 = np.random.default_rng(11)
    rnd_amp = []
    for j in range(24):
        idx = rng2.choice(len(los), 10, replace=False)
        in_tr = np.isin(fa.lo_hz, los[idx])
        st = fit_report(
            fa, G.rung_model("L26", G.TAU_FLEET), in_tr, ~in_tr,
            f"stage-A random10 #{j} frozen", fa.g1 != fa.g2,
        )
        rnd_amp.append(st)
    out["prereg_vs_random_stage_a"] = {
        "prereg": fits,
        "random_frozen_ripple_rms_median_deg": float(
            np.median([r["ripple_rms_amp_deg"] for r in rnd_amp])
        ),
        "random_frozen_uneq_mae_median_deg": float(
            np.median([r["uneq_mae_deg"] for r in rnd_amp])
        ),
        "random_frozen": rnd_amp,
    }
    for st in fits:
        print(
            f"{st['label']:34s} uneq {st['uneq_mae_deg']:8.4f} "
            f"ripple_rms {st['ripple_rms_amp_deg']:8.2f} deg  "
            f"max {st['ripple_max_amp_deg']:8.2f} deg"
        )
    print(
        "random10 frozen: median ripple_rms "
        f"{out['prereg_vs_random_stage_a']['random_frozen_ripple_rms_median_deg']:.2f} deg, "
        f"median uneq MAE "
        f"{out['prereg_vs_random_stage_a']['random_frozen_uneq_mae_median_deg']:.3f}"
    )

    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
