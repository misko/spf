"""E-GSC5 -- what should ship as the default rung: L26, L30, or both?

RESCOPED TWICE, and the rescoping matters more than the numbers.

1. LEAKAGE. The entry as written in docs/future_experiments.md would select the
   default by scoring on the prospective 103-LO capture. That is model selection
   on the only clean test set that exists, and MODEL_FITTING_AND_EVALUATION.md
   section 8 forbids it. So: the criterion is PRE-REGISTERED below and evaluated
   on A-G data ONLY. The prospective numbers are computed by a separate script
   stage and reported once, as confirmation or refutation.

2. THE ORIGINAL RULE CANNOT FIRE. "Promote L30 if it is within 0.1 deg of L26 on
   unseen-frequency error AND better on band transfer AND at least equal on
   coverage" is conjunctive, and the first clause fails by roughly 1.28 deg
   (stage-A LOFO: L26 2.262 vs L30 3.539). The rule would stop there and never
   reach the band-transfer evidence it exists to weigh. So the question is
   re-framed as: should the shipped default be BAND-CONDITIONAL?

PRE-REGISTERED CRITERION (fixed before any prospective number was computed):

  P1 within-band default. Among rungs with >= 99% coverage on the same mask,
     take the lowest unseen-frequency error (stage-A LOFO, confirmed by pooled
     LOFO and by stage-A LOBLK). Ties inside 0.1 deg go to the simpler rung.
  P2 cross-band default. A rung may be recommended across an UNMEASURED
     gain-table band only if, on the identical leave-one-band-out mask, it beats
     the anchor-only baseline by >= 0.5 deg at >= 95% coverage. Beating the
     other rung is not sufficient: both must clear the baseline. If none clears
     it, the cross-band default is "fail closed", i.e. anchor only.
  P3 second rung. L30 ships as an additionally-supported rung if it wins P2, or
     if it is strictly better than L26 in the rule-5 regime (both arms sharing
     the audited LNA/MIXER/TIA words), where it is neutral by construction and
     L26 is known to inject error.
  P4 every error is reported with coverage on the same mask, in both the
     all-cell and the unequal-gain-only convention.

ANCHORED convention throughout; fail-closed to the anchor.
"""

from __future__ import annotations

import json

import numpy as np

import features as FT
import gsc_common as G
import ladder as LD
import spflib as S
from models import build_design

RUNGS = ("L26", "L30", "L31")


def cv(f, split, rungs=RUNGS):
    """Cross-validate every rung on the SAME rows, and return per-rung stats
    plus the predictions, so downstream regime slices use identical masks."""
    uneq = f.g1 != f.g2
    out = {}
    preds = {}
    for name in rungs:
        m = G.rung_model(name)
        d = build_design(f, m.terms)
        pred = np.zeros(len(f))
        sup = np.zeros(len(f), dtype=bool)
        seen = np.zeros(len(f), dtype=bool)
        taus, npar = [], []
        for _lbl, tr, te in LD.SPLITS[split](f):
            tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
            if not len(tri) or not len(tei):
                continue
            p, sp, tt, nc = m.fit_eval(d, f.D, tri, tei)
            pred[tei] = p
            sup[tei] = sp
            seen[tei] = True
            npar.append(nc)
            if tt.size:
                taus.append(tt)
        st = G.score(f.D, pred, sup, seen, uneq)
        st["params"] = int(np.median(npar)) if npar else 0
        st["tau_ns"] = (
            np.round(np.median(np.array(taus), axis=0) * 1e9, 4).tolist()
            if taus else []
        )
        out[name] = st
        preds[name] = (pred, sup, seen)
    return out, preds


def show(title, res):
    print(f"\n{title}")
    print(f"  {'rung':6s} {'p':>5s} {'cov':>6s} {'MAE':>8s} {'uneqMAE':>8s} "
          f"{'P95':>8s}   (baseline all/uneq "
          f"{res[RUNGS[0]]['baseline_mae_deg']:.3f}/"
          f"{res[RUNGS[0]]['baseline_uneq_mae_deg']:.3f})")
    for k, v in res.items():
        print(f"  {k:6s} {v['params']:5d} {v['coverage']:6.3f} {v['mae_deg']:8.4f} "
              f"{v['uneq_mae_deg']:8.4f} {v['p95_deg']:8.3f}")


def rule5_regime(f, preds):
    """Cells where the audited (LNA, MIXER, TIA) words are identical on both
    arms and the gains differ. REPORT.md section 8 rule 5 says do not apply the
    correction there; L30/L31 are neutral there by construction."""
    same = np.ones(len(f), dtype=bool)
    for fld in ("lna", "mixer", "tia"):
        v1 = FT.HW.vec(f.band, f.g1, fld)
        v2 = FT.HW.vec(f.band, f.g2, fld)
        same &= (v1 == v2) & (v1 != -999)
    m = same & (f.g1 != f.g2)
    out = {"n_cells": int(m.sum()), "baseline_mae_deg": S.cmae_deg(f.D[m])}
    for k, (pred, sup, seen) in preds.items():
        mm = m & seen
        fc = np.where(sup, pred, 0.0)
        err = S.wrap(f.D[mm] - fc[mm])
        out[k] = {
            "mae_deg": S.cmae_deg(err),
            "max_deg": S.cmax_deg(err),
            "mean_injected_deg": float(
                np.mean(np.abs(np.degrees(S.wrap(fc[mm]))))
            ),
            "frac_made_worse": float(
                np.mean(np.abs(err) > np.abs(S.wrap(f.D[mm])))
            ),
        }
    return out


def main(out_path="gsc5_default.json"):
    res = {
        "preregistered_criterion": __doc__.split("PRE-REGISTERED CRITERION")[1]
        .split("ANCHORED")[0]
        .strip(),
        "selection_data": "A-G campaign only (stage A and the pooled set). "
                          "No prospective number entered the selection.",
    }

    # ---------------- P1: within-band, unseen frequency ---------------------
    fa = G.load_anchored(["A"])
    a_lofo, a_lofo_preds = cv(fa, "LOFO leave-one-frequency-out")
    show("stage A, leave-one-frequency-out (113 LOs, 3389 rows)", a_lofo)
    a_loblk, _ = cv(fa, "LOBLOCK leave-frequency-block-out")
    show("stage A, leave-frequency-block-out (~690 MHz gaps)", a_loblk)
    a_loro, _ = cv(fa, "LORO leave-one-radio-out")
    show("stage A, leave-one-radio-out", a_loro)

    fp = FT.add_anchor(
        S.load_stages(["A", "F", "E_tx_0", "rate_pilot"]), ref=None, per_epoch=True
    )
    print(
        f"\npooled rows={len(fp)} LOs={len(np.unique(fp.lo_hz))} "
        f"gains={len(set(fp.g1.tolist())|set(fp.g2.tolist()))}"
    )
    p_lofo, p_lofo_preds = cv(fp, "LOFO leave-one-frequency-out")
    show("pooled A+F+E+rate_pilot, leave-one-frequency-out", p_lofo)

    # ---------------- P2: cross-band portability ----------------------------
    p_loband, _ = cv(fp, "LOBAND leave-one-gain-table-band-out")
    show("pooled, leave-one-gain-table-band-out (A-G only)", p_loband)

    # ---------------- P3: the rule-5 regime ---------------------------------
    r5 = rule5_regime(fp, p_lofo_preds)
    print(f"\nrule-5 regime (audited LNA/MIXER/TIA identical on both arms, "
          f"unequal requested dB): {r5['n_cells']} pooled cells, "
          f"anchor-only {r5['baseline_mae_deg']:.4f} deg")
    for k in RUNGS:
        print(f"  {k:6s} MAE {r5[k]['mae_deg']:7.4f}  injects "
              f"{r5[k]['mean_injected_deg']:6.3f} deg on average, makes "
              f"{r5[k]['frac_made_worse']*100:5.1f}% of them worse")

    res["P1_stage_a_lofo"] = a_lofo
    res["P1_stage_a_loblk"] = a_loblk
    res["P1_stage_a_loro"] = a_loro
    res["P1_pooled_lofo"] = p_lofo
    res["P2_pooled_loband_AG_only"] = p_loband
    res["P3_rule5_regime_pooled"] = r5

    # ---------------- the decision, computed mechanically -------------------
    def p1_pick(table):
        elig = {k: v for k, v in table.items() if v["coverage"] >= 0.99}
        best = min(elig, key=lambda k: elig[k]["uneq_mae_deg"])
        near = [
            k for k in elig
            if elig[k]["uneq_mae_deg"] - elig[best]["uneq_mae_deg"] <= 0.1
        ]
        order = {"L30": 0, "L31": 1, "L26": 2}  # simpler first
        return sorted(near, key=lambda k: order[k])[0], best, near

    pick, best, near = p1_pick(a_lofo)
    base_band = p_loband[RUNGS[0]]["baseline_uneq_mae_deg"]
    p2 = {
        k: {
            "margin_vs_baseline_uneq_deg": base_band - p_loband[k]["uneq_mae_deg"],
            "coverage": p_loband[k]["coverage"],
            "passes": bool(
                (base_band - p_loband[k]["uneq_mae_deg"]) >= 0.5
                and p_loband[k]["coverage"] >= 0.95
            ),
        }
        for k in RUNGS
    }
    p3_l30_better_in_rule5 = r5["L30"]["mae_deg"] < r5["L26"]["mae_deg"]
    res["decision"] = {
        "P1_within_band_default": pick,
        "P1_lowest_error_rung": best,
        "P1_rungs_within_0p1_deg": near,
        "P2_cross_band": p2,
        "P2_any_rung_passes": any(v["passes"] for v in p2.values()),
        "P2_cross_band_default": (
            next((k for k in RUNGS if p2[k]["passes"]), "FAIL CLOSED (anchor only)")
        ),
        "P3_L30_better_in_rule5_regime": bool(p3_l30_better_in_rule5),
        "P3_ship_L30_as_second_rung": bool(
            any(p2[k]["passes"] for k in ("L30",)) or p3_l30_better_in_rule5
        ),
    }
    print("\n--- pre-registered decision, on A-G data only ---")
    print(json.dumps(res["decision"], indent=1))

    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
