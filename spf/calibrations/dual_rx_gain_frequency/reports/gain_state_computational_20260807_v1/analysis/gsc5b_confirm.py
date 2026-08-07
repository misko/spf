"""E-GSC5, confirmation stage -- the prospective numbers, reported ONCE.

Nothing here fed the selection. gsc5_default.py fixed the pre-registered
criterion and made the choice on A-G data only; this script exists purely to
confirm or refute that choice against the 2026-08-07 capture.

Read the numbers as confirmation or refutation of an already-made decision, not
as a second selection round.
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
COEFFS = ("l26_stage_a_v1", "l26_pooled_v1", "l30_pooled_v1", "l31_pooled_v1")


def transfer_block(train_stages, test_stages, ref, label):
    f = G.load_anchored(sorted(set(train_stages) | set(test_stages)), ref=ref)
    tr = np.isin(f.stage.astype(str), train_stages)
    te = np.isin(f.stage.astype(str), test_stages)
    uneq = f.g1 != f.g2
    out = {}
    print(f"\n{label}  train n={tr.sum()}  test n={te.sum()}  "
          f"baseline all/uneq {S.cmae_deg(f.D[te]):.4f}/{S.cmae_deg(f.D[te&uneq]):.4f}")
    for name in RUNGS:
        m = G.rung_model(name)
        d = build_design(f, m.terms)
        p, sp, tt, nc = m.fit_eval(d, f.D, np.nonzero(tr)[0], np.nonzero(te)[0])
        pred = np.zeros(len(f))
        sup = np.zeros(len(f), dtype=bool)
        pred[np.nonzero(te)[0]] = p
        sup[np.nonzero(te)[0]] = sp
        st = G.score(f.D, pred, sup, te, uneq)
        st["params"] = int(nc)
        st["tau_ns"] = (np.asarray(tt) * 1e9).round(4).tolist()
        out[name] = st
        print(f"  {name:6s} p={st['params']:4d} cov={st['coverage']:6.3f} "
              f"MAE={st['mae_deg']:8.4f} uneq={st['uneq_mae_deg']:8.4f} "
              f"P95={st['p95_deg']:8.3f}")
    return out


def main(out_path="gsc5b_confirm.json"):
    res = {
        "status": "CONFIRMATION ONLY -- no number here entered the E-GSC5 selection",
        "convention": "ANCHORED, fail-closed, degrees",
    }

    # 1/2. train on A-G, test on the prospective dense capture
    res["A_stageA_to_prospective"] = transfer_block(
        ["A"], ["P_dense"], 26,
        "stage A -> 2026-08-07 prospective dense (113 LOs)",
    )
    res["A_pooled_to_prospective"] = transfer_block(
        ["A", "F", "E_tx_0", "rate_pilot"], ["P_dense"], None,
        "pooled A+F+E+rate_pilot -> prospective dense",
    )

    # 3. augmented leave-one-band-out, with the E-CAL2 state fill included
    stages = ["A", "F", "E_tx_0", "rate_pilot",
              "P_cal2_low", "P_cal2_middle", "P_cal2_high"]
    fa = FT.add_anchor(S.load_stages(stages), ref=None, per_epoch=True)
    uneq = fa.g1 != fa.g2
    print(f"\naugmented pooled rows={len(fa)} LOs={len(np.unique(fa.lo_hz))}")
    aug = {}
    for name in RUNGS:
        m = G.rung_model(name)
        d = build_design(fa, m.terms)
        pred = np.zeros(len(fa))
        sup = np.zeros(len(fa), dtype=bool)
        seen = np.zeros(len(fa), dtype=bool)
        per_band = {}
        for lbl, tr, te in LD.splits_leave_one_band_out(fa):
            tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
            p, sp, _tt, _nc = m.fit_eval(d, fa.D, tri, tei)
            pred[tei] = p
            sup[tei] = sp
            seen[tei] = True
            fc = np.where(sp, p, 0.0)
            per_band[lbl] = {
                "mae_deg": S.cmae_deg(S.wrap(fa.D[tei] - fc)),
                "coverage": float(sp.mean()),
                "baseline_mae_deg": S.cmae_deg(fa.D[tei]),
            }
        st = G.score(fa.D, pred, sup, seen, uneq)
        st["per_band"] = per_band
        aug[name] = st
        print(f"  {name:6s} cov={st['coverage']:6.4f} MAE={st['mae_deg']:8.4f} "
              f"uneq={st['uneq_mae_deg']:8.4f}  "
              + "  ".join(f"{k.split('=')[1]} {v['mae_deg']:.2f}"
                          for k, v in per_band.items()))
    aug["baseline_all_deg"] = S.cmae_deg(fa.D)
    aug["baseline_uneq_deg"] = S.cmae_deg(fa.D[uneq])
    print(f"  baseline all/uneq {aug['baseline_all_deg']:.4f}/"
          f"{aug['baseline_uneq_deg']:.4f}")
    res["augmented_leave_one_band_out"] = aug

    # 4. the actually-shipped coefficient files, on the prospective capture
    fp = G.load_anchored(["P_dense"])
    uneq_p = fp.g1 != fp.g2
    ship = {}
    print("\ncommitted coefficient sets on the prospective dense capture "
          "(113 LOs, identical mask):")
    for cname in COEFFS:
        for guard in (True, False):
            pred, sup = G.committed_predictions(
                fp, cname, apply_rf_state_guard=guard
            )
            st = G.score(fp.D, pred, sup, np.ones(len(fp), bool), uneq_p)
            key = f"{cname}{'' if guard else ' (rule-5 guard OFF)'}"
            ship[key] = st
            print(f"  {key:44s} cov={st['coverage']:6.4f} "
                  f"MAE={st['mae_deg']:8.4f} uneq={st['uneq_mae_deg']:8.4f} "
                  f"P95={st['p95_deg']:8.3f}")
    ship["baseline"] = {
        "all_deg": S.cmae_deg(fp.D),
        "uneq_deg": S.cmae_deg(fp.D[uneq_p]),
    }
    print(f"  {'anchor only (baseline)':44s} cov= 1.0000 "
          f"MAE={ship['baseline']['all_deg']:8.4f} "
          f"uneq={ship['baseline']['uneq_deg']:8.4f}")
    res["committed_coefficients_on_prospective"] = ship

    # 5. committed coefficients on the E-CAL2 state-fill stages
    fc2 = G.load_anchored(
        ["P_cal2_low", "P_cal2_middle", "P_cal2_high"], ref=26
    )
    uneq_c = fc2.g1 != fc2.g2
    c2 = {}
    print("\ncommitted coefficient sets on the E-CAL2 state-fill stages:")
    for cname in COEFFS:
        pred, sup = G.committed_predictions(fc2, cname)
        st = G.score(fc2.D, pred, sup, np.ones(len(fc2), bool), uneq_c)
        c2[cname] = st
        print(f"  {cname:44s} cov={st['coverage']:6.4f} MAE={st['mae_deg']:8.4f} "
              f"uneq={st['uneq_mae_deg']:8.4f}")
    c2["baseline"] = {
        "all_deg": S.cmae_deg(fc2.D),
        "uneq_deg": S.cmae_deg(fc2.D[uneq_c]),
    }
    print(f"  {'anchor only (baseline)':44s} cov= 1.0000 "
          f"MAE={c2['baseline']['all_deg']:8.4f} uneq={c2['baseline']['uneq_deg']:8.4f}")
    res["committed_coefficients_on_ecal2"] = c2

    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
