"""E-GSC3 -- decompose the retrospective 2.26 deg vs prospective 4.79 deg gap.

Rescoped after review. The two published figures differ in TWO ways at once:

  2.26 deg  within-session leave-one-frequency-out on stage A, ALL cells
  4.79 deg  cross-session transfer to a new capture, UNEQUAL-GAIN cells only

So the first job is to put every number in one convention, and the second is to
compare the prospective transfer against the campaign's own transfers rather
than against a refit cross-validation number.

Stage semantics (campaign config): B = 11 dB pad on the treated RX1 arm,
C = 30 cm jumper on the treated RX1 arm, D = harness removed and restored.
They form one ordered chain, so they are cumulative harness treatments, not
independent session draws. G is the only clean elapsed-time repeat of stage A
on an unchanged harness.

ANCHORED convention throughout; fail-closed; degrees.
"""

from __future__ import annotations

import json

import numpy as np

import gsc_common as G
import ladder as LD
import spflib as S
from models import build_design


def both_conventions(D, pred, sup, mask, uneq):
    s = G.score(D, pred, sup, mask, uneq)
    s["ratio_all"] = s["baseline_mae_deg"] / s["mae_deg"]
    s["ratio_uneq"] = s["baseline_uneq_mae_deg"] / s["uneq_mae_deg"]
    return s


def transfer(train_stages, test_stages, ref=26, label=""):
    f = G.load_anchored(sorted(set(train_stages) | set(test_stages)), ref=ref)
    tr = np.isin(f.stage.astype(str), train_stages)
    te = np.isin(f.stage.astype(str), test_stages)
    uneq = f.g1 != f.g2
    m = G.rung_model("L26")
    d = build_design(f, m.terms)
    tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
    p, sp, taus, ncol = m.fit_eval(d, f.D, tri, tei)
    pred = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    pred[tei] = p
    sup[tei] = sp
    out = both_conventions(f.D, pred, sup, te, uneq)
    out.update(
        {
            "label": label or f"{'+'.join(train_stages)} -> {'+'.join(test_stages)}",
            "tau_ns": (np.asarray(taus) * 1e9).round(4).tolist(),
            "n_columns": int(ncol),
            "n_train_rows": int(tr.sum()),
        }
    )
    return out


def within_session_cv(split_name, label):
    fa = G.load_anchored(["A"])
    uneq = fa.g1 != fa.g2
    m = G.rung_model("L26")
    d = build_design(fa, m.terms)
    pred = np.zeros(len(fa))
    sup = np.zeros(len(fa), dtype=bool)
    seen = np.zeros(len(fa), dtype=bool)
    taus = []
    for _lbl, tr, te in LD.SPLITS[split_name](fa):
        tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
        if not len(tri) or not len(tei):
            continue
        p, sp, tt, _n = m.fit_eval(d, fa.D, tri, tei)
        pred[tei] = p
        sup[tei] = sp
        seen[tei] = True
        if tt.size:
            taus.append(tt)
    out = both_conventions(fa.D, pred, sup, seen, uneq)
    out["label"] = label
    out["tau_ns"] = np.round(np.median(np.array(taus), axis=0) * 1e9, 4).tolist()
    return out


def main(out_path="gsc3_gap.json"):
    res = {
        "convention": "ANCHORED D = phi - measured equal-gain anchor; "
                      "fail-closed; MAE deg; ratio = baseline / model",
        "note": "every row reports BOTH the all-cell and the unequal-gain-only "
                "convention, so no comparison mixes them",
    }

    rows = []
    print("=== within-session cross-validation on stage A (REFIT each fold) ===")
    for split, lbl in (
        ("LOFO leave-one-frequency-out", "stage A LOFO (unseen frequency, same session)"),
        ("LOEO leave-one-epoch-out", "stage A LOEO (unseen epoch, same session)"),
        ("LOBLOCK leave-frequency-block-out", "stage A LOBLK (unseen ~690 MHz block)"),
    ):
        r = within_session_cv(split, lbl)
        rows.append(r)
        print(
            f"{r['label']:52s} base {r['baseline_mae_deg']:6.3f}/{r['baseline_uneq_mae_deg']:6.3f}"
            f"  L26 {r['mae_deg']:6.3f}/{r['uneq_mae_deg']:6.3f}"
            f"  ratio {r['ratio_all']:5.3f}/{r['ratio_uneq']:5.3f}  cov {r['coverage']:.3f}"
        )

    print("\n=== cross-session transfer, unchanged harness (TRANSFER, no refit at test) ===")
    for tr_s, te_s, lbl in (
        (["A"], ["G"], "A -> G  12 h later, hot, unchanged harness"),
        (["D"], ["G"], "D -> G  later pair, unchanged harness"),
    ):
        r = transfer(tr_s, te_s, label=lbl)
        rows.append(r)
        print(
            f"{r['label']:52s} base {r['baseline_mae_deg']:6.3f}/{r['baseline_uneq_mae_deg']:6.3f}"
            f"  L26 {r['mae_deg']:6.3f}/{r['uneq_mae_deg']:6.3f}"
            f"  ratio {r['ratio_all']:5.3f}/{r['ratio_uneq']:5.3f}  cov {r['coverage']:.3f}"
        )

    print("\n=== cross-session transfer WITH a deliberate harness treatment ===")
    print("    (B, C, D are one cumulative ordered chain -- not independent draws)")
    for tr_s, te_s, lbl in (
        (["A"], ["B"], "A -> B  11 dB pad added to treated RX1"),
        (["A"], ["C"], "A -> C  30 cm jumper added to treated RX1"),
        (["A"], ["D"], "A -> D  harness removed and restored"),
    ):
        r = transfer(tr_s, te_s, label=lbl)
        r["harness_treatment"] = True
        rows.append(r)
        print(
            f"{r['label']:52s} base {r['baseline_mae_deg']:6.3f}/{r['baseline_uneq_mae_deg']:6.3f}"
            f"  L26 {r['mae_deg']:6.3f}/{r['uneq_mae_deg']:6.3f}"
            f"  ratio {r['ratio_all']:5.3f}/{r['ratio_uneq']:5.3f}  cov {r['coverage']:.3f}"
        )

    # ---------------- the prospective transfer, on matched masks -------------
    print("\n=== prospective transfer: stage-A L26 -> 2026-08-07 dense capture ===")
    fa = G.load_anchored(["A"])
    fp = G.load_anchored(["P_dense"])
    cols = {k: np.concatenate([fa.cols[k], fp.cols[k]]) for k in fa.cols}
    f = S.Frames(cols)
    is_A = np.concatenate([np.ones(len(fa), bool), np.zeros(len(fp), bool)])
    uneq = f.g1 != f.g2
    m = G.rung_model("L26")
    d = build_design(f, m.terms)
    tri, tei = np.nonzero(is_A)[0], np.nonzero(~is_A)[0]
    p, sp, taus, ncol = m.fit_eval(d, f.D, tri, tei)
    pred = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    pred[tei] = p
    sup[tei] = sp

    prereg = np.isin((f.lo_hz / 1e6).round().astype(int), G.PREREG_10_MHZ)
    for lbl, mask in (
        ("prospective, all 113 LOs (PAIRED with stage-A CV)", ~is_A),
        ("prospective, 103 LOs excl. the E-CAL3 training comb", ~is_A & ~prereg),
        ("prospective, the 10 E-CAL3 comb LOs only", ~is_A & prereg),
    ):
        r = both_conventions(f.D, pred, sup, mask, uneq)
        r["label"] = lbl
        r["tau_ns"] = (np.asarray(taus) * 1e9).round(4).tolist()
        rows.append(r)
        print(
            f"{r['label']:52s} base {r['baseline_mae_deg']:6.3f}/{r['baseline_uneq_mae_deg']:6.3f}"
            f"  L26 {r['mae_deg']:6.3f}/{r['uneq_mae_deg']:6.3f}"
            f"  ratio {r['ratio_all']:5.3f}/{r['ratio_uneq']:5.3f}  cov {r['coverage']:.3f}"
        )

    # ---------------- (b) paired cell-level restriction ----------------------
    def cells(fr):
        return set(
            f"{s}|{int(lo)}|{a}|{b}"
            for s, lo, a, b in zip(fr.serial, fr.lo_hz, fr.g1, fr.g2)
        )

    ca, cp = cells(fa), cells(fp)
    common = ca & cp
    res["cell_overlap"] = {
        "stage_A_cells": len(ca),
        "prospective_cells": len(cp),
        "common_cells": len(common),
        "stage_A_only": len(ca - cp),
        "prospective_only": len(cp - ca),
    }
    print(
        f"\ncell overlap: stage A {len(ca)}, prospective {len(cp)}, "
        f"common {len(common)}, A-only {len(ca-cp)}, P-only {len(cp-ca)}"
    )

    key = np.array(
        [
            f"{s}|{int(lo)}|{a}|{b}"
            for s, lo, a, b in zip(f.serial, f.lo_hz, f.g1, f.g2)
        ],
        dtype=object,
    )
    in_common = np.array([k in common for k in key])
    r = both_conventions(f.D, pred, sup, (~is_A) & in_common, uneq)
    r["label"] = "prospective, restricted to cells stage A also covered"
    rows.append(r)
    print(
        f"{r['label']:52s} base {r['baseline_mae_deg']:6.3f}/{r['baseline_uneq_mae_deg']:6.3f}"
        f"  L26 {r['mae_deg']:6.3f}/{r['uneq_mae_deg']:6.3f}"
        f"  ratio {r['ratio_all']:5.3f}/{r['ratio_uneq']:5.3f}"
    )

    # ---------------- (c) the arithmetic of the published gap ----------------
    lofo = next(r for r in rows if r["label"].startswith("stage A LOFO"))
    prosp = next(r for r in rows if r["label"].startswith("prospective, all 113"))
    a2g = next(r for r in rows if r["label"].startswith("A -> G"))
    d2g = next(r for r in rows if r["label"].startswith("D -> G"))
    res["decomposition"] = {
        "published_retrospective_2p26": lofo["mae_deg"],
        "published_prospective_4p79": prosp["uneq_mae_deg"],
        "step1_convention_only": {
            "note": "same stage-A LOFO fold, restated on unequal-gain cells",
            "all_cells": lofo["mae_deg"],
            "unequal_gain": lofo["uneq_mae_deg"],
            "delta_deg": lofo["uneq_mae_deg"] - lofo["mae_deg"],
        },
        "step2_refit_to_transfer": {
            "note": "same convention (unequal-gain), same campaign, "
                    "within-session CV vs unchanged-harness transfer to G",
            "stage_A_LOFO_uneq": lofo["uneq_mae_deg"],
            "A_to_G_uneq": a2g["uneq_mae_deg"],
            "delta_deg": a2g["uneq_mae_deg"] - lofo["uneq_mae_deg"],
        },
        "step3_session_difficulty": {
            "note": "anchor-only baselines, unequal-gain",
            "stage_A": lofo["baseline_uneq_mae_deg"],
            "stage_G": a2g["baseline_uneq_mae_deg"],
            "prospective": prosp["baseline_uneq_mae_deg"],
            "prospective_vs_stage_A_pct": 100
            * (prosp["baseline_uneq_mae_deg"] / lofo["baseline_uneq_mae_deg"] - 1),
            "model_error_vs_stage_A_LOFO_pct": 100
            * (prosp["uneq_mae_deg"] / lofo["uneq_mae_deg"] - 1),
        },
        "ratios_uneq": {
            "stage_A_LOFO_refit": lofo["ratio_uneq"],
            "A_to_G_transfer": a2g["ratio_uneq"],
            "D_to_G_transfer": d2g["ratio_uneq"],
            "prospective_transfer": prosp["ratio_uneq"],
        },
    }
    res["rows"] = rows
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)

    dd = res["decomposition"]
    print("\n--- decomposition (unequal-gain convention unless stated) ---")
    print(
        f"  stage-A LOFO           all {lofo['mae_deg']:.4f}  uneq {lofo['uneq_mae_deg']:.4f}"
    )
    print(
        f"  A->G transfer          all {a2g['mae_deg']:.4f}  uneq {a2g['uneq_mae_deg']:.4f}"
    )
    print(
        f"  prospective transfer   all {prosp['mae_deg']:.4f}  uneq {prosp['uneq_mae_deg']:.4f}"
    )
    print(f"  ratios uneq: {dd['ratios_uneq']}")
    print(
        f"  baseline +{dd['step3_session_difficulty']['prospective_vs_stage_A_pct']:.1f}%, "
        f"model error +{dd['step3_session_difficulty']['model_error_vs_stage_A_LOFO_pct']:.1f}%"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
