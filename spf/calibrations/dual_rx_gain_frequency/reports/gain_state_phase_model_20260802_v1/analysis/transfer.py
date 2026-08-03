"""Transfer tests: train on one stage / gain set, predict a genuinely different one.

T1  train A            -> test D, G, B, C   (later session, hot repeat, and
                                             two deliberate harness treatments)
T2  train A            -> test F, E         (gain ranges A never visited)
T3  train F+E          -> test A
T4  leave-one-gain-out on the pooled set    (can the model predict an unseen
                                             requested gain?)
T5  leave-one-LNA-state-out                 (can it predict an unseen RF state?)

All models predict D = phase - measured equal-gain anchor.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from models import build_design


def pooled(stages, ref=None):
    f = S.load_stages(stages)
    return FT.add_anchor(f, ref=ref, per_epoch=True)


def run_pair(train_stages, test_stages, ladder, label, ref=None):
    f = pooled(sorted(set(train_stages) | set(test_stages)), ref=ref)
    tr = np.isin(f.stage.astype(str), train_stages)
    te = np.isin(f.stage.astype(str), test_stages)
    print(f"\n{label}: train={train_stages} n={tr.sum()}  "
          f"test={test_stages} n={te.sum()}")
    print(f"   baseline (no gain correction) on the test set: "
          f"MAE {S.cmae_deg(f.D[te]):.3f}  P95 {S.cp95_deg(f.D[te]):.3f}")
    rows = []
    for m in ladder:
        d = build_design(f, m.terms)
        p, sp, tt, nc = m.fit_eval(d, f.D, np.nonzero(tr)[0], np.nonzero(te)[0])
        err = S.wrap(f.D[te] - p)
        st = S.err_stats(err)
        rows.append({"model": m.name, "params": nc, "coverage": float(sp.mean()),
                     **st, "sup_mae_deg": S.cmae_deg(err[sp]) if sp.any() else None,
                     "tau_ns": np.round(tt * 1e9, 3).tolist() if tt.size else []})
        r = rows[-1]
        print(f"   {r['model']:52s} p={r['params']:5d} cov={r['coverage']:5.3f} "
              f"MAE={r['mae_deg']:7.3f} P95={r['p95_deg']:7.3f} "
              f"supMAE={(r['sup_mae_deg'] or float('nan')):7.3f}")
    return rows


STAGE_REF = {"A": 26, "B": 26, "C": 26, "D": 26, "G": 26, "rate_pilot": 26,
             "F": 5, "F_neg": 5, "E_tx_0": 33, "E_tx_n7": 33, "E_tx_n14": 33,
             "E_tx_n21": 33, "E_tx_n28": 33, "E_tx_n80": 33}


def probe_gain(f):
    """The non-reference gain of a cross-design cell. Holding out a gain by
    (g1==g or g2==g) is degenerate here: every stage-A pair contains 26, so
    that split would delete the whole stage."""
    ref = np.array([STAGE_REF[s] for s in f.stage.astype(str)])
    return np.where(f.g2 == ref, f.g1, f.g2), ref


def leave_one_gain_out(f, ladder, label):
    """Hold out every cell whose PROBE gain equals g.

    Reports both the supported-only error and the fail-closed error, where an
    unsupported cell falls back to the anchor (prediction 0) as the repo's
    prediction contract requires. A raw extrapolated prediction on an
    unsupported cell is never a legitimate deployment number.
    """
    print(f"\n{label}")
    probe, ref = probe_gain(f)
    gains = sorted(set(probe.tolist()))
    out = {}
    print(f"   {'model':52s} {'p':>5s} {'cov':>5s} {'failclosedMAE':>13s} "
          f"{'supMAE':>7s} {'supP95':>7s}")
    for m in ladder:
        d = build_design(f, m.terms)
        preds = np.zeros(len(f))
        sup = np.zeros(len(f), dtype=bool)
        seen = np.zeros(len(f), dtype=bool)
        npar = []
        for g in gains:
            te = probe == g
            tr = ~te
            if tr.sum() == 0 or te.sum() == 0:
                continue
            p, sp, _tt, nc = m.fit_eval(d, f.D, np.nonzero(tr)[0], np.nonzero(te)[0])
            idx = np.nonzero(te)[0]
            preds[idx] = p
            sup[idx] = sp
            seen[idx] = True
            npar.append(nc)
        fail_closed = np.where(sup, preds, 0.0)
        err_fc = S.wrap(f.D[seen] - fail_closed[seen])
        err_sup = S.wrap(f.D[seen & sup] - preds[seen & sup])
        out[m.name] = {
            "params": int(np.median(npar)) if npar else 0,
            "coverage": float(sup[seen].mean()),
            **S.err_stats(err_fc, prefix="failclosed_"),
            **S.err_stats(err_sup, prefix="sup_"),
        }
        r = out[m.name]
        print(f"   {m.name:52s} {r['params']:5d} {r['coverage']:5.3f} "
              f"{r['failclosed_mae_deg']:13.3f} "
              f"{r.get('sup_mae_deg', float('nan')):7.3f} "
              f"{r.get('sup_p95_deg', float('nan')):7.3f}")
    return out


def main():
    ladder = [m for m in LD.make_ladder() if m.name.split()[0] in {
        "L00", "L01", "L05", "L06", "L08", "L09", "L11", "L14", "L16",
        "L18", "L19", "L25", "L26", "L27", "L28", "L29", "L23", "L24"}]
    results = {}

    print("=" * 100)
    print("T1  TEMPORAL AND HARNESS TRANSFER  (train on stage A only)")
    print("=" * 100)
    for tests, lbl in ((["G"], "A -> G  (12 h later, hot, same harness)"),
                       (["D"], "A -> D  (harness removed and restored)"),
                       (["B"], "A -> B  (11 dB pad on treated RX1)"),
                       (["C"], "A -> C  (30 cm jumper on treated RX1)")):
        results[lbl] = run_pair(["A"], tests, ladder, lbl, ref=26)

    print("\n" + "=" * 100)
    print("T2/T3  GAIN-RANGE TRANSFER")
    print("=" * 100)
    results["A -> F"] = run_pair(["A"], ["F"], ladder, "A -> F  (low/TIA gains)")
    results["A -> E"] = run_pair(["A"], ["E_tx_0"], ladder, "A -> E  (27-40 dB)")
    results["F+E -> A"] = run_pair(["F", "E_tx_0"], ["A"], ladder, "F+E -> A")

    print("\n" + "=" * 100)
    print("T4  LEAVE-ONE-GAIN-OUT on the pooled A+F+E+pilot set")
    print("=" * 100)
    fp = pooled(["A", "F", "E_tx_0", "rate_pilot"])
    print(f"pooled rows={len(fp)}  LOs={len(np.unique(fp.lo_hz))}  "
          f"gains={sorted(set(fp.g1.tolist())|set(fp.g2.tolist()))}")
    print(f"baseline D: MAE {S.cmae_deg(fp.D):.3f}  P95 {S.cp95_deg(fp.D):.3f}")
    results["leave-one-gain-out"] = leave_one_gain_out(
        fp, ladder, "leave-one-requested-gain-out (pooled)")

    with open("transfer_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
