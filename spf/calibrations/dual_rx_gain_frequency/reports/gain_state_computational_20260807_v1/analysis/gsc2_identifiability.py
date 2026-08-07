"""E-GSC2 -- the identifiability curve: how many LOs does L26 actually need?

Refit L26 from N of the 113 dense stage-A LOs and score on every held-out LO,
in two variants:

  (a) delays FREE   -- grid-searched on the training fold only
  (b) delays FROZEN -- fixed at the committed fleet values 2.56 / 0.92 ns

Each (N, variant) is repeated over the same set of random LO subsets, so the
two variants are paired. Every fit is also scored on the 2026-08-07 prospective
dense capture, restricted to the same held-out LOs -- a genuinely external
session that no fit here has ever seen.

All errors are ANCHORED (D = phi - measured equal-gain anchor) and fail closed
to the anchor on unsupported cells.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import gsc_common as G
import spflib as S
from models import build_design

N_LIST = (6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
N_SUBSETS = 24
SEED = 20260807


def main(out_path="gsc2_identifiability.json"):
    fa = G.load_anchored(["A"])
    fp = G.load_anchored(["P_dense"])

    los = np.unique(fa.lo_hz)
    assert np.array_equal(los, np.unique(fp.lo_hz)), "LO grids must match"
    n_lo = len(los)

    # one combined table so the design has identical columns for both sessions
    cols = {}
    for k in fa.cols:
        cols[k] = np.concatenate([fa.cols[k], fp.cols[k]])
    f = S.Frames(cols)
    is_A = np.concatenate([np.ones(len(fa), bool), np.zeros(len(fp), bool)])
    is_P = ~is_A
    uneq = f.g1 != f.g2

    designs = {
        "free": build_design(f, G.rung_model("L26").terms),
        "frozen": build_design(f, G.rung_model("L26", G.TAU_FLEET).terms),
    }
    models = {
        "free": G.rung_model("L26"),
        "frozen": G.rung_model("L26", G.TAU_FLEET),
    }

    rng = np.random.default_rng(SEED)
    subsets = {}
    for N in N_LIST:
        subsets[N] = [
            np.sort(rng.choice(n_lo, N, replace=False)) for _ in range(N_SUBSETS)
        ]
    # the deterministic pre-registered uniform 10-LO comb, as a labelled extra
    prereg_idx = np.array(
        [int(np.argmin(np.abs(los / 1e6 - m))) for m in G.PREREG_10_MHZ]
    )

    records = []
    t0 = time.time()

    def one_fit(variant, train_lo_idx, label, N):
        train_lo = set(los[train_lo_idx].tolist())
        in_train_lo = np.array([x in train_lo for x in f.lo_hz])
        tr = is_A & in_train_lo
        te = ~in_train_lo  # held-out LOs, BOTH sessions
        d = designs[variant]
        m = models[variant]
        pred, sup, taus, ncol = np.zeros(len(f)), np.zeros(len(f), bool), None, 0
        tri = np.nonzero(tr)[0]
        tei = np.nonzero(te)[0]
        p, sp, tt, ncol = m.fit_eval(d, f.D, tri, tei)
        pred[tei] = p
        sup[tei] = sp
        taus = (np.asarray(tt) * 1e9).round(4).tolist()
        rec = {
            "N": N,
            "variant": variant,
            "subset": label,
            "train_lo_mhz": sorted((los[train_lo_idx] / 1e6).round().astype(int).tolist()),
            "n_train_rows": int(tr.sum()),
            "tau_ns": taus,
            "n_columns": ncol,
            "in_campaign": G.score(f.D, pred, sup, te & is_A, uneq),
            "prospective": G.score(f.D, pred, sup, te & is_P, uneq),
        }
        records.append(rec)
        return rec

    for N in N_LIST:
        for variant in ("free", "frozen"):
            for j, idx in enumerate(subsets[N]):
                one_fit(variant, idx, f"rand{j:02d}", N)
        done = [r for r in records if r["N"] == N]
        fr = [r for r in done if r["variant"] == "free"]
        fz = [r for r in done if r["variant"] == "frozen"]
        print(
            f"N={N:3d}  free  uneq med {np.median([r['in_campaign']['uneq_mae_deg'] for r in fr]):7.3f}"
            f"  prosp {np.median([r['prospective']['uneq_mae_deg'] for r in fr]):7.3f}"
            f" | frozen uneq med {np.median([r['in_campaign']['uneq_mae_deg'] for r in fz]):7.3f}"
            f"  prosp {np.median([r['prospective']['uneq_mae_deg'] for r in fz]):7.3f}"
            f"   [{time.time()-t0:6.1f}s]",
            flush=True,
        )

    for variant in ("free", "frozen"):
        r = one_fit(variant, prereg_idx, "prereg10", 10)
        print(
            f"prereg-10 {variant:6s} tau={r['tau_ns']} "
            f"in-campaign uneq {r['in_campaign']['uneq_mae_deg']:.3f} "
            f"(base {r['in_campaign']['baseline_uneq_mae_deg']:.3f})  "
            f"prospective uneq {r['prospective']['uneq_mae_deg']:.3f} "
            f"(base {r['prospective']['baseline_uneq_mae_deg']:.3f})"
        )

    # ---- summary: median / IQR / win-rate against the anchor-only baseline ---
    summary = []
    for N in N_LIST:
        for variant in ("free", "frozen"):
            rr = [
                r for r in records
                if r["N"] == N and r["variant"] == variant
                and r["subset"].startswith("rand")
            ]
            row = {"N": N, "variant": variant, "n_subsets": len(rr)}
            for key in ("in_campaign", "prospective"):
                e = np.array([r[key]["uneq_mae_deg"] for r in rr])
                b = np.array([r[key]["baseline_uneq_mae_deg"] for r in rr])
                ea = np.array([r[key]["mae_deg"] for r in rr])
                ba = np.array([r[key]["baseline_mae_deg"] for r in rr])
                cov = np.array([r[key]["coverage_uneq"] for r in rr])
                row[key] = {
                    "uneq_mae_median": float(np.median(e)),
                    "uneq_mae_q25": float(np.percentile(e, 25)),
                    "uneq_mae_q75": float(np.percentile(e, 75)),
                    "uneq_mae_min": float(e.min()),
                    "uneq_mae_max": float(e.max()),
                    "baseline_uneq_median": float(np.median(b)),
                    "all_mae_median": float(np.median(ea)),
                    "baseline_all_median": float(np.median(ba)),
                    "coverage_min": float(cov.min()),
                    "coverage_median": float(np.median(cov)),
                    "beats_anchor_frac": float(np.mean(e < b)),
                    "margin_vs_anchor_median": float(np.median(b - e)),
                }
            taus = np.array([r["tau_ns"] for r in rr])
            row["tau1_ns"] = {
                "median": float(np.median(taus[:, 0])),
                "q25": float(np.percentile(taus[:, 0], 25)),
                "q75": float(np.percentile(taus[:, 0], 75)),
            }
            row["tau2_ns"] = {
                "median": float(np.median(taus[:, 1])),
                "q25": float(np.percentile(taus[:, 1], 25)),
                "q75": float(np.percentile(taus[:, 1], 75)),
            }
            summary.append(row)

    def nstar(variant, key):
        for N in N_LIST:
            row = next(
                r for r in summary if r["N"] == N and r["variant"] == variant
            )
            if row[key]["beats_anchor_frac"] >= 0.90:
                return N
        return None

    result = {
        "convention": "ANCHORED D = phi - measured equal-gain anchor; "
                      "fail-closed to anchor; MAE in degrees",
        "n_los_total": int(n_lo),
        "n_subsets_per_N": N_SUBSETS,
        "seed": SEED,
        "tau_fleet_ns": [t * 1e9 for t in G.TAU_FLEET],
        "N_star": {
            "free_in_campaign": nstar("free", "in_campaign"),
            "free_prospective": nstar("free", "prospective"),
            "frozen_in_campaign": nstar("frozen", "in_campaign"),
            "frozen_prospective": nstar("frozen", "prospective"),
        },
        "summary": summary,
        "fits": records,
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("\nN* (free delays, >=90% of subsets beat anchor-only):", result["N_star"])
    print(f"wrote {out_path}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main(*sys.argv[1:])
