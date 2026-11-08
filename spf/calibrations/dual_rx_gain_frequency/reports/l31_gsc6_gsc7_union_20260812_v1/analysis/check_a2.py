"""Check A, part 2 -- is the LOFO/LOBLK gap explained by the tau-search subsample?

Hypothesis: the SOURCE analysis' ``models.LadderModel`` grid-searches the ripple
delays on a random 1600-row subsample of each training fold
(``TAU_SEARCH_ROWS = 1600``, ``np.random.default_rng(0)`` per fold), while the
SHIPPED ``GainStatePhaseModel.fit`` searches on every training row. If that is
the whole difference, re-enabling the subsample must recover the published
numbers on exactly the splits that failed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitlib as FL  # noqa: E402
from check_a import PUBLISHED, POOLED_STAGES, STAGE_A, build  # noqa: E402


def main(out_path: str):
    pooled = build(POOLED_STAGES)
    stage_a = build(STAGE_A)
    out = {"hypothesis": "source pipeline subsamples the tau search to 1600 rows",
           "measured": {}}

    rungs = {"L26": (FL.L26_FIELDS, 2), "L30": (FL.L30_FIELDS, 0),
             "L31": (FL.L31_FIELDS, 2)}

    for mode, nsub in (("package_fit_all_rows", None),
                       ("source_pipeline_1600_row_tau_search", 1600)):
        out["measured"][mode] = {"pooled_LOFO": {}, "stage_a": {}}
        for rung, (fields, nrip) in rungs.items():
            r = FL.evaluate(pooled, list(FL.folds_leave_one_out(pooled, "lo_hz")),
                            static_fields=fields, n_ripples=nrip,
                            tau_search_rows=nsub)
            out["measured"][mode]["pooled_LOFO"][rung] = {
                "mae_deg": r["all_cells"]["mae_deg"],
                "p95_deg": r["all_cells"]["p95_deg"],
            }
            print(f"{mode:36s} pooled LOFO {rung}: "
                  f"mae={r['all_cells']['mae_deg']:.4f} "
                  f"p95={r['all_cells']['p95_deg']:.4f}")
        splits = {
            "LOEO": list(FL.folds_leave_one_out(stage_a, "epoch")),
            "LOFO": list(FL.folds_leave_one_out(stage_a, "lo_hz")),
            "LOBLK": list(FL.folds_leave_freq_block_out(stage_a, 8)),
            "LORO": list(FL.folds_leave_one_out(stage_a, "serial")),
        }
        for rung, (fields, nrip) in rungs.items():
            out["measured"][mode]["stage_a"][rung] = {}
            for sname, folds in splits.items():
                r = FL.evaluate(stage_a, folds, static_fields=fields,
                                n_ripples=nrip, tau_search_rows=nsub)
                out["measured"][mode]["stage_a"][rung][sname] = {
                    "mae_deg": r["all_cells"]["mae_deg"],
                    "uneq_mae_deg": r["unequal_gain_cells"]["mae_deg"],
                    "p95_deg": r["all_cells"]["p95_deg"],
                }
                print(f"{mode:36s} stage-A {rung} {sname}: "
                      f"mae={r['all_cells']['mae_deg']:.4f} "
                      f"uneq={r['unequal_gain_cells']['mae_deg']:.4f}")

    # ---- grade both modes against the published values ----------------------
    def grade(mode):
        m = out["measured"][mode]
        rows = []

        def chk(label, meas, pub, tol=0.005):
            rows.append({"check": label, "measured": meas, "published": pub,
                         "pass": abs(meas - pub) <= tol})

        for rung in ("L26", "L30", "L31"):
            chk(f"pooled LOFO {rung} MAE", m["pooled_LOFO"][rung]["mae_deg"],
                PUBLISHED["pooled_LOFO"][rung])
            chk(f"pooled LOFO {rung} P95", m["pooled_LOFO"][rung]["p95_deg"],
                PUBLISHED["pooled_LOFO_p95"][rung])
        for rung, pub in (("L26", PUBLISHED["stage_a_L26"]),
                          ("L30", PUBLISHED["stage_a_L30"]),
                          ("L31", PUBLISHED["stage_a_L31"])):
            for sname, val in pub.items():
                chk(f"stage-A {rung} {sname} MAE",
                    m["stage_a"][rung][sname]["mae_deg"], val)
        for rung, pub in (("L26", PUBLISHED["stage_a_L26_uneq"]),
                          ("L31", PUBLISHED["stage_a_L31_uneq"])):
            for sname, val in pub.items():
                chk(f"stage-A {rung} {sname} unequal-gain MAE",
                    m["stage_a"][rung][sname]["uneq_mae_deg"], val)
        return rows

    out["grading"] = {}
    for mode in out["measured"]:
        rows = grade(mode)
        out["grading"][mode] = {
            "checks": rows,
            "n": len(rows),
            "n_pass": sum(r["pass"] for r in rows),
        }
        print(f"\n{mode}: {out['grading'][mode]['n_pass']}/{len(rows)} match")
        for r in rows:
            if not r["pass"]:
                print(f"   MISS {r['check']}: {r['measured']:.4f} "
                      f"vs {r['published']}")

    Path(out_path).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
