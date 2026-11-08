"""Check A -- does the fitting path reproduce the published L31 rung?

Refit the EXISTING rungs on the EXISTING data (the A-G campaign's pooled
stages A + F + E_tx_0 + rate_pilot) and compare against the numbers published in
the shipped package README section 4.2 and in PROVENANCE.md section 2.

If this fails, nothing downstream is trustworthy and the report says so.
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
from spf.calibrations.gain_state_phase_model_v1.fit_from_extracted import (  # noqa: E402
    attach_anchor, load_stage,
)

EXTRACTED = Path(os.environ.get("SPF_EXTRACTED", "/tmp/spf_extracted"))

POOLED_STAGES = [
    "spectroscopy_20260730_full/A",
    "spectroscopy_20260730_full_r2/F",
    "spectroscopy_20260730_full/E_tx_0",
    "spectroscopy_20260730_full/rate_pilot",
]
STAGE_A = ["spectroscopy_20260730_full/A"]

# Published values, quoted from the committed package.
PUBLISHED = {
    # README section 4.2 pooled table / PROVENANCE.md section 2
    "pooled_LOFO": {"L26": 2.109, "L30": 2.985, "L31": 2.261},
    "pooled_LOFO_p95": {"L26": 6.686, "L30": 11.388, "L31": 7.363},
    "pooled_baseline_mae": 5.556,
    "pooled_baseline_p95": 17.861,
    "pooled_rows": 4641,
    "pooled_los": 119,
    # stage A, L26 (README section 4.2 / PROVENANCE section 2)
    "stage_a_rows": 3389,
    "stage_a_baseline_mae": 6.647,
    "stage_a_L26": {"LOEO": 2.08, "LOFO": 2.26, "LOBLK": 2.47, "LORO": 2.22},
    "stage_a_L26_uneq": {"LOEO": 2.60, "LOFO": 2.83},
    "stage_a_L31": {"LOEO": 2.45, "LOFO": 2.58, "LOBLK": 2.79, "LORO": 2.54},
    "stage_a_L31_uneq": {"LOEO": 3.06, "LOFO": 3.22},
    "stage_a_L30": {"LOEO": 3.49, "LOFO": 3.54, "LOBLK": 3.66, "LORO": 3.52},
}


def build(stages) -> dict:
    parts = [load_stage(EXTRACTED, s) for s in stages]
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = f["completed"] & f["qvalid"]
    f = {k: v[keep] for k, v in f.items()}
    f = attach_anchor(f, None)
    f = {k: v[f["has_anchor"]] for k, v in f.items()}
    return f


def main(out_path: str) -> None:
    out: dict = {"published": PUBLISHED, "measured": {}}

    pooled = build(POOLED_STAGES)
    stage_a = build(STAGE_A)

    out["measured"]["dataset"] = {
        "pooled_rows": int(len(pooled["D"])),
        "pooled_los": int(len(np.unique(pooled["lo_hz"]))),
        "pooled_baseline": FL.circ_stats(pooled["D"]),
        "stage_a_rows": int(len(stage_a["D"])),
        "stage_a_los": int(len(np.unique(stage_a["lo_hz"]))),
        "stage_a_baseline": FL.circ_stats(stage_a["D"]),
    }
    print("dataset:", json.dumps(out["measured"]["dataset"], indent=1))

    out["measured"]["fitter_selfcheck"] = FL.selfcheck(pooled)
    print("selfcheck:", out["measured"]["fitter_selfcheck"])

    rungs = {
        "L26": FL.L26_FIELDS,
        "L30": FL.L30_FIELDS,
        "L31": FL.L31_FIELDS,
    }
    ripples = {"L26": 2, "L30": 0, "L31": 2}

    # ---- pooled LOFO, the number l31_pooled_v1 is published against ---------
    out["measured"]["pooled_LOFO"] = {}
    for rung, fields in rungs.items():
        r = FL.evaluate(
            pooled, list(FL.folds_leave_one_out(pooled, "lo_hz")),
            static_fields=fields, n_ripples=ripples[rung], name=rung,
        )
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        out["measured"]["pooled_LOFO"][rung] = r
        print(f"pooled LOFO {rung}: mae={r['all_cells']['mae_deg']:.4f} "
              f"p95={r['all_cells']['p95_deg']:.4f} cov={r['coverage']:.3f}")

    # ---- stage A, all four splits ------------------------------------------
    out["measured"]["stage_a"] = {}
    splits = {
        "LOEO": list(FL.folds_leave_one_out(stage_a, "epoch")),
        "LOFO": list(FL.folds_leave_one_out(stage_a, "lo_hz")),
        "LOBLK": list(FL.folds_leave_freq_block_out(stage_a, 8)),
        "LORO": list(FL.folds_leave_one_out(stage_a, "serial")),
    }
    for rung, fields in rungs.items():
        out["measured"]["stage_a"][rung] = {}
        for sname, folds in splits.items():
            r = FL.evaluate(stage_a, folds, static_fields=fields,
                            n_ripples=ripples[rung], name=rung)
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            out["measured"]["stage_a"][rung][sname] = r
            print(f"stage-A {rung} {sname}: mae={r['all_cells']['mae_deg']:.4f} "
                  f"uneq={r['unequal_gain_cells']['mae_deg']:.4f} "
                  f"cov={r['coverage']:.3f}")

    # ---- verdict ------------------------------------------------------------
    checks = []

    def chk(label, measured, published, tol):
        ok = abs(measured - published) <= tol
        checks.append({"check": label, "measured": measured,
                       "published": published, "tolerance": tol, "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'} {label}: "
              f"{measured:.4f} vs {published} (tol {tol})")

    m = out["measured"]
    chk("pooled rows", m["dataset"]["pooled_rows"], PUBLISHED["pooled_rows"], 0)
    chk("pooled LOs", m["dataset"]["pooled_los"], PUBLISHED["pooled_los"], 0)
    chk("pooled baseline MAE", m["dataset"]["pooled_baseline"]["mae_deg"],
        PUBLISHED["pooled_baseline_mae"], 0.005)
    chk("pooled baseline P95", m["dataset"]["pooled_baseline"]["p95_deg"],
        PUBLISHED["pooled_baseline_p95"], 0.005)
    chk("stage-A rows", m["dataset"]["stage_a_rows"], PUBLISHED["stage_a_rows"], 0)
    chk("stage-A baseline MAE", m["dataset"]["stage_a_baseline"]["mae_deg"],
        PUBLISHED["stage_a_baseline_mae"], 0.005)
    for rung in ("L26", "L30", "L31"):
        chk(f"pooled LOFO {rung} MAE",
            m["pooled_LOFO"][rung]["all_cells"]["mae_deg"],
            PUBLISHED["pooled_LOFO"][rung], 0.005)
        chk(f"pooled LOFO {rung} P95",
            m["pooled_LOFO"][rung]["all_cells"]["p95_deg"],
            PUBLISHED["pooled_LOFO_p95"][rung], 0.005)
    for rung, pub in (("L26", PUBLISHED["stage_a_L26"]),
                      ("L30", PUBLISHED["stage_a_L30"]),
                      ("L31", PUBLISHED["stage_a_L31"])):
        for sname, val in pub.items():
            chk(f"stage-A {rung} {sname} MAE",
                m["stage_a"][rung][sname]["all_cells"]["mae_deg"], val, 0.005)
    for rung, pub in (("L26", PUBLISHED["stage_a_L26_uneq"]),
                      ("L31", PUBLISHED["stage_a_L31_uneq"])):
        for sname, val in pub.items():
            chk(f"stage-A {rung} {sname} unequal-gain MAE",
                m["stage_a"][rung][sname]["unequal_gain_cells"]["mae_deg"],
                val, 0.005)

    out["checks"] = checks
    out["n_checks"] = len(checks)
    out["n_pass"] = sum(c["pass"] for c in checks)
    out["verdict"] = "REPRODUCES" if out["n_pass"] == len(checks) else "MISMATCH"
    print(f"\n=== CHECK A: {out['verdict']} ({out['n_pass']}/{len(checks)}) ===")

    Path(out_path).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
