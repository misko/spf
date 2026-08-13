"""Merge every measurement in this run into one machine-readable results.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def parse_source_ladder(log_path: Path, section: str) -> dict:
    """Pull the source pipeline's own printed rung table out of its log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    blocks = text.split("=" * 100)
    want = None
    for i, b in enumerate(blocks):
        if section in b and i + 1 < len(blocks):
            want = blocks[i + 1]
            break
    if want is None:
        return {}
    out = {}
    pat = re.compile(
        r"^\s+(L\d\d)\b.*?p=\s*(\d+)\s+cov=([\d.]+)\s+MAE=\s*([\d.]+)\s+"
        r"uneqMAE=\s*([\d.]+)\s+P95=\s*([\d.]+)\s+max=\s*([\d.]+)")
    for line in want.splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = {
                "params": int(m.group(2)), "coverage": float(m.group(3)),
                "mae_deg": float(m.group(4)), "uneq_mae_deg": float(m.group(5)),
                "p95_deg": float(m.group(6)), "max_deg": float(m.group(7)),
            }
    return out


def main(out_path: str):
    srcpipe = HERE.parent / "srcpipe"
    res = {
        "run": "l31_gsc6_gsc7_union_20260812_v1",
        "date_utc": "2026-08-12",
        "check_A": {
            "question": "does the fitting path reproduce l31_pooled_v1's "
                        "published holdout numbers?",
            "reimplementation_selfcheck": load("check_a.json")["measured"][
                "fitter_selfcheck"],
            "dataset_reproduction": load("check_a.json")["measured"]["dataset"],
            "published": load("check_a2.json")["measured"] and
                         load("check_a.json")["published"],
            "two_fitting_rules": load("check_a2.json")["measured"],
            "grading": load("check_a2.json")["grading"],
            "source_pipeline_rerun": {
                "what": "the SOURCE analysis' own ladder.py / models.py, copied "
                        "unmodified to a scratch directory and run against this "
                        "run's own read-only re-extraction of the campaign stores",
                "stage_A_LOFO": parse_source_ladder(
                    srcpipe / "srccheck.log", "LOFO leave-one-frequency-out"),
                "stage_A_LOEO": parse_source_ladder(
                    srcpipe / "srccheck.log", "LOEO leave-one-epoch-out"),
                "stage_A_LOBLOCK": parse_source_ladder(
                    srcpipe / "srccheck.log", "LOBLOCK leave-frequency-block-out"),
                "stage_A_LORO": parse_source_ladder(
                    srcpipe / "srccheck.log", "LORO leave-one-radio-out"),
                "pooled_LOFO": parse_source_ladder(
                    srcpipe / "runmin.log", "POOLED leave-one-frequency-out"),
            },
        },
        "check_B_new_rung": load("fit_new.json"),
        "check_B_sensitivity": load("sensitivity.json"),
        "check_C_rover_coverage": load("rover_coverage_new.json"),
        "check_C_rover_anchor": load("rover_anchor.json"),
        "coefficients": load("coeffs/emit_manifest.json"),
    }
    Path(out_path).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")
    sp = res["check_A"]["source_pipeline_rerun"]
    for k, v in sp.items():
        if isinstance(v, dict) and v:
            print(k, {r: round(x["mae_deg"], 4)
                      for r, x in v.items() if r in ("L26", "L30", "L31")})


if __name__ == "__main__":
    main(sys.argv[1])
