"""Consolidate every measurement in this run into one results.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402

RAW = ("/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency")


def load(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main(out="results.json"):
    base = load_gsc.load()
    census = {}
    for st in sorted(set(base.stage)):
        m = base.stage == st
        census[st] = {
            "rows": int(m.sum()),
            "los_mhz": sorted({float(v) / 1e6 for v in base.lo_hz[m]}),
            "radios": sorted({str(v)[:12] for v in base.serial[m]}),
            "t_start": float(base.timestamp[m].min()),
            "t_end": float(base.timestamp[m].max()),
        }
    anchored = {}
    for ref in (26, 52, 62):
        f = FT.add_anchor(base, ref=ref, per_epoch=True)
        anchored[ref] = {
            "rows": int(len(f)),
            "D_mae_deg": S.cmae_deg(f.D),
            "D_p95_deg": S.cp95_deg(f.D),
        }
    doc = {
        "run": "ladder_frames_gsc678_20260813_v1",
        "source": "FRAMES from the QNAP raw V7 stores, not fitted reconstructions",
        "raw_root": RAW,
        "read_only": True,
        "spf_git_sha": subprocess.check_output(
            ["git", "-C", "/home/mouse9911/gits/spf", "rev-parse", "HEAD"]
        ).decode().strip(),
        "census": census,
        "anchored": anchored,
        "antisymmetry_vs_anchor": load("antisym_vs_anchor.json"),
        "ladder_ref62": load("ladder_gsc678_ref62.json"),
        "ladder_ref26": load("ladder_gsc678.json"),
        "carrier_ref62": load("carrier_eval_ref62.json"),
        "carrier_ref26": load("carrier_eval_ref26.json"),
        "epoch_ref62": load("epoch_eval_ref62.json"),
    }
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, default=float)
    print(f"wrote {out} ({os.path.getsize(out)/1e3:.1f} kB)")
    for k, v in doc.items():
        if isinstance(v, dict) and k.startswith(("ladder", "carrier", "epoch")):
            print(f"  {k}: {len(v)} blocks")
        elif v is None:
            print(f"  {k}: MISSING")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json")
