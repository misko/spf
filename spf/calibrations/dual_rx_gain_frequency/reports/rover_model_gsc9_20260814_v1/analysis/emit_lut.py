"""Emit deployable per-radio, per-carrier, per-arm gain LUTs from E-GSC9 session A.

SCHEMA NOTE, READ FIRST. This is NOT a GainStatePhaseModel coefficient file and
cannot be loaded by spf/calibrations/gain_state_phase_model_v1/model.py. That
class encodes D = H(s1) - H(s2) with ONE shared H over audited RF words. The
model E-GSC9 selects is arm-specific:

    D(f, radio, g1, g2) = d1[f, radio, g1] - d2[f, radio, g2]

with independent tables per arm. On the rover's own cells the shared-H form is
~10x worse (1.615 deg vs 0.149 on the clean radio at 5766 MHz), so the shape
difference is the whole point and cannot be flattened into the existing class.
A consumer for this file does not exist yet; that is stated in the report rather
than papered over.

Read-only inputs. Writes only the output path given on argv.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402
from gsc9_models import fit_additive  # noqa: E402

R18 = "1040007c4a94000211000b009186843ef2"
R17 = "104000bac4950008230026001b440a003a"
NAME = {R18: "R18", R17: "R17"}
DAMAGED = {R17}
CARRIERS = (5_766_000_000.0, 5_840_000_000.0)


def main(out_dir: str, ref: int = 62):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = json.load(open("rover_cell_weights.json"))
    scores = json.load(open(f"gsc9_models_{'rover62' if ref == 62 else 'best56'}.json"))
    sha = subprocess.check_output(
        ["git", "-C", "/home/mouse9911/gits/spf", "rev-parse", "HEAD"]).decode().strip()

    base = load_gsc.load()
    base = base.sel(base.stage == "GSC9A")
    manifest = {}
    for ser in (R18, R17):
        for lo in CARRIERS:
            m = (base.serial == ser) & (base.lo_hz == lo)
            if not m.sum():
                continue
            fa = FT.add_anchor(base.sel(m), ref=ref, per_epoch=True)
            pred = fit_additive(fa.g1, fa.g2, fa.D, ref)
            gains = sorted(set(fa.g1.tolist()) | set(fa.g2.tolist()))
            # recover the two arm tables by evaluating against the reference
            d1 = {int(g): float(np.degrees(pred(g, ref))) for g in gains}
            d2 = {int(g): float(np.degrees(-pred(ref, g))) for g in gains}
            key = f"{NAME[ser]}|{int(lo)}|{ref}"
            s = scores.get(key, {})
            w = weights.get(str(int(lo)), {})
            covered = sum(n for (c, n) in w.items()
                          if all(ref <= 62 for _ in [0])
                          and 26 <= int(c.split(",")[0]) <= 62
                          and 26 <= int(c.split(",")[1]) <= 62)
            total = sum(w.values())
            doc = {
                "model": "ARM_LUT_v1",
                "form": "D(f,radio,g1,g2) = d1[g1] - d2[g2], degrees",
                "radio": NAME[ser],
                "serial": ser,
                "lo_hz": float(lo),
                "anchor_gain_db": ref,
                "d1_deg": d1,
                "d2_deg": d2,
                "provenance": {
                    "source_capture": "e_gsc9_session_a_20260813_v1",
                    "raw_root": "/mnt/qnap01/mouse9911/spf/calibration_data/raw/"
                                "dual_rx_gain_frequency",
                    "grid": "[26,62]^2 = 1369 ordered cells, 5 epochs",
                    "n_frames": int(len(fa)),
                    "spf_git_sha": sha,
                    "held_out_weighted_mae_deg": s.get("ADD"),
                    "baseline_L00_weighted_mae_deg": s.get("L00"),
                    "full_cell_lookup_mae_deg": s.get("FULL"),
                    "mechanistic_rfword_mae_deg": s.get("MECH"),
                    "scoring": "leave-one-epoch-out within session A, error weighted "
                               "by rover cell usage over all 42 RX captures",
                    "rover_frame_coverage": covered / total if total else None,
                    "NOT_A_GainStatePhaseModel_FILE": (
                        "arm-specific; model.py encodes a single shared H and "
                        "cannot represent this. No consumer exists yet."),
                    "DEPLOYMENT_STATUS": (
                        "USABLE FOR THIS UNIT AND CARRIER ONLY. "
                        + ("This is the DAMAGED unit R17: E-GSC9 falsified additivity "
                           "on it (H2), its equal-gain step is -59 to -62 deg at the "
                           "40->41 dB transition, and its across-epoch drift failed "
                           "gate G9 at 4.468 deg. Do not use it to define absolute "
                           "coefficients. "
                           if ser in DAMAGED else
                           "Clean control unit. ")
                        + "Cross-radio transfer was measured at 1.16x, i.e. none, so "
                          "this table must not be applied to any other unit. It "
                          "carries no frequency model: a new carrier needs a new "
                          "capture. It is a residual to a MEASURED equal-gain anchor "
                          "at "
                        + str(ref)
                        + " dB, which the rover cannot yet measure in flight -- that "
                          "remains the deployment blocker."),
                },
            }
            fn = out / f"arm_lut_{NAME[ser].lower()}_{int(lo/1e6)}_anchor{ref}_20260814_v1.json"
            fn.write_text(json.dumps(doc, indent=1) + "\n")
            manifest[fn.name] = {
                "bytes": fn.stat().st_size,
                "held_out_mae_deg": s.get("ADD"),
                "vs_no_correction": (s.get("L00") / s.get("ADD")) if s.get("ADD") else None,
            }
            print(f"  {fn.name:<52} MAE={s.get('ADD'):.3f} deg  "
                  f"({s.get('L00')/s.get('ADD'):.1f}x vs no correction)")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nwrote {len(manifest)} LUTs + MANIFEST.json -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 62)
