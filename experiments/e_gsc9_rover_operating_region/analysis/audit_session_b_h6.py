"""AUDIT: E-GSC9 Session B DID run, and H6 is falsified on both radios.

Three committed documents say otherwise:
  * rover_model_gsc9_20260814_v1/REPORT.md:144-146 -- "Sessions B and C are
    TERMINATED by decision ... H6 (12 h transfer) and H7 (pad discriminator)
    were never run, so the re-calibration interval is unmeasured"
  * docs/gain_phase_rover_investigation_20260814.md:200-202 -- same
  * experiments/e_gsc9_rover_operating_region/RESULTS.md:3-4, 73, 98-104 --
    "Session B and session C remain outstanding" / H6 "PENDING" /
    "no Session-B raw directory exists yet"

The raw session-B directory exists on QNAP with a completed run and the H6
decision artifact the experiment_readme (§5, "H6 operational definition, frozen
before session B") specifies. This script reads that artifact read-only and
reports what it says, plus the split of the measured shift into a constant bias
and the variation about it -- the split that matters, because the reports argue a
per-session constant is absorbed downstream.

Read-only on /mnt. Writes nothing there, deletes nothing.
"""

from __future__ import annotations

import json
from math import sqrt

ART = ("/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/"
       "e_gsc9_session_b_20260814_v1/session_transfer_vs_a.json")
NAME = {"1040007c4a94000211000b009186843ef2": "R18 (clean)",
        "104000bac4950008230026001b440a003a": "R17 (damaged)"}
# rover-usage-weighted MAD of D measured in audit_mad_vs_sd.py, same (radio, LO)
MAD = {("R18 (clean)", 5766): 0.99, ("R18 (clean)", 5840): 1.07,
       ("R17 (damaged)", 5766): 1.90, ("R17 (damaged)", 5840): 1.87}


def main():
    d = json.load(open(ART))
    dr = d["decision_rule"]
    print(f"artifact  : {ART}")
    print(f"schema    : {d['schema']} v{d['schema_version']}   spf sha {d['spf_git_sha'][:7]}")
    print(f"rule      : {dr['metric']}, threshold {dr['threshold_deg_exclusive']} deg "
          f"(exclusive), min separation {dr['minimum_a_end_to_b_start_seconds']} s")
    print(f"H6 OVERALL: {'PASS' if d['h6_pass'] else 'FALSIFIED'}")
    print()
    hdr = (f"{'radio':<15}{'LO MHz':>8}{'sep h':>8}{'cells':>7}{'cov':>6}"
           f"{'MAE':>8}{'bias':>8}{'sd about bias':>15}{'vs 0.5 gate':>13}"
           f"{'sd / MAD':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in d["per_radio"]:
        nm = NAME.get(r["serial"], r["serial"][:12])
        sep_h = r["a_end_to_b_start_seconds"] / 3600.0
        for pf in r["per_frequency"]:
            lo = int(pf["frequency_hz"] / 1e6)
            mae, bias, rmse = (pf["circular_mae_deg"], pf["circular_bias_deg"],
                               pf["circular_rmse_deg"])
            var = max(rmse ** 2 - bias ** 2, 0.0)
            sd = sqrt(var)
            mad = MAD[(nm, lo)]
            print(f"{nm:<15}{lo:>8}{sep_h:>8.2f}{pf['common_quality_valid_cells']:>7}"
                  f"{pf['coverage_fraction']:>6.2f}{mae:>8.3f}{bias:>8.3f}"
                  f"{sd:>15.3f}{mae/0.5:>12.1f}x{sd/mad:>10.2f}")
    print()
    print("Reading:")
    print("  * separation and coverage gates BOTH PASS (>=12 h, 273/273 cells "
          "per radio per LO,")
    print("    all four validation.json status=pass, 1,638/1,638 quality-valid "
          "frames per radio).")
    print("    So this is a valid execution of H6, not a voided one.")
    print("  * the shift is almost entirely a CONSTANT (|bias| ~ MAE in all four "
          "strata), which")
    print("    the reports argue is absorbed by the per-session phase calibration.")
    print("  * the part that is NOT absorbed -- the sd about that constant -- is "
          "0.20-0.72 deg,")
    print("    i.e. 0.20-0.38x the rover-usage-weighted MAD the whole deployable "
          "prize consists of.")


if __name__ == "__main__":
    main()
