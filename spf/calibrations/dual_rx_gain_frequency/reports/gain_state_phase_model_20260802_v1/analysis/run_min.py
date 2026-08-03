"""Focused evaluation of the MINIMAL mechanistic models (no baseband-LPF
categorical term), which are the only rung that can predict a requested gain
the model has never seen.

Splits: leave-one-requested-gain-out on the pooled set, plus leave-one-frequency
and leave-one-gain-table-band on stage A.
"""

from __future__ import annotations

import json

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from models import build_design
from transfer import leave_one_gain_out, pooled

KEEP = {"L00", "L01", "L05", "L06", "L16", "L26", "L30", "L31", "L32", "L33",
        "L34", "L24"}


def main():
    ladder = [m for m in LD.make_ladder() if m.name.split()[0] in KEEP]
    out = {}

    print("=" * 100)
    print("LEAVE-ONE-REQUESTED-GAIN-OUT, pooled A + F + E + rate_pilot")
    print("  every cell using that gain in either RX role is held out")
    print("=" * 100)
    fp = pooled(["A", "F", "E_tx_0", "rate_pilot"])
    print(f"rows={len(fp)}  LOs={len(np.unique(fp.lo_hz))}  "
          f"gains={sorted(set(fp.g1.tolist())|set(fp.g2.tolist()))}")
    print(f"baseline D: MAE {S.cmae_deg(fp.D):.3f}  P95 {S.cp95_deg(fp.D):.3f}\n")
    out["leave_one_gain_out_pooled"] = leave_one_gain_out(fp, ladder, "")

    for split in ("LOFO", "LOBAND", "LOEO"):
        name = [k for k in LD.SPLITS if k.startswith(split)][0]
        print("\n" + "=" * 100)
        print(f"{name}  (stage A)")
        print("=" * 100)
        fa = FT.add_anchor(S.load_stages(["A"]), ref=26)
        out[name] = LD.evaluate(fa, ladder, name)

    print("\n" + "=" * 100)
    print("POOLED leave-one-frequency-out (A+F+E+pilot, 119 LOs)")
    print("=" * 100)
    out["pooled_LOFO"] = LD.evaluate(fp, ladder, "LOFO leave-one-frequency-out")

    with open("min_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
