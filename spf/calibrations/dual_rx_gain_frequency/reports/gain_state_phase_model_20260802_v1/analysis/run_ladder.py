"""Run the ladder over stage A (and optionally other stages) and dump results."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import features as FT
import ladder as LD
import spflib as S


def prep(stages, ref=26):
    f = S.load_stages(stages)
    f = FT.add_anchor(f, ref=ref, per_epoch=True)
    return f


def main():
    stages = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A"]
    ref = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(LD.SPLITS)
    tag = sys.argv[4] if len(sys.argv) > 4 else "A"

    f = prep(stages, ref=ref)
    print(f"stages={stages} ref={ref}dB  rows={len(f)}  "
          f"radios={len(np.unique(f.serial))}  LOs={len(np.unique(f.lo_hz))}  "
          f"gain pairs={len(np.unique([f'{a},{b}' for a,b in zip(f.g1,f.g2)]))}")
    print(f"D (phase - measured equal-gain anchor): "
          f"MAE {S.cmae_deg(f.D):.3f} deg  P95 {S.cp95_deg(f.D):.3f}  "
          f"max {S.cmax_deg(f.D):.3f}")

    ladder = LD.make_ladder()
    out = {}
    for split in which:
        name = [k for k in LD.SPLITS if k.startswith(split)]
        if not name:
            continue
        split_name = name[0]
        print("\n" + "=" * 100)
        print(split_name)
        print("=" * 100)
        t0 = time.time()
        rows = LD.evaluate(f, ladder, split_name)
        out[split_name] = rows
        print(f"  [{time.time()-t0:.1f}s]")
    with open(f"ladder_results_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
