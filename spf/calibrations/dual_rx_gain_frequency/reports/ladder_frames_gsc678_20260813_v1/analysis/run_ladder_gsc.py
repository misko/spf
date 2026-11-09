"""Run the published ladder over the frame-level E-GSC6+7+8 union."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import ladder as LD  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402


def main():
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else list(LD.SPLITS)
    tag = sys.argv[2] if len(sys.argv) > 2 else "gsc678"
    # Anchor gain. 26 dB is the published convention; the rover parks at 62 dB
    # and its equal-gain frames are 83%/96% there, so 62 is the deployment-
    # relevant anchor and 52 is the compromise. Measured on frames: the choice
    # moves R17's antisymmetry violation from 56-62 deg to under 1 deg.
    ref = int(sys.argv[3]) if len(sys.argv) > 3 else 26

    f = FT.add_anchor(load_gsc.load(), ref=ref, per_epoch=True)
    print(f"anchor ref={ref} dB", flush=True)
    print(f"rows={len(f)} radios={len(np.unique(f.serial))} LOs={len(np.unique(f.lo_hz))} "
          f"epochs={len(np.unique(f.epoch))}", flush=True)

    ladder = LD.make_ladder()
    print(f"{len(ladder)} rungs", flush=True)
    out = {}
    for split in which:
        name = [k for k in LD.SPLITS if k.startswith(split)]
        if not name:
            print(f"  (no split matching {split})")
            continue
        sn = name[0]
        t0 = time.time()
        print("\n" + "=" * 90 + f"\n{sn}\n" + "=" * 90, flush=True)
        out[sn] = LD.evaluate(f, ladder, sn)
        print(f"  [{time.time()-t0:.1f}s]", flush=True)
        with open(f"ladder_{tag}.json", "w") as fh:
            json.dump(out, fh, indent=1)   # checkpoint after every split
    print("done")


if __name__ == "__main__":
    main()
