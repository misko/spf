"""How fast does a gain-phase fit go stale?

The E-GSC8 'independent repeat' is 4.7 MINUTES after the primary, same cabling,
same thermal state. Any number from that pair is an upper bound on what a
deployed correction achieves, not an estimate of it. These three regimes span the
real separations available in the corpus:

    4.7 min   GSC8a  -> GSC8b     (E-GSC8's own repeat)
    3.2 h     GSC7usb2 -> GSC8a   (different session, same night)
    2 days    GSC6   -> GSC8b     (different session, different day)

All at 5766 MHz, the only carrier every campaign shares. 5840 MHz exists ONLY in
the two E-GSC8 sessions, so at that carrier the 4.7 min test is the only one that
can be run -- which is itself a finding about what the corpus can support.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import ladder as LD  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402
from carrier_eval import score  # noqa: E402

REGIMES = [
    ("4.7min_GSC8a->GSC8b", "GSC8a", "GSC8b", 5_766_000_000.0),
    ("3.2h_GSC7->GSC8a", "GSC7usb2", "GSC8a", 5_766_000_000.0),
    ("2day_GSC6->GSC8b", "GSC6", "GSC8b", 5_766_000_000.0),
    ("4.7min_GSC8a->GSC8b@5840", "GSC8a", "GSC8b", 5_840_000_000.0),
]


def main():
    ref = int(sys.argv[1]) if len(sys.argv) > 1 else 62
    f = FT.add_anchor(load_gsc.load(), ref=ref, per_epoch=True)
    y = f.D
    ladder = LD.make_ladder()
    designs = {m.name: LD.build_design(f, m.terms) for m in ladder}
    print(f"anchor ref={ref} dB", flush=True)

    out = {}
    for name, tr_stage, te_stage, lo in REGIMES:
        tr = (f.stage == tr_stage) & (f.lo_hz == lo)
        te = (f.stage == te_stage) & (f.lo_hz == lo)
        print(f"\n{name}: train={tr.sum()} test={te.sum()}", flush=True)
        rows = []
        for m in ladder:
            r = score(f, y, designs[m.name], m, tr, te, np.random.default_rng(0))
            if r:
                rows.append({"model": m.name, **r})
        rows.sort(key=lambda r: r["mae_deg"])
        out[name] = rows
        for r in rows[:5]:
            print(f"   {r['model'][:50]:<50} {r['mae_deg']:7.3f} deg "
                  f"(L00 {r['l00_mae_deg']:7.3f}, {r['ratio_vs_l00']:6.2f}x) "
                  f"cov={r['coverage']:.0%} p={r['params']}", flush=True)
        with open(f"epoch_eval_ref{ref}.json", "w") as fh:
            json.dump(out, fh, indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
