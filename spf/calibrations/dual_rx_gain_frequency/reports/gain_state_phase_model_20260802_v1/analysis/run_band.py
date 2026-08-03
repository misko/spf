"""Decisive band-portability test.

Train on two AD9361 gain-table bands, predict the third. On stage A alone this
is unidentifiable (only 3 requested gains per band, so most LMT states are seen
in one band only). On the pooled A+F+E+pilot set every LMT state except
LNA=3 appears in at least two bands, so the hardware-state parameterisation
actually has a chance.

This is the strongest available test of the claim that phase is a function of
the AD9361 RF state rather than of the requested dB.
"""

from __future__ import annotations

import json

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from transfer import pooled

KEEP = {"L00", "L01", "L05", "L06", "L08", "L11", "L14", "L16", "L26",
        "L30", "L31", "L32", "L33", "L29", "L24"}


def main():
    ladder = [m for m in LD.make_ladder() if m.name.split()[0] in KEEP]
    fp = pooled(["A", "F", "E_tx_0", "rate_pilot"])
    out = {}

    print("=" * 100)
    print("POOLED leave-one-gain-table-band-out  (train on 2 bands, predict the 3rd)")
    print("=" * 100)
    for b in (0, 1, 2):
        n = int((fp.band == b).sum())
        print(f"  {FT.BANDS[b]:7s}: {n} rows, baseline MAE "
              f"{S.cmae_deg(fp.D[fp.band == b]):.3f} deg")
    print()
    out["pooled_LOBAND"] = LD.evaluate(fp, ladder,
                                       "LOBAND leave-one-gain-table-band-out")

    # per-band breakdown for the best few
    print("\n" + "=" * 100)
    print("Per-held-out-band detail")
    print("=" * 100)
    from models import build_design
    for m in ladder:
        if m.name.split()[0] not in {"L00", "L01", "L08", "L16", "L26", "L31",
                                     "L32"}:
            continue
        d = build_design(fp, m.terms)
        line = [f"  {m.name:52s}"]
        for b in (0, 1, 2):
            te = fp.band == b
            tr = ~te
            p, sp, _t, _n = m.fit_eval(d, fp.D, np.nonzero(tr)[0],
                                       np.nonzero(te)[0])
            fc = np.where(sp, p, 0.0)
            line.append(f"{FT.BANDS[b][:3]}={S.cmae_deg(S.wrap(fp.D[te]-fc)):6.3f}"
                        f"(cov{sp.mean():4.2f})")
        print(" ".join(line))

    with open("band_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
