"""Quantify the selection optimism in choosing L26 as the LOBLK argmin.

Rotate the contiguous frequency-block boundaries by offsets that played no part
in the selection, and compute paired per-fold margins against the runners-up.
"""

from __future__ import annotations

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from models import build_design

KEEP = {"L16", "L26", "L27", "L31", "L08"}


def rotated_blocks(f, n_blocks=8, offset=0):
    los = np.unique(f.lo_hz)
    order = np.roll(np.arange(len(los)), offset)
    for idx in np.array_split(order, n_blocks):
        block = set(los[idx].tolist())
        te = np.array([x in block for x in f.lo_hz])
        yield f"blk", ~te, te


def score(f, m, splitter):
    d = build_design(f, m.terms)
    preds = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    per_fold = []
    for _lbl, tr, te in splitter:
        tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
        if not len(tri) or not len(tei):
            continue
        p, sp, _t, _n = m.fit_eval(d, f.D, tri, tei)
        preds[tei] = p
        sup[tei] = sp
        fc = np.where(sp, p, 0.0)
        per_fold.append(S.cmae_deg(S.wrap(f.D[tei] - fc)))
    fc = np.where(sup, preds, 0.0)
    return S.cmae_deg(S.wrap(f.D - fc)), np.array(per_fold)


def main():
    f = FT.add_anchor(S.load_stages(["A"]), ref=26)
    ladder = [m for m in LD.make_ladder() if m.name.split()[0] in KEEP]

    print("=" * 88)
    print("LOBLK under rotated block boundaries (offsets never used for selection)")
    print("=" * 88)
    folds = {}
    for off in (0, 3, 7, 10):
        print(f"\n  offset {off:2d} LO indices:")
        for m in ladder:
            mae, pf = score(f, m, rotated_blocks(f, 8, off))
            folds.setdefault((m.name, off), pf)
            print(f"    {m.name:52s} {mae:6.3f}")

    print("\n" + "=" * 88)
    print("Paired per-fold margins at the selection partition (offset 0)")
    print("=" * 88)
    ref = [m for m in ladder if m.name.startswith("L26")][0].name
    a = folds[(ref, 0)]
    for m in ladder:
        if m.name == ref:
            continue
        b = folds[(m.name, 0)]
        d = b - a  # positive => L26 better
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  L26 vs {m.name.split()[0]:5s}: mean margin {d.mean():+6.3f} "
              f"± {sem:5.3f} deg (sem over {len(d)} folds), "
              f"L26 better in {int((d>0).sum())}/{len(d)} folds")


if __name__ == "__main__":
    main()
