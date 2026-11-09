"""Score every rung with the support rule corrected -- section 4's headline table.

The shipped rule (``models.LadderModel.fit_eval``) refuses a test row when it
REFERENCES a parameter training could not estimate::

    needed = design.I[test_idx] > 0

In a signed design ``I`` and ``S`` differ: a gain-table level present identically on
both arms cancels, contributing ``S = 0`` while still setting ``I > 0``. On a
single-band high-gain fit that is true of exactly one column, and it is zero on the
TEST rows as well -- so it cannot change any prediction, yet it fails 100% of rows
closed. Every mechanistic rung then scores exactly the L00 baseline, which is what
made this worth chasing.

Corrected rule: refuse a row only when a column that ACTUALLY CONTRIBUTES to it
(``|S| > 0`` on that row) was unestimable in training. This is the version
FAVOURABLE to the shipped mechanistic family, and section 4 uses it throughout so
the comparison cannot be accused of scoring them under a broken rule.

This module does NOT patch the shipped analysis package; it reimplements the two
lines locally and leaves ``models.py`` untouched.
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

REGIMES = [
    ("4.7min@5766", "GSC8a", "GSC8b", 5_766_000_000.0),
    ("3.2h@5766", "GSC7usb2", "GSC8a", 5_766_000_000.0),
    ("2day@5766", "GSC6", "GSC8b", 5_766_000_000.0),
    ("4.7min@5840", "GSC8a", "GSC8b", 5_840_000_000.0),
]


def main(ref=62, out="corrected_support.json"):
    f = FT.add_anchor(load_gsc.load(), ref=ref, per_epoch=True)
    y = f.D
    out_doc = {}
    for m in LD.make_ladder():
        d = LD.build_design(f, m.terms)
        rec = {}
        for lbl, tr_s, te_s, lo in REGIMES:
            tr = (f.stage == tr_s) & (f.lo_hz == lo)
            te = (f.stage == te_s) & (f.lo_hz == lo)
            tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
            if not len(tri) or not len(tei):
                continue
            p, _shipped_sp, _, _ = m.fit_eval(d, y, tri, tei, rng=np.random.default_rng(0))
            active = np.any(np.abs(d.S[tri]) > 0, axis=0)
            contributes = np.abs(d.S[tei]) > 0            # <-- S, not I
            sp = ~np.any(contributes & ~active[None, :], axis=1)
            uneq = f.g1[tei] != f.g2[tei]
            err = S.wrap(y[tei][uneq] - np.where(sp, p, 0.0)[uneq])
            base = S.wrap(y[tei][uneq])
            rec[lbl] = {
                "mae_deg": S.cmae_deg(err),
                "l00": S.cmae_deg(base),
                "cov": float(sp[uneq].mean()),
            }
        out_doc[m.name] = rec
    with open(out, "w") as fh:
        json.dump(out_doc, fh, indent=1)
    arm = ("L04", "L10", "L20", "L24")
    rows = sorted(((n, r) for n, r in out_doc.items() if "4.7min@5766" in r),
                  key=lambda t: t[1]["4.7min@5766"]["mae_deg"])
    print(f"{'rung':<46}{'shape':<14}" + "".join(f"{r[0]:>14}" for r in REGIMES[:1] + REGIMES[3:]))
    for n, r in rows:
        shape = "arm-specific" if n.split(" ")[0] in arm else "symmetric"
        print(f"{n[:46]:<46}{shape:<14}"
              + "".join(f"{r[k]['mae_deg']:>8.2f} ({r[k]['l00']/r[k]['mae_deg']:4.1f}x)"
                        for k in ("4.7min@5766", "4.7min@5840") if k in r))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
