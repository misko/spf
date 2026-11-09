"""Staleness measured ONLY on cells the fitting session actually supports.

``epoch_eval.py`` reports the deployable fail-closed number: unsupported cells fall
back to the anchor and are scored as such. That is the right number to ship, but it
conflates two different things when the two sessions used different gain grids.

E-GSC6 at 5766 MHz measured gains {-1, 8, 20, 22, 23, 25, 26, 27, 29, 30, 31, 32,
33, 40, 41, 45, 49, 50, 51, 52, 62} and never 53..61, so only {26, 52, 62} overlap
E-GSC8's schedule. The raw 2-day figure of 23.199 deg is therefore 82% correct
fail-closed refusal, NOT drift. This module scores the overlap only, which is the
comparison that isolates staleness -- and says so about its own small n.
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
    ("4.7min_GSC8a->GSC8b", "GSC8a", "GSC8b", 5_766_000_000.0),
    ("3.2h_GSC7->GSC8a", "GSC7usb2", "GSC8a", 5_766_000_000.0),
    ("2day_GSC6->GSC8b", "GSC6", "GSC8b", 5_766_000_000.0),
    ("4.7min@5840", "GSC8a", "GSC8b", 5_840_000_000.0),
]


def main(ref=62, rung="L04 arm d1,d2 per radio", out="staleness_supported.json"):
    f = FT.add_anchor(load_gsc.load(), ref=ref, per_epoch=True)
    y = f.D
    m = {x.name: x for x in LD.make_ladder()}[rung]
    d = LD.build_design(f, m.terms)
    res = {}
    print(f"{rung}, anchor {ref} dB, supported cells only\n")
    print(f"{'regime':<24}{'n_sup':>6}{'n_test':>7}{'cov':>6}{'MAE':>9}{'L00':>9}{'ratio':>8}")
    for name, tr_s, te_s, lo in REGIMES:
        tr = (f.stage == tr_s) & (f.lo_hz == lo)
        te = (f.stage == te_s) & (f.lo_hz == lo)
        tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
        p, sp, _, _ = m.fit_eval(d, y, tri, tei, rng=np.random.default_rng(0))
        uneq = f.g1[tei] != f.g2[tei]
        sel = uneq & sp
        if sel.sum() == 0:
            continue
        err = S.wrap(y[tei][sel] - p[sel])
        base = S.wrap(y[tei][sel])
        res[name] = {
            "n_supported": int(sel.sum()),
            "n_unequal": int(uneq.sum()),
            "coverage": float(sp[uneq].mean()),
            "mae_deg": S.cmae_deg(err),
            "rmse_deg": S.crmse_deg(err),
            "l00_mae_deg": S.cmae_deg(base),
            "ratio_vs_l00": S.cmae_deg(base) / S.cmae_deg(err),
        }
        r = res[name]
        print(f"{name:<24}{r['n_supported']:>6}{r['n_unequal']:>7}{r['coverage']:>5.0%}"
              f"{r['mae_deg']:>9.3f}{r['l00_mae_deg']:>9.3f}{r['ratio_vs_l00']:>7.1f}x")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
