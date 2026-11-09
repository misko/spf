"""Expected performance on a radio the model was NOT fitted on.

This is the number that governs the rover, whose radios are uncalibrated: the
committed tables are for R17/R18 and the rover flies different units.

Four transfer policies, each fitted on one radio and applied to the other, scored
on the rover's own cells and weighted by rover cell usage:

  FULL_XFER    take the donor's mixer AND LNA tables wholesale
  MIXER_XFER   take the donor's mixer table; refit the recipient's LNA only
  LNA_XFER     take the donor's LNA table; refit the recipient's mixer only
  SAME_RADIO   fitted on the recipient itself -- the ceiling, not a transfer

MIXER_XFER is the interesting one. R17 and R18 have near-identical mixer
behaviour (arm-difference span 2.85 vs 3.15 deg) but wildly different LNA
behaviour (-59.0 vs +1.6 deg arm asymmetry), so if the mixer table travels and
the LNA does not, a per-unit calibration could be a handful of cells rather than
a 2.6 h grid.

n = 2 RADIOS, ONE OF THEM DAMAGED. Every number here is a single observation in
each direction and neither direction is "fit healthy unit A, apply to healthy
unit B", because there is only one healthy unit. Read as an order of magnitude.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402

R18 = "1040007c4a94000211000b009186843ef2"
R17 = "104000bac4950008230026001b440a003a"
NAME = {R18: "R18 (clean)", R17: "R17 (damaged)"}
HS = FT.HardwareStates()
MIX = lambda g: HS.state(2, int(g))[1]   # noqa: E731
LNA = lambda g: HS.state(2, int(g))[0]   # noqa: E731


def design(g1, g2, use_mix=True, use_lna=True):
    feats = {}

    def col(arm, nm):
        k = (arm, nm)
        if k not in feats:
            feats[k] = len(feats)
        return feats[k]

    rows = []
    for a, b in zip(g1, g2):
        r = defaultdict(float)
        if use_mix:
            r[col(1, f"mix{MIX(a)}")] += 1.0
            r[col(2, f"mix{MIX(b)}")] -= 1.0
        if use_lna:
            r[col(1, f"lna{LNA(a)}")] += 1.0
            r[col(2, f"lna{LNA(b)}")] -= 1.0
        rows.append(r)
    X = np.zeros((len(rows), len(feats)))
    for i, r in enumerate(rows):
        for j, v in r.items():
            X[i, j] = v
    return X, feats


def solve(X, y):
    keep = np.any(np.abs(X) > 0, axis=0)
    th, *_ = np.linalg.lstsq(X[:, keep], np.asarray(y), rcond=None)
    full = np.zeros(X.shape[1])
    full[keep] = th
    return full


def evaluate(coef, feats, g1, g2, D, w):
    err, wt = [], []
    for a, b, d in zip(g1, g2, D):
        q = w.get(f"{int(a)},{int(b)}", 0)
        if not q:
            continue
        p = 0.0
        for arm, gg, sgn in ((1, a, 1.0), (2, b, -1.0)):
            for nm in (f"mix{MIX(gg)}", f"lna{LNA(gg)}"):
                if (arm, nm) in feats:
                    p += sgn * coef[feats[(arm, nm)]]
        err.append(abs(S.wrap(d - p)))
        wt.append(q)
    e, wv = np.asarray(err), np.asarray(wt, dtype=float)
    return float(np.degrees(np.sum(e * wv) / np.sum(wv))), int(np.sum(wv))


def main():
    weights = json.load(open("rover_cell_weights.json"))
    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")
    out = {}
    print("Held-out RADIO performance on rover cells, anchor 62 dB, usage-weighted\n")
    print(f"{'donor -> recipient':<34}{'carrier':>8}{'FULL':>9}{'MIXER':>9}{'LNA':>9}"
          f"{'SAME':>9}{'none':>9}")
    for donor, recip in ((R18, R17), (R17, R18)):
        for lo in (5_766_000_000.0, 5_840_000_000.0):
            fd = FT.add_anchor(f.sel((f.serial == donor) & (f.lo_hz == lo)), ref=62,
                               per_epoch=True)
            fr = FT.add_anchor(f.sel((f.serial == recip) & (f.lo_hz == lo)), ref=62,
                               per_epoch=True)
            w = weights.get(str(int(lo)), {})
            res = {}

            # FULL: donor's whole model
            Xd, fe = design(fd.g1, fd.g2)
            cd = solve(Xd, fd.D)
            res["FULL"], n = evaluate(cd, fe, fr.g1, fr.g2, fr.D, w)

            # SAME: recipient fitted on itself (ceiling)
            Xr, fer = design(fr.g1, fr.g2)
            cr = solve(Xr, fr.D)
            res["SAME"], _ = evaluate(cr, fer, fr.g1, fr.g2, fr.D, w)

            # MIXER_XFER: donor mixer fixed, recipient LNA refitted on its own data
            Xm_d, fe_m = design(fd.g1, fd.g2, use_lna=False)
            c_mix = solve(Xm_d, fd.D)
            resid = np.asarray(fr.D) - np.asarray(
                [sum(s * c_mix[fe_m[(arm, f"mix{MIX(g)}")]]
                     for arm, g, s in ((1, a, 1.0), (2, b, -1.0))
                     if (arm, f"mix{MIX(g)}") in fe_m)
                 for a, b in zip(fr.g1, fr.g2)])
            Xl_r, fe_l = design(fr.g1, fr.g2, use_mix=False)
            c_lna = solve(Xl_r, resid)
            comb = {**{k: c_mix[v] for k, v in fe_m.items()},
                    **{k: c_lna[v] for k, v in fe_l.items()}}
            keys = list(comb)
            fe_c = {k: i for i, k in enumerate(keys)}
            res["MIXER"], _ = evaluate(np.array([comb[k] for k in keys]), fe_c,
                                       fr.g1, fr.g2, fr.D, w)

            # LNA_XFER: donor LNA fixed, recipient mixer refitted
            Xl_d, fe_l2 = design(fd.g1, fd.g2, use_mix=False)
            c_l2 = solve(Xl_d, fd.D)
            resid2 = np.asarray(fr.D) - np.asarray(
                [sum(s * c_l2[fe_l2[(arm, f"lna{LNA(g)}")]]
                     for arm, g, s in ((1, a, 1.0), (2, b, -1.0))
                     if (arm, f"lna{LNA(g)}") in fe_l2)
                 for a, b in zip(fr.g1, fr.g2)])
            Xm_r, fe_m2 = design(fr.g1, fr.g2, use_lna=False)
            c_m2 = solve(Xm_r, resid2)
            comb2 = {**{k: c_l2[v] for k, v in fe_l2.items()},
                     **{k: c_m2[v] for k, v in fe_m2.items()}}
            keys2 = list(comb2)
            res["LNA"], _ = evaluate(np.array([comb2[k] for k in keys2]),
                                     {k: i for i, k in enumerate(keys2)},
                                     fr.g1, fr.g2, fr.D, w)

            res["none"], _ = evaluate(np.zeros(1), {}, fr.g1, fr.g2, fr.D, w)
            lab = f"{NAME[donor].split()[0]} -> {NAME[recip]}"
            print(f"{lab:<34}{lo/1e6:>8.0f}" +
                  "".join(f"{res[k]:>9.3f}" for k in ("FULL", "MIXER", "LNA", "SAME", "none")))
            out[f"{lab}|{int(lo)}"] = res
    json.dump(out, open("heldout_radio.json", "w"), indent=1)
    print("\nFULL  = donor's mixer+LNA wholesale        MIXER = donor mixer, LNA refitted locally")
    print("LNA   = donor LNA, mixer refitted locally  SAME  = fitted on the recipient (ceiling)")
    print("\nwrote heldout_radio.json")


if __name__ == "__main__":
    main()
