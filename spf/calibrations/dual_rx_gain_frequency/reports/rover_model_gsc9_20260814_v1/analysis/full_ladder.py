"""The complete ladder: parameters, same-radio performance, held-out-radio performance.

Every rung is scored two ways on the rover's own cells, usage-weighted, anchor 62 dB:

  SAME-RADIO   leave-one-epoch-out within the radio itself. The ceiling.
  HELD-OUT     fitted on one radio, applied to the other. What the rover gets,
               because its units are uncalibrated.

Held-out is averaged over all four donor->recipient x carrier combinations, and the
range is reported because they differ a lot. n = 2 radios and one is damaged, so
neither direction is healthy->healthy; read the held-out column as an order of
magnitude, not an estimate.
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
HS = FT.HardwareStates()
W = json.load(open("rover_cell_weights.json"))


def sfield(g, i):
    return HS.state(2, int(g))[i]


# name -> (feature fn, arm_specific?)
BASES = [
    ("no correction (baseline)", None, None),
    ("linear in dB", lambda g: [("lin", float(g))], True),
    ("quadratic in dB", lambda g: [("lin", float(g)), ("q", float(g) ** 2)], True),
    ("mixer only", lambda g: [(f"mix{sfield(g,1)}", 1.0)], True),
    ("LNA only", lambda g: [(f"lna{sfield(g,0)}", 1.0)], True),
    ("LPF only", lambda g: [(f"lpf{sfield(g,3)}", 1.0)], True),
    ("mixer + LNA", lambda g: [(f"mix{sfield(g,1)}", 1.0), (f"lna{sfield(g,0)}", 1.0)], True),
    ("mixer + LNA + LPF slope",
     lambda g: [(f"mix{sfield(g,1)}", 1.0), (f"lna{sfield(g,0)}", 1.0),
                ("lpflin", float(sfield(g, 3)))], True),
    ("mixer + LPF", lambda g: [(f"mix{sfield(g,1)}", 1.0), (f"lpf{sfield(g,3)}", 1.0)], True),
    ("all four RF words",
     lambda g: [(f"lna{sfield(g,0)}", 1.0), (f"mix{sfield(g,1)}", 1.0),
                (f"tia{sfield(g,2)}", 1.0), (f"lpf{sfield(g,3)}", 1.0)], True),
    ("gain LUT (per dB)", lambda g: [(f"g{int(g)}", 1.0)], True),
    ("SHARED-H RF words (shipped L26/L30/L31 shape)",
     lambda g: [(f"lna{sfield(g,0)}", 1.0), (f"mix{sfield(g,1)}", 1.0),
                (f"tia{sfield(g,2)}", 1.0), (f"lpf{sfield(g,3)}", 1.0)], False),
    ("SHARED-H gain LUT", lambda g: [(f"g{int(g)}", 1.0)], False),
]


def build(basis, g1, g2, arm_specific):
    feats = {}

    def col(arm, nm):
        k = (arm if arm_specific else 0, nm)
        if k not in feats:
            feats[k] = len(feats)
        return feats[k]

    rows = []
    for a, b in zip(g1, g2):
        r = defaultdict(float)
        for nm, v in basis(a):
            r[col(1, nm)] += v
        for nm, v in basis(b):
            r[col(2, nm)] -= v
        rows.append(r)
    X = np.zeros((len(rows), len(feats)))
    for i, r in enumerate(rows):
        for j, v in r.items():
            X[i, j] = v
    return X, feats


def predict(coef, feats, basis, g1, g2, arm_specific):
    out = []
    for a, b in zip(g1, g2):
        s = 0.0
        for nm, v in basis(a):
            k = (1 if arm_specific else 0, nm)
            if k in feats:
                s += v * coef[feats[k]]
        for nm, v in basis(b):
            k = (2 if arm_specific else 0, nm)
            if k in feats:
                s -= v * coef[feats[k]]
        out.append(s)
    return np.asarray(out)


def wmae(D, pred, g1, g2, lo):
    w = W.get(str(int(lo)), {})
    e, q = [], []
    for a, b, d, p in zip(g1, g2, D, pred):
        n = w.get(f"{int(a)},{int(b)}", 0)
        if n:
            e.append(abs(S.wrap(d - p)))
            q.append(n)
    e, q = np.asarray(e), np.asarray(q, dtype=float)
    return float(np.degrees(np.sum(e * q) / np.sum(q)))


def main():
    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")
    fits = {}
    for ser in (R18, R17):
        for lo in (5_766_000_000.0, 5_840_000_000.0):
            fits[(ser, lo)] = FT.add_anchor(
                f.sel((f.serial == ser) & (f.lo_hz == lo)), ref=62, per_epoch=True)
    rows = []
    for name, basis, arm in BASES:
        if basis is None:
            same = [wmae(fits[(s, lo)].D, np.zeros(len(fits[(s, lo)])),
                         fits[(s, lo)].g1, fits[(s, lo)].g2, lo)
                    for s in (R18, R17) for lo in (5.766e9, 5.84e9)]
            rows.append((name, 0, same, same))
            continue
        # same-radio: leave-one-epoch-out
        same, held = [], []
        npar = 0
        for ser in (R18, R17):
            for lo in (5.766e9, 5.84e9):
                fa = fits[(ser, lo)]
                errs = []
                for te in sorted(set(fa.epoch.tolist())):
                    tr = fa.epoch != te
                    X, fe = build(basis, fa.g1[tr], fa.g2[tr], arm)
                    keep = np.any(np.abs(X) > 0, axis=0)
                    th, *_ = np.linalg.lstsq(X[:, keep], fa.D[tr], rcond=None)
                    c = np.zeros(X.shape[1])
                    c[keep] = th
                    npar = max(npar, int(keep.sum()))
                    m = fa.epoch == te
                    errs.append((fa.D[m], predict(c, fe, basis, fa.g1[m], fa.g2[m], arm),
                                 fa.g1[m], fa.g2[m]))
                D = np.concatenate([e[0] for e in errs])
                P = np.concatenate([e[1] for e in errs])
                G1 = np.concatenate([e[2] for e in errs])
                G2 = np.concatenate([e[3] for e in errs])
                same.append(wmae(D, P, G1, G2, lo))
        # held-out radio: fit donor, apply to recipient
        for donor, recip in ((R18, R17), (R17, R18)):
            for lo in (5.766e9, 5.84e9):
                fd, fr = fits[(donor, lo)], fits[(recip, lo)]
                X, fe = build(basis, fd.g1, fd.g2, arm)
                keep = np.any(np.abs(X) > 0, axis=0)
                th, *_ = np.linalg.lstsq(X[:, keep], fd.D, rcond=None)
                c = np.zeros(X.shape[1])
                c[keep] = th
                held.append(wmae(fr.D, predict(c, fe, basis, fr.g1, fr.g2, arm),
                                 fr.g1, fr.g2, lo))
        rows.append((name, npar, same, held))
    print(f"{'model':<46}{'par':>5}{'same-radio':>22}{'held-out radio':>24}")
    print(f"{'':<46}{'':>5}{'mean (range)':>22}{'mean (range)':>24}")
    for name, npar, same, held in rows:
        s, h = np.asarray(same), np.asarray(held)
        print(f"{name:<46}{npar:>5}"
              f"{s.mean():>10.2f} ({s.min():.2f}-{s.max():.2f})"
              f"{h.mean():>12.2f} ({h.min():.2f}-{h.max():.2f})")
    json.dump({n: {"params": p, "same": s, "held_out": h} for n, p, s, h in rows},
              open("full_ladder.json", "w"), indent=1)
    print("\nwrote full_ladder.json")


if __name__ == "__main__":
    main()
