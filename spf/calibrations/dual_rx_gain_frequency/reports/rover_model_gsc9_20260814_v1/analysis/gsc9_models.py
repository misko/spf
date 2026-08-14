"""E-GSC9 changes the model question: the rover's cells are now MEASURED.

Every previous rung had to reach the rover's operating cells by ADDITIVITY,
D(g1,g2) = d1(g1) - d2(g2), because no capture had ever visited them. E-GSC9
session A measures all 1,369 cells of [26,62]^2 directly, five times, at both
carriers. So the additive form is no longer a necessity -- it is now a choice
that can be priced.

This scores four predictors under leave-one-epoch-out inside session A, on the
rover's OWN cells, WEIGHTED BY HOW OFTEN THE ROVER USES EACH CELL:

  L00   predict 0            the measured equal-gain anchor, no model
  ADD   d1(g1) - d2(g2)      the additive per-arm LUT (the L24 form)
  FULL  cell mean            direct lookup of the measured cell
  MECHA RF-word additive     H(state1) - H(state2) over the audited gain table

Weighting matters: an unweighted cell average treats (62,26), which the rover
never uses, the same as (62,49), which is 16.3% of its frames.

Read-only. Writes only into this scratch directory.
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
NAME = {R18: "R18", R17: "R17"}
CARRIERS = (5_766_000_000.0, 5_840_000_000.0)


def circ_mean(a):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(a)))))


def fit_additive(g1, g2, D, ref):
    """Least-squares per-arm LUT: D ~ d1(g1) - d2(g2), d1(ref)=d2(ref)=0."""
    gains = sorted(set(g1.tolist()) | set(g2.tolist()))
    idx = {g: i for i, g in enumerate(gains)}
    n = len(gains)
    X = np.zeros((len(D), 2 * n))
    for r, (a, b) in enumerate(zip(g1, g2)):
        X[r, idx[int(a)]] += 1.0
        X[r, n + idx[int(b)]] -= 1.0
    keep = [i for i in range(2 * n) if i != idx[ref] and i != n + idx[ref]]
    X = X[:, keep]
    theta, *_ = np.linalg.lstsq(X + 0.0, np.asarray(D), rcond=None)
    full = np.zeros(2 * n)
    full[keep] = theta

    def predict(a, b):
        return full[idx[int(a)]] - full[n + idx[int(b)]]

    return predict


def fit_mech(g1, g2, D, band, tables, ref):
    """RF-word additive: D ~ sum_field [H_field(s1) - H_field(s2)]."""
    fields = ("lna", "mixer", "tia", "lpf")
    lv = {f: sorted({getattr_state(tables, band, int(g), f)
                     for g in set(g1.tolist()) | set(g2.tolist())}) for f in fields}
    cols, idx = [], {}
    for f in fields:
        for v in lv[f]:
            idx[(f, v)] = len(cols)
            cols.append((f, v))
    X = np.zeros((len(D), len(cols)))
    for r, (a, b) in enumerate(zip(g1, g2)):
        for f in fields:
            X[r, idx[(f, getattr_state(tables, band, int(a), f))]] += 1.0
            X[r, idx[(f, getattr_state(tables, band, int(b), f))]] -= 1.0
    active = np.any(np.abs(X) > 0, axis=0)
    A = X[:, active]
    theta, *_ = np.linalg.lstsq(A, np.asarray(D), rcond=None)
    full = np.zeros(len(cols))
    full[active] = theta

    def predict(a, b):
        s = 0.0
        for f in fields:
            s += full[idx[(f, getattr_state(tables, band, int(a), f))]]
            s -= full[idx[(f, getattr_state(tables, band, int(b), f))]]
        return s

    return predict


_HS = None


def getattr_state(tables, band, gain, field):
    global _HS
    if _HS is None:
        import features as _f
        _HS = _f.HardwareStates()
    st = _HS.state(band, gain)
    return st[{"lna": 0, "mixer": 1, "tia": 2, "lpf": 3}[field]]


def main(anchor_mode="rover62"):
    weights = json.load(open("rover_cell_weights.json"))
    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")
    tables = None
    out = {}
    print(f"anchor mode: {anchor_mode}\n")
    hdr = (f"{'radio':<5}{'carrier':>8}{'anchor':>7}{'cells':>7}{'wframes':>10}"
           f"{'L00':>9}{'ADD':>9}{'FULL':>9}{'MECH':>9}   {'FULL vs L00':>12}")
    print(hdr)
    for ser in (R18, R17):
        for lo in CARRIERS:
            m = (f.serial == ser) & (f.lo_hz == lo)
            if not m.sum():
                continue
            sub = f.sel(m)
            band = int(sub.band[0])
            anchors = {"rover62": 62, "best56": 56}
            ref = anchors.get(anchor_mode, 62)
            fa = FT.add_anchor(sub, ref=ref, per_epoch=True)
            w = weights.get(str(int(lo)), {})
            epochs = sorted(set(fa.epoch.tolist()))
            err = defaultdict(list)
            wts = []
            for te in epochs:
                tr = fa.epoch != te
                tem = fa.epoch == te
                g1t, g2t, Dt = fa.g1[tr], fa.g2[tr], fa.D[tr]
                p_add = fit_additive(g1t, g2t, Dt, ref)
                p_mech = fit_mech(g1t, g2t, Dt, band, tables, ref)
                cell = defaultdict(list)
                for a, b, d in zip(g1t, g2t, Dt):
                    cell[(int(a), int(b))].append(d)
                cellmean = {k: circ_mean(v) for k, v in cell.items()}
                for a, b, d in zip(fa.g1[tem], fa.g2[tem], fa.D[tem]):
                    key = f"{int(a)},{int(b)}"
                    wt = w.get(key, 0)
                    if wt == 0:
                        continue
                    wts.append(wt)
                    err["L00"].append(S.wrap(d))
                    err["ADD"].append(S.wrap(d - p_add(a, b)))
                    err["MECH"].append(S.wrap(d - p_mech(a, b)))
                    cm = cellmean.get((int(a), int(b)))
                    err["FULL"].append(S.wrap(d - cm) if cm is not None else S.wrap(d))
            if not wts:
                continue
            wa = np.asarray(wts, dtype=float)
            res = {}
            for k, v in err.items():
                e = np.abs(np.asarray(v))
                res[k] = float(np.degrees(np.sum(e * wa) / np.sum(wa)))
            ncell = len({(int(a), int(b)) for a, b in zip(fa.g1, fa.g2)
                         if f"{int(a)},{int(b)}" in w})
            print(f"{NAME[ser]:<5}{lo/1e6:>8.0f}{ref:>7}{ncell:>7}{int(np.sum(wa)):>10}"
                  f"{res['L00']:>9.3f}{res['ADD']:>9.3f}{res['FULL']:>9.3f}"
                  f"{res['MECH']:>9.3f}   {res['L00']/res['FULL']:>11.1f}x")
            out[f"{NAME[ser]}|{int(lo)}|{ref}"] = {**res, "n_cells": ncell,
                                                   "weighted_frames": float(np.sum(wa))}
    json.dump(out, open(f"gsc9_models_{anchor_mode}.json", "w"), indent=1)
    print(f"\nwrote gsc9_models_{anchor_mode}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rover62")
