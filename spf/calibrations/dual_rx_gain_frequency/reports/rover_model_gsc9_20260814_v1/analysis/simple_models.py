"""Which is the SIMPLEST physically-motivated model that keeps the correction?

Over 26..62 dB at band 2 the AD9361 realises gain in three disjoint regimes:
    26->40  baseband LPF attenuator only (0->14), LNA and mixer fixed
    40->41  a single LNA step (2->3)
    42->51  LPF continues (14->24)
    52->62  mixer only (4->15), LPF pinned at 24
TIA and RF_DC never move in this range.

Physically, the LPF is a BASEBAND attenuator after the mixer, so it should carry
little RF phase; the mixer and LNA are RF-side and should carry most of it. The
rover runs RX1 at 62 (LNA 3, mixer 15, LPF 24) against RX2 at 46..51 (LNA 3,
mixer 4, LPF 19..24), so in the rover's own regime the dominant term is the MIXER
difference plus a small LPF difference, with both arms on the same LNA.

Every model here is ARM-SPECIFIC and fitted per (radio, carrier), because that is
what E-GSC9 showed matters. They differ only in how each arm's table is indexed.
Scored identically to the committed result: leave-one-epoch-out inside session A,
error on the rover's own cells, weighted by rover cell usage, anchor 62 dB.
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
HS = FT.HardwareStates()
BAND = 2


def st(g, i):
    return HS.state(BAND, int(g))[i]


# Each basis maps a gain to a list of (feature_name, value) contributions.
def b_gain(g):
    return [(f"g{int(g)}", 1.0)]


def b_words(g):
    return [(f"lna{st(g,0)}", 1.0), (f"mix{st(g,1)}", 1.0),
            (f"tia{st(g,2)}", 1.0), (f"lpf{st(g,3)}", 1.0)]


def b_mix_lpf(g):
    return [(f"mix{st(g,1)}", 1.0), (f"lpf{st(g,3)}", 1.0)]


def b_mix_lna(g):
    return [(f"mix{st(g,1)}", 1.0), (f"lna{st(g,0)}", 1.0)]


def b_mix(g):
    return [(f"mix{st(g,1)}", 1.0)]


def b_lpf(g):
    return [(f"lpf{st(g,3)}", 1.0)]


def b_mix_lna_lpflin(g):
    """RF words discrete + baseband LPF as ONE linear slope (physically: a delay)."""
    return [(f"mix{st(g,1)}", 1.0), (f"lna{st(g,0)}", 1.0), ("lpf_lin", float(st(g, 3)))]


def b_linear(g):
    return [("lin", float(g))]


def b_quad(g):
    return [("lin", float(g)), ("quad", float(g) ** 2)]


MODELS = [
    ("gain LUT (committed)", b_gain),
    ("RF words: lna+mix+tia+lpf", b_words),
    ("mixer + lpf", b_mix_lpf),
    ("mixer + lna + lpf-as-slope", b_mix_lna_lpflin),
    ("mixer + lna", b_mix_lna),
    ("mixer only", b_mix),
    ("lpf only", b_lpf),
    ("quadratic in dB", b_quad),
    ("linear in dB", b_linear),
]


def fit(basis, g1, g2, D):
    """D ~ f_arm1(g1) - f_arm2(g2); separate feature sets per arm."""
    feats = {}

    def col(arm, name):
        k = (arm, name)
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
    keep = np.any(np.abs(X) > 0, axis=0)
    theta, *_ = np.linalg.lstsq(X[:, keep], np.asarray(D), rcond=None)
    full = np.zeros(len(feats))
    full[keep] = theta

    def predict(a, b):
        s = 0.0
        for nm, v in basis(a):
            s += v * full[feats[(1, nm)]]
        for nm, v in basis(b):
            s -= v * full[feats[(2, nm)]]
        return s

    return predict, int(keep.sum())


def main(ref=62):
    weights = json.load(open("rover_cell_weights.json"))
    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")
    out = {}
    print(f"anchor {ref} dB; leave-one-epoch-out; rover cells; rover-usage weighted\n")
    print(f"{'model':<30}{'par':>5}" + "".join(f"{NAME[s]+' '+str(int(l/1e6)):>12}"
          for s in (R18, R17) for l in (5766e6, 5840e6)))
    for label, basis in MODELS:
        cells = []
        npar = 0
        for ser in (R18, R17):
            for lo in (5766e6, 5840e6):
                m = (f.serial == ser) & (f.lo_hz == lo)
                fa = FT.add_anchor(f.sel(m), ref=ref, per_epoch=True)
                w = weights.get(str(int(lo)), {})
                errs, wts = [], []
                for te in sorted(set(fa.epoch.tolist())):
                    tr = fa.epoch != te
                    p, nc = fit(basis, fa.g1[tr], fa.g2[tr], fa.D[tr])
                    npar = max(npar, nc)
                    tem = fa.epoch == te
                    for a, b, d in zip(fa.g1[tem], fa.g2[tem], fa.D[tem]):
                        wt = w.get(f"{int(a)},{int(b)}", 0)
                        if wt:
                            errs.append(abs(S.wrap(d - p(a, b))))
                            wts.append(wt)
                e = np.asarray(errs)
                wv = np.asarray(wts, dtype=float)
                cells.append(float(np.degrees(np.sum(e * wv) / np.sum(wv))))
        out[label] = {"params": npar, "mae_deg": cells}
        print(f"{label:<30}{npar:>5}" + "".join(f"{c:>12.3f}" for c in cells))
    json.dump(out, open(f"simple_models_ref{ref}.json", "w"), indent=1)
    print(f"\nwrote simple_models_ref{ref}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 62)
