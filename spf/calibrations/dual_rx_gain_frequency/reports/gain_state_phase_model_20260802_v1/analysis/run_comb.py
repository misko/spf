"""Two engineering questions.

Q1  How coarse may the calibration frequency comb be?
    Hold out contiguous frequency blocks of increasing width and measure the
    error on the held-out interior. The block width is the effective comb gap.

Q2  What is the minimal RADIO-SPECIFIC parameter set?
    Take the best mechanistic model and promote one parameter family at a time
    from universal to per-radio, scoring on leave-one-epoch-out (does it help
    the radios you have?) and leave-one-radio-out (does it survive an unseen
    radio?).
"""

from __future__ import annotations

import json

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from models import LadderModel, Term, build_design

TAU = LD.TAU_GRID


def comb_sweep(f, models, blocks=(2, 4, 8, 16, 32, 57)):
    los = np.unique(f.lo_hz)
    print(f"\n{'model':46s} " + " ".join(f"{int(round((los[-1]-los[0])/1e6/b)):5d}"
                                        for b in blocks))
    print(f"{'  (held-out contiguous block width, MHz)':46s}")
    out = {}
    for m in models:
        d = build_design(f, m.terms)
        row = []
        for nb in blocks:
            preds = np.zeros(len(f))
            sup = np.zeros(len(f), dtype=bool)
            for _lbl, tr, te in LD.splits_leave_freq_block_out(f, n_blocks=nb):
                tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
                if not len(tri) or not len(tei):
                    continue
                p, sp, _t, _n = m.fit_eval(d, f.D, tri, tei)
                preds[tei] = p
                sup[tei] = sp
            fc = np.where(sup, preds, 0.0)
            row.append(S.cmae_deg(S.wrap(f.D - fc)))
        out[m.name] = dict(zip([f"{nb}blocks" for nb in blocks], row))
        print(f"{m.name:46s} " + " ".join(f"{v:5.2f}" for v in row))
    return out


def radio_specificity(f):
    """Promote one family at a time to per-radio."""
    base = [Term("lna"), Term("mixer"), Term("tia"), Term("lpf_lin"),
            Term("lna", basis="delay"), Term("mixer", basis="delay"),
            Term("lna", basis="cos0", groups=("band",)),
            Term("lna", basis="sin0", groups=("band",)),
            Term("lna", basis="cos1", groups=("band",)),
            Term("lna", basis="sin1", groups=("band",))]

    def with_groups(which):
        t = []
        for term in base:
            g = term.groups
            fam = ("ripple" if term.basis.startswith(("cos", "sin"))
                   else "delay" if term.basis == "delay" else "static")
            if fam in which:
                g = tuple(sorted(set(g) | {"serial"}))
            t.append(Term(term.value_field, term.basis, g, term.arm_specific))
        return t

    variants = {
        "all universal": set(),
        "+ per-radio static H": {"static"},
        "+ per-radio delay": {"delay"},
        "+ per-radio ripple amplitudes": {"ripple"},
        "+ everything per-radio": {"static", "delay", "ripple"},
    }
    print(f"\n{'variant':34s} {'params':>7s} {'LOEO MAE':>9s} {'LORO MAE':>9s} "
          f"{'LORO cov':>9s}")
    out = {}
    for name, which in variants.items():
        m = LadderModel(name, with_groups(which), tau_grids=[TAU, TAU])
        r_loeo = LD.evaluate(f, [m], "LOEO leave-one-epoch-out", verbose=False)[0]
        r_loro = LD.evaluate(f, [m], "LORO leave-one-radio-out", verbose=False)[0]
        out[name] = {"loeo": r_loeo, "loro": r_loro}
        print(f"{name:34s} {r_loeo['params']:7d} {r_loeo['mae_deg']:9.3f} "
              f"{r_loro['mae_deg']:9.3f} {r_loro['coverage']:9.3f}")
    return out


def main():
    fa = FT.add_anchor(S.load_stages(["A"]), ref=26)
    models = [
        LadderModel("L00 anchor only", []),
        LadderModel("L08 sym H(band,g)", [Term("g", groups=("band",))]),
        LadderModel("L16 MECH H(state)+ripple/LNA",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0")],
                    tau_grids=[TAU]),
        LadderModel("L26 MECH H(state)+2 ripples/LNA",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                     Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                    tau_grids=[TAU, TAU]),
        LadderModel("L27 MECH +delay +2 ripples/(band,LNA)",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band",)),
                     Term("lna", basis="sin0", groups=("band",)),
                     Term("lna", basis="cos1", groups=("band",)),
                     Term("lna", basis="sin1", groups=("band",))],
                    tau_grids=[TAU, TAU]),
        LadderModel("L24 per-frequency additive LUT",
                    [Term("g", groups=("serial", "lo_hz"), arm_specific=True)]),
    ]
    print("=" * 100)
    print("Q1  CALIBRATION COMB SPACING  (stage A, 113 LOs over 400-5900 MHz)")
    print("    held-out MAE in deg, all cells, fail-closed")
    print("=" * 100)
    r1 = comb_sweep(fa, models)

    print("\n" + "=" * 100)
    print("Q2  MINIMAL RADIO-SPECIFIC PARAMETER SET (stage A)")
    print("=" * 100)
    r2 = radio_specificity(fa)

    with open("comb_results.json", "w") as fh:
        json.dump({"comb": r1, "radio_specificity": r2}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
