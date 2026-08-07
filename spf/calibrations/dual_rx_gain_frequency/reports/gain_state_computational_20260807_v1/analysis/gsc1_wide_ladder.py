"""E-GSC1 -- fit the gain-state ladder to the wide 53-LO integer-gain survey.

The survey's raw Zarr stores are gone from this machine, but its per-radio,
per-frequency, per-arm additive fit is committed, and the anchored residual it
implies is available in closed form (see wide_survey.py). That is enough to run
the ladder, with the caveats recorded there and repeated in the output.

What this CAN answer: whether the hardware-state parameterisation survives a
64-gain axis and full LNA-state coverage, and whether band portability is a
frequency-extrapolation limit rather than a coverage hole.

What this CANNOT answer: leave-one-epoch-out (no epoch structure in the file),
and the 48 genuinely off-axis held-out pairs (their observations are not in the
file -- only the published summary metrics, which are quoted rather than
recomputed).
"""

from __future__ import annotations

import json

import numpy as np

import features as FT
import ladder as LD
import spflib as S
from models import LadderModel, Term
from wide_survey import WideSurvey

TAU = LD.TAU_GRID


def build_frames(ws: WideSurvey) -> S.Frames:
    """One row per additive-cross axis cell: (g, 26) and (26, g)."""
    serial, stage, lo, rf, band, g1, g2, D = [], [], [], [], [], [], [], []
    ref_i = ws.ref_idx
    ref_g = ws.ref_gain
    bands = S.gain_band(ws.freqs)
    for r, ser in enumerate(ws.serials):
        for fi, f_hz in enumerate(ws.freqs):
            for gi, g in enumerate(ws.gains):
                pairs = [(int(g), ref_g, gi, ref_i)]
                if gi != ref_i:
                    pairs.append((ref_g, int(g), ref_i, gi))
                for a, b, ai, bi in pairs:
                    serial.append(ser)
                    stage.append("W")
                    lo.append(float(f_hz))
                    rf.append(float(f_hz))
                    band.append(int(bands[fi]))
                    g1.append(a)
                    g2.append(b)
                    D.append(ws.rx1[r, fi, ai] + ws.rx2[r, fi, bi])
    n = len(D)
    return S.Frames(
        {
            "serial": np.array(serial, dtype=object),
            "stage": np.array(stage, dtype=object),
            "lo_hz": np.array(lo),
            "rf_hz": np.array(rf),
            "band": np.array(band, dtype=np.int8),
            "g1": np.array(g1, dtype=np.int64),
            "g2": np.array(g2, dtype=np.int64),
            "epoch": np.zeros(n, dtype=np.int64),
            "D": S.wrap(np.array(D)),
            "anchor": np.zeros(n),
        }
    )


def ladder():
    L = []
    L.append(LadderModel("L00 anchor only", []))
    L.append(LadderModel("L01 sym H(g) universal", [Term("g")]))
    L.append(LadderModel("L05 sym H(lna,mixer,tia,lpf)",
                         [Term("lna"), Term("mixer"), Term("tia"), Term("lpf")]))
    L.append(LadderModel("L06 sym H(gain-table row)", [Term("row")]))
    L.append(LadderModel("L08 sym H(band,g)", [Term("g", groups=("band",))]))
    L.append(LadderModel("L16 MECH H(state)+1 ripple/LNA",
                         [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                          Term("lna", basis="cos0"), Term("lna", basis="sin0")],
                         tau_grids=[TAU]))
    L.append(LadderModel("L26 MECH H(state)+2 ripples/LNA",
                         [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                          Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                          Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                         tau_grids=[TAU, TAU]))
    L.append(LadderModel("L27 MECH +delay +2 ripples/(band,LNA)",
                         [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                          Term("lna", basis="delay"), Term("mixer", basis="delay"),
                          Term("lna", basis="cos0", groups=("band",)),
                          Term("lna", basis="sin0", groups=("band",)),
                          Term("lna", basis="cos1", groups=("band",)),
                          Term("lna", basis="sin1", groups=("band",))],
                         tau_grids=[TAU, TAU]))
    L.append(LadderModel("L30 MIN H(lna,mixer,tia)",
                         [Term("lna"), Term("mixer"), Term("tia")]))
    L.append(LadderModel("L30b MIN H(lna,mixer) -- no h_tia",
                         [Term("lna"), Term("mixer")]))
    L.append(LadderModel("L31 MIN + 2 ripples/LNA",
                         [Term("lna"), Term("mixer"), Term("tia"),
                          Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                          Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                         tau_grids=[TAU, TAU]))
    L.append(LadderModel("L31b MIN + 2 ripples/LNA -- no h_tia",
                         [Term("lna"), Term("mixer"),
                          Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                          Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                         tau_grids=[TAU, TAU]))
    L.append(LadderModel("L33 L32 + linear LPF slope",
                         [Term("lna"), Term("mixer"), Term("tia"), Term("lpf_lin"),
                          Term("lna", basis="delay"), Term("mixer", basis="delay"),
                          Term("lna", basis="cos0", groups=("band",)),
                          Term("lna", basis="sin0", groups=("band",)),
                          Term("lna", basis="cos1", groups=("band",)),
                          Term("lna", basis="sin1", groups=("band",))],
                         tau_grids=[TAU, TAU]))
    # L24 (the per-frequency additive LUT) is deliberately NOT re-fitted here:
    # it is the model this reconstruction was derived FROM, so refitting it
    # would only recover its own coefficients, and under any frequency holdout
    # it is unsupported and fails closed to the anchor anyway. Its published
    # in-sample and leave-one-epoch-out numbers are quoted instead.
    return L


def main(out_path="gsc1_wide_ladder.json"):
    ws = WideSurvey()
    f = build_frames(ws)
    uneq = f.g1 != f.g2
    print(
        f"wide-survey reconstruction: rows={len(f)}  LOs={len(np.unique(f.lo_hz))}  "
        f"radios={len(np.unique(f.serial))}  gains={len(np.unique(f.g1))}"
    )
    print(f"baseline D (anchor only): all {S.cmae_deg(f.D):.4f} deg, "
          f"unequal-gain {S.cmae_deg(f.D[uneq]):.4f} deg")

    # state coverage: which LNA / TIA levels are reached at all?
    lna1 = FT.HW.vec(f.band, f.g1, "lna")
    tia1 = FT.HW.vec(f.band, f.g1, "tia")
    cover = {
        "lna_levels_present": sorted(int(x) for x in set(lna1.tolist()) - {-999}),
        "tia_levels_present": sorted(int(x) for x in set(tia1.tolist()) - {-999}),
        "lna_levels_per_band": {
            FT.BANDS[b]: sorted(
                int(x) for x in set(lna1[f.band == b].tolist()) - {-999}
            )
            for b in (0, 1, 2)
        },
    }
    print("state coverage:", json.dumps(cover))

    L = ladder()
    results = {}
    for split in (
        "LOFO leave-one-frequency-out",
        "LOBLOCK leave-frequency-block-out",
        "LORO leave-one-radio-out",
        "LOBAND leave-one-gain-table-band-out",
    ):
        print(f"\n--- {split} ---")
        results[split] = LD.evaluate(f, L, split)

    out = {
        "source": "committed wide_integer_gain_cross_band_20260730_v1/model_matrix.json",
        "convention": "ANCHORED D = phi - equal-gain(26/26) cell, fail-closed, degrees",
        "caveats": [
            "reconstructed from FITTED coefficients, not frames: the underlying "
            "additive fit's own residual is 0.514 deg MAE in-sample / 0.713 deg "
            "leave-one-epoch-out, so all errors here are optimistic relative to "
            "frame-level errors",
            "no epoch structure in the committed file, so no leave-one-epoch-out",
            "anchor fixed at the fit's 26 dB reference; cannot be re-anchored",
            "quality masks cannot be re-derived; both stores carry "
            "validation_status = fail_quality (25 and 43 of 27,825 frames)",
            "the 48 off-axis held-out pairs are NOT in the committed file; their "
            "published reference numbers are quoted, not recomputed",
            "the committed frequency list is used as the ripple basis frequency; "
            "the ~100 kHz tone offset is not recorded and is worth <0.1 deg here",
            "different session and dates from the A-G campaign -- independent "
            "corroboration, never pooled",
        ],
        "n_rows": len(f),
        "n_los": int(len(np.unique(f.lo_hz))),
        "baseline_all_deg": S.cmae_deg(f.D),
        "baseline_uneq_deg": S.cmae_deg(f.D[uneq]),
        "state_coverage": cover,
        "published_offaxis_reference": {
            "note": "from INTEGER_GAIN_CROSS_2P4_20260729.md, 48 off-axis pairs at "
                    "2412/2467 MHz -- quoted, not recomputed",
            "independent_rx1_rx2_curves_mae_deg": [1.48, 1.48],
            "shared_symmetric_H_mae_deg": [1.41, 1.25],
        },
        "results": results,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
