"""Emit the E-GSC9 arm LUT in the EXISTING spf.calibration.phase_offset_model schema.

NO CODE CHANGE IS REQUIRED TO CONSUME THIS. PhaseOffsetModel already implements
kind="frequency_specific_gain_additive":

    phase = intercept[f] + RX1[f, g1] + RX2[f, g2]

which is exactly how the model was fitted -- a separate arm table per carrier -- so
one file covers both 5766 and 5840 MHz.

Our fit is D = d1(g1) - d2(g2); the schema ADDS RX2, so rx2_phase = -d2.

A NEW family directory (gsc9_arm_lut_per_radio) is used rather than overwriting
gain_additive_lut_per_radio/<serial>.json, which is a different measurement from the
2026-07 cross-band survey. Artifacts are append-only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
from full_ladder import build, sfield  # noqa: E402

R18 = "1040007c4a94000211000b009186843ef2"
R17 = "104000bac4950008230026001b440a003a"
CARRIERS = (5_766_000_000, 5_840_000_000)
GAINS = list(range(26, 63))
ANCHOR = 62


def main(out_root: str, serial: str = R18):
    out = Path(out_root)
    mdir = out / "spf/calibrations/models/gsc9_arm_lut_per_radio"
    sdir = out / "spf/calibrations/models/gsc9_radio_support"
    mdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(
        ["git", "-C", "/home/mouse9911/gits/spf", "rev-parse", "HEAD"]).decode().strip()

    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")
    basis = lambda g: [(f"mix{sfield(g,1)}", 1.0), (f"lna{sfield(g,0)}", 1.0)]  # noqa: E731

    freqs = list(CARRIERS)
    coeffs: dict[str, float] = {}
    npar = 0
    n_frames = 0
    for k, lo in enumerate(freqs):
        fa = FT.add_anchor(f.sel((f.serial == serial) & (f.lo_hz == float(lo))),
                           ref=ANCHOR, per_epoch=True)
        n_frames += len(fa)
        X, fe = build(basis, fa.g1, fa.g2, True)
        keep = np.any(np.abs(X) > 0, axis=0)
        th, *_ = np.linalg.lstsq(X[:, keep], fa.D, rcond=None)
        c = np.zeros(X.shape[1])
        c[keep] = th
        npar += int(keep.sum())

        def arm(a, g, _fe=fe, _c=c):
            s = 0.0
            for nm, v in basis(g):
                if (a, nm) in _fe:
                    s += v * _c[_fe[(a, nm)]]
            return float(s)

        coeffs[f"frequency[{k}].intercept"] = 0.0
        for i, g in enumerate(GAINS):
            coeffs[f"frequency[{k}].rx1_phase[{i}]"] = arm(1, g)
            coeffs[f"frequency[{k}].rx2_phase[{i}]"] = -arm(2, g)

    support = {
        "schema": "spf.calibration.phase_offset_support",
        "schema_version": 1,
        "radio_serial": serial,
        "phase_convention": "RX1 minus RX2",
        "frequencies_hz": freqs,
        "gains_db": GAINS,
        "support_kind": "cartesian_product",
        "supported_cell_count": len(freqs) * len(GAINS) ** 2,
        "expected_cells": len(freqs) * len(GAINS) ** 2,
        "source": {"capture": "e_gsc9_session_a_20260813_v1", "spf_git_sha": sha},
    }
    spath = sdir / f"{serial}.json"
    spath.write_text(json.dumps(support, indent=1) + "\n")
    ssha = hashlib.sha256(spath.read_bytes()).hexdigest()

    doc = {
        "schema": "spf.calibration.phase_offset_model",
        "schema_version": 1,
        "scope": "per_radio",
        "kind": "frequency_specific_gain_additive",
        "model_name": "gsc9_arm_lut_per_radio",
        "label": "E-GSC9 arm gain LUT (mixer+LNA fit, per carrier, expressed per dB)",
        "radio_serial": serial,
        "phase_convention": "RX1 minus RX2",
        "formula": "phase = intercept[f] + RX1[f,g1] + RX2[f,g2]",
        "frequencies_hz": freqs,
        "gains_db": GAINS,
        "reference_gain_db": ANCHOR,
        "reference_frequency_hz": float(freqs[0]),
        "can_predict_unseen_frequency": False,
        "parameter_count": npar,
        "coefficients_rad": coeffs,
        "support_profile": {
            "path": f"../gsc9_radio_support/{serial}.json",
            "sha256": ssha,
            "strict_prediction_default": True,
        },
        "source": {
            "capture": "e_gsc9_session_a_20260813_v1",
            "grid": "[26,62]^2, 5 epochs, 2 carriers, anchor 62 dB",
            "n_frames": n_frames,
            "spf_git_sha": sha,
            "fit": "mixer+LNA arm-specific (28 params per carrier), tabulated per dB",
        },
        "evaluation": {
            "same_radio_rover_cell_weighted_mae_deg": 0.149,
            "held_out_radio_rover_cell_weighted_mae_deg": 2.51,
            "note": "HELD-OUT is the figure relevant to rover use: the rover's radios "
                    "are not this serial, and cross-radio transfer was measured at "
                    "1.16x on the bench ladder. Applying this table to a rover radio "
                    "is a deliberate held-out application.",
        },
    }
    mpath = mdir / f"{serial}.json"
    mpath.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {mpath.name}  ({mpath.stat().st_size} B, {npar} params)")
    print(f"wrote {spath.name}  ({support['supported_cell_count']} supported cells)")

    sys.path.insert(0, str(out))
    from spf.calibrations.models import load_phase_model
    m = load_phase_model(mpath)
    for lo in freqs:
        v = m.predict_phase_offset(frequency_hz=lo, gain_rx1_db=62, gain_rx2_db=49)
        v0 = m.predict_phase_offset(frequency_hz=lo, gain_rx1_db=62, gain_rx2_db=62)
        print(f"  loader OK @{lo/1e6:.0f} MHz  kind={m.kind}  "
              f"predict(62,49)={np.degrees(v):+7.3f} deg   anchor(62,62)={np.degrees(v0):+7.3f} deg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else R18)
