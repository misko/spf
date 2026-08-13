"""Build the E-GSC6 + E-GSC7 union dataset of anchored residuals D.

BOTH SOURCES ARE COMMITTED FITTED ARTEFACTS, NOT FRAMES. The raw V7/Zarr stores
for E-GSC6 (2026-08-11) and E-GSC7 (2026-08-12) are not on this machine; only the
per-report JSON they emitted is in Git. That is the same position
``gain_state_computational_20260807_v1/analysis/wide_survey.py`` was in for the
53-LO survey, and this module follows its documented reconstruction exactly.

E-GSC6
------
``equal_gain_diagonal_20260811_v1/additive_cross_<serial>.json`` stores, per
radio and per exact LO, an intercept plus one coefficient per (arm, requested
gain) with the 26 dB reference dropped to zero:

    phase(f, g1, g2) = intercept[f] + RX1[f, g1] + RX2[f, g2]
    D(f, g1, g2)     = phase(f,g1,g2) - phase(f,26,26) = RX1[f,g1] + RX2[f,g2]

The fit's training schedule is the 41 axis cells per frequency -- (g, 26) for all
21 gains and (26, g) for the 20 non-reference gains -- so those are exactly the
rows rebuilt here. Reconstruction error inherits the additive fit's own residual
(0.70 deg training MAE, 0.71/0.75 deg held-out), so every holdout number computed
on these rows is OPTIMISTIC relative to a frame-level number by roughly that much.

E-GSC7
------
``e_gsc7_iio_20260812_v1/analysis.json`` stores far less: ten adjacent 1 dB
*shared-effect* steps over 52->62 dB (mixer 5->15), at 5766 MHz ONLY, for two
radios x two transports. There is no per-arm split and no other frequency, so:

  * the cumulative shared curve S(g) = sum of steps, S(52) = 0, is recoverable;
  * the ladder's antisymmetric prediction for a pair is D(g1,g2) = S(g1) - S(g2),
    so pseudo-rows (g, 52) and (52, g) are exact under the model's own shape;
  * mixer levels 6..14 are therefore estimable AT ONE FREQUENCY ONLY. They carry
    zero frequency leverage, and E-GSC7's own H5 measured that a 5766 MHz curve
    does not transfer 466 MHz (9.06 deg RMS on the clean radio).

The two transports are treated as repeat epochs of the same radio, which is what
E-GSC7's own paired test showed them to be (Wilcoxon p = 0.322 / 0.695).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

# this file lives at <repo>/spf/calibrations/dual_rx_gain_frequency/reports/
#                     l31_gsc6_gsc7_union_20260812_v1/analysis/union.py
REPO = Path(os.environ.get("SPF_REPO", Path(__file__).resolve().parents[6]))
REPORTS = REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
GSC6_DIR = REPORTS / "equal_gain_diagonal_20260811_v1"
GSC7_JSON = REPORTS / "e_gsc7_iio_20260812_v1" / "analysis.json"

SERIAL_R17 = "104000bac4950008230026001b440a003a"
SERIAL_R18 = "1040007c4a94000211000b009186843ef2"
SHORT = {SERIAL_R17: "R17", SERIAL_R18: "R18"}

GSC7_LO_HZ = 5_766_000_000.0


def _wrap(x):
    return (np.asarray(x, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def load_gsc6() -> list[dict]:
    """One row per additive-cross axis cell, per radio, per LO."""
    rows = []
    for path in sorted(GSC6_DIR.glob("additive_cross_*.json")):
        doc = json.loads(path.read_text())
        serial = doc["serial"]
        gains = doc["gain_values_db"]
        ref_g = int(doc["reference_gain_db"])
        ref_i = gains.index(ref_g)
        for fr in doc["frequency_results"]:
            assert fr["status"] == "fit", fr["status"]
            f_hz = float(fr["frequency_hz"])
            rx1 = np.asarray(fr["rx1_effect_rad"], dtype=float)
            rx2 = np.asarray(fr["rx2_effect_rad"], dtype=float)
            assert rx1[ref_i] == 0.0 and rx2[ref_i] == 0.0
            for gi, g in enumerate(gains):
                pairs = [(int(g), ref_g, gi, ref_i)]
                if gi != ref_i:
                    pairs.append((ref_g, int(g), ref_i, gi))
                for a, b, ai, bi in pairs:
                    rows.append({
                        "source": "E-GSC6",
                        "serial": serial,
                        "radio": SHORT[serial],
                        "epoch": "gsc6",
                        "lo_hz": f_hz,
                        "rf_hz": f_hz,
                        "g1": a,
                        "g2": b,
                        "D": float(rx1[ai] + rx2[bi]),
                    })
    return rows


def load_gsc7(runs=("usb", "ip"), radios=("R17", "R18")) -> list[dict]:
    """Pseudo-rows from the cumulative shared-effect curve at 5766 MHz."""
    doc = json.loads(GSC7_JSON.read_text())
    gains = doc["rf_word_audit"]["gain_db"]           # 52 .. 62
    assert gains == list(range(52, 63)), gains
    rows = []
    for run in doc["results_5766"]:
        if run["transport"] not in runs or run["radio"] not in radios:
            continue
        steps = np.asarray(run["steps_deg"], dtype=float)
        assert len(steps) == 10
        curve = np.concatenate([[0.0], np.cumsum(steps)])   # S(52) = 0
        curve_rad = np.radians(curve)
        for gi, g in enumerate(gains):
            if g == 52:
                continue
            for a, b, v in ((int(g), 52, curve_rad[gi]),
                            (52, int(g), -curve_rad[gi])):
                rows.append({
                    "source": "E-GSC7",
                    "serial": SERIAL_R17 if run["radio"] == "R17" else SERIAL_R18,
                    "radio": run["radio"],
                    "epoch": f"gsc7_{run['transport']}",
                    "lo_hz": GSC7_LO_HZ,
                    "rf_hz": GSC7_LO_HZ,
                    "g1": a,
                    "g2": b,
                    "D": float(v),
                })
    return rows


def to_frames(rows: list[dict]) -> dict:
    keys = ("source", "serial", "radio", "epoch")
    out = {k: np.array([r[k] for r in rows], dtype=object) for k in keys}
    out["lo_hz"] = np.array([r["lo_hz"] for r in rows], dtype=float)
    out["rf_hz"] = np.array([r["rf_hz"] for r in rows], dtype=float)
    out["g1"] = np.array([r["g1"] for r in rows], dtype=np.int64)
    out["g2"] = np.array([r["g2"] for r in rows], dtype=np.int64)
    out["D"] = _wrap([r["D"] for r in rows])
    return out


def gsc6_heldout_diagonal() -> dict:
    """The genuine frame-level held-out cells E-GSC6 kept back.

    Every one of the 20 held-out pairs per frequency is an EQUAL-GAIN cell
    (g, g). Any antisymmetric rung -- L26, L30, L31, all of them -- predicts
    exactly 0 there by construction, so the observed |D(g,g)| IS the rung's
    held-out frame-level error on this data, with nothing fitted to it.
    """
    out = {}
    for path in sorted(GSC6_DIR.glob("additive_cross_*.json")):
        doc = json.loads(path.read_text())
        serial = doc["serial"]
        recs = []
        for fr in doc["frequency_results"]:
            ref = float(fr["reference_cell_mean_rad"])
            for c in fr["held_out_cells"]:
                assert c["gain_rx1_db"] == c["gain_rx2_db"]
                recs.append({
                    "lo_hz": float(fr["frequency_hz"]),
                    "g": int(c["gain_rx1_db"]),
                    "n": int(c["n_quality_valid"]),
                    "D_obs_rad": float(_wrap(c["observed_mean_rad"] - ref)),
                })
        out[SHORT[serial]] = {"serial": serial, "cells": recs}
    return out


def subset(frames: dict, mask) -> dict:
    return {k: v[mask] for k, v in frames.items()}


def build(include_gsc7=True, radios=("R17", "R18")) -> dict:
    rows = [r for r in load_gsc6() if r["radio"] in radios]
    if include_gsc7:
        rows += load_gsc7(radios=radios)
    return to_frames(rows)


if __name__ == "__main__":
    f = build()
    print(f"rows={len(f['D'])}  LOs={len(np.unique(f['lo_hz']))}  "
          f"radios={sorted(set(f['radio']))}  "
          f"gains={sorted(set(f['g1'].tolist()) | set(f['g2'].tolist()))}")
    for src in ("E-GSC6", "E-GSC7"):
        m = f["source"] == src
        print(f"  {src}: {int(m.sum())} rows, "
              f"{len(np.unique(f['lo_hz'][m]))} LOs")
