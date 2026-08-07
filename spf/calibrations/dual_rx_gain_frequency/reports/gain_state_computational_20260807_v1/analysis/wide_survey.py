"""Reconstruct the wide 53-LO integer-gain survey's gain curves from the
COMMITTED fitted coefficients, because its raw Zarr stores no longer exist.

Source (in Git):
  spf/calibrations/dual_rx_gain_frequency/reports/
    wide_integer_gain_cross_band_20260730_v1/model_matrix.json

The committed `frequency_specific_additive_gain_per_radio` fit stores, per radio
and per exact LO, an intercept plus one coefficient per (arm, requested gain),
with the reference gain 26 dB dropped to zero:

    phase = intercept[f] + RX1[f, g1] + RX2[f, g2]

Because the reference coefficients are identically zero, the anchored residual
this repository models is available in closed form:

    D(f, g1, g2) = phase(f,g1,g2) - phase(f,26,26) = RX1[f,g1] + RX2[f,g2]
    H(f, g)      = [ D(g,26) - D(26,g) ] / 2 = [ RX1[f,g] - RX2[f,g] ] / 2
    A(f, g)      =   D(g,26) + D(26,g)       =   RX1[f,g] + RX2[f,g]

WHAT THIS IS NOT. These are FITTED cell values, not frames. Consequences that
every downstream claim must respect:

  * The per-frequency additive fit's own residual is 0.514 deg MAE in-sample and
    0.713 deg leave-one-epoch-out, so this reconstruction has had roughly that
    much measurement noise smoothed out of it. Errors computed against it are
    therefore optimistic relative to frame-level errors.
  * Epoch structure is absent from the file, so leave-one-epoch-out is
    impossible here.
  * The anchor is fixed at the fit's reference gain of 26 dB. It cannot be
    re-anchored, and per-epoch anchoring cannot be reproduced.
  * Quality masks cannot be re-derived. Both stores carry
    `validation_status = "fail_quality"` (25 and 43 of 27,825 frames failed).
  * This is a DIFFERENT SESSION on different dates from the A-G campaign. Its
    numbers are independent corroboration and must not be pooled with the
    campaign's H statistics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

MODEL_MATRIX = Path(
    "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/reports/"
    "wide_integer_gain_cross_band_20260730_v1/model_matrix.json"
)


class WideSurvey:
    def __init__(self, path: Path = MODEL_MATRIX):
        doc = json.loads(Path(path).read_text())
        self.doc = doc
        self.freqs = np.array(doc["frequencies_hz"], dtype=np.float64)
        self.gains = np.array(doc["gains_db"], dtype=np.int64)
        self.ref_gain = int(doc["reference_gain_db"])
        nf, ng = len(self.freqs), len(self.gains)
        fit = doc["models"]["frequency_specific_additive_gain_per_radio"]["fits"]
        self.serials = [
            p["serial"]
            for p in sorted(doc["provenance"], key=lambda x: x["radio_index"])
        ]
        self.rx1 = np.zeros((len(fit), nf, ng))
        self.rx2 = np.zeros((len(fit), nf, ng))
        self.intercept = np.zeros((len(fit), nf))
        pat = re.compile(r"frequency\[(\d+)\]\.(intercept|rx1_phase|rx2_phase)(?:\[(\d+)\])?")
        for r, f in enumerate(fit):
            for k, v in f["coefficients_rad"].items():
                m = pat.fullmatch(k)
                assert m, k
                fi = int(m.group(1))
                if m.group(2) == "intercept":
                    self.intercept[r, fi] = v
                elif m.group(2) == "rx1_phase":
                    self.rx1[r, fi, int(m.group(3))] = v
                else:
                    self.rx2[r, fi, int(m.group(3))] = v
        ref_idx = int(np.nonzero(self.gains == self.ref_gain)[0][0])
        assert np.all(self.rx1[:, :, ref_idx] == 0)
        assert np.all(self.rx2[:, :, ref_idx] == 0)
        self.ref_idx = ref_idx

    # ------------------------------------------------------------- curves ---
    def H_deg(self):
        """H[radio, freq, gain] in degrees. H(ref) == 0 by construction."""
        return np.degrees((self.rx1 - self.rx2) / 2.0)

    def A_deg(self):
        """Arm asymmetry A[radio, freq, gain] in degrees."""
        return np.degrees(self.rx1 + self.rx2)

    def D_deg(self, r, fi, g1_idx, g2_idx):
        return np.degrees(self.rx1[r, fi, g1_idx] + self.rx2[r, fi, g2_idx])
