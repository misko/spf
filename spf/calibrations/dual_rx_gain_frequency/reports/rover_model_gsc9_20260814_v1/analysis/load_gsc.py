"""Build a spflib.Frames over the extracted E-GSC6/7/8 npz files.

Column names and dtypes match ``spflib.load_stage`` exactly, so every published
analysis module (features.add_anchor, ladder.make_ladder, ladder.evaluate) runs
against this union unmodified.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

ANALYSIS = ("/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
            "reports/gain_state_phase_model_20260802_v1/analysis")
sys.path.insert(0, ANALYSIS)

import spflib as S  # noqa: E402

# Transport repeats of one bench session are separate ANCHOR groups (they were
# re-cabled), but the same "epoch family" for leave-one-epoch-out.
EPOCH_FAMILY = {"GSC7usb1": 0, "GSC7usb2": 1, "GSC7ip": 2, "GSC8a": 0, "GSC8b": 1}


def load(extracted="extracted", quality_only=True) -> S.Frames:
    parts = []
    for p in sorted(glob.glob(os.path.join(extracted, "*", "*.npz"))):
        stage = os.path.basename(os.path.dirname(p))
        serial = os.path.basename(p)[: -len(".npz")]
        d = np.load(p, allow_pickle=True)
        n = len(d["sweep_completed"])
        g = d["sweep_requested_gain_db"].astype(np.int64)
        lo = d["sweep_lo_frequency_hz"].astype(np.float64)
        # Within a run the capture's own sweep_epoch separates repeat passes;
        # across runs the run itself is the epoch. Combine so LOEO holds out a
        # whole independent session rather than an interleaved sub-pass.
        ep = d["sweep_epoch"].astype(np.int64) + 100 * EPOCH_FAMILY.get(stage, 0)
        cols = {
            "serial": np.full(n, serial, dtype=object),
            "stage": np.full(n, stage, dtype=object),
            "lo_hz": lo,
            "rf_hz": d["sweep_rf_tone_frequency_hz"].astype(np.float64),
            "band": S.gain_band(lo),
            "g1": g[:, 0],
            "g2": g[:, 1],
            "epoch": ep,
            "phase": S.wrap(d["phase_difference_rad"]),
            "completed": d["sweep_completed"].astype(bool),
            "qvalid": d["sweep_quality_valid"].astype(bool),
            "qmask": d["sweep_quality_reason_mask"].astype(np.int64),
            "snr1": d["tone_snr_db"][:, 0].astype(np.float64),
            "snr2": d["tone_snr_db"][:, 1].astype(np.float64),
            "dbfs1": d["tone_dbfs"][:, 0].astype(np.float64),
            "dbfs2": d["tone_dbfs"][:, 1].astype(np.float64),
            "amp_ratio_db": d["amplitude_ratio_db_rx1_over_rx2"].astype(np.float64),
            "coherence": d["coherence"].astype(np.float64),
            "phase_std": d["within_capture_phase_std_rad"].astype(np.float64),
            "tx_gain_db": d["sweep_tx_gain_db"].astype(np.float64),
            "timestamp": d["system_timestamp"].astype(np.float64),
            "sched": d["sweep_schedule_index"].astype(np.int64),
        }
        f = S.Frames(cols)
        m = f.completed & (f.qvalid if quality_only else True)
        parts.append(f.sel(m))
    cols = {k: np.concatenate([p.cols[k] for p in parts]) for k in parts[0].cols}
    return S.Frames(cols)
