"""E-GSP7 -- score PRE-REGISTERED sparse combs on a fresh dense capture.

Fits `L26` on each pre-registered training comb and scores it on every LO NOT in
that comb, then compares:

  chosen-10   the conditioning-chosen comb   (cond  1.0899)
  ecal3-10    the actual E-CAL3 comb         (cond 17.9208)  <- primary control
  linspace-10 a uniform reconstruction       (cond 21.7782)  <- secondary control
  chosen-16   the conditioning-chosen N=16   (cond  1.0463)
  dense       all 111 LOs, leave-one-frequency-out is NOT attempted here; the
              dense row is the in-sample reference only

against the anchor-only baseline on the same held-out LOs, and against the
COMMITTED coefficients scored with no refit (which measures cross-session
transfer on this capture and is comparable to the 4.79-4.80 deg prospective
figure).

Both a free-delay and a frozen-delay (tau = fleet) fit are run, because E-GSC
puts N* at 16 free and 8 frozen and predicts 73.4% recovery at N = 10 frozen.

The model, design and scoring code is the committed E-GSC/E-GSP machinery,
imported unmodified -- only the data loader is new, because that code reads
`.npz` extracts of the old campaign and this is a fresh V7 capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS_GSC = (
    REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
    "/gain_state_computational_20260807_v1/analysis"
)
ANALYSIS_GSP = (
    REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
    "/gain_state_phase_model_20260802_v1/analysis"
)
for path in (ANALYSIS_GSC, ANALYSIS_GSP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as FT  # noqa: E402
import gsc_common as G  # noqa: E402
import spflib as S  # noqa: E402

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store  # noqa: E402

# Pre-registered in experiment_readme.md before the capture existed.
COMBS_MHZ = {
    "chosen-10": (900, 1050, 1200, 1350, 1750, 3050, 3800, 4500, 5200, 5900),
    "ecal3-10": G.PREREG_10_MHZ,
    "linspace-10": (400, 1000, 1600, 2250, 2850, 3450, 4050, 4700, 5300, 5900),
    "chosen-16": (
        750, 800, 1050, 1600, 2600, 2650, 2850, 2900,
        3250, 3450, 3550, 3650, 3700, 4150, 4450, 5500,
    ),
}
REFERENCE_GAIN = 26


def load_frames(root: Path) -> S.Frames:
    """V7 calibration zarrs -> the column table spflib/features expect."""
    parts = []
    for zpath in sorted(root.glob("*/calibration.v7.zarr")):
        serial = zpath.parent.name
        z = zarr_open_from_lmdb_store(str(zpath), mode="r")
        try:
            r = z["receivers/r0"]
            g = np.asarray(r["sweep_requested_gain_db"][:], dtype=np.int64)
            lo = np.asarray(r["sweep_lo_frequency_hz"][:], dtype=np.float64)
            n = len(lo)
            cols = {
                "serial": np.full(n, serial, dtype=object),
                "stage": np.full(n, "e_gsp7", dtype=object),
                "lo_hz": lo,
                "rf_hz": np.asarray(
                    r["sweep_rf_tone_frequency_hz"][:], dtype=np.float64
                ),
                "band": S.gain_band(lo),
                "g1": g[:, 0],
                "g2": g[:, 1],
                "epoch": np.asarray(r["sweep_epoch"][:], dtype=np.int64),
                "phase": S.wrap(
                    np.asarray(r["phase_difference_rad"][:], dtype=np.float64)
                ),
                "completed": np.asarray(r["sweep_completed"][:], dtype=bool),
                "qvalid": np.asarray(r["sweep_quality_valid"][:], dtype=bool),
                "coherence": np.asarray(r["coherence"][:], dtype=np.float64),
            }
        finally:
            z.store.close()
        f = S.Frames(cols)
        parts.append(f.sel(f.completed & f.qvalid))
    if not parts:
        raise SystemExit(f"no calibration datasets under {root}")
    merged = {
        k: np.concatenate([p.cols[k] for p in parts]) for k in parts[0].cols
    }
    return S.Frames(merged)


def run(root: Path, output: Path) -> int:
    frames = load_frames(root)
    f = FT.add_anchor(frames, ref=REFERENCE_GAIN, per_epoch=True)
    los = G.lo_mhz(f)
    uneq = f.g1 != f.g2
    present = sorted(set(los.tolist()))
    print(f"frames: {len(f)}  LOs present: {len(present)}  "
          f"radios: {len(set(f.serial))}\n")

    result = {
        "n_frames": int(len(f)),
        "n_los": len(present),
        "los_mhz": present,
        "reference_gain_db": REFERENCE_GAIN,
        "combs_mhz": {k: list(v) for k, v in COMBS_MHZ.items()},
        "arms": {},
    }

    for name, comb in COMBS_MHZ.items():
        missing = sorted(set(comb) - set(present))
        if missing:
            print(f"{name}: SKIPPED, LOs absent from capture: {missing}")
            result["arms"][name] = {"status": "skipped", "missing_mhz": missing}
            continue
        train = np.isin(los, np.asarray(comb))
        test = ~train
        arm = {
            "status": "ok",
            "n_train_los": len(comb),
            "n_test_los": len(set(present) - set(comb)),
            "n_train_frames": int(train.sum()),
            "n_test_frames": int(test.sum()),
        }
        for variant, model in (
            ("free", G.rung_model("L26")),
            ("frozen", G.rung_model("L26", G.TAU_FLEET)),
        ):
            pred, sup, taus, ncol = G.fit_and_predict(f, model, train, test)
            stats = G.score(f.D, pred, sup, test, uneq)
            stats["taus_ns"] = taus
            stats["n_columns"] = ncol
            arm[variant] = stats
            print(
                f"{name:12s} {variant:6s} "
                f"held-out MAE {stats['mae_deg']:6.3f}  "
                f"uneq {stats['uneq_mae_deg']:6.3f}  "
                f"baseline {stats['baseline_mae_deg']:6.3f}  "
                f"cov {stats['coverage']:.3f}  tau {np.round(taus, 2).tolist()}"
            )
        result["arms"][name] = arm
        print()

    # Committed coefficients, no refit: cross-session transfer on this capture.
    for coeff in ("l26_stage_a_v1", "l26_pooled_v1"):
        try:
            pred, sup = G.committed_predictions(f, coeff)
        except Exception as error:  # pragma: no cover - environment dependent
            result[f"committed_{coeff}"] = {"status": "error", "error": str(error)}
            print(f"committed {coeff}: ERROR {error}")
            continue
        everything = np.ones(len(f), dtype=bool)
        stats = G.score(f.D, pred, sup, everything, uneq)
        result[f"committed_{coeff}"] = stats
        print(
            f"committed {coeff:16s} MAE {stats['mae_deg']:6.3f}  "
            f"uneq {stats['uneq_mae_deg']:6.3f}  "
            f"baseline {stats['baseline_mae_deg']:6.3f}  cov {stats['coverage']:.3f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    return run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
