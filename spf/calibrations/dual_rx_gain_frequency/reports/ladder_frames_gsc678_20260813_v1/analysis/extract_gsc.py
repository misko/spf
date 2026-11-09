"""Extract E-GSC6 / E-GSC7 / E-GSC8 frames from the QNAP raw stores, read-only.

The 2026-08-12 union analysis had to reconstruct rows from committed *fitted*
JSON because "the raw V7/Zarr stores are not on this machine". E-GSC8's
canonicalisation put every calibration store under
/mnt/qnap01/mouse9911/spf/calibration_data/raw/, so this pulls FRAMES instead --
which removes the additive-fit reconstruction residual (~0.70-0.75 deg) that made
every previous holdout number optimistic, and makes LOEO computable for the first
time.

Read-only throughout. Writes only into the scratch dir given on argv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import extract as EX  # noqa: E402  (reuses the published read-only extractor)

RAW = Path("/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency")

# stage label -> run directory. The label becomes Frames.stage, which is what
# add_anchor() groups on, so each run gets its own measured anchor -- correct,
# since they are separate bench sessions with their own cabling.
RUNS = {
    "GSC6":      "e_gsc6_equal_gain_diagonal_20260811_r17_r18_v1",
    "GSC7usb1":  "e_gsc7_iio_usb_20260812_v1",
    "GSC7usb2":  "e_gsc7_iio_usb_20260812_v2",
    "GSC7ip":    "e_gsc7_iio_ip_20260812_v1",
    "GSC8a":     "e_gsc8_iio_usb_20260813_v1",
    "GSC8b":     "e_gsc8_iio_usb_20260813_v2",
}


def main(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for stage, run in RUNS.items():
        run_dir = RAW / run
        if not run_dir.is_dir():
            print(f"MISSING {run_dir}")
            continue
        for serial_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            zp = serial_dir / "calibration.v7.zarr"
            if not zp.exists():
                continue
            out = out_root / stage / f"{serial_dir.name}.npz"
            if out.exists():
                print(f"have {stage}/{serial_dir.name[:12]}")
                continue
            meta = EX.extract(zp, out)
            meta.update(stage=stage, run=run, serial=serial_dir.name)
            summary.append(meta)
            print(f"OK {stage:9s} {serial_dir.name[:12]} frames={meta['n_frames']:5d} "
                  f"done={meta['n_completed']:5d} qv={meta['n_quality_valid']:5d}")
    (out_root / "extract_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n{len(summary)} stores extracted -> {out_root}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
