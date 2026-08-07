"""Extract the 2026-08-07 prospective follow-up campaign, read-only.

Uses the SOURCE analysis' extract.extract() unchanged, so the extraction path is
byte-for-byte the one that produced the A-G scalars. Writes only under the
output directory given on the command line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from extract import extract

ROOT = Path(
    "/mnt/qnap01/mouse9911/share/spf_campaigns/gain_state_followups_20260807_v1"
)


def main():
    out_root = Path(sys.argv[1])
    summary = []
    stages_dir = ROOT / "stages"
    for stage_dir in sorted(stages_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        for serial_dir in sorted(stage_dir.iterdir()):
            if not serial_dir.is_dir():
                continue
            zp = serial_dir / "calibration.v7.zarr"
            if not zp.exists():
                continue
            out = out_root / ROOT.name / stage_dir.name / f"{serial_dir.name}.npz"
            if out.exists():
                print(f"skip {out}")
                continue
            try:
                meta = extract(zp, out)
                meta["stage"] = stage_dir.name
                meta["campaign"] = ROOT.name
                summary.append(meta)
                print(
                    f"OK {stage_dir.name}/{serial_dir.name[:12]} "
                    f"frames={meta['n_frames']} done={meta['n_completed']} "
                    f"qv={meta['n_quality_valid']}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {zp}: {type(exc).__name__}: {exc}")
    with open(out_root / "extract_summary_followup.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
