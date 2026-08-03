"""Extract scalar calibration measurements from V7 LMDB/Zarr calibration stores.

Read-only. Writes a compact .npz per (stage, serial) into the scratchpad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

SCALAR = [
    "sweep_completed",
    "sweep_epoch",
    "sweep_frequency_index",
    "sweep_schedule_index",
    "sweep_attempts",
    "sweep_quality_valid",
    "sweep_quality_reason_mask",
    "sweep_lo_frequency_hz",
    "sweep_rf_tone_frequency_hz",
    "sweep_tx_gain_db",
    "sweep_tx_digital_amplitude",
    "tone_frequency_hz",
    "tone_frequency_error_hz",
    "phase_difference_rad",
    "within_capture_phase_std_rad",
    "within_capture_resultant_length",
    "coherence",
    "amplitude_ratio_db_rx1_over_rx2",
    "system_timestamp",
    "rx_lo",
    "rx_bandwidth",
]

TWOX = [
    "sweep_requested_gain_db",
    "tone_dbfs",
    "tone_snr_db",
    "dc_dbfs",
    "clipping_fraction",
    "gains",
    "rssis",
]


def open_ro(path: str):
    store = zarr.LMDBStore(
        path,
        map_size=2**37,
        writemap=False,
        readonly=True,
        max_readers=1024,
        lock=False,
        meminit=False,
        readahead=False,
    )
    return zarr.open(store, mode="r"), store


def extract(zarr_path: Path, out_path: Path) -> dict:
    z, store = open_ro(str(zarr_path))
    try:
        rx = z["receivers/r0"]
        data = {}
        for k in SCALAR + TWOX:
            if k in rx:
                data[k] = np.asarray(rx[k][:])
        attrs = dict(rx.attrs)
        root_attrs = dict(z.attrs)
    finally:
        store.close()

    meta = {
        "zarr_path": str(zarr_path),
        "serial": attrs.get("serial") or attrs.get("pluto_serial"),
        "n_frames": int(len(data["sweep_completed"])),
        "n_completed": int(data["sweep_completed"].sum()),
        "n_quality_valid": int(data["sweep_quality_valid"].sum()),
        "receiver_attr_keys": sorted(attrs.keys()),
        "root_attr_keys": sorted(root_attrs.keys()),
        "phase_convention": root_attrs.get("phase_convention"),
    }
    # keep small json-able subset of attrs
    small = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            small[k] = v
        elif isinstance(v, dict):
            small[k] = {
                kk: vv
                for kk, vv in v.items()
                if isinstance(vv, (str, int, float, bool)) or vv is None
            }
    meta["receiver_attrs"] = small

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    with open(str(out_path) + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return meta


def main():
    roots = [
        Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full"),
        Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full_r2"),
    ]
    out_root = Path(sys.argv[1])
    summary = []
    for root in roots:
        stages_dir = root / "stages"
        if not stages_dir.exists():
            continue
        for stage_dir in sorted(stages_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            for serial_dir in sorted(stage_dir.iterdir()):
                zp = serial_dir / "calibration.v7.zarr"
                if not zp.exists():
                    continue
                out = out_root / root.name / stage_dir.name / f"{serial_dir.name}.npz"
                if out.exists():
                    print(f"skip {out}")
                    continue
                try:
                    meta = extract(zp, out)
                    meta["stage"] = stage_dir.name
                    meta["campaign"] = root.name
                    summary.append(meta)
                    print(
                        f"OK {root.name}/{stage_dir.name}/{serial_dir.name[:12]} "
                        f"frames={meta['n_frames']} done={meta['n_completed']} "
                        f"qv={meta['n_quality_valid']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {zp}: {type(exc).__name__}: {exc}")
    with open(out_root / "extract_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
