"""Validate a v4 Zarr written through direct-USB protocol v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def validate_capture(path: Path, expected_frames: int) -> dict:
    z = zarr_open_from_lmdb_store(str(path))
    try:
        receiver_names = sorted(z.receivers.keys())
        if receiver_names != ["r0"]:
            raise ValueError(f"expected one receiver, found {receiver_names}")
        receiver = z.receivers.r0
        signal = receiver.signal_matrix
        expected_shape = (expected_frames, 2, 524288)
        if signal.shape != expected_shape:
            raise ValueError(
                f"signal shape is {signal.shape}, expected {expected_shape}"
            )
        if signal.dtype != np.dtype("complex64"):
            raise ValueError(f"signal dtype is {signal.dtype}, expected complex64")
        for frame_index in range(expected_frames):
            frame = signal[frame_index]
            if not np.isfinite(frame).all():
                raise ValueError(f"frame {frame_index} contains non-finite IQ")
            if not np.any(frame[0]) or not np.any(frame[1]):
                raise ValueError(f"frame {frame_index} has an all-zero channel")

        gains = receiver.gains[:]
        rssis = receiver.rssis[:]
        for name, values in (("gains", gains), ("rssis", rssis)):
            if values.shape != (expected_frames, 2):
                raise ValueError(f"{name} shape is {values.shape}")
            if values.dtype != np.dtype("float64"):
                raise ValueError(f"{name} dtype is {values.dtype}, expected float64")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains invalid metadata")
        if np.any(rssis < 0):
            raise ValueError(
                "RSSI does not use the legacy positive-magnitude convention"
            )

        timestamps = receiver.system_timestamp[:]
        intervals = np.diff(timestamps)
        elapsed = float(intervals.sum()) if intervals.size else 0.0
        logical_bytes = expected_frames * 2 * 524288 * np.dtype("complex64").itemsize
        return {
            "status": "pass",
            "frames": expected_frames,
            "signal_shape": list(signal.shape),
            "signal_dtype": str(signal.dtype),
            "gains_shape": list(gains.shape),
            "gains_dtype": str(gains.dtype),
            "rssis_shape": list(rssis.shape),
            "rssis_dtype": str(rssis.dtype),
            "gain_minmax_db": [float(gains.min()), float(gains.max())],
            "rssi_minmax_db": [float(rssis.min()), float(rssis.max())],
            "first_to_last_seconds": (
                float(timestamps[-1] - timestamps[0]) if expected_frames > 1 else 0.0
            ),
            "median_frame_interval_seconds": (
                float(np.median(intervals)) if intervals.size else None
            ),
            "median_frame_rate_hz": (
                float(1.0 / np.median(intervals)) if intervals.size else None
            ),
            "logical_iq_mib_per_second": (
                logical_bytes / elapsed / (1024**2) if elapsed > 0 else None
            ),
        }
    finally:
        z.store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()
    try:
        report = validate_capture(args.path, args.expected_frames)
    except (ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
