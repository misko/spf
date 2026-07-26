"""Validate a v4 Zarr written through direct-USB protocol v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def _validate_receiver(receiver, expected_frames: int) -> dict:
    if receiver.attrs.get("sdr_family") != "pluto":
        raise ValueError("receiver is missing Pluto hardware identity")
    serial = receiver.attrs.get("sdr_serial")
    usb_port_path = receiver.attrs.get("usb_port_path")
    if not serial:
        raise ValueError("receiver is missing Pluto serial")
    if not usb_port_path:
        raise ValueError("receiver is missing Pluto physical USB path")
    if receiver.attrs.get("rx_transport") != "direct_usb":
        raise ValueError("receiver was not recorded through direct USB")
    if receiver.attrs.get("direct_usb_serial") != serial:
        raise ValueError("direct USB and generic Pluto serials disagree")
    signal = receiver.signal_matrix
    expected_shape = (expected_frames, 2, 524288)
    if signal.shape != expected_shape:
        raise ValueError(f"signal shape is {signal.shape}, expected {expected_shape}")
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
        raise ValueError("RSSI does not use the legacy positive-magnitude convention")

    timestamps = receiver.system_timestamp[:]
    intervals = np.diff(timestamps)
    elapsed = float(intervals.sum()) if intervals.size else 0.0
    logical_bytes = expected_frames * 2 * 524288 * np.dtype("complex64").itemsize
    return {
        "frames": expected_frames,
        "sdr_serial": serial,
        "usb_port_path": list(usb_port_path),
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


def validate_capture(
    path: Path, expected_frames: int, expected_receivers: int = 1
) -> dict:
    z = zarr_open_from_lmdb_store(str(path))
    try:
        if z.attrs.get("sdr_identity_version") != 1:
            raise ValueError("capture is missing SDR identity version 1")
        receiver_names = sorted(z.receivers.keys())
        if len(receiver_names) != expected_receivers:
            raise ValueError(
                f"expected {expected_receivers} receivers, found {receiver_names}"
            )
        receivers = {
            name: _validate_receiver(z.receivers[name], expected_frames)
            for name in receiver_names
        }
        serials = [report["sdr_serial"] for report in receivers.values()]
        usb_paths = [
            tuple(report["usb_port_path"]) for report in receivers.values()
        ]
        if len(serials) != len(set(serials)):
            raise ValueError(f"duplicate receiver serials: {serials}")
        if len(usb_paths) != len(set(usb_paths)):
            raise ValueError(f"duplicate receiver USB paths: {usb_paths}")
        return {
            "status": "pass",
            "data_version": 4,
            "frames_per_receiver": expected_frames,
            "receiver_count": expected_receivers,
            "receivers": receivers,
        }
    finally:
        z.store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-receivers", type=int, default=1)
    args = parser.parse_args()
    try:
        report = validate_capture(
            args.path, args.expected_frames, args.expected_receivers
        )
    except (ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
