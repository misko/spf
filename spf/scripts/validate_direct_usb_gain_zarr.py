"""Validate and report throughput for an SPF v6 direct-USB capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.direct_usb_protocol import MetadataFlags


UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
)


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

        for frame_idx in range(expected_frames):
            frame = signal[frame_idx]
            if not np.isfinite(frame).all():
                raise ValueError(f"frame {frame_idx} contains non-finite IQ")
            if not np.any(frame[0]) or not np.any(frame[1]):
                raise ValueError(f"frame {frame_idx} has an all-zero channel")

        valid = receiver.gain_metadata_valid[:]
        if not valid.all():
            bad = np.flatnonzero(~valid).tolist()
            raise ValueError(f"invalid gain metadata at frames {bad}")

        starts = receiver.gain_index_start[:]
        ends = receiver.gain_index_end[:]
        if np.any(starts > 0x7F) or np.any(ends > 0x7F):
            raise ValueError("valid gain indices must be in [0, 127]")

        endpoint_equal = receiver.gain_endpoints_equal[:]
        if not np.array_equal(endpoint_equal, starts == ends):
            raise ValueError("gain_endpoints_equal disagrees with endpoint indices")

        flags = receiver.gain_metadata_flags[:].astype(np.uint32)
        unsafe_mask = int(UNSAFE_FLAGS)
        unsafe = np.flatnonzero((flags & unsafe_mask) != 0)
        if unsafe.size:
            raise ValueError(f"unsafe metadata flags at frames {unsafe.tolist()}")

        stream_ids = receiver.stream_id[:]
        buffer_sequences = receiver.buffer_sequence[:]
        sample_sequences = receiver.sample_sequence[:]
        for frame_idx in range(1, expected_frames):
            if stream_ids[frame_idx] != stream_ids[frame_idx - 1]:
                if buffer_sequences[frame_idx] != 0:
                    raise ValueError(
                        f"new stream at frame {frame_idx} does not start at buffer 0"
                    )
                if sample_sequences[frame_idx] != 0:
                    raise ValueError(
                        f"new stream at frame {frame_idx} does not start at sample 0"
                    )
                continue
            if buffer_sequences[frame_idx] != buffer_sequences[frame_idx - 1] + 1:
                raise ValueError(f"buffer sequence gap at frame {frame_idx}")
            if sample_sequences[frame_idx] != sample_sequences[frame_idx - 1] + 524288:
                raise ValueError(f"sample sequence gap at frame {frame_idx}")

        if not np.isnan(receiver.rssis[:]).all():
            raise ValueError("direct capture unexpectedly contains legacy RSSI")
        if not np.isnan(receiver.gains[:]).all():
            raise ValueError("direct capture unexpectedly contains legacy gains")

        timestamps = receiver.system_timestamp[:]
        intervals = np.diff(timestamps)
        duration = float(timestamps[-1] - timestamps[0])
        logical_bytes = expected_frames * 2 * 524288 * np.dtype("complex64").itemsize
        elapsed_for_rate = float(intervals.sum()) if intervals.size else 0.0
        return {
            "status": "pass",
            "frames": expected_frames,
            "signal_shape": list(signal.shape),
            "signal_dtype": str(signal.dtype),
            "logical_iq_bytes": logical_bytes,
            "first_to_last_seconds": duration,
            "median_frame_interval_seconds": (
                float(np.median(intervals)) if intervals.size else None
            ),
            "median_frame_rate_hz": (
                float(1.0 / np.median(intervals)) if intervals.size else None
            ),
            "logical_iq_mib_per_second": (
                logical_bytes / elapsed_for_rate / (1024**2)
                if elapsed_for_rate > 0
                else None
            ),
            "unique_streams": int(np.unique(stream_ids).size),
            "endpoint_changed_frames": int(
                np.count_nonzero(~endpoint_equal.all(axis=1))
            ),
            "gain_read_duration_ns": {
                "start_p50": float(
                    np.percentile(receiver.gain_start_read_duration_ns[:], 50)
                ),
                "start_p99": float(
                    np.percentile(receiver.gain_start_read_duration_ns[:], 99)
                ),
                "end_p50": float(
                    np.percentile(receiver.gain_end_read_duration_ns[:], 50)
                ),
                "end_p99": float(
                    np.percentile(receiver.gain_end_read_duration_ns[:], 99)
                ),
            },
        }
    finally:
        z.store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--expected-frames", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_capture(args.zarr, args.expected_frames)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
