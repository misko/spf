"""Finite direct-USB capture smoke test used during firmware RAM bring-up."""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

from spf.sdrpluto.direct_usb_protocol import (
    MetadataFlags,
    RadioMetadataV2,
)
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--samples", type=int, default=524288)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--bulk-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--protocol-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    with PlutoDirectUsbReceiver(
        serial=args.serial,
        bulk_chunk_bytes=args.bulk_chunk_bytes,
        protocol_version=args.protocol_version,
    ) as receiver:
        capture = receiver.capture(
            samples_per_channel=args.samples,
            frame_count=args.frames,
        )

    frame_results = []
    stream_ids = set()
    for frame in capture.frames:
        metadata = frame.metadata
        signal_matrix = iq_payload_to_complex64(
            frame.iq_payload, metadata.samples_per_channel
        )
        stream_ids.add(metadata.stream_id)
        frame_result = {
            "buffer_sequence": metadata.buffer_sequence,
            "first_sample_sequence": metadata.first_sample_sequence,
            "samples_per_channel": metadata.samples_per_channel,
            "iq_payload_bytes": len(frame.iq_payload),
            "flags": int(metadata.flags),
            "dummy_gains": bool(metadata.flags & MetadataFlags.DUMMY_GAINS),
            "gain_metadata_valid": metadata.gain_metadata_valid,
            "shape": list(signal_matrix.shape),
            "dtype": str(signal_matrix.dtype),
            "finite": bool(np.isfinite(signal_matrix).all()),
            "channel_nonzero": [
                bool(np.any(signal_matrix[channel] != 0)) for channel in range(2)
            ],
            "channel_rms": [
                float(
                    np.sqrt(
                        np.mean(np.abs(signal_matrix[channel]).astype(np.float64) ** 2)
                    )
                )
                for channel in range(2)
            ],
        }
        if isinstance(metadata, RadioMetadataV2):
            frame_result.update(
                {
                    "gain_db_start": list(metadata.gain_db_start),
                    "gain_db_end": list(metadata.gain_db_end),
                    "rssi_db_start": list(metadata.rssi_db_start),
                    "rssi_db_end": list(metadata.rssi_db_end),
                    "rssi_metadata_valid": metadata.rssi_metadata_valid,
                    "gain_endpoints_equal": list(metadata.gain_endpoints_equal),
                }
            )
        else:
            frame_result.update(
                {
                    "gain_index_start": list(metadata.gain_index_start),
                    "gain_index_end": list(metadata.gain_index_end),
                }
            )
        frame_results.append(frame_result)

    result = {
        "identity": dataclasses.asdict(capture.identity),
        "capabilities": {
            "protocol_min": capture.capabilities.protocol_min,
            "protocol_max": capture.capabilities.protocol_max,
            "supported_features": int(capture.capabilities.supported_features),
            "max_samples_per_channel": (capture.capabilities.max_samples_per_channel),
            "max_finite_frames": capture.capabilities.max_finite_frames,
            "capability_flags": int(capture.capabilities.capability_flags),
        },
        "elapsed_seconds": capture.elapsed_seconds,
        "frame_count": len(capture.frames),
        "one_stream_id": len(stream_ids) == 1,
        "frames": frame_results,
    }
    print(json.dumps(result, indent=2))

    expected_sequences = list(range(args.frames))
    actual_sequences = [frame.metadata.buffer_sequence for frame in capture.frames]
    return (
        0
        if (
            len(capture.frames) == args.frames
            and actual_sequences == expected_sequences
            and len(stream_ids) == 1
            and all(item["finite"] for item in frame_results)
            and all(all(item["channel_nonzero"]) for item in frame_results)
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
