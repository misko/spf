"""Validate a multi-radio v7 Zarr written through direct-USB protocol v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.hardware_fingerprint import (
    HardwareFingerprintError,
    validate_public_hardware_fingerprint,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.direct_usb_protocol import MetadataFlags


UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
)


def _validate_receiver(receiver, expected_frames: int) -> dict:
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
        for channel_index in range(2):
            if np.all(frame[channel_index] == frame[channel_index, 0]):
                raise ValueError(
                    f"frame {frame_index} channel {channel_index} is constant"
                )
        if np.array_equal(frame[0], frame[1]):
            raise ValueError(f"frame {frame_index} has duplicated RX channels")

    gain_valid = receiver.gain_metadata_valid[:]
    rssi_valid = receiver.rssi_metadata_valid[:]
    if not gain_valid.all():
        raise ValueError(
            "invalid gain metadata at frames " f"{np.flatnonzero(~gain_valid).tolist()}"
        )
    if not rssi_valid.all():
        raise ValueError(
            "invalid RSSI metadata at frames " f"{np.flatnonzero(~rssi_valid).tolist()}"
        )

    gain_start = receiver.gain_db_start[:]
    gain_end = receiver.gain_db_end[:]
    rssi_start = receiver.rssi_db_start[:]
    rssi_end = receiver.rssi_db_end[:]
    for name, values in (
        ("gain_db_start", gain_start),
        ("gain_db_end", gain_end),
        ("rssi_db_start", rssi_start),
        ("rssi_db_end", rssi_end),
    ):
        if values.shape != (expected_frames, 2):
            raise ValueError(f"{name} shape is {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    if not np.allclose(receiver.gains[:], gain_end):
        raise ValueError("legacy gains do not match frame gain_db_end")
    if not np.allclose(receiver.rssis[:], rssi_end):
        raise ValueError("legacy RSSIs do not match frame rssi_db_end")

    flags = receiver.gain_metadata_flags[:].astype(np.uint32)
    unsafe = np.flatnonzero((flags & int(UNSAFE_FLAGS)) != 0)
    if unsafe.size:
        raise ValueError(f"unsafe metadata flags at frames {unsafe.tolist()}")
    endpoint_equal = receiver.gain_endpoints_equal[:]
    expected_equal = np.column_stack(
        (
            (flags & int(MetadataFlags.RX1_ENDPOINT_CHANGED)) == 0,
            (flags & int(MetadataFlags.RX2_ENDPOINT_CHANGED)) == 0,
        )
    )
    if not np.array_equal(endpoint_equal, expected_equal):
        raise ValueError("gain_endpoints_equal disagrees with raw-index flags")

    stream_ids = receiver.stream_id[:]
    buffer_sequences = receiver.buffer_sequence[:]
    sample_sequences = receiver.sample_sequence[:]
    for frame_index in range(1, expected_frames):
        if stream_ids[frame_index] != stream_ids[frame_index - 1]:
            if buffer_sequences[frame_index] != 0:
                raise ValueError(
                    f"new stream at frame {frame_index} does not start at buffer 0"
                )
            if sample_sequences[frame_index] != 0:
                raise ValueError(
                    f"new stream at frame {frame_index} does not start at sample 0"
                )
            continue
        if buffer_sequences[frame_index] != buffer_sequences[frame_index - 1] + 1:
            raise ValueError(f"buffer sequence gap at frame {frame_index}")
        if sample_sequences[frame_index] != sample_sequences[frame_index - 1] + 524288:
            raise ValueError(f"sample sequence gap at frame {frame_index}")

    timestamps = receiver.system_timestamp[:]
    intervals = np.diff(timestamps)
    if intervals.size and not np.all(intervals > 0):
        raise ValueError("system timestamps are not strictly increasing")
    return {
        "serial": receiver.attrs.get("sdr_serial"),
        "usb_port_path": list(receiver.attrs.get("usb_port_path", [])),
        "frames": expected_frames,
        "signal_shape": list(signal.shape),
        "gain_minmax_db": [float(gain_end.min()), float(gain_end.max())],
        "rssi_minmax_db": [float(rssi_end.min()), float(rssi_end.max())],
        "unique_streams": int(np.unique(stream_ids).size),
        "endpoint_changed_frames": int(np.count_nonzero(~endpoint_equal.all(axis=1))),
        "median_frame_rate_hz": (
            float(1.0 / np.median(intervals)) if intervals.size else None
        ),
        "interval_p99_seconds": (
            float(np.percentile(intervals, 99)) if intervals.size else None
        ),
    }


def validate_capture(path: Path, expected_frames: int, expected_receivers: int) -> dict:
    z = zarr_open_from_lmdb_store(str(path))
    try:
        if z.attrs.get("radio_metadata_schema_version") != 2:
            raise ValueError("capture is missing radio metadata schema version 2")
        if z.attrs.get("sdr_identity_version") != 1:
            raise ValueError("capture is missing SDR identity version 1")
        receiver_names = sorted(z.receivers.keys())
        if len(receiver_names) != expected_receivers:
            raise ValueError(
                f"expected {expected_receivers} receivers, found {receiver_names}"
            )
        reports = {}
        serials = []
        usb_paths = []
        stable_fingerprints = []
        for name in receiver_names:
            receiver = z.receivers[name]
            if receiver.attrs.get("rx_transport") != "direct_usb":
                raise ValueError(f"{name}: receiver was not captured by direct USB")
            if receiver.attrs.get("gain_metadata_protocol_version") != 2:
                raise ValueError(f"{name}: receiver did not negotiate protocol v2")
            if receiver.attrs.get("firmware_verified") is not True:
                raise ValueError(f"{name}: capture firmware was not boot-verified")
            for attribute in (
                "firmware_release_tag",
                "firmware_image_sha256",
                "firmware_git_sha",
                "firmware_gadget_git_sha",
                "firmware_boot_mode",
            ):
                if not receiver.attrs.get(attribute):
                    raise ValueError(
                        f"{name}: missing firmware provenance attribute {attribute}"
                    )
            fingerprint = receiver.attrs.get("hardware_fingerprint_v1")
            if not isinstance(fingerprint, dict):
                raise ValueError(f"{name}: hardware fingerprint is missing")
            try:
                fingerprint = validate_public_hardware_fingerprint(fingerprint)
            except HardwareFingerprintError as error:
                raise ValueError(
                    f"{name}: hardware fingerprint is invalid: {error}"
                ) from error
            if (
                fingerprint.get("fingerprint_timing")
                != "post_firmware_before_recording"
                or fingerprint.get("acquisition_binding") is not True
            ):
                raise ValueError(
                    f"{name}: hardware fingerprint is not bound to this acquisition"
                )
            if fingerprint.get("stable_identity", {}).get(
                "pluto_serial"
            ) != receiver.attrs.get("sdr_serial"):
                raise ValueError(f"{name}: hardware fingerprint serial mismatch")
            stable_hash = fingerprint.get("stable_fingerprint_sha256")
            report = _validate_receiver(receiver, expected_frames)
            if not report["serial"] or not report["usb_port_path"]:
                raise ValueError(f"{name}: missing Pluto serial or physical USB path")
            serials.append(report["serial"])
            usb_paths.append(tuple(report["usb_port_path"]))
            stable_fingerprints.append(stable_hash)
            report["stable_fingerprint_sha256"] = stable_hash
            report["fingerprint_session_id"] = fingerprint.get("fingerprint_session_id")
            reports[name] = report
        if len(serials) != len(set(serials)):
            raise ValueError(f"duplicate receiver serials: {serials}")
        if len(usb_paths) != len(set(usb_paths)):
            raise ValueError(f"duplicate receiver USB paths: {usb_paths}")
        if len(stable_fingerprints) != len(set(stable_fingerprints)):
            raise ValueError(
                f"duplicate stable hardware fingerprints: {stable_fingerprints}"
            )
        return {
            "status": "pass",
            "data_version": 7,
            "frames_per_receiver": expected_frames,
            "receiver_count": expected_receivers,
            "receivers": reports,
        }
    finally:
        z.store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--expected-frames", type=int, default=100)
    parser.add_argument("--expected-receivers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_capture(
            args.zarr, args.expected_frames, args.expected_receivers
        )
    except (ValueError, KeyError, IndexError) as error:
        result = {"status": "fail", "error": str(error)}
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(rendered)
        print(rendered, end="")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
