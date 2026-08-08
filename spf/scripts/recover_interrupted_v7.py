"""Recover the verified common prefix of an interrupted V7 capture."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from typing import Callable

import numpy as np
import yaml

from spf.dataset.v7_data import v7rx_keys, v7rx_new_dataset
from spf.scripts.validate_direct_usb_v7_zarr import (
    UNSAFE_FLAGS,
    validate_capture,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store, zarr_shrink
from spf.sdrpluto.direct_usb_protocol import MetadataFlags


RECOVERY_SCHEMA = "spf.interrupted_v7_recovery"
RECOVERY_VERSION = 1
SOURCE_HASH_ALGORITHM = "spf.sparse-file-sha256.v1"


def _sha256_file(path: Path) -> str:
    """Hash sparse LMDB content without reading potentially enormous holes."""

    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(SOURCE_HASH_ALGORITHM.encode("ascii") + b"\0")
    digest.update(struct.pack("<Q", size))
    descriptor = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while offset < size:
            try:
                data_offset = os.lseek(descriptor, offset, os.SEEK_DATA)
            except OSError as error:
                if error.errno == errno.ENXIO:
                    break
                if error.errno in {errno.EINVAL, errno.ENOTSUP}:
                    data_offset = offset
                else:
                    raise
            try:
                hole_offset = os.lseek(descriptor, data_offset, os.SEEK_HOLE)
            except OSError as error:
                if error.errno in {errno.EINVAL, errno.ENOTSUP}:
                    hole_offset = size
                else:
                    raise
            hole_offset = min(hole_offset, size)
            digest.update(struct.pack("<QQ", data_offset, hole_offset - data_offset))
            read_offset = data_offset
            while read_offset < hole_offset:
                block = os.pread(
                    descriptor,
                    min(1024 * 1024, hole_offset - read_offset),
                    read_offset,
                )
                if not block:
                    raise OSError(f"short read while hashing {path}")
                digest.update(block)
                read_offset += len(block)
            offset = hole_offset
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _first_invalid_reason(receiver, index: int, previous: dict | None) -> str | None:
    timestamp = float(receiver.system_timestamp[index])
    if not np.isfinite(timestamp) or timestamp <= 0:
        return "missing system timestamp"
    if previous is not None and timestamp <= previous["timestamp"]:
        return "non-monotonic system timestamp"
    if not bool(receiver.gain_metadata_valid[index]):
        return "invalid gain metadata"
    if not bool(receiver.rssi_metadata_valid[index]):
        return "invalid RSSI metadata"
    flags = int(receiver.gain_metadata_flags[index])
    if flags & int(UNSAFE_FLAGS):
        return f"unsafe metadata flags 0x{flags:x}"

    gain_start = np.asarray(receiver.gain_db_start[index])
    gain_end = np.asarray(receiver.gain_db_end[index])
    rssi_start = np.asarray(receiver.rssi_db_start[index])
    rssi_end = np.asarray(receiver.rssi_db_end[index])
    if not all(
        values.shape == (2,) and np.isfinite(values).all()
        for values in (gain_start, gain_end, rssi_start, rssi_end)
    ):
        return "non-finite or malformed gain/RSSI metadata"
    if not np.allclose(receiver.gains[index], gain_end):
        return "legacy gain does not match frame endpoint"
    if not np.allclose(receiver.rssis[index], rssi_end):
        return "legacy RSSI does not match frame endpoint"
    expected_equal = np.asarray(
        [
            (flags & int(MetadataFlags.RX1_ENDPOINT_CHANGED)) == 0,
            (flags & int(MetadataFlags.RX2_ENDPOINT_CHANGED)) == 0,
        ]
    )
    if not np.array_equal(receiver.gain_endpoints_equal[index], expected_equal):
        return "gain endpoint equality disagrees with flags"

    stream_id = int(receiver.stream_id[index])
    buffer_sequence = int(receiver.buffer_sequence[index])
    sample_sequence = int(receiver.sample_sequence[index])
    if previous is not None:
        if stream_id == previous["stream_id"]:
            if buffer_sequence != previous["buffer_sequence"] + 1:
                return "buffer sequence discontinuity"
            if sample_sequence != previous["sample_sequence"] + 524288:
                return "sample sequence discontinuity"
        elif buffer_sequence != 0 or sample_sequence != 0:
            return "new stream does not begin at zero"

    frame = np.asarray(receiver.signal_matrix[index])
    if frame.shape != (2, 524288) or frame.dtype != np.dtype("complex64"):
        return "unexpected IQ shape or dtype"
    if not np.isfinite(frame).all():
        return "non-finite IQ"
    if not np.any(frame[0]) or not np.any(frame[1]):
        return "all-zero IQ channel"
    if np.all(frame[0] == frame[0, 0]) or np.all(frame[1] == frame[1, 0]):
        return "constant IQ channel"
    if np.array_equal(frame[0], frame[1]):
        return "duplicated RX channels"
    return None


def valid_receiver_prefix(receiver, *, include_gain_series=False) -> tuple[int, str]:
    """Independently count contiguous fully valid records for one receiver."""

    required = set(v7rx_keys(include_gain_series=include_gain_series))
    missing = required - set(receiver.keys())
    if missing:
        raise ValueError(f"receiver is missing V7 fields: {sorted(missing)}")
    frame_count = int(receiver.signal_matrix.shape[0])
    previous = None
    for index in range(frame_count):
        reason = _first_invalid_reason(receiver, index, previous)
        if reason is not None:
            return index, reason
        previous = {
            "timestamp": float(receiver.system_timestamp[index]),
            "stream_id": int(receiver.stream_id[index]),
            "buffer_sequence": int(receiver.buffer_sequence[index]),
            "sample_sequence": int(receiver.sample_sequence[index]),
        }
    return frame_count, "allocated capture is completely valid"


def recover_capture(
    source_path: Path,
    output_path: Path,
    *,
    reason: str,
    strict_validator: Callable[[Path, int, int], dict] = validate_capture,
) -> dict:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not reason.strip():
        raise ValueError("a non-empty recovery reason is required")
    if not source_path.is_dir() or not (source_path / "data.mdb").is_file():
        raise ValueError(f"source is not an LMDB Zarr: {source_path}")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    temporary_path = output_path.with_name(output_path.name + ".recovery.tmp")
    if temporary_path.exists():
        raise ValueError(f"recovery temporary output already exists: {temporary_path}")
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("output must be separate from the immutable source")

    source_data_path = source_path / "data.mdb"
    source_sha256_before = _sha256_file(source_data_path)
    source = zarr_open_from_lmdb_store(str(source_path), mode="r")
    destination = None
    try:
        source_status = source.attrs.get("capture_status")
        if source_status not in {"incomplete", "in_progress"}:
            raise ValueError(
                f"source capture_status must be incomplete/in_progress, found "
                f"{source_status!r}"
            )
        if source.attrs.get("radio_metadata_schema_version") != 2:
            raise ValueError("source is not a protocol-v2 V7 capture")
        receiver_names = sorted(source.receivers.keys())
        if not receiver_names or receiver_names != [
            f"r{index}" for index in range(len(receiver_names))
        ]:
            raise ValueError(f"receiver groups are not contiguous: {receiver_names}")
        include_gain_series = source.attrs.get("gain_series_schema_version") == 1
        detected = []
        stopping_reasons = []
        serials = []
        for receiver_name in receiver_names:
            receiver = source.receivers[receiver_name]
            prefix, stopping_reason = valid_receiver_prefix(
                receiver, include_gain_series=include_gain_series
            )
            detected.append(prefix)
            stopping_reasons.append(stopping_reason)
            serial = receiver.attrs.get("sdr_serial")
            if not serial:
                raise ValueError(f"{receiver_name} is missing its SDR serial")
            serials.append(serial)
        if len(serials) != len(set(serials)):
            raise ValueError(f"source contains duplicate receiver serials: {serials}")
        common_prefix = min(detected)
        if common_prefix < 1:
            raise ValueError(f"source has no recoverable common prefix: {detected}")

        source_config_text = str(source.config[0])
        config = yaml.safe_load(source_config_text)
        if not isinstance(config, dict) or config.get("data-version") != 7:
            raise ValueError("source config is not V7")
        config["n-records-per-receiver"] = common_prefix
        config["recovery"] = {
            "schema": RECOVERY_SCHEMA,
            "version": RECOVERY_VERSION,
            "source_path": str(source_path),
            "reason": reason.strip(),
        }
        buffer_sizes = {
            int(source.receivers[name].signal_matrix.shape[2])
            for name in receiver_names
        }
        if buffer_sizes != {524288}:
            raise ValueError(f"unexpected V7 buffer sizes: {sorted(buffer_sizes)}")

        destination = v7rx_new_dataset(
            filename=str(temporary_path),
            timesteps=common_prefix,
            buffer_size=524288,
            n_receivers=len(receiver_names),
            config=config,
            chunk_size=512,
            compressor=None,
        )
        destination.attrs.update(dict(source.attrs))
        destination.attrs.update(
            {
                "capture_status": "recovered_incomplete",
                "capture_records_written_by_receiver": [common_prefix]
                * len(receiver_names),
                "recovery_schema": RECOVERY_SCHEMA,
                "recovery_version": RECOVERY_VERSION,
                "recovery_source_path": str(source_path),
                "recovery_source_capture_status": source_status,
                "recovery_source_data_sha256": source_sha256_before,
                "recovery_source_data_hash_algorithm": SOURCE_HASH_ALGORITHM,
                "recovery_source_config_sha256": hashlib.sha256(
                    source_config_text.encode("utf-8")
                ).hexdigest(),
                "recovery_detected_valid_records_by_receiver": detected,
                "recovery_common_prefix_records": common_prefix,
                "recovery_reason": reason.strip(),
                "recovery_unix": time.time(),
            }
        )
        destination.attrs["recovery_stopping_reasons_by_receiver"] = stopping_reasons
        for receiver_name in receiver_names:
            source_receiver = source.receivers[receiver_name]
            destination_receiver = destination.receivers[receiver_name]
            destination_receiver.attrs.update(dict(source_receiver.attrs))
            for key in v7rx_keys(include_gain_series=include_gain_series):
                destination_receiver[key][:] = source_receiver[key][:common_prefix]
    finally:
        if destination is not None:
            destination.store.close()
        source.store.close()

    source_sha256_after = _sha256_file(source_data_path)
    if source_sha256_after != source_sha256_before:
        raise RuntimeError("source data changed during read-only recovery")
    zarr_shrink(str(temporary_path))
    validation = strict_validator(
        temporary_path,
        common_prefix,
        len(receiver_names),
    )
    if validation.get("status") != "pass":
        raise ValueError(f"strict recovered V7 validation failed: {validation}")
    os.rename(temporary_path, output_path)
    output_sha256 = _sha256_file(output_path / "data.mdb")
    report = {
        "status": "pass",
        "recovery_schema": RECOVERY_SCHEMA,
        "recovery_version": RECOVERY_VERSION,
        "source_path": str(source_path),
        "source_data_sha256": source_sha256_before,
        "source_data_hash_algorithm": SOURCE_HASH_ALGORITHM,
        "output_path": str(output_path),
        "output_data_sha256": output_sha256,
        "output_data_hash_algorithm": SOURCE_HASH_ALGORITHM,
        "source_capture_status": source_status,
        "detected_valid_records_by_receiver": detected,
        "stopping_reasons_by_receiver": stopping_reasons,
        "common_prefix_records": common_prefix,
        "serials": serials,
        "reason": reason.strip(),
        "strict_validation": validation,
    }
    report_path = output_path.with_name(output_path.name + ".recovery.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        report = recover_capture(args.source, args.output, reason=args.reason)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
