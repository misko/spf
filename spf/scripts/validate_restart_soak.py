"""Validate identity and aggregate receiver time across restart-separated Zarrs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def validate(paths: list[Path], minimum_sessions: int, minimum_seconds: float) -> dict:
    if minimum_sessions < 1 or minimum_seconds < 0:
        raise ValueError("minimum sessions must be positive and seconds non-negative")
    sessions = []
    identity_bindings = None
    total_capture_seconds = 0.0
    fingerprint_sessions = set()
    for path in paths:
        zarr = zarr_open_from_lmdb_store(str(path), mode="r", readahead=False)
        try:
            if zarr.attrs.get("capture_status") != "complete":
                raise ValueError(f"{path}: capture is not complete")
            receiver_names = sorted(zarr["receivers"].keys())
            serials = tuple(
                zarr[f"receivers/{name}"].attrs["sdr_serial"] for name in receiver_names
            )
            fingerprints = tuple(
                zarr[f"receivers/{name}"].attrs["hardware_fingerprint_v1"][
                    "stable_fingerprint_sha256"
                ]
                for name in receiver_names
            )
            ports = tuple(
                tuple(zarr[f"receivers/{name}"].attrs["usb_port_path"])
                for name in receiver_names
            )
            session_ids = tuple(
                zarr[f"receivers/{name}"].attrs["hardware_fingerprint_v1"][
                    "fingerprint_session_id"
                ]
                for name in receiver_names
            )
            spans = []
            frame_counts = []
            for name in receiver_names:
                receiver = zarr[f"receivers/{name}"]
                timestamps = np.asarray(
                    receiver["system_timestamp"][:], dtype=np.float64
                )
                if timestamps.size < 2 or not np.all(np.diff(timestamps) > 0):
                    raise ValueError(
                        f"{path}/{name}: timestamps are not strictly increasing"
                    )
                spans.append(float(timestamps[-1] - timestamps[0]))
                frame_counts.append(int(timestamps.size))
            session_span = min(spans)
            total_capture_seconds += session_span
            sessions.append(
                {
                    "path": str(path),
                    "serials": serials,
                    "usb_port_paths": ports,
                    "fingerprint_session_ids": session_ids,
                    "frames_per_receiver": frame_counts,
                    "capture_span_seconds": session_span,
                }
            )
        finally:
            zarr.store.close()

        current_bindings = tuple(sorted(zip(serials, fingerprints, ports, strict=True)))
        if identity_bindings is None:
            identity_bindings = current_bindings
        elif current_bindings != identity_bindings:
            raise ValueError(f"{path}: durable radio identity changed between sessions")
        if len(set(session_ids)) != 1:
            raise ValueError(f"{path}: receivers do not share one fingerprint session")
        fingerprint_session = session_ids[0]
        if fingerprint_session in fingerprint_sessions:
            raise ValueError(f"{path}: fingerprint session was reused across restarts")
        fingerprint_sessions.add(fingerprint_session)

    complete = (
        len(sessions) >= minimum_sessions and total_capture_seconds >= minimum_seconds
    )
    return {
        "status": "pass" if complete else "incomplete",
        "session_count": len(sessions),
        "minimum_sessions": minimum_sessions,
        "total_capture_seconds": total_capture_seconds,
        "minimum_capture_seconds": minimum_seconds,
        "radio_identity_bindings": [
            {
                "serial": serial,
                "stable_fingerprint_sha256": fingerprint,
                "usb_port_path": port,
            }
            for serial, fingerprint, port in (identity_bindings or ())
        ],
        "sessions": sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-list", required=True, type=Path)
    parser.add_argument("--minimum-sessions", type=int, default=2)
    parser.add_argument("--minimum-capture-seconds", type=float, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [Path(line) for line in args.zarr_list.read_text().splitlines() if line]
    try:
        result = validate(paths, args.minimum_sessions, args.minimum_capture_seconds)
    except (ValueError, KeyError, OSError) as error:
        result = {"status": "fail", "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0 if result["status"] == "pass" else 3
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
