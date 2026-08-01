"""Atomic, durable operator status for a rover radio capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Sequence


CAPTURE_STATUS_VERSION = 1


def _atomic_json_write(path: Path, payload: dict) -> None:
    """Replace one JSON file atomically and flush the file plus parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary_name = target.name
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


class CaptureStatusWriter:
    """Publish bounded-rate status while retaining exact per-receiver progress."""

    def __init__(
        self,
        path: Path | str,
        *,
        capture_name: str,
        expected_records_per_receiver: int,
        receiver_count: int,
        minimum_write_interval_seconds: float = 5.0,
        wall_time: Callable[[], float] = time.time,
        monotonic_time: Callable[[], float] = time.monotonic,
    ):
        if expected_records_per_receiver < 1:
            raise ValueError("expected record count must be positive")
        if receiver_count < 1:
            raise ValueError("receiver count must be positive")
        if minimum_write_interval_seconds < 0:
            raise ValueError("minimum write interval cannot be negative")
        self.path = Path(path)
        self.capture_name = capture_name
        self.expected_records_per_receiver = int(expected_records_per_receiver)
        self.receiver_count = int(receiver_count)
        self.minimum_write_interval_seconds = float(minimum_write_interval_seconds)
        self.wall_time = wall_time
        self.monotonic_time = monotonic_time
        self.started_unix = float(wall_time())
        self.started_monotonic = float(monotonic_time())
        self.last_write_monotonic: float | None = None
        self.last_payload: dict | None = None

    def publish(
        self,
        state: str,
        records_written_by_receiver: Sequence[int],
        *,
        error: BaseException | None = None,
        artifact: str | None = None,
        force: bool = False,
    ) -> bool:
        counts = [int(value) for value in records_written_by_receiver]
        if len(counts) != self.receiver_count:
            raise ValueError(
                f"expected {self.receiver_count} receiver counts, found {counts}"
            )
        if any(value < 0 for value in counts):
            raise ValueError("record counts cannot be negative")
        now_monotonic = float(self.monotonic_time())
        final_state = state in {"complete", "failed"}
        if (
            not force
            and not final_state
            and self.last_write_monotonic is not None
            and now_monotonic - self.last_write_monotonic
            < self.minimum_write_interval_seconds
        ):
            return False

        elapsed = max(0.0, now_monotonic - self.started_monotonic)
        common_records = min(counts)
        frames_per_second = common_records / elapsed if elapsed > 0 else None
        remaining = max(0, self.expected_records_per_receiver - common_records)
        eta = (
            remaining / frames_per_second
            if frames_per_second is not None and frames_per_second > 0
            else None
        )
        payload = {
            "capture_status_version": CAPTURE_STATUS_VERSION,
            "capture_name": self.capture_name,
            "state": state,
            "started_unix": self.started_unix,
            "updated_unix": float(self.wall_time()),
            "elapsed_seconds": elapsed,
            "expected_records_per_receiver": self.expected_records_per_receiver,
            "records_written_by_receiver": counts,
            "common_records_written": common_records,
            "frames_per_second": frames_per_second,
            "estimated_remaining_seconds": eta,
        }
        if artifact is not None:
            payload["artifact"] = artifact
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
            error_number = getattr(error, "errno", None)
            if error_number is not None:
                payload["error_errno"] = int(error_number)
        _atomic_json_write(self.path, payload)
        self.last_write_monotonic = now_monotonic
        self.last_payload = payload
        return True


def mark_failed(path: Path | str, *, exit_code: int) -> dict:
    """Make a launcher-observed process failure durable without hiding prior state."""

    status_path = Path(path)
    try:
        payload = json.loads(status_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {
            "capture_status_version": CAPTURE_STATUS_VERSION,
            "capture_name": None,
            "records_written_by_receiver": [],
        }
    payload.update(
        {
            "state": "failed",
            "updated_unix": time.time(),
            "launcher_exit_code": int(exit_code),
        }
    )
    payload.setdefault("error_type", "CaptureProcessExit")
    payload.setdefault(
        "error_message", f"capture process exited with status {int(exit_code)}"
    )
    _atomic_json_write(status_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    failed = subparsers.add_parser("mark-failed")
    failed.add_argument("--path", type=Path, required=True)
    failed.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.command == "mark-failed":
        print(
            json.dumps(
                mark_failed(args.path, exit_code=args.exit_code),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
