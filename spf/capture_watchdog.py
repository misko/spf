"""Independent low-overhead evidence journal for rover radio captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable


WATCHDOG_VERSION = 1
DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_MAXIMUM_BYTES = 16 * 1024 * 1024
DEFAULT_OUTPUT = Path("/home/pi/preflight/capture_watchdog.jsonl")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError, PermissionError):
        return None


def _read_key_values(path: Path) -> dict[str, str]:
    text = _read_text(path)
    if text is None:
        return {}
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key] = value.strip()
    return result


def _process_sample(pid: int, proc_root: Path = Path("/proc")) -> dict:
    process_root = proc_root / str(pid)
    status = _read_key_values(process_root / "status")
    stat = _read_text(process_root / "stat")
    alive = bool(status or stat)
    result = {"pid": int(pid), "alive": alive}
    if not alive:
        return result
    if stat is not None and ") " in stat:
        result["state"] = stat.split(") ", 1)[1].split()[0]
    if "Threads" in status:
        result["threads"] = int(status["Threads"].split()[0])
    if "VmRSS" in status:
        result["rss_bytes"] = int(status["VmRSS"].split()[0]) * 1024
    if "VmSize" in status:
        result["virtual_bytes"] = int(status["VmSize"].split()[0]) * 1024
    return result


def _psi_sample(proc_root: Path = Path("/proc")) -> dict:
    result = {}
    for resource in ("cpu", "memory", "io"):
        text = _read_text(proc_root / "pressure" / resource)
        if text is None:
            continue
        resource_sample = {}
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            resource_sample[fields[0]] = {
                key: float(value)
                for key, value in (field.split("=", 1) for field in fields[1:])
                if key.startswith("avg")
            }
        result[resource] = resource_sample
    return result


def _temperature_sample(thermal_root: Path = Path("/sys/class/thermal")):
    values = []
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        raw = _read_text(zone / "temp")
        if raw is None:
            continue
        try:
            celsius = float(raw) / 1000.0
        except ValueError:
            continue
        values.append(
            {
                "zone": zone.name,
                "type": _read_text(zone / "type"),
                "celsius": celsius,
            }
        )
    return values


def _throttled_sample() -> str | None:
    executable = shutil.which("vcgencmd")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "get_throttled"],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _usb_sample(root: Path, expected_plutos: int) -> dict:
    devices = []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            vendor = _read_text(entry / "idVendor")
            product = _read_text(entry / "idProduct")
            if vendor != "0456" or product != "b673":
                continue
            device = {
                "sysfs_name": entry.name,
                "serial": _read_text(entry / "serial"),
                "bus": _read_text(entry / "busnum"),
                "address": _read_text(entry / "devnum"),
            }
            devices.append(device)
    observed = len(devices)
    return {
        "expected_plutos": int(expected_plutos),
        "observed_plutos": observed,
        "missing": observed != int(expected_plutos),
        "devices": devices,
    }


def _capture_status_sample(path: Path, now: float) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"status_readable": False, "status_path": str(path)}
    result = {
        "status_readable": True,
        "status_path": str(path),
        "capture_name": payload.get("capture_name"),
        "state": payload.get("state"),
        "records_written_by_receiver": payload.get("records_written_by_receiver"),
        "incident_id": payload.get("incident_id"),
        "error_source": payload.get("error_source"),
        "error_type": payload.get("error_type"),
        "error_message": payload.get("error_message"),
    }
    updated = payload.get("updated_unix")
    if isinstance(updated, (int, float)):
        result["status_age_seconds"] = max(0.0, now - float(updated))
    return result


def collect_watchdog_sample(
    *,
    pid: int,
    status_path: Path,
    storage_path: Path,
    expected_plutos: int,
    usb_sysfs_root: Path = Path("/sys/bus/usb/devices"),
    proc_root: Path = Path("/proc"),
    thermal_root: Path = Path("/sys/class/thermal"),
    wall_time: Callable[[], float] = time.time,
    monotonic_time: Callable[[], float] = time.monotonic,
    previous_monotonic: float | None = None,
    expected_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> dict:
    """Collect one read-only snapshot whose domains support root-cause triage."""

    now = float(wall_time())
    monotonic_now = float(monotonic_time())
    disk = shutil.disk_usage(storage_path)
    host = {
        "load_average": list(os.getloadavg()),
        "psi": _psi_sample(proc_root),
        "temperatures": _temperature_sample(thermal_root),
        "throttled": _throttled_sample(),
    }
    if previous_monotonic is not None:
        host["watchdog_scheduling_gap_seconds"] = max(
            0.0, monotonic_now - float(previous_monotonic)
        )
    sample = {
        "watchdog_version": WATCHDOG_VERSION,
        "unix_time": now,
        "monotonic_time": monotonic_now,
        "process": _process_sample(pid, proc_root),
        "host": host,
        "usb": _usb_sample(usb_sysfs_root, expected_plutos),
        "storage": {
            "path": str(storage_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "capture": _capture_status_sample(status_path, now),
    }
    conditions = []
    if sample["usb"]["missing"]:
        conditions.append("pluto_count_mismatch")
    if disk.free < 1024 * 1024 * 1024:
        conditions.append("storage_below_1gib")
    gap = host.get("watchdog_scheduling_gap_seconds")
    if gap is not None and gap > 2 * float(expected_interval_seconds):
        conditions.append("watchdog_scheduling_gap")
    if sample["process"].get("state") == "D":
        conditions.append("capture_process_uninterruptible_io")
    sample["conditions"] = conditions
    return sample


def rotate_jsonl_if_needed(path: Path, *, maximum_bytes: int) -> Path | None:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    if size <= maximum_bytes:
        return None
    rotated = path.with_name(path.name + ".1")
    os.replace(path, rotated)
    return rotated


def monitor(
    *,
    pid: int,
    status_path: Path,
    storage_path: Path,
    output: Path,
    expected_plutos: int,
    interval_seconds: float,
    maximum_bytes: int,
) -> int:
    if pid < 1:
        raise ValueError("pid must be positive")
    if expected_plutos < 0:
        raise ValueError("expected_plutos cannot be negative")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(output, maximum_bytes=maximum_bytes)
    stopping = False

    def stop(_signal_number, _frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {
        number: signal.signal(number, stop)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    previous_monotonic = None
    samples_since_sync = 0
    try:
        with output.open("a", encoding="utf-8", buffering=1) as journal:
            while not stopping:
                sample = collect_watchdog_sample(
                    pid=pid,
                    status_path=status_path,
                    storage_path=storage_path,
                    expected_plutos=expected_plutos,
                    previous_monotonic=previous_monotonic,
                    expected_interval_seconds=interval_seconds,
                )
                previous_monotonic = sample["monotonic_time"]
                journal.write(json.dumps(sample, sort_keys=True) + "\n")
                journal.flush()
                samples_since_sync += 1
                if samples_since_sync >= 5 or not sample["process"]["alive"]:
                    os.fsync(journal.fileno())
                    samples_since_sync = 0
                if not sample["process"]["alive"]:
                    break
                time.sleep(interval_seconds)
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    return 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("monitor")
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--status-file", type=Path, required=True)
    command.add_argument("--storage-path", type=Path, required=True)
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--expected-plutos", type=int, required=True)
    command.add_argument(
        "--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS
    )
    command.add_argument("--maximum-bytes", type=int, default=DEFAULT_MAXIMUM_BYTES)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.command == "monitor":
        return monitor(
            pid=args.pid,
            status_path=args.status_file,
            storage_path=args.storage_path,
            output=args.output,
            expected_plutos=args.expected_plutos,
            interval_seconds=args.interval_seconds,
            maximum_bytes=args.maximum_bytes,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
