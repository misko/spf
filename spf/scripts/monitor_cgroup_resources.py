"""Sample aggregate process memory for a systemd unit's cgroup."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time


MEMORY_FIELDS = ("VmRSS", "RssAnon", "RssFile", "VmSize")


def parse_process_status(text: str) -> dict[str, int]:
    values = {field: 0 for field in MEMORY_FIELDS}
    for line in text.splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in values:
            fields = remainder.split()
            if fields:
                values[name] = int(fields[0])
    return values


def sample_cgroup_processes(
    cgroup: Path, *, proc_root: Path = Path("/proc")
) -> dict[str, int]:
    totals = {field: 0 for field in MEMORY_FIELDS}
    pids = []
    try:
        pids = [int(value) for value in (cgroup / "cgroup.procs").read_text().split()]
    except (FileNotFoundError, ValueError):
        return {"pid_count": 0, **totals}
    for pid in pids:
        try:
            values = parse_process_status((proc_root / str(pid) / "status").read_text())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for field in totals:
            totals[field] += values[field]
    return {"pid_count": len(pids), **totals}


def _unit_property(unit: str, property_name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, f"-p{property_name}", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _unit_is_active(unit: str) -> bool:
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
        ).returncode
        == 0
    )


def _available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/meminfo lacks MemAvailable")


def _artifact_kib(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sk", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])


def monitor(unit: str, output: Path, artifact_root: Path, interval: float) -> None:
    if interval <= 0:
        raise ValueError("interval must be positive")
    control_group = _unit_property(unit, "ControlGroup")
    if not control_group:
        raise RuntimeError(f"{unit}: ControlGroup is empty")
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp_unix",
        "pid",
        "pid_count",
        "rss_kib",
        "rss_anon_kib",
        "rss_file_kib",
        "vmsize_kib",
        "available_kib",
        "artifact_kib",
    )
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        target.flush()
        while _unit_is_active(unit):
            sample = sample_cgroup_processes(cgroup)
            writer.writerow(
                {
                    "timestamp_unix": time.time(),
                    "pid": 0,
                    "pid_count": sample["pid_count"],
                    "rss_kib": sample["VmRSS"],
                    "rss_anon_kib": sample["RssAnon"],
                    "rss_file_kib": sample["RssFile"],
                    "vmsize_kib": sample["VmSize"],
                    "available_kib": _available_kib(),
                    "artifact_kib": _artifact_kib(artifact_root),
                }
            )
            target.flush()
            time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    monitor(args.unit, args.output, args.artifact_root, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
