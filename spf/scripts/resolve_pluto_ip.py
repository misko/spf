"""Resolve a Pluto's current LAN address by immutable hardware serial."""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import re
import subprocess
from collections.abc import Callable, Iterable


SERIAL_RE = re.compile(r"^hw_serial:\s*(\S+)\s*$", re.MULTILINE)


def neighbor_candidates(interface: str = "eth0") -> tuple[str, ...]:
    result = subprocess.run(
        ["ip", "-4", "neigh", "show", "dev", interface],
        check=True,
        text=True,
        capture_output=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields or any(state in fields for state in ("FAILED", "INCOMPLETE")):
            continue
        try:
            address = str(ipaddress.ip_address(fields[0]))
        except ValueError:
            continue
        candidates.append(address)
    return tuple(sorted(set(candidates), key=ipaddress.ip_address))


def probe_iio_serial(host: str, timeout_seconds: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            ["iio_attr", "-u", f"ip:{host}", "-C", "hw_serial"],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    match = SERIAL_RE.search(result.stdout)
    return None if match is None else match.group(1)


def resolve_pluto_ip(
    serial: str,
    candidates: Iterable[str],
    *,
    probe: Callable[[str], str | None] = probe_iio_serial,
) -> str:
    ordered = tuple(dict.fromkeys(candidates))
    if not ordered:
        raise RuntimeError("no candidate LAN addresses are available")
    matches = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(16, len(ordered))
    ) as executor:
        futures = {executor.submit(probe, host): host for host in ordered}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            if future.result() == serial:
                matches.append(host)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one LAN Pluto with serial {serial}, found "
            f"{sorted(matches)} among {list(ordered)}"
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a Pluto LAN address by immutable hw_serial"
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--preferred-host")
    parser.add_argument("--interface", default="eth0")
    args = parser.parse_args(argv)
    candidates = []
    if args.preferred_host:
        candidates.append(args.preferred_host)
    candidates.extend(neighbor_candidates(args.interface))
    print(resolve_pluto_ip(args.serial, candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
