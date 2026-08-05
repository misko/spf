"""Which physical radio is r0, which is r1, and is each one still there?

Answering that took a code read on 2026-08-05. A Pluto on Rover 1 was dropping
off the USB bus minutes after boot, and the only symptom was::

    Exception: No device found

Nothing said which receiver had gone, which USB port it was on, or which serial
to look for on the bench. The chain that answers it spans three files:

    capture config    receivers[i]["receiver-port"]        -> r0, r1 in order
    ~/device_mapping  "<port> <dev>" per attached radio    -> port -> dev
    collector         pluto://usb:1.<dev>.5                -> dev  -> URI

and the kernel calls the same device ``usb 1-1.<port>``. This prints all of it
at once, marks anything missing, and exits non-zero when a configured receiver
has no radio behind it.

Presence is checked against ``lsusb -t``, not ``lsusb``: a half-dropped Pluto
still answers the latter while no longer presenting its mass-storage/IIO
interface, which is exactly the state that produces "No device found".

Usage:
  python -m spf.scripts.rover_radio_map --config <yaml> [--device-mapping PATH]
                                        [--ready-manifest PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import yaml

DEFAULT_DEVICE_MAPPING = "/home/pi/device_mapping"
DEFAULT_READY_MANIFEST = "/run/spf/direct_usb_ready.json"
# Matches the URI the collector builds in mavlink_radio_collection.py.
URI_TEMPLATE = "pluto://usb:1.{dev}.5"


def read_device_mapping(path: str) -> dict[int, str]:
    """port -> dev, from the file the collector reads."""
    mapping: dict[int, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
            elif len(parts) == 3:
                # bus/dev form; the collector accepts it, so accept it here.
                mapping[int(parts[0])] = parts[2]
    return mapping


def present_ports() -> dict[int, str]:
    """port -> dev for radios currently presenting a mass-storage interface."""
    try:
        out = subprocess.run(
            ["lsusb", "-t"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return {}
    found = {}
    for line in out.splitlines():
        if "usb-storage" not in line:
            continue
        match = re.search(r"Port (\d+): Dev (\d+)", line)
        if match:
            found[int(match.group(1))] = match.group(2)
    return found


def serials_by_uri(path: str) -> dict[str, str]:
    """iio_uri -> pluto serial, from the boot firmware attestation."""
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}

    out: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            uri = node.get("iio_uri")
            serial = node.get("pluto_serial") or node.get("serial")
            if isinstance(uri, str) and isinstance(serial, str):
                out[uri] = serial
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(manifest)
    return out


def build(config_path: str, mapping_path: str, manifest_path: str) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        mapping = read_device_mapping(mapping_path)
        mapping_error = None
    except OSError as error:
        mapping, mapping_error = {}, str(error)

    live = present_ports()
    serials = serials_by_uri(manifest_path)

    rows = []
    for index, receiver in enumerate(config.get("receivers", [])):
        port = receiver.get("receiver-port")
        dev = mapping.get(port)
        uri = URI_TEMPLATE.format(dev=dev) if dev else None
        live_dev = live.get(port)
        rows.append(
            {
                "receiver": f"r{index}",
                "receiver_port": port,
                "mapped_dev": dev,
                "live_dev": live_dev,
                "uri": uri,
                "kernel_name": f"usb 1-1.{port}" if port is not None else None,
                "serial": serials.get(uri.replace("pluto://", ""), None)
                if uri
                else None,
                "present": live_dev is not None,
                # A mapping written before a re-enumeration points at a device
                # number that no longer exists; the collector then opens the
                # wrong URI or none at all.
                "mapping_stale": (
                    live_dev is not None and dev is not None and live_dev != dev
                ),
                "antenna_spacing_m": receiver.get("antenna-spacing-m"),
            }
        )
    return {
        "config": config_path,
        "device_mapping": mapping_path,
        "device_mapping_error": mapping_error,
        "receivers": rows,
        "live_ports": live,
    }


def render(report: dict) -> int:
    print(f"\n  config  : {report['config']}")
    print(f"  mapping : {report['device_mapping']}")
    if report["device_mapping_error"]:
        print(f"  WARNING : cannot read mapping: {report['device_mapping_error']}")
    print()
    header = (f"  {'rx':<4}{'port':>5}{'dev':>5}  {'kernel':<11}{'uri':<22}"
              f"{'serial':<36}{'state'}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    missing, stale = [], []
    for row in report["receivers"]:
        if not row["present"]:
            state, mark = "MISSING", "✗"
            missing.append(row["receiver"])
        elif row["mapping_stale"]:
            state = f"STALE MAP (live dev {row['live_dev']})"
            mark = "!"
            stale.append(row["receiver"])
        else:
            state, mark = "ok", "✓"
        print(f"  {mark} {row['receiver']:<2}{str(row['receiver_port']):>5}"
              f"{str(row['mapped_dev'] or '-'):>5}  {row['kernel_name'] or '-':<11}"
              f"{(row['uri'] or '-'):<22}{(row['serial'] or '-'):<36}{state}")

    print()
    if missing:
        print(f"  FAIL: no radio behind {', '.join(missing)}.")
        print("    A Pluto can vanish from the bus minutes after boot and the")
        print("    collector reports only 'No device found'. Check:")
        print("      dmesg -T | grep -i 'usb disconnect'")
        print("    then reseat or replace that port's USB cable.")
        return 1
    if stale:
        print(f"  WARN: {', '.join(stale)} mapped to a device number that has "
              "changed.")
        print("    The collector will open the wrong URI. Refresh with:")
        print("      rover radio remap")
        return 1
    print("  PASS: every configured receiver has a radio behind it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device-mapping", default=DEFAULT_DEVICE_MAPPING)
    parser.add_argument("--ready-manifest", default=DEFAULT_READY_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build(args.config, args.device_mapping, args.ready_manifest)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if all(r["present"] for r in report["receivers"]) else 1
    return render(report)


if __name__ == "__main__":
    raise SystemExit(main())
