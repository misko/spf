"""Write and verify the boot-time direct-USB Pluto readiness manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from spf.scripts.pluto_multi_firmware import discover_runtime_plutos
from spf.scripts.rover_capture_config import resolve_capture_plan


READY_MANIFEST_VERSION = 1
DEFAULT_READY_PATH = Path("/run/spf/direct_usb_ready.json")
DEFAULT_MAPPING_PATH = Path("/home/pi/device_mapping")
REPO_ROOT = Path(__file__).resolve().parents[2]
V7_REQUIRED_FEATURES = 0x37


class ReadyManifestError(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spf_git_sha() -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT}",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            "--verify",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _probe_v7_capabilities(serial: str) -> dict:
    from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver

    with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as receiver:
        capabilities = receiver.capabilities
    supported_features = int(capabilities.supported_features)
    if not capabilities.protocol_min <= 2 <= capabilities.protocol_max:
        raise ReadyManifestError(f"{serial}: direct-USB protocol v2 is not supported")
    if supported_features & V7_REQUIRED_FEATURES != V7_REQUIRED_FEATURES:
        raise ReadyManifestError(
            f"{serial}: required V7 metadata features are unavailable"
        )
    return {
        "protocol_min": capabilities.protocol_min,
        "protocol_max": capabilities.protocol_max,
        "supported_features": supported_features,
        "capability_flags": int(capabilities.capability_flags),
    }


def build_manifest(
    rover_id: int,
    *,
    config_override: str | Path | None = None,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> dict:
    plan = resolve_capture_plan(rover_id, config_override)
    devices = discover_runtime_plutos()
    if len(devices) != plan.expected_radios:
        raise ReadyManifestError(
            f"configured {plan.expected_radios} receivers but found "
            f"{len(devices)} attached runtime Plutos"
        )
    if any(not device.direct_usb for device in devices):
        missing = [device.serial for device in devices if not device.direct_usb]
        raise ReadyManifestError(
            f"direct-USB interface 6 is absent on radios: {missing}"
        )
    if not mapping_path.is_file():
        raise ReadyManifestError(f"device mapping is missing: {mapping_path}")
    mapping_rows = [
        line.strip() for line in mapping_path.read_text().splitlines() if line.strip()
    ]
    if len(mapping_rows) != plan.expected_radios:
        raise ReadyManifestError(
            f"device mapping has {len(mapping_rows)} rows, "
            f"expected {plan.expected_radios}"
        )

    return {
        "ready_manifest_version": READY_MANIFEST_VERSION,
        "rover_id": rover_id,
        "spf_git_sha": _spf_git_sha(),
        "config_path": plan.config_path,
        "config_sha256": plan.config_sha256,
        "configured_receiver_count": plan.expected_radios,
        "attached_radio_count": len(devices),
        "device_mapping_sha256": _file_sha256(mapping_path),
        "firmware": {
            "release_tag": plan.firmware_release_tag,
            "asset_name": plan.firmware_asset_name,
            "image_url": plan.firmware_image_url,
            "image_sha256": plan.firmware_image_sha256,
            "firmware_git_sha": plan.firmware_git_sha,
            "gadget_git_sha": plan.gadget_git_sha,
            "boot_mode": plan.firmware_boot_mode,
        },
        "radios": [
            {
                "serial": device.serial,
                "usb_sysfs_name": device.sysfs_name,
                "usb_bus": device.bus,
                "usb_port_path": device.port_path,
                "direct_usb": device.direct_usb,
                "firmware_verified": True,
                **_probe_v7_capabilities(device.serial),
            }
            for device in devices
        ],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_manifest(path: Path = DEFAULT_READY_PATH) -> dict:
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ReadyManifestError(f"ready manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReadyManifestError(f"ready manifest is invalid JSON: {path}") from exc
    if manifest.get("ready_manifest_version") != READY_MANIFEST_VERSION:
        raise ReadyManifestError(
            "unsupported ready manifest version: "
            f"{manifest.get('ready_manifest_version')}"
        )
    return manifest


def verify_manifest(
    rover_id: int,
    *,
    config_override: str | Path | None = None,
    path: Path = DEFAULT_READY_PATH,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> dict:
    expected = build_manifest(
        rover_id,
        config_override=config_override,
        mapping_path=mapping_path,
    )
    actual = load_manifest(path)
    for key in (
        "rover_id",
        "spf_git_sha",
        "config_path",
        "config_sha256",
        "configured_receiver_count",
        "attached_radio_count",
        "device_mapping_sha256",
        "firmware",
        "radios",
    ):
        if actual.get(key) != expected.get(key):
            raise ReadyManifestError(
                f"ready manifest field {key!r} is stale or mismatched"
            )
    return actual


def firmware_for_serial(manifest: dict, serial: str) -> dict | None:
    matching = [
        radio for radio in manifest.get("radios", []) if radio.get("serial") == serial
    ]
    if len(matching) != 1:
        return None
    result = dict(manifest.get("firmware", {}))
    result.update(matching[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify", "show"))
    parser.add_argument("--rover-id", required=True, type=int)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, default=DEFAULT_READY_PATH)
    parser.add_argument("--device-mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    args = parser.parse_args()
    try:
        if args.command == "write":
            manifest = build_manifest(
                args.rover_id,
                config_override=args.config,
                mapping_path=args.device_mapping,
            )
            write_manifest(args.output, manifest)
        elif args.command == "verify":
            manifest = verify_manifest(
                args.rover_id,
                config_override=args.config,
                path=args.output,
                mapping_path=args.device_mapping,
            )
        else:
            manifest = load_manifest(args.output)
    except (
        ReadyManifestError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
