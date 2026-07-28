"""Write and verify the boot-time direct-USB Pluto readiness manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from spf.hardware_fingerprint import (
    DEFAULT_HMAC_KEY_PATH,
    HARDWARE_FINGERPRINT_SCHEMA,
    HARDWARE_FINGERPRINT_VERSION,
    HardwareFingerprintError,
    collect_hardware_fingerprint,
    load_or_create_hmac_key,
    stable_identity_sha256,
    validate_public_hardware_fingerprint,
)
from spf.scripts.pluto_multi_firmware import (
    discover_runtime_plutos,
    read_passive_device_facts,
)
from spf.scripts.rover_capture_config import resolve_capture_plan


READY_MANIFEST_VERSION = 2
DEFAULT_READY_PATH = Path("/run/spf/direct_usb_ready.json")
DEFAULT_MAPPING_PATH = Path("/home/pi/device_mapping")
DEFAULT_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
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


def _boot_id(path: Path = DEFAULT_BOOT_ID_PATH) -> str:
    value = path.read_text().strip()
    if not value:
        raise ReadyManifestError(f"host boot ID is empty: {path}")
    return value


def _firmware_document(plan) -> dict:
    return {
        "release_tag": plan.firmware_release_tag,
        "asset_name": plan.firmware_asset_name,
        "image_url": plan.firmware_image_url,
        "image_sha256": plan.firmware_image_sha256,
        "firmware_git_sha": plan.firmware_git_sha,
        "gadget_git_sha": plan.gadget_git_sha,
        "boot_mode": plan.firmware_boot_mode,
    }


def _default_device_fact_reader(serial: str) -> dict[str, str]:
    return read_passive_device_facts(
        serial,
        ssh_config=(REPO_ROOT / "data_collection/rover/rover_v3.1/ssh_config"),
        ssh_password=os.environ.get("SPF_PLUTO_SSH_PASSWORD", "analog"),
    )


def build_manifest(
    rover_id: int,
    *,
    config_override: str | Path | None = None,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    hmac_key_path: Path = DEFAULT_HMAC_KEY_PATH,
    boot_id_path: Path = DEFAULT_BOOT_ID_PATH,
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

    firmware = _firmware_document(plan)
    session_id = str(uuid.uuid4())
    host_boot_id = _boot_id(boot_id_path)
    hmac_key = load_or_create_hmac_key(hmac_key_path)
    radios = []
    for device in devices:
        device_facts = _default_device_fact_reader(device.serial)
        fingerprint = collect_hardware_fingerprint(
            device,
            expected_firmware=firmware,
            session_id=session_id,
            hmac_key=hmac_key,
            boot_id_path=boot_id_path,
            device_facts=device_facts,
        )
        direct_usb = fingerprint["direct_usb"]
        radios.append(
            {
                "serial": device.serial,
                "usb_sysfs_name": device.sysfs_name,
                "usb_bus": device.bus,
                "usb_address": device.address,
                "usb_port_path": device.port_path,
                "direct_usb": device.direct_usb,
                "firmware_verified": True,
                **direct_usb,
                "hardware_fingerprint": validate_public_hardware_fingerprint(
                    fingerprint
                ),
            }
        )
    stable_hashes = [
        radio["hardware_fingerprint"]["stable_fingerprint_sha256"] for radio in radios
    ]
    if len(stable_hashes) != len(set(stable_hashes)):
        raise ReadyManifestError(
            "multiple attached radios produced one stable hardware fingerprint"
        )
    spi_nor_hmacs = [
        radio["hardware_fingerprint"]["stable_identity"][
            "spi_nor_unique_id_hmac_sha256"
        ]
        for radio in radios
    ]
    if len(spi_nor_hmacs) != len(set(spi_nor_hmacs)):
        raise ReadyManifestError(
            "multiple attached radios reported one SPI-NOR UniqueID identity"
        )
    hmac_key_ids = {radio["hardware_fingerprint"]["hmac_key_id"] for radio in radios}
    if len(hmac_key_ids) != 1:
        raise ReadyManifestError("attached radio fingerprints used different HMAC keys")

    return {
        "ready_manifest_version": READY_MANIFEST_VERSION,
        "created_at_unix_ns": time.time_ns(),
        "host_boot_id": host_boot_id,
        "fingerprint_session_id": session_id,
        "rover_id": rover_id,
        "spf_git_sha": _spf_git_sha(),
        "config_path": plan.config_path,
        "config_sha256": plan.config_sha256,
        "configured_receiver_count": plan.expected_radios,
        "attached_radio_count": len(devices),
        "device_mapping_sha256": _file_sha256(mapping_path),
        "firmware": firmware,
        "radios": radios,
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
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
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
    boot_id_path: Path = DEFAULT_BOOT_ID_PATH,
) -> dict:
    actual = load_manifest(path)
    plan = resolve_capture_plan(rover_id, config_override)
    devices = discover_runtime_plutos()
    if not mapping_path.is_file():
        raise ReadyManifestError(f"device mapping is missing: {mapping_path}")
    expected_scalars = {
        "rover_id": rover_id,
        "spf_git_sha": _spf_git_sha(),
        "config_path": plan.config_path,
        "config_sha256": plan.config_sha256,
        "configured_receiver_count": plan.expected_radios,
        "attached_radio_count": len(devices),
        "device_mapping_sha256": _file_sha256(mapping_path),
        "firmware": _firmware_document(plan),
        "host_boot_id": _boot_id(boot_id_path),
    }
    if len(devices) != plan.expected_radios:
        raise ReadyManifestError(
            f"configured {plan.expected_radios} receivers but found "
            f"{len(devices)} attached runtime Plutos"
        )
    for key, expected in expected_scalars.items():
        if actual.get(key) != expected:
            raise ReadyManifestError(
                f"ready manifest field {key!r} is stale or mismatched"
            )
    session_id = actual.get("fingerprint_session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ReadyManifestError("ready manifest has no fingerprint session ID")
    actual_radios = actual.get("radios")
    if not isinstance(actual_radios, list) or len(actual_radios) != len(devices):
        raise ReadyManifestError("ready manifest radio list is stale or mismatched")
    radios_by_serial = {
        radio.get("serial"): radio for radio in actual_radios if isinstance(radio, dict)
    }
    if len(radios_by_serial) != len(actual_radios):
        raise ReadyManifestError("ready manifest has duplicate or invalid radio rows")
    stable_hashes = []
    spi_nor_hmacs = []
    optional_dna_hmacs = []
    hmac_key_ids = []
    for device in devices:
        radio = radios_by_serial.get(device.serial)
        if radio is None:
            raise ReadyManifestError(
                f"ready manifest is missing attached radio {device.serial}"
            )
        expected_attachment = {
            "usb_sysfs_name": device.sysfs_name,
            "usb_bus": device.bus,
            "usb_address": device.address,
            "usb_port_path": device.port_path,
            "direct_usb": True,
            "firmware_verified": True,
        }
        for key, expected in expected_attachment.items():
            if radio.get(key) != expected:
                raise ReadyManifestError(
                    f"{device.serial}: ready field {key!r} is stale or mismatched"
                )
        fingerprint = radio.get("hardware_fingerprint")
        if not isinstance(fingerprint, dict):
            raise ReadyManifestError(
                f"{device.serial}: hardware fingerprint is missing"
            )
        try:
            fingerprint = validate_public_hardware_fingerprint(fingerprint)
        except HardwareFingerprintError as error:
            raise ReadyManifestError(
                f"{device.serial}: hardware fingerprint is invalid: {error}"
            ) from error
        if (
            fingerprint.get("schema") != HARDWARE_FINGERPRINT_SCHEMA
            or fingerprint.get("schema_version") != HARDWARE_FINGERPRINT_VERSION
            or fingerprint.get("fingerprint_session_id") != session_id
            or fingerprint.get("host_boot_id") != actual["host_boot_id"]
            or fingerprint.get("fingerprint_timing") != "post_firmware_before_recording"
            or fingerprint.get("acquisition_binding") is not True
            or fingerprint.get("tx_operations_performed") is not False
            or fingerprint.get("compatibility", {}).get("status") != "compatible"
        ):
            raise ReadyManifestError(
                f"{device.serial}: hardware fingerprint is stale or invalid"
            )
        if fingerprint.get("stable_identity", {}).get("pluto_serial") != device.serial:
            raise ReadyManifestError(
                f"{device.serial}: fingerprint serial is mismatched"
            )
        stable_identity = fingerprint["stable_identity"]
        stable_hash = fingerprint.get("stable_fingerprint_sha256")
        if stable_hash != stable_identity_sha256(stable_identity):
            raise ReadyManifestError(
                f"{device.serial}: stable fingerprint hash is invalid"
            )
        stable_hashes.append(stable_hash)
        spi_nor_hmacs.append(stable_identity["spi_nor_unique_id_hmac_sha256"])
        optional_dna_hmac = stable_identity.get("fpga_device_dna_hmac_sha256")
        if optional_dna_hmac is not None:
            optional_dna_hmacs.append(optional_dna_hmac)
        hmac_key_ids.append(fingerprint["hmac_key_id"])
        firmware_session = fingerprint.get("firmware_session", {})
        for key, expected in actual["firmware"].items():
            if firmware_session.get(key) != expected:
                raise ReadyManifestError(
                    f"{device.serial}: fingerprint firmware field {key!r} "
                    "is mismatched"
                )
        if fingerprint.get("direct_usb") != {
            "protocol_min": radio.get("protocol_min"),
            "protocol_max": radio.get("protocol_max"),
            "supported_features": radio.get("supported_features"),
            "capability_flags": radio.get("capability_flags"),
        }:
            raise ReadyManifestError(
                f"{device.serial}: fingerprint capabilities are mismatched"
            )
        attachment = fingerprint.get("attachment", {})
        if (
            attachment.get("usb_bus") != device.bus
            or attachment.get("usb_address") != device.address
            or attachment.get("usb_port_path") != device.port_path
        ):
            raise ReadyManifestError(
                f"{device.serial}: fingerprint attachment is stale"
            )
    if len(stable_hashes) != len(set(stable_hashes)):
        raise ReadyManifestError(
            "multiple attached radios have one stable hardware fingerprint"
        )
    if len(spi_nor_hmacs) != len(set(spi_nor_hmacs)):
        raise ReadyManifestError(
            "multiple attached radios have one SPI-NOR UniqueID identity"
        )
    if len(optional_dna_hmacs) != len(set(optional_dna_hmacs)):
        raise ReadyManifestError(
            "multiple attached radios have one optional FPGA Device DNA identity"
        )
    if len(set(hmac_key_ids)) != 1:
        raise ReadyManifestError("attached radio fingerprints use different HMAC keys")
    return actual


def firmware_for_serial(manifest: dict, serial: str) -> dict | None:
    matching = [
        radio for radio in manifest.get("radios", []) if radio.get("serial") == serial
    ]
    if len(matching) != 1:
        return None
    result = dict(manifest.get("firmware", {}))
    result.update(matching[0])
    result.pop("hardware_fingerprint", None)
    return result


def fingerprint_for_serial(manifest: dict, serial: str) -> dict | None:
    matching = [
        radio for radio in manifest.get("radios", []) if radio.get("serial") == serial
    ]
    if len(matching) != 1:
        return None
    fingerprint = matching[0].get("hardware_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    try:
        return validate_public_hardware_fingerprint(fingerprint)
    except HardwareFingerprintError:
        return None


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
        HardwareFingerprintError,
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
