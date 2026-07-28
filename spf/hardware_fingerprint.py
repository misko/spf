"""Passive, privacy-preserving hardware fingerprints for Rover Plutos."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Callable

from spf.scripts.pluto_multi_firmware import UsbPluto
from spf.sdrpluto.direct_usb_protocol import (
    CapabilityFlags,
    HardwareIdentityFlags,
    HardwareIdentityV1,
)


HARDWARE_FINGERPRINT_SCHEMA = "spf.hardware_compatibility_fingerprint"
HARDWARE_FINGERPRINT_VERSION = 1
DEFAULT_HMAC_KEY_PATH = Path("/etc/spf/hardware_fingerprint_hmac.key")
V7_REQUIRED_FEATURES = 0x37
_MAX_TEXT_BYTES = 512
_IIO_ATTRIBUTE_ALLOWLIST = (
    "hw_model",
    "hw_model_variant",
    "hw_serial",
    "fw_version",
    "ad9361-phy,xo_correction",
    "ad9361-phy,model",
    "local,kernel",
    "usb,idVendor",
    "usb,idProduct",
    "usb,release",
    "usb,vendor",
    "usb,product",
    "usb,serial",
    "usb,libusb",
)
_USB_ATTRIBUTE_ALLOWLIST = (
    "idVendor",
    "idProduct",
    "bcdDevice",
    "manufacturer",
    "product",
    "serial",
    "busnum",
    "devnum",
    "devpath",
    "bNumConfigurations",
    "speed",
)
_DEVICE_FACT_ALLOWLIST = (
    "device_tree_model",
    "memory_total_kib",
    "mtd0_size_bytes",
    "mtd1_size_bytes",
    "sd_present",
    "uboot_attr_name",
    "uboot_attr_val",
    "uboot_compatible",
    "uboot_mode",
    "device_fw",
    "linux_version",
    "uboot_version",
)


class HardwareFingerprintError(RuntimeError):
    """A passive fingerprint could not satisfy its fail-closed contract."""


def _safe_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise HardwareFingerprintError(f"{label} exceeds {_MAX_TEXT_BYTES} bytes")
    if any(ord(character) < 0x20 and character not in "\t" for character in value):
        raise HardwareFingerprintError(f"{label} contains control characters")
    return value


def _canonical_json(document: Any) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256_document(document: Any) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def stable_identity_sha256(stable_identity: dict[str, Any]) -> str:
    return _sha256_document(stable_identity)


def load_or_create_hmac_key(path: Path = DEFAULT_HMAC_KEY_PATH) -> bytes:
    """Load a root-owned key, creating one atomically on first provisioning."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        pass
    else:
        try:
            key = secrets.token_bytes(32)
            os.write(descriptor, key)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise HardwareFingerprintError(f"HMAC key is not a regular file: {path}")
    if info.st_mode & 0o077:
        raise HardwareFingerprintError(
            f"HMAC key permissions are too broad: {path} "
            f"({stat.S_IMODE(info.st_mode):04o})"
        )
    key = path.read_bytes()
    if len(key) < 32:
        raise HardwareFingerprintError("HMAC key must contain at least 32 bytes")
    return key


def hmac_key_id(key: bytes) -> str:
    return hashlib.sha256(b"spf-hardware-fingerprint-key-v1\0" + key).hexdigest()[:16]


def fpga_dna_hmac(key: bytes, fpga_device_dna: int) -> str:
    if fpga_device_dna <= 0 or fpga_device_dna >> 57:
        raise HardwareFingerprintError("FPGA Device DNA is outside the 57-bit range")
    message = b"spf-fpga-device-dna-v1\0" + fpga_device_dna.to_bytes(
        8, byteorder="little"
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def spi_nor_uid_hmac(key: bytes, spi_nor_unique_id: str) -> str:
    serial = _safe_text(spi_nor_unique_id, label="SPI-NOR UniqueID")
    if not serial:
        raise HardwareFingerprintError("SPI-NOR UniqueID is empty")
    message = b"spf-spi-nor-unique-id-v1\0" + serial.encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _read_usb_attributes(device: UsbPluto, usb_root: Path) -> dict[str, str]:
    device_path = Path(usb_root) / device.sysfs_name
    result: dict[str, str] = {}
    for name in _USB_ATTRIBUTE_ALLOWLIST:
        path = device_path / name
        if path.is_file():
            result[name] = _safe_text(path.read_text(), label=f"USB {name}")
    required = {
        "idVendor": "0456",
        "idProduct": "b673",
        "serial": device.serial,
        "busnum": str(device.bus),
        "devpath": device.port_path,
    }
    mismatches = {
        name: {"expected": expected, "actual": result.get(name)}
        for name, expected in required.items()
        if result.get(name, "").lower() != expected.lower()
    }
    if mismatches:
        raise HardwareFingerprintError(
            f"{device.serial}: USB identity mismatch: {mismatches}"
        )
    return result


def _read_iio_attributes(uri: str) -> dict[str, str]:
    try:
        import iio
    except ImportError as exc:
        raise HardwareFingerprintError(
            "pylibiio is required to collect the Pluto fingerprint"
        ) from exc
    try:
        context = iio.Context(uri)
    except Exception as exc:
        raise HardwareFingerprintError(
            f"failed to open IIO context {uri}: {exc}"
        ) from exc
    try:
        raw = context.attrs
        return {
            name: _safe_text(raw[name], label=f"IIO {name}")
            for name in _IIO_ATTRIBUTE_ALLOWLIST
            if name in raw
        }
    finally:
        del context


def _query_direct_usb_identity(
    device: UsbPluto,
) -> tuple[dict[str, int], HardwareIdentityV1]:
    from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver

    with PlutoDirectUsbReceiver(serial=device.serial, protocol_version=2) as receiver:
        capabilities = receiver.capabilities
        identity = receiver.query_hardware_identity()
    return (
        {
            "protocol_min": int(capabilities.protocol_min),
            "protocol_max": int(capabilities.protocol_max),
            "supported_features": int(capabilities.supported_features),
            "capability_flags": int(capabilities.capability_flags),
        },
        identity,
    )


def _sanitize_device_facts(device_facts: dict[str, Any] | None) -> dict[str, str]:
    if device_facts is None:
        return {}
    unknown = set(device_facts) - set(_DEVICE_FACT_ALLOWLIST)
    if unknown:
        raise HardwareFingerprintError(
            f"device facts contain non-allowlisted keys: {sorted(unknown)}"
        )
    return {
        name: _safe_text(value, label=f"device fact {name}")
        for name, value in device_facts.items()
        if value is not None
    }


def _compatibility_failures(
    *,
    device: UsbPluto,
    usb: dict[str, str],
    iio_attrs: dict[str, str],
    capabilities: dict[str, int],
    hardware_identity: HardwareIdentityV1,
    device_facts: dict[str, str],
    expected_gadget_git_sha: str,
) -> list[str]:
    failures: list[str] = []
    if not device.direct_usb:
        failures.append("direct_usb_interface_missing")
    if iio_attrs.get("hw_serial") != device.serial:
        failures.append("iio_serial_mismatch")
    if iio_attrs.get("usb,serial") != device.serial:
        failures.append("iio_usb_serial_mismatch")
    if iio_attrs.get("usb,idVendor", "").lower() != usb["idVendor"].lower():
        failures.append("iio_usb_vendor_mismatch")
    if iio_attrs.get("usb,idProduct", "").lower() != usb["idProduct"].lower():
        failures.append("iio_usb_product_mismatch")
    if iio_attrs.get("ad9361-phy,model") != "ad9361":
        failures.append("ad9361_model_not_enabled")
    if iio_attrs.get("ad9361-phy,xo_correction") != "40000000":
        failures.append("unexpected_reference_clock")
    if not capabilities["protocol_min"] <= 2 <= capabilities["protocol_max"]:
        failures.append("direct_usb_protocol_v2_missing")
    if (
        capabilities["supported_features"] & V7_REQUIRED_FEATURES
        != V7_REQUIRED_FEATURES
    ):
        failures.append("v7_metadata_capabilities_missing")
    if not capabilities["capability_flags"] & int(CapabilityFlags.HARDWARE_IDENTITY):
        failures.append("hardware_identity_capability_missing")
    if not hardware_identity.flags & HardwareIdentityFlags.GADGET_BUILD_ID_VALID:
        failures.append("gadget_build_identity_missing")
    if hardware_identity.gadget_build_id != expected_gadget_git_sha:
        failures.append("gadget_build_id_mismatch")
    if device_facts and device_facts.get("uboot_mode") != "2r2t":
        failures.append("uboot_not_configured_2r2t")
    if device_facts and device_facts.get("uboot_compatible") != "ad9361":
        failures.append("uboot_not_configured_ad9361")
    return failures


def collect_hardware_fingerprint(
    device: UsbPluto,
    *,
    expected_firmware: dict[str, Any],
    session_id: str,
    hmac_key: bytes,
    usb_root: Path = Path("/sys/bus/usb/devices"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    device_facts: dict[str, Any] | None = None,
    captured_at_unix_ns: int | None = None,
    iio_reader: Callable[[str], dict[str, str]] = _read_iio_attributes,
    direct_identity_reader: Callable[
        [UsbPluto], tuple[dict[str, int], HardwareIdentityV1]
    ] = _query_direct_usb_identity,
) -> dict[str, Any]:
    """Collect one radio after firmware loading, without starting either DMA path."""

    if device.address is None:
        device_path = Path(usb_root) / device.sysfs_name / "devnum"
        address = int(device_path.read_text().strip())
    else:
        address = device.address
    uri = f"usb:{device.bus}.{address}.5"
    usb = _read_usb_attributes(device, usb_root)
    iio_attrs = iio_reader(uri)
    capabilities, hardware_identity = direct_identity_reader(device)
    sanitized_facts = _sanitize_device_facts(device_facts)
    expected_gadget_sha = _safe_text(
        expected_firmware.get("gadget_git_sha", ""),
        label="expected gadget Git SHA",
    )
    if len(expected_gadget_sha) != 40:
        raise HardwareFingerprintError("expected gadget Git SHA is not 40 characters")
    failures = _compatibility_failures(
        device=device,
        usb=usb,
        iio_attrs=iio_attrs,
        capabilities=capabilities,
        hardware_identity=hardware_identity,
        device_facts=sanitized_facts,
        expected_gadget_git_sha=expected_gadget_sha,
    )
    if failures:
        raise HardwareFingerprintError(
            f"{device.serial}: incompatible fingerprint: {failures}"
        )

    spi_nor_hmac = spi_nor_uid_hmac(hmac_key, device.serial)
    stable_identity = {
        "pluto_serial": device.serial,
        "spi_nor_unique_id_hmac_sha256": spi_nor_hmac,
        "usb_vendor_id": usb["idVendor"].lower(),
        "usb_product_id": usb["idProduct"].lower(),
        "usb_device_release": usb.get("bcdDevice"),
        "usb_manufacturer": usb.get("manufacturer"),
        "usb_product": usb.get("product"),
        "iio_hardware_model": iio_attrs.get("hw_model"),
        "iio_hardware_model_variant": iio_attrs.get("hw_model_variant"),
        "ad936x_model": iio_attrs.get("ad9361-phy,model"),
        "reference_clock_hz": int(iio_attrs["ad9361-phy,xo_correction"]),
        "device_tree_model": sanitized_facts.get("device_tree_model"),
        "memory_total_kib": sanitized_facts.get("memory_total_kib"),
        "mtd0_size_bytes": sanitized_facts.get("mtd0_size_bytes"),
        "mtd1_size_bytes": sanitized_facts.get("mtd1_size_bytes"),
    }
    if hardware_identity.flags & HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID:
        stable_identity["fpga_device_dna_hmac_sha256"] = fpga_dna_hmac(
            hmac_key,
            hardware_identity.fpga_device_dna,
        )
    stable_identity = {
        key: value for key, value in stable_identity.items() if value is not None
    }
    boot_id = _safe_text(boot_id_path.read_text(), label="host boot ID")
    if not boot_id:
        raise HardwareFingerprintError("host boot ID is empty")
    captured_ns = time.time_ns() if captured_at_unix_ns is None else captured_at_unix_ns
    return {
        "schema": HARDWARE_FINGERPRINT_SCHEMA,
        "schema_version": HARDWARE_FINGERPRINT_VERSION,
        "fingerprint_timing": "post_firmware_before_recording",
        "acquisition_binding": True,
        "passive_observation": True,
        "tx_operations_performed": False,
        "2r2t_configured": sanitized_facts.get("uboot_mode") == "2r2t",
        "2r2t_functionally_verified": False,
        "captured_at_unix_ns": captured_ns,
        "host_boot_id": boot_id,
        "fingerprint_session_id": session_id,
        "hmac_key_id": hmac_key_id(hmac_key),
        "stable_identity": stable_identity,
        "stable_fingerprint_sha256": stable_identity_sha256(stable_identity),
        "attachment": {
            "usb_bus": device.bus,
            "usb_address": address,
            "usb_port_path": device.port_path,
            "usb_sysfs_name": device.sysfs_name,
            "iio_uri": uri,
        },
        "firmware_session": {
            **expected_firmware,
            "reported_gadget_build_id": hardware_identity.gadget_build_id,
            "iio_firmware_version": iio_attrs.get("fw_version"),
            "iio_kernel_version": iio_attrs.get("local,kernel"),
        },
        "direct_usb": capabilities,
        "device_facts": sanitized_facts,
        "compatibility": {
            "status": "compatible",
            "failures": [],
        },
    }


def public_fingerprint_copy(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON round-tripped copy suitable for a Zarr attribute."""

    copy = json.loads(_canonical_json(fingerprint))
    forbidden = {"fpga_device_dna", "raw_device_dna", "hmac_key"}

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden.intersection(value)
            if overlap:
                raise HardwareFingerprintError(
                    f"fingerprint contains private keys: {sorted(overlap)}"
                )
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(copy)
    return copy


def validate_public_hardware_fingerprint(
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Validate an untrusted serialized fingerprint and return a safe copy."""

    copy = public_fingerprint_copy(fingerprint)
    if (
        copy.get("schema") != HARDWARE_FINGERPRINT_SCHEMA
        or copy.get("schema_version") != HARDWARE_FINGERPRINT_VERSION
    ):
        raise HardwareFingerprintError("unsupported hardware fingerprint schema")
    timing = copy.get("fingerprint_timing")
    expected_binding = {
        "post_firmware_before_recording": True,
        "post_run_backfill": False,
    }.get(timing)
    if (
        expected_binding is None
        or copy.get("acquisition_binding") is not expected_binding
    ):
        raise HardwareFingerprintError(
            "hardware fingerprint timing/binding semantics are invalid"
        )
    if (
        copy.get("passive_observation") is not True
        or copy.get("tx_operations_performed") is not False
    ):
        raise HardwareFingerprintError("hardware fingerprint is not passive")
    stable_identity = copy.get("stable_identity")
    if not isinstance(stable_identity, dict):
        raise HardwareFingerprintError("stable hardware identity is missing")
    serial = stable_identity.get("pluto_serial")
    spi_nor_hmac = stable_identity.get("spi_nor_unique_id_hmac_sha256")
    if not isinstance(serial, str) or not serial:
        raise HardwareFingerprintError("stable Pluto serial is missing")
    if not isinstance(spi_nor_hmac, str) or not re.fullmatch(
        r"[0-9a-f]{64}", spi_nor_hmac
    ):
        raise HardwareFingerprintError("SPI-NOR UniqueID HMAC is invalid")
    dna_hmac = stable_identity.get("fpga_device_dna_hmac_sha256")
    if dna_hmac is not None and (
        not isinstance(dna_hmac, str) or not re.fullmatch(r"[0-9a-f]{64}", dna_hmac)
    ):
        raise HardwareFingerprintError("optional FPGA Device DNA HMAC is invalid")
    key_id = copy.get("hmac_key_id")
    if not isinstance(key_id, str) or not re.fullmatch(r"[0-9a-f]{16}", key_id):
        raise HardwareFingerprintError("hardware fingerprint HMAC key ID is invalid")
    stable_hash = copy.get("stable_fingerprint_sha256")
    if stable_hash != stable_identity_sha256(stable_identity):
        raise HardwareFingerprintError("stable hardware fingerprint hash is invalid")
    compatibility = copy.get("compatibility")
    if compatibility != {"status": "compatible", "failures": []}:
        raise HardwareFingerprintError(
            "hardware fingerprint compatibility result is invalid"
        )
    return copy
