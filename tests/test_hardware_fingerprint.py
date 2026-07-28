import json
from pathlib import Path
import struct

import pytest

from spf.hardware_fingerprint import (
    HardwareFingerprintError,
    collect_hardware_fingerprint,
    fpga_dna_hmac,
    load_or_create_hmac_key,
    public_fingerprint_copy,
    spi_nor_uid_hmac,
)
from spf.scripts.pluto_multi_firmware import UsbPluto
from spf.sdrpluto.direct_usb_protocol import (
    HARDWARE_IDENTITY_BYTES,
    HARDWARE_IDENTITY_MAGIC,
    HARDWARE_IDENTITY_VERSION,
    HardwareIdentityFlags,
    HardwareIdentityV1,
    ProtocolError,
)


SERIAL = "104000f6ad020002fdff3a00bba2f096a1"
GADGET_SHA = "c" * 40
DNA = 0x0123456789ABCD


def _device() -> UsbPluto:
    return UsbPluto(
        serial=SERIAL,
        sysfs_name="1-1.1",
        bus=1,
        port_path="1.1",
        direct_usb=True,
        address=8,
    )


def _usb_tree(root: Path) -> None:
    device = root / "1-1.1"
    device.mkdir()
    values = {
        "idVendor": "0456",
        "idProduct": "b673",
        "bcdDevice": "0515",
        "manufacturer": "Analog Devices Inc.",
        "product": "PlutoSDR+ with timestamp support",
        "serial": SERIAL,
        "busnum": "1",
        "devnum": "8",
        "devpath": "1.1",
        "bNumConfigurations": "1",
        "speed": "480",
    }
    for name, value in values.items():
        (device / name).write_text(value + "\n")


def _iio(uri: str) -> dict[str, str]:
    assert uri == "usb:1.8.5"
    return {
        "hw_model": "Analog Devices PlutoSDR Rev.C (Z7010-AD9361)",
        "hw_model_variant": "1",
        "hw_serial": SERIAL,
        "fw_version": "v0.38-test",
        "ad9361-phy,xo_correction": "40000000",
        "ad9361-phy,model": "ad9361",
        "local,kernel": "5.15-test",
        "usb,idVendor": "0456",
        "usb,idProduct": "b673",
        "usb,serial": SERIAL,
    }


def _direct(device: UsbPluto):
    assert device.serial == SERIAL
    return (
        {
            "protocol_min": 1,
            "protocol_max": 2,
            "supported_features": 0x37,
            "capability_flags": 5,
        },
        HardwareIdentityV1(
            flags=(
                HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
                | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
            ),
            fpga_device_dna=DNA,
            gadget_build_id=GADGET_SHA,
        ),
    )


def _facts():
    return {
        "device_tree_model": "Analog Devices PlutoSDR Rev.C",
        "memory_total_kib": "506000",
        "mtd0_size_bytes": "1048576",
        "mtd1_size_bytes": "32768000",
        "sd_present": "true",
        "uboot_attr_name": "compatible",
        "uboot_attr_val": "ad9361",
        "uboot_compatible": "ad9361",
        "uboot_mode": "2r2t",
        "device_fw": "v0.38-test",
        "linux_version": "5.15-test",
        "uboot_version": "test",
    }


def test_hardware_identity_golden_wire_response():
    payload = struct.pack(
        "<IHHIIQ40s",
        HARDWARE_IDENTITY_MAGIC,
        HARDWARE_IDENTITY_BYTES,
        HARDWARE_IDENTITY_VERSION,
        int(
            HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
            | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
        ),
        0,
        DNA,
        GADGET_SHA.encode("ascii"),
    )

    assert HardwareIdentityV1.unpack(payload) == _direct(_device())[1]


def test_hardware_identity_rejects_invalid_dna_and_build_id():
    payload = struct.pack(
        "<IHHIIQ40s",
        HARDWARE_IDENTITY_MAGIC,
        HARDWARE_IDENTITY_BYTES,
        HARDWARE_IDENTITY_VERSION,
        int(
            HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
            | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
        ),
        0,
        1 << 60,
        b"not-a-sha".ljust(40, b"\x00"),
    )

    with pytest.raises(ProtocolError):
        HardwareIdentityV1.unpack(payload)


def test_hardware_identity_accepts_build_identity_without_optional_fpga_dna():
    payload = struct.pack(
        "<IHHIIQ40s",
        HARDWARE_IDENTITY_MAGIC,
        HARDWARE_IDENTITY_BYTES,
        HARDWARE_IDENTITY_VERSION,
        int(HardwareIdentityFlags.GADGET_BUILD_ID_VALID),
        0,
        0,
        GADGET_SHA.encode("ascii"),
    )

    identity = HardwareIdentityV1.unpack(payload)

    assert identity.fpga_device_dna == 0
    assert identity.flags == HardwareIdentityFlags.GADGET_BUILD_ID_VALID
    assert identity.gadget_build_id == GADGET_SHA


def test_hmac_key_is_private_and_dna_hmac_is_deterministic(tmp_path):
    key_path = tmp_path / "private" / "key"

    key = load_or_create_hmac_key(key_path)

    assert len(key) == 32
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert fpga_dna_hmac(key, DNA) == fpga_dna_hmac(key, DNA)
    assert fpga_dna_hmac(key, DNA) != fpga_dna_hmac(key, DNA + 1)
    assert spi_nor_uid_hmac(key, SERIAL) == spi_nor_uid_hmac(key, SERIAL)


def test_collect_fingerprint_is_passive_sanitized_and_deterministic(tmp_path):
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    _usb_tree(usb_root)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("BOOT-A\n")
    firmware = {
        "release_tag": "release",
        "image_sha256": "a" * 64,
        "firmware_git_sha": "b" * 40,
        "gadget_git_sha": GADGET_SHA,
        "boot_mode": "ram",
    }

    first = collect_hardware_fingerprint(
        _device(),
        expected_firmware=firmware,
        session_id="SESSION-A",
        hmac_key=b"k" * 32,
        usb_root=usb_root,
        boot_id_path=boot_id,
        device_facts=_facts(),
        captured_at_unix_ns=123,
        iio_reader=_iio,
        direct_identity_reader=_direct,
    )
    second = collect_hardware_fingerprint(
        _device(),
        expected_firmware=firmware,
        session_id="SESSION-B",
        hmac_key=b"k" * 32,
        usb_root=usb_root,
        boot_id_path=boot_id,
        device_facts=_facts(),
        captured_at_unix_ns=456,
        iio_reader=_iio,
        direct_identity_reader=_direct,
    )

    assert first["passive_observation"] is True
    assert first["tx_operations_performed"] is False
    assert first["2r2t_configured"] is True
    assert first["2r2t_functionally_verified"] is False
    assert first["stable_fingerprint_sha256"] == second["stable_fingerprint_sha256"]
    assert first["fingerprint_session_id"] != second["fingerprint_session_id"]
    encoded = json.dumps(public_fingerprint_copy(first), sort_keys=True)
    assert str(DNA) not in encoded
    assert '"fpga_device_dna":' not in encoded
    assert first["stable_identity"]["spi_nor_unique_id_hmac_sha256"]
    assert first["stable_identity"]["fpga_device_dna_hmac_sha256"]


def test_collect_fingerprint_does_not_require_optional_fpga_dna(tmp_path):
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    _usb_tree(usb_root)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("BOOT-A\n")

    def direct_without_dna(device):
        capabilities, identity = _direct(device)
        return capabilities, HardwareIdentityV1(
            flags=HardwareIdentityFlags.GADGET_BUILD_ID_VALID,
            fpga_device_dna=0,
            gadget_build_id=identity.gadget_build_id,
        )

    fingerprint = collect_hardware_fingerprint(
        _device(),
        expected_firmware={"gadget_git_sha": GADGET_SHA},
        session_id="SESSION-A",
        hmac_key=b"k" * 32,
        usb_root=usb_root,
        boot_id_path=boot_id,
        device_facts=_facts(),
        iio_reader=_iio,
        direct_identity_reader=direct_without_dna,
    )

    stable = fingerprint["stable_identity"]
    assert stable["spi_nor_unique_id_hmac_sha256"]
    assert "fpga_device_dna_hmac_sha256" not in stable


def test_collect_fingerprint_rejects_wrong_reported_gadget(tmp_path):
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    _usb_tree(usb_root)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("BOOT-A\n")

    with pytest.raises(HardwareFingerprintError, match="gadget_build_id_mismatch"):
        collect_hardware_fingerprint(
            _device(),
            expected_firmware={"gadget_git_sha": "d" * 40},
            session_id="SESSION-A",
            hmac_key=b"k" * 32,
            usb_root=usb_root,
            boot_id_path=boot_id,
            device_facts=_facts(),
            iio_reader=_iio,
            direct_identity_reader=_direct,
        )


def test_collect_fingerprint_rejects_non_allowlisted_device_fact(tmp_path):
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    _usb_tree(usb_root)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("BOOT-A\n")
    facts = _facts()
    facts["ssh_private_key"] = "must not be collected"

    with pytest.raises(HardwareFingerprintError, match="non-allowlisted"):
        collect_hardware_fingerprint(
            _device(),
            expected_firmware={"gadget_git_sha": GADGET_SHA},
            session_id="SESSION-A",
            hmac_key=b"k" * 32,
            usb_root=usb_root,
            boot_id_path=boot_id,
            device_facts=facts,
            iio_reader=_iio,
            direct_identity_reader=_direct,
        )
