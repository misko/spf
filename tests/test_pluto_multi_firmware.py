from pathlib import Path

import pytest

from spf.scripts.pluto_multi_firmware import (
    FirmwareError,
    MultiPlutoFirmwareManager,
    discover_runtime_plutos,
    parse_uboot_environment,
)


def _add_usb_device(
    root: Path,
    *,
    name: str,
    serial: str,
    bus: int,
    port_path: str,
    direct_usb: bool,
) -> None:
    device = root / name
    device.mkdir()
    (device / "idVendor").write_text("0456\n")
    (device / "idProduct").write_text("b673\n")
    (device / "serial").write_text(f"{serial}\n")
    (device / "busnum").write_text(f"{bus}\n")
    (device / "devpath").write_text(f"{port_path}\n")
    if direct_usb:
        interface = root / f"{name}:1.6"
        interface.mkdir()
        (interface / "bInterfaceClass").write_text("ff\n")


def test_discover_runtime_plutos_is_sorted_and_detects_direct_interface(tmp_path):
    _add_usb_device(
        tmp_path,
        name="1-1.2",
        serial="SERIAL_B",
        bus=1,
        port_path="1.2",
        direct_usb=False,
    )
    _add_usb_device(
        tmp_path,
        name="1-1.1",
        serial="SERIAL_A",
        bus=1,
        port_path="1.1",
        direct_usb=True,
    )

    devices = discover_runtime_plutos(tmp_path)

    assert [device.serial for device in devices] == ["SERIAL_A", "SERIAL_B"]
    assert devices[0].direct_usb is True
    assert devices[1].direct_usb is False


def test_discover_runtime_plutos_rejects_duplicate_serials(tmp_path):
    for suffix in ("1", "2"):
        _add_usb_device(
            tmp_path,
            name=f"1-1.{suffix}",
            serial="DUPLICATE",
            bus=1,
            port_path=f"1.{suffix}",
            direct_usb=False,
        )

    with pytest.raises(FirmwareError, match="duplicate Pluto serials"):
        discover_runtime_plutos(tmp_path)


def test_firmware_manager_rejects_wrong_image_checksum(tmp_path):
    image = tmp_path / "pluto.dfu"
    image.write_bytes(b"not the accepted image")
    manager = MultiPlutoFirmwareManager(
        image=image,
        image_sha256="0" * 64,
        ssh_config=tmp_path / "ssh_config",
        ssh_password="analog",
        state_root=tmp_path / "state",
        expected_count=2,
    )

    with pytest.raises(FirmwareError, match="SHA-256 mismatch"):
        manager._check_image()


def test_parse_uboot_environment_ignores_diagnostics():
    assert parse_uboot_environment(
        "Warning: Bad CRC\ncompatible=ad9361\nmode=2r2t\n"
    ) == {
        "compatible": "ad9361",
        "mode": "2r2t",
    }
