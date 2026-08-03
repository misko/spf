from pathlib import Path
from types import SimpleNamespace

import pytest

import spf.scripts.pluto_multi_firmware as firmware_module
from spf.scripts.pluto_multi_firmware import (
    FirmwareError,
    MultiPlutoFirmwareManager,
    UsbPluto,
    discover_runtime_plutos,
    parse_passive_device_facts,
    parse_device_fw_version,
    parse_uboot_environment,
    read_passive_device_facts,
    wait_for_network_interface,
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
    (device / "devnum").write_text(f"{8 + len(list(root.iterdir()))}\n")
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


def test_parse_device_fw_version():
    assert (
        parse_device_fw_version(
            "build  abcdef\n" "device-fw  v0.37-dirty\n" "linux  4.14\n"
        )
        == "v0.37-dirty"
    )


def test_network_interface_wait_retries_transient_restore_gap(monkeypatch):
    observations = iter([[], [], ["eth1"]])
    monkeypatch.setattr(
        firmware_module,
        "find_network_interfaces",
        lambda serial: next(observations),
    )
    monkeypatch.setattr(firmware_module.time, "sleep", lambda seconds: None)

    assert (
        wait_for_network_interface("SERIAL_A", timeout=5, poll_interval=0.1)
        == "eth1"
    )


def test_network_interface_wait_rejects_ambiguous_identity_without_retry(
    monkeypatch,
):
    monkeypatch.setattr(
        firmware_module,
        "find_network_interfaces",
        lambda serial: ["eth1", "eth2"],
    )

    with pytest.raises(FirmwareError, match=r"found \['eth1', 'eth2'\]"):
        wait_for_network_interface("SERIAL_A", timeout=5)


def test_parse_passive_device_facts_uses_strict_allowlist():
    output = "\n".join(
        [
            "device_tree_model=Analog Devices PlutoSDR Rev.C",
            "memory_total_kib=506000",
            "mtd0_size_bytes=1048576",
            "mtd1_size_bytes=32768000",
            "sd_present=true",
            "uboot_attr_name=compatible",
            "uboot_attr_val=ad9361",
            "uboot_compatible=ad9361",
            "uboot_mode=2r2t",
            "device_fw=v0.38",
            "linux_version=5.15",
            "uboot_version=2022.01",
            "private_key=must be ignored",
        ]
    )

    facts = parse_passive_device_facts(output)

    assert facts["uboot_mode"] == "2r2t"
    assert "private_key" not in facts


def test_parse_passive_device_facts_rejects_missing_field():
    with pytest.raises(FirmwareError, match="incomplete"):
        parse_passive_device_facts("uboot_mode=2r2t\n")


def test_passive_fingerprint_waits_for_serial_specific_ssh_readiness(
    tmp_path, monkeypatch
):
    calls = []
    output = "\n".join(
        [
            "device_tree_model=Analog Devices PlutoSDR Rev.C",
            "memory_total_kib=506000",
            "mtd0_size_bytes=1048576",
            "mtd1_size_bytes=32768000",
            "sd_present=true",
            "uboot_attr_name=compatible",
            "uboot_attr_val=ad9361",
            "uboot_compatible=ad9361",
            "uboot_mode=2r2t",
            "device_fw=v0.38",
            "linux_version=5.15",
            "uboot_version=2022.01",
        ]
    )
    monkeypatch.setattr(
        MultiPlutoFirmwareManager,
        "_wait_for_ssh",
        lambda self, serial, timeout: calls.append(("wait", serial, timeout)),
    )
    monkeypatch.setattr(
        MultiPlutoFirmwareManager,
        "_ssh",
        lambda self, serial, command, timeout: (
            calls.append(("read", serial, timeout)) or SimpleNamespace(stdout=output)
        ),
    )

    facts = read_passive_device_facts(
        "SERIAL_A",
        ssh_config=tmp_path / "ssh_config",
        ssh_password="analog",
    )

    assert facts["uboot_mode"] == "2r2t"
    assert calls == [("wait", "SERIAL_A", 60), ("read", "SERIAL_A", 15)]


def _manager(tmp_path, expected_count=1):
    return MultiPlutoFirmwareManager(
        image=tmp_path / "pluto.dfu",
        image_sha256="0" * 64,
        ssh_config=tmp_path / "ssh_config",
        ssh_password="analog",
        state_root=tmp_path / "state",
        expected_count=expected_count,
    )


def _device(serial="SERIAL_A", path="1-1.1"):
    return UsbPluto(
        serial=serial,
        sysfs_name=path,
        bus=1,
        port_path=path.removeprefix("1-"),
        direct_usb=False,
    )


def test_wait_product_tolerates_sysfs_disappearing_during_reenumeration(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path)
    reads = iter(
        [
            OSError(19, "No such device"),
            "b674",
        ]
    )

    def flaky_read(_path):
        result = next(reads)
        if isinstance(result, OSError):
            raise result
        return result

    monkeypatch.setattr(firmware_module, "_read", flaky_read)
    monkeypatch.setattr(firmware_module.time, "sleep", lambda _seconds: None)

    manager._wait_product("1-1.1", "b674", timeout=1)


def test_provision_dry_run_identifies_only_incorrect_radio(tmp_path, monkeypatch):
    manager = _manager(tmp_path, expected_count=2)
    devices = [_device("SERIAL_A", "1-1.1"), _device("SERIAL_B", "1-1.2")]
    environments = {
        "SERIAL_A": {
            "attr_name": "compatible",
            "attr_val": "ad9361",
            "compatible": "ad9361",
            "mode": "2r2t",
        },
        "SERIAL_B": {
            "attr_name": "compatible",
            "attr_val": "ad9361",
            "compatible": "ad9361",
            "mode": "1r1t",
        },
    }
    calls = []
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(manager, "_devices", lambda: devices)
    monkeypatch.setattr(
        manager,
        "_device",
        lambda serial: next(device for device in devices if device.serial == serial),
    )
    monkeypatch.setattr(
        manager,
        "_read_persistent_state",
        lambda serial: ("v0.37-dirty", environments[serial], "backup"),
    )
    monkeypatch.setattr(
        manager, "_verify_dual_rx", lambda serial: calls.append(("verify", serial))
    )
    monkeypatch.setattr(
        manager,
        "_back_up_provisioning_state",
        lambda *args: calls.append(("backup", args[0].serial)),
    )
    monkeypatch.setattr(manager, "_ssh", lambda *args, **kwargs: calls.append(("ssh",)))

    manager.provision_config_all(dry_run=True)

    assert calls == [("verify", "SERIAL_A")]


def test_provision_writes_reboots_and_verifies_incorrect_radio(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    device = _device()
    before = {
        "attr_name": "compatible",
        "attr_val": "ad9361",
        "compatible": "ad9361",
        "mode": "1r1t",
    }
    after = {**before, "mode": "2r2t"}
    states = iter(
        [
            ("v0.37-dirty", before, "persistent backup"),
            ("v0.37-dirty", after, "persistent after"),
        ]
    )
    calls = []
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(manager, "_devices", lambda: [device])
    monkeypatch.setattr(manager, "_device", lambda serial: device)
    monkeypatch.setattr(manager, "_read_persistent_state", lambda serial: next(states))
    monkeypatch.setattr(
        manager,
        "_back_up_provisioning_state",
        lambda target, state: calls.append(("backup", target.serial, state)),
    )
    monkeypatch.setattr(
        manager,
        "_ssh",
        lambda serial, command, **kwargs: calls.append(("ssh", serial, command)),
    )
    monkeypatch.setattr(
        manager, "_wait_absent", lambda *args: calls.append(("absent",))
    )
    monkeypatch.setattr(
        manager, "_wait_product", lambda *args: calls.append(("product",))
    )
    monkeypatch.setattr(
        manager, "_wait_for_ssh", lambda *args: calls.append(("wait-ssh",))
    )
    monkeypatch.setattr(
        manager, "_verify_dual_rx", lambda serial: calls.append(("verify", serial))
    )

    manager.provision_config_all()

    assert ("backup", "SERIAL_A", "persistent backup") in calls
    write = next(call for call in calls if call[0] == "ssh" and "fw_setenv" in call[2])
    assert "fw_setenv mode 2r2t" in write[2]
    assert any(call[0] == "ssh" and "device_reboot reset" in call[2] for call in calls)
    assert ("verify", "SERIAL_A") in calls


def test_restart_all_requires_absence_and_preserves_each_radio_identity(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, expected_count=2)
    originals = [
        UsbPluto("SERIAL_A", "1-1.1", 1, "1.1", True, 8),
        UsbPluto("SERIAL_B", "1-1.2", 1, "1.2", True, 9),
    ]
    returned = {
        "SERIAL_A": UsbPluto("SERIAL_A", "1-1.1", 1, "1.1", True, 18),
        "SERIAL_B": UsbPluto("SERIAL_B", "1-1.2", 1, "1.2", True, 19),
    }
    calls = []
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(firmware_module, "_require_commands", lambda commands: None)
    monkeypatch.setattr(manager, "_devices", lambda: originals)
    monkeypatch.setattr(manager, "_device", lambda serial: returned[serial])
    monkeypatch.setattr(
        manager,
        "_ssh",
        lambda serial, command, **kwargs: calls.append(("ssh", serial, command)),
    )
    monkeypatch.setattr(
        manager,
        "_wait_absent",
        lambda path, timeout: calls.append(("absent", path, timeout)),
    )
    monkeypatch.setattr(
        manager,
        "_wait_product",
        lambda path, product, timeout: calls.append(
            ("product", path, product, timeout)
        ),
    )
    monkeypatch.setattr(
        manager,
        "_wait_for_ssh",
        lambda serial, timeout: calls.append(("wait-ssh", serial, timeout)),
    )
    monkeypatch.setattr(
        manager,
        "_verify_device",
        lambda serial: calls.append(("verify", serial)),
    )

    manager.restart_all()

    for radio in originals:
        serial = radio.serial
        assert ("ssh", serial, "sync; /usr/sbin/device_reboot reset") in calls
        assert ("absent", radio.sysfs_name, 30) in calls
        assert ("wait-ssh", serial, 60) in calls
        assert ("verify", serial) in calls


def test_restart_all_fails_closed_if_radio_returns_on_another_port(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path)
    original = UsbPluto("SERIAL_A", "1-1.1", 1, "1.1", True, 8)
    wrong_port = UsbPluto("SERIAL_A", "1-1.2", 1, "1.2", True, 18)
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(firmware_module, "_require_commands", lambda commands: None)
    monkeypatch.setattr(manager, "_devices", lambda: [original])
    monkeypatch.setattr(manager, "_device", lambda serial: wrong_port)
    monkeypatch.setattr(manager, "_ssh", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_wait_absent", lambda *args: None)
    monkeypatch.setattr(manager, "_wait_product", lambda *args: None)
    monkeypatch.setattr(manager, "_wait_for_ssh", lambda *args: None)
    monkeypatch.setattr(
        manager,
        "_verify_device",
        lambda serial: pytest.fail("wrong physical port must reject before verify"),
    )

    with pytest.raises(FirmwareError, match="unexpected USB identity"):
        manager.restart_all()


def test_provision_rejects_unapproved_qspi_before_write(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    device = _device()
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(manager, "_devices", lambda: [device])
    monkeypatch.setattr(manager, "_device", lambda serial: device)
    monkeypatch.setattr(
        manager,
        "_read_persistent_state",
        lambda serial: ("v0.32-1-g7bdc-dirty", {"mode": "1r1t"}, "backup"),
    )
    monkeypatch.setattr(
        manager,
        "_ssh",
        lambda *args, **kwargs: pytest.fail("must not write an unapproved radio"),
    )

    with pytest.raises(FirmwareError, match="is not approved"):
        manager.provision_config_all()


@pytest.mark.parametrize("direct_usb", [False, True])
def test_boot_config_check_ignores_active_runtime_version(
    tmp_path, monkeypatch, direct_usb
):
    manager = _manager(tmp_path)
    device = UsbPluto(
        serial="SERIAL_A",
        sysfs_name="1-1.1",
        bus=1,
        port_path="1.1",
        direct_usb=direct_usb,
    )
    environment = {
        "attr_name": "compatible",
        "attr_val": "ad9361",
        "compatible": "ad9361",
        "mode": "2r2t",
    }
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(manager, "_devices", lambda: [device])
    monkeypatch.setattr(manager, "_read_uboot_environment", lambda serial: environment)
    monkeypatch.setattr(
        manager,
        "_read_persistent_state",
        lambda serial: pytest.fail("boot must not inspect the active image version"),
    )
    monkeypatch.setattr(
        manager,
        "_verify_dual_rx",
        lambda serial: pytest.fail("dual RX is verified after the exact RAM load"),
    )

    manager.check_config_all()


def test_boot_config_check_still_rejects_persistent_uboot_mismatch(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_check_root", lambda: None)
    monkeypatch.setattr(manager, "_devices", lambda: [_device()])
    monkeypatch.setattr(
        manager,
        "_read_uboot_environment",
        lambda serial: {
            "attr_name": "compatible",
            "attr_val": "ad9361",
            "compatible": "ad9361",
            "mode": "1r1t",
        },
    )

    with pytest.raises(FirmwareError, match="U-Boot environment mismatch"):
        manager.check_config_all()


def test_direct_firmware_is_reloaded_instead_of_only_trusted(tmp_path, monkeypatch):
    manager = MultiPlutoFirmwareManager(
        image=tmp_path / "pluto.dfu",
        image_sha256="0" * 64,
        ssh_config=tmp_path / "ssh_config",
        ssh_password="analog",
        state_root=tmp_path / "state",
        expected_count=1,
    )
    device = UsbPluto(
        serial="SERIAL_A",
        sysfs_name="1-1.1",
        bus=1,
        port_path="1.1",
        direct_usb=True,
    )
    calls = []
    monkeypatch.setattr(
        manager,
        "_back_up",
        lambda serial: calls.append(("backup", serial)),
    )
    monkeypatch.setattr(
        manager,
        "_ssh",
        lambda serial, command, **kwargs: calls.append(("ssh", serial, command)),
    )
    monkeypatch.setattr(manager, "_wait_product", lambda *args: None)
    monkeypatch.setattr(manager, "_wait_for_ssh", lambda *args: None)
    monkeypatch.setattr(
        manager,
        "_verify_device",
        lambda serial: calls.append(("verify", serial)),
    )
    monkeypatch.setattr(
        "spf.scripts.pluto_multi_firmware._run",
        lambda command, **kwargs: calls.append(("run", command)),
    )

    manager._load_device(device)

    assert ("backup", "SERIAL_A") in calls
    assert any(
        call[0] == "ssh" and "/usr/sbin/device_reboot ram" in call[2] for call in calls
    )
    assert ("verify", "SERIAL_A") in calls


def test_boot_preparation_uses_configured_boot_mode_with_environment_override():
    boot_script = (
        Path(__file__).resolve().parents[1]
        / "data_collection/rover/rover_v3.1/prepare_direct_usb_boot.sh"
    ).read_text()
    # The canonical config selects QSPI or RAM; an explicit environment override
    # is retained for field recovery.
    ensure = "ensure_pluto_qspi.sh"
    config_mode = 'firmware_boot_mode="${config_values[14]}"'
    device_fw = 'firmware_device_fw="${config_values[15]}"'
    override = 'RAM_LOAD_OVERRIDE="${SPF_PLUTO_RAM_LOAD:-}"'
    ram_gate = 'if is_true "$ram_load"; then'
    load = 'run_loader load-all "$attached_radios"'

    assert ensure in boot_script
    assert config_mode in boot_script
    assert device_fw in boot_script
    assert override in boot_script
    assert ram_gate in boot_script
    # The volatile path remains isolated from persistent QSPI preparation.
    assert load in boot_script
    assert (
        boot_script.index(ram_gate)
        < boot_script.index(load)
        < boot_script.index(ensure)
    )
    # The per-boot ssh check-config-all was removed (it raced the shared-IP ssh);
    # a wrong AD9361/2r2t config is still caught by verify-all's dual-RX check.
    assert "run_loader check-config-all" not in boot_script
    # Readiness is still invalidated before config resolution.
    invalidate = 'rm -f -- "$READY_FILE"'
    resolver = "resolver_args=("
    assert invalidate in boot_script
    assert boot_script.index(invalidate) < boot_script.index(resolver)


@pytest.mark.parametrize("rover_id", (1, 2, 3, 4))
def test_canonical_v7_rover_config_declares_persistent_qspi(rover_id):
    from spf.scripts.rover_capture_config import resolve_capture_plan

    plan = resolve_capture_plan(rover_id)

    assert plan.data_version == 7
    assert plan.firmware_boot_mode == "qspi"


@pytest.mark.parametrize(
    ("channels", "passes"),
    [
        ((0, 1, 2, 3), True),
        ((0, 1), False),
    ],
)
def test_dual_rx_gate_requires_four_dma_scan_elements(
    tmp_path, monkeypatch, channels, passes
):
    manager = MultiPlutoFirmwareManager(
        image=tmp_path / "pluto.dfu",
        image_sha256="0" * 64,
        ssh_config=tmp_path / "ssh_config",
        ssh_password="analog",
        state_root=tmp_path / "state",
        expected_count=1,
    )
    monkeypatch.setattr(manager, "_iio_uri_for_serial", lambda serial: "usb:1.2.5")
    rendered_channels = "\n".join(
        f"\t\t\tvoltage{channel}:  " f"(input, index: {channel}, format: le:S12/16>>0)"
        for channel in channels
    )
    output = (
        "\tiio:device3: cf-ad9361-lpc (buffer capable)\n"
        f"{rendered_channels}\n"
        "\tiio:device4: other\n"
    )
    monkeypatch.setattr(
        "spf.scripts.pluto_multi_firmware._run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    if passes:
        manager._verify_dual_rx("SERIAL_A")
    else:
        with pytest.raises(FirmwareError, match="dual RX is unavailable"):
            manager._verify_dual_rx("SERIAL_A")
