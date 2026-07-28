import json

import pytest

from spf.hardware_fingerprint import stable_identity_sha256
from spf.scripts import pluto_ready_manifest
from spf.scripts.pluto_multi_firmware import UsbPluto


DEVICES = [
    UsbPluto(
        serial="SERIAL_A",
        sysfs_name="1-1.1",
        bus=1,
        port_path="1.1",
        direct_usb=True,
    ),
    UsbPluto(
        serial="SERIAL_B",
        sysfs_name="1-1.2",
        bus=1,
        port_path="1.2",
        direct_usb=True,
    ),
]


def _fingerprint(device, *, expected_firmware, session_id, **kwargs):
    stable_identity = {
        "pluto_serial": device.serial,
        "spi_nor_unique_id_hmac_sha256": (
            "a" * 64 if device.serial == "SERIAL_A" else "b" * 64
        ),
    }
    return {
        "schema": "spf.hardware_compatibility_fingerprint",
        "schema_version": 1,
        "fingerprint_timing": "post_firmware_before_recording",
        "acquisition_binding": True,
        "passive_observation": True,
        "tx_operations_performed": False,
        "host_boot_id": "BOOT_A",
        "fingerprint_session_id": session_id,
        "hmac_key_id": "c" * 16,
        "stable_identity": stable_identity,
        "stable_fingerprint_sha256": stable_identity_sha256(stable_identity),
        "attachment": {
            "usb_bus": device.bus,
            "usb_address": device.address,
            "usb_port_path": device.port_path,
        },
        "firmware_session": expected_firmware,
        "direct_usb": {
            "protocol_min": 1,
            "protocol_max": 2,
            "supported_features": 0x37,
            "capability_flags": 1,
        },
        "compatibility": {"status": "compatible", "failures": []},
    }


def _mock_passive_fingerprint(monkeypatch):
    monkeypatch.setattr(pluto_ready_manifest, "_boot_id", lambda path: "BOOT_A")
    monkeypatch.setattr(
        pluto_ready_manifest,
        "load_or_create_hmac_key",
        lambda path: b"k" * 32,
    )
    monkeypatch.setattr(
        pluto_ready_manifest,
        "_default_device_fact_reader",
        lambda serial: {},
    )
    monkeypatch.setattr(
        pluto_ready_manifest,
        "collect_hardware_fingerprint",
        _fingerprint,
    )


def test_ready_manifest_records_config_firmware_and_each_serial(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    _mock_passive_fingerprint(monkeypatch)

    manifest = pluto_ready_manifest.build_manifest(
        1,
        mapping_path=mapping,
        boot_id_path=tmp_path / "boot_id",
        hmac_key_path=tmp_path / "key",
    )

    assert manifest["ready_manifest_version"] == 2
    assert manifest["host_boot_id"] == "BOOT_A"
    assert manifest["configured_receiver_count"] == 2
    assert manifest["attached_radio_count"] == 2
    assert (
        manifest["firmware"]["image_sha256"]
        == "0a6a8939b31babed2ad7093d83941ebc809323d69804adcd8da5bcae0e48d3e9"
    )
    assert [radio["serial"] for radio in manifest["radios"]] == [
        "SERIAL_A",
        "SERIAL_B",
    ]
    assert all(radio["firmware_verified"] for radio in manifest["radios"])
    assert all("hardware_fingerprint" in radio for radio in manifest["radios"])


def test_ready_manifest_verification_fails_on_stale_firmware(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    ready = tmp_path / "ready.json"
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    _mock_passive_fingerprint(monkeypatch)
    manifest = pluto_ready_manifest.build_manifest(
        1,
        mapping_path=mapping,
        boot_id_path=tmp_path / "boot_id",
        hmac_key_path=tmp_path / "key",
    )
    manifest["firmware"]["image_sha256"] = "0" * 64
    ready.write_text(json.dumps(manifest))

    with pytest.raises(
        pluto_ready_manifest.ReadyManifestError,
        match="firmware",
    ):
        pluto_ready_manifest.verify_manifest(
            1,
            path=ready,
            mapping_path=mapping,
            boot_id_path=tmp_path / "boot_id",
        )


def test_ready_manifest_rejects_attached_config_count_mismatch(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n")
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES[:1],
    )

    with pytest.raises(
        pluto_ready_manifest.ReadyManifestError,
        match="configured 2 receivers",
    ):
        pluto_ready_manifest.build_manifest(1, mapping_path=mapping)


def test_ready_verification_uses_session_cache_without_requery(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    ready = tmp_path / "ready.json"
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    _mock_passive_fingerprint(monkeypatch)
    manifest = pluto_ready_manifest.build_manifest(
        1,
        mapping_path=mapping,
        boot_id_path=tmp_path / "boot_id",
        hmac_key_path=tmp_path / "key",
    )
    pluto_ready_manifest.write_manifest(ready, manifest)
    monkeypatch.setattr(
        pluto_ready_manifest,
        "collect_hardware_fingerprint",
        lambda *args, **kwargs: pytest.fail("fingerprint was queried twice"),
    )
    monkeypatch.setattr(
        pluto_ready_manifest,
        "_default_device_fact_reader",
        lambda serial: pytest.fail("device facts were queried twice"),
    )

    verified = pluto_ready_manifest.verify_manifest(
        1,
        path=ready,
        mapping_path=mapping,
        boot_id_path=tmp_path / "boot_id",
    )

    assert verified["fingerprint_session_id"] == manifest["fingerprint_session_id"]


def test_ready_verification_rejects_tampered_spi_nor_identity(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    ready = tmp_path / "ready.json"
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    _mock_passive_fingerprint(monkeypatch)
    manifest = pluto_ready_manifest.build_manifest(
        1,
        mapping_path=mapping,
        boot_id_path=tmp_path / "boot_id",
        hmac_key_path=tmp_path / "key",
    )
    manifest["radios"][0]["hardware_fingerprint"]["stable_identity"][
        "spi_nor_unique_id_hmac_sha256"
    ] = ("f" * 64)
    ready.write_text(json.dumps(manifest))

    with pytest.raises(
        pluto_ready_manifest.ReadyManifestError,
        match="fingerprint hash",
    ):
        pluto_ready_manifest.verify_manifest(
            1,
            path=ready,
            mapping_path=mapping,
            boot_id_path=tmp_path / "boot_id",
        )


def test_ready_build_rejects_duplicate_spi_nor_identity(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    _mock_passive_fingerprint(monkeypatch)

    def duplicate_spi_nor_uid(device, *, expected_firmware, session_id, **kwargs):
        result = _fingerprint(
            device,
            expected_firmware=expected_firmware,
            session_id=session_id,
        )
        result["stable_identity"]["spi_nor_unique_id_hmac_sha256"] = "a" * 64
        result["stable_fingerprint_sha256"] = stable_identity_sha256(
            result["stable_identity"]
        )
        return result

    monkeypatch.setattr(
        pluto_ready_manifest,
        "collect_hardware_fingerprint",
        duplicate_spi_nor_uid,
    )

    with pytest.raises(
        pluto_ready_manifest.ReadyManifestError,
        match="one SPI-NOR UniqueID",
    ):
        pluto_ready_manifest.build_manifest(
            1,
            mapping_path=mapping,
            boot_id_path=tmp_path / "boot_id",
            hmac_key_path=tmp_path / "key",
        )
