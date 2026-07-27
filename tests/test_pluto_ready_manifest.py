import json

import pytest

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


def test_ready_manifest_records_config_firmware_and_each_serial(tmp_path, monkeypatch):
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 8\n2 9\n")
    monkeypatch.setattr(
        pluto_ready_manifest,
        "discover_runtime_plutos",
        lambda: DEVICES,
    )
    monkeypatch.setattr(pluto_ready_manifest, "_spf_git_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        pluto_ready_manifest,
        "_probe_v7_capabilities",
        lambda serial: {
            "protocol_min": 1,
            "protocol_max": 2,
            "supported_features": 0x37,
            "capability_flags": 1,
        },
    )

    manifest = pluto_ready_manifest.build_manifest(1, mapping_path=mapping)

    assert manifest["configured_receiver_count"] == 2
    assert manifest["attached_radio_count"] == 2
    assert (
        manifest["firmware"]["image_sha256"]
        == "f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4"
    )
    assert [radio["serial"] for radio in manifest["radios"]] == [
        "SERIAL_A",
        "SERIAL_B",
    ]
    assert all(radio["firmware_verified"] for radio in manifest["radios"])


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
    monkeypatch.setattr(
        pluto_ready_manifest,
        "_probe_v7_capabilities",
        lambda serial: {
            "protocol_min": 1,
            "protocol_max": 2,
            "supported_features": 0x37,
            "capability_flags": 1,
        },
    )
    manifest = pluto_ready_manifest.build_manifest(1, mapping_path=mapping)
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
