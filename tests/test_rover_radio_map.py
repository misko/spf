"""`rover radio map` — which physical radio is r0, which is r1, is it there?

Written after a Pluto on Rover 1 dropped off the USB bus minutes after boot on
2026-08-05 and the only symptom was `Exception: No device found`. Working out
that the missing radio was r1 -- and which serial to look for on the bench --
required reading the capture config, ~/device_mapping and the collector's URI
construction together.

Presence deliberately keys on `lsusb -t` (mass-storage interface), not `lsusb`:
a half-dropped Pluto still answers the latter, which is the exact state that
produces "No device found".
"""

import json

import pytest
import yaml

from spf.scripts import rover_radio_map as rrm


@pytest.fixture
def rover1(tmp_path):
    """Rover 1's real shape: r0 on receiver-port 2, r1 on receiver-port 1."""
    config = tmp_path / "rover1.yaml"
    config.write_text(yaml.safe_dump({
        "receivers": [
            {"receiver-port": 2, "antenna-spacing-m": 0.035},
            {"receiver-port": 1, "antenna-spacing-m": 0.035},
        ]
    }))
    mapping = tmp_path / "device_mapping"
    mapping.write_text("1 3\n2 4\n")
    manifest = tmp_path / "ready.json"
    manifest.write_text(json.dumps({
        "receivers": [
            {"iio_uri": "usb:1.3.5", "pluto_serial": "SERIAL-R1"},
            {"iio_uri": "usb:1.4.5", "pluto_serial": "SERIAL-R0"},
        ]
    }))
    return config, mapping, manifest


def test_receiver_port_maps_to_the_right_radio(rover1, monkeypatch):
    config, mapping, manifest = rover1
    monkeypatch.setattr(rrm, "present_ports", lambda: {1: "3", 2: "4"})
    rows = {r["receiver"]: r for r in
            rrm.build(str(config), str(mapping), str(manifest))["receivers"]}

    # r0 is the FIRST config block, which carries receiver-port 2.
    assert rows["r0"]["receiver_port"] == 2
    assert rows["r0"]["uri"] == "pluto://usb:1.4.5"
    assert rows["r0"]["kernel_name"] == "usb 1-1.2"
    assert rows["r0"]["serial"] == "SERIAL-R0"

    assert rows["r1"]["receiver_port"] == 1
    assert rows["r1"]["uri"] == "pluto://usb:1.3.5"
    assert rows["r1"]["kernel_name"] == "usb 1-1.1"
    assert rows["r1"]["serial"] == "SERIAL-R1"


def test_the_rover1_disconnect_is_attributed_to_r1(rover1, monkeypatch, capsys):
    """usb 1-1.1 vanished; that is r1, and the report must say so."""
    config, mapping, manifest = rover1
    monkeypatch.setattr(rrm, "present_ports", lambda: {2: "4"})   # port 1 gone

    report = rrm.build(str(config), str(mapping), str(manifest))
    status = rrm.render(report)
    out = capsys.readouterr().out

    assert status == 1
    rows = {r["receiver"]: r for r in report["receivers"]}
    assert rows["r1"]["present"] is False
    assert rows["r0"]["present"] is True
    assert "no radio behind r1" in out
    assert "usb disconnect" in out         # tells the operator where to look
    assert "SERIAL-R1" in out              # and which unit to find on the bench


def test_stale_mapping_is_distinguished_from_a_missing_radio(rover1, monkeypatch, capsys):
    """A renumbered device is present but unopenable -- a different fix."""
    config, mapping, manifest = rover1
    monkeypatch.setattr(rrm, "present_ports", lambda: {1: "7", 2: "4"})

    report = rrm.build(str(config), str(mapping), str(manifest))
    status = rrm.render(report)
    out = capsys.readouterr().out

    assert status == 1
    assert report["receivers"][1]["mapping_stale"] is True
    assert report["receivers"][1]["present"] is True
    assert "rover radio remap" in out
    assert "no radio behind" not in out


def test_all_present_passes(rover1, monkeypatch, capsys):
    config, mapping, manifest = rover1
    monkeypatch.setattr(rrm, "present_ports", lambda: {1: "3", 2: "4"})
    assert rrm.render(rrm.build(str(config), str(mapping), str(manifest))) == 0
    assert "PASS" in capsys.readouterr().out


def test_missing_mapping_file_does_not_crash(rover1, monkeypatch, capsys):
    config, _mapping, manifest = rover1
    monkeypatch.setattr(rrm, "present_ports", lambda: {1: "3", 2: "4"})
    report = rrm.build(str(config), "/nonexistent/device_mapping", str(manifest))
    rrm.render(report)
    assert report["device_mapping_error"]


def test_present_ports_reads_the_storage_interface(monkeypatch):
    """lsusb (not -t) still lists a half-dropped Pluto; -t is the honest check."""
    sample = (
        "    |__ Port 1: Dev 3, If 2, Class=Mass Storage, Driver=usb-storage, 480M\n"
        "    |__ Port 2: Dev 4, If 2, Class=Mass Storage, Driver=usb-storage, 480M\n"
        "    |__ Port 3: Dev 5, If 1, Class=Communications, Driver=cdc_acm, 12M\n"
    )

    class Result:
        stdout = sample

    monkeypatch.setattr(rrm.subprocess, "run", lambda *a, **k: Result())
    assert rrm.present_ports() == {1: "3", 2: "4"}
