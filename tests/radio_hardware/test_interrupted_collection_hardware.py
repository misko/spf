"""Explicit real-radio SIGTERM and partial-Zarr recovery gate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.scripts.pluto_ready_manifest import load_manifest
from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver


pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_interrupt]


def _committed_counts(path):
    capture = zarr_open_from_lmdb_store(str(path), mode="r", readahead=False)
    try:
        return list(capture.attrs["capture_records_written_by_receiver"])
    finally:
        capture.store.close()


def _mapped_usb_devices(path):
    devices = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            _receiver_port, address = fields
            devices.add((1, int(address)))
        elif len(fields) == 3:
            _receiver_port, bus, address = fields
            devices.add((int(bus), int(address)))
        else:
            pytest.fail(f"invalid device mapping row: {line!r}")
    return devices


def test_real_collector_sigterm_is_incomplete_readable_and_releases_radios(
    attached_plutos, pytestconfig, tmp_path
):
    config = pytestconfig.getoption("--radio-capture-config")
    mapping = pytestconfig.getoption("--radio-device-mapping")
    manifest = pytestconfig.getoption("--radio-ready-manifest")
    if config is None:
        pytest.fail("--radio-interrupt requires --radio-capture-config")
    for label, path in (
        ("capture config", config),
        ("device mapping", mapping),
        ("ready manifest", manifest),
    ):
        if not path.is_file():
            pytest.fail(f"{label} does not exist: {path}")

    attached_devices = {(radio.bus, radio.address) for radio in attached_plutos}
    mapped_devices = _mapped_usb_devices(mapping)
    if mapped_devices != attached_devices:
        pytest.fail(
            "device mapping is stale: "
            f"mapped={sorted(mapped_devices)} attached={sorted(attached_devices)}; "
            "rerun prepare_direct_usb_boot.sh"
        )
    ready = load_manifest(manifest)
    manifest_devices = {
        (int(radio["usb_bus"]), int(radio["usb_address"])) for radio in ready["radios"]
    }
    if manifest_devices != attached_devices:
        pytest.fail(
            "ready manifest is stale: "
            f"manifest={sorted(manifest_devices)} "
            f"attached={sorted(attached_devices)}; rerun prepare_direct_usb_boot.sh"
        )
    mapping_sha256 = hashlib.sha256(mapping.read_bytes()).hexdigest()
    assert ready["device_mapping_sha256"] == mapping_sha256

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "collector"
    output_dir.mkdir()
    log_path = tmp_path / "collector-subprocess.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), environment.get("PYTHONPATH", "")]
    )
    environment["SPF_DIRECT_USB_READY_FILE"] = str(manifest)
    command = [
        sys.executable,
        str(repo_root / "spf/mavlink_radio_collection.py"),
        "--fake-drone",
        "--no-ultrasonic",
        "-c",
        str(config),
        "-m",
        str(mapping),
        "-r",
        "center",
        "-t",
        "INTERRUPT_TEST",
        "-n",
        "10000",
        "--temp",
        str(output_dir),
    ]

    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    deadline = time.monotonic() + 90
    partial_path = None
    committed_before_interrupt = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"collector exited before interruption ({process.returncode}); "
                f"log follows:\n{log_path.read_text()}"
            )
        stores = list(output_dir.glob("*.zarr.tmp"))
        if len(stores) == 1:
            partial_path = stores[0]
            try:
                counts = _committed_counts(partial_path)
            except Exception:
                counts = []
            if len(counts) == len(attached_plutos) and min(counts) >= 2:
                committed_before_interrupt = counts
                break
        time.sleep(0.1)
    if committed_before_interrupt is None:
        process.kill()
        process.wait(timeout=10)
        pytest.fail(
            f"collector made no interruptible progress:\n{log_path.read_text()}"
        )

    process.terminate()
    try:
        return_code = process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        pytest.fail(
            f"collector did not clean up after SIGTERM:\n{log_path.read_text()}"
        )
    assert return_code != 0
    assert partial_path is not None and partial_path.is_dir()
    assert not list(output_dir.glob("*.zarr"))

    partial = zarr_open_from_lmdb_store(str(partial_path), mode="r", readahead=False)
    try:
        assert partial.attrs["capture_status"] == "incomplete"
        assert partial.attrs["capture_error_type"] == "CaptureInterrupted"
        assert "SIGTERM" in partial.attrs["capture_error_message"]
        committed_after_interrupt = list(
            partial.attrs["capture_records_written_by_receiver"]
        )
        assert all(
            after >= before
            for before, after in zip(
                committed_before_interrupt,
                committed_after_interrupt,
                strict=True,
            )
        )
    finally:
        partial.store.close()

    # Cleanup must release every claimed vendor interface immediately. A new
    # one-frame request by serial is the practical field-recovery assertion.
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            capture = receiver.capture(samples_per_channel=16_384, frame_count=1)
            assert len(capture.frames) == 1
