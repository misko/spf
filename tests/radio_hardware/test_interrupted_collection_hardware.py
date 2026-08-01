"""Explicit real-radio SIGTERM and partial-Zarr recovery gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.scripts.interrupted_capture_timing import (
    interruption_progress_timeout_seconds,
)
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


def _assert_committed_prefix_is_valid(capture, counts, attached_plutos):
    receiver_names = sorted(capture.receivers.keys())
    assert len(receiver_names) == len(attached_plutos)
    attached_serials = {radio.serial for radio in attached_plutos}
    recorded_serials = set()
    for name, committed in zip(receiver_names, counts, strict=True):
        receiver = capture.receivers[name]
        recorded_serials.add(receiver.attrs["sdr_serial"])
        assert receiver.attrs["rx_transport"] == "direct_usb"
        assert receiver.attrs["gain_metadata_protocol_version"] == 2
        assert receiver.attrs["firmware_verified"] is True
        assert 0 < committed <= receiver.signal_matrix.shape[0]
        timestamps = receiver.system_timestamp[:committed]
        assert len(timestamps) == committed
        assert all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))
        assert receiver.gain_metadata_valid[:committed].all()
        assert receiver.rssi_metadata_valid[:committed].all()
    assert recorded_serials == attached_serials


def test_real_collector_interruption_is_fail_closed_readable_and_releases_radios(
    attached_plutos, pytestconfig, tmp_path, radio_report_dir
):
    config = pytestconfig.getoption("--radio-capture-config")
    mapping = pytestconfig.getoption("--radio-device-mapping")
    manifest = pytestconfig.getoption("--radio-ready-manifest")
    if config is None:
        pytest.fail("--radio-interrupt requires --radio-capture-config")
    interrupt_name = pytestconfig.getoption("--radio-interrupt-signal")
    minimum_records = pytestconfig.getoption("--radio-interrupt-min-records")
    if minimum_records < 1:
        pytest.fail("--radio-interrupt-min-records must be positive")
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

    progress_timeout_seconds = interruption_progress_timeout_seconds(minimum_records)
    deadline = time.monotonic() + progress_timeout_seconds
    partial_path = None
    committed_before_interrupt = None
    last_counts = []
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
            last_counts = counts
            if (
                len(counts) == len(attached_plutos)
                and min(counts) >= minimum_records
            ):
                committed_before_interrupt = counts
                break
        time.sleep(0.1)
    if committed_before_interrupt is None:
        process.kill()
        process.wait(timeout=10)
        pytest.fail(
            "collector did not reach the requested interruption boundary "
            f"within {progress_timeout_seconds:.1f}s; "
            f"last committed counts={last_counts!r}:\n{log_path.read_text()}"
        )

    started_wait = time.monotonic()
    if interrupt_name == "sigint":
        process.send_signal(signal.SIGINT)
    elif interrupt_name == "sigterm":
        process.terminate()
    elif interrupt_name == "sigkill":
        process.kill()
    else:  # argparse choices make this defensive only.
        pytest.fail(f"unsupported interruption signal: {interrupt_name}")
    try:
        return_code = process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        pytest.fail(
            f"collector did not exit after {interrupt_name}:\n{log_path.read_text()}"
        )
    exit_seconds = time.monotonic() - started_wait
    expected_return_code = {
        "sigint": 128 + signal.SIGINT,
        "sigterm": 128 + signal.SIGTERM,
        "sigkill": -signal.SIGKILL,
    }[interrupt_name]
    assert return_code == expected_return_code
    assert partial_path is not None and partial_path.is_dir()
    assert not list(output_dir.glob("*.zarr"))

    partial = zarr_open_from_lmdb_store(str(partial_path), mode="r", readahead=False)
    try:
        expected_status = "in_progress" if interrupt_name == "sigkill" else "incomplete"
        assert partial.attrs["capture_status"] == expected_status
        if interrupt_name == "sigkill":
            assert "capture_error_type" not in partial.attrs
            assert "capture_error_message" not in partial.attrs
        else:
            assert partial.attrs["capture_error_type"] == "CaptureInterrupted"
            assert interrupt_name.upper() in partial.attrs["capture_error_message"]
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
        _assert_committed_prefix_is_valid(
            partial, committed_after_interrupt, attached_plutos
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

    report = {
        "status": "pass",
        "signal": interrupt_name,
        "minimum_records": minimum_records,
        "committed_before_interrupt": committed_before_interrupt,
        "committed_after_interrupt": committed_after_interrupt,
        "capture_status": expected_status,
        "return_code": return_code,
        "exit_seconds": exit_seconds,
        "progress_timeout_seconds": progress_timeout_seconds,
        "partial_zarr": str(partial_path),
        "serials": [radio.serial for radio in attached_plutos],
    }
    report_path = radio_report_dir / (
        f"interruption-{interrupt_name}-{minimum_records}-records.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
