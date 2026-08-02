from __future__ import annotations

import json
import os
from pathlib import Path

from spf.capture_watchdog import collect_watchdog_sample, rotate_jsonl_if_needed


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_watchdog_sample_distinguishes_process_usb_storage_and_host(tmp_path):
    sysfs = tmp_path / "sys" / "bus" / "usb" / "devices"
    radio = sysfs / "1-1.4"
    _write(radio / "idVendor", "0456\n")
    _write(radio / "idProduct", "b673\n")
    _write(radio / "serial", "radio-a\n")
    _write(radio / "busnum", "1\n")
    _write(radio / "devnum", "7\n")
    status_path = tmp_path / "capture_status.json"
    status_path.write_text(
        json.dumps(
            {
                "capture_name": "capture.zarr.tmp",
                "state": "collecting",
                "records_written_by_receiver": [42, 41],
                "updated_unix": 995.0,
                "incident_id": "incident-42",
                "error_source": "receiver:usb:1.4.5",
                "error_type": "DirectUsbTransferTimeoutError",
                "error_message": "deadline expired",
            }
        )
    )

    sample = collect_watchdog_sample(
        pid=os.getpid(),
        status_path=status_path,
        storage_path=tmp_path,
        expected_plutos=2,
        usb_sysfs_root=sysfs,
        wall_time=lambda: 1000.0,
        monotonic_time=lambda: 20.0,
        previous_monotonic=17.5,
    )

    assert sample["watchdog_version"] == 1
    assert sample["process"]["alive"] is True
    assert sample["process"]["threads"] >= 1
    assert sample["host"]["watchdog_scheduling_gap_seconds"] == 2.5
    assert sample["storage"]["free_bytes"] > 0
    assert sample["capture"]["status_age_seconds"] == 5.0
    assert sample["capture"]["records_written_by_receiver"] == [42, 41]
    assert sample["capture"]["incident_id"] == "incident-42"
    assert sample["capture"]["error_source"] == "receiver:usb:1.4.5"
    assert sample["capture"]["error_type"] == "DirectUsbTransferTimeoutError"
    assert sample["usb"]["expected_plutos"] == 2
    assert sample["usb"]["observed_plutos"] == 1
    assert sample["usb"]["missing"] is True
    assert sample["usb"]["devices"][0]["serial"] == "radio-a"


def test_watchdog_journal_rotation_is_bounded_and_recoverable(tmp_path):
    journal = tmp_path / "watchdog.jsonl"
    journal.write_bytes(b"x" * 101)

    rotated = rotate_jsonl_if_needed(journal, maximum_bytes=100)

    assert rotated == tmp_path / "watchdog.jsonl.1"
    assert rotated.read_bytes() == b"x" * 101
    assert not journal.exists()


def test_scheduling_gap_threshold_tracks_configured_sample_interval(tmp_path):
    sample = collect_watchdog_sample(
        pid=os.getpid(),
        status_path=tmp_path / "missing-status.json",
        storage_path=tmp_path,
        expected_plutos=0,
        usb_sysfs_root=tmp_path / "missing-sysfs",
        wall_time=lambda: 1000.0,
        monotonic_time=lambda: 15.0,
        previous_monotonic=10.0,
        expected_interval_seconds=5.0,
    )

    assert "watchdog_scheduling_gap" not in sample["conditions"]
