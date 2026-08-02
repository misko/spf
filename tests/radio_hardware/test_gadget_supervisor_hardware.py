"""Opt-in crash/rebind gate for the on-Pluto direct-USB supervisor.

The test kills only ``sdr_usb_gadget``. It never resets the radio, changes RF
configuration, enables TX, or writes firmware. Passing proves the supervisor
publishes a fresh process nonce, rebinds the composite gadget, and preserves
standard USB-IIO.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import usb1

from spf.scripts.pluto_multi_firmware import MultiPlutoFirmwareManager
from spf.sdrpluto.direct_usb_protocol import RuntimeState
from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver


pytestmark = [
    pytest.mark.radio_hardware,
    pytest.mark.radio_crash_recovery,
]


def _manager(expected_count: int) -> MultiPlutoFirmwareManager:
    root = Path(__file__).resolve().parents[2]
    return MultiPlutoFirmwareManager(
        image=Path("/tmp/pluto.dfu"),
        image_sha256="unused-by-ssh-only-crash-gate",
        ssh_config=root / "data_collection" / "rover" / "rover_v3.1" / "ssh_config",
        ssh_password="analog",
        state_root=Path("/tmp/spf-gadget-supervisor-state"),
        expected_count=expected_count,
    )


def _usb_iio_has_serial(serial: str) -> bool:
    result = subprocess.run(
        ["iio_info", "-s"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and any(
        serial in line and "[usb:" in line for line in result.stdout.splitlines()
    )


def _wait_for_new_process(serial: str, old_nonce: bytes, timeout: float = 45.0):
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as receiver:
                status = receiver.query_runtime_status()
                if status.process_nonce != old_nonce and _usb_iio_has_serial(serial):
                    return receiver.identity, status
        except (RuntimeError, usb1.USBError) as error:
            last_error = error
        time.sleep(0.25)
    raise AssertionError(
        f"{serial}: fresh direct-USB process and USB-IIO did not return: {last_error}"
    )


def test_gadget_crash_rebinds_composite_and_preserves_iio(
    attached_plutos, pytestconfig, radio_report_dir
):
    manager = _manager(len(attached_plutos))
    samples = min(pytestconfig.getoption("--radio-samples"), 524_288)
    report = []
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            before_identity = receiver.identity
            before = receiver.query_runtime_status()

        crash = manager._ssh(
            radio.serial,
            "child=$(pidof sdr_usb_gadget); "
            'test -n "$child"; '
            "parent=$(awk '/^PPid:/ {print $2}' /proc/$child/status); "
            'printf \'supervisor=%s child=%s\\n\' "$parent" "$child"; '
            "kill -KILL $child",
            check=False,
            timeout=15,
        )
        after_identity, after = _wait_for_new_process(
            radio.serial, before.process_nonce
        )
        assert after.lifecycle_state == RuntimeState.IDLE
        assert after.process_nonce != before.process_nonce
        assert after.boot_id == before.boot_id
        assert after_identity.serial == before_identity.serial
        assert after_identity.port_path == before_identity.port_path

        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            frames = list(
                receiver.stream_frames(
                    samples_per_channel=samples,
                    frame_count=3,
                    queue_depth=1,
                )
            )
            final = receiver.query_runtime_status()
        assert [frame.metadata.buffer_sequence for frame in frames] == [0, 1, 2]
        assert final.completed_frame_count == after.completed_frame_count + 3

        services = manager._ssh(
            radio.serial,
            "pidof iiod; pidof sdr_usb_gadget; "
            "logread | grep 'rebound composite USB gadget' | tail -1",
            timeout=15,
        )
        assert "rebound composite USB gadget" in services.stdout
        report.append(
            {
                "serial": radio.serial,
                "port_path": list(radio.port_path),
                "address_before": before_identity.address,
                "address_after": after_identity.address,
                "boot_id": before.boot_id.hex(),
                "process_nonce_before": before.process_nonce.hex(),
                "process_nonce_after": after.process_nonce.hex(),
                "crash_command_stdout": crash.stdout.strip(),
                "service_evidence": services.stdout.strip().splitlines(),
            }
        )

    (radio_report_dir / "gadget_supervisor_crash_recovery.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
