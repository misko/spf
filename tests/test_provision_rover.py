"""provision_rover.sh replaces setup.sh; setup.sh must refuse to run.

setup.sh is unsafe on current Raspberry Pi OS images — it runs both board
blocks ungated, writes /boot/config.txt (a "DO NOT EDIT" stub since 2024), and
disables wifi before the static address is ever proven. It is kept only as a
historical reference, so the guard that stops it executing is load-bearing and
is asserted here.

These are text assertions on the scripts, matching the idiom in
test_rover_capture_profile.py: the scripts touch real hardware, so their
contents are what CI can check.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"
PROVISION = ROVER_ROOT / "provision_rover.sh"
SETUP = ROVER_ROOT / "setup.sh"

STAGES = ("identity", "network", "wifi-off", "base", "firmware", "audit", "all")


def test_setup_sh_refuses_to_run():
    """The deprecation guard must exit non-zero before doing anything."""
    result = subprocess.run(
        ["bash", str(SETUP), "1"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0, "setup.sh must fail closed, not run"
    combined = result.stdout + result.stderr
    assert "DEPRECATED" in combined
    assert "provision_rover.sh" in combined, "must name its replacement"


def test_setup_sh_guard_precedes_every_side_effect():
    """`exit 1` must come before the first EXECUTABLE line that mutates a rover.

    Comment lines are skipped deliberately: the deprecation block above the
    guard names the very commands it is protecting against, so a naive substring
    search reports them as unguarded.
    """
    lines = SETUP.read_text().splitlines()
    guard_line = next(
        i for i, line in enumerate(lines) if line.strip() == "exit 1"
    )
    for destructive in ("git clone", "apt-get", "/boot/config.txt", "reboot"):
        offenders = [
            i
            for i, line in enumerate(lines)
            if destructive in line and not line.lstrip().startswith("#")
        ]
        assert all(i > guard_line for i in offenders), (
            f"{destructive!r} runs at line(s) "
            f"{[i + 1 for i in offenders if i < guard_line]} before the "
            f"deprecation exit (line {guard_line + 1}); setup.sh could still "
            "mutate a rover"
        )


@pytest.mark.parametrize("stage", STAGES)
def test_provision_rover_declares_every_documented_stage(stage):
    body = PROVISION.read_text()
    assert f"{stage})" in body, f"stage {stage!r} is documented but not dispatched"


def test_provision_rover_rejects_unknown_rover_id():
    result = subprocess.run(
        ["bash", str(PROVISION), "9", "--stage", "audit"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Unsupported rover_id" in result.stdout + result.stderr


def test_provision_rover_requires_a_stage():
    result = subprocess.run(
        ["bash", str(PROVISION), "1"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert "--stage is required" in result.stdout + result.stderr


def test_provision_rover_delegates_rather_than_reimplementing():
    """Each hardware step must call the existing, separately-tested script."""
    body = PROVISION.read_text()
    for delegate in (
        "configure_rover_network.sh",
        "install_deps.sh",
        "flash_ardupilot.sh",
        "check_and_set_pluto.sh",
        "configure_direct_usb_boot.sh",
        "audit_rover.sh",
        "device_mapping.sh",
    ):
        assert delegate in body, f"provision_rover.sh should delegate to {delegate}"


def test_wifi_is_never_disabled_in_the_same_stage_that_sets_the_address():
    """The gate that stops a rover being stranded with no reachable path.

    `network` must leave wifi up; only the separate `wifi-off` stage disables
    it, and configure_rover_network.sh refuses that unless the static address
    already answers.
    """
    body = PROVISION.read_text()
    network_body = body.split("stage_network() {", 1)[1].split("\n}", 1)[0]
    assert "static-only" in network_body
    assert "disable-wifi" not in network_body, (
        "the network stage must not disable wifi — that belongs in wifi-off, "
        "after the static address is proven"
    )

    all_body = body.split("    all)", 1)[1].split(";;", 1)[0]
    assert "stage_wifi_off" not in all_body, (
        "--stage all must stop before wifi-off; it needs a human gate"
    )
    assert "stage_firmware" not in all_body, (
        "--stage all must stop before firmware; it needs a reboot first"
    )


def test_provision_rover_guards_rover_id_like_the_boot_scripts():
    """Same ^[1-4]$ contract as drone_run.sh and the direct-USB scripts."""
    assert "^[1-4]$" in PROVISION.read_text()
    for script in (
        "drone_run.sh",
        "prepare_direct_usb_boot.sh",
        "configure_direct_usb_boot.sh",
    ):
        assert "^[1-4]$" in (ROVER_ROOT / script).read_text(), (
            f"{script} disagrees with provision_rover.sh about supported rover ids"
        )
