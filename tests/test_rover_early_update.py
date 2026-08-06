import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"
UPDATER = ROVER_ROOT / "update_spf_before_boot.sh"


def _write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    return path


def _make_checkout(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "config", "user.email", "test@example.com"],
        check=True,
    )
    (seed / "version").write_text("one\n")
    subprocess.run(["git", "-C", str(seed), "add", "version"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "one"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "remote", "add", "origin", str(remote)],
        check=True,
    )
    subprocess.run(["git", "-C", str(seed), "push", "-u", "origin", "main"], check=True)
    subprocess.run(["git", "clone", "-b", "main", str(remote), str(checkout)], check=True)
    return seed, checkout


def _run_updater(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    events = tmp_path / "events"
    reconciler = _write_executable(
        tmp_path / "reconcile", 'printf "reconcile\\n" >>"$SPF_TEST_EVENTS"\n',
    )
    installer = _write_executable(
        tmp_path / "install-deps", 'printf "deps\\n" >>"$SPF_TEST_EVENTS"\n',
    )
    python = _write_executable(
        tmp_path / "python", 'printf "pip %s\\n" "$*" >>"$SPF_TEST_EVENTS"\n',
    )
    reboot = _write_executable(
        tmp_path / "reboot", 'printf "reboot\\n" >>"$SPF_TEST_EVENTS"\n',
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SPF_UPDATE_REPO_ROOT": str(checkout),
            "SPF_UPDATE_PROFILE_ENV": str(tmp_path / "no-profile-env"),
            "SPF_BOOT_UNIT_RECONCILER": str(reconciler),
            "SPF_INSTALL_DEPS": str(installer),
            "SPF_PYTHON": str(python),
            "SPF_TEST_REBOOT_COMMAND": str(reboot),
            "SPF_UPDATE_REBOOT_DELAY_SECONDS": "0",
            "SPF_UPDATE_REMOTE_WAIT_SECONDS": "1",
            # 0 disables the eth0 uplink gate: these tests exercise the
            # git/reboot logic against a local clone, and the dev box and
            # CI do not have a rover's interface names.
            "SPF_UPDATE_UPLINK_WAIT_SECONDS": "0",
            "SPF_UPDATE_GIT_TIMEOUT_SECONDS": "1",
            "SPF_TEST_EVENTS": str(events),
        }
    )
    result = subprocess.run(
        [str(UPDATER)],
        cwd=checkout,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    result.events = events.read_text().splitlines() if events.exists() else []
    return result


def test_repository_update_is_a_hard_prerequisite_of_radio_preparation():
    updater_unit = (ROVER_ROOT / "spf-rover-update.service").read_text()
    loader_unit = (ROVER_ROOT / "spf-pluto-direct-usb.service").read_text()
    mavlink_unit = (ROVER_ROOT / "mavlink_controller.service").read_text()

    assert "ExecStart=/home/pi/spf/data_collection/rover/rover_v3.1/update_spf_before_boot.sh" in updater_unit
    assert "Before=spf-pluto-direct-usb.service" in updater_unit
    assert "Requires=spf-rover-update.service" in loader_unit
    assert "After=spf-rover-update.service" in loader_unit
    assert "Requires=spf-pluto-direct-usb.service" in mavlink_unit


def test_remote_update_refreshes_installation_then_requests_one_reboot(tmp_path):
    seed, checkout = _make_checkout(tmp_path)
    old_head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    (seed / "version").write_text("two\n")
    subprocess.run(["git", "-C", str(seed), "commit", "-am", "two"], check=True)
    subprocess.run(["git", "-C", str(seed), "push"], check=True)

    result = _run_updater(tmp_path, checkout)

    assert result.returncode == 0, result.stderr
    new_head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert new_head != old_head
    assert (checkout / "version").read_text() == "two\n"
    assert result.events == ["deps", f"pip -m pip install -e {checkout}", "reconcile", "reboot"]


def test_unchanged_checkout_skips_expensive_install_and_reboot(tmp_path):
    _, checkout = _make_checkout(tmp_path)

    result = _run_updater(tmp_path, checkout)

    assert result.returncode == 0, result.stderr
    assert result.events == ["reconcile"]
    assert "Repository is unchanged" in result.stdout


def test_offline_remote_is_bounded_and_does_not_block_local_boot(tmp_path):
    _, checkout = _make_checkout(tmp_path)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "set-url", "origin", str(tmp_path / "absent")],
        check=True,
    )

    result = _run_updater(tmp_path, checkout)

    assert result.returncode == 0, result.stderr
    assert result.events == ["reconcile"]
    assert "continuing with checked-out code" in result.stdout


def test_tracked_dirty_checkout_fails_closed_without_mutation(tmp_path):
    _, checkout = _make_checkout(tmp_path)
    (checkout / "version").write_text("local operator change\n")

    result = _run_updater(tmp_path, checkout)

    assert result.returncode != 0
    assert result.events == []
    assert "Tracked working-tree changes" in result.stderr
    assert (checkout / "version").read_text() == "local operator change\n"


def test_late_mission_launcher_no_longer_updates_repository():
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()

    assert "maybe_self_update" not in launcher
    assert "git pull" not in launcher


# ---------------------------------------------------------------- uplink gate ---
#
# Rover 1, 2026-08-06: spf-rover-update.service started 29 ms after
# network.target and its 12 s Git budget expired before eth0 even had carrier
# (+3.4 s). network-online.target was no help -- it went green 800 ms into boot,
# satisfied by a DHCP lease on eth1 at 192.168.2.10, which is a PLUTO's USB-net
# interface. Three boots running, the rover stayed on stale code and kept
# capturing at the old carrier frequency.


def test_the_updater_waits_on_the_named_lan_interface_not_a_timer():
    updater = UPDATER.read_text()

    assert 'UPLINK_INTERFACE="${SPF_UPDATE_UPLINK_INTERFACE:-eth0}"' in updater
    # Interface-specific by name: an any-interface check is satisfied by a Pluto.
    assert 'ip -4 addr show dev "$UPLINK_INTERFACE" scope global' in updater
    assert '/sys/class/net/${UPLINK_INTERFACE}/carrier' in updater
    # The gate runs BEFORE the git budget starts.
    assert updater.index("wait_for_uplink") < updater.index("fetch_main_bounded()")
    main_body = updater.rsplit("main() {", 1)[1]
    assert main_body.index("wait_for_uplink") < main_body.index("fetch_main_bounded")


def test_an_offline_boot_gives_up_on_carrier_instead_of_burning_the_budget():
    """A rover in a field has no cable; it must not stall every boot waiting."""
    updater = UPDATER.read_text()
    body = updater.split("wait_for_uplink() {", 1)[1].split("\n}\n", 1)[0]

    assert "UPLINK_CARRIER_GRACE_SECONDS" in body
    assert "offline boot" in body
    # And it still fails open -- no uplink must never block the boot.
    assert "return 1" in body
    assert "No usable uplink on ${UPLINK_INTERFACE}" in updater
