import hashlib
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"
RECONCILER = ROVER_ROOT / "reconcile_rover_boot_units.sh"
MANAGED_UNITS = (
    "spf-rover-update.service",
    "spf-pluto-direct-usb.service",
    "spf-direct-usb-preflight.service",
    "mavlink_controller.service",
)
ENABLED_UNITS = (
    "spf-rover-update.service",
    "spf-pluto-direct-usb.service",
    "mavlink_controller.service",
)


def make_fake_systemctl(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    enabled_dir = tmp_path / "enabled"
    # exist_ok, so a test may build the shim more than once for the same
    # tmp_path -- the journald tests run the reconciler repeatedly.
    enabled_dir.mkdir(exist_ok=True)
    log = tmp_path / "systemctl.log"
    fake = tmp_path / "systemctl"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"$SPF_TEST_SYSTEMCTL_LOG"
operation="$1"
shift
case "$operation" in
    is-enabled)
        [[ "${1:-}" != "--quiet" ]] || shift
        [[ -f "$SPF_TEST_ENABLED_DIR/$1" ]]
        ;;
    enable)
        for unit in "$@"; do
            touch "$SPF_TEST_ENABLED_DIR/$unit"
        done
        ;;
    daemon-reload)
        ;;
    *)
        exit 2
        ;;
esac
"""
    )
    fake.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "SPF_TEST_ENABLED_DIR": str(enabled_dir),
            "SPF_TEST_SYSTEMCTL_LOG": str(log),
        }
    )
    return fake, environment


def run_reconciler(
    systemd_dir: Path,
    state_dir: Path,
    fake_systemctl: Path,
    environment: dict[str, str],
    cli_link: Path | None = None,
    journald_dropin: Path | None = None,
    journal_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    # --cli-link is never optional in practice: the reconciler also converges
    # the /usr/local/bin/rover symlink, and a test that let it fall through to
    # the default would reconcile the developer's real system. The same is true
    # of the journald paths, which default to /etc and /var/log.
    if cli_link is None:
        cli_link = state_dir.parent / "unused-cli-link"
    if journald_dropin is None:
        journald_dropin = state_dir.parent / "unused-journald.conf.d" / "spf.conf"
    if journal_dir is None:
        journal_dir = state_dir.parent / "unused-journal"
    return subprocess.run(
        [
            str(RECONCILER),
            "--systemd-dir",
            str(systemd_dir),
            "--state-dir",
            str(state_dir),
            "--systemctl",
            str(fake_systemctl),
            "--cli-link",
            str(cli_link),
            "--journald-dropin",
            str(journald_dropin),
            "--journal-dir",
            str(journal_dir),
            "--unprivileged",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_stale_units_are_verified_and_request_exactly_one_reboot(tmp_path):
    systemd_dir = tmp_path / "systemd"
    state_dir = tmp_path / "state"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)

    first = run_reconciler(systemd_dir, state_dir, fake_systemctl, environment)

    assert first.returncode == 10, first.stderr
    assert "single reconciliation reboot" in first.stdout
    for unit in MANAGED_UNITS:
        assert (systemd_dir / unit).read_bytes() == (ROVER_ROOT / unit).read_bytes()
    for unit in ENABLED_UNITS:
        assert (tmp_path / "enabled" / unit).exists()

    marker = state_dir / "boot-unit-reconcile-attempt"
    marker_text = marker.read_text()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    assert f"git_commit={git_commit}" in marker_text
    for unit in MANAGED_UNITS:
        digest = hashlib.sha256((ROVER_ROOT / unit).read_bytes()).hexdigest()
        assert f"unit_sha256[{unit}]={digest}" in marker_text

    second = run_reconciler(systemd_dir, state_dir, fake_systemctl, environment)

    assert second.returncode == 0, second.stderr
    assert "already match Git state" in second.stdout
    assert not marker.exists()


def test_persistent_drift_refuses_a_second_reboot_attempt(tmp_path):
    systemd_dir = tmp_path / "systemd"
    state_dir = tmp_path / "state"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    first = run_reconciler(systemd_dir, state_dir, fake_systemctl, environment)
    assert first.returncode == 10, first.stderr

    stale_unit = systemd_dir / MANAGED_UNITS[0]
    stale_unit.write_text("changed after verified installation\n")
    log_before = (tmp_path / "systemctl.log").read_text()

    second = run_reconciler(systemd_dir, state_dir, fake_systemctl, environment)

    assert second.returncode == 75
    assert "Refusing another reboot" in second.stderr
    assert stale_unit.read_text() == "changed after verified installation\n"
    assert (tmp_path / "systemctl.log").read_text() == log_before


def test_install_failure_never_requests_a_reboot(tmp_path):
    systemd_dir = tmp_path / "systemd"
    state_dir = tmp_path / "state"
    systemd_dir.write_text("not a directory\n")
    fake_systemctl, environment = make_fake_systemctl(tmp_path)

    result = run_reconciler(systemd_dir, state_dir, fake_systemctl, environment)

    assert result.returncode not in {0, 10}
    assert not (state_dir / "boot-unit-reconcile-attempt").exists()
    assert not (tmp_path / "systemctl.log").exists()


# ------------------------------------------------------ rover CLI symlink ---
#
# `rover install` joined provision_rover.sh's base stage hours after Rover 4 had
# already run that stage, and nothing re-runs a provisioning stage -- so Rover 4
# had no /usr/local/bin/rover at all. The reconciler converges it for the same
# reason it converges units: a rover provisioned at an older commit must still
# end up in the current desired state without anyone remembering to intervene.


def test_absent_cli_symlink_is_created(tmp_path):
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    cli_link = tmp_path / "bin" / "rover"
    cli_link.parent.mkdir()

    result = run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )

    assert result.returncode == 10, result.stderr
    assert cli_link.is_symlink()
    assert cli_link.resolve() == (ROVER_ROOT / "rover").resolve()


def test_cli_symlink_reconciliation_is_idempotent(tmp_path):
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    cli_link = tmp_path / "bin" / "rover"
    cli_link.parent.mkdir()

    run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )
    second = run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )

    assert "rover CLI symlink is current" in second.stdout
    assert cli_link.resolve() == (ROVER_ROOT / "rover").resolve()


def test_stale_cli_symlink_is_repointed(tmp_path):
    """A symlink into a moved or renamed checkout is as broken as a missing one."""
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    cli_link = tmp_path / "bin" / "rover"
    cli_link.parent.mkdir()
    cli_link.symlink_to("/bin/true")

    run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )

    assert cli_link.resolve() == (ROVER_ROOT / "rover").resolve()


def test_a_real_file_at_the_cli_path_is_never_clobbered(tmp_path):
    """Someone put that file there by hand; `rover install` refuses too."""
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    cli_link = tmp_path / "bin" / "rover"
    cli_link.parent.mkdir()
    cli_link.write_text("handwritten\n")

    result = run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )

    assert cli_link.read_text() == "handwritten\n"
    assert "not a symlink" in result.stderr


def test_cli_symlink_failure_never_changes_the_unit_exit_status(tmp_path):
    """The 0/10/75 contract belongs to the units; the CLI link must not touch it."""
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    # An unwritable parent makes reconciliation fail the way a non-root boot
    # would, without needing to be root to arrange it.
    unwritable = tmp_path / "unwritable"
    unwritable.mkdir(mode=0o500)
    cli_link = unwritable / "rover"

    first = run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )
    second = run_reconciler(
        systemd_dir, tmp_path / "state", fake_systemctl, environment, cli_link
    )

    assert first.returncode == 10, first.stderr
    assert second.returncode == 0, second.stderr
    assert not cli_link.exists()


# ---------------------------------------------------- journald persistence ---
#
# Rovers 1 and 4 ran journald with Storage=volatile and lost the journal at
# every reboot; rovers 2 and 3 did not. On 2026-08-04 Rover 4 recorded a capture
# with one receive channel 24 dB down and the AGC railed, and the journal for
# that boot no longer existed by the time anyone looked. The reconciler
# converges the setting for the same reason it converges the CLI symlink: a
# provisioning stage never re-runs, so only boot convergence reaches the fleet.


def journald_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "journald.conf.d" / "10-spf-persistent.conf", tmp_path / "journal"


def run_with_journald(tmp_path: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir(exist_ok=True)
    fake_systemctl, environment = kwargs.pop("systemctl", (None, None))
    if fake_systemctl is None:
        fake_systemctl, environment = make_fake_systemctl(tmp_path)
    dropin, journal_dir = journald_paths(tmp_path)
    return run_reconciler(
        systemd_dir,
        tmp_path / "state",
        fake_systemctl,
        environment,
        journald_dropin=kwargs.pop("journald_dropin", dropin),
        journal_dir=kwargs.pop("journal_dir", journal_dir),
        **kwargs,
    )


def test_journald_persistence_is_configured_when_absent(tmp_path):
    dropin, journal_dir = journald_paths(tmp_path)

    result = run_with_journald(tmp_path)

    assert result.returncode == 10, result.stderr
    text = dropin.read_text()
    assert "Storage=persistent" in text
    assert "SystemMaxUse=1G" in text
    assert journal_dir.is_dir()


def test_journald_reconciliation_is_idempotent(tmp_path):
    dropin, _ = journald_paths(tmp_path)
    run_with_journald(tmp_path)
    written = dropin.read_text()

    second = run_with_journald(tmp_path)

    assert second.returncode == 0, second.stderr
    assert "journald persistence is configured" in second.stdout
    assert dropin.read_text() == written


def test_a_hand_edited_journald_dropin_is_rewritten(tmp_path):
    """The drop-in says it is managed; drift back to volatile must not stick."""
    dropin, _ = journald_paths(tmp_path)
    run_with_journald(tmp_path)
    dropin.write_text("[Journal]\nStorage=volatile\n")

    run_with_journald(tmp_path)

    assert "Storage=persistent" in dropin.read_text()


def test_journald_is_never_restarted(tmp_path):
    """A journald restart cuts the log streams of services already running.

    Buying one boot of persistence at the price of silencing the capture chain
    for the rest of that boot is the wrong trade; the setting takes effect at
    the next boot instead.
    """
    fake_systemctl, environment = make_fake_systemctl(tmp_path)

    run_with_journald(tmp_path, systemctl=(fake_systemctl, environment))

    log = (tmp_path / "systemctl.log").read_text()
    assert "journald" not in log
    assert "restart" not in log


def test_journald_failure_never_changes_the_unit_exit_status(tmp_path):
    """As with the CLI symlink, the 0/10/75 contract belongs to the units."""
    unwritable = tmp_path / "unwritable"
    unwritable.mkdir(mode=0o500)

    first = run_with_journald(
        tmp_path,
        journald_dropin=unwritable / "conf.d" / "spf.conf",
        journal_dir=unwritable / "journal",
    )
    second = run_with_journald(
        tmp_path,
        journald_dropin=unwritable / "conf.d" / "spf.conf",
        journal_dir=unwritable / "journal",
    )

    assert first.returncode == 10, first.stderr
    assert second.returncode == 0, second.stderr
    assert "journal stays volatile" in first.stderr


def test_journald_only_converges_nothing_else(tmp_path):
    """provision_rover.sh's base stage uses this; it must not install units."""
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    fake_systemctl, environment = make_fake_systemctl(tmp_path)
    dropin, journal_dir = journald_paths(tmp_path)
    cli_link = tmp_path / "bin" / "rover"
    cli_link.parent.mkdir()

    result = subprocess.run(
        [
            str(RECONCILER),
            "--systemd-dir",
            str(systemd_dir),
            "--state-dir",
            str(tmp_path / "state"),
            "--systemctl",
            str(fake_systemctl),
            "--cli-link",
            str(cli_link),
            "--journald-dropin",
            str(dropin),
            "--journal-dir",
            str(journal_dir),
            "--journald-only",
            "--unprivileged",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Storage=persistent" in dropin.read_text()
    assert list(systemd_dir.iterdir()) == []
    assert not cli_link.exists()
    assert not (tmp_path / "systemctl.log").exists()
