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
    enabled_dir.mkdir()
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
) -> subprocess.CompletedProcess[str]:
    # --cli-link is never optional in practice: the reconciler also converges
    # the /usr/local/bin/rover symlink, and a test that let it fall through to
    # the default would reconcile the developer's real system.
    if cli_link is None:
        cli_link = state_dir.parent / "unused-cli-link"
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
