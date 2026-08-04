"""Guards for the `rover` CLI dispatcher.

No hardware and no rover: every test here runs against the checkout, the same
way tests/test_provision_rover.py exercises provision_rover.sh. The point is to
catch the failure modes that actually bit us -- a stale or unreachable command,
a subcommand that is documented but broken, and a script that quietly never
gets a front door.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_DIR = REPO_ROOT / "data_collection/rover/rover_v3.1"
CLI = ROVER_DIR / "rover"


def run_cli(*args: str, env: dict | None = None, cwd: Path | None = None):
    environment = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", str(CLI), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        cwd=str(cwd) if cwd else None,
    )


def subcommands() -> list[str]:
    result = run_cli("--subcommands")
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


# --------------------------------------------------------------- dispatch ---


def test_cli_is_executable():
    assert CLI.is_file(), f"missing CLI: {CLI}"
    assert os.access(CLI, os.X_OK), "rover CLI must be executable (chmod +x)"


def test_no_arguments_prints_usage_and_succeeds():
    result = run_cli()
    assert result.returncode == 0
    assert "usage: rover <command>" in result.stdout


def test_unknown_command_fails_with_usage():
    result = run_cli("definitely-not-a-command")
    assert result.returncode == 2, "an unknown command must not exit 0"
    assert "unknown command" in result.stderr


@pytest.mark.parametrize("command", subcommands())
def test_every_subcommand_answers_help(command: str):
    """A documented subcommand that cannot answer --help is a broken one."""
    result = run_cli(command, "--help")
    assert result.returncode == 0, f"'rover {command} --help' failed: {result.stderr}"
    assert result.stdout.strip(), f"'rover {command} --help' printed nothing"


def test_usage_lists_every_subcommand():
    listed = run_cli("help").stdout
    for command in subcommands():
        if command == "help":
            continue
        assert command in listed, f"'{command}' dispatches but is undocumented in usage"


# --------------------------------------------------------------- identity ---


def test_rover_commands_refuse_to_run_off_a_rover():
    """Guessing an id is worse than refusing; a wrong id targets a live rover."""
    result = run_cli("audit")
    assert result.returncode != 0
    assert "not a rover" in result.stderr


# ------------------------------------------------------- symlink resolution ---


def test_cli_resolves_repo_through_a_symlink(tmp_path):
    """Installed as /usr/local/bin/rover, the CLI must still find its checkout.

    It resolves BASH_SOURCE with readlink -f rather than hardcoding /home/pi/spf,
    so this is the property that makes `rover install` a symlink instead of a
    copy -- and a copy is what goes stale.
    """
    link = tmp_path / "rover"
    link.symlink_to(CLI)
    result = subprocess.run(
        ["bash", str(link), "version"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert str(REPO_ROOT) in result.stdout, "symlinked CLI did not resolve its repo"


# ------------------------------------------------- update-blocking changes ---


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("original\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def test_untracked_files_are_not_reported_as_update_blocking(tmp_path):
    """Every rover carries untracked operational files and always will.

    pluto_env, this_rover.params, device_mapping and friends are untracked on
    rovers 1 and 2 right now. update_spf_before_boot.sh gates on `git diff
    --quiet` and `git diff --cached --quiet`, neither of which looks at
    untracked files, so reporting them as blocking would put a permanent false
    warning on every rover in the fleet.
    """
    repo = make_repo(tmp_path)
    (repo / "pluto_env").write_text("untracked\n")
    (repo / "this_rover.params").write_text("untracked\n")

    result = run_cli("version", env={"SPF_REPO_ROOT": str(repo)})
    assert result.returncode == 0, result.stderr
    assert "no update-blocking changes" in result.stdout
    assert "will refuse" not in result.stdout


def test_modified_tracked_file_is_reported_as_update_blocking(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("locally edited\n")

    result = run_cli("version", env={"SPF_REPO_ROOT": str(repo)})
    assert "tracked files modified" in result.stdout
    assert "will refuse" in result.stdout


def test_staged_change_is_reported_as_update_blocking(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "new.txt").write_text("staged\n")
    subprocess.run(["git", "add", "new.txt"], cwd=repo, check=True)

    result = run_cli("version", env={"SPF_REPO_ROOT": str(repo)})
    assert "staged changes present" in result.stdout


def test_unreachable_remote_is_a_warning_not_a_failure(tmp_path):
    """A rover in the field may have no route to GitHub; the CLI must still run."""
    repo = make_repo(tmp_path)
    result = run_cli("version", env={"SPF_REPO_ROOT": str(repo)})
    assert result.returncode == 0, "no network must not make the CLI fail"
    assert "staleness unknown" in result.stdout


# ---------------------------------------------------------------- install ---


def test_install_refuses_without_root():
    result = run_cli("install")
    assert result.returncode != 0
    assert "must run as root" in result.stderr


def test_install_refuses_to_clobber_a_real_file(tmp_path):
    """Never silently replace a real /usr/local/bin/rover we did not create."""
    occupied = tmp_path / "rover"
    occupied.write_text("#!/bin/sh\necho not ours\n")
    result = run_cli(
        "install", env={"SPF_ROVER_CLI_PATH": str(occupied)}
    )
    assert result.returncode != 0
    # Fails as non-root first; the clobber guard is asserted by inspection of
    # the file surviving unchanged.
    assert occupied.read_text() == "#!/bin/sh\necho not ours\n"


# --------------------------------------------------------- coverage guard ---

# Every script in rover_v3.1 gets a disposition. Adding a script without
# classifying it fails this test, which is the point: the CLI cannot silently
# fall behind the directory it fronts. Same discipline as FIRMWARE_KEYS ->
# FIRMWARE_KEY_TO_PLAN_ATTR in spf/scripts/rover_capture_config.py.
SCRIPT_DISPOSITION = {
    # reachable from the CLI today
    "rover": "cli",
    "audit_rover.sh": "exposed",
    # fronted by `rover ardupilot ...` and `rover radio ...`
    "check_ardupilot_prearm.sh": "exposed",
    "check_compass_policy.sh": "exposed",
    "flash_ardupilot.sh": "exposed",
    "run_motor_test.py": "exposed",
    "mavlink_set_guided_mode.py": "utility",
    "check_pluto_firmware.sh": "exposed",
    "check_and_set_pluto.sh": "internal",
    "ensure_pluto_qspi.sh": "internal",
    "prepare_direct_usb_boot.sh": "internal",
    "load_direct_usb_firmware.sh": "internal",
    "configure_direct_usb_boot.sh": "internal",
    "run_direct_usb_boot_preflight.sh": "exposed",
    "drone_run.sh": "exposed",
    # provisioning: deliberately NOT behind the CLI. These run once, as root,
    # on a machine that may not have a working CLI yet.
    "provision_rover.sh": "provisioning",
    "configure_rover_network.sh": "provisioning",
    "install_deps.sh": "provisioning",
    "compare_rovers.sh": "provisioning",
    "setup.sh": "deprecated",
    # invoked by systemd units or by other scripts, never by an operator
    "update_spf_before_boot.sh": "internal",
    "radio_missing_shutdown.sh": "internal",
    "reconcile_rover_boot_units.sh": "internal",
    "device_mapping.sh": "internal",
    # long-running soaks, driven by hand with their own arguments
    "run_direct_usb_restart_soak.sh": "soak",
    "run_interrupted_capture_campaign.sh": "soak",
    "run_interrupted_capture_soak.sh": "soak",
    # generators and one-off utilities
    "make_schematic.py": "utility",
    "make_taranis_map.py": "utility",
    "make_tones.py": "utility",
    "make_pluto_frm.sh": "utility",
    "telem.sh": "utility",
    "debug_drone_run.sh": "utility",
}


GROUPS = ("ardupilot", "radio", "sitl")


@pytest.mark.parametrize("group", GROUPS)
def test_group_help_lists_subcommands(group: str):
    result = run_cli(group, "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: rover {group}" in result.stdout


@pytest.mark.parametrize("group", GROUPS)
def test_group_rejects_unknown_subcommand(group: str):
    result = run_cli(group, "not-a-subcommand")
    assert result.returncode != 0
    assert "unknown" in result.stderr


def test_ardupilot_alias_ap_works():
    result = run_cli("ap", "--help")
    assert result.returncode == 0
    assert "usage: rover ardupilot" in result.stdout


def test_exposed_scripts_are_actually_reachable_from_the_cli():
    """A disposition of 'exposed' must mean the CLI really dispatches to it.

    Without this the table is a comment: someone could mark a script exposed,
    never wire it, and the guard would still pass.
    """
    cli_source = CLI.read_text()
    exposed = [n for n, d in SCRIPT_DISPOSITION.items() if d == "exposed"]
    assert exposed, "expected some scripts to be exposed"
    unreachable = [name for name in exposed if name not in cli_source]
    assert not unreachable, f"marked 'exposed' but never dispatched to: {unreachable}"


def test_every_script_has_a_cli_disposition():
    on_disk = {
        path.name
        for path in ROVER_DIR.iterdir()
        if path.is_file() and (path.suffix in {".sh", ".py"} or path.name == "rover")
    }
    unclassified = on_disk - SCRIPT_DISPOSITION.keys()
    assert not unclassified, (
        f"new script(s) with no CLI disposition: {sorted(unclassified)}. "
        f"Add each to SCRIPT_DISPOSITION -- expose it via the CLI, or mark it "
        f"internal/utility/provisioning with a reason."
    )


def test_disposition_table_has_no_stale_entries():
    on_disk = {path.name for path in ROVER_DIR.iterdir() if path.is_file()}
    missing = SCRIPT_DISPOSITION.keys() - on_disk
    assert not missing, f"SCRIPT_DISPOSITION names deleted script(s): {sorted(missing)}"
