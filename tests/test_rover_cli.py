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


# ----------------------------------------------------------------- config ---

# ~/.rover_config is optional and normally absent -- on a rover, and on the
# base station, every setting falls back to a built-in default. It exists so
# tests (and an unusual machine) can inject known values.


def test_absent_config_falls_back_to_built_in_defaults(tmp_path):
    """The normal case: no file anywhere, dev defaults apply."""
    missing = tmp_path / "nope" / ".rover_config"
    result = run_cli("config", env={"SPF_ROVER_CONFIG": str(missing)})
    assert result.returncode == 0, result.stderr
    assert "absent" in result.stdout
    assert "192.168.1.141" in result.stdout      # default SITL host
    assert "14590 14591 14592" in result.stdout  # default SITL ports
    assert result.stdout.count("default") >= 8


def test_config_file_supplies_values(tmp_path):
    config = tmp_path / ".rover_config"
    config.write_text(
        "# a comment\n"
        "SPF_SITL_HOST=10.0.0.9\n"
        "SPF_SITL_PORTS=25590 25591\n"
        "\n"
    )
    result = run_cli("config", env={"SPF_ROVER_CONFIG": str(config)})
    assert result.returncode == 0, result.stderr
    assert "10.0.0.9" in result.stdout
    assert "25590 25591" in result.stdout
    assert str(config) in result.stdout  # provenance points at the file


def test_environment_beats_the_config_file(tmp_path):
    """Precedence: real env > file > default, so an explicit override wins."""
    config = tmp_path / ".rover_config"
    config.write_text("SPF_SITL_HOST=10.0.0.9\n")
    result = run_cli(
        "config",
        env={"SPF_ROVER_CONFIG": str(config), "SPF_SITL_HOST": "172.16.0.1"},
    )
    assert "172.16.0.1" in result.stdout
    assert "10.0.0.9" not in result.stdout
    assert "environment" in result.stdout


def test_config_file_is_parsed_not_executed(tmp_path):
    """A sourced config would run as root under `sudo rover install`."""
    canary = tmp_path / "canary"
    config = tmp_path / ".rover_config"
    config.write_text(
        f"SPF_SITL_HOST=$(touch {canary})\n"
        f"SPF_SITL_IMAGE=`touch {canary}`\n"
    )
    result = run_cli("config", env={"SPF_ROVER_CONFIG": str(config)})
    assert result.returncode == 0, result.stderr
    assert not canary.exists(), "config file was EXECUTED, not parsed"


def test_config_ignores_keys_outside_the_spf_namespace(tmp_path):
    """A stray line must not be able to clobber PATH, HOME, or an internal."""
    config = tmp_path / ".rover_config"
    config.write_text("PATH=/nowhere\nHOME=/nowhere\nSPF_SITL_HOST=10.0.0.9\n")
    result = run_cli("config", env={"SPF_ROVER_CONFIG": str(config)})
    assert result.returncode == 0, result.stderr
    assert "10.0.0.9" in result.stdout  # the SPF_ key still applied


# -------------------------------------------------------- capture env ---
#
# `rover env` writes /etc/spf/rover_collection.env -- the file systemd hands
# every unit via EnvironmentFile= and the only one a running capture reads.
# These tests point SPF_PROFILE_ENV at a tempdir, the same seam
# SPF_ROVER_CONFIG provides for ~/.rover_config.


def env_cli(*args, profile, config=None, **kwargs):
    environment = {"SPF_PROFILE_ENV": str(profile)}
    if config is not None:
        environment["SPF_ROVER_CONFIG"] = str(config)
    environment.update(kwargs.pop("env", {}))
    return run_cli("env", *args, env=environment, **kwargs)


@pytest.mark.parametrize("setting", ["crash_detect", "crash_recovery", "ultrasonic"])
def test_env_reads_the_built_in_default_when_the_profile_is_absent(
    tmp_path, setting: str
):
    result = env_cli(setting, profile=tmp_path / "absent.env")
    assert result.returncode == 0, result.stderr
    assert "unset" in result.stdout
    assert "default" in result.stdout


def test_crash_detect_defaults_on_and_recovery_defaults_off(tmp_path):
    """Detection is safe everywhere; autonomous reversing is opt-in per rover."""
    profile = tmp_path / "absent.env"
    detect = env_cli("crash_detect", profile=profile).stdout
    recovery = env_cli("crash_recovery", profile=profile).stdout
    assert "effective: enabled" in detect
    assert "effective: disabled" in recovery
    assert "rover 4" in recovery, "the read must explain WHY it is off here"


@pytest.mark.parametrize("setting", ["crash_detect", "crash_recovery", "ultrasonic"])
def test_env_writes_and_reads_back(tmp_path, setting: str):
    profile = tmp_path / "rover_collection.env"
    assert env_cli(setting, "enable", profile=profile).returncode == 0
    assert "effective: enabled" in env_cli(setting, profile=profile).stdout
    assert env_cli(setting, "disable", profile=profile).returncode == 0
    assert "effective: disabled" in env_cli(setting, profile=profile).stdout


def test_env_write_is_idempotent_and_preserves_other_settings(tmp_path):
    """A capture profile carries unrelated knobs; a write must not disturb them."""
    profile = tmp_path / "rover_collection.env"
    profile.write_text("SPF_RADIO_WAIT_SECONDS=600\nSPF_OUTPUT_ROOT=/home/pi/temp\n")

    for _ in range(3):
        env_cli("crash_recovery", "enable", profile=profile)
    env_cli("crash_detect", "disable", profile=profile)

    lines = [line for line in profile.read_text().splitlines() if line.strip()]
    assert lines.count("SPF_CRASH_RECOVERY=1") == 1, lines
    assert "SPF_CRASH_DETECT=0" in lines
    assert "SPF_RADIO_WAIT_SECONDS=600" in lines
    assert "SPF_OUTPUT_ROOT=/home/pi/temp" in lines


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses file permissions, so nothing is refused"
)
def test_env_refuses_to_write_an_unwritable_profile(tmp_path):
    """On a rover /etc/spf is root-owned, so this is the sudo path."""
    locked = tmp_path / "locked"
    locked.mkdir()
    profile = locked / "rover_collection.env"
    profile.write_text("SPF_RADIO_WAIT_SECONDS=600\n")
    profile.chmod(0o444)
    locked.chmod(0o555)
    try:
        result = env_cli("crash_recovery", "enable", profile=profile)
        assert result.returncode != 0
        assert "sudo rover env crash_recovery enable" in result.stderr
        assert profile.read_text() == "SPF_RADIO_WAIT_SECONDS=600\n"
    finally:
        locked.chmod(0o755)
        profile.chmod(0o644)


def test_env_rejects_an_unknown_action(tmp_path):
    result = env_cli("crash_detect", "toggle", profile=tmp_path / "p.env")
    assert result.returncode != 0
    assert "unknown action" in result.stderr


def test_capture_settings_in_rover_config_are_flagged_as_ineffective(tmp_path):
    """The footgun the two-file split creates, and the guard that closes it.

    ~/.rover_config is read by this CLI only. A capture setting placed there
    would be silently ignored by the rover, so saying nothing would make
    `rover config` answer "which value am I getting" wrongly.
    """
    config = tmp_path / ".rover_config"
    config.write_text("SPF_CRASH_RECOVERY=1\n")
    profile = tmp_path / "rover_collection.env"

    result = env_cli("crash_recovery", profile=profile, config=config)
    assert result.returncode == 0, result.stderr
    assert "which capture never reads" in result.stdout
    assert "effective: disabled" in result.stdout, "the file must NOT take effect"
    assert "sudo rover env crash_recovery enable|disable" in result.stdout


def test_config_shows_capture_settings_in_their_own_table(tmp_path):
    profile = tmp_path / "rover_collection.env"
    env_cli("crash_recovery", "enable", profile=profile)

    result = run_cli("config", env={"SPF_PROFILE_ENV": str(profile)})
    assert result.returncode == 0, result.stderr
    assert "capture services" in result.stdout
    assert "SPF_CRASH_RECOVERY" in result.stdout
    assert str(profile) in result.stdout


def test_capture_env_defaults_have_exactly_one_definition():
    """The rover-4 default must not be recomputed in two scripts.

    Independently derived per-rover defaults are what produced the ARMING_CHECK
    drift fixed in b4fa14a, so both consumers source rover_env_defaults.sh.
    """
    defaults = ROVER_DIR / "rover_env_defaults.sh"
    assert defaults.is_file()
    for consumer in (CLI, ROVER_DIR / "drone_run.sh"):
        assert "rover_env_defaults.sh" in consumer.read_text(), consumer

    others = [
        path
        for path in ROVER_DIR.iterdir()
        if path.is_file()
        and path != defaults
        and "spf_default_crash_recovery()" in path.read_text(errors="ignore")
    ]
    assert not others, f"crash_recovery default redefined in {others}"


# ------------------------------------------------------------ sitl status ---


def test_sitl_status_reports_free_ports(tmp_path):
    """Point the check at a port nothing owns."""
    config = tmp_path / ".rover_config"
    config.write_text("SPF_SITL_PORTS=59999\n")
    result = run_cli("sitl", "status", env={"SPF_ROVER_CONFIG": str(config)})
    assert result.returncode == 0, result.stderr
    assert "are free" in result.stdout


def test_sitl_status_detects_a_bound_port(tmp_path):
    """The real behaviour: bind a port, then assert the CLI notices.

    This is why SPF_SITL_PORTS is injectable -- the check can be exercised
    against a port the test controls instead of requiring a live simulator on
    14590. Uses an ephemeral port so concurrent runs cannot collide, which is
    the same lesson as the SITL fixture itself.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        bound = sock.getsockname()[1]

        config = tmp_path / ".rover_config"
        config.write_text(f"SPF_SITL_PORTS={bound}\n")
        result = run_cli("sitl", "status", env={"SPF_ROVER_CONFIG": str(config)})

    assert result.returncode == 0, result.stderr
    assert "already bound" in result.stdout
    assert str(bound) in result.stdout


# ------------------------------------------------------------------ stage ---
#
# `rover stage` fronts stage_captures.sh, which copies a capture WITH its .yaml
# and .log sidecars. Its own behaviour is covered by tests/test_stage_captures.py;
# these pin the front door and the base-station-only rule.


def test_stage_dispatches_to_stage_captures(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "cap_tag_RO1.zarr").mkdir()
    (source / "cap_tag_RO1.zarr" / "data.mdb").write_bytes(b"x")
    (source / "cap_tag_RO1.yaml").write_text("receivers: []\n")
    (source / "cap_tag_RO1.log").write_text("log\n")

    result = run_cli("stage", "--from", str(source), "--to", str(tmp_path / "dst"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "dst" / "cap_tag_RO1.yaml").exists()


def test_stage_refuses_to_run_on_a_rover(tmp_path):
    """There is no rsync or scp on a rover; offload is a pull from the base."""
    environment = rover_hostname_shim(tmp_path)
    result = run_cli("stage", "--from", "/x", "--to", "/y", env=environment)
    assert result.returncode != 0
    assert "base station" in result.stderr


def test_stage_help_works_on_a_rover_too(tmp_path):
    """Refusing to act must not refuse to explain -- and --help is asserted
    for every subcommand, on whatever machine the suite runs."""
    result = run_cli("stage", "--help", env=rover_hostname_shim(tmp_path))
    assert result.returncode == 0, result.stderr
    assert ".yaml" in result.stdout


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
    # sourced by both the CLI and drone_run.sh; never invoked directly, which
    # is the point -- it is the one place a capture-env default is defined.
    "rover_env_defaults.sh": "internal",
    "check_and_set_pluto.sh": "internal",
    "ensure_pluto_qspi.sh": "internal",
    "prepare_direct_usb_boot.sh": "internal",
    "load_direct_usb_firmware.sh": "internal",
    "configure_direct_usb_boot.sh": "internal",
    "run_direct_usb_boot_preflight.sh": "exposed",
    "run_rx_signal_check.sh": "exposed",
    "update_device_mapping.sh": "exposed",
    "drone_run.sh": "exposed",
    # base-station side, like sitl: offload PULLS from a rover, because a rover
    # has no rsync/scp (ROVER_RUNBOOK 12.2). Behaviour: tests/test_stage_captures.py
    "stage_captures.sh": "exposed",
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


GROUPS = ("ardupilot", "radio", "sitl", "env")


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


# ------------------------------------------------------- CLI symlink health ---
#
# Rover 4 shipped with no /usr/local/bin/rover at all: `rover install` joined
# provision_rover.sh's base stage after that rover had already run it. The boot
# reconciler now heals this, but doctor still has to be able to say so -- an
# operator who cannot run `rover` by name needs to be told why, not left to
# guess at PATH.


def rover_hostname_shim(tmp_path: Path) -> dict:
    """Make the checkout look like a rover, so doctor runs its full report."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "hostname").write_text("#!/bin/sh\necho roverpi4\n")
    (shim / "hostname").chmod(0o755)
    return {"PATH": f"{shim}:{os.environ['PATH']}"}


def doctor_cli_section(tmp_path: Path, cli_path: Path) -> str:
    environment = rover_hostname_shim(tmp_path)
    environment["SPF_ROVER_CLI_PATH"] = str(cli_path)
    result = run_cli("doctor", env=environment)
    section = result.stdout.split("== cli ==")[-1].split("== services ==")[0]
    return section


def test_doctor_reports_a_missing_cli_symlink(tmp_path):
    section = doctor_cli_section(tmp_path, tmp_path / "absent" / "rover")
    assert "missing" in section
    assert "install" in section


def test_doctor_accepts_a_correct_cli_symlink(tmp_path):
    link = tmp_path / "rover"
    link.symlink_to(CLI)
    section = doctor_cli_section(tmp_path, link)
    assert "missing" not in section
    assert str(CLI.resolve()) in section


def test_doctor_flags_a_symlink_into_a_different_checkout(tmp_path):
    link = tmp_path / "rover"
    link.symlink_to("/bin/true")
    section = doctor_cli_section(tmp_path, link)
    assert "but this CLI is" in section


def test_doctor_does_not_fail_the_mission_over_a_missing_symlink(tmp_path):
    """doctor's non-zero exit is reserved for things that stop a mission."""
    environment = rover_hostname_shim(tmp_path)
    environment["SPF_ROVER_CLI_PATH"] = str(tmp_path / "absent" / "rover")
    result = run_cli("doctor", env=environment)
    assert "== cli ==" in result.stdout


# ------------------------------------------------- ardupilot pass-throughs ---
#
# Calibration was only reachable as `python -m spf.ardupilot.ardu_cli ...` while
# the wrapper exposed a subset, so the documented sequence kept dropping out of
# `rover` half way through. These pin that every advertised ardupilot
# subcommand actually dispatches, with its arguments intact.

ARDUPILOT_PASSTHROUGH = ("status", "prearm", "compass", "rc", "accelcal", "magcal",
                         "calibrate", "reboot")


def fake_venv(tmp_path: Path) -> dict:
    """A venv whose python3 echoes its arguments instead of running them.

    Two dispatch paths exist and both must be stubbed: the ardu_cli
    pass-throughs resolve the interpreter themselves via SPF_VENV, while
    `prearm` and `compass` go through shell wrappers that read SPF_PYTHON.
    """
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python3"
    python.write_text("#!/bin/sh\necho \"DISPATCH $*\"\n")
    python.chmod(0o755)
    return {"SPF_VENV": str(tmp_path / "venv"), "SPF_PYTHON": str(python)}


@pytest.mark.parametrize("subcommand", ARDUPILOT_PASSTHROUGH)
def test_ardupilot_subcommand_dispatches_to_ardu_cli(subcommand: str, tmp_path):
    result = run_cli("ardupilot", subcommand, env=fake_venv(tmp_path))
    assert "ardu_cli.py" in result.stdout, result.stdout + result.stderr
    assert f" {subcommand}" in result.stdout, result.stdout


def test_ardupilot_passthrough_preserves_arguments(tmp_path):
    """magcal start --mask 1 must arrive intact, not collapsed to `magcal`."""
    result = run_cli(
        "ardupilot", "magcal", "start", "--yes", "--mask", "1",
        env=fake_venv(tmp_path),
    )
    assert "magcal start --yes --mask 1" in result.stdout, result.stdout


@pytest.mark.parametrize("subcommand", ARDUPILOT_PASSTHROUGH)
def test_every_advertised_ardupilot_subcommand_is_documented(subcommand: str):
    """A subcommand that works but is undocumented is one nobody finds."""
    result = run_cli("ardupilot", "--help")
    assert f"  {subcommand}" in result.stdout, result.stdout


def test_unknown_ardupilot_subcommand_still_fails(tmp_path):
    result = run_cli("ardupilot", "definitely-not-a-subcommand", env=fake_venv(tmp_path))
    assert result.returncode != 0
    assert "unknown ardupilot subcommand" in result.stderr


# ------------------------------------------------------- journal-persistence ---
#
# Rover 4's journal was volatile, so the boot that recorded a capture with one
# receive channel 24 dB down was gone before anyone could read it. The boot
# reconciler now converges the setting; doctor has to be able to say which of
# the three states the rover is in, because "configured" and "in effect" are
# not the same thing -- journald only reads Storage= at start.


def doctor_journal_section(tmp_path: Path, dropin: Path, journal_dir: Path) -> str:
    environment = rover_hostname_shim(tmp_path)
    environment["SPF_ROVER_CLI_PATH"] = str(tmp_path / "absent" / "rover")
    environment["SPF_JOURNALD_DROPIN"] = str(dropin)
    environment["SPF_JOURNAL_DIR"] = str(journal_dir)
    result = run_cli("doctor", env=environment)
    assert "== journal ==" in result.stdout, result.stdout + result.stderr
    return result.stdout.split("== journal ==")[-1].split("== hardware ==")[0]


def desired_dropin() -> str:
    result = subprocess.run(
        ["bash", "-c", f"source '{ROVER_DIR}/rover_env_defaults.sh'; "
                       "spf_journald_dropin_content"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_doctor_reports_a_volatile_journal(tmp_path):
    section = doctor_journal_section(tmp_path, tmp_path / "absent.conf", tmp_path / "none")
    assert "VOLATILE" in section
    assert "--journald-only" in section


def test_doctor_separates_configured_from_in_effect(tmp_path):
    """The drop-in is written, but journald still holds the journal in RAM."""
    dropin = tmp_path / "10-spf-persistent.conf"
    dropin.write_text(desired_dropin())
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()

    section = doctor_journal_section(tmp_path, dropin, journal_dir)

    assert "still VOLATILE" in section
    assert "next reboot" in section


def test_doctor_accepts_a_persistent_journal(tmp_path):
    dropin = tmp_path / "10-spf-persistent.conf"
    dropin.write_text(desired_dropin())
    journal_dir = tmp_path / "journal" / "0123456789abcdef"
    journal_dir.mkdir(parents=True)
    (journal_dir / "system.journal").write_bytes(b"")

    section = doctor_journal_section(tmp_path, dropin, journal_dir.parent)

    assert "persistent" in section
    assert "VOLATILE" not in section


def test_doctor_does_not_fail_the_mission_over_a_volatile_journal(tmp_path):
    """A lost journal costs the post-mortem, not the mission."""
    environment = rover_hostname_shim(tmp_path)
    environment["SPF_JOURNALD_DROPIN"] = str(tmp_path / "absent.conf")
    environment["SPF_JOURNAL_DIR"] = str(tmp_path / "none")
    result = run_cli("doctor", env=environment)
    # The report must carry on past the journal section. (It stops in the
    # hardware section on a dev box, which has no flight controller -- that is
    # the pre-existing behaviour of every doctor test here.)
    assert "== hardware ==" in result.stdout, result.stdout


# ------------------------------------------------------- ultrasonic default ---
#
# The obstacle stop is OFF fleet-wide as of 2026-08-05, and the env setting is
# authoritative over the transmitter: with it off the collector runs
# --no-ultrasonic, never constructs the distance finder, and handle_RC_CHANNELS
# returns on `distance_finder is None` before reading CH12. That last part
# matters because an R9 SX in failsafe holds its last channel values, so a
# frozen-high CH12 must not be able to switch a disabled sensor back on.


def test_ultrasonic_is_off_by_default_fleet_wide(tmp_path):
    """A silent flip back to on would re-arm the obstacle stop everywhere."""
    for rover_id in ("1", "2", "3", "4"):
        result = subprocess.run(
            ["bash", "-c",
             f'source "{ROVER_DIR}/rover_env_defaults.sh"; '
             f'spf_capture_env_default SPF_ULTRASONIC {rover_id}'],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "0", f"rover {rover_id}"


def test_crash_detect_stays_on_as_the_remaining_backstop(tmp_path):
    """Turning off the obstacle stop leaves stall detection carrying safety."""
    result = subprocess.run(
        ["bash", "-c",
         f'source "{ROVER_DIR}/rover_env_defaults.sh"; '
         f'spf_capture_env_default SPF_CRASH_DETECT 1'],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "1"
