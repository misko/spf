import os
from pathlib import Path
import subprocess
import time

from spf.mavlink.mavlink_controller import tones
from spf.scripts.rover_capture_config import resolve_capture_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"
HANDLER = ROVER_ROOT / "radio_missing_shutdown.sh"


def test_missing_radio_alarm_repeats_for_configured_grace_without_poweroff(tmp_path):
    call_log = tmp_path / "controller.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$SPF_TEST_CONTROLLER_LOG\"\n"
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "SPF_PYTHON": str(fake_python),
            "SPF_RADIO_MISSING_ACTION": "log-only",
            "SPF_RADIO_MISSING_GRACE_SECONDS": "1",
            "SPF_RADIO_MISSING_TONE_GAP_SECONDS": "1",
            "SPF_RADIO_MISSING_CANCEL_FILE": str(tmp_path / "cancel"),
            "SPF_TEST_CONTROLLER_LOG": str(call_log),
        }
    )

    result = subprocess.run(
        [str(HANDLER), "2", "1"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert len(calls) >= 1
    assert all(call.endswith("--buzzer radio-missing") for call in calls)
    assert "for 1 seconds before poweroff" in result.stderr
    assert "TEST MODE: system poweroff inhibited" in result.stdout


def test_operator_cancel_file_stops_alarm_and_prevents_poweroff(tmp_path):
    call_log = tmp_path / "controller.log"
    systemctl_log = tmp_path / "systemctl.log"
    cancel_file = tmp_path / "cancel"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$SPF_TEST_CONTROLLER_LOG\"\n"
        "sleep 0.1\n"
    )
    fake_python.chmod(0o755)
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$SPF_TEST_SYSTEMCTL_LOG\"\n"
    )
    fake_systemctl.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "SPF_PYTHON": str(fake_python),
            "SPF_RADIO_MISSING_ACTION": "poweroff",
            "SPF_RADIO_MISSING_GRACE_SECONDS": "45",
            "SPF_RADIO_MISSING_TONE_GAP_SECONDS": "1",
            "SPF_RADIO_MISSING_CANCEL_FILE": str(cancel_file),
            "SPF_TEST_CONTROLLER_LOG": str(call_log),
            "SPF_TEST_SYSTEMCTL_LOG": str(systemctl_log),
        }
    )

    process = subprocess.Popen(
        [str(HANDLER), "2", "1"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(50):
        if call_log.exists():
            break
        time.sleep(0.02)
    assert call_log.exists(), "alarm did not start"
    cancel_file.touch()
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert "Operator cancelled missing-radio poweroff" in stdout
    assert not systemctl_log.exists()
    assert not cancel_file.exists()


def test_uncancelled_grace_requests_clean_nonblocking_poweroff(tmp_path):
    call_log = tmp_path / "controller.log"
    systemctl_log = tmp_path / "systemctl.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$SPF_TEST_CONTROLLER_LOG\"\n"
    )
    fake_python.chmod(0o755)
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$SPF_TEST_SYSTEMCTL_LOG\"\n"
    )
    fake_systemctl.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "SPF_PYTHON": str(fake_python),
            "SPF_RADIO_MISSING_ACTION": "poweroff",
            "SPF_RADIO_MISSING_GRACE_SECONDS": "1",
            "SPF_RADIO_MISSING_TONE_GAP_SECONDS": "1",
            "SPF_RADIO_MISSING_CANCEL_FILE": str(tmp_path / "cancel"),
            "SPF_TEST_CONTROLLER_LOG": str(call_log),
            "SPF_TEST_SYSTEMCTL_LOG": str(systemctl_log),
        }
    )

    result = subprocess.run(
        [str(HANDLER), "2", "1"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert systemctl_log.read_text().splitlines() == ["--no-block poweroff"]
    assert "1-second missing-radio grace period completed" in result.stderr


def test_canonical_fleet_radio_counts_apply_two_radio_policy_only_where_needed():
    assert resolve_capture_plan(1).expected_radios == 2
    assert resolve_capture_plan(2).expected_radios == 1
    assert resolve_capture_plan(3).expected_radios == 2


def test_boot_preparation_only_shuts_down_when_radio_count_is_low():
    source = (ROVER_ROOT / "prepare_direct_usb_boot.sh").read_text()

    assert "radio_missing_shutdown.sh" in source
    assert '[[ "$attached_radios" -lt "$configured_radios" ]]' in source
    assert '[[ "$attached_radios" -gt "$configured_radios" ]]' in source
    low_count_branch = source.split(
        'if [[ "$attached_radios" -lt "$configured_radios" ]]', 1
    )[1].split("elif", 1)[0]
    high_count_branch = source.split(
        'elif [[ "$attached_radios" -gt "$configured_radios" ]]', 1
    )[1].split("fi", 1)[0]
    assert '"$RADIO_MISSING_HANDLER"' in low_count_branch
    assert '"$RADIO_MISSING_HANDLER"' not in high_count_branch


def test_radio_missing_tone_is_named_and_distinct():
    assert "radio-missing" in tones
    assert tones["radio-missing"] not in {tones["failure"], tones["gps-time"]}
