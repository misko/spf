import os
from pathlib import Path
import subprocess

from spf.mavlink.mavlink_controller import tones


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"
HANDLER = ROVER_ROOT / "radio_missing_shutdown.sh"


def test_missing_radio_alarm_plays_exactly_three_times_without_poweroff(tmp_path):
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
            "SPF_RADIO_MISSING_TONE_GAP_SECONDS": "0",
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
    assert len(calls) == 3
    assert all(call.endswith("--buzzer radio-missing") for call in calls)
    assert result.stdout.count("radio-missing alarm 1/3") == 1
    assert result.stdout.count("radio-missing alarm 2/3") == 1
    assert result.stdout.count("radio-missing alarm 3/3") == 1
    assert "TEST MODE: system poweroff inhibited" in result.stdout


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
