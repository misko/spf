from pathlib import Path
import os
import subprocess
import sys

import pytest
import yaml

from spf.scripts.rover_capture_profile import resolve_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
ROVER_ROOT = REPO_ROOT / "data_collection/rover/rover_v3.1"


@pytest.mark.parametrize(
    ("rover_id", "records", "radios", "routine"),
    [
        (1, 3000, 2, "bounce"),
        (2, 3500, 1, "circle"),
        (3, 3000, 2, "bounce"),
    ],
)
def test_legacy_profile_preserves_original_boot_capture(
    rover_id, records, radios, routine
):
    profile = resolve_profile("legacy_iio_v4", rover_id)
    assert profile.records_per_receiver == records
    assert profile.expected_radios == radios
    assert profile.routine == routine
    assert profile.rx_transport == "iio"
    assert profile.data_version == 4


@pytest.mark.parametrize(
    ("name", "data_version"),
    [("direct_usb_v4", 4), ("direct_usb_v7", 7)],
)
def test_direct_profiles_change_schema_without_changing_capture_shape(
    name, data_version
):
    profile = resolve_profile(name, 1)
    assert profile.records_per_receiver == 3000
    assert profile.expected_radios == 2
    assert profile.routine == "bounce"
    assert profile.rx_transport == "direct_usb"
    assert profile.data_version == data_version
    config = yaml.safe_load(Path(profile.config_path).read_text())
    assert {receiver["buffer-size"] for receiver in config["receivers"]} == {524288}
    assert {receiver["f-sampling"] for receiver in config["receivers"]} == {30.0e6}
    assert {receiver["bandwidth"] for receiver in config["receivers"]} == {3.0e6}
    assert {
        receiver["direct-usb"]["protocol-version"] for receiver in config["receivers"]
    } == {2}


def test_direct_profile_fails_closed_on_unqualified_rover():
    with pytest.raises(ValueError, match="qualified only for Rover 1"):
        resolve_profile("direct_usb_v4", 3)


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="unknown capture profile"):
        resolve_profile("surprise", 1)


def test_boot_launcher_prints_resolved_direct_v4_plan_without_hardware(tmp_path):
    rover_id_file = tmp_path / "rover_id"
    rover_id_file.write_text("1\n")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "SPF_PYTHON": sys.executable,
            "SPF_ROVER_ID_FILE": str(rover_id_file),
            "SPF_CAPTURE_PROFILE": "direct_usb_v4",
            "SPF_SKIP_SELF_UPDATE": "1",
        }
    )
    result = subprocess.run(
        [str(ROVER_ROOT / "drone_run.sh"), "--print-plan"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert plan["capture_profile"] == "direct_usb_v4"
    assert plan["routine"] == "bounce"
    assert plan["records_per_receiver"] == "3000"
    assert plan["expected_radios"] == "2"
    assert plan["rx_transport"] == "direct_usb"
    assert plan["data_version"] == "4"


def test_direct_production_unit_waits_for_firmware_loader():
    dropin = (ROVER_ROOT / "mavlink_controller.direct_usb.conf").read_text()
    assert "Requires=spf-pluto-direct-usb.service" in dropin
    assert "After=spf-pluto-direct-usb.service" in dropin
    production_unit = (ROVER_ROOT / "mavlink_controller.service").read_text()
    assert "EnvironmentFile=-/etc/spf/rover_collection.env" in production_unit
