from pathlib import Path
import os
import subprocess
import sys

import pytest
import yaml

from spf.scripts.rover_capture_profile import resolve_profile
from spf.scripts.rover_capture_config import resolve_capture_plan


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


def test_boot_launcher_prints_canonical_v7_plan_without_hardware(tmp_path):
    rover_id_file = tmp_path / "rover_id"
    rover_id_file.write_text("1\n")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "SPF_PYTHON": sys.executable,
            "SPF_ROVER_ID_FILE": str(rover_id_file),
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
    assert plan["config"].endswith("rover1_production_v7.yaml")
    assert plan["routine"] == "bounce"
    assert plan["records_per_receiver"] == "3000"
    assert plan["expected_radios"] == "2"
    assert plan["rx_transport"] == "direct_usb"
    assert plan["data_version"] == "7"
    assert plan["firmware_release_tag"] == "v0.38-plutoplus-spf-gain-rssi-v2"


def test_every_production_boot_waits_for_firmware_loader_by_default():
    production_unit = (ROVER_ROOT / "mavlink_controller.service").read_text()
    assert "Requires=spf-pluto-direct-usb.service" in production_unit
    assert "After=network-online.target spf-pluto-direct-usb.service" in production_unit
    assert "EnvironmentFile=-/etc/spf/rover_collection.env" in production_unit

    loader_unit = (ROVER_ROOT / "spf-pluto-direct-usb.service").read_text()
    assert "ConditionPathExists=" not in loader_unit
    assert "EnvironmentFile=-/etc/spf/direct_usb_boot.env" in loader_unit
    assert "EnvironmentFile=-/etc/spf/rover_collection.env" in loader_unit
    assert "ExecStart=" in loader_unit


def test_fresh_setup_installs_and_enables_loader_with_mavlink():
    setup = (ROVER_ROOT / "setup.sh").read_text()
    assert "configure_direct_usb_boot.sh" in setup
    assert "production-default" in setup
    configure = (ROVER_ROOT / "configure_direct_usb_boot.sh").read_text()
    assert "spf-pluto-direct-usb.service" in configure
    assert "cache_firmware_image" in configure
    assert 'systemctl enable "$LOADER_UNIT" "$PRODUCTION_UNIT"' in configure
    launcher = (ROVER_ROOT / "drone_run.sh").read_text()
    assert "reconcile_rover_boot_units.sh" in launcher
    main_body = launcher.rsplit("main() {", 1)[1]
    assert main_body.index("reconcile_boot_units_or_reboot") < main_body.index(
        "maybe_self_update"
    )
    reconciler = (ROVER_ROOT / "reconcile_rover_boot_units.sh").read_text()
    assert "spf-pluto-direct-usb.service" in reconciler
    assert "mavlink_controller.service" in reconciler
    enabled_block = reconciler.split("readonly -a ENABLED_UNITS=(", 1)[1].split(")", 1)[
        0
    ]
    assert "spf-pluto-direct-usb.service" in enabled_block
    assert "mavlink_controller.service" in enabled_block


def test_stock_firmware_restore_is_an_explicit_opt_out():
    configure = (ROVER_ROOT / "configure_direct_usb_boot.sh").read_text()
    assert "production-default" in configure
    assert "cache_firmware_image" in configure
    assert "set_ram_load_disabled 0" in configure
    assert "set_ram_load_disabled 1" in configure
    environment = (ROVER_ROOT / "direct_usb_boot.env.example").read_text()
    assert "SPF_DIRECT_USB_DISABLE=0" in environment
    collection = (ROVER_ROOT / "rover_collection.env.example").read_text()
    assert "SPF_CAPTURE_PROFILE=" not in collection
    assert "SPF_CAPTURE_CONFIG=" in collection


@pytest.mark.parametrize(
    ("rover_id", "records", "radios", "routine", "spacing"),
    [
        (1, 3000, 2, "bounce", 0.035),
        (2, 3500, 1, "circle", 0.05075),
        (3, 3000, 2, "bounce", 0.043),
    ],
)
def test_canonical_v7_configs_are_self_contained(
    rover_id, records, radios, routine, spacing
):
    plan = resolve_capture_plan(rover_id)
    assert plan.data_version == 7
    assert plan.rx_transport == "direct_usb"
    assert plan.records_per_receiver == records
    assert plan.expected_radios == radios
    assert plan.routine == routine
    config = yaml.safe_load(Path(plan.config_path).read_text())
    assert "rx-transport" not in config["receivers"][0]
    assert "direct-usb" not in config["receivers"][0]
    assert {receiver["antenna-spacing-m"] for receiver in config["receivers"]} == {
        spacing
    }
    assert config["pluto-firmware"]["image-sha256"] == plan.firmware_image_sha256
