from contextlib import contextmanager
import logging
from pathlib import Path

import pytest

from spf.mavlink.mavlink_controller import (
    VehicleParameterVerificationError,
    check_compass_policy,
    log_compass_policy_report,
    prepare_vehicle_parameters,
)


EXTERNAL_COMPASS_DEVICE_ID = 658953


def healthy_parameters():
    return {
        "COMPASS_ENABLE": 1,
        "COMPASS_CAL_FIT": 16,
        "COMPASS_DISBLMSK": 0,
        "COMPASS_OFFS_MAX": 1800,
        "COMPASS_PRIO1_ID": EXTERNAL_COMPASS_DEVICE_ID,
        "COMPASS_PRIO2_ID": 658945,
        "COMPASS_PRIO3_ID": 0,
        "COMPASS_DEV_ID": EXTERNAL_COMPASS_DEVICE_ID,
        "COMPASS_EXTERNAL": 1,
        "COMPASS_USE": 1,
        "COMPASS_OFS_X": -19,
        "COMPASS_OFS_Y": 156,
        "COMPASS_OFS_Z": -82,
        "COMPASS_DEV_ID2": 658945,
        "COMPASS_EXTERN2": 0,
        "COMPASS_USE2": 0,
        "COMPASS_OFS2_X": 0,
        "COMPASS_OFS2_Y": 0,
        "COMPASS_OFS2_Z": 0,
        "COMPASS_DEV_ID3": 0,
        "COMPASS_EXTERN3": 0,
        "COMPASS_USE3": 0,
        "COMPASS_OFS3_X": 0,
        "COMPASS_OFS3_Y": 0,
        "COMPASS_OFS3_Z": 0,
        "MANAGED_VALUE": 2,
    }


class FakeParameters(dict):
    def __init__(self, values, *, managed_value=2, fail_verification=False):
        super().__init__(values)
        self["MANAGED_VALUE"] = managed_value
        self.fail_verification = fail_verification
        self.loaded_path = None

    def diff(self, path):
        expected = float(Path(path).read_text().split()[1])
        if self.fail_verification:
            return 1
        return int(self["MANAGED_VALUE"] != expected)

    def load(self, path, *, mav):
        self.loaded_path = Path(path)
        self["MANAGED_VALUE"] = float(Path(path).read_text().split()[1])
        return 1, 1


class FakeDrone:
    def __init__(self, params, *, armed=False):
        self.params = params
        self.armed = armed
        self.downloads = 0
        self.events = []
        self.connection = object()

    def update_all_parameters(self):
        self.downloads += 1
        self.events.append("download")
        return True

    def single_operation_mode_on(self):
        self.events.append("pause")

    def single_operation_mode_off(self):
        self.events.append("resume")

    @contextmanager
    def _command_connection(self):
        yield self.connection


def test_prepare_vehicle_clean_boot_uses_one_read_and_no_write(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    drone = FakeDrone(FakeParameters(healthy_parameters()))

    report = prepare_vehicle_parameters(drone, managed_path)

    assert report.ok, report.errors
    assert drone.downloads == 1
    assert drone.events == ["download"]
    assert drone.params.loaded_path is None


def test_prepare_vehicle_applies_and_verifies_changes_from_same_snapshot(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    params = FakeParameters(healthy_parameters(), managed_value=1)
    drone = FakeDrone(params)

    report = prepare_vehicle_parameters(drone, managed_path)

    assert report.ok, report.errors
    assert drone.downloads == 2
    assert drone.events == ["download", "pause", "resume", "download"]
    assert params.loaded_path == managed_path


def test_prepare_vehicle_refuses_writes_while_armed(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    drone = FakeDrone(FakeParameters(healthy_parameters(), managed_value=1), armed=True)

    with pytest.raises(VehicleParameterVerificationError, match="vehicle is armed"):
        prepare_vehicle_parameters(drone, managed_path)


def test_prepare_vehicle_fails_closed_on_unverified_write(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    drone = FakeDrone(
        FakeParameters(healthy_parameters(), managed_value=1, fail_verification=True)
    )

    with pytest.raises(VehicleParameterVerificationError, match="1 differences"):
        prepare_vehicle_parameters(drone, managed_path)


def test_read_only_check_and_log_include_every_slot_and_priority(caplog):
    caplog.set_level(logging.INFO)
    drone = FakeDrone(FakeParameters(healthy_parameters()))

    report = check_compass_policy(drone)
    log_compass_policy_report(report)

    assert drone.downloads == 1
    assert "Compass priorities: PRIO1=658953 PRIO2=658945 PRIO3=0" in caplog.text
    assert "Compass slot 1: device_id=658953" in caplog.text
    assert "Compass slot 2: device_id=658945" in caplog.text
    assert "Compass slot 3: device_id=0" in caplog.text
    assert "Compass policy PASS" in caplog.text
