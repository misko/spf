from pathlib import Path

import pytest

from spf.mavlink.mavlink_controller import (
    VehicleParameterVerificationError,
    prepare_vehicle_parameters,
)


EXTERNAL_COMPASS_DEVICE_ID = 658953


def _healthy_parameters():
    params = {
        "COMPASS_ENABLE": 1,
        "COMPASS_CAL_FIT": 16,
        "COMPASS_DISBLMSK": 0,
        "COMPASS_PRIO1_ID": EXTERNAL_COMPASS_DEVICE_ID,
        "COMPASS_PRIO2_ID": 131594,
        "COMPASS_PRIO3_ID": 0,
    }
    for slot in (1, 2, 3):
        suffix = "" if slot == 1 else str(slot)
        external_key = "COMPASS_EXTERNAL" if slot == 1 else f"COMPASS_EXTERN{slot}"
        params.update(
            {
                f"COMPASS_DEV_ID{suffix}": 0,
                f"COMPASS_USE{suffix}": 0,
                external_key: 0,
                f"COMPASS_OFS{suffix}_X": 0,
                f"COMPASS_OFS{suffix}_Y": 0,
                f"COMPASS_OFS{suffix}_Z": 0,
            }
        )
    params.update(
        {
            "COMPASS_DEV_ID": EXTERNAL_COMPASS_DEVICE_ID,
            "COMPASS_USE": 1,
            "COMPASS_EXTERNAL": 1,
            "COMPASS_OFS_X": -19,
            "COMPASS_OFS_Y": 156,
            "COMPASS_OFS_Z": -82,
            "COMPASS_DEV_ID2": 131594,
        }
    )
    return params


class FakeParameters(dict):
    def __init__(self, values, *, differences=0, load_error=None):
        super().__init__(values)
        self.differences = differences
        self.load_error = load_error
        self.loaded_path = None
        self.diffed_path = None

    def load(self, path, *, mav):
        self.loaded_path = Path(path)
        if self.load_error is not None:
            raise self.load_error
        self["MANAGED_VALUE"] = 2
        return 1, 1

    def diff(self, path):
        self.diffed_path = Path(path)
        return self.differences


class FakeDrone:
    def __init__(self, params):
        self.params = params
        self.connection = object()
        self.downloads = 0
        self.events = []

    def update_all_parameters(self):
        self.downloads += 1
        self.events.append("download")
        return True

    def single_operation_mode_on(self):
        self.events.append("pause")

    def disarm(self):
        self.events.append("disarm")

    def single_operation_mode_off(self):
        self.events.append("resume")


def test_prepare_vehicle_uses_one_download_and_acknowledged_values(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    parameters = FakeParameters(_healthy_parameters())
    drone = FakeDrone(parameters)

    report = prepare_vehicle_parameters(drone, managed_path)

    assert report.ok, report.errors
    assert drone.downloads == 1
    assert drone.events == ["download", "pause", "disarm", "resume"]
    assert parameters.loaded_path == managed_path
    assert parameters.diffed_path == managed_path
    assert parameters["MANAGED_VALUE"] == 2


def test_prepare_vehicle_fails_closed_on_unverified_parameter(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    drone = FakeDrone(
        FakeParameters(_healthy_parameters(), differences=1)
    )

    with pytest.raises(VehicleParameterVerificationError, match="1 differences"):
        prepare_vehicle_parameters(drone, managed_path)


def test_prepare_vehicle_restores_message_processing_after_write_error(tmp_path):
    managed_path = tmp_path / "managed.params"
    managed_path.write_text("MANAGED_VALUE 2\n")
    drone = FakeDrone(
        FakeParameters(_healthy_parameters(), load_error=RuntimeError("write failed"))
    )

    with pytest.raises(RuntimeError, match="write failed"):
        prepare_vehicle_parameters(drone, managed_path)

    assert drone.events == ["download", "pause", "disarm", "resume"]
