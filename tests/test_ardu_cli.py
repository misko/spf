import json
from types import SimpleNamespace

import pytest
from pymavlink import mavutil

from spf.ardupilot import ardu_cli


class FakeMessage(SimpleNamespace):
    def get_type(self):
        return self.message_type

    def to_dict(self):
        return {
            key: value for key, value in vars(self).items() if key != "message_type"
        }


class FakeMav:
    def __init__(self):
        self.commands = []
        self.parameter_requests = []

    def command_long_send(self, *args):
        self.commands.append(args)

    def param_request_list_send(self, *args):
        self.parameter_requests.append(args)


class FakeConnection:
    target_system = 1
    target_component = 1

    def __init__(self, messages=()):
        self.messages = list(messages)
        self.mav = FakeMav()

    def recv_match(self, *, blocking, type=None, timeout=None):
        if not blocking:
            return None
        if not self.messages:
            return None
        if type is None:
            return self.messages.pop(0)
        allowed = {type} if isinstance(type, str) else set(type)
        for index, message in enumerate(self.messages):
            if message.get_type() in allowed:
                return self.messages.pop(index)
        return None


def heartbeat(*, armed=False):
    return FakeMessage(
        message_type="HEARTBEAT",
        base_mode=(mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED if armed else 0),
        system_status=mavutil.mavlink.MAV_STATE_STANDBY,
        type=mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
        autopilot=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        custom_mode=0,
    )


def healthy_compass_params():
    params = {
        "COMPASS_ENABLE": 1,
        "COMPASS_CAL_FIT": 16,
        "COMPASS_DISBLMSK": 0,
        "COMPASS_PRIO1_ID": 658953,
        "COMPASS_PRIO2_ID": 131594,
        "COMPASS_PRIO3_ID": 0,
        "COMPASS_DEV_ID": 658953,
        "COMPASS_EXTERNAL": 1,
        "COMPASS_USE": 1,
        "COMPASS_OFS_X": -20,
        "COMPASS_OFS_Y": 100,
        "COMPASS_OFS_Z": -30,
        "COMPASS_DEV_ID2": 131594,
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
    }
    return params


def test_direct_serial_refuses_active_production_service(monkeypatch):
    monkeypatch.setattr(ardu_cli, "_service_is_active", lambda: True)
    args = SimpleNamespace(
        master=None,
        allow_active_service=False,
        baud=115200,
        heartbeat_timeout=1,
    )

    with pytest.raises(ardu_cli.CliError, match="may own the ArduPilot serial link"):
        ardu_cli._connect(args)


def test_network_fanout_does_not_conflict_with_active_service(monkeypatch):
    connection = FakeConnection()
    expected_heartbeat = heartbeat()
    connection.wait_heartbeat = lambda timeout: expected_heartbeat
    monkeypatch.setattr(ardu_cli, "_service_is_active", lambda: True)
    monkeypatch.setattr(
        ardu_cli.mavutil,
        "mavlink_connection",
        lambda *args, **kwargs: connection,
    )
    args = SimpleNamespace(
        master="udp:127.0.0.1:14550",
        allow_active_service=False,
        baud=115200,
        heartbeat_timeout=1,
    )

    actual_connection, actual_heartbeat, master = ardu_cli._connect(args)

    assert actual_connection is connection
    assert actual_heartbeat is expected_heartbeat
    assert master == "udp:127.0.0.1:14550"


def test_parameter_download_requires_every_reported_index():
    connection = FakeConnection(
        [
            FakeMessage(
                message_type="PARAM_VALUE",
                param_id=b"COMPASS_ENABLE\x00",
                param_value=1,
                param_index=0,
                param_count=2,
            ),
            FakeMessage(
                message_type="PARAM_VALUE",
                param_id="COMPASS_USE",
                param_value=1,
                param_index=1,
                param_count=2,
            ),
        ]
    )

    params, complete = ardu_cli.download_parameters(connection, timeout_s=0.01)

    assert complete
    assert params == {"COMPASS_ENABLE": 1.0, "COMPASS_USE": 1.0}
    assert connection.mav.parameter_requests == [(1, 1)]


def test_status_snapshot_combines_arm_gps_ekf_and_compass_health():
    sensor_bits = (
        mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
        | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
        | mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
    )
    connection = FakeConnection(
        [
            FakeMessage(
                message_type="GPS_RAW_INT",
                fix_type=3,
                satellites_visible=14,
                lat=378353836,
                lon=-1224785680,
                alt=12000,
            ),
            FakeMessage(
                message_type="EKF_STATUS_REPORT",
                flags=1 | 16 | 32,
            ),
            FakeMessage(
                message_type="SYS_STATUS",
                onboard_control_sensors_present=sensor_bits,
                onboard_control_sensors_enabled=sensor_bits,
                onboard_control_sensors_health=sensor_bits,
            ),
        ]
    )

    report = ardu_cli.collect_status(connection, heartbeat(), timeout_s=0.01)

    assert report["complete"] is True
    assert report["armed"] is False
    assert report["gps"]["fix_type_name"] == "3D_FIX"
    assert report["gps"]["satellites_visible"] == 14
    assert report["ekf"]["flag_names"] == [
        "attitude",
        "pos_horiz_abs",
        "pos_vert_abs",
    ]
    assert report["sensors"]["compass_healthy"] is True
    assert report["sensors"]["gps_healthy"] is True
    assert report["sensors"]["prearm_healthy"] is True
    assert len(connection.mav.commands) == 3


def test_offline_compass_command_emits_policy_json(tmp_path, capsys):
    parameter_file = tmp_path / "rover.params"
    parameter_file.write_text(
        "\n".join(f"{key} {value}" for key, value in healthy_compass_params().items())
        + "\n"
    )

    assert ardu_cli.main(["compass", "--params", str(parameter_file), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["policy"]["ok"] is True
    assert report["policy"]["external_compass"]["device_id"] == 658953
    assert report["parameter_download_complete"] is True


@pytest.mark.parametrize("action", ["start", "accept", "cancel"])
def test_magcal_mutations_require_explicit_confirmation(action, capsys):
    assert ardu_cli.main(["magcal", action]) == 2
    assert "repeat with --yes" in capsys.readouterr().err


def test_magcal_refuses_armed_vehicle(monkeypatch, capsys):
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (FakeConnection(), heartbeat(armed=True), "fake"),
    )

    assert ardu_cli.main(["magcal", "start", "--yes"]) == 2
    assert "vehicle is armed" in capsys.readouterr().err


def test_magcal_start_wire_parameters_match_guarded_options():
    connection = FakeConnection()

    ardu_cli.send_magcal_command(
        connection,
        "start",
        mask=3,
        retry=True,
        autosave=False,
    )

    command = connection.mav.commands[0]
    assert command[:4] == (
        1,
        1,
        mavutil.mavlink.MAV_CMD_DO_START_MAG_CAL,
        0,
    )
    assert command[4:] == (3, 1, 0, 0, 0, 0, 0)
