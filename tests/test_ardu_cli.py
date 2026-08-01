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
        "COMPASS_OFFS_MAX": 1800,
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


def test_compass_repair_requires_confirmation_before_connect(monkeypatch, capsys):
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: pytest.fail("repair without --yes must not connect"),
    )

    assert ardu_cli.main(["compass", "--repair"]) == 2
    assert "repeat with --yes" in capsys.readouterr().err


def test_compass_repair_writes_acknowledged_priority_and_use_changes(
    monkeypatch, capsys
):
    params = healthy_compass_params()
    params.update(
        {
            "COMPASS_PRIO1_ID": 131594,
            "COMPASS_PRIO2_ID": 658953,
            "COMPASS_USE": 0,
            "COMPASS_USE2": 1,
        }
    )
    connection = FakeConnection()
    applied = {}
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (connection, heartbeat(), "fake"),
    )
    monkeypatch.setattr(
        ardu_cli,
        "download_parameters",
        lambda connection, timeout: (params.copy(), True),
    )

    def fake_apply(_connection, mutable_params, changes):
        for key, value in changes.items():
            applied[key] = {"before": mutable_params[key], "after": value}
            mutable_params[key] = value
        return applied

    monkeypatch.setattr(ardu_cli, "apply_parameter_changes", fake_apply)

    assert ardu_cli.main(["compass", "--repair", "--yes"]) == 0
    output = capsys.readouterr().out
    assert applied == {
        "COMPASS_PRIO1_ID": {"before": 131594, "after": 658953},
        "COMPASS_PRIO2_ID": {"before": 658953, "after": 131594},
        "COMPASS_USE": {"before": 0, "after": 1},
        "COMPASS_USE2": {"before": 1, "after": 0},
    }
    assert "PENDING: stored policy is correct" in output
    assert "REBOOT REQUIRED" in output


def test_compass_repair_refuses_duplicate_device_ids(monkeypatch, capsys):
    params = healthy_compass_params()
    params["COMPASS_DEV_ID2"] = params["COMPASS_DEV_ID"]
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (FakeConnection(), heartbeat(), "fake"),
    )
    monkeypatch.setattr(
        ardu_cli,
        "download_parameters",
        lambda connection, timeout: (params, True),
    )

    assert ardu_cli.main(["compass", "--repair", "--yes"]) == 2
    assert "duplicate detected" in capsys.readouterr().err


def test_compass_repair_refuses_armed_vehicle(monkeypatch, capsys):
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (FakeConnection(), heartbeat(armed=True), "fake"),
    )
    monkeypatch.setattr(
        ardu_cli,
        "download_parameters",
        lambda connection, timeout: (healthy_compass_params(), True),
    )

    assert ardu_cli.main(["compass", "--repair", "--yes"]) == 2
    assert "vehicle is armed" in capsys.readouterr().err


def test_apply_parameter_changes_waits_for_matching_acknowledgements():
    class AckConnection:
        def __init__(self):
            self.pending = None
            self.writes = []

        def param_set_send(self, name, value):
            self.writes.append((name, value))
            self.pending = FakeMessage(
                message_type="PARAM_VALUE", param_id=name, param_value=value
            )

        def recv_match(self, *, type, blocking, timeout):
            message, self.pending = self.pending, None
            return message

    connection = AckConnection()
    params = {"COMPASS_PRIO1_ID": 131594.0, "COMPASS_USE": 0.0}

    applied = ardu_cli.apply_parameter_changes(
        connection,
        params,
        {"COMPASS_PRIO1_ID": 658953, "COMPASS_USE": 1},
    )

    assert connection.writes == [
        ("COMPASS_PRIO1_ID", 658953.0),
        ("COMPASS_USE", 1.0),
    ]
    assert params == {"COMPASS_PRIO1_ID": 658953.0, "COMPASS_USE": 1.0}
    assert applied["COMPASS_PRIO1_ID"] == {
        "before": 131594.0,
        "after": 658953.0,
    }


def test_apply_parameter_changes_rejects_coerced_acknowledgement():
    class RejectingConnection:
        def __init__(self):
            self.writes = 0
            self.pending = None

        def param_set_send(self, name, value):
            self.writes += 1
            self.pending = FakeMessage(
                message_type="PARAM_VALUE", param_id=name, param_value=0
            )

        def recv_match(self, *, type, blocking, timeout):
            message, self.pending = self.pending, None
            return message

    connection = RejectingConnection()

    with pytest.raises(ardu_cli.CliError, match="did not acknowledge"):
        ardu_cli.apply_parameter_changes(
            connection,
            {"COMPASS_PRIO1_ID": 131594.0},
            {"COMPASS_PRIO1_ID": 658953},
        )

    assert connection.writes == 3


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


def test_monitor_magcal_streams_progress_and_exits_on_terminal_report():
    progress = FakeMessage(
        message_type="MAG_CAL_PROGRESS",
        compass_id=0,
        cal_status=2,
        completion_pct=37,
    )
    report = FakeMessage(
        message_type="MAG_CAL_REPORT",
        compass_id=0,
        cal_status=4,
        fitness=5.25,
        autosaved=1,
    )
    event_after_terminal_report = FakeMessage(
        message_type="MAG_CAL_PROGRESS",
        compass_id=0,
        cal_status=2,
        completion_pct=99,
    )
    connection = FakeConnection([progress, report, event_after_terminal_report])
    streamed = []

    events = ardu_cli.monitor_magcal(
        connection,
        timeout_s=300,
        on_event=streamed.append,
    )

    assert [event["message_type"] for event in events] == [
        "MAG_CAL_PROGRESS",
        "MAG_CAL_REPORT",
    ]
    assert streamed == events
    assert connection.messages == [event_after_terminal_report]


def test_magcal_command_prints_ack_and_events_without_waiting_for_timeout(
    monkeypatch, capsys
):
    connection = FakeConnection(
        [
            FakeMessage(
                message_type="MAG_CAL_PROGRESS",
                compass_id=0,
                cal_status=2,
                completion_pct=51,
            ),
            FakeMessage(
                message_type="MAG_CAL_REPORT",
                compass_id=0,
                cal_status=4,
                fitness=4.5,
                autosaved=1,
            ),
        ]
    )
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (connection, heartbeat(), "fake"),
    )
    monkeypatch.setattr(
        ardu_cli,
        "_wait_command_ack",
        lambda connection, command, timeout: FakeMessage(
            message_type="COMMAND_ACK",
            command=command,
            result=mavutil.mavlink.MAV_RESULT_ACCEPTED,
        ),
    )

    assert (
        ardu_cli.main(
            [
                "magcal",
                "start",
                "--yes",
                "--mask",
                "1",
                "--monitor-seconds",
                "300",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.count("magcal start: MAV_RESULT_ACCEPTED") == 1
    assert "Compass 0: MAG_CAL_RUNNING_STEP_ONE, 51%" in output
    assert "Compass 0: MAG_CAL_SUCCESS, fitness=4.5 autosaved=1" in output
