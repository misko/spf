import argparse
import io
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
        self.closed = False

    def close(self):
        self.closed = True

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


def test_no_arguments_prints_operational_cheatsheet(capsys):
    assert ardu_cli.main([]) == 0

    output = capsys.readouterr().out
    assert "quick reference:" in output
    assert "sudo systemctl stop mavlink_controller.service" in output
    assert "compass --repair --yes --parameter-timeout 60" in output
    assert "magcal start --yes --mask 1 --retry --monitor-seconds 300" in output
    assert "accelcal start --yes" in output
    assert "sudo systemctl restart mavlink_controller.service" in output
    assert "exit codes:" in output


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


def test_direct_serial_is_claimed_exclusively_before_heartbeat(monkeypatch):
    events = []

    class Port:
        def fileno(self):
            events.append("fileno")
            return 42

    connection = FakeConnection()
    connection.port = Port()

    def wait_heartbeat(timeout):
        events.append("heartbeat")
        return heartbeat()

    connection.wait_heartbeat = wait_heartbeat
    monkeypatch.setattr(ardu_cli, "_service_is_active", lambda: False)
    monkeypatch.setattr(
        ardu_cli.mavutil,
        "mavlink_connection",
        lambda *args, **kwargs: connection,
    )
    monkeypatch.setattr(
        ardu_cli.fcntl,
        "ioctl",
        lambda fd, operation: events.append(("ioctl", fd, operation)),
    )
    args = SimpleNamespace(
        master="/dev/serial/by-id/usb-ArduPilot-test",
        allow_active_service=False,
        baud=115200,
        heartbeat_timeout=1,
    )

    ardu_cli._connect(args)

    assert events == [
        "fileno",
        ("ioctl", 42, ardu_cli.termios.TIOCEXCL),
        "heartbeat",
    ]


def test_exclusive_claim_failure_closes_connection(monkeypatch):
    closed = []
    connection = SimpleNamespace(
        port=SimpleNamespace(fileno=lambda: 42),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        ardu_cli.fcntl,
        "ioctl",
        lambda fd, operation: (_ for _ in ()).throw(OSError("busy")),
    )

    with pytest.raises(ardu_cli.CliError, match="exclusive MAVLink ownership"):
        ardu_cli._claim_serial_exclusive(connection, "/dev/ttyACM2")

    assert closed == [True]


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


def command_ack(command, result=mavutil.mavlink.MAV_RESULT_ACCEPTED):
    return FakeMessage(
        message_type="COMMAND_ACK",
        command=command,
        result=result,
    )


def accelcal_position(position):
    return FakeMessage(
        message_type="COMMAND_LONG",
        command=mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS,
        param1=float(position),
    )


def test_accelcal_requires_explicit_confirmation_before_connect(monkeypatch, capsys):
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: pytest.fail("accelcal without --yes must not connect"),
    )

    assert ardu_cli.main(["accelcal", "start"]) == 2
    assert "repeat with --yes" in capsys.readouterr().err


def test_accelcal_refuses_armed_vehicle(monkeypatch, capsys):
    monkeypatch.setattr(
        ardu_cli,
        "_connect",
        lambda args: (FakeConnection(), heartbeat(armed=True), "fake"),
    )

    assert ardu_cli.main(["accelcal", "start", "--yes"]) == 2
    assert "vehicle is armed" in capsys.readouterr().err


def test_accelcal_wire_commands_match_ardupilot_protocol():
    connection = FakeConnection()

    ardu_cli.send_accelcal_start(connection)
    ardu_cli.send_accelcal_position(connection, 4)

    start, position = connection.mav.commands
    assert start[:4] == (
        1,
        1,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
        0,
    )
    assert start[4:] == (0, 0, 0, 0, 1, 0, 0)
    assert position[:4] == (
        1,
        1,
        mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS,
        0,
    )
    assert position[4:] == (4, 0, 0, 0, 0, 0, 0)


def test_run_accelcal_advances_without_optional_pose_acks():
    messages = [command_ack(mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION)]
    for position in range(1, 7):
        messages.append(accelcal_position(position))
    messages.append(accelcal_position(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS))
    connection = FakeConnection(messages)
    prompts = []
    output = []

    report = ardu_cli.run_accelcal(
        connection,
        command_timeout_s=0.01,
        pose_timeout_s=0.01,
        result_timeout_s=0.01,
        input_fn=lambda prompt: prompts.append(prompt) or "",
        output_fn=output.append,
    )

    assert report["success"] is True
    assert report["poses_completed"] == [1, 2, 3, 4, 5, 6]
    assert len(prompts) == 6
    rendered = "\n".join(output)
    for title in (
        "LEVEL",
        "LEFT SIDE",
        "RIGHT SIDE",
        "NOSE DOWN",
        "NOSE UP",
        "UPSIDE DOWN",
    ):
        assert title in rendered
    assert [command[2] for command in connection.mav.commands] == [
        mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
        *([mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS] * 6),
    ]


def test_run_accelcal_fails_closed_on_out_of_order_pose():
    connection = FakeConnection(
        [
            command_ack(mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION),
            accelcal_position(2),
        ]
    )

    with pytest.raises(ardu_cli.CliError, match="requested pose 2, expected 1"):
        ardu_cli.run_accelcal(
            connection,
            command_timeout_s=0.01,
            pose_timeout_s=0.01,
            result_timeout_s=0.01,
            input_fn=lambda prompt: "",
        )


def test_run_accelcal_reports_terminal_failure():
    messages = [command_ack(mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION)]
    for position in range(1, 7):
        messages.append(accelcal_position(position))
    messages.append(accelcal_position(mavutil.mavlink.ACCELCAL_VEHICLE_POS_FAILED))
    connection = FakeConnection(messages)

    report = ardu_cli.run_accelcal(
        connection,
        command_timeout_s=0.01,
        pose_timeout_s=0.01,
        result_timeout_s=0.01,
        input_fn=lambda prompt: "",
        output_fn=lambda message: None,
    )

    assert report["success"] is False
    assert report["failure"] == "flight controller reported failure"


# ------------------------------------------------- accelcal trace + verify ---
#
# Rover 4 (2026-08-04) finished all six poses, saved INS_ACCSCAL_* at
# 0.992/0.993/0.995 -- and `run_accelcal` still raised "no terminal calibration
# result". Stored values alone cannot establish the outcome of that run.
# These tests cover evidence gathering, repeated pose requests at completion,
# and parameter inspection without certifying calibration success.


class RecordingConnection(FakeConnection):
    """FakeConnection that remembers which message types were asked for."""

    def __init__(self, messages=()):
        super().__init__(messages)
        self.requested_types = []

    def recv_match(self, *, blocking, type=None, timeout=None):
        self.requested_types.append(type)
        return super().recv_match(blocking=blocking, type=type, timeout=timeout)


def statustext(text, severity=6):
    return FakeMessage(message_type="STATUSTEXT", severity=severity, text=text)


def full_pose_messages(terminal=None):
    messages = [command_ack(mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION)]
    for position in range(1, 7):
        messages.append(accelcal_position(position))
    if terminal is not None:
        messages.append(accelcal_position(terminal))
    return messages


def run_traced_accelcal(connection, tracer):
    return ardu_cli.run_accelcal(
        connection,
        command_timeout_s=0.01,
        pose_timeout_s=0.01,
        result_timeout_s=0.01,
        input_fn=lambda prompt: "",
        output_fn=lambda message: None,
        tracer=tracer,
    )


def test_untraced_accelcal_keeps_the_narrow_message_filter():
    connection = RecordingConnection(
        full_pose_messages(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS)
    )

    run_traced_accelcal(connection, ardu_cli.MessageTracer(enabled=False))

    pose_filters = [
        wanted for wanted in connection.requested_types if wanted is not None
    ]
    assert pose_filters
    assert all(
        set(wanted) == {"COMMAND_LONG", "STATUSTEXT"}
        for wanted in pose_filters
        if not isinstance(wanted, str)
    )


def test_trace_takes_delivery_of_every_message_not_just_the_matched_two():
    """A filter that hides COMMAND_INT/COMMAND_ACK would hide the bug itself."""
    messages = full_pose_messages(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS)
    messages.insert(
        1,
        FakeMessage(
            message_type="COMMAND_INT",
            command=mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS,
            param1=17.0,
        ),
    )
    connection = RecordingConnection(messages)
    stream = io.StringIO()
    tracer = ardu_cli.MessageTracer(enabled=True, stream=stream)

    report = run_traced_accelcal(connection, tracer)
    tracer.close()

    assert report["success"] is True
    # Everything after the start ACK is fetched unfiltered while tracing.
    assert None in connection.requested_types
    rendered = stream.getvalue()
    assert "COMMAND_INT" in rendered
    assert "MAV_CMD_ACCELCAL_VEHICLE_POS" in rendered
    assert "param1=17" in rendered
    assert tracer.counts["COMMAND_INT"] == 1


def test_trace_records_command_ack_statustext_and_pose_detail():
    messages = full_pose_messages(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS)
    messages.insert(1, statustext("Calibration step 1", severity=5))
    messages.insert(
        2, command_ack(mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS, result=0)
    )
    stream = io.StringIO()
    tracer = ardu_cli.MessageTracer(enabled=True, stream=stream)

    run_traced_accelcal(FakeConnection(messages), tracer)
    tracer.close()

    rendered = stream.getvalue()
    assert "COMMAND_ACK" in rendered and "MAV_RESULT_ACCEPTED" in rendered
    assert "Calibration step 1" in rendered
    assert "severity=5" in rendered
    assert f"COMMAND_LONG command={mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS}" in (
        rendered
    )
    assert "pose-wait" in rendered
    assert "message totals" in rendered


def test_trace_counts_telemetry_without_printing_every_frame():
    messages = full_pose_messages(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS)
    messages.insert(1, heartbeat())
    messages.insert(2, heartbeat())
    stream = io.StringIO()
    tracer = ardu_cli.MessageTracer(enabled=True, stream=stream)

    run_traced_accelcal(FakeConnection(messages), tracer)
    tracer.close()

    assert tracer.counts["HEARTBEAT"] == 2
    printed = [line for line in stream.getvalue().splitlines() if "HEARTBEAT" in line]
    # Counted in the totals line only -- never one line per frame.
    assert len(printed) == 1
    assert "message totals" in printed[0]


def test_trace_output_file_is_written_and_implies_tracing(tmp_path):
    path = tmp_path / "traces" / "accelcal.txt"
    tracer = ardu_cli.tracer_from_args(
        argparse.Namespace(trace=False, trace_output=str(path))
    )
    assert tracer.enabled

    run_traced_accelcal(
        FakeConnection(
            full_pose_messages(mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS)
        ),
        tracer,
    )
    tracer.close()

    assert "COMMAND_LONG" in path.read_text()


def test_tracer_is_disabled_by_default():
    tracer = ardu_cli.tracer_from_args(argparse.Namespace())

    assert not tracer.enabled
    tracer.note("ignored")
    tracer.close()
    assert tracer.lines == []


def test_missing_terminal_result_still_fails_but_says_what_to_do():
    """Fail closed, but name the parameters and the flag that closes this out."""
    connection = FakeConnection(full_pose_messages(terminal=None))

    with pytest.raises(ardu_cli.CliError) as error:
        run_traced_accelcal(connection, ardu_cli.MessageTracer(enabled=False))

    text = str(error.value)
    assert "not confirmed" in text
    assert "earlier run" in text
    assert "PROBABLY SAVED" not in text
    assert "INS_ACCOFFS_" in text and "INS_ACCSCAL_" in text
    assert "accelcal verify" in text
    assert "--trace" in text


def accel_params(offsets=(0.058, -0.107, -0.836), scales=(0.992, 0.993, 0.995)):
    axes = ("X", "Y", "Z")
    params = {f"INS_ACCOFFS_{axis}": offsets[i] for i, axis in enumerate(axes)}
    params.update({f"INS_ACCSCAL_{axis}": scales[i] for i, axis in enumerate(axes)})
    return params


def test_rover4_values_are_observations_not_a_calibration_verdict():
    report = ardu_cli.evaluate_accel_calibration(accel_params())
    assert report["status"] == "complete"
    assert report["imus"][0]["status"] == "nondefault_values"
    assert "calibrated" not in report
    assert "calibrated" not in report["imus"][0]
    assert "do not confirm" in report["notice"]


def test_unity_scales_with_nonzero_offsets_are_nondefault_values():
    report = ardu_cli.evaluate_accel_calibration(accel_params(scales=(1.0, 1.0, 1.0)))
    assert report["status"] == "complete"
    assert report["imus"][0]["status"] == "nondefault_values"


def test_factory_defaults_are_reported_without_a_calibration_verdict():
    report = ardu_cli.evaluate_accel_calibration(
        accel_params(offsets=(0.0, 0.0, 0.0), scales=(1.0, 1.0, 1.0))
    )
    assert report["status"] == "complete"
    assert report["imus"][0]["status"] == "default_values"


@pytest.mark.parametrize(
    "field,value",
    [
        ("INS_ACCOFFS_X", float("nan")),
        ("INS_ACCOFFS_Y", float("inf")),
        ("INS_ACCOFFS_Z", -float("inf")),
        ("INS_ACCSCAL_X", float("nan")),
        ("INS_ACCSCAL_Y", float("inf")),
        ("INS_ACCSCAL_Z", 4.0),
        ("INS_ACCSCAL_X", 0.0),
    ],
)
def test_invalid_values_are_flagged_and_json_is_finite(field, value):
    params = accel_params()
    params[field] = value
    report = ardu_cli.evaluate_accel_calibration(params)
    assert report["status"] == "invalid_values"
    assert field in report["imus"][0]["invalid_parameters"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "params",
    [
        {"INS_ACCSCAL_X": 0.992, "INS_ACCOFFS_X": 0.058},
        {"INS_ACCOFFS_X": 0.058},
        {"INS_ACCSCAL_X": 0.992},
    ],
)
def test_partial_imu_values_are_unknown_even_in_a_complete_download(params):
    report = ardu_cli.evaluate_accel_calibration(params)
    assert report["status"] == "unknown"
    assert len(report["imus"]) == 1
    assert report["imus"][0]["missing_parameters"]


def test_incomplete_download_is_unknown_even_with_a_complete_imu():
    report = ardu_cli.evaluate_accel_calibration(
        accel_params(), snapshot_complete=False
    )
    assert report["status"] == "unknown"
    assert report["imus"][0]["status"] == "nondefault_values"


def test_absent_ins_parameters_are_unknown():
    report = ardu_cli.evaluate_accel_calibration({"COMPASS_USE": 1})
    assert report["imus"] == []
    assert report["status"] == "unknown"


def test_good_imu_does_not_hide_an_invalid_second_slot():
    params = accel_params()
    params.update(
        {
            key.replace("INS_ACC", "INS_ACC2"): value
            for key, value in accel_params().items()
        }
    )
    params["INS_ACC2SCAL_Z"] = 4.0
    report = ardu_cli.evaluate_accel_calibration(params)
    assert report["status"] == "invalid_values"
    assert [imu["status"] for imu in report["imus"]] == [
        "nondefault_values",
        "invalid_values",
    ]


def test_default_secondary_slot_is_described_without_inferring_sensor_activity():
    params = accel_params()
    defaults = accel_params(offsets=(0.0, 0.0, 0.0), scales=(1.0, 1.0, 1.0))
    params.update(
        {key.replace("INS_ACC", "INS_ACC2"): value for key, value in defaults.items()}
    )
    report = ardu_cli.evaluate_accel_calibration(params)
    assert report["status"] == "complete"
    assert [imu["status"] for imu in report["imus"]] == [
        "nondefault_values",
        "default_values",
    ]
    assert "all active sensors" in report["notice"]


@pytest.mark.parametrize(
    "params,complete,expected_code,expected_status",
    [
        (accel_params(), True, 0, "complete"),
        (
            accel_params(offsets=(0.0, 0.0, 0.0), scales=(1.0, 1.0, 1.0)),
            True,
            0,
            "complete",
        ),
        (accel_params(scales=(1.0, 1.0, 4.0)), True, 1, "invalid_values"),
        ({"INS_ACCSCAL_X": 0.992, "INS_ACCOFFS_X": 0.058}, False, 2, "unknown"),
        (accel_params(), False, 2, "unknown"),
        ({"COMPASS_USE": 1}, False, 2, "unknown"),
        ({}, False, 2, "unknown"),
    ],
)
@pytest.mark.parametrize("json_output", [False, True])
def test_verify_reports_inspection_status_without_writes(
    monkeypatch, capsys, params, complete, expected_code, expected_status, json_output
):
    connection = FakeConnection()
    monkeypatch.setattr(
        ardu_cli, "_connect", lambda args: (connection, heartbeat(), "fake")
    )
    monkeypatch.setattr(
        ardu_cli, "download_parameters", lambda _c, _t: (params, complete)
    )
    args = ["accelcal", "verify"] + (["--json"] if json_output else [])
    assert ardu_cli.main(args) == expected_code
    output = capsys.readouterr().out
    if json_output:
        report = json.loads(output)
        assert report["status"] == expected_status
        assert report["parameter_snapshot_complete"] == complete
        assert "calibrated" not in report
    else:
        assert f"Parameter inspection: {expected_status}" in output
        assert "do not confirm" in output
        assert "PASS" not in output
    assert connection.mav.commands == []
    assert connection.closed


@pytest.mark.parametrize("tracing", [False, True])
@pytest.mark.parametrize(
    "terminal",
    [
        mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS,
        mavutil.mavlink.ACCELCAL_VEHICLE_POS_FAILED,
    ],
)
def test_final_wait_ignores_repeated_poses_and_receives_terminal_result(
    tracing, terminal
):
    messages = full_pose_messages(terminal)
    messages[-1:-1] = [accelcal_position(position) for position in (6, 5, 1, 6)]
    connection = FakeConnection(messages)
    tracer = ardu_cli.MessageTracer(enabled=tracing, stream=io.StringIO())
    report = run_traced_accelcal(connection, tracer)
    assert report["success"] == (
        terminal == mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS
    )
    assert report["poses_completed"] == list(range(1, 7))
    assert (
        len(connection.mav.commands) == 7
    ), "repeated poses must not request more samples"
    assert not connection.messages
    if tracing:
        assert "ignoring repeated pose 6" in "\n".join(tracer.lines)


def test_repeated_final_poses_do_not_extend_the_original_deadline(monkeypatch):
    now = [0.0]
    poses = iter(range(1, 7))
    terminal_budgets = []
    monkeypatch.setattr(ardu_cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        ardu_cli,
        "_wait_command_ack",
        lambda *_a: command_ack(ardu_cli.ACCELCAL_START_COMMAND),
    )

    def event(_connection, timeout_s, tracer=None, phase="pose-wait"):
        if phase != "terminal-wait":
            position = next(poses)
            return ("pose", position, accelcal_position(position))
        terminal_budgets.append(timeout_s)
        assert len(terminal_budgets) <= 4, "final wait restarted its timeout"
        now[0] += 0.25
        return ("pose", 6, accelcal_position(6))

    monkeypatch.setattr(ardu_cli, "_wait_accelcal_event", event)
    with pytest.raises(ardu_cli.CliError, match="not confirmed within 1s"):
        ardu_cli.run_accelcal(
            FakeConnection(),
            command_timeout_s=1,
            pose_timeout_s=1,
            result_timeout_s=1,
            input_fn=lambda _p: "",
            output_fn=lambda _m: None,
        )
    assert terminal_budgets == [1.0, 0.75, 0.5, 0.25]


def test_trace_is_saved_when_calibration_raises(monkeypatch, tmp_path):
    connection = FakeConnection()
    monkeypatch.setattr(
        ardu_cli, "_connect", lambda _a: (connection, heartbeat(), "fake")
    )

    def fail(_connection, **kwargs):
        kwargs["tracer"].note("terminal result missing")
        raise ardu_cli.CliError("completion not confirmed")

    monkeypatch.setattr(ardu_cli, "run_accelcal", fail)
    path = tmp_path / "failed-trace.txt"
    assert (
        ardu_cli.main(["accelcal", "start", "--yes", "--trace-output", str(path)]) == 2
    )
    assert "terminal result missing" in path.read_text()
