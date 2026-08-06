"""`ardu_cli calibrate` — the whole mission calibration as one guarded command.

Two things are pinned here above all else:

1.  **Order.** accel (+gyro) -> reboot -> compass. Compass calibration fits mag
    samples using the AHRS attitude estimate, which comes from the accel/gyro,
    and accel offsets do not take effect until the FC reboots. Running compass
    first, or skipping the reboot, silently bakes accel error into the compass
    offsets -- a bad calibration that still reports success.

2.  **Failing closed.** Every stage that cannot be confirmed must stop the run.
    A calibration that half-completed and exited 0 is worse than one that
    failed, because nobody re-runs it.
"""

import argparse
from types import SimpleNamespace

import pytest

from spf.ardupilot import ardu_cli


def calibrate_args(**overrides):
    settings = {
        "yes": True,
        "json": False,
        "json_output": None,
        "command_timeout": 1.0,
        "pose_timeout": 1.0,
        "result_timeout": 1.0,
        "settle": 0.0,
        "reboot_timeout": 1.0,
        "magcal_timeout": 1.0,
        "prearm_timeout": 1.0,
        "ready_timeout": 0.0,
        "parameter_timeout": 1.0,
        "mask": 1,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


class Recorder:
    """Records the calibration stages in the order they actually happened."""

    def __init__(self):
        self.stages = []


@pytest.fixture
def harness(monkeypatch):
    recorder = Recorder()
    connection = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        ardu_cli, "_connect", lambda _a: (connection, SimpleNamespace(base_mode=0), "/dev/fake")
    )
    monkeypatch.setattr(ardu_cli, "_armed", lambda _h: False)
    monkeypatch.setattr(ardu_cli, "_print_accelcal_pose_plan", lambda: None)

    def fake_accelcal(_conn, **_kw):
        recorder.stages.append("accel")
        return {"success": True, "failure": None}

    def fake_reboot(_args, _conn, **_kw):
        recorder.stages.append("reboot")
        return connection

    def fake_magcal_send(_conn, action, **_kw):
        recorder.stages.append(f"magcal:{action}")

    def fake_ack(_conn, _command, _timeout):
        return SimpleNamespace(result=0)  # MAV_RESULT_ACCEPTED

    def fake_monitor(_conn, _timeout, on_event=None):
        recorder.stages.append("magcal:monitor")
        return [
            {
                "message_type": "MAG_CAL_REPORT",
                "cal_status_name": "MAG_CAL_SUCCESS",
            }
        ]

    def fake_prearm(_conn, timeout_s=None):
        recorder.stages.append("prearm")
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(ardu_cli, "run_accelcal", fake_accelcal)
    monkeypatch.setattr(ardu_cli, "reboot_flight_controller", fake_reboot)
    monkeypatch.setattr(ardu_cli, "send_magcal_command", fake_magcal_send)
    monkeypatch.setattr(ardu_cli, "_wait_command_ack", fake_ack)
    monkeypatch.setattr(ardu_cli, "monitor_magcal", fake_monitor)
    monkeypatch.setattr(ardu_cli, "run_prearm_checks", fake_prearm)
    monkeypatch.setattr(
        ardu_cli, "_calibration_compass_policy", lambda *_a: (True, [])
    )
    return recorder, monkeypatch


def test_the_happy_path_runs_accel_then_reboot_then_compass(harness):
    recorder, _ = harness
    code = ardu_cli.command_calibrate(calibrate_args())

    assert code == 0
    assert recorder.stages.index("accel") < recorder.stages.index("reboot")
    assert recorder.stages.index("reboot") < recorder.stages.index("magcal:start")
    assert recorder.stages[-1] == "prearm"


def test_a_reboot_separates_accel_from_compass(harness):
    """Without it, magcal fits against the pre-calibration attitude."""
    recorder, _ = harness
    ardu_cli.command_calibrate(calibrate_args())

    between = recorder.stages[
        recorder.stages.index("accel") + 1 : recorder.stages.index("magcal:start")
    ]
    assert "reboot" in between


def test_failed_accel_never_starts_the_compass(harness):
    recorder, monkeypatch = harness
    monkeypatch.setattr(
        ardu_cli,
        "run_accelcal",
        lambda _c, **_k: {"success": False, "failure": "flight controller said no"},
    )

    code = ardu_cli.command_calibrate(calibrate_args())

    assert code == 1
    assert not any(stage.startswith("magcal") for stage in recorder.stages)


def test_a_missing_mag_cal_report_is_a_failure(harness):
    """Silence is not success: no report means the compass is not calibrated."""
    recorder, monkeypatch = harness
    monkeypatch.setattr(ardu_cli, "monitor_magcal", lambda *a, **k: [])

    code = ardu_cli.command_calibrate(calibrate_args())

    assert code == 1
    assert "prearm" not in recorder.stages


def test_an_unsuccessful_mag_cal_report_is_a_failure(harness):
    _recorder, monkeypatch = harness
    monkeypatch.setattr(
        ardu_cli,
        "monitor_magcal",
        lambda *a, **k: [
            {"message_type": "MAG_CAL_REPORT", "cal_status_name": "MAG_CAL_FAILED"}
        ],
    )

    assert ardu_cli.command_calibrate(calibrate_args()) == 1


def test_a_dirty_prearm_fails_the_whole_calibration(harness):
    _recorder, monkeypatch = harness
    monkeypatch.setattr(
        ardu_cli, "run_prearm_checks", lambda _c, timeout_s=None: SimpleNamespace(passed=False)
    )

    assert ardu_cli.command_calibrate(calibrate_args()) == 1


def test_calibrate_refuses_without_yes(harness):
    with pytest.raises(ardu_cli.CliError, match="--yes"):
        ardu_cli.command_calibrate(calibrate_args(yes=False))


def test_calibrate_refuses_an_armed_vehicle(harness):
    _recorder, monkeypatch = harness
    monkeypatch.setattr(ardu_cli, "_armed", lambda _h: True)

    with pytest.raises(ardu_cli.CliError, match="armed"):
        ardu_cli.command_calibrate(calibrate_args())


def test_calibrate_rejects_json_output(harness):
    """The run is interactive; a JSON flag would silently swallow the prompts."""
    with pytest.raises(ardu_cli.CliError):
        ardu_cli.command_calibrate(calibrate_args(json=True))


# ------------------------------------------------------------------ reboot ---


def test_reboot_returns_a_reconnected_link(monkeypatch):
    closed = []
    old = SimpleNamespace(close=lambda: closed.append(True))
    new = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(ardu_cli, "send_fc_reboot", lambda _c: None)

    result = ardu_cli.reboot_flight_controller(
        argparse.Namespace(),
        old,
        settle_s=0.0,
        timeout_s=5.0,
        output_fn=lambda _m: None,
        sleep_fn=lambda _s: None,
        connect_fn=lambda _a: (new, SimpleNamespace(), "/dev/fake"),
    )

    assert result is new
    assert closed, "the pre-reboot link must be closed, not leaked"


def test_reboot_fails_closed_when_the_fc_never_returns(monkeypatch):
    """Returning a dead link would run the next calibration against nothing."""
    monkeypatch.setattr(ardu_cli, "send_fc_reboot", lambda _c: None)

    def never(_args):
        raise OSError("no such device")

    with pytest.raises(ardu_cli.CliError, match="did not return"):
        ardu_cli.reboot_flight_controller(
            argparse.Namespace(),
            SimpleNamespace(close=lambda: None),
            settle_s=0.0,
            timeout_s=0.2,
            output_fn=lambda _m: None,
            sleep_fn=lambda _s: None,
            connect_fn=never,
        )


# ------------------------------------------- readiness and compass policy ---
#
# Both of these were found on real hardware (Rover 4, 2026-08-04) after the
# first version of `calibrate` shipped, and neither was caught by stubs.


def test_reboot_waits_for_imu_health_not_just_a_heartbeat(monkeypatch):
    """A heartbeat arrives long before gyros are healthy; magcal must not start then."""
    waited = []
    monkeypatch.setattr(ardu_cli, "send_fc_reboot", lambda _c: None)
    monkeypatch.setattr(
        ardu_cli,
        "wait_for_sensor_health",
        lambda conn, timeout, **kw: waited.append(timeout) or True,
    )
    new = SimpleNamespace(close=lambda: None)

    ardu_cli.reboot_flight_controller(
        argparse.Namespace(),
        SimpleNamespace(close=lambda: None),
        settle_s=0.0,
        timeout_s=5.0,
        ready_timeout_s=7.0,
        output_fn=lambda _m: None,
        sleep_fn=lambda _s: None,
        connect_fn=lambda _a: (new, SimpleNamespace(), "/dev/fake"),
    )

    assert waited == [7.0]


def test_sensor_health_reports_a_dropped_link_as_transport_loss():
    """pyserial's raw SerialException must not escape as a traceback.

    Found on Rover 4 (2026-08-06): `rover ardupilot reboot --yes` printed
    "heartbeat received; waiting for the IMU to settle" and then died with
    `SerialException: device reports readiness to read but returned no data`.
    The reboot had in fact worked -- the fmuv3 simply re-enumerated a second
    time while settling.
    """

    class Dropping:
        def recv_match(self, **_kw):
            raise OSError(
                "device reports readiness to read but returned no data "
                "(device disconnected or multiple access on port?)"
            )

        mav = SimpleNamespace(command_long_send=lambda *a: None)
        target_system = target_component = 1

    with pytest.raises(ardu_cli.TransportLost):
        ardu_cli.wait_for_sensor_health(Dropping(), 5.0, output_fn=lambda _m: None)


def test_reboot_survives_a_second_re_enumeration_while_settling(monkeypatch):
    """fmuv3 enumerates twice; the first heartbeat can arrive on a doomed handle."""
    monkeypatch.setattr(ardu_cli, "send_fc_reboot", lambda _c: None)
    first = SimpleNamespace(close=lambda: None)
    second = SimpleNamespace(close=lambda: None)
    handed_out = [first, second]
    calls = []

    def flaky_health(conn, timeout, **_kw):
        calls.append(conn)
        if conn is first:
            raise ardu_cli.TransportLost("re-enumerated")
        return True

    monkeypatch.setattr(ardu_cli, "wait_for_sensor_health", flaky_health)

    result = ardu_cli.reboot_flight_controller(
        argparse.Namespace(),
        SimpleNamespace(close=lambda: None),
        settle_s=0.0,
        timeout_s=5.0,
        ready_timeout_s=7.0,
        output_fn=lambda _m: None,
        sleep_fn=lambda _s: None,
        connect_fn=lambda _a: (handed_out.pop(0), SimpleNamespace(), "/dev/fake"),
    )

    assert result is second, "must return the link that survived, not the first one"
    assert calls == [first, second]


def test_sensor_health_requires_both_gyro_and_accel():
    healthy = (
        ardu_cli.mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO
        | ardu_cli.mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL
    )
    gyro_only = ardu_cli.mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO

    class Conn:
        def __init__(self, values):
            self.values = list(values)

        def recv_match(self, **_kw):
            if not self.values:
                return None
            return SimpleNamespace(
                onboard_control_sensors_health=self.values.pop(0)
            )

        mav = SimpleNamespace(command_long_send=lambda *a: None)
        target_system = target_component = 1

    assert ardu_cli.wait_for_sensor_health(Conn([gyro_only, healthy]), 5.0)
    assert not ardu_cli.wait_for_sensor_health(
        Conn([gyro_only]), 0.2, output_fn=lambda _m: None
    )


def test_magcal_default_mask_is_the_external_compass(harness):
    """mask=0 would calibrate the internal compass the policy excludes from yaw."""
    recorder, monkeypatch = harness
    seen = {}
    monkeypatch.setattr(
        ardu_cli,
        "send_magcal_command",
        lambda _c, action, **kw: seen.update(kw) or recorder.stages.append("magcal:start"),
    )
    monkeypatch.setattr(
        ardu_cli, "_calibration_compass_policy", lambda *_a: (True, [])
    )

    ardu_cli.command_calibrate(
        calibrate_args(mask=ardu_cli.DEFAULT_MAGCAL_MASK, parameter_timeout=1.0, ready_timeout=0.0)
    )

    assert ardu_cli.DEFAULT_MAGCAL_MASK == 1
    assert seen["mask"] == 1


def test_a_broken_compass_policy_blocks_calibration(harness):
    """Calibrating while a non-primary slot drives yaw is a confidently wrong result."""
    recorder, monkeypatch = harness
    monkeypatch.setattr(
        ardu_cli,
        "_calibration_compass_policy",
        lambda *_a: (False, ["non-primary compass slot 2 is enabled for yaw"]),
    )

    code = ardu_cli.command_calibrate(
        calibrate_args(mask=1, parameter_timeout=1.0, ready_timeout=0.0)
    )

    assert code == 1
    assert not any(stage == "magcal:start" for stage in recorder.stages)


def test_an_uncalibrated_compass_does_not_block_its_own_calibration(monkeypatch):
    """The gate must ignore 'has no calibration' or calibrate could never run."""
    monkeypatch.setattr(
        ardu_cli, "download_parameters", lambda _c, _t: ({}, True)
    )
    monkeypatch.setattr(
        ardu_cli,
        "evaluate_compass_policy",
        lambda _p: SimpleNamespace(
            errors=["external compass slot 1 has no calibration"]
        ),
    )

    ok, blocking = ardu_cli._calibration_compass_policy(object(), 1.0)

    assert ok
    assert blocking == []
