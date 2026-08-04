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
