import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spf.mavlink.mavlink_controller import Drone


def _drone():
    drone = Drone.__new__(Drone)
    drone.armed = False
    drone.motor_active = False
    drone.disable_distance_finder = False
    drone.distance_finder = object()
    drone.run_compass_calibration = Mock()
    drone.reboot = Mock()
    return drone


def test_ultrasonic_rc_is_ignored_when_capture_disabled_the_sensor(monkeypatch, caplog):
    drone = _drone()
    drone.distance_finder = None
    drone.disable_distance_finder = True
    _set_monotonic(monkeypatch, [0.0, 0.1, 0.2])

    for _ in range(3):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=0))

    assert drone.disable_distance_finder is True
    assert "ULTRASONIC" not in caplog.text


def test_ultrasonic_rc_requires_three_consistent_samples(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    drone = _drone()
    _set_monotonic(monkeypatch, [0.0, 0.1, 0.2, 0.3, 0.4])

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=900))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    assert drone.disable_distance_finder is False
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))

    assert drone.disable_distance_finder is True
    assert caplog.text.count("DISABLE ULTRASONIC") == 1
    assert "ch12_raw=1800" in caplog.text


def _rc_message(*, ch9_raw, ch7_raw=0, ch10_raw=0, ch12_raw=0):
    return SimpleNamespace(
        chan7_raw=ch7_raw,
        chan9_raw=ch9_raw,
        chan10_raw=ch10_raw,
        chan12_raw=ch12_raw,
    )


def _set_monotonic(monkeypatch, values):
    clock = iter(values)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.time.monotonic",
        lambda: next(clock),
    )


def _complete_shutdown_hold(drone, *, final_message=None):
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(
        final_message if final_message is not None else _rc_message(ch9_raw=1800)
    )


def test_single_shutdown_packet_does_not_power_off(monkeypatch):
    shutdown = Mock()
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0])
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_not_called()


def test_shutdown_requires_release_continuous_hold_and_safe_rover(monkeypatch):
    shutdown = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 1.1, 2.1, 3.1])
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_called_once_with(["sudo", "shutdown", "0"], check=False)


def test_shutdown_hold_is_cancelled_by_release(monkeypatch):
    shutdown = Mock()
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 1.1, 1.2, 4.0])
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_not_called()


def test_shutdown_requires_release_after_unsafe_high(monkeypatch):
    shutdown = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 3.1, 3.2, 3.3, 4.3, 5.3])
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.armed = True
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.armed = False
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_called_once_with(["sudo", "shutdown", "0"], check=False)


def test_shutdown_hold_resets_after_rc_message_gap(monkeypatch):
    shutdown = Mock()
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 2.0, 3.0])
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_not_called()


def test_nonzero_shutdown_command_is_logged_and_latched(monkeypatch, caplog):
    shutdown = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 1.1, 2.1, 3.1])
    drone = _drone()
    caplog.set_level(logging.ERROR)

    _complete_shutdown_hold(drone)
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_called_once_with(["sudo", "shutdown", "0"], check=False)
    assert "RC shutdown command failed with return code 7" in caplog.messages


@pytest.mark.parametrize(
    "command_error",
    [
        pytest.param(OSError("shutdown executable unavailable"), id="oserror"),
        pytest.param(
            subprocess.SubprocessError("shutdown execution failed"),
            id="subprocess-error",
        ),
    ],
)
def test_shutdown_command_exception_is_contained_and_owns_message(
    monkeypatch,
    caplog,
    command_error,
):
    shutdown = Mock(side_effect=command_error)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 1.1, 2.1, 2.2, 2.3, 2.4])
    drone = _drone()
    caplog.set_level(logging.ERROR)

    _complete_shutdown_hold(
        drone,
        final_message=_rc_message(
            ch9_raw=1800,
            ch7_raw=1800,
            ch10_raw=1800,
            ch12_raw=1800,
        ),
    )

    shutdown.assert_called_once_with(["sudo", "shutdown", "0"], check=False)
    drone.run_compass_calibration.assert_not_called()
    drone.reboot.assert_not_called()
    assert drone.disable_distance_finder is False
    assert any(
        message.startswith("RC shutdown command failed:") for message in caplog.messages
    )

    for _ in range(3):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))

    assert drone.disable_distance_finder is True


def test_shutdown_command_is_not_repeated_while_latched(monkeypatch):
    shutdown = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(monkeypatch, [0.0, 0.1, 1.1, 2.1, 3.1, 4.1])
    drone = _drone()

    _complete_shutdown_hold(drone)
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_called_once_with(["sudo", "shutdown", "0"], check=False)


def test_shutdown_retry_requires_release_and_new_valid_hold(monkeypatch):
    shutdown = Mock(
        side_effect=[
            OSError("shutdown executable unavailable"),
            SimpleNamespace(returncode=0),
        ]
    )
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    _set_monotonic(
        monkeypatch,
        [0.0, 0.1, 1.1, 2.1, 3.1, 3.2, 3.3, 4.3, 5.3],
    )
    drone = _drone()

    _complete_shutdown_hold(drone)
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    assert shutdown.call_count == 1

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    assert shutdown.call_count == 2
