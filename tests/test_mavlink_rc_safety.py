from types import SimpleNamespace
from unittest.mock import Mock

from spf.mavlink.mavlink_controller import Drone


def _drone():
    drone = Drone.__new__(Drone)
    drone.armed = False
    drone.motor_active = False
    drone.disable_distance_finder = False
    drone.run_compass_calibration = Mock()
    drone.reboot = Mock()
    return drone


def _rc_message(*, ch9_raw):
    return SimpleNamespace(
        chan7_raw=0,
        chan9_raw=ch9_raw,
        chan10_raw=0,
        chan12_raw=0,
    )


def _set_monotonic(monkeypatch, values):
    clock = iter(values)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.time.monotonic",
        lambda: next(clock),
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
