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


def test_single_shutdown_packet_does_not_power_off(monkeypatch):
    shutdown = Mock()
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.subprocess.run",
        shutdown,
    )
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    shutdown.assert_not_called()
