from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import Mock

import numpy as np

from spf.mavlink.mavlink_controller import (
    MOVE_ABORTED,
    Drone,
    MavlinkConnectionError,
)


def _moving_drone() -> Drone:
    drone = Drone.__new__(Drone)
    drone.gps = np.asarray([0.0, 0.0])
    drone.tolerance_in_m = 1.0
    drone.distance_finder = None
    drone.disable_distance_finder = True
    drone.motor_active = True
    drone.mav_mode = "ROVER_MODE_GUIDED"
    drone.armed = True
    drone.connection_failure = None
    drone.planner_should_move = True
    drone.reposition = Mock()
    drone.distance_to_target = Mock(return_value=10.0)
    drone.set_mode = Mock()
    drone._initialize_motion_stop_state()
    return drone


def test_capture_failure_interrupts_an_active_waypoint_and_enters_hold():
    drone = _moving_drone()
    result = []
    moving = threading.Thread(
        target=lambda: result.append(drone.move_to_point(np.asarray([1.0, 1.0])))
    )
    moving.start()

    deadline = time.monotonic() + 1.0
    while not drone.reposition.called and time.monotonic() < deadline:
        time.sleep(0.005)
    assert drone.reposition.called

    drone.request_motion_stop("capture incident incident-test")
    moving.join(timeout=1.0)

    assert not moving.is_alive()
    # move_to_point reports an outcome rather than a bool: a waypoint skipped
    # by stall recovery must not read the same as an aborted capture.
    assert result == [MOVE_ABORTED]
    assert drone.planner_should_move is False
    drone.set_mode.assert_called_once_with("HOLD")


def test_pending_hold_is_retried_after_mavlink_recovers():
    drone = _moving_drone()
    drone.set_mode.side_effect = [OSError("MAVLink unavailable"), None]

    drone.request_motion_stop("capture failed")

    assert drone._try_enter_abort_hold() is False
    assert drone._try_enter_abort_hold() is True
    assert drone.set_mode.call_count == 2


def test_abort_hold_wait_is_bounded_when_mavlink_stays_unavailable():
    drone = _moving_drone()
    drone.set_mode.side_effect = OSError("MAVLink unavailable")
    drone.request_motion_stop("capture failed")

    started = time.monotonic()
    assert drone.wait_for_abort_hold(timeout_seconds=0.05) is False

    assert time.monotonic() - started < 0.5


def test_active_waypoint_fails_when_mavlink_reconnect_is_exhausted():
    drone = _moving_drone()
    drone.connection_failure = None
    outcome = []

    def move():
        try:
            drone.move_to_point(np.asarray([1.0, 1.0]))
        except BaseException as error:
            outcome.append(error)

    moving = threading.Thread(target=move)
    moving.start()
    deadline = time.monotonic() + 1.0
    while not drone.reposition.called and time.monotonic() < deadline:
        time.sleep(0.005)
    drone.connection_failure = MavlinkConnectionError("heartbeat lost")
    moving.join(timeout=2.0)

    assert not moving.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], MavlinkConnectionError)


def test_ordinary_capture_failure_exits_despite_lingering_non_daemon_thread():
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), environment.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tests/helpers/run_capture_failure_exit.py"),
        ],
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert combined.count("Traceback (most recent call last)") == 1
    assert "incident-test" in combined
    assert "receiver:test-radio" in combined
