"""The capture must record exactly while the planner is driving the vehicle.

Two defects seen in the field on 2026-08-07, both about the boundary between
"the planner is in control" and "the collector is recording":

  1. `planner_in_control` is assigned at the BOTTOM of the planner's waypoint
     loop, so it stays False through the drive-to-home AND the whole first
     waypoint. Rover 1 logged `Planner starting to issue move commands` at
     01:12:42 and was still logging `Waiting for drone to start moving` at
     01:14:27 -- driving for two minutes while nothing was recorded.

  2. The collector reads `is_planner_in_control()` once, to decide when to
     START. Nothing re-checks it. When the operator took MANUAL, rover 4 kept
     writing snapshots of a stationary vehicle (progress 320 -> 578 of 3000),
     so those records carry near-static gps_lat/long -- and rx/tx ground truth
     is derived from exactly those fields.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from spf.mavlink.mavlink_controller import MOVE_REACHED, Drone


SITE = np.asarray([-122.4791383, 37.83550286])


def _planner_drone(monkeypatch, *, waypoints=3):
    """A Drone with just enough stubbed to run the real run_planner()."""
    drone = Drone.__new__(Drone)

    # Distinct points: run_planner sleeps instead of moving when the next point
    # equals the last, so repeating one point spins forever.
    def _points():
        for step in itertools.count(1):
            yield SITE + np.asarray([step * 1e-4, step * 1e-4])

    drone.planner = SimpleNamespace(
        get_home_point=lambda: SITE,
        yield_points=_points,
    )
    drone.planner_in_control = False
    drone.planner_should_move = True
    drone.drone_ready = True
    drone.ignore_mode = False
    drone.armed = True
    drone.mav_mode = "ROVER_MODE_GUIDED"
    drone.connection_failure = None
    drone._motion_stop_requested = SimpleNamespace(clear=lambda: None)
    drone._motion_hold_sent = SimpleNamespace(clear=lambda: None)
    drone._motion_stop_reason = None

    for name in ("single_operation_mode_on", "single_operation_mode_off", "arm"):
        monkeypatch.setattr(drone, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(drone, "set_home", lambda **k: None, raising=False)
    monkeypatch.setattr(
        drone, "_wait_for_mode", lambda *a, **k: None, raising=False
    )

    # Record what planner_in_control was AT THE MOMENT each move began, and
    # stop the loop after `waypoints` moves so the test terminates.
    seen: list[bool] = []

    def fake_move_to_point(point, log_interval=5):
        seen.append(bool(drone.planner_in_control))
        if len(seen) >= waypoints:
            drone.planner_should_move = False
        return MOVE_REACHED

    monkeypatch.setattr(drone, "move_to_point", fake_move_to_point, raising=False)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.time.sleep", lambda _s: None
    )
    return drone, seen


def test_planner_is_in_control_from_the_first_move_not_after_it(monkeypatch):
    """RED: the flag lags a whole waypoint, so the first leg is never recorded.

    The collector gates the START of recording on this flag. Setting it at the
    bottom of the loop means the drive-to-home and the entire first waypoint
    happen with recording still stopped.
    """
    drone, seen = _planner_drone(monkeypatch)

    drone.run_planner()

    assert seen, "run_planner should have issued at least one move"
    assert seen[0] is True, (
        "planner_in_control was False while the planner was already driving; "
        f"observed across moves: {seen}"
    )


def test_planner_control_is_released_when_the_planner_stops(monkeypatch):
    """The flag must not stay True once the planner has finished."""
    drone, _seen = _planner_drone(monkeypatch)

    drone.run_planner()

    assert drone.planner_in_control is False


# --------------------------------------------------------- collector gating ---


def test_capture_stops_when_the_planner_loses_control_mid_run():
    """RED: nothing re-checks planner control once recording has started.

    An operator taking MANUAL is a designed interlock, not a fault -- but the
    rover then sits still while the collector keeps writing snapshots whose
    gps_lat/long barely change. Those records become ground truth for a vehicle
    that was not moving.
    """
    from spf.mavlink_radio_collection import planner_control_lost

    # In control -> keep recording.
    assert planner_control_lost(SimpleNamespace(is_planner_in_control=lambda: True)) is False
    # Control lost -> the capture must not keep writing.
    assert planner_control_lost(SimpleNamespace(is_planner_in_control=lambda: False)) is True


def test_a_fake_drone_capture_is_never_gated_on_planner_control():
    """--fake-drone has no vehicle; bench captures must still run."""
    from spf.mavlink_radio_collection import planner_control_lost

    assert planner_control_lost(None) is False


# ------------------------------------------- accounting for lost control ------
#
# Detection alone does not help if nothing records WHEN control was lost. The
# records written during a takeover are not corrupt -- the IQ and the GPS are
# both accurate -- they simply describe a stationary vehicle, so they carry no
# bearing diversity and bias any aggregate. The capture must therefore say how
# much of itself was recorded that way, so it can be filtered rather than
# discovered by someone puzzling over a flat bearing track months later.


def _tracker():
    from spf.mavlink_radio_collection import LostIntervalTracker

    return LostIntervalTracker()


def test_a_capture_with_continuous_control_reports_no_lost_time():
    t = _tracker()
    for tick in range(10):
        t.update(False, now=float(tick))
    assert t.lost_seconds == 0.0
    assert t.episodes == 0


def test_lost_time_is_accumulated_across_a_takeover():
    t = _tracker()
    t.update(False, now=0.0)
    t.update(True, now=10.0)   # takeover begins
    t.update(True, now=25.0)
    t.update(False, now=40.0)  # control returns
    t.update(False, now=50.0)

    assert t.episodes == 1
    assert t.lost_seconds == pytest.approx(30.0)


def test_multiple_takeovers_are_counted_separately():
    t = _tracker()
    for now, lost in [(0, False), (5, True), (10, False), (20, True), (25, False)]:
        t.update(lost, now=float(now))

    assert t.episodes == 2
    assert t.lost_seconds == pytest.approx(10.0)


def test_a_takeover_still_open_at_the_end_is_counted():
    """Rover 4 was still in MANUAL when the capture kept advancing."""
    t = _tracker()
    t.update(False, now=0.0)
    t.update(True, now=10.0)
    t.finish(now=45.0)

    assert t.episodes == 1
    assert t.lost_seconds == pytest.approx(35.0)


def test_both_edges_are_reported_so_a_recovery_is_visible():
    """"Degraded and never came back" and "degraded for 4s" want different responses."""
    from spf.mavlink_radio_collection import LOST, RECOVERED

    t = _tracker()
    assert t.update(False, now=0.0) == ""
    assert t.update(True, now=5.0) == LOST
    assert t.update(True, now=6.0) == ""
    assert t.update(False, now=9.0) == RECOVERED


def test_the_capture_records_how_much_of_it_had_no_planner_control():
    """The number has to reach the artifact, or nobody will ever see it."""
    import inspect

    from spf.data_collector import DataCollector

    source = inspect.getsource(DataCollector._mark_capture_state)
    assert "planner_control_lost_seconds" in source, (
        "the zarr must carry how long the planner was not driving"
    )
    assert "navigation_unhealthy_seconds" in source, (
        "the zarr must carry how long the vehicle did not know where it was"
    )


# ------------------------------------- control is a live signal, not a latch ---
#
# The two tests above only exercise the ACCOUNTING. They pass even if the
# signal being accounted never moves -- which is exactly what shipped:
# `planner_in_control` is set by run_planner before its first move and cleared
# only on return or MOVE_ABORTED. An operator taking MANUAL does not touch it,
# so on 2026-08-07 rover 4 recorded a stationary vehicle for 258 records and
# the takeover counter read zero.


def _live_drone(**overrides):
    """A Drone with only the fields planner_control_loss_reason reads."""
    drone = Drone.__new__(Drone)
    drone.planner_in_control = True
    drone.ignore_mode = False
    drone.mav_mode = "ROVER_MODE_GUIDED"
    drone.connection_factory = None
    drone.ekf_healthy = True
    drone.sensors_enabled = [
        "MAV_SYS_STATUS_SENSOR_GPS",
        "MAV_SYS_STATUS_SENSOR_3D_MAG",
    ]
    drone.sensors_health = list(drone.sensors_enabled)
    for key, value in overrides.items():
        setattr(drone, key, value)
    return drone


def test_a_guided_vehicle_under_the_planner_is_in_control():
    assert _live_drone().is_planner_in_control() is True


def test_an_operator_taking_manual_is_control_lost_immediately():
    """RED: this is the rover 4 case, and it used to report in-control."""
    drone = _live_drone(mav_mode="ROVER_MODE_MANUAL")

    assert drone.is_planner_in_control() is False
    assert "MANUAL" in drone.planner_control_loss_reason()


@pytest.mark.parametrize(
    "mode", ["ROVER_MODE_HOLD", "ROVER_MODE_RTL", "ROVER_MODE_AUTO", None]
)
def test_any_mode_that_is_not_guided_is_control_lost(mode):
    """HOLD is what an EKF or RC failsafe drops the rover into; None is a dead link."""
    assert _live_drone(mav_mode=mode).is_planner_in_control() is False


def test_control_returns_by_itself_when_the_mode_comes_back():
    """Resume must need no unwinding -- the signal is read, not latched."""
    drone = _live_drone(mav_mode="ROVER_MODE_MANUAL")
    assert drone.is_planner_in_control() is False

    drone.mav_mode = "ROVER_MODE_GUIDED"  # operator flips CH8 back
    assert drone.is_planner_in_control() is True


def test_ignore_mode_does_not_watch_the_mode_at_all():
    """--ignore-mode drives without an operator; there is no handback to wait for."""
    drone = _live_drone(ignore_mode=True, mav_mode="ROVER_MODE_MANUAL")

    assert drone.is_planner_in_control() is True


def test_control_is_lost_before_the_planner_has_started():
    assert _live_drone(planner_in_control=False).is_planner_in_control() is False


def test_the_latch_is_still_available_for_waiting_out_the_planner_thread():
    """Post-capture parking waits for run_planner to EXIT, not for GUIDED.

    Using the live signal there would end the wait the instant an operator took
    MANUAL, and the parking commands would then race a planner still issuing
    repositions.
    """
    drone = _live_drone(mav_mode="ROVER_MODE_MANUAL")

    assert drone.is_planner_in_control() is False
    assert drone.planner_is_still_driving() is True

    drone.planner_in_control = False  # run_planner returned
    assert drone.planner_is_still_driving() is False


# ----------------------------------------- navigation health, counted apart ---


def test_a_healthy_vehicle_reports_no_navigation_warning():
    assert _live_drone().navigation_health_warning() is None


@pytest.mark.parametrize(
    "failed", ["MAV_SYS_STATUS_SENSOR_GPS", "MAV_SYS_STATUS_SENSOR_3D_MAG"]
)
def test_an_enabled_navigation_sensor_reporting_unhealthy_is_a_warning(failed):
    """GPS loss and compass loss: gps_lat/long/heading ARE the ground truth."""
    drone = _live_drone()
    drone.sensors_health = [s for s in drone.sensors_enabled if s != failed]

    warning = drone.navigation_health_warning()
    assert warning is not None and failed in warning


def test_an_unhealthy_compass_does_not_by_itself_mean_control_was_lost():
    """The two facts are counted separately; conflating them helps neither."""
    drone = _live_drone()
    drone.sensors_health = ["MAV_SYS_STATUS_SENSOR_GPS"]

    assert drone.is_planner_in_control() is True
    assert drone.navigation_health_warning() is not None


def test_a_sensor_the_airframe_does_not_have_is_not_a_fault():
    """Absent != broken. Treating it as broken marks every capture bad forever."""
    drone = _live_drone()
    drone.sensors_enabled = ["MAV_SYS_STATUS_SENSOR_GPS"]
    drone.sensors_health = ["MAV_SYS_STATUS_SENSOR_GPS"]

    assert drone.navigation_health_warning() is None


def test_an_unhappy_ekf_is_a_navigation_warning():
    assert _live_drone(ekf_healthy=False).navigation_health_warning() is not None


def test_navigation_health_recovers_when_the_sensor_does():
    drone = _live_drone()
    drone.sensors_health = ["MAV_SYS_STATUS_SENSOR_GPS"]
    assert drone.navigation_health_warning() is not None

    drone.sensors_health = list(drone.sensors_enabled)
    assert drone.navigation_health_warning() is None
