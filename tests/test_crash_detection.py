"""Stall detection, escape maneuvers, and the MANUAL handback.

No hardware and no simulator: every test drives the real `move_to_point` loop
on a hand-built Drone with a scripted GPS track and a fake clock, following the
`Drone.__new__` idiom already used by tests/test_mavlink_rc_safety.py and
tests/test_capture_failure_coordination.py.

The invariant under test is deliberately DISPLACEMENT from an anchor, not
distance-to-target: with TURN_RADIUS 5.0 a healthy rover can arc for twenty
seconds without closing on its waypoint, and calling that a stall would yank a
working rover into MANUAL mid-capture. test_no_stall_while_arcing_away_from_the
_target is the guard for exactly that.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from spf.mavlink.mavlink_controller import (
    MOVE_ABORTED,
    MOVE_REACHED,
    MOVE_SKIPPED,
    STALL_ESCAPE,
    STALL_MANUAL,
    STALL_OK,
    STALL_PROGRESS_RADIUS_M,
    Drone,
    bearing_to_unit_vector,
    degrees_to_meters,
    meters_to_degrees,
)
from spf.mavlink import mavlink_controller

SITE_LAT = 37.765
SITE_LONG = -122.409
# Escape-leg length, read from the source rather than hardcoded: a literal
# 3.0 here is exactly what let the leg-inside-WP_RADIUS bug pass unnoticed.
LEG_M = mavlink_controller.STALL_ESCAPE_DISTANCE_M


class FakeClock:
    """Monotonic time the test advances explicitly.

    Every sleep in the code under test advances the clock instead of blocking,
    so a forty-second escalation runs in microseconds and the timings asserted
    are exact rather than approximate.
    """

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _at(east_m=0.0, north_m=0.0):
    """A (long, lat) point offset from the site origin by metres."""
    return np.asarray([SITE_LONG, SITE_LAT]) + meters_to_degrees(
        east_m, north_m, SITE_LAT
    )


def _drone(monkeypatch, *, crash_detect=True, crash_recovery=False, **overrides):
    clock = FakeClock()
    monkeypatch.setattr(mavlink_controller.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(mavlink_controller.time, "sleep", clock.sleep)

    drone = Drone.__new__(Drone)
    drone.gps = _at()
    drone.heading = 0.0  # pointing due North
    drone.tolerance_in_m = 1.0
    drone.distance_finder = None
    drone.disable_distance_finder = True
    drone.motor_active = True
    drone.mav_mode = "ROVER_MODE_GUIDED"
    drone.armed = True
    drone.connection_failure = None
    drone.planner_should_move = True
    drone.crash_detect = crash_detect
    drone.crash_recovery = crash_recovery
    drone.stall_detect_seconds = 10.0
    drone.stall_manual_seconds = 40.0

    drone.reposition = _Recorder(drone)
    drone.reverse = _Recorder(drone)
    drone.set_mode = _Recorder(drone)
    drone.arm = _Recorder(drone)
    drone.buzzer = _Recorder(drone)
    drone.status_text = _Recorder(drone)
    drone._initialize_stall_state()
    drone._initialize_motion_stop_state()
    for name, value in overrides.items():
        setattr(drone, name, value)
    drone.clock = clock
    return drone


class _Recorder:
    """Mock that also lets a test react to the call (e.g. free the rover)."""

    def __init__(self, drone):
        self.drone = drone
        self.calls = []
        self.side_effect = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            return self.side_effect(*args, **kwargs)

    @property
    def called(self):
        return bool(self.calls)


def _target(drone, east_m=1000.0):
    """Give the drone a waypoint it cannot reach until the test says so.

    Distance to the WAYPOINT is scripted (the rover is jammed, so it never
    arrives); distance to the stall ANCHOR is computed for real from the GPS
    track, because that is the quantity under test.
    """
    drone._target = _at(east_m=east_m)
    drone._target_distance = 1e6

    def distance_to_target(point):
        if point is drone._target:
            return drone._target_distance
        return float(
            np.linalg.norm(degrees_to_meters(*(drone.gps - np.asarray(point)), SITE_LAT))
        )

    drone.distance_to_target = distance_to_target
    return drone._target


def _drive(drone):
    return drone.move_to_point(drone._target)


def _run_planner(drone, max_waypoints=12):
    """Mimic run_planner: keep issuing waypoints until one sticks or aborts.

    An escape returns MOVE_SKIPPED so the planner advances, which is precisely
    why the stall clock has to live on the Drone rather than in move_to_point.
    """
    outcomes = []
    for _ in range(max_waypoints):
        outcome = drone.move_to_point(drone._target)
        outcomes.append(outcome)
        if outcome in (MOVE_ABORTED, MOVE_REACHED):
            break
        # A handover also returns MOVE_SKIPPED, and a rover still jammed after
        # the operator gives control back will correctly escalate all over
        # again -- so stop here rather than measuring the second cycle.
        if _manual_calls(drone):
            break
    return outcomes


def _manual_calls(drone):
    return [call for call in drone.set_mode.calls if call[0] == ("MANUAL",)]


# ------------------------------------------------------------- the invariant ---


def test_stall_trips_after_the_detect_threshold(monkeypatch):
    drone = _drone(monkeypatch)
    drone._reset_stall_anchor()

    assert drone._stall_verdict() == STALL_OK
    drone.clock.now += 9.9
    assert drone._stall_verdict() == STALL_OK
    drone.clock.now += 0.2
    assert drone._stall_verdict() == STALL_MANUAL


def test_no_stall_while_covering_ground(monkeypatch):
    drone = _drone(monkeypatch)
    drone._reset_stall_anchor()

    # 1.5 m/s for a minute: the anchor advances every few seconds and the clock
    # never gets near the threshold.
    for step in range(1, 41):
        drone.clock.now += 1.5
        drone.gps = _at(north_m=2.25 * step)
        assert drone._stall_verdict() == STALL_OK


def test_no_stall_while_arcing_away_from_the_target(monkeypatch):
    """Distance-to-target rises for 20 s while the rover covers ground.

    This is the case that rules out the simpler "distance to target stopped
    shrinking" invariant. WP_PIVOT_ANGLE is 0 and TURN_RADIUS is 5.0, so a
    turn is an arc, and during it the rover legitimately moves away from its
    waypoint.
    """
    drone = _drone(monkeypatch)
    drone._reset_stall_anchor()

    for step in range(1, 21):
        angle = np.radians(18 * step)
        drone.clock.now += 1.0
        drone.gps = _at(
            east_m=5.0 * np.sin(angle), north_m=5.0 * (1 - np.cos(angle))
        )
        assert drone._stall_verdict() == STALL_OK


def test_gps_jitter_under_the_radius_does_not_reset_the_clock(monkeypatch):
    drone = _drone(monkeypatch)
    drone._reset_stall_anchor()

    for step in range(20):
        drone.clock.now += 0.5
        wobble = 1.2 * np.sin(step)  # well inside STALL_PROGRESS_RADIUS_M
        drone.gps = _at(east_m=wobble, north_m=-wobble)
    assert drone._stall_verdict() == STALL_MANUAL


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("armed", False),
        ("motor_active", False),
        ("mav_mode", "ROVER_MODE_MANUAL"),
        ("mav_mode", "ROVER_MODE_HOLD"),
        ("crash_detect", False),
    ],
)
def test_never_trips_unless_actually_driving(monkeypatch, attribute, value):
    drone = _drone(monkeypatch)
    drone._reset_stall_anchor()
    setattr(drone, attribute, value)

    drone.clock.now += 600.0
    assert drone._stall_verdict() == STALL_OK


# --------------------------------------------------------------- flag matrix ---


def _script_operator(drone):
    """MANUAL sticks; the operator returns CH8 to GUIDED on the next chirp."""

    def on_set_mode(mode):
        if mode == "MANUAL":
            drone.mav_mode = "ROVER_MODE_MANUAL"
            drone._manual_at = drone.clock.now
        elif mode == "HOLD":
            # The inter-leg settle; GUIDED resumes when the next target lands.
            drone.mav_mode = "ROVER_MODE_GUIDED"

    drone.set_mode.side_effect = on_set_mode
    drone.buzzer.side_effect = lambda _tone: (
        setattr(drone, "mav_mode", "ROVER_MODE_GUIDED")
        if drone.mav_mode == "ROVER_MODE_MANUAL"
        else None
    )


def test_recovery_off_hands_over_at_the_detect_threshold(monkeypatch):
    """crash_detect on, crash_recovery off: straight to MANUAL, never reverses."""
    drone = _drone(monkeypatch, crash_recovery=False)
    _target(drone)
    _script_operator(drone)
    started = drone.clock.now

    assert _drive(drone) == MOVE_SKIPPED

    assert drone._manual_at - started == pytest.approx(10.0, abs=0.5)
    assert not drone.reverse.called, "recovery is off; nothing may reverse"
    assert len(_manual_calls(drone)) == 1


def test_recovery_on_makes_three_attempts_then_hands_over(monkeypatch):
    drone = _drone(monkeypatch, crash_recovery=True)
    _target(drone)
    _script_operator(drone)
    started = drone.clock.now

    _run_planner(drone)

    engaged = [call[0][0] for call in drone.reverse.calls]
    assert engaged.count(True) == 3, "expected exactly three escape attempts"
    assert engaged.count(False) == 3, "every attempt must clear the reverse flag"
    assert drone._manual_at - started == pytest.approx(40.0, abs=1.0)


def test_escalation_survives_an_escape_that_moves_the_rover(monkeypatch):
    """The contract the runbook states: three attempts, then the operator.

    The clock alone cannot deliver that. An escape that shifts the rover a metre
    resets the anchor and so defers the very escalation it should lead to -- a
    rover creeping a metre per attempt would escape forever and never reach a
    human. Caught in SITL, where the jammed-but-creeping rover logged
    "reversing out" indefinitely and never handed over.
    """
    drone = _drone(monkeypatch, crash_recovery=True)
    _target(drone)
    _script_operator(drone)

    # Every escape nudges the rover just past the progress radius, which is the
    # worst case for the clock: it always looks like fresh progress.
    def creep(engaged):
        if engaged:
            attempt = len([c for c in drone.reverse.calls if c[0][0]])
            drone.gps = _at(north_m=4.0 * attempt)

    drone.reverse.side_effect = creep

    _run_planner(drone)

    engaged = [call[0][0] for call in drone.reverse.calls]
    assert engaged.count(True) == mavlink_controller.STALL_MAX_ESCAPES, (
        f"expected exactly {mavlink_controller.STALL_MAX_ESCAPES} attempts before "
        f"handing over, got {engaged.count(True)}"
    )
    assert _manual_calls(drone), "the rover must reach the operator, not escape forever"


def test_driving_clear_under_its_own_power_clears_the_escape_count(monkeypatch):
    """A rover that recovers must not carry escape credits into its next stall.

    The threshold is deliberately far beyond one escape leg. Clearing on the
    progress radius instead would let each escape clear its own count, which is
    no cap at all -- and leaving it uncleared makes a recovered rover escalate
    early on an unrelated stall later.
    """
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._reset_stall_anchor()
    drone._stall_escapes = mavlink_controller.STALL_MAX_ESCAPES - 1

    # Just past the progress radius is NOT proof of recovery: an escape could
    # have produced it, so the count must survive.
    nudged = mavlink_controller.STALL_PROGRESS_RADIUS_M + 0.5
    drone.gps = _at(north_m=nudged)
    assert drone._stall_verdict() == STALL_OK
    assert drone._stall_escapes == mavlink_controller.STALL_MAX_ESCAPES - 1

    # Beyond anything an escape can shove it, so the slate is clean. Measured
    # from the anchor the call above just moved, not from the original stall.
    drone.gps = _at(north_m=nudged + mavlink_controller.STALL_RECOVERED_M + 1.0)
    assert drone._stall_verdict() == STALL_OK
    assert drone._stall_escapes == 0


def test_reaching_a_waypoint_clears_the_escape_count(monkeypatch):
    """Arriving somewhere is the one signal that the rover is genuinely moving.

    Counting since the last anchor reset instead would be circular -- the escape
    is what moves the anchor.
    """
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._stall_escapes = mavlink_controller.STALL_MAX_ESCAPES - 1
    drone._target = _at(east_m=5.0)
    drone.distance_to_target = lambda point: 0.0

    assert drone.move_to_point(drone._target) == MOVE_REACHED
    assert drone._stall_escapes == 0


def test_escapes_are_rate_limited_not_run_every_tick(monkeypatch):
    """Ten seconds between attempts, not one per 100 ms poll."""
    drone = _drone(monkeypatch, crash_recovery=True)
    _target(drone)
    _script_operator(drone)
    attempt_times = []
    drone.reverse.side_effect = lambda engaged: (
        attempt_times.append(drone.clock.now) if engaged else None
    )

    _run_planner(drone)

    gaps = np.diff(attempt_times)
    assert len(attempt_times) == 3
    assert all(gap == pytest.approx(10.0, abs=1.0) for gap in gaps), attempt_times


def test_motion_between_attempts_cancels_the_escalation(monkeypatch):
    """The single most important behaviour: moving resets the clock.

    A rover that frees itself on an escape must never reach MANUAL. This is the
    difference between a watchdog that helps and one that yanks a working rover
    away from its operator mid-capture.
    """
    drone = _drone(monkeypatch, crash_recovery=True)
    _target(drone)
    _script_operator(drone)

    def on_reverse(engaged):
        # The second attempt frees it: the rover covers ground and reaches the
        # waypoint it was driving to.
        if engaged and len([c for c in drone.reverse.calls if c[0][0]]) == 2:
            drone.gps = _at(north_m=40.0)
            drone._target_distance = 0.0

    drone.reverse.side_effect = on_reverse

    outcomes = _run_planner(drone)

    assert outcomes[-1] == MOVE_REACHED
    assert _manual_calls(drone) == [], "the rover moved; it must not reach MANUAL"


def test_crash_detect_off_leaves_the_loop_untouched(monkeypatch):
    drone = _drone(monkeypatch, crash_detect=False, crash_recovery=True)
    _target(drone)
    drone.clock.now += 600.0

    assert drone._stall_verdict() == STALL_OK
    assert not drone.reverse.called
    assert not drone.set_mode.called


# ----------------------------------------------------------------- geometry ---


def test_degrees_to_meters_round_trips():
    for east, north in [(3.0, -4.0), (0.0, 3.0), (-2.5, 0.0), (11.0, 7.0)]:
        degrees = meters_to_degrees(east, north, SITE_LAT)
        back = degrees_to_meters(degrees[0], degrees[1], SITE_LAT)
        assert back == pytest.approx([east, north], abs=1e-9)


def test_rotating_in_degree_space_would_corrupt_the_escape_leg():
    """Why degrees_to_meters exists at all.

    Longitude degrees are ~21% shorter than latitude degrees at the SPF sites,
    so rotating a (dlong, dlat) pair 90 degrees skews the bearing on an oblique
    axis and corrupts the length even on an axis-aligned one. Both errors would
    aim the lateral escape leg at the wrong place.
    """
    bearing = lambda vec: np.degrees(np.arctan2(vec[0], vec[1]))

    oblique = np.asarray([3.0, 3.0])  # 3 m East, 3 m North
    as_degrees = meters_to_degrees(*oblique, SITE_LAT)
    naive = degrees_to_meters(-as_degrees[1], as_degrees[0], SITE_LAT)
    correct = np.asarray([-oblique[1], oblique[0]])
    assert abs(bearing(naive) - bearing(correct)) > 10.0

    axis_aligned = np.asarray([3.0, 0.0])
    as_degrees = meters_to_degrees(*axis_aligned, SITE_LAT)
    naive = degrees_to_meters(-as_degrees[1], as_degrees[0], SITE_LAT)
    assert bearing(naive) == pytest.approx(0.0, abs=1e-6)  # bearing survives...
    assert np.linalg.norm(naive) > 3.7  # ...but a 3 m leg becomes 3.8 m


def test_escape_distance_exceeds_the_fleet_waypoint_radius():
    """An escape target inside WP_RADIUS is one ArduPilot is already "at".

    This bit for real: at 3 m against the fleet's WP_RADIUS 5.0, the vehicle
    judged every escape target already reached and never drove the leg -- the
    maneuver logged correctly and did nothing. Caught in SITL, where the escape
    produced small steering corrections instead of pegged reverse output.
    """
    params = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data_collection/rover/rover_v3.1/rover3_base_parameters.params"
    )
    wp_radius = None
    for line in params.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "WP_RADIUS":
            wp_radius = float(fields[1])
    assert wp_radius is not None, f"WP_RADIUS not found in {params}"
    assert mavlink_controller.STALL_ESCAPE_DISTANCE_M > wp_radius, (
        f"escape legs are {mavlink_controller.STALL_ESCAPE_DISTANCE_M} m but "
        f"WP_RADIUS is {wp_radius} m, so the vehicle counts the escape target as "
        "already reached and the maneuver never drives"
    )


def test_bearing_to_unit_vector_is_compass_convention():
    assert bearing_to_unit_vector(0) == pytest.approx([0.0, 1.0], abs=1e-9)
    assert bearing_to_unit_vector(90) == pytest.approx([1.0, 0.0], abs=1e-9)
    assert bearing_to_unit_vector(180) == pytest.approx([0.0, -1.0], abs=1e-9)


# ------------------------------------------------------------ escape legs ---


def test_reverse_leg_aims_behind_and_lateral_leg_is_orthogonal_to_it(monkeypatch):
    drone = _drone(monkeypatch, crash_recovery=True)
    drone.heading = 90.0  # pointing due East
    drone._reset_stall_anchor()
    stall_point = drone.gps

    def on_set_mode(mode):
        if mode == "HOLD":
            # The reverse leg achieved 2 m due West -- short of the progress
            # radius, but a perfectly usable axis.
            drone.gps = _at(east_m=-2.0)

    drone.set_mode.side_effect = on_set_mode

    assert drone._escape_jam() is True

    first, second = [call[1] for call in drone.reposition.calls]
    reverse_offset = degrees_to_meters(
        *(np.asarray([first["long"], first["lat"]]) - stall_point), SITE_LAT
    )
    assert reverse_offset == pytest.approx([-LEG_M, 0.0], abs=0.05)

    reverse_point = _at(east_m=-2.0)
    lateral_offset = degrees_to_meters(
        *(np.asarray([second["long"], second["lat"]]) - reverse_point), SITE_LAT
    )
    # The measured axis is due West, so orthogonal is North or South.
    assert np.linalg.norm(lateral_offset) == pytest.approx(LEG_M, abs=0.05)
    assert abs(lateral_offset[0]) < 0.05
    assert abs(lateral_offset[1]) == pytest.approx(LEG_M, abs=0.05)


def test_lateral_leg_is_skipped_when_reversing_already_freed_the_rover(monkeypatch):
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._reset_stall_anchor()
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "gps", _at(north_m=25.0)) if mode == "HOLD" else None
    )

    assert drone._escape_jam() is True
    assert len(drone.reposition.calls) == 1, "no lateral leg once the rover is free"


def test_lateral_leg_falls_back_to_heading_when_the_axis_is_undefined(monkeypatch):
    """Reverse moved the rover a few centimetres: there is no axis to rotate."""
    drone = _drone(monkeypatch, crash_recovery=True)
    drone.heading = 0.0  # North; reversing aims South
    drone._reset_stall_anchor()
    stall_point = drone.gps
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "gps", _at(north_m=-0.05)) if mode == "HOLD" else None
    )

    assert drone._escape_jam() is True

    lateral = drone.reposition.calls[1][1]
    offset = degrees_to_meters(
        *(np.asarray([lateral["long"], lateral["lat"]]) - drone.gps), SITE_LAT
    )
    # Fallback axis is the heading reciprocal (South), so orthogonal is E/W.
    assert np.linalg.norm(offset) == pytest.approx(LEG_M, abs=0.05)
    assert abs(offset[0]) == pytest.approx(LEG_M, abs=0.05)
    assert abs(offset[1]) < 0.05
    assert stall_point is not drone.gps


def test_escape_sides_alternate(monkeypatch):
    drone = _drone(monkeypatch, crash_recovery=True)
    drone.heading = 0.0
    drone._reset_stall_anchor()
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "gps", _at(north_m=-2.0)) if mode == "HOLD" else None
    )

    sides = []
    for _ in range(2):
        drone.reposition.calls.clear()
        drone._escape_jam()
        lateral = drone.reposition.calls[1][1]
        offset = degrees_to_meters(
            *(np.asarray([lateral["long"], lateral["lat"]]) - drone.gps), SITE_LAT
        )
        sides.append(np.sign(offset[0]))

    assert sides[0] == -sides[1], "a rover blocked on one side must try the other"


# -------------------------------------------------------------- safety ---


def test_reverse_is_always_cleared_even_when_the_leg_raises(monkeypatch):
    """A leaked reverse flag would drive the entire next leg backwards."""
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._reset_stall_anchor()
    drone.reposition.side_effect = _raise(RuntimeError("link died mid-leg"))

    with pytest.raises(RuntimeError):
        drone._escape_jam()

    assert [call[0][0] for call in drone.reverse.calls] == [True, False]


def test_escape_leaves_the_vehicle_in_guided_not_hold(monkeypatch):
    """The settle HOLD must be temporary, or the maneuver disables itself.

    In HOLD the vehicle ignores guided targets, so a leaked HOLD makes the
    lateral leg a no-op, makes every later reposition a no-op, and fails
    _stall_verdict's GUIDED gate -- which resets the anchor and puts MANUAL
    permanently out of reach. Seen in SITL as mode timelines ending
    [..., 'GUIDED', 'HOLD'] with the rover parked.
    """
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._reset_stall_anchor()
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "gps", _at(north_m=-2.0)) if mode == "HOLD" else None
    )

    drone._escape_jam()

    modes = [call[0][0] for call in drone.set_mode.calls]
    assert "HOLD" in modes, "the integrator settle must still happen"
    assert modes[-1] == "GUIDED", (
        f"escape left the vehicle in {modes[-1]}; guided targets are ignored "
        f"outside GUIDED, so the next leg and every later waypoint are dropped "
        f"(mode calls: {modes})"
    )


def test_settle_hold_follows_every_reverse(monkeypatch):
    """The HOLD is what makes the firmware zero its wound-up speed integrator."""
    drone = _drone(monkeypatch, crash_recovery=True)
    drone._reset_stall_anchor()
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "gps", _at(north_m=-2.0)) if mode == "HOLD" else None
    )

    drone._escape_jam()

    assert ("HOLD",) in [call[0] for call in drone.set_mode.calls]


def test_stall_recovery_never_requests_a_capture_abort(monkeypatch):
    """A stall is not a capture failure and must not send the abort HOLD."""
    drone = _drone(monkeypatch, crash_recovery=True)
    _target(drone)
    drone.set_mode.side_effect = lambda mode: (
        setattr(drone, "mav_mode", "ROVER_MODE_MANUAL")
        if mode == "MANUAL"
        else setattr(drone, "mav_mode", "ROVER_MODE_GUIDED")
    )
    drone.buzzer.side_effect = lambda _tone: setattr(
        drone, "mav_mode", "ROVER_MODE_GUIDED"
    )

    _drive(drone)

    assert not drone._motion_stop_requested.is_set()
    assert not drone._motion_hold_sent.is_set()


def test_capture_abort_during_the_manual_park_still_wins(monkeypatch):
    drone = _drone(monkeypatch, crash_recovery=False)
    _target(drone)

    def on_set_mode(mode):
        if mode == "MANUAL":
            drone.mav_mode = "ROVER_MODE_MANUAL"

    drone.set_mode.side_effect = on_set_mode
    # The capture fails while the rover is parked waiting for the operator.
    drone.buzzer.side_effect = lambda _tone: drone.request_motion_stop("capture died")

    assert _drive(drone) == MOVE_ABORTED


def test_reaching_the_target_reports_reached(monkeypatch):
    drone = _drone(monkeypatch)
    drone._target = _at(east_m=5.0)
    drone.distance_to_target = lambda point: 0.0

    assert drone.move_to_point(drone._target) == MOVE_REACHED
    assert not drone.set_mode.called


def test_arriving_resets_the_clock_but_skipping_does_not(monkeypatch):
    """The asymmetry that keeps a healthy waypoint transition from tripping.

    Observed in SITL before this: the rover reached a waypoint, decelerated to
    a stop, the planner issued the next one, and the stop-and-go transition was
    indistinguishable from a stall. Arriving has to reset the clock. Skipping
    must NOT, or every abandoned waypoint buys a fresh window and the
    escalation to MANUAL is unreachable.
    """
    drone = _drone(monkeypatch, crash_recovery=True)

    # Arrival resets.
    drone._target = _at(east_m=5.0)
    drone.distance_to_target = lambda point: 0.0
    drone._reset_stall_anchor()
    drone.clock.now += 9.0
    assert drone.move_to_point(drone._target) == MOVE_REACHED
    assert drone.stalled_seconds() == pytest.approx(0.0, abs=1e-6)

    # A skip does not: the clock keeps running toward MANUAL.
    _target(drone)
    _script_operator(drone)
    before = drone.clock.now
    assert _drive(drone) == MOVE_SKIPPED  # first escape attempt
    assert drone.stalled_seconds() > drone.clock.now - before - 1.0


def _raise(error):
    def raiser(*args, **kwargs):
        raise error

    return raiser


# ------------------------------------------------------- link loss vs stall ---
#
# Rover 4, 2026-08-05 23:10 (report E1): a 48 s "stall" fired in the same second
# as a heartbeat timeout. The rover had not stopped -- the telemetry had. When
# heartbeats stop, self.gps freezes at its last value, so the anchor distance
# stays ~0 while the clock runs, and an operator is sent after a phantom.


def test_a_dead_link_is_not_reported_as_a_stall(monkeypatch):
    import time as _time

    drone = _drone(monkeypatch)
    drone.reconnect_heartbeat_timeout = 5.0
    drone.last_heartbeat = _time.time() - 30.0  # link gone well past the timeout
    drone._reset_stall_anchor()
    drone.clock.now += 60.0  # far beyond the 10 s detect threshold

    assert drone._stall_verdict() == STALL_OK


def test_a_live_link_still_detects_a_real_stall(monkeypatch):
    """The suppression must not cost us true positives."""
    import time as _time

    drone = _drone(monkeypatch)
    drone.reconnect_heartbeat_timeout = 5.0
    drone.last_heartbeat = _time.time()  # heartbeats arriving normally
    drone._reset_stall_anchor()
    drone.clock.now += 60.0

    assert drone._stall_verdict() == STALL_MANUAL


def test_a_drone_that_never_tracks_heartbeats_still_detects_stalls(monkeypatch):
    """last_heartbeat == 0 means "not tracked", NOT "link is dead".

    Reading 0 as a dead link disables stall detection on every fake-drone and
    test path at once -- which is exactly what the first version of this
    suppression did.
    """
    drone = _drone(monkeypatch)
    assert getattr(drone, "last_heartbeat", 0) == 0
    drone._reset_stall_anchor()
    drone.clock.now += 60.0

    assert drone._stall_verdict() == STALL_MANUAL
