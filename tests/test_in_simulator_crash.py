"""Stall detection and recovery against a real ArduPilot vehicle in SITL.

Green as of 2026-08-04 (10/10, ~6m40s). Getting there turned up three product
bugs, none of which produced a single wrong log line -- the Pi logged correct
intent every time and the vehicle ignored it. That is what these tests are for,
and why they assert on VEHICLE TELEMETRY (servo PWM, mode timeline) rather than
on the collector's own output:

  * escape legs were 3 m against WP_RADIUS 5.0, so ArduPilot counted every
    escape target as already reached and never drove the leg;
  * the escape set HOLD to settle the speed integrator and never restored
    GUIDED, so it disabled itself after one attempt and parked the rover;
  * the escape-attempt cap could be cleared by the escape's own motion, so a
    creeping rover escaped forever and never reached the operator.

Three things had to be right before any of this could detect a stall at all,
and each was found the hard way. They are documented at their point of use, but
collected here because every one of them fails SILENTLY -- you get a motionless
rover and no detection, which reads like a broken watchdog and is not one:

  * the jam must clamp SERVOn_MIN/MAX, not MOT_THR_MAX (see `jam()`);
  * the sim must run at -S 1, not -S 5 (see SIMULATOR_SPEEDUP);
  * params must be set over MAVLink, not via `--load-params` -- that path
    takes ~25 s because it verifies by downloading all 1281 params, long
    enough for the rover to reach its waypoint and park before the jam lands.

Deliberately a separate file with its own container from test_in_simulator.py.
These cases are slow and opt-in (`--sitl-crash`), and isolating them means they
cannot destabilise the seven simulator tests that b80de20 just repaired.
Running two sims side by side is safe precisely because that commit floated the
host ports.

Every assertion that matters is made against VEHICLE TELEMETRY via SitlObserver
rather than the collector's stdout: a reverse that is logged but never reaches
the motors is exactly the failure worth catching, and it is also how
MAV_CMD_DO_SET_REVERSE gets validated on the firmware in the image.

The jam is injected by clamping the throttle channels' SERVO output range (see
`jam()` for why, and for the measurement that ruled out the obvious
alternative). Those params are in NEITHER enforced params file, so every test
restores them in a finally; a leaked clamp would silently poison every later
case.
"""

from __future__ import annotations

import glob
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import docker
import pytest

import spf.mavlink.mavlink_controller
from spf.mavlink import mavlink_controller
from spf import mavlink_radio_collection
from tests.helpers.sitl_observer import SitlObserver

root_dir = os.path.dirname(os.path.dirname(spf.__file__))

# Real time, not sim time: the stall clock is time.monotonic() on the HOST, so
# -S speedup does not shorten it. Short thresholds keep each case ~15-30 s.
#
# The RADIUS is scaled with the time, not left at the production 3 m. What the
# detector really tests is a speed floor -- production's 3 m / 10 s is 0.3 m/s.
# Compressing only the time to 3 s while keeping 3 m would demand 1.0 m/s, over
# three times stricter, and a rover slowing through a bounce turn then trips a
# stall it never would in the field. That is a false positive invented by the
# test, and it is what test_healthy_driving_is_never_called_a_stall caught.
STALL_DETECT_SECONDS = 3.0
STALL_MANUAL_SECONDS = 12.0
STALL_PROGRESS_RADIUS_M = 1.0  # 1.0 m / 3 s = 0.33 m/s, matching production

# The crash suite must run the simulator in REAL TIME, unlike
# test_in_simulator.py which uses -S 5.
#
# The detector compares host wall-clock elapsed time against distance the
# VEHICLE covered in sim time. Any speedup multiplies the ground covered per
# wall-second and defeats the displacement threshold: measured at -S 5, a
# clamped rover crawling at 0.26 m/s of sim speed still covered 15.8 m per 12 s
# of wall time -- five times the 3 m progress radius, so the anchor kept
# resetting and no stall was ever detected. At -S 1 the same clamp covers
# 0.8 m in the detect window, which is the field behaviour these tests exist to
# reproduce.
SIMULATOR_SPEEDUP = 1

pytestmark = pytest.mark.sitl_crash


@dataclass(frozen=True)
class CrashSimEndpoints:
    """Host ports for the sim's three MAVLink endpoints, keyed by ROLE.

    Same reasoning as SimEndpoints in test_in_simulator.py -- Docker neither
    preserves order nor contiguity, so a positional tuple could silently swap
    two of these and the symptom would be a timeout rather than an error. The
    third endpoint is what the observer attaches to; each `tcpin` accepts a
    single client, so it cannot share the collector's or the commander's.
    """

    collect: int  # container 14590 -- the collector under test
    command: int  # container 14591 -- mode changes and parameter loads
    observe: int  # container 14592 -- read-only telemetry for assertions


@pytest.fixture(scope="session")
def crash_simulator():
    client = docker.from_env()
    container = client.containers.run(
        "csmisko/ardupilotspf:latest",
        "/ardupilot/Tools/autotest/sim_vehicle.py -l 37.76509485,-122.40940127,0,0 "
        "-v rover -f rover-skid "
        "--out tcpin:0.0.0.0:14590 --out tcpin:0.0.0.0:14591 "
        f"--out tcpin:0.0.0.0:14592 -S {SIMULATOR_SPEEDUP}",
        stdin_open=True,
        # Host side floats (None); container side stays fixed because
        # sim_vehicle.py bakes those numbers into the --out arguments.
        ports={
            "14590/tcp": ("127.0.0.1", None),
            "14591/tcp": ("127.0.0.1", None),
            "14592/tcp": ("127.0.0.1", None),
        },
        detach=True,
        remove=True,
        auto_remove=True,
    )
    try:
        container.reload()

        def host_port(container_port: int) -> int:
            bindings = container.ports.get(f"{container_port}/tcp")
            if not bindings:
                raise RuntimeError(
                    f"docker published no host port for {container_port}/tcp; "
                    f"got {container.ports!r}"
                )
            return int(bindings[0]["HostPort"])

        endpoints = CrashSimEndpoints(
            collect=host_port(14590),
            command=host_port(14591),
            observe=host_port(14592),
        )

        online = False
        for line in container.attach(stdout=True, stream=True, logs=True):
            if "Detected vehicle" in line.decode():
                online = True
                break
        if not online:
            raise RuntimeError("simulator never reported a detected vehicle")

        yield endpoints
    finally:
        container.stop()


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def controller(port):
    return (
        f"python3 {spf.mavlink.mavlink_controller.__file__} "
        f"--ip 127.0.0.1 --port {port} --proto tcp"
    )


def set_mode(mode, port):
    subprocess.check_output(
        f"{controller(port)} --mode {mode}", timeout=30, shell=True, env=get_env()
    )


# Clamp the SERVO OUTPUT RANGE, not MOT_THR_MAX.
#
# Collapsing MOT_THR_MAX was the obvious choice and it does not work: measured
# in this simulator, MOT_THR_MAX<=5 does stop the rover (0.03 m/s) but drives
# both throttle channels to exactly 1500 for ~90% of samples. motor_active then
# flickers false, the detector correctly stands down, and nothing is under test.
#
# SimRover converts PWM with a FIXED mapping, `2*((servo-1000)/1000 - 0.5)`,
# independent of SERVOn_MIN/MAX. So narrowing SERVOn_MIN/MAX makes ArduPilot
# emit a saturated-but-narrow PWM -- pegged at the clamp, hence clearly off
# neutral and motor_active true -- while the physics sees only a couple of
# percent throttle. That is the field signature: commanded hard, going nowhere.
#
# The clamp is ASYMMETRIC, and it has to be. Two requirements pull opposite
# ways and a symmetric clamp cannot satisfy both:
#
#   forward must be throttled HARD -- the rover has to cover less than
#     STALL_PROGRESS_RADIUS_M in STALL_DETECT_SECONDS, i.e. stay under a
#     0.33 m/s floor, or it is simply driving and no stall exists to detect;
#   reverse needs HEADROOM -- the escape leg's output must land more than the
#     observer's 25 us reverse margin below neutral, or a genuinely reversing
#     rover reads as neutral.
#
# Measured, symmetric, both wrong:
#   1480/1520 -> 0.26 m/s, stalls fine, but only 20 us of reverse (undetectable)
#   1450/1550 -> 0.41 m/s, reverse detectable, but 1.2 m > 1.0 m so NOT stalled,
#                and the rover then oscillates stalled/progressing, resetting
#                the anchor and never escalating to MANUAL
#
# So: cap forward tightly, open reverse wide.
JAM_PWM_LOW = 1440  # 60 us of reverse headroom -- well past the 25 us margin
JAM_PWM_HIGH = 1510  # +10 us ~= 2% throttle ~= 0.13 m/s, far under the floor
# SITL defaults; the sim never loads rover3_rc_servo_parameters.params (800/2200).
FREE_PWM_LOW = 1000
FREE_PWM_HIGH = 2000

# Skid-steer throttle channels: SERVO1_FUNCTION 74, SERVO3_FUNCTION 73.
THROTTLE_CHANNELS = (1, 3)


def _servo_range(low, high):
    return {
        f"SERVO{channel}_{bound}": value
        for channel in THROTTLE_CHANNELS
        for bound, value in (("MIN", low), ("MAX", high))
    }


def jam(vehicle):
    """Commanded hard, going nowhere -- see the note above.

    Goes through the vehicle handle, not `--load-params`. That path took ~25 s
    (it verifies by downloading all 1281 params), and the rover kept driving
    throughout -- so the jam landed after it had already reached its waypoint
    and parked, and the measurement saw zero throttle for an innocent reason.
    """
    vehicle.params(**_servo_range(JAM_PWM_LOW, JAM_PWM_HIGH))


def unjam(vehicle):
    vehicle.params(**_servo_range(FREE_PWM_LOW, FREE_PWM_HIGH))


def collector_command(
    collect_port,
    tmpdir,
    *,
    crash_detect=True,
    crash_recovery=False,
    detect_seconds=None,
    manual_seconds=None,
    radius_m=None,
):
    # --drone-uri is explicit: tests/rover_config.yaml hardcodes
    # tcp:127.0.0.1:14591, which on the CI box is the developer's own sim.
    return (
        f"python3 {mavlink_radio_collection.__file__} "
        f"-c {root_dir}/tests/rover_config.yaml "
        f"-m {root_dir}/tests/device_mapping "
        f"--drone-uri tcp:127.0.0.1:{collect_port} "
        f"-r bounce --temp {tmpdir} "
        f"{'--crash-detect' if crash_detect else '--no-crash-detect'} "
        f"{'--crash-recovery' if crash_recovery else '--no-crash-recovery'} "
        f"--stall-detect-seconds {detect_seconds or STALL_DETECT_SECONDS} "
        f"--stall-manual-seconds {manual_seconds or STALL_MANUAL_SECONDS} "
        f"--stall-progress-radius-m {radius_m or STALL_PROGRESS_RADIUS_M}"
    )


class Capture:
    """A running collector, its stdout, and the vehicle observer beside it."""

    def __init__(self, endpoints, tmpdir, **flags):
        self.endpoints = endpoints
        self.detect_seconds = flags.get("detect_seconds") or STALL_DETECT_SECONDS
        self.manual_seconds = flags.get("manual_seconds") or STALL_MANUAL_SECONDS
        self.command = collector_command(endpoints.collect, tmpdir, **flags)
        self.lines = []

    def __enter__(self):
        self.observer = SitlObserver(
            f"tcp:127.0.0.1:{self.endpoints.observe}"
        ).start()
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            env=get_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Own process group, so teardown can kill the COLLECTOR and not
            # just the shell wrapping it. Each `tcpin` accepts exactly one
            # client: an orphaned collector keeps the collect endpoint claimed
            # and the next test in the session dies with "no heartbeat within
            # 10.0s" -- which looks like a simulator fault and is not one.
            start_new_session=True,
        )
        return self

    def __exit__(self, *_exc):
        self._terminate_group()
        leaked = self.observer.restore_params()
        self.observer.stop()
        self._wait_for_endpoint_release()
        assert not leaked, f"could not restore vehicle params: {leaked}"

    # Each `tcpin` serves ONE client, so the sim needs a moment to notice the
    # old collector is gone before the next test's collector can attach --
    # otherwise that one connects, gets no heartbeat, and dies with "MAVLink
    # unavailable", which looks like a broken simulator rather than a test that
    # did not clean up after itself.
    #
    # Deliberately a sleep rather than a probe: any probe would have to CONNECT
    # to the endpoint to learn anything, which claims the single slot it is
    # checking is free. The cure would cause the disease.
    ENDPOINT_RELEASE_SECONDS = 5.0

    def _wait_for_endpoint_release(self):
        time.sleep(self.ENDPOINT_RELEASE_SECONDS)

    def _terminate_group(self):
        try:
            group = os.getpgid(self.process.pid)
        except ProcessLookupError:
            return
        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, signal_number)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                continue

    def wait_for_line(self, needle, timeout=120):
        """Read collector stdout until `needle` appears. Returns False on EOF."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                return False
            self.lines.append(line)
            if needle in line:
                return True
        return False

    @property
    def output(self):
        return "".join(self.lines)

    def drain(self, seconds):
        """Keep reading for a while so the pipe cannot fill and block."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                return
            self.lines.append(line)

    def hand_to_guided(self):
        set_mode("guided", self.endpoints.command)

    def wait_for_guided(self, timeout=25.0):
        """Get the rover back under autonomous control, whatever it is doing.

        Covers both idle-but-innocent states: parked in MANUAL after a handover
        (the operator would flip CH8 back, so do that and wait for the collector
        to acknowledge) and sitting in the HOLD that _escape_jam uses between
        legs (which the rover leaves by itself once the next target lands).
        """
        deadline = time.monotonic() + timeout
        handed_back = False
        while time.monotonic() < deadline:
            current = (
                self.observer.modes[-1].mode if self.observer.modes else None
            )
            if current == "GUIDED":
                return True
            if current == "MANUAL" and not handed_back:
                self.hand_to_guided()
                handed_back = True
                if not self.wait_for_line("operator returned control", timeout=20):
                    return False
                continue
            self.drain(1)
        return False


def start_driving(capture):
    """Drive the collector through the operator handshake into GUIDED."""
    assert capture.wait_for_line(
        "waiting for rover to move into guided mode"
    ), capture.output
    capture.hand_to_guided()
    assert capture.wait_for_line(
        "Planner starting to issue move commands"
    ), capture.output


# ------------------------------------------------------------------ cases ---


def test_the_jam_injection_produces_a_real_stall_signature(crash_simulator):
    """Precondition for every other case in this file.

    The detector requires throttle to be COMMANDED (servo1/servo3 off neutral,
    i.e. motor_active) while the rover covers no ground. Both must hold at once
    and for the whole detect window. The first injection tried here satisfied
    only the second -- it stopped the rover but let the outputs fall to exactly
    1500 -- so motor_active flickered false, the detector correctly stood down,
    and every case below would have failed for an unexplained reason. Assert
    the injection reproduces the real signature before trusting anything else
    in this file.
    """
    def commanded_fraction(capture, since):
        samples = [s for s in capture.observer.servos if s.at >= since]
        assert samples, "no SERVO_OUTPUT_RAW in the measurement window"
        off = sum(1 for s in samples if s.servo1 != 1500 or s.servo3 != 1500)
        return off / len(samples), samples

    # crash_detect OFF on purpose: this test characterises the INJECTION, so
    # nothing may intervene. With the watchdog on it fires (correctly) part way
    # through, parks the rover in MANUAL, and the servos then read neutral
    # because the sticks are centred -- which looks exactly like a failed
    # injection and is in fact the feature working.
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_detect=False) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                # Baseline FIRST. If throttle is already neutral before the jam
                # then the rover simply is not driving, and any conclusion about
                # the injection would be wrong.
                assert capture.observer.wait_for_servos(), "no SERVO_OUTPUT_RAW"
                baseline_mark = time.monotonic()
                capture.drain(6)
                baseline, baseline_samples = commanded_fraction(
                    capture, baseline_mark
                )
                baseline_moved = capture.observer.displacement_m(since=baseline_mark)
                assert baseline > 0.5, (
                    "the rover is not driving BEFORE the jam is applied, so this "
                    "test cannot say anything about the injection. mode timeline="
                    f"{capture.observer.mode_timeline()}, moved={baseline_moved:.1f}m, "
                    f"samples={baseline_samples[-5:]}"
                )

                jam(capture.observer)
                capture.drain(4)  # shed momentum from before the clamp
                # Measure over EXACTLY the detect window. Draining longer and
                # then comparing the total against the radius is
                # apples-to-oranges: at 0.15 m/s the rover covers 1.05 m in 7 s
                # but only 0.45 m in the 3 s the detector actually looks at, so
                # a perfectly good jam reads as "progress, not a stall".
                mark = time.monotonic()
                capture.drain(STALL_DETECT_SECONDS)
                commanded, samples = commanded_fraction(capture, mark)
                moved = capture.observer.displacement_m(since=mark)

                assert commanded > 0.5, (
                    "throttle fell to neutral during the jam, so motor_active "
                    "flickers false and the detector (rightly) stands down -- "
                    f"this injection cannot test a stall. commanded={commanded:.0%} "
                    f"(baseline {baseline:.0%}), moved={moved:.1f}m, "
                    f"mode={capture.observer.mode_timeline()}, last={samples[-5:]}"
                )
                # Against the radius this run CONFIGURED, not the production
                # constant: a hardcoded 3.0 here silently passed a jam that the
                # detector, running at a 1 m radius, would rightly call motion.
                assert moved <= STALL_PROGRESS_RADIUS_M, (
                    f"rover covered {moved:.1f} m during the jam, over the "
                    f"{STALL_PROGRESS_RADIUS_M} m radius this run uses; that is "
                    "progress, not a stall"
                )
            finally:
                unjam(capture.observer)


def test_healthy_driving_is_never_called_a_stall(crash_simulator):
    """The false-positive guard, and the most important test in this file.

    A watchdog that yanks a working rover into MANUAL mid-capture is worse than
    no watchdog. Drive normally well past the escalation deadline and assert
    nothing fires.

    Uses PRODUCTION thresholds, unlike every other case here, and that is the
    point: this test exists to validate the production margin, and a compressed
    threshold cannot do that. Acceleration and turn dynamics do not scale with
    the window -- after arriving at a bounce point the next waypoint lies on a
    new bearing, and with WP_PIVOT_ANGLE 0 and TURN_RADIUS 5.0 the rover must
    arc from a standstill, covering almost no ground for seconds. At 1 m / 3 s
    that reads as a stall; at the real 3 m / 10 s it has ample room. Scaling the
    radius with the time is not sufficient, which is why this one case pays the
    ~50 s.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(
            crash_simulator,
            tmpdir,
            detect_seconds=mavlink_controller.STALL_DETECT_SECONDS,
            manual_seconds=mavlink_controller.STALL_MANUAL_SECONDS,
            radius_m=mavlink_controller.STALL_PROGRESS_RADIUS_M,
        ) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            started = time.monotonic()
            capture.drain(capture.manual_seconds * 1.5)

            assert "STALL" not in capture.output, capture.output
            assert capture.observer.entered("MANUAL", since=started) is None
            assert not capture.observer.saw_reverse(since=started)


def test_detect_only_hands_over_without_ever_reversing(crash_simulator):
    """crash_detect on, crash_recovery off: MANUAL, and nothing autonomous."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=False) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                jammed_at = time.monotonic()
                assert capture.wait_for_line("handing control to the operator"), (
                    capture.output
                )
                manual_at = capture.observer.wait_for_mode(
                    "MANUAL", timeout=20, since=jammed_at
                )
                assert manual_at is not None, capture.observer.mode_timeline()
                assert not capture.observer.saw_reverse(since=jammed_at), (
                    "recovery is off; the rover must not drive itself"
                )
            finally:
                unjam(capture.observer)


def test_recovery_reverses_then_escalates_when_still_jammed(crash_simulator):
    """The full escalation, observed at the motors rather than in the log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=True) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                jammed_at = time.monotonic()

                assert capture.wait_for_line("reversing out"), capture.output
                assert capture.wait_for_line("handing control to the operator"), (
                    capture.output
                )
                assert capture.observer.wait_for_mode(
                    "MANUAL", timeout=20, since=jammed_at
                ), capture.observer.mode_timeline()

                # At least one full attempt -- reverse leg AND lateral leg --
                # before the hand-over.
                #
                # Deliberately not "at least two". The compressed thresholds
                # here leave room for exactly one: detect at 3 s plus a ~9 s
                # maneuver (4 s reverse + 1 s settle + 4 s lateral) reaches the
                # 12 s manual deadline, so the second attempt never starts.
                # Production has room for three (10 s detect, ~9 s maneuver,
                # 40 s deadline) and that count is pinned by
                # test_recovery_on_makes_three_attempts_then_hands_over, where a
                # fake clock makes the arithmetic exact. Asserting a count here
                # only re-tests the thresholds this file happens to pick.
                assert capture.output.count("reversing out") >= 1, capture.output
                assert "stepping" in capture.output, capture.output
                # HOLD is the inter-leg settle that lets the firmware reset its
                # wound-up speed integrator.
                assert "HOLD" in capture.observer.mode_timeline()
            finally:
                unjam(capture.observer)


def test_reverse_actually_reaches_the_motors(crash_simulator):
    """Validates MAV_CMD_DO_SET_REVERSE on this firmware, not on the docs.

    Rover-4.5.7 source routes it with no mode check, so it should apply in
    GUIDED -- but the fleet runs 4.5.0 and users have reported reverse failing
    there. Both throttle channels below neutral is the only proof that counts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=True) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                jammed_at = time.monotonic()
                assert capture.wait_for_line("reversing out"), capture.output
                capture.drain(6)
                assert capture.observer.saw_reverse(since=jammed_at), (
                    "DO_SET_REVERSE did not produce reverse PWM on this firmware; "
                    f"servo samples={capture.observer.servos[-20:]}"
                )
            finally:
                unjam(capture.observer)


def test_freeing_the_rover_cancels_the_escalation(crash_simulator):
    """Motion resets the clock, so a rover that frees itself never sees MANUAL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=True) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            jam(capture.observer)
            jammed_at = time.monotonic()
            assert capture.wait_for_line("reversing out"), capture.output

            # Restore authority mid-maneuver: the rover drives out of the jam.
            unjam(capture.observer)
            capture.drain(STALL_MANUAL_SECONDS * 2)

            assert capture.observer.entered("MANUAL", since=jammed_at) is None, (
                "the rover moved again; it must not have been handed over"
            )


def test_the_reverse_flag_is_never_left_set(crash_simulator):
    """A leaked DO_SET_REVERSE would drive the whole next leg backwards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=True) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            jam(capture.observer)
            assert capture.wait_for_line("reversing out"), capture.output

            unjam(capture.observer)
            # The rover must be back under GUIDED and moving before "did it go
            # forward?" means anything. Two ways it can be idle and innocent:
            # parked in MANUAL after a handover (sticks centred), or sitting in
            # the HOLD that _escape_jam uses as its inter-leg settle. Either
            # reads as zero throttle, and the assertion below would blame a
            # leaked reverse flag for a rover that was simply not being driven.
            capture.drain(4)
            if not capture.wait_for_guided(timeout=25):
                pytest.fail(
                    "rover never returned to GUIDED, so forward motion cannot "
                    f"be judged; modes={capture.observer.mode_timeline()}"
                )
            resumed_at = time.monotonic()
            capture.drain(12)

            assert capture.observer.saw_forward(since=resumed_at), (
                "after recovery the rover must drive forward again, not backwards; "
                f"modes={capture.observer.mode_timeline()}"
            )


def test_crash_detect_off_leaves_the_rover_stuck(crash_simulator):
    """Proves the flag really disables the feature rather than merely muting it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_detect=False) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                jammed_at = time.monotonic()
                capture.drain(STALL_MANUAL_SECONDS * 2)

                assert "STALL" not in capture.output, capture.output
                assert capture.observer.entered("MANUAL", since=jammed_at) is None
                assert not capture.observer.saw_reverse(since=jammed_at)
            finally:
                unjam(capture.observer)


def test_capture_resumes_after_the_operator_hands_control_back(crash_simulator):
    """The handshake is the whole point: MANUAL is a pause, not an abort."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=False) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                assert capture.wait_for_line(
                    "waiting for the operator to hand control back"
                ), capture.output

                unjam(capture.observer)
                capture.hand_to_guided()

                assert capture.wait_for_line("operator returned control"), (
                    capture.output
                )
                assert capture.wait_for_line("Dist (m) to target"), capture.output
            finally:
                unjam(capture.observer)


def test_capture_still_writes_a_dataset_through_a_stall(crash_simulator):
    """A stall must not cost the recording."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Capture(crash_simulator, tmpdir, crash_recovery=True) as capture:
            set_mode("manual", crash_simulator.command)
            start_driving(capture)
            try:
                jam(capture.observer)
                assert capture.wait_for_line("reversing out"), capture.output
                unjam(capture.observer)
                capture.drain(10)
            finally:
                unjam(capture.observer)

        assert glob.glob(f"{tmpdir}/*.zarr") or glob.glob(f"{tmpdir}/*.zarr.tmp")
