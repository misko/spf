"""Record what the simulated vehicle actually did, from a third MAVLink link.

The existing simulator tests assert on the collector's stdout. That proves the
Pi *believes* it acted, which is not the same claim: a reverse that is logged
but never reaches the motors looks identical. This observer watches
SERVO_OUTPUT_RAW, so `saw_reverse()` means both throttle channels genuinely
went below neutral -- which is also what validates MAV_CMD_DO_SET_REVERSE on
the firmware the rovers actually run (4.5.0; the source read during design was
4.5.7).

It is strictly a reader. It never commands the vehicle, so attaching it cannot
change the behaviour under test.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from pymavlink import mavutil

# Skid-steer throttle channels: SERVO1_FUNCTION 74 (throttle right) and
# SERVO3_FUNCTION 73 (throttle left). 1500 is neutral. Steering shows up as the
# two disagreeing, so "both below neutral" is specifically reverse and not a
# hard turn.
NEUTRAL_PWM = 1500
REVERSE_MARGIN_PWM = 25  # ignore controller dither around neutral


@dataclass
class ServoSample:
    at: float
    servo1: int
    servo3: int

    @property
    def reversing(self) -> bool:
        return (
            self.servo1 < NEUTRAL_PWM - REVERSE_MARGIN_PWM
            and self.servo3 < NEUTRAL_PWM - REVERSE_MARGIN_PWM
        )

    @property
    def forward(self) -> bool:
        return (
            self.servo1 > NEUTRAL_PWM + REVERSE_MARGIN_PWM
            and self.servo3 > NEUTRAL_PWM + REVERSE_MARGIN_PWM
        )


@dataclass
class ModeChange:
    at: float
    mode: str


@dataclass
class SitlObserver:
    """Reader AND commander over one `tcpin` endpoint of the simulator.

    Commanding lives here rather than in a `mavlink_controller.py` subprocess
    for two measured reasons. Speed: setting four params over this connection
    takes 0.07 s, where `--load-params` takes ~25 s because it downloads all
    1281 params to verify -- long enough that a jam meant to catch a driving
    rover instead lands after it has parked at its waypoint. And safety: each
    `tcpin` accepts exactly one client, so a subprocess that outlives its
    `terminate()` keeps an endpoint claimed and the next test dies with "no
    heartbeat", which reads as a simulator fault and is not one.
    """

    uri: str
    servos: list[ServoSample] = field(default_factory=list)
    modes: list[ModeChange] = field(default_factory=list)
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    statustexts: list[tuple[float, str]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    # PARAM_VALUE is consumed by the reader thread, never by a competing
    # recv_match on the caller's thread -- see _await_param.
    _params: dict = field(default_factory=dict)
    _param_event: threading.Event = field(default_factory=threading.Event)
    # Original values of everything this handle wrote, so a test can hand the
    # vehicle back as it found it. That is what makes reusing one vehicle --
    # or attaching to a running one -- safe rather than reckless.
    _original_params: dict = field(default_factory=dict)

    # Rover custom_mode values; see custom_mode_mapping in mavlink_controller.
    MODES = {0: "MANUAL", 4: "HOLD", 10: "AUTO", 11: "RTL", 15: "GUIDED"}

    # ArduPilot streams telemetry per LINK, at whatever rate that link asked
    # for. A bare mavlink_connection asks for nothing, so an observer that just
    # listens sees heartbeats and little else -- notably no SERVO_OUTPUT_RAW,
    # which is the one message this class exists to read. The collector only
    # gets it because its link requests streams.
    STREAM_RATE_HZ = 10

    def start(self):
        self.connection = mavutil.mavlink_connection(self.uri)
        self.connection.wait_heartbeat(timeout=30)
        self.connection.mav.request_data_stream_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            self.STREAM_RATE_HZ,
            1,  # start
        )
        # Belt and braces: on builds that ignore the legacy stream request, ask
        # for the two messages the assertions depend on by id.
        for message_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        ):
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                int(1e6 / self.STREAM_RATE_HZ),  # microseconds between messages
                0,
                0,
                0,
                0,
                0,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def wait_for_servos(self, timeout=15.0) -> bool:
        """Block until the vehicle is actually streaming servo output."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.servos:
                return True
            time.sleep(0.1)
        return False

    # ------------------------------------------------------- commanding ---

    def _await_param(self, name, predicate, timeout):
        """Wait for the reader thread to report a PARAM_VALUE we accept.

        The caller must NOT call recv_match itself: the reader thread already
        owns the socket, so a second consumer would race it and PARAM_VALUE
        would be swallowed at random.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self._params.get(name)
            if value is not None and predicate(value):
                return value
            self._param_event.clear()
            self._param_event.wait(timeout=0.2)
        return None

    def read_param(self, name, timeout=5.0):
        self._params.pop(name, None)
        self.connection.mav.param_request_read_send(
            self.connection.target_system,
            self.connection.target_component,
            name.encode(),
            -1,
        )
        return self._await_param(name, lambda _value: True, timeout)

    def set_param(self, name, value, timeout=5.0):
        """Set one param and confirm the vehicle echoed the new value back."""
        value = float(value)
        if name not in self._original_params:
            self._original_params[name] = self.read_param(name, timeout=timeout)
        self._params.pop(name, None)
        self.connection.mav.param_set_send(
            self.connection.target_system,
            self.connection.target_component,
            name.encode(),
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        confirmed = self._await_param(
            name, lambda actual: abs(actual - value) < 1e-6, timeout
        )
        if confirmed is None:
            raise AssertionError(
                f"vehicle did not accept {name}={value} within {timeout}s "
                f"(last seen {self._params.get(name)})"
            )
        return confirmed

    def params(self, **values):
        """Set and verify named params. ~0.07s for four, vs ~25s via a file."""
        for name, value in values.items():
            self.set_param(name, value)

    def restore_params(self):
        """Put back every param this handle changed, newest write first."""
        failures = []
        for name, original in reversed(list(self._original_params.items())):
            if original is None:
                continue
            try:
                self._original_params.pop(name, None)
                self.set_param(name, original)
            except AssertionError as error:
                failures.append(str(error))
        self._original_params.clear()
        return failures

    def speedup(self, rate):
        """Change the simulation rate at RUNTIME.

        Verified: settable in 0.04 s, and the physics rate really follows --
        5.02 sim-seconds per wall-second at 5, 1.00 at 1. So the sim rate is
        not a launch-time property, and one container can serve both a suite
        that wants to run fast and one whose thresholds are wall-clock.
        """
        return self.set_param("SIM_SPEEDUP", rate)

    def set_mode(self, mode, timeout=10.0):
        """Set a flight mode and wait until the vehicle reports it."""
        self.connection.set_mode(mode.upper())
        expected = mode.upper()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.modes and self.modes[-1].mode == expected:
                return True
            time.sleep(0.1)
        raise AssertionError(
            f"vehicle did not enter {expected} within {timeout}s; "
            f"timeline={self.mode_timeline()}"
        )

    def _run(self):
        while not self._stop.is_set():
            try:
                message = self.connection.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue
            if message is None:
                continue
            now = time.monotonic()
            kind = message.get_type()
            if kind == "SERVO_OUTPUT_RAW":
                self.servos.append(
                    ServoSample(now, message.servo1_raw, message.servo3_raw)
                )
            elif kind == "HEARTBEAT":
                mode = self.MODES.get(message.custom_mode, str(message.custom_mode))
                if not self.modes or self.modes[-1].mode != mode:
                    self.modes.append(ModeChange(now, mode))
            elif kind == "GLOBAL_POSITION_INT":
                self.positions.append((now, message.lat / 1e7, message.lon / 1e7))
            elif kind == "STATUSTEXT":
                text = message.text
                if isinstance(text, bytes):
                    text = text.decode(errors="replace")
                self.statustexts.append((now, text))
            elif kind == "PARAM_VALUE":
                name = message.param_id
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                self._params[name.rstrip("\x00")] = message.param_value
                self._param_event.set()

    def stop(self):
        self._stop.set()
        if getattr(self, "_thread", None) is not None:
            self._thread.join(timeout=2.0)
        try:
            self.connection.close()
        except Exception:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.stop()

    # ---------------------------------------------------------- queries ---

    def saw_reverse(self, since: float = 0.0) -> bool:
        """Did BOTH throttle channels go below neutral together?"""
        return any(s.reversing for s in self.servos if s.at >= since)

    def saw_forward(self, since: float = 0.0) -> bool:
        return any(s.forward for s in self.servos if s.at >= since)

    def mode_timeline(self) -> list[str]:
        return [change.mode for change in self.modes]

    def entered(self, mode: str, since: float = 0.0) -> float | None:
        """When the vehicle first entered `mode`, or None."""
        for change in self.modes:
            if change.mode == mode and change.at >= since:
                return change.at
        return None

    def wait_for_mode(self, mode: str, timeout: float, since: float = 0.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            at = self.entered(mode, since=since)
            if at is not None:
                return at
            # 20 Hz is plenty: the vehicle streams heartbeats at 1 Hz.
            time.sleep(0.05)
        return None

    def displacement_m(self, since: float = 0.0) -> float:
        """Straight-line metres between the first and last fix at/after `since`.

        The same quantity the detector anchors on, so a test can assert the
        vehicle really did (or did not) cover ground.
        """
        from haversine import Unit, haversine

        fixes = [p for p in self.positions if p[0] >= since]
        if len(fixes) < 2:
            return 0.0
        _, lat0, long0 = fixes[0]
        _, lat1, long1 = fixes[-1]
        return haversine((lat0, long0), (lat1, long1), unit=Unit.METERS)

    def matching_statustext(self, needle: str) -> list[str]:
        return [text for _, text in self.statustexts if needle.lower() in text.lower()]
