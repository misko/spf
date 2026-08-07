# Import mavutil
import argparse
import fcntl
import glob
import json
import logging
import math
import os
import subprocess
import sys
import termios
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from haversine import Unit, haversine
from pymavlink import mavutil

from spf.gps.boundaries import boundary_to_diamond  # crissy_boundary_convex
from spf.gps.boundaries import franklin_safe
from spf.gps.gps_utils import swap_lat_long
from spf.mavlink.compass_policy import (
    evaluate_compass_policy,
    format_compass_inventory,
)
from spf.mavlink.mavparm import MAVParmDict
from spf.mavlink.pymavlink_compat import harden_pymavlink_instance_messages
from spf.motion_planners.dynamics import Dynamics
from spf.motion_planners.planner import (
    BouncePlanner,
    CirclePlanner,
    PointCycle,
    StationaryPlanner,
)

# Before any vehicle link is opened. run_compass_calibration() is reachable from
# Taranis CH10 in the field, and the MAG_CAL_PROGRESS messages it provokes are
# what trip the pymavlink instance-message defect this repairs.
harden_pymavlink_instance_messages()

logging.basicConfig(
    filename="logs.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s: %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

EKF_STATUS_DEC_TO_STRING = {
    1: "EKF_ATTITUDE",
    2: "EKF_VELOCITY_HORIZ",
    4: "EKF_VELOCITY_VERT",
    8: "EKF_POS_HORIZ_REL",
    16: "EKF_POS_HORIZ_ABS",
    32: "EKF_POS_VERT_ABS",
    64: "EKF_POS_VERT_AGL",
    128: "EKF_CONST_POS_MODE",
    256: "EKF_PRED_POS_HORIZ_REL",
    512: "EKF_PRED_POS_HORIZ_ABS",
    1024: "EKF_UNINITIALIZED",
}
EKF_STATUS_STRING_TO_DEC = {v: k for k, v in EKF_STATUS_DEC_TO_STRING.items()}

gps_fix_type_str_to_num = {
    0: "GPS_FIX_TYPE_NO_GPS",
    1: "GPS_FIX_TYPE_NO_FIX",
    2: "GPS_FIX_TYPE_2D_FIX",
    3: "GPS_FIX_TYPE_3D_FIX",
    4: "GPS_FIX_TYPE_DGPS",
    5: "GPS_FIX_TYPE_RTK_FLOAT",
    6: "GPS_FIX_TYPE_RTK_FIXED",
    7: "GPS_FIX_TYPE_STATIC",
    8: "GPS_FIX_TYPE_PPP",
}
gps_fix_type_num_to_str = {v: k for k, v in gps_fix_type_str_to_num.items()}

mav_states_list = [
    "MAV_STATE_UNINIT",
    "MAV_STATE_BOOT",
    "MAV_STATE_CALIBRATING",
    "MAV_STATE_STANDBY",
    "MAV_STATE_ACTIVE",
    "MAV_STATE_CRITICAL",
    "MAV_STATE_EMERGENCY",
    "MAV_STATE_POWEROFF",
    "MAV_STATE_FLIGHT_TERMINATION",
]

mission_states = [
    "MISSION_STATE_UNKNOWN",
    "MISSION_STATE_NO_MISSION",
    "MISSION_STATE_NOT_STARTED",
    "MISSION_STATE_ACTIVE",
    "MISSION_STATE_PAUSED",
    "MISSION_STATE_COMPLETE",
]

mav_mission_states = [
    "MAV_MISSION_ACCEPTED",
    "MAV_MISSION_ERROR",
    "MAV_MISSION_UNSUPPORTED_FRAME",
    "MAV_MISSION_UNSUPPORTED",
    "MAV_MISSION_NO_SPACE",
    "MAV_MISSION_INVALID",
    "MAV_MISSION_INVALID_PARAM1",
    "MAV_MISSION_INVALID_PARAM2",
    "MAV_MISSION_INVALID_PARAM3",
    "MAV_MISSION_INVALID_PARAM4",
    "MAV_MISSION_INVALID_PARAM5_X",
    "MAV_MISSION_INVALID_PARAM6_Y",
    "MAV_MISSION_INVALID_PARAM7",
    "MAV_MISSION_INVALID_SEQUENCE",
    "MAV_MISSION_DENIED",
    "MAV_MISSION_OPERATION_CANCELLED",
]

sensors_list = [
    "MAV_SYS_STATUS_SENSOR_3D_GYRO",
    "MAV_SYS_STATUS_SENSOR_3D_ACCEL",
    "MAV_SYS_STATUS_SENSOR_3D_MAG",
    "MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE",
    "MAV_SYS_STATUS_SENSOR_DIFFERENTIAL_PRESSURE",
    "MAV_SYS_STATUS_SENSOR_GPS",
    "MAV_SYS_STATUS_SENSOR_OPTICAL_FLOW",
    "MAV_SYS_STATUS_SENSOR_VISION_POSITION",
    "MAV_SYS_STATUS_SENSOR_LASER_POSITION",
    "MAV_SYS_STATUS_SENSOR_EXTERNAL_GROUND_TRUTH",
    "MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL",
    "MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION",
    "MAV_SYS_STATUS_SENSOR_YAW_POSITION",
    "MAV_SYS_STATUS_SENSOR_Z_ALTITUDE_CONTROL",
    "MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL",
    "MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS",
    "MAV_SYS_STATUS_SENSOR_RC_RECEIVER",
    "MAV_SYS_STATUS_SENSOR_3D_GYRO2",
    "MAV_SYS_STATUS_SENSOR_3D_ACCEL2",
    "MAV_SYS_STATUS_SENSOR_3D_MAG2",
    "MAV_SYS_STATUS_GEOFENCE",
    "MAV_SYS_STATUS_AHRS",
    "MAV_SYS_STATUS_TERRAIN",
    "MAV_SYS_STATUS_REVERSE_MOTOR",
    "MAV_SYS_STATUS_LOGGING",
    "MAV_SYS_STATUS_SENSOR_BATTERY",
    "MAV_SYS_STATUS_SENSOR_PROXIMITY",
    "MAV_SYS_STATUS_SENSOR_SATCOM",
    "MAV_SYS_STATUS_PREARM_CHECK",
    "MAV_SYS_STATUS_OBSTACLE_AVOIDANCE",
    "MAV_SYS_STATUS_SENSOR_PROPULSION",
    # "MAV_SYS_STATUS_EXTENSION_USED",
]


custom_mode_mapping = {
    0: "ROVER_MODE_MANUAL",
    1: "ROVER_MODE_ACRO",
    3: "ROVER_MODE_STEERING",
    4: "ROVER_MODE_HOLD",
    5: "ROVER_MODE_LOITER",
    6: "ROVER_MODE_FOLLOW",
    7: "ROVER_MODE_SIMPLE",
    10: "ROVER_MODE_AUTO",
    11: "ROVER_MODE_RTL",
    12: "ROVER_MODE_SMART_RTL",
    15: "ROVER_MODE_GUIDED",
    16: "ROVER_MODE_INITIALIZING",
}

ROVER_MODE_RTL = 11

switchable_modes = {
    "GUIDED": "ROVER_MODE_GUIDED",
    "MANUAL": "ROVER_MODE_MANUAL",
    "HOLD": "ROVER_MODE_HOLD",
}

mav_cmds_num2name = {}


tones = {
    "gps-time": "MFT240L8 C C C P4 C C C P4 L8dcdcdcdc",
    "check-diff": "MFT240L8 A B P4 A B P4 L8dcdc",
    "git": "MFT240L4 < F P2 F P4 L8dcdc",
    "planner": "MFT240L8 G G F F P4 G G F F P4 L8dc",
    "wait": "MFT240L8 G P4 < G P4 < G P4 > > G P4 < G P4 < G",
    "ready": "MFT240L8 G P8 < G P8 < G P8 > > G P8 < G P8 < G",
    "failure": "MFT240L8 D D D P4 D D D P4 L8dddddc",
    # A short, unobtrusive double chirp for the 15-second readiness watchdog.
    "readiness-wait": "MFT240L8 G P8 G",
    # Three long descending notes, deliberately unlike the GPS/check tones.
    "radio-missing": "MFT180L4 A F D P2",
    # Rising then held: the operator's confirmation that CH9 was accepted and
    # the Pi is going down. Must be distinguishable from "failure" at a
    # distance, because it is the last thing the rover says.
    "shutdown": "MFT180L8 C E G L2 > C",
}
tones = {k: v.replace(" ", "").encode() for k, v in tones.items()}

LOG_ERASE = 121

RC_SHUTDOWN_THRESHOLD = 1500
RC_REBOOT_THRESHOLD = 1000
RC_COMPASS_THRESHOLD = 1500
# Samples above threshold before a press counts. 1 is deliberate: a level test
# needs no clock, so the trigger behaves identically at 0.5 Hz and 10 Hz --
# which a timed gesture cannot. Raise to 2 to trade ~250ms at 4 Hz for immunity
# to a single corrupt frame; both SBUS and MAVLink already checksum, so 1 is
# the honest default.
RC_PRESS_CONFIRM_SAMPLES = 1
# Bounds on the pre-halt vehicle-safing sequence. Both exist to be exceeded
# safely: a failed HOLD or a failed disarm must never prevent the poweroff,
# because the operator's fallback when the switch is ignored is pulling the
# battery on a live rover -- which is the failure this whole path removes.
RC_SHUTDOWN_HOLD_TIMEOUT_SECONDS = 2.0
RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS = 2.0
# Ask the flight controller for the RC rate we want rather than inheriting
# whatever SR*_RC_CHAN each FC happens to carry -- that parameter is in no
# enforced .params file, so it is per-rover luck.
RC_CHANNELS_INTERVAL_US = 100_000
RC_ULTRASONIC_LOW_THRESHOLD = 1000
RC_ULTRASONIC_HIGH_THRESHOLD = 1500
RC_ULTRASONIC_STABLE_SAMPLES = 3
RC_ULTRASONIC_MAX_SAMPLE_GAP_SECONDS = 1.5

# Stall detection. The rover can high-center or jam a wheel while driving a
# GUIDED waypoint; ATC_SPEED_I then winds up and the throttle pins at 100%
# until someone pulls the battery. The invariant is deliberately displacement
# from an anchor rather than distance-to-target: with TURN_RADIUS 5.0 the rover
# can arc for over twenty seconds without closing on its target, and that is
# healthy driving, not a stall.
STALL_PROGRESS_RADIUS_M = 3.0  # displacement that counts as "it moved"
STALL_DETECT_SECONDS = 10.0  # no progress for this long == stalled
STALL_MANUAL_SECONDS = 40.0  # recovery only: give up, hand to the operator
# The same invariant, but for a rover the autopilot is commanding to NEUTRAL
# throttle while a destination is still outstanding -- the parked-in-GUIDED
# failure (see _stall_verdict). It has to be longer than STALL_DETECT_SECONDS
# for two independent reasons:
#   * stop_vehicle() produces a real coast-down, up to ~12 m over 25 s, so the
#     rover keeps moving for a while after the throttle goes neutral; and
#   * a waypoint pivot legitimately shows motor-off for several seconds while
#     the rover turns in place and covers no ground.
# Sized from the 2026-08-07 captures, by two independent measurements that
# disagree about the worst healthy case -- so this takes the larger one.
#
#   (a) Replaying every GUIDED, planner-driven second of the seventeen captures
#       with usable GPS through _stall_verdict itself: the longest window in
#       which a HEALTHY rover failed to clear STALL_PROGRESS_RADIUS_M was
#       11.5 s. This is the watchdog's own semantics -- windows are split
#       wherever real motion resets the anchor -- so it is the number that
#       actually binds.
#   (b) A cruder upper bound over all 23 logs: the longest unbroken run of
#       GUIDED samples with the motor commanded off, ignoring anchor resets
#       entirely, was 20.0 s (RO3 00:08:06). It cannot bind, because a rover
#       creeping past the radius inside that run resets the clock -- but it is
#       measured rather than modelled, and it is nearly twice (a).
#
# The three parked tails ran 70.9 / 203.4 / 240.9 s, so the populations are
# separated by a factor of three either way. 30.0 clears (b) by 1.5x and (a) by
# 2.6x while still catching the SHORTEST real incident with 2.4x to spare. The
# asymmetry is deliberate: a false stall sends an operator across a field after
# a healthy rover, and this watchdog's whole credibility is that it does not.
STALL_PARKED_SECONDS = 30.0
# Leg length, both reverse and lateral. MUST exceed WP_RADIUS (5.0 in
# rover3_base_parameters.params) or the maneuver silently does nothing:
# ArduPilot judges arrival on WP_RADIUS, so a target 3 m away is already
# "reached" the moment it is issued, and the rover never drives the leg. Found
# in SITL, where the escape produced only small steering corrections
# (servo1=1502, servo3=1487) instead of the pegged reverse output expected.
STALL_ESCAPE_DISTANCE_M = 8.0
STALL_ESCAPE_DRIVE_SECONDS = 4.0  # hard bound per leg
STALL_ESCAPE_SETTLE_SECONDS = 1.0  # HOLD between legs, so the firmware resets I
STALL_AXIS_MIN_M = 0.5  # shortest usable stall->reverse axis
STALL_MAX_ESCAPES = 3  # escapes without real recovery before handing over
# Displacement that clears the escape count, i.e. proof the rover recovered
# under its OWN power. Deliberately far beyond what an escape can produce
# (one leg is STALL_ESCAPE_DISTANCE_M): that is what keeps the reset
# non-circular. Resetting on the progress radius instead would let each
# escape clear its own count, which is no cap at all.
STALL_RECOVERED_M = 3 * STALL_ESCAPE_DISTANCE_M
STALL_MANUAL_ATTEMPTS = 3  # set_mode("MANUAL") retries before giving up loudly
STALL_MANUAL_RETRY_SECONDS = 2.0

# DO_REPOSITION's param2 flags word. Bit 0 tells the autopilot it may change
# mode to honour the destination; ArduPilot's Rover/GCS_Mavlink.cpp
# handle_command_int_do_reposition() reads it as
#
#   const bool change_modes = ((int32_t)packet.param2 &
#         MAV_DO_REPOSITION_FLAGS_CHANGE_MODE) == MAV_DO_REPOSITION_FLAGS_CHANGE_MODE;
#   if (!rover.control_mode->in_guided_mode() && !change_modes) {
#       return MAV_RESULT_DENIED;
#   }
#
# so with param2=0 -- what this file sent until 2026-08-07 -- every waypoint
# issued while the vehicle was in anything but GUIDED was refused before the
# destination was even read. That is not a corner case: the EKF failsafe parks
# the rover in HOLD, _escape_jam passes through HOLD deliberately, and a capture
# abort sends HOLD itself.
#
# Taken from pymavlink's enum rather than written as a literal, and the value
# was read out of the firmware the rovers actually run
# (csmisko/ardupilotspf:latest, common.xml entry value="1") rather than from
# memory. What really pins it is
# tests/test_in_simulator.py::test_a_reposition_out_of_hold_is_accepted_and_drives,
# which proves the flag against that firmware instead of against a constant.
REPOSITION_CHANGE_MODE = mavutil.mavlink.MAV_DO_REPOSITION_FLAGS_CHANGE_MODE

# A command answered with either of these was not refused. IN_PROGRESS is an
# interim ack (long-running commands send it before a final result), so treating
# it as a failure would cry wolf on every one of them.
COMMAND_RESULTS_OK = (
    mavutil.mavlink.MAV_RESULT_ACCEPTED,
    mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
)

# _stall_verdict outcomes.
STALL_OK = "ok"
STALL_ESCAPE = "escape"
STALL_MANUAL = "manual"

# move_to_point outcomes. Three states rather than a bool: a skipped waypoint
# must not be confused with an aborted capture, since the planner continues in
# the first case and stops in the second.
MOVE_REACHED = "reached"
MOVE_SKIPPED = "skipped"
MOVE_ABORTED = "aborted"

# Sensors whose failure makes the vehicle's own position and heading untrue.
# gps_lat/gps_long/heading ARE the ground truth a capture is labelled with, so
# a record written while one of these is unhealthy is not merely uninformative
# -- it is wrong, and nothing downstream can tell.
NAVIGATION_SENSORS = (
    "MAV_SYS_STATUS_SENSOR_GPS",
    "MAV_SYS_STATUS_SENSOR_3D_MAG",
)


class _RCPressTrigger:
    """Fire once per press of an RC switch, never on one that was already high.

    Deliberately level-triggered and clockless. The interlock this replaced
    required a >2s continuous hold, measured with time.monotonic() at the
    moment each frame was *processed* -- so it depended on the RC stream rate
    (below ~0.67 Hz the 1.5s gap window reset on every sample and no hold could
    ever complete) and on the message loop not stalling (a loop that stalls and
    then drains buffered frames collapses a genuine 2s hold to ~10ms). Neither
    dependency is observable from the cockpit, and both fail silently. A level
    test has neither: one frame above threshold is one frame above threshold at
    any rate, in any burst.

    `released_seen` is the single bit of memory kept, and it carries the whole
    safety argument. A receiver on failsafe Hold keeps reporting the last
    values it saw, so a process that connects mid-press -- or after the
    operator's last act before walking away was a press -- would otherwise act
    on a press nobody is making. Since the capture process restarts on every
    capture iteration, that is a shutdown boot-loop, not a rare edge. SH is
    spring-loaded, so in normal use this bit is set within one frame and the
    operator never perceives it.
    """

    def __init__(self, label, threshold, confirm_samples=RC_PRESS_CONFIRM_SAMPLES):
        self.label = label
        self.threshold = threshold
        self.confirm_samples = confirm_samples
        self._released_seen = False
        self._high_samples = 0
        self._latched = False
        self._warned_stale_high = False

    def update(self, value):
        if value <= self.threshold:
            self._released_seen = True
            self._high_samples = 0
            self._latched = False  # re-arm for the next press
            return False

        if not self._released_seen:
            # Once, not once per frame: at 10 Hz this would otherwise bury the
            # journal, and the condition can persist for the life of the link.
            if not self._warned_stale_high:
                logging.warning(
                    "RC %s has been high since connect; ignoring it until the "
                    "switch is released (stale failsafe values, or a switch "
                    "left up, look exactly like a press)",
                    self.label,
                )
                self._warned_stale_high = True
            return False

        if self._latched:
            return False

        self._high_samples += 1
        if self._high_samples >= self.confirm_samples:
            self._latched = True
            return True
        return False


def _build_rc_triggers():
    """One press trigger per destructive RC channel.

    CH7 and CH10 share the trigger rather than testing their raw value inline,
    which is not cosmetic: `elif msg.chan7_raw > 1000: reboot(); sys.exit(1)`
    fires for any resting value in (1000, 1500] -- a centred 3-position switch
    reads 1500 -- and had nothing to latch it, so it re-fired on every frame.
    The only thing keeping that quiet was CH7 happening to rest at <=1000.
    """

    return SimpleNamespace(
        shutdown=_RCPressTrigger("CH9 (shutdown)", RC_SHUTDOWN_THRESHOLD),
        reboot=_RCPressTrigger("CH7 (reboot)", RC_REBOOT_THRESHOLD),
        compass=_RCPressTrigger("CH10 (compass calibration)", RC_COMPASS_THRESHOLD),
    )


class _RCStableSwitch:
    """Apply hysteresis and consecutive-sample debounce to an RC switch."""

    def __init__(
        self,
        *,
        initial_state,
        low_threshold,
        high_threshold,
        stable_samples,
        max_sample_gap_seconds,
    ):
        self.state = bool(initial_state)
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.stable_samples = int(stable_samples)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)
        self.candidate = None
        self.candidate_samples = 0
        self.last_sample_at = None

    def update(self, *, value, now):
        if value >= self.high_threshold:
            observed = True
        elif value <= self.low_threshold:
            observed = False
        else:
            self.candidate = None
            self.candidate_samples = 0
            self.last_sample_at = now
            return None

        if (
            self.last_sample_at is None
            or now - self.last_sample_at > self.max_sample_gap_seconds
            or observed != self.candidate
        ):
            self.candidate = observed
            self.candidate_samples = 1
        else:
            self.candidate_samples += 1
        self.last_sample_at = now

        if self.candidate_samples < self.stable_samples or observed == self.state:
            return None
        self.state = observed
        self.candidate = None
        self.candidate_samples = 0
        return self.state


# Mean earth radius, matching the `haversine` library used by
# distance_to_target/move_to_point — so a rest offset expressed in metres
# means the same metres the rover uses to decide it has arrived. (The WGS84
# equatorial radius 6378137 would make a requested 2 m read as 1.9978 m.)
EARTH_RADIUS_M = 6371008.8

DEFAULT_MAVLINK_RECONNECT_ATTEMPTS = 3
DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS = 1.0
DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS = 10.0


class MavlinkConnectionError(RuntimeError):
    """The vehicle link could not be established or recovered safely."""


class MavlinkMessageHandlingError(MavlinkConnectionError):
    """A decoded MAVLink message could not be applied to controller state."""


class MavlinkParameterError(MavlinkConnectionError):
    """The complete vehicle parameter set could not be read."""


class VehicleParameterVerificationError(MavlinkParameterError):
    """Managed vehicle parameters were not acknowledged and verified."""


def resolve_ardupilot_serial(configured="", available_pilots=None):
    """Return a stable ArduPilot ``/dev/serial/by-id`` endpoint when possible.

    Linux ``ttyACM`` numbers are allocation-order identifiers and can change
    after a flight-controller reboot.  A caller may still provide a tty name,
    but it is promoted to the by-id symlink that resolves to the same device.
    """
    if available_pilots is None:
        available_pilots = sorted(glob.glob("/dev/serial/by-id/usb-ArduPilot*"))
    else:
        available_pilots = sorted(available_pilots)

    if configured in ("", "serial", None):
        if len(available_pilots) != 1:
            raise MavlinkConnectionError(
                "Expected exactly one ArduPilot serial device; "
                f"found {len(available_pilots)}: {available_pilots}"
            )
        return available_pilots[0]

    if configured in available_pilots or configured.startswith("/dev/serial/by-id/"):
        return configured

    configured_realpath = os.path.realpath(configured)
    matches = [
        candidate
        for candidate in available_pilots
        if os.path.realpath(candidate) == configured_realpath
    ]
    if len(matches) == 1:
        logging.info(
            "Promoted unstable MAVLink endpoint %s to %s",
            configured,
            matches[0],
        )
        return matches[0]
    if len(matches) > 1:
        raise MavlinkConnectionError(
            f"Serial endpoint {configured} maps to multiple by-id devices: {matches}"
        )

    logging.warning(
        "No stable /dev/serial/by-id identity found for %s; reconnects may "
        "select the wrong tty after re-enumeration",
        configured,
    )
    return configured


def _claim_serial_exclusive(connection, endpoint):
    """Ask the kernel to reject a second opener of this serial port."""
    if ":" in endpoint or not hasattr(connection, "port"):
        return
    try:
        fd = connection.port.fileno()
        fcntl.ioctl(fd, termios.TIOCEXCL)
    except Exception as error:
        try:
            connection.close()
        except Exception:
            pass
        raise MavlinkConnectionError(
            f"Could not claim exclusive MAVLink ownership of {endpoint}: {error}. "
            "Stop mavproxy/QGroundControl serial access and retry."
        ) from error


def open_mavlink_connection(endpoint, baud=115200):
    """Open one MAVLink endpoint with exclusive local-serial ownership."""
    connection = mavutil.mavlink_connection(endpoint, baud=baud)
    _claim_serial_exclusive(connection, endpoint)
    return connection


def mavlink_connection_factory(endpoint, baud=115200):
    return lambda: open_mavlink_connection(endpoint, baud=baud)


def connect_with_heartbeat(
    connection_factory,
    *,
    attempts=DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
    heartbeat_timeout=DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
    retry_backoff=DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
):
    """Open a link and require a real heartbeat, with a bounded retry count."""
    last_error = None
    for attempt in range(1, attempts + 1):
        connection = None
        try:
            connection = connection_factory()
            heartbeat = connection.wait_heartbeat(
                blocking=True,
                timeout=heartbeat_timeout,
            )
            if heartbeat is None:
                raise MavlinkConnectionError(
                    f"no heartbeat within {heartbeat_timeout:.1f}s"
                )
            return connection, heartbeat
        except Exception as error:
            last_error = error
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            logging.warning(
                "MAVLink connection attempt %d/%d failed: %s",
                attempt,
                attempts,
                error,
            )
            if attempt < attempts:
                time.sleep(retry_backoff)
    raise MavlinkConnectionError(
        f"MAVLink unavailable after {attempts} attempts: {last_error}"
    ) from last_error


def meters_to_degrees(east_m, north_m, latitude_deg):
    """Convert an (East, North) metre offset to (dlong, dlat) degrees.

    Longitude degrees shrink by cos(latitude), so the two axes are NOT
    interchangeable — at the SPF sites (~37.8 deg) the anisotropy is ~26%.
    Returns (dlong, dlat) to match the (long, lat) convention used by
    spf/gps/boundaries.py and every planner coordinate.
    """
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_long = m_per_deg_lat * np.cos(np.radians(latitude_deg))
    return np.array([east_m / m_per_deg_long, north_m / m_per_deg_lat])


def degrees_to_meters(dlong, dlat, latitude_deg):
    """Convert a (dlong, dlat) degree offset to (East, North) metres.

    The exact inverse of meters_to_degrees. Needed because a bearing cannot be
    rotated in degree-space: longitude degrees shrink by cos(latitude), so at
    the SPF sites (~37.8 deg) a 90-degree rotation applied to a (dlong, dlat)
    pair comes out up to ~15 degrees off. Convert to metres, rotate there,
    convert back.
    """
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_long = m_per_deg_lat * np.cos(np.radians(latitude_deg))
    return np.array([dlong * m_per_deg_long, dlat * m_per_deg_lat])


def bearing_to_unit_vector(bearing_deg):
    """Compass bearing (degrees, clockwise from North) to an (East, North) unit."""
    radians = np.radians(bearing_deg)
    return np.array([np.sin(radians), np.cos(radians)])


def rest_offset_to_degrees(rest_offset_m, boundary):
    """Per-rover resting offset in degrees, or None when unconfigured.

    rest_offset_m is (east_m, north_m). None/absent -> None, which preserves
    the historical centroid behaviour exactly.
    """
    if rest_offset_m is None:
        return None
    offset = np.asarray(rest_offset_m, dtype=float)
    if offset.shape != (2,):
        raise ValueError(
            f"rest-offset-m must be [east_m, north_m], got {rest_offset_m}"
        )
    if not np.isfinite(offset).all():
        raise ValueError(f"rest-offset-m must be finite, got {rest_offset_m}")
    if np.all(offset == 0.0):
        return None
    centroid = boundary.mean(axis=0)
    return meters_to_degrees(offset[0], offset[1], centroid[1])


def drone_get_planner(routine, boundary, rest_offset_m=None):
    """Build the motion planner for a routine.

    rest_offset_m (east_m, north_m) shifts this vehicle's RESTING position —
    start point, stationary point and home/RTL — away from the boundary
    centroid so that co-located rovers do not converge on the same spot.
    It deliberately does NOT move `circle_center` or the diamond points: those
    are pattern geometry measured against the fence, and CirclePlanner performs
    no bounds check (planner.py yield_points never calls Dynamics.to_steps), so
    shifting the ring would silently drive outside the geofence.
    """
    centroid = boundary.mean(axis=0)
    offset_deg = rest_offset_to_degrees(rest_offset_m, boundary)
    rest_point = centroid if offset_deg is None else centroid + offset_deg
    # None keeps Planner.get_home_point() on its original centroid expression,
    # so an unconfigured rover takes a bit-identical code path.
    home_point = None if offset_deg is None else rest_point
    if offset_deg is not None:
        logging.info(
            f"Rest offset {rest_offset_m} (east_m, north_m) -> "
            f"centroid {centroid} shifted to {rest_point}"
        )

    if routine == "circle":
        return CirclePlanner(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000001,
            ),
            start_point=rest_point,
            step_size=0.0001,
            circle_diameter=0.0003,
            circle_center=centroid,  # pattern geometry: stays fence-centred
            home_point=home_point,
        )
    elif routine == "center":
        return StationaryPlanner(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000001,
            ),
            start_point=rest_point,
            stationary_point=rest_point,
            step_size=0.0002,
            home_point=home_point,
        )
    elif routine == "bounce":
        return BouncePlanner(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000000001,
            ),
            start_point=rest_point,
            epsilon=0.0000001,
            step_size=0.1,
            home_point=home_point,
        )
    elif routine == "diamond":
        base_points = boundary_to_diamond(boundary)
        points = base_points * 0.85 + centroid * 0.15  # fence-relative geometry
        if np.random.rand() > 0.5:
            points = np.flip(points, axis=0)
        return PointCycle(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000000001,
            ),
            start_point=rest_point,
            step_size=0.1,
            points=points,
            home_point=home_point,
        )

    else:
        raise Exception("Missing planner")


def lookup_bits(x, table):
    return [y for y in table if x & getattr(mavutil.mavlink, y)]


def lookup_exact(x, table):
    return [y for y in table if x == getattr(mavutil.mavlink, y)]


def _enum_name(enum, value, prefix):
    """MAVLink enum value -> name, falling back to the raw number.

    Never raises. These names exist to make a log line readable, and an id this
    build of pymavlink has not heard of is exactly when the line matters most.
    """
    entry = mavutil.mavlink.enums.get(enum, {}).get(value)
    if entry is None:
        return f"{prefix}{value}"
    return entry.name


def mav_command_name(command):
    return _enum_name("MAV_CMD", command, "MAV_CMD_")


def mav_result_name(result):
    return _enum_name("MAV_RESULT", result, "MAV_RESULT_")


@dataclass(frozen=True)
class CommandResult:
    """The vehicle's answer to one command, kept by command NAME.

    Keyed by name and not by number so that a caller, a log line and a test all
    say the same thing. `at` is wall clock because it is compared against the
    log, not against the stall watchdog's monotonic clock.
    """

    command: str
    result: str
    at: float

    @property
    def accepted(self):
        return self.result == "MAV_RESULT_ACCEPTED"


class Drone:
    def __init__(
        self,
        connection,
        tolerance_in_m=5,
        distance_finder=None,
        fake=False,
        ignore_mode=False,
        connection_factory=None,
        reconnect_attempts=DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
        reconnect_backoff=DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
        reconnect_heartbeat_timeout=DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
        crash_detect=True,
        crash_recovery=False,
        stall_detect_seconds=STALL_DETECT_SECONDS,
        stall_manual_seconds=STALL_MANUAL_SECONDS,
        stall_progress_radius_m=STALL_PROGRESS_RADIUS_M,
        stall_parked_seconds=STALL_PARKED_SECONDS,
    ):
        self.connection = connection
        self.connection_factory = connection_factory
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff = reconnect_backoff
        self.reconnect_heartbeat_timeout = reconnect_heartbeat_timeout
        self.connection_failure = None
        self.connection_epoch = 0
        self._connection_healthy = threading.Event()
        self._connection_lock = threading.RLock()
        self.param_count = 0
        self.heading = 0
        self.gps_time = 0
        self.time_since_boot = 0
        # Latest COMMAND_ACK per command name; see handle_COMMAND_ACK.
        self.command_results = {}
        self.distance_finder = distance_finder
        self.ignore_mode = ignore_mode
        if self.distance_finder is not None:
            self.distance_finder.run_in_new_thread()

        logging.getLogger("numba").setLevel(logging.WARNING)
        self.fake = fake
        if not self.fake:
            self.mav_mode_mapping_name2num = connection.mode_mapping()
            self.mav_mode_mapping_num2name = mavutil.mode_mapping_bynumber(
                connection.sysid_state[connection.sysid].mav_type
            )
        # breakpoint()
        self.reset_params()

        self.healthy_ekf_flag = (
            EKF_STATUS_STRING_TO_DEC["EKF_ATTITUDE"]
            | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
            | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
        )

        self.motor_active = False
        # Owned by move_to_point, not by the stall state: it is "the planner
        # has issued a destination and has not reached it yet", which is what
        # makes a motionless rover a fault rather than a rover with nothing to
        # do. Deliberately NOT reset by _initialize_stall_state, so a stall
        # re-initialisation can never silently disarm the watchdog.
        self._destination_outstanding = False
        self.crash_detect = crash_detect
        self.crash_recovery = crash_recovery
        self.stall_detect_seconds = float(stall_detect_seconds)
        self.stall_manual_seconds = float(stall_manual_seconds)
        self.stall_progress_radius_m = float(stall_progress_radius_m)
        self.stall_parked_seconds = float(stall_parked_seconds)
        self._initialize_stall_state()
        self._initialize_motion_stop_state()
        self._rc_triggers = _build_rc_triggers()

        self.ekf_healthy = False
        self.disable_distance_finder = False
        self.mav_states = []
        self.gps = np.zeros(2)
        self.mav_mode = None
        self.mav_cmd_name2num = {
            "MAV_CMD_DO_SET_MODE": 176,
        }
        self.mav_cmd_num2name = {176: "MAV_CMD_DO_SET_MODE"}

        self.message_condition = threading.Condition()  # can set message_loop=False,
        self.single_condition = threading.Condition()  # can set message_loop=True
        # self.drone_ready_condition = threading.Condition()
        self.drone_ready = (
            False  # Are all the systems good (outside of armed and guided mode)
        )

        self.message_loop = True
        self.single_operation = False

        self.sensors_present = []
        self.sensors_enabled = []
        self.sensors_health = []

        self.last_heartbeat = 0

        self.timeout = 0.5
        self.tolerance_in_m = tolerance_in_m
        self.ignore_messages = [
            # "AHRS2",
            "ATTITUDE",
            "BATTERY_STATUS",
            # "EKF_STATUS_REPORT",
            "ESC_TELEMETRY_1_TO_4",
            # "GPS_RAW_INT",
            "HWSTATUS",
            "LOCAL_POSITION_NED",
            "MEMINFO",
            # "MISSION_CURRENT",
            # "NAV_CONTROLLER_OUTPUT",
            "POSITION_TARGET_GLOBAL_INT",
            "POWER_STATUS",
            "RAW_IMU",
            # "RC_CHANNELS",
            # "RC_CHANNELS_SCALED",
            "SCALED_IMU2",
            "SCALED_IMU3",
            "SCALED_PRESSURE",
            "SCALED_PRESSURE2",
            "SERVO_OUTPUT_RAW",
            "SIMSTATE",
            # "SYSTEM_TIME",
            "SYS_STATUS",
            "VFR_HUD",
            "VIBRATION",
            # "PARAM_VALUE",
            "BAD_DATA",
        ]

        # self.erase_logs()

        self.message_loop_thread = threading.Thread(
            target=self.process_messages, daemon=True
        )

        self.last_heartbeat_log = None
        self.armed = False

        self.gps_satellites = -1
        self.gps_fix_type = "NOT_SET_YET"
        # self.mission_item_condition = threading.Condition()
        # self.mission_item_reached = False

    @property
    def connection_healthy(self):
        return self._connection_healthy.is_set()

    def _mark_connection_unhealthy(self, error):
        with self._connection_lock:
            logging.error("MAVLink connection lost: %s", error)
            self._connection_healthy.clear()
            self.connection_failure = None
            # Never preserve an armed/ready assumption across a broken link.
            self.armed = False
            self.drone_ready = False
            self.planner_in_control = False
            self.mav_mode = None
            self.mav_states = []
            self.gps = np.zeros(2)
            self.gps_satellites = -1
            self.gps_fix_type = "NOT_SET_YET"
            self.ekf_healthy = False
            self.sensors_health = []
            if hasattr(self, "_motion_hold_sent"):
                # A HOLD sent through the vanished connection is not proof that
                # the flight controller received it.  Retry after a fresh
                # heartbeat if a capture abort is still pending.
                self._motion_hold_sent.clear()

    def _require_healthy_connection(self):
        if self.connection_factory is not None and not self.connection_healthy:
            raise MavlinkConnectionError(
                "MAVLink connection is not healthy; refusing vehicle command"
            )
        return self.connection

    @contextmanager
    def _command_connection(self, *, allow_replacement=False):
        """Hold one verified connection stable for a runtime operation.

        Vehicle commands fail closed if a reconnect begins while they are
        waiting for the lock.  Advisory operations such as the buzzer may wait
        for a bounded reconnect and opt into the replacement connection.
        """
        expected_connection = self.connection
        if (
            not allow_replacement
            and self.connection_factory is not None
            and not self.connection_healthy
        ):
            raise MavlinkConnectionError(
                "MAVLink connection is not healthy; refusing vehicle command"
            )
        with self._connection_lock:
            if not allow_replacement and self.connection is not expected_connection:
                raise MavlinkConnectionError(
                    "MAVLink connection changed; refusing stale vehicle command"
                )
            yield self._require_healthy_connection()

    def raise_if_connection_failed(self):
        if self.connection_failure is not None:
            if isinstance(self.connection_failure, MavlinkConnectionError):
                raise self.connection_failure
            raise MavlinkConnectionError(
                str(self.connection_failure)
            ) from self.connection_failure

    def _recover_connection(self, cause):
        self._mark_connection_unhealthy(cause)
        if self.connection_factory is None:
            self.connection_failure = MavlinkConnectionError(
                f"MAVLink receive loop stopped: {cause}"
            )
            return False

        with self._connection_lock:
            stale_connection = self.connection
            try:
                stale_connection.close()
            except Exception:
                pass

            try:
                connection, heartbeat = connect_with_heartbeat(
                    self.connection_factory,
                    attempts=self.reconnect_attempts,
                    heartbeat_timeout=self.reconnect_heartbeat_timeout,
                    retry_backoff=self.reconnect_backoff,
                )
            except MavlinkConnectionError as error:
                self.connection_failure = error
                logging.critical("MAVLink reconnect exhausted: %s", error)
                return False

            self.connection = connection
            self.connection_epoch += 1
            if not self.fake:
                self.mav_mode_mapping_name2num = connection.mode_mapping()
                self.mav_mode_mapping_num2name = mavutil.mode_mapping_bynumber(
                    connection.sysid_state[connection.sysid].mav_type
                )
            self.process_message(heartbeat)
            # A reconnect means the switch positions during the outage are
            # unknown, and a receiver on failsafe Hold will report whatever it
            # last saw. Re-arming the press triggers forces a fresh release
            # before any destructive channel can act on the new link.
            self._rc_triggers = _build_rc_triggers()
            self._request_rc_channel_rate()
            logging.info(
                "MAVLink reconnected after a fresh flight-controller heartbeat"
            )
            return True

    def _request_rc_channel_rate(self, interval_us=RC_CHANNELS_INTERVAL_US):
        """Ask the FC for the RC_CHANNELS rate we want, rather than inherit one.

        SR*_RC_CHAN is in no enforced .params file, so the rate is per-rover
        luck -- rover 3's dump has 4 Hz on SERIAL0 and 1 Hz on SERIAL2. The
        press trigger works at any rate by construction, so this is latency and
        uniformity, not correctness. Advisory: a failure must not stop the
        receive loop from starting.
        """

        try:
            with self._command_connection(allow_replacement=True) as connection:
                connection.mav.command_long_send(
                    connection.target_system,
                    connection.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                    interval_us,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            return True
        except Exception as error:
            logging.warning(
                "Could not request an RC_CHANNELS rate; falling back to "
                "whatever SR*_RC_CHAN this flight controller carries: %s",
                error,
            )
            return False

    def set_and_start_planner(self, planner):
        self.planner = planner
        self.planner_in_control = False
        assert self.planner is not None

        self.planner_thread = threading.Thread(target=self.run_planner, daemon=True)
        self.planner_thread.start()

    def buzzer(self, tone_bytes):
        for _ in range(5):
            try:
                with self._command_connection(allow_replacement=True) as connection:
                    connection.mav.play_tune_send(
                        connection.target_system,
                        connection.target_component,
                        tone_bytes,
                    )
                return True
            except AttributeError:
                pass
            except MavlinkConnectionError as error:
                logging.info("Skipping buzzer while MAVLink is unavailable: %s", error)
                return False
            except Exception as error:
                # The buzzer is advisory.  A write failure must not kill the
                # planner, and receive-loop recovery remains the sole owner of
                # replacing the transport.
                logging.warning("MAVLink buzzer command failed: %s", error)
                return False
            time.sleep(0.1)
        return False

    def reset_params(self):
        self.params = MAVParmDict()

    def update_all_parameters(
        self,
        timeout=50,
        max_attempts=3,
        retry_backoff=1.0,
    ):
        """Fetch a complete, internally consistent parameter set.

        A stalled partial download is discarded and restarted from zero. This
        prevents parameters received before a flight-controller reset from
        being combined with a later boot's parameter stream.
        """
        for attempt in range(1, max_attempts + 1):
            self.raise_if_connection_failed()
            self.reset_params()
            self.param_count = 0
            with self._command_connection() as connection:
                try:
                    connection.param_fetch_all()
                except Exception as error:
                    logging.warning(
                        "Parameter fetch attempt %d/%d could not start: %s",
                        attempt,
                        max_attempts,
                        error,
                    )
                    if attempt < max_attempts:
                        time.sleep(retry_backoff)
                        continue
                    raise MavlinkParameterError(
                        f"Could not start parameter fetch: {error}"
                    ) from error

            progress_deadline = time.monotonic() + timeout
            params_read = 0
            while self.param_count == 0 or params_read < self.param_count:
                self.raise_if_connection_failed()
                current_count = len(self.params)
                if current_count != params_read:
                    params_read = current_count
                    progress_deadline = time.monotonic() + timeout
                if self.param_count > 0 and params_read == self.param_count:
                    break
                if time.monotonic() >= progress_deadline:
                    break
                time.sleep(min(0.1, max(timeout / 10.0, 0.001)))

            if self.param_count > 0 and len(self.params) == self.param_count:
                logging.info(
                    "Done loading drone parameters: have %d, wanted %d "
                    "(attempt %d/%d)",
                    len(self.params),
                    self.param_count,
                    attempt,
                    max_attempts,
                )
                return True

            logging.warning(
                "Incomplete parameter fetch attempt %d/%d: have %d, expected %d; "
                "discarding partial set and retrying",
                attempt,
                max_attempts,
                len(self.params),
                self.param_count,
            )
            if attempt < max_attempts:
                time.sleep(retry_backoff)

        raise MavlinkParameterError(
            "Failed to read a complete parameter set after "
            f"{max_attempts} attempts (last attempt had {len(self.params)}/"
            f"{self.param_count})"
        )

    # motion interface
    def start(self):
        if not self.fake:
            self._request_rc_channel_rate()
        self.message_loop_thread.start()
        return self

    def send_status(self, text):
        with self._command_connection() as connection:
            connection.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_CRITICAL, text.encode()
            )

    def planner_control_loss_reason(self):
        """Why the planner is not driving the vehicle right now, or None.

        `planner_in_control` alone is a LATCH: run_planner sets it True before
        its first move and clears it only on return or MOVE_ABORTED. Nothing
        clears it when the operator takes MANUAL -- so on 2026-08-07 rover 4
        recorded 258 snapshots of a vehicle sitting still under an operator's
        thumb, and the takeover accounting added for that incident could never
        observe it, because the signal it watches never moved.

        Mode is therefore read live from the heartbeat. That also makes resume
        automatic: hand control back and this returns None again with no state
        to unwind, whether the handback came from the operator's switch or from
        _hand_over_to_manual's own wait.
        """

        if not self.planner_in_control:
            return "the planner is not driving"
        if (
            getattr(self, "connection_factory", None) is not None
            and not self.connection_healthy
        ):
            return "the MAVLink connection is not healthy"
        if not self.ignore_mode and self.mav_mode != "ROVER_MODE_GUIDED":
            return f"the vehicle is in {self.mav_mode}, not GUIDED"
        return None

    def is_planner_in_control(self):
        return self.planner_control_loss_reason() is None

    def planner_is_still_driving(self):
        """Has run_planner left its driving section yet?

        The LATCH, deliberately: unlike is_planner_in_control() this ignores
        the live mode. Callers waiting for the planner thread to finish before
        they park the vehicle must not stop waiting merely because the operator
        happens to be holding MANUAL at that moment -- the planner is still in
        its loop and would race whatever they do next.
        """
        return bool(self.planner_in_control)

    def unhealthy_navigation_sensors(self):
        """Enabled navigation sensors the flight controller reports unhealthy.

        Enabled-but-unhealthy, never merely absent: a sensor the airframe does
        not have is not a fault, and treating it as one would mark every
        capture bad forever.
        """

        enabled = set(getattr(self, "sensors_enabled", None) or ())
        healthy = set(getattr(self, "sensors_health", None) or ())
        return tuple(
            sensor
            for sensor in NAVIGATION_SENSORS
            if sensor in enabled and sensor not in healthy
        )

    def navigation_health_warning(self):
        """Why this vehicle's reported position/heading is untrustworthy, or None.

        Deliberately NOT folded into planner_control_loss_reason. "Nobody was
        driving" and "we were driving but did not know where we were" are
        different facts with different fixes, and one number that means either
        one can be acted on for neither.
        """

        unhealthy = self.unhealthy_navigation_sensors()
        if unhealthy:
            return "unhealthy navigation sensors: " + ", ".join(sorted(unhealthy))
        if not self.ekf_healthy:
            return "the EKF is not healthy"
        return None

    def get_position_bearing_and_time(self):
        return {"gps": self.gps, "heading": self.heading, "gps_time": self.gps_time}

    # drone specific

    def distance_to_target(self, target_point):
        # points are long , lat
        return haversine(
            swap_lat_long(self.gps), swap_lat_long(target_point), unit=Unit.METERS
        )
        return np.linalg.norm(target_point - self.gps)

    def erase_logs(self):
        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                LOG_ERASE,
                0,  # set position
                0,  # param1
                0,  # param2
                0,  # param3
                0,  # param4
                0,  # 37.8047122,  # lat
                0,  # long,  # -122.4659164,  # lon
                0,  # 0,
            )

    # point long/lat
    def _initialize_motion_stop_state(self):
        self._motion_stop_requested = threading.Event()
        self._motion_hold_sent = threading.Event()
        self._motion_stop_reason = None

    def request_motion_stop(self, reason="capture stopped"):
        """Cooperatively abort planner motion; HOLD is retried until accepted."""

        if not hasattr(self, "_motion_stop_requested"):
            self._initialize_motion_stop_state()
        if not self._motion_stop_requested.is_set():
            self._motion_stop_reason = str(reason)
            logging.error("Vehicle motion stop requested: %s", self._motion_stop_reason)
        self.planner_should_move = False
        self._motion_stop_requested.set()

    def _try_enter_abort_hold(self):
        if not getattr(self, "_motion_stop_requested", threading.Event()).is_set():
            return False
        if self._motion_hold_sent.is_set():
            return True
        try:
            self.set_mode("HOLD")
        except Exception as error:
            logging.warning(
                "Capture abort HOLD is pending until MAVLink is available: %s",
                error,
            )
            return False
        self._motion_hold_sent.set()
        logging.warning(
            "Vehicle entered HOLD after capture abort: %s",
            self._motion_stop_reason,
        )
        return True

    def wait_for_abort_hold(self, *, timeout_seconds=2.0):
        """Bound the in-process HOLD attempt before hard process teardown."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._try_enter_abort_hold():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logging.error(
                    "Vehicle HOLD was not confirmed before the capture process "
                    "exit deadline"
                )
                return False
            time.sleep(min(0.1, remaining))

    # ------------------------------------------------------------- stall ---
    #
    # Detection is unconditional and identical in every configuration; the
    # crash_recovery flag only decides what happens after a stall is detected.
    # Detection is safe everywhere -- its worst outcome is handing a working
    # rover to a human -- while autonomous reversing is not, so that stays
    # opt-in per rover.

    def _initialize_stall_state(self):
        self._stall_anchor = None
        self._stall_anchor_at = None
        self._stall_last_escape_at = None
        self._stall_escape_left = True
        self._stall_escapes = 0
        self._stall_motor_was_off = False

    def _stall_radius(self):
        return getattr(self, "stall_progress_radius_m", STALL_PROGRESS_RADIUS_M)

    def _reset_stall_anchor(self, now=None):
        """Re-anchor at the current position; the stall clock restarts here."""

        self._stall_anchor = self.gps
        self._stall_anchor_at = time.monotonic() if now is None else now
        self._stall_last_escape_at = None
        # Which deadline the next window gets is decided per window, so the
        # "the motor was commanded off at some point in here" memory has to die
        # with the window it belongs to.
        self._stall_motor_was_off = False

    def _stall_verdict(self):
        """OK / ESCAPE / MANUAL for the current position and elapsed time.

        The anchor is reset ONLY by real motion, never merely by attempting a
        recovery -- that is what makes "stuck in reverse too" safe: a maneuver
        that moves the rover nowhere leaves the clock untouched.

        But the clock alone is not sufficient, and assuming it was cost a
        contract violation. An escape that DOES shift the rover a metre resets
        the anchor, which defers the escalation the escape is supposed to lead
        to -- so a rover creeping a metre per attempt escapes forever and never
        reaches the operator, while the runbook promised three attempts then
        MANUAL. Hence STALL_MAX_ESCAPES, counted since the last waypoint
        actually REACHED (see move_to_point) rather than since the last anchor
        reset; counting the latter would be circular, because the escape is
        what moves the anchor.
        """

        # getattr throughout, matching the _rc_triggers pattern: a
        # reconnected or hand-built Drone must not crash the motion loop over a
        # missing attribute.
        if not getattr(self, "crash_detect", True):
            return STALL_OK
        if not hasattr(self, "_stall_anchor"):
            self._initialize_stall_state()
        detect_seconds = getattr(self, "stall_detect_seconds", STALL_DETECT_SECONDS)
        manual_seconds = getattr(self, "stall_manual_seconds", STALL_MANUAL_SECONDS)
        parked_seconds = getattr(self, "stall_parked_seconds", STALL_PARKED_SECONDS)
        radius_m = getattr(
            self, "stall_progress_radius_m", STALL_PROGRESS_RADIUS_M
        )

        # A dead link looks exactly like a dead rover. When heartbeats stop,
        # self.gps freezes at its last value, so the anchor distance stays ~0
        # while the clock keeps running -- and armed/motor_active/mav_mode are
        # equally stale. Rover 4's 48s "stall" on 2026-08-05 fired in the same
        # second as a heartbeat timeout (report E1); the rover had not stopped,
        # the telemetry had, and an operator was sent after a phantom.
        #
        # Reset rather than freeze the clock: a rover that genuinely stalled
        # during an outage then needs detect_seconds of FRESH data before it
        # counts, which delays a true detection but can never invent one.
        # `> 0` matters and is not defensive noise: 0 means "no heartbeat has
        # ever been recorded", which is a Drone that does not track them (the
        # fake-drone and test paths) -- not a dead link. Treating it as dead
        # would silently disable stall detection everywhere. Same guard the
        # connection-health check uses.
        last_heartbeat = getattr(self, "last_heartbeat", 0) or 0
        stale_after = getattr(
            self,
            "reconnect_heartbeat_timeout",
            DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
        )
        if last_heartbeat > 0 and (time.time() - last_heartbeat) > stale_after:
            self._reset_stall_anchor()
            return STALL_OK

        # The gate used to include `self.motor_active`, and that single term
        # made the watchdog structurally incapable of catching the failure it
        # was written for. motor_active is False EXACTLY when the autopilot
        # commands neutral throttle (handle_SERVO_OUTPUT_RAW: servo1_raw ==
        # servo3_raw == 1500) -- and commanded-neutral throttle IS the
        # signature. ModeGuided::_enter() calls start_stop()
        # (Rover/mode_guided.cpp:3-20, :392), which sets SubMode::Stop, and
        # stop_vehicle() (Rover/mode.cpp:336-363) then pins both channels at
        # 1500 while the rover sits on a destination the firmware has thrown
        # away. start_stop() has exactly two call sites, both in _enter(), so
        # every re-entry into GUIDED -- an operator's CH8 handback, most of all
        # -- produces it, on every ArduRover >= 4.2.0. The old gate therefore
        # re-anchored its own clock on precisely the state it was watching for:
        # zero stall lines across rover 1's 15.6-minute and rover 4's
        # 17.2-minute dead tails on 2026-08-07, while the same code fired
        # correctly eleven times that night on motor_active=True jams.
        #
        # So progress is now read from GPS alone, and the gate asks only
        # whether the rover is SUPPOSED to be going somewhere:
        #   armed + GUIDED            -- the Pi's commands can take effect at
        #                                all. This is also the live half of
        #                                planner_control_loss_reason(): if the
        #                                operator has CH8'd to MANUAL the mode
        #                                is not GUIDED and nothing here fires.
        #   _destination_outstanding  -- move_to_point issued a destination and
        #                                has not reached it, so standing still
        #                                is a fault rather than a rover with
        #                                nothing to do.
        # (Connection health, the third part of that question, is already
        # covered by the heartbeat-staleness guard above.)
        #
        # Deliberately NOT `planner_in_control`: that flag is the run_planner
        # LATCH, and run_planner clears it before move_to_home() drives the
        # rover back (mavlink_radio_collection waits on planner_is_still_driving
        # precisely so the two do not fight). Gating on it would switch the
        # watchdog off for the entire return leg -- an unattended drive across
        # the same field, where a jam has nobody watching it and move_to_home
        # would just burn its 300 s max_wait. _destination_outstanding is set by
        # move_to_point, which is the ONE function that drives to a point, so it
        # covers the planner and the return home alike.
        #
        # Servo output has not been discarded -- it decides how long to wait,
        # not whether to look. See the threshold below.
        driving_to_a_destination = (
            self.armed
            and self.mav_mode == "ROVER_MODE_GUIDED"
            and getattr(self, "_destination_outstanding", False)
        )
        if not driving_to_a_destination:
            self._reset_stall_anchor()
            return STALL_OK

        now = time.monotonic()
        if self._stall_anchor is None or self._stall_anchor_at is None:
            self._reset_stall_anchor(now=now)
            return STALL_OK
        travelled = self.distance_to_target(self._stall_anchor)
        if travelled > radius_m:
            if travelled > STALL_RECOVERED_M:
                # Further than any escape could have shoved it, so the rover is
                # genuinely under way again and has earned a clean slate.
                self._stall_escapes = 0
            self._reset_stall_anchor(now=now)
            return STALL_OK

        # Two deadlines, chosen by whether the motor was EVER commanded off
        # since the anchor -- not by what it is doing this instant. Sticky is
        # the whole point: a waypoint pivot shows motor-off for several seconds
        # (measured 6.00-9.66 s on the clean 2026-08-07 captures) and then
        # drives away, and a "current sample" rule would hand that window the
        # 10 s deadline the moment the throttle came back, firing on a rover
        # accelerating out of a legitimate turn. One-way and therefore safe:
        # any motor-off sample buys the whole window the longer deadline, and
        # real motion clears it for free by resetting the anchor.
        #
        # A window that was motor-on throughout keeps the original 10 s, so by
        # construction this change cannot make a motor-on jam fire LATER than
        # it used to. Note what that is and is not: it is an argument from the
        # code path, NOT a measurement. The offline replay built for this fix
        # reproduced 0 of the 11 real 2026-08-07 firings in its "before" arm
        # (and invented one on a capture whose log has no STALL line at all),
        # so the replay can size the parked threshold -- which only needs the
        # dwell distribution -- but it cannot testify about firing times.
        # Anyone claiming those eleven are bit-for-bit untouched needs a
        # harness that can first reproduce them.
        if not self.motor_active:
            self._stall_motor_was_off = True
        detect_threshold = (
            parked_seconds
            if getattr(self, "_stall_motor_was_off", False)
            else detect_seconds
        )

        stalled_seconds = now - self._stall_anchor_at
        if stalled_seconds < detect_threshold:
            return STALL_OK
        if not getattr(self, "crash_recovery", False):
            return STALL_MANUAL
        if stalled_seconds >= manual_seconds:
            return STALL_MANUAL
        if getattr(self, "_stall_escapes", 0) >= STALL_MAX_ESCAPES:
            return STALL_MANUAL
        if (
            self._stall_last_escape_at is None
            or now - self._stall_last_escape_at >= detect_seconds
        ):
            self._stall_last_escape_at = now
            return STALL_ESCAPE
        return STALL_OK

    def stalled_seconds(self):
        if getattr(self, "_stall_anchor_at", None) is None:
            return 0.0
        return time.monotonic() - self._stall_anchor_at

    def reverse(self, reversed_travel):
        """Tell the waypoint navigator to drive to destinations backwards.

        MAV_CMD_DO_SET_REVERSE is a command, not a parameter -- Rover routes it
        to Mode::set_reversed() -> AR_WPNav, a runtime flag with no EEPROM
        backing. Mode::enter() clears it on every mode change, but within a
        mode it persists, so callers must clear it on every exit path or the
        next leg is driven backwards.
        """

        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_REVERSE,
                0,
                1 if reversed_travel else 0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def status_text(self, text, severity=mavutil.mavlink.MAV_SEVERITY_WARNING):
        """Surface a decision to any attached GCS, not just the Pi's log file.

        Never raises: a missing GCS must not break motion control.
        """

        try:
            with self._command_connection() as connection:
                connection.mav.statustext_send(severity, text[:50].encode())
        except Exception as error:  # pragma: no cover - best effort only
            logging.debug("statustext_send failed: %s", error)

    def _wait_for_escape_progress(self, deadline_seconds):
        """Drive a leg until it covers ground or runs out of time.

        Time-bounded rather than success-checked, on purpose: a leg that moves
        the rover nothing simply leaves the stall clock running.
        """

        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            self.raise_if_connection_failed()
            if self._motion_stop_requested.is_set():
                return False
            if self.distance_to_target(self._stall_anchor) > self._stall_radius():
                return True
            time.sleep(0.1)
        return False

    def _escape_jam(self):
        """Back out of a jam, then step off the axis we backed out along.

        Leg 1 reverses STALL_ESCAPE_DISTANCE_M. Leg 2 aims the same distance
        perpendicular to the MEASURED stall->reverse axis rather than to the
        compass heading, so it reflects the displacement actually achieved.
        Returns False only if a capture abort landed mid-maneuver.
        """

        stall_point = self.gps
        stalled = self.stalled_seconds()
        self._stall_escapes = getattr(self, "_stall_escapes", 0) + 1
        logging.warning(
            "STALL: no progress for %.0fs at %s; reversing out"
            " (attempt %d/%d, side=%s)",
            stalled,
            str(stall_point),
            self._stall_escapes,
            STALL_MAX_ESCAPES,
            "left" if self._stall_escape_left else "right",
        )
        self.status_text(f"STALL {stalled:.0f}s: reversing out")

        try:
            self.reverse(True)
            behind = stall_point + meters_to_degrees(
                *(
                    STALL_ESCAPE_DISTANCE_M
                    * bearing_to_unit_vector(self.heading + 180)
                ),
                stall_point[1],
            )
            self.reposition(lat=behind[1], long=behind[0])
            if not self._wait_for_escape_progress(STALL_ESCAPE_DRIVE_SECONDS):
                if self._motion_stop_requested.is_set():
                    return False
        finally:
            # Unconditional: leaving the flag set would drive the entire next
            # leg backwards. The HOLD is not cosmetic either -- after reversing
            # the speed integrator sits well negative and would fight the next
            # forward command, and get_throttle_out_speed() resets it for free
            # once speed control goes inactive.
            self.reverse(False)
            self.set_mode("HOLD")
            time.sleep(STALL_ESCAPE_SETTLE_SECONDS)
            # GUIDED must come back, and this is not optional bookkeeping. In
            # HOLD the vehicle ignores guided targets, so leaving it there makes
            # the lateral leg a no-op, makes every later reposition a no-op, and
            # -- worst -- fails _stall_verdict's GUIDED gate, which resets the
            # anchor and puts MANUAL permanently out of reach. The maneuver would
            # disable itself after one attempt and park the rover. Observed in
            # SITL as mode timelines ending [..., 'GUIDED', 'HOLD'] with faint
            # forward output (servo1=1503, servo3=1504) instead of reverse.
            self.set_mode("GUIDED")

        reverse_point = self.gps
        if self.distance_to_target(self._stall_anchor) > self._stall_radius():
            logging.warning("STALL: reversing freed the rover; skipping lateral leg")
            return True

        offset_m = degrees_to_meters(*(reverse_point - stall_point), reverse_point[1])
        axis_m = float(np.linalg.norm(offset_m))
        if axis_m >= STALL_AXIS_MIN_M:
            unit = offset_m / axis_m
        else:
            # The reverse leg achieved nothing usable, so there is no measured
            # axis to be orthogonal to; fall back to the heading.
            unit = bearing_to_unit_vector(self.heading + 180)
        if self._stall_escape_left:
            orthogonal = np.array([-unit[1], unit[0]])
        else:
            orthogonal = np.array([unit[1], -unit[0]])
        self._stall_escape_left = not self._stall_escape_left

        lateral = reverse_point + meters_to_degrees(
            *(STALL_ESCAPE_DISTANCE_M * orthogonal), reverse_point[1]
        )
        logging.warning(
            "STALL: reverse covered %.1fm; stepping %.0fm off that axis to %s",
            axis_m,
            STALL_ESCAPE_DISTANCE_M,
            str(lateral),
        )
        self.reposition(lat=lateral[1], long=lateral[0])
        self._wait_for_escape_progress(STALL_ESCAPE_DRIVE_SECONDS)
        return not self._motion_stop_requested.is_set()

    def _hand_over_to_manual(self):
        """Put the rover in MANUAL and wait for the operator to hand it back.

        MODE_CH 8 only re-applies on switch MOVEMENT, so a Pi-set MANUAL
        persists until a human physically moves CH8 -- that is the interlock.
        Deliberately does not touch _motion_stop_requested: a stall is not a
        capture abort and must not send HOLD.
        """

        stalled = self.stalled_seconds()
        logging.error(
            "STALL: no progress for %.0fs at %s; handing control to the operator",
            stalled,
            str(self.gps),
        )
        self.status_text(
            f"STALL {stalled:.0f}s: handing to MANUAL",
            severity=mavutil.mavlink.MAV_SEVERITY_CRITICAL,
        )
        self.buzzer(tones["failure"])

        for _ in range(STALL_MANUAL_ATTEMPTS):
            self.set_mode("MANUAL")
            time.sleep(STALL_MANUAL_RETRY_SECONDS)
            if self.mav_mode == "ROVER_MODE_MANUAL":
                break
        else:
            logging.error(
                "STALL: MANUAL was not confirmed after %d attempts; mode is %s",
                STALL_MANUAL_ATTEMPTS,
                self.mav_mode,
            )

        while self.mav_mode != "ROVER_MODE_GUIDED":
            self.raise_if_connection_failed()
            if self._motion_stop_requested.is_set():
                self._try_enter_abort_hold()
                return False
            logging.info(
                "STALL: waiting for the operator to hand control back (CH8 -> guided)"
            )
            self.buzzer(tones["wait"])
            time.sleep(2)

        if not self.armed:
            self.arm()
            time.sleep(0.1)
        self._reset_stall_anchor()
        self._stall_escapes = 0
        logging.info("STALL: operator returned control; resuming")
        return True

    def _handle_stall(self, verdict):
        """Apply a stall verdict. Returns the move outcome, or None to carry on."""

        if verdict == STALL_ESCAPE:
            if not self._escape_jam():
                return MOVE_ABORTED
            return MOVE_SKIPPED
        if verdict == STALL_MANUAL:
            if not self._hand_over_to_manual():
                return MOVE_ABORTED
            return MOVE_SKIPPED
        return None

    def move_to_point(self, point, log_interval=5):
        # logging.info(f"GPS: current position {self.gps} target position {str(point)}")
        self.raise_if_connection_failed()
        if self._motion_stop_requested.is_set():
            self._try_enter_abort_hold()
            self._destination_outstanding = False
            return MOVE_ABORTED
        self.reposition(lat=point[1], long=point[0])
        # From here until the target is reached, standing still is a fault and
        # not merely an idle rover -- which is what lets _stall_verdict stop
        # asking the servos whether the vehicle is meant to be moving. Set
        # AFTER reposition() so a raising send cannot arm the watchdog against
        # a destination the vehicle was never given.
        self._destination_outstanding = True
        # The stall anchor is deliberately NOT reset here. It is owned by
        # _stall_verdict and reset only by real motion, so the clock survives
        # across the waypoints an escape abandons -- which is what makes the
        # escalation to MANUAL reachable at all.
        last_message = None
        # ArduPilot THROWS THIS DESTINATION AWAY on every re-entry into GUIDED.
        # ModeGuided::_enter() calls start_stop() (Rover/mode_guided.cpp:3-20),
        # start_stop() sets SubMode::Stop (:392), and stop_vehicle() pins both
        # throttle channels at exactly 1500 (Rover/mode.cpp:336-363).
        # start_stop() has no call site outside _enter(), so SubMode::Stop is
        # reachable ONLY this way -- and re-entering GUIDED is exactly what the
        # operator's CH8 switch does on the way back from a MANUAL excursion.
        # Behaviour of every ArduRover >= 4.2.0, not a rover-specific fault.
        #
        # So the mode is latched and the destination re-issued on every
        # observed transition INTO GUIDED. RO1 on 2026-08-07 is what this
        # costs otherwise: one waypoint held for 939 s after a takeover, 1101
        # of 3000 records written parked, and nothing in the log to show for it
        # -- armed, GUIDED, EKF healthy, motionless.
        #
        # The stall watchdog cannot cover this, which is why the recovery lives
        # here rather than there: `driving` above requires motor_active, and
        # motor_active is False precisely when the autopilot commands neutral
        # throttle, so this failure resets the watchdog's own anchor on every
        # tick and its clock never starts.
        #
        # EDGE-triggered, deliberately, following the MOVE_SKIPPED re-issue in
        # move_to_home rather than inventing a periodic refresh. An
        # unconditional 1 Hz refresh was measured and costs no ground speed in
        # SITL (14.949 vs 14.944 m/s steady state; 4.953 vs 4.950 s per 60 m
        # leg, n=4 each -- numbers in docs/learnings.md), so the case against it
        # is not speed: it is that SITL's link is a local TCP socket while the
        # fleet's is a shared 915 MHz radio, where a command per second is real
        # airtime spent to buy exactly what one command per handback buys.
        last_mode = self.mav_mode
        while self.distance_to_target(point) > self.tolerance_in_m:
            self.raise_if_connection_failed()
            if self._motion_stop_requested.is_set():
                self._try_enter_abort_hold()
                logging.warning(
                    "Aborted active waypoint after capture failure; target=%s",
                    point,
                )
                # HOLD was just requested, so nothing is driving anywhere; a
                # watchdog left armed here would report the deliberate stop.
                self._destination_outstanding = False
                return MOVE_ABORTED
            # Outside the distance_finder guard below on purpose, same as the
            # stall check: the field runs --no-ultrasonic, so anything inside
            # that guard is dead code on a real rover.
            observed_mode = self.mav_mode
            if observed_mode != last_mode:
                if observed_mode == "ROVER_MODE_GUIDED":
                    logging.warning(
                        "Re-entered GUIDED (was %s); re-issuing the destination "
                        "%s, which ArduPilot discarded on mode entry",
                        last_mode,
                        point,
                    )
                    self.reposition(lat=point[1], long=point[0])
                last_mode = observed_mode
            # Outside the distance_finder guard below on purpose: the stall
            # watchdog must work with --no-ultrasonic, unlike the "Are we
            # sleeping somwehere?" heuristic it supersedes.
            outcome = self._handle_stall(self._stall_verdict())
            if outcome is not None:
                return outcome
            if last_message is None or time.time() - last_message > log_interval:
                logging.info(
                    f"\tDist (m) to target {str(self.distance_to_target(point))} {self.motor_active} {self.mav_mode}"
                )
                last_message = time.time()
            # safety
            if self.distance_finder is not None:
                distance = self.distance_finder.distance
                collision_soon = (
                    (not self.disable_distance_finder)
                    and self.distance_finder is not None
                    and distance < 30
                )
                if self.mav_mode == "ROVER_MODE_GUIDED":
                    if self.armed and collision_soon:
                        logging.info(f"AVOIDING COLLISION! {distance}")
                        self.disarm()
                        time.sleep(2)
                    elif not self.armed and not collision_soon:
                        logging.info("RESUMING FROM NEAR COLLISION!")
                        self.arm()
                    elif self.armed and not self.motor_active:
                        logging.info("Are we sleeping somwehere?")
                        self.reposition(lat=point[1], long=point[0])
                        time.sleep(0.5)
            time.sleep(0.1)
        logging.info(f"\tReached target {str(point)} , current gps {str(self.gps)}")
        # Arriving IS progress, so the clock restarts here -- but note that a
        # SKIPPED waypoint deliberately does not reset it (see _handle_stall).
        # That asymmetry is the whole point: without the reset, the slow final
        # approach plus the stop-and-go of a waypoint transition reads exactly
        # like a stall and the watchdog fires on a healthy rover (observed in
        # SITL: "Reached target" then "no progress for 3s" on the next leg).
        # Without the skip case NOT resetting, escalation to MANUAL could never
        # be reached, because every abandoned waypoint would buy a fresh window.
        self._reset_stall_anchor()
        self._stall_escapes = 0
        # Nothing is outstanding until run_planner issues the next waypoint,
        # and the gap between legs is exactly where a rover is allowed to sit
        # still. MOVE_SKIPPED deliberately does NOT clear it, for the same
        # reason it does not reset the anchor.
        self._destination_outstanding = False
        return MOVE_REACHED

    def run_planner(self):
        # self.single_operation_mode_on()
        logging.info("Start planner...")
        self._motion_stop_requested.clear()
        self._motion_hold_sent.clear()
        self._motion_stop_reason = None
        self.planner_should_move = True

        # self.single_operation_mode_on()
        # logging.info("SINGLE OPERATION MODE")
        # Per-rover rest offset lives on the planner so home, the S5 rendezvous,
        # the post-run park and the RTL destination cannot drift apart.
        home = self.planner.get_home_point()

        self.single_operation_mode_on()
        # self.connection.waypoint_clear_all_send()
        # logging.info("SINGLE OPERATION MODE 2")

        self.set_home(lat=home[1], long=home[0])
        self.single_operation_mode_off()

        # self.single_operation_mode_off()
        # drone.request_home()
        logging.info("Planer main loop...")
        while not self.drone_ready:
            logging.info(
                f"Planner wait for drone ready: gps:{str(self.gps)} , ekf:{str(self.ekf_healthy)}"
            )
            time.sleep(2)

        if self.ignore_mode:
            time.sleep(2)
            while self.planner_should_move:
                self.planner_in_control = True
                time.sleep(1)

        else:
            self._wait_for_mode(
                "ROVER_MODE_MANUAL",
                tones["wait"],
                "waiting for rover to move into manual mode...",
            )
            self._wait_for_mode(
                "ROVER_MODE_GUIDED",
                tones["ready"],
                "waiting for rover to move into guided mode...",
            )

            if not self.armed:
                self.arm()
                time.sleep(0.1)
            logging.info("Planner starting to issue move commands...")

            # BEFORE the first move, not after it. The collector gates the start
            # of recording on this flag, and it used to be assigned only at the
            # BOTTOM of the waypoint loop below -- so it stayed False through the
            # drive-to-home and the whole first waypoint. Rover 1 on 2026-08-07
            # logged "Planner starting to issue move commands" at 01:12:42 and
            # was still logging "Waiting for drone to start moving" at 01:14:27:
            # two minutes of driving with nothing recorded.
            self.planner_in_control = True

            if self.move_to_point(home) == MOVE_ABORTED:
                self.planner_in_control = False
                return
            time.sleep(2)

            # drone is now ready
            # point is long, lat
            yp = self.planner.yield_points()
            point = None
            logging.info(f"About to enter planned main loop {self.planner}")
            # breakpoint()
            while self.planner_should_move:
                next_point = next(yp)
                # logging.info(f"In planner main loop {next_point} {point}")
                if (
                    point is not None
                    and np.isclose(next_point, point, atol=1e-10, rtol=1e-10).all()
                ):
                    time.sleep(0.2)
                else:
                    point = next_point
                    outcome = self.move_to_point(point)
                    if outcome == MOVE_ABORTED:
                        break
                    if outcome == MOVE_SKIPPED:
                        # The rover jammed on this waypoint. Drop one more so
                        # it does not immediately drive back at whatever it
                        # just backed out of -- for `bounce` that is a whole
                        # leg, and for `circle` about a tenth of a lap.
                        next(yp)
                        point = None
                self.planner_in_control = True
                # time.sleep(2)

        self.planner_in_control = False
        # A MOVE_SKIPPED on the last waypoint leaves a destination outstanding
        # that nobody is driving to any more. Harmless today -- _stall_verdict
        # is only reached from inside move_to_point -- but a stale True here
        # would arm the watchdog against a parked, finished rover the moment
        # anything else consults it.
        self._destination_outstanding = False

    def _wait_for_mode(self, expected_mode, tone, message, poll_interval=2.0):
        """Wait for operator mode changes without making the buzzer critical."""
        while self.mav_mode != expected_mode:
            self.raise_if_connection_failed()
            time.sleep(poll_interval)
            logging.info(message)
            self.buzzer(tone)

    def get_cmd(self, cmd):
        v = getattr(mavutil.mavlink, cmd)
        self.mav_cmd_name2num[cmd] = v
        self.mav_cmd_num2name[v] = cmd
        return v

    def move_to_home(self, max_wait=300):
        """
        Sets the drone mode to RTL and blocks until the drone has reached home
        or the max_wait time (in seconds) has elapsed.
        Returns True if the drone reaches home within max_wait, else False.
        """
        # Make sure we know the home location
        if not hasattr(self, "home"):
            logging.error("Home location is not known. Set home first.")
            return False

        # Switch to RTL mode
        # self.set_rtl_mode()
        # logging.info("RTL mode set. Waiting for the drone to reach home...")
        # monotonic, not time.time(): GPS time sync moves the wall clock on the
        # rover (see handle_SYSTEM_TIME), which would corrupt this deadline.
        deadline = time.monotonic() + max_wait
        while True:
            outcome = self.move_to_point(self.home)
            if outcome == MOVE_ABORTED:
                return False
            if outcome == MOVE_REACHED:
                break
            # MOVE_SKIPPED: a stall escape or MANUAL handback ran. Unlike the
            # planner there is no next waypoint to fall through to, so re-issue
            # home until it is reached or the deadline passes.
            if time.monotonic() >= deadline:
                logging.warning("Timed out driving home after a stall recovery.")
                return False

        while time.monotonic() < deadline:
            dist = self.distance_to_target(self.home)
            logging.debug(f"Distance to home: {dist:.2f} meters")
            if dist <= self.tolerance_in_m:
                logging.info("Drone has reached home!")
                return True
            time.sleep(1)

        logging.warning("Timed out waiting for the drone to reach home.")
        return False

    def set_rtl_mode(self):
        """
        Sets the drone mode to RTL (Return to Launch).
        """
        # According to the custom_mode_mapping, mode 11 is ROVER_MODE_RTL.
        # We can use the underlying MAV_CMD_DO_SET_MODE command to switch modes.
        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                self.get_cmd("MAV_CMD_DO_SET_MODE"),
                0,  # Confirmation
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,  # Base mode flags
                ROVER_MODE_RTL,  # Custom mode index for RTL
                0,
                0,
                0,
                0,
                0,
            )

    def run_compass_calibration(self):
        with self._command_connection() as connection:
            message = connection.mav.command_long_encode(
                connection.target_system,  # Target system ID
                connection.target_component,  # Target component ID
                self.get_cmd("MAV_CMD_DO_START_MAG_CAL"),  # ID of command to send
                0,
                3,  # first two
                0,
                1,
                0,
                1,
                0,
                0,
            )
            connection.mav.send(message)

    def request_home(self):
        with self._command_connection() as connection:
            message = connection.mav.command_long_encode(
                connection.target_system,  # Target system ID
                connection.target_component,  # Target component ID
                self.get_cmd("MAV_CMD_GET_HOME_POSITION"),  # ID of command to send
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

            # msg = connection.mav.command_long_encode(
            #    0, 0, mavutil.mavlink.MAV_CMD_GET_HOME_POSITION, 0, 0, 0, 0, 0, 0, 0, 0
            # )

            # Send the COMMAND_LONG
            connection.mav.send(message)

    def set_home(self, lat, long):
        """Set the vehicle home (and therefore the RTL destination) to lat/long.

        param1 MUST be 0 = "use specified location". With param1=1 ArduPilot
        reads it as "use current location" and silently DISCARDS lat/long —
        verified in SITL, where home stayed at the spawn point 14.14 m away
        from the requested position.

        Sent as COMMAND_INT (degrees x 1e7, integer) rather than COMMAND_LONG:
        COMMAND_LONG carries lat/long as float32, which quantizes to ~0.67 m at
        these longitudes — larger than the per-rover rest offsets themselves.
        This mirrors reposition(), which already uses command_int_send.
        """
        with self._command_connection() as connection:
            connection.mav.command_int_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL,
                self.get_cmd("MAV_CMD_DO_SET_HOME"),
                0,  # current
                0,  # autocontinue
                0,  # param1: 0 = use SPECIFIED location (1 would discard lat/long)
                0,  # param2
                0,  # param3
                0,  # param4
                int(lat * 1e7),
                int(long * 1e7),
                0,
            )
        # self.ack("COMMAND_ACK")
        self.home = np.array([long, lat])  # Store home for later use

    def turn_off_hardware_safety(self):
        with self._command_connection() as connection:
            connection.mav.set_mode_send(
                connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY,
                0,
            )

    def reboot(self, force=False, hold_in_bootloader=False):
        with self._command_connection() as connection:
            if not force:
                connection.reboot_autopilot()
                return
            if hold_in_bootloader:
                param1 = 3
            else:
                param1 = 1
            param6 = 20190226
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                0,
                param1,
                0,
                0,
                0,
                0,
                param6,
                0,
            )

    def disarm(self, force=False):
        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                self.get_cmd("MAV_CMD_COMPONENT_ARM_DISARM"),
                0,
                0,
                1 if force else 0,
                0,
                0,
                0,
                0,
                0,
            )
        # self.ack("COMMAND_ACK")

    def arm(self, force=False):
        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                self.get_cmd("MAV_CMD_COMPONENT_ARM_DISARM"),
                0,
                1,
                1 if force else 0,
                0,
                0,
                0,
                0,
                0,
            )
        # self.ack("COMMAND_ACK")

    def reposition(self, lat, long, change_mode=None):
        """Send the vehicle to one destination, and let it switch to GUIDED.

        The flags word is the point (see REPOSITION_CHANGE_MODE): without it
        the autopilot answers MAV_RESULT_DENIED to every waypoint issued from
        HOLD, which is where the EKF failsafe, _escape_jam and a capture abort
        all leave the rover. Since handle_COMMAND_ACK was a bare `pass` at the
        time, every one of those refusals was silent.

        MANUAL is the one mode this will NOT drag the vehicle out of, and that
        exclusion is a safety interlock rather than caution. MANUAL means an
        operator is holding CH8 and driving; ArduPilot's MODE_CH only re-applies
        on switch MOVEMENT (the mechanism _hand_over_to_manual depends on), so a
        GCS-forced GUIDED would take the rover out of the operator's hands and
        leave it there until they thought to flick the switch twice. A waypoint
        refused while a human is driving is the correct outcome -- and now a
        logged one.

        `change_mode` overrides the decision for callers that know better;
        None means decide from the mode the heartbeat last reported.
        """
        if change_mode is None:
            change_mode = self.mav_mode != "ROVER_MODE_MANUAL"
        # self.connection.mav.command_long_send(
        #    self.connection.target_system,
        #    self.connection.target_component,
        #    self.get_cmd("MAV_CMD_DO_REPOSITION"),
        #    0,
        #    -1,  # default ground speed
        #    0,  # reposition flags
        #    0,  # loiter radius, 0 is ignore
        #    math.nan,  # yaw
        #    lat,
        #    long,
        #    0.0,  # altitude
        # )
        # self.mission_item_reached = False
        with self._command_connection() as connection:
            connection.mav.command_int_send(
                connection.target_system,
                connection.target_component,
                0,  # frame
                self.get_cmd("MAV_CMD_DO_REPOSITION"),  # cmd
                0,  # not used
                0,  # not used
                -1,  # default ground speed
                REPOSITION_CHANGE_MODE if change_mode else 0,  # reposition flags
                0,  # loiter radius, 0 is ignore
                math.nan,  # yaw
                int(lat * 1e7),
                int(long * 1e7),
                0.0,  # altitude
            )

    def do_mission(self, restart_mission=True):
        with self._command_connection() as connection:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                self.get_cmd("MAV_CMD_DO_SET_MISSION_CURRENT"),
                0,
                -1,
                1 if restart_mission else 0,
                0,
                0,
                0,
                0,
                0,
            )
        # self.ack("COMMAND_ACK")

    def ack(self, keyword):
        with self._command_connection() as connection:
            return connection.recv_match(type=keyword, blocking=True, timeout=5) is None

    def single_operation_mode_on(self):
        assert not self.single_operation
        with self.single_condition:
            self.single_operation = True  # request single operation mode
            while self.message_loop:
                self.single_condition.wait()
            return True

    def single_operation_mode_off(self, turn_on_messages=True):
        assert self.single_operation
        with self.single_condition:
            self.single_operation = False  # request single operation mode
            if turn_on_messages:
                self.message_loop = turn_on_messages
                with self.message_condition:
                    self.message_condition.notify_all()

    def process_messages(self):
        with self.message_condition:
            while True:  # try not to leave context too often
                # if we are not supposed to run  message loop or the single operation mode is requested
                # chill out
                if self.single_operation:
                    self.message_loop = False  # lets chill for a bit
                    with self.single_condition:
                        self.single_condition.notify_all()
                    while not self.message_loop:
                        self.message_condition.wait()
                try:
                    with self._connection_lock:
                        connection = self.connection
                    msg = connection.recv_match(blocking=True, timeout=0.5)
                    if (
                        self.connection_healthy
                        and self.last_heartbeat > 0
                        and time.time() - self.last_heartbeat
                        > self.reconnect_heartbeat_timeout
                    ):
                        raise MavlinkConnectionError(
                            "flight-controller heartbeat timed out"
                        )
                except Exception as error:
                    if not self._recover_connection(error):
                        return
                    continue
                try:
                    self.process_message(msg)
                except Exception as error:
                    message_type = msg.get_type() if msg is not None else "NO_MESSAGE"
                    failure = MavlinkMessageHandlingError(
                        "MAVLink message handler failed: "
                        f"message_type={message_type} "
                        f"error={type(error).__name__}: {error}"
                    )
                    # Preserve the original traceback for the capture's single
                    # owning traceback without printing a second one here.
                    failure.__cause__ = error
                    self.connection_failure = failure
                    self._connection_healthy.clear()
                    logging.error("%s", failure)
                    return

    def handle_HOME_POSITION(self, msg):
        # HOME_POSITION message fields are in 1e7 scaled integers
        self.launch_home = np.array([msg.longitude / 1e7, msg.latitude / 1e7])
        logging.debug(f"Launch home position set to: {self.launch_home}")

    def handle_NAV_CONTROLLER_OUTPUT(self, msg):
        # breakpoint()
        # self.target_
        pass

    def handle_GLOBAL_POSITION_INT(self, msg):
        self.lat = msg.lat / 1e7
        self.long = msg.lon / 1e7
        self.gps = np.array([self.long, self.lat])
        self.heading = msg.hdg / 100

    def handle_GPS_RAW_INT(self, msg):
        self.gps_satellites = msg.satellites_visible
        self.gps_fix_type = gps_fix_type_str_to_num[msg.fix_type]

    def handle_EKF_STATUS_REPORT(self, msg):
        if msg.flags & self.healthy_ekf_flag == self.healthy_ekf_flag:
            self.ekf_healthy = True
        else:
            self.ekf_healthy = False

    def handle_COMMAND_ACK(self, msg):
        """Record what the vehicle said about a command, and shout if it refused.

        This was a bare `pass`, which made a refused command and an obeyed one
        the same observable event -- silence -- for everything this controller
        sends: every waypoint, every arm(), every mode-adjacent command. The
        DENIED repositions that defect F3 is about could have been read off the
        wire from the first flight; nobody was listening.

        Refusals are logged, not raised. This runs on the receive loop, where an
        exception tears the link down (see process_messages), and a rejected
        buzzer must never do that. The result is also kept as STATE so a caller
        can check its own command afterwards rather than scrape the log.
        """
        if not hasattr(self, "command_results"):
            # Same defensiveness as _initialize_stall_state's callers: a
            # hand-built or partially constructed Drone must not kill the
            # receive loop over a missing attribute.
            self.command_results = {}
        name = mav_command_name(msg.command)
        result = mav_result_name(msg.result)
        self.command_results[name] = CommandResult(
            command=name, result=result, at=time.time()
        )
        if msg.result not in COMMAND_RESULTS_OK:
            logging.error("Vehicle refused %s: %s", name, result)

    def handle_PARAM_VALUE(self, msg):
        # print("param", msg.param_id)
        self.params[msg.param_id] = msg.param_value
        # {
        #    "value": msg.param_value,
        #    "index": msg.param_index,
        #    "type": msg.param_type,
        # }
        self.param_count = msg.param_count

    def handle_HEARTBEAT(self, msg, log_interval=5):
        self.last_heartbeat = time.time()
        self.connection_failure = None
        self._connection_healthy.set()
        self.mav_states = lookup_exact(msg.system_status, mav_states_list)
        self.mav_mode = custom_mode_mapping[
            msg.custom_mode
        ]  # self.mav_mode_mapping_num2name[msg.base_mode]
        self.armed = (
            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        ) == mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        if not self.drone_ready:
            mav_state_check = (
                "MAV_STATE_STANDBY" in self.mav_states
                or "MAV_STATE_ACTIVE" in self.mav_states
            )
            gps_check = self.gps is not None and self.gps[0] != 0
            gps_healthy = "MAV_SYS_STATUS_SENSOR_GPS" in self.sensors_health
            guided_mode = self.mav_mode == "ROVER_MODE_GUIDED"
            if (
                self.last_heartbeat_log is None
                or time.time() - self.last_heartbeat_log > log_interval
            ):
                logging.info(
                    f"HEARTBEAT: mav_state:{mav_state_check}"
                    + f"gps:{gps_check}({self.gps_satellites}sats,{self.gps_fix_type}),"
                    + f"gps_healthy:{gps_healthy}, guided_mode:{guided_mode}, ekf:{self.ekf_healthy}"
                )
                self.last_heartbeat_log = time.time()
            if (
                mav_state_check
                and gps_check
                and gps_healthy
                # and guided_mode
                and self.ekf_healthy
            ):
                logging.info(
                    "Navigation ready for capture: connection_epoch=%d "
                    "gps=%s satellites=%s fix=%s gps_healthy=%s ekf=%s",
                    self.connection_epoch,
                    self.gps,
                    self.gps_satellites,
                    self.gps_fix_type,
                    gps_healthy,
                    self.ekf_healthy,
                )
                self.drone_ready = True
        if getattr(self, "_motion_stop_requested", None) is not None:
            self._try_enter_abort_hold()

    def handle_SYSTEM_TIME(self, msg):
        self.gps_time = msg.time_unix_usec / 1e6  # time in seconds since epoch
        self.time_since_boot = msg.time_boot_ms / 1e3

    def handle_SYS_STATUS(self, msg):
        self.sensors_present = lookup_bits(
            msg.onboard_control_sensors_present, sensors_list
        )
        self.sensors_enabled = lookup_bits(
            msg.onboard_control_sensors_enabled, sensors_list
        )
        self.sensors_health = lookup_bits(
            msg.onboard_control_sensors_health, sensors_list
        )

    def handle_STATUSTEXT(self, msg):
        if "SmartRTL" in msg.text and "space" in msg.text.lower():
            logging.warning(
                "ARDUPILOT_SMARTRTL_ADVISORY: %s:%s:%s "
                "(flight-controller breadcrumb capacity, not Pi disk space)",
                self.connection.target_system,
                self.connection.target_component,
                msg.text,
            )
            return
        logging.info(
            f"{self.connection.target_system}:{self.connection.target_component}:{msg.text}"
        )

    def handle_RC_CHANNELS_SCALED(self, msg):
        # print(msg.to_dict())
        pass

    def _rc_shutdown(self, msg):
        """Safe the vehicle, then power the Pi down. Every wait is bounded.

        Vehicle state is an ACTION here, not a veto. It used to be a veto --
        `permitted = not armed and not motor_active` -- which meant the switch
        did nothing during a capture, because the planner arms the vehicle and
        re-arms it within 0.1s of any operator disarm. Worse, the veto branch
        cleared the release bit, so a press while armed poisoned the *next*
        attempt too. The operator's fallback when a shutdown switch ignores
        them is pulling the battery on a live rover, which is the mechanism
        behind the fleet's 32 unclean shutdowns (field report 2026-08-05, D1/D2)
        and every unfinalised capture that came with them.

        So nothing below may prevent the poweroff. HOLD and disarm are
        attempted, bounded, and logged; whatever their outcome, the Pi goes
        down. Refusing to halt is the failure being removed, not a safe default.
        """

        logging.warning(
            "RC shutdown accepted: ch9_raw=%d armed=%s motor_active=%s mode=%s",
            msg.chan9_raw,
            self.armed,
            self.motor_active,
            self.mav_mode,
        )
        # Advisory, and deliberately before the safing sequence: the operator
        # needs to know the press registered even if MAVLink then goes away.
        try:
            self.send_status("RC SHUTDOWN: safing vehicle and powering down")
        except Exception as error:
            logging.warning("RC shutdown status text failed: %s", error)
        self.buzzer(tones["shutdown"])

        self._rc_shutdown_safe_vehicle()

        # poweroff, not halt: halt parks the CPU with the rails still up and
        # the battery still draining. systemd SIGTERMs the units on the way
        # down, which is what closes the zarr through capture_signal_handlers.
        try:
            result = subprocess.run(["sudo", "poweroff"], check=False)
        except (OSError, subprocess.SubprocessError) as error:
            logging.exception("RC shutdown command failed: %s", error)
        else:
            if result.returncode != 0:
                logging.error(
                    "RC shutdown command failed with return code %d",
                    result.returncode,
                )

    def _rc_shutdown_safe_vehicle(self):
        """Best-effort HOLD + disarm ahead of a poweroff. Never raises."""

        # request_motion_stop/wait_for_abort_hold is the existing cooperative
        # stop used at capture teardown -- it clears planner_should_move, so
        # the planner stops issuing repositions instead of racing the disarm
        # below. Reusing it is why an operator's CH5 disarm loses today and
        # this one does not.
        try:
            self.request_motion_stop(reason="RC shutdown (CH9)")
            self.wait_for_abort_hold(
                timeout_seconds=RC_SHUTDOWN_HOLD_TIMEOUT_SECONDS
            )
        except Exception as error:
            logging.error("RC shutdown could not stop planner motion: %s", error)

        if not self.armed:
            return

        try:
            self.disarm()
        except Exception as error:
            logging.error("RC shutdown disarm command failed: %s", error)
            return

        deadline = time.monotonic() + RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS
        while self.armed and time.monotonic() < deadline:
            # Pump heartbeats inline rather than sleeping. handle_RC_CHANNELS
            # runs ON the receive loop thread, and self.armed is only ever set
            # by handle_HEARTBEAT on that same thread -- so a plain sleep here
            # blocks the one loop that could observe the disarm, and the wait
            # could never do anything but time out.
            if not self._pump_one_heartbeat(timeout_seconds=0.2):
                break
        if self.armed:
            logging.error(
                "RC shutdown: vehicle still armed after %.1fs; powering down anyway",
                RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS,
            )
        else:
            logging.warning("RC shutdown: vehicle disarmed before poweroff")

    def _pump_one_heartbeat(self, *, timeout_seconds):
        """Process one HEARTBEAT from inside the receive loop. False to stop.

        Safe only because the sole caller already runs on the receive loop
        thread, so nothing else is reading this connection concurrently.
        Non-heartbeat messages queued behind it are discarded by recv_match,
        which is acceptable here and only here: the Pi is powering off.
        """

        try:
            with self._connection_lock:
                connection = self.connection
            heartbeat = connection.recv_match(
                type="HEARTBEAT", blocking=True, timeout=timeout_seconds
            )
        except Exception as error:
            logging.warning("RC shutdown could not read a heartbeat: %s", error)
            return False
        if heartbeat is not None:
            self.process_message(heartbeat)
        return True

    def handle_RC_CHANNELS(self, msg):
        # getattr/hasattr, matching the pattern used throughout: a reconnected
        # or hand-built Drone must not crash the receive loop over a missing
        # attribute.
        if not hasattr(self, "_rc_triggers"):
            self._rc_triggers = _build_rc_triggers()
        triggers = self._rc_triggers

        if triggers.shutdown.update(msg.chan9_raw):
            self._rc_shutdown(msg)
            # An accepted shutdown owns this RC message. Do not combine it
            # with reboot, compass-calibration, or distance-finder actions.
            return

        if triggers.compass.update(msg.chan10_raw):
            self.run_compass_calibration()

        if triggers.reboot.update(msg.chan7_raw):
            force = msg.chan7_raw > RC_SHUTDOWN_THRESHOLD
            logging.info("Request %sreboot", "force " if force else "")
            self.reboot(force=force)
            if not force:
                sys.exit(1)
        # If --no-ultrasonic omitted the sensor entirely, ignore RC12. This
        # prevents reconnect/default channel values from producing misleading
        # ENABLE/DISABLE messages for hardware that is not in use.
        if self.distance_finder is None:
            return
        if not hasattr(self, "_ultrasonic_rc_switch"):
            self._ultrasonic_rc_switch = _RCStableSwitch(
                initial_state=self.disable_distance_finder,
                low_threshold=RC_ULTRASONIC_LOW_THRESHOLD,
                high_threshold=RC_ULTRASONIC_HIGH_THRESHOLD,
                stable_samples=RC_ULTRASONIC_STABLE_SAMPLES,
                max_sample_gap_seconds=RC_ULTRASONIC_MAX_SAMPLE_GAP_SECONDS,
            )
        disabled = self._ultrasonic_rc_switch.update(
            value=msg.chan12_raw,
            # The ultrasonic switch keeps its clock: it debounces a *setting*,
            # where a wrong answer is recoverable, so consecutive-sample
            # stability is worth the rate dependency. The destructive channels
            # above deliberately have no clock at all.
            now=time.monotonic(),
        )
        if disabled is None:
            return
        self.disable_distance_finder = disabled
        logging.info(
            "%s ULTRASONIC after %d stable RC samples: ch12_raw=%d",
            "DISABLE" if disabled else "ENABLE",
            RC_ULTRASONIC_STABLE_SAMPLES,
            msg.chan12_raw,
        )

    def handle_SERVO_OUTPUT_RAW(self, msg):
        if msg.servo1_raw == 1500 and msg.servo3_raw == 1500:
            self.motor_active = False
        else:
            self.motor_active = True

    message_handlers = {
        "GLOBAL_POSITION_INT": handle_GLOBAL_POSITION_INT,
        "GPS_RAW_INT": handle_GPS_RAW_INT,
        "EKF_STATUS_REPORT": handle_EKF_STATUS_REPORT,
        "COMMAND_ACK": handle_COMMAND_ACK,
        "HEARTBEAT": handle_HEARTBEAT,
        "SYSTEM_TIME": handle_SYSTEM_TIME,
        "SYS_STATUS": handle_SYS_STATUS,
        "STATUSTEXT": handle_STATUSTEXT,
        "RC_CHANNELS": handle_RC_CHANNELS,
        "SERVO_OUTPUT_RAW": handle_SERVO_OUTPUT_RAW,
        "NAV_CONTROLLER_OUTPUT": handle_NAV_CONTROLLER_OUTPUT,
        "RC_CHANNELS_SCALED": handle_RC_CHANNELS_SCALED,
        "PARAM_VALUE": handle_PARAM_VALUE,
        "HOME_POSITION": handle_HOME_POSITION,
    }

    def process_message(self, msg):
        if msg is None:
            time.sleep(0.01)
            return
        msg_type = msg.get_type()
        if msg_type in self.message_handlers:
            self.message_handlers[msg_type](self, msg)

    def set_mode(self, mode):
        with self._command_connection() as connection:
            connection.set_mode(mode)


def get_ardupilot_serial():
    try:
        return resolve_ardupilot_serial()
    except MavlinkConnectionError as error:
        logging.error("%s", error)
        return None


def prepare_vehicle_parameters(drone, parameter_file):
    """Verify managed parameters and compass policy from live readback."""
    parameter_file = os.fspath(parameter_file)
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Parameter file does not exist: {parameter_file}")
    drone.update_all_parameters()

    differences = drone.params.diff(parameter_file)
    if differences:
        if drone.armed:
            raise VehicleParameterVerificationError(
                "vehicle is armed; refusing managed parameter writes"
            )
        drone.single_operation_mode_on()
        try:
            with drone._command_connection() as connection:
                drone.params.load(parameter_file, mav=connection)
        finally:
            drone.single_operation_mode_off()
        # MAVParmDict updates its local cache on PARAM_VALUE acknowledgement.
        # Re-download instead of trusting that cache so rejected/coerced values
        # cannot masquerade as verified flight-controller state.
        drone.update_all_parameters()

    remaining_differences = drone.params.diff(parameter_file)
    if remaining_differences:
        raise VehicleParameterVerificationError(
            "managed parameter verification found "
            f"{remaining_differences} differences"
        )
    return evaluate_compass_policy(drone.params)


def check_compass_policy(drone):
    """Evaluate a fresh, complete, read-only vehicle parameter snapshot."""
    drone.update_all_parameters()
    return evaluate_compass_policy(drone.params)


def write_compass_policy_report(report, output_path):
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def log_compass_policy_report(report):
    """Put the full detected inventory and priority state in the boot journal."""
    for line in format_compass_inventory(report):
        logging.info(line)
    for warning in report.warnings:
        logging.warning("Compass policy: %s", warning)
    for error in report.errors:
        logging.error("Compass policy: %s", error)
    if report.ok:
        logging.info("Compass policy PASS: external GPS compass is the sole yaw source")


def get_mavlink_controller_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial", type=str, help="Serial port", required=False, default=""
    )
    parser.add_argument("--ip", type=str, help="ip address", required=False, default="")
    parser.add_argument("--port", type=int, help="port", required=False, default=14552)
    parser.add_argument(
        "--save-params",
        type=str,
        help="save params to this file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--load-params",
        type=str,
        help="save params to this file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--diff-params",
        type=str,
        help="save params to this file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prepare-vehicle-params",
        type=str,
        help=(
            "download parameters once, apply this managed parameter file if "
            "needed, verify it, and evaluate compass policy"
        ),
        default=None,
    )
    parser.add_argument(
        "--check-compass-policy",
        action="store_true",
        help="download parameters once and evaluate compass policy without writes",
    )
    parser.add_argument(
        "--compass-policy-json",
        type=str,
        help="write the compass inventory and policy report to this path",
        default=None,
    )
    parser.add_argument(
        "--planner",
        type=str,
        help="which planner",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--proto",
        type=str,
        help="udpin/tcp",
        required=False,
        default="udpin",
    )

    parser.add_argument(
        "--get-time",
        type=str,
        help="time to file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--buzzer",
        type=str,
        help="buzz",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="set mode to this and exit",
        required=False,
        default=None,
    )
    parser.add_argument("--skip-heartbeat", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--connect-attempts",
        type=int,
        default=DEFAULT_MAVLINK_RECONNECT_ATTEMPTS,
        help="bounded initial/reconnect attempt count",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=DEFAULT_MAVLINK_HEARTBEAT_TIMEOUT_SECONDS,
        help="seconds to wait for a fresh flight-controller heartbeat",
    )
    parser.add_argument(
        "--reconnect-backoff",
        type=float,
        default=DEFAULT_MAVLINK_RECONNECT_BACKOFF_SECONDS,
        help="seconds between bounded reconnect attempts",
    )
    parser.add_argument("--reboot", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--time-since-boot",
        type=str,
        help="write time since boot to file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--status-json",
        type=str,
        help="write one read-only heartbeat status snapshot and exit",
        required=False,
        default=None,
    )
    return parser


def mavlink_controller_run(args):
    if args.serial == "" and args.ip == "":
        args.serial = resolve_ardupilot_serial()

    logging.info("Connecting to mavlink (drone)...")
    if args.serial != "":
        endpoint = resolve_ardupilot_serial(args.serial)
    elif args.ip != "":
        endpoint = f"{args.proto}:{args.ip}:{args.port}"
    else:
        raise MavlinkConnectionError("need ip or serial")

    connection_factory = mavlink_connection_factory(endpoint)
    initial_heartbeat = None
    if args.skip_heartbeat:
        connection = connection_factory()
    else:
        logging.info("Waiting for heartbeat...")
        connection, initial_heartbeat = connect_with_heartbeat(
            connection_factory,
            attempts=args.connect_attempts,
            heartbeat_timeout=args.heartbeat_timeout,
            retry_backoff=args.reconnect_backoff,
        )

    if args.buzzer is not None:
        assert not args.skip_heartbeat
        if args.buzzer.lower() in tones:
            tone_bytes = tones[args.buzzer.lower()]
        else:
            tone_bytes = args.buzzer.replace(" ", "").encode()
        drone = Drone(
            connection=connection,
            connection_factory=connection_factory,
            reconnect_attempts=args.connect_attempts,
            reconnect_backoff=args.reconnect_backoff,
            reconnect_heartbeat_timeout=args.heartbeat_timeout,
        )
        drone.process_message(initial_heartbeat)
        if not drone.buzzer(tone_bytes):
            logging.error("Could not send MAVLink buzzer command")
            sys.exit(1)
        sys.exit(0)

    logging.info("Listening...")

    boundary = franklin_safe

    # planner = None
    # if args.planner is not None:
    #    planner = drone_get_planner(args.planner, boundary=boundary)

    drone = Drone(
        connection,
        connection_factory=connection_factory,
        reconnect_attempts=args.connect_attempts,
        reconnect_backoff=args.reconnect_backoff,
        reconnect_heartbeat_timeout=args.heartbeat_timeout,
    )
    if initial_heartbeat is not None:
        drone.process_message(initial_heartbeat)

    logging.info("Drone start()")
    drone.start()
    # upload_waypoints(connection)

    if args.status_json is not None:
        deadline = time.time() + 10
        while drone.last_heartbeat == 0 and time.time() < deadline:
            time.sleep(0.05)
        if drone.last_heartbeat == 0:
            logging.error("Timed out waiting for a processed heartbeat")
            sys.exit(1)
        status = {
            "armed": bool(drone.armed),
            "mav_mode": drone.mav_mode,
            "mav_states": drone.mav_states,
            "heartbeat_age_seconds": time.time() - drone.last_heartbeat,
        }
        with open(args.status_json, "w") as status_file:
            json.dump(status, status_file, indent=2, sort_keys=True)
            status_file.write("\n")
        print(json.dumps(status, sort_keys=True))
        sys.exit(0)

    if args.time_since_boot is not None:
        with open(args.time_since_boot, "w") as f:
            while drone.time_since_boot == 0:
                time.sleep(0.01)
            f.write("%0.2f\n" % drone.time_since_boot)
        sys.exit(0)

    if args.reboot:
        drone.reboot(force=True)
        time.sleep(0.1)
        sys.exit(0)

    # do_mission(connection)

    #   connection.set_mode_auto()  # MAV_CMD_MISSION_START

    # connection.set_mode_auto()

    # connection.set_mode_auto()
    # breakpoint()
    # logging.info("Waiting for the vehicle to arm")
    # connection.motors_armed_wait()
    # logging.info("Armed!")

    if args.get_time is not None:
        drone.buzzer(tones["gps-time"])
        logging.info("GPS-time: waiting for heartbeat")
        while drone.last_heartbeat == 0:
            time.sleep(0.1)
        logging.info("GPS-time: waiting for gps time")
        # Wait for real UTC, never merely for a fix. The previous condition also
        # exited on a 3D fix, which is a race: ArduPilot reports a fix before
        # SYSTEM_TIME has delivered UTC, so this returned with gps_time == 0 and
        # wrote an epoch-0 timestamp. drone_run.sh was expected to catch that by
        # its "1970-" prefix, but datetime.fromtimestamp(0) renders as naive
        # LOCAL time -- "1970-01-01 01:00:00" in Europe/London but
        # "1969-12-31 16:00:00" west of UTC, where the prefix misses and the
        # clock really is set to epoch 0. Do not produce the bad value at all.
        #
        # The old literal was doubly wrong: "GPS_FIX_TYPE_3D_FIX" also fails to
        # match DGPS/RTK_FLOAT/RTK_FIXED, so a BETTER fix did not satisfy it.
        #
        # Unbounded by design -- callers bound it (drone_run.sh wraps this in
        # `timeout ${SPF_GPS_TIME_TIMEOUT:-180}` and retries).
        while drone.gps_time == 0:
            time.sleep(1)

        gps_time = datetime.fromtimestamp(drone.gps_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        # Open only once a real value exists. Opening "w" before the wait meant
        # every timeout-killed attempt truncated this file to zero bytes and
        # destroyed whatever a previous successful sync had written -- and
        # `date -d ""` parses to today-at-midnight, so an empty file silently
        # became a plausible-looking clock up to 24h wrong.
        with open(args.get_time, "w") as f:
            f.write(gps_time + "\n")
        sys.exit(0)

    if (
        args.save_params is not None
        or args.load_params is not None
        or args.diff_params is not None
        or args.prepare_vehicle_params is not None
        or args.check_compass_policy
    ):
        while drone.last_heartbeat == 0:
            time.sleep(3)
        drone.buzzer(tones["check-diff"])

        if args.prepare_vehicle_params is not None or args.check_compass_policy:
            if args.compass_policy_json is None:
                logging.error(
                    "--prepare-vehicle-params/--check-compass-policy requires "
                    "--compass-policy-json"
                )
                sys.exit(2)
            if args.prepare_vehicle_params is not None and args.check_compass_policy:
                logging.error(
                    "--prepare-vehicle-params and --check-compass-policy are "
                    "mutually exclusive"
                )
                sys.exit(2)
            try:
                if args.prepare_vehicle_params is not None:
                    report = prepare_vehicle_parameters(
                        drone, args.prepare_vehicle_params
                    )
                else:
                    report = check_compass_policy(drone)
            except (OSError, MavlinkParameterError) as error:
                logging.error(
                    "Vehicle/compass parameter verification failed: %s", error
                )
                sys.exit(1)
            write_compass_policy_report(report, args.compass_policy_json)
            log_compass_policy_report(report)
            sys.exit(0 if report.ok else 1)

        drone.update_all_parameters()

        if args.diff_params is not None:
            if not os.path.isfile(args.diff_params):
                logging.error(f"File {args.diff_params} does not exist!")
                sys.exit(1)
            diffs = drone.params.diff(args.diff_params)
            logging.info(f"Detected {diffs} differences")
            sys.exit(diffs)
        if args.save_params is not None:
            drone.params.save(args.save_params)
        if args.load_params is not None:
            if not os.path.isfile(args.load_params):
                logging.error(f"File {args.load_params} does not exist!")
                sys.exit(1)
            drone.single_operation_mode_on()
            drone.disarm()
            time.sleep(0.02)
            with drone._command_connection() as connection:
                count, changed = drone.params.load(args.load_params, mav=connection)
            drone.single_operation_mode_off()
        sys.exit(0)

    if args.mode is not None:
        return_code = set_drone_mode(drone, args.mode)
        sys.exit(return_code)

    while True:
        drone.raise_if_connection_failed()
        time.sleep(1)


def set_drone_mode(drone, mode):
    target_mode = mode.upper()
    if target_mode not in switchable_modes:
        logging.error("Not a valid switchable mode")
        sys.exit(1)
    result_mode = switchable_modes[target_mode]
    return_code = 1
    for _ in range(3):  # 3 retries
        drone.set_mode(target_mode)
        time.sleep(2)
        if drone.mav_mode == result_mode:
            return_code = 0
            break
    return return_code


if __name__ == "__main__":
    args = get_mavlink_controller_parser().parse_args()
    mavlink_controller_run(args)
    # Create the connection
    # Need to provide the serial port and baudrate

    # logging.info(f"MODE {drone.mav_mode}")

    # logging.info("DONE")
    # drone.process_messages()
