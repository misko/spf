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
from datetime import datetime

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
from spf.motion_planners.dynamics import Dynamics
from spf.motion_planners.planner import (
    BouncePlanner,
    CirclePlanner,
    PointCycle,
    StationaryPlanner,
)

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
}
tones = {k: v.replace(" ", "").encode() for k, v in tones.items()}

LOG_ERASE = 121

RC_SHUTDOWN_THRESHOLD = 1500
RC_SHUTDOWN_HOLD_SECONDS = 2.0
RC_SHUTDOWN_MAX_SAMPLE_GAP_SECONDS = 1.5
RC_SHUTDOWN_MIN_HIGH_SAMPLES = 3
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


class _RCHoldInterlock:
    """Debounce a destructive RC action and require an intentional release."""

    def __init__(
        self,
        *,
        threshold,
        hold_seconds,
        max_sample_gap_seconds,
        min_high_samples,
    ):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.max_sample_gap_seconds = max_sample_gap_seconds
        self.min_high_samples = min_high_samples
        self._released_seen = False
        self._high_since = None
        self._last_high_at = None
        self._high_samples = 0
        self._latched = False

    def _reset_hold(self):
        self._high_since = None
        self._last_high_at = None
        self._high_samples = 0

    def update(self, *, value, now, permitted):
        if value <= self.threshold:
            self._released_seen = True
            self._latched = False
            self._reset_hold()
            return False, 0.0

        if self._latched or not self._released_seen:
            return False, 0.0

        if not permitted:
            # A switch held while the rover is unsafe must be released before
            # it can begin a later shutdown request.
            self._released_seen = False
            self._reset_hold()
            return False, 0.0

        if (
            self._last_high_at is None
            or now - self._last_high_at > self.max_sample_gap_seconds
        ):
            self._high_since = now
            self._high_samples = 1
        else:
            self._high_samples += 1

        self._last_high_at = now
        held_seconds = now - self._high_since
        if (
            held_seconds >= self.hold_seconds
            and self._high_samples >= self.min_high_samples
        ):
            self._latched = True
            self._reset_hold()
            return True, held_seconds

        return False, held_seconds


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
        self.crash_detect = crash_detect
        self.crash_recovery = crash_recovery
        self.stall_detect_seconds = float(stall_detect_seconds)
        self.stall_manual_seconds = float(stall_manual_seconds)
        self.stall_progress_radius_m = float(stall_progress_radius_m)
        self._initialize_stall_state()
        self._initialize_motion_stop_state()
        self._rc_shutdown_interlock = _RCHoldInterlock(
            threshold=RC_SHUTDOWN_THRESHOLD,
            hold_seconds=RC_SHUTDOWN_HOLD_SECONDS,
            max_sample_gap_seconds=RC_SHUTDOWN_MAX_SAMPLE_GAP_SECONDS,
            min_high_samples=RC_SHUTDOWN_MIN_HIGH_SAMPLES,
        )

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
            logging.info(
                "MAVLink reconnected after a fresh flight-controller heartbeat"
            )
            return True

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
        self.message_loop_thread.start()
        return self

    def send_status(self, text):
        with self._command_connection() as connection:
            connection.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_CRITICAL, text.encode()
            )

    def is_planner_in_control(self):
        return self.planner_in_control

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

    def _stall_radius(self):
        return getattr(self, "stall_progress_radius_m", STALL_PROGRESS_RADIUS_M)

    def _reset_stall_anchor(self, now=None):
        """Re-anchor at the current position; the stall clock restarts here."""

        self._stall_anchor = self.gps
        self._stall_anchor_at = time.monotonic() if now is None else now
        self._stall_last_escape_at = None

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

        # getattr throughout, matching the _rc_shutdown_interlock pattern: a
        # reconnected or hand-built Drone must not crash the motion loop over a
        # missing attribute.
        if not getattr(self, "crash_detect", True):
            return STALL_OK
        if not hasattr(self, "_stall_anchor"):
            self._initialize_stall_state()
        detect_seconds = getattr(self, "stall_detect_seconds", STALL_DETECT_SECONDS)
        manual_seconds = getattr(self, "stall_manual_seconds", STALL_MANUAL_SECONDS)
        radius_m = getattr(
            self, "stall_progress_radius_m", STALL_PROGRESS_RADIUS_M
        )

        driving = (
            self.armed and self.motor_active and self.mav_mode == "ROVER_MODE_GUIDED"
        )
        if not driving:
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

        stalled_seconds = now - self._stall_anchor_at
        if stalled_seconds < detect_seconds:
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
            return MOVE_ABORTED
        self.reposition(lat=point[1], long=point[0])
        # The stall anchor is deliberately NOT reset here. It is owned by
        # _stall_verdict and reset only by real motion, so the clock survives
        # across the waypoints an escape abandons -- which is what makes the
        # escalation to MANUAL reachable at all.
        last_message = None
        while self.distance_to_target(point) > self.tolerance_in_m:
            self.raise_if_connection_failed()
            if self._motion_stop_requested.is_set():
                self._try_enter_abort_hold()
                logging.warning(
                    "Aborted active waypoint after capture failure; target=%s",
                    point,
                )
                return MOVE_ABORTED
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

    def reposition(self, lat, long):
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
                0,  # reposition flags
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
        # logging.info(f"COMMAND ACK {str(msg)}")
        # if msg.command in self.mav_cmd_num2name:
        #    logging.info(f"COMMAND {self.mav_cmd_num2name[msg.command]}")
        pass

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

    def handle_RC_CHANNELS(self, msg):
        if not hasattr(self, "_rc_shutdown_interlock"):
            self._rc_shutdown_interlock = _RCHoldInterlock(
                threshold=RC_SHUTDOWN_THRESHOLD,
                hold_seconds=RC_SHUTDOWN_HOLD_SECONDS,
                max_sample_gap_seconds=RC_SHUTDOWN_MAX_SAMPLE_GAP_SECONDS,
                min_high_samples=RC_SHUTDOWN_MIN_HIGH_SAMPLES,
            )
        now = time.monotonic()
        shutdown_requested, held_seconds = self._rc_shutdown_interlock.update(
            value=msg.chan9_raw,
            now=now,
            permitted=not self.armed and not self.motor_active,
        )
        if shutdown_requested:
            logging.warning(
                "RC shutdown accepted: ch9_raw=%d held=%.3fs armed=%s motor_active=%s",
                msg.chan9_raw,
                held_seconds,
                self.armed,
                self.motor_active,
            )
            try:
                result = subprocess.run(["sudo", "shutdown", "0"], check=False)
            except (OSError, subprocess.SubprocessError) as error:
                logging.exception("RC shutdown command failed: %s", error)
            else:
                if result.returncode != 0:
                    logging.error(
                        "RC shutdown command failed with return code %d",
                        result.returncode,
                    )
            # An accepted shutdown owns this RC message. Do not combine it
            # with reboot, compass-calibration, or distance-finder actions.
            return
        if msg.chan10_raw > 1500:  # run compass calibration
            self.run_compass_calibration()
        if msg.chan7_raw > 1500:
            # reboot ardupilot
            logging.info("Request force reboot")
            self.reboot(force=True)
        elif msg.chan7_raw > 1000:
            logging.info("Request reboot")
            self.reboot()
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
            now=now,
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
        with open(args.get_time, "w") as f:
            drone.buzzer(tones["gps-time"])
            logging.info("GPS-time: waiting for heartbeat")
            while drone.last_heartbeat == 0:
                time.sleep(0.1)
            logging.info("GPS-time: waiting for gps time")
            # NB: this can exit on a 3D fix while gps_time is still 0, yielding a
            # 1970 timestamp; drone_run.sh sync_gps_time guards `date -s` against
            # that so the system clock is never set to 1970 (poll shortened 5->1s).
            while drone.gps_time == 0 and drone.gps_fix_type != "GPS_FIX_TYPE_3D_FIX":
                time.sleep(1)

            gps_time = datetime.fromtimestamp(drone.gps_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
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
