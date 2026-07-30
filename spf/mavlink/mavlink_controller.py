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
from datetime import datetime

import numpy as np
from haversine import Unit, haversine
from pymavlink import mavutil

from spf.gps.boundaries import boundary_to_diamond  # crissy_boundary_convex
from spf.gps.boundaries import franklin_safe
from spf.gps.gps_utils import swap_lat_long
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

switchable_modes = {"GUIDED": "ROVER_MODE_GUIDED", "MANUAL": "ROVER_MODE_MANUAL"}

mav_cmds_num2name = {}


tones = {
    "gps-time": "MFT240L8 C C C P4 C C C P4 L8dcdcdcdc",
    "check-diff": "MFT240L8 A B P4 A B P4 L8dcdc",
    "git": "MFT240L4 < F P2 F P4 L8dcdc",
    "planner": "MFT240L8 G G F F P4 G G F F P4 L8dc",
    "wait": "MFT240L8 G P4 < G P4 < G P4 > > G P4 < G P4 < G",
    "ready": "MFT240L8 G P8 < G P8 < G P8 > > G P8 < G P8 < G",
    "failure": "MFT240L8 D D D P4 D D D P4 L8dddddc",
}
tones = {k: v.replace(" ", "").encode() for k, v in tones.items()}

LOG_ERASE = 121

RC_SHUTDOWN_THRESHOLD = 1500
RC_SHUTDOWN_HOLD_SECONDS = 2.0
RC_SHUTDOWN_MAX_SAMPLE_GAP_SECONDS = 1.5
RC_SHUTDOWN_MIN_HIGH_SAMPLES = 3


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


class MavlinkParameterError(MavlinkConnectionError):
    """The complete vehicle parameter set could not be read."""


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
    ):
        self.connection = connection
        self.connection_factory = connection_factory
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff = reconnect_backoff
        self.reconnect_heartbeat_timeout = reconnect_heartbeat_timeout
        self.connection_failure = None
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

    def _require_healthy_connection(self):
        if self.connection_factory is not None and not self.connection_healthy:
            raise MavlinkConnectionError(
                "MAVLink connection is not healthy; refusing vehicle command"
            )
        return self.connection

    def raise_if_connection_failed(self):
        if self.connection_failure is not None:
            raise MavlinkConnectionError(str(self.connection_failure))

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
                self.connection.mav.play_tune_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    tone_bytes,
                )
                return True
            except AttributeError:
                pass
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
            with self._connection_lock:
                connection = self._require_healthy_connection()
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
        self.connection.mav.statustext_send(
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
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
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
    def move_to_point(self, point, log_interval=5):
        # logging.info(f"GPS: current position {self.gps} target position {str(point)}")
        self.reposition(lat=point[1], long=point[0])
        last_message = None
        while self.distance_to_target(point) > self.tolerance_in_m:
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
        return True

    def run_planner(self):
        # self.single_operation_mode_on()
        logging.info("Start planner...")
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
            time.sleep(10)

        if self.ignore_mode:
            time.sleep(2)
            while self.planner_should_move:
                self.planner_in_control = True
                time.sleep(1)

        else:
            while self.mav_mode != "ROVER_MODE_MANUAL":
                time.sleep(10)
                logging.info("waiting for rover to move into manual mode...")
                self.buzzer(tones["wait"])

            while self.mav_mode != "ROVER_MODE_GUIDED":
                time.sleep(10)
                logging.info("waiting for rover to move into guided mode...")
                self.buzzer(tones["ready"])

            if not self.armed:
                self.arm()
                time.sleep(0.1)
            logging.info("Planner starting to issue move commands...")

            self.move_to_point(home)
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
                    self.move_to_point(point)
                self.planner_in_control = True
                # time.sleep(2)

        self.planner_in_control = False

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
        self.move_to_point(self.home)

        start_time = time.time()
        while time.time() - start_time < max_wait:
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
        connection = self._require_healthy_connection()
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
        message = self.connection.mav.command_long_encode(
            self.connection.target_system,  # Target system ID
            self.connection.target_component,  # Target component ID
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
        self.connection.mav.send(message)

    def request_home(self):
        message = self.connection.mav.command_long_encode(
            self.connection.target_system,  # Target system ID
            self.connection.target_component,  # Target component ID
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
        self.connection.mav.send(message)

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
        connection = self._require_healthy_connection()
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
        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY,
            0,
        )

    def reboot(self, force=False, hold_in_bootloader=False):
        if not force:
            self.connection.reboot_autopilot()
            return
        if hold_in_bootloader:
            param1 = 3
        else:
            param1 = 1
        param6 = 20190226
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
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
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
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
        connection = self._require_healthy_connection()
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
        connection = self._require_healthy_connection()
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
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
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
        return (
            self.connection.recv_match(type=keyword, blocking=True, timeout=5) is None
        )

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
                self.process_message(msg)

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
                logging.info("Drone ready (gps + gps_healthy + ekf_healthy)")
                self.drone_ready = True

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
        shutdown_requested, held_seconds = self._rc_shutdown_interlock.update(
            value=msg.chan9_raw,
            now=time.monotonic(),
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
            result = subprocess.run(["sudo", "shutdown", "0"], check=False)
            if result.returncode != 0:
                logging.error(
                    "RC shutdown command failed with return code %d",
                    result.returncode,
                )
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
        if msg.chan12_raw > 1000:
            if not self.disable_distance_finder:
                logging.info("DISABLE ULTRASONIC")
                self.disable_distance_finder = True
        else:
            if self.disable_distance_finder:
                logging.info("ENABLE ULTRASONIC")
                self.disable_distance_finder = False

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
        self._require_healthy_connection().set_mode(mode)


def get_ardupilot_serial():
    try:
        return resolve_ardupilot_serial()
    except MavlinkConnectionError as error:
        logging.error("%s", error)
        return None


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
        drone.buzzer(tone_bytes)
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
            while drone.gps_time == 0 and drone.gps_fix_type != "GPS_FIX_TYPE_3D_FIX":
                drone.buzzer(tones["gps-time"])
                time.sleep(5)

            gps_time = datetime.fromtimestamp(drone.gps_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            f.write(gps_time + "\n")
            sys.exit(0)

    if (
        args.save_params is not None
        or args.load_params is not None
        or args.diff_params is not None
    ):
        while drone.last_heartbeat == 0:
            time.sleep(3)
        drone.buzzer(tones["check-diff"])
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
            count, changed = drone.params.load(args.load_params, mav=drone.connection)
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
