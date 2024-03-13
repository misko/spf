# Import mavutil
import argparse
import glob
import logging
import math
import subprocess
import sys
import threading
import time

import numpy as np
from haversine import Unit, haversine
from pymavlink import mavutil, mavwp

from spf.gps.boundaries import franklin_safe  # crissy_boundary_convex
from spf.gps.gps_utils import swap_lat_long
from spf.grbl.grbl_interactive import BouncePlanner, Dynamics

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

gps_fix_type = {
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

mav_cmds_num2name = {}


LOG_ERASE = 121


def lookup_bits(x, table):
    return [y for y in table if x & getattr(mavutil.mavlink, y)]


def lookup_exact(x, table):
    return [y for y in table if x == getattr(mavutil.mavlink, y)]


class Drone:
    def __init__(
        self,
        connection,
        planner=None,
        boundary=None,
        tolerance_in_m=5,
        distance_finder=None,
    ):
        self.connection = connection
        self.boundary = boundary

        self.heading = 0
        self.gps_time = 0
        self.distance_finder = distance_finder
        if self.distance_finder is not None:
            self.distance_finder.run_in_new_thread()

        logging.getLogger("numba").setLevel(logging.WARNING)
        self.mav_mode_mapping_name2num = connection.mode_mapping()
        self.mav_mode_mapping_num2name = mavutil.mode_mapping_bynumber(
            connection.sysid_state[connection.sysid].mav_type
        )
        # breakpoint()

        self.healthy_ekf_flag = (
            EKF_STATUS_STRING_TO_DEC["EKF_ATTITUDE"]
            | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
            | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
        )

        self.motor_active = False

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
        self.drone_ready = False

        self.message_loop = True
        self.single_operation = False

        self.sensors_present = []
        self.sensors_enabled = []
        self.sensors_health = []

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
            "PARAM_VALUE",
            "BAD_DATA",
        ]

        # self.erase_logs()

        self.message_loop_thread = threading.Thread(
            target=self.process_messages, daemon=True
        )

        self.planner = planner
        self.planner_started_moving = False

        if self.planner is not None:
            self.planner_thread = threading.Thread(target=self.run_planner, daemon=True)

        self.last_heartbeat_log = None
        self.armed = False

        self.gps_satellites = -1
        self.gps_fix_type = "NOT_SET_YET"
        # self.mission_item_condition = threading.Condition()
        # self.mission_item_reached = False

    # motion interface
    def start(self):
        self.message_loop_thread.start()
        if self.planner is not None:
            self.planner_thread.start()
        return self

    def send_status(self, text):
        self.connection.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_CRITICAL, text.encode()
        )

    def has_planner_started_moving(self):
        return self.planner_started_moving

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
        logging.info(f"CURRENT {self.gps} TARGET {str(point)}")
        self.reposition(lat=point[1], long=point[0])
        last_message = None
        while self.distance_to_target(point) > self.tolerance_in_m:
            if last_message is None or time.time() - last_message > log_interval:
                logging.info(
                    f"distance (m) TO TARGET {str(self.distance_to_target(point))} {self.motor_active} {self.mav_mode}"
                )
                last_message = time.time()
            # safety
            distance = self.distance_finder.distance
            collision_soon = (
                (not self.disable_distance_finder)
                and self.distance_finder is not None
                and distance < 130
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
        logging.info(f"REACHED TARGET {str(point)} Current {str(self.gps)}")
        return True

    def run_planner(self):
        # self.single_operation_mode_on()
        logging.info("Start planner")
        # self.single_operation_mode_on()
        # logging.info("SINGLE OPERATION MODE")
        home = self.boundary.mean(axis=0)

        self.single_operation_mode_on()
        # self.connection.waypoint_clear_all_send()
        # logging.info("SINGLE OPERATION MODE 2")

        self.set_home(lat=home[1], long=home[0])
        self.single_operation_mode_off()

        # self.single_operation_mode_off()
        # drone.request_home()
        logging.info("Planer main loop")
        while not self.drone_ready:
            if (
                self.gps is not None
                and "MAV_SYS_STATUS_SENSOR_GPS" in self.sensors_health
                and self.ekf_healthy
                and not self.armed
            ):
                self.arm()
            else:
                logging.info(
                    f"wait for ready: gps:{str(self.gps)} , ekf:{str(self.ekf_healthy)}"
                )
            time.sleep(10)
        logging.info("DRONE IS READY TO ROLL")

        # for point in self.boundary:
        #    self.move_to_point(point)
        # self.move_to_point(self.boundary[0])

        self.move_to_point(home)
        time.sleep(2)

        # drone is now ready
        # point is long, lat
        yp = self.planner.yield_points()
        while True:
            point = next(yp)
            self.move_to_point(point)
            self.planner_started_moving = True
            time.sleep(2)

    def get_cmd(self, cmd):
        v = getattr(mavutil.mavlink, cmd)
        self.mav_cmd_name2num[cmd] = v
        self.mav_cmd_num2name[v] = cmd
        return v

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
        # set home position
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            self.get_cmd("MAV_CMD_DO_SET_HOME"),
            0,  # set position
            1,  # param1
            0,  # param2
            0,  # param3
            0,  # param4
            lat,  # 37.8047122,  # lat
            long,  # -122.4659164,  # lon
            0,
        )
        # self.ack("COMMAND_ACK")

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
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
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
        self.connection.mav.command_int_send(
            self.connection.target_system,
            self.connection.target_component,
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

    def upload_waypoints(self):
        wp = mavwp.MAVWPLoader()
        seq = 1
        radius = 10
        for long, lat in self.boundary:
            wp.add(
                mavutil.mavlink.MAVLink_mission_item_message(
                    self.connection.target_system,
                    self.onnection.target_component,
                    seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    self.get_cmd("MAV_CMD_NAV_WAYPOINT"),
                    0,
                    0,
                    0,
                    radius,
                    0,
                    0,
                    lat,
                    long,
                    0.0,
                )
            )  # altitude = 0.0
            seq += 1
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(wp.count())

        assert wp.count() > 0

        msg = self.connection.recv_match(type=["MISSION_ACK"], blocking=True, timeout=3)
        if msg is None or mav_mission_states[msg.type] != "MAV_MISSION_ACCEPTED":
            return False

        while True:
            msg = self.connection.recv_match(
                type=["MISSION_REQUEST", "MISSION_ACK"], blocking=True
            )
            if msg.get_type() == "MISSION_ACK":
                if mav_mission_states[msg.type] == "MAV_MISSION_ACCEPTED":
                    break
                continue
            logging.info(f"Sending waypoint {msg.seq}")
            self.connection.mav.send(wp.wp(msg.seq))

        logging.info("DONE UPLOAD")

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
                msg = self.connection.recv_match(blocking=True, timeout=0.5)
                self.process_message(msg)

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
        self.gps_fix_type = gps_fix_type[msg.fix_type]

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

    def handle_HEARTBEAT(self, msg, log_interval=5):
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
                    f"HEARTBEAT STATUS: mav_state:{mav_state_check}"
                    + f"gps:{gps_check}({self.gps_satellites}sats,{self.gps_fix_type}),"
                    + f"gps_healthy:{gps_healthy}, guided_mode:{guided_mode}, ekf:{self.ekf_healthy}"
                )
                self.last_heartbeat_log = time.time()
            if (
                mav_state_check
                and gps_check
                and gps_healthy
                and guided_mode
                and self.ekf_healthy
            ):
                logging.info("SETTING DRONE TO READY!")
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
        # print(msg.to_dict())
        if msg.chan9_raw > 1500:
            subprocess.run(["sudo", "shutdown", "0"])
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
        elif msg.chan12_raw > 1000:
            logging.info("DISABLE ULTRASONIC")
            self.disable_distance_finder = True

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
    }

    def process_message(self, msg):
        if msg is None:
            time.sleep(0.01)
            return
        msg_type = msg.get_type()
        if msg_type in self.message_handlers:
            self.message_handlers[msg_type](self, msg)

    def set_mode(self, mode):
        self.connection.set_mode(mode)


def get_ardupilot_serial():
    available_pilots = glob.glob("/dev/serial/by-id/usb-ArduPilot*")
    if len(available_pilots) != 1:
        logging.error(f"Strange number of autopilots found {len(available_pilots)}")
        return None
    return available_pilots[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial", type=str, help="Serial port", required=False, default=""
    )
    parser.add_argument("--ip", type=str, help="ip address", required=False, default="")
    parser.add_argument("--port", type=int, help="port", required=False, default=14552)

    args = parser.parse_args()
    logging.info("WTF")
    # Create the connection
    # Need to provide the serial port and baudrate
    if args.serial == "" and args.ip == "":
        args.serial = get_ardupilot_serial()
        if args.serial is None:
            sys.exit(1)

    logging.info("Connecting...")
    if args.serial != "":
        connection = mavutil.mavlink_connection(args.serial, baud=115200)
    elif args.ip != "":
        connection = mavutil.mavlink_connection(
            f"udpin:{args.ip}:{args.port}"
        )  # tcp is 5670
        # connection = mavutil.mavlink_connection(f"udpout:{args.ip}:14550")
    else:
        logging.error("need ip or serial")
        exit(1)

    logging.info("Wait heartbeat...")
    while True:
        if connection.wait_heartbeat(blocking=True, timeout=1):
            break

    logging.info("Listening...")

    # get rid of the top?
    msg = connection.recv_match(blocking=True, timeout=0.5)
    while msg is None or msg.get_type() == "BAD_DATA":
        msg = connection.recv_match(blocking=True, timeout=0.5)

    logging.info("really listening")

    boundary = franklin_safe
    drone = Drone(
        connection,
        planner=BouncePlanner(
            dynamics=Dynamics(
                bounding_box=boundary,
                bounds_radius=0.000000001,
            ),
            start_point=boundary.mean(axis=0),
            epsilon=0.0000001,
            step_size=0.1,
        ),
        boundary=boundary,
    )

    logging.info("drone object created")
    drone.start()
    # upload_waypoints(connection)

    # do_mission(connection)

    #   connection.set_mode_auto()  # MAV_CMD_MISSION_START

    # connection.set_mode_auto()

    # connection.set_mode_auto()
    # breakpoint()
    # logging.info("Waiting for the vehicle to arm")
    # connection.motors_armed_wait()
    # logging.info("Armed!")

    while True:
        time.sleep(200)
    # logging.info(f"MODE {drone.mav_mode}")

    # logging.info("DONE")
    # drone.process_messages()
