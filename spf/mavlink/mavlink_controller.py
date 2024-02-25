# Import mavutil
import argparse
import logging
import math
import signal
import threading
import time

import numpy as np
from pymavlink import mavutil, mavwp

from spf.gps.boundaries import crissy_boundary_convex
from spf.grbl.grbl_interactive import BouncePlanner, Dynamics

keep_running = True


def signal_handler(sig, frame):
    global keep_running
    keep_running = False


signal.signal(signal.SIGINT, signal_handler)


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


mav_cmds_num2name = {}


def lookup_bits(x, table):
    return [y for y in table if x & getattr(mavutil.mavlink, y)]


def lookup_exact(x, table):
    return [y for y in table if x == getattr(mavutil.mavlink, y)]


class Drone:
    def __init__(self, connection, planner, boundary, tolerance=0.0001):
        self.connection = connection
        self.boundary = boundary

        self.mav_mode_mapping_name2num = connection.mode_mapping()
        self.mav_mode_mapping_num2name = mavutil.mode_mapping_bynumber(
            connection.sysid_state[connection.sysid].mav_type
        )

        self.mav_states = []
        self.gps = None
        self.mav_mode = None
        self.mav_cmd_name2num = {
            "MAV_CMD_DO_SET_MODE": 176,
        }
        self.mav_cmd_num2name = {176: "MAV_CMD_DO_SET_MODE"}

        self.switch_condition = threading.Condition()
        self.drone_ready_condition = threading.Condition()
        self.drone_ready = False

        self.message_loop = True
        self.single_operation = False

        self.timeout = 0.5
        self.tolerance = tolerance
        self.ignore_messages = [
            "AHRS2",
            "ATTITUDE",
            "BATTERY_STATUS",
            "EKF_STATUS_REPORT",
            "ESC_TELEMETRY_1_TO_4",
            "GPS_RAW_INT",
            "HWSTATUS",
            "LOCAL_POSITION_NED",
            "MEMINFO",
            # "MISSION_CURRENT",
            "NAV_CONTROLLER_OUTPUT",
            "POSITION_TARGET_GLOBAL_INT",
            "POWER_STATUS",
            "RAW_IMU",
            "RC_CHANNELS",
            "RC_CHANNELS_SCALED",
            "SCALED_IMU2",
            "SCALED_IMU3",
            "SCALED_PRESSURE",
            "SCALED_PRESSURE2",
            "SERVO_OUTPUT_RAW",
            "SIMSTATE",
            "SYSTEM_TIME",
            "SYS_STATUS",
            "VFR_HUD",
            "VIBRATION",
            "PARAM_VALUE",
        ]

        self.message_loop_thread = threading.Thread(target=self.process_messages)

        self.planner_thread = threading.Thread(target=self.run_planner)

        self.planner = planner

        self.mission_item_condition = threading.Condition()
        self.mission_item_reached = False

        self.message_loop_thread.start()
        self.planner_thread.start()

    def distance_to_target(self, target_point):
        # points are long , lat
        return np.linalg.norm(target_point - self.gps)

    # point long/lat
    def move_to_point(self, point):
        print("CURRENT", self.gps, "TARGET", point)
        with self.mission_item_condition:
            self.reposition(lat=point[1], long=point[0])
            while keep_running and self.distance_to_target(point) > self.tolerance:
                print("D TO TARGET", self.distance_to_target(point))
                self.mission_item_condition.wait(timeout=5.0)
        if keep_running:
            print("REACHED TARGET", point, "Current", self.gps)
            return True
        return False

    def run_planner(self):

        home = self.boundary.mean(axis=0)

        self.connection.waypoint_clear_all_send()

        self.set_home(lat=home[1], long=home[0])

        # drone.request_home()

        global keep_running
        with self.drone_ready_condition:
            while keep_running and not self.drone_ready:
                self.turn_off_hardware_safety()
                self.arm()
                self.set_mode("GUIDED")
                print("WAIT FOR READY DRONE")
                self.drone_ready_condition.wait(timeout=5.0)

        for point in self.boundary:
            self.move_to_point(point)
        self.move_to_point(self.boundary[0])

        self.move_to_point(home)
        time.sleep(2)

        # drone is now ready
        # point is long, lat
        yp = self.planner.yield_points()
        while keep_running:
            point = next(yp)
            self.move_to_point(point)
            time.sleep(2)

    def get_cmd(self, cmd):
        v = getattr(mavutil.mavlink, cmd)
        self.mav_cmd_name2num[cmd] = v
        self.mav_cmd_num2name[v] = cmd
        return v

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
            1,  # set position
            0,  # param1
            0,  # param2
            0,  # param3
            0,  # param4
            lat,  # 37.8047122,  # lat
            long,  # -122.4659164,  # lon
            0,
        )
        self.ack("COMMAND_ACK")

    def turn_off_hardware_safety(self):
        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY,
            0,
        )

    def arm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            self.get_cmd("MAV_CMD_COMPONENT_ARM_DISARM"),
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.ack("COMMAND_ACK")

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
        self.mission_item_reached = False
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
        self.ack("COMMAND_ACK")

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

        while True:
            msg = self.connection.recv_match(type=["MISSION_ACK"], blocking=True)
            if mav_mission_states[msg.type] == "MAV_MISSION_ACCEPTED":
                break

        while True:
            msg = self.connection.recv_match(
                type=["MISSION_REQUEST", "MISSION_ACK"], blocking=True
            )
            if msg.get_type() == "MISSION_ACK":
                if mav_mission_states[msg.type] == "MAV_MISSION_ACCEPTED":
                    break
                continue
            print(f"Sending waypoint {msg.seq}")
            self.connection.mav.send(wp.wp(msg.seq))

        print("DONE UPLOAD")

    def ack(self, keyword):
        self.connection.recv_match(type=keyword, blocking=True)

    def single_operation_mode_on(self):
        global keep_running
        assert not self.single_operation
        self.single_operation = True  # request single operation mode
        with self.switch_condition:
            while keep_running and self.message_loop:
                self.switch_condition.wait(timeout=self.timeout)
            if not keep_running:
                return None
            return True

    def single_operation_mode_off(self, turn_on_messages=True):
        assert self.single_operation
        self.single_operation = False  # request single operation mode
        self.message_loop = turn_on_messages
        with self.switch_condition:
            self.switch_condition.notify_all()

    def process_messages(self):
        global keep_running
        while keep_running:
            with self.switch_condition:  # try not to leave context too often
                # if we are not supposed to run  message loop or the single operation mode is requested
                # chill out
                if self.single_operation:
                    self.message_loop = False  # lets chill for a bit
                while keep_running and (not self.message_loop or self.single_operation):
                    print("Message loop waiting")
                    self.switch_condition.wait(timeout=self.timeout)
                if not keep_running:
                    return
                msg = self.connection.recv_match()
                self.process_message(msg)

    def process_message(self, msg):
        if msg is None:
            time.sleep(0.01)
            return
        # .to_dict())
        if msg.get_type() == "GLOBAL_POSITION_INT":
            d = msg.to_dict()
            self.lat = msg.lat / 1e7
            self.long = msg.lon / 1e7
            self.gps = np.array([self.long, self.lat])
        elif msg.get_type() == "COMMAND_ACK":
            d = msg.to_dict()
            if msg.command in drone.mav_cmd_num2name:
                print("COMMAND", drone.mav_cmd_num2name[msg.command])
        elif msg.get_type() == "HOME_POSITION":  # also maybe GPS_GLOBAL_ORIGIN
            d = msg.to_dict()
            # print(d)
        elif msg.get_type() == "AHRS":
            d = msg.to_dict()
            # print(d)
        elif msg.get_type() == "HEARTBEAT":
            d = msg.to_dict()
            # print(d)
            self.mav_states = lookup_exact(d["system_status"], mav_states_list)
            self.mav_mode = self.mav_mode_mapping_num2name[msg.custom_mode]
            if (
                not self.drone_ready
                and (
                    "MAV_STATE_STANDBY" in self.mav_states
                    or "MAV_STATE_ACTIVE" in self.mav_states
                )
                and self.gps is not None
            ):
                self.drone_ready = True
                with self.drone_ready_condition:
                    self.drone_ready_condition.notify_all()

        elif msg.get_type() == "SYS_STATUS":
            d = msg.to_dict()
            self.sensors_present = lookup_bits(
                d["onboard_control_sensors_present"], sensors_list
            )
            self.ensors_enabled = lookup_bits(
                d["onboard_control_sensors_enabled"], sensors_list
            )
            self.sensors_health = lookup_bits(
                d["onboard_control_sensors_health"], sensors_list
            )
            # print(d)
            # for sensor in sensors_present:
            #    enabled = sensor in sensors_enabled
            #    health = sensor in sensors_health
            #    print(f"\t{sensor}\t{enabled}\t{health}")
        elif msg.get_type() == "STATUSTEXT":
            # {'mavpackettype': 'STATUSTEXT', 'severity': 6, 'text': 'Throttle disarmed', 'id': 0, 'chunk_seq': 0}
            d = msg.to_dict()
            print("\n\n")
            print(d)
            print("\n\n")
        elif msg.get_type() == "MISSION_ITEM_REACHED":
            self.mission_item_reached = True
            with self.mission_item_condition:
                self.mission_item_condition.notify_all()
        elif msg.get_type() == "MISSION_CURRENT":
            # print(
            #    "MISSION CURRENT",
            #    mission_states[msg.mission_state],
            #    msg.mission_mode,
            #    msg.total,
            #    msg.seq,
            # )
            pass
        elif msg.get_type() in self.ignore_messages:
            pass
        else:
            print(f"\t{msg.get_type()}")
            # pass
        # if

    def set_mode(self, mode):
        self.connection.set_mode(mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial", type=str, help="Serial port", required=False, default=""
    )
    parser.add_argument("--ip", type=str, help="ip address", required=False, default="")
    parser.add_argument("--port", type=int, help="port", required=False, default=14552)

    args = parser.parse_args()
    print("WTF")
    # Create the connection
    # Need to provide the serial port and baudrate
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
    connection.wait_heartbeat()

    logging.info("Listening...")

    # get rid of the top?
    msg = connection.recv_match()
    while msg is None or msg.get_type() == "BAD_DATA":
        msg = connection.recv_match()

    drone = Drone(
        connection,
        planner=BouncePlanner(
            dynamics=Dynamics(
                bounding_box=crissy_boundary_convex,
                bounds_radius=0.000000001,
            ),
            start_point=crissy_boundary_convex.mean(axis=0),
            epsilon=0.0000001,
            step_size=0.1,
        ),
        boundary=crissy_boundary_convex,
    )

    # upload_waypoints(connection)

    # do_mission(connection)

    time.sleep(1)

    #   connection.set_mode_auto()  # MAV_CMD_MISSION_START

    # connection.set_mode_auto()

    # connection.set_mode_auto()
    # breakpoint()
    # print("Waiting for the vehicle to arm")
    # connection.motors_armed_wait()
    # print("Armed!")

    time.sleep(2)
    print("MODE", drone.mav_mode)
    # for long, lat in crissy_boundary_convex[1:]:
    #    print(lat, long)
    #    drone.reposition(lat, long)
    #    break

    # print("DONE")
    # drone.process_messages()
