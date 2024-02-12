# Import mavutil
import argparse
import logging
import time

from pymavlink import mavutil

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


def lookup_bits(x, table):
    return [y for y in table if x & getattr(mavutil.mavlink, y)]


def lookup_exact(x, table):
    return [y for y in table if x == getattr(mavutil.mavlink, y)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial",
        type=str,
        help="Serial port",
        required=True,
    )

    args = parser.parse_args()

    # Create the connection
    # Need to provide the serial port and baudrate
    logging.info("Connecting...")
    pixhawk = mavutil.mavlink_connection(args.serial, baud=115200)

    logging.info("Wait heartbeat...")
    pixhawk.wait_heartbeat()

    logging.info("Listening...")

    def request_home(connection):
        message = connection.mav.command_long_encode(
            connection.target_system,  # Target system ID
            connection.target_component,  # Target component ID
            mavutil.mavlink.MAV_CMD_GET_HOME_POSITION,  # ID of command to send
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

    # get rid of the top?
    msg = pixhawk.recv_match()
    while msg is None or msg.get_type() == "BAD_DATA":
        msg = pixhawk.recv_match()

    request_home(pixhawk)

    pixhawk.mav.set_mode_send(
        pixhawk.target_system, mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY, 0
    )

    pixhawk.mav.command_long_send(
        pixhawk.target_system,
        pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    # print("Waiting for the vehicle to arm")
    # pixhawk.motors_armed_wait()
    # print("Armed!")

    while True:
        if True:
            msg = pixhawk.recv_match()
            if msg is None:
                time.sleep(0.01)
                continue
            # .to_dict())
            if msg.get_type() == "GLOBAL_POSITION_INT":
                d = msg.to_dict()
                lat = d["lat"] * 1e-7
                lon = d["lon"] * 1e-7
                print("GPS", lat, lon)
            elif msg.get_type() == "COMMAND_ACK":
                d = msg.to_dict()
                print(d)
            elif msg.get_type() == "HOME_POSITION":  # also maybe GPS_GLOBAL_ORIGIN
                d = msg.to_dict()
                print(d)
            elif msg.get_type() == "AHRS":
                d = msg.to_dict()
                print(d)
            elif msg.get_type() == "HEARTBEAT":
                d = msg.to_dict()
                print(d)
                print(lookup_exact(d["system_status"], mav_states_list))
            elif msg.get_type() == "SYS_STATUS":
                d = msg.to_dict()
                sensors_present = lookup_bits(
                    d["onboard_control_sensors_present"], sensors_list
                )
                sensors_enabled = lookup_bits(
                    d["onboard_control_sensors_enabled"], sensors_list
                )
                sensors_health = lookup_bits(
                    d["onboard_control_sensors_health"], sensors_list
                )
                print(d)
                for sensor in sensors_present:
                    enabled = sensor in sensors_enabled
                    health = sensor in sensors_health
                    print(f"\t{sensor}\t{enabled}\t{health}")
            elif msg.get_type() == "STATUSTEXT":
                # {'mavpackettype': 'STATUSTEXT', 'severity': 6, 'text': 'Throttle disarmed', 'id': 0, 'chunk_seq': 0}
                d = msg.to_dict()
                print("\n\n")
                print(d)
                print("\n\n")
            else:
                # print(f"\t{msg.get_type()}")
                pass
            # if
