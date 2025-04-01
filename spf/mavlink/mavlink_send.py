import random
import time

from pymavlink import mavutil

# -- User configuration --
serial_port = "/dev/ttyACM1"  # Serial port to Pixhawk (e.g., '/dev/ttyAMA0' or '/dev/ttyS0' on Raspberry Pi)
baud_rate = 57600  # Baud rate matching SERIAL2_BAUD on Pixhawk (57600 is an example)

# Optional: define the MAVLink system and component IDs for the Raspberry Pi
# It's good practice to use an ID not equal to the Pixhawk's system ID (1) or GCS's ID (255) to avoid conflicts.
system_id = 2  # Arbitrary non-1, non-255 ID for the Pi
component_id = 191  # Component ID for the Pi (191 is just an example)

# Connect to the Pixhawk over serial
print(f"Connecting to Pixhawk on {serial_port} at {baud_rate} baud...")
master = mavutil.mavlink_connection(
    serial_port,
    baud=baud_rate,
    source_system=system_id,
    source_component=component_id,
    dialect="ardupilotmega",
)

# Wait for the Pixhawk to send a heartbeat message, to ensure the connection is established
print("Waiting for heartbeat from Pixhawk...")
master.wait_heartbeat(timeout=10)
print(f"Heartbeat received from Pixhawk (system {master.target_system})")

# Now continuously read (or generate) sensor data and send MAVLink messages
# In this example, we send a float value under the name "TEMP" once per second.
start_time = time.time()
try:
    while True:
        # 1. Get sensor reading (replace this with actual sensor code as needed)
        sensor_value = random.uniform(
            20.0, 30.0
        )  # e.g., simulate a temperature between 20 and 30°C

        # 2. Prepare a timestamp in milliseconds since boot (relative to this script’s start)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 3. Send the NAMED_VALUE_FLOAT MAVLink message with a name and value
        # 'name' should be up to 10 characters. It's sent as a byte string in MAVLink.
        master.mav.named_value_float_send(
            elapsed_ms,  # time since system boot (ms)
            b"TEMP",  # name of the variable (max 10 bytes, ideally <= 9 chars to leave room for null terminator)
            float(sensor_value),  # float value
        )

        # Print debug info to console (optional)
        print(f"Sent sensor value: {sensor_value:.2f}")

        # 4. Wait before sending the next reading
        time.sleep(1.0)
except KeyboardInterrupt:
    print("Stopped by user")
