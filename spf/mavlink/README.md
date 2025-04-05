# MAVLink subsystem

Goal of this subsystem is to provide an interface for the onboard computer to control motion through ardupilot VIA (MAVlink)

## Components

### mavlink_controller.py

This is the core component that facilitates communication with ArduPilot via MAVLink. It provides:
- Drone class for handling MAVLink messages
- Various planners (BouncePlanner, CirclePlanner, StationaryPlanner) for autonomous movement
- Capabilities for parameter management, mode switching, and system monitoring
- Safety features like collision avoidance using distance finder

### mavlink_radio_collection.py

Located in the parent directory, this script allows data collection using radio devices while the drone is in motion. It:
- Connects to the drone via MAVLink
- Sets up and configures radio receivers and emitters
- Collects data while the drone moves according to planned paths
- Handles system time synchronization and data saving

### mavparm.py

Provides parameter management functionality for MAVLink devices.

## Development

### Simulator Setup

Use Windows machine to run Mission Planner, then start rover simulator, setup mavlink mirror (setup->advanced) to forward traffic to a network IP over UDP. **MAKE SURE WRITE ACCESS CHECK BOX IS CLICKED!!!**

For example, forward to 192.168.1.139 UDP port 14551

### Docker Simulator (for Testing)

The test suite uses a Docker container to run an ArduPilot simulator:
```
docker run --rm -it -p 14590-14595:14590-14595 csmisko/ardupilotspf:latest /ardupilot/Tools/autotest/sim_vehicle.py \
   -l 37.76509485,-122.40940127,0,0 -v rover -f rover-skid \
   --out tcpin:0.0.0.0:14590 --out tcpin:0.0.0.0:14591 -S 1
```

### MAVLink Proxy

Use mavproxy.py to route MAVLink messages between devices:

```
mavproxy.py --master=udp:192.168.1.139:14551 --out 127.0.0.1:14550 --out 192.168.1.139:14552
mavproxy.py --master=tcp:192.168.1.127:14560 --out 127.0.0.1:14550 --out 127.0.0.1:14552
mavproxy.py --master=tcp:192.168.1.127:14560 --out 127.0.0.1:14550 --out 127.0.0.1:14552
```

## Usage Examples

### Mavlink Controller Script

This script takes in a listening IP (or serial interface, live on rover) that it uses to connect to the drone VIA MAVLink. Once connected the script uses the same BouncePlanner from the WallArray to bounce the drone in a convex boundary defined by GPS.

Youtube [link](https://youtu.be/b0P2JzziI_M)

```
python mavlink_controller.py --ip 192.168.1.139
```

Additional options:
```
--port PORT           # Port number
--proto PROTO         # Protocol (tcp/udp)
--routine ROUTINE     # Movement routine (bounce/circle/center)
--boundary BOUNDARY   # GPS boundary to use
--timeout TIMEOUT     # Connection timeout
--mode MODE           # Set drone mode (guided/manual)
--buzzer TONE         # Play a buzzer tone
```

### Radio Data Collection

To collect radio data while the drone is moving:

```
python mavlink_radio_collection.py -c config.yaml -m device_mapping -r circle
```

Key options:
```
-c, --yaml-config CONFIG    # YAML configuration file
-m, --device-mapping MAP    # Device mapping file
-r, --routine ROUTINE       # Movement routine
-s, --run-for-seconds SEC   # Run duration in seconds
-n, --records-per-receiver N # Number of records per receiver
--fake-drone               # Use simulated drone
```

## Testing

The system includes comprehensive test scripts:

### test_in_simulator.py

Tests the MAVLink controller and radio collection in a simulated environment. Verifies:
- GPS time synchronization
- System parameter management
- Mode switching
- Buzzer functionality
- Data collection in both stationary and moving modes

### test_mavlink_radio_collect.py

Tests the radio collection system with a simulated drone, verifying that:
- Data collection works properly
- All required data fields are present and valid
- File output is correctly formatted