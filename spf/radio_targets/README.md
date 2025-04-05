# Radio Targets

This module provides implementations for various radio targets used in wireless communication experiments and testing. The module supports multiple radio protocols and hardware configurations for different frequency bands and modulation types.

## Overview

The radio_targets module contains implementations for different radio communication protocols:

- **WiFi**: ESP32-based targets for WiFi signal generation
- **LoRa**: Long Range radio implementations 
- **5.8GHz VTX**: Video transmitter targets used in drone/FPV applications

## WiFi Targets

The WiFi implementation uses ESP32 WROOM 32U modules in FCC test mode to emit continuous WiFi data on specific channels with controllable duty cycles.

### Features

- Configurable WiFi channels
- Adjustable transmission power levels
- Variable duty cycle patterns (10%, 50%, 90%)
- Multiple operation modes (continuous transmission, packet-based, tone)
- Arduino companion code for autonomous operation

### Hardware Components

- ESP32 WROOM 32U modules
- IPEX to SMA antenna adapters
- USB to Serial converters for programming
- XIAO microcontroller for companion control

### Setup and Operation

The WiFi targets can be programmed using the ESPTestTool from Espressif and controlled either through direct commands or via a companion Arduino board. See the [detailed WiFi README](/home/mouse9911/gits/spf/spf/radio_targets/wifi/README.md) for specific commands and setup instructions.

## LoRa Targets

The LoRa implementation provides a sender module that transmits LoRa packets with configurable parameters.

### Features

- Operates on 915MHz band (configurable to other bands like 868MHz)
- Random packet generation
- Variable transmission timing
- Adjustable transmission power
- OLED display for status information

### Hardware Components

- Heltec LoRa module with integrated display
- Built-in PABOOST for extended range

### Operation

The LoRa sender operates in cycles of broadcasting and sleeping, with randomized timing to simulate real-world transmission patterns. The device displays current status information on its built-in OLED screen.

## 5.8GHz Video Transmitter (VTX)

This implementation uses drone FPV video transmitters operating in the 5.8GHz band.

### Features

- Multiple frequency channels
- Power level selection
- Color-coded status indication

### Hardware Components

- Zeus Nano VTX transmitter module

### Operation

The VTX module is controlled through long button presses: 
- Long press (2+ seconds) to enter selection mode
- Second long press to move to next color/parameter
- Within each color selection, choose the specific value

## Usage in Signal Processing Framework

These radio targets can be used as signal sources for testing signal processing algorithms, direction finding applications, and RF characterization. They can be integrated with the SPF framework's motion planners to create mobile signal sources for more dynamic testing scenarios.

```python
# Example of integrating a WiFi target with motion planning
from spf.motion_planners.dynamics import Dynamics
from spf.motion_planners.planner import CirclePlanner
import subprocess

# Define boundary for motion
boundary = np.array([
    [1, 1], [1, -1], [-1, -1], [-1, 1]
])

# Create motion planner
dynamics = Dynamics(boundary)
planner = CirclePlanner(
    dynamics=dynamics,
    start_point=np.array([0, 0.8]),
    step_size=0.1,
    circle_diameter=1.5
)

# Control radio target as we move
for point in planner.yield_points():
    # Update position
    position_x, position_y = point
    
    # Adjust radio parameters based on position
    if position_x > 0:
        # Increase power when in positive x region
        subprocess.run(["arduino_command", "--power", "high"])
    else:
        # Decrease power when in negative x region
        subprocess.run(["arduino_command", "--power", "low"])
```

## Dependencies

- Arduino IDE for companion code development
- ESP32 toolchain for WiFi target programming
- Heltec Arduino library for LoRa modules 