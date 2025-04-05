# GRBL Control System

This directory contains the control system for GRBL-based CNC/motion control hardware. The system is designed to control stepper motors with various driver configurations and includes support for different movement patterns and calibration routines.

## Key Components

### Core Files
- `grbl_interactive.py`: Main control interface that provides classes for:
  - Motion dynamics and boundary management
  - Various movement planners (Bounce, Stationary, Circle, etc.)
  - GRBL controller and manager implementation
  - Safety features including boundary checking

- `run_grbl.py`: Simple G-code streaming utility to send commands to GRBL hardware
- `mm_per_step.py`: Configuration calculator for stepper motor steps/mm based on:
  - Gear ratios
  - Microstepping settings
  - Physical parameters (teeth count, tooth spacing)

### Configuration Files
- `grbl_settings_A4988`: Settings for A4988 stepper driver
- `grbl_settings_A4988_slow`: Conservative settings for A4988 driver
- `grbl_settings_drv8225`: Settings for DRV8225 stepper driver
- `grbl_settings_tmc2208`: Settings for TMC2208 stepper driver
- `grbl_setup`: Basic GRBL setup instructions

### Test Files
- `x30cm.gcode`: Simple G-code test file

## Features

### Movement Planners
The system includes several movement planning strategies:
- Bounce planning for random motion within boundaries
- Circular motion planning
- Stationary point planning
- Point cycle planning
- Calibration routines

### Safety Features
- Boundary checking and enforcement
- Convex hull validation for movement areas
- Emergency stop functionality
- Real-time position monitoring

### Hardware Support
- Compatible with multiple stepper driver types:
  - A4988
  - DRV8225
  - TMC2208
- Configurable microstepping
- Adjustable motion parameters

## Usage

### Basic Setup
1. Choose appropriate settings file for your stepper driver
2. Configure steps/mm using `mm_per_step.py`
3. Upload settings to GRBL controller

### Running G-code
```bash
python run_grbl.py /dev/ttyUSB0 your_gcode_file.gcode
```

### Interactive Control
Use `grbl_interactive.py` for advanced motion control and testing:
```python
from grbl_interactive import GRBLManager, get_default_gm
# See grbl_interactive.py for detailed usage examples
```

## Hardware Documentation
Additional hardware-specific documentation can be found in the `hardware_documentation/` directory.

## Configuration Parameters

### Motor Settings
- Default microstepping: 16
- Steps per revolution calculation based on:
  - Base step angle: 1.8°
  - Gear ratio: 26 + 103/121
  - Teeth per revolution: 20
  - Tooth spacing: 2mm

### Motion Parameters
- Configurable acceleration
- Adjustable maximum speeds
- Boundary enforcement for safe operation

## Safety Notes
1. Always test movements with slower speeds first
2. Ensure proper boundary configuration before operation
3. Keep emergency stop access available
4. Verify stepper driver settings match configuration files 