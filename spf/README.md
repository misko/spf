# Signal Processing Framework (SPF)

SPF is a comprehensive framework for radio signal acquisition, processing, and analysis with a focus on direction finding and mobile applications. The framework integrates hardware control, motion planning, data collection, and signal processing in a modular architecture.

## Overview

The SPF framework provides tools for:

- Radio data acquisition from various sources (SDR, WiFi, LoRa)
- Motion planning and control for mobile platforms (rovers, drones)
- Signal processing and analysis
- Machine learning model training and inference
- Data management and visualization

## Key Components

### Core Systems

- **Radio Targets (`radio_targets/`)**: Implementations for various radio signal sources
- **Motion Planners (`motion_planners/`)**: Algorithms for trajectory planning and boundary management
- **Data Collection**: Scripts for coordinated movement and signal acquisition
- **Signal Processing (`rf.py`)**: Radio frequency signal processing utilities
- **GRBL and MAVLink Controllers**: Hardware control interfaces

### Hardware Support

- **SDR (Software Defined Radio)**: Support for PlutoSDR devices
- **GRBL**: Control for CNC-based positioning systems
- **MAVLink**: Drone/rover control using ArduPilot
- **GPS**: Position tracking and geofencing
- **Distance Finder**: Ultrasonic distance sensors

## Data Collection Scripts

### GRBL Radio Collection (CNC-based systems)

The `grbl_radio_collection.py` script is used for collecting radio data with GRBL-controlled positioning systems.

#### Usage

```bash
python spf/grbl_radio_collection.py -c CONFIG_FILE -r ROUTINE -s SERIAL_PORT [OPTIONS]
```

#### Key Parameters

- `-c, --yaml-config`: YAML configuration file (required)
- `-r, --routine`: Motion routine to execute (e.g., 'bounce', 'circle')
- `-s, --serial`: Serial port for GRBL controller
- `-t, --tag`: Tag for output files
- `-n, --n-records`: Number of records to collect per receiver (-1 for unlimited)
- `--tx-gain`: Transmission gain for SDR emitters
- `-o, --output-dir`: Output directory for collected data

#### Example

```bash
# Run a bounce pattern with 10000 records
python spf/grbl_radio_collection.py -c spf/v5_configs/wall_array_config.yaml -r bounce -s /dev/ttyACM0 --n-records 10000

# Dry run to test configuration
python spf/grbl_radio_collection.py -c spf/v5_configs/wall_array_config.yaml -r bounce -s /dev/ttyACM0 --dry-run --n-records 1
```

### MAVLink Radio Collection (Drone/Rover systems)

The `mavlink_radio_collection.py` script is used for collecting radio data with MAVLink-controlled vehicles (drones/rovers).

#### Usage

```bash
python spf/mavlink_radio_collection.py -c CONFIG_FILE -m DEVICE_MAPPING [OPTIONS]
```

#### Key Parameters

- `-c, --yaml-config`: YAML configuration file (required)
- `-m, --device-mapping`: Device mapping file for SDR URIs (required)
- `-r, --routine`: Motion routine to execute (e.g., 'bounce', 'circle')
- `-t, --tag`: Tag for output files
- `-n, --records-per-receiver`: Number of records to collect per receiver
- `-d, --drone-uri`: URI for MAVLink connection
- `--ultrasonic`: Enable/disable ultrasonic distance sensor
- `--tx-gain`: Transmission gain for SDR emitters

#### Example

```bash
# Run a bounce pattern on a rover
python spf/mavlink_radio_collection.py -c spf/rover_configs/rover_config.yaml -m /home/pi/device_mapping -r bounce -t "RO1" -n 3000

# Run with a specific drone connection and without ultrasonic sensor
python spf/mavlink_radio_collection.py -c spf/rover_configs/rover_config.yaml -m /home/pi/device_mapping -r circle -t "RO2" -n 40 --drone-uri tcp:192.168.1.141:14590 --no-ultrasonic
```

## Configuration Files

SPF uses YAML configuration files to define:

1. **Radio Settings**: Frequencies, bandwidths, gains, and other SDR parameters
2. **Receiver/Emitter Setup**: Configuration for one or more radio devices
3. **Motion Parameters**: Step sizes, boundaries, and other motion control settings
4. **Data Collection**: Sampling rates, record counts, and file formats

Example configuration structure:

```yaml
data-version: 5
routine: bounce
n-records-per-receiver: 10000
receivers:
  - type: sdr
    receiver-uri: pluto://usb:1.3.5
    center-freq: 2464000000
    sample-rate: 6000000
    bandwidth: 3000000
emitter:
  type: sdr
  emitter-uri: pluto://usb:1.4.5
  center-freq: 2464000000
  tx-gain: 43
  sample-rate: 6000000
  bandwidth: 3000000
```

## Motion Planning

SPF includes a motion planning module that provides various trajectory types:

- **Bounce**: Bounces within boundaries with random direction changes
- **Circle**: Follows a circular path with configurable diameter
- **Point Cycle**: Moves through a predefined sequence of points
- **Stationary**: Remains at a fixed position

These planners can be used with both GRBL and MAVLink controllers.

## Data Processing Pipeline

The typical workflow with SPF involves:

1. **Configuration**: Set up hardware and create configuration files
2. **Data Collection**: Run collection scripts to gather radio data
3. **Processing**: Apply signal processing algorithms to the collected data
4. **Analysis/Inference**: Run models to extract insights or make predictions
5. **Visualization**: Generate plots and visualizations of results

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyMAVLink (for drone/rover control)
- PySDR libraries (for SDR control)
- Machine learning libraries (TensorFlow, PyTorch)

## Examples

See the `notebooks/` directory for examples and tutorials on using different aspects of the SPF framework. 