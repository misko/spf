# SDR Interface Library

## Introduction

This library provides a unified interface for controlling Software-Defined Radios (SDRs) in the SPF project. It currently supports the following SDR platforms:

- PlutoPlus SDR - based on the AD9361 with two coherent TX and RX ports
- BladeRF - supporting similar functionality for signal processing tasks

## Key Interface Classes

### Configuration Classes

- `ReceiverConfig` - Configuration parameters for SDR receivers
- `EmitterConfig` - Configuration parameters for SDR emitters

### SDR Controller Classes

- `PPlus` - Controller for PlutoPlus SDR
- `BladeRFSdr` - Controller for BladeRF devices
- `FakePPlus` - Emulates an SDR for testing purposes

## Core Interface Functions

### SDR Setup Functions

```python
def setup_rx(rx_config, provided_pplus_rx=None):
    """
    Initialize and configure a receiver SDR.
    
    Args:
        rx_config (ReceiverConfig): Configuration for the receiver
        provided_pplus_rx (PPlus, optional): Existing SDR object to configure
        
    Returns:
        PPlus: Configured receiver SDR object
    """
```

```python
def setup_rxtx(rx_config, tx_config, leave_tx_on=False, provided_pplus_rx=None):
    """
    Initialize and configure both receiver and transmitter SDRs.
    
    Args:
        rx_config (ReceiverConfig): Configuration for the receiver
        tx_config (EmitterConfig): Configuration for the transmitter
        leave_tx_on (bool): Whether to leave the transmitter active after setup
        provided_pplus_rx (PPlus, optional): Existing receiver SDR object
        
    Returns:
        tuple: (receiver_sdr, transmitter_sdr)
    """
```

```python
def setup_rxtx_and_phase_calibration(rx_config, tx_config, tolerance=0.01, 
                                    n_calibration_frames=800, leave_tx_on=False, 
                                    using_tx_already_on=None):
    """
    Setup receiver and transmitter SDRs and perform phase calibration.
    
    The function assumes the emitter is equidistant from both RX ports,
    calculates the phase difference, and sets calibration values.
    
    Args:
        rx_config (ReceiverConfig): Configuration for the receiver
        tx_config (EmitterConfig): Configuration for the transmitter
        tolerance (float): Maximum allowed standard deviation for phase calibration
        n_calibration_frames (int): Number of frames to use for calibration
        leave_tx_on (bool): Whether to leave the transmitter active after setup
        using_tx_already_on (PPlus, optional): Existing transmitter object
        
    Returns:
        tuple: (receiver_sdr, transmitter_sdr)
    """
```

### Configuration Helper Functions

```python
def rx_config_from_receiver_yaml(receiver_yaml):
    """Convert YAML configuration to ReceiverConfig object"""
```

```python
def args_to_rx_config(args):
    """Convert command line arguments to ReceiverConfig object"""
```

```python
def args_to_tx_config(args):
    """Convert command line arguments to EmitterConfig object"""
```

## Integration with Data Collection

The SDR interface is used in two primary data collection scenarios:

### GRBL-based Data Collection

In `grbl_radio_collection.py`, SDRs are configured and used to collect data while a GRBL controller moves emitters or receivers. The workflow is:

1. Load configuration from YAML file
2. Set up GRBL controller
3. Initialize data collector with SDR settings
4. Call `radios_to_online()` to set up all SDRs
5. Start data collection process
6. Data collector uses `ThreadedRX` classes to continuously capture SDR data

### MAVLink-based Data Collection

In `mavlink_radio_collection.py`, SDRs are used to collect data from drones or rovers controlled via MAVLink. The workflow is:

1. Load configuration from YAML and device mapping file
2. Set up drone/rover connection
3. Initialize data collector with SDR settings
4. Start data collection process
5. Continuously capture SDR data while the vehicle moves

## Command Line Usage

For basic testing and manual control, use the `sdr_controller.py` script directly:

```bash
# Receiver only mode
python sdr_controller.py --receiver-ip <IP_ADDRESS> --mode rx

# Transmitter only mode 
python sdr_controller.py --emitter-ip <IP_ADDRESS> --mode tx

# Receiver with phase calibration
python sdr_controller.py --receiver-ip <IP_ADDRESS> --emitter-ip <IP_ADDRESS> --mode rxcal
```

## Additional Documentation

For specific information about the PlutoSDR implementation, please see the [plutosdr documentation](../docs/plutosdr.md).

For BladeRF specific details, refer to the [bladerf documentation](../docs/bladerf.md).