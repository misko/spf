# SPF (Signal Processing Fun)

## Youtube explanation

[Overview](https://www.youtube.com/watch?v=vj99KvB2AcA)

[Wall array v1 video](https://youtu.be/ljlRKGjBUoE) and [design+data+files](data_collection/2d_wall_array/2d_wall_array_v1)

[Wall array v2 video](https://youtu.be/9S3mgBD47kw) and [design+data+files](data_collection/2d_wall_array/2d_wall_array_v2)

[Rover v1 video](https://youtu.be/aGzOBduHrvo) and [design+data+files](data_collection/rover/rover_v1)

[Rover v2 video](https://youtube.com/shorts/qZmtBzATahE?feature=share) and [design+data+files](data_collection/rover/rover_v2)

Rover v3.1 [design+data+files](data_collection/rover/rover_v3.1)


## Problem statement:

Given a mobile radio receiver (agent/detector) find the optimal (minimal) path of movement to accurately locate physical radio emitters in the operating space

![Example session](images/01_example_animation.gif)
*(top left): The physical X,Y location of 3 static (non-mobile) emitters and one mobile detector (radio receiver). (top right): The raw radio signal from each antenna in the receiver radio array is [processed](spf/notebooks/03_beamformer_wNoise.ipynb) to produce a mapping from angle of incidence to signal strength. The larger the signal strength the more likely the signal originated from the corresponding angle. (bottom left): The ground truth the detector should output over enough time has elapsed. This represents the probability of finding an emitter at any X/Y coordinate. (bottom right): The processed radio signal from (top right) mapped into an image centered around the detectors position.*

## Background:

### The Challenge of Drone Detection

Detecting and locating small drones is increasingly important for security and airspace management. Traditional detection methods face significant challenges:

- Modern drones are small and becoming smaller
- Drones are often made from materials that don't reflect radar effectively
- Their extreme mobility makes tracking difficult with conventional systems

Most drones rely on radio frequency (RF) signals to communicate with operators, transmitting telemetry data and receiving commands. This RF communication provides an opportunity for detection even when visual or radar methods fail.

### Project Approach

This project develops optimal control algorithms for mobile radio receivers to locate RF emitters (such as drones) in an operational area. Key aspects of our approach:

1. **Mobile Detection**: Using airborne or ground-based mobile radio receivers that can dynamically position themselves for optimal signal reception
2. **Signal Integration**: Combining multiple signal measurements over time to build accurate emitter location maps
3. **Path Optimization**: Finding minimal movement paths that maximize information gain about emitter locations

### Technical Implementation

The system works by collecting and processing RF signals over time:

- **Signal Snapshots**: The receiver captures radio signals at discrete time intervals (snapshots)
- **Beamforming**: Raw antenna array data is processed through a [beamformer](spf/notebooks/03_beamformer_wNoise.ipynb) to map signal strength to angle of incidence
- **Spatial Mapping**: Processed signals are projected into a 2D probability map centered on the detector
- **Integration**: Multiple snapshots are combined to improve accuracy and reduce noise

The example below shows a single snapshot from our system. The blue circle represents the detector's position (with its path shown as a tail), while red circles mark emitter locations. The yellow areas in the bottom right image indicate high probability regions where an emitter is likely located based on the current signal processing.

![Example snapshot](images/01_emitter_right_example.png)

## Current Progress

### Signal Processing & Algorithms

The project has implemented several sophisticated algorithms for emitter detection and localization:

#### Filtering Techniques
- **Extended Kalman Filters (EKF)**: Multiple implementations available in `spf/filters/`:
  - Single radio configuration (`ekf_single_radio_filter.py`)
  - Dual radio setup for improved accuracy (`ekf_dualradio_filter.py`)
  - XY coordinate tracking variants (`ekf_dualradioXY_filter.py`)

- **Particle Filters**: Probabilistic approach for robust tracking in noisy environments:
  - Single and dual radio implementations (`particle_single_radio_filter.py`, `particle_dualradio_filter.py`)
  - Neural network enhanced variants (`particle_single_radio_nn_filter.py`, `particle_dual_radio_nn_filter.py`)
  - XY coordinate variants for direct position estimation (`particle_dualradioXY_filter.py`)

#### Motion Planning
- Path planning algorithms in `spf/motion_planners/`:
  - Vehicle dynamics modeling (`dynamics.py`)
  - Optimal trajectory planning (`planner.py`)

#### Neural Network Models
- Advanced deep learning architectures in `spf/model_training_and_inference/models/`:
  - Point-based networks for signal processing (`single_point_networks.py`)
  - Optimized inference implementations (`single_point_networks_inference.py`) 
- Multiple model configurations in the `model_configs/` directory with variations in:
  - Network architecture (single vs. paired)
  - Regularization techniques (dropout, weight decay)
  - Training strategies (data augmentation, normalization)

### Data Collection & Hardware

#### Data Collection Systems
- **Wall-mounted Arrays**:
  - Fixed position antenna arrays for controlled experiments
  - Multiple versions (v1, v2) with different configurations

- **Mobile Platforms**:
  - Rover-based systems for dynamic data collection (v1, v2, v3.1)
  - Movement controlled via GRBL (`spf/grbl/`) or MAVLink (`spf/mavlink/`)
  - Flying drone platforms in experimental phase

#### Hardware Integration
- **SDR Support**:
  - **PlutoSDR** integration (`spf/sdrpluto/sdr_controller.py`)
  - **BladeRF 2.0** integration (`spf/sdrpluto/sdr_controller.py` with the `BladeRFSdr` class)
    - Advanced register-level configurations for phase alignment between channels
    - Support for multiple receivers with serial number identification
  - Signal processing and detrending utilities (`detrend.py`)
  - Abstracted hardware interfaces that work across SDR platforms

- **GPS Integration**:
  - Geolocation utilities (`spf/gps/gps_utils.py`)
  - Boundary definition for operation areas (`spf/gps/boundaries.py`)
  - Support for geofencing and automatic boundary detection

#### Drone Control
- **MAVLink Protocol**:
  - Advanced drone control interface (`spf/mavlink/mavlink_controller.py`)
  - Parameter management (`mavparm.py`)
  - Telemetry collection (`mavlink_radio_collection.py`)
  - Support for both real and simulated drones with the same API
  - Safety features including return-to-home functionality

### Dataset Management & Visualization

- **Dataset Generation & Processing**:
  - Custom dataset implementations (`spf/dataset/spf_dataset.py`)
    - Support for multiple SDR devices (PLUTO, BLADERF2, SIMULATION)
    - Data augmentation (flipping, striding, etc.)
  - Segmentation algorithms (`spf/dataset/segmentation.py`)
  - Synthetic data generation (`spf/dataset/fake_dataset.py`)

- **Analysis Tools**:
  - Extensive notebook collection in `spf/notebooks/` for:
    - Beamformer verification
    - Filter development and testing
    - Data visualization
    - Performance analysis

- **Evaluation Reports**:
  - Results tracked in `reports/` directory with multiple iterations
  - Analysis pipelines for algorithm comparison

### Cloud & Infrastructure

- **AWS Integration**:
  - S3 utilities for data storage and retrieval (`s3_utils.py`)
  - Cloud-based model training

- **Docker Support**:
  - Containerized development and deployment
  - Reproducible environments

## Test Suite

The project includes comprehensive testing to ensure reliability of all components:

### Signal Processing Tests
- **Beamformer Tests** (`test_beamformer.py`): Verifies beamforming algorithms for both uniform linear arrays (ULA) and uniform circular arrays (UCA) across multiple signal conditions
- **Segmentation Tests** (`test_beamformer.py`): Validates signal segmentation algorithms that identify continuous signal windows with stable phase relationships
- **Circular Mean Tests** (`test_beamformer.py`): Ensures accuracy of circular statistics used for angle estimation

### Filter Tests
- **Particle Filter Tests** (`test_particle_filter.py`): Tests single and dual radio particle filter implementations, including:
  - Single theta tracking with single radio
  - Single theta tracking with dual radio
  - XY-coordinate tracking with dual radio
- **EKF Tests** (`test_ekf.py`): Validates Extended Kalman Filter implementations for emitter tracking

### Motion Planning Tests
- **Dynamics Tests** (`test_dyanmics_and_planners.py`): Verifies vehicle dynamics models for both GRBL and GPS-based systems
- **Bounce Planning Tests** (`test_dyanmics_and_planners.py`): Tests the bounce trajectory planning algorithm to ensure efficient area coverage
- **GRBL Tests** (`test_grbl.py`): Detailed testing of the GRBL motion control system:
  - Coordinate transformations (steps to/from real-world coordinates)
  - Movement boundaries and collision avoidance
  - Simple move and bounce motion patterns
  - Radio collection during movement

### Hardware Integration Tests
- **MAVLink Tests** (`test_in_simulator.py`): Tests drone control via MAVLink in a simulated environment:
  - GPS time and boot time retrieval
  - Parameter loading and verification
  - Mode setting (manual, guided)
  - Drone movement commands
  - Signal recording during movement
- **GRBL Radio Collection Tests** (`test_grbl.py`): Tests collection of radio signals while using GRBL motion control

### Data Processing Tests
- **GPS Conversion Tests** (`test_gps.py`): Validates GPS coordinate processing, including:
  - GPS circle generation and bearing calculations
  - GPS to XY coordinate conversion
- **Dataset Tests** (`test_dataset.py`): Tests dataset creation and manipulation:
  - Data generation
  - Ground truth validation
  - Live data generation
  - V4 and V5 dataset format compatibility
- **Zarr Tools Tests** (`test_zarr_tools.py`): Validates zarr-based data storage operations
  
### Other Specialized Tests
- **Model Tests** (`test_models.py`, `test_new_models.py`): Verifies neural network model implementations
- **Rotation Distribution Tests** (`test_rotate_dist.py`, `test_paired_rotate_dist.py`): Tests rotation-based data augmentation
- **Fake Data Tests** (`test_fake_data.py`): Validates synthetic data generation
- **Invariance Tests** (`test_invariance.py`): Tests invariance properties of models to various transformations

## FAQ 

### Why use an airborne radio receiver?

To avoid multipath issues and noise from ground-based sources. Airborne receivers have a clearer line of sight to emitters and experience less signal reflection and interference from buildings, terrain, and other obstacles.

### What is a radio receiver in this context?

It can be any array of antenna receivers configured to a tunable radio so that the agent can scan across frequencies. In the current simulation setup, the array can be simulated as any number of linear or circular elements and processed through a relatively simple beamformer algorithm to produce the map of angle to signal strength required for input.

### What frequency bands does the system operate in?

The system is flexible and can be configured for different frequency bands. Current implementations primarily focus on 2.4 GHz (common for drone communications) but the architecture supports other frequencies that the SDR hardware can tune to.

### What's the difference between the different filter methods?

- **Extended Kalman Filters (EKF)** provide optimal state estimation when the system dynamics and measurement models are approximately linear and noise is Gaussian. They're computationally efficient but can fail with highly non-linear systems.
- **Particle Filters** use a set of weighted samples (particles) to represent the probability distribution of the state estimate. They're more robust to non-linearities and non-Gaussian noise but typically require more computational resources.
- **Neural Network Enhanced Filters** combine traditional filtering approaches with learned components to better handle complex signal environments and improve tracking accuracy.

### Can the system track multiple emitters simultaneously?

Not yet, soon.

### What are the hardware requirements for running the system?

For basic simulation and algorithm development:
- A standard computer with at least 8GB RAM
- Python 3.10 or newer
- CUDA-compatible GPU (optional, but recommended for neural network training)

For hardware deployment:
- SDR hardware (PlutoSDR or BladeRF 2.0)
- Mobile platform (rover or drone)
- Raspberry Pi or similar embedded computer for onboard processing
- GPS module for outdoor deployments

### How does the data format work?

The project uses Zarr arrays stored in LMDB databases for efficient data storage and access. The data format has evolved through several versions (currently v5) with enhancements for better performance and features. Each dataset contains:
- Raw IQ samples from the radio
- Metadata about detector position and orientation
- Signal processing results
- Ground truth information (for simulated data)

## Installation

### Software Setup

#### For Ubuntu/Debian Linux:

```bash
# Install system dependencies
sudo apt-get install git screen libiio-dev libiio-utils vim python3-dev uhubctl libusb-dev libusb-1.0-0-dev sshpass python3.10-venv -y

# Create and activate Python virtual environment
python3 -m venv spf_venv
source spf_venv/bin/activate

# Install the package in development mode
pip install -e .

# Run the tests to verify installation
pytest tests
```


### Hardware Setup

#### PlutoSDR Setup:

1. Connect your PlutoSDR to your computer via USB
2. Verify the connection:
   ```bash
   iio_info -s
   ```
3. Configure the PlutoSDR (if needed):
   ```bash
   python -m spf.sdrpluto.sdr_controller --setup
   ```

#### BladeRF 2.0 Setup:

1. Install the BladeRF libraries and Python bindings:
   ```bash
   # Clone the BladeRF repository
   git clone --depth 1 https://github.com/Nuand/bladeRF.git
   
   pushd bladeRF/host
   mkdir build && cd build
   cmake ..
   make
   sudo make install
   sudo ldconfig
   popd 

   # Install the Python bindings
   cd bladeRF/host/libraries/libbladeRF_bindings/python
   python setup.py install
   ```




