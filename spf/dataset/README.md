# Dataset


## Dataset Loader

The SPF dataset module provides tools for loading and processing signal processing datasets, particularly for radio signal analysis and direction finding applications. The main class is `v5spfdataset`, which handles loading, preprocessing, and providing access to the dataset.

### Installation

The dataset loader is part of the SPF package. Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Key Components

- **v5spfdataset**: Main dataset class that loads and processes SPF data
- **v5spfdataset_manager**: Context manager for easier resource management
- **Utility functions**: Various utility functions for data transformation, collation, and processing

### Basic Usage

```python
from spf.dataset.spf_dataset import v5spfdataset

# Initialize the dataset
dataset = v5spfdataset(
    prefix="path/to/dataset",          # Base path for dataset files (without .zarr extension)
    nthetas=65,                        # Number of theta angles for beamforming discretization
    precompute_cache="path/to/cache",  # Directory to store/load precomputed segmentation data
    ignore_qc=False,                   # Whether to skip quality control checks
    paired=False,                      # Whether to return paired samples from all receivers at once
    gpu=False                          # Whether to use GPU for computation
)

# Access a sample from the dataset
sample = dataset[0]

# Get the total number of samples
length = len(dataset)

# Close the dataset (release resources)
dataset.close()
```

### Using the Context Manager

For proper resource management, use the context manager:

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

with v5spfdataset_manager(
    "path/to/dataset",
    precompute_cache="path/to/cache",
    nthetas=65,
    paired=True,
    ignore_qc=True
) as dataset:
    # Use dataset here
    sample = dataset[0]
    # Resources will be automatically released when exiting the context
```

### Important Parameters

- **prefix**: Base path for dataset files (without .zarr extension)
- **nthetas**: Number of theta angles for beamforming discretization
- **precompute_cache**: Directory to store/load precomputed segmentation data
- **phi_drift_max**: Maximum allowable phase drift for quality control (default: 0.2)
- **min_mean_windows**: Minimum average number of windows required (default: 10)
- **ignore_qc**: Skip quality control checks if True (default: False)
- **paired**: Return paired samples from all receivers at once (default: False)
- **gpu**: Use GPU for segmentation computation if available (default: False)
- **snapshots_per_session**: Number of snapshots to include in each returned session (default: 1)
- **tiled_sessions**: If True, sessions overlap with stride; if False, sessions are disjoint (default: True)
- **snapshots_stride**: Step size between consecutive session starting points (default: 1)
- **readahead**: Enable read-ahead for zarr storage I/O optimization (default: False)
- **temp_file**: Whether dataset is using temporary files (default: False)
- **temp_file_suffix**: Suffix for temporary files (default: ".tmp")
- **skip_fields**: Data fields to exclude during loading to save memory (default: [])
- **n_parallel**: Number of parallel processes for segmentation (default: 20)
- **segment_if_not_exist**: Generate segmentation cache if missing (default: False)

### Data Format

Each item in the dataset is a dictionary containing various tensors and metadata:

- **x**: Raw signal data
- **y_rad**: Ground truth angle in radians
- **y_phi**: Phase difference
- **rx_pos_xy**: Receiver position
- **craft_y_rad**: Craft ground truth angle
- **windowed_beamformer**: Beamformer data for each window
- **all_windows_stats**: Statistics for each window
- **segmentation_mask**: Segmentation mask for the data

### Example: Working with Paired Data

```python
from spf.dataset.spf_dataset import v5spfdataset

dataset = v5spfdataset(
    prefix="path/to/dataset",
    nthetas=65,
    precompute_cache="path/to/cache",
    paired=True  # Return paired data from all receivers
)

# Get a paired sample (returns a list with data from each receiver)
paired_sample = dataset[0]

# Access data from first receiver
receiver_0_data = paired_sample[0]

# Access data from second receiver
receiver_1_data = paired_sample[1]

# Access specific fields
theta_receiver_0 = receiver_0_data["y_rad"]
```

### Example: Using Temporary Files

```python
from spf.dataset.spf_dataset import v5spfdataset

dataset = v5spfdataset(
    prefix="path/to/dataset",
    nthetas=65,
    precompute_cache="path/to/cache",
    temp_file=True,
    temp_file_suffix=".tmp"  # Files with this suffix will be considered temporary
)
```

### Advanced: Performance Optimization

To optimize memory usage:

```python
dataset = v5spfdataset(
    prefix="path/to/dataset",
    nthetas=65,
    precompute_cache="path/to/cache",
    skip_fields=["signal_matrix"],  # Skip loading large fields not needed
    readahead=True,                 # Enable read-ahead for I/O optimization
    gpu=True                        # Use GPU for computations if available
)
```

### Example: Working with Segmentation

```python
dataset = v5spfdataset(
    prefix="path/to/dataset",
    nthetas=65,
    precompute_cache="path/to/cache",
    segment_if_not_exist=True,  # Generate segmentation if it doesn't exist
    windows_per_snapshot=256    # Set maximum number of windows per snapshot
)

# Access segmentation masks
sample = dataset[0]
mask = sample["segmentation_mask"]
downsampled_mask = sample["downsampled_segmentation_mask"]
```