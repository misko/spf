# SPF Dataset Usage Examples

This document demonstrates how to use the `v5spfdataset` class with different `temp_file` configurations, including data loading, pre-processing, and filtering options.

## Understanding the `temp_file` Parameter

The `temp_file` parameter in `v5spfdataset` controls how the dataset handles files that are still being written or are incomplete:

- When `temp_file=True`: Used for accessing **in-progress recordings** where data is actively being written. Files are expected to have a temporary extension (default `.tmp`).
- When `temp_file=False`: Used for accessing **completed recordings** where all data is finalized and no new data will be added.

## Example 1: Using `temp_file=False` (Default)

This is the standard mode for processing completed datasets:

```python
from spf.dataset.spf_dataset import v5spfdataset

# Load a complete dataset
ds = v5spfdataset(
    "/path/to/your/dataset",  # Path to your dataset without .zarr extension
    nthetas=65,                # Number of angle bins for beamforming
    precompute_cache="/path/to/cache_dir",  # Where to store preprocessed data
    paired=True,               # Return paired samples from all receivers at once
    ignore_qc=True,            # Skip quality control checks
    skip_fields=set(["signal_matrix"]),  # Skip loading raw signals to save memory
    segment_if_not_exist=True, # Generate segmentation cache if missing
    gpu=True,                  # Use GPU for signal processing if available
)

# Access a sample (returns data from all receivers for this snapshot)
sample = ds[100]  

# Access specific data fields
theta_rx0 = sample[0]["rx_theta_in_pis"]   # Receiver 0's orientation
theta_rx1 = sample[1]["rx_theta_in_pis"]   # Receiver 1's orientation
beamformer_rx0 = sample[0]["windowed_beamformer"]  # Beamforming results from rx0
```

Benefits of `temp_file=False`:
- Assumes all data is available and generates full preprocessing cache
- Better for training models or performing analysis on complete recordings
- All available snapshots will be processed
- Preprocessing cache is saved permanently for faster future access

## Example 2: Using `temp_file=True` for Live Data

This mode is used for accessing datasets that are actively being recorded:

```python
import tempfile
from spf.dataset.spf_dataset import v5spfdataset

# Create a temporary directory for processing cache
with tempfile.TemporaryDirectory() as tmp_cache:
    # Load a dataset that's currently being recorded
    ds = v5spfdataset(
        "/path/to/active/dataset",  # Path without .zarr extension
        nthetas=65,
        precompute_cache=tmp_cache,
        paired=True,
        ignore_qc=True,
        temp_file=True,             # Indicate this is a temporary/in-progress file
        temp_file_suffix=".tmp",    # The suffix used for the temporary files
        snapshots_per_session=1,    # Must be 1 for temp_file=True
        skip_fields=set(["signal_matrix"]),
    )
    
    # The dataset automatically detects how many valid entries are available
    print(f"Valid entries: {ds.valid_entries}")
    
    # Access the latest available sample (assuming idx is within valid range)
    latest_idx = min(ds.valid_entries) - 1
    if latest_idx >= 0:
        latest_sample = ds[latest_idx]
        
        # Process data from each receiver
        for rx_idx, rx_data in enumerate(latest_sample):
            print(f"Receiver {rx_idx} beamformer peak: {rx_data['windowed_beamformer'].argmax()}")
            print(f"Receiver {rx_idx} phase: {rx_data['mean_phase_segmentation']}")
```

Benefits of `temp_file=True`:
- Works with in-progress recordings that are still being written
- Automatically detects how many valid entries are available in the dataset
- Returns `None` for indices beyond valid entries
- Uses temporary cache for processing (ideal for monitoring live data)
- Can be used for real-time processing during data collection

## Example 3: Using with a Context Manager

For better resource management, you can use the `v5spfdataset_manager` context manager:

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

# Example with temp_file=False (complete dataset)
with v5spfdataset_manager(
    "/path/to/dataset",
    precompute_cache="/path/to/cache",
    nthetas=65,
    paired=True,
    ignore_qc=True,
    gpu=False,
    temp_file=False,
    segment_if_not_exist=True,
) as ds:
    # Process all data
    for i in range(len(ds)):
        sample = ds[i]
        # Process sample...

# Example with temp_file=True (in-progress dataset)
with v5spfdataset_manager(
    "/path/to/active/dataset",
    precompute_cache="/tmp/cache",
    nthetas=65,
    paired=True,
    ignore_qc=True,
    gpu=False,
    temp_file=True,
    temp_file_suffix=".tmp",
    segment_if_not_exist=False,  # Don't regenerate segmentation for temp files
) as ds:
    # Process only valid entries
    for i in range(min(ds.valid_entries)):
        sample = ds[i]
        # Process sample...
```

## Pre-processing and Filtering

Both modes support the same pre-processing and filtering options:

### Key Pre-processing Features

1. **Signal Segmentation**:
   - Identifies valid signal windows vs. noise
   - Computes phase differences between antenna elements
   - Extracts statistical features from signal windows

2. **Beamforming**:
   - Creates power profiles across different angular directions
   - Typically discretized into 65 angle bins (nthetas=65)

3. **Data Field Selection**:
   - Use `skip_fields` to avoid loading large data (like raw signals)
   - Common fields to skip: `"signal_matrix"`, `"windowed_beamformer"`

### Common Data Fields

Both modes provide access to these pre-processed fields:

```python
# Common fields available in each sample
sample = ds[idx][rx_idx]  # Get sample for receiver rx_idx

# Positional information
tx_pos_x = sample["tx_pos_x_mm"]  # Transmitter X position in mm
tx_pos_y = sample["tx_pos_y_mm"]  # Transmitter Y position in mm
rx_pos_x = sample["rx_pos_x_mm"]  # Receiver X position in mm
rx_pos_y = sample["rx_pos_y_mm"]  # Receiver Y position in mm

# Direction information
rx_heading = sample["rx_heading_in_pis"]  # Receiver heading in π units
rx_theta = sample["rx_theta_in_pis"]      # Receiver angle to transmitter in π units

# Signal processing results
beamformer = sample["windowed_beamformer"]  # Beamforming output (if not skipped)
phase_seg = sample["mean_phase_segmentation"]  # Mean phase from segmentation
```

## Practical Considerations

1. When using `temp_file=True`:
   - Always check that indices are within `valid_entries` before accessing
   - Set `snapshots_per_session=1` (required for temp files)
   - Consider using a temporary directory for the cache

2. When using `temp_file=False`:
   - You can use any value for `snapshots_per_session`
   - Pre-processing cache is persistent for future use
   - Use `segment_if_not_exist=True` to ensure segmentation is available

3. For both modes:
   - Use `paired=True` to get data from all receivers simultaneously
   - Set `skip_fields` to optimize memory usage based on your needs 