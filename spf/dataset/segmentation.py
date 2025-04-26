import os
import pickle
from multiprocessing import Pool, cpu_count

from spf.utils import SEGMENTATION_VERSION

try:
    import cupy as cp
except:
    pass
import numpy as np
import torch
import tqdm
from scipy.stats import trim_mean

from spf.rf import (
    beamformer_given_steering_nomean,
    beamformer_given_steering_nomean_cp,
    get_phase_diff,
    get_stats_for_signal,
    mean_phase_mean,
    pi_norm,
    reduce_theta_to_positive_y,
    windowed_trimmed_circular_mean_and_stddev,
)
from spf.scripts.zarr_utils import (
    new_yarr_dataset,
    zarr_open_from_lmdb_store,
    zarr_open_from_lmdb_store_cm,
    zarr_shrink,
)
from spf.sdrpluto.detrend import detrend_np

DEFAULT_SEGMENT_ARGS = {
    "window_size": 2048,
    "stride": 2048,
    "trim": 20.0,
    "mean_diff_threshold": 0.2,
    "max_stddev_threshold": 0.5,
    "drop_less_than_size": 3000,
    "min_abs_signal": 40,
}


def segment_single_session(
    zarr_fn, steering_vectors_for_receiver, session_idx, receiver_idx, gpu=False
):
    """
    Wrapper function to segment a single session of data from one receiver.

    Args:
        zarr_fn: Path to the zarr file containing raw signal data
        steering_vectors_for_receiver: Precomputed steering vectors for beamforming
        session_idx: Index of the session to process
        receiver_idx: Index of the receiver to process
        gpu: Whether to use GPU acceleration

    Returns:
        Dictionary containing segmentation results and beamforming outputs
    """
    args = {
        "zarr_fn": zarr_fn,
        "receiver": receiver_idx,
        "session_idx": session_idx,
        "steering_vectors": steering_vectors_for_receiver,
        "gpu": gpu,
        **DEFAULT_SEGMENT_ARGS,
    }
    return segment_session(**args)


def mean_phase_from_simple_segmentation(segmentation):
    mean_phases = []
    for result in segmentation:
        means = []
        weights = []
        for x in result["simple_segmentation"]:
            if x["type"] == "signal":
                means.append(x["mean"])
                weights.append(
                    (x["end_idx"] - x["start_idx"])
                    * x["abs_signal_median"]
                    / (x["stddev"] + 1e-6)  # weight by signal strength and region
                )
        if len(means) == 0:
            mean_phases.append(torch.nan)  # No signal detected
        else:
            means = np.array(means)
            weights = np.array(weights)
            # Calculate weighted circular mean of phase differences
            mean_phases.append(mean_phase_mean(angles=means, weights=weights))
    return np.hstack(mean_phases)


# MultiProcess segmentation and pre-processing
# This contains the heart of pre-processing
# The input here is a zarr file containing the raw radio signal
# The output is another zarr file (yarr) containing the segmentation
# and other pre-processed features. This yarr file can then be loaded
# during training, avoiding touching any files containing the very large
# raw radio features
def mp_segment_zarr(
    zarr_fn,
    results_fn,
    steering_vectors_for_all_receivers,
    precompute_to_idx=-1,
    gpu=False,
    n_parallel=20,
    skip_beamformer=False,
    temp_file=False,
    skip_detrend=False,
):
    """
    Multiprocessing function to segment and precompute features for all sessions.

    This function is the core of the preprocessing pipeline and performs:
    1. Signal segmentation to identify segments containing RF signal
    2. Beamforming across multiple directions using steering vectors
    3. Feature extraction for model training

    Args:
        zarr_fn: Path to the zarr file containing raw signal data
        results_fn: Path to save the segmentation results (PKL file)
        steering_vectors_for_all_receivers: Precomputed steering vectors for all receivers
        precompute_to_idx: Index until which to precompute (-1 for all)
        gpu: Whether to use GPU acceleration for beamforming
        n_parallel: Number of parallel processes to use
        skip_beamformer: Whether to skip beamforming computation (saves time if not needed)

    Returns:
        Number of sessions processed
    """
    # Open the zarr file containing raw signal data
    z = zarr_open_from_lmdb_store(zarr_fn)
    yarr_fn = results_fn.replace(".pkl", ".yarr")
    already_computed = 0

    # Initialize empty segmentation or load existing one
    previous_simple_segmentation = {
        f"r{r_idx}": [] for r_idx in range(len(z["receivers"]))
    }
    precomputed_zarr = None

    # Check if results already exist and load them
    if os.path.exists(results_fn) and os.path.exists(yarr_fn):
        # this might be really slow, like O(n^2)
        previous_simple_segmentation = pickle.load(open(results_fn, "rb"))[
            "segmentation_by_receiver"
        ]
        precomputed_zarr = zarr_open_from_lmdb_store(
            yarr_fn, mode="rw", map_size=2**32, lock=False
        )

        # Determine how many sessions have already been computed
        # by checking which entries have non-zero values
        already_computed = min(
            [
                (
                    np.sum(precomputed_zarr[f"r{r_idx}/all_windows_stats"], axis=(1, 2))
                    > 0
                ).sum()
                for r_idx in range(len(z["receivers"]))
            ]
        )

    # Determine how many sessions to process
    if precompute_to_idx < 0:
        # Process all available sessions
        precompute_to_idx = min(
            [
                (z[f"receivers/r{ridx}/system_timestamp"][:] > 0).sum()
                for ridx in range(len(z["receivers"]))
            ]
        )
    else:
        # Add 1 to make the range inclusive
        precompute_to_idx += 1

    # Skip processing if everything is already computed
    if precompute_to_idx <= already_computed:
        return already_computed

    # Currently only supports exactly 2 receivers
    assert len(z["receivers"]) == 2

    # Get total number of sessions
    n_sessions = z.receivers["r0"].system_timestamp.shape[0]

    # Process each receiver in sequence
    results_by_receiver = {}
    for r_idx in [0, 1]:
        r_name = f"r{r_idx}"

        # Create a list of processing tasks - one for each session
        inputs = [
            {
                "zarr_fn": zarr_fn,
                "receiver": r_name,
                "session_idx": idx,
                "steering_vectors": steering_vectors_for_all_receivers[r_idx],
                "gpu": gpu,
                "skip_beamformer": skip_beamformer,
                "skip_detrend": skip_detrend,
                **DEFAULT_SEGMENT_ARGS,
            }
            for idx in range(already_computed, precompute_to_idx)
        ]

        # Run segmentation in parallel if requested
        if n_parallel > 0:
            with Pool(min(cpu_count(), n_parallel)) as pool:
                results_by_receiver[r_name] = list(
                    tqdm.tqdm(
                        pool.imap(segment_session_from_zarr_star, inputs),
                        desc=f"Segmenting {r_name}",
                        total=len(inputs),
                    )
                )
        else:
            # Run sequentially if n_parallel is 0
            monitor_fn = tqdm.tqdm
            if temp_file:
                monitor_fn = lambda x, desc, total: x
            results_by_receiver[r_name] = list(
                monitor_fn(
                    map(segment_session_from_zarr_star, inputs),
                    desc=f"Segmenting {r_name}",
                    total=len(inputs),
                )
            )

    # Prepare to save results
    segmentation_zarr_fn = results_fn.replace(".pkl", ".yarr")

    # Determine shapes for output arrays based on first result
    all_windows_stats_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "all_windows_stats"
    ].shape[1:]
    windowed_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape[1:]
    weighted_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape[2:]
    downsampled_segmentation_mask_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "downsampled_segmentation_mask"
    ].shape[1:]

    # Create new zarr dataset if one doesn't exist
    if precomputed_zarr is None:
        os.makedirs(os.path.dirname(segmentation_zarr_fn), exist_ok=True)
        precomputed_zarr = new_yarr_dataset(
            filename=segmentation_zarr_fn,
            n_receivers=2,
            all_windows_stats_shape=all_windows_stats_shape,
            windowed_beamformer_shape=windowed_beamformer_shape,
            weighted_beamformer_shape=weighted_beamformer_shape,
            downsampled_segmentation_mask_shape=downsampled_segmentation_mask_shape,
            mean_phase_shape=(n_sessions,),
        )

    # Process and save results for each receiver
    for r_idx in [0, 1]:
        # Store all_windows_stats: contains mean, stddev, and median signal strength for each window
        precomputed_zarr[f"r{r_idx}/all_windows_stats"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["all_windows_stats"] for x in results_by_receiver[f"r{r_idx}"]]
        )

        # Store windowed_beamformer: beamforming output for each window and direction
        precomputed_zarr[f"r{r_idx}/windowed_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["windowed_beamformer"] for x in results_by_receiver[f"r{r_idx}"]]
        )

        # Store weighted_beamformer: session-level beamforming weighted by signal quality
        precomputed_zarr[f"r{r_idx}/weighted_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["weighted_beamformer"] for x in results_by_receiver[f"r{r_idx}"]]
        )

        # Store weighted_windows_stats: statistics weighted by signal quality
        precomputed_zarr[f"r{r_idx}/weighted_windows_stats"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["weighted_windows_stats"] for x in results_by_receiver[f"r{r_idx}"]]
        )

        # Store downsampled_segmentation_mask: binary mask of signal vs noise windows
        precomputed_zarr[f"r{r_idx}/downsampled_segmentation_mask"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [
                x["downsampled_segmentation_mask"]
                for x in results_by_receiver[f"r{r_idx}"]
            ]
        )

        # Calculate mean phase across all signal windows, weighted by:
        # - window length (longer segments are more reliable)
        # - signal strength (stronger signals have better SNR)
        # - inverse stddev (less variance means more reliable phase)

        # Store the mean phase for each session
        precomputed_zarr[f"r{r_idx}/mean_phase"][already_computed:precompute_to_idx] = (
            mean_phase_from_simple_segmentation(results_by_receiver[f"r{r_idx}"])
        )

    # Build the final segmentation dictionary by combining:
    # 1. Previously computed sessions (if any)
    # 2. Newly processed sessions
    # 3. Empty placeholders for unprocessed sessions (if any)
    simple_segmentation = {}
    for r_idx in [0, 1]:
        simple_segmentation[f"r{r_idx}"] = (
            previous_simple_segmentation[f"r{r_idx}"][:already_computed]
            + [
                {"simple_segmentation": x["simple_segmentation"]}
                for x in results_by_receiver[f"r{r_idx}"]
            ]
            + [
                {"simple_segmentation": []}
                for _ in range(n_sessions - precompute_to_idx)
            ]
        )

    # Clean up and close resources
    precomputed_zarr.store.close()
    precomputed_zarr = None

    # Optimize zarr storage by removing unnecessary parts
    if not temp_file:
        zarr_shrink(segmentation_zarr_fn)

    # Save segmentation metadata to pickle file
    pickle.dump(
        {
            "version": SEGMENTATION_VERSION,
            "segmentation_by_receiver": simple_segmentation,
        },
        open(results_fn, "wb"),
    )

    return precompute_to_idx


# helper function to expand a dictionary in keywords args
def segment_session_from_zarr_star(arg_dict):
    return segment_session_from_zarr(**arg_dict)


# take a single zarr, receiver and session_idx and segment it
def segment_session_from_zarr(
    zarr_fn,
    receiver,
    session_idx,
    **kwrgs,
):
    with zarr_open_from_lmdb_store_cm(zarr_fn, mode="r", readahead=True) as z:
        # Load raw signal from the specified session and receiver
        # Signal matrix shape is (2, N) where 2 represents the two antenna elements
        # and N is the number of IQ samples in the session
        v = z.receivers[receiver].signal_matrix[session_idx][:].astype(np.complex64)
        return segment_session(v, **kwrgs)


def segment_session(
    v,
    gpu=True,
    skip_beamformer=False,
    skip_detrend=False,
    skip_segmentation=False,
    **kwrgs,
):
    """
    Process a single session's radio signal data to identify signal segments and calculate beamforming outputs.

    This function is at the core of the signal processing pipeline and performs several critical operations:

    1. Load and detrend the raw signal to remove DC offsets and linear trends
    2. Segment the signal to identify regions containing the target RF signal versus noise
    3. Calculate phase differences between antenna elements
    4. Apply beamforming across multiple angles to determine signal direction
    5. Extract statistical features for further processing

    Args:
        gpu: Whether to use GPU for beamforming calculations (faster)
        skip_beamformer: If True, skip the beamforming calculation (saves time if not needed)
        **kwrgs: Additional arguments passed to the simple_segment function

    Returns:
        Dictionary containing segmentation results and beamforming outputs
    """
    # Load raw signal from the specified session and receiver
    # Signal matrix shape is (2, N) where 2 represents the two antenna elements
    # and N is the number of IQ samples in the session
    # v = z.receivers[receiver].signal_matrix[session_idx][:].astype(np.complex64)

    # Detrend the signal to remove DC offsets and linear trends
    # The detrend_np function:
    # 1. Divides the signal into windows (1024 samples by default)
    # 2. Fits a linear trend to both the real and imaginary parts of each antenna's signal
    # 3. Subtracts the trend to center the signal around zero
    # 4. This improves phase estimation by removing hardware biases and drift
    if not skip_detrend:
        v = detrend_np(v)

    segmentation_results = {}

    # Perform beamforming if not skipped
    # Beamforming is the process of combining signals from multiple antennas
    # to enhance reception from a specific direction while suppressing others

    # BEAMFORMING EXPLANATION FOR 2-ELEMENT ARRAYS
    #
    # Beamforming is a signal processing technique used to direct or receive signals
    # in a specific direction by combining signals from multiple antenna elements.
    #
    # For a 2-element antenna array (as used in this codebase), here's how it works:
    #
    # ASCII Diagram of a 2-element array with arriving signals:
    #
    #    Signal from           Signal from
    #     θ = -45°              θ = +30°
    #       \                     /
    #        \                   /
    #         \                 /
    #          \               /
    #           v             v
    #          ┌─┐           ┌─┐
    #          │ │           │ │
    #          └─┘           └─┘
    #        Element 0     Element 1
    #          <------ d ------>
    #
    # PHYSICAL PRINCIPLES:
    #
    # 1. Path Length Difference:
    #    When a signal arrives at angle θ, it reaches the two elements at different times.
    #    - Path difference = d·sin(θ), where d is the spacing between elements
    #    - This creates a phase difference between signals at each antenna
    #
    # 2. Phase Difference:
    #    - Phase difference (φ) = 2π·d·sin(θ)/λ, where λ is the wavelength
    #    - For our 2-element array, we measure this phase difference directly
    #    - This is a key equation: it maps angle θ to phase difference φ
    #
    # 3. Direction Finding:
    #    - Forward problem: Given angle θ, calculate expected phase φ
    #    - Inverse problem: Given measured phase φ, estimate arrival angle θ
    #    - For a 2-element array: θ = arcsin(φ·λ/(2π·d))
    #
    # MATHEMATICAL FORMULATION:
    #
    # 1. Signal Representation:
    #    - Let x = [x₀, x₁]ᵀ be the complex signals received at the two antenna elements
    #
    # 2. Steering Vectors:
    #    - For each potential arrival angle θ, we compute a steering vector a(θ):
    #    - a(θ) = [1, e^(-j·2π·d·sin(θ)/λ)]ᵀ
    #    - Where: d = antenna spacing, λ = wavelength
    #
    # 3. Beamforming Operation:
    #    - The beamformer output for angle θ is: y(θ) = a(θ)ᴴ · x
    #    - In expanded form: y(θ) = x₀ + x₁·e^(j·2π·d·sin(θ)/λ)
    #
    # 4. Power at angle θ:
    #    - The power at each angle is: P(θ) = |y(θ)|²
    #
    # 5. Finding Direction of Arrival:
    #    - The angle with maximum power is the estimated signal direction:
    #    - θ̂ = argmax(P(θ))
    #
    # For a 2-element array, the phase difference φ between elements relates to angle θ:
    #    φ = 2π·d·sin(θ)/λ
    #
    # This is the key relationship that allows us to estimate signal direction from
    # the measured phase difference between antenna elements.
    #
    # In the code below:
    # 1. We use precomputed steering vectors for many possible angles
    # 2. Apply these steering vectors to the signal matrix
    # 3. Calculate the beamformer power output for each potential angle
    # 4. The angle with maximum power indicates the likely signal direction

    nthetas = kwrgs["steering_vectors"].shape[0]  # Number of angle bins for beamforming
    if not skip_beamformer:
        if gpu:
            # GPU version of beamforming (much faster for large arrays)
            # The beamformer_given_steering_nomean_cp function:
            # 1. Applies the precomputed steering vectors to the signal matrix
            # 2. For each angle θ in the steering vectors, calculates the array response
            # 3. Returns the absolute value (magnitude) of the beamformed output
            # 4. This creates a "spatial spectrum" showing signal power vs angle
            segmentation_results["windowed_beamformer"] = cp.asnumpy(
                beamformer_given_steering_nomean_cp(
                    steering_vectors=cp.asarray(kwrgs["steering_vectors"]),
                    signal_matrix=cp.asarray(v.astype(np.complex64)),
                )
                .reshape(nthetas, -1, kwrgs["window_size"])
                .mean(axis=2)
                .T
            )
        else:
            # CPU version of beamforming (same algorithm but slower)
            segmentation_results["windowed_beamformer"] = (
                beamformer_given_steering_nomean(
                    steering_vectors=kwrgs["steering_vectors"],
                    signal_matrix=v.astype(np.complex64),
                )
                .reshape(nthetas, -1, kwrgs["window_size"])
                .mean(axis=2)
                .T
            )

        # Convert to float16 to save memory
        segmentation_results["windowed_beamformer"] = segmentation_results[
            "windowed_beamformer"
        ].astype(np.float16)

    else:
        # If skipping beamforming, create empty placeholders
        windowed_beamformer = np.zeros(
            (v.shape[1] // kwrgs["window_size"], nthetas), dtype=np.float16
        )
        windowed_beamformer.fill(np.nan)
        segmentation_results["windowed_beamformer"] = windowed_beamformer

    if not skip_segmentation:
        # Simple segmentation identifies regions of the signal that likely contain useful data
        # The simple_segment function:
        # 1. Calculates phase difference between the two antenna elements
        # 2. Divides the signal into windows (typically 2048 samples) with configurable stride
        # 3. For each window, calculates trimmed circular mean and standard deviation of phase
        #    and median of signal amplitude
        # 4. Identifies windows as "signal" vs "noise" based on phase stability and amplitude
        # 5. Combines adjacent windows with similar phase characteristics
        # 6. Returns a list of segment information and signal statistics
        segmentation_results.update(simple_segment(v, **kwrgs))

        # Transpose the window statistics for easier processing
        # all_windows_stats shape is (3, N_windows) where:
        # - Row 0: Trimmed circular mean of phase difference
        # - Row 1: Trimmed standard deviation of phase difference
        # - Row 2: Median absolute signal amplitude
        segmentation_results["all_windows_stats"] = (
            segmentation_results["all_windows_stats"].astype(np.float16).T
        )

        # Calculate a weighted beamformer output for the entire session
        # by combining window-level beamformer outputs, using the segmentation mask
        # as weights (so only signal windows contribute)
        weighted_beamformer = (
            segmentation_results["windowed_beamformer"].astype(np.float32)
            * segmentation_results["downsampled_segmentation_mask"][:, None]
        ).sum(axis=0) / (
            segmentation_results["downsampled_segmentation_mask"].sum() + 0.001
        )
        # Store the session-level weighted beamformer
        segmentation_results["weighted_beamformer"] = weighted_beamformer
        # Calculate session-level statistics from the identified signal windows
        if segmentation_results["downsampled_segmentation_mask"].sum() > 0:
            # Calculate trimmed mean of phase differences from signal windows only
            # Trimming removes extreme values to make the mean more robust
            mean_phase = trim_mean(
                reduce_theta_to_positive_y(
                    segmentation_results["all_windows_stats"][0]
                )[segmentation_results["downsampled_segmentation_mask"]].astype(
                    np.float32
                ),
                0.1,  # Trim 10% from both ends
            )

            # Calculate trimmed mean of standard deviation and signal amplitude
            stddev_and_abs_signal = trim_mean(
                segmentation_results["all_windows_stats"][1:][
                    :, segmentation_results["downsampled_segmentation_mask"]
                ].astype(np.float32),
                0.1,
                axis=1,
            )

            # Store the session-level statistics
            segmentation_results["weighted_windows_stats"] = np.array(
                [mean_phase, stddev_and_abs_signal[0], stddev_and_abs_signal[1]],
                dtype=np.float32,
            )
        else:
            # If no signal windows were identified, use placeholder values
            segmentation_results["weighted_windows_stats"] = np.array([-1, -1, -1])
    else:
        segmentation_results['all_windows_stats']=get_all_windows_stats(v,**kwrgs)[1]

        # Transpose the window statistics for easier processing
        # all_windows_stats shape is (3, N_windows) where:
        # - Row 0: Trimmed circular mean of phase difference
        # - Row 1: Trimmed standard deviation of phase difference
        # - Row 2: Median absolute signal amplitude
        segmentation_results["all_windows_stats"] = (
            segmentation_results["all_windows_stats"].astype(np.float16).T
        )

    # add a singleton in front of each of these
    segmentation_results = {
        k: v[None] if isinstance(v, np.ndarray) else v
        for k, v in segmentation_results.items()
    }

    return segmentation_results

def get_all_windows_stats(
    v,
    window_size,
    stride,
    trim,
    mean_diff_threshold,
    max_stddev_threshold,
    drop_less_than_size,
    min_abs_signal,
    steering_vectors=None,  # not used but passed in
):
    # Verify the signal matrix has the expected shape (2 antennas)
    assert v.ndim == 2 and v.shape[0] == 2

    # Calculate phase differences between the two antenna elements
    # This gives us the phase shift that can be used to determine angle of arrival
    pd = get_phase_diff(v).astype(np.float32)

    # Calculate statistics for each window:
    # - Trimmed circular mean of phase differences
    # - Trimmed standard deviation of phase differences
    # - Median absolute signal amplitude
    step_idxs, step_stats = windowed_trimmed_circular_mean_and_stddev(
        v, pd, window_size=window_size, stride=stride, trim=trim
    )
    return step_idxs, step_stats

def simple_segment(
    v,
    window_size,
    stride,
    trim,
    mean_diff_threshold,
    max_stddev_threshold,
    drop_less_than_size,
    min_abs_signal,
    steering_vectors=None,  # not used but passed in
):
    """
    Segment a radio signal into regions containing valid signal vs. noise.

    This is a critical processing step that identifies meaningful signals
    within potentially noisy radio data. It works by analyzing phase differences
    between antenna elements to find stable signal regions.

    The algorithm:
    1. Calculates phase differences between antenna pairs
    2. Divides the signal into overlapping windows
    3. Computes statistical properties of each window
    4. Identifies windows with stable phase (signal) vs. unstable phase (noise)
    5. Combines adjacent windows with similar phase characteristics
    6. Applies filtering to remove small segments or unreliable detections

    Args:
        v: Complex signal matrix of shape (2, N) - two antenna elements
        window_size: Size of each window in samples
        stride: Step size between windows
        trim: Percentage of extreme values to trim when calculating circular statistics
        mean_diff_threshold: Phase mean difference threshold for signal detection
        max_stddev_threshold: Maximum allowable phase standard deviation for signal windows
        drop_less_than_size: Minimum size (in samples) for a valid signal segment
        min_abs_signal: Minimum absolute signal amplitude to consider as valid signal
        steering_vectors: Not used in this function, but passed for API compatibility

    Returns:
        Dictionary containing segmentation results:
        - simple_segmentation: List of identified signal/noise regions
        - downsampled_segmentation_mask: Boolean mask indicating signal windows
        - all_windows_stats: Statistics for each window (phase mean, stddev, amplitude)
    """
    # Verify the signal matrix has the expected shape (2 antennas)
    assert v.ndim == 2 and v.shape[0] == 2

    # Calculate phase differences between the two antenna elements
    # This gives us the phase shift that can be used to determine angle of arrival
    pd = get_phase_diff(v).astype(np.float32)

    # Create empty list to hold window information
    candidate_windows = []

    # Calculate statistics for each window:
    # - Trimmed circular mean of phase differences
    # - Trimmed standard deviation of phase differences
    # - Median absolute signal amplitude
    # window_idxs_and_stats = windowed_trimmed_circular_mean_and_stddev(
    #     v, pd, window_size=window_size, stride=stride, trim=trim
    # )
    window_idxs_and_stats = get_all_windows_stats(v=v,window_size=window_size,stride=stride,trim=trim)
    # window_idxs_and_stats has two components:
    # [0] = list of window indices (start_idx, end_idx)
    # [1] = array of statistics (trimmed_mean, trimmed_stddev, abs_signal_median)

    # Create structured information for each window
    original_windows = [
        {
            "start_idx": idx[0],  # Start index of the window
            "end_idx": idx[1],  # End index of the window
            "mean": stats[0],  # Circular mean of phase difference
            "stddev": stats[1],  # Standard deviation of phase difference
            "abs_signal_median": stats[2],  # Median absolute signal amplitude
        }
        for idx, stats in zip(window_idxs_and_stats[0], window_idxs_and_stats[1])
    ]

    # Combine windows with similar phase characteristics
    # Windows are classified as "signal" if they have:
    # - Low phase standard deviation (stable phase) OR
    # - High signal amplitude (strong signal)
    # Adjacent signal windows with similar phase mean are merged
    candidate_windows = combine_windows(
        original_windows, max_stddev_threshold, min_abs_signal
    )

    # Filter out short segments that might be false positives
    # Small windows are more likely to be noise or transient effects
    candidate_windows = drop_windows_smaller_than(
        candidate_windows, drop_less_than_size
    )

    # Apply additional filtering to retain only signal segments surrounded by noise
    # This helps eliminate false positives that might occur in transitions
    candidate_windows = keep_signal_surrounded_by_noise(candidate_windows)

    # Recalculate statistics for the final windows based on the original signal
    # This gives more accurate statistics for the identified segments
    simple_segmentation = recompute_stats_for_windows(candidate_windows, v, pd, trim)

    # Create a binary mask of signal vs. noise windows at the window level
    # This is used for weighted averaging and visualization
    downsampled_segmentation_mask = compute_downsampled_segmentation_mask(
        simple_segmentation,
        window_size=window_size,
        n_windows=window_idxs_and_stats[1].shape[0],
    )

    # Return all the computed results
    return {
        "simple_segmentation": simple_segmentation,  # Detailed segment info
        "downsampled_segmentation_mask": downsampled_segmentation_mask,  # Binary mask
        "all_windows_stats": window_idxs_and_stats[1],  # Window-level statistics
    }


def drop_noise_windows(windows):
    """
    Remove all windows labeled as "noise" from the list of windows.

    Args:
        windows: List of window dictionaries with "type" field

    Returns:
        List containing only windows of type "signal"
    """
    valid_windows = []
    for _, window in enumerate(windows):
        if window["type"] == "signal":
            valid_windows.append(window)
    return valid_windows


def keep_signal_surrounded_by_noise(windows):
    """
    Filter signal windows to keep only those surrounded by noise windows.

    This is a quality control step that helps reject false positives by requiring
    that true signals are isolated (not adjacent to other signals). Isolated
    signal windows surrounded by noise are more likely to be genuine signals
    rather than processing artifacts.

    Args:
        windows: List of window dictionaries with "type" field

    Returns:
        Filtered list of windows where signal windows are surrounded by noise
    """
    valid_windows = []
    for window_idx, window in enumerate(windows):
        if window["type"] == "signal":
            # check if one before was signal
            if window_idx > 0 and windows[window_idx - 1]["type"] == "signal":
                continue
            # check if one after was signal
            if (
                window_idx + 1 < len(windows)
                and windows[window_idx + 1]["type"] == "signal"
            ):
                continue
            valid_windows.append(window)
    return valid_windows


def drop_windows_smaller_than(windows, drop_less_than_size):
    """
    Remove windows that are smaller than the specified size.

    Small segments are more likely to be noise spikes or processing artifacts
    rather than genuine signals. This function filters out windows that are
    too small to be considered reliable.

    Args:
        windows: List of window dictionaries with "start_idx" and "end_idx" fields
        drop_less_than_size: Minimum window size in samples

    Returns:
        Filtered list of windows that meet the minimum size requirement
    """
    return [w for w in windows if (w["end_idx"] - w["start_idx"]) > drop_less_than_size]


def combine_windows(windows, max_stddev_threshold, min_abs_signal):
    """
    Classify windows as signal or noise and combine adjacent similar signal windows.

    This function:
    1. Classifies each window as "signal" or "noise" based on phase stability and signal strength
    2. Merges adjacent signal windows that have similar phase characteristics
    3. Merges adjacent noise windows to simplify the segmentation

    Windows are classified as "signal" if they have either:
    - Low phase standard deviation (stable phase) OR
    - High signal amplitude (strong signal)

    Args:
        windows: List of window dictionaries with statistical information
        max_stddev_threshold: Maximum acceptable phase standard deviation for signal windows
        min_abs_signal: Minimum absolute signal strength for a window to be considered signal

    Returns:
        List of combined windows with type classification (signal/noise)
    """
    # combine windows
    new_windows = []
    for window in windows:
        # Check if this window should be classified as a signal window
        # based on either low phase variance or high signal strength
        if (
            window["stddev"] < max_stddev_threshold
            or window["abs_signal_median"] >= min_abs_signal
        ):
            # Check if we can merge with the previous window (if it was also signal)
            # Windows are merged if they have similar phase mean and standard deviation
            if (
                len(new_windows) > 0
                and new_windows[-1]["type"] == "signal"
                and pi_norm(abs(new_windows[-1]["mean"] - window["mean"])) < 0.2
                and abs(new_windows[-1]["stddev"] - window["stddev"]) < 0.1
            ):
                # Extend the previous window to include this one
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                # Add as a new signal window
                window["type"] = "signal"
                new_windows.append(window)
        else:
            # This is a noise window
            # Check if the previous window was also noise, if so merge them
            if len(new_windows) > 0 and new_windows[-1]["type"] == "noise":
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                # Add as a new noise window
                window["type"] = "noise"
                new_windows.append(window)
    return new_windows


def recompute_stats_for_windows(windows, v, pd, trim):
    """
    Recalculate statistical properties for each window based on the original signal.

    After windows have been merged and filtered, this function recomputes the
    statistical properties (mean phase, standard deviation, signal amplitude)
    for each window using the original signal data.

    Args:
        windows: List of window dictionaries with start/end indices
        v: Original complex signal matrix
        pd: Phase differences from original signal
        trim: Percentage to trim from circular statistics

    Returns:
        Windows with updated statistical properties
    """
    for window in windows:
        # Extract the phase differences for this window
        _pd = pd[window["start_idx"] : window["end_idx"]]
        # Extract the signal for this window
        _v = v[:, window["start_idx"] : window["end_idx"]]

        # Calculate circular mean, standard deviation, and signal amplitude
        r = get_stats_for_signal(_v, _pd, trim)
        window["mean"] = r[0]
        window["stddev"] = r[1]
        window["abs_signal_median"] = r[2]
    return windows


def compute_downsampled_segmentation_mask(simple_segmentation, n_windows, window_size):
    """
    Create a binary mask indicating which windows contain signal vs. noise.

    This function converts the detailed segmentation information into a simple
    binary mask at the window level. This mask is used for:
    1. Weighting in beamforming calculations
    2. Visualization of signal vs. noise regions
    3. Simplifying further processing

    Args:
        simple_segmentation: List of window dictionaries with start/end indices
        n_windows: Total number of windows
        window_size: Size of each window in samples

    Returns:
        Boolean array with True for windows containing signal
    """
    # Initialize an empty mask (all False/noise)
    seg_mask = np.zeros(n_windows).astype(bool)

    # Set True for windows that overlap with signal segments
    for window in simple_segmentation:
        seg_mask[
            window["start_idx"] // window_size : window["end_idx"] // window_size,
        ] = True
    return seg_mask


# BEAMFORMING EXPLANATION FOR 2-ELEMENT ARRAYS
#
# Beamforming is a signal processing technique used to direct or receive signals
# in a specific direction by combining signals from multiple antenna elements.
#
# For a 2-element antenna array, here's how it works:
#
# ASCII Diagram:
#
#    Signal from           Signal from
#     θ = -45°              θ = +30°
#       \                     /
#        \                   /
#         \                 /
#          \               /
#           v             v
#          ┌─┐           ┌─┐
#          │ │           │ │
#          └─┘           └─┘
#        Element 0     Element 1
#          <------ d ------>
#
# When a signal arrives at angle θ, it reaches Element 0 and Element 1 at different times
# due to the path length difference (d·sin(θ)), causing a phase difference between them.
#
# Linear Algebra Explanation:
#
# 1. Signal Representation:
#    - Let x = [x₀, x₁]ᵀ be the complex signals received at the two antenna elements
#
# 2. Steering Vectors:
#    - For each potential arrival angle θ, we compute a steering vector a(θ):
#    - a(θ) = [1, e^(-j·2π·d·sin(θ)/λ)]ᵀ
#    - Where: d = antenna spacing, λ = wavelength
#
# 3. Beamforming Operation:
#    - The beamformer output for angle θ is: y(θ) = a(θ)ᴴ · x
#    - In expanded form: y(θ) = x₀ + x₁·e^(j·2π·d·sin(θ)/λ)
#
# 4. Power at angle θ:
#    - The power at each angle is: P(θ) = |y(θ)|²
#
# 5. Finding Direction of Arrival:
#    - The angle with maximum power is the estimated signal direction:
#    - θ̂ = argmax(P(θ))
#
# For a 2-element array, the phase difference φ between elements relates to angle θ:
#    φ = 2π·d·sin(θ)/λ
#
# This is the key relationship that allows us to estimate signal direction from
# the measured phase difference between antenna elements.
#
# In the code, we:
# 1. Compute phase differences between antenna elements
# 2. Generate steering vectors for many possible angles
# 3. Compute beamformer output for each angle
# 4. Find peaks in the beamforming output to estimate signal directions


def beamform_signal_cpu(v, steering_vectors, n_windows=None):
    """
    Perform beamforming on CPU to estimate signal direction.

    This function computes the beamformer output for each possible direction
    represented by the steering vectors.

    Args:
        v: Complex signal matrix of shape (n_antennas, n_samples)
        steering_vectors: Precomputed steering vectors for different angles
        n_windows: Number of windows to process (defaults to all)

    Returns:
        Beamformer output power for each angle and window
    """
    # ... existing code ...
