import numpy as np
import torch

from spf.dataset.segmentation import simple_segment
from spf.rf import (
    IQSource,
    UCADetector,
    ULADetector,
    beamformer,
    beamformer_given_steering_nomean,
    beamformer_given_steering_nomean_fast,
    c,
    circular_diff_to_mean_single,
    circular_diff_to_mean_single_fast,
    circular_mean,
    circular_mean_simple_fast,
    fast_median_abs,
    fast_percentile,
    rotation_matrix,
    torch_circular_mean,
)
from spf.utils import random_signal_matrix


def test_beamformer():
    # carrier_frequency=2.4e9
    carrier_frequency = 100e3
    wavelength = c / carrier_frequency
    sampling_frequency = 10e6

    import matplotlib.pyplot as plt

    """
           X (source_pos)
           |
           |
           |
           |
    ----------------

    Lets rotate on the source around to the left(counter clockwise) by theta radians
    """

    source_pos = np.array([[0, 10000]])
    rotations = [
        -np.pi / 2,
        -np.pi,
        -np.pi / 4,
        0,
        np.pi / 2,
        np.pi / 2,
        np.pi,
    ] + list(np.random.uniform(-np.pi, np.pi, 10))
    spacings = [
        wavelength / 4
    ]  # ,wavelength/2,wavelength,wavelength/3]+list(np.random.uniform(0,1,10))

    nelements = [2, 3, 4]  # ,16]
    np.random.seed(1442)

    for Detector in [ULADetector, UCADetector]:
        for nelement in nelements:
            for spacing in spacings:
                d = Detector(sampling_frequency, nelement, spacing, sigma=0.0)

                for rot_theta in rotations:
                    rot_mat = rotation_matrix(rot_theta)
                    _source_pos = (
                        rot_mat @ source_pos.T
                    ).T  # rotate right by rot_theta

                    d.rm_sources()
                    d.add_source(
                        IQSource(_source_pos, carrier_frequency, 100e3)  # x, y position
                    )

                    signal_matrix = d.get_signal_matrix(
                        start_time=100, duration=3 / d.sampling_frequency
                    )

                    thetas_at_t, beam_former_outputs_at_t, _ = beamformer(
                        d.all_receiver_pos(),
                        signal_matrix,
                        carrier_frequency,
                        spacing=1024 + 1,
                    )

                    closest_angle_answer = np.argmin(np.abs(thetas_at_t - rot_theta))
                    potential_error = np.abs(
                        beam_former_outputs_at_t[:-1] - beam_former_outputs_at_t[1:]
                    ).max()

                    assert (
                        beam_former_outputs_at_t.max() - potential_error
                    ) <= beam_former_outputs_at_t[closest_angle_answer]

                    # plt.figure()
                    # plt.plot(thetas_at_t,beam_former_outputs_at_t)
                    # plt.axvline(x=-rot_theta)
                    # plt.axhline(y=beam_former_outputs_at_t.max()-potential_error)
                    # plt.show()


def test_simple_segment():
    rng = np.random.default_rng(12345)
    signal_matrix = random_signal_matrix(1200, rng=rng).reshape(2, 600)

    ground_truth_windows = [
        {"start_idx": 0, "end_idx": 100, "mean": 0.5},
        {"start_idx": 300, "end_idx": 400, "mean": 1.3},
        {"start_idx": 500, "end_idx": 600, "mean": 1.25},
    ]

    for window in ground_truth_windows:
        signal_matrix[0, window["start_idx"] : window["end_idx"]] *= 10
        signal_matrix[1, window["start_idx"] : window["end_idx"]] = signal_matrix[
            0, window["start_idx"] : window["end_idx"]
        ] * np.exp(-1j * window["mean"])

    mean_diff_threshold = 0.05
    segmented_windows = simple_segment(
        signal_matrix,
        window_size=100,
        stride=50,
        trim=10,
        # mean_diff_threshold=mean_diff_threshold,
        max_stddev_threshold=0.1,
        drop_less_than_size=0,
        min_abs_signal=10,
    )["simple_segmentation"]

    assert segmented_windows[0]["start_idx"] == ground_truth_windows[0]["start_idx"]
    assert segmented_windows[0]["end_idx"] == ground_truth_windows[0]["end_idx"]
    assert np.isclose(
        segmented_windows[0]["mean"],
        ground_truth_windows[0]["mean"],
        mean_diff_threshold,
    )

    assert segmented_windows[1]["start_idx"] == ground_truth_windows[1]["start_idx"]
    # assert segmented_windows[1]["end_idx"] == ground_truth_windows[2]["end_idx"]
    assert np.isclose(
        segmented_windows[1]["mean"],
        ground_truth_windows[1]["mean"],
        mean_diff_threshold,
    )


def test_simple_segment_separate():
    rng = np.random.default_rng(12345)
    signal_matrix = random_signal_matrix(800, rng=rng).reshape(2, 400)

    ground_truth_windows = [
        {"start_idx": 0, "end_idx": 100, "mean": 0.5},
        {"start_idx": 200, "end_idx": 300, "mean": 1.3},
        {"start_idx": 300, "end_idx": 400, "mean": 1.1},
    ]

    for window in ground_truth_windows:
        signal_matrix[0, window["start_idx"] : window["end_idx"]] *= 10
        signal_matrix[1, window["start_idx"] : window["end_idx"]] = signal_matrix[
            0, window["start_idx"] : window["end_idx"]
        ] * np.exp(-1j * window["mean"])

    mean_diff_threshold = 0.05
    segmented_windows = simple_segment(
        signal_matrix,
        window_size=100,
        stride=50,
        trim=10,
        # mean_diff_threshold=mean_diff_threshold,
        max_stddev_threshold=0.1,
        drop_less_than_size=0,
        min_abs_signal=10,
    )["simple_segmentation"]

    assert segmented_windows[0]["start_idx"] == ground_truth_windows[0]["start_idx"]
    assert segmented_windows[0]["end_idx"] == ground_truth_windows[0]["end_idx"]
    assert np.isclose(
        segmented_windows[0]["mean"],
        ground_truth_windows[0]["mean"],
        mean_diff_threshold,
    )

    assert segmented_windows[1]["start_idx"] == ground_truth_windows[1]["start_idx"]
    assert segmented_windows[1]["end_idx"] == ground_truth_windows[1]["end_idx"]
    assert np.isclose(
        segmented_windows[1]["mean"],
        ground_truth_windows[1]["mean"],
        mean_diff_threshold,
    )

    assert segmented_windows[2]["start_idx"] == ground_truth_windows[2]["start_idx"]
    assert segmented_windows[2]["end_idx"] == ground_truth_windows[2]["end_idx"]
    assert np.isclose(
        segmented_windows[2]["mean"],
        ground_truth_windows[2]["mean"],
        mean_diff_threshold,
    )


def test_circular_mean():
    gt_mean = np.array([0, np.pi / 2]).reshape(-1, 1)

    gt_mean = np.random.uniform(low=-np.pi, high=np.pi, size=(128, 1))

    for n in [1000]:
        angles = np.random.randn(gt_mean.shape[0], n) * 0.2 + gt_mean
        m, _m = circular_mean(angles, 20)
        assert np.isclose(_m, gt_mean[:, 0], atol=1e-1).all()

        _angles = torch.from_numpy(angles)
        m, _m = torch_circular_mean(_angles, 20)
        assert torch.isclose(
            _m.double(), torch.from_numpy(gt_mean[:, 0]), atol=1e-1
        ).all()

        # try with and without weights
        add_noise_angles = np.hstack(
            [angles, np.random.randn(gt_mean.shape[0], n * 10) + 0.5]
        )
        # add_noise_angles = 10*np.random.randn(2, n*40)
        weights = np.zeros(add_noise_angles.shape)
        weights[:, :n] = 1

        m, _m = circular_mean(add_noise_angles, 20, weights=weights)
        assert np.isclose(_m, gt_mean[:, 0], atol=1e-1).all()

        m, _m = torch_circular_mean(
            torch.from_numpy(add_noise_angles), 20, weights=weights
        )
        assert torch.isclose(
            _m.double(), torch.from_numpy(gt_mean[:, 0]), atol=1e-1
        ).all()


def test_beamformer_functions():
    np.random.seed(42)  # for reproducibility

    # Generate random inputs
    n_thetas = 65
    n_antennas = 2
    n_samples = 1000
    n_iter=10

    for _ in range(n_iter):
        steering_vectors = np.random.randn(n_thetas, n_antennas) + 1j * np.random.randn(n_thetas, n_antennas)
        steering_vectors = steering_vectors.astype(np.complex64)

        signal_matrix = np.random.randn(n_antennas, n_samples) + 1j * np.random.randn(n_antennas, n_samples)
        signal_matrix = signal_matrix.astype(np.complex64)

        # Run both versions
        slow_output = beamformer_given_steering_nomean(steering_vectors, signal_matrix)
        fast_output = beamformer_given_steering_nomean_fast(steering_vectors, signal_matrix)

        assert np.isclose(slow_output,fast_output,rtol=1e-4).all()



def test_fast_percentile():
    np.random.seed(42)
    
    # Test cases
    arrays = [
        np.array([1, 2, 3, 4, 5]),
        np.array([10, 20, 30]),
        np.random.randn(100),
        np.random.uniform(-10, 10, size=1000),
        np.array([])  # Empty array case
    ]

    percentiles = [0.0, 25.0, 50.0, 75.0, 100.0]

    for arr_idx, arr in enumerate(arrays):
        arr_float32 = arr.astype(np.float32)

        print(f"Testing array #{arr_idx}, size {arr_float32.size}")
        for p in percentiles:
            fast_val = fast_percentile(arr_float32, p)
            numpy_val = np.percentile(arr_float32, p, interpolation='nearest')  # match "nearest" behavior

            diff = abs(fast_val - numpy_val)
            print(f"  Percentile {p:.1f}% -> fast: {fast_val:.5f}, numpy: {numpy_val:.5f}, diff: {diff:.2e}")

            # Allow small floating point rounding errors
            if arr_float32.size > 0:
                assert diff < 1e-3, f"Mismatch at array {arr_idx}, percentile {p}"
            else:
                assert fast_val == 0.0, "Empty array should return 0.0"


# --- Test function ---
def test_circular_diff_to_mean_single():
    np.random.seed(42)

    means = [0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for mean in means:
        print(f"Testing with mean = {mean:.4f}")

        angles = np.random.uniform(0, 2*np.pi, size=1000).astype(np.float32)

        output_orig = circular_diff_to_mean_single(angles, mean)
        output_opt = circular_diff_to_mean_single_fast(angles, mean)

        max_abs_error = np.max(np.abs(output_orig - output_opt))
        print(f"  Max abs error: {max_abs_error:.2e}")

        # Allow only tiny floating point differences
        assert max_abs_error < 1e-6, f"Mismatch! Max error: {max_abs_error}"

def test_fast_median_abs():
    np.random.seed(42)

    arrays = [
        np.array([], dtype=np.float32),                          # Empty
        np.array([3.0], dtype=np.float32),                       # Single element
        np.array([-2.0, 5.0], dtype=np.float32),                 # Two elements
        np.random.uniform(-10, 10, size=5).astype(np.float32),   # Odd size
        np.random.uniform(-10, 10, size=6).astype(np.float32),   # Even size
        np.random.randn(1000).astype(np.float32),                # Large array
    ]

    for idx, arr in enumerate(arrays):
        computed = fast_median_abs(arr)

        if arr.size == 0:
            # Special case for empty array
            print(f"Array #{idx}: Empty array, Computed={computed:.6f}")
            assert computed == 0.0, f"Empty array should return 0.0, got {computed}"
        else:
            expected = np.median(np.abs(arr))
            diff = abs(expected - computed)
            print(f"Array #{idx}: Expected={expected:.6f}, Computed={computed:.6f}, Diff={diff:.2e}")
            assert diff < 1e-6, f"Mismatch in array #{idx}! Diff={diff}"

    print("✅ All tests passed for fast_median_abs!")


# === Test function ===
def test_circular_mean():
    np.random.seed(42)

    test_cases = [
        np.random.uniform(0, 2*np.pi, size=(1, 5)).astype(np.float32),     # small 1×5
        np.random.uniform(0, 2*np.pi, size=(10, 64)).astype(np.float32),   # medium 10×64
        np.random.uniform(0, 2*np.pi, size=(100, 128)).astype(np.float32), # larger 100×128
    ]

    for idx, angles in enumerate(test_cases):
        cm_orig, _ = circular_mean(angles, trim=0.0)  # Your original slow
        cm_opt, _ = circular_mean_simple_fast(angles) # Your new fast

        # Correct circular difference!
        diff = np.abs((cm_orig - cm_opt + np.pi) % (2*np.pi) - np.pi)
        max_diff = np.max(diff)
        print(f"Test case {idx}: max circular diff = {max_diff:.2e}")

        assert max_diff < 1e-6, f"Mismatch in test case {idx}! Max circular diff = {max_diff}"

    print("✅ All tests passed for optimized circular_mean!")
