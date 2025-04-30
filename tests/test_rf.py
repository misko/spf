import numpy as np
import torch

from spf.rf import (
    beamformer_given_steering_nomean,
    beamformer_given_steering_nomean_fast,
    circular_diff_to_mean_single,
    circular_diff_to_mean_single_fast,
    circular_mean,
    circular_mean_simple_fast,
    fast_median_abs,
    fast_percentile,
    torch_circular_mean,
)

# lots of test code from chat gpt

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

        assert max_diff < 1e-6, f"Mismatch in test case {idx}! Max circular diff = {max_diff}"

