import numpy as np
import torch

from spf.dataset.segmentation import simple_segment
from spf.rf import (
    IQSource,
    UCADetector,
    ULADetector,
    beamformer,
    c,
    circular_mean,
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
        mean_diff_threshold=mean_diff_threshold,
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
        mean_diff_threshold=mean_diff_threshold,
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
