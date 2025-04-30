import numpy as np
from spf.dataset.segmentation import simple_segment
from spf.utils import random_signal_matrix

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

