# from numba import jit
import functools
import os
import pickle
import compress_pickle
from multiprocessing import Pool

import numpy as np
import torch
from numba import jit, njit
from tqdm import tqdm

try:
    import cupy as cp
except:
    pass
from spf.utils import zarr_open_from_lmdb_store, zarr_open_from_lmdb_store_cm

SEGMENTATION_VERSION = 2.0
# numba = False

"""

Given some guess of the source of direction we can shift the carrier frequency
phase of received samples at the N different receivers. If the guess of the
source direction is correct, the signal from the N different receivers should
interfer constructively.

"""


@functools.lru_cache(maxsize=1024)
def rf_linspace(s, e, i):
    return np.linspace(
        s, e, i, dtype=np.float64
    )  # this affects tests :'( keep higher precision


"""
Rotate by orientation
If we are left multiplying then its a right (clockwise) rotation

"""


@njit
def pi_norm(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@njit
def pi_norm_halfpi(x):
    return ((x + np.pi / 2) % (2 * np.pi / 2)) - np.pi / 2


@torch.jit.script
def torch_circular_diff_to_mean(angles: torch.Tensor, means: torch.Tensor):
    assert means.ndim == 1
    a = torch.abs(means[:, None] - angles) % (2 * torch.pi)
    b = 2 * torch.pi - a
    m, _ = torch.min(torch.vstack([a[None], b[None]]), dim=0)
    return m


# def torch_pi_norm(x, max_angle=torch.pi):
#     return ((x + max_angle) % (2 * max_angle)) - max_angle


@torch.jit.script
def torch_pi_norm_pi(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


@torch.jit.script
def torch_pi_norm(x: torch.Tensor, max_angle: float = torch.pi):
    return ((x + max_angle) % (2 * max_angle)) - max_angle


# returns circular_stddev and trimmed cricular stddev
@njit
def circular_stddev(v, u, trim=50.0):
    diff_from_mean = circular_diff_to_mean_single(angles=v, mean=u)

    diff_from_mean_squared = diff_from_mean**2

    stddev = 0
    trimmed_stddev = 0

    if diff_from_mean.shape[0] > 1:
        stddev = np.sqrt(diff_from_mean_squared.sum() / (diff_from_mean.shape[0] - 1))

    mask = diff_from_mean <= np.percentile(diff_from_mean, 100.0 - trim)
    _diff_from_mean_squared = diff_from_mean_squared[mask]

    if _diff_from_mean_squared.shape[0] > 1:
        trimmed_stddev = np.sqrt(
            _diff_from_mean_squared.sum() / (_diff_from_mean_squared.shape[0] - 1)
        )
    return stddev, trimmed_stddev


# returns circular_stddev and trimmed cricular stddev
@torch.jit.script
def torch_circular_stddev(v: torch.Tensor, u: torch.Tensor, trim: float):  # =50.0):
    diff_from_mean = torch_circular_diff_to_mean(angles=v, means=u.reshape(-1))

    diff_from_mean_squared = diff_from_mean**2

    stddev = torch.tensor(0)
    trimmed_stddev = torch.tensor(0)

    if diff_from_mean.shape[0] > 1:
        stddev = torch.sqrt(
            diff_from_mean_squared.sum() / (diff_from_mean.shape[0] - 1)
        )

    mask = diff_from_mean <= torch.quantile(diff_from_mean, 1.0 - trim / 100)
    _diff_from_mean_squared = diff_from_mean_squared[mask]

    if _diff_from_mean_squared.shape[0] > 1:
        trimmed_stddev = torch.sqrt(
            _diff_from_mean_squared.sum() / (_diff_from_mean_squared.shape[0] - 1)
        )
    return stddev, trimmed_stddev


@torch.jit.script
def torch_reduce_theta_to_positive_y(ground_truth_thetas):
    reduced_thetas = ground_truth_thetas.clone()

    # |theta|>np.pi/2 means its on the y<0
    reduced_ground_truth_thetas_mask = abs(reduced_thetas) > np.pi / 2
    reduced_ground_truth_thetas_at_mask = reduced_thetas[
        reduced_ground_truth_thetas_mask
    ]
    # reduced_thetas[reduced_ground_truth_thetas_mask] = (
    #     np.sign(reduced_ground_truth_thetas_at_mask) * np.pi
    #     - reduced_ground_truth_thetas_at_mask
    # )
    if isinstance(ground_truth_thetas, torch.Tensor):
        reduced_thetas[reduced_ground_truth_thetas_mask] = (
            torch.sign(reduced_ground_truth_thetas_at_mask) * torch.pi
            - reduced_ground_truth_thetas_at_mask
        )
    else:
        reduced_thetas[reduced_ground_truth_thetas_mask] = (
            np.sign(reduced_ground_truth_thetas_at_mask) * np.pi
            - reduced_ground_truth_thetas_at_mask
        )
    return reduced_thetas


def reduce_theta_to_positive_y(ground_truth_thetas):
    if isinstance(ground_truth_thetas, torch.Tensor):
        reduced_thetas = ground_truth_thetas.clone()
    else:
        reduced_thetas = ground_truth_thetas.copy()

    # |theta|>np.pi/2 means its on the y<0
    reduced_ground_truth_thetas_mask = abs(reduced_thetas) > np.pi / 2
    reduced_ground_truth_thetas_at_mask = reduced_thetas[
        reduced_ground_truth_thetas_mask
    ]
    # reduced_thetas[reduced_ground_truth_thetas_mask] = (
    #     np.sign(reduced_ground_truth_thetas_at_mask) * np.pi
    #     - reduced_ground_truth_thetas_at_mask
    # )
    if isinstance(ground_truth_thetas, torch.Tensor):
        reduced_thetas[reduced_ground_truth_thetas_mask] = (
            torch.sign(reduced_ground_truth_thetas_at_mask) * torch.pi
            - reduced_ground_truth_thetas_at_mask
        )
    else:
        reduced_thetas[reduced_ground_truth_thetas_mask] = (
            np.sign(reduced_ground_truth_thetas_at_mask) * np.pi
            - reduced_ground_truth_thetas_at_mask
        )
    return reduced_thetas


@njit
def circular_diff_to_mean_single(angles, mean: float):
    assert angles.ndim == 1
    a = np.abs(mean - angles) % (2 * np.pi)
    b = 2 * np.pi - a
    mask = a < b
    b[mask] = a[mask]
    return b


# @njit
def circular_diff_to_mean(angles, means):
    assert means.ndim == 1
    a = np.abs(means[:, None] - angles) % (2 * np.pi)
    b = 2 * np.pi - a
    return np.min(np.vstack([a[None], b[None]]), axis=0)


# TODO remove
def circular_mean(angles, trim, weights=None):
    assert angles.ndim == 2
    _sin_angles = np.sin(angles)
    _cos_angles = np.cos(angles)
    if weights is not None:
        _sin_angles = _sin_angles * weights
        _cos_angles = _cos_angles * weights
    cm = np.arctan2(_sin_angles.sum(axis=1), _cos_angles.sum(axis=1)) % (2 * np.pi)

    if trim == 0.0:
        r = pi_norm(cm)
        return r, r

    dists = circular_diff_to_mean(angles=angles, means=cm)

    mask = dists <= np.percentile(dists, 100.0 - trim, axis=1, keepdims=True)
    _cm = np.zeros(angles.shape[0])
    for idx in range(angles.shape[0]):
        _cm[idx] = np.arctan2(
            _sin_angles[idx][mask[idx]].sum(),
            _cos_angles[idx][mask[idx]].sum(),
        ) % (2 * np.pi)

    return pi_norm(cm), pi_norm(_cm)


@njit
def circular_mean_single(angles, trim, weights=None):
    assert angles.ndim == 1
    _sin_angles = np.sin(angles)
    _cos_angles = np.cos(angles)
    if weights is not None:
        _sin_angles = _sin_angles * weights
        _cos_angles = _cos_angles * weights
    cm = np.arctan2(_sin_angles.sum(), _cos_angles.sum()) % (2 * np.pi)

    if trim == 0.0:
        r = pi_norm(cm)
        return r, r

    dists = circular_diff_to_mean_single(angles=angles, mean=cm)

    mask = dists <= np.percentile(dists, 100.0 - trim)
    _cm = np.arctan2(_sin_angles[mask].sum(), _cos_angles[mask].sum()) % (2 * np.pi)

    return pi_norm(cm), pi_norm(_cm)


@torch.jit.script
def torch_circular_mean_notrim(angles: torch.Tensor):
    assert angles.ndim == 2
    _sin_angles = torch.sin(angles)
    _cos_angles = torch.cos(angles)
    cm = torch.arctan2(_sin_angles.sum(dim=1), _cos_angles.sum(dim=1)) % (2 * torch.pi)

    r = torch_pi_norm_pi(cm)
    return r, r


@torch.jit.script
def torch_circular_mean_noweight(angles: torch.Tensor, trim: float):
    assert angles.ndim == 2
    _sin_angles = torch.sin(angles)
    _cos_angles = torch.cos(angles)

    cm = torch.arctan2(_sin_angles.sum(dim=1), _cos_angles.sum(dim=1)) % (2 * torch.pi)

    if trim == 0.0:
        r = torch_pi_norm_pi(cm)
        return r, r

    dists = torch_circular_diff_to_mean(angles=angles, means=cm)

    mask = dists <= torch.quantile(dists, (1.0 - trim / 100), dim=1, keepdim=True)
    _cm = torch.zeros(angles.shape[0])
    for idx in torch.arange(angles.shape[0]):
        _cm[idx] = torch.arctan2(
            _sin_angles[idx][mask[idx]].sum(),
            _cos_angles[idx][mask[idx]].sum(),
        ) % (2 * torch.pi)

    return torch_pi_norm_pi(cm), torch_pi_norm_pi(_cm)


def torch_circular_mean(angles: torch.Tensor, trim: float, weights=None):
    assert angles.ndim == 2
    _sin_angles = torch.sin(angles)
    _cos_angles = torch.cos(angles)
    if weights is not None:
        _sin_angles = _sin_angles * weights
        _cos_angles = _cos_angles * weights

    cm = torch.arctan2(_sin_angles.sum(dim=1), _cos_angles.sum(dim=1)) % (2 * torch.pi)

    if trim == 0.0:
        r = torch_pi_norm_pi(cm)
        return r, r

    dists = torch_circular_diff_to_mean(angles=angles, means=cm)

    mask = dists <= torch.quantile(dists, (1.0 - trim / 100), dim=1, keepdim=True)
    _cm = torch.zeros(angles.shape[0])
    for idx in range(angles.shape[0]):
        _cm[idx] = torch.arctan2(
            _sin_angles[idx][mask[idx]].sum(),
            _cos_angles[idx][mask[idx]].sum(),
        ) % (2 * torch.pi)

    return torch_pi_norm_pi(cm), torch_pi_norm_pi(_cm)


# helper function to expand a dictionary in keywords args
def segment_session_star(arg_dict):
    return segment_session(**arg_dict)


# take a single zarr, receiver and session_idx and segment it
def segment_session(
    zarr_fn, receiver, session_idx, gpu=True, skip_beamformer=False, **kwrgs
):
    with zarr_open_from_lmdb_store_cm(zarr_fn, mode="r") as z:
        # z[f"receivers/r{receiver}/system_timestamp"][session_idx] > 0

        v = z.receivers[receiver].signal_matrix[session_idx][:].astype(np.complex64)

        segmentation_results = simple_segment(v, **kwrgs)

        segmentation_results["all_windows_stats"] = (
            segmentation_results["all_windows_stats"].astype(np.float16).T
        )

        _, windows = segmentation_results["all_windows_stats"].shape
        nthetas = kwrgs["steering_vectors"].shape[0]

        if not skip_beamformer:
            if gpu:
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
                segmentation_results["windowed_beamformer"] = (
                    beamformer_given_steering_nomean(
                        steering_vectors=kwrgs["steering_vectors"],
                        signal_matrix=v.astype(np.complex64),
                    )
                    .reshape(nthetas, -1, kwrgs["window_size"])
                    .mean(axis=2)
                    .T
                )
            segmentation_results["windowed_beamformer"] = segmentation_results[
                "windowed_beamformer"
            ].astype(np.float16)
        else:
            windowed_beamformer = np.zeros((windows, nthetas), dtype=np.float16)
            windowed_beamformer.fill(np.nan)
            segmentation_results["windowed_beamformer"] = windowed_beamformer

        return segmentation_results


@torch.jit.script
def torch_get_stats_for_signal(v: torch.Tensor, pd: torch.Tensor, trim: float):
    trimmed_cm = torch_circular_mean_noweight(pd.reshape(1, -1), trim=trim)[1][
        0
    ]  # get the value for trimmed
    trimmed_stddev = torch_circular_stddev(pd, trimmed_cm, trim=trim)[1]
    abs_signal_median = (
        torch.median(torch.abs(v).reshape(-1)) if v.numel() > 0 else torch.tensor(0)
    )
    return torch.hstack((trimmed_cm, trimmed_stddev, abs_signal_median))


@njit
def get_stats_for_signal(v, pd, trim):
    trimmed_cm = circular_mean_single(pd, trim=trim)[1]  # get the value for trimmed
    trimmed_stddev = circular_stddev(pd, trimmed_cm, trim=trim)[1]
    abs_signal_median = np.median(np.abs(v)) if v.size > 0 else 0
    return trimmed_cm, trimmed_stddev, abs_signal_median


@torch.jit.script
def torch_windowed_trimmed_circular_mean_and_stddev(
    v: torch.Tensor, pd: torch.Tensor, window_size: int, stride: int, trim: float
):
    assert (pd.shape[0] - window_size) % stride == 0
    n_steps = 1 + (pd.shape[0] - window_size) // stride

    step_idxs = torch.zeros((n_steps, 2), dtype=torch.int32)
    step_stats = torch.zeros((n_steps, 3), dtype=torch.float32)
    steps = torch.arange(n_steps)

    # start_idx, end_idx
    step_idxs[:, 0] = steps * stride
    step_idxs[:, 1] = step_idxs[:, 0] + window_size

    for step in torch.arange(n_steps):
        start_idx = step * stride
        end_idx = start_idx + window_size

        _pd = pd[start_idx:end_idx]
        _v = v[:, start_idx:end_idx]
        # trimmed_cm, trimmed_stddev, abs_signal_median
        step_stats[step] = torch_get_stats_for_signal(
            _v, _pd, trim
        )  # trimmed_cm, trimmed_stddev, abs_signal_median
    return step_idxs, step_stats


@njit
def windowed_trimmed_circular_mean_and_stddev(v, pd, window_size, stride, trim=50.0):
    assert (pd.shape[0] - window_size) % stride == 0
    n_steps = 1 + (pd.shape[0] - window_size) // stride

    step_idxs = np.zeros((n_steps, 2), dtype=np.int32)
    step_stats = np.zeros((n_steps, 3), dtype=np.float32)
    steps = np.arange(n_steps)

    # start_idx, end_idx
    step_idxs[:, 0] = steps * stride
    step_idxs[:, 1] = step_idxs[:, 0] + window_size
    for step in range(n_steps):
        start_idx, end_idx = step_idxs[step][:2]
        _pd = pd[start_idx:end_idx]
        _v = v[:, start_idx:end_idx]
        # trimmed_cm, trimmed_stddev, abs_signal_median
        step_stats[step] = get_stats_for_signal(
            _v, _pd, trim
        )  # trimmed_cm, trimmed_stddev, abs_signal_median
    return step_idxs, step_stats


def keep_signal_surrounded_by_noise(windows):
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
    return [w for w in windows if (w["end_idx"] - w["start_idx"]) > drop_less_than_size]


def combine_windows(windows, max_stddev_threshold, min_abs_signal):
    # combine windows
    new_windows = []
    for window in windows:
        if (
            window["stddev"] < max_stddev_threshold
            or window["abs_signal_median"] >= min_abs_signal
        ):
            if (
                len(new_windows) > 0
                and new_windows[-1]["type"] == "signal"
                and pi_norm(abs(new_windows[-1]["mean"] - window["mean"])) < 0.2
                and abs(new_windows[-1]["stddev"] - window["stddev"]) < 0.1
            ):
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                window["type"] = "signal"
                new_windows.append(window)
        else:
            # previous window was also noise
            if len(new_windows) > 0 and new_windows[-1]["type"] == "noise":
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                window["type"] = "noise"
                new_windows.append(window)
    return new_windows


def recompute_stats_for_windows(windows, v, pd, trim):
    for window in windows:
        _pd = pd[window["start_idx"] : window["end_idx"]]
        _v = v[window["start_idx"] : window["end_idx"]]
        r = get_stats_for_signal(_v, _pd, trim)
        window["mean"] = r[0]
        window["stddev"] = r[1]
        window["abs_signal_median"] = r[2]
    return windows


def compute_downsampled_segmentation_mask(simple_segmentation, n_windows, window_size):
    seg_mask = np.zeros(n_windows).astype(bool)
    for window in simple_segmentation:
        seg_mask[
            window["start_idx"] // window_size : window["end_idx"] // window_size,
        ] = True
    return seg_mask


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
    assert v.ndim == 2 and v.shape[0] == 2

    # numba incorrectly compiles this and it returns a float64 :'(
    pd = get_phase_diff(v).astype(np.float32)
    candidate_windows = []
    window_idxs_and_stats = windowed_trimmed_circular_mean_and_stddev(
        v, pd, window_size=window_size, stride=stride, trim=trim
    )
    # window_idxs_and_stats[1]= trimmed_cm, trimmed_stddev, abs_signal_median

    original_windows = [
        {
            "start_idx": idx[0],
            "end_idx": idx[1],
            "mean": stats[0],
            "stddev": stats[1],
            "abs_signal_median": stats[2],
        }
        for idx, stats in zip(window_idxs_and_stats[0], window_idxs_and_stats[1])
    ]

    # combine windows
    candidate_windows = combine_windows(
        original_windows, max_stddev_threshold, min_abs_signal
    )

    # drop all noise windows less than 3windows in size
    candidate_windows = drop_windows_smaller_than(
        candidate_windows, drop_less_than_size
    )

    # only keep signal windows surounded by noise
    candidate_windows = keep_signal_surrounded_by_noise(candidate_windows)

    simple_segmentation = recompute_stats_for_windows(candidate_windows, v, pd, trim)
    downsampled_segmentation_mask = compute_downsampled_segmentation_mask(
        simple_segmentation,
        window_size=window_size,
        n_windows=window_idxs_and_stats[1].shape[0],
    )

    return {
        "simple_segmentation": simple_segmentation,
        # "all_windows": original_windows,
        "downsampled_segmentation_mask": downsampled_segmentation_mask,
        "all_windows_stats": window_idxs_and_stats[
            1
        ],  # trimmed_cm, trimmed_stddev, abs_signal_median
    }


@njit
def phase_diff_to_theta(
    phase_diff, wavelength, distance_between_receivers, large_phase_goes_right
):
    if not large_phase_goes_right:
        phase_diff = phase_diff.copy()
        phase_diff *= -1
    phase_diff = pi_norm(phase_diff)
    # clip the values to reasonable?
    edge = 1 - 1e-8
    sin_arg = np.clip(
        wavelength * phase_diff / (distance_between_receivers * np.pi * 2),
        -edge,
        edge,
    )
    x = np.arcsin(sin_arg)
    return x, np.pi - x, (pi_norm(x) + pi_norm(np.pi - x)) / 2 - np.pi


# """
# Do not use pi_norm here!! this can flip the sign on observations near the edge
# """


@njit
def get_phase_diff(signal_matrix):
    return pi_norm(np.angle(signal_matrix[0]) - np.angle(signal_matrix[1]))


@torch.jit.script
def torch_get_phase_diff(signal_matrix: torch.Tensor):
    return torch_pi_norm_pi(signal_matrix[:, 0].angle() - signal_matrix[:, 1].angle())


# @njit
def get_avg_phase(signal_matrix, trim=0.0):
    return np.array(
        circular_mean(get_phase_diff(signal_matrix=signal_matrix)[None], trim=trim)
    ).reshape(-1)


@torch.jit.script
def torch_get_avg_phase_notrim(signal_matrix: torch.Tensor):
    return torch.hstack(
        torch_circular_mean_notrim(
            torch_get_phase_diff(signal_matrix=signal_matrix)[None],
        )
    )


# @torch.jit.script
def torch_get_avg_phase(signal_matrix: torch.Tensor, trim: float):
    return torch.tensor(
        torch_circular_mean(
            torch_get_phase_diff(signal_matrix=signal_matrix)[None], trim
        )
    )


@functools.lru_cache(maxsize=1024)
def rotation_matrix(orientation):
    s = np.sin(orientation)
    c = np.cos(orientation)
    return np.array([c, s, -s, c], dtype=np.float64).reshape(2, 2)


speed_of_light = 299792458  # m/s speed of light
c = speed_of_light  # speed of light


class Source(object):
    def __init__(self, pos):
        self.pos = np.array(pos)
        assert self.pos.shape[1] == 2

    def signal(self, sampling_times):
        return (
            np.cos(2 * np.pi * sampling_times) + np.sin(2 * np.pi * sampling_times) * 1j
        )

    def demod_signal(self, signal, demod_times):
        return signal


class IQSource(Source):
    def __init__(self, pos, frequency, phase=0, amplitude=1):
        super().__init__(pos)
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude

    def signal(self, sampling_times):
        return (
            np.cos(2 * np.pi * sampling_times * self.frequency + self.phase)
            + np.sin(2 * np.pi * sampling_times * self.frequency + self.phase) * 1j
        )


class SinSource(Source):
    def __init__(self, pos, frequency, phase=0, amplitude=1):
        super().__init__(pos)
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude

    def signal(self, sampling_times):
        return self.amplitude * np.sin(
            2 * np.pi * sampling_times * self.frequency + self.phase
        )  # .reshape(1,-1)


class MixedSource(Source):
    def __init__(self, pos, source_a, source_b, h=None):
        super().__init__(pos)
        self.source_a = source_a
        self.source_b = source_b
        self.h = h

    def signal(self, sampling_times):
        return self.source_a(sampling_times) * self.source_b(sampling_times)


class NoiseWrapper(Source):
    def __init__(self, internal_source, sigma=1):
        super().__init__(internal_source.pos)
        self.internal_source = internal_source
        self.sigma = sigma

    def signal(self, sampling_times):
        assert sampling_times.ndim == 2  # receivers x time
        return (
            self.internal_source.signal(sampling_times)
            + (
                np.random.randn(*sampling_times.shape)
                + np.random.randn(*sampling_times.shape) * 1j
            )
            * self.sigma
        )


class Detector(object):
    def __init__(self, sampling_frequency, orientation=0, sigma=0.0):
        self.sources = []
        self.source_positions = None
        self.receiver_positions = None
        self.sampling_frequency = sampling_frequency
        self.position_offset = np.zeros(2, dtype=np.float32)
        self.orientation = orientation  # rotation to the right in radians to apply to receiver array coordinate system
        self.sigma = sigma

    def add_source(self, source):
        self.sources.append(source)
        if self.source_positions is None:
            self.source_positions = np.array(source.pos, dtype=np.float32).reshape(1, 2)
        else:
            self.source_positions = np.vstack(
                [
                    self.source_positions,
                    np.array(source.pos, dtype=np.float32).reshape(1, 2),
                ]
            )

    def distance_receiver_to_source(self):
        return np.linalg.norm(
            self.all_receiver_pos()[:, None] - self.source_positions[None],
            axis=2,
        )

    def rm_sources(self):
        self.sources = []
        self.source_positions = None

    def set_receiver_positions(self, receiver_positions):
        self.receiver_positions = receiver_positions

    def add_receiver(self, receiver_position):
        if self.receiver_positions is None:
            self.receiver_positions = np.array(
                receiver_position, dtype=np.float32
            ).reshape(1, 2)
        else:
            self.receiver_positions = np.vstack(
                [
                    self.receiver_positions,
                    np.array(receiver_position, dtype=np.float32).reshape(1, 2),
                ]
            )

    def n_receivers(self):
        return self.receiver_positions.shape[0]

    # returns n_receivers, 2(x,y)
    def all_receiver_pos(self, with_offset=True):
        if with_offset:
            return (
                self.position_offset
                + (rotation_matrix(self.orientation) @ self.receiver_positions.T).T
            ).astype(np.float32)
        else:
            return (
                rotation_matrix(self.orientation) @ self.receiver_positions.T
            ).T.astype(np.float32)

    def receiver_pos(self, receiver_idx, with_offset=True):
        if with_offset:
            return (
                self.position_offset
                + (
                    rotation_matrix(self.orientation)
                    @ self.receiver_positions[receiver_idx].T
                ).T
            ).astype(np.float32)
        else:
            return (
                rotation_matrix(self.orientation)
                @ self.receiver_positions[receiver_idx].T
            ).T.astype(np.float32)

    def get_signal_matrix(self, start_time, duration, rx_lo=0):
        n_samples = int(duration * self.sampling_frequency)
        base_times = (
            start_time
            + rf_linspace(0, n_samples - 1, n_samples) / self.sampling_frequency
        )

        if self.sigma == 0.0:
            sample_matrix = np.zeros(
                (self.receiver_positions.shape[0], n_samples), dtype=np.cdouble
            )  # receivers x samples
        else:
            sample_matrix = (
                np.random.randn(self.receiver_positions.shape[0], n_samples, 2)
                .view(np.cdouble)
                .reshape(self.receiver_positions.shape[0], n_samples)
                * self.sigma
            )

        if len(self.sources) == 0:
            return sample_matrix

        distances = (
            self.distance_receiver_to_source().T
        )  # sources x receivers # TODO numerical stability,  maybe project angle and calculate diff
        # TODO diff can be small relative to absolute distance
        # time_delays = distances / c
        base_time_offsets = (
            base_times[None, None] - (distances / c)[..., None]
        )  # sources x receivers x sampling intervals
        distances_squared = distances**2
        for source_index, _source in enumerate(self.sources):
            # get the signal from the source for these times
            signal = _source.signal(
                base_time_offsets[source_index]
            )  # .reshape(base_time_offsets[source_index].shape) # receivers x sampling intervals
            normalized_signal = signal / distances_squared[source_index][..., None]
            _base_times = np.broadcast_to(
                base_times, normalized_signal.shape
            )  # broadcast the basetimes for rx_lo on all receivers
            demod_times = np.broadcast_to(
                _base_times.mean(axis=0, keepdims=True), _base_times.shape
            )  # TODO this just takes the average?
            ds = _source.demod_signal(
                normalized_signal, demod_times
            )  # TODO nested demod?
            sample_matrix += ds
        return sample_matrix  # ,raw_signal,demod_times,base_time_offsets[0]


"""
Spacing is the full distance between each antenna
This zero centers the array, for two elements we get
spacing*([-0.5, 0.5]) for the two X positions 
and 0 on the Y positions
"""


@functools.lru_cache(maxsize=1024)
def linear_receiver_positions(n_elements, spacing):
    receiver_positions = np.zeros((n_elements, 2), dtype=np.float32)
    receiver_positions[:, 0] = spacing * (np.arange(n_elements) - (n_elements - 1) / 2)
    return receiver_positions


class ULADetector(Detector):
    def __init__(
        self, sampling_frequency, n_elements, spacing, sigma=0.0, orientation=0.0
    ):
        super().__init__(sampling_frequency, sigma=sigma, orientation=orientation)
        self.set_receiver_positions(linear_receiver_positions(n_elements, spacing))


@functools.lru_cache(maxsize=1024)
def circular_receiver_positions(n_elements, radius):
    theta = (rf_linspace(0, 2 * np.pi, n_elements + 1)[:-1]).reshape(-1, 1)
    return radius * np.hstack([np.cos(theta), np.sin(theta)])


class UCADetector(Detector):
    def __init__(
        self, sampling_frequency, n_elements, radius, sigma=0.0, orientation=0.0
    ):
        super().__init__(sampling_frequency, sigma=sigma, orientation=orientation)
        self.set_receiver_positions(circular_receiver_positions(n_elements, radius))


@functools.lru_cache(maxsize=1024)
def get_thetas(spacing):
    thetas = rf_linspace(-np.pi, np.pi, spacing)
    return thetas, np.vstack([np.cos(thetas)[None], np.sin(thetas)[None]]).T


"""
if numba:

    @jit(nopython=True)
    def beamformer_numba_helper(
        receiver_positions,
        signal_matrix,
        carrier_frequency,
        spacing,
        thetas,
        source_vectors,
    ):
        steer_dot_signal = np.zeros(thetas.shape[0])
        carrier_wavelength = c / carrier_frequency

        projection_of_receiver_onto_source_directions = (
            source_vectors @ receiver_positions.T
        )
        args = (
            2
            * np.pi
            * projection_of_receiver_onto_source_directions
            / carrier_wavelength
        )
        steering_vectors = np.exp(-1j * args)
        steer_dot_signal = (
            np.absolute(steering_vectors @ signal_matrix).sum(axis=1)
            / signal_matrix.shape[1]
        )

        return thetas, steer_dot_signal, steering_vectors


def beamformer_numba(
    receiver_positions, signal_matrix, carrier_frequency, spacing=64 + 1
):
    thetas, source_vectors = get_thetas(spacing)
    return beamformer_numba_helper(
        receiver_positions,
        signal_matrix,
        carrier_frequency,
        spacing,
        thetas,
        source_vectors,
    )
"""


# from Jon Kraft github
def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20 * np.log10(
        np.abs(s_shift) / (2**11)
    )  # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs


###
"""
Beamformer takes as input the
  receiver positions
  signal marix representing signal received at those positions
  carrier_frequency
  calibration between receivers
  spacing of thetas
  offset (subtracted from thetas, often needs to be supplied with (-) to counter the actual angle)

(1) compute spacing different directions in the unit circle to test signal strength at
(2) compute the source unit vectors for each of the directions in (1)
(3) project the receiver positions onto source unit vectors, this tells us relative distance to each receiver
(4) using the above distances, normalize to wavelength units, and compute phase adjustments
(5) transform phase adjustments into complex matrix
(6) apply adjustment matrix to signals and take the mean of the absolute values

**Beamformer theta output is right rotated (clockwise)**

Beamformer assumes,
0 -> x=0, y=1
pi/2 -> x=1, y=0
-pi/2 -> x=-1, y=0
"""


"""
              Y+
              0deg
              |
              |
              |
              |    
       .......*........  X+ ( on axis right is +pi/2) 
              |
              |
              |
              |   
        (-pi (counter clockwise) or +pi (clockwise))

        by default for ULA

              Y+
              0deg
              |
              |
              |
              |    
    RX0.......*........ RX1 X+ ( on axis right is +pi/2) 
              |
              |
              |
              |   
    
"""


def precompute_steering_vectors(
    receiver_positions,
    carrier_frequency,
    spacing=64 + 1,
    calibration=None,
    offset=0.0,
):
    thetas = np.linspace(-np.pi, np.pi, spacing, dtype=np.float32)  # -offset
    source_vectors = np.vstack(
        [np.sin(thetas + offset)[None], np.cos(thetas + offset)[None]]
    ).T

    projection_of_receiver_onto_source_directions = (
        source_vectors @ receiver_positions.T
    )

    carrier_wavelength = c / carrier_frequency
    args = (
        2 * np.pi * projection_of_receiver_onto_source_directions / carrier_wavelength
    )
    steering_vectors = np.exp(-1j * args)
    if calibration is not None:
        steering_vectors = steering_vectors * calibration[None]

    return steering_vectors


def thetas_from_nthetas(nthetas):
    return np.linspace(-np.pi, np.pi, nthetas)


def beamformer_given_steering_nomean_cp(
    steering_vectors,
    signal_matrix,
):
    # the delay sum is performed in the matmul step, the absolute is over the summed value
    phase_adjusted = np.dot(
        steering_vectors, signal_matrix
    )  # this is adjust and sum in one step
    return np.absolute(phase_adjusted)


@njit
def beamformer_given_steering_nomean(
    steering_vectors,
    signal_matrix,
):
    # the delay sum is performed in the matmul step, the absolute is over the summed value
    phase_adjusted = np.dot(
        steering_vectors, signal_matrix
    )  # this is adjust and sum in one step
    return np.absolute(phase_adjusted)


# @njit
def beamformer_given_steering(
    steering_vectors,
    signal_matrix,
):
    return beamformer_given_steering_nomean(steering_vectors, signal_matrix).mean(
        axis=1
    )


def beamformer_thetas(spacing):
    return np.linspace(-np.pi, np.pi, spacing)


def beamformer(
    receiver_positions,  # recievers X 2[X,Y]
    signal_matrix,  # receivers X samples
    carrier_frequency,
    calibration=None,
    spacing=64 + 1,
    offset=0.0,
):
    thetas = beamformer_thetas(spacing)  # -offset
    source_vectors = np.vstack(
        [np.sin(thetas + offset)[None], np.cos(thetas + offset)[None]]
    ).T

    projection_of_receiver_onto_source_directions = (
        source_vectors @ receiver_positions.T
    )

    carrier_wavelength = c / carrier_frequency
    args = (
        2 * np.pi * projection_of_receiver_onto_source_directions / carrier_wavelength
    )
    steering_vectors = np.exp(-1j * args)
    if calibration is not None:
        steering_vectors = steering_vectors * calibration[None]
    # the delay sum is performed in the matmul step, the absolute is over the summed value
    phase_adjusted = np.matmul(
        steering_vectors, signal_matrix
    )  # this is adjust and sum in one step
    steer_dot_signal = np.absolute(phase_adjusted).mean(axis=1)  # mean over samples
    return thetas, steer_dot_signal, steering_vectors


def get_peaks_for_2rx(beam_former_output):
    n = beam_former_output.shape[0]
    first_peak = np.argmax(beam_former_output)

    pivot = n // 4
    third_peak = n // 4 + n // 2
    if first_peak > n // 2:
        pivot = n // 2 + n // 4
        third_peak = n // 4

    d = np.abs(first_peak - pivot)

    return pivot + d, pivot - d, third_peak
