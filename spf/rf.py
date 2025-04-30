# from numba import jit
import functools

import numpy as np
import torch
from numba import njit
from scipy.signal import find_peaks

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store_cm
from spf.sdrpluto.detrend import detrend_np, merge_dynamic_windows_np
from numba import njit, prange
try:
    import cupy as cp
except:
    pass
import math

import torch
from numba import jit
from scipy.stats import trim_mean

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


# @torch.jit.script
def torch_circular_diff_to_mean(angles: torch.Tensor, means: torch.Tensor):
    assert means.ndim == 1
    a = torch.abs(means[:, None] - angles) % (2 * torch.pi)
    b = 2 * torch.pi - a
    m, _ = torch.min(torch.vstack([a[None], b[None]]), dim=0)
    return m


# def torch_pi_norm(x, max_angle=torch.pi):
#     return ((x + max_angle) % (2 * max_angle)) - max_angle


# @torch.jit.script
def torch_pi_norm_pi(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


# @torch.jit.script
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
# @torch.jit.script
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


# @torch.jit.script
def torch_reduce_theta_to_positive_y(ground_truth_thetas):
    reduced_thetas = ground_truth_thetas.clone()
    # |theta|>np.pi/2 means its on the y<0
    reduced_ground_truth_thetas_mask = abs(reduced_thetas) > torch.pi / 2
    reduced_ground_truth_thetas_at_mask = reduced_thetas[
        reduced_ground_truth_thetas_mask
    ]
    reduced_thetas[reduced_ground_truth_thetas_mask] = (
        torch.sign(reduced_ground_truth_thetas_at_mask) * torch.pi
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


# @torch.jit.script
def torch_circular_mean_notrim(angles: torch.Tensor):
    assert angles.ndim == 2
    _sin_angles = torch.sin(angles)
    _cos_angles = torch.cos(angles)
    cm = torch.arctan2(_sin_angles.sum(dim=1), _cos_angles.sum(dim=1)) % (2 * torch.pi)

    r = torch_pi_norm_pi(cm)
    return r, r


# @torch.jit.script
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


def mean_phase_mean(angles, weights):
    assert np.isfinite(weights).all()
    _sin_angles = np.sin(angles) * weights
    _cos_angles = np.cos(angles) * weights
    cm = np.arctan2(_sin_angles.sum(), _cos_angles.sum()) % (2 * np.pi)
    return pi_norm(cm)


def torch_mean_phase_mean(angles, weights):
    assert weights.isfinite().all()
    _sin_angles = np.sin(angles) * weights
    _cos_angles = np.cos(angles) * weights
    cm = np.arctan2(_sin_angles.sum(), _cos_angles.sum()) % (2 * np.pi)
    return torch_pi_norm(cm)


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


# @torch.jit.script
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


# @torch.jit.script
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


# @torch.jit.script
def torch_get_phase_diff(signal_matrix: torch.Tensor):
    return torch_pi_norm_pi(signal_matrix[:, 0].angle() - signal_matrix[:, 1].angle())


# @njit
def get_avg_phase(signal_matrix, trim=0.0):
    return np.array(
        circular_mean(get_phase_diff(signal_matrix=signal_matrix)[None], trim=trim)
    ).reshape(-1)


# @torch.jit.script
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
    
@njit(parallel=True)
def beamformer_given_steering_nomean_fast(
    steering_vectors,  # [n_thetas, n_antennas] (complex64 or complex128)
    signal_matrix,     # [n_antennas, n_samples] (complex64 or complex128)
):
    n_thetas, n_antennas = steering_vectors.shape
    _, n_samples = signal_matrix.shape

    output = np.empty((n_thetas, n_samples), dtype=np.float32)

    for theta_idx in prange(n_thetas):
        for sample_idx in range(n_samples):
            real_sum = 0.0
            imag_sum = 0.0
            for ant_idx in range(n_antennas):
                sv = steering_vectors[theta_idx, ant_idx]
                sig = signal_matrix[ant_idx, sample_idx]
                # complex multiply conjugate(steering) * signal
                real_sum += (sv.real * sig.real + sv.imag * sig.imag)
                imag_sum += (sv.real * sig.imag - sv.imag * sig.real)
            # Compute magnitude (sqrt(real^2 + imag^2))
            power = (real_sum**2 + imag_sum**2)**0.5
            output[theta_idx, sample_idx] = power

    return output

@jit(nopython=True)
def beamformer_given_steering_nomean_cp(
    steering_vectors,  # shape: [n_thetas, n_antennas]
    signal_matrix,  # shape: [n_antennas, n_samples]
):
    """
    GPU-accelerated implementation of the beamforming algorithm using CuPy.

    This function applies precomputed steering vectors to a signal matrix to perform
    beamforming. For each potential angle (theta), it calculates the beamformer output
    which represents the signal power in that direction.

    Mathematical operation:
    - For each angle θ with steering vector a(θ), compute: y(θ) = a(θ)ᴴ · signal
    - Then calculate power: P(θ) = |y(θ)|

    Args:
        steering_vectors: Complex matrix of steering vectors [n_thetas, n_antennas]
                          Each row is a steering vector for a specific angle
        signal_matrix: Complex signal data from antenna array [n_antennas, n_samples]

    Returns:
        Array of shape [n_thetas, n_samples] containing beamformer power output
        for each angle and time sample
    """
    # The matrix multiplication performs phase alignment and summation in one step
    # For each theta, it multiplies the steering vector with all signal samples simultaneously
    phase_adjusted = np.dot(steering_vectors, signal_matrix)

    # Calculate magnitude (absolute value) to get power
    # This converts complex values to real power measurements
    return np.absolute(phase_adjusted)


def beamformer_given_steering_nomean(
    steering_vectors,  # shape: [n_thetas, n_antennas]
    signal_matrix,  # shape: [n_antennas, n_samples]
):
    """
    CPU implementation of the beamforming algorithm.

    This is the CPU equivalent of beamformer_given_steering_nomean_cp. It applies the
    precomputed steering vectors to the signal matrix to perform beamforming.

    The beamforming principle works as follows:
    1. Each steering vector contains complex weights for each antenna element
    2. These weights delay/phase-shift signals to align them for a specific direction
    3. When signals are summed after applying these weights:
       - Signals from the target direction add constructively
       - Signals from other directions add destructively

    Args:
        steering_vectors: Complex matrix of steering vectors [n_thetas, n_antennas]
                          Each row is a steering vector for a specific angle
        signal_matrix: Complex signal data from antenna array [n_antennas, n_samples]

    Returns:
        Array of shape [n_thetas, n_samples] containing beamformer power output
        for each angle and time sample
    """
    # The dot product performs two operations:
    # 1. Phase alignment: steering_vectors align signal phases for each potential angle
    # 2. Summation: signals from all antennas are combined into a single output
    phase_adjusted = np.dot(steering_vectors, signal_matrix)

    # Convert to power (magnitude of complex values)
    # Higher values indicate more energy from that direction
    return np.absolute(phase_adjusted)


def beamformer_given_steering(
    steering_vectors,  # shape: [n_thetas, n_antennas]
    signal_matrix,  # shape: [n_antennas, n_samples]
):
    """
    High-level beamforming function that returns angle-power spectrum averaged over time.

    This function extends beamformer_given_steering_nomean by averaging the
    beamforming output across all time samples. This improves SNR and
    provides a single power spectrum as a function of angle.

    Args:
        steering_vectors: Complex matrix of steering vectors [n_thetas, n_antennas]
        signal_matrix: Complex signal data from antenna array [n_antennas, n_samples]

    Returns:
        Array of shape [n_thetas] containing time-averaged beamformer power output
        for each angle
    """
    # First calculate power for each angle and time sample
    beam_output = beamformer_given_steering_nomean(steering_vectors, signal_matrix)

    # Then average over time (axis=1) to get a single power value per angle
    # This improves SNR by averaging out temporal noise
    return beam_output.mean(axis=1)


def beamformer_thetas(spacing):
    """
    Generate an array of theta angles for beamforming.

    Creates a uniform angular grid from -π to +π with the specified number of points.
    These angles represent the potential directions of arrival to be evaluated.

    Args:
        spacing: Number of angular points to generate

    Returns:
        Array of theta values in radians
    """
    return np.linspace(-np.pi, np.pi, spacing)


def beamformer(
    receiver_positions,  # shape: [n_receivers, 2] - X,Y coordinates
    signal_matrix,  # shape: [n_receivers, n_samples]
    carrier_frequency,  # RF carrier frequency in Hz
    calibration=None,  # Optional per-receiver calibration factors
    spacing=64 + 1,  # Number of angular points (65 by default)
    offset=0.0,  # Angular offset in radians
):
    """
    Complete beamforming implementation that handles the entire process:
    1. Generate angles to scan
    2. Compute steering vectors for each angle
    3. Apply steering vectors to signal matrix
    4. Calculate power spectrum as a function of angle

    This function implements the delay-and-sum beamforming algorithm for
    arbitrary antenna geometries. It works as follows:

    1. For each potential angle θ, calculate the expected phase offset at each antenna
    2. Create steering vectors that compensate for these phase offsets
    3. Apply steering vectors to align signals from direction θ
    4. Measure power after alignment - higher power indicates likely signal direction

    Mathematical basis:
    - Path difference between antennas: d·sin(θ)
    - Phase difference: 2π·d·sin(θ)/λ  where λ is wavelength
    - Steering vector: e^(-j·phase_difference)

    Args:
        receiver_positions: Array of antenna positions [n_receivers, 2]
        signal_matrix: Complex signal data [n_receivers, n_samples]
        carrier_frequency: RF carrier frequency in Hz
        calibration: Optional complex calibration factors for each receiver
        spacing: Number of angular points to evaluate
        offset: Angular offset to apply to all angles in radians

    Returns:
        thetas: Array of angles evaluated
        steer_dot_signal: Power spectrum as a function of angle
        steering_vectors: Computed steering vectors for each angle
    """
    # Generate set of angles to scan for signals
    thetas = beamformer_thetas(spacing)

    # Convert angles to unit vectors representing propagation directions
    # Each row is a [sin(θ), cos(θ)] vector pointing in direction θ
    source_vectors = np.vstack(
        [np.sin(thetas + offset)[None], np.cos(thetas + offset)[None]]
    ).T

    # Calculate projection of each receiver position onto each source direction
    # This determines the path length difference for signals arriving from each direction
    projection_of_receiver_onto_source_directions = (
        source_vectors @ receiver_positions.T  # Matrix multiplication
    )

    # Convert physical path differences to phase differences
    # Phase = 2π · distance / wavelength
    carrier_wavelength = c / carrier_frequency
    args = (
        2 * np.pi * projection_of_receiver_onto_source_directions / carrier_wavelength
    )

    # Create steering vectors using complex exponentials
    # steering_vectors[i,j] = e^(-j·phase) for angle i and receiver j
    # The negative sign aligns phases to compensate for the path differences
    steering_vectors = np.exp(-1j * args)

    # Apply calibration factors if provided
    # These correct for hardware phase/amplitude imbalances between receivers
    if calibration is not None:
        steering_vectors = steering_vectors * calibration[None]

    # Apply steering vectors to signal matrix
    # For each angle, this aligns and sums signals from all receivers
    phase_adjusted = np.matmul(steering_vectors, signal_matrix)

    # Calculate magnitude and average over time
    # Higher values indicate more energy from that direction
    steer_dot_signal = np.absolute(phase_adjusted).mean(axis=1)

    return thetas, steer_dot_signal, steering_vectors


def get_peaks_for_2rx(beam_former_output):
    # beam_former_output /= beam_former_output.max()
    # beam_former_output = beam_former_output.round(decimals=2)
    n = beam_former_output.shape[0]
    first_peak = np.argmax(beam_former_output)

    pivot = n // 4
    third_peak = n // 4 + n // 2
    if first_peak > n // 2:
        pivot = n // 2 + n // 4
        third_peak = n // 4

    d = np.abs(first_peak - pivot)

    return pivot + d, pivot - d, third_peak


# chat gpt
def get_top_3_peaks(beam_former_output, distance=1, prominence=0.1):
    """
    Return indices of the 3 highest local maxima in beam_former_output.

    Parameters:
    -----------
    beam_former_output : 1D array-like
        Beamformer output.
    distance : int
        Minimum distance between peaks. Increase if you want to avoid
        finding peaks that are very close to each other.
    prominence : float
        Required prominence of peaks. Increase if you want to filter out
        small (noisy) peaks.

    Returns:
    --------
    peaks[:3] : list
        Indices of up to 3 highest local maxima (by amplitude).
    """
    # 1. Find all local maxima with constraints
    peaks, properties = find_peaks(
        beam_former_output, distance=distance, prominence=prominence
    )

    # 2. Sort the found peaks by their amplitude in descending order
    peaks_sorted = sorted(peaks, key=lambda idx: beam_former_output[idx], reverse=True)

    # 3. Return up to 3 highest
    return peaks_sorted[:3]


# Example usage
# beam_former_output = ...
# top_3_peaks = get_top_3_peaks(beam_former_output)
# print("Indices of top 3 peaks:", top_3_peaks)


# chatgpt
def rotate_dist(input_dist: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Rotate discrete distributions defined over angles -pi to +pi by the
    specified rotations (in radians).

    Arguments:
    ----------
    input_dist : (B, 65) tensor
        Each row is a discrete distribution over 65 equally spaced angles in [-pi, pi].
    rotations : (B,) or (B, 1) tensor
        The rotation (in radians) for each distribution in the batch.

    Returns:
    --------
    rotated : (B, 65) tensor
        The input distributions rotated by the specified angles (with interpolation).
    """

    rotations = rotations.view(-1)  # Make sure rotations has shape (B,)

    B, n_bins = input_dist.shape
    # assert n_bins == 65, "Expected distributions of size 65 along axis=1."

    # 1) Define the angle grid for the 65 bins
    # angles = torch.linspace(-math.pi, math.pi, steps=n_bins, device=input_dist.device)
    dtheta = 2.0 * math.pi / n_bins  # 2π / 65
    angles = (
        torch.arange(n_bins, device=input_dist.device, dtype=input_dist.dtype) * dtheta
    )

    # Bin width
    bin_width = angles[1] - angles[0]  # ~ 2*pi/64

    # 2) For each bin j in [0..64], the "rotated" angle is angles[j] - rotation.
    angles_2d = angles.unsqueeze(0)  # shape (1, 65)
    rotations_2d = rotations.unsqueeze(1)  # shape (B, 1)

    # (B, 65)
    target_angles = angles_2d - rotations_2d  # %(2*torch.pi)

    # 3) Convert these target angles to float indices in [0..64]
    float_indices = torch.round(
        (target_angles - angles[0]) / bin_width, decimals=4
    )  # shape (B, 65)

    # 4) Floor to get lower bin index, then +1 for upper bin
    idx0 = torch.floor(float_indices).long()  # can be negative
    idx1 = idx0 + 1

    # Wrap both with modulo 65
    idx0_mod = idx0 % n_bins
    idx1_mod = idx1 % n_bins

    # 5) Interpolation weights
    w1 = float_indices - idx0.float()
    w0 = 1.0 - w1

    # 6) Gather the corresponding values from input_dist
    dist_gather_0 = input_dist.gather(1, idx0_mod)
    dist_gather_1 = input_dist.gather(1, idx1_mod)

    # 7) Linear interpolation
    rotated = w0 * dist_gather_0 + w1 * dist_gather_1
    return torch.nn.functional.normalize(rotated, p=1.0, dim=1)
