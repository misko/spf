from functools import cache

import numpy as np
import torch
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import systematic_resample

from spf.rf import pi_norm


@cache
def Q_discrete_white_noise_cached(**kwargs):
    return Q_discrete_white_noise(**kwargs)


@cache
def F_cached(dt):
    return np.eye(2) + np.array([[0, 1], [0, 0]]) * dt


def residual(a, b):
    # we are dealing in phi space here, not theta space
    # in phi space lets make sure we use the closer of the
    # two points
    return pi_norm(a - b)


"""
x = [ theta dtheta/dt ]
z = [ phi ]

F = [ [ 1 dt ],
      [ 0  1 ]]

h(x) = sin(x[0]) *  2 * pi  * (d/ wavelength )

H(x) = [ dh/dx_1 , dh/dx_2 ] = [ cos(x[0]) * 2 * pi (d/ wavelength ) , 0]

"""


"""
Convert a state (x) representing [ theta dtheta/dt ] into an observation of phi
given the spacing between antennas as a fraction of wavelength
"""


def h_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offset=0
):
    assert x.ndim == 2 and x.shape[0] == 2 and x.shape[1] == 1
    return np.array(
        [
            np.sin(x[0, 0] - radio_array_angle_offset)
            * 2
            * np.pi
            * antenna_spacing_in_wavelengths
        ]
    )


"""
Compute the derivative of the observation generated from state x (theta,dtheta/dt)
with respect to state variables (theta,dtheta/dt)
"""


def hjacobian_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offset=0
):
    assert x.ndim == 2 and x.shape[0] == 2 and x.shape[1] == 1
    return np.array(
        [
            [
                np.cos(x[0, 0] - radio_array_angle_offset)
                * 2
                * np.pi
                * antenna_spacing_in_wavelengths,
                0,
            ]
        ]
    )


def single_h_phi_observation_from_theta_state(x, antenna_spacing_in_wavelengths):
    assert x[0, 0] >= -np.pi / 2 and x[0, 0] <= np.pi / 2
    return h_phi_observation_from_theta_state(x, antenna_spacing_in_wavelengths)


def single_hjacobian_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths
):
    assert x[0, 0] >= -np.pi / 2 and x[0, 0] <= np.pi / 2
    return hjacobian_phi_observation_from_theta_state(x, antenna_spacing_in_wavelengths)


def paired_h_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offsets
):
    return np.vstack(
        [
            h_phi_observation_from_theta_state(
                x,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[0],
            ),
            h_phi_observation_from_theta_state(
                x,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[1],
            ),
        ]
    )


def paired_hjacobian_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offsets
):
    return np.vstack(
        [
            hjacobian_phi_observation_from_theta_state(
                x,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[0],
            ),
            hjacobian_phi_observation_from_theta_state(
                x,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[1],
            ),
        ]
    )


def pairedXY_h_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offsets
):
    rel_x, rel_y = x[0, 0] - x[4, 0], x[1, 0] - x[5, 0]
    target_theta = pi_norm(np.arctan2(rel_x, rel_y))
    two_state_theta_dtheta = np.array([[target_theta], [0]])
    return np.vstack(
        [
            h_phi_observation_from_theta_state(
                two_state_theta_dtheta,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[0],
            ),
            h_phi_observation_from_theta_state(
                two_state_theta_dtheta,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[1],
            ),
        ]
    )


def pairedXY_hjacobian_phi_observation_from_theta_state(
    x, antenna_spacing_in_wavelengths, radio_array_angle_offsets
):
    rel_x, rel_y = x[0, 0] - x[4, 0], x[1, 0] - x[5, 0]
    target_theta = pi_norm(np.arctan2(rel_x, rel_y))
    # print("Target", target_theta)
    two_state_theta_dtheta = np.array([[target_theta], [0]])
    two_state_jacobian = np.vstack(
        [
            hjacobian_phi_observation_from_theta_state(
                two_state_theta_dtheta,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[0],
            ),
            hjacobian_phi_observation_from_theta_state(
                two_state_theta_dtheta,
                antenna_spacing_in_wavelengths=antenna_spacing_in_wavelengths,
                radio_array_angle_offset=radio_array_angle_offsets[1],
            ),
        ]
    )
    d2 = rel_x**2 + rel_y**2
    dtheta_drel_x = rel_y / d2
    dtheta_drel_y = -rel_x / d2
    dphi0_dtheta = two_state_jacobian[0, 0]
    dphi1_dtheta = two_state_jacobian[1, 0]
    jacobian = np.zeros((2, 6))
    jacobian[0, 0] = dphi0_dtheta * dtheta_drel_x  #  * drel_x/dx
    jacobian[0, 1] = dphi0_dtheta * dtheta_drel_y  # * drel_y/dy
    jacobian[1, 0] = dphi1_dtheta * dtheta_drel_x  # * drel_x/dx
    jacobian[1, 1] = dphi1_dtheta * dtheta_drel_y  # * drel_y/dy
    # print("jacobian",jacobian)
    return jacobian


class SPFFilter:
    def __init__(self, ds):
        self.ds = ds

    """
    Given an idx return the known state : i.e. RX position
    """

    def our_state(self, idx):
        return None

    """
    Given current RX known state, time difference and noise level
    """

    def predict(self, our_state, dt, noise_std) -> None:
        pass

    """
    Given an idx use the internally
    """

    def update(self) -> None:
        pass

    """
    Given an idx return the observation at that point
    """

    def observation(self, idx):
        pass

    """
    Given a trajectory compute metrics over it
    """

    def metrics(self, trajectory):
        pass

    def setup(self, initial_conditions):
        pass

    def posterior_to_mu_var(self, posterior):
        return {"var": None, "mu": None}

    """
    Iterate over the dataset and generate a trajectory
    """

    def trajectory(self, initial_conditions={}, dt=1.0, noise_std=0.01):
        self.setup(initial_conditions)
        trajectory = []
        for idx in range(len(self.ds)):
            prior = self.predict(
                dt=dt,
                noise_std=noise_std,
                our_state=self.our_state(idx),
            )

            posterior = self.update(prior=prior, observation=self.observation(idx))

            trajectory.append(self.posterior_to_mu_var(posterior))

        return {"trajectory": trajectory}


class ParticleFilter(SPFFilter):

    def fix_particles(self):
        return self.particles

    def trajectory(
        self,
        mean,
        std,
        N=128,
        noise_std=None,
        return_particles=False,
        debug=False,
        seed=0,
    ):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.particles = create_gaussian_particles_xy(
            mean, std, N, generator=self.generator
        )
        self.weights = torch.ones((N,), dtype=torch.float64) / N
        trajectory = []
        for idx in range(len(self.ds)):
            self.predict(
                dt=1.0,
                noise_std=noise_std,
                our_state=self.our_state(idx),
            )
            self.fix_particles()

            self.update(z=self.observation(idx))

            # resample if too few effective particles
            if neff(self.weights) < N / 2:
                indexes = torch.as_tensor(systematic_resample(self.weights.numpy()))
                resample_from_index(self.particles, self.weights, indexes)

            mu, var = estimate(self.particles, self.weights)

            trajectory.append({"var": var, "mu": mu})
            if return_particles:
                trajectory[-1]["particles"] = self.particles.copy()
            if debug:
                trajectory[-1]["observation"] = self.observation(idx)
        return trajectory


# @torch.jit.script
def create_gaussian_particles_xy(
    mean: torch.Tensor, std: torch.Tensor, N: int, generator
):
    assert mean.ndim == 2 and mean.shape[0] == 1
    assert std.ndim == 2 and std.shape[0] == 1
    return mean + (torch.randn(N, mean.shape[1], generator=generator) * std)


@torch.jit.script
def theta_phi_to_bins(theta_phi: torch.Tensor, nbins: int):
    if isinstance(theta_phi, float):
        return int(nbins * (theta_phi + torch.pi) / (2 * torch.pi)) % nbins
    return (nbins * (theta_phi + torch.pi) / (2 * torch.pi)).to(torch.long) % nbins


@torch.jit.script
def theta_phi_to_p_vec(thetas, phis, full_p):
    theta_bin = theta_phi_to_bins(thetas, nbins=full_p.shape[0])
    phi_bin = theta_phi_to_bins(phis, nbins=full_p.shape[1])
    return torch.take(full_p[:, phi_bin], theta_bin)


# @torch.jit.script
def add_noise(particles: torch.Tensor, noise_std: torch.Tensor, generator):
    particles[:] += (
        torch.randn(particles.shape, generator=generator) * noise_std
    )  # theta_noise=0.1, theta_dot_noise=0.001


@torch.jit.script
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.fill_(1.0 / len(weights))
    weights /= torch.sum(weights)  # normalize


@torch.jit.script
def neff(weights: torch.Tensor):
    return 1.0 / torch.sum(torch.square(weights))


@torch.jit.script
def weighted_mean(inputs: torch.Tensor, weights: torch.Tensor):
    return torch.sum(weights * inputs, dim=0) / weights.sum()


@torch.jit.script
def estimate(particles: torch.Tensor, weights: torch.Tensor):
    # mean = torch.mean(self.particles, weights=self.weights, axis=0)
    # var = torch.mean((self.particles - mean) ** 2, weights=self.weights, axis=0)
    mean = weighted_mean(particles, weights.reshape(-1, 1))
    var = weighted_mean((particles - mean) ** 2, weights.reshape(-1, 1))
    return mean, var


@torch.jit.script
def theta_phi_to_p(theta, phi, full_p):
    theta_bins = full_p.shape[0]
    phi_bins = full_p.shape[1]
    theta_bin = int(theta_bins * (theta + torch.pi) / (2 * torch.pi)) % theta_bins
    phi_bin = int(phi_bins * (phi + torch.pi) / (2 * torch.pi)) % phi_bins
    return full_p[theta_bin, phi_bin]


@torch.jit.script
def fix_particles_single(particles):
    # this is required! because we need to flip the velocity of theta
    while torch.abs(particles[:, 0]).max() > torch.pi / 2:
        mask = torch.abs(particles[:, 0]) > torch.pi / 2
        particles[mask, 0] = (
            torch.sign(particles[mask, 0]) * torch.pi - particles[mask, 0]
        )
        particles[mask, 1] *= -1
    return particles
