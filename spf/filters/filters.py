from functools import cache

import numpy as np
from filterpy.common import Q_discrete_white_noise

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
