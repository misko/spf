from functools import cache, partial

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter

from spf.filters.filters import (
    SPFFilter,
    pairedXY_h_phi_observation_from_theta_state,
    pairedXY_hjacobian_phi_observation_from_theta_state,
    residual,
)
from spf.rf import pi_norm


@cache
def Qxy_discrete_white_noise_cached(**kwargs):
    q = np.zeros((6, 6))
    q[:4, :4] = Q_discrete_white_noise(**kwargs)
    return q


@cache
def Fxy_cached(dt):
    return (
        np.eye(6)
        + np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],  # no updates for dtx_x/dt
                [0, 0, 0, 0, 0, 0],  # no updates for dtx_y/dt
                [0, 0, 0, 0, 0, 0],  # no updates for rx_x
                [0, 0, 0, 0, 0, 0],  # no updates for rx_y
            ]
        )
        * dt
    )


class SPFPairedXYKalmanFilter(ExtendedKalmanFilter, SPFFilter):
    def __init__(self, ds, phi_std=0.5, p=5, dynamic_R=False, **kwargs):
        # state x = [ tx_x, tx_y, dtx_x/dt, dtx_y/dt, rx_x, rx_y] , x,y relative to (0,0) not craft
        super().__init__(dim_x=6, dim_z=2, **kwargs)
        self.R *= phi_std**2
        self.P *= p  # initialized as identity?

        self.ds = ds
        # flip the sign of antennas
        assert (
            ds.yaml_config["receivers"][0]["antenna-spacing-m"]
            == ds.yaml_config["receivers"][1]["antenna-spacing-m"]
        )
        antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]

        assert ds.wavelengths[0] == ds.wavelengths[1]
        wavelength = ds.wavelengths[0]

        self.radio_array_angle_offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * np.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * np.pi,
        ]

        self.antenna_spacing_in_wavelengths = antenna_spacing / wavelength

        self.dynamic_R = dynamic_R

    def R_at_x(self, angle):
        return 2.5 * np.exp(-((abs(pi_norm(angle)) - np.pi / 2) ** 2))

    def fix_x(self):
        # self.x[[2, 3]] = np.clip(self.x[[2, 3]], a_min=-0.1, a_max=0.1)
        pass

    """
    Given current RX known state, time difference and noise level
    Predict and return prior
    """

    def predict(self, dt, noise_std):  # q_var -> noise_std
        self.F = Fxy_cached(dt)
        self.Q = Qxy_discrete_white_noise_cached(
            dim=2, dt=dt, var=noise_std, block_size=2, order_by_dim=False
        )  # TODO Cache thisQ_discrete_white_noise(dim=2, dt=1,block_size=2,order_by_dim=False)
        ### predict self.x
        self.x = np.dot(self.F, self.x)
        self.fix_x()
        ###
        # print("P", self.P.shape, "Q", self.Q.shape)

        # update covar
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

    def update(self, observation):

        rel_x, rel_y = self.x[0, 0] - self.x[4, 0], self.x[1, 0] - self.x[5, 0]
        target_theta = np.arctan2(rel_x, rel_y)
        r = np.array(
            [
                [
                    self.R_at_x(
                        pi_norm(target_theta - self.radio_array_angle_offsets[0])
                    ),
                    0,
                ],
                [
                    0,
                    self.R_at_x(
                        pi_norm(target_theta - self.radio_array_angle_offsets[1])
                    ),
                ],
            ]
        )
        # print("Update", self.x)
        super().update(
            np.array(observation),
            partial(
                pairedXY_hjacobian_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                radio_array_angle_offsets=self.radio_array_angle_offsets,
            ),
            partial(
                pairedXY_h_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                radio_array_angle_offsets=self.radio_array_angle_offsets,
            ),
            residual=residual,
            R=self.R if not self.dynamic_R else r,
        )
        self.fix_x()

    """
    Given an idx return the observation at that point
    """

    def observation(self, idx):
        return np.vstack(
            [
                self.ds[idx][0]["mean_phase_segmentation"],
                self.ds[idx][1]["mean_phase_segmentation"],
            ]
        )

    """
    Given a trajectory compute metrics over it
    """

    def metrics(self, trajectory):
        pass

    def setup(self, initial_conditions={}):
        self.x = np.array(
            [
                [self.ds[0][0]["tx_pos_x_mm"].item()],
                [self.ds[0][0]["tx_pos_y_mm"].item()],
                [0],
                [0],
                [self.ds[0][0]["rx_pos_x_mm"].item()],
                [self.ds[0][0]["rx_pos_y_mm"].item()],
            ]
        )

    def posterior_to_mu_var(self, posterior):
        return {"var": None, "mu": None}

    """
    Iterate over the dataset and generate a trajectory
    """

    def trajectory(
        self,
        initial_conditions={},
        dt=1.0,
        noise_std=0.01,
        max_iterations=None,
        debug=False,
    ):
        self.setup(initial_conditions)
        trajectory = []
        n = (
            len(self.ds)
            if max_iterations is None
            else min(max_iterations, len(self.ds))
        )
        for idx in range(n):
            self.P[4:, 4:] = 0
            self.predict(
                dt=dt,
                noise_std=noise_std,
            )

            observation = self.observation(idx)

            self.x[4] = self.ds[idx][0]["rx_pos_x_mm"].item()
            self.x[5] = self.ds[idx][0]["rx_pos_y_mm"].item()
            if debug:
                hx = pairedXY_h_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                    radio_array_angle_offsets=self.radio_array_angle_offsets,
                )
                jacobian = pairedXY_hjacobian_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                    radio_array_angle_offsets=self.radio_array_angle_offsets,
                )

            self.update(observation=observation)

            current_instance = {
                "mu": self.x,
                "var": self.P[:4, :4],
            }
            if debug:
                self.x[4] = self.ds[idx][0]["rx_pos_x_mm"].item()
                self.x[5] = self.ds[idx][0]["rx_pos_y_mm"].item()
                rel_x, rel_y = self.x[0, 0] - self.x[4, 0], self.x[1, 0] - self.x[5, 0]
                target_theta = pi_norm(np.arctan2(rel_x, rel_y))
                current_instance.update(
                    {
                        "jacobian": jacobian[0, 0],
                        "hx": hx,
                        "theta": target_theta,
                        "P_theta": 0.01,  # self.P[0, 0],
                        "observation": observation,
                    }
                )

            trajectory.append(current_instance)

        return trajectory
