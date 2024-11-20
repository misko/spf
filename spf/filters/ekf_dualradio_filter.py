import time
from functools import partial

import numpy as np
import torch
from filterpy.kalman import ExtendedKalmanFilter
from matplotlib import pyplot as plt

from spf.filters.filters import (
    F_cached,
    Q_discrete_white_noise_cached,
    SPFFilter,
    dual_radio_mse_theta_metrics,
    paired_h_phi_observation_from_theta_state,
    paired_hjacobian_phi_observation_from_theta_state,
    residual,
)
from spf.rf import pi_norm


class SPFPairedKalmanFilter(ExtendedKalmanFilter, SPFFilter):
    def __init__(self, ds, phi_std=0.5, p=5, dynamic_R=0.0, **kwargs):
        super().__init__(dim_x=2, dim_z=2, **kwargs)
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

        if not self.ds.temp_file:
            self.all_observations = torch.vstack(
                [self.ds.mean_phase["r0"], self.ds.mean_phase["r1"]]
            ).T

    def R_at_x(self, angle):
        return 2.5 * np.exp(-((abs(pi_norm(angle)) - np.pi / 2) ** 2))

    def fix_x(self):
        self.x[0] = pi_norm(self.x[0])

    """
    Given current RX known state, time difference and noise level
    Predict and return prior
    """

    def predict(self, dt, noise_std):  # q_var -> noise_std
        self.F = F_cached(dt)
        self.Q = Q_discrete_white_noise_cached(
            dim=2, dt=dt, var=noise_std
        )  # TODO Cache this
        ### predict self.x
        self.x = np.dot(self.F, self.x)
        self.fix_x()
        ###

        # update covar
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

    def update(self, observation):
        r = np.array(
            [
                [
                    self.R_at_x(
                        pi_norm(self.x[0, 0] - self.radio_array_angle_offsets[0])
                    ),
                    0,
                ],
                [
                    0,
                    self.R_at_x(
                        pi_norm(self.x[0, 0] - self.radio_array_angle_offsets[1])
                    ),
                ],
            ]
        )
        super().update(
            np.array(observation),
            partial(
                paired_hjacobian_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                radio_array_angle_offsets=self.radio_array_angle_offsets,
            ),
            partial(
                paired_h_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                radio_array_angle_offsets=self.radio_array_angle_offsets,
            ),
            residual=residual,
            R=self.R if self.dynamic_R == 0 else r * self.dynamic_R,
        )
        self.fix_x()

    """
    Given an idx return the observation at that point
    """

    def observation(self, idx):
        if not self.ds.temp_file:
            return self.all_observations[idx].reshape(2, 1)
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
        return dual_radio_mse_theta_metrics(
            trajectory, self.ds.craft_ground_truth_thetas
        )

    def setup(self, initial_conditions={}):
        # self.x = np.array([[self.ds[0][0]["craft_ground_truth_theta"].item()], [0]])
        self.x = np.array([[0.0], [0]])

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
            # compute the prior
            self.predict(
                dt=dt,
                noise_std=noise_std,
            )

            if debug:
                hx = paired_h_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                    radio_array_angle_offsets=self.radio_array_angle_offsets,
                )
                jacobian = paired_hjacobian_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                    radio_array_angle_offsets=self.radio_array_angle_offsets,
                )

            # compute update = likelihood * prior
            observation = self.observation(idx)

            self.update(observation=observation)

            current_instance = {
                "mu": self.x,
                "var": self.P,
                "craft_theta": self.x[0, 0],
            }
            if debug:
                current_instance.update(
                    {
                        "jacobian": jacobian[0, 0],
                        "hx": hx,
                        "P_theta": self.P[0, 0],
                        "observation": observation,
                    }
                )

            trajectory.append(current_instance)

        return trajectory


def run_and_plot_dualradio_EKF(ds, trajectory=None):

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[1].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    kf = SPFPairedKalmanFilter(ds=ds, phi_std=5.0, p=5, dynamic_R=0.0)
    trajectory = (
        kf.trajectory(max_iterations=None, debug=True)
        if trajectory is None
        else trajectory
    )

    n = len(trajectory)
    for rx_idx in range(2):
        ax[0].scatter(
            range(min(n, ds.mean_phase[f"r{rx_idx}"].shape[0])),
            ds.mean_phase[f"r{rx_idx}"][:n],
            label=f"r{rx_idx} estimated phi",
            s=1.0,
            alpha=1.0,
            color="red",
        )
        ax[0].plot(ds.ground_truth_phis[rx_idx][:n], label="perfect phi")
    ground_truth_theta = [
        pi_norm(ds[idx][0]["craft_y_rad"].item()) for idx in range(len(trajectory))
    ]
    ax[1].plot(
        ground_truth_theta,
        label="craft gt theta",
    )

    xs = np.array([x["craft_theta"] for x in trajectory])
    stds = np.sqrt(np.array([x["P_theta"] for x in trajectory]))
    zscores = (xs - np.array(ground_truth_theta)) / stds

    ax[1].plot(xs, label="EKF-x", color="orange")
    ax[1].fill_between(
        np.arange(xs.shape[0]),
        xs - stds,
        xs + stds,
        label="EKF-std",
        color="orange",
        alpha=0.2,
    )

    ax[0].set_ylabel("radio phi")

    ax[0].legend()
    ax[0].set_title(f"Radio")
    ax[1].legend()
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("radio theta")

    ax[2].hist(zscores.reshape(-1), bins=25)
    # fig.suptitle("Single ladies (radios) EKF")
    # fig.savefig(f"{output_prefix}_single_ladies_ekf.png")
    return fig
