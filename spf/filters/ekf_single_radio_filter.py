from functools import partial

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from matplotlib import pyplot as plt

from spf.filters.filters import (
    F_cached,
    Q_discrete_white_noise_cached,
    SPFFilter,
    residual,
    single_h_phi_observation_from_theta_state,
    single_hjacobian_phi_observation_from_theta_state,
    single_radio_mse_theta_metrics,
)
from spf.rf import pi_norm, reduce_theta_to_positive_y, torch_pi_norm_pi


class SPFKalmanFilter(ExtendedKalmanFilter, SPFFilter):
    def __init__(self, ds, rx_idx, phi_std=0.5, p=5, dynamic_R=False, **kwargs):
        super().__init__(dim_x=2, dim_z=1, **kwargs)
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

        self.antenna_spacing_in_wavelengths = antenna_spacing / wavelength
        self.rx_idx = rx_idx

        self.dynamic_R = dynamic_R

    def R_at_x(self):
        return 2.5 * np.exp(-((abs(pi_norm(self.x[0, 0])) - np.pi / 2) ** 2))

    def fix_x(self):
        self.x[0] = reduce_theta_to_positive_y(pi_norm(self.x[0]))
        assert self.x[0] >= -np.pi / 2 and self.x[0] <= np.pi / 2

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
        super().update(
            np.array(observation),
            partial(
                single_hjacobian_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
            ),
            partial(
                single_h_phi_observation_from_theta_state,
                antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
            ),
            residual=residual,
            R=self.R if not self.dynamic_R else (np.array([[self.R_at_x()]]) ** 2) * 5,
        )
        self.fix_x()

    """
    Given an idx return the observation at that point
    """

    def observation(self, idx):
        return self.ds[idx][self.rx_idx]["mean_phase_segmentation"]

    """
    Given a trajectory compute metrics over it
    """

    def metrics(self, trajectory):
        return single_radio_mse_theta_metrics(
            trajectory, self.ds.ground_truth_thetas[self.rx_idx]
        )

    def setup(self, initial_conditions={}):
        self.x = np.array([[self.ds[self.rx_idx][0]["ground_truth_theta"].item()], [0]])

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
                hx = single_h_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                )
                jacobian = single_hjacobian_phi_observation_from_theta_state(
                    x=self.x,
                    antenna_spacing_in_wavelengths=self.antenna_spacing_in_wavelengths,
                )

            # compute update = likelihood * prior
            observation = self.observation(idx)
            self.update(observation=observation)

            current_instance = {
                "mu": self.x,
                "var": self.P,
                "theta": self.x[0, 0],
            }
            if debug:
                current_instance.update(
                    {
                        "jacobian": jacobian[0, 0],
                        "hx": hx,
                        "P_theta": self.P[0, 0],
                        "observation": observation.item(),
                    }
                )

            trajectory.append(current_instance)

        return trajectory


def run_and_plot_single_radio_EKF(ds, trajectories=None):

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    for rx_idx in [0, 1]:  # [0, 1]:
        ax[1, rx_idx].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
        ax[1, rx_idx].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))

        kf = SPFKalmanFilter(
            ds=ds, rx_idx=rx_idx, phi_std=5.0, p=5, dynamic_R=True
        )  # , phi_std=0.5, p=5, **kwargs):
        trajectory = (
            kf.trajectory(max_iterations=None, debug=True)
            if trajectories is None
            else trajectories[rx_idx]
        )
        jacobian = [x["jacobian"] for x in trajectory]
        zs = [x["observation"] for x in trajectory]
        # trajectory, jacobian, zs = trajectory_for_phi(rx_idx, ds)
        jacobian = np.array(jacobian)
        zs = np.array(zs)
        n = len(trajectory)
        ax[0, rx_idx].scatter(
            range(min(n, ds.mean_phase[f"r{rx_idx}"].shape[0])),
            ds.mean_phase[f"r{rx_idx}"][:n],
            label=f"r{rx_idx} estimated phi",
            s=1.0,
            alpha=1.0,
            color="red",
        )
        ax[0, rx_idx].plot(ds.ground_truth_phis[rx_idx][:n], label="perfect phi")
        ax[0, rx_idx].plot(jacobian, label="jacobian")
        ax[0, rx_idx].plot(zs, label="zs")
        ax[0, rx_idx].plot(np.clip(zs / jacobian, a_min=-5, a_max=5), label="zs/j")
        ax[1, rx_idx].plot(
            [ds[idx][rx_idx]["ground_truth_theta"] for idx in range(min(n, len(ds)))],
            label=f"r{rx_idx} gt theta",
        )
        reduced_gt_theta = np.array(
            [
                reduce_theta_to_positive_y(ds[idx][rx_idx]["ground_truth_theta"])
                for idx in range(min(n, len(ds)))
            ]
        )
        ax[1, rx_idx].plot(
            reduced_gt_theta,
            label=f"r{rx_idx} reduced gt theta",
        )

        xs = np.array([x["theta"] for x in trajectory])
        stds = np.sqrt(np.array([x["P_theta"] for x in trajectory]))
        zscores = (xs - reduced_gt_theta) / stds

        ax[1, rx_idx].plot(xs, label="EKF-x", color="orange")
        ax[1, rx_idx].fill_between(
            np.arange(xs.shape[0]),
            xs - stds,
            xs + stds,
            label="EKF-std",
            color="orange",
            alpha=0.2,
        )

        ax[0, rx_idx].set_ylabel("radio phi")

        ax[0, rx_idx].legend()
        ax[0, rx_idx].set_title(f"Radio {rx_idx}")
        ax[1, rx_idx].legend()
        ax[1, rx_idx].set_xlabel("time step")
        ax[1, rx_idx].set_ylabel("radio theta")

        ax[2, rx_idx].hist(zscores.reshape(-1), bins=25)
    return fig
