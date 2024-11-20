from functools import cache, partial

import numpy as np
import torch
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from matplotlib import pyplot as plt

from spf.filters.filters import (
    SPFFilter,
    pairedXY_h_phi_observation_from_theta_state,
    pairedXY_hjacobian_phi_observation_from_theta_state,
    residual,
)
from spf.rf import pi_norm, torch_pi_norm_pi


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
    def __init__(self, ds, phi_std=0.5, p=5, dynamic_R=0.0, **kwargs):
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
            R=self.R if self.dynamic_R == 0.0 else r * self.dynamic_R,
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

    def _ground_truth_xy(self):
        return torch.vstack(
            [
                self.ds.cached_keys[0]["tx_pos_x_mm"],
                self.ds.cached_keys[0]["tx_pos_y_mm"],
            ]
        ).T

    """
    Given a trajectory compute metrics over it
    """

    def metrics(self, trajectory):
        pred_theta = torch.tensor(np.hstack([x["mu"][0] for x in trajectory]))
        pred_xy = torch.tensor(np.hstack([x["mu"][[0, 1]] for x in trajectory]).T)
        return {
            "mse_craft_theta": (
                torch_pi_norm_pi(self.ds.craft_ground_truth_thetas - pred_theta) ** 2
            )
            .mean()
            .item(),
            "mse_xy": ((self._ground_truth_xy() - pred_xy) ** 2).mean().item(),
        }

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

            self.x[4] = self.ds[idx][0]["rx_pos_x_mm"].item()
            self.x[5] = self.ds[idx][0]["rx_pos_y_mm"].item()
            rel_x, rel_y = self.x[0, 0] - self.x[4, 0], self.x[1, 0] - self.x[5, 0]
            target_theta = pi_norm(np.arctan2(rel_x, rel_y))

            current_instance = {
                "mu": self.x,
                "var": self.P[:4, :4],
                "craft_theta": target_theta,
            }
            if debug:

                current_instance.update(
                    {
                        "jacobian": jacobian[0, 0],
                        "hx": hx,
                        "P_theta": 0.01,  # self.P[0, 0],
                        "observation": observation,
                    }
                )

            trajectory.append(current_instance)

        return trajectory


def run_and_plot_dualradioXY_EKF(ds, trajectory=None):

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[1].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    kf = SPFPairedXYKalmanFilter(ds=ds, phi_std=5, p=0.1, dynamic_R=True)
    trajectory = (
        kf.trajectory(debug=True, noise_std=10.0, dt=1)
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
    # zscores = (xs - np.array(ground_truth_theta)) / stds

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

    # ax[2].hist(zscores.reshape(-1), bins=25)
    # fig.suptitle("Single ladies (radios) EKF")
    # fig.savefig(f"{output_prefix}_single_ladies_ekf.png")

    pos_x = np.array([x["mu"][0] for x in trajectory])
    pos_y = np.array([x["mu"][1] for x in trajectory])
    rpos_x = np.array([x["mu"][4] for x in trajectory])
    rpos_y = np.array([x["mu"][5] for x in trajectory])
    vel_x = np.array([x["mu"][2] for x in trajectory])
    vel_y = np.array([x["mu"][3] for x in trajectory])
    gt_x = [ds[idx][0]["tx_pos_x_mm"].item() for idx in range(len(trajectory))]
    gt_y = [ds[idx][0]["tx_pos_y_mm"].item() for idx in range(len(trajectory))]
    ax[2].plot(gt_x, color="red")
    ax[2].plot(gt_y, color="green")
    ax[2].scatter(range(len(pos_x)), pos_x, color="red")
    ax[2].scatter(range(len(pos_y)), pos_y, color="green")
    ax[2].plot(range(len(rpos_x)), rpos_x, color="blue")
    ax[2].plot(range(len(rpos_y)), rpos_y, color="black")
    # ax[2].scatter(range(len(vel_x)), vel_x, color="red")
    # ax[2].scatter(range(len(vel_y)), vel_y, color="green")
    return fig
