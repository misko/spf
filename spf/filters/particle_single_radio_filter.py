from functools import cache

import numpy as np
import torch
from matplotlib import pyplot as plt

from spf.filters.filters import (
    ParticleFilter,
    add_noise,
    fix_particles_single,
    single_radio_mse_theta_metrics,
    theta_phi_to_p_vec,
)
from spf.rf import reduce_theta_to_positive_y


class PFSingleThetaSingleRadio(ParticleFilter):
    """
    particle state is [ theta, dtheta/dt]
    """

    def __init__(self, ds, rx_idx):
        self.ds = ds
        self.rx_idx = rx_idx
        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        if not self.ds.temp_file:
            self.all_observations = self.ds.mean_phase[f"r{self.rx_idx}"]

    def observation(self, idx):
        if not self.ds.temp_file:
            return self.all_observations[idx]
        return self.ds[idx][self.rx_idx]["mean_phase_segmentation"]

    def fix_particles(self):
        self.particles = fix_particles_single(self.particles)
        # self.particles[:, 0] = reduce_theta_to_positive_y(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std, generator=self.generator)

    @cache
    def cached_empirical_dist(self):
        return self.ds.get_empirical_dist(self.rx_idx).T

    def update(self, z):
        self.weights *= theta_phi_to_p_vec(
            self.particles[:, 0],
            z,
            self.cached_empirical_dist(),
        )
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        return single_radio_mse_theta_metrics(
            trajectory, self.ds.ground_truth_thetas[self.rx_idx]
        )

    def trajectory(self, **kwargs):
        trajectory = super().trajectory(**kwargs)
        for x in trajectory:
            x["theta"] = x["mu"][0]
            x["P_theta"] = x["var"][0]
        return trajectory


def plot_single_theta_single_radio(ds):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for rx_idx in [0, 1]:
        ax[1, rx_idx].axhline(y=torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
        ax[1, rx_idx].axhline(y=-torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))

        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx)
        trajectory = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[2, 0.1]]),
            return_particles=False,
        )
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
        ax[1, rx_idx].plot(
            ds.ground_truth_thetas[rx_idx].reshape(-1),
            label=f"r{rx_idx} gt theta",
        )

        xs = torch.hstack([x["mu"][0] for x in trajectory])
        stds = torch.sqrt(torch.hstack([x["var"][0] for x in trajectory]))

        ax[1, rx_idx].fill_between(
            torch.arange(xs.shape[0]),
            xs - stds,
            xs + stds,
            label="PF-std",
            color="red",
            alpha=0.2,
        )

        ax[1, rx_idx].scatter(
            range(xs.shape[0]), xs, label="PF-x", color="orange", s=0.5
        )

        ax[1, rx_idx].plot(
            reduce_theta_to_positive_y(ds.ground_truth_thetas[rx_idx]),
            label=f"r{rx_idx} reduced gt theta",
            color="black",
            linestyle="dashed",
            linewidth=1,
        )

        ax[0, rx_idx].set_ylabel("radio phi")
        ax[0, rx_idx].set_title(f"Radio {rx_idx}")
        ax[1, rx_idx].set_xlabel("time step")
        ax[1, rx_idx].set_ylabel("radio theta")
        ax[0, rx_idx].legend()
        ax[1, rx_idx].legend()
    return fig
