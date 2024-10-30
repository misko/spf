from functools import cache

import scipy
import torch
from matplotlib import pyplot as plt

from spf.filters.filters import ParticleFilter, add_noise, theta_phi_to_p_vec
from spf.rf import torch_pi_norm_pi


class PFXYDualRadio(ParticleFilter):
    def __init__(self, ds):
        super().__init__(ds)
        self.offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * torch.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * torch.pi,
        ]
        self.speed_dist = scipy.stats.norm(1.5, 3)

        self.generator = self.generator.manual_seed(0)

    def our_state(self, idx):
        return torch.vstack(
            [
                self.ds.cached_keys[0]["rx_pos_x_mm"][idx],
                self.ds.cached_keys[0]["rx_pos_y_mm"][idx],
            ]
        ).reshape(-1)

    def ground_truth_xy(self):
        return torch.vstack(
            [
                self.ds.cached_keys[0]["tx_pos_x_mm"],
                self.ds.cached_keys[0]["tx_pos_y_mm"],
            ]
        ).T

    def ground_truth_thetas(self):
        return self.ds.craft_ground_truth_thetas

    def observation(self, idx):
        return torch.concatenate(
            [
                self.ds[idx][0]["mean_phase_segmentation"].reshape(1),
                self.ds[idx][1]["mean_phase_segmentation"].reshape(1),
            ],
            axis=0,
        )

    @cache
    def cached_empirical_dist(self, rx_idx):
        return self.ds.get_empirical_dist(rx_idx).T

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, dt, our_state, noise_std):
        add_noise(self.particles, noise_std, generator=self.generator)
        # update movement
        self.particles[:, [1, 2]] += dt * self.particles[:, [3, 4]]
        # recomput theta
        rx_pos = our_state.reshape(1, 2)
        tx_pos = self.particles[:, [1, 2]]
        d = tx_pos - rx_pos
        self.particles[:, 0] = torch_pi_norm_pi(torch.arctan2(d[:, 0], d[:, 1]))

    def update(self, z):
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[0]),
            z[0],
            self.cached_empirical_dist(0),
        )
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[1]),
            z[1],
            self.cached_empirical_dist(1),
        )
        # prior on velocity
        self.weights *= self.speed_dist.pdf(
            torch.linalg.norm(self.particles[:, [3, 4]], axis=1)
        )
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= self.weights.sum()  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.hstack([x["mu"][0] for x in trajectory])
        pred_xy = torch.vstack([x["mu"][[1, 2]] for x in trajectory])
        ground_truth_theta = self.ground_truth_thetas()
        ground_truth_xy = self.ground_truth_xy()
        # breakpoint()
        return {
            "mse_theta": (torch_pi_norm_pi(ground_truth_theta - pred_theta) ** 2)
            .mean()
            .item(),
            "mse_xy": ((ground_truth_xy - pred_xy) ** 2).mean().item(),
        }


def plot_xy_dual_radio(ds):
    pf = PFXYDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        N=128 * 16,
        mean=torch.tensor([[0, 0, 0, 0, 0]]),
        std=torch.tensor([[0, 2, 2, 0.1, 0.1]]),
        return_particles=False,
        noise_std=torch.tensor([[0, 15, 15, 0.5, 0.5]]),
    )

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[1].axhline(y=torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    n = len(traj_paired)
    colors = ["blue", "orange"]
    for rx_idx in (0, 1):
        ax[0].scatter(
            range(min(n, ds.mean_phase[f"r{rx_idx}"].shape[0])),
            ds.mean_phase[f"r{rx_idx}"][:n],
            label=f"r{rx_idx} estimated phi",
            s=1.0,
            alpha=0.1,
            color=colors[rx_idx],
        )
        ax[0].plot(
            ds.ground_truth_phis[rx_idx][:n],
            color=colors[rx_idx],
            label=f"r{rx_idx} perfect phi",
            linestyle="dashed",
        )

    # ax[1].plot(torch_pi_norm_pi(ds[0][0]["craft_y_rad"][0]))
    ax[1].plot(torch_pi_norm_pi(ds.craft_ground_truth_thetas))

    xs = torch.hstack([x["mu"][0] for x in traj_paired])
    stds = torch.sqrt(torch.hstack([x["var"][0] for x in traj_paired]))

    ax[1].fill_between(
        torch.arange(xs.shape[0]),
        xs - stds,
        xs + stds,
        label="PF-std",
        color="red",
        alpha=0.2,
    )
    ax[1].scatter(range(xs.shape[0]), xs, label="PF-x", color="orange", s=0.5)

    gt_xy = torch.vstack(
        [
            ds.cached_keys[0]["tx_pos_x_mm"],
            ds.cached_keys[0]["tx_pos_y_mm"],
        ]
    )
    xys = torch.vstack([x["mu"][[1, 2]] for x in traj_paired])
    ax[2].scatter(range(xys.shape[0]), xys[:, 0], label="PF-x", color="orange", s=0.5)
    ax[2].scatter(range(xys.shape[0]), xys[:, 1], label="PF-y", color="blue", s=0.5)
    # tx = np.vstack([ds[0][0]["tx_pos_x_mm"], ds[0][0]["tx_pos_y_mm"]])
    ax[2].plot(gt_xy[0, :], label="gt-x", color="red")
    ax[2].plot(gt_xy[1, :], label="gt-y", color="black")

    ax[0].set_ylabel("radio phi")

    ax[0].legend()
    ax[0].set_title(f"Radio 0 & 1")
    ax[1].legend()
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("Theta between target and receiver craft")
    ax[2].legend()
    return fig
