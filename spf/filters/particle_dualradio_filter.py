import torch
from matplotlib import pyplot as plt

from spf.filters.filters import (
    ParticleFilter,
    add_noise,
    dual_radio_mse_theta_metrics,
    theta_phi_to_p_vec,
)
from spf.rf import torch_pi_norm_pi


@torch.jit.script
def pf_single_theta_dual_radio_update(weights, particles, z, empirical_dist, offsets):
    z0 = theta_phi_to_p_vec(
        torch_pi_norm_pi(particles[:, 0] - offsets[0]),
        z[0],
        empirical_dist[0],
    )
    z1 = theta_phi_to_p_vec(
        torch_pi_norm_pi(particles[:, 0] - offsets[1]),
        z[1],
        empirical_dist[1],
    )
    weights = weights * z0 * z1 + 1.0e-30  # avoid round-off to zero
    return weights / torch.sum(weights)  # normalize


class PFSingleThetaDualRadio(ParticleFilter):
    def __init__(self, ds):
        self.ds = ds
        self.offsets = torch.tensor(
            [
                ds.yaml_config["receivers"][0]["theta-in-pis"] * torch.pi,
                ds.yaml_config["receivers"][1]["theta-in-pis"] * torch.pi,
            ]
        )

        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        self.cached_empirical_dist = torch.vstack(
            [
                self.ds.get_empirical_dist(0).T.unsqueeze(0),
                self.ds.get_empirical_dist(1).T.unsqueeze(0),
            ]
        )
        if not self.ds.temp_file:
            self.all_observations = torch.vstack(
                [self.ds.mean_phase["r0"], self.ds.mean_phase["r1"]]
            ).T

    def observation(self, idx):
        if not self.ds.temp_file:
            return self.all_observations[idx]
        return torch.concatenate(
            [
                self.ds[idx][0]["mean_phase_segmentation"].reshape(1),
                self.ds[idx][1]["mean_phase_segmentation"].reshape(1),
            ],
            axis=0,
        )

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std, generator=self.generator)

    def update(self, z):
        self.weights = pf_single_theta_dual_radio_update(
            self.weights,
            self.particles,
            z,
            self.cached_empirical_dist,
            self.offsets,
        )

    def metrics(self, trajectory):
        return dual_radio_mse_theta_metrics(
            trajectory, self.ds.craft_ground_truth_thetas
        )

    def trajectory(self, **kwargs):
        trajectory = super().trajectory(**kwargs)
        for x in trajectory:
            x["craft_theta"] = x["mu"][0]
            x["P_theta"] = x["var"][0]
        return trajectory


def plot_single_theta_dual_radio(ds):

    pf = PFSingleThetaDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        std=torch.tensor([[2, 0.1]]),
        return_particles=False,
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

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

    ax[1].plot(
        # torch_pi_norm_pi(ds[0][0]["craft_y_rad"][0]),
        torch_pi_norm_pi(ds.craft_ground_truth_thetas),
        label="craft gt theta",
        linestyle="dashed",
    )

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

    ax[0].set_ylabel("radio phi")

    ax[0].legend()
    ax[0].set_title(f"Radio 0 & 1")
    ax[1].legend()
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("Theta between target and receiver craft")
    return fig
