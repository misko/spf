from functools import cache
import pickle
import numpy as np
import argparse
from filterpy.monte_carlo import systematic_resample
import scipy
import torch
import matplotlib.pyplot as plt
import random
from spf.dataset.spf_dataset import v5spfdataset
import matplotlib.pyplot as plt

from spf.rf import reduce_theta_to_positive_y

import pickle
import random
import tqdm
from multiprocessing import Pool, cpu_count
from spf.rf import pi_norm, reduce_theta_to_positive_y, torch_pi_norm_pi

"""
SINGLE THETA SINGLE RADIO
"""

torch.set_num_threads(1)


@torch.jit.script
def create_gaussian_particles_xy(mean: torch.Tensor, std: torch.Tensor, N: int):
    assert mean.ndim == 2 and mean.shape[0] == 1
    assert std.ndim == 2 and std.shape[0] == 1
    return mean + (torch.randn(N, mean.shape[1]) * std)


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


@torch.jit.script
def add_noise(particles: torch.Tensor, noise_std: torch.Tensor):
    particles[:] += (
        torch.randn_like(particles) * noise_std
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
    while torch.abs(particles[:, 0]).max() > torch.pi / 2:
        mask = torch.abs(particles[:, 0]) > torch.pi / 2
        particles[mask, 0] = (
            torch.sign(particles[mask, 0]) * torch.pi - particles[mask, 0]
        )
        particles[mask, 1] *= -1
    return particles


class ParticleFilter:
    def __init__(self, ds, full_p_fn):
        self.full_p = torch.as_tensor(pickle.load(open(full_p_fn, "rb"))["full_p"])
        self.ds = ds

    def our_state(self, idx):
        return None

    def fix_particles(self):
        return self.particles

    def predict(self, our_state, dt, noise_std):
        pass

    def update(self):
        pass

    def metrics(self, trajectory):
        pass

    def trajectory(self, mean, std, N=128, noise_std=None, return_particles=False):
        self.particles = create_gaussian_particles_xy(mean, std, N)
        self.weights = torch.ones((N,), dtype=torch.float64) / N
        trajectory = []
        all_particles = []
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
                all_particles.append(self.particles.copy())
        return trajectory, all_particles


class PFSingleThetaSingleRadio(ParticleFilter):
    def __init__(self, ds, full_p_fn, rx_idx):
        super().__init__(ds, full_p_fn)
        self.rx_idx = rx_idx
        # self.ground_truth_theta = ds[0][rx_idx]["ground_truth_theta"]  # 1 x 5000
        # self.ground_truth_reduced_theta = reduce_theta_to_positive_y(
        #     self.ground_truth_theta
        # )  # 1 x 5000
        # self.observations = self.ds[0][rx_idx]["mean_phase_segmentation"]  # 1 x 5000

    def ground_truth_theta(self, idx):
        # _r = (
        #     self.ds[idx][self.rx_idx]["ground_truth_theta"].detach().numpy().reshape(-1)
        # )
        return self.ds.ground_truth_thetas[self.rx_idx][idx].reshape(-1)

    def ground_truth_theta_reduced_theta(self, idx):
        return reduce_theta_to_positive_y(self.ground_truth_theta(idx))

    def all_ground_truth_theta_reduced_theta(self):
        # self.ground_truth_thetas[receiver_idx][snapshot_start_idx:snapshot_end_idx]
        return reduce_theta_to_positive_y(self.ds.ground_truth_thetas[self.rx_idx])
        # vs = []
        # for idx in range(len(self.ds)):
        #     vs.append(self.ground_truth_theta_reduced_theta(idx))
        # _r = np.hstack(vs)
        # breakpoint()
        # return _r

    def observation(self, idx):
        # self.mean_phase[f"r{receiver_idx}"][snapshot_start_idx:snapshot_end_idx]
        return self.ds.mean_phase[f"r{self.rx_idx}"][idx]
        # breakpoint()
        # return (
        #     self.ds[idx][self.rx_idx]["mean_phase_segmentation"]
        #     .detach()
        #     .numpy()
        #     .reshape(-1)
        # )

    def fix_particles(self):
        self.particles = fix_particles_single(self.particles)

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std)

    def update(self, z):
        self.weights *= theta_phi_to_p_vec(self.particles[:, 0], z, self.full_p)
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.hstack([x["mu"][0] for x in trajectory])
        ground_truth_reduced_theta = torch.as_tensor(
            self.all_ground_truth_theta_reduced_theta()
        )
        return {
            "mse_theta": (
                torch_pi_norm_pi(ground_truth_reduced_theta - pred_theta) ** 2
            )
            .mean()
            .item()
        }


"""
PAIRED PF
"""


class PFSingleThetaDualRadio(ParticleFilter):
    def __init__(self, ds, full_p_fn):
        super().__init__(ds, full_p_fn)
        # self.ground_truth_theta = ds[0][0]["craft_y_rad"][0]
        # self.observations = np.vstack(
        #     [
        #         ds[0][0]["mean_phase_segmentation"],
        #         ds[0][1]["mean_phase_segmentation"],
        #     ]
        # ).T
        self.offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * torch.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * torch.pi,
        ]
        # breakpoint()
        # a = 1

    def ground_truth_thetas(self):
        return self.ds.craft_ground_truth_thetas
        # vs = []
        # for idx in range(len(self.ds)):
        #     vs.append(self.ds[idx][0]["craft_y_rad"][0])
        # _r = torch.hstack(vs)
        # breakpoint()
        # return torch.hstack(vs)

    def observation(self, idx):
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
        add_noise(self.particles, noise_std=noise_std)

    def update(self, z):
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[0]), z[0], self.full_p
        )
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[1]), z[1], self.full_p
        )
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.hstack([x["mu"][0] for x in trajectory])
        ground_truth_thetas = self.ground_truth_thetas()
        return {
            "mse_theta": (torch_pi_norm_pi(ground_truth_thetas - pred_theta) ** 2)
            .mean()
            .item()
        }


# # flip the order of the antennas
# antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]
# assert antenna_spacing == -ds.yaml_config["receivers"][1]["antenna-spacing-m"]

# wavelength = ds.wavelengths[0]
# assert wavelength == ds.wavelengths[1]

# offsets = [
#     ds.yaml_config["receivers"][0]["theta-in-pis"] * torch.pi,
#     ds.yaml_config["receivers"][1]["theta-in-pis"] * torch.pi,
# ]


"""
Predict XY
"""


class PFXYDualRadio(ParticleFilter):
    def __init__(self, ds, full_p_fn):
        super().__init__(ds, full_p_fn)
        # self.observations = np.vstack(
        #     [
        #         ds[0][0]["mean_phase_segmentation"],
        #         ds[0][1]["mean_phase_segmentation"],
        #     ]
        # ).T
        self.offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * torch.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * torch.pi,
        ]
        # self.our_states = np.vstack(
        #     [ds[0][0]["rx_pos_x_mm"], ds[0][0]["rx_pos_y_mm"]]
        # ).T
        self.speed_dist = scipy.stats.norm(1.5, 3)
        # self.ground_truth_theta = ds[0][0]["craft_y_rad"][0]
        # self.ground_truth_xy = torch.tensor(
        #     np.vstack([ds[0][0]["tx_pos_x_mm"], ds[0][0]["tx_pos_y_mm"]]).T
        # )

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

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, dt, our_state, noise_std):
        add_noise(self.particles, noise_std)
        # update movement
        self.particles[:, [1, 2]] += dt * self.particles[:, [3, 4]]
        # recomput theta
        rx_pos = our_state.reshape(1, 2)
        tx_pos = self.particles[:, [1, 2]]
        d = tx_pos - rx_pos
        self.particles[:, 0] = torch_pi_norm_pi(torch.arctan2(d[:, 0], d[:, 1]))

    def update(self, z):
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[0]), z[0], self.full_p
        )
        self.weights *= theta_phi_to_p_vec(
            torch_pi_norm_pi(self.particles[:, 0] - self.offsets[1]), z[1], self.full_p
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


"""
Plotting
"""


def plot_single_theta_single_radio(ds, full_p_fn):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for rx_idx in [0, 1]:
        ax[1, rx_idx].axhline(y=torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
        ax[1, rx_idx].axhline(y=-torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))

        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx, full_p_fn=full_p_fn)
        trajectory, _ = pf.trajectory(
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


def plot_single_theta_dual_radio(ds, full_p_fn):

    pf = PFSingleThetaDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
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


def plot_xy_dual_radio(ds, full_p_fn):
    pf = PFXYDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
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


def run_single_theta_single_radio(
    ds_fn, precompute_fn, full_p_fn, theta_err=0.1, theta_dot_err=0.001, N=128
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        skip_signal_matrix=True,
        skip_simple_segmentations=True,
        snapshots_per_session=1,
        readahead=True,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
            ]
        ),
    )
    metrics = []
    for rx_idx in [0, 1]:
        pf = PFSingleThetaSingleRadio(
            ds=ds,
            full_p_fn=full_p_fn,
            rx_idx=1,
        )
        trajectory, _ = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[2, 0.1]]),
            noise_std=torch.tensor([[theta_err, theta_dot_err]]),
            return_particles=False,
            N=N,
        )
        metrics.append(
            {
                "type": "single_theta_single_radio",
                "ds_fn": ds_fn,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": pf.metrics(trajectory=trajectory),
            }
        )
    return metrics


def run_single_theta_dual_radio(
    ds_fn, precompute_fn, full_p_fn, theta_err=0.1, theta_dot_err=0.001, N=128
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        skip_signal_matrix=True,
        snapshots_per_session=1,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
            ]
        ),
    )
    pf = PFSingleThetaDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[2, 0.1]]),
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
    )

    return [
        {
            "type": "single_theta_dual_radio",
            "ds_fn": ds_fn,
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def run_xy_dual_radio(
    ds_fn, precompute_fn, full_p_fn, pos_err=15, vel_err=0.5, N=128 * 16
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        skip_signal_matrix=True,
        snapshots_per_session=1,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
            ]
        ),
    )
    # dual radio dual
    pf = PFXYDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
        N=N,
        mean=torch.tensor([[0, 0, 0, 0, 0]]),
        std=torch.tensor([[0, 200, 200, 0.1, 0.1]]),
        return_particles=False,
        noise_std=torch.tensor([[0, pos_err, pos_err, vel_err, vel_err]]),
    )
    return [
        {
            "type": "xy_dual_radio",
            "ds_fn": ds_fn,
            "vel_err": vel_err,
            "pos_err": pos_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def runner(x):
    fn, args = x
    return fn(**args)


if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--datasets",
            type=str,
            help="dataset prefixes",
            nargs="+",
            required=True,
        )
        parser.add_argument(
            "--nthetas",
            type=int,
            required=False,
            default=65,
        )
        parser.add_argument(
            "--device",
            type=str,
            required=False,
            default="cpu",
        )
        parser.add_argument(
            "--skip-qc",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--precompute-cache",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--full-p-fn",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            required=False,
        )
        parser.add_argument(
            "--debug",
            action=argparse.BooleanOptionalAction,
            default=False,
        )

        return parser

    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    jobs = []

    for ds_fn in args.datasets:
        for N in [128, 128 * 4, 128 * 8, 128 * 16]:
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_single_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dot_err,
                            },
                        )
                    )
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_dual_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dot_err,
                            },
                        )
                    )
    for ds_fn in args.datasets:
        for N in [128, 128 * 4, 128 * 8, 128 * 16, 128 * 32]:
            for pos_err in [1000, 100, 50, 30, 15, 5, 0.5]:
                for vel_err in [50, 5, 0.5, 0.05, 0.01, 0.001]:
                    jobs.append(
                        (
                            run_xy_dual_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "pos_err": pos_err,
                                "vel_err": vel_err,
                            },
                        )
                    )

    random.shuffle(jobs)

    if args.debug:
        results = list(tqdm.tqdm(map(runner, jobs), total=len(jobs)))
    else:
        with Pool(20) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(tqdm.tqdm(pool.imap(runner, jobs), total=len(jobs)))
    pickle.dump(results, open("particle_filter_results2.pkl", "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
