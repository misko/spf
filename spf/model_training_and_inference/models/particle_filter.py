import pickle
import numpy as np
import argparse
from filterpy.monte_carlo import systematic_resample
import scipy
import torch
import matplotlib.pyplot as plt

from spf.dataset.spf_dataset import v5spfdataset
import matplotlib.pyplot as plt

from spf.rf import reduce_theta_to_positive_y

import pickle
import random
import tqdm
from multiprocessing import Pool, cpu_count
from spf.rf import pi_norm, reduce_theta_to_positive_y, torch_pi_norm

"""
SINGLE THETA SINGLE RADIO
"""


def create_gaussian_particles_xy(mean, std, N):
    assert mean.ndim == 2 and mean.shape[0] == 1
    assert std.ndim == 2 and std.shape[0] == 1
    return mean + (np.random.randn(N, mean.shape[1]) * std)


def theta_phi_to_bins(theta_phi, nbins):
    if isinstance(theta_phi, float):
        return int(nbins * (theta_phi + np.pi) / (2 * np.pi)) % nbins
    return (nbins * (theta_phi + np.pi) / (2 * np.pi)).astype(int) % nbins


class ParticleFilter:
    def __init__(self, ds, full_p_fn):
        self.full_p = pickle.load(open(full_p_fn, "rb"))["full_p"]
        self.ds = ds
        self.our_states = None

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / len(self.weights))
        self.weights /= sum(self.weights)  # normalize

    def fix_particles(self):
        return self.particles

    def theta_phi_to_p_vec(self, thetas, phis):
        theta_bin = theta_phi_to_bins(thetas, nbins=self.full_p.shape[0])
        phi_bin = theta_phi_to_bins(phis, nbins=self.full_p.shape[1])
        return np.take(self.full_p[:, phi_bin], theta_bin)

    def neff(self):
        return 1.0 / np.sum(np.square(self.weights))

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var = np.average((self.particles - mean) ** 2, weights=self.weights, axis=0)
        return mean, var

    def predict(self, our_state, dt, noise_std):
        pass

    def update(self):
        pass

    def add_noise(self, noise_std):
        self.particles[:] += (
            np.random.randn(*self.particles.shape) * noise_std
        )  # theta_noise=0.1, theta_dot_noise=0.001

    def metrics(self, trajectory):
        pass

    def trajectory(self, mean, std, N=128, noise_std=None, return_particles=False):
        self.particles = create_gaussian_particles_xy(mean, std, N)
        self.weights = np.ones((N,)) / N
        trajectory = []
        all_particles = []

        for idx in range(self.ds.snapshots_per_session):
            self.predict(
                dt=1.0,
                noise_std=noise_std,
                our_state=None if self.our_states is None else self.our_states[idx],
            )
            self.fix_particles()

            self.update(z=self.observations[idx])

            # resample if too few effective particles
            if self.neff() < N / 2:
                indexes = systematic_resample(self.weights)
                self.resample_from_index(indexes)

            mu, var = self.estimate()

            trajectory.append({"var": var, "mu": mu})
            if return_particles:
                all_particles.append(self.particles.copy())
        return trajectory, all_particles

    def theta_phi_to_p(self, theta, phi):
        theta_bins = self.full_p.shape[0]
        phi_bins = self.full_p.shape[1]
        theta_bin = int(theta_bins * (theta + np.pi) / (2 * np.pi)) % theta_bins
        phi_bin = int(phi_bins * (phi + np.pi) / (2 * np.pi)) % phi_bins
        return self.full_p[theta_bin, phi_bin]


class PFSingleThetaSingleRadio(ParticleFilter):
    def __init__(self, ds, full_p_fn, rx_idx):
        super().__init__(ds, full_p_fn)
        self.ground_truth_theta = ds[0][rx_idx]["ground_truth_theta"]
        self.ground_truth_reduced_theta = reduce_theta_to_positive_y(
            self.ground_truth_theta
        )
        self.observations = self.ds[0][rx_idx]["mean_phase_segmentation"]

    def fix_particles(self):
        while np.abs(self.particles[:, 0]).max() > np.pi / 2:
            mask = np.abs(self.particles[:, 0]) > np.pi / 2
            self.particles[mask, 0] = (
                np.sign(self.particles[mask, 0]) * np.pi - self.particles[mask, 0]
            )
            self.particles[mask, 1] *= -1

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = np.array([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        self.add_noise(noise_std=noise_std)

    def update(self, z):
        self.weights *= self.theta_phi_to_p_vec(self.particles[:, 0], z)
        self.weights += 1.0e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.tensor([x["mu"][0] for x in trajectory])
        return {
            "mse_theta": (
                torch_pi_norm(self.ground_truth_reduced_theta - pred_theta) ** 2
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
        self.ground_truth_theta = ds[0][0]["craft_y_rad"][0]
        self.observations = np.vstack(
            [
                ds[0][0]["mean_phase_segmentation"],
                ds[0][1]["mean_phase_segmentation"],
            ]
        ).T
        self.offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * np.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * np.pi,
        ]

    def fix_particles(self):
        self.particles[:, 0] = pi_norm(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = np.array([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        self.add_noise(noise_std=noise_std)

    def update(self, z):
        self.weights *= self.theta_phi_to_p_vec(
            pi_norm(self.particles[:, 0] - self.offsets[0]), z[0]
        )
        self.weights *= self.theta_phi_to_p_vec(
            pi_norm(self.particles[:, 0] - self.offsets[1]), z[1]
        )
        self.weights += 1.0e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.tensor([x["mu"][0] for x in trajectory])
        return {
            "mse_theta": (torch_pi_norm(self.ground_truth_theta - pred_theta) ** 2)
            .mean()
            .item()
        }


# # flip the order of the antennas
# antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]
# assert antenna_spacing == -ds.yaml_config["receivers"][1]["antenna-spacing-m"]

# wavelength = ds.wavelengths[0]
# assert wavelength == ds.wavelengths[1]

# offsets = [
#     ds.yaml_config["receivers"][0]["theta-in-pis"] * np.pi,
#     ds.yaml_config["receivers"][1]["theta-in-pis"] * np.pi,
# ]


"""
Predict XY
"""


class PFXYDualRadio(ParticleFilter):
    def __init__(self, ds, full_p_fn):
        super().__init__(ds, full_p_fn)
        self.observations = np.vstack(
            [
                ds[0][0]["mean_phase_segmentation"],
                ds[0][1]["mean_phase_segmentation"],
            ]
        ).T
        self.offsets = [
            ds.yaml_config["receivers"][0]["theta-in-pis"] * np.pi,
            ds.yaml_config["receivers"][1]["theta-in-pis"] * np.pi,
        ]
        self.our_states = np.vstack(
            [ds[0][0]["rx_pos_x_mm"], ds[0][0]["rx_pos_y_mm"]]
        ).T
        self.speed_dist = scipy.stats.norm(1.5, 3)
        self.ground_truth_theta = ds[0][0]["craft_y_rad"][0]
        self.ground_truth_xy = torch.tensor(
            np.vstack([ds[0][0]["tx_pos_x_mm"], ds[0][0]["tx_pos_y_mm"]]).T
        )

    def fix_particles(self):
        self.particles[:, 0] = pi_norm(self.particles[:, 0])

    def predict(self, dt, our_state, noise_std):
        rx_pos = our_state.reshape(1, 2)
        self.add_noise(noise_std)
        # update movement
        self.particles[:, [1, 2]] += dt * self.particles[:, [3, 4]]
        # recomput theta
        tx_pos = self.particles[:, [1, 2]]
        d = tx_pos - rx_pos
        self.particles[:, 0] = pi_norm(np.arctan2(d[:, 0], d[:, 1]))

    def update(self, z):
        self.weights *= self.theta_phi_to_p_vec(
            pi_norm(self.particles[:, 0] - self.offsets[0]), z[0]
        )
        self.weights *= self.theta_phi_to_p_vec(
            pi_norm(self.particles[:, 0] - self.offsets[1]), z[1]
        )
        # prior on velocity
        self.weights *= self.speed_dist.pdf(
            np.linalg.norm(self.particles[:, [3, 4]], axis=1)
        )
        self.weights += 1.0e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def metrics(self, trajectory):
        pred_theta = torch.tensor(np.array([x["mu"][0] for x in trajectory]))
        pred_xy = torch.tensor(np.array([x["mu"][[1, 2]] for x in trajectory]))
        return {
            "mse_theta": (torch_pi_norm(self.ground_truth_theta - pred_theta) ** 2)
            .mean()
            .item(),
            "mse_xy": ((self.ground_truth_xy - pred_xy) ** 2).mean().item(),
        }


"""
Plotting
"""


def plot_single_theta_single_radio(ds, full_p_fn):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for rx_idx in [0, 1]:
        ax[1, rx_idx].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
        ax[1, rx_idx].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))

        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx, full_p_fn=full_p_fn)
        trajectory, _ = pf.trajectory(
            mean=np.array([[0, 0]]), std=np.array([[2, 0.1]]), return_particles=False
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
            ds[0][rx_idx]["ground_truth_theta"],
            label=f"r{rx_idx} gt theta",
        )

        xs = np.array([x["mu"][0] for x in trajectory])
        stds = np.sqrt(np.array([x["var"][0] for x in trajectory]))

        ax[1, rx_idx].fill_between(
            np.arange(xs.shape[0]),
            xs - stds,
            xs + stds,
            label="PF-std",
            color="red",
            alpha=0.2,
        )
        ax[1, rx_idx].scatter(
            range(ds.snapshots_per_session), xs, label="PF-x", color="orange", s=0.5
        )

        ax[1, rx_idx].plot(
            reduce_theta_to_positive_y(ds[0][rx_idx]["ground_truth_theta"]),
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
        mean=np.array([[0, 0]]), std=np.array([[2, 0.1]]), return_particles=False
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[1].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
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
        torch_pi_norm(ds[0][0]["craft_y_rad"][0]),
        label="craft gt theta",
        linestyle="dashed",
    )

    xs = np.array([x["mu"][0] for x in traj_paired])
    stds = np.sqrt(np.array([x["var"][0] for x in traj_paired]))

    ax[1].fill_between(
        np.arange(xs.shape[0]),
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
        mean=np.array([[0, 0, 0, 0, 0]]),
        std=np.array([[0, 2, 2, 0.1, 0.1]]),
        return_particles=False,
        noise_std=np.array([[0, 15, 15, 0.5, 0.5]]),
    )

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[1].axhline(y=np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
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

    ax[1].plot(torch_pi_norm(ds[0][0]["craft_y_rad"][0]))

    xs = np.array([x["mu"][0] for x in traj_paired])
    stds = np.sqrt(np.array([x["var"][0] for x in traj_paired]))

    ax[1].fill_between(
        np.arange(xs.shape[0]),
        xs - stds,
        xs + stds,
        label="PF-std",
        color="red",
        alpha=0.2,
    )
    ax[1].scatter(
        range(ds.snapshots_per_session), xs, label="PF-x", color="orange", s=0.5
    )

    xys = np.array([x["mu"][[1, 2]] for x in traj_paired])
    ax[2].scatter(
        range(ds.snapshots_per_session), xys[:, 0], label="PF-x", color="orange", s=0.5
    )
    ax[2].scatter(
        range(ds.snapshots_per_session), xys[:, 1], label="PF-y", color="blue", s=0.5
    )
    tx = np.vstack([ds[0][0]["tx_pos_x_mm"], ds[0][0]["tx_pos_y_mm"]])
    ax[2].plot(tx[0, :], label="gt-x", color="red")
    ax[2].plot(tx[1, :], label="gt-y", color="black")

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
        snapshots_per_session=-1,
    )
    metrics = []
    for rx_idx in [0, 1]:
        pf = PFSingleThetaSingleRadio(
            ds=ds,
            full_p_fn=full_p_fn,
            rx_idx=1,
        )
        trajectory, _ = pf.trajectory(
            mean=np.array([[0, 0]]),
            std=np.array([[2, 0.1]]),
            noise_std=np.array([[theta_err, theta_dot_err]]),
            return_particles=False,
            N=N,
        )
        metrics.append(
            {
                "type": "single_theta_single_radio",
                "ds_fn": ds_fn,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dor_err": theta_dot_err,
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
        snapshots_per_session=-1,
    )
    pf = PFSingleThetaDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
        mean=np.array([[0, 0]]),
        N=N,
        std=np.array([[2, 0.1]]),
        noise_std=np.array([[theta_err, theta_dot_err]]),
        return_particles=False,
    )

    return [
        {
            "type": "single_theta_dual_radio",
            "ds_fn": ds_fn,
            "theta_err": theta_err,
            "theta_dor_err": theta_dot_err,
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
        snapshots_per_session=-1,
    )
    # dual radio dual
    pf = PFXYDualRadio(ds=ds, full_p_fn=full_p_fn)
    traj_paired, _ = pf.trajectory(
        N=N,
        mean=np.array([[0, 0, 0, 0, 0]]),
        std=np.array([[0, 200, 200, 0.1, 0.1]]),
        return_particles=False,
        noise_std=np.array([[0, pos_err, pos_err, vel_err, vel_err]]),
    )
    return [
        {
            "type": "xy_dual_radio",
            "ds_fn": ds_fn,
            "vel_err": vel_err,
            "pos_err": pos_err,
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
        return parser

    parser = get_parser()
    args = parser.parse_args()

    jobs = []

    for ds_fn in args.datasets:
        for N in [128, 128 * 4, 128 * 8, 128 * 16]:
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dor_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_single_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dor_err,
                            },
                        )
                    )
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dor_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_dual_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dor_err,
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
    with Pool(20) as pool:  # cpu_count())  # cpu_count() // 4)
        results = list(tqdm.tqdm(pool.imap(runner, jobs), total=len(jobs)))
    pickle.dump(results, open("particle_filter_results.pkl", "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
