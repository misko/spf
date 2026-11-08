from spf.filters.filters import dual_radio_mse_theta_metrics
import torch
from spf.filters.filters import (
    ParticleFilter,
    add_noise,
    theta_phi_to_bins,
)
from spf.dataset.spf_dataset import v5spfdataset, segmentation_based_keys
from spf.rf import torch_pi_norm_pi
import torch


class PFSingleThetaDualRadioNN(ParticleFilter):
    def __init__(
        self,
        nn_ds,
    ):
        self.ds = nn_ds
        self.absolute = nn_ds.absolute
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def observation(self, idx):
        return self.ds.get_inference_for_idx(idx)[0]["paired"][0]

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std, generator=self.generator)

    def update(self, z):
        # z is not the raw observation, but the processed model output
        theta_bin = theta_phi_to_bins(self.particles[:, 0], nbins=z.shape[0])
        prob_theta_given_observation = torch.take(z, theta_bin)

        self.weights *= prob_theta_given_observation
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        return dual_radio_mse_theta_metrics(
            trajectory,
            (
                self.ds.craft_ground_truth_thetas
                if not self.absolute
                else self.ds.absolute_thetas.mean(axis=0)
            ),
        )

    def trajectory(self, **kwargs):
        trajectory = super().trajectory(**kwargs)
        for x in trajectory:
            x["craft_theta"] = x["mu"][0]
            x["P_theta"] = x["var"][0]
        return trajectory

import matplotlib.pyplot as plt


def plot_traj(ds, traj_paired, n=None):

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[1].axhline(y=torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(y=-torch.pi / 2, ls=":", c=(0.7, 0.7, 0.7))

    if n is None:
        n = len(traj_paired)
    n = min(n, len(ds))
    colors = ["blue", "orange"]
    for rx_idx in (0, 1):
        if ds.ds.realtime:
            mean_phase = []
            ground_truth_phi = []
            craft_ground_truth_thetas = []
            absolute_thetas = []
            for idx in range(n):
                d = ds[idx][rx_idx]
                mean_phase.append(d["mean_phase_segmentation"])
                ground_truth_phi.append(d["ground_truth_phi"])
                craft_ground_truth_thetas.append(d["craft_ground_truth_theta"])
                absolute_thetas.append(d["absolute_theta"])
            mean_phase = torch.tensor(mean_phase)
            ground_truth_phi = torch.tensor(ground_truth_phi)
            craft_ground_truth_thetas = torch.tensor(craft_ground_truth_thetas)
            absolute_thetas = torch.tensor(absolute_thetas)
        else:
            mean_phase = ds.mean_phase[f"r{rx_idx}"][:n]
            ground_truth_phi = ds.ground_truth_phis[rx_idx][:n]
            craft_ground_truth_thetas = torch_pi_norm_pi(ds.craft_ground_truth_thetas)[
                :n
            ]
            absolute_thetas = torch_pi_norm_pi(ds.absolute_thetas.mean(axis=0))[:n]

        print(mean_phase.shape)
        ax[0].scatter(
            range(min(n, mean_phase.shape[0])),
            mean_phase,
            label=f"r{rx_idx} estimated phi",
            s=1.0,
            # alpha=0.1,
            color=colors[rx_idx],
        )
        ax[0].plot(
            ground_truth_phi,
            color=colors[rx_idx],
            label=f"r{rx_idx} perfect phi",
            linestyle="dashed",
            alpha=0.5,
        )

    ax[1].plot(
        craft_ground_truth_thetas,
        label="craft gt theta",
        linestyle="dashed",
    )
    ax[1].plot(
        absolute_thetas,
        label="absolute theta",
        linestyle="dashed",
    )

    xs = torch.hstack([x["mu"][0] for x in traj_paired])[:n]
    stds = torch.sqrt(torch.hstack([x["var"][0] for x in traj_paired]))[:n]

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

config_fn = "/home/mouse9911/gits/spf/checkpoints/march16/paired_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2/config.yml"
checkpoint_fn = "/home/mouse9911/gits/spf/checkpoints/march16/paired_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2/best.pth"

empirical_data_fn = "/home/mouse9911/gits/spf/empirical_dists/full.pkl"

precompute_cache = "/mnt/md2/cache/precompute_cache_3p5_chunk1/"
precompute_cache = "/mnt/md2/cache/precompute_cache_3p6/"
inference_cache = "/mnt/md2/cache/inference"

from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper

from spf.dataset.spf_dataset import v5spfdataset, segmentation_based_keys

ds = v5spfdataset(
    # "/mnt/md2/cache/nosig_data/wallarrayv3_2024_08_21_03_09_04_nRX2_rx_circle_spacing0p05075.zarr",
    "/mnt/md1/2d_wallarray_v2_data/aug/wallarrayv3_2024_08_21_03_09_04_nRX2_rx_circle_spacing0p05075.zarr",
    # "/mnt/md2/rpi5inference/rpi5inference_2025_04_09_15_36_19_nRX2_bounce_spacing0p035.zarr",
    nthetas=65,
    ignore_qc=True,
    precompute_cache=precompute_cache,
    empirical_data_fn=empirical_data_fn,
    paired=True,
    skip_fields=set(["simple_segmentations"]),  # ["signal_matrix"] + segmentation_based_keys),
    vehicle_type="wallarray",
    segmentation_version=3.6,
    realtime=False,
)
nn_ds = v5spfdataset_nn_wrapper(
    ds,
    config_fn,
    checkpoint_fn,
    inference_cache,
    device="cpu",
    v4=False,
    absolute=True,
)
pf = PFSingleThetaDualRadioNN(
    nn_ds,
)

theta_err = 0.075
theta_dot_err = 0.002

trajectory = pf.trajectory(
    mean=torch.tensor([[0, 0]]),
    std=torch.tensor([[20, 0.1]]),
    noise_std=torch.tensor([[theta_err, theta_dot_err]]),
    return_particles=False,
    debug=True,
    N=512 * 16,  # * 8,
)

from spf.dataset.spf_dataset import v5inferencedataset


v5inf = v5inferencedataset(
    yaml_fn=ds.zarr_fn.replace(".zarr", "") + ".yaml",
    nthetas=65,
    gpu=False,
    n_parallel=8,
    paired=True,
    model_config_fn="",
    # skip_fields=["signal_matrix"],
    vehicle_type="wallarray",
    skip_segmentation=False,
    skip_detrend=False,
)

n = len(ds)

for idx in range(n):
    d = ds[idx]
    print(idx)
    for ridx in range(2):
        v5inf.write_to_idx(idx, ridx, d[ridx])
