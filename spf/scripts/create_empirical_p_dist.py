import argparse
import logging
import math
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from spf.dataset.spf_dataset import v5spfdataset
from spf.utils import rx_spacing_to_str


def get_heatmap_for_radio(dss, radio_idx, bins):
    ground_truth_thetas = np.hstack([ds.ground_truth_thetas[radio_idx] for ds in dss])
    mean_phase = np.hstack([ds.mean_phase[f"r{radio_idx}"] for ds in dss])
    mask = np.isfinite(mean_phase)
    return np.histogram2d(
        ground_truth_thetas[mask], mean_phase[mask], bins=bins
    )  # heatmap, xedges, yedges


def get_heatmap(dss, bins=50):
    heatmaps = []
    for ridx in [0, 1]:
        heatmaps.append(get_heatmap_for_radio(dss, ridx, bins=bins)[0])
    return (heatmaps[0].copy() + heatmaps[1].copy()) / 2


def create_heatmaps_and_plot(dss, bins, save_fig_to_prefix=None):
    # theta norm is where if you sum over all phi for a specific theta
    # you get back 1.0
    fig_theta_norm, axs_theta_norm = plt.subplots(2, 3, figsize=(15, 10))
    fig_phi_norm, axs_phi_norm = plt.subplots(2, 3, figsize=(15, 10))
    row_idx = 0
    heatmaps = {"r0": {}, "r1": {}, "r": {}}
    eps = 1e-10
    for symmetry in [False, True]:
        r0, _, _ = get_heatmap_for_radio(dss, 0, bins)
        r1, _, _ = get_heatmap_for_radio(dss, 1, bins)
        r = (r0 + r1) / 2
        if symmetry:
            r0 = apply_symmetry_rules_to_heatmap(r0)
            r1 = apply_symmetry_rules_to_heatmap(r1)
            r = apply_symmetry_rules_to_heatmap(r)
        extent = [-torch.pi, torch.pi, -torch.pi, torch.pi]
        # r0,r1,r are matricies of format m[theta][phi]
        # normalizing by dividing by sum of axis=0 (theta)
        # results in r[:,0].sum()==1
        # then taking transpose so r[0].sum()==1 and r[phi][theta]
        r0_phi_norm = (r0 / (r0.sum(axis=0, keepdims=True) + eps)).T
        r1_phi_norm = (r1 / (r1.sum(axis=0, keepdims=True) + eps)).T
        r_phi_norm = (r / (r.sum(axis=0, keepdims=True) + eps)).T

        heatmaps["r0"]["sym" if symmetry else "nosym"] = torch.tensor(r0_phi_norm)
        heatmaps["r1"]["sym" if symmetry else "nosym"] = torch.tensor(r1_phi_norm)
        heatmaps["r"]["sym" if symmetry else "nosym"] = torch.tensor(r_phi_norm)

        # write maps in map[phi][theta] = pr(theta | phi)
        axs_phi_norm[row_idx, 0].imshow(r0_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 0].set_title(f"Radio0,sym={symmetry}")
        axs_phi_norm[row_idx, 1].imshow(r1_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 1].set_title(f"Radio1,sym={symmetry}")
        axs_phi_norm[row_idx, 2].imshow(r_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 2].set_title(f"Radio0+1,sym={symmetry}")
        for _x in range(3):
            axs_phi_norm[row_idx, _x].set_xlabel("Theta (gt)")
            axs_phi_norm[row_idx, _x].set_ylabel("Phase diff (obs)")

        # r0_theta_norm is such that
        # r[0].sum()==1 and r[theta][phi]
        r0_theta_norm = r0 / (r0.sum(axis=1, keepdims=True) + eps)
        r1_theta_norm = r1 / (r1.sum(axis=1, keepdims=True) + eps)
        r_theta_norm = r / (r.sum(axis=1, keepdims=True) + eps)

        # write maps in map[phi][theta] = pr(theta | phi)
        axs_theta_norm[row_idx, 0].imshow(r0_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 0].set_title(f"Radio0,sym={symmetry}")
        axs_theta_norm[row_idx, 1].imshow(r1_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 1].set_title(f"Radio1,sym={symmetry}")
        axs_theta_norm[row_idx, 2].imshow(r_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 2].set_title(f"Radio0+1,sym={symmetry}")
        for _x in range(3):
            axs_theta_norm[row_idx, _x].set_xlabel("Theta (gt)")
            axs_theta_norm[row_idx, _x].set_ylabel("Phase diff (obs)")

        row_idx += 1
    fig_phi_norm.suptitle(f"theta conditional on phi")
    fig_theta_norm.suptitle(f"phi conditional on theta")
    if save_fig_to_prefix is not None:
        fig_phi_norm.savefig(f"{save_fig_to_prefix}_phi_norm.png")
        fig_theta_norm.savefig(f"{save_fig_to_prefix}_theta_norm.png")
        plt.close(fig_phi_norm)
        plt.close(fig_theta_norm)
    return heatmaps


def apply_symmetry_rules_to_heatmap(h):
    bins = h.shape[0]
    # h[theta][phi]
    # half is restricting to positive y_rad, -theta -> theta
    # positive theta , phi is same as negative theta, - phi
    half = h[: math.ceil(bins / 2)] + np.flip(h[math.floor(bins // 2) :])
    # pi/2+epsilon is same as pi/2-epsilon
    half = half + np.flip(half, axis=0)
    full = np.vstack([half[:-1], np.flip(half)])
    return full  # / full.sum(axis=1, keepdims=True)


def get_empirical_p_dist_parser():
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
        "--out",
        type=str,
        default="empirical-dist.pkl",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--nthetas",
        type=int,
        default=65,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--precompute-cache",
        type=str,
        required=True,
    )
    parser.add_argument("--output-fig-prefix", type=str, required=False, default=None)
    return parser


def create_empirical_p_dist(args):
    if args.output_fig_prefix is not None:
        os.makedirs(os.path.dirname(args.output_fig_prefix), exist_ok=True)
    datasets = []

    for prefix in tqdm(args.datasets, total=len(args.datasets)):
        try:
            ds = v5spfdataset(
                prefix,
                precompute_cache=args.precompute_cache,
                nthetas=args.nthetas,
                skip_fields=set(["signal_matrix"]),
                paired=False,
                ignore_qc=True,
                gpu=args.device == "cuda",
            )
            datasets.append(ds)
        except ValueError as e:
            logging.error(f"Failed to load {prefix} with error {str(e)}")

    datasets_by_spacing = {}

    counts = {}

    for dataset in datasets:
        check0 = (
            (
                dataset.cached_keys[0]["rx_wavelength_spacing"]
                == dataset.cached_keys[0]["rx_wavelength_spacing"].median()
            )
            .to(torch.float)
            .mean()
        )
        check1 = (
            (
                dataset.cached_keys[1]["rx_wavelength_spacing"]
                == dataset.cached_keys[1]["rx_wavelength_spacing"].median()
            )
            .to(torch.float)
            .mean()
        )
        if check0 != 1.0 or check1 != 1.0:
            breakpoint()
            logging.warning(
                f"{dataset.zarr_fn} Failed consistentcy check for rx spacing! {check0} {check1}"
            )

        rx_spacing_str = rx_spacing_to_str(
            dataset.cached_keys[0]["rx_wavelength_spacing"].median()
        )
        assert rx_spacing_str == rx_spacing_to_str(
            dataset.cached_keys[1]["rx_wavelength_spacing"].median()
        )

        if rx_spacing_str not in counts:
            counts[rx_spacing_str] = {}
        rx_lo_and_spacing = dataset.get_spacing_identifier()
        if rx_lo_and_spacing not in counts[rx_spacing_str]:
            counts[rx_spacing_str][rx_lo_and_spacing] = 0
        counts[rx_spacing_str][rx_lo_and_spacing] += 1

        if rx_spacing_str not in datasets_by_spacing:
            datasets_by_spacing[rx_spacing_str] = []
        datasets_by_spacing[rx_spacing_str].append(dataset)

    print("Found spacings:", datasets_by_spacing.keys())
    for rx_spacing_str in counts:
        print(rx_spacing_str)
        for rx_lo_and_spacing, count in counts[rx_spacing_str].items():
            print("\t", rx_lo_and_spacing, count)

    heatmaps = {}
    for rx_spacing_str, _datasets in datasets_by_spacing.items():
        heatmaps[rx_spacing_str] = create_heatmaps_and_plot(
            _datasets,
            args.nbins,
            save_fig_to_prefix=f"{args.output_fig_prefix}_rxwavelengthspacing{rx_spacing_str}_nbins{args.nbins}",
        )

    pickle.dump(heatmaps, open(args.out, "wb"))


if __name__ == "__main__":

    parser = get_empirical_p_dist_parser()
    args = parser.parse_args()
    create_empirical_p_dist(args)
