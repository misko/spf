import argparse
import math
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from spf.dataset.spf_dataset import v5spfdataset
from spf.utils import rx_spacing_to_str


def get_heatmap_for_radio(dss, radio_idx, bins):
    ground_truth_thetas = np.hstack([ds.ground_truth_thetas[radio_idx] for ds in dss])
    mean_phase = np.hstack([ds.mean_phase[f"r{radio_idx}"] for ds in dss])
    return np.histogram2d(
        ground_truth_thetas, mean_phase, bins=bins
    )  # heatmap, xedges, yedges


def get_heatmap(dss, bins=50):
    heatmaps = []
    for ridx in [0, 1]:
        heatmaps.append(get_heatmap_for_radio(dss, ridx, bins=bins)[0])
    return (heatmaps[0].copy() + heatmaps[1].copy()) / 2


def create_heatmaps_and_plot(dss, bins, save_fig_to=None):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
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
        r0 = r0 / (r0.sum(axis=0, keepdims=True) + eps)
        r1 = r1 / (r1.sum(axis=0, keepdims=True) + eps)
        r = r / (r.sum(axis=0, keepdims=True) + eps)
        heatmaps["r0"]["sym" if symmetry else "nosym"] = torch.tensor(r0.T)
        heatmaps["r1"]["sym" if symmetry else "nosym"] = torch.tensor(r1.T)
        heatmaps["r"]["sym" if symmetry else "nosym"] = torch.tensor(r.T)
        # write maps in map[phi][theta] = pr(theta | phi)
        # axs[2 + 3 * ridx].imshow(heatmap.T, extent=extent, origin="lower")
        axs[row_idx, 0].imshow(r0.T, extent=extent)
        axs[row_idx, 0].set_title(f"Radio0,sym={symmetry}")
        axs[row_idx, 1].imshow(r1.T, extent=extent)
        axs[row_idx, 1].set_title(f"Radio1,sym={symmetry}")
        axs[row_idx, 2].imshow(r.T, extent=extent)
        axs[row_idx, 2].set_title(f"Radio0+1,sym={symmetry}")
        row_idx += 1
    if save_fig_to is not None:
        fig.savefig(save_fig_to)
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

    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=args.precompute_cache,
            nthetas=args.nthetas,
            skip_fields=set(["signal_matrix"]),
            paired=False,
            ignore_qc=True,
            gpu=args.device == "cuda",
        )
        for prefix in args.datasets
    ]

    datasets_by_spacing = {}

    for dataset in datasets:
        rx_wavelength_spacing = dataset.cached_keys[0]["rx_wavelength_spacing"][
            0
        ].item()
        print("CREATE", rx_wavelength_spacing)
        assert (
            dataset.cached_keys[0]["rx_wavelength_spacing"] == rx_wavelength_spacing
        ).all()
        assert (
            dataset.cached_keys[1]["rx_wavelength_spacing"] == rx_wavelength_spacing
        ).all()
        rx_spacing_str = rx_spacing_to_str(rx_wavelength_spacing)
        if rx_spacing_str not in datasets_by_spacing:
            datasets_by_spacing[rx_spacing_str] = []
        datasets_by_spacing[rx_spacing_str].append(dataset)

    print("Found spacings:", datasets_by_spacing.keys())

    heatmaps = {}
    for rx_spacing_str, _datasets in datasets_by_spacing.items():
        heatmaps[rx_spacing_str] = create_heatmaps_and_plot(
            _datasets,
            args.nbins,
            save_fig_to=f"{args.output_fig_prefix}_rxwavelengthspacing{rx_spacing_str}_nbins{args.nbins}.png",
        )

    pickle.dump(heatmaps, open(args.out, "wb"))


if __name__ == "__main__":

    parser = get_empirical_p_dist_parser()
    args = parser.parse_args()
    create_empirical_p_dist(args)
