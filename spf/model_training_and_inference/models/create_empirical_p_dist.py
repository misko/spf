import argparse

from spf.dataset.spf_dataset import v5spfdataset

import numpy as np
import pickle


def get_heatmap(dss, bins=50):
    heatmaps = []
    for ridx in [0, 1]:
        ground_truth_thetas = np.hstack([ds.ground_truth_thetas[ridx] for ds in dss])
        mean_phase = np.hstack([ds.mean_phase[f"r{ridx}"] for ds in dss])
        heatmap, _, _ = np.histogram2d(ground_truth_thetas, mean_phase, bins=bins)
        heatmaps.append(heatmap)
    return heatmaps[0].copy() + heatmaps[1].copy()


def apply_symmetry_rules_to_heatmap(h, bins=50):
    half = h[: bins // 2] + np.flip(h[bins // 2 :])
    half = half + np.flip(half, axis=0)
    full = np.vstack([half, np.flip(half)])
    return full / full.sum(axis=1, keepdims=True)


if __name__ == "__main__":
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
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="full_p.pkl",
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

    args = parser.parse_args()

    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=args.precompute_cache,
            nthetas=args.nthetas,
            skip_signal_matrix=True,
            paired=False,
            ignore_qc=True,
            gpu=args.device == "cuda",
        )
        for prefix in args.datasets
    ]

    heatmap = get_heatmap(datasets, bins=args.nbins)
    if args.symmetry:
        heatmap = apply_symmetry_rules_to_heatmap(heatmap)

    pickle.dump({"full_p": heatmap}, open(args.out, "wb"))
