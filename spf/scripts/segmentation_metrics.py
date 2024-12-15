import argparse
import glob
import logging
from multiprocessing import Pool

import torch
from tqdm import tqdm

from spf.dataset.spf_dataset import v5spfdataset, v5spfdataset_manager
from spf.rf import torch_pi_norm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-zarrs", type=str, nargs="+", help="input zarr", required=True
    )
    parser.add_argument(
        "-c", "--precompute-cache", type=str, help="precompute cache", required=True
    )
    parser.add_argument(
        "--segmentation-version",
        type=float,
        help="segmentation version",
        required=True,
    )
    parser.add_argument(
        "-p", "--parallel", type=int, default=8, help="precompute cache"
    )
    return parser


def ds_to_metrics(args):
    try:
        with v5spfdataset_manager(
            args["ds_fn"],
            nthetas=65,
            ignore_qc=True,
            precompute_cache=args["precompute_cache"],
            snapshots_per_session=1,
            skip_fields=["signal_matrix"],
            paired=True,
            segmentation_version=args["segmentation_version"],
        ) as ds:
            diffs = torch_pi_norm(
                ds.ground_truth_phis
                - torch.vstack([ds.mean_phase["r0"], ds.mean_phase["r1"]])
            )
            mask = diffs.isfinite()
            return (
                ds.yaml_config["routine"],
                ds.carrier_frequencies[0],
                torch.as_tensor(
                    [
                        diffs[mask].std(),
                        mask.to(torch.float).mean(),
                        ds.mean_phase["r0"].shape[0],
                    ]
                ),
            )
    except Exception as e:
        logging.error(f"Failed to load... {args['ds_fn']} with exception: {e}")
        return None, None, None


if __name__ == "__main__":
    args = get_parser().parse_args()
    if len(args.input_zarrs) == 1 and args.input_zarrs[0][-4:] == ".txt":
        args.input_zarrs = [line.strip() for line in open(args.input_zarrs[0], "r")]

    jobs = [
        {
            "ds_fn": fn,
            "segmentation_version": args.segmentation_version,
            "precompute_cache": args.precompute_cache,
        }
        for fn in args.input_zarrs
    ]

    with Pool(8) as p:
        metrics_list = list(tqdm(p.imap(ds_to_metrics, jobs), total=len(jobs)))

    results = {}
    for routine, frequency, metrics in metrics_list:
        if metrics is not None:
            if frequency not in results:
                results[frequency] = {}
            if routine not in results[frequency]:
                results[frequency][routine] = []
            results[frequency][routine].append(metrics)
    for frequency in results:
        for routine in results[frequency]:
            metrics = torch.vstack(results[frequency][routine])
            std = ((metrics[:, 0] * metrics[:, 2]) / metrics[:, 2].sum()).sum()
            notnan = ((metrics[:, 1] * metrics[:, 2]) / metrics[:, 2].sum()).sum()
            results[frequency][routine] = {"std": std, "notnan": notnan}

    print(f"freq\troutine\tstd\tnotnan")
    for frequency in results:
        for routine in results[frequency]:
            print(
                f"{frequency}\t{routine}\t{results[frequency][routine]['std']}\t{results[frequency][routine]['notnan']}"
            )
