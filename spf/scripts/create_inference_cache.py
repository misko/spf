import argparse
from functools import partial
from multiprocessing import Pool

import tqdm

from spf.model_training_and_inference.models.single_point_networks_inference import (
    get_inference_on_ds_noexceptions,
    get_nn_inference_on_ds_and_cache,
)

if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config-fn",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--checkpoint-fn",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--inference-cache",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--segmentation-version",
            type=float,
            required=True,
        )
        parser.add_argument("--parallel", type=int, required=False, default=16)

        parser.add_argument("--precompute-cache", type=str, required=True)
        parser.add_argument("--workers", type=int, required=False, default=0)
        parser.add_argument("--device", type=str, required=False, default="cuda")
        parser.add_argument(
            "-d",
            "--datasets",
            type=str,
            help="dataset prefixes",
            nargs="+",
            required=True,
        )
        parser.add_argument(
            "--debug",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        return parser

    parser = get_parser()
    args = parser.parse_args()

    run_fn = partial(
        (
            get_inference_on_ds_noexceptions
            if not args.debug
            else get_nn_inference_on_ds_and_cache
        ),
        config_fn=args.config_fn,
        checkpoint_fn=args.checkpoint_fn,
        device=args.device,
        inference_cache=args.inference_cache,
        batch_size=64,
        workers=0,
        precompute_cache=args.precompute_cache,
        crash_if_not_cached=False,
        segmentation_version=args.segmentation_version,
    )

    if len(args.datasets) == 1 and args.datasets[0][-4:] == ".txt":
        with open(args.datasets[0], "r") as f:
            args.datasets = [x.strip() for x in f]

    if args.debug:
        results = list(
            tqdm.tqdm(
                map(run_fn, args.datasets),
                total=len(args.datasets),
            )
        )
    else:
        # list(map(run_fn, args.datasets))
        with Pool(args.parallel) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(run_fn, args.datasets),
                    total=len(args.datasets),
                )
            )
            # list(pool.imap(run_fn, args.datasets))
