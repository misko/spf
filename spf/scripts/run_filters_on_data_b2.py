#!/usr/bin/env python3

import argparse
import logging
import tempfile
from dotenv import load_dotenv  # Project Must install Python Package:  python-dotenv
import os

from spf.s3_utils import (
    b2_download_folder_cache,
    b2_file_to_local_with_cache,
    b2_get_or_set_cache,
    b2_reset_cache,
    b2path_to_bucket_and_path,
    get_b2_client,
)
from spf.scripts.run_filters_on_data import generate_configs_to_run, run_filter_jobs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        required=False,
    )
    parser.add_argument(
        "--empirical-pkl-fn",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        required=False,
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_dir",
        required=False,
    )

    return parser


def main():
    load_dotenv()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = get_parser()
    args = parser.parse_args()

    b2_client = get_b2_client()

    bucket = "projectspf"

    # first need to download the model and get checksums

    with tempfile.TemporaryDirectory() as tmpdirname:

        if args.empirical_pkl_fn[:3] == "b2:":
            # download to local
            local_empirical_pkl_fn = f"{tmpdirname}/empirical_dist.pkl"
            _, b2_path = b2path_to_bucket_and_path(args.empirical_pkl_fn)
            # print(bucket, b2_path)
            b2_client.download_file(bucket, b2_path, local_empirical_pkl_fn)
        else:
            local_empirical_pkl_fn = args.empirical_pkl_fn

        # download inference cache /mnt/md2$ ls cache/inference/dec18_mission1_rover1.zarr/3.500/dc0661eb09c048996e81545363ff8e33/d1655af080f3721a7e4852221955950e.npz
        prefix = "md2/cache/nosig_data"

        # Use the *custom* client for listing and downloading:
        resp = b2_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            b2_reset_cache()
            filename = obj["Key"]
            if ".yaml" in filename:
                remote_zarr_fn = filename.replace(".yaml", ".zarr")

                _ = b2_file_to_local_with_cache(f'b2://{bucket}/{obj["Key"]}')
                local_zarr_fn = b2_download_folder_cache(
                    f"b2://{bucket}/{remote_zarr_fn}"
                )

                jobs = generate_configs_to_run(
                    yaml_config_fn=args.config,
                    work_dir=args.work_dir,
                    dataset_fns=[local_zarr_fn],
                    seed=args.seed,
                    empirical_pkl_fn=local_empirical_pkl_fn,
                )

                run_filter_jobs(jobs, nparallel=args.parallel, debug=args.debug)


if __name__ == "__main__":
    main()
