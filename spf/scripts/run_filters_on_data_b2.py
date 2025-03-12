#!/usr/bin/env python3

import argparse
import hashlib
import logging
import os
import random
import tempfile

import boto3
from boto3.dynamodb.conditions import Attr
from dotenv import load_dotenv  # Project Must install Python Package:  python-dotenv

from spf.s3_utils import (
    b2_download_folder_cache,
    b2_file_as_local,
    b2_file_to_local_with_cache,
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
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
    )

    return parser


def get_all_items_by_bucket_scan_full_name(prefix):
    return get_all_items_by_bucket_scan(prefix, "full_name")


def get_all_items_by_bucket_scan(prefix, projection_expression=None):
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table("filter_metrics")

    if projection_expression == None:
        response = table.scan(
            FilterExpression=Attr("bucket").eq(prefix),
        )
    else:
        response = table.scan(
            FilterExpression=Attr("bucket").eq(prefix),
            ProjectionExpression=projection_expression,
        )

    items = response.get("Items", [])

    n = 50000
    k = 0
    while "LastEvaluatedKey" in response:
        if projection_expression == None:
            response = table.scan(
                FilterExpression=Attr("bucket").eq(prefix),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
        else:
            response = table.scan(
                FilterExpression=Attr("bucket").eq(prefix),
                ProjectionExpression=projection_expression,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
        items.extend(response.get("Items", []))
        if len(items) // n != k:
            print(len(items))
            k = len(items) // n

    return items


def get_chunk_index(filename: str, total_chunks: int) -> int:
    """
    Take the MD5 hash of the filename, interpret it as a large integer,
    and mod by total_chunks to get a chunk index.
    """
    # Encode the filename to bytes
    encoded = filename.encode("utf-8")
    # Get the hexadecimal MD5 digest
    md5_hex = hashlib.md5(encoded).hexdigest()
    # Convert hex digest to an integer
    hash_int = int(md5_hex, 16)
    # Take mod of total_chunks to pick which chunk/subjob handles this file
    return hash_int % total_chunks


def main():
    load_dotenv()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = get_parser()
    args = parser.parse_args()

    chunk_idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
    chunks_total = int(os.environ.get("AWS_BATCH_JOB_ARRAY_SIZE", "1"))

    files_to_process = set()
    with b2_file_as_local(args.manifest, "r") as f:
        files_to_process = set(
            [os.path.basename(line.strip()).replace(".zarr", ".yaml") for line in f]
        )

    b2_client = get_b2_client()

    bucket = "projectspf"

    assert args.work_dir[-1] != "/"

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

        already_processed = set(
            [
                item["full_name"]
                for item in get_all_items_by_bucket_scan_full_name(args.work_dir)
            ]
        )
        # breakpoint()

        files_on_remote = resp.get("Contents", [])
        random.shuffle(files_on_remote)
        for obj in files_on_remote:
            # b2_reset_cache()
            filename = obj["Key"]

            if (
                get_chunk_index(filename=filename, total_chunks=chunks_total)
                != chunk_idx
            ):
                continue
            if os.path.basename(filename) not in files_to_process:
                print("Skipping", filename)
                continue
            if ".yaml" in filename:  #  and "_tag_" in filename:
                # print(filename)
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

                try:
                    run_filter_jobs(
                        jobs,
                        nparallel=args.parallel,
                        debug=args.debug,
                        already_processed=set(
                            [
                                x
                                for x in already_processed
                                if os.path.basename(local_zarr_fn) in x
                            ]
                        ),
                    )
                except Exception as e:
                    print("Failed to process", str(e))


if __name__ == "__main__":
    main()
