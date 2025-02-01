import argparse
import os
import shutil
import sys

import yaml

from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.scripts.zarr_utils import (
    compare_and_check,
    compare_and_copy,
    zarr_new_dataset,
    zarr_open_from_lmdb_store,
    zarr_shrink,
)


def zarr_fix(
    zarr_filename_in,
):
    prefix = zarr_filename_in.replace(".zarr", "")
    yaml_fn = f"{prefix}.yaml"
    config = yaml.dump(yaml.safe_load(open(yaml_fn, "r")))
    antenna_spacing_in_m = yaml.safe_load(open(yaml_fn, "r"))["receivers"][0][
        "antenna-spacing-m"
    ]
    original_zarr = zarr_open_from_lmdb_store(
        zarr_filename_in, readahead=True, mode="rw"
    )
    if (
        original_zarr.receivers.r0.rx_spacing[0] != antenna_spacing_in_m
        or original_zarr.receivers.r1.rx_spacing[0] != antenna_spacing_in_m
    ):
        print(
            f"{zarr_filename_in} {antenna_spacing_in_m} {original_zarr.receivers.r0.rx_spacing[0]} {original_zarr.receivers.r1.rx_spacing[0]}"
        )
        original_zarr.receivers.r0.rx_spacing[:] = antenna_spacing_in_m
        original_zarr.receivers.r1.rx_spacing[:] = antenna_spacing_in_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_filename_in", type=str, help="input zarr")
    args = parser.parse_args()

    return_code = zarr_fix(
        args.zarr_filename_in,
    )
    sys.exit(return_code)
