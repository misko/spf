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


def zarr_rechunk(
    zarr_filename_in,
    zarr_filename_out,
    skip_signal_matrix,
    check_contents_if_exists=True,
):

    prefix = zarr_filename_in.replace(".zarr", "")
    yaml_fn = f"{prefix}.yaml"
    config = yaml.dump(yaml.safe_load(open(yaml_fn, "r")))

    original_zarr = zarr_open_from_lmdb_store(
        zarr_filename_in, readahead=True, mode="r"
    )

    if os.path.exists(zarr_filename_out):
        if not check_contents_if_exists:
            return 0
        new_zarr = zarr_open_from_lmdb_store(
            zarr_filename_out, readahead=True, mode="r"
        )
        if not compare_and_check(
            "", original_zarr, new_zarr, skip_signal_matrix=skip_signal_matrix
        ):
            print(zarr_filename_out, "Output file exists and is not carbon copy!")
            return 1
        else:
            print(zarr_filename_out, "Output file exists and is carbon copy!")
            return 0

    os.makedirs(os.path.dirname(zarr_filename_out), exist_ok=True)

    timesteps = original_zarr["receivers/r0/system_timestamp"].shape[0]
    buffer_size = original_zarr["receivers/r0/signal_matrix"].shape[-1]
    n_receivers = 2
    keys_f64 = v5rx_f64_keys
    keys_2xf64 = v5rx_2xf64_keys
    chunk_size = 512

    shutil.copyfile(yaml_fn, zarr_filename_out.replace(".zarr", ".yaml"))
    new_zarr = zarr_new_dataset(
        zarr_filename_out,
        timesteps,
        buffer_size,
        n_receivers,
        keys_f64,
        keys_2xf64,
        original_zarr["config"],
        chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
        compressor=None,
        skip_signal_matrix=skip_signal_matrix,
    )
    new_zarr["config"][0] = config
    compare_and_copy(
        "",
        original_zarr,
        new_zarr,
        skip_signal_matrix=skip_signal_matrix,
    )
    if original_zarr["config"].shape == ():
        new_zarr["config"][0] = config
    new_zarr.store.close()
    new_zarr = None
    zarr_shrink(zarr_filename_out)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_filename_in", type=str, help="input zarr")
    parser.add_argument("zarr_filename_out", type=str, help="output zarr")
    parser.add_argument(
        "-s", "--skip-signal-matrix", action="store_true", help="skip signal matrix"
    )
    parser.add_argument(
        "--check-contents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    return_code = zarr_rechunk(
        args.zarr_filename_in,
        args.zarr_filename_out,
        skip_signal_matrix=args.skip_signal_matrix,
        check_contents_if_exists=args.check_contents,
    )
    sys.exit(return_code)
