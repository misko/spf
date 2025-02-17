import argparse
import multiprocessing
import os
import shutil
import sys

import yaml
from tqdm import tqdm

from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.scripts.zarr_utils import (
    compare_and_check,
    compare_and_copy,
    new_yarr_dataset,
    zarr_new_dataset,
    zarr_open_from_lmdb_store,
    zarr_shrink,
)


def yarr_3p5_to_3p6(
    yarr_filename_in,
    output_folder,
):

    original_yarr = zarr_open_from_lmdb_store(
        yarr_filename_in, readahead=True, mode="r"
    )

    yarr_filename_out = f"{output_folder}/{os.path.basename(yarr_filename_in)}"
    if os.path.exists(yarr_filename_out):
        print(yarr_filename_out, "Output file exists!")
        return 1

    os.makedirs(os.path.dirname(yarr_filename_out), exist_ok=True)
    new_yarr = new_yarr_dataset(
        filename=yarr_filename_out,
        n_receivers=2,
        all_windows_stats_shape=original_yarr.r0.all_windows_stats.shape,
        windowed_beamformer_shape=original_yarr.r0.windowed_beamformer.shape,
        weighted_beamformer_shape=original_yarr.r0.weighted_beamformer.shape,
        downsampled_segmentation_mask_shape=original_yarr.r0.downsampled_segmentation_mask.shape,
        mean_phase_shape=original_yarr.r0.mean_phase.shape,
        chunk_size=1,
    )

    compare_and_copy(
        "",
        original_yarr,
        new_yarr,
    )
    new_yarr.store.close()
    new_yarr = None
    zarr_shrink(yarr_filename_out)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs", type=str, help="input yarrs", nargs="+", required=True
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, help="output folder", required=True
    )
    args = parser.parse_args()

    tasks = [(input_filename, args.output_folder) for input_filename in args.inputs]
    with multiprocessing.Pool(processes=30) as pool:
        results = list(tqdm(pool.starmap(yarr_3p5_to_3p6, tasks), total=len(tasks)))
    # sys.exit(return_code)
