import argparse
import concurrent
import pickle

import numpy as np
import torch
from tqdm import tqdm

# from spf.dataset.spf_dataset import v5spfdataset
from spf.utils import zarr_open_from_lmdb_store


def process_yarr(yarr_fn):
    precomputed_zarr = zarr_open_from_lmdb_store(yarr_fn, mode="rw", map_size=2**32)

    precomputed_pkl = pickle.load(open(yarr_fn.replace(".yarr", ".pkl"), "rb"))

    for r_idx in range(2):
        mean_phase = np.hstack(
            [
                (
                    torch.tensor(
                        [x["mean"] for x in result["simple_segmentation"]]
                    ).mean()
                    if len(result) > 0
                    else torch.tensor(float("nan"))
                )
                for result in precomputed_pkl["segmentation_by_receiver"][f"r{r_idx}"]
            ]
        )
        # TODO THIS SHOULD BE FIXED!!!
        mean_phase[~np.isfinite(mean_phase)] = 0
        # diff_std = nanstd(precomputed_zarr[f"r{r_idx}"]["mean_phase"][:] - mean_phase)
        # print(r_idx, diff_std)
        precomputed_zarr[f"r{r_idx}"]["mean_phase"][:] = mean_phase
    precomputed_zarr["version"][:] = 3.31


def nanstd(x):
    return x[np.isfinite(x)].std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yarrs", type=str, help="yarrs", nargs="+", required=True
    )
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-w", "--workers", type=int, default=16, help="n workers")

    args = parser.parse_args()

    if args.debug:
        list(map(process_yarr, args.yarrs))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            list(tqdm(executor.map(process_yarr, args.yarrs), total=len(args.yarrs)))
