import argparse

import numpy as np

from spf.gps.boundaries import find_closest_boundary_with_distance
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-zarrs", nargs="+", type=str, help="input zarr", required=True
    )
    args = parser.parse_args()

    for zarr_fn in args.input_zarrs:
        z = zarr_open_from_lmdb_store(
            zarr_fn,
            mode="r",
        )

        for ridx in [0, 1]:
            try:
                lats = z["receivers"][f"r{ridx}"]["gps_lat"][:]
                longs = z["receivers"][f"r{ridx}"]["gps_long"][:]
                zero_counts = 0
                out_of_bounds_counts = 0
                total = lats.shape[0]
                for idx in range(total):
                    if np.isclose(longs[idx], 0.0) or np.isclose(lats[idx], 0.0):
                        zero_counts += 1
                    else:
                        d, boundary = find_closest_boundary_with_distance(
                            [longs[idx], lats[idx]]
                        )
                        if d > 200:
                            out_of_bounds_counts += 1
                print(f"{zarr_fn},r{ridx},{zero_counts},{out_of_bounds_counts}")
            except KeyError as e:
                print(f"{zarr_fn}: failed {e}")
