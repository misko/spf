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
                out_of_bounds = 0
                lim = 300000
                for k in ("tx_pos_x_mm", "tx_pos_y_mm", "rx_pos_x_mm", "rx_pos_y_mm"):
                    _out_of_bounds = (
                        np.abs(z["receivers"][f"r{ridx}"][k][:]) > lim
                    ).sum()
                    out_of_bounds += _out_of_bounds
                if out_of_bounds > 10:
                    print(f"{zarr_fn},r{ridx},{out_of_bounds}")
            except KeyError as e:
                print(f"{zarr_fn}: failed {e}")
