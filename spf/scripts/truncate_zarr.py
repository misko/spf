import argparse
import sys

from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys
from spf.scripts.zarr_utils import truncate_zarr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_filename_in", type=str, help="input zarr")
    args = parser.parse_args()

    # this only works for version 4

    return_code = truncate_zarr(
        args.zarr_filename_in, f64_keys=v4rx_f64_keys, f64x2_keys=v4rx_2xf64_keys
    )
    sys.exit(return_code)
