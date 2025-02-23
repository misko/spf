import argparse
import datetime
import logging
import os

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    z = zarr_open_from_lmdb_store(args.filename, readahead=True, mode="r")
    times = z["receivers/r0/gps_timestamp"][:]
    first_time = times[np.where(times > 0)[0]].min()
    datetime_object = datetime.datetime.fromtimestamp(
        first_time, tz=datetime.timezone.utc
    )
    datetime_string = datetime_object.strftime("%Y_%m_%d_%H_%M_%S")
    print(datetime_string)
