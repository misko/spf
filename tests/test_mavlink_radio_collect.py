import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

import spf
from spf.dataset.v4_data import v4rx_f64_keys
from spf.mavlink.mavlink_controller import (
    get_mavlink_controller_parser,
    mavlink_controller_run,
)
from spf.utils import zarr_open_from_lmdb_store

root_dir = os.path.dirname(os.path.dirname(spf.__file__))


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def test_mavlink_radio_collect():
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.check_output(
            f"python3 {root_dir}/spf/mavlink_radio_collection.py --fake-drone --exit -c "
            + f"{root_dir}/tests/test_config.yaml -m {root_dir}/tests/test_device_mapping "
            + f"-r center -n 50 --temp {tmpdirname}",
            timeout=180,
            shell=True,
            env=get_env(),
            stderr=subprocess.STDOUT,
        ).decode()

        # load output and make sure entries are not obiously wrong
        zarr_fn = glob.glob(f"{tmpdirname}/*.zarr")[0]
        z = zarr_open_from_lmdb_store(zarr_fn)
        keys_with_nans = []
        for key in v4rx_f64_keys:
            if not np.isfinite(z["receivers/r0"][key]).all():
                keys_with_nans.append(key)
        assert len(keys_with_nans) == 0


# def test_mavlink_radio_collect_direct():
#     parser = get_mavlink_controller_parser()
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         args = parser.parse_args(
#             args=[
#                 "--fake-drone",
#                 "--exit",
#                 "-c",
#                 f"{root_dir}/tests/test_config.yaml",
#                 "-m",
#                 "{root_dir}/tests/test_device_mapping",
#                 "-r",
#                 "center",
#                 "-n",
#                 "50",
#                 "--temp",
#                 tmpdirname,
#             ]
#         )
#         mavlink_controller_run(args)
