import os
import subprocess
import sys
import tempfile

import spf

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
