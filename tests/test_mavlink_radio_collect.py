import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import yaml

import spf
from spf.dataset.v4_data import v4rx_f64_keys
from spf.dataset.v6_data import v6rx_2x_keys, v6rx_scalar_keys
from spf.capture_schema import normalize_capture_config
from spf.mavlink_radio_collection import validate_transport_schema
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

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


def test_mavlink_radio_collect_v6_iio_compatibility():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f"{root_dir}/tests/test_config.yaml") as source:
            config = yaml.safe_load(source)
        config["data-version"] = 6
        config["n-records-per-receiver"] = 2
        config_path = f"{tmpdirname}/config_v6.yaml"
        with open(config_path, "w") as destination:
            yaml.safe_dump(config, destination)

        subprocess.check_output(
            f"python3 {root_dir}/spf/mavlink_radio_collection.py --fake-drone --exit -c "
            + f"{config_path} -m {root_dir}/tests/test_device_mapping "
            + f"-r center -n 2 --temp {tmpdirname}",
            timeout=180,
            shell=True,
            env=get_env(),
            stderr=subprocess.STDOUT,
        ).decode()

        zarr_fn = glob.glob(f"{tmpdirname}/*.zarr")[0]
        z = zarr_open_from_lmdb_store(zarr_fn)
        for receiver_idx in range(2):
            receiver = z[f"receivers/r{receiver_idx}"]
            assert receiver.signal_matrix.shape == (2, 2, 4096)
            for key in v6rx_scalar_keys:
                assert key in receiver
            for key in v6rx_2x_keys:
                assert key in receiver
            assert not receiver.gain_metadata_valid[:].any()
            np.testing.assert_array_equal(
                receiver.gain_index_start[:],
                np.full((2, 2), 0xFF, dtype=np.uint8),
            )


def test_direct_usb_refuses_legacy_dataset_schema():
    with pytest.raises(ValueError, match="requires data-version: 6"):
        validate_transport_schema(
            {
                "data-version": 4,
                "receivers": [{"rx-transport": "direct_usb"}],
            }
        )


def test_direct_usb_v2_accepts_compatibility_and_full_metadata_schemas():
    for data_version in (4, 7):
        validate_transport_schema(
            {
                "data-version": data_version,
                "receivers": [
                    {
                        "rx-transport": "direct_usb",
                        "direct-usb": {"protocol-version": 2},
                    }
                ],
            }
        )


def test_v7_infers_direct_usb_protocol_v2_and_rejects_conflicts():
    normalized = normalize_capture_config(
        {
            "data-version": 7,
            "receivers": [{"receiver-port": 1}],
        }
    )
    receiver = normalized["receivers"][0]
    assert receiver["rx-transport"] == "direct_usb"
    assert receiver["direct-usb"] == {
        "protocol-version": 2,
        "require-gain-metadata": True,
        "frame-count-per-request": 1,
    }
    validate_transport_schema(normalized)

    with pytest.raises(ValueError, match="requires rx-transport: direct_usb"):
        normalize_capture_config(
            {
                "data-version": 7,
                "receivers": [{"rx-transport": "iio"}],
            }
        )


def test_mavlink_radio_collect_v7_requires_boot_verified_firmware():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f"{root_dir}/tests/test_config.yaml") as source:
            config = yaml.safe_load(source)
        config["data-version"] = 7
        config["n-records-per-receiver"] = 2
        config_path = f"{tmpdirname}/config_v7.yaml"
        with open(config_path, "w") as destination:
            yaml.safe_dump(config, destination)

        with pytest.raises(subprocess.CalledProcessError) as raised:
            subprocess.check_output(
                f"python3 {root_dir}/spf/mavlink_radio_collection.py "
                "--fake-drone --exit -c "
                + f"{config_path} -m {root_dir}/tests/test_device_mapping "
                + f"-r center -n 2 --temp {tmpdirname}",
                timeout=180,
                shell=True,
                env=get_env(),
                stderr=subprocess.STDOUT,
            )

        assert b"V7 capture requires boot-verified firmware" in raised.value.output


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
