import collections.abc
import os
import pathlib
import tempfile

import pytest
import yaml

from spf.scripts.train_single_point import (
    get_parser_filter,
    load_config_from_fn,
    train_single_point,
)


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


@pytest.fixture
def single_net_config():
    return str(pathlib.Path(__file__).parent / "model_configs/test_single_net.yaml")


@pytest.fixture
def paired_net_config():
    return str(pathlib.Path(__file__).parent / "model_configs/test_paired_net.yaml")


def update_config(input_fn, updates, output_fn):
    base_config = load_config_from_fn(str(input_fn))
    merged_config = merge_dictionary(base_config, updates)
    with open(output_fn, "w") as f:
        yaml.dump(merged_config, f)


def test_simple(
    perfect_circle_dataset_n7_with_empirical, single_net_config, paired_net_config
):
    root_dir, empirical_pkl_fn, zarr_fn = perfect_circle_dataset_n7_with_empirical

    with tempfile.TemporaryDirectory() as tmpdirname:
        input_yaml_fn = tmpdirname + f"/input.yaml"
        update_config(
            single_net_config,
            {
                "datasets": {
                    "train_paths": [f"{zarr_fn}.zarr"],
                    "precompute_cache": root_dir,
                    "train_on_val": True,
                    "empirical_data_fn": empirical_pkl_fn,
                }
            },
            input_yaml_fn,
        )

        # dump a single radio checkpoint
        single_checkpoints_dir = f"{tmpdirname}/single_checkpoints"
        parser = get_parser_filter()
        args = parser.parse_args(
            ["-c", input_yaml_fn, "--debug", "--output", single_checkpoints_dir]
        )
        train_single_point(args)
        assert os.path.exists(f"{single_checkpoints_dir}/best.pth")
        assert os.path.exists(f"{single_checkpoints_dir}/checkpoint_e1_s10.pth")

        # dump a paired raido checkpoint
        input_yaml_fn = tmpdirname + f"/input.yaml"
        update_config(
            paired_net_config,
            {
                "datasets": {
                    "train_paths": [f"{zarr_fn}.zarr"],
                    "precompute_cache": root_dir,
                    "train_on_val": True,
                    "empirical_data_fn": empirical_pkl_fn,
                },
                "optim": {"checkpoint": f"{single_checkpoints_dir}/best.pth"},
            },
            input_yaml_fn,
        )
        paired_checkpoints_dir = f"{tmpdirname}/paired_checkpoints"
        parser = get_parser_filter()
        args = parser.parse_args(
            ["-c", input_yaml_fn, "--debug", "--output", paired_checkpoints_dir]
        )
        train_single_point(args)
        assert os.path.exists(f"{paired_checkpoints_dir}/best.pth")
        assert os.path.exists(f"{paired_checkpoints_dir}/checkpoint_e1_s10.pth")
