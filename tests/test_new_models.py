import collections.abc
import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch
import yaml

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.scripts.create_empirical_p_dist import (
    create_empirical_p_dist,
    get_empirical_p_dist_parser,
)
from spf.scripts.train_single_point import (
    get_parser_filter,
    load_config_from_fn,
    train_single_point,
)


@pytest.fixture
def perfect_circle_dataset_n33():
    n = 7
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.0)
        yield tmpdirname, fn


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_base_config(updates, output_fn):
    base_yaml_fn = pathlib.Path(__file__).parent / "model_configs/test_single_net.yaml"
    base_config = load_config_from_fn(str(base_yaml_fn))
    merged_config = merge_dictionary(base_config, updates)
    with open(output_fn, "w") as f:
        yaml.dump(merged_config, f)


# def test_simple(perfect_circle_dataset_n33):
#     root_dir, zarr_fn = perfect_circle_dataset_n33

#     datasets = [f"{zarr_fn}.zarr"]
#     parser = get_empirical_p_dist_parser()
#     args = parser.parse_args(["-d", ""])
#     create_empirical_p_dist(args)

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         input_yaml_fn = tmpdirname + f"/input.yaml"
#         update_base_config(
#             {
#                 "datasets": {
#                     "train_paths": [f"{zarr_fn}.zarr"],
#                     "precompute_cache": root_dir,
#                     "train_on_val": True,
#                 }
#             },
#             input_yaml_fn,
#         )

#         # save_prefix = f"{root_dir}/test_simple_filter_save"
#         # base_args = []
#         # chkpnt_fn = save_prefix + "_step0.chkpnt"
#         parser = get_parser_filter()
#         args = parser.parse_args(["-c", input_yaml_fn, "--debug"])
#         train_single_point(args)
