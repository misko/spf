import collections
import collections.abc
import pathlib
import tempfile

import pytest
import yaml

from spf.dataset.fake_dataset import (
    create_empirical_dist_for_datasets,
    create_fake_dataset,
    fake_yaml_v4,
    fake_yaml_v5,
)
from spf.dataset.spf_dataset import v5spfdataset
from spf.scripts.train_single_point import (
    get_parser_filter,
    load_config_from_fn,
    train_single_point,
)


def update_config(input_fn, updates, output_fn):
    base_config = load_config_from_fn(str(input_fn))
    merged_config = merge_dictionary(base_config, updates)
    with open(output_fn, "w") as f:
        yaml.dump(merged_config, f)


@pytest.fixture(scope="session")
def noise1_n128_obits2():
    with tempfile.TemporaryDirectory() as tmpdirname:
        n = 128
        nthetas = 65
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p3"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.3, orbits=2
        )

        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=nthetas,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=nthetas
        )
        yield tmpdirname, empirical_pkl_fn, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n1025_orbits4_noise0p3():
    n = 1025
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p3"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.3, orbits=4
        )
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=65
        )

        yield tmpdirname, empirical_pkl_fn, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n33():
    n = 33
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_n50_0p01():
    n = 50
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=50, noise=0.01)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_n50_0p01_v4():
    n = 50
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v4, n=50, noise=0.01)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
            v4=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=5, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0p001():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p001"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=5)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n7_with_empirical():
    n = 7
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=7
        )

        yield tmpdirname, empirical_pkl_fn, fn


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


@pytest.fixture(scope="session")
def single_net_config():
    return str(pathlib.Path(__file__).parent / "model_configs/test_single_net.yaml")


@pytest.fixture(scope="session")
def paired_net_config():
    return str(pathlib.Path(__file__).parent / "model_configs/test_paired_net.yaml")


@pytest.fixture(scope="session")
def single_net_checkpoint(perfect_circle_dataset_n7_with_empirical, single_net_config):
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
        yield single_checkpoints_dir


@pytest.fixture(scope="session")
def paired_net_checkpoint_using_single_checkpoint(
    perfect_circle_dataset_n7_with_empirical, paired_net_config, single_net_checkpoint
):
    root_dir, empirical_pkl_fn, zarr_fn = perfect_circle_dataset_n7_with_empirical
    with tempfile.TemporaryDirectory() as tmpdirname:
        # dump a paired raido checkpoint
        single_checkpoints_dir = single_net_checkpoint
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
        yield paired_checkpoints_dir
