import os
import tempfile
import numpy as np
import pytest
import torch
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.model_training_and_inference.models.beamsegnet import (
    normal_correction_for_bounded_range,
)
from spf.notebooks.simple_train_filter import get_parser_filter, simple_train_filter
from spf.notebooks.simple_train import get_parser, simple_train


@pytest.fixture
def perfect_circle_dataset_n33():
    n = 33
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.0)
        yield tmpdirname, fn


def base_args():
    return [
        "--device",
        "cpu",
        "--seed",
        "0",
        "--nthetas",
        "65",
        "--batch",
        "128",
        "--workers",
        "0",
        "--act",
        "leaky",
        "--segmentation-level",
        "downsampled",
        "--seg-net",
        "conv",
        "--no-shuffle",
        "--skip-qc",
        "--no-sigmoid",
        "--val-on-train",
        "--segmentation-lambda",
        "0",
        "--independent",
    ]


def test_simple_filter_save_load(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    save_prefix = f"{root_dir}/test_simple_filter_save"
    base_args = [
        "-d",
        zarr_fn,
        "--precompute-cache",
        root_dir,
        "--act",
        "leaky",
        "--skip-qc",
        "--debug-model",
        "--save-every",
        "1",
        "--snapshots-per-session",
        "1",
        "--save-prefix",
        save_prefix,
    ]
    chkpnt_fn = save_prefix + "_step0.chkpnt"
    save_args = base_args + [
        "--save-prefix",
        save_prefix,
        "--steps",
        "1",
    ]
    load_args = base_args + [
        "--load-checkpoint",
        chkpnt_fn,
        "--steps",
        "3",
    ]

    # train and save
    args = get_parser_filter().parse_args(save_args)
    simple_train_filter(args)
    assert os.path.exists(chkpnt_fn)

    # load checkpoint
    args = get_parser_filter().parse_args(load_args)
    simple_train_filter(args)


def test_beamnet_downsampled():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn = f"{tmpdirname}/test_circle"
        create_fake_dataset(filename=ds_fn, yaml_config_str=fake_yaml, n=5)
        args_list = [
            "--device",
            "cpu",
            "--seed",
            "0",
            "--nthetas",
            "11",
            "--datasets",
            ds_fn,
            "--batch",
            "128",
            "--workers",
            "0",
            # "--batch-norm",
            "--act",
            "leaky",
            "--shuffle",
            "--segmentation-level",
            "downsampled",
            "--type",
            "direct",
            "--seg-net",
            "conv",
            "--epochs",
            "1200",
            # "--skip-segmentation",
            "--no-shuffle",
            "--symmetry",
            # "--sigmoid",
            "--no-sigmoid",
            "--block",
            # "--wandb-project",
            # "test123",
            "--plot-every",
            "50",
            "--lr",
            "0.0005",
            "--precompute-cache",
            tmpdirname,
        ]
        args = get_parser().parse_args(args_list)

        train_results = simple_train(args)
        assert np.array(train_results["losses"])[-10:].mean() < 0.2


def test_beamnet_direct_positional(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--positional",
        "--type",
        "direct",
        "--hidden",
        "16",  # "256",
        "--depth",
        "1",  # "6",
        "--min-sigma",
        "0.0",
        "--precompute-cache",
        root_dir,
        "--lr",
        "0.001",
        "--epochs",
        "300",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < -1.0


def test_beamnet_direct_positional_and_symmetry(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--positional",
        "--type",
        "direct",
        "--symmetry",
        "--hidden",
        "16",  # "256",
        "--min-sigma",
        "0.0",
        "--depth",
        "1",  # "6",
        "--precompute-cache",
        root_dir,
        "--lr",
        "0.001",
        "--epochs",
        "300",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < -1.0


def test_beamnet_direct(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--type",
        "direct",
        "--min-sigma",
        "0.0",
        "--hidden",
        "16",  # "256",
        "--depth",
        "1",  # "6",
        "--precompute-cache",
        root_dir,
        "--lr",
        "0.001",
        "--epochs",
        "300",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < -1.0


def test_beamnet_discrete(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--type",
        "discrete",
        "--precompute-cache",
        root_dir,
        "--batch-norm",  # important for discrete model
        "--hidden",
        "128",
        "--depth",
        "5",
        "--lr",
        "0.001",
        "--epochs",
        "75",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < 1.5


def test_beamnet_discrete_symmetry(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--type",
        "discrete",
        "--precompute-cache",
        root_dir,
        "--batch-norm",  # important for discrete model
        "--hidden",
        "128",
        "--depth",
        "5",
        "--lr",
        "0.001",
        "--epochs",
        "75",
        "--symmetry",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < 1.5


def test_beamnet_discrete_symmetry_and_positional(perfect_circle_dataset_n33):
    root_dir, zarr_fn = perfect_circle_dataset_n33
    args_list = base_args() + [
        "--datasets",
        zarr_fn,
        "--type",
        "discrete",
        "--precompute-cache",
        root_dir,
        "--batch-norm",  # important for discrete model
        "--hidden",
        "128",
        "--depth",
        "5",
        "--lr",
        "0.001",
        "--epochs",
        "75",
        "--symmetry",
        "--positional",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert train_results["losses"][-1] < 1.5


def test_normal_correction():
    # mean is on the boundary of the bounded range, half of the dist is out of bounds
    # factor should be 2
    assert np.isclose(
        normal_correction_for_bounded_range(
            mean=np.pi, sigma=torch.tensor(1), max_y=np.pi
        ),
        2,
    )
    # mean is in center of range with tight sigma, almost all of dist is in range
    # factor should be 1
    assert np.isclose(
        normal_correction_for_bounded_range(
            mean=0, sigma=torch.tensor(0.001), max_y=np.pi
        ),
        1,
    )
    # mean is in center of range with box at 1std
    # 68% of density is in the box
    # correction factor should be 1/0.68
    # factor should be ~1.47
    assert np.isclose(
        normal_correction_for_bounded_range(mean=0, sigma=torch.tensor(1), max_y=1),
        1.47,
        atol=0.01,
    )
