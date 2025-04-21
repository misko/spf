import os

import torch

from spf.model_training_and_inference.models.single_point_networks_inference import (
    convert_datasets_config_to_inference,
    dataloader_inference,
    get_nn_inference_on_ds_and_cache,
    load_model_and_config_from_config_fn_and_checkpoint,
    single_example_inference,
)
from spf.utils import SEGMENTATION_VERSION


def test_single_checkpoints_exist(
    single_net_checkpoint,
):
    single_checkpoints_dir = single_net_checkpoint
    assert os.path.exists(f"{single_checkpoints_dir}/best.pth")
    assert os.path.exists(f"{single_checkpoints_dir}/checkpoint_e1_s10.pth")


def test_paired_checkpoints_from_single_exist(
    paired_net_checkpoint_using_single_checkpoint,
):
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint
    assert os.path.exists(f"{paired_checkpoints_dir}/best.pth")
    assert os.path.exists(f"{paired_checkpoints_dir}/checkpoint_e1_s10.pth")


def test_inference_single_checkpoint(
    single_net_checkpoint, perfect_circle_dataset_n7_with_empirical
):
    single_checkpoints_dir = single_net_checkpoint
    precompute_cache, _, zarr_fn = perfect_circle_dataset_n7_with_empirical

    ds_fn = f"{zarr_fn}.zarr"
    config_fn = f"{single_checkpoints_dir}/config.yml"
    checkpoint_fn = f"{single_checkpoints_dir}/best.pth"

    # load model and model config
    model, config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn=config_fn, checkpoint_fn=checkpoint_fn
    )

    # prepare inference configs
    optim_config = {"device": "cpu", "dtype": torch.float32}
    datasets_config = convert_datasets_config_to_inference(
        config["datasets"],
        ds_fn=ds_fn,
        batch_size=3,
        precompute_cache=precompute_cache,
        segmentation_version=SEGMENTATION_VERSION,
    )

    # inference using dataloader
    dataloader_results = dataloader_inference(
        model=model,
        global_config=config["global"],
        datasets_config=datasets_config,
        optim_config=optim_config,
        model_config=config["model"],
    )

    # run inference one at a time
    single_example_results = single_example_inference(
        model, config["global"], datasets_config, optim_config
    )

    assert dataloader_results["single"].isclose(single_example_results["single"]).all()


def test_inference_single_checkpoint_against_ds_inference(
    single_net_checkpoint, perfect_circle_dataset_n7_with_empirical
):
    single_checkpoints_dir = single_net_checkpoint
    precompute_cache, _, zarr_fn = perfect_circle_dataset_n7_with_empirical

    ds_fn = f"{zarr_fn}.zarr"
    config_fn = f"{single_checkpoints_dir}/config.yml"
    checkpoint_fn = f"{single_checkpoints_dir}/best.pth"

    # load model and model config
    model, config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn=config_fn, checkpoint_fn=checkpoint_fn
    )

    # prepare inference configs
    optim_config = {"device": "cpu", "dtype": torch.float32}
    datasets_config = convert_datasets_config_to_inference(
        config["datasets"],
        ds_fn=ds_fn,
        batch_size=3,
        precompute_cache=precompute_cache,
        segmentation_version=SEGMENTATION_VERSION,
    )

    # run inference one at a time
    single_example_results = single_example_inference(
        model, config["global"], datasets_config, optim_config
    )

    results = get_nn_inference_on_ds_and_cache(
        ds_fn,
        config_fn,
        checkpoint_fn,
        inference_cache=None,
        device="cpu",
        batch_size=4,
        workers=0,
        precompute_cache=None,
        crash_if_not_cached=False,
    )
    assert results["single"].isclose(single_example_results["single"]).all()


def test_inference_paired_checkpoint(
    single_net_checkpoint,
    paired_net_checkpoint_using_single_checkpoint,
    perfect_circle_dataset_n7_with_empirical,
):
    # get single checkpoint results
    single_checkpoints_dir = single_net_checkpoint
    precompute_cache, _, zarr_fn = perfect_circle_dataset_n7_with_empirical

    ds_fn = f"{zarr_fn}.zarr"
    single_config_fn = f"{single_checkpoints_dir}/config.yml"
    single_checkpoint_fn = f"{single_checkpoints_dir}/best.pth"

    # load model and model config
    single_model, single_config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn=single_config_fn, checkpoint_fn=single_checkpoint_fn
    )

    # prepare inference configs
    optim_config = {"device": "cpu", "dtype": torch.float32}
    single_datasets_config = convert_datasets_config_to_inference(
        single_config["datasets"],
        ds_fn=ds_fn,
        batch_size=3,
        precompute_cache=precompute_cache,
        segmentation_version=SEGMENTATION_VERSION,
    )

    # inference using dataloader
    dataloader_single_results = dataloader_inference(
        model=single_model,
        global_config=single_config["global"],
        datasets_config=single_datasets_config,
        optim_config=optim_config,
        model_config=single_config["model"],
    )

    # get paired checkpoint results
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

    paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
    paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

    # load model and model config
    paired_model, paired_config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn=paired_config_fn, checkpoint_fn=paired_checkpoint_fn
    )

    # prepare inference configs
    paired_datasets_config = convert_datasets_config_to_inference(
        paired_config["datasets"],
        ds_fn=ds_fn,
        precompute_cache=precompute_cache,
        segmentation_version=SEGMENTATION_VERSION,
    )

    # inference using dataloader
    dataloader_paired_results = dataloader_inference(
        model=paired_model,
        global_config=paired_config["global"],
        datasets_config=paired_datasets_config,
        optim_config=optim_config,
        model_config=paired_config["model"],
    )

    # run inference one at a time
    single_example_paired_results = single_example_inference(
        paired_model, paired_config["global"], paired_datasets_config, optim_config
    )

    assert (
        dataloader_paired_results["single"]
        .isclose(single_example_paired_results["single"])
        .all()
    )

    assert (
        dataloader_paired_results["paired"]
        .isclose(single_example_paired_results["paired"])
        .all()
    )

    assert (
        dataloader_paired_results["single"]
        .isclose(dataloader_single_results["single"])
        .all()
    )
