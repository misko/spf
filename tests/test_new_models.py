import collections.abc
import os
import pathlib
import tempfile

import pytest
import torch
import yaml
from tqdm import tqdm

from spf.dataset.spf_dataset import v5_collate_keys_fast, v5spfdataset
from spf.model_training_and_inference.models.single_point_networks_inference import (
    convert_datasets_config_to_inference,
    get_inference_on_ds,
    load_model_and_config_from_config_fn_and_checkpoint,
)
from spf.scripts.train_single_point import (
    get_parser_filter,
    global_config_to_keys_used,
    load_config_from_fn,
    load_dataloaders,
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


@pytest.fixture
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


@pytest.fixture
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


def update_config(input_fn, updates, output_fn):
    base_config = load_config_from_fn(str(input_fn))
    merged_config = merge_dictionary(base_config, updates)
    with open(output_fn, "w") as f:
        yaml.dump(merged_config, f)


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


def dataloader_inference(
    model, global_config, datasets_config, optim_config, model_config
):

    _, val_dataloader = load_dataloaders(
        datasets_config=datasets_config,
        optim_config=optim_config,
        global_config=global_config,
        model_config=model_config,
        step=0,
        epoch=0,
    )

    model.eval()
    outputs = []
    with torch.no_grad():
        for _, val_batch_data in enumerate(tqdm(val_dataloader, leave=False)):
            val_batch_data = val_batch_data.to(optim_config["device"])
            outputs.append(model(val_batch_data))
    results = {"single": torch.vstack([output["single"] for output in outputs])}

    sessions_times_radios, snapshots, ntheta = results["single"].shape
    sessions = sessions_times_radios // 2
    radios = 2
    results["single"] = results["single"].reshape(sessions, radios, snapshots, ntheta)

    if "paired" in outputs[0]:
        results["paired"] = torch.vstack([output["paired"] for output in outputs])
        results["paired"] = results["paired"].reshape(
            sessions, radios, snapshots, ntheta
        )
    return results


def single_example_inference(model, global_config, datasets_config, optim_config):

    ds = v5spfdataset(
        datasets_config["train_paths"][0],
        nthetas=global_config["nthetas"],
        ignore_qc=True,
        precompute_cache=datasets_config["precompute_cache"],
        empirical_data_fn=datasets_config["empirical_data_fn"],
        paired=True,
        skip_fields=set(["signal_matrix"]),
    )

    keys_to_get = global_config_to_keys_used(global_config=global_config)
    outputs = []
    with torch.no_grad():
        for idx in range(len(ds)):
            single_example = v5_collate_keys_fast(keys_to_get, [ds[idx]]).to(
                optim_config["device"]
            )
            outputs.append(model(single_example))
    results = {"single": torch.vstack([output["single"] for output in outputs])}

    sessions_times_radios, snapshots, ntheta = results["single"].shape
    sessions = sessions_times_radios // 2
    radios = 2
    results["single"] = results["single"].reshape(sessions, radios, snapshots, ntheta)

    if "paired" in outputs[0]:
        results["paired"] = torch.vstack([output["paired"] for output in outputs])
        results["paired"] = results["paired"].reshape(
            sessions, radios, snapshots, ntheta
        )
    return results


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
        config["datasets"], ds_fn=ds_fn, batch_size=3, precompute_cache=precompute_cache
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
        config["datasets"], ds_fn=ds_fn, batch_size=3, precompute_cache=precompute_cache
    )

    # run inference one at a time
    single_example_results = single_example_inference(
        model, config["global"], datasets_config, optim_config
    )

    results = get_inference_on_ds(
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
