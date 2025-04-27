import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from spf.dataset.spf_dataset import (
    global_config_to_keys_used,
    v5_collate_keys_fast,
    v5spfdataset,
)
from spf.s3_utils import b2_file_to_local_with_cache, spf_exists
from spf.scripts.train_single_point import (
    load_checkpoint,
    load_config_from_fn,
    load_dataloaders,
    load_model,
)
from spf.utils import SEGMENTATION_VERSION, get_md5_of_file


def load_model_and_config_from_config_fn_and_checkpoint(
    config_fn, checkpoint_fn, device=None
):
    config = load_config_from_fn(config_fn)
    config["optim"]["checkpoint"] = checkpoint_fn
    if device is not None:
        config["optim"]["device"] = device
    m = load_model(config["model"], config["global"]).to(config["optim"]["device"])
    m, _, _, _, _ = load_checkpoint(
        checkpoint_fn=config["optim"]["checkpoint"],
        config=config,
        model=m,
        optimizer=None,
        scheduler=None,
        force_load=True,
    )
    return m, config


def single_example_realtime_inference(model, global_config, optim_config, realtime_ds):
    keys_to_get = global_config_to_keys_used(global_config=global_config)
    outputs = []
    with torch.no_grad():
        for sample in realtime_ds:
            # get a sample from realtime
            single_example = v5_collate_keys_fast(keys_to_get, [sample]).to(
                optim_config["device"]
            )
            yield model(single_example)
            outputs.append(model(single_example))
    return None


def single_example_inference(model, global_config, datasets_config, optim_config):
    model.eval()

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


def dataloader_inference(
    model, global_config, datasets_config, optim_config, model_config
):

    _, val_dataloader, _ = load_dataloaders(
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


def convert_datasets_config_to_inference(
    datasets_config,
    ds_fn,
    precompute_cache,
    segmentation_version,
    batch_size=1,
    workers=1,
    v4=False,
):
    datasets_config = datasets_config.copy()
    datasets_config.update(
        {
            "batch_size": batch_size,
            "flip": False,
            "double_flip": False,
            "shuffle": False,
            "skip_qc": True,
            "snapshots_adjacent_stride": 1,
            "train_snapshots_per_session": 1,
            "val_snapshots_per_session": 1,
            "random_snapshot_size": False,
            "snapshots_stride": 1,
            "train_paths": [ds_fn],
            # "val_paths": [ds_fn],
            "train_on_val": True,
            "workers": workers,
            "segmentation_version": segmentation_version,
        }
    )
    if precompute_cache is not None:
        datasets_config.update({"precompute_cache": precompute_cache})
    datasets_config.update({"v4": v4})
    return datasets_config


def get_inference_on_ds_noexceptions(
    ds_fn,
    config_fn,
    checkpoint_fn,
    inference_cache=None,
    device="cpu",
    batch_size=128,
    workers=8,
    precompute_cache=None,
    crash_if_not_cached=True,
    segmentation_version=None,
):
    try:
        get_nn_inference_on_ds_and_cache(
            ds_fn,
            config_fn,
            checkpoint_fn,
            inference_cache=inference_cache,
            device=device,
            batch_size=batch_size,
            workers=workers,
            precompute_cache=precompute_cache,
            crash_if_not_cached=crash_if_not_cached,
            segmentation_version=segmentation_version,
        )
    except Exception as e:
        logging.error(f"Failed to process {ds_fn} with {str(e)}")


def get_nn_inference_on_ds_and_cache(
    ds_fn,
    config_fn,
    checkpoint_fn,
    inference_cache=None,
    device="cpu",
    batch_size=128,
    workers=8,
    precompute_cache=None,
    crash_if_not_cached=True,
    segmentation_version=None,
    v4=False,
):
    if segmentation_version is None:
        logging.warning(
            f"Segmentation version not specified using default {SEGMENTATION_VERSION}"
        )
        segmentation_version = SEGMENTATION_VERSION
    if inference_cache is None:
        assert not crash_if_not_cached
        logging.debug("Inference cache: Skipping cache because not specified")
        return run_nn_inference_on_ds(
            ds_fn=ds_fn,
            config_fn=config_fn,
            checkpoint_fn=checkpoint_fn,
            device=device,
            batch_size=batch_size,
            workers=workers,
            precompute_cache=precompute_cache,
            segmentation_version=segmentation_version,
            v4=v4,
        )
    config_checksum = get_md5_of_file(config_fn)
    checkpoint_checksum = get_md5_of_file(checkpoint_fn)
    logging.debug(
        f"checkpoint_checksum {checkpoint_checksum} config_checksum {config_checksum}"
    )
    ds_basename = os.path.basename(ds_fn)
    inference_cache_fn = f"{inference_cache}/{ds_basename}/{segmentation_version:0.3f}/{checkpoint_checksum}/{config_checksum}.npz"
    if spf_exists(inference_cache_fn):
        logging.debug("Inference cache: Using cached results")
        inference_cache_local_fn = b2_file_to_local_with_cache(inference_cache_fn)
        with open(inference_cache_local_fn, "rb") as f:
            return {k: v for k, v in np.load(f).items()}
    # run inference
    assert not crash_if_not_cached, inference_cache_fn
    os.makedirs(os.path.dirname(inference_cache_fn), exist_ok=True)
    logging.debug("Inference cache: Computing results for cache")
    results = run_nn_inference_on_ds(
        ds_fn=ds_fn,
        config_fn=config_fn,
        checkpoint_fn=checkpoint_fn,
        device=device,
        batch_size=batch_size,
        workers=workers,
        precompute_cache=precompute_cache,
        segmentation_version=segmentation_version,
        v4=v4,
    )
    results = {key: value.numpy() for key, value in results.items()}
    np.savez_compressed(inference_cache_fn + ".tmp", **results)
    os.rename(inference_cache_fn + ".tmp.npz", inference_cache_fn)
    return results


def run_nn_inference_on_ds(
    ds_fn,
    config_fn,
    checkpoint_fn,
    device,
    batch_size,
    workers,
    precompute_cache,
    segmentation_version,
    v4=False,
):
    # load model and model config
    model, config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn=config_fn, checkpoint_fn=checkpoint_fn, device=device
    )

    # prepare inference configs
    optim_config = {"device": device, "dtype": torch.float32}
    datasets_config = convert_datasets_config_to_inference(
        config["datasets"],
        ds_fn=ds_fn,
        batch_size=batch_size,
        workers=workers,
        precompute_cache=precompute_cache,
        segmentation_version=segmentation_version,
        v4=v4,
    )
    try:
        _, val_dataloader, _ = load_dataloaders(
            datasets_config=datasets_config,
            optim_config=optim_config,
            global_config=config["global"],
            model_config=config["model"],
            step=0,
            epoch=0,
            no_tqdm=True,
        )
    except Exception as e:
        logging.error(f"Failed to load file {ds_fn} {e}")
        raise e
    model.eval()
    outputs = []
    with torch.no_grad():
        for _, val_batch_data in enumerate(val_dataloader):
            val_batch_data = val_batch_data.to(optim_config["device"])
            outputs.append(model(val_batch_data))
    results = {"single": torch.vstack([output["single"] for output in outputs]).cpu()}
    sessions_times_radios, snapshots, single_ntheta = results["single"].shape
    sessions = sessions_times_radios // 2
    radios = 2
    results["single"] = results["single"].reshape(
        sessions, radios, snapshots, single_ntheta
    )

    if "paired" in outputs[0]:
        results["paired"] = torch.vstack([output["paired"] for output in outputs]).cpu()
        _, _, paired_ntheta = results["paired"].shape
        results["paired"] = results["paired"].reshape(
            sessions, radios, snapshots, paired_ntheta
        )
    return results
