import hashlib
import os

import numpy as np
import torch
from tqdm import tqdm

from spf.scripts.train_single_point import (
    load_checkpoint,
    load_config_from_fn,
    load_dataloaders,
    load_model,
)


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


def convert_datasets_config_to_inference(
    datasets_config, ds_fn, precompute_cache, batch_size=1, workers=1
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
            "train_on_val": True,
            "workers": workers,
        }
    )
    if precompute_cache is not None:
        datasets_config.update({"precompute_cache": precompute_cache})
    return datasets_config


def get_md5_of_file(fn, cache_md5=True):
    if os.path.exists(fn + ".md5"):
        return open(fn + ".md5", "r").readlines()[0].strip()
    hash_md5 = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    md5 = hash_md5.hexdigest()
    if cache_md5:
        with open(fn + ".md5", "w") as f:
            f.write(md5)
    return md5


def get_inference_on_ds(
    ds_fn,
    config_fn,
    checkpoint_fn,
    inference_cache=None,
    device="cpu",
    batch_size=128,
    workers=8,
    precompute_cache=None,
):
    if inference_cache is None:
        return run_inference_on_ds(
            ds_fn=ds_fn,
            config_fn=config_fn,
            checkpoint_fn=checkpoint_fn,
            device=device,
            batch_size=batch_size,
            workers=workers,
            precompute_cache=precompute_cache,
        )
    config_checksum = get_md5_of_file(config_fn)
    checkpoint_checksum = get_md5_of_file(checkpoint_fn)
    ds_basename = os.path.basename(ds_fn)
    inference_cache_fn = (
        f"{inference_cache}/{ds_basename}/{checkpoint_checksum}/{config_checksum}.npz"
    )
    if os.path.exists(inference_cache_fn):
        return np.load(inference_cache_fn)
    # run inference
    os.makedirs(os.path.dirname(inference_cache_fn), exist_ok=True)
    results = run_inference_on_ds(
        ds_fn=ds_fn,
        config_fn=config_fn,
        checkpoint_fn=checkpoint_fn,
        device=device,
        batch_size=batch_size,
        workers=workers,
        precompute_cache=precompute_cache,
    )
    results = {key: value.numpy() for key, value in results.items()}
    np.savez_compressed(inference_cache_fn + ".tmp", **results)
    os.rename(inference_cache_fn + ".tmp.npz", inference_cache_fn)
    return results


def run_inference_on_ds(
    ds_fn, config_fn, checkpoint_fn, device, batch_size, workers, precompute_cache
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
    )

    _, val_dataloader = load_dataloaders(
        datasets_config, optim_config, config["global"], step=0, epoch=0
    )
    model.eval()
    outputs = []
    with torch.no_grad():
        for _, val_batch_data in enumerate(val_dataloader):
            val_batch_data = val_batch_data.to(optim_config["device"])
            outputs.append(model(val_batch_data))
    results = {"single": torch.vstack([output["single"] for output in outputs]).cpu()}
    sessions_times_radios, snapshots, ntheta = results["single"].shape
    sessions = sessions_times_radios // 2
    radios = 2
    results["single"] = results["single"].reshape(sessions, radios, snapshots, ntheta)

    if "paired" in outputs[0]:
        results["paired"] = torch.vstack([output["paired"] for output in outputs]).cpu()
        results["paired"] = results["paired"].reshape(
            sessions, radios, snapshots, ntheta
        )
    return results
