from spf.scripts.train_single_point import (
    load_checkpoint,
    load_config_from_fn,
    load_model,
)


def load_model_and_config_from_config_fn_and_checkpoint(config_fn, checkpoint_fn):
    config = load_config_from_fn(config_fn)
    config["optim"]["checkpoint"] = checkpoint_fn
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


def convert_datasets_config_to_inference(datasets_config, ds_fn):
    datasets_config = datasets_config.copy()
    datasets_config.update(
        {
            "batch_size": 1,
            "flip": False,
            "double_flip": False,
            "precompute_cache": "/home/mouse9911/precompute_cache_chunk16_sept",
            "shuffle": False,
            "skip_qc": True,
            "snapshots_adjacent_stride": 1,
            "train_snapshots_per_session": 1,
            "val_snapshots_per_session": 1,
            "random_snapshot_size": False,
            "snapshots_stride": 1,
            "train_paths": [ds_fn],
            "train_on_val": True,
            "workers": 1,
        }
    )
    return datasets_config
