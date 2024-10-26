import argparse
import copy
import datetime
import glob
import logging
import math
import os
import random
from functools import partial

import numpy as np
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import v5_collate_keys_fast, v5spfdataset
from spf.model_training_and_inference.models.single_point_networks import (
    PairedMultiPointWithBeamformer,
    PairedSinglePointWithBeamformer,
    SinglePointPassThrough,
    SinglePointWithBeamformer,
    TrajPairedMultiPointWithBeamformer,
)
from spf.rf import torch_pi_norm, torch_reduce_theta_to_positive_y
from spf.utils import StatefulBatchsampler


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    for (
        _dataset
    ) in dataset.dataset.datasets:  # subset_dataset.concat_dataset.v5spfdataset
        _dataset.reinit()


def load_config_from_fn(fn):
    with open(fn, "r") as f:
        return yaml.safe_load(f)
    return None


def load_seed(global_config):
    torch.manual_seed(global_config["seed"])
    random.seed(global_config["seed"])


def expand_wildcards_and_join(paths):
    expanded_paths = []
    for path in paths:
        expanded_paths += glob.glob(path)
    return expanded_paths


def load_dataloaders(datasets_config, optim_config, global_config, step=0, epoch=0):
    skip_fields = set(["signal_matrix", "simple_segmentations"])
    # if not global_config["beamformer_input"]:
    skip_fields |= set(["windowed_beamformer"])
    # import glob
    # glob.glob('./[0-9].*')

    train_dataset_filenames = expand_wildcards_and_join(datasets_config["train_paths"])
    random.shuffle(train_dataset_filenames)

    if datasets_config.get("train_on_val", False):
        val_paths = train_dataset_filenames
        train_paths = train_dataset_filenames
    else:
        n_val_files = max(
            1,
            int(datasets_config["val_holdout_fraction"] * len(train_dataset_filenames)),
        )
        val_paths = train_dataset_filenames[-n_val_files:]
        train_paths = train_dataset_filenames[:-n_val_files]

    val_adjacent_stride = datasets_config.get(
        "val_snapshots_adjacent_stride", datasets_config["snapshots_adjacent_stride"]
    )
    logging.info(f"Using validation stride of {val_adjacent_stride}")
    val_datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=datasets_config["precompute_cache"],
            nthetas=global_config["nthetas"],
            paired=global_config["n_radios"] > 1,
            ignore_qc=datasets_config["skip_qc"],
            gpu=optim_config["device"] == "cuda",
            snapshots_per_session=datasets_config["val_snapshots_per_session"],
            snapshots_stride=datasets_config["snapshots_stride"],
            snapshots_adjacent_stride=val_adjacent_stride,
            readahead=False,
            skip_fields=skip_fields,
            empirical_data_fn=datasets_config.get("empirical_data_fn", None),
            empirical_individual_radio=datasets_config.get(
                "empirical_individual_radio"
            ),
            empirical_symmetry=datasets_config.get("empirical_symmetry", None),
            target_dtype=optim_config["dtype"],
        )
        for prefix in val_paths
    ]
    train_datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=datasets_config["precompute_cache"],
            nthetas=global_config["nthetas"],
            paired=global_config["n_radios"] > 1,
            ignore_qc=datasets_config["skip_qc"],
            gpu=optim_config["device"] == "cuda",
            snapshots_per_session=datasets_config["train_snapshots_per_session"],
            snapshots_stride=datasets_config["snapshots_stride"],
            snapshots_adjacent_stride=datasets_config["snapshots_adjacent_stride"],
            readahead=False,
            skip_fields=skip_fields,
            empirical_data_fn=datasets_config.get("empirical_data_fn", None),
            empirical_individual_radio=datasets_config.get(
                "empirical_individual_radio"
            ),
            empirical_symmetry=datasets_config.get("empirical_symmetry", None),
            target_dtype=optim_config["dtype"],
            # difference
            flip=datasets_config["flip"],
            double_flip=datasets_config["double_flip"],
            random_adjacent_stride=datasets_config.get("random_adjacent_stride", False),
        )
        for prefix in train_paths
    ]
    assert len(train_paths) > 0
    assert len(val_paths) > 0
    for ds in val_datasets + train_datasets:
        ds.get_segmentation()

    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    train_ds = torch.utils.data.ConcatDataset(train_datasets)

    train_idxs = range(len(train_ds))
    val_idxs = list(range(len(val_ds)))

    random.shuffle(val_idxs)
    val_idxs = val_idxs[
        : max(
            1, int(len(val_idxs) * datasets_config.get("val_subsample_fraction", 1.0))
        )
    ]

    train_ds = torch.utils.data.Subset(train_ds, train_idxs)
    val_ds = torch.utils.data.Subset(val_ds, val_idxs)

    logging.info(f"Train-dataset size {len(train_ds)}, Val dataset size {len(val_ds)}")

    def params_for_ds(ds, batch_size):
        sampler = StatefulBatchsampler(
            ds,
            shuffle=datasets_config["shuffle"],
            seed=global_config["seed"],
            batch_size=batch_size,
        )
        keys_to_get = [
            "all_windows_stats",
            # "rx_pos_xy",
            # "tx_pos_xy",
            # "downsampled_segmentation_mask",
            "rx_spacing",
            "y_rad",
            "craft_y_rad",
            "y_phi",
            "system_timestamp",
            "empirical",
            "y_rad_binned",
            "craft_y_rad_binned",
            "weighted_windows_stats",
            "rx_pos_xy",
            "tx_pos_xy",
        ]
        if global_config["beamformer_input"]:
            # keys_to_get += ["windowed_beamformer"]
            keys_to_get += ["weighted_beamformer"]
        return {
            # "batch_size": args.batch,
            "num_workers": datasets_config["workers"],
            "collate_fn": partial(v5_collate_keys_fast, keys_to_get),
            "worker_init_fn": worker_init_fn,
            # "pin_memory": True,
            "prefetch_factor": 2 if datasets_config["workers"] > 0 else None,
            "batch_sampler": sampler,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        **params_for_ds(
            train_ds,
            batch_size=datasets_config["batch_size"],
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        **params_for_ds(
            val_ds,
            batch_size=datasets_config["batch_size"],
        ),
    )

    logging.info(
        f"Train dataloader size: {len(train_dataloader)}, Val dataloader size: {len(val_dataloader)}"
    )

    return train_dataloader, val_dataloader


def load_optimizer(optim_config, params):
    optimizer = torch.optim.AdamW(
        params,
        lr=optim_config["learning_rate"],
        weight_decay=optim_config["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=optim_config.get("scheduler_step", 1),
        gamma=0.5,
        verbose=True,
    )
    return optimizer, scheduler


class FrozenModule(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

    def train(self, _):
        return self

    def forward(self, x):
        self.net.eval()
        return self.net(x)


def save_model(
    prefix,
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    config,
    running_config,
    checkpoint_fn=None,
):
    model_checksum("save: ", model)
    if checkpoint_fn is None:
        checkpoint_fn = f"checkpoint_e{epoch}_s{step}.pth"

    torch.save(
        {
            "epoch": epoch,  # Optional: save the current epoch
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
            "scheduler_state_dict": scheduler.state_dict(),
        },
        f"{prefix}/{checkpoint_fn}",
    )
    with open(f"{prefix}/config.yml", "w") as outfile:
        yaml.dump(running_config, outfile)


def load_checkpoint(checkpoint_fn, config, model, optimizer, scheduler):
    logging.info(f"Loading checkpoint {checkpoint_fn}")
    checkpoint = torch.load(checkpoint_fn, map_location=torch.device("cpu"))

    # config_being_loaded = checkpoint["config"]

    ### FIRST LOAD MODEL FULLY #####
    # model_being_loaded = load_model(
    #    config_being_loaded["model"], config_being_loaded["global"]
    # )
    # model_being_loaded.load_state_dict(checkpoint["model_state_dict"])
    # model_being_loaded = model_being_loaded.to(config["optim"]["device"])

    ### THEN LOAD OPTIMIZER #####
    # optimizer_being_loaded, scheduler_being_loaded = load_optimizer(
    #     config_being_loaded["optim"], model_being_loaded.parameters()
    # )
    # optimizer_being_loaded.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler_being_loaded.load_state_dict(checkpoint["scheduler_state_dict"])

    # check if we loading a single network
    if config["model"].get("load_single", False):
        logging.info("Loading single_radio_net only")
        model.single_radio_net.load_state_dict(checkpoint["model_state_dict"])
        for param in model.single_radio_net.parameters():
            param.requires_grad = False
        # model.single_radio_net = FrozenModule(model_being_loaded)
        return (model, optimizer, scheduler, 0, 0)  # epoch  # step
    elif config["model"].get("load_paired", False):
        # check if we loading a paired network
        logging.info("Loading paired_radio net only")
        model.multi_radio_net.load_state_dict(checkpoint["model_state_dict"])
        for param in model.multi_radio_net.parameters():
            param.requires_grad = False
        # breakpoint()
        return (model, optimizer, scheduler, 0, 0)  # epoch  # step

    # else
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        model,
        optimizer,
        scheduler,
        checkpoint["epoch"],
        checkpoint["step"],
    )


def load_model(model_config, global_config):
    if model_config["name"] == "pass_through":
        return SinglePointPassThrough(model_config, global_config)
    elif model_config["name"] == "beamformer":
        return SinglePointWithBeamformer(model_config, global_config)
    elif model_config["name"] == "pairedbeamformer":
        return PairedSinglePointWithBeamformer(model_config, global_config)
    elif model_config["name"] == "multipairedbeamformer":
        return PairedMultiPointWithBeamformer(model_config, global_config)
    elif model_config["name"] == "trajmultipairedbeamformer":
        return TrajPairedMultiPointWithBeamformer(model_config, global_config)

    raise ValueError


def uniform_preds(batch):
    return (
        torch.ones(*batch["empirical"].shape, device=batch["empirical"].device)
        / batch["empirical"].shape[-1]
    )


def target_from_scatter(batch, y_rad, y_rad_binned, sigma, k):
    assert y_rad.max() <= torch.pi and y_rad.min() >= -torch.pi
    torch.zeros(*batch["empirical"].shape, device=batch["empirical"].device)
    n = batch["empirical"].shape[-1]
    assert n % 2 == 1
    padding = n  # easier if padding is just n
    # try to deal with the tails of the distribution wrapping around
    effective_n = 2 * padding + n

    m = (
        (torch.arange(effective_n, device=batch["empirical"].device) - effective_n // 2)
        * (2 / n)
    ).reshape(1, 1, effective_n) * torch.pi
    diff = ((m - y_rad) / sigma) ** 2
    rescaled = torch.nn.functional.normalize((-diff / 2).exp(), p=1.0, dim=2)

    batch, session, _ = rescaled.shape
    return rescaled.reshape(batch, session, 3, n).sum(axis=2)


def target_from_scatter_binned(batch, y_rad, y_rad_binned, sigma, k):
    # if paired:
    #     y_rad_binned = batch["craft_y_rad_binned"][..., None]
    # else:
    #     y_rad_binned = batch["y_rad_binned"][..., None]
    scattered_targets = torch.zeros(
        *batch["empirical"].shape, device=batch["empirical"].device
    ).scatter(
        2,
        index=y_rad_binned,
        src=torch.ones(*y_rad_binned.shape, device=batch["empirical"].device),
    )
    if sigma > 0.0:
        scattered_targets = torchvision.transforms.GaussianBlur((k, 1), sigma=sigma)(
            scattered_targets
        )
        scattered_targets /= scattered_targets.sum(axis=2, keepdims=True)
    return scattered_targets


def weighted_beamformer(batch):
    seg_mask = batch["downsampled_segmentation_mask"]
    windowed_beam_former_scaled = batch["windowed_beamformer"] / (
        seg_mask.sum(axis=3, keepdim=True) + 0.1
    )
    return (windowed_beam_former_scaled * seg_mask.transpose(2, 3)).mean(axis=2)


def mse_loss(output, target):
    return ((output - target) ** 2).sum(axis=2).mean()


def discrete_loss(output, target):
    # return -torch.sqrt((target * output).sum(axis=2).mean())
    return -(target * output).sum(axis=2).mean()
    # return -torch.log((target * output).sum(axis=2)).mean()


def nn_checksum(net):
    return torch.vstack([p.abs().mean() for p in net.parameters()]).mean().item()


def model_checksum(prefix, m):
    if hasattr(m, "single_radio_net"):
        logging.info(
            f"{prefix} checksum: single_radio_net {nn_checksum(m.single_radio_net)}"
        )
    if hasattr(m, "multi_radio_net"):
        logging.info(
            f"{prefix} checksum: multi_radio_net {nn_checksum(m.multi_radio_net)}"
        )
    logging.info(f"{prefix} checksum: model {nn_checksum(m)}")


def new_log():
    return {
        "loss": [],
        "epoch": [],
        "data_seen": [],
        "passthrough_loss": [],
        "uniform_loss": [],
        "single_loss": [],
        "paired_loss": [],
        "multipaired_loss": [],
        "learning_rate": [],
        "multipaired_direct_loss": [],
        "multipaired_tx_pos_loss": [],
    }


class SimpleLogger:
    def __init__(self, args, logger_config, full_config):
        pass

    def log(self, data, step, prefix=""):

        losses = {
            f"{prefix}{key}": np.array(value).mean()
            for key, value in data.items()
            if "plot" not in key and len(value) > 0
        }

        for k, v in losses.items():
            logging.info(f"{step}:{k} {v}")

        for key, value in data.items():
            if "plot" in key:
                plt.close(value)

        return losses


class WNBLogger:
    def __init__(self, args, logger_config, full_config):
        wandb_name = None
        if args.name is not None:
            wandb_name = args.name
        elif "run_name" in logger_config:
            wandb_name = logger_config["run_name"]
        else:
            wandb_name = (
                os.path.basename(args.config)
                + " "
                + datetime.datetime.now().strftime("%Y-%m-%d")
            )

        wandb.init(
            # set the wandb project where this run will be logged
            project=logger_config["project"],
            # track hyperparameters and run metadata
            config=full_config,
            name=wandb_name,
            # id="d7i47byn",  # "iconic-fog-63",
        )

    def log(self, data, step, prefix=""):
        losses = {
            f"{prefix}{key}": np.array(value).mean()
            for key, value in data.items()
            if "plot" not in key and len(value) > 0
        }
        losses["step"] = step
        logging.info(f"SUBMIT losses to wandb, {losses}")
        for key, value in data.items():
            if "plot" in key:
                losses[key] = value
        wandb.log(
            losses,
            step=step,
        )
        for key, value in data.items():
            if "plot" in key:
                plt.close(value)
        return losses


def compute_loss(
    output,
    batch_data,
    datasets_config,
    loss_fn,
    scatter_fn,
    single=True,
    paired=True,
    plot=False,
    direct_loss=True,
):
    loss_d = {}
    loss = 0

    fig = None
    if plot:
        fig, axs = plt.subplots(3, 2, figsize=(10, 30))
        n = output["single"].shape[0] * output["single"].shape[1]
        d = output["single"].shape[2]
        show_n = min(n, 80)

    if single and "single" in output:
        target = scatter_fn(
            batch_data,
            y_rad=batch_data["y_rad"][..., None],
            y_rad_binned=batch_data["y_rad_binned"][..., None],
            sigma=datasets_config["sigma"],
            k=datasets_config["scatter_k"],
        )
        loss_d["single_loss"] = loss_fn(output["single"], target)
        loss += loss_d["single_loss"]

        if plot:
            axs[0, 0].imshow(
                output["single"].reshape(n, -1).cpu().detach().numpy()[:show_n],
                aspect="auto",
            )
            axs[0, 1].imshow(
                target.reshape(n, -1).cpu().detach().numpy()[:show_n], aspect="auto"
            )
            axs[0, 0].set_title("1radio x 1timestep (Pred)")
            axs[0, 0].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])
            axs[0, 1].set_title("1radio x 1timestep (label)")
            axs[0, 1].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])

    if paired and "paired" in output or "multipaired" in output:
        paired_target = scatter_fn(
            batch_data,
            y_rad=batch_data["craft_y_rad"][..., None],
            y_rad_binned=batch_data["craft_y_rad_binned"][..., None],
            sigma=datasets_config["sigma"],
            k=datasets_config["scatter_k"],
        )
    if paired and "paired" in output:
        loss_d["paired_loss"] = loss_fn(output["paired"], paired_target)
        loss += loss_d["paired_loss"]
        if plot:
            axs[1, 0].imshow(
                output["paired"].reshape(n, -1).cpu().detach().numpy()[:show_n],
                aspect="auto",
            )
            axs[1, 1].imshow(
                paired_target.reshape(n, -1).cpu().detach().numpy()[:show_n],
                aspect="auto",
            )
            axs[1, 0].set_title("2radio x 1timestep (pred)")
            axs[1, 1].set_title("2radio x 1timestep (label)")
            axs[1, 0].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])
            axs[1, 1].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])

    if paired and "multipaired" in output:
        loss_d["multipaired_loss"] = loss_fn(output["multipaired"], paired_target)
        loss += loss_d["multipaired_loss"]

        if "multipaired_tx_pos" in output:
            target_tx_pos = (
                batch_data["tx_pos_xy"] - batch_data["rx_pos_xy"]
            )  # relative to each
            loss_d["multipaired_tx_pos_loss"] = loss_fn(
                output["multipaired_tx_pos"], target_tx_pos
            )
            loss += loss_d["multipaired_tx_pos_loss"] * 0.1

        if direct_loss:
            loss_d["multipaired_direct_loss"] = (
                (
                    torch_pi_norm(
                        output["multipaired_direct"]
                        - batch_data["craft_y_rad"][..., None]
                    )
                )
                ** 2
            ).mean()
            loss += (
                loss_d["multipaired_direct_loss"] / 30
            )  # TODO constant to keep losses balanced
        if plot:
            axs[2, 0].imshow(
                output["multipaired"].reshape(n, -1).cpu().detach().numpy()[:show_n],
                aspect="auto",
            )
            axs[2, 1].imshow(
                paired_target.reshape(n, -1).cpu().detach().numpy()[:show_n],
                aspect="auto",
            )
            axs[2, 0].set_title(
                f"2radio x {output['multipaired'].shape[1]}timestep (pred)"
            )
            axs[2, 1].set_title(
                f"2radio x {output['multipaired'].shape[1]}timestep (target)"
            )
            axs[2, 0].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])
            axs[2, 1].set_xticks([0, d // 2, d - 1], labels=["-pi", "0", "+pi"])
    loss_d["loss"] = loss
    if plot:
        fig.tight_layout()
    return loss_d, fig


def train_single_point(args):
    config = load_config_from_fn(args.config)
    config["args"] = vars(args)
    logging.info(config)

    running_config = copy.deepcopy(config)

    output_from_config = config["optim"].get("output", None)
    if args.output is None and output_from_config is not None:
        args.output = output_from_config
    if args.output is None:
        args.output = datetime.datetime.now().strftime("spf-run-%Y-%m-%d_%H-%M-%S")
    try:
        os.mkdir(args.output)
    except FileExistsError:
        pass

    torch_device_str = config["optim"]["device"]
    config["optim"]["device"] = torch.device(config["optim"]["device"])

    dtype = torch.float16
    if config["optim"]["dtype"] == "torch.float32":
        dtype = torch.float32
    elif config["optim"]["dtype"] == "torch.float16":
        dtype = torch.float16
    else:
        raise ValueError
    config["optim"]["dtype"] = dtype

    load_seed(config["global"])
    if config["datasets"].get("flip", False):
        # Cant flip when doing paired!
        assert config["model"]["name"] == "beamformer"
    if config["model"]["name"] in (
        "multipairedbeamformer",
        "trajmultipairedbeamformer",
    ):
        assert config["datasets"]["train_snapshots_per_session"] > 1
        assert config["datasets"]["val_snapshots_per_session"] > 1
    else:
        assert config["datasets"]["random_snapshot_size"] is False

    m = load_model(config["model"], config["global"]).to(config["optim"]["device"])

    model_checksum("load_model:", m)
    optimizer, scheduler = load_optimizer(config["optim"], m.parameters())

    load_seed(config["global"])

    if config["logger"]["name"] == "simple" or args.debug:
        logger = SimpleLogger(args, config.get("logger", {}), config)
    elif config["logger"]["name"] == "wandb":
        logger = WNBLogger(args, config["logger"], config)

    step = 0
    start_epoch = 0

    if "checkpoint" in config["optim"]:
        m, optimizer, scheduler, start_epoch, step = load_checkpoint(
            checkpoint_fn=config["optim"]["checkpoint"],
            config=config,
            model=m,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # start_epoch = checkpoint["epoch"]
        # step = checkpoint["step"]

    load_seed(config["global"])

    train_dataloader, val_dataloader = load_dataloaders(
        config["datasets"], config["optim"], config["global"]
    )

    scaler = torch.GradScaler(
        torch_device_str,
        enabled=config["optim"]["amp"],
    )
    losses = new_log()

    loss_fn = None
    if config["optim"]["loss"] == "mse":
        loss_fn = mse_loss
    elif config["optim"]["loss"] == "discrete":
        loss_fn = discrete_loss
    else:
        raise ValueError("Not a valid loss function")

    scatter_fn = None
    if config["datasets"]["scatter"] == "continuous":
        scatter_fn = target_from_scatter
    elif config["datasets"]["scatter"] == "onehot":
        scatter_fn = target_from_scatter_binned
    else:
        raise ValueError(f"Not a valid scatter fn, {config['datasets']['scatter']}")

    best_val_loss_so_far = None

    last_val_plot = 0
    for epoch in range(start_epoch, config["optim"]["epochs"]):
        train_dataloader.batch_sampler.set_epoch_and_start_iteration(
            epoch=epoch, start_iteration=step % len(train_dataloader)
        )

        for _, batch_data in enumerate(tqdm(train_dataloader)):
            if config["datasets"].get("random_snapshot_size", False):
                effective_snapshots_per_session = max(
                    1,
                    math.ceil(
                        torch.rand(1).item()
                        * config["datasets"]["train_snapshots_per_session"]
                    ),
                )
                batch_data = batch_data[:, :effective_snapshots_per_session]
                logging.debug(
                    f"effective_snapshots_per_session: {effective_snapshots_per_session}"
                )
            if step % config["optim"]["val_every"] == 0:
                model_checksum(f"val.e{epoch}.s{step}: ", m)
                m.eval()
                with torch.no_grad():
                    val_losses = new_log()
                    # for  in val_dataloader:
                    logging.info("Running validation:")
                    for _, val_batch_data in enumerate(
                        tqdm(val_dataloader, leave=False)
                    ):
                        # for val_batch_data in val_dataloader:
                        val_batch_data = val_batch_data.to(config["optim"]["device"])
                        # run beamformer and segmentation
                        should_we_plot = (
                            "val_plot_pred" not in losses
                            and (step - last_val_plot) > config["logger"]["plot_every"]
                        )
                        if should_we_plot:
                            last_val_plot = step

                        output = m(val_batch_data)
                        loss_d, fig = compute_loss(
                            output,
                            val_batch_data,
                            loss_fn=loss_fn,
                            datasets_config=config["datasets"],
                            scatter_fn=scatter_fn,
                            single=True,
                            plot=should_we_plot,  # only take one batch?
                            paired=epoch >= config["optim"]["head_start"],
                            direct_loss=config["optim"]["direct_loss"],
                        )

                        if fig is not None:
                            assert "val_plot_pred" not in losses
                            losses["val_plot_pred"] = fig
                        # scheduler.step(loinss_d["loss"])
                        # val_losses["learning_rate"] = [scheduler.get_last_lr()]

                        # TODO refactor
                        target = target_from_scatter(
                            val_batch_data,
                            y_rad=val_batch_data["y_rad"][..., None],
                            y_rad_binned=val_batch_data["y_rad_binned"][..., None],
                            sigma=config["datasets"]["sigma"],
                            k=config["datasets"]["scatter_k"],
                        )
                        loss_d["passthrough_loss"] = loss_fn(
                            val_batch_data["empirical"],
                            target,
                        )
                        loss_d["uniform_loss"] = loss_fn(
                            uniform_preds(val_batch_data), target
                        )

                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())

                        val_losses["epoch"].append(step / len(train_dataloader))
                        val_losses["data_seen"].append(
                            step
                            * config["datasets"]["val_snapshots_per_session"]
                            * config["datasets"]["batch_size"]
                        )

                    reported_losses = logger.log(val_losses, step=step, prefix="val_")
                    if config["optim"].get("save_on", "") != "":
                        this_loss = reported_losses[config["optim"].get("save_on")]
                        if (
                            best_val_loss_so_far == None
                            or this_loss < best_val_loss_so_far
                        ):
                            best_val_loss_so_far = this_loss
                            save_model(
                                args.output + "/",
                                m,
                                optimizer=optimizer,
                                epoch=epoch,
                                step=step,
                                config=config,
                                scheduler=scheduler,
                                running_config=running_config,
                                checkpoint_fn="best.pth",
                            )

            m.train()
            batch_data = batch_data.to(config["optim"]["device"])

            optimizer.zero_grad()
            with torch.autocast(torch_device_str, enabled=config["optim"]["amp"]):
                output = m(batch_data)

                loss_d, fig = compute_loss(
                    output,
                    batch_data,
                    datasets_config=config["datasets"],
                    loss_fn=loss_fn,
                    plot=step == 0 or (step + 1) % config["logger"]["plot_every"] == 0,
                    scatter_fn=scatter_fn,
                    single=True,
                    paired=epoch >= config["optim"]["head_start"],
                    direct_loss=config["optim"]["direct_loss"],
                )
            if fig is not None:
                assert "train_plot_pred" not in losses
                losses["train_plot_pred"] = fig

            scaler.scale(loss_d["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                for key, value in loss_d.items():
                    losses[key].append(value.item())
                losses["learning_rate"].append(scheduler.get_lr()[0])

                if step == 0 or (step + 1) % config["logger"]["log_every"] == 0:

                    losses["epoch"].append(step / len(train_dataloader))
                    losses["data_seen"].append(
                        step
                        * config["datasets"]["train_snapshots_per_session"]
                        * config["datasets"]["batch_size"]
                    )
                    logger.log(losses, step=step, prefix="train_")
                    losses = new_log()
            step += 1
            if step % config["optim"]["checkpoint_every"] == 0:
                save_model(
                    args.output + "/",
                    m,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=step,
                    config=config,
                    scheduler=scheduler,
                    running_config=running_config,
                )

            if "steps" in config["optim"] and step >= config["optim"]["steps"]:
                return
        scheduler.step()
        logging.info(f"LR STEP: {scheduler.get_lr()}")


def get_parser_filter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="config file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output folder",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="logger name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--beamnet-latent",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--debug-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-prefix", type=str, default="./this_model_")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = get_parser_filter()
    args = parser.parse_args()
    train_single_point(args)
