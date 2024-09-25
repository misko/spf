import argparse
import glob
import random
from functools import partial

import numpy as np
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import v5_collate_keys_fast, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import FFNN
from spf.model_training_and_inference.models.single_point_networks import (
    PairedMultiPointWithBeamformer,
    PairedSinglePointWithBeamformer,
    SinglePointPassThrough,
    SinglePointWithBeamformer,
)
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


def load_dataloaders(datasets_config, optim_config, global_config):
    skip_fields = set(["signal_matrix", "simple_segmentations"])
    # if not global_config["beamformer_input"]:
    skip_fields |= set(["windowed_beamformer"])
    # import glob
    # glob.glob('./[0-9].*')

    train_dataset_filenames = expand_wildcards_and_join(datasets_config["train_paths"])
    random.shuffle(train_dataset_filenames)

    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=datasets_config["precompute_cache"],
            nthetas=global_config["nthetas"],
            paired=global_config["n_radios"] > 1,
            ignore_qc=datasets_config["skip_qc"],
            gpu=optim_config["device"] == "cuda",
            snapshots_per_session=datasets_config["snapshots_per_session"],
            snapshots_stride=datasets_config["snapshots_stride"],
            readahead=False,
            skip_fields=skip_fields,
            empirical_data_fn=datasets_config["empirical_data_fn"],
            empirical_individual_radio=datasets_config["empirical_individual_radio"],
            empirical_symmetry=datasets_config["empirical_symmetry"],
            target_dtype=optim_config["dtype"],
        )
        for prefix in train_dataset_filenames
    ]
    for ds in datasets:
        ds.get_segmentation()
    complete_ds = torch.utils.data.ConcatDataset(datasets)

    n = len(complete_ds)
    train_idxs = range(int((1.0 - datasets_config["val_holdout_fraction"]) * n))
    val_idxs = list(range(train_idxs[-1] + 1, n))

    random.shuffle(val_idxs)
    val_idxs = val_idxs[
        : max(1, int(len(val_idxs) * datasets_config["val_subsample_fraction"]))
    ]

    train_ds = torch.utils.data.Subset(complete_ds, train_idxs)
    val_ds = torch.utils.data.Subset(complete_ds, val_idxs)
    print(f"Train-dataset size {len(train_ds)}, Val dataset size {len(val_ds)}")

    def params_for_ds(ds, batch_size, resume_step):
        sampler = StatefulBatchsampler(
            ds,
            shuffle=datasets_config["shuffle"],
            seed=global_config["seed"],
            batch_size=datasets_config["batch_size"],
        )
        # sampler.set_epoch_and_start_iteration(epoch=epoch, start_iteration=0)
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
            resume_step=optim_config["resume_step"],
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        **params_for_ds(
            val_ds, batch_size=datasets_config["batch_size"], resume_step=0
        ),
    )

    return train_dataloader, val_dataloader


def load_optimizer(optim_config, params):
    return torch.optim.AdamW(
        params,
        lr=optim_config["learning_rate"],
        weight_decay=optim_config["weight_decay"],
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

    raise ValueError


def uniform_preds(batch):
    return (
        torch.ones(*batch["empirical"].shape, device=batch["empirical"].device)
        / batch["empirical"].shape[-1]
    )


def target_from_scatter(batch, y_rad_binned, sigma, k):
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
    }


class SimpleLogger:
    def __init__(self, logger_config, full_config):
        pass

    def log(self, data, step, prefix=""):
        for key, value in data.items():
            if "plot" not in key:
                print(f"{step}:{prefix}{key}{np.array(value).mean()}")
            else:
                plt.close(value)


class WNBLogger:
    def __init__(self, logger_config, full_config):
        wandb.init(
            # set the wandb project where this run will be logged
            project=logger_config["project"],
            # track hyperparameters and run metadata
            config=full_config,
            # name=args.wandb_name,
            # id="d7i47byn",  # "iconic-fog-63",
        )

    def log(self, data, step, prefix=""):
        losses = {
            f"{prefix}{key}": np.array(value).mean()
            for key, value in data.items()
            if "plot" not in key and len(value) > 0
        }
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


def compute_loss(output, batch_data, datasets_config, loss_fn, plot=False):
    loss_d = {}
    loss = 0

    fig = None
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(6, 9))
        n = output["single"].shape[0] * output["single"].shape[1]
        d = output["single"].shape[2]
        show_n = min(n, 10)

    if "single" in output:
        target = target_from_scatter(
            batch_data,
            y_rad_binned=batch_data["y_rad_binned"][..., None],
            sigma=datasets_config["sigma"],
            k=datasets_config["scatter_k"],
        )
        loss_d["single_loss"] = loss_fn(output["single"], target)
        loss += loss_d["single_loss"]

        if plot:
            im = np.zeros((show_n * 2, d))
            im[1::2] = output["single"].reshape(n, -1).cpu().detach().numpy()[:show_n]
            im[::2] = target.reshape(n, -1).cpu().detach().numpy()[:show_n]
            axs[0].imshow(im)
            axs[0].set_title("Single")

    if "paired" in output or "multipaired" in output:
        paired_target = target_from_scatter(
            batch_data,
            y_rad_binned=batch_data["craft_y_rad_binned"][..., None],
            sigma=datasets_config["sigma"],
            k=datasets_config["scatter_k"],
        )
    if "paired" in output:
        loss_d["paired_loss"] = loss_fn(output["paired"], paired_target)
        loss += loss_d["paired_loss"]
        if plot:
            im = np.zeros((show_n * 2, d))
            im[1::2] = output["paired"].reshape(n, -1).cpu().detach().numpy()[:show_n]
            im[::2] = paired_target.reshape(n, -1).cpu().detach().numpy()[:show_n]
            axs[1].imshow(im)
            axs[1].set_title("Paired")
    if "multipaired" in output:
        loss_d["multipaired_loss"] = loss_fn(output["multipaired"], paired_target)
        loss += loss_d["multipaired_loss"]
        if plot:
            im = np.zeros((show_n * 2, d))
            im[1::2] = (
                output["multipaired"].reshape(n, -1).cpu().detach().numpy()[:show_n]
            )
            im[::2] = paired_target.reshape(n, -1).cpu().detach().numpy()[:show_n]
            axs[2].imshow(im)
            axs[2].set_title("MultiPaired")
    loss_d["loss"] = loss
    return loss_d, fig


def train_single_point(args):
    config = load_config_from_fn(args.config)
    print(config)

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

    train_dataloader, val_dataloader = load_dataloaders(
        config["datasets"], config["optim"], config["global"]
    )

    if "logger" not in config or config["logger"]["name"] == "simple":
        logger = SimpleLogger(config["logger"], config)
    elif config["logger"]["name"] == "wandb":
        logger = WNBLogger(config["logger"], config)

    m = load_model(config["model"], config["global"]).to(config["optim"]["device"])
    optimizer = load_optimizer(config["optim"], m.parameters())
    # scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.05, threshold=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=500)
    scheduler = StepLR(
        optimizer,
        step_size=config["optim"].get("scheduler_step", 1),
        gamma=0.5,
        verbose=True,
    )

    step = 0
    assert config["optim"]["resume_step"] == 0

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

    for epoch in range(config["optim"]["epochs"]):
        train_dataloader.batch_sampler.set_epoch_and_start_iteration(
            epoch=epoch, start_iteration=step % len(train_dataloader)
        )

        for _, batch_data in enumerate(tqdm(train_dataloader)):
            if step % config["optim"]["val_every"] == 0:
                m.eval()
                with torch.no_grad():
                    val_losses = new_log()
                    # for  in val_dataloader:
                    print("Running validation:")
                    for _, val_batch_data in enumerate(
                        tqdm(val_dataloader, leave=False)
                    ):
                        # for val_batch_data in val_dataloader:
                        val_batch_data = val_batch_data.to(config["optim"]["device"])

                        # run beamformer and segmentation
                        output = m(val_batch_data)
                        loss_d, _ = compute_loss(
                            output,
                            val_batch_data,
                            loss_fn=loss_fn,
                            datasets_config=config["datasets"],
                        )

                        # scheduler.step(loss_d["loss"])
                        # val_losses["learning_rate"] = [scheduler.get_last_lr()]

                        # TODO refactor
                        target = target_from_scatter(
                            val_batch_data,
                            y_rad_binned=val_batch_data["y_rad_binned"][..., None],
                            sigma=config["datasets"]["sigma"],
                            k=config["datasets"]["scatter_k"],
                        )
                        loss_d["passthrough_loss"] = loss_fn(
                            val_batch_data["empirical"],
                            target,
                        )
                        # fig, axs = plt.subplots(1, 1)
                        # axs.imshow(
                        #     val_batch_data["empirical"]
                        #     .reshape(512 * 2, -1)[:20]
                        #     .detach()
                        #     .numpy()
                        # )
                        # fig.savefig("test.png")
                        # breakpoint()
                        # fig, axs = plt.subplots(1, 1)
                        # axs.imshow(target.reshape(512 * 2, -1)[:20].detach().numpy())
                        # fig.savefig("test2.png")
                        # breakpoint()
                        loss_d["uniform_loss"] = loss_fn(
                            uniform_preds(val_batch_data), target
                        )

                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())
                        val_losses["epoch"].append(step / len(train_dataloader))
                        val_losses["data_seen"].append(
                            step
                            * config["datasets"]["snapshots_per_session"]
                            * config["datasets"]["batch_size"]
                        )

                    logger.log(val_losses, step=step, prefix="val_")

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
                    plot=step == 0 or (step + 1) % config["logger"]["log_every"] == 0,
                )

                if fig is not None:
                    losses["plot_pred"] = fig

                scaler.scale(loss_d["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                for key, value in loss_d.items():
                    losses[key].append(value.item())
                losses["learning_rate"].append(scheduler.get_lr()[0])

                if step == 0 or (step + 1) % config["logger"]["log_every"] == 0:

                    losses["epoch"].append(step / len(train_dataloader))
                    losses["data_seen"].append(
                        step
                        * config["datasets"]["snapshots_per_session"]
                        * config["datasets"]["batch_size"]
                    )
                    logger.log(losses, step=step, prefix="train_")
                    losses = new_log()
            step += 1
        scheduler.step()
        print("LR STEP:", scheduler.get_lr())


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
        "--beamnet-latent",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--head-start",
        type=int,
        default=10000,
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
    parser = get_parser_filter()
    args = parser.parse_args()
    train_single_point(args)
