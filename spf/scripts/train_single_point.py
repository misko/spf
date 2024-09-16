import argparse
import glob
import random
from functools import partial

import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import v5_collate_keys_fast, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import FFNN
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
            "rx_pos_xy",
            "tx_pos_xy",
            "downsampled_segmentation_mask",
            "rx_spacing",
            "y_rad",
            "craft_y_rad",
            "y_phi",
            "system_timestamp",
            "empirical",
            "y_rad_binned",
            "craft_y_rad_binned",
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


class SinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.net = FFNN(
            inputs=global_config["nthetas"] * 2,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=global_config["nthetas"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            bn=model_config["bn"],  # False , bool
        )
        self.paired = False

    def forward(self, batch):

        # x = self.net(
        #     torch.concatenate([batch["empirical"], weighted_beamformer(batch)], dim=2)
        # )
        # breakpoint()
        x = self.net(
            torch.concatenate(
                [batch["empirical"], batch["weighted_beamformer"] / 256], dim=2
            )
        )
        # first dim odd / even is the radios
        return torch.nn.functional.softmax(x, dim=2)


class PairedSinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.single_radio_net = SinglePointWithBeamformer(model_config, global_config)
        self.net = FFNN(
            inputs=global_config["nthetas"] * 2,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=global_config["nthetas"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            bn=model_config["bn"],  # False , bool
        )
        self.paired = True

    def forward(self, batch):
        single_radio_estimates = self.single_radio_net(batch)
        x = self.net(
            torch.concatenate(
                [single_radio_estimates[::2], single_radio_estimates[1::2]], dim=2
            )
        )
        idxs = torch.arange(x.shape[0]).reshape(-1, 1).repeat(1, 2).reshape(-1)
        # first dim odd / even is the radios
        return (
            torch.nn.functional.softmax(x[idxs], dim=2) * 0.5
            + single_radio_estimates * 0.5
        )


class SinglePointPassThrough(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.paired = False

    def forward(self, batch):
        # weighted_beamformer(batch)
        return batch["empirical"] + self.w * 0.0


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

    raise ValueError


def uniform_preds(batch):
    return (
        torch.ones(*batch["empirical"].shape, device=batch["empirical"].device)
        / batch["empirical"].shape[-1]
    )


def target_from_scatter(batch, paired, sigma, k):
    if paired:
        y_rad_binned = batch["craft_y_rad_binned"][..., None]
    else:
        y_rad_binned = batch["y_rad_binned"][..., None]

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


def discrete_loss(output, target):
    return -(target * output).sum(axis=2).mean()


def new_log():
    return {
        "loss": [],
        "epoch": [],
        "data_seen": [],
        "passthrough_loss": [],
        "uniform_loss": [],
    }


class SimpleLogger:
    def __init__(self, logger_config, full_config):
        pass

    def log(self, data, step, prefix=""):
        for key, value in data.items():
            print(f"{step}:{prefix}{key}{np.array(value).mean()}")


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
        wandb.log(
            {
                f"{prefix}{key}": np.array(value).mean()
                for key, value in data.items()
                if len(value) > 0
            },
            step=step,
        )


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

    step = 0
    assert config["optim"]["resume_step"] == 0

    scaler = torch.GradScaler(
        torch_device_str,
        enabled=config["optim"]["amp"],
    )
    losses = new_log()

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
                        target = target_from_scatter(
                            val_batch_data,
                            paired=m.paired,
                            sigma=config["datasets"]["sigma"],
                            k=config["datasets"]["scatter_k"],
                        )

                        # compute the loss
                        loss_d = {
                            "loss": discrete_loss(output, target),
                            "passthrough_loss": discrete_loss(
                                val_batch_data["empirical"], target
                            ),
                            "uniform_loss": discrete_loss(
                                uniform_preds(val_batch_data), target
                            ),
                        }
                        # breakpoint()
                        # breakpoint()
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

                target = target_from_scatter(
                    batch_data,
                    paired=m.paired,
                    sigma=config["datasets"]["sigma"],
                    k=config["datasets"]["scatter_k"],
                )

                # compute the loss
                loss_d = {
                    "loss": discrete_loss(output, target),
                }
                # print(loss_d)

                # loss = loss_d["beamnet_loss"]
                # loss.backward()

                # optimizer.step()

                scaler.scale(loss_d["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                for key, value in loss_d.items():
                    losses[key].append(value.item())

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
