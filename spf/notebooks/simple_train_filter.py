import argparse
import gc
import math
import random
from cProfile import Profile
from functools import cache, partial
from math import ceil
from pstats import SortKey, Stats
from random import shuffle

import numpy as np
import tensordict
import torch
from matplotlib import pyplot as plt
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import BatchSampler, DistributedSampler, Sampler
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import (
    v5_collate_beamsegnet,
    v5_collate_keys_fast,
    v5spfdataset,
)
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    SimpleNet,
)
from spf.model_training_and_inference.models.funkynet1 import DebugFunkyNet, FunkyNet
from spf.rf import (
    reduce_theta_to_positive_y,
    torch_pi_norm,
    torch_pi_norm_pi,
    torch_reduce_theta_to_positive_y,
)
from spf.utils import PositionalEncoding, StatefulBatchsampler

torch.set_float32_matmul_precision("high")


# torch.autograd.set_detect_anomaly(True)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    for (
        _dataset
    ) in dataset.dataset.datasets:  # subset_dataset.concat_dataset.v5spfdataset
        _dataset.reinit()


def save_everything(model, optimizer, config, step, epoch, path):
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# @torch.compile
def batch_data_to_x_y_seg(
    batch_data: tensordict.tensordict.TensorDict,
    torch_device: torch.device,
    dtype: torch.dtype,
    beamformer_input: bool,
):
    # x ~ # trimmed_cm, trimmed_stddev, abs_signal_median
    batch_data = batch_data.to(torch_device)
    x = batch_data["all_windows_stats"].to(dtype=dtype)
    rx_pos = batch_data["rx_pos_xy"].to(dtype=dtype)
    tx_pos = batch_data["tx_pos_xy"].to(dtype=dtype)
    seg_mask = batch_data["downsampled_segmentation_mask"].to(dtype=dtype)
    rx_spacing = batch_data["rx_spacing"].to(dtype=dtype)
    if beamformer_input:
        windowed_beamformer = batch_data["windowed_beamformer"].to(dtype=dtype)
    else:
        windowed_beamformer = None
    y_rad = batch_data["y_rad"].to(dtype=dtype)
    craft_y_rad = batch_data["craft_y_rad"].to(dtype=dtype)
    y_phi = batch_data["y_phi"].to(dtype=dtype)
    timestamps = batch_data["system_timestamp"].to(dtype=dtype)
    # rx_spacing = batch_data["rx_spacing"].to(dtype=dtype)
    # assert seg_mask.ndim == 4 and seg_mask.shape[2] == 1
    return (
        x,
        y_rad,
        craft_y_rad,
        y_phi,
        seg_mask,
        rx_spacing,
        windowed_beamformer,
        rx_pos,
        timestamps,
        tx_pos,
    )


def simple_train_filter(args):
    # torch.autograd.detect_anomaly()
    assert args.n_radios == 2
    # torch.autograd.detect_anomaly()
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    torch_device = torch.device(args.device)

    torch.manual_seed(args.seed)

    random.seed(args.seed)

    assert args.n_radios in [1, 2]

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    dtype = torch.float16
    if args.dtype == "float32":
        dtype = torch.float32

    if args.act == "relu":
        act = torch.nn.ReLU
    elif args.act == "selu":
        act = torch.nn.SELU
    elif args.act == "leaky":
        act = torch.nn.LeakyReLU
    else:
        raise NotImplementedError

    # init model here
    #######
    if args.debug_model:
        m = DebugFunkyNet().to(torch_device, dtype=dtype)
    else:
        m = FunkyNet(
            args=args,
            d_hid=args.tformer_dhid,
            d_model=args.tformer_dmodel,
            dropout=args.tformer_dropout,
            token_dropout=args.tformer_snapshot_dropout,
            n_layers=args.tformer_layers,
            latent=args.beamnet_latent,
            beamformer_input=args.beamformer_input,
            include_input=args.include_input,
            only_beamnet=args.only_beamnet,
        ).to(torch_device, dtype=dtype)
    ########

    def no_change(m):
        return

    init_function = no_change

    if args.weight_init == "":
        pass
    else:
        assert 1 == 0

    # weight init
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            # print("Init", m)
            init_function(m.weight.data)
            init_function(m.bias.data)

    m.beam_m.apply(weights_init)

    #####
    if args.debug:
        args.wandb_project = None
    if args.wandb_project:
        # start a new wandb run to track this script
        config = vars(args)
        config["pytorch_model"] = str(m)
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # track hyperparameters and run metadata
            config=config,
            name=args.wandb_name,
            # id="d7i47byn",  # "iconic-fog-63",
        )
    else:
        print("model:")
        print(m)

    if args.compile:
        m = torch.compile(m)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    x[1]
                    for x in filter(
                        lambda x: "beam_m.beam_net" in x[0], m.named_parameters()
                    )
                ],
                "lr": args.lr_beamnet,
            },
            {
                "params": [
                    x[1]
                    for x in filter(
                        lambda x: "beam_m.beam_net" not in x[0], m.named_parameters()
                    )
                ],
                "lr": args.lr,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    step = 0
    losses = []

    def new_log():
        return {
            "loss": [],
            "beamnet_loss": [],
            "beamnet_mse_loss": [],
            "beamnet_mse_random_loss": [],
            "epoch": [],
            "data_seen": [],
            "transformer_mse_random_loss": [],
            "transformer_mse_loss": [],
            "all_transformer_mse_loss": [],
            "pos_transformer_mse_loss": [],
            "mm_pos_transformer_mse_loss": [],
        }

    to_log = new_log()
    step = 0
    epoch = 0
    just_loaded = False
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        m.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        config = checkpoint["config"]
        just_loaded = True
        # loop over and concat datasets here
    elif args.load_beamnet:
        checkpoint = torch.load(args.load_beamnet)
        state_dict = checkpoint["model_state_dict"]
        for key in list(state_dict.keys()):
            if "beam_m" not in key:
                state_dict.pop(key)
            else:
                state_dict[key.replace("beam_m.", "")] = state_dict.pop(key)
        m.beam_m.load_state_dict(state_dict)
    skip_fields = set(["signal_matrix", "simple_segmentations"])
    if not args.beamformer_input:
        skip_fields |= set(["windowed_beamformer"])
        print(skip_fields)
    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=args.precompute_cache,
            nthetas=args.nthetas,
            paired=args.n_radios > 1,
            ignore_qc=args.skip_qc,
            gpu=args.device == "cuda",
            snapshots_per_session=args.snapshots_per_session,
            snapshots_stride=args.snapshots_stride,
            readahead=False,
            skip_fields=skip_fields,
        )
        for prefix in args.datasets
    ]
    for ds in datasets:
        ds.get_segmentation()
    complete_ds = torch.utils.data.ConcatDataset(datasets)

    if args.val_on_train:
        train_ds = complete_ds
        val_ds = complete_ds
    else:
        n = len(complete_ds)
        train_idxs = range(int((1.0 - args.val_holdout_fraction) * n))
        val_idxs = list(range(train_idxs[-1] + 1, n))

        shuffle(val_idxs)
        val_idxs = val_idxs[: max(1, int(len(val_idxs) * args.val_subsample_fraction))]

        train_ds = torch.utils.data.Subset(complete_ds, train_idxs)
        val_ds = torch.utils.data.Subset(complete_ds, val_idxs)
    print(f"Train-dataset size {len(train_ds)}, Val dataset size {len(val_ds)}")

    def params_for_ds(ds, batch_size, resume_step):
        sampler = StatefulBatchsampler(
            ds, shuffle=args.shuffle, seed=args.seed, batch_size=batch_size
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
        ]
        if args.beamformer_input:
            keys_to_get += ["windowed_beamformer"]
        return {
            # "batch_size": args.batch,
            "num_workers": args.workers,
            "collate_fn": partial(v5_collate_keys_fast, keys_to_get),
            "worker_init_fn": worker_init_fn,
            # "pin_memory": True,
            "prefetch_factor": 2 if args.workers > 0 else None,
            "batch_sampler": sampler,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, **params_for_ds(train_ds, batch_size=args.batch, resume_step=step)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, **params_for_ds(val_ds, batch_size=args.batch, resume_step=0)
    )
    for epoch in range(args.epochs):
        train_dataloader.batch_sampler.set_epoch_and_start_iteration(
            epoch=epoch, start_iteration=step % len(train_dataloader)
        )
        if args.steps >= 0 and step >= args.steps:
            break

        for _, batch_data in enumerate(
            tqdm(train_dataloader)
        ):  # , total=len(train_dataloader)):
            # if step > 200:
            #     return
            if args.steps >= 0 and step >= args.steps:
                break
            # if torch.rand(1).item() < 0.002:
            #     gc.collect()
            if just_loaded is False and step % args.save_every == 0:
                m.eval()
                save_everything(
                    model=m,
                    optimizer=optimizer,
                    config=args,
                    step=step,
                    epoch=epoch,
                    path=f"{args.save_prefix}_step{step}.chkpnt",
                )
            just_loaded = False
            # breakpoint()
            if step % args.val_every == 0:
                beamnet_params = [
                    x[1].reshape(-1)
                    for x in filter(
                        lambda x: "beam_m.beam_net" in x[0],
                        m.named_parameters(),
                    )
                ]
                if len(beamnet_params) > 0:
                    beamnet_checksum = torch.hstack(beamnet_params).mean().item()
                    print("Beamnet checksum:", beamnet_checksum)
                    if not math.isfinite(beamnet_checksum):
                        if args.wandb_project:
                            wandb.finish()
                        return {"losses": losses}
                m.eval()
                with torch.no_grad():
                    val_losses = new_log()
                    # for  in val_dataloader:
                    print("Running validation:")
                    for _, val_batch_data in enumerate(
                        tqdm(val_dataloader, leave=False)
                    ):
                        # for val_batch_data in val_dataloader:
                        (
                            x,
                            y_rad,
                            craft_y_rad,
                            y_phi,
                            seg_mask,
                            rx_spacing,
                            windowed_beamformer,
                            rx_pos,
                            timestamps,
                            tx_pos,
                        ) = batch_data_to_x_y_seg(
                            val_batch_data, torch_device, dtype, args.beamformer_input
                        )

                        # run beamformer and segmentation
                        output = m(
                            x=x,
                            seg_mask=seg_mask,
                            rx_spacing=rx_spacing,
                            y_rad=y_rad,
                            windowed_beam_former=windowed_beamformer,
                            rx_pos=rx_pos,
                            timestamps=timestamps,
                        )

                        # compute the loss
                        loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask, tx_pos)
                        # breakpoint()
                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())
                        val_losses["epoch"].append(step / len(train_dataloader))
                        val_losses["data_seen"].append(
                            step * args.snapshots_per_session * args.batch
                        )

                        # plot the first batch
                        if "val_unpaired_output" not in to_log:
                            pass

                    if args.wandb_project:
                        wandb.log(
                            {
                                f"val_{key}": np.array(value).mean()
                                for key, value in val_losses.items()
                            },
                            step=step,
                            # epoch=epoch,
                        )
                        for key, value in to_log.items():
                            if "_output" in key:
                                plt.close(value)

            m.train()

            optimizer.zero_grad()

            (
                x,
                y_rad,
                craft_y_rad,
                y_phi,
                seg_mask,
                rx_spacing,
                windowed_beamformer,
                rx_pos,
                timestamps,
                tx_pos,
            ) = batch_data_to_x_y_seg(
                batch_data, torch_device, dtype, beamformer_input=args.beamformer_input
            )

            with torch.autocast(
                device_type=args.device, dtype=torch.float16, enabled=args.amp
            ):
                output = m(
                    x=x,
                    seg_mask=seg_mask,
                    rx_spacing=rx_spacing,
                    y_rad=y_rad,
                    windowed_beam_former=windowed_beamformer,
                    rx_pos=rx_pos,
                    timestamps=timestamps,
                )
                loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask, tx_pos)

                loss = loss_d["beamnet_loss"] * args.beam_net_lambda
                if step > args.head_start:
                    loss += loss_d["loss"]

                # loss = loss_d["beamnet_loss"]
                # loss.backward()

                # optimizer.step()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                # accumulate the loss for returning to caller
                losses.append(loss_d["loss"].item())

                # for accumulaing and averaging
                for key, value in loss_d.items():
                    to_log[key].append(value.item())

                if step == 0 or (step + 1) % args.plot_every == 0:
                    # segmentation output
                    pass

                if args.wandb_project and (
                    step == 0 or (step + 1) % args.log_every == 0
                ):
                    for key, value in to_log.items():
                        if "loss" in key and len(value) > 0:
                            to_log[key] = np.array(value).mean()

                    to_log["epoch"] = step / len(train_dataloader)
                    to_log["data_seen"] = step * args.snapshots_per_session * args.batch
                    wandb.log(to_log, step=step)
                    # for key, value in to_log.items():
                    #    if "_output" in key:
                    #        plt.close(value)
                    to_log = new_log()
            step += 1

    # [optional] finish the wandb run, necessary in notebooks
    if args.wandb_project:
        wandb.finish()
    return {"losses": losses}


def get_parser_filter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="dataset prefixes",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--nthetas",
        type=int,
        required=False,
        default=65,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--snapshots-per-session",
        type=int,
        required=False,
        default=500,
    )
    parser.add_argument(
        "--snapshots-stride",
        type=float,
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        "--include-input",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lr-beamnet",
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=1337,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        default=3,
    )
    parser.add_argument(
        "--val-holdout-fraction",
        type=float,
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--val-subsample-fraction",
        type=float,
        required=False,
        default=0.05,
    )
    parser.add_argument(
        "--hidden",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=False,
        default=0.00001,
    )
    parser.add_argument(
        "--weight-init",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        required=False,
        default=200,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="batch",
    )
    parser.add_argument(
        "--n-radios",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--beamformer-ntheta",
        type=int,
        default=65,
    )
    parser.add_argument(
        "--act",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--beam-net-depth",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--beam-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--beam-type",
        type=str,
        default="direct",
    )

    parser.add_argument(
        "--beam-net-hidden",
        type=int,
        default=128,
    )
    parser.add_argument("--load-checkpoint", type=str, required=False, default=None)
    parser.add_argument("--load-beamnet", type=str, required=False, default=None)
    parser.add_argument(
        "--precompute-cache",
        type=str,
        required=True,
    )
    parser.add_argument("--wandb-project", type=str, required=False, default=None)
    parser.add_argument("--wandb-name", type=str, required=False, default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--beamformer-input",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--only-beamnet",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--val-on-train",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-qc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--batch-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rx-spacing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--positional",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tformer-layers",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--beam-net-lambda",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--beam-norm-type",
        type=str,
        default="layer",
    )
    parser.add_argument(
        "--tformer-dmodel",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--tformer-dhid",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
    )
    parser.add_argument(
        "--tformer-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tformer-snapshot-dropout",
        type=float,
        default=0.5,
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


# from pyinstrument import Profiler
# from pyinstrument.renderers import ConsoleRenderer

if __name__ == "__main__":
    parser = get_parser_filter()
    args = parser.parse_args()
    # with Profile() as profile:
    # profiler = Profiler()
    # profiler.start()
    simple_train_filter(args)

    # session = profiler.stop()
    #    # (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(200))
    # profile_renderer = ConsoleRenderer(unicode=True, color=True, show_all=True)
    # print(profile_renderer.render(session))
