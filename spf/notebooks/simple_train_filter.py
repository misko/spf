import argparse
from functools import cache, partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from random import shuffle
import wandb
from spf.dataset.spf_dataset import (
    v5_collate_beamsegnet,
    v5_collate_keys_fast,
    v5spfdataset,
)
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
)
import gc
from cProfile import Profile
from pstats import SortKey, Stats

from spf.rf import reduce_theta_to_positive_y, torch_pi_norm

import random

from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    LayerNorm,
)

from torch.utils.data import DistributedSampler, Sampler, BatchSampler


# from fair-chem repo
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset=dataset, **kwargs)

        self.start_iter = 0
        self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size not set for the sampler"
        # logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # TODO: For very large datasets, even virtual datasets this might slow down
        # or not work correctly. The issue is that we enumerate the full list of all
        # samples in a single epoch, and manipulate this list directly. A better way
        # of doing this would be to keep this sequence strictly as an iterator
        # that stores the current state (instead of the full sequence)
        distributed_sampler_sequence = super().__iter__()
        if self.start_iter > 0:
            for i, _ in enumerate(distributed_sampler_sequence):
                if i == self.start_iter * self.batch_size - 1:
                    break
        return distributed_sampler_sequence

    def set_epoch_and_start_iteration(self, epoch, start_iter):
        self.set_epoch(epoch)
        self.start_iter = start_iter


class StatefulBatchsampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0, shuffle=False):
        self.single_sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            drop_last=False,
            batch_size=batch_size,
            seed=seed,
        )
        self.batch_sampler = BatchSampler(
            self.single_sampler,
            batch_size,
            drop_last=False,
        )

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int) -> None:
        self.single_sampler.set_epoch_and_start_iteration(epoch, start_iteration)

    def __iter__(self):
        yield from self.batch_sampler


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


@cache
def get_time_split(batch_size, snapshots_per_sessions, device, dtype):
    return (
        torch.linspace(-1, 0, snapshots_per_sessions, device=device, dtype=dtype)
        .reshape(1, -1, 1)
        .expand(batch_size // 2, snapshots_per_sessions, 1)
    )


class DebugFunkyNet(torch.nn.Module):
    def __init__(
        self,
        input_dim=10,
        d_model=2048,
        d_hid=512,
        dropout=0.1,
        n_heads=16,
        n_layers=24,
        output_dim=1,
        token_dropout=0.0,
    ):
        super(DebugFunkyNet, self).__init__()
        self.l = torch.nn.Linear(3, 1).to(torch.float32)

    def forward(self, x, seg_mask, rx_spacing, y_rad, windowed_beam_former, rx_pos):
        return {
            "transformer_output": None,
            "pred_theta": None,
            "fake_out": self.l(x[:, 0, 0, :3]),
        }

    def loss(self, output, y_rad, craft_y_rad, seg_mask):
        loss = output["fake_out"].mean()
        return {
            "loss": loss,
            "transformer_mse_loss": loss,
            "beamnet_loss": loss,
            "beamnet_mse_loss": loss,
            "beamnet_mse_random_loss": loss,
            "transformer_mse_random_loss": loss,
        }


class FunkyNet(torch.nn.Module):
    def __init__(
        self,
        d_model=2048,
        d_hid=512,
        dropout=0.1,
        n_heads=16,
        n_layers=24,
        output_dim=1,
        token_dropout=0.5,
        latent=0,
        beamformer_input=False,
    ):
        super(FunkyNet, self).__init__()

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            n_layers,
            LayerNorm(d_model),
        )
        self.beamformer_input = beamformer_input
        self.output_dim = output_dim

        if beamformer_input:
            self.beam_m = BeamNetDirect(
                nthetas=65,
                depth=args.beam_net_depth,
                hidden=args.beam_net_hidden,
                symmetry=False,
                act=torch.nn.LeakyReLU,
                other=True,
                bn=False,  # True
                no_sigmoid=True,
                block=True,
                rx_spacing_track=-1,
                pd_track=-1,
                mag_track=-1,
                stddev_track=-1,
                inputs=65,
                latent=latent,
                norm="layer",
                max_angle=np.pi / 2,
                linear_sigmas=True,
                correction=True,
                min_sigma=0.0001,
            )

            input_dim = (65 + 2) * 2
            # (65,65,2,2) # 65 for R0 signal, 65 for R1 signal, 2 for pos0, 2 for pos1
            self.input_net = torch.nn.Sequential(
                torch.nn.Linear(
                    input_dim + (5 + latent) * 2 + 1, d_model
                )  # 5 output beam_former R1+R2, time
            )

        else:
            self.beam_m = BeamNetDirect(
                nthetas=65,
                depth=args.beam_net_depth,
                hidden=args.beam_net_hidden,
                symmetry=False,
                act=torch.nn.LeakyReLU,
                other=True,
                bn=True,
                no_sigmoid=True,
                block=True,
                inputs=3,  # + (1 if args.rx_spacing else 0),
                norm="layer",
                positional_encoding=False,
                latent=latent,
                max_angle=np.pi / 2,
                linear_sigmas=True,
                correction=True,
                min_sigma=0.0001,
            )  # .to(torch_device)
            input_dim = 10  # (3,3,2,2) # 3 for R0 signal, 3 for R1 signal, 2 for pos0, 2 for pos1
            self.input_net = torch.nn.Sequential(
                torch.nn.Linear(
                    input_dim + (5 + latent) * 2 + 1, d_model
                )  # 5 output beam_former R1+R2, time
            )

        self.paired_drop_in_gt = 0.00
        self.token_dropout = token_dropout

    def forward(self, x, seg_mask, rx_spacing, y_rad, windowed_beam_former, rx_pos):
        rx_pos = rx_pos.detach().clone() / 4000

        batch_size, snapshots_per_sessions = y_rad.shape
        # weighted_input = torch.mul(x, seg_mask).sum(axis=3) / (
        #     seg_mask.sum(axis=3) + 0.001
        # )
        if self.beamformer_input:
            windowed_beam_former_scaled = windowed_beam_former / (
                seg_mask.sum(axis=3, keepdim=True) + 0.1
            )
            weighted_input = (
                windowed_beam_former_scaled * seg_mask[:, 0][..., None]
            ).mean(axis=2)

        else:
            weighted_input = (
                torch.mul(x, seg_mask) / (seg_mask.sum(axis=3, keepdim=True) + 0.001)
            ).sum(axis=3)
            # weighted_input ~ (batch_size*2,1,3)

        pred_theta = self.beam_m(
            weighted_input.reshape(batch_size * snapshots_per_sessions, -1)
        )
        # if not pred_theta.isfinite().all():
        #    breakpoint()
        #    a = 1
        detached_pred_theta = torch.hstack(
            [
                pred_theta[:, : self.beam_m.outputs - self.beam_m.latent].detach(),
                pred_theta[:, self.beam_m.outputs - self.beam_m.latent :],
            ]
        )
        # TODO inject correct means sometimes!
        # if self.train() randomly inject
        # if self.eval() never inject!
        if y_rad is not None and self.training and self.paired_drop_in_gt > 0.0:
            y_rad_reduced = reduce_theta_to_positive_y(y_rad).reshape(-1, 1)
            mask = torch.rand(detached_pred_theta.shape[0]) < self.paired_drop_in_gt
            detached_pred_theta[mask, 0] = y_rad_reduced[mask, 0]
            detached_pred_theta[mask, 1:3] = 0
        detached_pred_theta = detached_pred_theta.reshape(
            batch_size, snapshots_per_sessions, -1
        )

        weighted_input_by_example = weighted_input.reshape(
            batch_size, snapshots_per_sessions, weighted_input.shape[-1]
        )
        rx_pos_by_example = rx_pos.reshape(batch_size, snapshots_per_sessions, 2)

        input = torch.concatenate(
            [
                weighted_input_by_example[::2],
                weighted_input_by_example[1::2],
                rx_pos_by_example[::2],
                rx_pos_by_example[1::2],
                detached_pred_theta[::2],
                detached_pred_theta[1::2],
                get_time_split(
                    batch_size,
                    snapshots_per_sessions,
                    weighted_input_by_example.device,
                    dtype=weighted_input_by_example.dtype,
                ),
            ],
            axis=2,
        )
        # drop out 1/4 of the sequence, except the last (element we predict on)
        if self.training:
            src_key_padding_mask = (
                torch.rand(batch_size // 2, snapshots_per_sessions, device=input.device)
                < self.token_dropout  # True here means skip
            )
            src_key_padding_mask[:, -1] = False  # True is not allowed to attend
            transformer_output = self.transformer_encoder(
                self.input_net(input), src_key_padding_mask=src_key_padding_mask
            )[:, 0, : self.output_dim]
        else:
            transformer_output = self.transformer_encoder(self.input_net(input))[
                :, 0, : self.output_dim
            ]
        return {
            "transformer_output": transformer_output,
            "pred_theta": pred_theta,
        }

    def loss(self, output, y_rad, craft_y_rad, seg_mask):
        target = craft_y_rad[::2, [-1]]
        transformer_loss = (
            torch_pi_norm(target - output["transformer_output"]) ** 2
        ).mean()

        random_target = (
            (torch.rand(target.shape, device=target.device) - 0.5) * 2 * np.pi
        )
        transformer_random_loss = (torch_pi_norm(target - random_target) ** 2).mean()

        y_rad_reduced = reduce_theta_to_positive_y(y_rad).reshape(-1, 1)
        # x to beamformer loss (indirectly including segmentation)
        beamnet_loss = -self.beam_m.loglikelihood(
            output["pred_theta"], y_rad_reduced
        ).mean()

        beamnet_mse = self.beam_m.mse(output["pred_theta"], y_rad_reduced)
        beamnet_mse_random = (
            torch_pi_norm(
                y_rad_reduced
                - (torch.rand(y_rad_reduced.shape, device=target.device) - 0.5)
                * 2
                * np.pi
                / 2,
                max_angle=torch.pi / 2,
            )
            ** 2
        ).mean()
        loss = transformer_loss + beamnet_loss
        return {
            "loss": loss,
            "transformer_mse_loss": transformer_loss,
            "beamnet_loss": beamnet_loss,
            "beamnet_mse_loss": beamnet_mse,
            "beamnet_mse_random_loss": beamnet_mse_random,
            "transformer_mse_random_loss": transformer_random_loss,
        }


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
            d_hid=args.tformer_dhid,
            d_model=args.tformer_dmodel,
            dropout=args.tformer_dropout,
            token_dropout=args.tformer_snapshot_dropout,
            n_layers=args.tformer_layers,
            latent=args.beamnet_latent,
            beamformer_input=args.beamformer_input,
        ).to(torch_device, dtype=dtype)
    ########

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
        m.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def batch_data_to_x_y_seg(batch_data):
        # x ~ # trimmed_cm, trimmed_stddev, abs_signal_median
        x = batch_data["all_windows_stats"].to(torch_device, dtype=dtype)
        rx_pos = batch_data["rx_pos_xy"].to(torch_device, dtype=dtype)
        seg_mask = batch_data["downsampled_segmentation_mask"].to(
            torch_device, dtype=dtype
        )
        rx_spacing = batch_data["rx_spacing"].to(torch_device, dtype=dtype)
        windowed_beamformer = batch_data["windowed_beamformer"].to(
            torch_device, dtype=dtype
        )
        y_rad = batch_data["y_rad"].to(torch_device, dtype=dtype)
        craft_y_rad = batch_data["craft_y_rad"].to(torch_device, dtype=dtype)
        y_phi = batch_data["y_phi"].to(torch_device, dtype=dtype)
        assert seg_mask.ndim == 4 and seg_mask.shape[2] == 1
        return (
            x,
            y_rad,
            craft_y_rad,
            y_phi,
            seg_mask,
            rx_spacing,
            windowed_beamformer,
            rx_pos,
        )

    step = 0
    losses = []

    def new_log():
        return {
            "loss": [],
            "transformer_mse_loss": [],
            "beamnet_loss": [],
            "beamnet_mse_loss": [],
            "beamnet_mse_random_loss": [],
            "transformer_mse_random_loss": [],
        }

    to_log = new_log()
    step = 0
    epoch = 0

    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        m.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        config = checkpoint["config"]

        # loop over and concat datasets here
    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=args.precompute_cache,
            nthetas=args.nthetas,
            skip_signal_matrix=True,
            paired=args.n_radios > 1,
            ignore_qc=args.skip_qc,
            gpu=args.device == "cuda",
            snapshots_per_session=args.snapshots_per_session,
            readahead=False,
            skip_simple_segmentations=True,
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

    def params_for_ds(ds):
        sampler = StatefulBatchsampler(
            ds, shuffle=args.shuffle, seed=args.seed, batch_size=args.batch
        )
        sampler.set_epoch_and_start_iteration(epoch=epoch, start_iteration=step)
        return {
            # "batch_size": args.batch,
            "num_workers": args.workers,
            "collate_fn": partial(
                v5_collate_keys_fast,
                [
                    "all_windows_stats",
                    "rx_pos_xy",
                    "downsampled_segmentation_mask",
                    "rx_spacing",
                    "windowed_beamformer",
                    "y_rad",
                    "craft_y_rad",
                    "y_phi",
                    "windowed_beamformer",
                ],
            ),
            "worker_init_fn": worker_init_fn,
            # "pin_memory": True,
            "prefetch_factor": 2 if args.workers > 0 else None,
            "batch_sampler": sampler,
        }

    train_dataloader = torch.utils.data.DataLoader(train_ds, **params_for_ds(train_ds))
    val_dataloader = torch.utils.data.DataLoader(val_ds, **params_for_ds(val_ds))

    for epoch in range(args.epochs):
        # breakpoint()
        if args.steps >= 0 and step >= args.steps:
            break

        for _, batch_data in enumerate(
            tqdm(train_dataloader)
        ):  # , total=len(train_dataloader)):
            # if step > 200:
            #     return
            if args.steps >= 0 and step >= args.steps:
                break
            if torch.rand(1).item() < 0.02:
                gc.collect()
            if step % args.save_every == 0:
                m.eval()
                save_everything(
                    model=m,
                    optimizer=optimizer,
                    config=args,
                    step=step,
                    epoch=epoch,
                    path=f"{args.save_prefix}_step{step}.chkpnt",
                )
            # breakpoint()
            if step % args.val_every == 0:
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
                        ) = batch_data_to_x_y_seg(val_batch_data)

                        # run beamformer and segmentation
                        output = m(
                            x=x,
                            seg_mask=seg_mask,
                            rx_spacing=rx_spacing,
                            y_rad=y_rad,
                            windowed_beam_former=windowed_beamformer,
                            rx_pos=rx_pos,
                        )

                        # compute the loss
                        loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask)
                        # breakpoint()
                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())

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
            ) = batch_data_to_x_y_seg(batch_data)

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
                )
                loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask)

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
        default=1000,
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
        "--beam-net-hidden",
        type=int,
        default=128,
    )
    parser.add_argument("--load-checkpoint", type=str, required=False, default=None)
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
        default="float16",
    )
    parser.add_argument(
        "--tformer-dropout",
        type=int,
        default=0.1,
    )
    parser.add_argument(
        "--tformer-snapshot-dropout",
        type=int,
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
