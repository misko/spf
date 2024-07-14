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

# torch.autograd.set_detect_anomaly(True)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    for (
        _dataset
    ) in dataset.dataset.datasets:  # subset_dataset.concat_dataset.v5spfdataset
        _dataset.reinit()


def save_everything(model, optimizer, config, step, path):
    torch.save(
        {
            "step": step,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


@cache
def get_time_split(batch_size, snapshots_per_sessions, device):
    return (
        torch.linspace(-1, 0, snapshots_per_sessions)
        .to(device)
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
        self.l = torch.nn.Linear(3, 1).to(torch.float16)

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
        input_dim=10,
        d_model=2048,
        d_hid=512,
        dropout=0.1,
        n_heads=16,
        n_layers=24,
        output_dim=1,
        token_dropout=0.5,
    ):
        super(FunkyNet, self).__init__()
        self.z = torch.nn.Linear(10, 3)

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
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim + 5 * 2 + 1, d_model
            )  # 5 output beam_former R1+R2, time
        )
        self.output_dim = output_dim

        self.beam_m = BeamNetDirect(
            nthetas=65,
            depth=6,
            hidden=128,
            symmetry=False,
            act=torch.nn.LeakyReLU,
            other=True,
            bn=True,
            no_sigmoid=True,
            block=True,
            inputs=3,  # + (1 if args.rx_spacing else 0),
            norm="layer",
            positional_encoding=False,
            latent=0,
            max_angle=np.pi / 2,
            linear_sigmas=True,
            correction=True,
            min_sigma=0.0001,
        )  # .to(torch_device)

        self.paired_drop_in_gt = 0.00
        self.token_dropout = token_dropout

    def forward(self, x, seg_mask, rx_spacing, y_rad, windowed_beam_former, rx_pos):
        rx_pos = rx_pos.detach().clone() / 4000

        batch_size, snapshots_per_sessions = y_rad.shape
        # weighted_input = torch.mul(x, seg_mask).sum(axis=3) / (
        #     seg_mask.sum(axis=3) + 0.001
        # )
        weighted_input = (
            torch.mul(x, seg_mask) / (seg_mask.sum(axis=3, keepdim=True) + 0.001)
        ).sum(axis=3)

        pred_theta = self.beam_m(
            weighted_input.reshape(batch_size * snapshots_per_sessions, -1)
        )
        # if not pred_theta.isfinite().all():
        #     breakpoint()
        #     a = 1
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
            batch_size, snapshots_per_sessions, 3
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
                    batch_size, snapshots_per_sessions, weighted_input_by_example.device
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
            (
                y_rad_reduced
                - (torch.rand(y_rad_reduced.shape, device=target.device) - 0.5)
                * 2
                * np.pi
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


def simple_train(args):
    assert args.n_radios == 2
    # torch.autograd.detect_anomaly()
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    torch_device = torch.device(args.device)

    torch.manual_seed(args.seed)

    random.seed(args.seed)

    assert args.n_radios in [1, 2]
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

    dataloader_params = {
        "batch_size": args.batch,
        "shuffle": args.shuffle,
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
            ],
        ),
        "worker_init_fn": worker_init_fn,
        # "pin_memory": True,
        "prefetch_factor": 2 if args.workers > 0 else None,
    }
    train_dataloader = torch.utils.data.DataLoader(train_ds, **dataloader_params)
    val_dataloader = torch.utils.data.DataLoader(val_ds, **dataloader_params)

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
    m = FunkyNet().to(torch_device)
    # m = DebugFunkyNet().to(torch_device)
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
        x = batch_data["all_windows_stats"].to(torch_device)
        rx_pos = batch_data["rx_pos_xy"].to(torch_device)
        seg_mask = batch_data["downsampled_segmentation_mask"].to(torch_device)
        rx_spacing = batch_data["rx_spacing"].to(torch_device)
        windowed_beamformer = batch_data["windowed_beamformer"].to(torch_device)
        y_rad = batch_data["y_rad"].to(torch_device)
        craft_y_rad = batch_data["craft_y_rad"].to(torch_device)
        y_phi = batch_data["y_phi"].to(torch_device)
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
    for _ in range(args.epochs):
        # breakpoint()
        for step, batch_data in enumerate(
            tqdm(train_dataloader)
        ):  # , total=len(train_dataloader)):
            # if step > 200:
            #     return
            if torch.rand(1).item() < 0.05:
                gc.collect()
            if step % args.save_every == 0:
                m.eval()
                save_everything(
                    model=m,
                    optimizer=optimizer,
                    config=args,
                    step=step,
                    path=f"{args.save_prefix}_step{step}.chkpnt",
                )
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

            output = m(
                x=x,
                seg_mask=seg_mask,
                rx_spacing=rx_spacing,
                y_rad=y_rad,
                windowed_beam_former=windowed_beamformer,
                rx_pos=rx_pos,
            )
            loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask)

            loss = loss_d["loss"]
            # loss = loss_d["beamnet_loss"]
            loss.backward()

            optimizer.step()

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


def get_parser():
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
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--save-prefix", type=str, default="./this_model_")
    return parser


# from pyinstrument import Profiler
# from pyinstrument.renderers import ConsoleRenderer

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # with Profile() as profile:
    # profiler = Profiler()
    # profiler.start()
    simple_train(args)

    # session = profiler.stop()
    #    # (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(200))
    # profile_renderer = ConsoleRenderer(unicode=True, color=True, show_all=True)
    # print(profile_renderer.render(session))
