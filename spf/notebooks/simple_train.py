import argparse
from functools import cache

import numpy as np
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import v5_collate_beamsegnet, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    BeamNSegNet,
    ConvNet,
    UNet1D,
)
from spf.rf import reduce_theta_to_positive_y


def plot_instance(_x, _output_seg, _seg_mask, idx, first_n):
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))
    s = 0.3
    idx = 0
    axs[0].set_title("input (track 0/1)")
    axs[0].scatter(range(first_n), _x[idx, 0, :first_n], s=s)
    axs[0].scatter(range(first_n), _x[idx, 1, :first_n], s=s)
    axs[1].set_title("input (track 2)")
    axs[1].scatter(range(first_n), _x[idx, 2, :first_n], s=s)
    # mw = mask_weights.cpu().detach().numpy()

    axs[2].set_title("output vs gt")
    axs[2].scatter(range(first_n), _output_seg[idx, 0, :first_n], s=s)
    axs[2].scatter(range(first_n), _seg_mask[idx, 0, :first_n], s=s)
    return fig


def simple_train(args):
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    torch_device = torch.device(args.device)

    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)

    assert args.n_radios in [1, 2]
    # loop over and concat datasets here
    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache="/home/mouse9911/precompute_cache",
            nthetas=args.nthetas,
            skip_signal_matrix=args.segmentation_level == "downsampled",
            paired=args.n_radios > 1,
            ignore_qc=args.skip_qc,
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
        val_idxs = range(train_idxs[-1] + 1, n)

        train_ds = torch.utils.data.Subset(complete_ds, train_idxs)
        val_ds = torch.utils.data.Subset(complete_ds, val_idxs)
    print(f"Train-dataset size {len(train_ds)}, Val dataset size {len(val_ds)}")

    dataloader_params = {
        "batch_size": args.batch,
        "shuffle": args.shuffle,
        "num_workers": args.workers,
        "collate_fn": v5_collate_beamsegnet,
    }
    train_dataloader = torch.utils.data.DataLoader(train_ds, **dataloader_params)
    val_dataloader = torch.utils.data.DataLoader(val_ds, **dataloader_params)

    if args.wandb_project:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # track hyperparameters and run metadata
            config=args,
        )

    if args.act == "relu":
        act = torch.nn.ReLU
    elif args.act == "selu":
        act = torch.nn.SELU
    elif args.act == "leaky":
        act = torch.nn.LeakyReLU
    else:
        raise NotImplementedError

    if args.segmentation_level == "full":
        first_n = 10000
        seg_m = UNet1D(bn=args.batch_norm).to(torch_device, act=act)
    elif args.segmentation_level == "downsampled":
        first_n = 256
        if args.seg_net == "conv":
            seg_m = ConvNet(3, 1, hidden=args.hidden, act=act, bn=args.batch_norm).to(
                torch_device
            )
        elif args.seg_net == "unet":
            seg_m = UNet1D(step=4, act=act, hidden=args.hidden, bn=args.batch_norm).to(
                torch_device
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.type == "direct":
        beam_m = BeamNetDirect(
            nthetas=args.nthetas,
            depth=args.depth,
            hidden=args.hidden,
            symmetry=args.symmetry,
            act=act,
            other=args.other,
            bn=args.batch_norm,
            no_sigmoid=not args.sigmoid,
            block=args.block,
            inputs=3,  # + (1 if args.rx_spacing else 0),
            norm=args.norm,
        ).to(torch_device)
        paired_net = BeamNetDirect(
            nthetas=args.nthetas,
            depth=args.depth,
            hidden=args.hidden,
            symmetry=False,
            act=act,
            other=args.other,
            bn=args.batch_norm,
            no_sigmoid=not args.sigmoid,
            block=args.block,
            rx_spacing_track=-1,
            pd_track=-1,
            mag_track=-1,
            stddev_track=-1,
            inputs=args.n_radios * beam_m.outputs,
            norm=args.norm,
        )
    elif args.type == "discrete":
        beam_m = BeamNetDiscrete(
            nthetas=args.nthetas,
            hidden=args.hidden,
            act=act,
            symmetry=args.symmetry,
            bn=args.batch_norm,
        ).to(torch_device)
        paired_net = BeamNetDiscrete(
            nthetas=args.nthetas,
            depth=args.depth,
            hidden=args.hidden,
            symmetry=False,
            act=act,
            other=args.other,
            bn=args.batch_norm,
            no_sigmoid=not args.sigmoid,
            block=args.block,
            rx_spacing_track=-1,
            pd_track=-1,
            mag_track=-1,
            stddev_track=-1,
            inputs=args.n_radios * beam_m.outputs,
            norm=args.norm,
        )
    m = BeamNSegNet(
        segnet=seg_m,
        beamnet=beam_m,
        circular_mean=args.circular_mean,
        segmentation_lambda=args.segmentation_lambda,
        independent=args.independent,
        n_radios=args.n_radios,
        paired_net=paired_net,
        rx_spacing=args.rx_spacing,
    ).to(torch_device)

    if args.compile:
        m = torch.compile(m)

    optimizer = torch.optim.AdamW(
        m.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def batch_data_to_x_y_seg(batch_data, segmentation_level):
        if segmentation_level == "full":
            x = batch_data["x"].to(torch_device)
            seg_mask = batch_data["segmentation_mask"].to(torch_device)
        elif segmentation_level == "downsampled":
            x = batch_data["all_windows_stats"].to(torch_device).to(torch.float32)
            seg_mask = batch_data["downsampled_segmentation_mask"].to(torch_device)
        else:
            raise NotImplementedError

        rx_spacing = batch_data["rx_spacing"].to(torch_device)

        y_rad = batch_data["y_rad"].to(torch_device)
        assert seg_mask.ndim == 3 and seg_mask.shape[1] == 1
        return x, y_rad, seg_mask, rx_spacing

    step = 0
    losses = []
    to_log = None
    for _ in range(args.epochs):
        for _, batch_data in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            if step % args.val_every == 0:
                m.eval()
                with torch.no_grad():
                    val_losses = {
                        "loss": [],
                        "segmentation_loss": [],
                        "beamformer_loss": [],
                        "paired_beamformer_loss": [],
                    }
                    # for  in val_dataloader:
                    print("Running validation:")
                    for _, val_batch_data in tqdm(
                        enumerate(val_dataloader),
                        total=len(val_dataloader),
                        leave=False,
                    ):
                        # for val_batch_data in val_dataloader:
                        x, y_rad, seg_mask, rx_spacing = batch_data_to_x_y_seg(
                            val_batch_data, args.segmentation_level
                        )

                        y_rad_reduced = reduce_theta_to_positive_y(y_rad)
                        # run beamformer and segmentation
                        output = m(x, seg_mask, rx_spacing)

                        # compute the loss
                        loss_d = m.loss(output, y_rad_reduced, seg_mask)

                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())
                    if args.wandb_project:
                        wandb.log(
                            {f"val_{key}": value for key, value in loss_d.items()},
                            step=step,
                        )

            m.train()

            if to_log is None:
                to_log = {
                    "loss": [],
                    "segmentation_loss": [],
                    "beamformer_loss": [],
                    "paired_beamformer_loss": [],
                }

            optimizer.zero_grad()

            x, y_rad, seg_mask, rx_spacing = batch_data_to_x_y_seg(
                batch_data, args.segmentation_level
            )
            y_rad_reduced = reduce_theta_to_positive_y(y_rad)

            output = m(x, seg_mask, rx_spacing)

            loss_d = m.loss(output, y_rad_reduced, seg_mask)

            loss = loss_d["beamformer_loss"]
            if step > args.seg_start:
                loss += loss_d["segmentation_loss"] * args.segmentation_lambda
            if args.n_radios > 1 and step > args.paired_start:
                loss += loss_d["paired_beamformer_loss"] * args.paired_lambda

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                # accumulate the loss for returning to caller
                losses.append(loss_d["loss"].item())

                # for accumulaing and averaging
                for key, value in loss_d.items():
                    to_log[key].append(value.item())

                if step % args.plot_every == 0:
                    img_beam_output = (
                        (beam_m.render_discrete_x(output["pred_theta"]) * 255)
                        .cpu()
                        .byte()
                    )
                    img_beam_gt = (beam_m.render_discrete_y(y_rad) * 255).cpu().byte()
                    train_target_image = torch.zeros(
                        (img_beam_output.shape[0] * 3, img_beam_output.shape[1]),
                    ).byte()
                    for row_idx in range(img_beam_output.shape[0]):
                        train_target_image[row_idx * 3] = img_beam_output[row_idx]
                        train_target_image[row_idx * 3 + 1] = img_beam_gt[row_idx]
                    if args.wandb_project:
                        output_image = wandb.Image(
                            train_target_image, caption="train vs target (interleaved)"
                        )
                        to_log["output"] = output_image
                    else:
                        ax, fig = plt.subplots(1, 1)
                        fig.imshow(train_target_image)

                    # segmentation output
                    _x = x.detach().cpu().numpy()
                    _seg_mask = seg_mask.detach().cpu().numpy()
                    _output_seg = output["segmentation"].detach().cpu().numpy()

                    to_log["fig"] = plot_instance(
                        _x, _output_seg, _seg_mask, idx=0, first_n=first_n
                    )
                if args.wandb_project and step % args.log_every == 0:
                    for key, value in to_log.items():
                        if "loss" in key:
                            to_log[key] = np.array(value).mean()
                    wandb.log(to_log, step=step)
                    to_log = None

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
        default=33,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
    )
    parser.add_argument(
        "--seg-start",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--paired-start",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=False,
        default=4,
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
        "--hidden",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--segmentation-lambda",
        type=float,
        required=False,
        default=10.0,
    )
    parser.add_argument(
        "--paired-lambda",
        type=float,
        required=False,
        default=1.0,
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
        "--type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="batch",
    )
    parser.add_argument(
        "--n-radios",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--act",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seg-net",
        type=str,
        required=True,
    )
    parser.add_argument("--wandb-project", type=str, required=False, default=None)
    parser.add_argument(
        "--segmentation-level",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--circular-mean",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--val-on-train",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--block",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--other",
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
        "--independent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rx-spacing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sigmoid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    simple_train(args)
