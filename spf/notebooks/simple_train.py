import argparse
from functools import cache

import numpy as np
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb
from spf.dataset.spf_dataset import (
    v5_collate_beamsegnet,
    v5_thetas_to_targets,
    v5_collate_all_fast,
    v5spfdataset,
)
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    BeamNSegNet,
    ConvNet,
    UNet1D,
)
from spf.rf import reduce_theta_to_positive_y


def imshow_predictions_pi(
    beam_m,
    output,
    y_rad,
):
    img_beam_output = (beam_m.render_discrete_x(output) * 255).cpu().byte()

    output_dim = img_beam_output.shape[1]

    img_beam_gt = (
        (v5_thetas_to_targets(y_rad, output_dim, range_in_rad=2, sigma=0.3) * 255)
        .cpu()
        .byte()
    )

    examples_to_plot = min(128, img_beam_output.shape[0])
    train_target_image = torch.zeros(
        (examples_to_plot * 4, output_dim),
    ).byte()

    for row_idx in range(examples_to_plot):
        train_target_image[row_idx * 4] = img_beam_output[row_idx]
        train_target_image[row_idx * 4 + 1] = img_beam_gt[row_idx]

    fig, ax = plt.subplots(1, 1, figsize=(16, examples_to_plot / 2))
    ax.imshow(train_target_image, interpolation="none")

    ax.set_xticks(np.linspace(0, output_dim - 1, 5))
    ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])

    # Labels for major ticks
    ax.grid(
        which="major",
        color="w",
        linestyle="-",
        linewidth=0.5,
        axis="x",
    )
    return fig


def imshow_predictions_half_pi(
    beam_m,
    output,
    y_rad,
):
    img_beam_output = (beam_m.render_discrete_x(output) * 255).cpu().byte()

    output_dim = img_beam_output.shape[1]
    assert output_dim % 2 == 1
    full_output_dim = output_dim * 2 - 1
    half_pi_output_offset = output_dim // 2

    img_beam_gt_reduced = (
        (
            v5_thetas_to_targets(
                reduce_theta_to_positive_y(y_rad), output_dim, range_in_rad=1, sigma=0.3
            )
            * 255
        )
        .cpu()
        .byte()
    )
    img_beam_gt = (
        (v5_thetas_to_targets(y_rad, full_output_dim, range_in_rad=2, sigma=0.3) * 255)
        .cpu()
        .byte()
    )

    examples_to_plot = min(128, img_beam_output.shape[0])
    train_target_image = torch.zeros(
        (examples_to_plot * 7, full_output_dim),
    ).byte()

    for row_idx in range(examples_to_plot):
        train_target_image[
            row_idx * 7,
            half_pi_output_offset : half_pi_output_offset + output_dim,
        ] = img_beam_output[row_idx]
        train_target_image[
            row_idx * 7 + 2,
            half_pi_output_offset : half_pi_output_offset + output_dim,
        ] = img_beam_gt_reduced[row_idx]
        train_target_image[row_idx * 7 + 3] = img_beam_gt[row_idx]

    fig, ax = plt.subplots(1, 1, figsize=(16, examples_to_plot / 2))
    ax.imshow(train_target_image, interpolation="none")

    ax.set_xticks(np.linspace(0, full_output_dim - 1, 5))
    ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])

    # Labels for major ticks
    ax.grid(
        which="major",
        color="w",
        linestyle="-",
        linewidth=0.5,
        axis="x",
    )
    return fig


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
    if args.n_radios == 1 and args.latent > 0:
        raise ValueError("Cannot have latent space when n_radios==1")
    # torch.autograd.detect_anomaly()
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    torch_device = torch.device(args.device)

    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)

    assert args.n_radios in [1, 2]
    # loop over and concat datasets here
    skip_fields = set(["simple_segmentations"])
    if args.segmentation_level == "downsampled":
        skip_fields |= set(["signal_matrix"])
    datasets = [
        v5spfdataset(
            prefix,
            precompute_cache=args.precompute_cache,
            nthetas=args.nthetas,
            skip_fields=skip_fields,
            paired=args.n_radios > 1,
            ignore_qc=args.skip_qc,
            gpu=args.device == "cuda",
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
        "collate_fn": v5_collate_all_fast,
        # "collate_fn": v5_collate_beamsegnet,
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
        if args.beamformer_input:
            beam_m = BeamNetDirect(
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
                inputs=args.beamformer_ntheta,
                norm=args.norm,
                max_angle=np.pi / 2,
                linear_sigmas=args.linear_sigmas,
                correction=args.normal_correction,
                min_sigma=args.min_sigma,
            )
        else:
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
                positional_encoding=args.positional,
                latent=args.latent,
                max_angle=np.pi / 2,
                linear_sigmas=args.linear_sigmas,
                correction=args.normal_correction,
                min_sigma=args.min_sigma,
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
            max_angle=np.pi,
            linear_sigmas=args.linear_sigmas,
            correction=args.normal_correction,
            min_sigma=args.min_sigma,
        )
    elif args.type == "discrete":
        beam_m = BeamNetDiscrete(
            nthetas=args.nthetas,
            hidden=args.hidden,
            act=act,
            symmetry=args.symmetry,
            bn=args.batch_norm,
            positional_encoding=args.positional,
            latent=args.latent,
            max_angle=np.pi / 2,
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
            max_angle=np.pi,
            # positional_encoding=args.positional,
        )
    m = BeamNSegNet(
        segnet=seg_m,
        beamnet=beam_m,
        circular_mean=args.circular_mean,
        segmentation_lambda=args.segmentation_lambda,
        paired_lambda=args.paired_lambda,
        independent=args.independent,
        n_radios=args.n_radios,
        paired_net=paired_net,
        rx_spacing=args.rx_spacing,
        drop_in_gt=args.drop_in_gt,
        paired_drop_in_gt=args.paired_drop_in_gt,
        mse_lambda=args.mse_lambda,
        paired_mse_lambda=args.paired_mse_lambda,
        beamnet_lambda=args.beam_net_lambda,
        beamformer_input=args.beamformer_input,
    ).to(torch_device)

    if args.wandb_project:
        # start a new wandb run to track this script
        config = vars(args)
        config["pytorch_model"] = str(m)
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # track hyperparameters and run metadata
            config=config,
        )
    else:
        print("model:")
        print(m)

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
        windowed_beamformer = batch_data["windowed_beamformer"].to(torch_device)
        y_rad = batch_data["y_rad"].to(torch_device)
        craft_y_rad = batch_data["craft_y_rad"].to(torch_device)
        y_phi = batch_data["y_phi"].to(torch_device)
        # todo fix this properly, or only support single snapshots?
        assert seg_mask.ndim == 4 and seg_mask.shape[2] == 1 and seg_mask.shape[1] == 1
        assert x.ndim == 4 and x.shape[1] == 1
        seg_mask = seg_mask[:, 0]
        x = x[:, 0]
        return x, y_rad, craft_y_rad, y_phi, seg_mask, rx_spacing, windowed_beamformer

    step = 0
    losses = []

    def new_log():
        return {
            "loss": [],
            "segmentation_loss": [],
            "beamnet_loss": [],
            "paired_beamnet_loss": [],
            "paired_beamnet_mse_loss": [],
            "mse_loss": [],
        }

    to_log = new_log()
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
                        "beamnet_loss": [],
                        "paired_beamnet_loss": [],
                        "paired_beamnet_mse_loss": [],
                        "mse_loss": [],
                    }
                    # for  in val_dataloader:
                    print("Running validation:")
                    for _, val_batch_data in tqdm(
                        enumerate(val_dataloader),
                        total=len(val_dataloader),
                        leave=False,
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
                        ) = batch_data_to_x_y_seg(
                            val_batch_data, args.segmentation_level
                        )

                        # run beamformer and segmentation
                        output = m(
                            x=x,
                            gt_seg_mask=seg_mask,
                            rx_spacing=rx_spacing,
                            y_rad=y_rad,
                            windowed_beam_former=windowed_beamformer,
                        )

                        # compute the loss
                        loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask)

                        # for accumulaing and averaging
                        for key, value in loss_d.items():
                            val_losses[key].append(value.item())

                        # plot the first batch
                        if "val_unpaired_output" not in to_log:
                            to_log["val_unpaired_output"] = imshow_predictions_half_pi(
                                beam_m, output["pred_theta"], y_rad
                            )

                            if "paired_pred_theta" in output:
                                to_log["val_paired_output"] = imshow_predictions_pi(
                                    paired_net,
                                    output["paired_pred_theta"],
                                    craft_y_rad[::2],
                                )

                    if args.wandb_project:
                        wandb.log(
                            {
                                f"val_{key}": np.array(value).mean()
                                for key, value in val_losses
                            },
                            step=step,
                        )
                        for key, value in to_log.items():
                            if "_output" in key:
                                plt.close(value)

            m.train()

            optimizer.zero_grad()

            x, y_rad, craft_y_rad, y_phi, seg_mask, rx_spacing, windowed_beamformer = (
                batch_data_to_x_y_seg(batch_data, args.segmentation_level)
            )

            output = m(
                x=x,
                gt_seg_mask=seg_mask,
                rx_spacing=rx_spacing,
                y_rad=y_rad,
                windowed_beam_former=windowed_beamformer,
            )
            loss_d = m.loss(output, y_rad, craft_y_rad, seg_mask)

            loss = loss_d["loss"]
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                # accumulate the loss for returning to caller
                losses.append(loss_d["loss"].item())

                # for accumulaing and averaging
                for key, value in loss_d.items():
                    to_log[key].append(value.item())

                if step == 0 or (step + 1) % args.plot_every == 0:
                    to_log["unpaired_output"] = imshow_predictions_half_pi(
                        beam_m, output["pred_theta"], y_rad
                    )
                    if "paired_pred_theta" in output:
                        to_log["paired_output"] = imshow_predictions_pi(
                            paired_net, output["paired_pred_theta"], craft_y_rad[::2]
                        )

                    # segmentation output
                    _x = x.detach().cpu().numpy()
                    _seg_mask = seg_mask.detach().cpu().numpy()
                    _output_seg = output["segmentation"].detach().cpu().numpy()

                    to_log["fig"] = plot_instance(
                        _x, _output_seg, _seg_mask, idx=0, first_n=first_n
                    )
                if args.wandb_project and (
                    step == 0 or (step + 1) % args.log_every == 0
                ):
                    for key, value in to_log.items():
                        if "loss" in key and len(value) > 0:
                            to_log[key] = np.array(value).mean()
                    wandb.log(to_log, step=step)
                    for key, value in to_log.items():
                        if "_output" in key:
                            plt.close(value)
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
        "--beam-net-lambda",
        type=float,
        required=False,
        default=1.0,
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
        "--mse-lambda",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--paired-mse-lambda",
        type=float,
        required=False,
        default=0.0,
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
        "--latent",
        type=int,
        default=0,
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
        "--linear-sigmas",
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
        "--beamformer-input",
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
        "--normal-correction",
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
    parser.add_argument(
        "--paired-drop-in-gt",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--drop-in-gt",
        type=float,
        default=0.0,
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
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    simple_train(args)
