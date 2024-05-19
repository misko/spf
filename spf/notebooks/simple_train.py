import argparse
from functools import cache

import torch
import torch.nn.functional as f
import wandb
from matplotlib import pyplot as plt

from spf.dataset.spf_dataset import v5_collate_beamsegnet, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    BeamNSegNet,
    ConvNet,
    UNet1D,
)


def plot_instance(_x, _output_seg, _seg_mask, idx):
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


if __name__ == "__main__":
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
        "--type",
        type=str,
        required=True,
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
    )
    parser.add_argument(
        "--circular-mean",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--other",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--batch-norm",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--skip-segmentation",
        action=argparse.BooleanOptionalAction,
    )
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    args = parser.parse_args()
    torch_device = torch.device(args.device)

    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)

    # loop over and concat datasets here
    datasets = [v5spfdataset(prefix, nthetas=args.nthetas) for prefix in args.datasets]
    for ds in datasets:
        ds.get_segmentation()
    ds = torch.utils.data.ConcatDataset(datasets)

    dataloader_params = {
        "batch_size": args.batch,
        "shuffle": True,
        "num_workers": args.workers,
        "collate_fn": v5_collate_beamsegnet,
    }
    train_dataloader = torch.utils.data.DataLoader(ds, **dataloader_params)

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

    if args.type == "direct":
        beam_m = BeamNetDirect(
            nthetas=args.nthetas,
            depth=args.depth,
            hidden=args.hidden,
            symmetry=args.symmetry,
            act=act,
            other=args.other,
            bn=args.batch_norm,
        ).to(torch_device)
    elif args.type == "discrete":
        beam_m = BeamNetDiscrete(
            nthetas=args.nthetas,
            hidden=args.hidden,
            act=act,
            symmetry=args.symmetry,
            bn=args.batch_norm,
        ).to(torch_device)
    m = BeamNSegNet(segnet=seg_m, beamnet=beam_m, circular_mean=args.circular_mean).to(
        torch_device
    )

    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0)

    if args.compile:
        m = torch.compile(m)
    optimizer = torch.optim.AdamW(
        m.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    step = 0
    for epoch in range(args.epochs):
        for batch_data in train_dataloader:
            # for X, Y_rad in train_dataloader:
            optimizer.zero_grad()

            # copy to torch device
            if args.segmentation_level == "full":
                x = batch_data["x"].to(torch_device)
                y_rad = batch_data["y_rad"].to(torch_device)
                seg_mask = batch_data["segmentation_mask"].to(torch_device)
            elif args.segmentation_level == "downsampled":
                x = batch_data["all_windows_stats"].to(torch_device)
                y_rad = batch_data["y_rad"].to(torch_device)
                seg_mask = batch_data["downsampled_segmentation_mask"].to(torch_device)
            else:
                raise NotImplementedError

            assert seg_mask.ndim == 3 and seg_mask.shape[1] == 1

            # run beamformer and segmentation
            if not args.skip_segmentation:
                output = m(x)
            else:
                output = m(x, seg_mask)

            assert output["pred_theta"].isfinite().all()

            # x to beamformer loss (indirectly including segmentation)
            x_to_beamformer_loss = -beam_m.loglikelihood(output["pred_theta"], y_rad)
            assert x_to_beamformer_loss.shape == (args.batch, 1)
            x_to_beamformer_loss = x_to_beamformer_loss.mean()

            # segmentation loss
            x_to_segmentation_loss = (output["segmentation"] - seg_mask) ** 2
            assert (
                x_to_segmentation_loss.ndim == 3
                and x_to_segmentation_loss.shape[1] == 1
            )
            x_to_segmentation_loss = x_to_segmentation_loss.mean()

            if args.skip_segmentation:
                loss = x_to_beamformer_loss
            else:
                loss = x_to_beamformer_loss + 10 * x_to_segmentation_loss

            assert loss.isfinite().all()

            loss.backward()
            optimizer.step()

            to_log = {
                "loss": loss.item(),
                "segmentation_loss": x_to_segmentation_loss.item(),
                "beam_former_loss": x_to_beamformer_loss.item(),
            }
            if step % 200 == 0:
                # beam outputs
                img_beam_output = (
                    (beam_m.render_discrete_x(output["pred_theta"]) * 255).cpu().byte()
                )
                img_beam_gt = (beam_m.render_discrete_y(y_rad) * 255).cpu().byte()
                train_target_image = torch.zeros(
                    (img_beam_output.shape[0] * 2, img_beam_output.shape[1]),
                ).byte()
                for row_idx in range(img_beam_output.shape[0]):
                    train_target_image[row_idx * 2] = img_beam_output[row_idx]
                    train_target_image[row_idx * 2 + 1] = img_beam_gt[row_idx]
                if args.wandb_project:
                    output_image = wandb.Image(
                        train_target_image, caption="train vs target (interleaved)"
                    )
                    to_log["output"] = output_image

                # segmentation output
                _x = x.detach().cpu().numpy()
                _seg_mask = seg_mask.detach().cpu().numpy()
                _output_seg = output["segmentation"].detach().cpu().numpy()

                to_log["fig"] = plot_instance(_x, _output_seg, _seg_mask, idx=0)
            if args.wandb_project:
                wandb.log(to_log)
            step += 1

    # [optional] finish the wandb run, necessary in notebooks
    if args.wandb_project:
        wandb.finish()
