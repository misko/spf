import argparse
from functools import cache

import torch
import torch.nn.functional as f
import wandb

from spf.dataset.spf_dataset import v5_collate_beamsegnet, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNSegNetDirect,
    BeamNSegNetDiscrete,
)

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
        "--weight-decay",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--symmetry",
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

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="projectspf",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "batch": args.batch,
            "architecture": "beamsegnet1",
            "type": args.type,
        },
    )

    @cache
    def mean_guess(shape):
        return f.normalize(torch.ones(shape), p=1, dim=1)

    if args.type == "discrete":
        m = BeamNSegNetDiscrete(nthetas=args.nthetas).to(torch_device)
    elif args.type == "direct":
        m = BeamNSegNetDirect(nthetas=args.nthetas, symmetry=args.symmetry).to(
            torch_device
        )
    else:
        raise NotImplementedError
    if args.compile:
        m = torch.compile(m)
    optimizer = torch.optim.AdamW(
        m.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    step = 0
    for epoch in range(args.epochs):
        for X, Y_rad, segmentation in train_dataloader:
            optimizer.zero_grad()
            output = m(X.to(torch_device))
            loss = -m.loglikelihood(output, Y_rad.to(torch_device)).mean()
            loss.backward()
            optimizer.step()

            to_log = {"loss": loss.item()}
            if step % 200 == 0:
                _output = (m.render_discrete_x(output) * 255).cpu().byte()
                _Y = (m.render_discrete_y(Y_rad) * 255).cpu().byte()
                train_target_image = torch.zeros(
                    (_output.shape[0] * 2, _output.shape[1]),
                ).byte()
                for row_idx in range(_output.shape[0]):
                    train_target_image[row_idx * 2] = _output[row_idx]
                    train_target_image[row_idx * 2 + 1] = _Y[row_idx]
                output_image = wandb.Image(
                    train_target_image, caption="train vs target (interleaved)"
                )
                to_log["output"] = output_image
            wandb.log(to_log)
            step += 1

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
