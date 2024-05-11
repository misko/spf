import argparse
from functools import cache

import torch
import torch.nn.functional as f
import wandb

from spf.dataset.spf_dataset import v5_collate_beamsegnet, v5spfdataset
from spf.model_training_and_inference.models.beamsegnet import BeamNSegNet

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
    # "/Volumes/SPFData/missions/april5/wallarrayv3_2024_05_06_19_04_15_nRX2_bounce",
    args = parser.parse_args()
    torch_device = torch.device(args.device)

    # loop over and concat datasets here
    datasets = [v5spfdataset(prefix, nthetas=args.nthetas) for prefix in args.datasets]
    for ds in datasets:
        ds.get_segmentation()
    ds = torch.utils.data.ConcatDataset(datasets)

    nthetas = 11
    lr = 0.001

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
        },
    )

    @cache
    def mean_guess(shape):
        return f.normalize(torch.ones(shape), p=1, dim=1)

    loss_fn = torch.nn.MSELoss()
    m = BeamNSegNet(nthetas=nthetas).to(torch_device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=args.lr)
    step = 0
    for epoch in range(10):
        for X, Y in train_dataloader:
            optimizer.zero_grad()
            output = m(X.to(torch_device))
            loss = loss_fn(output, Y.to(torch_device))
            loss.backward()
            mean_loss = output
            optimizer.step()

            mean_loss = loss_fn(mean_guess(Y.shape), Y)

            to_log = {"loss": loss.item(), "mean_loss": mean_loss.item()}
            if step % 500 == 0:
                train_target_image = torch.zeros(
                    (output.shape[0] * 2, output.shape[1])
                ).byte()
                _output = (output * 255).cpu().byte()
                _Y = (Y * 255).cpu().byte()
                for row_idx in range(output.shape[0]):
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
