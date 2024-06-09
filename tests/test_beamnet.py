import gc
from functools import cache

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from spf.dataset.spf_dataset import v5_collate_beamsegnet, v5_thetas_to_targets
from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    BeamNSegNet,
    ConvNet,
    UNet1D,
)

torch.manual_seed(1337)

import pickle


def test_beamnet():

    torch_device = torch.device("cpu")
    nthetas = 11

    batch_data = pickle.load(open("tests/test_batch.pkl", "rb"))

    seg_m = ConvNet(3, 1, 32, bn=True).to(torch_device)

    beam_m = BeamNetDirect(
        nthetas=nthetas, hidden=16, symmetry=True, other=True, act=nn.SELU, bn=False
    ).to(torch_device)
    m = BeamNSegNet(segnet=seg_m, beamnet=beam_m, circular_mean=True).to(torch_device)

    optimizer = torch.optim.AdamW(seg_m.parameters(), lr=0.01, weight_decay=0)

    step = 0
    head_start = 200
    for epoch in range(4000):
        if step == head_start:
            optimizer = torch.optim.AdamW(beam_m.parameters(), lr=0.001, weight_decay=0)
            optimizer.zero_grad()
        # for X, Y_rad in train_dataloader:
        optimizer.zero_grad()

        # copy to torch device
        x = batch_data["all_windows_stats"].to(torch_device)
        y_rad = batch_data["y_rad"].to(torch_device)
        seg_mask = batch_data["downsampled_segmentation_mask"].to(torch_device)

        assert seg_mask.ndim == 3 and seg_mask.shape[1] == 1

        output = m(x, seg_mask, rx_spacing=None)

        # x to beamformer loss (indirectly including segmentation)
        x_to_beamformer_loss = -beam_m.loglikelihood(output["pred_theta"], y_rad)

        x_to_beamformer_loss = x_to_beamformer_loss.mean()

        # segmentation loss
        x_to_segmentation_loss = (output["segmentation"] - seg_mask) ** 2
        assert x_to_segmentation_loss.ndim == 3 and x_to_segmentation_loss.shape[1] == 1
        x_to_segmentation_loss = x_to_segmentation_loss.mean()

        if step >= head_start:
            loss = x_to_beamformer_loss
        else:
            loss = x_to_segmentation_loss
        assert torch.isfinite(loss).all()
        loss.backward()
        optimizer.step()

        print(
            step,
            loss.item(),
            x_to_beamformer_loss.item(),
            x_to_segmentation_loss.item(),
        )
        step += 1
    # -3.3521714210510254 -3.4109437465667725 0.005877225194126368
    # -3.6402697563171387 -3.662614345550537 0.0022344605531543493
    assert loss.item() < -1.0
    assert x_to_beamformer_loss < -1.0
    assert x_to_segmentation_loss < 0.01
