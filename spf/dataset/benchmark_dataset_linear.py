from functools import partial
import numpy as np
import scipy
from tqdm import tqdm
from filterpy.monte_carlo import systematic_resample

import pickle
import os
import sys
import time

from spf.dataset.spf_dataset import (
    v5_collate_beamsegnet,
    v5_collate_keys_fast,
    v5spfdataset,
)
import torch
import random

ds_fn = "/mnt/4tb_ssd/june_fix/wallarrayv3_2024_06_15_11_44_13_nRX2_bounce.zarr"
# ds_fn = "/mnt/4tb_ssd/wallarrayv3_2024_06_15_11_44_13_nRX2_bounce.zarr"


nthetas = 65
ds = v5spfdataset(
    ds_fn,
    nthetas=nthetas,
    ignore_qc=True,
    precompute_cache="/home/mouse9911/precompute_cache_chunk16_fresh",
    paired=True,
    skip_signal_matrix=True,
    snapshots_per_session=1,
    skip_simple_segmentations=True,
    skip_fields=set(
        [
            "windowed_beamformer",
            "all_windows_stats",
            "downsampled_segmentation_mask",
        ]
    ),
)

idxs = torch.arange(3000)  # len(ds))

for x in idxs:
    ds[x]
sys.exit(0)
random.shuffle(idxs)

random.seed(10)
ds = torch.utils.data.Subset(ds, idxs[:3000])
print("Getting")

if False:
    start_time = time.time()
    count = 1
    for idx in range(len(ds)):
        if count % 1000 == 0:
            print((time.time() - start_time) / count, "seconds per sample")
        ds[idx]
        count += 1
    print((time.time() - start_time) / len(ds), "seconds per sample")

    # x = batch_data["all_windows_stats"].to(torch_device).to(torch.float32)
    # rx_pos = batch_data["rx_pos_xy"].to(torch_device)
    # seg_mask = batch_data["downsampled_segmentation_mask"].to(torch_device)
    # rx_spacing = batch_data["rx_spacing"].to(torch_device)
    # windowed_beamformer = batch_data["windowed_beamformer"].to(torch_device)
    # y_rad = batch_data["y_rad"].to(torch_device)
    # craft_y_rad = batch_data["craft_y_rad"].to(torch_device)
    # y_phi = batch_data["y_phi"].to(torch_device)
workers = 0
dataloader_params = {
    "batch_size": 8,
    "shuffle": True,
    "num_workers": workers,
    "collate_fn": partial(
        v5_collate_keys_fast,
        [
            "all_windows_stats",
            "rx_pos_xy",
            "downsampled_segmentation_mask",
            "rx_spacing",
            # "windowed_beamformer",
            "y_rad",
            "craft_y_rad",
            "y_phi",
        ],
    ),
    "pin_memory": True,
    "prefetch_factor": 1 if workers > 0 else None,
}
train_dataloader = torch.utils.data.DataLoader(ds, **dataloader_params)

for step, batch_data in enumerate(tqdm(train_dataloader)):
    pass
