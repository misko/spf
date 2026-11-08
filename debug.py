from spf.dataset.spf_dataset import v5inferencedataset, data_single_radio_to_raw
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml_v4
from spf.dataset.spf_dataset import v5spfdataset

import pickle
from spf.dataset.spf_dataset import v5inferencedataset, data_single_radio_to_raw
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml_v4
from spf.dataset.spf_dataset import v5spfdataset

# create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v4, n=50, noise=0.01)
import logging
from spf.dataset.spf_dataset import v5_collate_keys_fast, global_config_to_keys_used
import torch

# try doing neural net inference online
# get paired checkpoint results
from spf.model_training_and_inference.models.single_point_networks_inference import (
    load_model_and_config_from_config_fn_and_checkpoint,
)

paired_checkpoint_fn = "/home/mouse9911/gits/spf/checkpoints/march16/paired_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2/best.pth"
paired_config_fn = "/home/mouse9911/gits/spf/checkpoints/march16/paired_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2/config.yml"
empirical_dist_fn = "/home/mouse9911/gits/spf/checkpoints/apr5/single_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2_updateddata_lowerdrop_rerun_3p6/empirical_dist.pkl"
inference_cache = "/tmp/infcache"
n = 50
tmpdirname = "/tmp/test1"
fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"

ds = v5spfdataset(  # make sure everything gets segmented here
    fn,
    nthetas=65,
    ignore_qc=True,
    precompute_cache=tmpdirname,
    paired=True,
    empirical_data_fn=empirical_dist_fn,
    # skip_fields=set(["signal_matrix"]),
    segment_if_not_exist=True,
    v4=True,
)

v5inf = v5inferencedataset(
    yaml_fn=fn + ".yaml",
    nthetas=65,
    gpu=False,
    n_parallel=8,
    paired=True,
    model_config_fn="",
    empirical_data_fn=empirical_dist_fn,
    skip_fields=["signal_matrix"],
    vehicle_type="wallarray",
    skip_segmentation=False,
    skip_detrend=False,
)
n = 3

for idx in range(n):
    d = ds[idx]
    for ridx in range(2):
        print(idx, ridx)
        # v5inf.write_to_idx(idx, ridx, data_single_radio_to_raw(d[ridx]))
        v5inf.write_to_idx(idx, ridx, d[ridx])
        # compare_two_entries(ds[0][1], v5inf[0][1])

from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper


nn_ds = v5spfdataset_nn_wrapper(
    v5inf,
    paired_config_fn,
    paired_checkpoint_fn,
    inference_cache,
    device="cpu",
    absolute=True
)

nn_ds[0]
