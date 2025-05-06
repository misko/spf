import tempfile
import time
from typing import List

import torch

from spf.dataset.spf_dataset import (
    data_single_radio_to_raw,
    training_only_keys,
    v5inferencedataset,
    v5spfdataset,
)
from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper
from spf.model_training_and_inference.models.single_point_networks_inference import (
    load_model_and_config_from_config_fn_and_checkpoint,
    single_example_realtime_inference,
)


def compare_two_entries(a, b):
    for k, v in a.items():
        if k in training_only_keys + ["craft_y_rad_binned"]:
            continue
        assert k in b, f"{k} missing!"
        vp = b[k]
        if isinstance(v, torch.Tensor):
            assert v.isclose(vp, rtol=1e-2).all(), f"{v} {vp}"
        elif isinstance(v, List):
            s = v[0]
            sp = vp[0]
            assert len(s) == len(sp)
            for idx in range(len(s)):
                assert s[idx]["start_idx"] == sp[idx]["start_idx"]
                assert s[idx]["end_idx"] == sp[idx]["end_idx"]


def test_preprocessing_equal(perfect_circle_n50_0p01_v4):
    tmpdirname, zarr_fn = perfect_circle_n50_0p01_v4

    ds = v5spfdataset(  # make sure everything gets segmented here
        zarr_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=tmpdirname,
        paired=True,
        segment_if_not_exist=True,
        v4=True,
    )

    with v5inferencedataset(
        yaml_fn=zarr_fn.replace(".zarr", "") + ".yaml",
        nthetas=65,
        gpu=False,
        paired=True,
        model_config_fn="",
        skip_fields=[],
        vehicle_type="wallarray",
        skip_segmentation=False,
        skip_detrend=False,
    ) as v5inf:

        for idx in range(20):
            d = ds[idx]
            for ridx in range(2):
                v5inf.write_to_idx(idx, ridx, data_single_radio_to_raw(d[ridx], ds))

        for idx in range(20):
            d = ds[idx]
            for ridx in range(2):
                compare_two_entries(d[ridx], v5inf[idx][ridx])


def test_v5inference_with_nn_wrapper(
    perfect_circle_n50_0p01_v4, paired_net_checkpoint_using_single_checkpoint
):
    tmpdirname, zarr_fn = perfect_circle_n50_0p01_v4

    ds = v5spfdataset(  # make sure everything gets segmented here
        zarr_fn,
        nthetas=7,
        ignore_qc=True,
        precompute_cache=tmpdirname,
        paired=True,
        # skip_fields=["signal_matrix"],
        segment_if_not_exist=True,
        v4=True,
    )

    # get paired checkpoint results
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

    paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
    paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

    def compare_two_nn_ds(a, b):
        for idx in range(len(a)):
            for ridx in range(2):
                assert a[idx][ridx]["single"].isclose(b[idx][ridx]["single"]).all()
                assert a[idx][ridx]["paired"].isclose(b[idx][ridx]["paired"]).all()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # this one should cache to disk
        nn_ds = v5spfdataset_nn_wrapper(
            ds,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=tmpdirname,
            device="cpu",
            v4=True,
        )
        # this one stays in memory
        nn_ds_no_disk = v5spfdataset_nn_wrapper(
            ds,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=None,
            device="cpu",
            v4=True,
        )
        compare_two_nn_ds(nn_ds, nn_ds_no_disk)
        # this one runs realtime
        nn_ds_realtime = v5spfdataset_nn_wrapper(
            ds,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=None,
            device="cpu",
            v4=True,
        )
        compare_two_nn_ds(nn_ds, nn_ds_realtime)


def test_v5inference_with_nn(
    perfect_circle_n50_0p01_v4, paired_net_checkpoint_using_single_checkpoint
):
    tmpdirname, zarr_fn = perfect_circle_n50_0p01_v4
    ds = v5spfdataset(  # make sure everything gets segmented here
        zarr_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=tmpdirname,
        paired=True,
        # skip_fields=["signal_matrix"],
        segment_if_not_exist=True,
        v4=True,
    )

    with v5inferencedataset(
        yaml_fn=zarr_fn.replace(".zarr", "") + ".yaml",
        nthetas=65,
        gpu=False,
        paired=True,
        model_config_fn="",
        # skip_fields=["signal_matrix"],
        vehicle_type="wallarray",
        skip_segmentation=False,
        skip_detrend=False,
    ) as v5inf:

        n = len(ds)

        for idx in range(n):
            d = ds[idx]
            for ridx in range(2):
                v5inf.write_to_idx(idx, ridx, d[ridx])

        # get paired checkpoint results
        paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

        paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
        paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

        # load model and model config
        model, config = load_model_and_config_from_config_fn_and_checkpoint(
            config_fn=paired_config_fn, checkpoint_fn=paired_checkpoint_fn
        )
        config["optim"]["device"] = "cpu"
        model.to(config["optim"]["device"])

        st = time.time()

        idx = 0
        for x in single_example_realtime_inference(
            model, config["global"], config["optim"], realtime_ds=v5inf
        ):
            idx += 1
            if idx >= len(ds):
                break
        print(time.time() - st)
