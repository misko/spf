from typing import List

import torch

from spf.dataset.spf_dataset import (
    data_single_radio_to_raw,
    training_only_keys,
    v5inferencedataset,
    v5spfdataset,
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

    v5inf = v5inferencedataset(
        yaml_fn=zarr_fn.replace(".zarr", "") + ".yaml",
        nthetas=65,
        gpu=False,
        n_parallel=8,
        paired=True,
        model_config_fn="",
        skip_fields=[],
        vehicle_type="wallarray",
        skip_segmentation=False,
        skip_detrend=False,
    )

    for idx in range(20):
        d = ds[idx]
        for ridx in range(2):
            v5inf.write_to_idx(idx, ridx, data_single_radio_to_raw(d[ridx]))

    for idx in range(20):
        d = ds[idx]
        for ridx in range(2):
            compare_two_entries(d[ridx], v5inf[idx][ridx])
