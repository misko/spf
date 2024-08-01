import tempfile

import torch
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml, partial_dataset
from spf.dataset.open_partial_ds import open_partial_dataset_and_check_some
from spf.dataset.spf_dataset import v5spfdataset
import random
from spf.model_training_and_inference.models.create_empirical_p_dist import (
    apply_symmetry_rules_to_heatmap,
    get_heatmap,
)
import pytest
import pickle

from spf.model_training_and_inference.models.particle_filter import (
    plot_single_theta_dual_radio,
    plot_single_theta_single_radio,
    plot_xy_dual_radio,
    run_single_theta_dual_radio,
    run_single_theta_single_radio,
    run_xy_dual_radio,
)


@pytest.fixture
def noise1_n128_obits2():
    with tempfile.TemporaryDirectory() as tmpdirname:
        n = 128
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.3, orbits=2
        )
        yield tmpdirname, fn


@pytest.fixture
def heatmap(noise1_n128_obits2):
    dirname, ds_fn = noise1_n128_obits2
    ds = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_signal_matrix=True,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    heatmap = get_heatmap([ds], bins=50)
    heatmap = apply_symmetry_rules_to_heatmap(heatmap)
    full_p_fn = f"{dirname}/full_p.pkl"
    pickle.dump({"full_p": heatmap}, open(full_p_fn, "wb"))
    return full_p_fn


def test_single_theta_single_radio(noise1_n128_obits2, heatmap):
    dirname, ds_fn = noise1_n128_obits2
    ds = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_signal_matrix=True,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    args = {
        "ds_fn": ds_fn,
        "precompute_fn": dirname,
        "full_p_fn": heatmap,
        "N": 1024 * 4,
        "theta_err": 0.01,
        "theta_dot_err": 0.01,
    }
    results = run_single_theta_single_radio(**args)
    for result in results:
        assert result["metrics"]["mse_theta"] < 0.05
    plot_single_theta_single_radio(ds, heatmap)


def test_single_theta_dual_radio(noise1_n128_obits2, heatmap):
    dirname, ds_fn = noise1_n128_obits2
    ds = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_signal_matrix=True,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    args = {
        "ds_fn": ds_fn,
        "precompute_fn": dirname,
        "full_p_fn": heatmap,
        "N": 1024 * 4,
        "theta_err": 0.01,
        "theta_dot_err": 0.01,
    }
    result = run_single_theta_dual_radio(**args)
    assert result[0]["metrics"]["mse_theta"] < 0.15
    plot_single_theta_dual_radio(ds, heatmap)


def test_single_theta_dual_radio(noise1_n128_obits2, heatmap):
    dirname, ds_fn = noise1_n128_obits2
    ds = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_signal_matrix=True,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    args = {
        "ds_fn": ds_fn,
        "precompute_fn": dirname,
        "full_p_fn": heatmap,
        "N": 1024 * 4,
        "pos_err": 50,
        "vel_err": 0.1,
    }

    result = run_xy_dual_radio(**args)
    assert result[0]["metrics"]["mse_theta"] < 0.25
    plot_xy_dual_radio(ds, heatmap)


def test_partial(noise1_n128_obits2):
    dirname, ds_fn = noise1_n128_obits2
    ds_og = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_signal_matrix=True,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn_out = f"{tmpdirname}/partial"
        for partial_n in [10, 100, 128]:
            partial_dataset(ds_fn, ds_fn_out, partial_n)
            ds = v5spfdataset(
                ds_fn_out,
                precompute_cache=tmpdirname,
                nthetas=65,
                skip_signal_matrix=True,
                paired=True,
                ignore_qc=True,
                gpu=False,
                temp_file=True,
                temp_file_suffix="",
            )
            assert min(ds.valid_entries) == partial_n
            random.seed(0)
            idxs = list(range(partial_n))
            random.shuffle(idxs)
            for idx in idxs[:8]:
                for r_idx in range(2):
                    for key in ds_og[0][0].keys():
                        if isinstance(ds_og[idx][r_idx][key], torch.Tensor):
                            assert (ds_og[idx][r_idx][key] == ds[idx][r_idx][key]).all()


def test_partial_script(noise1_n128_obits2):
    dirname, ds_fn = noise1_n128_obits2
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn_out = f"{tmpdirname}/partial"
        for partial_n in [10, 20]:
            partial_dataset(ds_fn, ds_fn_out, partial_n)
            # open_partial_dataset_and_check_some(ds_fn_out, suffix="", n_parallel=0)
            open_partial_dataset_and_check_some(
                ds_fn_out, suffix="", n_parallel=0, skip_fields=["windowed_beamformer"]
            )
