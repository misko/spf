import tempfile
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset

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
