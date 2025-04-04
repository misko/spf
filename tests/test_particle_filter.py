import pytest

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.filters.particle_dualradio_filter import plot_single_theta_dual_radio
from spf.filters.particle_dualradioXY_filter import plot_xy_dual_radio
from spf.filters.particle_single_radio_filter import plot_single_theta_single_radio
from spf.filters.run_filters_on_data import (
    run_PF_single_theta_dual_radio,
    run_PF_single_theta_single_radio,
    run_PF_xy_dual_radio,
)


@pytest.mark.parametrize("temp_file", [False, True])
def test_single_theta_single_radio(temp_file, noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    with v5spfdataset_manager(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
        temp_file=temp_file,
        temp_file_suffix="",
        segment_if_not_exist=not temp_file,
    ) as ds:
        args = {
            "ds": ds,
            "N": 1024 * 4,
            "theta_err": 0.01,
            "theta_dot_err": 0.01,
        }
        results = run_PF_single_theta_single_radio(**args)
        for result in results:
            assert result["metrics"]["mse_single_radio_theta"] < 0.05
        plot_single_theta_single_radio(ds)


@pytest.mark.parametrize("temp_file", [False, True])
def test_single_theta_dual_radio(temp_file, noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    with v5spfdataset_manager(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
        temp_file=temp_file,
        temp_file_suffix="",
        segment_if_not_exist=not temp_file,
    ) as ds:
        args = {
            "ds": ds,
            "N": 1024 * 4,
            "theta_err": 0.01,
            "theta_dot_err": 0.01,
        }
        result = run_PF_single_theta_dual_radio(**args)
        assert result[0]["metrics"]["mse_craft_theta"] < 0.15
        plot_single_theta_dual_radio(ds)


@pytest.mark.parametrize("temp_file", [False, True])
def test_XY_dual_radio(temp_file, noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    with v5spfdataset_manager(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
        temp_file=temp_file,
        temp_file_suffix="",
        segment_if_not_exist=not temp_file,
    ) as ds:
        args = {
            "ds": ds,
            "N": 1024 * 4,
            "pos_err": 50,
            "vel_err": 0.1,
        }

        result = run_PF_xy_dual_radio(**args)
        assert result[0]["metrics"]["mse_craft_theta"] < 0.25
        plot_xy_dual_radio(ds)
