import tempfile

import pytest
import torch

from spf.dataset.spf_dataset import (
    training_only_keys,
    v5inferencedataset,
    v5spfdataset_manager,
)
from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper
from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN
from spf.filters.particle_dualradio_filter import plot_single_theta_dual_radio
from spf.filters.particle_dualradioXY_filter import plot_xy_dual_radio
from spf.filters.particle_single_radio_filter import plot_single_theta_single_radio
from spf.filters.run_filters_on_data import (
    run_PF_single_theta_dual_radio,
    run_PF_single_theta_dual_radio_NN,
    run_PF_single_theta_single_radio,
    run_PF_xy_dual_radio,
)
from spf.model_training_and_inference.models.single_point_networks_inference import (
    get_nn_inference_on_ds_and_cache,
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


@pytest.mark.parametrize("temp_file", [False, True])
def test_single_theta_dual_radioNN(
    temp_file, noise1_n128_obits2, paired_net_checkpoint_using_single_checkpoint
):

    # get paired checkpoint results
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

    paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
    paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    ds_fn += ".zarr"

    inference_dir = tempfile.TemporaryDirectory()
    if not temp_file:
        _ = get_nn_inference_on_ds_and_cache(
            ds_fn,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=inference_dir.name,
            device="cpu",
            batch_size=4,
            workers=0,
            precompute_cache=dirname,
            crash_if_not_cached=False,
        )

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
            "theta_err": 0.1,
            "theta_dot_err": 0.001,
            "checkpoint_fn": paired_checkpoint_fn,
        }
        if not temp_file:
            args.update(
                {
                    "inference_cache": inference_dir.name,
                }
            )

        _ = run_PF_single_theta_dual_radio_NN(**args)
        plot_single_theta_dual_radio(ds)


@pytest.mark.parametrize("temp_file", [False, True])
def test_single_theta_dual_radioNN(
    temp_file, noise1_n128_obits2, paired_net_checkpoint_using_single_checkpoint
):

    # get paired checkpoint results
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

    paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
    paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    ds_fn += ".zarr"

    inference_dir = tempfile.TemporaryDirectory()
    if not temp_file:
        _ = get_nn_inference_on_ds_and_cache(
            ds_fn,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=inference_dir.name,
            device="cpu",
            batch_size=4,
            workers=0,
            precompute_cache=dirname,
            crash_if_not_cached=False,
        )

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
            "theta_err": 0.1,
            "theta_dot_err": 0.001,
            "checkpoint_fn": paired_checkpoint_fn,
        }
        if not temp_file:
            args.update(
                {
                    "inference_cache": inference_dir.name,
                }
            )

        _ = run_PF_single_theta_dual_radio_NN(**args)
        plot_single_theta_dual_radio(ds)


def test_single_theta_dual_radioNN_nnwrapper(
    noise1_n128_obits2, paired_net_checkpoint_using_single_checkpoint
):
    # get paired checkpoint results
    paired_checkpoints_dir = paired_net_checkpoint_using_single_checkpoint

    paired_config_fn = f"{paired_checkpoints_dir}/config.yml"
    paired_checkpoint_fn = f"{paired_checkpoints_dir}/best.pth"

    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    ds_fn += ".zarr"

    with v5spfdataset_manager(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set([]),  # set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
        segment_if_not_exist=True,
    ) as ds:

        v5inf = v5inferencedataset(
            yaml_fn=ds.zarr_fn.replace(".zarr", "") + ".yaml",
            nthetas=65,
            gpu=False,
            n_parallel=8,
            paired=True,
            model_config_fn="",
            skip_fields=["signal_matrix"] + training_only_keys,
            vehicle_type="wallarray",
            skip_segmentation=False,
            skip_detrend=False,
        )

        n_steps = len(ds) // 4

        for idx in range(n_steps):
            d = ds[idx]
            print("serving", idx)
            for ridx in range(2):
                v5inf.write_to_idx(idx, ridx, d[ridx])
        nn_ds = v5spfdataset_nn_wrapper(
            v5inf,
            paired_config_fn,
            paired_checkpoint_fn,
            inference_cache=None,
            device="cpu",
            v4=False,
            absolute=True,
        )
        pf = PFSingleThetaDualRadioNN(nn_ds=nn_ds)

        theta_err = 0.075
        theta_dot_err = 0.002
        N = 512
        _ = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            N=N,
            std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
            noise_std=torch.tensor([[theta_err, theta_dot_err]]),
            return_particles=False,
            steps=n_steps,
        )
