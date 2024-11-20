import pytest

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.filters.ekf_dualradio_filter import (
    SPFPairedKalmanFilter,
    run_and_plot_dualradio_EKF,
)
from spf.filters.ekf_dualradioXY_filter import (
    SPFPairedXYKalmanFilter,
    run_and_plot_dualradioXY_EKF,
)
from spf.filters.ekf_single_radio_filter import (
    SPFKalmanFilter,
    run_and_plot_single_radio_EKF,
)


@pytest.mark.parametrize("temp_file", [False, True])
def test_single_radio_ekf(temp_file, perfect_circle_dataset_n1025_orbits4_noise0p3):
    ds_dir, _, ds_fn = perfect_circle_dataset_n1025_orbits4_noise0p3
    with v5spfdataset_manager(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=ds_dir,
        paired=True,
        skip_fields=set(["signal_matrix"]),
        temp_file=temp_file,
        temp_file_suffix="",
    ) as ds:
        kfs = [
            SPFKalmanFilter(ds=ds, rx_idx=rx_idx, phi_std=5.0, p=5)
            for rx_idx in range(2)
        ]
        single_radio_trajectories = [kf.trajectory(debug=True) for kf in kfs]
        single_radio_metrics = [
            kf.metrics(trajectory)
            for kf, trajectory in zip(kfs, single_radio_trajectories)
        ]
        assert single_radio_metrics[0]["mse_single_radio_theta"] < 0.03
        assert single_radio_metrics[1]["mse_single_radio_theta"] < 0.03

        run_and_plot_single_radio_EKF(ds, trajectories=single_radio_trajectories)


@pytest.mark.parametrize("temp_file", [False, True])
def test_paired_radio_ekf(temp_file, perfect_circle_dataset_n1025_orbits4_noise0p3):
    ds_dir, _, ds_fn = perfect_circle_dataset_n1025_orbits4_noise0p3
    with v5spfdataset_manager(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=ds_dir,
        paired=True,
        skip_fields=set(["signal_matrix"]),
        temp_file=temp_file,
        temp_file_suffix="",
    ) as ds:
        kf = SPFPairedKalmanFilter(ds=ds, phi_std=5.0, p=5, dynamic_R=False)
        paired_trajectory = kf.trajectory(debug=True)
        paired_metrics = kf.metrics(paired_trajectory)
        assert paired_metrics["mse_craft_theta"] < 0.005

        run_and_plot_dualradio_EKF(ds, trajectory=paired_trajectory)


@pytest.mark.parametrize("temp_file", [False, True])
def test_pairedXY_radio_ekf(temp_file, perfect_circle_dataset_n1025_orbits4_noise0p3):
    ds_dir, _, ds_fn = perfect_circle_dataset_n1025_orbits4_noise0p3
    with v5spfdataset_manager(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=ds_dir,
        paired=True,
        skip_fields=set(["signal_matrix"]),
        temp_file=temp_file,
        temp_file_suffix="",
    ) as ds:

        kf = SPFPairedXYKalmanFilter(ds=ds, phi_std=5.0, p=0.1, dynamic_R=True)
        pairedXY_trajectory = kf.trajectory(debug=True, dt=1.0, noise_std=10)
        pairedXY_metrics = kf.metrics(pairedXY_trajectory)
        assert pairedXY_metrics["mse_craft_theta"] < 5  # we arent very good here

        run_and_plot_dualradioXY_EKF(ds, trajectory=pairedXY_trajectory)
