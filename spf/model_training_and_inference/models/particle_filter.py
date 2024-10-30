import argparse
import pickle
import random
from multiprocessing import Pool

import torch
import tqdm

from spf.dataset.spf_dataset import v5spfdataset
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio

torch.set_num_threads(1)


def run_single_theta_single_radio(
    ds_fn,
    precompute_fn,
    empirical_pkl_fn,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        snapshots_per_session=1,
        readahead=True,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
                "signal_matrix",
                "simple_segmentations",
            ]
        ),
        empirical_data_fn=empirical_pkl_fn,
    )
    metrics = []
    for rx_idx in [0, 1]:
        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx)
        trajectory = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[2, 0.1]]),
            noise_std=torch.tensor([[theta_err, theta_dot_err]]),
            return_particles=False,
            N=N,
        )
        metrics.append(
            {
                "type": "single_theta_single_radio",
                "ds_fn": ds_fn,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": pf.metrics(trajectory=trajectory),
            }
        )
    return metrics


def run_single_theta_dual_radio(
    ds_fn, precompute_fn, empirical_pkl_fn, theta_err=0.1, theta_dot_err=0.001, N=128
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        snapshots_per_session=1,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
                "signal_matrix",
            ]
        ),
        empirical_data_fn=empirical_pkl_fn,
    )
    pf = PFSingleThetaDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[2, 0.1]]),
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
    )

    return [
        {
            "type": "single_theta_dual_radio",
            "ds_fn": ds_fn,
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def run_xy_dual_radio(
    ds_fn, precompute_fn, empirical_pkl_fn, pos_err=15, vel_err=0.5, N=128 * 16
):
    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=precompute_fn,
        paired=True,
        snapshots_per_session=1,
        skip_fields=set(
            [
                "windowed_beamformer",
                "all_windows_stats",
                "downsampled_segmentation_mask",
                "signal_matrix",
            ]
        ),
        empirical_data_fn=empirical_pkl_fn,
    )
    # dual radio dual
    pf = PFXYDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        N=N,
        mean=torch.tensor([[0, 0, 0, 0, 0]]),
        std=torch.tensor([[0, 200, 200, 0.1, 0.1]]),
        return_particles=False,
        noise_std=torch.tensor([[0, pos_err, pos_err, vel_err, vel_err]]),
    )
    return [
        {
            "type": "xy_dual_radio",
            "ds_fn": ds_fn,
            "vel_err": vel_err,
            "pos_err": pos_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def runner(x):
    fn, args = x
    return fn(**args)


if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--datasets",
            type=str,
            help="dataset prefixes",
            nargs="+",
            required=True,
        )
        parser.add_argument(
            "--nthetas",
            type=int,
            required=False,
            default=65,
        )
        parser.add_argument(
            "--device",
            type=str,
            required=False,
            default="cpu",
        )
        parser.add_argument(
            "--skip-qc",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--precompute-cache",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--full-p-fn",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            required=False,
        )
        parser.add_argument(
            "--debug",
            action=argparse.BooleanOptionalAction,
            default=False,
        )

        return parser

    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    jobs = []

    for ds_fn in args.datasets:
        for N in [128, 128 * 4, 128 * 8, 128 * 16]:
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_single_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dot_err,
                            },
                        )
                    )
            for theta_err in [0.1, 0.01, 0.001, 0.2]:
                for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
                    jobs.append(
                        (
                            run_single_theta_dual_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "theta_err": theta_err,
                                "theta_dot_err": theta_dot_err,
                            },
                        )
                    )
    for ds_fn in args.datasets:
        for N in [128, 128 * 4, 128 * 8, 128 * 16, 128 * 32]:
            for pos_err in [1000, 100, 50, 30, 15, 5, 0.5]:
                for vel_err in [50, 5, 0.5, 0.05, 0.01, 0.001]:
                    jobs.append(
                        (
                            run_xy_dual_radio,
                            {
                                "ds_fn": ds_fn,
                                "precompute_fn": args.precompute_cache,
                                "full_p_fn": args.full_p_fn,
                                "N": N,
                                "pos_err": pos_err,
                                "vel_err": vel_err,
                            },
                        )
                    )

    random.shuffle(jobs)

    if args.debug:
        results = list(tqdm.tqdm(map(runner, jobs), total=len(jobs)))
    else:
        with Pool(20) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(tqdm.tqdm(pool.imap(runner, jobs), total=len(jobs)))
    pickle.dump(results, open("particle_filter_results2.pkl", "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
