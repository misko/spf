import argparse
import pickle
import random
from multiprocessing import Pool

import torch
import tqdm

from spf.dataset.spf_dataset import v5spfdataset, v5spfdataset_manager
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio

torch.set_num_threads(1)


def run_jobs_with_one_dataset(kwargs):
    results = []
    with v5spfdataset_manager(
        kwargs["ds_fn"],
        nthetas=65,
        ignore_qc=True,
        precompute_cache=kwargs["precompute_cache"],
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
        empirical_data_fn=kwargs["empirical_pkl_fn"],
    ) as ds:
        for fn, fn_kwargs in kwargs["jobs"]:

            fn_kwargs["ds"] = ds
            new_results = fn(**fn_kwargs)
            for result in new_results:
                result["ds_fn"] = kwargs["ds_fn"]
            results += new_results
        return results


def run_single_theta_single_radio(
    ds,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
):
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
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": pf.metrics(trajectory=trajectory),
            }
        )
    return metrics


def run_single_theta_dual_radio(ds, theta_err=0.1, theta_dot_err=0.001, N=128):

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
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def run_xy_dual_radio(ds, pos_err=15, vel_err=0.5, N=128 * 16):

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
            "--empirical-pkl-fn",
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
            "--output",
            type=str,
            required=True,
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

    jobs_per_ds_fn = []

    # for N in [128, 128 * 4, 128 * 8, 128 * 16]:
    #     for theta_err in [0.1, 0.01, 0.001, 0.2]:
    #         for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
    #             jobs_per_ds_fn.append(
    #                 (
    #                     run_single_theta_single_radio,
    #                     {
    #                         "N": N,
    #                         "theta_err": theta_err,
    #                         "theta_dot_err": theta_dot_err,
    #                     },
    #                 )
    #             )
    #     for theta_err in [0.1, 0.01, 0.001, 0.2]:
    #         for theta_dot_err in [0.001, 0.0001, 0.01, 0.1]:
    #             jobs_per_ds_fn.append(
    #                 (
    #                     run_single_theta_dual_radio,
    #                     {
    #                         "N": N,
    #                         "theta_err": theta_err,
    #                         "theta_dot_err": theta_dot_err,
    #                     },
    #                 )
    #             )
    for N in [128, 128 * 4, 128 * 8, 128 * 16, 128 * 32]:
        for pos_err in [1000, 100, 50, 30, 15, 5, 0.5]:
            for vel_err in [50, 5, 0.5, 0.05, 0.01, 0.001]:
                jobs_per_ds_fn.append(
                    (
                        run_xy_dual_radio,
                        {
                            "N": N,
                            "pos_err": pos_err,
                            "vel_err": vel_err,
                        },
                    )
                )

    random.shuffle(jobs_per_ds_fn)

    # one job per dataset
    jobs = [
        {
            "ds_fn": ds_fn,
            "precompute_cache": args.precompute_cache,
            "empirical_pkl_fn": args.empirical_pkl_fn,
            "jobs": jobs_per_ds_fn,
        }
        for ds_fn in args.datasets
    ]

    jobs = []
    for ds_fn in args.datasets:
        for job in jobs_per_ds_fn:
            jobs.append(
                {
                    "ds_fn": ds_fn,
                    "precompute_cache": args.precompute_cache,
                    "empirical_pkl_fn": args.empirical_pkl_fn,
                    "jobs": [job],
                }
            )

    if args.debug:
        results = list(
            tqdm.tqdm(
                map(run_jobs_with_one_dataset, jobs),
                total=len(jobs),
            )
        )
    else:
        with Pool(20) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(
                tqdm.tqdm(
                    pool.imap(run_jobs_with_one_dataset, jobs),
                    total=len(jobs),
                )
            )
    pickle.dump(results, open(args.output, "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
