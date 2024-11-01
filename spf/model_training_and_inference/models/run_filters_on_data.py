import argparse
import pickle
import random
from multiprocessing import Pool

import torch
import tqdm
import yaml

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.filters.ekf_dualradio_filter import SPFPairedKalmanFilter
from spf.filters.ekf_dualradioXY_filter import SPFPairedXYKalmanFilter
from spf.filters.ekf_single_radio_filter import SPFKalmanFilter
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
                "weighted_beamformer",
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


def run_EKF_single_theta_single_radio(ds, phi_std, p, noise_std, dynamic_R):
    metrics = []

    for rx_idx in [0, 1]:
        ekf = SPFKalmanFilter(
            ds=ds, rx_idx=rx_idx, phi_std=phi_std, p=p, dynamic_R=dynamic_R
        )
        trajectory = ekf.trajectory(
            dt=1.0,
            noise_std=0.01,
            max_iterations=None,
            debug=False,
        )

        metrics.append(
            {
                "type": "EKF_single_theta_single_radio",
                "rx_idx": rx_idx,
                "phi_std": phi_std,
                "p": p,
                "noise_std": noise_std,
                "dyanmic_R": dynamic_R,
                "metrics": ekf.metrics(trajectory=trajectory),
            }
        )
    return metrics


def run_EKF_single_theta_dual_radio(
    ds,
    phi_std,
    p,
    noise_std,
    dynamic_R,
):
    ekf = SPFPairedKalmanFilter(ds=ds, phi_std=phi_std, p=p, dynamic_R=dynamic_R)

    trajectory = ekf.trajectory(
        dt=1.0,
        noise_std=noise_std,
        max_iterations=None,
        debug=False,
    )

    return [
        {
            "type": "EKF_single_theta_dual_radio",
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dyanmic_R": dynamic_R,
            "metrics": ekf.metrics(trajectory=trajectory),
        }
    ]


def run_EKF_xy_dual_radio(
    ds,
    phi_std,
    p,
    noise_std,
    dynamic_R,
):
    ekf = SPFPairedXYKalmanFilter(ds=ds, phi_std=phi_std, p=p, dynamic_R=dynamic_R)

    trajectory = ekf.trajectory(
        dt=1.0,
        noise_std=noise_std,
        max_iterations=None,
        debug=False,
    )

    return [
        {
            "type": "EKF_XY_dual_radio",
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dyanmic_R": dynamic_R,
            "metrics": ekf.metrics(trajectory=trajectory),
        }
    ]


def run_PF_single_theta_single_radio(
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
                "type": "PF_single_theta_single_radio",
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": pf.metrics(trajectory=trajectory),
            }
        )
    return metrics


def run_PF_single_theta_dual_radio(ds, theta_err=0.1, theta_dot_err=0.001, N=128):

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
            "type": "PF_single_theta_dual_radio",
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def run_PF_xy_dual_radio(ds, pos_err=15, vel_err=0.5, N=128 * 16):

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
            "type": "PF_xy_dual_radio",
            "vel_err": vel_err,
            "pos_err": pos_err,
            "N": N,
            "metrics": pf.metrics(trajectory=traj_paired),
        }
    ]


def runner(x):
    fn, args = x
    return fn(**args)


fn_key_to_fn = {
    "run_EKF_single_theta_single_radio": run_EKF_single_theta_single_radio,
    "run_EKF_single_theta_dual_radio": run_EKF_single_theta_dual_radio,
    "run_EKF_xy_dual_radio": run_EKF_xy_dual_radio,
    "run_PF_single_theta_single_radio": run_PF_single_theta_single_radio,
    "run_PF_single_theta_dual_radio": run_PF_single_theta_dual_radio,
    "run_PF_xy_dual_radio": run_PF_xy_dual_radio,
}


def config_to_job_params(config):
    jobs = [{}]
    for key, values in config.items():
        new_jobs = []
        for job in jobs:
            for value in values:
                if isinstance(value, str):
                    value = eval(value)  # TODO this might be dangerous
                d = job.copy()
                d.update({key: value})
                new_jobs.append(d)
        jobs = new_jobs
    if len(jobs) == 1 and len(jobs[0]) == 0:
        return []
    return jobs


def config_to_jobs(config):
    jobs = []
    for fn_key, fn_config in config.items():
        print(fn_key, fn_config)
        fn = fn_key_to_fn[fn_key]
        jobs += [(fn, job_params) for job_params in config_to_job_params(fn_config)]
    return jobs


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
        parser.add_argument(
            "--config",
            type=str,
            required=True,
        )

        return parser

    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    yaml_config = yaml.safe_load(open(args.config, "r"))
    jobs_per_ds_fn = config_to_jobs(yaml_config)

    random.shuffle(jobs_per_ds_fn)

    # one job per dataset
    # jobs = [
    #     {
    #         "ds_fn": ds_fn,
    #         "precompute_cache": args.precompute_cache,
    #         "empirical_pkl_fn": args.empirical_pkl_fn,
    #         "jobs": jobs_per_ds_fn,
    #     }
    #     for ds_fn in args.datasets
    # ]

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
        with Pool(28) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(
                tqdm.tqdm(
                    pool.imap(run_jobs_with_one_dataset, jobs),
                    total=len(jobs),
                )
            )

    final_results = []
    for result in results:
        final_results += result
    pickle.dump(results, open(args.output, "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
