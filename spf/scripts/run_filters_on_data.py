import argparse
import logging
import os
import pickle
import random
import time
from multiprocessing import Pool

import torch
import tqdm
import yaml

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.filters.ekf_dualradio_filter import SPFPairedKalmanFilter
from spf.filters.ekf_dualradioXY_filter import SPFPairedXYKalmanFilter
from spf.filters.ekf_single_radio_filter import SPFKalmanFilter
from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio
from spf.filters.particle_single_radio_nn_filter import PFSingleThetaSingleRadioNN
from spf.utils import get_md5_of_file


def args_to_str(args):
    str = ""
    for key in sorted(args.keys()):
        if "_fn" in key:
            md5 = get_md5_of_file(args[key])
            str += f"_{key}x{md5}_"
        elif "inference_cache" in key:
            pass
        elif key != "ds":
            str += f"_{key}x{args[key]}_"
    return str


def fake_runner(ds, **kwargs):
    for idx in range(len(ds)):
        a = ds[idx][0]


def run_jobs_with_one_dataset(kwargs):
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
        # fake_runner(ds=ds)
        # return
        result_fns = []
        # logging.info(kwargs["jobs"])
        for fn, fn_kwargs in kwargs["jobs"]:
            workdir = fn_kwargs.pop("workdir")
            result_fn = (
                workdir
                + "/fn_"
                + str(fn.__name__)
                + "/"
                + "_ds_"
                + os.path.basename(kwargs["ds_fn"])
                + "/"
                + "_"
                + args_to_str(fn_kwargs)
                + "results.pkl"
            )

            os.makedirs(os.path.dirname(result_fn), exist_ok=True)
            if not os.path.exists(result_fn):
                fn_kwargs["ds"] = ds
                new_results = fn(**fn_kwargs)
                for result in new_results:
                    result["ds_fn"] = kwargs["ds_fn"]
                pickle.dump(new_results, open(result_fn + ".tmp", "wb"))
                os.rename(result_fn + ".tmp", result_fn)
                assert os.path.exists(result_fn)
            else:
                # print("SKIPPING", result_fn)
                pass
            result_fns.append(result_fn)
            # breakpoint()
            # a = 1
        return result_fns


def run_EKF_single_theta_single_radio(ds, phi_std, p, noise_std, dynamic_R):
    all_metrics = []

    for rx_idx in [0, 1]:
        start_time = time.time()
        ekf = SPFKalmanFilter(
            ds=ds, rx_idx=rx_idx, phi_std=phi_std, p=p, dynamic_R=dynamic_R
        )
        trajectory = ekf.trajectory(
            dt=1.0,
            noise_std=0.01,
            max_iterations=None,
            debug=False,
        )

        metrics = ekf.metrics(trajectory=trajectory)
        metrics["runtime"] = time.time() - start_time
        all_metrics.append(
            {
                "type": "EKF_single_theta_single_radio",
                "frequency": ds.cached_keys[0]["rx_lo"][0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "phi_std": phi_std,
                "p": p,
                "noise_std": noise_std,
                "dynamic_R": dynamic_R,
                "metrics": metrics,
            }
        )
    return all_metrics


def run_EKF_single_theta_dual_radio(
    ds,
    phi_std,
    p,
    noise_std,
    dynamic_R,
):
    start_time = time.time()
    ekf = SPFPairedKalmanFilter(ds=ds, phi_std=phi_std, p=p, dynamic_R=dynamic_R)

    trajectory = ekf.trajectory(
        dt=1.0,
        noise_std=noise_std,
        max_iterations=None,
        debug=False,
    )
    metrics = ekf.metrics(trajectory=trajectory)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "EKF_single_theta_dual_radio",
            "frequency": ds.cached_keys[0]["rx_lo"][0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dynamic_R": dynamic_R,
            "metrics": metrics,
        }
    ]


def run_EKF_xy_dual_radio(
    ds,
    phi_std,
    p,
    noise_std,
    dynamic_R,
):
    start_time = time.time()
    ekf = SPFPairedXYKalmanFilter(ds=ds, phi_std=phi_std, p=p, dynamic_R=dynamic_R)

    trajectory = ekf.trajectory(
        dt=1.0,
        noise_std=noise_std,
        max_iterations=None,
        debug=False,
    )
    metrics = ekf.metrics(trajectory=trajectory)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "EKF_XY_dual_radio",
            "frequency": ds.cached_keys[0]["rx_lo"][0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dynamic_R": dynamic_R,
            "metrics": metrics,
        }
    ]


def run_PF_single_theta_single_radio_NN(
    ds,
    checkpoint_fn,
    inference_cache,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
):

    all_metrics = []
    config_fn = f"{os.path.dirname(checkpoint_fn)}/config.yml"
    for rx_idx in [0, 1]:
        start_time = time.time()
        pf = PFSingleThetaSingleRadioNN(
            ds,
            rx_idx,
            checkpoint_fn,
            config_fn,
            inference_cache=inference_cache,
            device="cpu",
        )
        trajectory = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
            noise_std=torch.tensor([[theta_err, theta_dot_err]]),
            return_particles=False,
            N=N,
        )
        metrics = pf.metrics(trajectory=trajectory)
        metrics["runtime"] = time.time() - start_time
        all_metrics.append(
            {
                "type": "PF_single_theta_single_radio_NN",
                "frequency": ds.cached_keys[0]["rx_lo"][0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": metrics,
                "checkpoint_fn": checkpoint_fn,
                "config_fn": config_fn,
            }
        )
    return all_metrics


def run_PF_single_theta_single_radio(
    ds,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
):

    all_metrics = []

    for rx_idx in [0, 1]:
        start_time = time.time()
        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx)
        trajectory = pf.trajectory(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
            noise_std=torch.tensor([[theta_err, theta_dot_err]]),
            return_particles=False,
            N=N,
        )
        metrics = pf.metrics(trajectory=trajectory)
        metrics["runtime"] = time.time() - start_time
        all_metrics.append(
            {
                "type": "PF_single_theta_single_radio",
                "frequency": ds.cached_keys[0]["rx_lo"][0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": metrics,
            }
        )
    return all_metrics


def run_PF_single_theta_dual_radio(ds, theta_err=0.1, theta_dot_err=0.001, N=128):
    start_time = time.time()
    pf = PFSingleThetaDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_single_theta_dual_radio",
            "frequency": ds.cached_keys[0]["rx_lo"][0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": metrics,
        }
    ]


def run_PF_single_theta_dual_radio_NN(
    ds,
    checkpoint_fn,
    inference_cache,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
):
    config_fn = f"{os.path.dirname(checkpoint_fn)}/config.yml"
    start_time = time.time()
    pf = PFSingleThetaDualRadioNN(
        ds=ds,
        checkpoint_fn=checkpoint_fn,
        config_fn=config_fn,
        inference_cache=inference_cache,
    )
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_single_theta_dual_radio_NN",
            "frequency": ds.cached_keys[0]["rx_lo"][0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "theta_err": theta_err,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": metrics,
            "checkpoint_fn": checkpoint_fn,
            "config_fn": config_fn,
        }
    ]


def run_PF_xy_dual_radio(ds, pos_err=15, vel_err=0.5, N=128 * 16):

    start_time = time.time()
    # dual radio dual
    pf = PFXYDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        N=N,
        mean=torch.tensor([[0, 0, 0, 0, 0]]),
        std=torch.tensor([[0, 200, 200, 0.1, 0.1]]),
        return_particles=False,
        noise_std=torch.tensor([[0, pos_err, pos_err, vel_err, vel_err]]),
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_xy_dual_radio",
            "frequency": ds.cached_keys[0]["rx_lo"][0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "vel_err": vel_err,
            "pos_err": pos_err,
            "N": N,
            "metrics": metrics,
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
    "run_PF_single_theta_single_radio_NN": run_PF_single_theta_single_radio_NN,
    "run_PF_single_theta_dual_radio_NN": run_PF_single_theta_dual_radio_NN,
}


def config_to_job_params(config):
    jobs = [{}]
    for key, values in config.items():
        new_jobs = []
        for job in jobs:
            for value in values:
                if isinstance(value, str):
                    try:
                        value = eval(value)  # TODO this might be dangerous
                    except SyntaxError:
                        pass
                d = job.copy()
                d.update({key: value})
                new_jobs.append(d)
        jobs = new_jobs
    if len(jobs) == 1 and len(jobs[0]) == 0:
        return []
    return jobs


def config_to_jobs(list_config):
    jobs = []
    for config in list_config["runs"]:
        for fn_key, fn_config in config.items():
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
            "--parallel",
            type=int,
            default=30,
            required=False,
        )
        parser.add_argument(
            "--work-dir",
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

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    try:
        os.makedirs(args.work_dir)
    except FileExistsError as e:
        pass
    yaml_config = yaml.safe_load(open(args.config, "r"))
    jobs_per_ds_fn = config_to_jobs(yaml_config)

    for _, job_params in jobs_per_ds_fn:
        assert "workdir" not in job_params
        job_params["workdir"] = args.work_dir

    random.seed(args.seed)
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

    dataset_fns = sorted(args.datasets)
    if len(dataset_fns) == 1 and dataset_fns[0][-4:] == ".txt":
        dataset_fns = [x.strip() for x in open(dataset_fns[0]).readlines()]

    random.seed(args.seed)
    random.shuffle(dataset_fns)

    jobs = []
    # try to read the same ds back to back so that OS can cache it
    for ds_fn in dataset_fns:
        for job in jobs_per_ds_fn:
            jobs.append(
                {
                    "ds_fn": ds_fn,
                    "precompute_cache": args.precompute_cache,
                    "empirical_pkl_fn": args.empirical_pkl_fn,
                    "jobs": [[job[0], job[1].copy()]],
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
        torch.set_num_threads(1)
        with Pool(args.parallel) as pool:  # cpu_count())  # cpu_count() // 4)
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(run_jobs_with_one_dataset, jobs),
                    total=len(jobs),
                )
            )

    # final_results = []
    # for result in results:
    #     final_results += result
    # pickle.dump(results, open(args.output, "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)
