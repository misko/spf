import argparse
import logging
import os
import random
import time
from functools import partial
from multiprocessing import Pool

import torch
import tqdm
import yaml

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.dataset.phase_corrected_dataset import PhaseCorrectedDataset
from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper
from spf.filters.ekf_dualradio_filter import SPFPairedKalmanFilter
from spf.filters.ekf_dualradioXY_filter import SPFPairedXYKalmanFilter
from spf.filters.ekf_single_radio_filter import SPFKalmanFilter
from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio
from spf.filters.particle_single_radio_nn_filter import PFSingleThetaSingleRadioNN
from spf.s3_utils import b2_file_to_local_with_cache, b2path_to_bucket_and_path
from spf.filters.results_store import make_results_store
from spf.utils import get_md5_of_file

checkpoints_cache_dir = None


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


def run_jobs_with_one_dataset(
    kwargs, checkpoints_cache_dir, already_processed=[], results_backend="local"
):
    # b2_reset_cache()  # different processes need different folders for cache
    result_fns = []
    store = None  # built lazily: a fully-skipped dataset should touch no sink

    for fn, fn_kwargs in kwargs["jobs"]:
        precompute_cache = fn_kwargs.pop("precompute_cache")
        segmentation_version = fn_kwargs.pop("segmentation_version")

        original_b2_paths = {}
        # Only b2:// checkpoints need fetching. b2path_to_bucket_and_path on a
        # local path yields an empty bucket and the config fetch below then tries
        # to download "b2:///mnt/...", so without this guard a local checkpoint
        # cannot be used at all.
        if "checkpoint_fn" in fn_kwargs and str(fn_kwargs["checkpoint_fn"]).startswith(
            "b2://"
        ):

            b2_checkpoint_fn = fn_kwargs["checkpoint_fn"]
            local_checkpoint_fn = b2_file_to_local_with_cache(
                fn_kwargs["checkpoint_fn"], b2_cache_folder=checkpoints_cache_dir
            )
            bucket, b2_path = b2path_to_bucket_and_path(b2_checkpoint_fn)
            config_b2_path = os.path.join(os.path.dirname(b2_path), "config.yml")
            _ = b2_file_to_local_with_cache(
                f"b2://{bucket}/{config_b2_path}", b2_cache_folder=checkpoints_cache_dir
            )

            original_b2_paths["checkpoint_fn"] = fn_kwargs["checkpoint_fn"]
            original_b2_paths["config_fn"] = (
                f"{os.path.dirname(fn_kwargs['checkpoint_fn'])}/config.yml"
            )
            fn_kwargs["checkpoint_fn"] = local_checkpoint_fn

        # before downloading anything lets check that it isnt already finished!

        workdir = fn_kwargs.pop("workdir")
        result_fn_without_workdir = (
            f"fn_{fn.__name__}"
            + f"/{segmentation_version:0.3f}"
            + f"/ds_{os.path.basename(kwargs['ds_fn'])}/"
            + "_"
            + args_to_str(fn_kwargs)
            + "results.pkl"
        )
        result_fn = workdir + "/" + result_fn_without_workdir

        if result_fn_without_workdir in already_processed:
            # print(
            #    f"Item with full_name={result_fn} already exists in DynamoDB. Skipping..."
            # )
            continue

        # with get_local_precompute_cache(precompute_cache) as local_precompute_cache

        with v5spfdataset_manager(
            prefix=kwargs["ds_fn"],
            nthetas=65,
            ignore_qc=True,
            precompute_cache=precompute_cache,
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
            segmentation_version=segmentation_version,
        ) as ds:

            # popped here, NOT earlier: it must still be in fn_kwargs when
            # args_to_str() builds result_fn_without_workdir above, or every arm
            # of a phase_correction sweep collides on one filename.
            phase_correction = fn_kwargs.pop("phase_correction", "none")
            phase_model_fn = fn_kwargs.pop("phase_model_fn", None)
            # A table built with a correction and consumed without it (or the
            # reverse) is a silent ~7 deg mismatch, measured harmful in
            # phasecorr_direct_pf_20260814_v1. Fail loudly instead.
            table_pc = (
                getattr(ds, "empirical_data", {})
                .get("__provenance__", {})
                .get("params", {})
                .get("phase_correction", "none")
            )
            assert table_pc == phase_correction, (
                f"empirical table was built with phase_correction={table_pc!r} but "
                f"this run uses {phase_correction!r}; they must match"
            )
            fn_kwargs["ds"] = (
                ds
                if phase_correction == "none"
                else PhaseCorrectedDataset(ds, phase_correction, phase_model_fn)
            )
            new_results = fn(**fn_kwargs)
            for result in new_results:
                # NOTE: deliberately not storing result_fn_without_workdir on the
                # result. It embeds the dataset basename, and the reporter builds
                # its grouping key from every non-metric field -- storing it puts
                # each dataset in its own group and defeats the cross-dataset
                # averaging the report exists to do.
                result["ds_fn"] = kwargs["ds_fn"]
                result["segmentation_version"] = segmentation_version
                result["precompute_cache"] = precompute_cache
                result["phase_correction"] = phase_correction
                # report the b2 path the job was configured with, not the local
                # cache copy it was actually read from
                for replace_key, replace_value in original_b2_paths.items():
                    if replace_key in result:
                        result[replace_key] = replace_value

            if store is None:
                store = make_results_store(results_backend, workdir)
            store.put(result_fn_without_workdir, new_results)

            result_fns.append(result_fn)

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
                "frequency": ds.carrier_frequencies[0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "phi_std": phi_std,
                "p": p,
                "noise_std": noise_std,
                "dynamic_R": dynamic_R,
                "metrics": metrics,
                "routine": (
                    ds.yaml_config["routine"]
                    if "routine" in ds.yaml_config
                    else "unknown"
                ),
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
            "frequency": ds.carrier_frequencies[0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dynamic_R": dynamic_R,
            "metrics": metrics,
            "routine": (
                ds.yaml_config["routine"] if "routine" in ds.yaml_config else "unknown"
            ),
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
            "frequency": ds.carrier_frequencies[0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "phi_std": phi_std,
            "p": p,
            "noise_std": noise_std,
            "dynamic_R": dynamic_R,
            "metrics": metrics,
            "routine": (
                ds.yaml_config["routine"] if "routine" in ds.yaml_config else "unknown"
            ),
        }
    ]


def run_PF_single_theta_single_radio_NN(
    ds,
    checkpoint_fn,
    inference_cache,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
    seed=0,
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
            seed=seed,
        )
        metrics = pf.metrics(trajectory=trajectory)
        metrics["runtime"] = time.time() - start_time
        all_metrics.append(
            {
                "type": "PF_single_theta_single_radio_NN",
                "frequency": ds.carrier_frequencies[0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "seed": seed,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": metrics,
                "checkpoint_fn": checkpoint_fn,
                "config_fn": config_fn,
                "routine": (
                    ds.yaml_config["routine"]
                    if "routine" in ds.yaml_config
                    else "unknown"
                ),
            }
        )
    return all_metrics


def run_PF_single_theta_single_radio(
    ds,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
    seed=0,
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
            seed=seed,
        )
        metrics = pf.metrics(trajectory=trajectory)
        metrics["runtime"] = time.time() - start_time
        all_metrics.append(
            {
                "type": "PF_single_theta_single_radio",
                "frequency": ds.carrier_frequencies[0],
                "rx_wavelength_spacing": ds.rx_wavelength_spacing,
                "rx_idx": rx_idx,
                "theta_err": theta_err,
                "seed": seed,
                "theta_dot_err": theta_dot_err,
                "N": N,
                "metrics": metrics,
                "routine": (
                    ds.yaml_config["routine"]
                    if "routine" in ds.yaml_config
                    else "unknown"
                ),
            }
        )
    return all_metrics


def run_PF_single_theta_dual_radio(
    ds, theta_err=0.1, theta_dot_err=0.001, N=128, progress_bar=False, seed=0
):
    start_time = time.time()
    pf = PFSingleThetaDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
        progress_bar=progress_bar,
        seed=seed,
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_single_theta_dual_radio",
            "frequency": ds.carrier_frequencies[0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "theta_err": theta_err,
            "seed": seed,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": metrics,
            "routine": (
                ds.yaml_config["routine"] if "routine" in ds.yaml_config else "unknown"
            ),
        }
    ]


def run_PF_single_theta_dual_radio_NN(
    ds,
    checkpoint_fn,
    inference_cache=None,
    theta_err=0.1,
    theta_dot_err=0.001,
    N=128,
    absolute=False,
    steps=None,
    seed=0,
):
    config_fn = f"{os.path.dirname(checkpoint_fn)}/config.yml"
    start_time = time.time()

    nn_ds = v5spfdataset_nn_wrapper(
        ds,
        config_fn,
        checkpoint_fn,
        inference_cache,
        device="cpu",
        absolute=absolute,
    )
    pf = PFSingleThetaDualRadioNN(nn_ds=nn_ds)
    traj_paired = pf.trajectory(
        mean=torch.tensor([[0, 0]]),
        N=N,
        std=torch.tensor([[20, 0.1]]),  # 20 should be random enough to loop around
        noise_std=torch.tensor([[theta_err, theta_dot_err]]),
        return_particles=False,
        steps=steps,
        seed=seed,
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_single_theta_dual_radio_NN",
            "frequency": ds.carrier_frequencies[0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "theta_err": theta_err,
            "seed": seed,
            "theta_dot_err": theta_dot_err,
            "N": N,
            "metrics": metrics,
            "checkpoint_fn": checkpoint_fn,
            "config_fn": config_fn,
            "absolute": absolute,
            "routine": (
                ds.yaml_config["routine"] if "routine" in ds.yaml_config else "unknown"
            ),
        }
    ]


def run_PF_xy_dual_radio(ds, pos_err=15, vel_err=0.5, N=128 * 16, seed=0):

    start_time = time.time()
    # dual radio dual
    pf = PFXYDualRadio(ds=ds)
    traj_paired = pf.trajectory(
        N=N,
        mean=torch.tensor([[0, 0, 0, 0, 0]]),
        std=torch.tensor([[0, 200, 200, 0.1, 0.1]]),
        return_particles=False,
        noise_std=torch.tensor([[0, pos_err, pos_err, vel_err, vel_err]]),
        seed=seed,
    )
    metrics = pf.metrics(trajectory=traj_paired)
    metrics["runtime"] = time.time() - start_time
    return [
        {
            "type": "PF_xy_dual_radio",
            "frequency": ds.carrier_frequencies[0],
            "rx_wavelength_spacing": ds.rx_wavelength_spacing,
            "vel_err": vel_err,
            "seed": seed,
            "pos_err": pos_err,
            "N": N,
            "metrics": metrics,
            "routine": (
                ds.yaml_config["routine"] if "routine" in ds.yaml_config else "unknown"
            ),
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


#
# {
# kwarg1: v1 , v2
# kwarg2: v3 , v4
# }
#
# -> (kwarg1: v1, kwarg2: v3) , (kwarg1: v1, kwarg2: v4) ...
#
def config_to_job_params(config):
    jobs = [{}]
    for key, values in config.items():
        new_jobs = []
        for job in jobs:
            for value in values:
                if isinstance(value, str):
                    try:
                        value = eval(value)  # TODO this might be dangerous
                    except (SyntaxError, NameError):
                        # NameError matters for string-valued axes: a path
                        # like "/mnt/..." is a SyntaxError and was already
                        # tolerated, but a bare word like "none" is a valid
                        # identifier and raised out of the sweep.
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


def add_precompute_cache_to_job(job, precompute_caches):
    fn, args = job
    args = args.copy()
    if "segmentation_version" in args:
        args["precompute_cache"] = precompute_caches[args["segmentation_version"]]
    elif "checkpoint_fn_and_segmentation_version" in args:
        d = args["checkpoint_fn_and_segmentation_version"]
        args["checkpoint_fn"] = d["checkpoint_fn"]
        args["segmentation_version"] = d["segmentation_version"]
        args.pop("checkpoint_fn_and_segmentation_version")
        args["precompute_cache"] = precompute_caches[args["segmentation_version"]]
    else:
        raise ValueError(
            "Must have segmentation_version or checkpoint_fn_and_segmentation_version in job"
        )
    return (fn, args)


def generate_configs_to_run(
    yaml_config_fn, work_dir, dataset_fns, seed, empirical_pkl_fn
):

    try:
        os.makedirs(work_dir)
    except FileExistsError as e:
        pass
    yaml_config = yaml.safe_load(open(yaml_config_fn, "r"))
    jobs_per_ds_fn = config_to_jobs(yaml_config)

    # assign the precompute cache
    jobs_per_ds_fn = [
        add_precompute_cache_to_job(job, yaml_config["precompute_caches"])
        for job in jobs_per_ds_fn
    ]

    for _, job_params in jobs_per_ds_fn:
        assert "workdir" not in job_params
        job_params["workdir"] = work_dir

    random.seed(seed)
    random.shuffle(jobs_per_ds_fn)

    dataset_fns = sorted(dataset_fns)
    if len(dataset_fns) == 1 and dataset_fns[0][-4:] == ".txt":
        dataset_fns = [x.strip() for x in open(dataset_fns[0]).readlines()]

    random.seed(seed)
    random.shuffle(dataset_fns)

    jobs = []
    # try to read the same ds back to back so that OS can cache it
    # job [0] = fn, job[1] = args
    for ds_fn in dataset_fns:
        for job in jobs_per_ds_fn:
            jobs.append(
                {
                    "ds_fn": ds_fn,
                    "empirical_pkl_fn": empirical_pkl_fn,
                    "jobs": [[job[0], job[1].copy()]],
                }
            )
    return jobs

    # final_results = []
    # for result in results:
    #     final_results += result
    # pickle.dump(results, open(args.output, "wb"))

    # run_single_theta_single_radio()
    # run_single_theta_dual_radio(
    #     ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn
    # )
    # run_xy_dual_radio(ds_fn=ds_fn, precompute_fn=precompute_fn, full_p_fn=full_p_fn)


def run_filter_jobs(
    jobs,
    nparallel,
    debug=False,
    checkpoints_cache_dir=None,
    already_processed=[],
    results_backend="local",
):
    f = partial(
        run_jobs_with_one_dataset,
        checkpoints_cache_dir=checkpoints_cache_dir,
        already_processed=already_processed,
        results_backend=results_backend,
    )
    if debug:
        _ = list(
            tqdm.tqdm(
                map(f, jobs),
                total=len(jobs),
            )
        )
    else:
        torch.set_num_threads(1)
        with Pool(nparallel) as pool:  # cpu_count())  # cpu_count() // 4)
            _ = list(
                tqdm.tqdm(
                    pool.imap_unordered(f, jobs),
                    total=len(jobs),
                )
            )


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
        parser.add_argument(
            "--results-backend",
            type=str,
            choices=["local", "dynamodb"],
            default="local",
            help=(
                "local (default): one results.pkl per job under --work-dir, which "
                "is what run_filters_report.py reads. dynamodb: put_item into the "
                "'filter_metrics' table -- needs AWS credentials and writes to the "
                "network."
            ),
        )
        parser.add_argument(
            "--checkpoints-cache-dir",
            type=str,
            default=None,
            required=False,
            help="local cache for b2:// checkpoints; unused for local paths",
        )
        parser.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="skip jobs already present in the results store",
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
    jobs = generate_configs_to_run(
        yaml_config_fn=args.config,
        work_dir=args.work_dir,
        dataset_fns=args.datasets,
        seed=args.seed,
        empirical_pkl_fn=args.empirical_pkl_fn,
    )
    already_processed = []
    if args.resume:
        already_processed = make_results_store(
            args.results_backend, args.work_dir
        ).already_processed()
        if already_processed:
            logging.info(f"resuming: {len(already_processed)} results already present")
    run_filter_jobs(
        jobs,
        nparallel=args.parallel,
        debug=args.debug,
        checkpoints_cache_dir=args.checkpoints_cache_dir,
        already_processed=already_processed,
        results_backend=args.results_backend,
    )
