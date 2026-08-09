"""End-to-end: config -> local sweep -> results store -> report.

Guards the two defects that made a local run impossible:

* ``run_filter_jobs(jobs)`` in ``__main__`` omitted the required ``nparallel``,
  so invoking the runner directly raised TypeError before doing any work.
* The runner wrote to DynamoDB behind a hard-coded ``use_file_system = False``
  while ``run_filters_report.py`` read ``.pkl`` from the work dir, so the two
  halves of the pipeline pointed at different sinks.

Uses the tiny synthetic fixture, so it needs no /mnt access and no credentials.
"""

import os
import subprocess
import sys
import tempfile

import yaml

from spf.filters.report import build_report
from spf.filters.run_filters_on_data import (
    generate_configs_to_run,
    run_filter_jobs,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(precompute_cache):
    return {
        "precompute_caches": {3.7: precompute_cache},
        "runs": [
            {
                "run_EKF_single_theta_dual_radio": {
                    "phi_std": [10.0],
                    "p": [5.0],
                    "noise_std": [0.001],
                    "dynamic_R": [0.0],
                    "segmentation_version": [3.7],
                }
            },
            {
                "run_PF_single_theta_dual_radio": {
                    "N": [128],
                    "theta_err": [0.01],
                    "theta_dot_err": [0.01],
                    "segmentation_version": [3.7],
                }
            },
        ],
    }


def _run_sweep(noise1_n128_obits2, workdir, config_fn):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    with open(config_fn, "w") as f:
        yaml.safe_dump(_config(dirname), f)
    jobs = generate_configs_to_run(
        yaml_config_fn=config_fn,
        work_dir=workdir,
        dataset_fns=[ds_fn + ".zarr"],
        seed=0,
        empirical_pkl_fn=empirical_pkl_fn,
    )
    run_filter_jobs(jobs, nparallel=2, debug=True, results_backend="local")
    return jobs


def test_local_sweep_writes_results_the_reporter_can_read(noise1_n128_obits2):
    with tempfile.TemporaryDirectory() as d:
        workdir = os.path.join(d, "wd")
        jobs = _run_sweep(noise1_n128_obits2, workdir, os.path.join(d, "c.yml"))
        assert len(jobs) == 2

        written = [
            os.path.join(dp, f)
            for dp, _, fs in os.walk(workdir)
            for f in fs
            if f.endswith(".pkl")
        ]
        assert len(written) == 2, written

        report = build_report(workdir)
        assert report["n_results"] >= 2
        assert all("frame" in row for row in report["rows"])
        # every run must carry a real error metric
        for row in report["rows"]:
            assert any(k.startswith("mse") and k.endswith("_mean") for k in row), row


def test_sweep_is_resumable(noise1_n128_obits2):
    """A second pass with the same work dir must skip everything already done."""
    with tempfile.TemporaryDirectory() as d:
        workdir = os.path.join(d, "wd")
        config_fn = os.path.join(d, "c.yml")
        _run_sweep(noise1_n128_obits2, workdir, config_fn)
        before = {
            os.path.join(dp, f): os.path.getmtime(os.path.join(dp, f))
            for dp, _, fs in os.walk(workdir)
            for f in fs
        }

        from spf.filters.results_store import LocalResultsStore

        dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
        jobs = generate_configs_to_run(
            yaml_config_fn=config_fn,
            work_dir=workdir,
            dataset_fns=[ds_fn + ".zarr"],
            seed=0,
            empirical_pkl_fn=empirical_pkl_fn,
        )
        run_filter_jobs(
            jobs,
            nparallel=2,
            debug=True,
            already_processed=LocalResultsStore(workdir).already_processed(),
            results_backend="local",
        )
        after = {
            os.path.join(dp, f): os.path.getmtime(os.path.join(dp, f))
            for dp, _, fs in os.walk(workdir)
            for f in fs
        }
        assert before == after, "resumed sweep rewrote already-finished results"


def test_runner_main_accepts_its_own_arguments(noise1_n128_obits2):
    """`run_filter_jobs(jobs)` in __main__ used to raise TypeError immediately."""
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    with tempfile.TemporaryDirectory() as d:
        config_fn = os.path.join(d, "c.yml")
        with open(config_fn, "w") as f:
            yaml.safe_dump(_config(dirname), f)
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO, "spf", "filters", "run_filters_on_data.py"),
                "-d",
                ds_fn + ".zarr",
                "--empirical-pkl-fn",
                empirical_pkl_fn,
                "--work-dir",
                os.path.join(d, "wd"),
                "--config",
                config_fn,
                "--results-backend",
                "local",
                "--parallel",
                "2",
                "--debug",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=REPO,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "TypeError" not in proc.stderr
        produced = [
            f
            for _, _, fs in os.walk(os.path.join(d, "wd"))
            for f in fs
            if f.endswith(".pkl")
        ]
        assert len(produced) == 2, proc.stdout + proc.stderr
