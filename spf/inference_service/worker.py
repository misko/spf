"""One isolated scientific job using existing dataset, filter, and plotting code."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import copy
import shutil
import subprocess
import sys
import time

if sys.platform == "linux" and "SPF_INFERENCE_PARENT_PID" in os.environ:
    import ctypes
    import signal
    expected_parent = int(os.environ["SPF_INFERENCE_PARENT_PID"])
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "Could not arm parent-death signal")
    if os.getppid() != expected_parent:
        raise SystemExit("Inference server exited before worker startup")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from spf.inference_service.service import atomic_json, read_json


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def prepare_execution_request(request, directory, model_identity):
    execution = copy.deepcopy(request)
    if not request["model"]:
        return execution, None
    model_directory = directory / "model"
    model_directory.mkdir()
    checkpoint = Path(request["model"]["checkpoint"])
    staged = {}
    for name, source in [("checkpoint", checkpoint), ("config", checkpoint.with_name("config.yml"))]:
        destination = model_directory / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if sha256(temporary) != model_identity[name]["sha256"]:
            temporary.unlink()
            raise ValueError(f"Model input changed before execution: {source}")
        os.replace(temporary, destination)
        staged[name] = file_identity(destination)
    execution["model"]["checkpoint"] = staged["checkpoint"]["path"]
    staged["verified_against_source"] = True
    staged["policy"] = "Byte-identical job-local model inputs; existing checksum sidecars stay in the job directory."
    return execution, staged


def execute(directory):
    request = read_json(directory / "request.json")
    protected = [REPO / "spf"]
    for ds in request["datasets"].values():
        protected.extend([Path(ds["path"]), Path(ds["path"]).with_suffix(".yaml"), Path(ds["precompute_cache"]), Path(ds["empirical"])])
    if request["model"]:
        protected.append(Path(request["model"]["checkpoint"]).parent)
    protected = [x.resolve() for x in protected]

    def audit(event, values):
        paths = []
        if event == "open" and values[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            paths = [values[0]]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate"}:
            paths = [values[0]]
        elif event == "os.rename":
            paths = values[:2]
        elif event in {"os.link", "os.symlink"}:
            paths = [values[1]]
        for path in paths:
            if isinstance(path, (str, bytes)) and any(Path(os.fsdecode(path)).resolve().is_relative_to(root) for root in protected):
                raise PermissionError(f"Source write prohibited: {event} {path}")
    sys.addaudithook(audit)
    import numpy as np
    import torch
    import yaml
    from matplotlib import pyplot as plt
    from spf.filters.dump_tracks import save_track
    from spf.filters.plot_filter_run import open_dataset, run_filter, summarize_run, plot_filter_run

    torch.set_num_threads(2)
    seed = request.get("seed", 0)
    progress = lambda **kw: atomic_json(directory / "progress.json", {"updated_at": time.time(), **kw})
    progress(stage="recording provenance", completed=0, total=len(request["dataset_ids"]))
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    provenance = {"profile": request["profile"], "request": request, "entry_points": ["spf.filters.plot_filter_run.open_dataset", "spf.filters.plot_filter_run.run_filter", "spf.filters.plot_filter_run.summarize_run"], "device": "cpu", "torch_threads": 2, "seed": {"python": seed, "numpy": seed, "torch": seed, "filter": seed}, "dependencies": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "lmdb", "zarr", "filterpy", "matplotlib"]}, "code": {}, "inputs": {}, "metric_semantics": "Current-code circular angular metrics. Equal dataset weighting in aggregate. Not a frozen PDF-era execution profile.", "precision": "Existing run_filter outputs preserved without additional float32 casts", "cache_policy": "Job-local fresh prediction cache, independently namespaced by dataset identifier; completed outputs reused for viewing only"}
    try:
        provenance["git_revision"] = git("rev-parse", "HEAD")
        provenance["git_modified_paths"] = git("diff", "--name-only").splitlines()
    except subprocess.CalledProcessError:
        provenance["git_revision"] = None
    for path in sorted((REPO / "spf").rglob("*.py")):
        provenance["code"][str(path.relative_to(REPO))] = sha256(path)
    identities = {}
    def record(path):
        path = Path(path).resolve()
        if str(path) not in identities:
            identities[str(path)] = file_identity(path)
        return identities[str(path)]
    for ident, ds in request["datasets"].items():
        path = Path(ds["path"])
        stem = path.stem + "_segmentation_nthetas65"
        cache = Path(ds["precompute_cache"])
        provenance["inputs"][ident] = {"capture": record(path / "data.mdb"), "yaml": record(path.with_suffix(".yaml")), "segmentation": record(cache / (stem + ".pkl")), "features": record(cache / (stem + ".yarr") / "data.mdb"), "calibration": record(ds["empirical"]), "segmentation_version": ds["segmentation_version"]}
    if request["model"]:
        checkpoint = Path(request["model"]["checkpoint"])
        config = checkpoint.with_name("config.yml")
        model_config = yaml.safe_load(config.read_text())
        empirical = model_config["datasets"].get("empirical_data_fn")
        if empirical:
            protected.append(Path(empirical).resolve())
        provenance["model"] = {"checkpoint": record(checkpoint), "config": record(config), "inference_calibration": record(empirical) if empirical else None, "empirical_input": model_config["global"].get("empirical_input", False), "note": "NN loading uses the existing checkpoint configuration, including its own calibration path; the dataset catalog calibration applies to the outer filter dataset."}
    if request.get("kind") == "sweep":
        provenance["seed"] = {"seeds": request["seeds"], "policy": "Reset Python, NumPy, and Torch before each capture/configuration/seed trial; pass the seed to the existing filter."}
        provenance["cache_policy"] = "Fresh dataset-specific prediction cache reused only across configurations and seeds of this sweep"
    execution_request, execution_model = prepare_execution_request(request, directory, provenance.get("model"))
    if execution_model is not None:
        provenance["execution_model"] = execution_model
    atomic_json(directory / "provenance.json", clean(provenance))
    if request.get("kind") == "sweep":
        from spf.inference_service.sweeps import run_sweep
        result = run_sweep(execution_request, directory, progress)
    else:
        result = {"kind": "evaluation", "cohort_id": request.get("cohort_id"), "cohort": request.get("cohort"), "profile": request["profile"], "algorithm": request["algorithm"], "seed": seed, "datasets": [], "artifacts": ["provenance.json"]}
        for index, ident in enumerate(request["dataset_ids"]):
            ds_spec = request["datasets"][ident]
            progress(stage="inference", dataset_id=ident, completed=index, total=len(request["dataset_ids"]))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            ds = None
            try:
                ds = open_dataset(ds_spec["path"], ds_spec["precompute_cache"], ds_spec["empirical"], ds_spec["segmentation_version"])
                params = {**request["params"], "seed": seed}
                theta, sigma, gt, extras = run_filter(ds, request["algorithm"], params, checkpoint_fn=execution_request["model"]["checkpoint"] if execution_request["model"] else None, inference_cache=str(directory / "prediction_cache" / ident))
                if len(theta) != len(ds) or not len(theta) == len(sigma) == len(gt):
                    raise ValueError("Frame count mismatch; refusing a truncated evaluation")
                metrics = summarize_run(theta, sigma, gt)
                prefix = f"dataset_{index}"
                track_name, plot_name, json_name = prefix + ".npz", prefix + ".png", prefix + ".json"
                save_track(str(directory / track_name), theta=theta, sigma=sigma, gt=gt, frame=extras["frame"], seed=seed)
                arrays = {"theta": theta.tolist(), "sigma": sigma.tolist(), "gt": gt.tolist(), "frame": extras["frame"], "frames": len(theta), "dataset_id": ident}
                if extras.get("posterior") is not None:
                    arrays["posterior"] = extras["posterior"].tolist()
                atomic_json(directory / json_name, clean(arrays))
                fig = plot_filter_run(ds, theta, sigma, gt, extras, f"{ident} / {request['algorithm']} / current-code / seed {seed}")
                fig.savefig(directory / plot_name, dpi=100)
                plt.close(fig)
                result["datasets"].append({"dataset_id": ident, "frame": extras["frame"], "frames": len(theta), "metrics": clean(metrics), "tracks": json_name, "npz": track_name, "plot": plot_name})
                result["artifacts"].extend([track_name, plot_name, json_name])
            finally:
                if ds is not None:
                    ds.close()
        frames = {x["frame"] for x in result["datasets"]}
        if len(frames) == 1:
            result["aggregate"] = {"mean_dataset_mse": sum(x["metrics"]["mse"] for x in result["datasets"]) / len(result["datasets"]), "weighting": "equal weight per dataset", "frame": next(iter(frames))}
    for path, identity in identities.items():
        stat = Path(path).stat()
        if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Source changed during execution: {path}")
    atomic_json(directory / "results.json", clean(result))
    if request.get("kind") != "sweep":
        progress(stage="complete", completed=len(request["dataset_ids"]), total=len(request["dataset_ids"]))


if __name__ == "__main__":
    execute(Path(sys.argv[1]).resolve())
