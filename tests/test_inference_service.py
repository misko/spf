"""Transport and lifecycle tests. Real scientific parity is verified separately."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from spf.inference_service import service


@pytest.fixture
def catalog(tmp_path):
    capture = tmp_path / "capture.zarr"
    capture.mkdir()
    capture.with_suffix(".yaml").write_text("{}")
    cache = tmp_path / "precompute"
    cache.mkdir()
    empirical = tmp_path / "empirical.pkl"
    empirical.write_bytes(b"fixture")
    config = {"datasets": {"capture": {"path": str(capture), "precompute_cache": str(cache), "empirical": str(empirical), "segmentation_version": 3.5}}, "models": {}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(config))
    return service.load_catalog(path)


def wait_for(manager, ident, states):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = manager.get(ident)
        if record["state"] in states:
            return record
        time.sleep(.02)
    raise AssertionError(manager.get(ident))


def request():
    return {"dataset_ids": ["capture"], "algorithm": "ekf_dual"}


@pytest.mark.parametrize("change", [
    {"dataset_ids": ["missing"]}, {"dataset_ids": ["capture", "capture"]},
    {"params": {"phi_std": float("nan")}}, {"params": {"phi_std": 10**400}},
    {"params": {"noise_std": -1}}, {"params": {"wrong": 1}},
    {"params": {"phi_std": 0, "dynamic_R": 0}}, {"seed": True},
    {"seed": -1}, {"algorithm": "pf_dual_nn"}, {"path": "/tmp/anything"},
])
def test_reject_invalid_scientific_requests(catalog, change):
    with pytest.raises(ValueError):
        service.validate_request({**request(), **change}, catalog)


def test_defaults_are_recorded(catalog):
    spec = service.validate_request(request(), catalog)
    assert spec["profile"] == "current-code"
    assert spec["params"] == {"phi_std": 12., "p": .1, "noise_std": .01, "dynamic_R": 0.}
    assert spec["seed"] == 0


def test_source_output_overlap_rejected(catalog, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        service.JobManager(catalog, Path(catalog["datasets"]["capture"]["precompute_cache"]) / "jobs")


def test_restart_marks_abandoned_jobs_interrupted(catalog, tmp_path):
    root = tmp_path / "jobs"
    for state in ["queued", "running", "succeeded"]:
        directory = root / state
        directory.mkdir(parents=True)
        service.atomic_json(directory / "status.json", {"id": state, "state": state, "created_at": 1})
    manager = service.JobManager(catalog, root)
    try:
        assert manager.get("queued")["state"] == "interrupted"
        assert manager.get("running")["state"] == "interrupted"
        assert manager.get("succeeded")["state"] == "succeeded"
        with pytest.raises(ValueError, match="another service"):
            service.JobManager(catalog, root)
    finally:
        manager.close()


def test_cancellation_waits_for_process_and_skips_queued_job(catalog, tmp_path, monkeypatch):
    original = subprocess.Popen
    def sleeping_worker(*args, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
    monkeypatch.setattr(service.subprocess, "Popen", sleeping_worker)
    manager = service.JobManager(catalog, tmp_path / "jobs")
    try:
        first = manager.submit(request())
        wait_for(manager, first["id"], {"running"})
        process = manager.process
        second = manager.submit(request())
        assert manager.cancel(second["id"])["state"] == "cancelled"
        assert manager.cancel(first["id"])["state"] == "cancelled"
        assert process.poll() is not None
        assert manager.get(second["id"])["state"] == "cancelled"
    finally:
        manager.close()


def test_http_runs_results_and_boundary_errors(catalog, tmp_path, monkeypatch):
    original = subprocess.Popen
    def successful_worker(command, **kwargs):
        directory = command[-1]
        code = "import json,pathlib; p=pathlib.Path(" + repr(directory) + "); (p/'results.json').write_text(json.dumps({'datasets': [], 'artifacts': []}))"
        return original([sys.executable, "-c", code], **kwargs)
    monkeypatch.setattr(service.subprocess, "Popen", successful_worker)
    manager = service.JobManager(catalog, tmp_path / "jobs")
    server = service.make_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        all_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=all_headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        connection.close()
        return response.status, data
    try:
        assert call("GET", "/api/catalog")[0] == 200
        assert call("POST", "/api/runs", {**request(), "params": {"phi_std": 10**400}})[0] == 400
        assert call("POST", "/api/runs", request(), {"Origin": "https://example.com"})[0] == 400
        assert call("GET", "/api/runs", headers={"Host": "example.com"})[0] == 400
        status, row = call("POST", "/api/runs", request())
        assert status == 202
        wait_for(manager, row["id"], {"succeeded"})
        assert call("GET", f"/api/runs/{row['id']}/results") == (200, {"datasets": [], "artifacts": []})
        assert call("GET", f"/api/runs/{row['id']}/artifacts/../../request.json")[0] == 404
        assert call("GET", "/api/runs/missing")[0] == 404
        assert service.read_json(manager.root / row["id"] / "request.json")["params"]["noise_std"] == .01
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join()


def test_same_basename_captures_get_separate_prediction_caches(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from spf.inference_service import worker
    from spf.filters import plot_filter_run, dump_tracks

    directory = tmp_path / "job"
    directory.mkdir()
    datasets = {}
    for ident in ["capture_a", "capture_b"]:
        parent = tmp_path / ident
        capture = parent / "same.zarr"
        capture.mkdir(parents=True)
        (capture / "data.mdb").write_bytes(ident.encode())
        capture.with_suffix(".yaml").write_text("{}")
        cache = parent / "cache"
        cache.mkdir()
        (cache / "same_segmentation_nthetas65.pkl").write_bytes(b"fixture")
        yarr = cache / "same_segmentation_nthetas65.yarr"
        yarr.mkdir()
        (yarr / "data.mdb").write_bytes(b"fixture")
        empirical = parent / "empirical.pkl"
        empirical.write_bytes(b"fixture")
        datasets[ident] = {"path": str(capture), "precompute_cache": str(cache), "empirical": str(empirical), "segmentation_version": 3.5}
    service.atomic_json(directory / "request.json", {"profile": "current-code", "dataset_ids": list(datasets), "datasets": datasets, "algorithm": "pf_dual", "params": service.DEFAULTS["pf_dual"], "seed": 0, "model": None})
    class Dataset:
        def __len__(self):
            return 2
        def close(self):
            pass
    caches = []
    def run_filter(ds, algorithm, params, **kwargs):
        caches.append(kwargs["inference_cache"])
        return np.array([0., .1]), np.array([.2, .2]), np.array([0., .1]), {"frame": "craft_relative"}
    class Figure:
        def savefig(self, path, **kwargs):
            Path(path).write_bytes(b"plot")
    monkeypatch.setattr(plot_filter_run, "open_dataset", lambda *args: Dataset())
    monkeypatch.setattr(plot_filter_run, "run_filter", run_filter)
    monkeypatch.setattr(plot_filter_run, "summarize_run", lambda *args: {"mse": 0.})
    monkeypatch.setattr(plot_filter_run, "plot_filter_run", lambda *args: Figure())
    monkeypatch.setattr(worker.sys, "addaudithook", lambda *args: None)
    from matplotlib import pyplot
    monkeypatch.setattr(pyplot, "close", lambda *args: None)
    worker.execute(directory)
    assert len(set(caches)) == 2
    assert all(Path(path).is_relative_to(directory) for path in caches)
    result = service.read_json(directory / "results.json")
    assert [x["dataset_id"] for x in result["datasets"]] == list(datasets)
    assert result["aggregate"]["mean_dataset_mse"] == 0.


def test_job_model_copy_preserves_inputs_and_contains_checksum_sidecars(tmp_path):
    from spf.inference_service import worker
    from spf.utils import get_md5_of_file

    source = tmp_path / "source"
    source.mkdir()
    checkpoint, config = source / "best.pth", source / "config.yml"
    checkpoint.write_bytes(b"model weights")
    config.write_bytes(b"datasets:\n  empirical_data_fn: /original/table.pkl\n")
    request = {"model": {"checkpoint": str(checkpoint)}, "datasets": {"a": {"path": "original"}}}
    original = json.loads(json.dumps(request))
    directory = tmp_path / "job"
    directory.mkdir()
    identity = {"checkpoint": worker.file_identity(checkpoint), "config": worker.file_identity(config)}

    execution, staged = worker.prepare_execution_request(request, directory, identity)

    assert request == original
    copied_checkpoint = Path(execution["model"]["checkpoint"])
    assert copied_checkpoint.is_relative_to(directory)
    assert copied_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert copied_checkpoint.with_name("config.yml").read_bytes() == config.read_bytes()
    for path in [copied_checkpoint, copied_checkpoint.with_name("config.yml")]:
        assert get_md5_of_file(str(path))
        assert Path(str(path) + ".md5").is_file()
    assert not checkpoint.with_suffix(".pth.md5").exists()
    assert not config.with_suffix(".yml.md5").exists()
    assert staged["checkpoint"]["sha256"] == identity["checkpoint"]["sha256"]
    assert staged["config"]["sha256"] == identity["config"]["sha256"]
    execution["datasets"]["a"]["path"] = "changed"
    assert request == original


def test_model_input_change_is_rejected_before_execution(tmp_path):
    from spf.inference_service import worker

    checkpoint, config = tmp_path / "best.pth", tmp_path / "config.yml"
    checkpoint.write_bytes(b"original weights")
    config.write_bytes(b"{}")
    identities = {"checkpoint": worker.file_identity(checkpoint), "config": worker.file_identity(config)}
    checkpoint.write_bytes(b"changed weights")
    directory = tmp_path / "job"
    directory.mkdir()
    with pytest.raises(ValueError, match="Model input changed"):
        worker.prepare_execution_request({"model": {"checkpoint": str(checkpoint)}}, directory, identities)
    assert not (directory / "model" / "best.pth").exists()
