"""Local job management and HTTP transport; scientific imports live in the worker."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROFILE = "current-code"
DEFAULTS = {
    "ekf_dual": {"phi_std": 12.0, "p": 0.1, "noise_std": 0.01, "dynamic_R": 0.0},
    "pf_dual": {"N": 16384, "theta_err": 0.1, "theta_dot_err": 0.001},
    "pf_dual_nn": {"N": 8192, "theta_err": 0.075, "theta_dot_err": 0.002, "absolute": True},
}
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def load_catalog(path):
    catalog = read_json(path)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("datasets"), dict) or not catalog["datasets"]:
        raise ValueError("catalog requires a nonempty datasets object")
    catalog.setdefault("models", {})
    if not isinstance(catalog["models"], dict):
        raise ValueError("models must be an object")
    for kind in ("datasets", "models"):
        for key, value in catalog[kind].items():
            if not ID.fullmatch(key) or not isinstance(value, dict):
                raise ValueError(f"invalid {kind} entry: {key}")
            required = ["path", "precompute_cache", "empirical", "segmentation_version"] if kind == "datasets" else ["checkpoint"]
            for field in required:
                if field not in value:
                    raise ValueError(f"{key}: missing {field}")
            for field in required:
                if field == "segmentation_version":
                    if not finite_number(value[field]) or value[field] <= 0:
                        raise ValueError(f"{key}: invalid segmentation_version")
                else:
                    p = Path(value[field]).expanduser().resolve()
                    if not p.exists():
                        raise ValueError(f"{key}: {field} does not exist: {p}")
                    value[field] = str(p)
            if kind == "datasets":
                if not Path(value["path"]).is_dir() or not value["path"].endswith(".zarr"):
                    raise ValueError(f"{key}: path must be a .zarr directory")
                if not Path(value["path"]).with_suffix(".yaml").is_file():
                    raise ValueError(f"{key}: matching .yaml is missing")
                if not Path(value["precompute_cache"]).is_dir() or not Path(value["empirical"]).is_file():
                    raise ValueError(f"{key}: invalid preprocessing or calibration path")
            elif not Path(value["checkpoint"]).is_file() or not Path(value["checkpoint"]).with_name("config.yml").is_file():
                raise ValueError(f"{key}: checkpoint and sibling config.yml are required")
    cohorts = catalog.setdefault("cohorts", {})
    if not isinstance(cohorts, dict):
        raise ValueError("cohorts must be an object")
    for ident, cohort in cohorts.items():
        if not ID.fullmatch(ident) or not isinstance(cohort, dict):
            raise ValueError("invalid cohort identifier or specification")
        ids = cohort.get("dataset_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or x not in catalog["datasets"] for x in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{ident}: cohort requires distinct known dataset_ids")
        if len({catalog["datasets"][x]["path"] for x in ids}) != len(ids):
            raise ValueError(f"{ident}: cohort contains duplicate capture paths")
        if cohort.get("role") not in {"development_tuning", "development_validation", "development_reporting", "historical_reference"}:
            raise ValueError(f"{ident}: unsupported cohort role")
        if not isinstance(cohort.get("description", ""), str) or not isinstance(cohort.get("overlap_notes", []), list) or any(not isinstance(x, str) for x in cohort.get("overlap_notes", [])):
            raise ValueError(f"{ident}: invalid cohort description or overlap notes")
    return catalog


def validate_params(algorithm, params):
    if set(params) != DEFAULTS[algorithm].keys():
        raise ValueError("unknown or missing filter parameters")
    for key, value in params.items():
        if key == "absolute":
            if not isinstance(value, bool):
                raise ValueError("absolute must be boolean")
        elif key == "N":
            if type(value) is not int or not 16 <= value <= 131072:
                raise ValueError("N must be an integer between 16 and 131072")
        elif not finite_number(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if algorithm == "ekf_dual" and params["phi_std"] == 0 and params["dynamic_R"] == 0:
        raise ValueError("phi_std and dynamic_R cannot both be zero")


def validate_request(body, catalog):
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    kind = body.get("kind", "evaluation")
    if not isinstance(kind, str):
        raise ValueError("request kind must be a string")
    common = {"kind", "dataset_ids", "cohort_id", "algorithm", "model_id"}
    allowed = common | ({"params", "seed"} if kind == "evaluation" else {"fixed_params", "grid", "seeds"})
    if kind not in {"evaluation", "sweep"} or set(body) - allowed:
        raise ValueError("unknown request kind or fields")
    cohort_id = body.get("cohort_id")
    cohort = None
    if cohort_id is not None:
        if "dataset_ids" in body or not isinstance(cohort_id, str) or cohort_id not in catalog.get("cohorts", {}):
            raise ValueError("select one known cohort_id or explicit dataset_ids")
        cohort = json.loads(json.dumps(catalog["cohorts"][cohort_id]))
        ids = cohort["dataset_ids"]
    else:
        ids = body.get("dataset_ids")
    if not isinstance(ids, list) or not ids or len(ids) > 100 or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset_ids must contain 1 to 100 distinct catalog identifiers")
    if any(x not in catalog["datasets"] for x in ids):
        raise ValueError("unknown dataset identifier")
    if len({catalog["datasets"][x]["path"] for x in ids}) != len(ids):
        raise ValueError("dataset_ids contain aliases of the same capture path")
    algorithm = body.get("algorithm")
    if not isinstance(algorithm, str) or algorithm not in DEFAULTS:
        raise ValueError("unknown algorithm")
    model_id = body.get("model_id")
    if algorithm == "pf_dual_nn":
        if not isinstance(model_id, str) or model_id not in catalog["models"]:
            raise ValueError("pf_dual_nn requires a catalog model_id")
    elif model_id is not None:
        raise ValueError("model_id applies only to pf_dual_nn")
    spec = {"kind": kind, "profile": PROFILE, "dataset_ids": list(ids), "cohort_id": cohort_id, "cohort": cohort, "algorithm": algorithm, "model_id": model_id}
    if kind == "sweep":
        from .sweeps import validate_sweep
        defaults = dict(DEFAULTS[algorithm])
        if algorithm == "pf_dual_nn":
            defaults["absolute"] = False
        spec.update(validate_sweep(body, defaults, lambda p: validate_params(algorithm, p), len(ids)))
    else:
        supplied = body.get("params", {})
        if not isinstance(supplied, dict):
            raise ValueError("params must be an object")
        params = {**DEFAULTS[algorithm], **supplied}
        validate_params(algorithm, params)
        seed = body.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        spec.update(params=params, seed=seed)
    return spec


class JobManager:
    def __init__(self, catalog, output_dir):
        self.catalog = catalog
        self.root = Path(output_dir).resolve()
        sources = [Path(__file__).resolve().parents[1]]
        for ds in catalog["datasets"].values():
            sources += [Path(ds["path"]), Path(ds["precompute_cache"]), Path(ds["empirical"])]
        for model in catalog["models"].values():
            sources.append(Path(model["checkpoint"]).parent)
        if any(inside(self.root, p) or inside(p, self.root) for p in sources):
            raise ValueError("output directory must be separate from source data, caches, models, and SPF code")
        self.root.mkdir(parents=True, exist_ok=True)
        self._file_lock = (self.root / ".server.lock").open("a+")
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file_lock.close()
            raise ValueError("another service owns this output directory")
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.jobs = {}
        self.process = None
        self.active_id = None
        self.stopping = False
        for path in sorted(self.root.glob("*/status.json")):
            record = read_json(path)
            if not ID.fullmatch(path.parent.name) or record.get("id") != path.parent.name:
                continue
            if record["state"] not in TERMINAL:
                record.update(state="interrupted", error="Service stopped before this run completed", finished_at=time.time())
                atomic_json(path, record)
            self.jobs[record["id"]] = record
        self.thread = threading.Thread(target=self._loop, name="inference-jobs", daemon=True)
        self.thread.start()

    def _save(self, record):
        atomic_json(self.root / record["id"] / "status.json", record)

    def submit(self, body):
        spec = validate_request(body, self.catalog)
        with self.condition:
            if self.stopping:
                raise ValueError("service is stopping")
            if sum(x["state"] in {"queued", "running"} for x in self.jobs.values()) >= 100:
                raise ValueError("job queue is full")
            ident = uuid.uuid4().hex
            directory = self.root / ident
            directory.mkdir()
            request = {**spec, "datasets": {x: self.catalog["datasets"][x] for x in spec["dataset_ids"]}, "model": self.catalog["models"].get(spec["model_id"])}
            atomic_json(directory / "request.json", request)
            record = {"id": ident, "state": "queued", "created_at": time.time(), "request": spec}
            self.jobs[ident] = record
            self._save(record)
            self.condition.notify_all()
            return dict(record)

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs:
                raise KeyError(ident)
            record = dict(self.jobs[ident])
            progress = self.root / ident / "progress.json"
            if progress.exists():
                record["progress"] = read_json(progress)
            return record

    def list(self):
        with self.lock:
            return [dict(x) for x in sorted(self.jobs.values(), key=lambda x: x["created_at"], reverse=True)]

    def cancel(self, ident):
        with self.condition:
            record = self.jobs[ident]
            if record["state"] in TERMINAL:
                return dict(record)
            if record["state"] == "queued":
                record.update(state="cancelled", finished_at=time.time())
                self._save(record)
                return dict(record)
            record["cancel_requested"] = True
            self._save(record)
            process = self.process
        if process is not None:
            self._terminate(process)
        with self.condition:
            self.condition.wait_for(lambda: self.jobs[ident]["state"] in TERMINAL, timeout=10)
            return dict(self.jobs[ident])

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or any(x["state"] == "queued" for x in self.jobs.values()))
                if self.stopping:
                    return
                record = next(x for x in self.jobs.values() if x["state"] == "queued")
                directory = self.root / record["id"]
                log = (directory / "worker.log").open("wb")
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(directory / "mpl"), NUMBA_CACHE_DIR=str(directory / "numba"), TMPDIR=str(directory / "tmp"), SPF_INFERENCE_PARENT_PID=str(os.getpid()))
                (directory / "tmp").mkdir()
                try:
                    self.process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).with_name("worker.py")), str(directory)], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    self.active_id = record["id"]
                    record.update(state="running", started_at=time.time())
                    self._save(record)
                except Exception as error:
                    log.close()
                    record.update(state="failed", error=str(error), finished_at=time.time())
                    self._save(record)
                    continue
                process = self.process
            code = process.wait()
            log.close()
            with self.condition:
                state = "cancelled" if record.get("cancel_requested") else ("succeeded" if code == 0 and (directory / "results.json").is_file() else "failed")
                record.update(state=state, returncode=code, finished_at=time.time())
                if state == "failed":
                    record["error"] = "Worker failed; inspect worker.log"
                self._save(record)
                self.process = None
                self.active_id = None
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            process = self.process
            if self.active_id:
                self.jobs[self.active_id]["cancel_requested"] = True
            for record in self.jobs.values():
                if record["state"] == "queued":
                    record.update(state="interrupted", error="Service shut down", finished_at=time.time())
                    self._save(record)
            self.condition.notify_all()
        if process is not None:
            self._terminate(process)
        self.thread.join(timeout=10)
        self._file_lock.close()


def make_server(manager, host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("this local service binds only to loopback")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            blob = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(blob)

        def local_request(self):
            authority = self.headers.get("Host", "")
            parsed = urlparse("http://" + authority)
            if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.port != self.server.server_port:
                raise ValueError("invalid Host header")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + authority:
                raise ValueError("cross-origin requests are not allowed")

        def do_POST(self):
            try:
                self.local_request()
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request body must be 1 to 65536 bytes")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/runs":
                    self.reply(202, manager.submit(body))
                else:
                    parts = self.path.strip("/").split("/")
                    if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                        self.reply(200, manager.cancel(parts[2]))
                    else:
                        self.reply(404, {"error": "not found"})
            except (ValueError, RecursionError) as error:
                self.reply(400, {"error": str(error)})
            except KeyError:
                self.reply(404, {"error": "unknown run"})

        def do_GET(self):
            try:
                self.local_request()
                if self.path == "/":
                    blob = Path(__file__).with_name("index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(blob)))
                    self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'")
                    self.end_headers()
                    self.wfile.write(blob)
                    return
                if self.path == "/api/catalog":
                    self.reply(200, {"profile": PROFILE, "datasets": {k: {"label": v.get("label", k), "segmentation_version": v["segmentation_version"]} for k, v in manager.catalog["datasets"].items()}, "models": {k: {"label": v.get("label", k)} for k, v in manager.catalog["models"].items()}, "algorithms": DEFAULTS, "cohorts": manager.catalog.get("cohorts", {}), "sweep_limits": {"cells": 100, "seeds": 16, "trials": 10000}})
                    return
                if self.path == "/api/runs":
                    self.reply(200, manager.list())
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) < 3 or parts[:2] != ["api", "runs"]:
                    self.reply(404, {"error": "not found"})
                    return
                ident = parts[2]
                record = manager.get(ident)
                if len(parts) == 3:
                    self.reply(200, record)
                    return
                directory = manager.root / ident
                if len(parts) == 4 and parts[3] in {"provenance", "log"}:
                    path = directory / ("provenance.json" if parts[3] == "provenance" else "worker.log")
                    self.reply(200, read_json(path) if parts[3] == "provenance" else {"log": path.read_text(errors="replace")[-100000:]})
                    return
                if record["state"] != "succeeded":
                    self.reply(409, {"error": "results are available only after successful completion"})
                    return
                if len(parts) == 4 and parts[3] == "results":
                    self.reply(200, read_json(directory / "results.json"))
                    return
                if len(parts) == 5 and parts[3] == "artifacts" and re.fullmatch(r"[A-Za-z0-9_-]+\.(json|png|npz)", parts[4]):
                    allowed = read_json(directory / "results.json")["artifacts"]
                    if parts[4] not in allowed:
                        self.reply(404, {"error": "unknown artifact"})
                        return
                    path = directory / parts[4]
                    blob = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", {".json": "application/json", ".png": "image/png", ".npz": "application/octet-stream"}[path.suffix])
                    self.send_header("Content-Length", str(len(blob)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(blob)
                    return
                self.reply(404, {"error": "not found"})
            except ValueError as error:
                self.reply(400, {"error": str(error)})
            except (KeyError, FileNotFoundError):
                self.reply(404, {"error": "not found"})

    return ThreadingHTTPServer((host, port), Handler)
