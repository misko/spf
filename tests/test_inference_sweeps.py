"""Cohort selection, balanced sweep scoring, and reusable heatmap integration."""
import copy
import itertools
import json
import math
from pathlib import Path

import pytest

from spf.inference_service import service, sweeps


def catalog():
    return {"datasets": {"a": {"path": "/data/a.zarr"}, "b": {"path": "/data/b.zarr"}}, "models": {"paired": {"checkpoint": "/models/best.pth"}}, "cohorts": {"dev": {"dataset_ids": ["a", "b"], "role": "development_tuning", "description": "Development only", "overlap_notes": ["Used for calibration"]}}}


def body():
    return {"kind": "sweep", "cohort_id": "dev", "algorithm": "pf_dual", "fixed_params": {"theta_err": .1}, "grid": {"N": [128, 512], "theta_dot_err": [.001, .01]}, "seeds": [0, 1]}


def spec():
    return service.validate_request(body(), catalog())


def trials(request):
    rows = []
    for cell in sweeps.expand_grid(request):
        for seed in request["seeds"]:
            for ident, mse, frames in [("a", 1., 10), ("b", 3., 1000)]:
                rows.append({"cell_index": cell["cell_index"], "seed": seed, "dataset_id": ident, "frame": "craft_relative", "frames": frames, "metrics": {"mse": mse + seed}})
    return rows


def test_cohort_resolution_is_an_immutable_request_snapshot():
    c = catalog()
    request = service.validate_request(body(), c)
    assert request["dataset_ids"] == ["a", "b"]
    c["cohorts"]["dev"]["dataset_ids"].clear()
    assert request["dataset_ids"] == ["a", "b"]
    assert request["cohort"]["overlap_notes"] == ["Used for calibration"]
    assert request["n_cells"] == 4 and request["n_trials"] == 16


def test_cohort_can_drive_existing_evaluation_mode():
    request = service.validate_request({"cohort_id": "dev", "algorithm": "ekf_dual"}, catalog())
    assert request["kind"] == "evaluation"
    assert request["dataset_ids"] == ["a", "b"]
    assert request["seed"] == 0


@pytest.mark.parametrize("change", [
    {"dataset_ids": ["a"]}, {"cohort_id": "missing"},
    {"grid": {"N": [128, 512]}},
    {"grid": {"N": [128, 128], "theta_dot_err": [.001, .01]}},
    {"grid": {"N": [True, 512], "theta_dot_err": [.001, .01]}},
    {"grid": {"N": [128, 512], "absolute": [False, True]}},
    {"grid": {"N": [128, 512], "theta_dot_err": [float("inf"), .01]}},
    {"fixed_params": {"N": 128}}, {"fixed_params": {"typo": 1}},
    {"seeds": [0, 0]}, {"seeds": [True]}, {"seeds": []},
    {"seed": 0}, {"grid": {"N": list(range(16, 117)), "theta_dot_err": [.001, .01]}},
])
def test_invalid_sweep_is_rejected_before_execution(change):
    with pytest.raises(ValueError):
        service.validate_request({**body(), **change}, catalog())


def test_neural_sweep_defaults_to_craft_frame_and_ekf_does_not_repeat_seeds():
    request = service.validate_request({**body(), "algorithm": "pf_dual_nn", "model_id": "paired"}, catalog())
    assert request["fixed_params"]["absolute"] is False
    with pytest.raises(ValueError, match="deterministic"):
        service.validate_request({**body(), "algorithm": "ekf_dual", "fixed_params": {}, "grid": {"phi_std": [5, 12], "noise_std": [.001, .01]}}, catalog())


def test_aliases_cannot_count_the_same_capture_twice():
    c = catalog()
    c["datasets"]["b"]["path"] = c["datasets"]["a"]["path"]
    with pytest.raises(ValueError, match="aliases"):
        service.validate_request(body(), c)


def test_balanced_macro_score_then_seed_mean_not_frame_pooling():
    request = spec()
    summary = sweeps.summarize_trials(request, trials(request))
    assert len(summary["rows"]) == 8
    for cell in summary["cells"]:
        assert cell["mean_dataset_mse"] == 2.5
        assert cell["seed_std_mse"] == .5
        assert cell["capture_std_mse"] == 1.
        assert cell["rmse_rad"] == math.sqrt(2.5)
        assert cell["n_datasets"] == 2 and cell["n_seeds"] == 2
        assert [x["mean_dataset_mse"] for x in cell["per_seed"]] == [2., 3.]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nonfinite", "negative", "frame", "empty"])
def test_incomplete_or_invalid_cells_cannot_be_published(failure):
    request = spec()
    rows = trials(request)
    if failure == "missing": rows.pop()
    elif failure == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif failure == "nonfinite": rows[0]["metrics"]["mse"] = float("nan")
    elif failure == "negative": rows[0]["metrics"]["mse"] = -1
    elif failure == "frame": rows[0]["frame"] = "absolute_north"
    elif failure == "empty": rows[0]["frames"] = 0
    with pytest.raises(ValueError):
        sweeps.summarize_trials(request, rows)


def test_report_renders_through_existing_heatmap_code(tmp_path):
    summary = sweeps.summarize_trials(spec(), trials(spec()))
    heatmap = sweeps.write_heatmap(summary, tmp_path)
    assert (tmp_path / heatmap["image"]).is_file()
    assert heatmap["x"]["name"] == "N"
    assert heatmap["values"] == [[2.5, 2.5], [2.5, 2.5]]


def test_sweep_opens_each_capture_once_and_reuses_dataset_cache(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from spf.filters import plot_filter_run
    request = spec()
    request.update(datasets={key: {"path": key, "precompute_cache": "cache", "empirical": "table", "segmentation_version": 3.7} for key in request["dataset_ids"]}, model=None)
    opened, calls = [], []
    class Dataset:
        mean_phase = {"r0": torch.zeros(2), "r1": torch.zeros(2)}
        def __init__(self, ident): self.ident = ident
        def __len__(self): return 2
        def close(self): pass
    def open_dataset(path, *args):
        opened.append(path)
        return Dataset(path)
    def run_filter(ds, name, params, **kw):
        calls.append((ds.ident, kw["inference_cache"], params["seed"]))
        return np.zeros(2), np.ones(2), np.zeros(2), {"frame": "craft_relative"}
    monkeypatch.setattr(plot_filter_run, "open_dataset", open_dataset)
    monkeypatch.setattr(plot_filter_run, "run_filter", run_filter)
    monkeypatch.setattr(plot_filter_run, "summarize_run", lambda *args: {"mse": 1.})
    result = sweeps.run_sweep(request, tmp_path, lambda **kwargs: None)
    assert opened == ["a", "b"]
    assert len(calls) == 16
    assert len({cache for ident, cache, seed in calls if ident == "a"}) == 1
    assert len({cache for ident, cache, seed in calls}) == 2
    assert result["n_trials"] == 16
    assert all(row["mean_dataset_mse"] == 1. for row in result["cells"])
    assert len(json.loads((tmp_path / "trials.json").read_text())) == 16


def test_dynamic_measurement_noise_validates_complete_grid_configurations():
    request = service.validate_request({"kind": "sweep", "cohort_id": "dev", "algorithm": "ekf_dual", "fixed_params": {"dynamic_R": 1}, "grid": {"phi_std": [0, 5], "noise_std": [.001, .01]}, "seeds": [0]}, catalog())
    assert len(sweeps.expand_grid(request)) == 4
    assert request["fixed_params"]["dynamic_R"] == 1


@pytest.mark.parametrize("kind", [[], {}, True, None])
def test_malformed_kind_returns_validation_error(kind):
    with pytest.raises(ValueError, match="kind"):
        service.validate_request({**body(), "kind": kind}, catalog())
