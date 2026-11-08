"""Balanced cohort sweeps over existing filter execution and reporting code."""
import itertools
import math

MAX_CELLS = 100
MAX_SEEDS = 16
MAX_TRIALS = 10000
FAMILIES = {
    "ekf_dual": "EKF_single_theta_dual_radio",
    "pf_dual": "PF_single_theta_dual_radio",
    "pf_dual_nn": "PF_single_theta_dual_radio_NN",
}


def validate_sweep(body, defaults, validate_params, n_datasets):
    grid = body.get("grid")
    fixed = body.get("fixed_params", {})
    if not isinstance(grid, dict) or len(grid) != 2:
        raise ValueError("grid requires exactly two varying numeric axes")
    if not isinstance(fixed, dict) or set(fixed) & set(grid):
        raise ValueError("fixed_params must be an object and cannot overlap grid axes")
    if set(grid) - defaults.keys() or "absolute" in grid:
        raise ValueError("unknown or nonnumeric grid axis")
    normalized = {}
    for axis, values in sorted(grid.items()):
        if not isinstance(values, list) or not 2 <= len(values) <= MAX_CELLS:
            raise ValueError(f"{axis}: provide at least two distinct values")
        for value in values:
            try:
                finite = not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{axis}: grid values must be finite numbers")
        if len(set(values)) != len(values):
            raise ValueError(f"{axis}: duplicate grid values")
        normalized[axis] = sorted(values)
    base = {**defaults, **fixed}
    seeds = body.get("seeds", [0] if body["algorithm"] == "ekf_dual" else [0, 1, 2])
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_SEEDS:
        raise ValueError(f"seeds requires 1 to {MAX_SEEDS} distinct integers")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if body["algorithm"] == "ekf_dual" and seeds != [0]:
        raise ValueError("EKF is deterministic; use seeds: [0]")
    n_cells = math.prod(len(v) for v in normalized.values())
    n_trials = n_cells * len(seeds) * n_datasets
    if n_cells > MAX_CELLS or n_trials > MAX_TRIALS:
        raise ValueError(f"sweep exceeds {MAX_CELLS} cells or {MAX_TRIALS} capture trials")
    spec = {"fixed_params": {k: v for k, v in base.items() if k not in normalized}, "grid": normalized, "seeds": sorted(seeds), "n_cells": n_cells, "n_trials": n_trials}
    for cell in expand_grid(spec):
        validate_params(cell["params"])
    return spec


def expand_grid(spec):
    axes = list(spec["grid"])
    return [
        {"cell_index": index, "params": {**spec["fixed_params"], **dict(zip(axes, values))}}
        for index, values in enumerate(itertools.product(*(spec["grid"][axis] for axis in axes)))
    ]


def summarize_trials(spec, trials):
    import numpy as np
    from spf.evaluation.aggregate import aggregate

    cells = expand_grid(spec)
    expected = {(c["cell_index"], seed, ident) for c in cells for seed in spec["seeds"] for ident in spec["dataset_ids"]}
    observed = set()
    frames = set()
    rows = []
    for trial in trials:
        key = (trial["cell_index"], trial["seed"], trial["dataset_id"])
        if key not in expected or key in observed:
            raise ValueError("duplicate or unexpected sweep trial")
        observed.add(key)
        mse = trial["metrics"]["mse"]
        if isinstance(mse, bool) or not isinstance(mse, (int, float)) or not math.isfinite(mse) or mse < 0:
            raise ValueError("nonfinite or invalid trial MSE; refusing incomplete cell")
        if type(trial["frames"]) is not int or trial["frames"] <= 0:
            raise ValueError("trial requires a positive frame count")
        frame = trial["frame"]
        frames.add(frame)
        params = cells[trial["cell_index"]]["params"]
        rows.append({"type": FAMILIES[spec["algorithm"]], "frame": frame, "cell_index": trial["cell_index"], "seed": trial["seed"], **params, "metrics": {"mse_craft_theta": float(mse)}})
    if observed != expected:
        raise ValueError(f"incomplete sweep: {len(observed)}/{len(expected)} capture trials")
    if len(frames) != 1:
        raise ValueError("a sweep must have exactly one angular frame")
    frame = next(iter(frames))
    wanted_frame = "absolute_north" if spec["algorithm"] == "pf_dual_nn" and spec["fixed_params"].get("absolute") else "craft_relative"
    if frame != wanted_frame:
        raise ValueError(f"unexpected angular frame {frame}; expected {wanted_frame}")
    keys = ["type", "frame", "cell_index", "seed", *cells[0]["params"]]
    report_rows = aggregate(rows, keys)
    summaries = []
    for cell in cells:
        index = cell["cell_index"]
        selected = [t for t in trials if t["cell_index"] == index]
        seed_rows = sorted([r for r in report_rows if r["cell_index"] == index], key=lambda r: r["seed"])
        seed_means = [r["mse_craft_theta_mean"] for r in seed_rows]
        capture_means = [float(np.mean([t["metrics"]["mse"] for t in selected if t["dataset_id"] == ident])) for ident in spec["dataset_ids"]]
        mean = float(np.mean(seed_means))
        summaries.append({**cell, "frame": frame, "mean_dataset_mse": mean, "rmse_rad": math.sqrt(mean), "rmse_deg": math.degrees(math.sqrt(mean)), "seed_std_mse": float(np.std(seed_means)), "capture_std_mse": float(np.std(capture_means)), "n_datasets": len(spec["dataset_ids"]), "n_seeds": len(spec["seeds"]), "per_seed": [{"seed": r["seed"], "mean_dataset_mse": r["mse_craft_theta_mean"]} for r in seed_rows], "per_dataset": [{"dataset_id": ident, "mean_seed_mse": mean} for ident, mean in zip(spec["dataset_ids"], capture_means)]})
    return {"frame": frame, "cells": summaries, "rows": report_rows, "best_cell": min(summaries, key=lambda c: c["mean_dataset_mse"]), "weighting": "Within each seed: equal weight per dataset. Cell: equal weight per seed. No frame-count pooling.", "spread_semantics": "Population SD of per-seed cohort means; capture SD is over per-capture seed means. These are not confidence intervals."}


def write_heatmap(summary, directory):
    from pathlib import Path
    from spf.filters.plot_hyperparam_heatmap import family_configs, fig_family, choose_axes, grid_for

    grouped = family_configs(summary["rows"])
    if len(grouped) != 1:
        raise ValueError("expected exactly one algorithm and angular frame")
    (family, frame), configs = next(iter(grouped.items()))
    plotted = fig_family(family, frame, configs, str(directory))
    if plotted is None:
        raise ValueError("existing heatmap renderer could not plot this grid")
    xname, yname = choose_axes(configs)
    xs, ys, values = grid_for(configs, xname, yname)
    if not __import__("numpy").isfinite(values).all():
        raise ValueError("heatmap has a missing or nonfinite cell")
    return {"image": Path(plotted[0]).name, "x": {"name": xname, "values": xs}, "y": {"name": yname, "values": ys}, "values": values.tolist(), "notes": plotted[1], "interpretation": "Only the displayed two axes vary; every other parameter is fixed. Cells average seeds after equal-capture aggregation."}


def run_sweep(request, directory, progress):
    import random
    import numpy as np
    import torch
    from spf.filters.plot_filter_run import open_dataset, run_filter, summarize_run
    from spf.inference_service.service import atomic_json

    cells = expand_grid(request)
    trials = []
    artifacts = ["provenance.json", "trials.json", "sweep_report.json", "heatmap_values.json"]
    total = len(cells) * len(request["seeds"]) * len(request["dataset_ids"])
    try:
        for ident in request["dataset_ids"]:
            ds_spec = request["datasets"][ident]
            ds = open_dataset(ds_spec["path"], ds_spec["precompute_cache"], ds_spec["empirical"], ds_spec["segmentation_version"])
            try:
                finite = torch.isfinite(ds.mean_phase["r0"]) & torch.isfinite(ds.mean_phase["r1"])
                valid_fraction = float(finite.float().mean())
                for cell in cells:
                    for seed in request["seeds"]:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        progress(stage="sweep", dataset_id=ident, completed=len(trials), total=total, unit="capture trials", cell_index=cell["cell_index"], seed=seed)
                        theta, sigma, gt, extras = run_filter(ds, request["algorithm"], {**cell["params"], "seed": seed}, checkpoint_fn=request["model"]["checkpoint"] if request["model"] else None, inference_cache=str(directory / "prediction_cache" / ident))
                        if len(theta) != len(ds) or not len(theta) == len(sigma) == len(gt):
                            raise ValueError("Frame count mismatch; refusing a truncated sweep trial")
                        metrics = summarize_run(theta, sigma, gt)
                        mse = metrics["mse"]
                        if not math.isfinite(mse) or mse < 0:
                            raise ValueError("Nonfinite sweep trial MSE; refusing biased aggregate")
                        trials.append({"cell_index": cell["cell_index"], "params": cell["params"], "seed": seed, "dataset_id": ident, "frames": len(theta), "frame": extras["frame"], "valid_paired_phase_fraction": valid_fraction, "metrics": {"mse": float(mse)}})
                        if len(trials) % 10 == 0:
                            atomic_json(directory / "trials.json", trials)
            finally:
                ds.close()
    finally:
        atomic_json(directory / "trials.json", trials)
    summary = summarize_trials(request, trials)
    heatmap = write_heatmap(summary, directory)
    artifacts.append(heatmap["image"])
    atomic_json(directory / "sweep_report.json", summary)
    atomic_json(directory / "heatmap_values.json", heatmap)
    progress(stage="complete", completed=total, total=total, unit="capture trials")
    return {"kind": "sweep", "profile": request["profile"], "algorithm": request["algorithm"], "cohort_id": request.get("cohort_id"), "cohort": request.get("cohort"), "dataset_ids": request["dataset_ids"], "seeds": request["seeds"], "grid": request["grid"], "fixed_params": request["fixed_params"], "cells": summary["cells"], "best_cell": summary["best_cell"], "frame": summary["frame"], "weighting": summary["weighting"], "spread_semantics": summary["spread_semantics"], "n_trials": len(trials), "heatmap": heatmap, "artifacts": artifacts}
