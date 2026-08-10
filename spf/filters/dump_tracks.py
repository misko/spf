"""Persist per-timestep tracks for a small set of configurations.

The sweep stores scalars only -- one MSE and one calibration block per run. That
is the right thing for 26,112 runs, but it means every question that needs the
actual track has to re-run the sweep to ask it: reliability curves, *where in
time* a filter failed, whether two filters fail on the same frames, or any metric
invented later.

This writes the tracks for a named handful of configurations instead. One
``.npz`` per (configuration, dataset, seed), holding ``theta``, ``sigma`` and
``gt`` -- all three in the SAME angular frame, named in the file.

Deliberately NOT built into ``run_filters_on_data.py``. Which configurations are
worth keeping is decided *by* a sweep, not during one, and threading a
``store_tracks`` flag would mean eight near-identical edits to wrappers shared
with CI and three notebooks. ``plot_filter_run.run_filter`` already returns
exactly this tuple for all seven families and is already test-covered, so this
is a thin driver over it.

Size: ~1,800 runs x ~1,500 timesteps x 3 float32 arrays is roughly 33 MB.

Usage::

    python spf/filters/dump_tracks.py \\
        --datasets $(cat stage3_rover_all_n48.txt) \\
        --configs best_configs.json \\
        --precompute-cache <cache> --empirical-pkl-fn <table>.pkl \\
        --checkpoint-fn <ckpt>/best.pth --inference-cache <cache> \\
        --seeds 0 1 2 3 4 --output-dir /mnt/qnap01/.../tracks
"""

import argparse
import json
import logging
import os

import numpy as np

from spf.evaluation import calibration, metrics
from spf.filters.plot_filter_run import open_dataset, run_filter
from spf.filters.plot_trajectory_comparison import TYPE_TO_FILTER

# EKFs are deterministic; a seed axis would write five identical files.
DETERMINISTIC = ("EKF",)


def track_filename(output_dir, ds_name, _type, frame, seed):
    return os.path.join(
        output_dir, f"{ds_name}__{_type}__{frame}__seed{seed}.npz"
    )


def dump_one(ds, ds_name, key, spec, seed, args):
    """Run one configuration on one dataset and write its track. Returns a row."""
    _type, frame = key.split("|")
    filter_name = TYPE_TO_FILTER.get(_type)
    if filter_name is None:
        raise ValueError(f"unknown filter type {_type!r}")

    params = dict(spec["params"])
    params.pop("segmentation_version", None)
    params["seed"] = seed

    theta, sigma, gt, extras = run_filter(
        ds, filter_name, params,
        checkpoint_fn=args.checkpoint_fn,
        inference_cache=args.inference_cache,
    )
    # run_filter names the frame it actually produced; trust that over the key,
    # which is only a label from the sweep report.
    frame = extras["frame"]
    out_fn = track_filename(args.output_dir, ds_name, _type, frame, seed)

    tmp = f"{out_fn}.{os.getpid()}.tmp"
    np.savez_compressed(
        tmp,
        theta=np.asarray(theta, dtype=np.float32),
        sigma=np.asarray(sigma, dtype=np.float32),
        gt=np.asarray(gt, dtype=np.float32),
        frame=frame,
        type=_type,
        seed=seed,
        rx_idx=extras["rx_idx"],
        ds_fn=ds_name,
    )
    os.replace(tmp, out_fn)

    row = metrics.summarize(theta, gt)
    row.update({
        "ds_fn": ds_name, "type": _type, "frame": frame, "seed": seed,
        "calib_std_z": calibration.calibration_ratio(theta, gt, sigma),
        "calib_cov1": calibration.coverage(theta, gt, sigma, ks=(1,))[0]["measured"],
        "skill_vs_random": metrics.skill_vs_random(row["mse"]),
        "track_fn": os.path.basename(out_fn),
    })
    return row


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--configs", required=True,
                   help='JSON of {"<TYPE>|<frame>": {"params": {...}}}')
    p.add_argument("--precompute-cache", required=True)
    p.add_argument("--empirical-pkl-fn", required=True)
    p.add_argument("--segmentation-version", type=float, default=3.7)
    p.add_argument("--checkpoint-fn", default=None)
    p.add_argument("--inference-cache", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip a (config, dataset, seed) whose npz already exists")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    args = get_parser().parse_args()
    with open(args.configs) as f:
        configs = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    rows, failures = [], []
    for ds_fn in args.datasets:
        ds_name = os.path.basename(str(ds_fn)).replace(".zarr", "")
        ds = None
        for key, spec in sorted(configs.items()):
            _type = key.split("|")[0]
            seeds = [0] if _type.startswith(DETERMINISTIC) else args.seeds
            for seed in seeds:
                if args.resume:
                    frame = key.split("|")[1]
                    if os.path.exists(
                        track_filename(args.output_dir, ds_name, _type, frame, seed)
                    ):
                        continue
                if ds is None:  # only pay the open cost if there is work to do
                    ds = open_dataset(
                        ds_fn, args.precompute_cache, args.empirical_pkl_fn,
                        args.segmentation_version,
                    )
                try:
                    rows.append(dump_one(ds, ds_name, key, spec, seed, args))
                    logging.info(f"{ds_name} {key} seed={seed}: ok")
                except Exception as e:
                    failures.append((ds_name, key, seed, f"{type(e).__name__}: {e}"))
                    logging.error(f"{ds_name} {key} seed={seed}: {e}")

    index_fn = os.path.join(args.output_dir, "index.json")
    with open(index_fn, "w") as f:
        json.dump({"rows": rows, "failures": failures}, f, indent=2, sort_keys=True)

    logging.info(f"wrote {len(rows)} tracks and {index_fn}")
    if failures:
        # Loudly, and non-zero: a bulk tool that half-works and exits 0 is how
        # the inference cache silently built 31 of 48 stores.
        logging.error(f"{len(failures)} of {len(rows) + len(failures)} FAILED")
        for f_ in failures[:20]:
            logging.error(f"  {f_}")
        raise SystemExit(1)
