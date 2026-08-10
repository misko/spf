"""Every filter's bearing track over time, on one axis, with its confidence band.

``plot_filter_run.py`` shows one filter in depth. This answers the other
question: on this capture, which approach actually tracks the emitter, and does
its stated confidence mean anything?

Per angular frame (never across -- see ``spf.evaluation.frames``):

* ground truth as a black line
* each approach's estimate as a coloured line
* its +-1 sigma as a matching translucent fill
* a metrics table naming every approach, with the **random-prediction floor**

The floor matters. An MSE of 2.6 rad^2 sounds like a number until you know that
guessing uniformly at random scores pi^2/3 = 3.29. Two references are drawn:

``uniform random``
    predicting a uniform angle every step. E[e^2] = pi^2/3 rad^2, RMSE 103.9 deg.
    Zero information.
``best constant``
    the single fixed bearing minimising MSE on this capture -- the circular mean
    of ground truth. A tracker must beat this or it has learned nothing about
    *time*, only about where the emitter tends to be.

Usage::

    python spf/filters/plot_trajectory_comparison.py \\
        --dataset <prefix>.zarr \\
        --precompute-cache <cache> --empirical-pkl-fn <table>.pkl \\
        --checkpoint-fn <ckpt>/best.pth --inference-cache <cache> \\
        --configs best_configs.json \\
        --output-dir <report>/figures
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from spf.evaluation import calibration, metrics  # noqa: E402
from spf.filters.plot_filter_run import open_dataset, run_filter  # noqa: E402

UNIFORM_RANDOM_MSE = metrics.UNIFORM_RANDOM_MSE

# Map a stage-2 result "type" onto the plotter's filter name.
TYPE_TO_FILTER = {
    "EKF_single_theta_single_radio": "ekf_single",
    "EKF_single_theta_dual_radio": "ekf_dual",
    "PF_single_theta_single_radio": "pf_single",
    "PF_single_theta_dual_radio": "pf_dual",
    "PF_single_theta_single_radio_NN": "pf_single_nn",
    "PF_single_theta_dual_radio_NN": "pf_dual_nn",
}


def score(theta, sigma, gt):
    out = metrics.summarize(theta, gt)
    cov = calibration.coverage(theta, gt, sigma)
    out["cov1"] = cov[0]["measured"]
    out["cov2"] = cov[1]["measured"]
    out["calib_ratio"] = calibration.calibration_ratio(theta, gt, sigma)
    out["skill_vs_random"] = metrics.skill_vs_random(out["mse"])
    return out


def plot_comparison(ds, runs, title, max_steps=None):
    """``runs``: list of (label, theta, sigma, gt, frame). One panel per frame."""
    by_frame = {}
    for label, theta, sigma, gt, frame in runs:
        by_frame.setdefault(frame, []).append((label, theta, sigma, gt))

    frames = sorted(by_frame)
    # one track panel + one metrics panel per frame
    fig, axes = plt.subplots(
        len(frames) * 2, 1, figsize=(15, 5.0 * len(frames)),
        gridspec_kw={"height_ratios": [3, 1.5] * len(frames)},
    )
    axes = np.atleast_1d(axes)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange"]

    for fi, frame in enumerate(frames):
        ax, tax = axes[2 * fi], axes[2 * fi + 1]
        entries = by_frame[frame]
        gt = entries[0][3]
        n = len(gt) if max_steps is None else min(max_steps, len(gt))
        t = np.arange(n)

        ax.plot(t, gt[:n], color="black", lw=1.6, label="ground truth", zorder=5)
        rows = []
        for i, (label, theta, sigma, g) in enumerate(entries):
            c = colors[i % len(colors)]
            ax.fill_between(
                t, (theta - sigma)[:n], (theta + sigma)[:n],
                color=c, alpha=0.18, linewidth=0,
            )
            ax.plot(t, theta[:n], color=c, lw=1.0, alpha=0.9, label=label)
            s = score(theta, sigma, g)
            rows.append((label, s))

        ax.axhline(np.pi / 2, ls=":", c=(0.75, 0.75, 0.75), lw=0.8)
        ax.axhline(-np.pi / 2, ls=":", c=(0.75, 0.75, 0.75), lw=0.8)
        ax.set_ylim(-3.6, 4.6)  # headroom so the legend clears the data
        ax.set_ylabel("bearing theta (rad)")
        ax.set_xlabel("sample index (time)")
        ax.set_title(f"frame: {frame}   —   shaded band = ±1σ reported by that filter")
        ax.legend(fontsize=8, ncol=len(entries) + 1, loc="upper center", framealpha=0.9)

        # ---- metrics table, with the random floor
        base = metrics.baselines(gt)
        tax.axis("off")
        header = ["approach", "MSE rad²", "RMSE°", "median|e|°", "skill vs rand",
                  "±1σ cov", "std(z)"]
        body = []
        for label, s in sorted(rows, key=lambda r: r[1]["mse"]):
            body.append([
                label,
                f"{s['mse']:.3f}",
                f"{s['rmse_deg']:.1f}",
                f"{np.degrees(s['median_abs_err_rad']):.1f}",
                f"{s['skill_vs_random']:+.1%}",
                f"{s['cov1']:.2f}",
                f"{s['calib_ratio']:.2f}",
            ])
        body.append([
            "— best constant —", f"{base['best_constant']['mse']:.3f}",
            f"{base['best_constant']['rmse_deg']:.1f}", "—",
            f"{1 - base['best_constant']['mse'] / UNIFORM_RANDOM_MSE:+.1%}", "—", "—",
        ])
        body.append([
            "— uniform random —", f"{base['uniform_random']['mse']:.3f}",
            f"{base['uniform_random']['rmse_deg']:.1f}", "—", "0.0%", "—", "—",
        ])
        # Explicit bbox, not loc="center": a centred table grows downward with the
        # row count and swallows the note below it (5 rows did exactly that).
        tbl = tax.table(
            cellText=body, colLabels=header, cellLoc="center",
            bbox=[0, 0.16, 1, 0.84],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#dddddd")
        for i in range(len(body) - 2, len(body)):
            for j in range(len(header)):
                tbl[i + 1, j].set_facecolor("#f4f4f4")
        tax.text(
            0.5, 0.10,
            "nominal ±1σ coverage is 0.68 · std(z) 1.0 = calibrated, >1 overconfident · "
            "an approach that cannot beat 'best constant' has learned nothing about time",
            transform=tax.transAxes, ha="center", va="top",
            fontsize=8, color="#555555",
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--precompute-cache", required=True)
    p.add_argument("--empirical-pkl-fn", required=True)
    p.add_argument("--segmentation-version", type=float, default=3.7)
    p.add_argument("--checkpoint-fn", default=None)
    p.add_argument("--inference-cache", default=None)
    p.add_argument(
        "--configs",
        required=True,
        help='JSON of {"<TYPE>|<frame>": {"params": {...}}} as written by the '
        "stage-2 analysis -- the winning configuration per family",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    args = get_parser().parse_args()
    with open(args.configs) as f:
        configs = json.load(f)

    ds = open_dataset(
        args.dataset, args.precompute_cache, args.empirical_pkl_fn,
        args.segmentation_version,
    )

    runs = []
    for key, spec in sorted(configs.items()):
        _type, frame = key.split("|")
        name = TYPE_TO_FILTER.get(_type)
        if name is None:
            logging.warning(f"skipping unknown type {_type}")
            continue
        params = dict(spec["params"])
        params.pop("segmentation_version", None)
        try:
            theta, sigma, gt, extras = run_filter(
                ds, name, params,
                checkpoint_fn=args.checkpoint_fn,
                inference_cache=args.inference_cache,
            )
        except Exception as e:
            logging.error(f"{key}: {type(e).__name__}: {e}")
            continue
        label = _type.replace("_single_theta", "").replace("_", " ")
        if params.get("absolute"):
            label += " [abs]"
        runs.append((label, theta, sigma, gt, extras["frame"]))
        logging.info(f"ran {key}: {len(theta)} steps, frame {extras['frame']}")

    if not runs:
        raise SystemExit("no filters ran")

    basename = os.path.basename(str(args.dataset)).replace(".zarr", "")
    fig = plot_comparison(
        ds, runs,
        f"Filter comparison — {basename}\n"
        f"d/λ = {ds.rx_wavelength_spacing:.5f}   {len(ds)} timesteps",
        max_steps=args.max_steps,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{basename}__comparison.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
