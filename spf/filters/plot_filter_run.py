"""Visualise one filter run: observations, prediction, confidence, calibration.

The per-filter ``plot_*`` helpers scattered through ``spf/filters/*.py`` each
build their own filter with hard-coded hyperparameters and each draw a different
set of panels, so two filters cannot be put side by side. This runs a filter
under the hyperparameters you actually swept and draws the same panels for every
family, scoring through ``spf.evaluation`` so the numbers match the report.

Panels:

1. measured phi vs ground-truth phi per radio, with dropped (NaN) frames marked
2. the theta track with a +-1 sigma band, against ground truth
3. absolute error over time and the running RMSE
4. calibration -- z-score histogram against a unit normal, plus measured
   +-1/2/3 sigma coverage against nominal
5. for NN filters, the model posterior over theta as a heat map, with ground
   truth and the filter track drawn on top

Panel 4 is the point. A filter reporting a variance nobody has checked is
reporting decoration: pointed at the NN dual-radio PF on a rover capture, the
+-1 sigma band covered 25.6% of errors where 68.3% is nominal.

Example::

    python spf/filters/plot_filter_run.py \\
        --dataset /mnt/qnap01/mouse9911/rovers_2026/merged/<prefix>.zarr \\
        --precompute-cache /mnt/qnap01/mouse9911/rovers_2026/precompute \\
        --empirical-pkl-fn empirical_dists/full.pkl \\
        --segmentation-version 3.7 \\
        --filter pf_dual_nn --params '{"N": 4096, "theta_err": 0.01}' \\
        --checkpoint-fn <ckpt>/best.pth --inference-cache <cache> \\
        --output-dir /tmp/filter_plots
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from spf.dataset.spf_dataset import v5spfdataset
from spf.dataset.spf_nn_dataset_wrapper import v5spfdataset_nn_wrapper
from spf.evaluation import calibration, metrics
from spf.evaluation.frames import ABSOLUTE_NORTH, CRAFT_RELATIVE, RADIO_FOLDED
from spf.filters.ekf_dualradio_filter import SPFPairedKalmanFilter
from spf.filters.ekf_single_radio_filter import SPFKalmanFilter
from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio
from spf.filters.particle_single_radio_nn_filter import PFSingleThetaSingleRadioNN
from spf.rf import reduce_theta_to_positive_y

FILTERS = [
    "ekf_single",
    "ekf_dual",
    "pf_single",
    "pf_dual",
    "pf_single_nn",
    "pf_dual_nn",
]

# The filters never read these; loading them costs minutes of IO per dataset.
FILTER_SKIP_FIELDS = set(
    [
        "windowed_beamformer",
        "weighted_beamformer",
        "all_windows_stats",
        "downsampled_segmentation_mask",
        "signal_matrix",
        "simple_segmentations",
    ]
)


def open_dataset(
    prefix, precompute_cache, empirical_pkl_fn, segmentation_version, nthetas=65
):
    return v5spfdataset(
        prefix=str(prefix).replace(".zarr", ""),
        nthetas=nthetas,
        ignore_qc=True,
        precompute_cache=precompute_cache,
        paired=True,
        snapshots_per_session=1,
        readahead=False,
        skip_fields=FILTER_SKIP_FIELDS,
        empirical_data_fn=empirical_pkl_fn,
        segmentation_version=segmentation_version,
        segment_if_not_exist=False,
        gpu=False,
    )


def _scalars(values):
    return np.asarray([float(np.asarray(v).reshape(-1)[0]) for v in values])


def _nn_posterior(nn_ds, key, rx_idx):
    """(timesteps, ntheta) model posterior, or None when running realtime."""
    cached = getattr(nn_ds, "cached_model_inference", None)
    if cached is None or key not in cached:
        return None
    return np.asarray(cached[key])[:, rx_idx, 0, :]


def run_filter(ds, filter_name, params, checkpoint_fn=None, inference_cache=None):
    """Run one filter; return (theta, sigma, ground_truth, extras).

    All three arrays are in the SAME angular frame, named in ``extras["frame"]``.
    """
    params = dict(params)
    rx_idx = int(params.pop("rx_idx", 0))
    seed = int(params.pop("seed", 0))
    extras = {"rx_idx": rx_idx, "seed": seed}

    pf_kwargs = None
    if filter_name.startswith("pf_"):
        pf_kwargs = dict(
            mean=torch.tensor([[0.0, 0.0]]),
            std=torch.tensor([[20.0, 0.1]]),
            noise_std=torch.tensor(
                [[params.pop("theta_err", 0.01), params.pop("theta_dot_err", 0.001)]]
            ),
            N=int(params.pop("N", 128 * 32)),
            return_particles=False,
            seed=seed,
        )

    if filter_name == "pf_single":
        pf = PFSingleThetaSingleRadio(ds=ds, rx_idx=rx_idx)
        traj = pf.trajectory(**pf_kwargs)
        theta, sigma = _scalars([x["theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        gt = reduce_theta_to_positive_y(ds.ground_truth_thetas[rx_idx]).numpy()
        extras["frame"] = RADIO_FOLDED
    elif filter_name == "pf_single_nn":
        pf = PFSingleThetaSingleRadioNN(
            ds,
            rx_idx,
            checkpoint_fn,
            f"{os.path.dirname(checkpoint_fn)}/config.yml",
            inference_cache=inference_cache,
            device="cpu",
        )
        traj = pf.trajectory(**pf_kwargs)
        theta, sigma = _scalars([x["theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        gt = reduce_theta_to_positive_y(ds.ground_truth_thetas[rx_idx]).numpy()
        extras["frame"] = RADIO_FOLDED
        extras["posterior"] = _nn_posterior(pf.ds, "single", rx_idx)
    elif filter_name == "pf_dual":
        pf = PFSingleThetaDualRadio(ds=ds)
        traj = pf.trajectory(**pf_kwargs)
        theta, sigma = _scalars([x["craft_theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        gt = ds.craft_ground_truth_thetas.numpy()
        extras["frame"] = CRAFT_RELATIVE
    elif filter_name == "pf_dual_nn":
        absolute = bool(params.pop("absolute", False))
        nn_ds = v5spfdataset_nn_wrapper(
            ds,
            f"{os.path.dirname(checkpoint_fn)}/config.yml",
            checkpoint_fn,
            inference_cache,
            device="cpu",
            absolute=absolute,
        )
        pf = PFSingleThetaDualRadioNN(nn_ds=nn_ds)
        traj = pf.trajectory(**pf_kwargs)
        theta, sigma = _scalars([x["craft_theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        if absolute:
            # matches PFSingleThetaDualRadioNN.metrics: circular mean over radios
            gt = torch.atan2(
                torch.sin(ds.absolute_thetas).mean(axis=0),
                torch.cos(ds.absolute_thetas).mean(axis=0),
            ).numpy()
            extras["frame"] = ABSOLUTE_NORTH
        else:
            gt = ds.craft_ground_truth_thetas.numpy()
            extras["frame"] = CRAFT_RELATIVE
        extras["posterior"] = _nn_posterior(nn_ds, "paired", 0)
    elif filter_name == "ekf_dual":
        ekf = SPFPairedKalmanFilter(
            ds=ds,
            phi_std=params.pop("phi_std", 10.0),
            p=params.pop("p", 5.0),
            dynamic_R=params.pop("dynamic_R", 0.0),
        )
        # debug=True is what populates P_theta; without it the EKF reports no sigma
        traj = ekf.trajectory(
            dt=1.0, noise_std=params.pop("noise_std", 0.001), debug=True
        )
        theta, sigma = _scalars([x["craft_theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        gt = ds.craft_ground_truth_thetas.numpy()
        extras["frame"] = CRAFT_RELATIVE
    elif filter_name == "ekf_single":
        ekf = SPFKalmanFilter(
            ds=ds,
            rx_idx=rx_idx,
            phi_std=params.pop("phi_std", 10.0),
            p=params.pop("p", 5.0),
            dynamic_R=params.pop("dynamic_R", 0.0),
        )
        traj = ekf.trajectory(
            dt=1.0, noise_std=params.pop("noise_std", 0.001), debug=True
        )
        theta, sigma = _scalars([x["theta"] for x in traj]), np.sqrt(
            _scalars([x["P_theta"] for x in traj])
        )
        gt = reduce_theta_to_positive_y(ds.ground_truth_thetas[rx_idx]).numpy()
        extras["frame"] = RADIO_FOLDED
    else:
        raise ValueError(f"unknown filter {filter_name!r}; expected one of {FILTERS}")

    if params:
        raise ValueError(f"unused params for {filter_name}: {sorted(params)}")

    gt = np.asarray(gt).reshape(-1)[: theta.shape[0]]
    return theta, sigma, gt, extras


def summarize_run(theta, sigma, gt):
    """The numbers printed alongside the figure, from spf.evaluation."""
    out = metrics.summarize(theta, gt)
    out["coverage"] = calibration.coverage(theta, gt, sigma)
    out["calibration_ratio"] = calibration.calibration_ratio(theta, gt, sigma)
    return out


def plot_filter_run(ds, theta, sigma, gt, extras, title):
    from matplotlib import pyplot as plt

    posterior = extras.get("posterior")
    nrows = 5 if posterior is not None else 4
    fig, ax = plt.subplots(nrows, 1, figsize=(13, 3.1 * nrows))
    n = theta.shape[0]
    t = np.arange(n)

    # ---- observations
    colors = ["tab:blue", "tab:orange"]
    for ridx in (0, 1):
        mp = ds.mean_phase[f"r{ridx}"][:n].numpy()
        ax[0].scatter(
            t[: mp.shape[0]],
            mp,
            s=2.0,
            alpha=0.45,
            color=colors[ridx],
            label=f"r{ridx} measured phi",
        )
        ax[0].plot(
            ds.ground_truth_phis[ridx][:n].numpy(),
            color=colors[ridx],
            ls="dashed",
            lw=1.0,
            label=f"r{ridx} ground-truth phi",
        )
        nan_frac = float(np.isnan(mp).mean())
        for x in t[: mp.shape[0]][np.isnan(mp)]:
            ax[0].axvline(x, color=colors[ridx], alpha=0.05, lw=0.5)
        ax[0].plot([], [], " ", label=f"r{ridx} dropped frames: {nan_frac:.1%}")
    ax[0].set_ylabel("phi (rad)")
    ax[0].set_title("Observations: phase difference per radio (bands = no segment)")
    # phi fills [-pi, pi], so a legend inside the axes covers real data. Add
    # headroom above and put it there.
    ax[0].set_ylim(-3.4, 5.2)
    ax[0].legend(fontsize=7, ncol=3, loc="upper center", framealpha=0.9)

    # ---- track + confidence
    ax[1].axhline(np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].axhline(-np.pi / 2, ls=":", c=(0.7, 0.7, 0.7))
    ax[1].plot(gt, color="black", lw=1.2, label="ground truth")
    ax[1].fill_between(
        t, theta - sigma, theta + sigma, color="tab:red", alpha=0.22, label="+-1 sigma"
    )
    ax[1].scatter(t, theta, s=3.0, color="tab:red", label="filter estimate")
    ax[1].set_ylabel("theta (rad)")
    ax[1].set_title(f"Estimate and confidence -- frame: {extras['frame']}")
    ax[1].legend(fontsize=8, ncol=3)

    # ---- error
    err = metrics.angular_error(theta, gt)
    final_rmse = float(np.sqrt((err**2).mean()))
    ax[2].plot(np.abs(err), lw=0.8, color="tab:purple", label="|error|")
    ax[2].plot(
        np.sqrt(np.cumsum(err**2) / (np.arange(err.shape[0]) + 1)),
        lw=1.6,
        color="black",
        label="running RMSE",
    )
    ax[2].axhline(
        final_rmse,
        ls="dashed",
        color="tab:green",
        label=f"final RMSE {final_rmse:.3f} rad ({np.degrees(final_rmse):.1f} deg)",
    )
    ax[2].set_ylabel("|error| (rad)")
    ax[2].set_xlabel("sample index (time)")
    ax[2].set_title(f"Error -- MSE {float((err ** 2).mean()):.4f} rad^2")
    ax[2].legend(fontsize=8, ncol=3)

    # ---- calibration
    z = calibration.z_scores(theta, gt, sigma)
    if z.size:
        ax[3].hist(
            np.clip(z, -6, 6),
            bins=61,
            density=True,
            alpha=0.7,
            color="tab:blue",
            label="z = error / sigma (clipped to +-6)",
        )
        grid = np.linspace(-6, 6, 400)
        ax[3].plot(
            grid,
            np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi),
            color="black",
            label="unit normal (perfect calibration)",
        )
        # 3 decimals: nominal 3-sigma is 0.9973 and rounds to "1.00" at 2,
        # which reads as though the band is expected to hold everything
        cov = "  ".join(
            f"{r['k']}s: {r['measured']:.3f} (nom {r['nominal']:.3f})"
            for r in calibration.coverage(theta, gt, sigma)
        )
        ratio = calibration.calibration_ratio(theta, gt, sigma)
        ax[3].set_title(f"Calibration -- std(z)={ratio:.2f}   coverage {cov}")
    else:
        ax[3].set_title("Calibration -- this filter reported no usable sigma")
    ax[3].set_xlabel("z-score")
    ax[3].legend(fontsize=8)

    # ---- posterior heat map
    if posterior is not None:
        post = np.asarray(posterior)[:n]
        ax[4].imshow(
            post.T,
            aspect="auto",
            origin="lower",
            extent=[0, n, -np.pi, np.pi],
            cmap="viridis",
        )
        ax[4].plot(gt, color="white", lw=1.2, label="ground truth")
        ax[4].scatter(t, theta, s=2.0, color="red", label="filter estimate")
        ax[4].set_ylabel("theta (rad)")
        ax[4].set_xlabel("sample index (time)")
        ax[4].set_title(f"Network posterior over theta ({post.shape[-1]} bins)")
        ax[4].legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, help="zarr prefix")
    parser.add_argument("--precompute-cache", type=str, required=True)
    parser.add_argument("--empirical-pkl-fn", type=str, required=True)
    parser.add_argument("--segmentation-version", type=float, default=3.7)
    parser.add_argument("--nthetas", type=int, default=65)
    parser.add_argument("--filter", type=str, required=True, choices=FILTERS)
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help='JSON hyperparameters, e.g. \'{"N": 4096, "theta_err": 0.01}\'',
    )
    parser.add_argument("--checkpoint-fn", type=str, default=None)
    parser.add_argument("--inference-cache", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = get_parser().parse_args()
    params = json.loads(args.params)

    ds = open_dataset(
        args.dataset,
        args.precompute_cache,
        args.empirical_pkl_fn,
        args.segmentation_version,
        nthetas=args.nthetas,
    )
    theta, sigma, gt, extras = run_filter(
        ds,
        args.filter,
        params,
        checkpoint_fn=args.checkpoint_fn,
        inference_cache=args.inference_cache,
    )
    basename = os.path.basename(str(args.dataset)).replace(".zarr", "")
    fig = plot_filter_run(
        ds,
        theta,
        sigma,
        gt,
        extras,
        f"{args.filter}  {json.dumps(params, sort_keys=True)}\n{basename}",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_fn = os.path.join(args.output_dir, f"{basename}__{args.filter}.png")
    fig.savefig(out_fn, dpi=110)
    plt.close(fig)
    logging.info(f"wrote {out_fn}")
    print(json.dumps(summarize_run(theta, sigma, gt), indent=2, default=str))
