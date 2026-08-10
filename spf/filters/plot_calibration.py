"""Is the filter accurate, and does it know how accurate it is?

Two questions, two axes, and the whole point of these figures is that they are
independent. MSE against the pi^2/3 floor says whether an estimate is useful;
``std(z)`` against 1.0 says whether the stated sigma is honest. A filter can win
on one and lose badly on the other -- E-INF1's most accurate family is among its
worst calibrated.

``std(z)`` cannot rank filters by skill, by construction. A uniform-random
guesser that honestly reports sigma = pi/sqrt(3) = 1.814 rad scores exactly 1.00,
better than every real filter measured here. It is a scale-free self-consistency
check and is meaningless read alone.

Three figures:

``accuracy_vs_calibration``
    every configuration as one point, MSE against std(z), with both reference
    lines. The quadrants are the argument.
``reliability``
    measured against nominal coverage, from the dumped per-timestep tracks. The
    diagonal is perfect calibration; below it is overconfident.
``sigma_vs_error``
    what the filter claimed against what it delivered, binned. Answers whether a
    filter is uniformly overconfident or only wrong in particular regimes --
    which decides whether a scalar rescaling could fix it.

Usage::

    python spf/filters/plot_calibration.py \\
        --results <report>/results.json \\
        --tracks-dir <tracks>/ --output-dir <report>/figures
"""

import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from spf.evaluation import calibration, metrics  # noqa: E402

UNIFORM_RANDOM_MSE = metrics.UNIFORM_RANDOM_MSE
# A uniform guess has error uniform on [-pi, pi); its std is pi/sqrt(3).
UNIFORM_RANDOM_SIGMA = np.pi / np.sqrt(3.0)


def short(name, frame):
    return name.replace("_single_theta", "").replace("_", " ") + f"\n[{frame}]"


def mse_of(row):
    return row.get("mse_craft_theta_mean", row.get("mse_single_radio_theta_mean"))


def fig_accuracy_vs_calibration(rows, out_dir):
    per = collections.defaultdict(list)
    for r in rows:
        m, z = mse_of(r), r.get("calib_std_z_mean")
        if m is None or z is None or not np.isfinite(z):
            continue
        per[(r["type"], r["frame"])].append((m, z))

    fig, ax = plt.subplots(figsize=(11.5, 7))
    for i, key in enumerate(sorted(per)):
        v = np.array(per[key])
        ax.scatter(v[:, 0], v[:, 1], s=14, alpha=0.5,
                   label=short(*key).replace("\n", " "))
    ax.axvline(UNIFORM_RANDOM_MSE, color="tab:red", ls="--", lw=1.5)
    ax.axhline(1.0, color="black", ls="--", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("corpus-mean MSE (rad²)  →  worse")
    ax.set_ylabel("std(z)  →  more overconfident   (log scale)")
    ax.set_title(
        "Accuracy and honesty are independent\n"
        "left of red = better than guessing · near black = honest error bars"
    )
    ymax = ax.get_ylim()[1]
    ax.text(UNIFORM_RANDOM_MSE, ymax, " uniform random MSE ", color="tab:red",
            fontsize=8, va="top", ha="left")
    ax.text(ax.get_xlim()[0], 1.0, " calibrated (std(z)=1) ", color="black",
            fontsize=8, va="bottom", ha="left")
    # A random guesser lands exactly on the intersection: no skill, perfectly
    # honest. Naming it stops the reader treating low std(z) as "good".
    ax.plot([UNIFORM_RANDOM_MSE], [1.0], marker="*", ms=18, color="tab:red",
            zorder=6)
    # Into empty space below-right with a leader; placed on the point itself it
    # sits on top of the EKF cloud.
    ax.annotate(
        "uniform random guesser\n(zero skill, perfectly calibrated)",
        xy=(UNIFORM_RANDOM_MSE, 1.0),
        xytext=(0.97, 0.06), textcoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="tab:red",
        arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0,
                    "shrinkA": 2, "shrinkB": 6},
    )
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "accuracy_vs_calibration.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def load_tracks(tracks_dir):
    """{(type, frame): [(theta, sigma, gt), ...]} from the dumped npz files."""
    out = collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(tracks_dir, "*.npz"))):
        with np.load(fn) as z:
            out[(str(z["type"]), str(z["frame"]))].append(
                (z["theta"], z["sigma"], z["gt"])
            )
    return out


def fig_reliability(tracks, out_dir):
    """Measured vs nominal coverage -- only computable because tracks are kept."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    q = np.linspace(0.05, 0.95, 19)
    for key in sorted(tracks):
        # pool every capture: one curve per family, not per dataset
        theta = np.concatenate([t[0] for t in tracks[key]])
        sigma = np.concatenate([t[1] for t in tracks[key]])
        gt = np.concatenate([t[2] for t in tracks[key]])
        curve = calibration.reliability_curve(theta, gt, sigma, quantiles=q)
        ax.plot([c["nominal"] for c in curve], [c["measured"] for c in curve],
                marker="o", ms=3, lw=1.3, label=short(*key).replace("\n", " "))
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1.5, label="perfect")
    ax.set_xlabel("nominal central mass the filter's σ claims")
    ax.set_ylabel("fraction of errors that actually landed inside")
    ax.set_title(
        "Reliability — below the diagonal is overconfident\n"
        "pooled over 48 rover captures, winning configuration per family"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    fn = os.path.join(out_dir, "reliability.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_sigma_vs_error(tracks, out_dir):
    """Claimed sigma against delivered |error|, binned by claimed sigma.

    If a family sits on one horizontal offset from the diagonal it is uniformly
    overconfident and a single scale factor would fix it. If the gap varies with
    sigma, the variance model itself is wrong and rescaling would not help.
    """
    keys = sorted(tracks)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        theta = np.concatenate([t[0] for t in tracks[key]])
        sigma = np.concatenate([t[1] for t in tracks[key]])
        gt = np.concatenate([t[2] for t in tracks[key]])
        err = np.abs(metrics.angular_error(theta, gt))
        ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(err)
        s, e = sigma[ok], err[ok]
        if s.size == 0:
            ax.axis("off")
            continue
        edges = np.quantile(s, np.linspace(0, 1, 13))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 20:
                xs.append(float(np.median(s[m])))
                ys.append(float(np.median(e[m])))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color="tab:blue")
        lim = max(max(xs + ys) if xs else 1.0, 1e-3)
        ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1.2)
        ax.set_title(short(*key), fontsize=8)
        ax.set_xlabel("claimed σ (rad)", fontsize=8)
        ax.set_ylabel("median |error| (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        "What the filter claimed vs what it delivered — points above the dashed "
        "line are overconfident\n"
        "a constant vertical offset could be rescaled away; a widening gap means "
        "the variance model is wrong",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fn = os.path.join(out_dir, "sigma_vs_error.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--tracks-dir", default=None)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    rows = json.load(open(a.results))["rows"]
    written = [fig_accuracy_vs_calibration(rows, a.output_dir)]
    if a.tracks_dir:
        tracks = load_tracks(a.tracks_dir)
        if tracks:
            written.append(fig_reliability(tracks, a.output_dir))
            written.append(fig_sigma_vs_error(tracks, a.output_dir))
    for fn in written:
        print("wrote", fn)
