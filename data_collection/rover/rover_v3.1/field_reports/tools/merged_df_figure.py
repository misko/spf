"""DF figure for a merged dataset: beamformer heatmap with ground-truth theta
overlaid, and estimated theta (beamformer peaks) vs ground truth. Loads via
v5spfdataset (builds the segmentation/beamformer cache if missing).
Usage: viz_merged_df.py <prefix_without_ext> <out_png> <cache_dir>
"""
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spf.dataset.spf_dataset import v5spfdataset
from spf.rf import pi_norm, reduce_theta_to_positive_y

prefix, out_png, cache = sys.argv[1], sys.argv[2], sys.argv[3]
NB = 65  # nthetas / angle bins


def theta_to_bin(theta):
    return (pi_norm(np.asarray(theta)) / np.pi + 1) / 2 * (NB - 1)


ds = v5spfdataset(
    prefix, nthetas=NB, ignore_qc=True, precompute_cache=cache, gpu=False,
    snapshots_per_session=1, n_parallel=8, paired=True, segment_if_not_exist=True,
)
est = ds.get_estimated_thetas()

fig, axs = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle(os.path.basename(ds.zarr_fn) + "  —  beamformer & AoA vs ground truth", fontsize=10)

for rx in range(2):
    gt = np.asarray(ds.ground_truth_thetas[rx])
    gtr = np.asarray(reduce_theta_to_positive_y(ds.ground_truth_thetas[rx]))
    n = gt.shape[0]

    # --- left: beamformer heatmap (angle x time) with GT overlaid ---
    bf = ds.precomputed_zarr[f"r{rx}"].windowed_beamformer[:].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        bf = np.nan_to_num(bf / bf.max(axis=2, keepdims=True)).mean(axis=1)  # (T, NB)
        bf = np.nan_to_num(bf / bf.sum(axis=1, keepdims=True))
    ax = axs[rx, 0]
    ax.imshow(bf.T, origin="lower", aspect="auto", cmap="viridis", extent=[0, n, 0, NB - 1])
    ax.plot(np.arange(n), theta_to_bin(gt), color="red", lw=0.8, label="ground truth")
    ax.plot(np.arange(n), theta_to_bin(gtr), color="cyan", lw=0.8, alpha=0.8, label="reduced GT")
    ax.set_title(f"Rx{rx}: beamformer power (angle x time) vs GT")
    ax.set_yticks([0, (NB - 1) / 2, NB - 1])
    ax.set_yticklabels(["-pi", "0", "pi"])
    ax.set_xlabel("snapshot (time)")
    ax.set_ylabel("angle")
    ax.legend(fontsize=7, loc="upper right")

    # --- right: estimated theta (2 peaks) vs GT ---
    ax = axs[rx, 1]
    ax.plot(gt, color="red", alpha=0.6, lw=1.0, label="ground truth")
    ax.plot(gtr, color="green", alpha=0.6, lw=1.0, label="reduced GT")
    ax.scatter(range(n), pi_norm(est[f"r{rx}"][0]), s=2, label="peak1")
    ax.scatter(range(n), pi_norm(est[f"r{rx}"][1]), s=2, label="peak2")
    ax.set_title(f"Rx{rx}: estimated theta vs GT")
    ax.set_xlabel("snapshot (time)")
    ax.set_ylabel("theta (rad)")
    ax.set_ylim(-np.pi, np.pi)
    ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(out_png, dpi=85, bbox_inches="tight")
print("wrote", out_png, "n=", ds.ground_truth_thetas[0].shape)
