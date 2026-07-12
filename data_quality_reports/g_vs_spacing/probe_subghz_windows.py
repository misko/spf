"""Probe: why is fitted g so spread at 868/915 MHz?

Tests the 'windowing locks onto standing background instead of the beacon'
hypothesis on 3 datasets (worst-disagreement 915, an 868, and a healthy 2.4 GHz
control). For each dataset x receiver:
  col 1: measured mean_phase vs predicted -2pi(d/lam)sin(theta_gt)  (should be identity)
  col 2: phase vs time with prediction overlay (drift/lock visible)
  col 3: window amplitude, signal-class vs noise-class + duty cycle
  col 4: raw spectrum of one snapshot, signal window vs noise window (DC tone?)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spf.dataset.spf_dataset import v5spfdataset
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

DATASETS = [
    ("915 MHz 4cm (worst rx disagreement)",
     "wallarrayv3_2024_11_26_20_08_28_nRX2_rx_random_circle_spacing0p04"),
    ("868 MHz 7.5cm",
     "wallarrayv3_2024_10_29_02_26_59_nRX2_bounce_spacing0p075"),
    ("2.412 GHz 7.5cm control (OK)",
     "wallarrayv3_2024_08_31_13_37_57_nRX2_rx_circle_spacing0p075"),
]
CACHE = "/mnt/md2/cache/precompute_cache_3p7"
ROOT = "/mnt/md2/cache/nosig_data"

fig, axes = plt.subplots(3, 4, figsize=(19, 11))

for row_i, (label, name) in enumerate(DATASETS):
    ds = v5spfdataset(
        f"{ROOT}/{name}.zarr", nthetas=65, ignore_qc=True,
        skip_fields=set(["signal_matrix"]), precompute_cache=CACHE, paired=False,
    )
    k = -2 * np.pi * float(ds.rx_wavelength_spacing)
    axs = axes[row_i]
    for r, col in ((0, "tab:blue"), (1, "tab:orange")):
        pm = ds.mean_phase[f"r{r}"].numpy()
        th = ds.ground_truth_thetas[r].numpy()
        ok = np.isfinite(pm) & np.isfinite(th)
        pred = k * np.sin(th[ok])
        axs[0].scatter(pred, np.angle(np.exp(1j * pm[ok])), s=3, alpha=0.3,
                       color=col, label=f"r{r}")
        axs[1].plot(np.where(ok)[0], np.angle(np.exp(1j * pm[ok])), ".", ms=2,
                    alpha=0.4, color=col)
    axs[1].plot(np.where(ok)[0], np.angle(np.exp(1j * pred)), "-", lw=0.7,
                color="k", alpha=0.6, label="predicted (r1 geom)")
    lim = max(1.2, np.abs(k) * 1.15)
    axs[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-np.pi, np.pi)
    axs[0].set_xlabel("predicted phase (rad)"); axs[0].set_ylabel("measured mean_phase (rad)")
    axs[0].set_title(f"{label}\nphase vs geometry", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("snapshot #"); axs[1].set_ylabel("phase (rad)")
    axs[1].set_title("phase vs time (black = predicted)", fontsize=9)
    axs[1].legend(fontsize=7)

    # window stats from the precompute store
    try:
        aws = ds.precomputed_zarr["r1/all_windows_stats"][:]      # (n, 3, n_windows) or (n, n_windows, 3)
        mask = ds.precomputed_zarr["r1/downsampled_segmentation_mask"][:]
        if aws.shape[1] == 3:
            amp = aws[:, 2, :]
        else:
            amp = aws[:, :, 2]
        m = mask.astype(bool)
        duty = float(m.mean())
        sig_amp, noi_amp = amp[m], amp[~m]
        bins = np.geomspace(max(1e-3, min(noi_amp.min(), sig_amp.min()) + 1e-3),
                            max(sig_amp.max(), noi_amp.max()), 60)
        axs[2].hist(noi_amp, bins=bins, alpha=0.5, label=f"noise-class ({(~m).sum()})",
                    color="gray", density=True)
        axs[2].hist(sig_amp, bins=bins, alpha=0.5, label=f"signal-class ({m.sum()})",
                    color="tab:red", density=True)
        axs[2].set_xscale("log")
        axs[2].set_xlabel("window |signal| median"); axs[2].set_ylabel("density")
        axs[2].set_title(f"window amplitude by class — duty {100*duty:.1f}%", fontsize=9)
        axs[2].legend(fontsize=7)
    except Exception as e:
        axs[2].set_title(f"window stats unavailable: {e}", fontsize=7)

    # raw spectrum: one snapshot, r1; signal window vs noise window
    try:
        z = zarr_open_from_lmdb_store(f"{ROOT}/{name}.zarr", mode="r")
        snap = int(np.argmax(mask.sum(axis=1)))  # snapshot with most signal windows
        smk = "r1/signal_matrix" if "r1" in z else "receivers/r1/signal_matrix"
        sm = np.array(z[smk][snap])  # (2, N) complex
        n_win = mask.shape[1]
        wsize = sm.shape[1] // n_win
        wm = mask[snap].astype(bool)
        wi_sig = int(np.argmax(wm))
        wi_noi = int(np.argmax(~wm))
        fs_spec = []
        for wi, lab, colr in ((wi_sig, "signal window", "tab:red"),
                              (wi_noi, "noise window", "gray")):
            x = sm[0, wi * wsize:(wi + 1) * wsize]
            spec = np.fft.fftshift(np.abs(np.fft.fft(x * np.hanning(len(x)))) ** 2)
            f = np.fft.fftshift(np.fft.fftfreq(len(x)))
            axs[3].plot(f, 10 * np.log10(spec + 1e-6), lw=0.7, color=colr, label=lab)
        axs[3].axvline(0, color="k", lw=0.6, ls=":")
        axs[3].set_xlabel("freq (fraction of fs)"); axs[3].set_ylabel("power (dB)")
        axs[3].set_title(f"raw spectrum, snapshot {snap}, ant0 of r1", fontsize=9)
        axs[3].legend(fontsize=7)
    except Exception as e:
        axs[3].set_title(f"raw unavailable: {type(e).__name__}: {e}", fontsize=6)
    # beamformer GT-alignment metric (uses the same representation the NN sees)
    try:
        for r in (0, 1):
            bf = np.asarray(ds.precomputed_zarr[f"r{r}/weighted_beamformer"][:], dtype=np.float64)
            th = ds.ground_truth_thetas[r].numpy()
            ok = np.isfinite(th) & np.isfinite(bf).all(axis=1)
            bf, thv = bf[ok], th[ok]
            # fold GT into [-pi/2, pi/2] (front-back) and map to 65 bins
            thf = np.arctan2(np.sin(thv), np.abs(np.cos(thv)))
            gt_bin = np.clip(((thf + np.pi / 2) / np.pi * (bf.shape[1] - 1)).round().astype(int),
                             0, bf.shape[1] - 1)
            gt_val = bf[np.arange(len(gt_bin)), gt_bin]
            rank = (bf < gt_val[:, None]).mean(axis=1)  # percentile of GT bin per snapshot
            print(f"BFMETRIC {label} r{r}: mean GT percentile={rank.mean():.3f} "
                  f"(0.5=uninformative)  frac>0.8={float((rank > 0.8).mean()):.3f}  n={len(rank)}")
    except Exception as e:
        print("BFMETRIC failed:", label, e)
    ds.close()

fig.suptitle("Sub-GHz g spread probe: is the segmenter locking onto a standing background?", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=110)
print("wrote", out)
