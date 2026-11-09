"""Audit the phase->bearing conversion used in the gain-phase reports.

The reports convert a phase quantity to a bearing quantity with a FIXED factor of
~0.40 deg-bearing per deg-phase.  For a 2-element array the forward map the
dataset itself uses (spf_dataset.get_ground_truth_phis) is

    phi = pi_norm( -2*pi*(d/lambda)*sin(theta) )

so the LOCAL derivative is

    |dtheta/dphi| = 1 / ( 2*pi*(d/lambda)*|cos(theta)| )

which depends on BOTH d/lambda and theta, and diverges at endfire (theta=+-90 deg).
This script measures the real framewise distribution over the rover corpus, using
each frame's real d/lambda (cached_keys[r]["rx_wavelength_spacing"]) and real
ground-truth bearing (ground_truth_thetas[r]).

It ALSO measures what the pipeline actually does, which is not a local inverse
sine at all: spf.filters.filters.theta_phi_to_p_vec looks phi up in a 65-bin
empirical table p(theta | phi_bin).  So we additionally report
  * the phi-bin width (the quantisation the correction has to clear at all)
  * the fraction of frames whose phi bin actually MOVES under a delta-degree shift
  * the shift in the table row's circular-mean theta and MAP theta
  * the total-variation distance between the pre/post likelihood rows

READ-ONLY.  Datasets are opened through v5spfdataset exactly as the committed
analysis scripts do; nothing under /mnt is written.

Usage:
  cd <worktree>; python3 <this> [capture_list.txt] > out.txt
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from spf.dataset.spf_dataset import get_empirical_dist, v5spfdataset  # noqa: E402

# Identical to the KW dict in
# spf/filters/reports/phasecorr_direct_pf_20260814_v1/analysis/geometry_conditioned.py
KW = dict(
    nthetas=65,
    ignore_qc=True,
    precompute_cache="/mnt/qnap01/mouse9911/rovers_2026/precompute",
    paired=True,
    snapshots_per_session=1,
    readahead=True,
    skip_fields=set(
        [
            "windowed_beamformer",
            "weighted_beamformer",
            "all_windows_stats",
            "downsampled_segmentation_mask",
            "signal_matrix",
            "simple_segmentations",
        ]
    ),
    empirical_data_fn="empirical_dists/full_20260809_v1.pkl",
    segmentation_version=3.7,
)

NBINS = 65
DELTAS_DEG = [0.5, 1.0, 1.9, 6.97]  # 6.97 = the report's median total correction


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def phi_bin(phi_rad, nbins=NBINS):
    """Exactly spf.filters.filters.theta_phi_to_bins."""
    return (np.floor(nbins * (phi_rad + np.pi) / (2 * np.pi)).astype(np.int64)) % nbins


def circ_mean(p, centers):
    """Circular mean of a discrete distribution p over angle centers."""
    z = (p * np.exp(1j * centers)).sum(axis=-1)
    return np.angle(z)


def pct(a, q):
    return float(np.percentile(a, q))


def main(list_fn):
    fns = [l.strip() for l in open(list_fn) if l.strip()]
    # dedup on the underlying RX capture, same rule as geometry_conditioned.py
    seen, use = set(), []
    for f in fns:
        rx = f.split("/")[-1].split(".")[0]
        if rx not in seen:
            seen.add(rx)
            use.append(f)
    print(f"# {len(fns)} merged stores -> {len(use)} unique RX captures")

    theta_centers = (np.arange(NBINS) + 0.5) * (2 * np.pi / NBINS) - np.pi

    J_all, dl_all, cos_all, th_all = [], [], [], []
    # table-based instrument, keyed by delta
    tab = {d: {"binmoved": [], "dmean": [], "dmap": [], "tv": []} for d in DELTAS_DEG}
    per_stream = []
    n_ds = 0

    for f in use:
        try:
            ds = v5spfdataset(prefix=f, **KW)
        except Exception as e:  # noqa: BLE001
            print(f"# SKIP {f}: {type(e).__name__} {e}")
            continue
        n_ds += 1
        for r in range(ds.n_receivers):
            dl = np.asarray(
                ds.cached_keys[r]["rx_wavelength_spacing"], dtype=float
            ).ravel()
            th = np.asarray(ds.ground_truth_thetas[r], dtype=float).ravel()
            phi_meas = np.asarray(ds.mean_phase[f"r{r}"], dtype=float).ravel()
            n = min(len(dl), len(th), len(phi_meas))
            dl, th, phi_meas = dl[:n], th[:n], phi_meas[:n]
            ok = np.isfinite(dl) & np.isfinite(th) & (dl > 0)
            if ok.sum() < 50:
                continue
            dlk, thk = dl[ok], th[ok]
            c = np.abs(np.cos(thk))
            # |dtheta/dphi| is dimensionless -> deg-bearing per deg-phase directly
            with np.errstate(divide="ignore"):
                J = 1.0 / (2 * np.pi * dlk * c)
            J_all.append(J)
            dl_all.append(dlk)
            cos_all.append(c)
            th_all.append(thk)
            per_stream.append(
                (float(np.median(J)), float(dlk.mean()), int(ok.sum()), f, r)
            )

            # ---- what the pipeline actually consumes ----
            okp = ok & np.isfinite(phi_meas)
            if okp.sum() >= 50:
                p = phi_meas[okp]
                ed = get_empirical_dist(ds, r)  # [phi_bin, theta_bin]
                ed = np.asarray(ed.numpy() if isinstance(ed, torch.Tensor) else ed,
                                dtype=np.float64)
                rows = ed / np.clip(ed.sum(axis=1, keepdims=True), 1e-30, None)
                b0 = phi_bin(wrap(p))
                r0 = rows[b0]
                m0 = circ_mean(r0, theta_centers)
                a0 = theta_centers[r0.argmax(axis=1)]
                for d in DELTAS_DEG:
                    b1 = phi_bin(wrap(p + np.radians(d)))
                    r1 = rows[b1]
                    moved = b1 != b0
                    tab[d]["binmoved"].append(moved)
                    tab[d]["dmean"].append(np.abs(np.degrees(wrap(circ_mean(r1, theta_centers) - m0))))
                    tab[d]["dmap"].append(np.abs(np.degrees(wrap(theta_centers[r1.argmax(axis=1)] - a0))))
                    tab[d]["tv"].append(0.5 * np.abs(r1 - r0).sum(axis=1))
        del ds

    J = np.concatenate(J_all)
    dl = np.concatenate(dl_all)
    cos = np.concatenate(cos_all)
    th = np.concatenate(th_all)
    N = len(J)

    out = {}
    print(f"\n# datasets loaded: {n_ds}   receiver-streams: {len(per_stream)}   frames: {N}")

    print("\n## 1. d/lambda actually present in the rover corpus")
    uq, cnt = np.unique(np.round(dl, 4), return_counts=True)
    for u, c in sorted(zip(uq, cnt), key=lambda t: -t[1]):
        print(f"   d/lambda = {u:.4f}   frames = {c:8d} ({100*c/N:5.1f}%)   "
              f"broadside |dtheta/dphi| = {1/(2*np.pi*u):.4f}")
    out["dlambda"] = {float(u): int(c) for u, c in zip(uq, cnt)}

    bs = 1.0 / (2 * np.pi * dl)
    print(f"\n   broadside factor 1/(2*pi*d/lambda): median {np.median(bs):.4f}  "
          f"min {bs.min():.4f}  max {bs.max():.4f}  frame-mean {bs.mean():.4f}")
    out["broadside"] = dict(median=float(np.median(bs)), min=float(bs.min()),
                            max=float(bs.max()), mean=float(bs.mean()))

    print("\n## 2. Framewise |dtheta/dphi| (deg bearing per deg phase)")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = {q: pct(J, q) for q in qs}
    for q in qs:
        print(f"   P{q:<3d} = {vals[q]:8.4f}")
    print(f"   IQR  = [{vals[25]:.4f}, {vals[75]:.4f}]  width {vals[75]-vals[25]:.4f}")
    print(f"   mean = {J.mean():.4f}   (NOT a usable statistic: E|dtheta/dphi| "
          f"diverges logarithmically at endfire)")
    print(f"   max  = {J.max():.4f}")
    print(f"   sqrt(mean J^2) = {np.sqrt((J**2).mean()):.4f}  (also divergent in the limit)")
    for t in [0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]:
        print(f"   fraction with |dtheta/dphi| > {t:<4} : {100*(J>t).mean():6.2f}%")
    out["J"] = dict(
        percentiles=vals, mean=float(J.mean()), rms=float(np.sqrt((J**2).mean())),
        max=float(J.max()),
        frac_gt={str(t): float((J > t).mean()) for t in [0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]},
        n=int(N),
    )

    print("\n## 3. The inverse direction: |dphi/dtheta| (deg phase per deg bearing)")
    K = 2 * np.pi * dl * cos
    kv = {q: pct(K, q) for q in qs}
    for q in qs:
        print(f"   P{q:<3d} = {kv[q]:8.4f}")
    print(f"   mean = {K.mean():.4f}  (this one IS well defined)")
    print(f"   -> 'phase budget per degree of bearing', mean-sense: {K.mean():.4f} deg-phase/deg-bearing")
    print(f"   -> reciprocal of the mean: {1/K.mean():.4f} deg-bearing/deg-phase")
    out["K"] = dict(percentiles=kv, mean=float(K.mean()), recip_mean=float(1 / K.mean()))

    print("\n## 4. Ground-truth bearing relative to the array (deg)")
    print(f"   |theta| percentiles: " + "  ".join(
        f"P{q}={pct(np.abs(np.degrees(th)), q):.1f}" for q in [5, 25, 50, 75, 95]))
    for w in [5, 10, 20, 30]:
        near = (np.abs(np.abs(np.degrees(th)) - 90.0) < w).mean()
        print(f"   fraction within {w:2d} deg of endfire (|theta|=90): {100*near:5.2f}%")
    out["endfire"] = {str(w): float((np.abs(np.abs(np.degrees(th)) - 90.0) < w).mean())
                      for w in [5, 10, 20, 30]}

    print("\n## 5. Per-stream medians of |dtheta/dphi| (n = %d streams)" % len(per_stream))
    sm = np.array([p[0] for p in per_stream])
    print(f"   median of stream medians {np.median(sm):.4f}   "
          f"min {sm.min():.4f}  max {sm.max():.4f}")
    print(f"   streams with median > 0.4 : {(sm>0.4).sum()}/{len(sm)}")
    out["stream_medians"] = dict(median=float(np.median(sm)), min=float(sm.min()),
                                 max=float(sm.max()), n=len(sm),
                                 n_gt_0p4=int((sm > 0.4).sum()))

    print("\n## 6. What the PIPELINE does: 65-bin empirical table p(theta | phi_bin)")
    print(f"   phi bin width  = {360/NBINS:.4f} deg of phase")
    print(f"   theta bin width= {360/NBINS:.4f} deg of bearing")
    print("   A phase correction smaller than one phi bin changes the likelihood the")
    print("   filter consumes on ONLY the frames whose phi happens to sit within the")
    print("   correction of a bin edge.  On every other frame the effect is EXACTLY zero.")
    tabout = {}
    for d in DELTAS_DEG:
        mv = np.concatenate(tab[d]["binmoved"])
        dm = np.concatenate(tab[d]["dmean"])
        da = np.concatenate(tab[d]["dmap"])
        tv = np.concatenate(tab[d]["tv"])
        print(f"\n   delta = {d:.2f} deg of phase   (n = {len(mv)} frames)")
        print(f"     frames whose phi bin moves at all : {100*mv.mean():6.2f}%  "
              f"(theory: delta/binwidth = {100*d/(360/NBINS):.2f}%)")
        print(f"     |shift in table circular-mean theta| : mean {dm.mean():.4f} deg, "
              f"median {np.median(dm):.4f}, P90 {pct(dm,90):.4f}, max {dm.max():.3f}")
        print(f"     |shift in table MAP theta|           : mean {da.mean():.4f} deg, "
              f"frames with any MAP move {100*(da>1e-9).mean():.2f}%")
        print(f"     total-variation dist between rows    : mean {tv.mean():.4f}")
        tabout[str(d)] = dict(
            frac_bin_moved=float(mv.mean()),
            dmean_mean=float(dm.mean()), dmean_median=float(np.median(dm)),
            dmean_p90=float(pct(dm, 90)), dmean_max=float(dm.max()),
            dmap_mean=float(da.mean()), frac_map_moved=float((da > 1e-9).mean()),
            tv_mean=float(tv.mean()), n=int(len(mv)),
        )
    out["table"] = tabout

    print("\n## 7. Report's fixed factor vs measured")
    for src, lo, hi in [("report 1.0-1.9 deg phase", 1.0, 1.9)]:
        print(f"   {src}:")
        print(f"     report's fixed 0.40 factor -> {0.40*lo:.3f} - {0.40*hi:.3f} deg bearing")
        print(f"     measured MEDIAN factor {vals[50]:.4f} -> "
              f"{vals[50]*lo:.3f} - {vals[50]*hi:.3f} deg bearing")
        print(f"     measured P90    factor {vals[90]:.4f} -> "
              f"{vals[90]*lo:.3f} - {vals[90]*hi:.3f} deg bearing")
        print(f"     measured P95    factor {vals[95]:.4f} -> "
              f"{vals[95]*lo:.3f} - {vals[95]*hi:.3f} deg bearing")

    print("\n## 8. Percentage-of-RMSE arithmetic (filter RMSEs from the committed report)")
    for rmse in [41.1, 55.6]:
        print(f"   RMSE {rmse} deg:")
        for name, fac in [("report 0.40", 0.40), ("median", vals[50]),
                          ("P90", vals[90]), ("P95", vals[95])]:
            print(f"     {name:<12} 1.0 deg -> {100*fac*1.0/rmse:5.2f}%   "
                  f"1.9 deg -> {100*fac*1.9/rmse:5.2f}%")
    out["pct"] = {
        str(rmse): {name: [100 * fac * 1.0 / rmse, 100 * fac * 1.9 / rmse]
                    for name, fac in [("report_0.40", 0.40), ("median", vals[50]),
                                      ("P90", vals[90]), ("P95", vals[95])]}
        for rmse in [41.1, 55.6]
    }

    with open("/tmp/claude-1000/-home-mouse9911-gits-spf/fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/audit2/jacobian_audit.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n# wrote jacobian_audit.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
