"""Geometry-conditioned test of the DONOR correction on rover data (canonical).

Supersedes ``analysis/geometry_conditioned.py``, which was right about the
estimand and wrong about two things around it. Both are fixed here.

WHAT IS MEASURED
----------------
For every receiver stream,

    e_none = wrap(mean_phase - ground_truth_phi)
    e_corr = wrap(mean_phase - correction - ground_truth_phi)

each centred circularly over the stream (which removes any CONSTANT per-radio
phase offset, so only a gain-*dependent* effect can survive). The reported
statistic is the change in mean |e| when the donor correction is applied, averaged
unweighted over receiver streams.

DEFECT (a) FIXED -- CONCATENATE, DO NOT KEEP-FIRST
--------------------------------------------------
The merged corpus names stores ``<rx_capture>.<tx_capture>.zarr`` and
``v7_tx_rx_merge.py`` copies only the RX rows that temporally overlap that TX
partner. So when one physical RX recording was merged against two different TX
rovers, the two merged stores hold DIFFERENT, NON-OVERLAPPING intervals of the
same recording -- not duplicates. The old script kept the first store per RX
prefix and silently discarded the other interval.

Verified on this corpus (``verify_disjoint.py``): all 6 multi-store RX captures
have ZERO shared timestamps and ZERO seconds of interval overlap between their
segments. This script therefore CONCATENATES the segments per physical RX
capture, in time order, and still deduplicates at frame level on
``system_timestamp`` -- reporting the count it drops, which on this corpus is 0.

DEFECT (b) FIXED -- BOOTSTRAP BY PHYSICAL CAPTURE
-------------------------------------------------
The two receiver streams of one physical capture are the same rover on the same
trajectory through the same environment, sharing GPS/heading error and multipath.
Resampling them independently treats one capture as two independent draws. The
canonical CI resamples PHYSICAL CAPTURES, carrying both streams together. The old
per-stream CI is printed alongside so the two can be compared directly.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
It bounds the DONOR model -- R18's table applied to rover radios explicitly held
out of it. A noisy or mismatched predictor is attenuated toward zero correlation
even when the underlying physical term is real, so a null here does NOT bound a
same-radio or sample-weighted correction.

READ-ONLY. Opens rover stores through the standard read-only path and writes
nothing, anywhere.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, ".")

from spf.dataset.phase_corrected_dataset import PhaseCorrectedDataset  # noqa: E402
from spf.dataset.spf_dataset import v5spfdataset  # noqa: E402

MODEL = ("spf/calibrations/models/gsc9_arm_lut_per_radio/"
         "1040007c4a94000211000b009186843ef2.json")
KW = dict(
    nthetas=65, ignore_qc=True,
    precompute_cache="/mnt/qnap01/mouse9911/rovers_2026/precompute",
    paired=True, snapshots_per_session=1, readahead=True,
    skip_fields=set(["windowed_beamformer", "weighted_beamformer", "all_windows_stats",
                     "downsampled_segmentation_mask", "signal_matrix",
                     "simple_segmentations"]),
    empirical_data_fn="empirical_dists/full_20260809_v1.pkl", segmentation_version=3.7,
)
MIN_FRAMES = 50
N_BOOT = 10000
SEED = 0


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def centre(x):
    return wrap(np.asarray(x) - np.angle(np.mean(np.exp(1j * np.asarray(x)))))


# --------------------------------------------------------------------------
# grouping: the unit of independence
# --------------------------------------------------------------------------
def group_stores_by_physical_capture(store_paths):
    """Map each physical RX recording to the merged stores cut from it.

    A merged store is named ``<rx_capture>.<tx_capture>.zarr``; the RX capture is
    the physical recording -- one rover, one trajectory, one pair of receivers,
    one environment -- and is the unit that is independent across the corpus. Two
    stores sharing an RX capture are two disjoint time segments of it, produced by
    merging against two different TX partners.

    Returns an ``OrderedDict`` in first-appearance order, values in file-list
    order (so ``value[0]`` is exactly what the old keep-first dedup retained).
    """
    groups = OrderedDict()
    for path in store_paths:
        rx_capture = path.split("/")[-1].split(".")[0]
        groups.setdefault(rx_capture, []).append(path)
    return groups


def capture_date(rx_capture):
    """``rover_2026_08_05_23_16_08_...`` -> ``2026_08_05``; the id itself if unparsed."""
    m = re.match(r"rover_(\d{4}_\d{2}_\d{2})_", rx_capture)
    return m.group(1) if m else rx_capture


def segment_streams(store_path):
    """Per-receiver arrays for one merged store: (ts, e_none_raw, correction).

    ``e_none_raw`` is the un-centred geometry-removed phase residual; centring
    happens once per concatenated stream, not per segment.
    """
    ds = v5spfdataset(prefix=store_path, **KW)
    try:
        w = PhaseCorrectedDataset(ds, "arm_lut", MODEL)
        out = {}
        for r in range(ds.n_receivers):
            phi = np.asarray(ds.mean_phase[f"r{r}"], dtype=float)
            gtp = np.asarray(ds.ground_truth_phis[r], dtype=float)
            corr = np.asarray(w._corr[r], dtype=float)
            ts = np.asarray(ds.cached_keys[r]["system_timestamp"], dtype=float)
            n = min(len(phi), len(gtp), len(corr), len(ts))
            phi, gtp, corr, ts = phi[:n], gtp[:n], corr[:n], ts[:n]
            ok = (np.isfinite(phi) & np.isfinite(gtp) & np.isfinite(corr)
                  & np.isfinite(ts) & (ts > 0))
            out[r] = (ts[ok], wrap(phi[ok] - gtp[ok]), corr[ok], int(n))
        return out
    finally:
        ds.close()


def stream_stats(ts, e_none_raw, corr):
    """Per-stream summary after frame-level dedup and circular centring."""
    order = np.argsort(ts, kind="stable")
    ts, e_none_raw, corr = ts[order], e_none_raw[order], corr[order]
    keep = np.ones(len(ts), dtype=bool)
    if len(ts) > 1:
        keep[1:] = ts[1:] != ts[:-1]
    n_dup = int((~keep).sum())
    ts, e_none_raw, corr = ts[keep], e_none_raw[keep], corr[keep]
    e0 = centre(e_none_raw)
    e1 = centre(wrap(e_none_raw - corr))
    return dict(
        n=len(ts), n_dup=n_dup,
        mae_none=float(np.degrees(np.abs(e0)).mean()),
        mae_corr=float(np.degrees(np.abs(e1)).mean()),
        corr_dm=corr - corr.mean(), e0=e0,
    )


def cluster_bootstrap_ci(d, cluster_of_stream, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% CI for mean(d), resampling whole clusters with replacement.

    Every stream of a drawn cluster comes along, so the two receiver streams of a
    physical capture are never split apart. With clusters of size 2 this is the
    right unit: the pair shares one trajectory, one GPS/heading error, one
    environment and one multipath realisation.
    """
    clusters = np.asarray(cluster_of_stream)
    members = [np.flatnonzero(clusters == u) for u in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        boot[b] = d[np.concatenate([members[i] for i in pick])].mean()
    return tuple(np.percentile(boot, [2.5, 97.5]))


def stream_bootstrap_ci(d, n_boot=N_BOOT, seed=SEED):
    """The OLD interval: receiver streams resampled independently of each other."""
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def within_capture_correlation(d, cluster_of_stream):
    """Pearson r between the two streams' deltas across captures (nan if none)."""
    clusters = np.asarray(cluster_of_stream)
    pairs = [d[np.flatnonzero(clusters == u)] for u in np.unique(clusters)]
    pairs = [p for p in pairs if len(p) == 2]
    if len(pairs) < 3:
        return float("nan"), 0
    a = np.array(pairs)
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), len(pairs)


def summarise(per_stream, cluster_of_stream, label, n_boot=N_BOOT, seed=SEED):
    """Print mean |e| with/without, the change, and both bootstrap CIs."""
    mae0 = np.array([s["mae_none"] for s in per_stream])
    mae1 = np.array([s["mae_corr"] for s in per_stream])
    nfr = np.array([s["n"] for s in per_stream])
    d = mae1 - mae0
    uniq = np.unique(np.asarray(cluster_of_stream))

    lo_c, hi_c = cluster_bootstrap_ci(d, cluster_of_stream, n_boot, seed)
    lo_s, hi_s = stream_bootstrap_ci(d, n_boot, seed)
    # secondary robustness only: calendar date of the capture (a session that
    # runs past midnight is split, so this is coarse and has few clusters)
    day_of_stream = [capture_date(c) for c in cluster_of_stream]
    lo_d, hi_d = cluster_bootstrap_ci(d, day_of_stream, n_boot, seed)
    rho, n_pairs = within_capture_correlation(d, cluster_of_stream)

    X = np.concatenate([s["corr_dm"] for s in per_stream])
    Y = np.concatenate([s["e0"] for s in per_stream])
    r = float(np.corrcoef(X, Y)[0, 1])

    print(f"=== {label} ===")
    print(f"  physical captures / receiver-streams / frames : "
          f"{len(uniq)} / {len(d)} / {int(nfr.sum()):,}")
    print(f"  mean |e| without correction : {mae0.mean():7.3f} deg")
    print(f"  mean |e| with correction    : {mae1.mean():7.3f} deg")
    print(f"  change                      : {d.mean():+7.3f} deg")
    print(f"    95% CI, capture-clustered : [{lo_c:+.3f}, {hi_c:+.3f}]   "
          f"(width {hi_c - lo_c:.4f})")
    print(f"    95% CI, per-stream (old)  : [{lo_s:+.3f}, {hi_s:+.3f}]   "
          f"(width {hi_s - lo_s:.4f})   "
          f"ratio {(hi_c - lo_c) / (hi_s - lo_s):.3f}x")
    print(f"    95% CI, date-clustered    : [{lo_d:+.3f}, {hi_d:+.3f}]   "
          f"(width {hi_d - lo_d:.4f}, {len(set(day_of_stream))} dates) "
          f"[secondary]")
    print(f"  within-capture corr of delta: {rho:+.4f}  over {n_pairs} pairs")
    print(f"  better with correction on   : {(d < 0).sum()}/{len(d)} streams")
    print(f"  corr(correction, residual)  : {r:+.4f}   r^2 = {100 * r ** 2:.3f}%")
    print(f"  sd correction {np.degrees(X.std()):.2f} deg vs "
          f"sd residual {np.degrees(Y.std()):.2f} deg")
    sd_stream = np.array([np.degrees(s["corr_dm"].std()) for s in per_stream])
    print(f"  per-stream sd of correction : median {np.median(sd_stream):.3f} deg, "
          f"{int((sd_stream < 1e-9).sum())} stream(s) constant "
          f"(a constant correction is nulled exactly by the centring)")
    print(f"  frame-weighted mean |e| without / with : "
          f"{np.average(mae0, weights=nfr):.3f} / "
          f"{np.average(mae1, weights=nfr):.3f} deg")
    print()
    return dict(mae0=mae0, mae1=mae1, d=d, nfr=nfr, ci_cluster=(lo_c, hi_c),
                ci_stream=(lo_s, hi_s), r=r)


def main(list_fn):
    store_paths = [line.strip() for line in open(list_fn) if line.strip()]
    groups = group_stores_by_physical_capture(store_paths)
    n_multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"{len(store_paths)} merged stores -> {len(groups)} physical RX captures "
          f"({n_multi} of them split across >1 store)\n")

    # ---- read every store once ------------------------------------------
    seg = {}
    failed = []
    for path in store_paths:
        try:
            seg[path] = segment_streams(path)
        except Exception as exc:  # a store that will not open is reported, not hidden
            failed.append((path, repr(exc)))
    if failed:
        print(f"WARNING: {len(failed)} stores failed to open:")
        for path, exc in failed:
            print(f"  {path}\n    {exc}")
        print()

    # ---- (i) keep-first, reproducing the committed number ---------------
    kf_stats, kf_cluster = [], []
    for cap, paths in groups.items():
        first = paths[0]
        if first not in seg:
            continue
        for r, (ts, e_raw, corr, _) in sorted(seg[first].items()):
            if len(ts) < MIN_FRAMES:
                continue
            kf_stats.append(stream_stats(ts, e_raw, corr))
            kf_cluster.append(cap)

    # ---- (ii) concatenated, the canonical configuration -----------------
    cc_stats, cc_cluster, n_dup_total, n_seg_used = [], [], 0, 0
    for cap, paths in groups.items():
        paths = [p for p in paths if p in seg]
        if not paths:
            continue
        n_seg_used += len(paths)
        rx_idxs = sorted({r for p in paths for r in seg[p]})
        for r in rx_idxs:
            parts = [seg[p][r] for p in paths if r in seg[p]]
            parts.sort(key=lambda t: t[0].min() if len(t[0]) else np.inf)
            ts = np.concatenate([p[0] for p in parts])
            e_raw = np.concatenate([p[1] for p in parts])
            corr = np.concatenate([p[2] for p in parts])
            if len(ts) < MIN_FRAMES:
                continue
            st = stream_stats(ts, e_raw, corr)
            n_dup_total += st["n_dup"]
            cc_stats.append(st)
            cc_cluster.append(cap)
    print(f"concatenation used {n_seg_used} segments across {len(set(cc_cluster))} "
          f"captures; frame-level duplicate timestamps removed: {n_dup_total}\n")

    # ---- (iii) diagnostic: every store its own stream, no concatenation --
    # Not canonical. A capture split across two stores gets TWO circular
    # centrings instead of one, and its short segment is weighted equal to a
    # full-length stream. Reported only so the difference from (ii) is visible.
    st_stats, st_cluster = [], []
    for cap, paths in groups.items():
        for p in paths:
            if p not in seg:
                continue
            for r, (ts, e_raw, corr, _) in sorted(seg[p].items()):
                if len(ts) < MIN_FRAMES:
                    continue
                st_stats.append(stream_stats(ts, e_raw, corr))
                st_cluster.append(cap)

    kf = summarise(kf_stats, kf_cluster, "KEEP-FIRST (old dedup, reproduction)")
    cc = summarise(cc_stats, cc_cluster, "CONCATENATED (canonical)")
    ps = summarise(st_stats, st_cluster,
                   "PER-STORE, NOT CONCATENATED (diagnostic only)")

    print("=== what the two fixes moved ===")
    print(f"  frames  : {int(kf['nfr'].sum()):,} -> {int(cc['nfr'].sum()):,}  "
          f"(+{int(cc['nfr'].sum() - kf['nfr'].sum()):,}, "
          f"+{100 * (cc['nfr'].sum() / kf['nfr'].sum() - 1):.2f}%)")
    print(f"  streams : {len(kf['d'])} -> {len(cc['d'])}")
    print(f"  change  : {kf['d'].mean():+.4f} deg -> {cc['d'].mean():+.4f} deg "
          f"(per-store variant, not canonical: {ps['d'].mean():+.4f} deg)")
    print(f"  CI      : per-stream [{kf['ci_stream'][0]:+.4f}, "
          f"{kf['ci_stream'][1]:+.4f}] -> capture-clustered "
          f"[{cc['ci_cluster'][0]:+.4f}, {cc['ci_cluster'][1]:+.4f}]")
    if len(cc["d"]) == len(kf["d"]) and kf_cluster == cc_cluster:
        # same (capture, receiver) order in both, so element-wise is meaningful
        n_changed = int((np.abs(cc["d"] - kf["d"]) > 1e-12).sum())
        print(f"  streams whose delta actually moved: {n_changed} of "
              f"{2 * n_multi} touched (a stream whose correction is CONSTANT is "
              f"nulled by the centring and cannot move)")
    print("\nBounds the DONOR only. Attenuation from predictor error means this does not\n"
          "bound a same-radio or sample-weighted correction.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
