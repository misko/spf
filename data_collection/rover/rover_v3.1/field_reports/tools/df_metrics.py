"""Direction-finding quality metrics for merged TX/RX datasets.

For every merged dataset, compare the RF-measured inter-antenna phase difference
(from segmentation) against the GPS-derived ground-truth phase, and summarise
signal strength and segmentation health. This produces the quantitative table in
the field reports.

Metrics, per receiver (r0, r1), over snapshots where both values are finite:
  R        Pearson correlation of measured mean_phase vs ground_truth_phi
  MAE      mean absolute circular error, in degrees
  nan_frac fraction of snapshots with no usable segmentation
plus per-dataset geometry (tx<->rx distance) and v7 signal metadata
(iq_power_dbfs, gains, rssis, gain_metadata_valid).

Reads the merged zarrs and the precompute cache READ-ONLY. `signal_matrix` is
skipped, so this is fast (no IQ is loaded).

Usage:
  python df_metrics.py <merged_dir> <precompute_cache_dir> [out.json] [--markdown]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from spf.dataset.spf_dataset import v5spfdataset
from spf.rf import torch_pi_norm


def label(name):
    """rover_..._tag_RO1.rover_..._tag_RO2 -> 'RO1@HH:MM:SS x RO2@HH:MM:SS'"""
    def one(x):
        t = x[15:23].replace("_", ":")
        tag = next((k for k in ("RO1", "RO2", "RO3") if f"tag_{k}" in x), "?")
        return f"{tag}@{t}"
    parts = name.split(".")
    return " x ".join(one(p) for p in parts[:2]) if len(parts) > 1 else one(parts[0])


def metrics_for(prefix, cache):
    ds = v5spfdataset(prefix, nthetas=65, ignore_qc=True, precompute_cache=cache,
                      snapshots_per_session=1, paired=True, gpu=False,
                      skip_fields=set(["signal_matrix"]))
    r = {"name": os.path.basename(prefix), "n": len(ds)}
    d = (ds.cached_keys[0]["tx_pos_mm"] / 1000 - ds.cached_keys[0]["rx_pos_mm"] / 1000)
    d = d.pow(2).sum(axis=1).sqrt().numpy()
    r["dist_m"] = [float(d.min()), float(np.median(d)), float(d.max())]

    for rx in [0, 1]:
        mp = ds.mean_phase[f"r{rx}"].numpy()
        gt = np.asarray(ds.ground_truth_phis[rx])
        m = np.isfinite(mp) & np.isfinite(gt)
        r[f"nan_frac_r{rx}"] = float(1 - m.mean())
        if m.sum() > 10:
            err = torch_pi_norm(torch.tensor(mp[m] - gt[m])).numpy()
            r[f"df_r{rx}"] = {"R": float(np.corrcoef(mp[m], gt[m])[0, 1]),
                              "mae_deg": float(np.degrees(np.abs(err).mean())),
                              "n_used": int(m.sum())}
        else:
            r[f"df_r{rx}"] = None
        g = ds.z[f"receivers/r{rx}"]
        if "iq_power_dbfs" in g:
            r[f"iq_power_dbfs_r{rx}"] = float(np.asarray(g["iq_power_dbfs"][:]).mean())
        r[f"gain_r{rx}"] = float(np.asarray(g["gains"][:]).mean())
        r[f"rssi_r{rx}"] = float(np.asarray(g["rssis"][:]).mean())
        if "gain_metadata_valid" in g:
            r[f"gain_valid_r{rx}"] = float(np.asarray(g["gain_metadata_valid"][:]).mean())
    ds.close()
    return r


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--markdown" in sys.argv
    merged, cache = args[0], args[1]
    out = args[2] if len(args) > 2 else None

    rows = []
    for zf in sorted(glob.glob(merged + "/*.zarr")):
        try:
            rows.append(metrics_for(zf[:-5], cache))
        except Exception as e:  # noqa
            rows.append({"name": os.path.basename(zf[:-5]), "error": f"{type(e).__name__}: {e}"})
            print("ERROR", zf, e, file=sys.stderr)

    ok = [r for r in rows if "error" not in r and r.get("df_r0")]
    if md:
        print("| RX x TX | Snapshots | R r0 | R r1 | MAE r0 | MAE r1 | NaN r0 | NaN r1 | median TX-RX |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(ok, key=lambda x: -x["n"]):
            print(f"| {label(r['name'])} | {r['n']} | {r['df_r0']['R']:.3f} | {r['df_r1']['R']:.3f} | "
                  f"{r['df_r0']['mae_deg']:.1f}° | {r['df_r1']['mae_deg']:.1f}° | "
                  f"{r['nan_frac_r0']:.2f} | {r['nan_frac_r1']:.2f} | {r['dist_m'][1]:.1f} m |")
    else:
        for r in sorted(ok, key=lambda x: -x["n"]):
            print(f"{label(r['name']):26s} n={r['n']:5d} R={r['df_r0']['R']:.3f}/{r['df_r1']['R']:.3f} "
                  f"MAE={r['df_r0']['mae_deg']:.1f}/{r['df_r1']['mae_deg']:.1f} deg")

    if ok:
        R = [r["df_r0"]["R"] for r in ok] + [r["df_r1"]["R"] for r in ok]
        M = [r["df_r0"]["mae_deg"] for r in ok] + [r["df_r1"]["mae_deg"] for r in ok]
        N = [r["nan_frac_r0"] for r in ok] + [r["nan_frac_r1"] for r in ok]
        IQ = [r["iq_power_dbfs_r0"] for r in ok if r.get("iq_power_dbfs_r0") is not None]
        G = [r["gain_r0"] for r in ok if r.get("gain_r0") is not None]
        print(f"\n{len(ok)} datasets, {sum(r['n'] for r in ok)} snapshots")
        print(f"  median R        = {np.median(R):.3f}")
        print(f"  median MAE      = {np.median(M):.1f} deg")
        print(f"  median nan_frac = {np.median(N):.2f}")
        if IQ:
            print(f"  median iq_power = {np.median(IQ):.1f} dBFS   median gain = {np.median(G):.1f} dB")

    if out:
        json.dump(rows, open(out, "w"), indent=1)
        print(f"\nwrote {out}")
