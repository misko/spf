"""Per-zarr field-report visualization + stats for one rover capture (raw v7 or merged).
Reads the zarr READ-ONLY (raw is immutable). Runs one file per process for crash isolation.
Usage: viz_one.py <zarr_path> <out_png> <out_json>
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Proj

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.rf import get_phase_diff

try:
    from spf.sdrpluto.detrend import detrend_np
except Exception:  # noqa
    def detrend_np(x):
        return x

path, out_png, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
st = {"path": path, "name": os.path.basename(path), "readable": False, "figure": None}


def nzmean(a):
    a = np.asarray(a)
    a = a[~np.isclose(a, 0.0)]
    return float(a.mean()) if a.size else float("nan")


def col(r, k):
    return r[k][:] if k in r else None


def rylim(ax, arrays, pad=0.06):
    vs = [np.asarray(a).ravel() for a in arrays if a is not None]
    if not vs:
        return
    v = np.concatenate(vs)
    v = v[np.isfinite(v)]
    if v.size:
        lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
        r = max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad * r, hi + pad * r)


try:
    z = zarr_open_from_lmdb_store(path, mode="r", readahead=True)
    recs = z["receivers"]
    rx_names = sorted(recs.keys())
    nrx = len(rx_names)
    st["n_receivers"] = nrx
    r0 = recs["r0"]
    gt = r0["gps_timestamp"][:]
    lat = r0["gps_lat"][:]
    lon = r0["gps_long"][:]
    n = int(gt.shape[0])
    valid = gt > 0.1
    st.update(n_snapshots=n, n_valid_gps=int(valid.sum()))
    if valid.any():
        st["gps_duration_s"] = float(gt[valid].max() - gt[valid].min())
        st["lat_mean"] = nzmean(lat)
        st["lon_mean"] = nzmean(lon)
    gains = col(r0, "gains")
    rssis = col(r0, "rssis")
    iqp = col(r0, "iq_power_dbfs")
    gds = col(r0, "gain_db_start")
    gde = col(r0, "gain_db_end")
    if gains is not None and (gains > 0).any():
        st["gain_mean"] = float(gains[gains > 0].mean())
    if rssis is not None:
        st["rssi_mean"] = float(np.asarray(rssis).mean())
    if iqp is not None:
        st["iq_power_dbfs_mean"] = float(np.asarray(iqp).mean())
    if gds is not None and gde is not None:
        st["gain_drift_abs_mean_db"] = float(np.abs(np.asarray(gds) - np.asarray(gde)).mean())
    txx = col(r0, "tx_pos_x_mm")
    merged = txx is not None and np.abs(np.asarray(txx)).sum() > 0
    st["merged"] = bool(merged)
    if merged:
        txy = r0["tx_pos_y_mm"][:]
        rxx = r0["rx_pos_x_mm"][:]
        rxy = r0["rx_pos_y_mm"][:]
        d = np.sqrt((txx - rxx) ** 2 + (txy - rxy) ** 2) / 1000.0
        st["txrx_dist_m"] = [float(d.min()), float(np.median(d)), float(d.max())]
    st["sig_shape"] = list(r0["signal_matrix"].shape)
    st["readable"] = True

    # ---- figure (only if there is real GPS data) ----
    if st["n_valid_gps"] >= 30:
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(os.path.basename(path), fontsize=10)

        # [0,0] track: tx/rx (merged) or GPS path (raw)
        ax = axs[0, 0]
        if merged:
            ax.scatter(txx / 1000, txy / 1000, s=4, label="tx (emitter)")
            ax.scatter(rxx / 1000, rxy / 1000, s=4, label="rx (receiver)")
            ax.set_title("TX / RX track (m)")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)
        else:
            vlat, vlon = lat[valid], lon[valid]
            good = (vlat != 0) & (vlon != 0)
            vlat, vlon = vlat[good], vlon[good]
            mlat, mlon = np.median(vlat), np.median(vlon)
            keep = (np.abs(vlat - mlat) < 0.02) & (np.abs(vlon - mlon) < 0.02)
            st["n_gps_glitches"] = int((~keep).sum())
            vlat, vlon = vlat[keep], vlon[keep]
            proj = Proj(proj="aeqd", lat_0=mlat, lon_0=mlon, datum="WGS84")
            x, y = proj(vlon, vlat)
            sc = ax.scatter(np.asarray(x), np.asarray(y), s=4, c=np.arange(len(x)), cmap="viridis")
            ax.set_title("GPS track (m, colored by time)")
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # [0,1] gain over time
        ax = axs[0, 1]
        gcollect = []
        for ri, rn in enumerate(rx_names):
            g = col(recs[rn], "gains")
            if g is not None:
                ax.plot(g[:, 0], lw=0.6, label=f"{rn}.g0")
                gcollect.append(g[:, 0])
                if g.shape[1] > 1:
                    ax.plot(g[:, 1], lw=0.6, label=f"{rn}.g1")
                    gcollect.append(g[:, 1])
        rylim(ax, gcollect)
        ax.set_title("gain (dB) over time")
        ax.set_xlabel("snapshot")
        ax.legend(fontsize=7, ncol=2)

        # [0,2] rssi over time
        ax = axs[0, 2]
        rcollect = []
        for rn in rx_names:
            rr = col(recs[rn], "rssis")
            if rr is not None:
                ax.plot(rr[:, 0], lw=0.5, label=f"{rn}.r0")
                rcollect.append(rr[:, 0])
        rylim(ax, rcollect)
        ax.set_title("RSSI over time")
        ax.set_xlabel("snapshot")
        ax.legend(fontsize=7)

        # [1,0] iq power dbfs
        ax = axs[1, 0]
        icollect = []
        for rn in rx_names:
            ip = col(recs[rn], "iq_power_dbfs")
            if ip is not None:
                ax.plot(ip[:, 0], lw=0.6, label=f"{rn}.e0")
                icollect.append(ip[:, 0])
                if ip.shape[1] > 1:
                    ax.plot(ip[:, 1], lw=0.6, label=f"{rn}.e1")
                    icollect.append(ip[:, 1])
        rylim(ax, icollect)
        ax.set_title("iq_power_dbfs over time")
        ax.set_xlabel("snapshot")
        ax.legend(fontsize=7)

        # sample snapshot for IQ/phase
        vidx = np.where(valid)[0]
        sidx = int(vidx[len(vidx) // 2])
        raw = None
        try:
            raw = detrend_np(np.asarray(r0["signal_matrix"][sidx]))
        except Exception as e:  # torn chunk
            st["iq_sample_error"] = f"{type(e).__name__}"
        ax = axs[1, 1]
        if raw is not None:
            m = raw.shape[1]
            ax.scatter(np.arange(m), np.abs(raw[0]), s=1, alpha=0.1, label="ant0")
            ax.scatter(np.arange(m), np.abs(raw[1]), s=1, alpha=0.1, label="ant1")
            ax.set_title(f"raw |IQ| (snapshot {sidx})")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "signal_matrix unreadable\n(torn chunk)", ha="center", va="center")
            ax.set_title("raw |IQ|")
        ax.set_xlabel("sample")

        ax = axs[1, 2]
        if raw is not None:
            pd = get_phase_diff(raw)
            ax.scatter(np.arange(len(pd)), pd, s=1, alpha=0.03)
            ax.set_title("inter-antenna phase diff")
            ax.set_ylim(-np.pi, np.pi)
        else:
            ax.set_title("phase diff (n/a)")
        ax.set_xlabel("sample")

        fig.tight_layout()
        fig.savefig(out_png, dpi=85, bbox_inches="tight")
        plt.close(fig)
        st["figure"] = os.path.basename(out_png)
except Exception as e:  # noqa
    st["error"] = f"{type(e).__name__}: {e}"

json.dump(st, open(out_json, "w"), indent=1)
print(json.dumps({k: st.get(k) for k in ["name", "readable", "n_valid_gps", "merged", "figure", "error"]}))
