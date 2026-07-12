"""Sample measured IF (beacon tone offset from DC) across the wall fleet.

For a stratified sample of wall datasets, read ONE snapshot of raw IQ per receiver
from the original zarr (nosig copies have signal_matrix stripped), FFT it, and record:
  - meas_if_frac: frequency of the strongest tone, as a fraction of fs (signed)
  - dc_frac:      fraction of window power within +-0.2% fs of DC
  - fs:           sampling rate from the embedded config (if parseable)
Joins are done later by dataset name against metrics_v2.csv.
"""

import glob
import os

import numpy as np
import pandas as pd
import yaml

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

HERE = os.path.dirname(os.path.abspath(__file__))
PER_BAND = 40
RAW_GLOBS = ["/mnt/md0/2d_wallarray_v2_data/*/{n}.zarr",
             "/mnt/md1/2d_wallarray_v2_data/*/{n}.zarr",
             "/mnt/md2/2d_wallarray_v2_data/*/{n}.zarr"]


def find_raw(name):
    for g in RAW_GLOBS:
        hits = glob.glob(g.format(n=name))
        if hits:
            return hits[0]
    return None


def measure(path):
    z = zarr_open_from_lmdb_store(path, mode="r")
    fs = None
    try:
        cfg = yaml.safe_load(str(z["config"][0]))
        rx = cfg["receivers"][0]
        for k in ("f-sampling", "sampling-frequency", "fs"):
            if k in rx:
                fs = float(rx[k])
                break
    except Exception:
        pass
    out = {}
    for r in ("r0", "r1"):
        try:
            arr = z[f"receivers/{r}/signal_matrix"]
            snap = arr.shape[0] // 2
            x = np.array(arr[snap][0])
            n = min(len(x), 65536)
            x = x[:n]
            sp = np.abs(np.fft.fft(x * np.hanning(n))) ** 2
            f = np.fft.fftfreq(n)
            tot = sp.sum()
            if tot <= 0:
                continue
            pk = int(np.argmax(sp))
            out[f"{r}_if_frac"] = float(f[pk])
            out[f"{r}_dc_frac"] = float(sp[np.abs(f) < 0.002].sum() / tot)
            out[f"{r}_tone_frac"] = float(
                sp[np.abs(f - f[pk]) < 0.002].sum() / tot)
        except Exception:
            continue
    out["fs"] = fs
    return out


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(os.path.join(HERE, "../../pdf_scripts/dataset/metrics_v2.csv"))
    w = df[(df.platform == "wall") & df.rx_lo.notna() & (df.rx_lo > 1e8)].copy()
    w["band"] = pd.cut(w.rx_lo, [0.8e9, 1e9, 2.45e9, 2.5e9, 6e9],
                       labels=["subGHz", "2.412", "2.46x", "5.8"])
    rows = []
    for band, grp in w.groupby("band", observed=True):
        take = grp.sample(min(PER_BAND, len(grp)), random_state=0)
        for _, r in take.iterrows():
            path = find_raw(r.dataset)
            if path is None:
                continue
            try:
                m = measure(path)
            except Exception:
                continue
            m.update(dataset=r.dataset, band=str(band), rx_lo=r.rx_lo,
                     r0_circstd_corr=r.r0_circstd_corr,
                     r1_circstd_corr=r.r1_circstd_corr,
                     r0_g=r.r0_g, r1_g=r.r1_g, status=r.status)
            rows.append(m)
            print(f"{str(band):7s} {r.dataset[:58]:58s} "
                  f"if={m.get('r0_if_frac', np.nan):+.4f} dc={m.get('r0_dc_frac', np.nan):.2f}")
    out = pd.DataFrame(rows)
    fn = os.path.join(HERE, "if_sample.csv")
    out.to_csv(fn, index=False)
    print(f"wrote {fn} ({len(out)} rows)")


if __name__ == "__main__":
    main()
