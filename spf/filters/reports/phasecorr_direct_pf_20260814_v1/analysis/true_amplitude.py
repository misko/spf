"""AUDIT: the true amplitude of the gain-dependent phase correction.

The committed report (spf/calibrations/dual_rx_gain_frequency/reports/
rover_model_gsc9_20260814_v1/REPORT.md, Addendum 2, lines 313-318) tabulates a
column headed "MAD about the mean" with 0.99 / 1.07 / 1.90 / 1.87 deg, and the
power calibration (spf/filters/reports/phasecorr_direct_pf_20260814_v1/analysis/
power_calibration.py, TERM_LO/TERM_HI = 1.0/1.9) feeds those same numbers into
Delta = 0.0101 * A^2 as though A were a GAUSSIAN RMS. MAD is not RMS.

This script recomputes, per radio x carrier, over the rover's real gain-cell
occupancy:

    weighted mean D, weighted MAD about that mean, weighted RMS (sd) about that
    mean, weighted P95 and max |deviation|, and the observed RMS/MAD ratio.

Three surfaces are scored, because "the correction" is ambiguous:

  MEASURED   the bench-measured anchored differential phase D from E-GSC9
             session A, exactly the quantity full_ladder.py's "no correction
             (baseline)" row scores. This is what Addendum 2's table used --
             its "mean |D|" column, 6.37/6.79/7.06/10.52, is that row's per
             cell-weighted MAE, term for term.
  ARM_LUT62  the deployed per-arm gain LUT, coefficients/luts62/*.json,
             D(g1,g2) = d1[g1] - d2[g2], evaluated per rover cell.
  RFBLOCK    the shipped mixer+LNA fit, coefficients/rfblock/*.json,
             which is what spf/calibrations/models/gsc9_arm_lut_per_radio/
             <serial>.json actually tabulates and deploys.

Weighting corpus: analysis/rover_cell_weights.json, which is the per-frame
(g1,g2) histogram over /mnt/qnap01/mouse9911/rovers_2026/merged/*.zarr, both
receivers, split by rx_lo. Verified byte-for-byte against the 'all' variant in
scratchpad/gsc9/rover_hist.pkl (the 'clean' variant, which drops 6 captures, is
NOT what was used).

Read-only. Writes only the JSON path given on argv (default: alongside this file).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPORT = ("/tmp/claude-1000/-home-mouse9911-gits-spf/"
          "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/impl/spf/calibrations/"
          "dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1")
ANALYSIS = REPORT + "/analysis"
LADDER2 = ("/tmp/claude-1000/-home-mouse9911-gits-spf/"
           "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/ladder2")
GSPM = ("/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
        "reports/gain_state_phase_model_20260802_v1/analysis")

sys.path.insert(0, ANALYSIS)
sys.path.insert(0, GSPM)

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402

R18 = "1040007c4a94000211000b009186843ef2"
R17 = "104000bac4950008230026001b440a003a"
NAME = {R18: "R18", R17: "R17"}
CARRIERS = (5_766_000_000.0, 5_840_000_000.0)
HS = FT.HardwareStates()

W = json.load(open(os.path.join(ANALYSIS, "rover_cell_weights.json")))


# --------------------------------------------------------------- statistics --
def wstats(x_deg, w, centre="circular"):
    """Occupancy-weighted dispersion of x about its own weighted mean.

    MAD here is MEAN absolute deviation about the weighted MEAN -- the same
    thing full_ladder.wmae computes when the prediction is a constant, which is
    how Addendum 2's column was built. Not median-absolute-deviation.

    The centre is the weighted CIRCULAR mean. That choice is not cosmetic: it is
    the one that reproduces all four published MADs to the printed two decimals
    (0.99 / 1.07 / 1.90 / 1.87); the arithmetic mean misses three of the four in
    the second decimal (0.99 / 1.08 / 1.89 / 1.86). See probe_center.py.
    """
    x = np.asarray(x_deg, dtype=float)
    w = np.asarray(w, dtype=float)
    tw = w.sum()
    if centre == "circular":
        mean = float(np.degrees(np.angle(
            (w * np.exp(1j * np.radians(x))).sum() / tw)))
    else:
        mean = float((w * x).sum() / tw)
    dev = x - mean
    ad = np.abs(dev)
    mad = float((w * ad).sum() / tw)
    rms = float(np.sqrt((w * dev ** 2).sum() / tw))
    # weighted quantile of |deviation|
    o = np.argsort(ad)
    cw = np.cumsum(w[o]) / tw
    p95 = float(ad[o][np.searchsorted(cw, 0.95)])
    return dict(
        mean_abs=float((w * np.abs(x)).sum() / tw),   # the "mean |D|" column
        mean=mean,
        mad=mad,
        rms=rms,
        p95=p95,
        max=float(ad.max()),
        rms_over_mad=rms / mad if mad else float("nan"),
        n_points=int(len(x)),
        total_weight=float(tw),
        centre=centre,
    )


# ------------------------------------------------------ surface definitions --
def lut62_surface(radio, lo):
    p = os.path.join(REPORT, "coefficients", "luts62",
                     f"arm_lut_{radio.lower()}_{int(lo/1e6)}_anchor62_20260814_v1.json")
    d = json.load(open(p))
    d1 = {int(k): v for k, v in d["d1_deg"].items()}
    d2 = {int(k): v for k, v in d["d2_deg"].items()}
    return lambda a, b: (d1[a] - d2[b]) if (a in d1 and b in d2) else None


def rfblock_surface(radio, lo):
    p = os.path.join(REPORT, "coefficients", "rfblock",
                     f"rfblock_{radio.lower()}_{int(lo/1e6)}_anchor62_20260814_v1.json")
    d = json.load(open(p))
    hm, hl = d["h_mixer_deg"], d["h_lna_deg"]

    def f(a, b):
        sa, sb = HS.state(2, a), HS.state(2, b)
        if sa is None or sb is None:
            return None
        try:
            v1 = hm["arm1"][str(sa[1])] + hl["arm1"][str(sa[0])]
            v2 = hm["arm2"][str(sb[1])] + hl["arm2"][str(sb[0])]
        except KeyError:
            return None
        return v1 - v2

    return f


def cell_scored(surface, lo):
    """Evaluate a correction surface once per rover cell, weight = frame count."""
    w = W[str(int(lo))]
    xs, ws, miss = [], [], 0
    for k, n in w.items():
        a, b = (int(t) for t in k.split(","))
        v = surface(a, b)
        if v is None:
            miss += n
            continue
        xs.append(v)
        ws.append(n)
    return np.asarray(xs), np.asarray(ws, dtype=float), miss, sum(w.values())


def main(out_path):
    os.chdir(LADDER2)                      # load_gsc.load() reads ./extracted
    f = load_gsc.load()
    f = f.sel(f.stage == "GSC9A")

    res = {"weighting_corpus": {
        "file": os.path.join(ANALYSIS, "rover_cell_weights.json"),
        "source": "/mnt/qnap01/mouse9911/rovers_2026/merged/*.zarr, both receivers, "
                  "per-frame (g1,g2) histogram split by rx_lo ('all' variant)",
        "cells": {lo: len(W[lo]) for lo in W},
        "frames": {lo: int(sum(W[lo].values())) for lo in W},
    }, "rows": []}

    for ser in (R18, R17):
        for lo in CARRIERS:
            radio = NAME[ser]
            m = (f.serial == ser) & (f.lo_hz == lo)
            fa = FT.add_anchor(f.sel(m), ref=62, per_epoch=True)

            # --- MEASURED: exactly full_ladder.wmae's weighting, per bench frame
            wf = np.array([W[str(int(lo))].get(f"{int(a)},{int(b)}", 0)
                           for a, b in zip(fa.g1, fa.g2)], dtype=float)
            keep = wf > 0
            Dd = np.degrees(S.wrap(fa.D[keep]))
            st = wstats(Dd, wf[keep])
            st["distinct_cells_with_weight"] = len(
                {(int(a), int(b)) for a, b in zip(fa.g1[keep], fa.g2[keep])})
            st["bench_frames_total"] = int(len(fa))
            st["bench_frames_on_rover_cells"] = int(keep.sum())
            st["rover_frames_covered_by_bench"] = float(
                wf[keep].sum() / sum(W[str(int(lo))].values()))
            res["rows"].append(dict(surface="MEASURED", radio=radio,
                                    carrier_mhz=int(lo / 1e6), **st))

            # --- deployed surfaces, one value per rover cell
            for tag, mk in (("ARM_LUT62", lut62_surface), ("RFBLOCK", rfblock_surface)):
                xs, ws, miss, tot = cell_scored(mk(radio, lo), lo)
                st = wstats(xs, ws)
                st["distinct_cells_with_weight"] = int(len(xs))
                st["rover_frames_covered"] = float(ws.sum() / tot)
                st["rover_frames_uncovered"] = int(miss)
                res["rows"].append(dict(surface=tag, radio=radio,
                                        carrier_mhz=int(lo / 1e6), **st))

    hdr = (f"{'surface':<10}{'radio':<6}{'car':>6}{'mean|D|':>9}{'meanD':>9}"
           f"{'MAD':>8}{'RMS':>8}{'P95':>8}{'max':>8}{'RMS/MAD':>9}{'cells':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in res["rows"]:
        print(f"{r['surface']:<10}{r['radio']:<6}{r['carrier_mhz']:>6}"
              f"{r['mean_abs']:>9.3f}{r['mean']:>9.3f}{r['mad']:>8.3f}{r['rms']:>8.3f}"
              f"{r['p95']:>8.3f}{r['max']:>8.3f}{r['rms_over_mad']:>9.3f}"
              f"{r['distinct_cells_with_weight']:>7d}")

    for tag in ("MEASURED", "ARM_LUT62", "RFBLOCK"):
        rs = [r for r in res["rows"] if r["surface"] == tag]
        mad = np.array([r["mad"] for r in rs])
        rms = np.array([r["rms"] for r in rs])
        print(f"\n{tag}: MAD range {mad.min():.3f}-{mad.max():.3f} deg, "
              f"RMS range {rms.min():.3f}-{rms.max():.3f} deg, "
              f"RMS/MAD {(rms/mad).min():.3f}-{(rms/mad).max():.3f} "
              f"(Gaussian would be 1.253)")
        res.setdefault("summary", {})[tag] = dict(
            mad_lo=float(mad.min()), mad_hi=float(mad.max()),
            rms_lo=float(rms.min()), rms_hi=float(rms.max()),
            ratio_lo=float((rms / mad).min()), ratio_hi=float((rms / mad).max()))

    # --- consequence for the power calibration
    K = 0.0101
    print("\n=== power_calibration.py ceiling, Delta = 0.0101 * A^2 deg ===")
    for tag, lo_, hi_ in (("as published (MAD mislabelled as rms)", 1.0, 1.9),
                          *[(f"corrected rms, {t}",
                             res["summary"][t]["rms_lo"], res["summary"][t]["rms_hi"])
                            for t in ("MEASURED", "ARM_LUT62", "RFBLOCK")]):
        print(f"  {tag:<42} A = {lo_:.3f}-{hi_:.3f} deg  ->  "
              f"Delta = {K*lo_**2:+.4f} to {K*hi_**2:+.4f} deg   "
              f"(vs fold-seed sd 0.033: "
              f"{'BELOW' if K*hi_**2 < 0.033 else 'EXCEEDS'})")
        res.setdefault("ceiling", {})[tag] = dict(
            A_lo=lo_, A_hi=hi_, delta_lo=K * lo_ ** 2, delta_hi=K * hi_ ** 2)

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/tmp/claude-1000/-home-mouse9911-gits-spf/"
         "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/audit2/true_amplitude.json")
