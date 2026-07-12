#!/usr/bin/env python3
"""
Build the full dataset-quality PDF report from a fleet-scan metrics.csv
(produced by spf/scripts/dataset_quality_scan.py).

Sections: metric documentation -> fleet quality bins -> per-issue deep dives
(text + figures) -> error table -> per-file quality appendix (all datasets).

Read-only w.r.t. datasets; writes only the PDF.
Run:  python spf/scripts/dataset_quality_report_pdf.py \
        --csv data_quality_reports/scan_2026_07_12/metrics.csv \
        --out data_quality_reports/scan_2026_07_12/quality_report.pdf
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PW, PH = 8.27, 11.69  # A4 portrait inches
INK, ACC, MUT = "#1a1a1a", "#1f5fa8", "#666666"
L, R, T, B = 0.07, 0.93, 0.945, 0.05
C = 299792458.0


class Rep:
    def __init__(self, path):
        self.pdf = PdfPages(path)
        self.pageno = 0
        self.fig = None
        self.toc = []

    def new_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.022, f"SPF dataset quality report · page {self.pageno}",
                          ha="center", fontsize=7.5, color=MUT)
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PW, PH))
        self.pageno += 1
        return self.fig

    def h1(self, title, y=T):
        self.toc.append((title, self.pageno))
        self.fig.text(L, y, title, fontsize=15, fontweight="bold", color=ACC, va="top")
        self.fig.add_artist(plt.Line2D([L, R], [y - 0.022, y - 0.022], color=ACC, lw=0.7, alpha=0.5))
        return y - 0.042

    def text(self, y, lines, size=9.3, dy=0.0158, color=INK, x=L, mono=False):
        fam = "DejaVu Sans Mono" if mono else None
        for ln in lines:
            self.fig.text(x, y, ln, fontsize=size, va="top", color=color, family=fam)
            y -= dy
        return y

    def para(self, y, txt, size=9.3, wrap=104, gap=0.008):
        words, line, lines = txt.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > wrap:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        return self.text(y, lines, size=size) - gap

    def ax(self, rect):
        return self.fig.add_axes(rect)

    def close(self):
        # flush the last real page's footer without creating a phantom page
        if self.fig is not None:
            self.fig.text(0.5, 0.022, f"SPF dataset quality report · page {self.pageno}",
                          ha="center", fontsize=7.5, color=MUT)
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None
        self.pdf.close()


# ---------------- data ----------------

def _trunc(s, n):
    """Truncate at the last ';' boundary before n and mark with an ellipsis."""
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    if ";" in cut:
        cut = cut[: cut.rindex(";") + 1]
    return cut + "\u2026"

def load(csv_fn):
    rows = list(csv.DictReader(open(csv_fn)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("dataset", "platform", "device", "status", "reasons", "frozen_tail",
                     "r0_g_at_bound", "r1_g_at_bound"):
                continue
            try:
                r[k] = float(v) if v not in (None, "", "nan") else np.nan
            except (ValueError, TypeError):
                pass
    return rows


def g(r, k, default=np.nan):
    v = r.get(k, default)
    return v if isinstance(v, float) else default


def month_of(r):
    m = re.search(r"wallarrayv3_(\d{4}_\d{2})", r["dataset"])
    return m.group(1) if m else None


# ---------------- sections ----------------
def sec_title(rep, rows):
    f = rep.new_page()
    f.text(0.5, 0.80, "SPF Dataset Quality Report", ha="center", fontsize=25,
           fontweight="bold", color=ACC)
    f.text(0.5, 0.745, "Fleet scan: metrics, quality bins, issue deep-dives, per-file appendix",
           ha="center", fontsize=12.5)
    n = len(rows)
    cnt = Counter(r["status"] for r in rows)
    f.text(0.5, 0.66, f"{n} datasets scanned (scan of 2026-07-12, scanner v2 gates)", ha="center", fontsize=11)
    f.text(0.5, 0.625, f"OK {cnt.get('OK',0)}   ·   FLAG {cnt.get('FLAG',0)}   ·   "
           f"QUARANTINE {cnt.get('QUARANTINE',0)}   ·   ERROR {cnt.get('ERROR',0)}",
           ha="center", fontsize=12, color=INK)
    f.text(0.5, 0.53, "Scanner: spf/scripts/dataset_quality_scan.py  (read-only)\n"
           "Method & plan: claude_docs/03_datasets/data_quality_plan.md\n"
           "Raw output: data_quality_reports/scan_2026_07_12/metrics.csv",
           ha="center", fontsize=9.5, color=MUT)
    f.text(0.5, 0.40, "Contents", ha="center", fontsize=12, fontweight="bold")
    toc = ["1   Metrics: what we measure and why",
           "2   Fleet overview: quality bins",
           "3   Wall-array breakdown (by device × spacing group)",
           "4   Rover breakdown (by era)",
           "5   Deep dive — wall effective-spacing (gain) systematic",
           "5b  Per-band g vs spacing — coupling-model fits (2 pp)",
           "6   Deep dive — NaN / no-signal snapshots",
           "7   Deep dive — the bad-capture months",
           "8   Deep dive — rover heading bias & mount anomalies",
           "9   Deep dive — stale positions (#42)",
           "10  Deep dive — timestamp monotonicity",
           "11  Deep dive — correction payoff (raw vs corrected)",
           "12  Deep dive — scan errors (integrity failures)",
           "13  Scanner improvements — v2/v3 roadmap",
           "14  Recommended actions",
           "Appendix — per-file quality and issues (all datasets)"]
    y = 0.375
    for t in toc:
        f.text(0.30, y, t, fontsize=9.5)
        y -= 0.019


def sec_metrics(rep):
    f = rep.new_page()
    y = rep.h1("1  Metrics: what we measure and why")
    y = rep.para(y, "Every metric derives from one idea: ground truth (tx/rx positions) predicts "
                 "the physical observable, so the wrapped residual between measured and predicted "
                 "phase difference exposes both noise and systematics. The forward direction is "
                 "immune to front/back ambiguity and d/λ>0.5 aliasing.")
    y = rep.text(y, [r"$\phi_{pred} = -2\pi\,(d/\lambda)\,\sin(\theta_{gt}-\theta_{mount})$"
                     r"$\qquad \delta\phi = \mathrm{pi\_norm}(\phi_{meas}-\phi_{pred})$"],
                 size=11.5, x=0.18) - 0.012
    y = rep.para(y, "phi_meas is the cached per-snapshot mean_phase from segmentation v3.7. All "
                 "residual statistics are circular (wrap at ±π). Fits are per dataset × "
                 "receiver; rover fits are distance-weighted (bearing noise ∝ 1/range).")
    rows_t1 = [
        ("TIER 1 — validity gates", ""),
        ("  M1 nan_frac (r0/r1)", "fraction of snapshots with no valid mean_phase (no signal windows)"),
        ("  M3 frozen_max_frac / frozen_tail", "longest run of identical tx+rx positions; tail = #42 signature"),
        ("  M3b rx_speed_p99", "physical speed plausibility from positions + timestamps"),
        ("  M4 ts_nonmono_frac / ts_med_dt", "timestamp order violations; median cadence"),
        ("TIER 2 — forward-model residual", ""),
        ("  M5 bias", "circular mean of residual: phase offset / heading bias"),
        ("  M6 circstd_raw / circstd_corr", "circular stddev before / after the fitted correction"),
        ("  M7 outlier_frac", "fraction with |residual-bias| > 1 rad (multipath bursts, GT glitches)"),
        ("  M8 g (gain)", "fitted scale on phi_pred: effective d/lambda = g × configured"),
        ("  M9 dtheta", "fitted mount-angle shift; rover: common part = heading bias"),
        ("  M11 drift_span", "spread of residual mean across 4 time-quarters (thermal drift)"),
        ("  M12 structure", "dispersion of residual means across 12 theta bins (multipath/geometry)"),
        ("FIT MODEL (≤3 params/receiver)", ""),
        ("  phi ~ c + g·(-2π d/λ)·sin(θ-Δθ)", "wall g∈[0.7,3.0], Δθ∈±0.35; rover g∈[0.9,1.1] (diagnostic), Δθ∈±0.9"),
    ]
    for a, b in rows_t1:
        bold = b == ""
        rep.fig.text(L, y, a, fontsize=8.6, va="top",
                     fontweight="bold" if bold else "normal", family="DejaVu Sans Mono")
        rep.fig.text(0.40, y, b, fontsize=8.6, va="top")
        y -= 0.0165
    y -= 0.008
    y = rep.para(y, "GATES (v2 thresholds; platform-branched). WALL: QUARANTINE if nan>20% or "
                 "frozen_tail; FLAG if nan 5-20%, |g-1|>0.25, corrected circstd>0.8, ts "
                 "non-monotonic >1% of pairs, or fit at grid bound. ROVER: QUARANTINE if nan>90% "
                 "or <100 valid snapshots; FLAG if |heading|>0.25, corrected circstd>0.7, ts>1%, "
                 "or fit at bound. INFO:low_coverage marks fits skipped by the identifiability "
                 "guard (does not change the bin).")
    y = rep.para(y, "SCANNER v2 (this scan): compared to v1, the wall g grid is widened to "
                 "[0.7, 3.0] (the cfg-0.20λ group no longer pins at a bound), the rover Δθ grid "
                 "to ±0.9 rad, the wall NaN gate is two-tier, the ts gate fires only above 1% "
                 "of pairs, fits are skipped under narrow angular coverage (offset-only + "
                 "INFO:low_coverage), errors are re-checked serially, and frozen-tail datasets "
                 "carry a truncation index for salvage. Section 13 lists the remaining v3 "
                 "roadmap.", size=8.8)


def sec_bins(rep, rows):
    f = rep.new_page()
    y = rep.h1("2  Fleet overview: quality bins")
    wall = [r for r in rows if r.get("platform") == "wall"]
    rover = [r for r in rows if r.get("platform") == "rover"]
    y = rep.para(y, f"2,250 datasets: {len(wall)} wall-array, {len(rover)} rover, "
                 f"{len(rows)-len(wall)-len(rover)} unclassifiable (scan errors). Quality bins "
                 "are assigned by the platform-specific gates of section 1. The bins mean: OK = "
                 "passes all gates; FLAG = usable but carries a measurable systematic or minor "
                 "defect (most FLAGs are the correctable gain systematic, section 5); "
                 "QUARANTINE = fails a validity gate (do not train on as-is); ERROR = could not "
                 "be read (integrity failure, section 12).")
    ax1 = rep.ax([0.08, 0.585, 0.40, 0.20])
    stat_order = ["OK", "FLAG", "QUARANTINE", "ERROR"]
    wc = Counter(r["status"] for r in wall)
    rc = Counter(r["status"] for r in rover)
    x = np.arange(4)
    ax1.bar(x - 0.18, [wc.get(s, 0) for s in stat_order], 0.36, label="wall", color=ACC)
    ax1.bar(x + 0.18, [rc.get(s, 0) for s in stat_order], 0.36, label="rover", color="#c0392b")
    ax1.set_xticks(x, stat_order, fontsize=8)
    ax1.set_ylabel("datasets", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_title("Quality bins by platform", fontsize=9)
    ax1.tick_params(labelsize=8)
    # reason histogram
    ax2 = rep.ax([0.56, 0.585, 0.38, 0.20])
    cr = Counter()
    for r in rows:
        for reason in (r.get("reasons") or "").split(";"):
            if reason:
                cr[re.sub(r"=[-\d.]+", "", reason).replace("r0_", "").replace("r1_", "")] += 1
    ks = [k for k, _ in cr.most_common(8)]
    ax2.barh(range(len(ks))[::-1], [cr[k] for k in ks], color=ACC)
    ax2.set_yticks(range(len(ks))[::-1], [k.replace("FLAG:", "F:").replace("QUAR:", "Q:") for k in ks], fontsize=7.5)
    ax2.set_title("Reason mentions (a dataset can have several)", fontsize=9)
    ax2.tick_params(labelsize=8)
    y = 0.545
    wall_ = [r for r in rows if r.get("platform") == "wall"]
    n_gain = sum(1 for r in wall_ if "gain" in (r.get("reasons") or ""))
    n_nanq = sum(1 for r in wall_ if "nan>20" in (r.get("reasons") or ""))
    n_nanf = sum(1 for r in wall_ if "nan5-20" in (r.get("reasons") or ""))
    n_noisy = sum(1 for r in rows if "noisy" in (r.get("reasons") or ""))
    n_frozen = sum(1 for r in rows if "frozen" in (r.get("reasons") or ""))
    n_err = sum(1 for r in rows if r.get("status") == "ERROR")
    cnt2 = Counter(r["status"] for r in rows)
    y = rep.para(y, f"DECOMPOSITION OF THE NON-OK BINS (v2 gates): the {cnt2.get('FLAG',0)} FLAG "
                 f"+ {cnt2.get('QUARANTINE',0)} QUARANTINE + {n_err} ERROR are six distinct "
                 "populations, most NOT bad data:")
    for t in [
        f"{n_gain} wall FLAG:gain — real, tight, CORRECTABLE spacing systematic (keep + sidecar). [sec 5]",
        f"{n_nanq} wall QUAR:nan>20% — month-clustered bad-capture era (Nov-24, Oct-24, Feb-25). [sec 7]",
        f"{n_nanf} wall FLAG:nan5-20% — borderline duty-cycle tier (v2 two-tier gate). [sec 6]",
        f"{n_noisy} noisy-valid-part mentions — largely the same bad months. [sec 7]",
        f"{n_frozen} QUAR:frozen_tail — real #42 casualties; truncation index now recorded. [sec 9]",
        f"{n_err} ERROR after serial retry — genuine integrity failures. [sec 12]",
    ]:
        rep.fig.text(L + 0.01, y, "• " + t, fontsize=8.8, va="top")
        y -= 0.0165
    y -= 0.01
    # month heatmap-ish bar
    ax3 = rep.ax([0.08, 0.12, 0.85, 0.20])
    months = sorted({month_of(r) for r in wall if month_of(r)})
    tot = Counter(month_of(r) for r in wall)
    bad = Counter(month_of(r) for r in wall if r["status"] in ("QUARANTINE",))
    flg = Counter(month_of(r) for r in wall if r["status"] == "FLAG")
    xm = np.arange(len(months))
    ax3.bar(xm, [tot[m] for m in months], color="#dddddd", label="all")
    ax3.bar(xm, [flg[m] for m in months], color="#f0ad4e", label="FLAG")
    ax3.bar(xm, [bad[m] for m in months], bottom=[flg[m] for m in months],
            color="#c0392b", label="QUARANTINE")
    ax3.set_xticks(xm, months, rotation=45, fontsize=7.5)
    ax3.legend(fontsize=8)
    ax3.set_title("Wall-array datasets by collection month: totals vs FLAG/QUARANTINE "
                  "(the quality story is era-structured, not random)", fontsize=9)
    ax3.tick_params(labelsize=8)


def sec_dive_gain(rep, rows):
    wall = [r for r in rows if r.get("platform") == "wall"]
    rep.new_page()
    y = rep.h1("5  Deep dive — wall effective-spacing (gain) systematic")
    y = rep.para(y, "THE FINDING. Fitting a gain g on the physics term shows the effective "
                 "antenna spacing disagrees with the configured spacing for a structured subset "
                 "of the wall fleet. This is not fit noise: the two independent receivers agree "
                 "(corr 0.64, median |Δg| = 0.08, fig. A2) and per-config groups are razor-tight "
                 "(e.g. cfg 0.32λ → IQR 1.56-1.66).")
    y = rep.para(y, "IN METERS (fig. A1): at 2.4 GHz every config below ~6 cm lands at an "
                 "effective 6.2-7.3 cm (configs ≥6 cm read correct) — a physical FLOOR, "
                 "consistent with antenna body width. At 5.8 GHz configs ≥4.3 cm are correct and "
                 "the 2.5 cm config floors at ~3.3 cm. At 915 MHz the effect INVERTS: large "
                 "configs (7-7.5 cm) read 1.3-1.6× larger — a floor cannot do that, pointing at "
                 "mutual coupling (or actually-larger mounting) for that band.")
    y = rep.para(y, "CONSEQUENCE: rx_spacing_input to the NN and the spacing keys of the "
                 "empirical tables are wrong for the sub-floor config groups. REMEDY: these "
                 "datasets are healthy — apply the fitted effective d/λ as a 1-parameter "
                 "sidecar; do NOT exclude. Root-cause (floor vs coupling) needs a physical rig "
                 "check; both hypotheses predict floors that scale with λ (~0.5-0.6 λ), as observed.")
    # fig A1: eff vs cfg meters by band
    ax = rep.ax([0.09, 0.30, 0.52, 0.30])
    bands = {0.9: ("#8e44ad", "915 MHz"), 2.4: (ACC, "2.4 GHz"), 5.8: ("#c0392b", "5.8 GHz")}
    by = defaultdict(list)
    for r in wall:
        lo, ws = g(r, "rx_lo"), g(r, "wavelength_spacing")
        if not (np.isfinite(lo) and lo > 1e8 and np.isfinite(ws)):
            continue
        for k in ("r0_g", "r1_g"):
            gv = g(r, k)
            if np.isfinite(gv) and 0.72 < gv < 2.98:
                band = min(bands, key=lambda b: abs(lo / 1e9 - b))
                by[(band, round(ws * C / lo, 4))].append(gv * ws * C / lo)
    for (band, cfg), v in sorted(by.items()):
        if len(v) < 20:
            continue
        col, _ = bands[band]
        ax.scatter([cfg * 100], [np.median(v) * 100], s=18 + len(v) / 12, color=col, zorder=3)
    ax.plot([1, 13], [1, 13], ls="--", color="#999", lw=1, label="effective = configured")
    for band, (col, lab) in bands.items():
        ax.scatter([], [], color=col, label=lab)
    ax.axhspan(6.2, 7.3, alpha=0.10, color=ACC)
    ax.text(1.4, 6.55, "2.4 GHz floor", fontsize=7, color=ACC)
    ax.set_xlabel("configured spacing (cm)", fontsize=8.5)
    ax.set_ylabel("median effective spacing (cm)", fontsize=8.5)
    ax.set_title("Fig. A1 — effective vs configured antenna spacing\n(marker size = # fits; bound-hitting fits excluded)", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=8)
    # fig A2: r0 vs r1 g
    ax2 = rep.ax([0.68, 0.30, 0.27, 0.30])
    g0 = np.array([g(r, "r0_g") for r in wall])
    g1 = np.array([g(r, "r1_g") for r in wall])
    m = np.isfinite(g0) & np.isfinite(g1)
    ax2.scatter(g0[m], g1[m], s=3, alpha=0.25, color=ACC)
    ax2.plot([0.7, 3], [0.7, 3], ls="--", color="#999", lw=1)
    ax2.set_xlabel("g (receiver 0)", fontsize=8.5)
    ax2.set_ylabel("g (receiver 1)", fontsize=8.5)
    ax2.set_title("Fig. A2 — independent receivers\nagree on g (corr 0.64)", fontsize=8.5)
    ax2.tick_params(labelsize=8)
    # bottom: g hist per selected cfg groups
    ax3 = rep.ax([0.09, 0.085, 0.86, 0.15])
    for sp, col in [("0.32182", ACC), ("0.56319", "#2e8b57"), ("0.28159", "#c0392b"), ("0.48273", "#f0ad4e")]:
        v = [g(r, "r0_g") for r in wall if r.get("wavelength_spacing") == float(sp)]
        v = [x for x in v if np.isfinite(x)]
        if v:
            ax3.hist(v, bins=np.arange(0.7, 3.05, 0.04), alpha=0.55, label=f"cfg d/λ={sp}", color=col)
    ax3.axvline(1.0, color="k", lw=1, ls=":")
    ax3.legend(fontsize=7.5)
    ax3.set_xlabel("fitted gain g (receiver 0)", fontsize=8.5)
    ax3.set_title("Fig. A3 — g distributions per config group: tight, group-specific, far from 1 for sub-floor configs", fontsize=8.5)
    ax3.tick_params(labelsize=8)


def _coupling_model_g(d, lam, A, psi0):
    """Phase-swing gain of a coupled 2-element array: V0=1+Ce^{jp}, V1=e^{jp}+C,
    C(d)=A e^{j(psi0-kd)}/(kd). Returns d(arg V1V0*)/dp at broadside."""
    k = 2 * np.pi / lam
    Cc = A * np.exp(1j * (psi0 - k * np.asarray(d))) / (k * np.asarray(d))
    eps = 1e-4
    v0 = 1 + Cc * np.exp(1j * eps)
    v1 = np.exp(1j * eps) + Cc
    return np.angle(v1 * np.conj(v0)) / eps


def sec_dive_gain_model(rep, rows):
    from scipy.optimize import least_squares

    wall = [r for r in rows if r.get("platform") == "wall"]
    bands = [
        ("2.412 GHz", 2.410e9, 2.414e9),
        ("2.464-2.467 GHz", 2.462e9, 2.469e9),
        ("5.77-5.84 GHz", 5.70e9, 5.90e9),
        ("868 + 915 MHz", 0.85e9, 0.93e9),
    ]

    rep.new_page()
    y = rep.h1("5b  Per-band g vs configured spacing — coupling-model fits")
    y = rep.para(y, "Every wall dataset as a dot (mean of r0/r1 fitted g; bound-hitting fits "
                 "excluded), per band. Overlaid: g=1 (config correct), the naive pin "
                 "g=(λ/2)/d (effective spacing stuck at half-wavelength), and a 2-parameter "
                 "MUTUAL-COUPLING model. Model: each channel hears its own antenna plus a "
                 "coupled copy of its neighbour, V0=1+Ce^{jφ}, V1=e^{jφ}+C, with "
                 "C(d)=A·e^{j(ψ0−kd)}/(kd) — the leading far-term of the classical mutual "
                 "impedance between parallel elements (C≈Z21/(Z11+ZL)). Measured phase is "
                 "arg(V1V0*); its swing vs the ideal gives closed-form "
                 "g=(1−|C|²)/(1+2ReC+|C|²). One (A, ψ0) pair fitted per band on config "
                 "medians, weighted by dataset count.")

    rects = [[0.09, 0.42, 0.38, 0.23], [0.58, 0.42, 0.38, 0.23],
             [0.09, 0.09, 0.38, 0.23], [0.58, 0.09, 0.38, 0.23]]
    fit_results = []
    rng = np.random.default_rng(0)
    for (name, flo, fhi), rect in zip(bands, rects):
        ax = rep.ax(rect)
        pts_d, pts_g, ag0, ag1 = [], [], [], []
        for r in wall:
            lo, ws = g(r, "rx_lo"), g(r, "wavelength_spacing")
            if not (np.isfinite(lo) and flo <= lo <= fhi and np.isfinite(ws)):
                continue
            gs = [g(r, k) for k in ("r0_g", "r1_g")]
            gs = [x for x in gs if np.isfinite(x) and 0.72 < x < 2.98]
            if len(gs) == 2:
                ag0.append(gs[0])
                ag1.append(gs[1])
            if gs:
                pts_d.append(ws * C / lo)
                pts_g.append(float(np.mean(gs)))
        pts_d, pts_g = np.array(pts_d), np.array(pts_g)
        rx_corr = float(np.corrcoef(ag0, ag1)[0, 1]) if len(ag0) > 10 else np.nan
        if len(pts_d) < 10:
            continue
        lam = C / ((flo + fhi) / 2)
        jit = 1 + rng.normal(0, 0.012, len(pts_d))
        ax.scatter(pts_d * jit * 100, pts_g, s=5, alpha=0.22, color=ACC, edgecolors="none")
        cfgs = sorted(set(np.round(pts_d, 4)))
        med_d = np.array(cfgs)
        med_g = np.array([np.median(pts_g[np.round(pts_d, 4) == c]) for c in cfgs])
        med_n = np.array([int((np.round(pts_d, 4) == c).sum()) for c in cfgs])
        ax.scatter(med_d * 100, med_g, s=55, marker="s", facecolor="none",
                   edgecolor="k", linewidth=1.2, zorder=5, label="config median")
        dd = np.linspace(med_d.min() * 0.7, max(med_d.max() * 1.25, lam / 2 * 1.05), 250)
        ax.plot(dd * 100, np.ones_like(dd), color="#999", lw=0.8, ls=":")
        ax.plot(dd * 100, (lam / 2) / dd, color="#2e8b57", lw=1.1, ls="--",
                label="g=(λ/2)/d")
        if len(med_d) >= 2:
            def resid(p):
                return (_coupling_model_g(med_d, lam, p[0], p[1]) - med_g) * np.sqrt(med_n)
            best = None
            for p0 in np.linspace(-np.pi, np.pi, 13):
                for A0 in (0.2, 0.5, 1.0, 2.0):
                    try:
                        rr = least_squares(resid, [A0, p0],
                                           bounds=([0, -2 * np.pi], [10, 2 * np.pi]))
                    except ValueError:
                        continue
                    if best is None or rr.cost < best.cost:
                        best = rr
            A, psi0 = best.x
            psi0 = (psi0 + np.pi) % (2 * np.pi) - np.pi
            rmse = float(np.sqrt(np.average(
                (_coupling_model_g(med_d, lam, A, psi0) - med_g) ** 2, weights=med_n)))
            trusted = np.isfinite(rx_corr) and rx_corr > 0.5
            if trusted:
                ax.plot(dd * 100, _coupling_model_g(dd, lam, A, psi0), color="#c0392b",
                        lw=1.6, label=f"coupling A={A:.2f} ψ0={psi0:+.2f}")
            else:
                ax.plot(dd * 100, _coupling_model_g(dd, lam, A, psi0), color="#999",
                        lw=1.2, ls="--", label="coupling fit — NOT trusted here")
            fit_results.append((name, A, psi0, rmse, int(med_n.sum()), len(med_d), rx_corr))
        ax.axvline(lam / 2 * 100, color="#2e8b57", lw=0.7, alpha=0.5)
        ax.set_title(f"{name}  (λ={lam*100:.1f} cm)   rx0/rx1 g-agreement ρ={rx_corr:+.2f}",
                     fontsize=8)
        ax.set_xlabel("configured spacing (cm)", fontsize=8)
        ax.set_ylabel("fitted g", fontsize=8)
        ax.set_ylim(0.6, max(2.6, med_g.max() + 0.4))
        ax.legend(fontsize=6.2, loc="upper right")
        ax.tick_params(labelsize=7.5)

    # ---- second page: is the fit reasonable? ----
    rep.new_page()
    y = rep.h1("5b (cont.)  Is the coupling fit reasonable?")
    y = rep.para(y, "FIT QUALITY (weighted rmse on config medians, 2 params/band; ρ = "
                 "agreement of the two independent receivers on per-dataset g): " +
                 ";  ".join(f"{n}: A={A:.2f}, ψ0={p:+.2f}, rmse={e:.3f}, ρ={c:+.2f}"
                            for n, A, p, e, _, k, c in fit_results) + ".")
    y = rep.para(y, "WHERE THE FIT IS NOT TRUSTWORTHY — 868/915 MHz. Per-dataset g is not a "
                 "reproducible measurement in this band: the two independent receivers of the "
                 "same rig agree at ρ=+0.97 (2.412 GHz), +0.91 (2.464), +0.85 (5.8) — but "
                 "ρ≈ 0.0 at sub-GHz (median |g_r0−g_r1| = 0.58). Monte-Carlo shows plain "
                 "Gaussian phase noise cannot do this (unbiased, sd≤0.1); the culprit is "
                 "STRUCTURED SLOW DRIFT: sub-GHz residuals carry 2× the drift span (0.36 vs "
                 "0.19 rad) at junk-level corrected circstd (1.0), and simulated slow drift "
                 "aliases into the fitted amplitude with sd≈0.47 when the geometric swing is "
                 "only ±0.9-1.4 rad (d/λ=0.12-0.23) — matching the observed per-receiver "
                 "scatter — while the same drift at d/λ=0.4 gives sd 0.28. Each receiver "
                 "pair drifts independently, hence zero correlation. All sub-GHz data also "
                 "comes from the Oct-2024–Jan-2025 degraded-hardware era (RX1 DC-offset / "
                 "IF≈0 issues, weak signal). Config medians (n=51-156, SE≈0.05) still sit "
                 "significantly above 1, so some expansion is probably real, but they may be "
                 "drift-biased — the sub-GHz curve is drawn dashed-grey and excluded from any "
                 "sidecar until re-measured in a healthy era.")
    y = rep.para(y, "WHY IT IS REASONABLE. (1) The functional form is not an ad-hoc curve: "
                 "C≈Z21/(Z11+ZL) with Z21 decaying as 1/(kd) and phase −kd is the leading "
                 "far-term of the textbook mutual impedance between parallel elements. "
                 "(2) Fitted coupling amplitudes are physical: A=0.2-0.5 keeps |C|<0.5 at "
                 "every measured spacing — coupling weaker than the direct path, as it must "
                 "be. (3) Cross-band consistency: A stays the same order of magnitude across "
                 "a 6.4× frequency range, and the distortion is largest exactly where kd is "
                 "smallest — the defining signature of coupling. (4) It predicts structure "
                 "the naive λ/2-pin cannot: the ±8% oscillation of g around 1 at 2.4 GHz "
                 "for 6-8 cm configs (Re C changes sign with kd); the moderate sub-λ/2 "
                 "expansion at 868/915 MHz (g=1.2-1.6 where pinning would demand 2.3-4); and "
                 "the near-λ/2 floors at 2.4/5.8 GHz. (5) g is a rig property (independent "
                 "receivers agree, corr 0.64), and coupling is a rig-level EM mechanism — "
                 "the right explanation class. (6) No overfitting: 2 parameters describe up "
                 "to 9 config medians per band.")
    y = rep.para(y, "WHY TO STAY CAUTIOUS. (1) The model computes the broadside phase-swing "
                 "slope; the scanner fits a full sine over the sweep — for the strongest "
                 "coupling (2.5 cm at 2.4 GHz) these differ, so read A and ψ0 as effective, "
                 "not literal. (2) A and ψ0 are lumped constants absorbing antenna type, "
                 "load impedance, mounts and ground plane; each band uses different physical "
                 "antennas, so per-band parameters cannot be checked against geometry "
                 "without a bench measurement. (3) Identifiability is thin where configs are "
                 "few: 5.8 GHz has effectively two informative spacings and 868 MHz only "
                 "one, so ψ0 there is weakly constrained. (4) Competing mechanisms — phase-"
                 "center displacement on the shared mount, near-field scattering off the "
                 "plate — produce similar g(d) shapes at small kd; this fit cannot exclude "
                 "them. (5) The 2.412 GHz rmse (0.12) is dominated by under-predicting the "
                 "6-8 cm wiggle: real Z21 has 1/(kd)² and 1/(kd)³ near-field terms we "
                 "dropped. (6) Observational data: config groups correlate with collection "
                 "eras, though within-config stability across eras argues against a "
                 "temporal artifact.")
    rep.new_page()
    y = rep.h1("5b (cont.)  Verdict and the universal view")
    y = rep.para(y, "VERDICT & USE. Treat the coupling fit as a physically-motivated "
                 "2-parameter summary and interpolator — good enough to generate an "
                 "effective-spacing sidecar g(d, band) for correcting rx_spacing_input, and "
                 "good enough to predict g for a NEW spacing config before collecting it. "
                 "Do not treat it as validated first-principles EM: the decisive test is a "
                 "bench VNA S21-vs-distance sweep on the actual mounts (one afternoon), or "
                 "a controlled spacing sweep at fixed era. If either matches the fitted "
                 "C(d), the sidecar can be trusted fleet-wide, including for spacings never "
                 "collected.")
    # universal d/lambda panel at bottom
    ax = rep.ax([0.09, 0.075, 0.83, 0.24])
    xs, ys, cs = [], [], []
    for r in wall:
        lo, ws = g(r, "rx_lo"), g(r, "wavelength_spacing")
        if not (np.isfinite(lo) and lo > 1e8 and np.isfinite(ws)):
            continue
        gs = [g(r, k) for k in ("r0_g", "r1_g")]
        gs = [x for x in gs if np.isfinite(x) and 0.72 < x < 2.98]
        if gs:
            xs.append(ws)
            ys.append(float(np.mean(gs)))
            cs.append(np.log10(lo))
    sc = ax.scatter(xs, ys, s=6, alpha=0.3, c=cs, cmap="viridis", edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("log10 carrier (Hz)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    dl = np.linspace(0.05, 1.7, 200)
    ax.plot(dl, np.ones_like(dl), color="#999", lw=0.8, ls=":")
    ax.plot(dl, 0.5 / dl, color="#2e8b57", lw=1.1, ls="--", label="g=0.5/(d/λ)")
    ax.axvline(0.5, color="#2e8b57", lw=0.7, alpha=0.5)
    ax.set_xlabel("configured spacing d/λ", fontsize=8.5)
    ax.set_ylabel("fitted g", fontsize=8.5)
    ax.set_ylim(0.5, 3.0)
    ax.set_title("Fig. B4 — all wall datasets, universal axis: distortion is a function of d/λ, onset below ≈0.5",
                 fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=8)


def sec_dive_badmonths(rep, rows):
    wall = [r for r in rows if r.get("platform") == "wall"]
    rep.new_page()
    y = rep.h1("7  Deep dive — the bad-capture months")
    qq = [r for r in wall if "nan" in (r.get("reasons") or "")]
    nq20 = sum(1 for r in qq if "nan>20" in r.get("reasons", ""))
    y = rep.para(y, f"THE FINDING. {len(qq)} wall datasets exceed 5% NaN mean_phase ({nq20} "
                 f"quarantined at >20%, {len(qq)-nq20} flagged at 5-20% under the v2 two-tier "
                 "gate; what NaN means: section 6). They are NOT randomly spread: they cluster "
                 "hard by collection month — Nov 2024: 100% of the month's datasets, Oct 2024: "
                 "~47%, Feb 2025: ~35%, against 0/855 for Jun-Sep 2024. Crucially the VALID "
                 "part of these datasets is also degraded (median corrected circular stddev "
                 "≈0.96 rad vs fleet ≈0.57, fig. B2) — a capture-quality era, not merely a "
                 "bursty emitter.")
    y = rep.para(y, "INTERPRETATION: a genuine capture-quality era — something about the rig, "
                 "emitter, or RF environment degraded in Oct-Nov 2024 (partially recovered "
                 "Dec, relapsed Feb 2025). This is not a segmentation-threshold artifact and not "
                 "related to the spacing mislabeling. RECOMMENDATION: quarantine stands for the "
                 ">20% NaN group; the 142 datasets at 5-20% NaN are borderline — decide after "
                 "checking whether their valid-part circstd is clean; investigate what changed "
                 "at the rig in those months (emitter power? antenna cable? gain settings?).")
    q = [r for r in wall if "nan" in (r.get("reasons") or "")]
    ax1 = rep.ax([0.09, 0.36, 0.38, 0.22])
    months_all = sorted({month_of(r) for r in wall if month_of(r)})
    med_by_m = []
    for m_ in months_all:
        v = [g(r, "r0_circstd_corr") for r in wall if month_of(r) == m_]
        v = [x for x in v if np.isfinite(x)]
        med_by_m.append(np.median(v) if v else np.nan)
    ax1.bar(np.arange(len(months_all)), med_by_m, color=ACC, alpha=0.85)
    ax1.set_xticks(np.arange(len(months_all)), months_all, rotation=55, fontsize=6.5)
    ax1.set_ylabel("median corrected circstd (rad)", fontsize=8)
    ax1.set_title("Fig. B1 — phase quality by month (ALL wall):\nthe bad months are noisy, not just NaN-heavy", fontsize=8.5)
    ax1.tick_params(labelsize=8)
    ax2 = rep.ax([0.56, 0.36, 0.38, 0.22])
    cs_q = [g(r, "r0_circstd_corr") for r in q]
    cs_ok = [g(r, "r0_circstd_corr") for r in wall if r["status"] == "OK"]
    ax2.hist([x for x in cs_ok if np.isfinite(x)], bins=np.arange(0, 1.6, 0.06), alpha=0.6, label="OK datasets", color=ACC)
    ax2.hist([x for x in cs_q if np.isfinite(x)], bins=np.arange(0, 1.6, 0.06), alpha=0.6, label="NaN-quarantined", color="#c0392b")
    ax2.set_xlabel("corrected circular stddev (rad), valid part, r0", fontsize=8.5)
    ax2.legend(fontsize=8)
    ax2.set_title("Fig. B2 — even their VALID snapshots\nare noisier: a capture-era problem", fontsize=8.5)
    ax2.tick_params(labelsize=8)
    ax3 = rep.ax([0.09, 0.105, 0.85, 0.19])
    months = sorted({month_of(r) for r in wall if month_of(r)})
    tot = Counter(month_of(r) for r in wall)
    qm = Counter(month_of(r) for r in q)
    frac = [100 * qm.get(m, 0) / tot[m] for m in months]
    ax3.bar(np.arange(len(months)), frac, color="#c0392b", alpha=0.85)
    ax3.set_xticks(np.arange(len(months)), months, rotation=45, fontsize=7.5)
    ax3.set_ylabel("% of month quarantined", fontsize=8.5)
    ax3.set_title("Fig. B3 — NaN-quarantine rate by month: Nov 2024 = 100%, Jun-Sep 2024 = 0%", fontsize=8.5)
    ax3.tick_params(labelsize=8)


def sec_dive_rover(rep, rows):
    rover = [r for r in rows if r.get("platform") == "rover"]
    rep.new_page()
    y = rep.h1("8  Deep dive — rover heading bias & mount anomalies")
    y = rep.para(y, "THE FINDING. The common component of the two receivers' fitted Δθ "
                 "isolates a HEADING bias (both arrays shift together only if the craft heading "
                 "is wrong). Dec-Feb missions consistently show −0.14..−0.33 rad; the later "
                 "'rover*'-era missions center near +0.04 (fig. C1). The bias is therefore "
                 "era/installation-dependent and stable within an era — an ideal 1-parameter "
                 "correction, fitted per day or era.")
    y = rep.para(y, "SCANNER CAVEAT + ANOMALIES: 73/139 rover fits hit the Δθ grid bound "
                 "(±0.35) — a v1 scanner limitation. Wide re-fits (±0.9) split these into (a) "
                 "larger-but-plausible heading biases (common −0.28..−0.34) and (b) a few "
                 "sessions with large DIFFERENTIAL mount anomalies — feb7_mission1_rover1 r1 = "
                 "−0.72 rad (≈41°), feb8_mission1_rover1 r1 = +0.54 rad — i.e. one array "
                 "rotated/miswired relative to config. Those specific sessions are genuinely "
                 "suspect and should be excluded or inspected individually.")
    y = rep.para(y, "Also note the rover noise floor: at 9-13 m range, 2-5 m GPS error alone "
                 "predicts 0.15-0.5 rad of bearing noise — the observed post-correction circstd "
                 "(~0.45-0.6) is consistent with GPS-dominated labels, so rover metrics compare "
                 "the residual against this predicted floor rather than against zero.")
    ax = rep.ax([0.09, 0.10, 0.85, 0.34])
    by_day = defaultdict(list)
    for r in rover:
        v = g(r, "heading_common")
        if np.isfinite(v):
            by_day[r["dataset"].split("_")[0]].append(v)
    days = sorted(by_day, key=lambda d: -len(by_day[d]))[:14]
    data = [by_day[d] for d in days]
    bp = ax.boxplot(data, tick_labels=days, showfliers=True, patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("#cfe0f2")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axhline(-0.15, color="#c0392b", lw=0.8, ls="--")
    ax.text(len(days) + 0.45, -0.135, "≈−0.15 rad Dec-Feb era bias",
            fontsize=7.5, color="#c0392b", ha="right")
    ax.set_ylabel("heading_common (rad)", fontsize=8.5)
    ax.set_title("Fig. C1 — common-mode Δθ (heading bias) by mission day/era: stable within era, differs across eras", fontsize=8.5)
    ax.tick_params(labelsize=7.5, axis="x", rotation=30)
    ax.tick_params(labelsize=8, axis="y")


def sec_dive_frozen(rep, rows):
    wall = [r for r in rows if r.get("platform") == "wall"]
    rep.new_page()
    y = rep.h1("9  Deep dive — stale positions (#42 in the wild)")
    y = rep.para(y, "THE FINDING. KNOWN_ISSUES #42: if the GRBL planner thread dies (e.g. an "
                 "out-of-bounds point), data collection keeps recording while the position cache "
                 "freezes — every subsequent snapshot is stamped with the same stale position, "
                 "silently corrupting labels. The scanner detects the signature (a long frozen "
                 "run that reaches the end of the dataset): 21 wall datasets carry it. These "
                 "should be excluded, or truncated at the freeze point (the data before the "
                 "freeze is fine).")
    ft = [r for r in wall if r.get("frozen_tail") == "True"]
    ax = rep.ax([0.09, 0.42, 0.40, 0.22])
    fr = [g(r, "frozen_max_frac") * 100 for r in wall if np.isfinite(g(r, "frozen_max_frac"))]
    ax.hist(fr, bins=np.arange(0, 105, 2.5), color=ACC, log=True)
    ax.axvline(20, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("longest frozen run (% of dataset)", fontsize=8.5)
    ax.set_ylabel("datasets (log)", fontsize=8.5)
    ax.set_title("Fig. D1 — frozen-run length across the wall fleet\n(the 21 tail-frozen sit in the right tail)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    y = 0.40
    rep.fig.text(0.54, 0.68, "The 21 frozen-tail datasets:", fontsize=9, fontweight="bold", va="top")
    yy = 0.66
    for r in ft:
        frac = g(r, "frozen_max_frac")
        rep.fig.text(0.54, yy, f"{r['dataset'][:40]}  ({frac*100:.0f}%)", fontsize=6.8,
                     va="top", family="DejaVu Sans Mono")
        yy -= 0.0125
    rep.para(0.335, "Note the histogram's mass near zero: normal bounce/circle motion updates "
             "positions every snapshot, so any frozen run beyond a few percent is anomalous. "
             "The red line marks the detector threshold; only runs that ALSO reach the final "
             "snapshot are classified as #42 tails (mid-run pauses exist, e.g. planner restarts).")


def sec_dive_ts(rep, rows):
    rep.new_page()
    y = rep.h1("10  Deep dive — timestamp monotonicity")
    nz = [g(r, "ts_nonmono_frac") for r in rows
          if isinstance(r.get("ts_nonmono_frac"), float) and np.isfinite(g(r, "ts_nonmono_frac"))
          and g(r, "ts_nonmono_frac") > 0]
    y = rep.para(y, f"THE FINDING. {len(nz)} datasets contain at least one non-monotonic "
                 "timestamp pair, but the distribution (fig. E1) shows the median affected "
                 "fraction is ~0.02% of pairs (a handful of samples out of hundreds of "
                 "thousands): benign clock jitter, likely NTP adjustments during collection. "
                 f"The v2 gate (applied in this scan) flags only the {sum(1 for x in nz if x > 0.01)} "
                 "datasets above 1% — those have real timestamp problems worth individual "
                 "review (cadence anomalies can corrupt time-interpolated ground truth).")
    ax = rep.ax([0.09, 0.42, 0.55, 0.26])
    v = [g(r, "ts_nonmono_frac") * 100 for r in rows
         if isinstance(r.get("ts_nonmono_frac"), float) and np.isfinite(g(r, "ts_nonmono_frac")) and g(r, "ts_nonmono_frac") > 0]
    ax.hist(v, bins=np.logspace(-4, np.log10(20), 40), color=ACC)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#c0392b", ls="--", lw=1)
    ax.text(0.86, ax.get_ylim()[1] * 0.86, "v2 gate (1%)",
        fontsize=7.5, color="#c0392b", ha="right")
    ax.set_xlabel("% of timestamp pairs out of order (log scale)", fontsize=8.5)
    ax.set_ylabel("datasets", fontsize=8.5)
    ax.set_title("Fig. E1 — timestamp violations: overwhelmingly trivial jitter", fontsize=8.5)
    ax.tick_params(labelsize=8)
    n_real = sum(1 for x in v if x > 1.0)
    rep.para(0.38, f"Datasets above the 1% gate: {n_real}. Everything below it is reported as "
             "info and no longer affects the quality bin.")


def sec_dive_payoff(rep, rows):
    rep.new_page()
    y = rep.h1("11  Deep dive — correction payoff (raw vs corrected)")
    meds = {}
    for plat_ in ("wall", "rover"):
        rs_ = [r for r in rows if r.get("platform") == plat_]
        a = np.array([g(r, "r0_circstd_raw") for r in rs_])
        b = np.array([g(r, "r0_circstd_corr") for r in rs_])
        m_ = np.isfinite(a) & np.isfinite(b)
        meds[plat_] = (np.median(a[m_]), np.median(b[m_]))
    y = rep.para(y, "For every dataset the scanner fits the ≤3-parameter systematic model and "
                 "reports the residual circular stddev before and after. Fleet-wide medians "
                 f"(v2 grids): wall {meds['wall'][0]:.3f} → {meds['wall'][1]:.3f} rad, rover "
                 f"{meds['rover'][0]:.3f} → {meds['rover'][1]:.3f} rad. Fig. F1 shows the "
                 "per-dataset improvement; points far below the diagonal are datasets whose "
                 "apparent noise was mostly a correctable systematic (the gain groups); points "
                 "on the diagonal are already-clean or genuinely noisy (the bad months).")
    for i, plat in enumerate(("wall", "rover")):
        rs = [r for r in rows if r.get("platform") == plat]
        raw = np.array([g(r, "r0_circstd_raw") for r in rs])
        cor = np.array([g(r, "r0_circstd_corr") for r in rs])
        m = np.isfinite(raw) & np.isfinite(cor)
        ax = rep.ax([0.09 + i * 0.47, 0.34, 0.38, 0.32])
        ax.scatter(raw[m], cor[m], s=4, alpha=0.3, color=ACC if plat == "wall" else "#c0392b")
        lim = 1.6
        ax.plot([0, lim], [0, lim], ls="--", color="#999", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("raw circstd (rad)", fontsize=8.5)
        ax.set_ylabel("corrected circstd (rad)", fontsize=8.5)
        ax.set_title(f"Fig. F1{'ab'[i]} — {plat} (n={m.sum()})", fontsize=9)
        ax.tick_params(labelsize=8)
    rep.para(0.27, "Reading: the wall cloud shows a dense band pulled well below the diagonal "
             "(the gain systematic absorbing 30-50% of apparent noise) plus a diagonal "
             "population near (1.0, 1.0) — the bad-month era whose noise is NOT explained by "
             "any low-dimensional systematic. The rover cloud improves more modestly (heading "
             "bias removal), sitting on its GPS-induced floor of ~0.45-0.6 rad.")


def sec_dive_errors(rep, rows):
    rep.new_page()
    y = rep.h1("12  Deep dive — scan errors (integrity failures)")
    y = rep.para(y, "23 datasets could not be scanned at all. Every error is itself a data "
                 "integrity finding, in four groups:")
    groups = [
        ("9×  yaml-vs-zarr mount-angle mismatch", "AssertionError comparing rx_theta_in_pis in the yaml "
         "config against the recorded zarr value — the dataset's mount metadata contradicts itself."),
        ("6×  unreadable/corrupt zarr", "the store fails to open — truncated or damaged LMDB."),
        ("6×  missing segmentation cache", "no v3.7 .yarr/.pkl in the precompute cache — never segmented."),
        ("2×  rx_spacing not constant within dataset", "the per-snapshot spacing field changes mid-dataset "
         "(thousands of mismatches) — collection-time config drift."),
    ]
    for a, b in groups:
        rep.fig.text(L, y, a, fontsize=9.5, fontweight="bold", va="top")
        y -= 0.017
        y = rep.para(y, b, size=8.8)
    y -= 0.005
    rep.fig.text(L, y, "Full list:", fontsize=9.5, fontweight="bold", va="top")
    y -= 0.018
    errs = [r for r in rows if r["status"] == "ERROR"]
    for r in errs:
        rep.fig.text(L, y, f"{r['dataset'][:52]:52s}  {_trunc((r.get('reasons') or ''), 58)}",
                     fontsize=6.6, va="top", family="DejaVu Sans Mono")
        y -= 0.0122


def sec_wall_breakdown(rep, rows):
    wall = [r for r in rows if r.get("platform") == "wall"]
    rep.new_page()
    y = rep.h1("3  Wall-array breakdown (by device × spacing group)")
    dev = Counter(r.get("device", "?") for r in wall)
    y = rep.para(y, f"{len(wall)} wall-array datasets ({dict(dev)}). Ground truth is GRBL "
                 "steps (sub-mm), so residuals here measure the RF chain itself. The fleet is "
                 "organized in (device × configured d/λ) groups — the natural calibration unit: "
                 "gain, mount shift, and quality are tight within a group and differ across "
                 "groups. Table W1 summarizes the largest groups; the stacked bars show where "
                 "the quality bins live.")
    groups = defaultdict(list)
    for r in wall:
        groups[(r.get("device", "?"), r.get("wavelength_spacing"))].append(r)
    top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:14]
    rep.fig.text(L, y, "Table W1 — largest device × d/λ groups", fontsize=9,
                 fontweight="bold", va="top")
    y -= 0.018
    hdr = f"{'device':9s} {'cfg d/λ':>8s} {'n':>4s} {'med g':>6s} {'med corr':>8s} {'med NaN%':>8s} {'OK/FL/QU':>9s}"
    rep.fig.text(L, y, hdr, fontsize=7.2, family="DejaVu Sans Mono", fontweight="bold", va="top")
    y -= 0.0145
    for (d_, sp), rs in top:
        med = lambda k: np.nanmedian([g(x, k) for x in rs])  # noqa: E731
        stat = Counter(x["status"] for x in rs)
        line = (f"{str(d_)[:9]:9s} {sp:8.3f} {len(rs):4d} {med('r0_g'):6.2f} "
                f"{med('r0_circstd_corr'):8.2f} {100*med('r0_nan_frac'):8.0f} "
                f"{stat.get('OK',0):3d}/{stat.get('FLAG',0):3d}/{stat.get('QUARANTINE',0):3d}")
        rep.fig.text(L, y, line, fontsize=7.2, family="DejaVu Sans Mono", va="top")
        y -= 0.0135
    y -= 0.010
    ax = rep.ax([0.09, 0.10, 0.85, max(0.14, y - 0.14)])
    labels, oks, fls, qs = [], [], [], []
    for (d_, sp), rs in top:
        labels.append(f"{str(d_)[0]}·{sp:.3f}")
        st = Counter(x["status"] for x in rs)
        oks.append(st.get("OK", 0)); fls.append(st.get("FLAG", 0)); qs.append(st.get("QUARANTINE", 0))
    xg = np.arange(len(labels))
    ax.bar(xg, oks, color="#2e8b57", label="OK")
    ax.bar(xg, fls, bottom=oks, color="#f0ad4e", label="FLAG")
    ax.bar(xg, qs, bottom=np.array(oks) + np.array(fls), color="#c0392b", label="QUAR")
    ax.set_xticks(xg, labels, rotation=45, fontsize=6.8)
    ax.legend(fontsize=8)
    ax.set_title("Fig. W1 — quality bins per device × d/λ group (P=PLUTO, B=BLADERF2)", fontsize=8.5)
    ax.tick_params(labelsize=8)


def sec_rover_breakdown(rep, rows):
    rover = [r for r in rows if r.get("platform") == "rover"]
    rep.new_page()
    y = rep.h1("4  Rover breakdown (by era)")
    y = rep.para(y, f"{len(rover)} rover datasets. Ground truth is GPS + compass, so residuals "
                 "are dominated by label noise (2-5 m GPS at 9-15 m range ⇒ 0.15-0.5 rad "
                 "bearing noise floor); systematics appear as biases beneath it. The natural "
                 "unit is the mission ERA (day/installation): heading bias and quality are "
                 "stable within an era and jump between eras. Rover NaN baseline is 60-70% "
                 "(bursty WiFi/o4 emitter) — high NaN alone is NOT a defect here (section 6).")
    eras = defaultdict(list)
    for r in rover:
        eras[r["dataset"].split("_")[0]].append(r)
    top = sorted(eras.items(), key=lambda kv: -len(kv[1]))[:12]
    rep.fig.text(L, y, "Table R1 — mission eras", fontsize=9, fontweight="bold", va="top")
    y -= 0.018
    hdr = (f"{'era':10s} {'n':>4s} {'med NaN%':>8s} {'med corr':>8s} {'med heading':>11s} "
           f"{'med range m':>11s} {'OK/FL/QU':>9s}")
    rep.fig.text(L, y, hdr, fontsize=7.2, family="DejaVu Sans Mono", fontweight="bold", va="top")
    y -= 0.0145
    for era, rs in top:
        med = lambda k: np.nanmedian([g(x, k) for x in rs])  # noqa: E731
        st = Counter(x["status"] for x in rs)
        hh = med("heading_common")
        line = (f"{era[:10]:10s} {len(rs):4d} {100*med('r0_nan_frac'):8.0f} "
                f"{med('r0_circstd_corr'):8.2f} {hh:+11.2f} {med('range_med_m'):11.1f} "
                f"{st.get('OK',0):3d}/{st.get('FLAG',0):3d}/{st.get('QUARANTINE',0):3d}")
        rep.fig.text(L, y, line, fontsize=7.2, family="DejaVu Sans Mono", va="top")
        y -= 0.0135
    y -= 0.008
    y = rep.para(y, "Reading: the 'rover' era (spring-2025 named runs) is the largest and "
                 "near-zero heading bias; the Dec-Feb named-day eras carry the ≈−0.15 rad "
                 "heading bias (section 8). Per-era OK/FLAG mixes are driven by circstd and "
                 "ts flags; QUAR here means no_signal (fewer than 100 valid snapshots).", size=8.8)
    ax = rep.ax([0.09, 0.10, 0.85, max(0.14, y - 0.14)])
    labels = [e for e, _ in top]
    oks = [Counter(x["status"] for x in rs).get("OK", 0) for _, rs in top]
    fls = [Counter(x["status"] for x in rs).get("FLAG", 0) for _, rs in top]
    qs = [Counter(x["status"] for x in rs).get("QUARANTINE", 0) for _, rs in top]
    xg = np.arange(len(labels))
    ax.bar(xg, oks, color="#2e8b57", label="OK")
    ax.bar(xg, fls, bottom=oks, color="#f0ad4e", label="FLAG")
    ax.bar(xg, qs, bottom=np.array(oks) + np.array(fls), color="#c0392b", label="QUAR")
    ax.set_xticks(xg, labels, rotation=30, fontsize=7.5)
    ax.legend(fontsize=8)
    ax.set_title("Fig. R1 — quality bins per rover era", fontsize=8.5)
    ax.tick_params(labelsize=8)


def sec_dive_nan(rep, rows):
    wall = [r for r in rows if r.get("platform") == "wall"]
    rover = [r for r in rows if r.get("platform") == "rover"]
    rep.new_page()
    y = rep.h1("6  Deep dive — NaN / no-signal snapshots")
    y = rep.para(y, "WHAT NaN MEANS. mean_phase is computed only over windows classified as "
                 "signal (coherent phase, stddev<0.5, or amplitude>40). If a snapshot has NO "
                 "signal windows, mean_phase = NaN — silence is represented explicitly, never "
                 "averaged in. NaN is therefore a DUTY-CYCLE descriptor, not by itself a "
                 "defect: the wall blaster transmits continuously (baseline ≈0% NaN) while the "
                 "rover's WiFi/o4 emitter is bursty (healthy baseline 60-70% NaN).")
    y = rep.para(y, "THE DISCRIMINATOR (fig. N2): plotting NaN% against the quality of the "
                 "VALID part separates two populations. Rover: high NaN with clean valid part "
                 "— intermittent but healthy. Wall bad-months: elevated NaN AND noisy valid "
                 "part — a degraded capture, not a bursty emitter. A NaN gate alone cannot "
                 "make this distinction; the pair (NaN%, valid-part circstd) can.")
    nb = sum(1 for r in wall if "nan5-20" in (r.get("reasons") or ""))
    y = rep.para(y, f"GATES (v2, applied in this scan): wall two-tier — 5-20% = FLAG ({nb} "
                 "datasets, kept usable), >20% = QUARANTINE; rover unchanged (>90% or <100 "
                 "valid snapshots = no_signal). Root-causing low-SNR vs absent-emitter needs "
                 "the planned burst-SNR metric (section 13): signal-window amplitude vs "
                 "noise-window amplitude.")
    ax1 = rep.ax([0.09, 0.30, 0.40, 0.26])
    wn = [100 * max(g(r, "r0_nan_frac"), g(r, "r1_nan_frac")) for r in wall]
    rn = [100 * max(g(r, "r0_nan_frac"), g(r, "r1_nan_frac")) for r in rover]
    ax1.hist([x for x in wn if np.isfinite(x)], bins=np.arange(0, 102, 4), alpha=0.65,
             label="wall", color=ACC, log=True)
    ax1.hist([x for x in rn if np.isfinite(x)], bins=np.arange(0, 102, 4), alpha=0.65,
             label="rover", color="#c0392b", log=True)
    ax1.axvline(5, color=ACC, ls=":", lw=1); ax1.axvline(20, color=ACC, ls="--", lw=1)
    ax1.axvline(90, color="#c0392b", ls="--", lw=1)
    ax1.set_xlabel("max NaN % (r0,r1)", fontsize=8.5)
    ax1.set_ylabel("datasets (log)", fontsize=8.5)
    ax1.legend(fontsize=8)
    ax1.set_title("Fig. N1 — NaN% by platform, with gates\n(wall 5/20% two-tier; rover 90%)", fontsize=8.5)
    ax1.tick_params(labelsize=8)
    ax2 = rep.ax([0.575, 0.30, 0.38, 0.26])
    for rs, col, lab in ((wall, ACC, "wall"), (rover, "#c0392b", "rover")):
        xs = [100 * max(g(r, "r0_nan_frac"), g(r, "r1_nan_frac")) for r in rs]
        ys = [g(r, "r0_circstd_corr") for r in rs]
        m = [i for i in range(len(xs)) if np.isfinite(xs[i]) and np.isfinite(ys[i])]
        ax2.scatter([xs[i] for i in m], [ys[i] for i in m], s=5, alpha=0.35, color=col, label=lab)
    ax2.set_xlabel("max NaN %", fontsize=8.5)
    ax2.set_ylabel("corrected circstd of VALID part (rad)", fontsize=8.5)
    ax2.legend(fontsize=8)
    ax2.set_title("Fig. N2 — the discriminator: duty cycle vs\ncapture quality are separable", fontsize=8.5)
    ax2.tick_params(labelsize=8)
    rep.para(0.24, "Reading fig. N2: the rover cloud sits at high NaN / moderate circstd (its "
             "GPS floor) — bursty but fine. The wall cloud splits: the dense mass at (0%, "
             "0.4-0.6) is healthy; the arm extending right AND up is the Oct/Nov-24 + Feb-25 "
             "era (section 7) — NaN and valid-part noise rise TOGETHER, the signature of a "
             "weak/degraded signal chain rather than an intermittent emitter.", size=8.8)


def sec_roadmap(rep):
    rep.new_page()
    y = rep.h1("13  Scanner improvements — v2/v3 roadmap")
    y = rep.para(y, "The v1 scanner did its job (it found the systematics and the bad eras) "
                 "but this scan also measured the scanner itself. Planned changes, with the "
                 "evidence that motivates each:")
    rep.fig.text(L, y, "v2 — gates and fits (IMPLEMENTED — this report uses the v2 scan)", fontsize=10.5,
                 fontweight="bold", va="top")
    y -= 0.021
    for t in [
        "1. Widen wall gain grid to [0.7, 3.0] — the cfg-0.20λ group pinned at the 2.0 bound (true g ≈ 2.0-2.5).",
        "2. Widen rover Δθ grid to ±0.9 rad — 73/139 rover fits hit the ±0.35 bound; real biases reach −0.72.",
        "3. Two-tier wall NaN gate: 5-20% = FLAG, >20% = QUARANTINE (142 borderline datasets were over-quarantined).",
        "4. Timestamp gate at >1% of pairs (median violation is 0.02% — benign NTP jitter; only ~3 datasets exceed 1%).",
        "5. Common/differential Δθ decomposition for the wall too (separates shared geometry error from per-array mount).",
        "6. Coverage-aware fitting: skip multi-parameter fits when angular coverage is narrow (g and Δθ degenerate).",
        "7. Serial re-check of ERROR datasets (12-way parallel LMDB opens can fake 'unreadable' via lock contention).",
        "8. Frozen-tail: emit the truncation index so #42 datasets can be salvaged (prefix is good data) instead of dropped.",
    ]:
        y = rep.para(y, t, size=8.9, gap=0.003)
    y -= 0.008
    rep.fig.text(L, y, "v3 — new metrics and detectors (planned)", fontsize=10.5, fontweight="bold", va="top")
    y -= 0.021
    for t in [
        "9. Duty-cycle descriptor: signal-window fraction + its time profile (a characteristic, never a failure).",
        "10. Burst SNR: median |signal| in signal-windows ÷ noise-windows — free per-dataset SNR; would root-cause the "
        "bad months (weak emitter vs receiver degradation) directly from cached data.",
        "11. Noise-floor tracking from the noise windows — a free receiver-floor calibration per dataset; drift across "
        "eras flags gain/AGC changes.",
        "12. SNR-weighted residual fits: weight each snapshot by window count × inverse phase-stddev.",
        "13. Time-lag Δt fit (plan M10): rover tx/rx clock alignment via residual concentration over lag.",
        "14. Interference detectors (for coherent in-band transmitters, which defeat the coherence test): "
        "(a) two-component von Mises + uniform mixture on the residual → interferer fraction + its angle; "
        "(b) static-source back-projection — contaminated snapshots cluster at one WORLD angle while GT moves, "
        "localizing the interferer; (c) cross-receiver coherent-outlier test (both receivers wrong consistently = "
        "interferer, independently = noise); (d) second-peak persistence in the cached windowed beamformer.",
        "15. Calibration sidecar emission: per-(dataset × receiver) fitted (c, g, Δθ) with the plan's guards — "
        "time-split stability, rig-consistency, bounded priors, minimum-improvement threshold.",
    ]:
        y = rep.para(y, t, size=8.9, gap=0.003)
    y -= 0.006
    rep.para(y, "Evidence for the interference detectors is already in this scan: the "
             "2025-04-05 rover session (7 datasets, structure 0.98-1.45, 55-65% outliers) is a "
             "discrete interference/multipath incident, and the 915 MHz band shows 2.3× the "
             "structured residual of the other bands fleet-wide.", size=8.9)


def sec_actions(rep):
    rep.new_page()
    y = rep.h1("14  Recommended actions")
    for t, d in [
        ("1. Physical rig check (10 minutes, highest leverage)",
         "Measure the actual minimum achievable antenna spacing on the 2.4 GHz wall arrays. If ~6.5 cm, "
         "the sub-floor configs are confirmed mislabeled; if smaller, mutual coupling is the cause. "
         "Either way the fitted effective d/λ is the number the pipeline should use."),
        ("2. Clean split files",
         "Exclude from the next training split: the >20% NaN bad-month datasets, the 21 frozen-tail "
         "datasets (or truncate at freeze), the 23 error datasets, and the rover mount-anomaly "
         "sessions (feb7/feb8 mission1). Roughly 480-500 datasets, all era-clustered."),
        ("3. Calibration sidecar",
         "Emit per-(dataset × receiver) fitted (c, g, Δθ) with the plan's guards (time-split "
         "stability, rig consistency, bounded priors, minimum improvement). Feed empirical tables "
         "and filters first; A/B a retrain with corrected spacing/labels after."),
        ("4. Scanner v2",
         "Widen grids (wall g → 3.0, rover Δθ → ±0.9), two-tier NaN gate (5-20% FLAG / >20% "
         "QUARANTINE), ts gate at >1%, add common/differential Δθ decomposition for wall too."),
        ("5. Collection-time gates",
         "Run Tier-1 metrics inside data_collector at capture time so a Nov-2024-style bad era or a "
         "frozen planner is caught in the field within minutes, not months later."),
    ]:
        rep.fig.text(L, y, t, fontsize=10.5, fontweight="bold", va="top")
        y -= 0.019
        y = rep.para(y, d, size=9.0) - 0.006


def sec_appendix(rep, rows):
    order = {"ERROR": 0, "QUARANTINE": 1, "FLAG": 2, "OK": 3}
    rs = sorted(rows, key=lambda r: (r.get("platform") or "zz", r["dataset"]))
    per_page = 48
    first_page_rows = 38
    header = (f"{'dataset':40s} {'plat':5s} {'stat':6s} {'nan%':7s} {'raw>cor':9s} "
              f"{'g0/g1':9s} {'dth0':5s} issues")
    npages = int(np.ceil(max(0, len(rs) - first_page_rows) / per_page)) + 1
    for p in range(npages):
        rep.new_page()
        if p == 0:
            y = rep.h1("Appendix — per-file quality and issues (all datasets)")
            y = rep.para(y, "Sorted by platform then name. stat: OK / FLAG (usable, has a "
                         "known systematic) / QUAR (fails validity gate) / ERR. nan% = "
                         "max(r0,r1) invalid-phase fraction. raw>cor = r0 circular stddev "
                         "before/after the fitted correction (rad). g0/g1 = fitted gain per "
                         "receiver (effective/configured spacing). dth0 = fitted mount shift "
                         "(rad). Issues = gate reasons.", size=8.2)
        else:
            y = T - 0.01
            rep.fig.text(L, T, f"Appendix — per-file quality (cont., page {p+1}/{npages})",
                         fontsize=9, color=MUT, va="top")
            y = T - 0.028
        rep.fig.text(L, y, header, fontsize=6.4, family="DejaVu Sans Mono",
                     fontweight="bold", va="top")
        y -= 0.014
        if p == 0:
            chunk = rs[:first_page_rows]
        else:
            a = first_page_rows + (p - 1) * per_page
            chunk = rs[a:a + per_page]
        for r in chunk:
            st = {"QUARANTINE": "QUAR", "ERROR": "ERR"}.get(r["status"], r["status"])
            nan = max(g(r, "r0_nan_frac", 1.0), g(r, "r1_nan_frac", 1.0))
            nan_s = f"{100*nan:.0f}" if np.isfinite(nan) else "-"
            rc = (f"{g(r,'r0_circstd_raw'):.2f}>{g(r,'r0_circstd_corr'):.2f}"
                  if np.isfinite(g(r, "r0_circstd_raw")) else "-")
            gg = (f"{g(r,'r0_g'):.2f}/{g(r,'r1_g'):.2f}"
                  if np.isfinite(g(r, "r0_g")) else "-")
            dt = f"{g(r,'r0_dtheta'):+.2f}" if np.isfinite(g(r, "r0_dtheta")) else "-"
            reasons = _trunc((r.get("reasons") or "").replace("FLAG:", "F:").replace("QUAR:", "Q:"), 38)
            line = (f"{r['dataset'][:40]:40s} {(r.get('platform') or '?')[:5]:5s} {st:6s} "
                    f"{nan_s:7s} {rc:9s} {gg:9s} {dt:5s} {reasons}")
            rep.fig.text(L, y, line, fontsize=6.4, family="DejaVu Sans Mono", va="top")
            y -= 0.0173


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.csv)
    rep = Rep(args.out)
    sec_title(rep, rows)
    sec_metrics(rep)
    sec_bins(rep, rows)
    sec_wall_breakdown(rep, rows)
    sec_rover_breakdown(rep, rows)
    sec_dive_gain(rep, rows)
    sec_dive_gain_model(rep, rows)
    sec_dive_nan(rep, rows)
    sec_dive_badmonths(rep, rows)
    sec_dive_rover(rep, rows)
    sec_dive_frozen(rep, rows)
    sec_dive_ts(rep, rows)
    sec_dive_payoff(rep, rows)
    sec_dive_errors(rep, rows)
    sec_roadmap(rep)
    sec_actions(rep)
    sec_appendix(rep, rows)
    rep.close()
    print(f"wrote {args.out} ({rep.pageno} pages)")


if __name__ == "__main__":
    main()
