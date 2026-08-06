"""Generate the README figure set from the extracted campaign scalars.

    python -m spf.calibrations.gain_state_phase_model_v1.make_figures \
        --extracted /path/to/extracted

Everything is drawn from real measurements plus the committed coefficients and
the source analysis' committed result JSONs. Nothing here is schematic.

The seven figures map onto the three things a reader needs to judge the model:

    data       fig1_data, fig2_mechanism
    modelling  fig3_ladder, fig7_calibration_cost
    error      fig4_fit, fig5_error, fig6_coverage
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fit_from_extracted import attach_anchor, load_stage  # noqa: E402
from .gain_tables import BANDS, band_for_lo, default_tables  # noqa: E402
from .model import GainStatePhaseModel  # noqa: E402

HERE = Path(__file__).resolve().parent
LADDER_JSON = (
    HERE.parent
    / "dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1"
)

POOLED = [
    "spectroscopy_20260730_full/A",
    "spectroscopy_20260730_full_r2/F",
    "spectroscopy_20260730_full/E_tx_0",
    "spectroscopy_20260730_full/rate_pilot",
]

# colourblind-safe, readable on white
C_R17, C_R18 = "#0072B2", "#D55E00"
C_BASE, C_MODEL, C_LUT = "#999999", "#009E73", "#CC79A7"
C_ACCENT = "#56B4E9"
BAND_EDGES = (1300, 4000)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def mark_bands(ax, y=None):
    for e in BAND_EDGES:
        ax.axvline(e, color="k", lw=0.8, ls=":", alpha=0.55, zorder=0)


def load_pooled(extracted: Path):
    parts = [load_stage(extracted, s) for s in POOLED]
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = f["completed"] & f["qvalid"]
    f = {k: v[keep] for k, v in f.items()}
    f = attach_anchor(f, None)
    f = {k: v[f["has_anchor"]] for k, v in f.items()}
    f["band"] = np.array([band_for_lo(x) for x in f["lo_hz"]])
    return f


# --------------------------------------------------------------------- fig 1
def fig_data(f, out: Path):
    """The measured data: what the correction is actually up against."""
    # NB: independent x-axes -- panel (a) is in MHz, panel (b) in dB.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8),
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    tab = default_tables()
    colour = {s: (C_R17 if s.startswith("104000bac") else C_R18)
              for s in set(f["serial"])}
    name = {s: ("R17" if s.startswith("104000bac") else "R18")
            for s in set(f["serial"])}

    ax = axes[0]
    stage_a = f["stage"] == POOLED[0]
    for g1, mk, ls in ((45, "o", "-"), (5, "^", "--")):
        for s in sorted(colour):
            m = stage_a & (f["g1"] == g1) & (f["g2"] == 26) & (f["serial"] == s)
            if not m.any():
                continue
            o = np.argsort(f["lo_hz"][m])
            ax.plot(f["lo_hz"][m][o] / 1e6, np.degrees(f["D"][m][o]), ls,
                    marker=mk, ms=2.2, lw=0.9, color=colour[s], alpha=0.85,
                    label=f"{name[s]}  (RX1,RX2)=({g1},26) dB")
    mark_bands(ax)
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel("LO frequency [MHz]")
    ax.set_ylabel("D = phase − equal-gain anchor  [deg]")
    ax.set_title("(a) Measured gain-dependent residual vs LO. Below 4 GHz the two "
                 "radios track each other closely (ρ≈0.99);\nabove it they diverge "
                 "(ρ≈0.45). Dotted lines are the gain-table band edges at "
                 "1300 / 4000 MHz.", loc="left")
    ax.legend(ncol=4, loc="lower left", fontsize=7.5)

    ax = axes[1]
    for b, col in zip(BANDS, (C_R17, C_R18, C_MODEL)):
        lo_db, hi_db = tab.gain_range_db(b)
        gs = np.arange(lo_db, hi_db + 1)
        lna = np.array([tab.state(b, int(g)).lna for g in gs], dtype=float)
        ax.step(gs, lna, where="post", lw=1.7, color=col, alpha=0.9,
                label=f"{b} table ({lo_db}…{hi_db} dB)")
    ax.axhspan(0.80, 1.20, color="k", alpha=0.11, zorder=0)
    ax.text(73, 1.0, "LNA index 1: never sampled by the\nA–G campaign "
                     "(the 2.4 GHz integer-gain\nruns do cover it)",
            fontsize=7, va="center", ha="right", color="#444")
    ax.set_xlabel("requested RX gain [dB]")
    ax.set_ylabel("LNA index")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.35, 3.6)
    ax.set_title("(b) The same requested dB is a different LNA state in each band "
                 "— which is why the model is indexed by\nhardware state, not by dB",
                 loc="left")
    ax.legend(ncol=3, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig1_data.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 2
# Published, carefully-derived values from the source report. They are plotted
# rather than recomputed here because both quantities need machinery this script
# does not reproduce: the symmetry split needs the paired additive-cross cells,
# and the ripple amplitude is read at each band's own best-fit delay. A crude
# variance proxy overstates the DeltaLNA = 0 cells by ~10x.
SYMMETRY = [  # (radio, g, mean|H| deg, mean|A| deg, asymmetric energy %)
    ("R17", 5, 6.41, 1.70, 3.5), ("R17", 45, 9.85, 2.29, 1.7),
    ("R18", 5, 7.09, 2.38, 6.0), ("R18", 45, 9.54, 1.66, 1.3),
]
ASYM_BY_BAND = [("low\n≤1300", 0.73), ("middle\n1301–4000", 1.24),
                ("high\n>4000", 3.72)]
RIPPLE = [  # (band, g, dLNA, amp R17, amp R18)
    ("low", 5, 0, 0.11, 0.36), ("middle", 5, 0, 0.19, 0.18),
    ("high", 5, -2, 4.6, 7.1), ("low", 45, +2, 10.7, 9.7),
    ("middle", 45, +2, 8.0, 8.1), ("high", 45, +1, 1.1, 3.3),
]


def fig_mechanism(out: Path):
    """The two structural choices, in published measured quantities."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8),
                             gridspec_kw={"width_ratios": [1.15, 0.8, 1.5]})

    # (a) the arms respond almost identically -- measured with no fit at all
    ax = axes[0]
    x = np.arange(len(SYMMETRY))
    w = 0.38
    ax.bar(x - w / 2, [r[2] for r in SYMMETRY], w, color=C_MODEL,
           label="common |H| (shared by both arms)")
    ax.bar(x + w / 2, [r[3] for r in SYMMETRY], w, color=C_LUT,
           label="arm asymmetry |A|")
    for i, r in enumerate(SYMMETRY):
        ax.text(i + w / 2, r[3] + 0.28, f"{r[4]}%", ha="center", fontsize=7.5,
                color=C_LUT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\ng={r[1]}" for r in SYMMETRY], fontsize=8)
    ax.set_ylabel("degrees")
    ax.set_ylim(0, 11.6)
    ax.set_title("(a) One shared curve covers both arms.\n"
                 "% = share of the energy left in |A|", loc="left")
    ax.legend(fontsize=7, loc="upper center")

    # (b) ... but not uniformly in frequency
    ax = axes[1]
    ax.bar(range(len(ASYM_BY_BAND)), [v for _, v in ASYM_BY_BAND],
           color=C_LUT, width=0.62)
    for i, (_, v) in enumerate(ASYM_BY_BAND):
        ax.text(i, v + 0.09, f"{v:.2f}°", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(ASYM_BY_BAND)))
    ax.set_xticklabels([k for k, _ in ASYM_BY_BAND], fontsize=7.5)
    ax.set_ylabel("mean |A|  [deg]")
    ax.set_ylim(0, 4.5)
    ax.set_title("(b) …but the residual\nasymmetry grows above 4 GHz", loc="left")

    # (c) ripple amplitude tracks the LNA index change, not the requested dB
    ax = axes[2]
    x = np.arange(len(RIPPLE))
    face = [("white" if r[2] == 0 else C_MODEL) for r in RIPPLE]
    ax.bar(x - w / 2, [r[3] for r in RIPPLE], w, color=face,
           edgecolor=C_MODEL, lw=1.1, label="R17")
    ax.bar(x + w / 2, [r[4] for r in RIPPLE], w, color=face, alpha=0.55,
           edgecolor=C_MODEL, lw=1.1, ls="--", label="R18")
    for i, r in enumerate(RIPPLE):
        ax.text(i, max(r[3], r[4]) + 0.35, f"ΔLNA {r[2]:+d}", ha="center",
                fontsize=7.5,
                fontweight="bold" if r[2] else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\ng={r[1]}" for r in RIPPLE], fontsize=7.5)
    ax.set_ylabel("fitted ripple amplitude  [deg]")
    ax.set_ylim(0, 13.2)
    ax.set_title("(c) Ripple appears only where the LNA index changes "
                 "(open bars = ΔLNA 0).\nSolid = R17, dashed = R18.", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig2_mechanism.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 3
def fig_ladder(out: Path):
    """Accuracy vs parameter count vs generalisation, from the committed JSON."""
    doc = json.loads((LADDER_JSON / "ladder_results_A_main.json").read_text())
    splits = [
        ("LOEO leave-one-epoch-out", "LOEO  (known cell)", "o", "-"),
        ("LOFO leave-one-frequency-out", "LOFO  (unseen frequency)", "s", "-"),
        ("LOBLOCK leave-frequency-block-out", "LOBLK (~690 MHz gap)", "^", "--"),
        ("LORO leave-one-radio-out", "LORO  (unseen radio)", "D", "--"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for key, lbl, mk, ls in splits:
        pts = [(r["params"], r["mae_deg"]) for r in doc[key]
               if r["coverage"] > 0.99 and r["params"] > 0]
        if not pts:
            continue
        pts.sort()
        # lower envelope: best error achievable at or below each parameter count
        env_x, env_y, best = [], [], math.inf
        for p, e in pts:
            best = min(best, e)
            env_x.append(p)
            env_y.append(best)
        ax.plot(env_x, env_y, ls, marker=mk, ms=3.5, lw=1.2, label=lbl, alpha=0.9)

    base = doc["LOEO leave-one-epoch-out"][0]["mae_deg"]
    ax.axhline(base, color=C_BASE, lw=1.4, ls="-")
    ax.set_ylim(0.5, base * 1.6)
    ax.text(3.6, base * 1.06, f"no gain correction (anchor only): {base:.2f}°",
            fontsize=8, color="#555")

    for key, name, dx, dy in (
        ("LOFO leave-one-frequency-out", "L26", 1.25, 1.16),
        ("LOFO leave-one-frequency-out", "L30", 0.62, 1.14),
        ("LOEO leave-one-epoch-out", "L24", 0.30, 1.02),
    ):
        r = [x for x in doc[key] if x["model"].startswith(name)]
        if not r:
            continue
        r = r[0]
        ax.scatter([r["params"]], [r["mae_deg"]], s=95, facecolors="none",
                   edgecolors="k", lw=1.3, zorder=5)
        ax.annotate(name, (r["params"], r["mae_deg"]),
                    xytext=(r["params"] * dx, r["mae_deg"] * dy),
                    fontsize=9, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("design columns (upper bound on parameters, log scale)")
    ax.set_ylabel("held-out circular MAE  [deg, log scale]")
    ax.set_title("Model ladder: what each extra parameter buys, and what it costs "
                 "in generalisation\n"
                 "(lower envelope per holdout; only rungs with 100% coverage)",
                 loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    # placed in the empty upper-right quadrant; the legend owns the lower-left
    ax.text(105, 6.15,
            "L30 = 8 hardware-state coefficients\n"
            "L26 = L30 + 2 LNA-indexed ripples\n"
            "L24 = per-frequency lookup table — known cells only. It fails closed\n"
            "         at an unseen frequency or radio, which is why only the blue\n"
            "         (known-cell) curve reaches it.",
            fontsize=7.4, color="#444", va="top", ha="left")
    ax.set_yticks([0.6, 1, 2, 3, 5, 7])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(out / "fig3_ladder.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 4
def fig_fit(f, model, out: Path):
    """Does the fit actually describe the data?"""
    pred, sup = predict_all(f, model)
    ok = sup
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    ax = axes[0]
    x = np.degrees(f["D"][ok])
    y = np.degrees(pred[ok])
    ax.scatter(x, y, s=3.2, alpha=0.18, color=C_MODEL, edgecolors="none")
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.9, alpha=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured D  [deg]")
    ax.set_ylabel("predicted D  [deg]")
    r = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"(a) Observed vs predicted  (in-sample, r = {r:.3f})", loc="left")

    ax = axes[1]
    before = np.degrees(np.abs(f["D"][ok]))
    after = np.degrees(np.abs(f["D"][ok] - pred[ok]))
    bins = np.linspace(0, np.percentile(before, 99.5), 60)
    ax.hist(before, bins=bins, color=C_BASE, alpha=0.75,
            label=f"anchor only          MAE {before.mean():.2f}°")
    ax.hist(after, bins=bins, color=C_MODEL, alpha=0.75,
            label=f"+ L26 (in-sample)  MAE {after.mean():.2f}°")
    ax.set_xlabel("|residual|  [deg]")
    ax.set_ylabel("frames  (one per LO × RX1 gain × RX2 gain × epoch)")
    ax.set_title("(b) Residual distribution, pooled set.  In-sample — the "
                 "held-out\nfigure is 2.11° (leave-one-frequency-out)", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig4_fit.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 5
def fig_error(f, model, out: Path):
    """Where the remaining error lives."""
    pred, sup = predict_all(f, model)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    los = np.unique(f["lo_hz"])
    edges = np.linspace(los.min(), los.max(), 26)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for label, err, col in (
        ("anchor only", np.abs(f["D"]), C_BASE),
        ("+ L26", np.abs(f["D"] - np.where(sup, pred, 0.0)), C_MODEL),
    ):
        m = [np.degrees(err[(f["lo_hz"] >= a) & (f["lo_hz"] < b)].mean())
             if ((f["lo_hz"] >= a) & (f["lo_hz"] < b)).any() else np.nan
             for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(ctr / 1e6, m, "-o", ms=3, lw=1.4, color=col, label=label)
    mark_bands(ax)
    ax.set_xlabel("LO frequency [MHz]")
    ax.set_ylabel("MAE  [deg]")
    ax.set_title("(a) The UNCORRECTED error concentrates above 4 GHz (2.5°→7.8°).\n"
                 "After L26 the error is nearly flat in frequency (1.2°→2.3°).",
                 loc="left")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    w = 0.38
    xs = np.arange(len(BANDS))
    for i, (label, err, col) in enumerate((
        ("anchor only", np.abs(f["D"]), C_BASE),
        ("+ L26", np.abs(f["D"] - np.where(sup, pred, 0.0)), C_MODEL),
    )):
        vals = [np.degrees(err[f["band"] == b].mean()) for b in BANDS]
        ax.bar(xs + (i - 0.5) * w, vals, w, color=col, label=label)
        for x_, v in zip(xs + (i - 0.5) * w, vals):
            ax.text(x_, v + 0.06, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["low\n≤1300", "middle\n1301–4000", "high\n>4000"],
                       fontsize=8)
    ax.set_ylabel("MAE  [deg]")
    ax.set_title("(b) By gain-table band", loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_error.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 6
def fig_coverage(model, out: Path):
    """Fail-closed behaviour made visible."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))
    tab = default_tables()
    for ax, lo, name in zip(axes, (900e6, 2_412e6, 5_100e6),
                            ("900 MHz (low)", "2412 MHz (middle)",
                             "5100 MHz (high)")):
        lo_db, hi_db = tab.gain_range_db(band_for_lo(lo))
        gs = np.arange(lo_db, hi_db + 1)
        img = np.full((len(gs), len(gs)), np.nan)
        for i, a in enumerate(gs):
            for j, b in enumerate(gs):
                p = model.predict(lo, int(a), int(b))
                if not p.supported:
                    img[i, j] = np.nan
                elif p.guarded:
                    img[i, j] = 0.0
                else:
                    img[i, j] = abs(p.residual_deg)
        im = ax.imshow(img, origin="lower", cmap="viridis",
                       extent=[gs[0], gs[-1], gs[0], gs[-1]], aspect="equal",
                       vmin=0, vmax=20)
        # refused cells read as white-with-hatch, so grey stays reserved for
        # "no correction / baseline" everywhere else in this figure set
        ax.set_facecolor("white")
        ax.patch.set_hatch("///")
        ax.patch.set_edgecolor("#d0d0d0")
        ax.plot([gs[0], gs[-1]], [gs[0], gs[-1]], color="k", lw=0.8, ls="--",
                alpha=0.55, zorder=3)
        n_ok = int(np.isfinite(img).sum())
        ax.set_title(f"{name}\n{n_ok} / {img.size} pairs supported", fontsize=9,
                     loc="left")
        ax.set_xlabel("RX2 gain [dB]")
        if ax is axes[0]:
            ax.set_ylabel("RX1 gain [dB]")
    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.015)
    cb.set_label("magnitude of CORRECTION applied |D|  [deg]\n(not an error — "
                 "dark is not 'good')", fontsize=8)
    fig.suptitle("Prediction coverage. Hatched = refused: the model fails closed "
                 "rather than extrapolating.\nRefusals are driven mostly by "
                 "unmeasured MIXER and LPF words (11 + 8–11 gains per band); only "
                 "2–3 come from the unmeasured LNA state.\nThe dashed line is the "
                 "equal-gain diagonal, where D is zero by construction. "
                 "(51 supported gains per band is a coincidence: 75−24 and 73−22.)",
                 fontsize=8.5, x=0.012, ha="left")
    fig.savefig(out / "fig6_coverage.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig 7
def fig_calibration_cost(out: Path):
    """How coarse may the calibration comb be?"""
    doc = json.loads((LADDER_JSON / "comb_results.json").read_text())["comb"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    span = 5900 - 400
    for name, style in (("L00 anchor only", (C_BASE, "-")),
                        ("L08 sym H(band,g)", ("#7f7fbf", "--")),
                        ("L16 MECH H(state)+ripple/LNA", (C_ACCENT, "-")),
                        ("L26 MECH H(state)+2 ripples/LNA", (C_MODEL, "-")),
                        ("L27 MECH rich", ("#B22222", "--"))):
        key = [k for k in doc if k.startswith(name.split()[0])]
        if not key:
            continue
        d = doc[key[0]]
        gaps, errs = [], []
        for k, v in d.items():
            nb = int(k.replace("blocks", ""))
            gaps.append(span / nb)
            errs.append(v)
        o = np.argsort(gaps)
        ax.plot(np.array(gaps)[o], np.array(errs)[o], style[1], marker="o",
                ms=4, lw=1.5, color=style[0], label=key[0])
    ax.axvspan(0, 690, color=C_MODEL, alpha=0.07)
    ax.set_ylim(1.2, 10.6)
    # no leader line: at this aspect ratio an arrow reads as another data series
    ax.text(2850, 10.2,
            "shaded: error is flat out to ~690 MHz gaps —\n"
            "a ~10-point comb recovers essentially all of\n"
            "the 113-point comb's benefit",
            fontsize=8, color="#2a6f5a", va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("held-out frequency gap  [MHz, log scale]")
    ax.set_ylabel("held-out MAE  [deg]")
    ax.set_title("Calibration cost: error vs how coarse the frequency comb is.\n"
                 "L27 (red) wins on a dense comb but is unstable at 690–1400 MHz "
                 "gaps (3.5° / 7.6° vs L26's 2.5° / 2.9°).", loc="left")
    leg = ax.legend(fontsize=7.2, loc="upper left", frameon=True,
                    framealpha=0.96, edgecolor="#cccccc")
    leg.get_frame().set_facecolor("white")
    ax.axvline(690, color=C_MODEL, lw=1.0, ls=":")
    ax.set_xticks([100, 200, 400, 690, 1400, 2800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(out / "fig7_calibration_cost.png", bbox_inches="tight")
    plt.close(fig)


def predict_all(f, model):
    pred = np.zeros(len(f["D"]))
    sup = np.zeros(len(f["D"]), dtype=bool)
    for i in range(len(pred)):
        p = model.predict(f["lo_hz"][i], int(f["g1"][i]), int(f["g2"][i]),
                          rf_hz=f["rf_hz"][i])
        pred[i] = p.residual_rad
        sup[i] = p.supported
    return pred, sup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f = load_pooled(args.extracted)
    model = GainStatePhaseModel.load_named("l26_pooled_v1")
    print(f"pooled rows={len(f['D'])}  LOs={len(np.unique(f['lo_hz']))}")

    fig_data(f, args.out)
    print("fig1_data")
    fig_mechanism(args.out)
    print("fig2_mechanism")
    fig_ladder(args.out)
    print("fig3_ladder")
    fig_fit(f, model, args.out)
    print("fig4_fit")
    fig_error(f, model, args.out)
    print("fig5_error")
    fig_coverage(model, args.out)
    print("fig6_coverage")
    fig_calibration_cost(args.out)
    print("fig7_calibration_cost")


if __name__ == "__main__":
    main()
