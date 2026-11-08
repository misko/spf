"""Regenerate the report's figures from the committed machine-readable results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

BLUE, ORANGE, GREEN, RED, GREY = ("#1f77b4", "#ff7f0e", "#2ca02c",
                                  "#d62728", "#7f7f7f")


def load(name):
    return json.loads((HERE / name).read_text())


# ------------------------------------------------------------------ figure 1
def fig1_check_a():
    d = load("check_a2.json")
    labels, pub, pkg, src = [], [], [], []
    m_pkg = d["measured"]["package_fit_all_rows"]
    m_src = d["measured"]["source_pipeline_1600_row_tau_search"]
    pub_map = {c["check"]: c["published"]
               for c in d["grading"]["package_fit_all_rows"]["checks"]}
    meas_pkg = {c["check"]: c["measured"]
                for c in d["grading"]["package_fit_all_rows"]["checks"]}
    meas_src = {c["check"]: c["measured"]
                for c in d["grading"]["source_pipeline_1600_row_tau_search"]["checks"]}
    order = [
        "pooled LOFO L31 MAE", "pooled LOFO L31 P95",
        "pooled LOFO L26 MAE", "pooled LOFO L30 MAE",
        "stage-A L31 LOEO MAE", "stage-A L31 LOFO MAE",
        "stage-A L31 LOBLK MAE", "stage-A L31 LORO MAE",
        "stage-A L26 LOEO MAE", "stage-A L26 LOFO MAE",
        "stage-A L26 LOBLK MAE", "stage-A L26 LORO MAE",
    ]
    for k in order:
        labels.append(k.replace("stage-A ", "A/").replace("pooled ", "pool/")
                       .replace(" MAE", "").replace(" P95", " (P95)"))
        pub.append(pub_map[k]); pkg.append(meas_pkg[k]); src.append(meas_src[k])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    ax = axes[0]
    w = 0.27
    ax.bar(x - w, pub, w, label="published (committed JSON)", color=GREY)
    ax.bar(x, pkg, w, label="refit: shipped GainStatePhaseModel.fit\n"
                            "(tau searched on ALL training rows)", color=BLUE)
    ax.bar(x + w, src, w, label="refit: source pipeline's rule\n"
                                "(tau searched on a 1600-row subsample)",
           color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylabel("holdout error (deg)")
    ax.set_title("Check A -- refitting the EXISTING rungs on the EXISTING data.\n"
                 "The shipped fit() misses the published pooled-L31 numbers; the "
                 "source pipeline's tau-search subsample recovers them exactly.\n"
                 "Under the source rule four stage-A columns remain outside "
                 "+-0.005 deg; the two largest are L26.", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(pub), max(pkg), max(src)) * 1.55)
    # column 1 is a P95, not a mean -- say so on the axis, not just in the label
    ax.annotate("this column alone is a P95,\nnot a mean -- do not compare\n"
                "its height to the others",
                xy=(1.27, pub[1] * 1.02), xytext=(4.4, pub[1] * 1.30),
                fontsize=8, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_ylabel("holdout MAE (deg), except col. 2 = P95")

    ax = axes[1]
    dp = np.array(pkg) - np.array(pub)
    ds = np.array(src) - np.array(pub)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-0.005, 0.005, color=GREEN, alpha=0.25,
               label="+-0.005 deg (published precision)")
    ax.plot(x, dp, "o-", color=BLUE, label="shipped fit() minus published")
    ax.plot(x, ds, "s-", color=ORANGE, label="source rule minus published")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("measured minus published (deg)")
    lim = max(0.30, float(np.abs(np.concatenate([dp, ds])).max()) * 1.9)
    ax.set_ylim(-lim, lim)
    ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
    for xi, v in zip(x, ds):
        if abs(v) > 0.005:
            ax.annotate(f"{v:+.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, -12 if v > 0 else 8), ha="center",
                        fontsize=7, color=ORANGE)
    ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_check_a.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig2_ladder():
    d = load("fit_new.json")
    variants = [
        ("U2_gsc6_plus_gsc7_R18_only", "R18 only\n(clean radio)"),
        ("U5_gsc6_plus_gsc7_R17_only", "R17 only\n(damaged harness)"),
        ("U1_gsc6_plus_gsc7_both_radios", "both radios pooled\n(the union as asked)"),
    ]
    splits = ["LOFO", "LOBLK", "LORO", "LOBAND"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(variants))
    w = 0.19
    colors = [BLUE, ORANGE, GREEN, RED]
    for j, sp in enumerate(splits):
        vals, hatch = [], []
        for v, _ in variants:
            r = d["ladder"][v]["L31"].get(sp, {})
            vals.append(r["all_cells"]["mae_deg"] if "all_cells" in r else np.nan)
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=f"L31 {sp}", color=colors[j])
        for xi, val in zip(x + (j - 1.5) * w, vals):
            if np.isnan(val):
                ax.bar([xi], [12.2], w, color="none", edgecolor=GREY,
                       hatch="//", lw=0.8)
                ax.text(xi, 6.1, "split does not\nexist (1 radio)",
                        ha="center", va="center", fontsize=7,
                        rotation=90, color=GREY)
    for i, (v, _) in enumerate(variants):
        b = d["datasets"][v]["baseline_all"]["mae_deg"]
        ax.plot([i - 0.45, i + 0.45], [b, b], color="k", lw=2.4,
                label="anchor-only baseline (L00)" if i == 0 else None)
        ax.annotate(f"L00 = {b:.2f}", xy=(i - 0.45, b),
                    xytext=(i - 0.45, 12.0), fontsize=8.5, fontweight="bold",
                    ha="left", va="top", color="k",
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.8, ls=":"))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in variants], fontsize=10)
    ax.set_ylabel("held-out MAE (deg)")
    ax.set_title("The L31-shaped rung, fitted SEPARATELY on each radio subset of the "
                 "E-GSC6 + E-GSC7 union.\nEach group has its own anchor-only "
                 "baseline (black); bars above it are WORSE than no correction "
                 "at all.", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.005), framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 13.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_union_ladder.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig3_coverage():
    d = load("rover_coverage_new.json")
    sets = ["l26_stage_a_v1", "l26_pooled_v1", "l30_pooled_v1", "l31_pooled_v1",
            "l31_gsc6_gsc7_r18_20260812_v1"]
    nice = ["l26_stage_a_v1", "l26_pooled_v1\n(shipped default)", "l30_pooled_v1",
            "l31_pooled_v1", "NEW\nl31_gsc6_gsc7_r18\n_20260812_v1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, carrier in zip(axes, ("5766000000", "5840000000")):
        c = d["carriers"][carrier]
        sup = [c["support"][s]["supported_fraction"] * 100 for s in sets]
        corr = [c["support"][s]["correcting_fraction"] * 100 for s in sets]
        x = np.arange(len(sets))
        ax.bar(x, sup, 0.62, color=BLUE, label="supported (predicted or rule-5 guarded)")
        ax.bar(x, corr, 0.62 * 0.5, color=ORANGE,
               label="actually corrected (guard not fired)")
        for xi, s_, cc in zip(x, sup, corr):
            ax.text(xi, s_ + 8.0, f"{s_:.2f}%", ha="center", fontsize=9,
                    fontweight="bold", color=BLUE)
            ax.text(xi, s_ + 2.0, f"{cc:.2f}%", ha="center", fontsize=8,
                    color=ORANGE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(nice, fontsize=8, rotation=14, ha="right")
        ax.set_title(f"{int(carrier)/1e6:.0f} MHz "
                     f"({c['arm_pair_observations']:,} arm-pair observations, "
                     f"{c['share_of_corpus']*100:.1f}% of corpus)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 128)
        ax.tick_params(labelleft=True)
    for a in axes:
        a.set_ylabel("share of that carrier's arm-pair observations (%)",
                     fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
    axes[0].text(0.02, 0.63, "coverage is APPLICABILITY, not accuracy:\n"
                             "it says the model knows what to predict,\n"
                             "not that predicting it helps",
                 transform=axes[0].transAxes, fontsize=7.5, color=RED,
                 style="italic")
    fig.suptitle("Measured fail-closed coverage of the 2026 rover corpus, "
                 "per carrier", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_rover_coverage.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig4_anchor():
    d = load("rover_anchor.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    hist = d["equal_gain_gain_histogram_dedup"]
    for carrier, color, lbl in (("5766000000", BLUE, "5766 MHz"),
                                ("5840000000", ORANGE, "5840 MHz")):
        h = hist[carrier]
        gains = sorted(int(k) for k in h)
        vals = [h[str(g)] for g in gains]
        ax.bar([g + (0 if carrier == "5766000000" else 0.42) for g in gains],
               vals, 0.42, color=color, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("requested gain, both arms equal (dB)")
    ax.set_ylabel("equal-gain arm-pair observations (log)")
    ax.set_title("Where the rover's equal-gain frames actually are:\n"
                 "overwhelmingly both arms at 62 dB, the top of the gain table",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.grid(axis="y", alpha=0.3)
    tot = {c: sum(hist[c].values()) for c in hist}
    for c, color in (("5766000000", BLUE), ("5840000000", ORANGE)):
        frac = hist[c].get("62", 0) / tot[c]
        ax.text(0.03, 0.94 if c == "5766000000" else 0.87,
                f"{int(c)/1e6:.0f} MHz: {frac*100:.0f}% of equal-gain "
                f"frames are 62/62 dB", transform=ax.transAxes, fontsize=8.5,
                color=color)

    ax = axes[1]
    rows = []
    for carrier in ("5766000000", "5840000000"):
        e = d["equal_gain"]["dedup"][carrier]
        rows.append((f"{int(carrier)/1e6:.0f} MHz",
                     e["streams_with_any"], e["n_streams"],
                     e["fraction"] * 100, e["median_equal_frames_per_stream"]))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=GREEN,
           label="receiver streams with ANY exact equal-gain frame")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=GREY,
           label="receiver streams total")
    for xi, r in zip(x, rows):
        ax.text(xi - 0.2, r[1] + 1.5, f"{r[1]}  ({r[1]/r[2]*100:.0f}%)",
                ha="center", fontsize=9.5, fontweight="bold", color=GREEN)
        ax.text(xi + 0.2, r[2] + 1.5, str(r[2]), ha="center", fontsize=9.5,
                color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r[0]}\n{r[3]:.2f}% of frames equal-gain\n"
         f"median {r[4]:.0f} per stream" for r in rows], fontsize=8.5)
    ax.set_ylim(0, 88)
    ax.set_ylabel("receiver streams")
    ax.set_title("Equal-gain frames exist at 5766 MHz (58 of 64 streams)\n"
                 "but at 5840 MHz 12 of 20 streams have none at all",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="upper center", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("The blocker that outranks the coefficients: the rover corpus "
                 "carries no USABLE harness anchor", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_anchor.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig5_diagonal():
    d = load("fit_new.json")
    hd = d["gsc6_heldout_diagonal"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    cats = ["all_bands", "high_band", "at_5766"]
    nice = ["all 24 LOs\n(480 cells / 1440 frames)",
            "high band, 9 LOs\n(a SUBSET of the left group)",
            "5766 MHz only\n(a SUBSET of the middle group)"]
    x = np.arange(len(cats))
    for i, (radio, color) in enumerate((("R18", BLUE), ("R17", RED))):
        vals = [hd[radio][c]["mae_deg"] for c in cats]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=color,
               label=f"{radio} ({'untouched control' if radio == 'R18' else 'connector-damaged'})")
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.4, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.368, color=GREEN, ls="--", lw=1.6,
               label="0.368 deg per-frame noise floor (E-GSC6, quoted)")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=8.5)
    ax.set_ylabel("|D(g,g)| MAE over held-out cells (deg)\n"
                  "1 cell = 1 (LO, gain), 3 quality-valid frames", fontsize=9)
    ax.set_title("The one measured, zero-parameter test available. Every "
                 "antisymmetric rung predicts\nexactly 0 on the equal-gain "
                 "diagonal, so E-GSC6's held-out |D(g,g)| IS the rung's error "
                 "there.\nThe three groups are NESTED, not independent.",
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 23.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_heldout_diagonal.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for fn in (fig1_check_a, fig2_ladder, fig3_coverage, fig4_anchor,
               fig5_diagonal):
        fn()
        print(f"ok {fn.__name__}")
