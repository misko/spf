"""Figures for the E-GSC computational report.

Every panel states in-sample vs held-out on the figure itself, labels both axes
with units, and keeps legends off the data.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

FREE = "#c0392b"
FROZ = "#1f6fb4"


def fig_identifiability(gsc2, gsc2b, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, key, title in (
        (
            axes[0],
            "in_campaign",
            "Held-out LOs, SAME session (A-G stage A)",
        ),
        (
            axes[1],
            "prospective",
            "Same held-out LOs, DIFFERENT session (2026-08-07)",
        ),
    ):
        for variant, colour, lbl in (
            ("free", FREE, "delays free (searched on train fold)"),
            ("frozen", FROZ, "delays frozen at 2.56 / 0.92 ns"),
        ):
            rows = [
                r for r in gsc2["summary"] if r["variant"] == variant
            ]
            rows.sort(key=lambda r: r["N"])
            N = np.array([r["N"] for r in rows])
            med = np.array([r[key]["uneq_mae_median"] for r in rows])
            q25 = np.array([r[key]["uneq_mae_q25"] for r in rows])
            q75 = np.array([r[key]["uneq_mae_q75"] for r in rows])
            ax.fill_between(N, q25, q75, color=colour, alpha=0.18, linewidth=0)
            ax.plot(N, med, "o-", color=colour, label=lbl, ms=4.5, lw=1.8)
        base = np.median(
            [r[key]["baseline_uneq_median"] for r in gsc2["summary"]]
        )
        ax.axhline(base, color="0.35", ls="--", lw=1.4)
        ax.text(
            27, base + 0.55, f"anchor only, no model: {base:.2f}°",
            color="0.25", fontsize=8.5,
        )
        if key == "in_campaign":
            gate = 2.8277
            gate_lbl = "dense 113-LO fit, leave-one-frequency-out: 2.83°"
        else:
            gate = gsc2b["asymptote_N113"][0]["uneq_mae_deg"]
            gate_lbl = f"dense 113-LO fit, transferred: {gate:.2f}°"
        ax.axhline(gate, color="#2e7d32", ls=":", lw=1.8)
        ax.text(6.2, gate - 1.15, gate_lbl, color="#2e7d32", fontsize=8.5)
        pre = [
            r for r in gsc2["fits"]
            if r["subset"] == "prereg10" and r["variant"] == "frozen"
        ][0]
        ax.plot(
            [10], [pre[key]["uneq_mae_deg"]], "v", color="black", ms=10,
            zorder=6, linestyle="none",
            label="the pre-registered uniform comb (N = 10)",
        )
        ax.annotate(
            f"the pre-registered uniform\n600 MHz comb, frozen delays:\n"
            f"{pre[key]['uneq_mae_deg']:.1f}°",
            xy=(10.4, pre[key]["uneq_mae_deg"] - 0.35),
            xytext=(12.5, 12.4),
            fontsize=8.5, color="black", ha="left",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        )
        ax.set_xscale("log")
        ax.set_xticks([6, 8, 10, 12, 16, 20, 24, 32, 48, 64])
        ax.set_xticklabels([str(x) for x in [6, 8, 10, 12, 16, 20, 24, 32, 48, 64]])
        ax.set_xlabel(
            "N = number of LOs used to refit L26  (of 113; log scale)"
        )
        ax.set_ylabel("held-out MAE on unequal-gain cells  (degrees)")
        ax.set_title(title)
        ax.set_ylim(0, 20)
    axes[0].legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    fig.suptitle(
        "E-GSC2: how many LOs does an L26 refit need?   "
        "median and inter-quartile range over 24 random LO subsets per N.   "
        "ALL POINTS ARE HELD-OUT (never in-sample).",
        fontsize=10.5, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_delays(gsc2, path):
    """The two ripple slots are EXCHANGEABLE -- the rung imposes no ordering and
    both slots search the same grid -- so a per-slot plot would show a spurious
    swap. Plot the sorted pair (longer, shorter) instead, and report the swap
    rate separately."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    fits = [
        r for r in gsc2["fits"]
        if r["variant"] == "free" and r["subset"].startswith("rand")
    ]
    Ns = sorted({r["N"] for r in fits})
    swap = [
        float(np.mean([r["tau_ns"][0] < r["tau_ns"][1]
                       for r in fits if r["N"] == n]))
        for n in Ns
    ]
    for k, (ax, fleet, name) in enumerate(
        ((axes[0], 2.56, "longer delay"), (axes[1], 0.92, "shorter delay"))
    ):
        data = [
            [sorted(r["tau_ns"], reverse=True)[k] for r in fits if r["N"] == n]
            for n in Ns
        ]
        bp = ax.boxplot(
            data, positions=range(len(Ns)), widths=0.6, showfliers=True,
            patch_artist=True,
            flierprops=dict(marker=".", ms=4, mfc="0.35", mec="none"),
            medianprops=dict(color="black", lw=1.6),
        )
        for b in bp["boxes"]:
            b.set_facecolor(FREE)
            b.set_alpha(0.35)
            b.set_edgecolor("0.3")
        ax.axhline(fleet, color="#2e7d32", ls="--", lw=1.6)
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("N = number of LOs in the training comb")
        ax.set_ylabel(f"fitted {name}  (ns)")
        ax.set_title(
            f"{name} of the fitted pair, searched on the TRAINING FOLD only\n"
            f"green dashes = committed fleet value {fleet:.2f} ns"
        )
        ax.set_ylim(0, 8.4)
    fig.suptitle(
        "E-GSC2: the nonlinear delays are what fail first.  24 free-delay refits "
        "per N, plotted as the SORTED pair because the two ripple slots are "
        "exchangeable\n(no ordering constraint in the rung; the slots are swapped "
        "relative to the fleet ordering in "
        + ", ".join(f"{100*sw:.0f}% of refits at N={n}"
                    for n, sw in zip(Ns, swap) if n in (6, 10, 24, 64))
        + ").\nBox = inter-quartile range over the 24 refits, bar = median, "
        "whiskers = 1.5 IQR, dots = individual refits beyond that.  Search grid "
        "0.10–8.00 ns (0.02 ns step below 4 ns); no refit reached the ceiling.",
        fontsize=9.5, y=1.10,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_aliasing(gsc2b, path, cond_samples=None, los=None, gsc2=None,
                 cond_fn=None, tau_fleet=None):
    al = gsc2b["aliasing"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    # (a) the conditioning distribution over 2,000 random 10-LO combs.
    #     Pure design property: no measured data enter this panel.
    ax = axes[0]
    if cond_samples is not None:
        ax.hist(
            cond_samples, bins=np.logspace(0, np.log10(45), 60),
            color="0.72", edgecolor="0.5", lw=0.4,
        )
    ax.axvline(
        al["cond_prereg_10"], color=FREE, lw=2.4,
        label=f"the comb that was scheduled: {al['cond_prereg_10']:.1f}",
    )
    ax.axvline(
        al["cond_random_10_median"], color=FROZ, lw=2.2, ls="--",
        label=f"random 10-LO comb, median: {al['cond_random_10_median']:.2f}",
    )
    ax.axvline(
        al["cond_all_113"], color="#2e7d32", lw=2.2, ls=":",
        label=f"all 113 LOs: {al['cond_all_113']:.3f}",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.95, 45)
    ax.set_xticks([1, 2, 3, 5, 10, 20, 40])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "40"])
    ax.set_ylim(0, 240)
    ax.set_ylabel("number of random 10-LO combs  (of 2,000)")
    ax.set_xlabel(
        "condition number of the 2-ripple frequency basis at the comb\n"
        "(dimensionless, log scale; 1 = perfectly separable)"
    )
    n_worse = int(round(al["frac_random_10_worse_than_prereg"] * 2000))
    ax.set_title(
        f"(a) DESIGN ONLY — no data.  Just {n_worse} of 2,000 random 10-LO\n"
        "combs is worse-conditioned than the one that was scheduled"
    )
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    # (b) does that design property predict the measured outcome? Every
    #     frozen-delay refit at N in {8..16}, its comb's condition number
    #     against its HELD-OUT error.
    ax = axes[1]
    if gsc2 is not None and cond_fn is not None:
        cmap = {8: "#8ecae6", 10: "#219ebc", 12: "#fb8500", 16: "#023047"}
        for N, colour in cmap.items():
            xs, ys = [], []
            for r in gsc2["fits"]:
                if (r["variant"] != "frozen" or r["N"] != N
                        or not r["subset"].startswith("rand")):
                    continue
                f_hz = np.array(r["train_lo_mhz"], dtype=float) * 1e6
                xs.append(cond_fn(f_hz, tau_fleet))
                ys.append(r["in_campaign"]["uneq_mae_deg"])
            ax.plot(xs, ys, "o", color=colour, ms=6, alpha=0.85,
                    label=f"random combs, N = {N}")
        pre = next(
            r for r in gsc2["fits"]
            if r["subset"] == "prereg10" and r["variant"] == "frozen"
        )
        f_hz = np.array(pre["train_lo_mhz"], dtype=float) * 1e6
        ax.plot(
            [cond_fn(f_hz, tau_fleet)], [pre["in_campaign"]["uneq_mae_deg"]],
            "v", color="black", ms=13, zorder=6, linestyle="none",
            label="the comb that was scheduled (N = 10)",
        )
    base = 8.31
    ax.axhline(base, color="0.35", ls="--", lw=1.4)
    ax.text(1.15, base + 0.55, f"anchor only, no model: {base:.2f}°",
            color="0.25", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1.1, 45)
    ax.set_xticks([2, 3, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "40"])
    ax.set_ylim(0, 20)
    ax.set_xlabel(
        "condition number of the 2-ripple basis at the training comb\n"
        "(dimensionless, log scale)"
    )
    ax.set_ylabel(
        "HELD-OUT MAE, unequal-gain cells  (degrees)"
    )
    ax.set_title(
        "(b) MEASURED. Frozen delays, scored on every LO not used to fit.\n"
        "Conditioning predicts held-out error (Spearman ρ = 0.403, "
        "p = 4.7e-05, n = 96)"
    )
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8)

    fig.suptitle(
        "E-GSC2: why the ten-frequency refit failed. The comb was near-uniform at "
        "600 MHz (one 700 MHz gap), and 600 MHz × (τ₁ − τ₂) = 0.984 ≈ one whole "
        "cycle,\nso the two ripple components become nearly proportional where "
        "they are sampled and least squares cannot separate them.",
        fontsize=9.8, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_gap(gsc3, path):
    rows = gsc3["rows"]

    def get(prefix):
        return next(r for r in rows if r["label"].startswith(prefix))

    lofo = get("stage A LOFO")
    a2g = get("A -> G")
    d2g = get("D -> G")
    prosp = get("prospective, all 113")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # (a) one change at a time. The first three bars are cumulative steps; the
    #     fourth is the destination, not a further step -- marked by the gap.
    ax = axes[0]
    steps = [
        ("published 2.26°\nall cells", lofo["mae_deg"], "0.62", 0),
        ("+ restate on\nunequal-gain\ncells", lofo["uneq_mae_deg"], "#7e57c2", 1),
        ("+ refit CV →\ntransfer\n(A → G)", a2g["uneq_mae_deg"], "#ef6c00", 2),
        ("prospective\ntransfer\npublished 4.79°",
         prosp["uneq_mae_deg"], "#2e7d32", 3.55),
    ]
    xs = [s[3] for s in steps]
    vals = [s[1] for s in steps]
    ax.bar(xs, vals, color=[s[2] for s in steps], width=0.66)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.10, f"{v:.3f}°", ha="center", fontsize=9.5)
    for i in range(3):
        ax.annotate(
            "", xy=(xs[i + 1] - 0.34 if i < 2 else xs[i + 1] - 0.34, vals[i]),
            xytext=(xs[i] + 0.34, vals[i]),
            arrowprops=dict(arrowstyle="-", lw=0.9, ls=":", color="0.4"),
        )
    ax.axvline(2.78, color="0.55", lw=1.0, ls="--")
    ax.text(2.80, 5.75, "different\nsession", fontsize=8, color="0.4")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.6)
    ax.set_ylabel("L26 error, held-out  (degrees)")
    ax.set_ylim(0, 6.4)
    ax.set_title(
        "E-GSC3: the 2.26° → 4.79° gap, one change at a time\n"
        "every bar is held-out; none is in-sample"
    )

    # (b) ratios
    ax = axes[1]
    entries = [
        (lofo, "stage A LOFO\nrefit, same session", "#7e57c2"),
        (a2g, "A → G\ntransfer, 12 h", "#ef6c00"),
        (d2g, "D → G\ntransfer", "#ef6c00"),
        (prosp, "→ 2026-08-07\ntransfer, 8 days", "#2e7d32"),
    ]
    lo = min(a2g["ratio_uneq"], d2g["ratio_uneq"])
    hi = max(a2g["ratio_uneq"], d2g["ratio_uneq"])
    ax.axhspan(lo, hi, color="#ef6c00", alpha=0.15, zorder=0)
    ax.axhline(hi, color="#ef6c00", lw=1.0, ls="--", zorder=1)
    x = np.arange(len(entries))
    ax.bar(
        x, [e[0]["ratio_uneq"] for e in entries],
        color=[e[2] for e in entries], width=0.62, zorder=3,
    )
    for xi, e in zip(x, entries):
        ax.text(xi, e[0]["ratio_uneq"] + 0.04, f"{e[0]['ratio_uneq']:.3f}×",
                ha="center", fontsize=9.5, zorder=4)
    ax.text(
        1.55, 2.90,
        f"shaded = the campaign's own unchanged-harness\n"
        f"transfers, {lo:.3f}× to {hi:.3f}×",
        fontsize=8.3, color="#a04000", ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([e[1] for e in entries], fontsize=8.2)
    ax.set_ylabel(
        "improvement ratio: baseline / L26\n(dimensionless, higher is better)"
    )
    ax.set_ylim(0, 3.5)
    ax.set_title(
        f"The prospective transfer ({prosp['ratio_uneq']:.3f}×) MATCHES the "
        f"campaign's own\ntransfers — {100*(prosp['ratio_uneq']/hi - 1):+.1f}% "
        "against the better of the two, not a regression"
    )
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_discriminator(gsc4, path):
    classes = [
        ("lna", "LNA word\nmoves"),
        ("mixer", "MIXER word\nmoves"),
        ("tia", "TIA word\nmoves"),
        ("lpf_only", "baseband LPF\nword only"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))

    # (a) dot plot, NOT bars: the axis is logarithmic, so bar length would not
    #     encode the ratio the panel is about.
    ax = axes[0]
    meds = [gsc4["by_class"][c]["median_abs_deg"] for c, _ in classes]
    p90s = [gsc4["by_class"][c]["p90_abs_deg"] for c, _ in classes]
    ns = [gsc4["by_class"][c]["n"] for c, _ in classes]
    cols = ["#c0392b", "#ef6c00", "#7e57c2", "0.45"]
    x = np.arange(len(classes))
    floor = gsc4["by_class"]["lpf_only"]["median_abs_deg"]
    ax.axhline(floor, color="0.45", ls="--", lw=1.3, zorder=1)
    for xi, m, p, n, c in zip(x, meds, p90s, ns, cols):
        ax.plot([xi, xi], [m, p], color=c, lw=2.0, zorder=2)
        ax.plot([xi], [p], "_", color=c, ms=13, mew=2.0, zorder=2)
        ax.plot([xi], [m], "o", color=c, ms=11, zorder=3)
        ax.text(xi + 0.17, m, f"{m:.3f}°\nn = {n}", fontsize=9, va="center")
        if c != "0.45":
            ax.text(xi + 0.17, p, f"{m/floor:.1f}× the floor", fontsize=8,
                    va="center", color=c)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in classes], fontsize=9)
    ax.set_xlim(-0.45, 3.9)
    ax.set_ylim(0.09, 30)
    ax.set_ylabel(
        "median |ΔH| per adjacent 1 dB step  (degrees, log scale)\n"
        "dot = median, tick = 90th percentile"
    )
    ax.set_title(
        "E-GSC4: what a 1 dB gain step costs, by which audited\n"
        "AD9361 word it moves.  Dashed line = the LPF-only floor"
    )

    # (b) every LNA transition
    ax = axes[1]
    det = gsc4["lna_transitions_detail"]
    keys = list(det.keys())
    keys = [keys[i] for i in np.argsort([det[k]["median_abs_deg"] for k in keys])]
    y = np.arange(len(keys))
    vals = [det[k]["median_abs_deg"] for k in keys]
    short = []
    for k in keys:
        parts = k.split()
        short.append(f"{parts[0]:6s} {parts[1]:>10s}   LNA {parts[4]}")
    ax.barh(y, vals, color="#c0392b", alpha=0.85, height=0.62, zorder=3)
    for yi, v, k in zip(y, vals, keys):
        ax.text(v + 0.3, yi, f"{v:.2f}°  n = {det[k]['n']}", va="center",
                fontsize=8.5)
    ax.axvline(floor, color="0.35", ls="--", lw=1.4, zorder=4)
    ax.annotate(
        f"same-dataset baseband\nLPF-only floor, {floor:.3f}°",
        xy=(floor, -0.55), xytext=(6.5, -1.35), fontsize=8, color="0.3",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"),
    )
    ax.set_ylim(-1.6, len(keys) - 0.35)
    ax.set_xlim(-0.7, 25.5)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=8.5, family="monospace")
    ax.set_xlabel("median |ΔH| over the surveyed LOs  (degrees)")
    ax.set_title(
        "All nine adjacent-1 dB LNA transitions the gain tables contain\n"
        "rows read: gain-table band, requested dB step, LNA index change"
    )
    fig.suptitle(
        "|ΔH| is the change in the shared antisymmetric gain response "
        "H(f,g) = [D(g,26) − D(26,g)]/2, where D is the phase measured against a "
        "same-session equal-gain anchor.\n"
        "Reconstructed from the COMMITTED wide-survey fit (53 LOs, both radios) — "
        "a different session from the A-G campaign, so this is independent "
        "corroboration and is never pooled.",
        fontsize=9.0, y=1.08,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(outdir="."):
    from pathlib import Path

    outdir = Path(outdir)
    g2 = json.loads(Path("gsc2_identifiability.json").read_text())
    g2b = json.loads(Path("gsc2b_extras.json").read_text())
    g3 = json.loads(Path("gsc3_gap.json").read_text())
    g4 = json.loads(Path("gsc4_wide_discriminator.json").read_text())

    # regenerate the conditioning sample used by panel (b) of fig 3, with the
    # same seed and the same helper the analysis used
    import gsc_common as G  # noqa: E402
    import spflib as S  # noqa: E402
    from gsc2b_extras import ripple_conditioning  # noqa: E402

    los = np.unique(G.load_anchored(["A"]).lo_hz)
    rng = np.random.default_rng(20260807)
    cond = np.array(
        [
            ripple_conditioning(los[rng.choice(len(los), 10, replace=False)],
                                G.TAU_FLEET)
            for _ in range(2000)
        ]
    )
    assert abs(float(np.median(cond)) - g2b["aliasing"]["cond_random_10_median"]) < 1e-9

    fig_identifiability(g2, g2b, outdir / "fig1_identifiability.png")
    fig_delays(g2, outdir / "fig2_fitted_delays.png")
    fig_aliasing(g2b, outdir / "fig3_comb_aliasing.png", cond_samples=cond,
                 los=los, gsc2=g2, cond_fn=ripple_conditioning,
                 tau_fleet=G.TAU_FLEET)
    fig_gap(g3, outdir / "fig4_gap_decomposition.png")
    fig_discriminator(g4, outdir / "fig5_word_discriminator.png")
    print("wrote 5 figures to", outdir)


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
