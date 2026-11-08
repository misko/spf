"""Figures for rover_applicability_ladder_20260812_v1.

Reads only committed JSON in this repository plus this report's own
``analysis.json``. Nothing under /mnt is opened here.

    python -m spf.calibrations.dual_rx_gain_frequency.reports.\
rover_applicability_ladder_20260812_v1.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# validated categorical slots 1-3 (dataviz reference palette, all-pairs pass)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
SEQ = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "font.size": 9,
    "axes.grid": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

A = json.loads((HERE / "analysis.json").read_text())
L = json.loads((HERE / "ladder_rebuilt.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


# ---------------------------------------------------------------- fig 1
def fig1_ladder():
    rungs = L["stage_a_ladder"]
    keep = ["L00", "L01", "L05", "L08", "L11", "L14", "L16", "L26", "L27",
            "L29", "L30", "L31", "L33", "L23", "L24"]
    pts = {}
    for short in keep:
        k = next((k for k in rungs if k.startswith(short + " ")), None)
        if k is None:
            continue
        pts[short] = rungs[k]
    base = pts["L00"]["LOEO"]["mae"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhline(base, color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.annotate("L00 anchor only, 6.65°", (1.6, base - 0.42), color=INK2,
                ha="left", fontsize=8)

    series = [("LOEO", C1, "o", "known cell (leave-one-epoch-out)"),
              ("LOBLK", C2, "s", "unmeasured ~690 MHz gap (leave-block-out)"),
              ("LORO", C3, "^", "unseen radio (leave-one-radio-out)")]
    for split, col, mk, lab in series:
        xs, ys = [], []
        for short, e in pts.items():
            s = e.get(split)
            if s is None or s["coverage"] < 0.999:
                continue
            xs.append(max(e["params"], 0.7))
            ys.append(s["mae"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, color=col, lw=2.0, marker=mk, ms=6, label=lab,
                zorder=3, markeredgecolor="white", markeredgewidth=1.2)

    # the fail-closed LUT rungs, known cell only
    for short in ("L23", "L24"):
        e = pts[short]
        ax.plot([e["params"]], [e["LOEO"]["mae"]], marker="o", ms=6, color=C1,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{short}\n{e['LOEO']['mae']:.2f}°",
                    (e["params"], e["LOEO"]["mae"]), textcoords="offset points",
                    xytext=(0, -26), ha="center", fontsize=8, color=INK2)

    offsets = {"L01": (0, 10), "L08": (-11, 10), "L30": (12, 8),
               "L16": (-13, 11), "L31": (11, 10), "L26": (10, 9),
               "L27": (14, -4)}
    for short, xy in offsets.items():
        e = pts[short]
        ax.annotate(short, (max(e["params"], 0.7), e["LOBLK"]["mae"]),
                    textcoords="offset points", xytext=xy, ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("non-zero design columns (log scale)")
    ax.set_ylabel("anchored MAE, degrees")
    ax.set_title("Stage-A model ladder, re-derived from ladder_results_A_main.json\n"
                 "only rungs at 100% coverage are drawn on the generalising splits",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left",
              bbox_to_anchor=(0.02, 0.02))
    ax.set_ylim(0, 7.6)
    save(fig, "fig1_ladder.png")


# ---------------------------------------------------------------- fig 2
def fig2_state_demand():
    lo = "5766000000"
    e = A["rover_corpus"]["per_lo"][lo]
    # exact per-arm demand from the decoded levels
    demand = e["mixer_arm_share"]
    levels = list(range(0, 16))
    share = [demand.get(str(m), 0.0) for m in levels]

    fitted = {
        "l26_pooled_v1 (shipped)": A["rover_corpus"]["fitted_levels"]
        ["l26_pooled_v1"]["mixer"],
        "l30 / l31_pooled_v1": A["rover_corpus"]["fitted_levels"]
        ["l30_pooled_v1"]["mixer"],
        "E-GSC6 gain list": e["refit_coverage"]["e_gsc6"]["levels"]["mixer"],
        "E-GSC6 + E-GSC7": e["refit_coverage"]["e_gsc6_plus_e_gsc7"]["levels"]
        ["mixer"],
    }

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.25], "hspace": 0.18})
    ax.bar(levels, [100 * s for s in share], color=SEQ, width=0.62)
    for m, s in zip(levels, share):
        if s > 0.005:
            ax.annotate(f"{100*s:.1f}%", (m, 100 * s), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
    ax.set_ylabel("share of receive arms, %")
    ax.set_title("What the rover actually asks for, and what has been fitted\n"
                 "5766 MHz, 134,374 frames x 2 arms, measured", loc="left",
                 fontsize=10)
    ax.set_ylim(0, 62)

    rows = list(fitted)
    for i, name in enumerate(rows):
        y = len(rows) - 1 - i
        for m in levels:
            present = m in fitted[name]
            ax2.add_patch(plt.Rectangle(
                (m - 0.34, y - 0.34), 0.68, 0.68,
                facecolor=SEQ if present else "#f0efec",
                edgecolor="white", linewidth=1.4))
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(rows[::-1], fontsize=8.5)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlim(-0.7, 15.7)
    ax2.set_xticks(levels)
    ax2.set_xlabel("AD9361 MIXER_GM_GAIN word")
    ax2.grid(False)
    ax2.set_title("filled = the fit estimated this mixer level", loc="left",
                  fontsize=8.5, color=INK2)
    save(fig, "fig2_state_demand.png")


# ---------------------------------------------------------------- fig 3
def fig3_coverage():
    entries = [
        ("l26_stage_a_v1", lambda e: e["coefficient_set_support"]
         ["l26_stage_a_v1"]["frame_weighted_fraction"]),
        ("l26_pooled_v1 (shipped default)", lambda e: e["coefficient_set_support"]
         ["l26_pooled_v1"]["frame_weighted_fraction"]),
        ("l30 / l31_pooled_v1", lambda e: e["coefficient_set_support"]
         ["l30_pooled_v1"]["frame_weighted_fraction"]),
        ("refit on E-GSC6, with an LPF term", lambda e: e["refit_coverage"]
         ["e_gsc6"]["with_lpf"]["all_frames"]),
        ("refit on E-GSC6, no LPF term", lambda e: e["refit_coverage"]
         ["e_gsc6"]["rf_only"]["all_frames"]),
        ("refit on E-GSC6 + E-GSC7, with an LPF term",
         lambda e: e["refit_coverage"]["e_gsc6_plus_e_gsc7"]["with_lpf"]["all_frames"]),
        ("refit on E-GSC6 + E-GSC7, no LPF term",
         lambda e: e["refit_coverage"]["e_gsc6_plus_e_gsc7"]["rf_only"]["all_frames"]),
    ]
    e66 = A["rover_corpus"]["per_lo"]["5766000000"]
    e84 = A["rover_corpus"]["per_lo"]["5840000000"]
    labels = [n for n, _ in entries]
    v5766 = [100 * f(e66) for _, f in entries]
    v5840 = [100 * f(e84) for _, f in entries]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    h = 0.36
    ax.barh(y + h / 2, v5766, height=h, color=C1, label="5766 MHz (134,374 frames)")
    ax.barh(y - h / 2, v5840, height=h, color=C2, label="5840 MHz (43,036 frames)")
    for yy, v in zip(y + h / 2, v5766):
        ax.annotate(f"{v:.1f}%", (v, yy), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    for yy, v in zip(y - h / 2, v5840):
        ax.annotate(f"{v:.1f}%", (v, yy), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of rover frames the model can predict at all, %")
    ax.set_title("Coverage of the 2026 rover corpus — measured, fail-closed\n"
                 "top four rows are the coefficient sets that exist today",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              bbox_to_anchor=(1.0, 0.30))
    save(fig, "fig3_coverage.png")


# ---------------------------------------------------------------- fig 4
def fig4_gsc7():
    g = A["e_gsc7"]
    steps = g["steps_by_run"]
    thr = g["h1_threshold_deg"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    x = np.arange(1, 11)
    styles = [("R18_usb", C1, "-", "o", "R18 (clean) USB"),
              ("R18_ip", C1, "--", "s", "R18 (clean) IP"),
              ("R17_usb", C2, "-", "o", "R17 (damaged) USB"),
              ("R17_ip", C2, "--", "s", "R17 (damaged) IP")]
    for key, col, ls, mk, lab in styles:
        ax.plot(x, np.abs(steps[key]), color=col, ls=ls, lw=1.8, marker=mk,
                ms=5, label=lab, markeredgecolor="white", markeredgewidth=1.0)
    ax.axhline(thr, color=INK2, lw=1.3, ls=":")
    ax.annotate(f"H1 resolution threshold {thr}°", (10.3, thr + 0.06),
                ha="right", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{51+i}→{52+i}" for i in range(10)], rotation=45,
                       fontsize=7.5)
    ax.set_xlabel("adjacent 1 dB step (mixer 5→15), dB")
    ax.set_ylabel("|phase step|, degrees")
    ax.set_title("(a) E-GSC7 at 5766 MHz: ten steps, most unresolved",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)

    rows = g["cross_lo_transfer"]["rows"]
    los = sorted({r["lo_hz"] for r in rows})
    for radio, col in (("R18", C1), ("R17", C2)):
        for transport, mk, ls in (("usb", "o", "-"), ("ip", "s", "--")):
            ys = [next(r["cross_lo_rms_deg"] for r in rows
                       if r["radio"] == radio and r["transport"] == transport
                       and r["lo_hz"] == lo) for lo in los]
            ax2.plot([l / 1e9 for l in los], ys, color=col, ls=ls, lw=1.8,
                     marker=mk, ms=5, markeredgecolor="white",
                     markeredgewidth=1.0,
                     label=f"{radio} {transport.upper()}")
    ax2.axhline(0.5146615736431825, color=C3, lw=1.6)
    ax2.annotate("R18 same-LO USB↔IP repeat, 0.51°", (4.99, 0.56), ha="left",
                 fontsize=7.5, color=INK2)
    ax2.set_yscale("log")
    ax2.set_ylim(0.4, 300)
    ax2.set_xlabel("LO the 5766 MHz curve is transferred to, GHz")
    ax2.set_ylabel("curve RMS error, degrees (log)")
    ax2.set_title("(b) H5: the 5766 MHz ladder does not transfer",
                  loc="left", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    save(fig, "fig4_gsc7.png")


# ---------------------------------------------------------------- fig 5
def fig5_fold():
    f = A["segmentation_fold_measured"]
    cs = [2.0, 5.0, 10.0, 20.0]
    folded = [f["post_hoc_error_on_folded_trimmed_mean_deg"][f"{c:g}"]["median"]
              for c in cs]
    circ = [f["post_hoc_error_on_circular_mean_deg"][f"{c:g}"]["median"]
            for c in cs]
    worst = max(f["post_hoc_error_on_circular_mean_deg"][f"{c:g}"]["max"]
                for c in cs)

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.plot(cs, folded, color=C2, lw=2.0, marker="o", ms=7,
            markeredgecolor="white", markeredgewidth=1.2,
            label="weighted_windows_stats[0]  (folded, then trimmed)")
    ax.plot(cs, [0] * len(cs), color=C1, lw=2.0, marker="s", ms=7,
            markeredgecolor="white", markeredgewidth=1.2,
            label="mean_phase / mean_phase_segmentation  (circular mean)")
    ax.plot(cs, cs, color=MUTED, ls="--", lw=1.2)
    ax.annotate("y = x: the whole correction is lost", (12.5, 12.5),
                xytext=(4, -12), textcoords="offset points", ha="left",
                fontsize=8, color=INK2)
    for c, v in zip(cs, folded):
        ax.annotate(f"{v:.0f}°", (c, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2)
    ax.annotate(f"exactly zero to machine precision (worst {worst:.0e}°)",
                (3.4, 0), xytext=(0, 9), textcoords="offset points",
                ha="left", fontsize=8, color=INK2)
    ax.set_ylim(-3, 46)
    ax.set_xlabel("size of the phase correction applied, degrees")
    ax.set_ylabel("median error of the post-hoc correction, degrees")
    ax.set_title("A correction subtracted from a stored scalar is only valid if\n"
                 "the stored scalar is the circular mean — measured on 14,350 "
                 "rover frames", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig5_fold.png")


if __name__ == "__main__":
    fig1_ladder()
    fig2_state_demand()
    fig3_coverage()
    fig4_gsc7()
    fig5_fold()
