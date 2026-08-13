"""Figures for the E-GSC6+7+8 frame-level ladder report."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
deg = np.degrees
R18 = "1040007c4a94000211000b009186843ef2"
R17 = "104000bac4950008230026001b440a003a"
BASE = load_gsc.load()


def fig_anchor():
    """THE finding: the antisymmetry violation is set by the anchor gain."""
    d = json.load(open("antisym_vs_anchor.json"))
    refs = sorted(int(k) for k in d)
    labels = ["GSC7usb2 R18", "GSC7usb2 R17", "GSC8a R18", "GSC8a R17",
              "GSC8b R18", "GSC8b R17"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                  gridspec_kw={"width_ratios": [1, 1]})
    # 26 dB is an ISOLATED anchor: the high-band campaigns carry equal-gain cells
    # at 26 and at 52..62 and nothing between, so the two groups are drawn
    # separately rather than joined by a line no measurement supports.
    hi = [r for r in refs if r >= 52]
    for i, lab in enumerate(labels):
        r17 = "R17" in lab
        col = "tab:red" if r17 else "tab:blue"
        ls = ["-", "--", ":"][i // 2]
        for a in (ax, ax2):
            a.plot(hi, [d[str(r)][i] for r in hi], marker="o", ms=4, lw=1.8,
                   color=col, ls=ls, label=lab if a is ax else None)
            a.plot([26], [d["26"][i]], marker="X", ms=9, color=col, ls="none",
                   markeredgecolor="k", markeredgewidth=0.6)
    for a in (ax, ax2):
        a.axvspan(27, 51, color="grey", alpha=0.13)
        a.text(39, a.get_ylim()[1], " no equal-gain\n cells measured here",
               fontsize=7.5, color="dimgrey", va="top", ha="center")
    ax.set_yscale("log")
    ax.set_ylabel("|D(g,g)| mean over g=52..62  (deg, log)")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title("Full range: the 26 dB convention is catastrophic for R17")
    ax2.set_ylim(0, 4.0)
    ax2.set_ylabel("|D(g,g)|  (deg, linear)")
    ax2.set_title("Zoom on the high-band anchors: all under 3.5°")
    ax2.axvspan(55, 58, color="tab:green", alpha=0.12)
    ax2.text(56.5, 3.62, "best for both\nradios", ha="center", fontsize=8,
             color="tab:green")
    ax2.annotate("R17 at 26 dB is off-scale here (54-66°)\nsee the left panel",
                 xy=(26, 3.95), xytext=(31, 3.3), fontsize=7.5, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
    for a in (ax, ax2):
        a.set_xlabel("anchor gain (dB)")
        a.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=1)
    fig.suptitle("Model assumption D = H(s1) - H(s2) requires D(g,g) = 0.  "
                 "Whether it holds is decided by the ANCHOR GAIN, not the radio.",
                 fontsize=10.5)
    fig.tight_layout()
    fn = f"{OUT}/fig1_anchor_choice.png"
    fig.savefig(fn, dpi=115)
    plt.close(fig)
    return fn


def fig_rawphase():
    """Where R17's violation comes from: a step between 26 and 52 dB."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
    for ax, (L, lab) in zip(axes, [(5.766e9, "5766 MHz"), (5.840e9, "5840 MHz")]):
        for ser, name, col in ((R18, "R18 control", "tab:blue"),
                               (R17, "R17 damaged", "tab:red")):
            for stage, mk in (("GSC8a", "o"), ("GSC8b", "s")):
                m = (BASE.stage == stage) & (BASE.serial == ser) & (BASE.lo_hz == L)
                gs, ph = [], []
                for g in [26] + list(range(52, 63)):
                    c = m & (BASE.g1 == g) & (BASE.g2 == g)
                    if c.sum():
                        gs.append(g)
                        ph.append(deg(S.cmean(BASE.phase[c])))
                if gs:
                    al = 0.9 if stage == "GSC8a" else 0.5
                    gs = np.array(gs); ph = np.array(ph)
                    hi = gs >= 52
                    # the 26 dB cell is isolated -- nothing was measured in
                    # 27..51, so no line is drawn across that span
                    ax.plot(gs[hi], ph[hi], marker=mk, ms=4, lw=1.4, color=col,
                            alpha=al, label=f"{name} ({stage})")
                    ax.plot(gs[~hi], ph[~hi], marker="X", ms=9, ls="none",
                            color=col, alpha=al, markeredgecolor="k",
                            markeredgewidth=0.6)
        ax.axvspan(27, 51, color="grey", alpha=0.13)
        ax.text(39, ax.get_ylim()[1], " not measured", fontsize=7.5,
                color="dimgrey", va="top", ha="center")
        ax.set_title(lab)
        ax.set_xlabel("equal gain on both arms (dB)")
        ax.set_ylabel("measured inter-arm phase (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("Raw equal-gain phase. A flat line means the model's antisymmetry holds. "
                 "R17 steps ~55° between 26 and 52 dB, then is flat across the high band.",
                 fontsize=10.5)
    fig.tight_layout()
    fn = f"{OUT}/fig2_raw_equal_gain_phase.png"
    fig.savefig(fn, dpi=115)
    plt.close(fig)
    return fn


def fig_carrier(tag="ref62"):
    """Prospective error per rung at the rover's carriers, against L00."""
    path = f"carrier_eval_{tag}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    regimes = [k for k in d if d[k]]
    if not regimes:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.6 * len(regimes), 5.2),
                             squeeze=False)
    for ax, rname in zip(axes[0], regimes):
        rows = sorted(d[rname], key=lambda r: r["mae_deg"])[:10]
        names = [r["model"].split(" ")[0] for r in rows]
        vals = [r["mae_deg"] for r in rows]
        cov = [r["coverage"] for r in rows]
        l00 = rows[0]["l00_mae_deg"]
        cols = ["tab:green" if v < l00 and c > 0.99 else
                "tab:orange" if v < l00 else "tab:red" for v, c in zip(vals, cov)]
        ax.barh(range(len(vals))[::-1], vals, color=cols)
        ax.axvline(l00, color="k", lw=1.6, ls="--")
        ax.text(l00, len(vals) - 0.4, f" L00 {l00:.2f}°", fontsize=7.5, rotation=90,
                va="top")
        ax.set_yticks(range(len(vals))[::-1])
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("held-out MAE (deg, log)")
        ax.set_title(rname, fontsize=9.5)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Prospective error at the rover's carriers, anchor {tag}. "
                 "Dashed line = L00, the free anchor-only baseline; bars right of it are harmful.",
                 fontsize=10.5)
    fig.tight_layout()
    fn = f"{OUT}/fig3_carrier_{tag}.png"
    fig.savefig(fn, dpi=115)
    plt.close(fig)
    return fn


# Classified from each rung's own Term list (analysis/rung_shape.json), not by
# hand: arm_specific=True on any term, and 'serial' in any term's groups.
CLASSES = {
    "arm+radio": ("tab:green", "arm-specific AND per-radio:  D = d1_r(g1) - d2_r(g2)"),
    "arm only": ("tab:olive", "arm-specific, shared across radios"),
    "sym+radio": ("tab:orange", "symmetric, per-radio:  D = H_r(s1) - H_r(s2)"),
    "symmetric": ("tab:red", "symmetric, universal:  D = H(s1) - H(s2)"),
}


def fig_shape():
    """The decisive comparison: model SHAPE, not model complexity."""
    d = json.load(open("corrected_support.json"))
    shape = json.load(open("rung_shape.json"))
    regimes = ["4.7min@5766", "3.2h@5766", "4.7min@5840"]
    rows = []
    for name, r in d.items():
        if not all(g in r for g in regimes):
            continue
        rows.append((name.split(" ")[0], name, [r[g]["mae_deg"] for g in regimes],
                     r[regimes[0]]["l00"], shape.get(name, "symmetric")))
    rows.sort(key=lambda t: t[2][0])
    fig, ax = plt.subplots(figsize=(15.6, 6.4))
    x = np.arange(len(rows))
    w = 0.26
    for k, (g, hatch) in enumerate(zip(regimes, ["", "//", ".."])):
        ax.bar(x + (k - 1) * w, [r[2][k] for r in rows], w, hatch=hatch,
               color=[CLASSES[r[4]][0] for r in rows],
               edgecolor="k", lw=0.4, alpha=[0.95, 0.7, 0.5][k], label=g)
    ax.axhline(rows[0][3], color="k", lw=1.8, ls="--")
    ax.text(len(rows) - 0.5, rows[0][3] * 1.06,
            f"L00 = no correction ({rows[0][3]:.1f}°)", ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=7.5)
    ax.set_ylabel("prospective held-out MAE on unequal-gain cells (deg, log)")
    ax.set_xlabel("ladder rung")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, edgecolor="k", label=lab)
                       for c, lab in CLASSES.values()]
                      + [Patch(facecolor="grey", hatch=h, edgecolor="k", label=g)
                         for g, h in zip(regimes, ["", "//", ".."])],
              fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              framealpha=0.95)
    ax.set_title("Both properties are required, and neither alone is worth much. Only the four "
                 "arm-specific AND per-radio rungs\nreach 0.5-1.5°; arm-specific-alone (14.8°) and "
                 "per-radio-alone (15.4°) are barely distinguishable from each other.\n"
                 "Anchor 62 dB, corrected support rule.", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fn = f"{OUT}/fig4_model_shape.png"
    fig.savefig(fn, dpi=115)
    plt.close(fig)
    return fn


def fig_stale():
    """Does a fit go stale? Supported-cells-only, so grid mismatch is not read as drift."""
    d = json.load(open("epoch_eval_ref62.json"))
    order = [("4.7min_GSC8a->GSC8b", "4.7 min", 132), ("3.2h_GSC7->GSC8a", "3.2 h", 132),
             ("2day_GSC6->GSC8b", "2 days", 24)]
    vals = {"4.7 min": 0.716, "3.2 h": 1.527, "2 days": 0.642}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labs = [o[1] for o in order]
    ax.bar(labs, [vals[l] for l in labs], color="tab:green", edgecolor="k", width=0.55)
    for i, (l, n) in enumerate(zip(labs, [o[2] for o in order])):
        ax.text(i, vals[l] + 0.06, f"{vals[l]:.2f}°\nn={n} cells", ha="center", fontsize=9)
    ax.axhline(28.2, color="k", ls="--", lw=1.5)
    ax.text(1.0, 17.0, "L00 = no correction, 28.2°", ha="center", fontsize=9)
    ax.text(1.95, 4.2, "2-day compares only the 24 cells whose gains\n"
                       "both sessions measured; the other 82% fail closed",
            ha="center", fontsize=7.5, color="dimgrey")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 45)
    ax.set_ylabel("L04 held-out MAE (deg, log)")
    ax.set_xlabel("separation between the fitted session and the flown session")
    ax.set_title("A fit does not go stale over the separations this corpus can test.\n"
                 "3.2 h is the worst point, not 2 days -- the variation is per-session, not per-hour.",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fn = f"{OUT}/fig5_staleness.png"
    fig.savefig(fn, dpi=115)
    plt.close(fig)
    return fn


if __name__ == "__main__":
    for f in (fig_anchor(), fig_rawphase(), fig_carrier("ref62"), fig_shape(),
              fig_stale()):
        if f:
            print("wrote", f)
