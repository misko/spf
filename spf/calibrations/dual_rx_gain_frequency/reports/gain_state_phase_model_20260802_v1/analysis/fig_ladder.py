"""Ladder summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# results live beside the report when run from analysis/, else in cwd
_HERE = Path(__file__).resolve().parent
_R = _HERE.parent if (_HERE.parent / "ladder_results_A_main.json").exists() else Path(".")
MAIN = json.load(open(_R / "ladder_results_A_main.json"))
COMB = json.load(open(_R / "comb_results.json"))

SPLIT_LABEL = {
    "LOEO leave-one-epoch-out": ("repeat of a measured cell (LOEO)", "#4c78a8"),
    "LOFO leave-one-frequency-out": ("unmeasured frequency, 50 MHz neighbours "
                                     "kept (LOFO)", "#f58518"),
    "LOBLOCK leave-frequency-block-out": ("unmeasured 690 MHz frequency block",
                                          "#e45756"),
    "LORO leave-one-radio-out": ("unmeasured radio (LORO)", "#54a24b"),
}
HILITE = {"L00": "anchor only", "L01": "H(g)", "L08": "H(band,g)",
          "L16": "MECH 1 ripple", "L26": "MECH 2 ripples",
          "L27": "MECH +delay", "L24": "per-freq LUT"}


def main(out="fig3_ladder.png"):
    fig = plt.figure(figsize=(12.0, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.05, 1.0], wspace=0.34)

    # (a) MAE vs parameter count
    ax = fig.add_subplot(gs[0, 0])
    for split, (lbl, col) in SPLIT_LABEL.items():
        rows = [r for r in MAIN[split] if r["coverage"] > 0.99]
        p = np.array([max(r["params"], 0.7) for r in rows])
        e = np.array([r["mae_deg"] for r in rows])
        o = np.argsort(p)
        ax.plot(p[o], e[o], "o", ms=4, color=col, alpha=0.75, label=lbl)
        # lower envelope
        best, xs, ys = np.inf, [], []
        for pi, ei in zip(p[o], e[o]):
            if ei < best:
                best = ei
                xs.append(pi)
                ys.append(ei)
        ax.plot(xs, ys, "-", color=col, lw=1.2, alpha=0.8)
    base = MAIN["LOEO leave-one-epoch-out"][0]["mae_deg"]
    ax.axhline(base, color="0.4", ls="--", lw=1.0)
    ax.text(1.1, base * 1.05, "no gain correction (per-session anchor only)",
            fontsize=6.8, color="0.35")
    for r in MAIN["LOEO leave-one-epoch-out"]:
        k = r["model"].split()[0]
        if k in HILITE and r["coverage"] > 0.99:
            ax.annotate(HILITE[k], (max(r["params"], 0.7), r["mae_deg"]),
                        textcoords="offset points", xytext=(5, -9), fontsize=6.2,
                        color="#333")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fitted parameters (log)")
    ax.set_ylabel("held-out circular MAE (deg, log)")
    ax.set_title("(a) the ladder: accuracy vs parameter count\n"
                 "     only models with 100% coverage shown", fontsize=9)
    ax.legend(fontsize=6.3, loc="lower left")
    ax.set_yticks([0.6, 1, 2, 3, 5, 7])
    ax.set_yticklabels(["0.6", "1", "2", "3", "5", "7"])

    # (b) comb spacing
    ax = fig.add_subplot(gs[0, 1])
    widths = [2750, 1375, 688, 344, 172, 96]
    keys = ["2blocks", "4blocks", "8blocks", "16blocks", "32blocks", "57blocks"]
    for name, col, mk in (("L08 sym H(band,g)", "#9e9e9e", "s"),
                          ("L16 MECH H(state)+ripple/LNA", "#f58518", "^"),
                          ("L26 MECH H(state)+2 ripples/LNA", "#e45756", "o"),
                          ("L27 MECH +delay +2 ripples/(band,LNA)", "#4c78a8", "v")):
        y = [COMB["comb"][name][k] for k in keys]
        ax.plot(widths, y, mk + "-", color=col, ms=4, lw=1.2,
                label=name.split(" ", 1)[1][:26])
    ax.axhline(COMB["comb"]["L00 anchor only"]["8blocks"], color="0.4", ls="--",
               lw=1.0)
    ax.text(100, 6.9, "no gain correction", fontsize=6.5, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("width of the held-out frequency gap (MHz)")
    ax.set_ylabel("held-out MAE (deg)")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths], fontsize=6.8)
    ax.set_title("(b) how coarse can the calibration comb be?\n"
                 "     error is flat from 96 to ~690 MHz gaps", fontsize=9)
    ax.legend(fontsize=6.2, loc="upper center", ncol=1)
    ax.set_ylim(1.5, 9.6)

    # (c) radio specificity
    ax = fig.add_subplot(gs[0, 2])
    rs = COMB["radio_specificity"]
    names = list(rs)
    x = np.arange(len(names))
    loeo = [rs[n]["loeo"]["mae_deg"] for n in names]
    loro = [rs[n]["loro"]["mae_deg"] for n in names]
    cov = [rs[n]["loro"]["coverage"] for n in names]
    ax.bar(x - 0.2, loeo, 0.38, color="#4c78a8", label="same radios (LOEO)")
    bars = ax.bar(x + 0.2, loro, 0.38, color="#54a24b", label="unseen radio (LORO)")
    for xi, (b, c) in enumerate(zip(bars, cov)):
        if c < 0.5:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    "no\ncoverage", ha="center", fontsize=5.8, color="#a00")
    ax.set_xticks(x)
    ax.set_xticklabels(["all\nuniversal", "static H", "delay", "ripple",
                        "everything"], fontsize=6.6)
    ax.set_xlabel("which family is made per-radio", fontsize=7.5)
    ax.set_ylabel("held-out MAE (deg)")
    ax.set_ylim(0, 8.6)
    ax.set_title("(c) nothing needs to be radio-specific\n"
                 "     per-radio terms add zero and break transfer", fontsize=9)
    ax.legend(fontsize=6.4, loc="upper center")
    ax.grid(axis="x", visible=False)

    fig.suptitle("A physically-parameterised gain model: 6.65° → 2.26° on an "
                 "unmeasured frequency, with 27 universal parameters",
                 fontsize=10.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
