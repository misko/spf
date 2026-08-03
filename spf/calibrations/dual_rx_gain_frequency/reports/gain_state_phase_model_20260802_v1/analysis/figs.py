"""Figures for the AD9361 gain/frequency phase investigation."""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features as FT  # noqa: E402
import spflib as S  # noqa: E402
from diag_gainsteps import common_H  # noqa: E402
from diag_Hspectrum import build_H  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

C17, C18 = "#1f77b4", "#d62728"
BAND_EDGE = (1300.0, 4000.0)


def fig1_mechanism(out="fig1_mechanism.png"):
    res = build_H("A")
    a, b = res[S.TREATED], res[S.CONTROL]
    lo = a["lo"] / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6))

    # (a) H(f) at g=45 and g=5
    ax = axes[0, 0]
    for g, ls in ((45, "-"), (5, "--")):
        ax.plot(lo, np.degrees(a[f"H{g}"]), ls, color=C17, lw=1.1,
                label=f"R17  g={g} dB")
        ax.plot(lo, np.degrees(b[f"H{g}"]), ls, color=C18, lw=1.1, alpha=0.75,
                label=f"R18  g={g} dB")
    ax.set_ylim(-27, 27)
    for e in BAND_EDGE:
        ax.axvline(e, color="k", lw=0.8, ls=":")
        ax.text(e + 60, 25, "gain-table\nedge", fontsize=6.2, va="top", ha="left",
                color="0.3")
    ax.set_xlabel("RX LO frequency (MHz)")
    ax.set_ylabel("common gain response H (deg)")
    ax.set_title("(a) H(f,g) = [D(g,26) − D(26,g)]/2 — the part both RX arms share\n"
                 "     R17 and R18 overlie each other: this is a chip-level curve",
                 fontsize=9)
    ax.legend(ncol=2, fontsize=7, loc="lower left")

    # (b) arm asymmetry
    ax = axes[0, 1]
    for g, ls in ((45, "-"), (5, "--")):
        ax.plot(lo, np.degrees(a[f"A{g}"]), ls, color=C17, lw=1.0,
                label=f"R17  g={g}")
        ax.plot(lo, np.degrees(b[f"A{g}"]), ls, color=C18, lw=1.0, alpha=0.75,
                label=f"R18  g={g}")
    for e in BAND_EDGE:
        ax.axvline(e, color="k", lw=0.8, ls=":")
    ax.set_xlabel("RX LO frequency (MHz)")
    ax.set_ylabel("arm asymmetry A (deg)")
    ax.set_ylim(-27, 27)
    ax.set_title("(b) A(f,g) = D(g,26) + D(26,g) — zero if the arms respond\n"
                 "     identically to gain. Same y-scale as (a): 1.3–6.0% of energy",
                 fontsize=9)
    ax.legend(ncol=2, fontsize=7, loc="lower left")

    # (c) staircase at 5766 MHz from stage F + E
    ax = axes[1, 0]
    resF = common_H("F", 5)
    resE = common_H("E_tx_0", 33)
    for serial, col, lbl in ((S.TREATED, C17, "R17"), (S.CONTROL, C18, "R18")):
        gsF, HF, _ = resF[serial][5766000000]
        gsE, HE, _ = resE[serial][5766000000]
        # align the two sweeps at their overlap-free reference by anchoring E
        ax.plot(gsF, np.degrees(HF), "o-", color=col, ms=3, lw=1.0,
                label=f"{lbl} stage F (ref 5 dB)")
        ax.plot(gsE, np.degrees(HE) + np.degrees(HF[-1]), "s--", color=col,
                ms=3, lw=1.0, alpha=0.6, label=f"{lbl} stage E (ref 33 dB)")
    tab = FT.HW.tab["high"]
    prev = None
    for g in range(-1, 41):
        r = int(tab.row_for_gain(np.array(g)))
        if r < 0:
            continue
        st = (int(tab.lna[r]), int(tab.mixer[r]), int(tab.tia[r]))
        if prev is not None and st != prev[1]:
            ax.axvline(g - 0.5, color="0.35", lw=0.8, ls="-", alpha=0.7)
            names = []
            if st[0] != prev[1][0]:
                names.append(f"LNA{prev[1][0]}→{st[0]}")
            if st[1] != prev[1][1]:
                names.append(f"MIX{prev[1][1]}→{st[1]}")
            if st[2] != prev[1][2]:
                names.append(f"TIA{prev[1][2]}→{st[2]}")
            ax.text(g - 0.35, -6.6, " ".join(names), fontsize=5.8, rotation=90,
                    va="bottom", ha="center", color="0.25")
        prev = (g, st)
    ax.axvspan(26.5, 40.5, color="0.85", alpha=0.45, zorder=0)
    ax.text(33.5, 1.2, "no RF-state change in 27–40 dB:\n"
                       "13 dB of gain costs < 1° of phase",
            fontsize=6.6, ha="center", va="center", color="0.25")
    ax.set_xlabel("requested RX gain (dB)")
    ax.set_ylabel("common gain response H (deg)")
    ax.set_ylim(-7, 17)
    ax.set_title("(c) 5766 MHz: H steps only where the audited gain table\n"
                 "     changes an RF state (vertical lines)", fontsize=9)
    ax.legend(fontsize=6.2, loc="upper left", ncol=2)

    # (d) 1 dB step magnitude by cause
    ax = axes[1, 1]
    hw, lpf = [], []
    for stage, ref in (("F", 5), ("E_tx_0", 33)):
        r = common_H(stage, ref)
        for serial, tabd in r.items():
            for f_hz, (gs, H, _A) in tabd.items():
                bi = int(S.gain_band(np.array([f_hz]))[0])
                for i in range(1, len(gs)):
                    if gs[i] - gs[i - 1] != 1:
                        continue
                    s0 = FT.HW.state(bi, int(gs[i - 1]))
                    s1 = FT.HW.state(bi, int(gs[i]))
                    if s0 is None or s1 is None:
                        continue
                    d = abs(np.degrees(S.wrap(H[i] - H[i - 1])))
                    rf = s0[:3] != s1[:3]
                    (hw if rf else lpf).append(d)
    hw, lpf = np.array(hw), np.array(lpf)
    rng = np.random.default_rng(0)
    for k, (arr, lbl, col) in enumerate((
            (lpf, f"baseband LPF word only\n(n={len(lpf)})", "#4c78a8"),
            (hw, f"RF state change\nLNA / mixer / TIA  (n={len(hw)})", "#e45756"))):
        x = k + rng.normal(0, 0.06, len(arr))
        ax.scatter(x, arr, s=9, color=col, alpha=0.6, edgecolors="none")
        ax.hlines(np.median(arr), k - 0.28, k + 0.28, color="k", lw=2, zorder=5)
        ax.text(k, np.median(arr) * 1.35, f"median {np.median(arr):.2f}°",
                ha="center", fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"baseband LPF word only\n(n={len(lpf)})",
                        f"RF state change\nLNA / mixer / TIA  (n={len(hw)})"],
                       fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("|ΔH| for a 1 dB gain step (deg)")
    ax.set_title("(d) what actually moves the phase: the RF state, not the dB",
                 fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.suptitle(
        "AD9361 dual-RX phase: the gain effect is antisymmetric between arms "
        "and is driven by discrete RF-state changes",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    print("wrote", out)


def fig2_ripple(out="fig2_ripple.png"):
    """Ripple amplitude tracks the LNA-state change predicted by the audited
    gain table, band by band."""
    res = build_H("A")
    lo = res[S.TREATED]["lo"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    bands = (("low ≤1300 MHz", lo <= 1300e6, 0),
             ("middle 1301–4000 MHz", (lo > 1300e6) & (lo <= 4000e6), 1),
             ("high >4000 MHz", lo > 4000e6, 2))
    for ax, (title, mask, bi) in zip(axes, bands):
        for serial, col, lbl in ((S.TREATED, C17, "R17"), (S.CONTROL, C18, "R18")):
            for g, ls in ((5, "--"), (45, "-")):
                H = res[serial][f"H{g}"][mask]
                x = lo[mask] / 1e6
                y = np.degrees(H)
                y = y - np.polyval(np.polyfit(x, y, 1), x)
                st_ref = FT.HW.state(bi, 26)
                st_g = FT.HW.state(bi, g)
                dlna = st_g[0] - st_ref[0]
                ax.plot(x, y, ls, color=col, lw=1.0, alpha=0.85,
                        label=f"{lbl} g={g}  ΔLNA={dlna:+d}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("RX LO frequency (MHz)")
        ax.set_ylim(-20, 22)
        ax.legend(fontsize=6.2, ncol=2, loc="upper left")
    axes[0].set_ylabel("H with linear trend removed (deg)")
    fig.suptitle(
        "Ripple amplitude follows the LNA-state change the audited gain table makes, "
        "not the requested dB\n"
        "(the same requested gain maps to a different LNA index in each band; "
        "ΔLNA = 0 → flat,   |ΔLNA| ≥ 1 → strong ripple)",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig1_mechanism()
    fig2_ripple()
