"""Additional figures.

fig4  does the recommended model actually work, and where does it fail
fig5  the RF-DC discriminator that already exists inside the excluded F_neg stage
fig6  how much of H is really shared between the two radios
fig7  the 27 fitted parameters, made concrete
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features as FT  # noqa: E402
import ladder as LD  # noqa: E402
import spflib as S  # noqa: E402
from diag_gainsteps import common_H  # noqa: E402
from diag_Hspectrum import build_H  # noqa: E402
from models import build_design  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
C17, C18 = "#1f77b4", "#d62728"
BANDC = {0: "#4c78a8", 1: "#f58518", 2: "#e45756"}
BANDN = {0: "low ≤1300", 1: "middle 1301–4000", 2: "high >4000"}


def _rung(key):
    return [m for m in LD.make_ladder() if m.name.split()[0] == key][0]


def _lofo(f, m):
    d = build_design(f, m.terms)
    preds = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    for _l, tr, te in LD.splits_leave_one_frequency_out(f):
        tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
        p, sp, _t, _n = m.fit_eval(d, f.D, tri, tei)
        preds[tei] = p
        sup[tei] = sp
    return np.where(sup, preds, 0.0)


def fig4(out="fig4_performance.png"):
    f = FT.add_anchor(S.load_stages(["A"]), ref=26)
    uneq = f.g1 != f.g2
    pred26 = _lofo(f, _rung("L26"))
    pred27 = _lofo(f, _rung("L27"))
    pred24 = _lofo(f, _rung("L24"))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))

    # (a) predicted vs measured, unequal-gain cells only
    ax = axes[0]
    lim = 45
    ax.plot([-lim, lim], [-lim, lim], "-", color="0.5", lw=0.9, zorder=1)
    for b in (0, 1, 2):
        k = uneq & (f.band == b)
        ax.scatter(np.degrees(f.D[k]), np.degrees(pred26[k]), s=5,
                   color=BANDC[b], alpha=0.45, edgecolors="none",
                   label=f"{BANDN[b]} MHz")
    e = S.wrap(f.D[uneq] - pred26[uneq])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("measured gain effect D (deg)")
    ax.set_ylabel("predicted by L26, frequency held out (deg)")
    ax.set_title("(a) 27 universal parameters, on a frequency\n"
                 "     the model never saw", fontsize=9)
    ax.legend(fontsize=6.4, loc="upper left")
    ax.text(0.97, 0.06, f"MAE {S.cmae_deg(e):.2f}°\nP95 {S.cp95_deg(e):.2f}°\n"
                        f"n = {int(uneq.sum())}",
            transform=ax.transAxes, ha="right", fontsize=7.2)

    # (b) residual vs frequency
    ax = axes[1]
    los = np.unique(f.lo_hz)
    # the per-frequency LUT is UNSUPPORTED at an unseen frequency, so it fails
    # closed to the anchor and its curve is identical to "no correction" --
    # plotting it as a competitor would be misleading
    assert np.allclose(pred24, 0.0)
    for lbl, pred, col in (
            ("no correction — and the 1356-param per-frequency LUT,\n"
             "which fails closed here", np.zeros(len(f)), "0.45"),
            ("L26, 27 params", pred26, "#e45756"),
            ("L27, 49 params", pred27, "#4c78a8")):
        y = [S.cmae_deg(S.wrap(f.D[uneq & (f.lo_hz == lo)]
                               - pred[uneq & (f.lo_hz == lo)])) for lo in los]
        ax.plot(los / 1e6, y, "-", color=col, lw=1.1, label=lbl)
    for edge in (1300, 4000):
        ax.axvline(edge, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("RX LO frequency (MHz)")
    ax.set_ylabel("MAE at that LO (deg, log)")
    ax.set_title("(b) where the error lives: above 4 GHz, and at the\n"
                 "     5100/5400 MHz cells the prior reports also flag", fontsize=9)
    ax.legend(fontsize=6.0, loc="lower right", ncol=1)

    # (c) error CDF
    ax = axes[2]
    for lbl, pred, col in (("no correction (= per-frequency LUT,\n"
                            "unsupported at an unseen LO)", np.zeros(len(f)), "0.45"),
                           ("L01 H(g), 3 params", None, None),
                           ("L26, 27 params", pred26, "#e45756"),
                           ("L27, 49 params", pred27, "#4c78a8")):
        if pred is None:
            pred = _lofo(f, _rung("L01"))
            col = "#f58518"
        v = np.sort(np.abs(np.degrees(S.wrap(f.D[uneq] - pred[uneq]))))
        ax.plot(v, np.linspace(0, 100, v.size), "-", color=col, lw=1.2, label=lbl)
    ax.axhline(95, color="0.4", ls="--", lw=0.8)
    ax.text(31, 96, "P95", fontsize=6.5, color="0.4")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 100)
    ax.set_xlabel("|error| on an unmeasured frequency (deg)")
    ax.set_ylabel("percent of cells below")
    ax.set_title("(c) the whole error distribution, not just the mean",
                 fontsize=9)
    ax.legend(fontsize=6.2, loc="lower right")

    fig.suptitle("Leave-one-frequency-out performance on stage A, unequal-gain "
                 "cells only (the cells a correction acts on)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    print("wrote", out)


def fig5(out="fig5_rfdc_discriminator.png"):
    tab = FT.HW.tab["high"]
    res = common_H("F_neg", 5)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    for serial, col, lbl in ((S.TREATED, C17, "R17"), (S.CONTROL, C18, "R18")):
        for f_hz, ls in ((5766000000, "-"), (5866000000, "--")):
            if f_hz not in res[serial]:
                continue
            gs, H, _ = res[serial][f_hz]
            k = gs <= 10
            ax.plot(gs[k], np.degrees(H[k]), ls, marker="o", ms=3, lw=1.0,
                    color=col, alpha=0.9 if ls == "-" else 0.55,
                    label=f"{lbl} {f_hz/1e6:.0f} MHz")
    ymin, ymax = ax.get_ylim()
    for g in range(-10, 11):
        r = int(tab.row_for_gain(np.array(g)))
        rp = int(tab.row_for_gain(np.array(g - 1)))
        if r < 0 or rp < 0:
            continue
        lmt = (tab.lna[r], tab.mixer[r], tab.tia[r])
        lmtp = (tab.lna[rp], tab.mixer[rp], tab.tia[rp])
        sampled = any(g in set(res[sv][fz][0].tolist())
                      for sv in res for fz in res[sv])
        if lmt != lmtp:
            ax.axvline(g - 0.5, color="#e45756", lw=1.4)
            ax.text(g - 0.6, ymax * 0.97,
                    f"MIX {lmtp[1]}→{lmt[1]}", fontsize=6.4, rotation=90,
                    color="#e45756", va="top", ha="right", fontweight="bold")
        elif tab.rfdc[r] and not tab.rfdc[rp]:
            ax.axvline(g - 0.5, color="#4c78a8", lw=1.4, ls="-.")
            tag = "RF-DC only" if sampled else "RF-DC only\n(never sampled)"
            ax.text(g - 0.15, ymin * 0.95, tag, fontsize=6.4, rotation=90,
                    color="#4c78a8", va="bottom", ha="left")
    ax.set_xlabel("requested RX gain (dB), high gain table")
    ax.set_ylabel("common gain response H (deg)")
    ax.set_ylim(ymin * 1.25, ymax * 1.35)
    ax.set_title("(a) stage F_neg, high band: at −3 dB the gain table sets RF_DC_CAL\n"
                 "     with the LNA/mixer/TIA words unchanged — and nothing happens",
                 fontsize=9)
    ax.legend(fontsize=6.2, ncol=2, loc="upper left")

    ax = axes[1]
    rows = []
    for serial, tabd in res.items():
        for f_hz, (gs, H, _A) in tabd.items():
            if int(S.gain_band(np.array([f_hz]))[0]) != 2:
                continue
            for i in range(1, len(gs)):
                if gs[i] - gs[i - 1] != 1:
                    continue
                s0 = FT.HW.state(2, int(gs[i - 1]))
                s1 = FT.HW.state(2, int(gs[i]))
                if s0 is None or s1 is None:
                    continue
                d = abs(np.degrees(S.wrap(H[i] - H[i - 1])))
                r1 = int(tab.rfdc[int(tab.row_for_gain(np.array(int(gs[i]))))])
                r0 = int(tab.rfdc[int(tab.row_for_gain(np.array(int(gs[i - 1]))))])
                if s0[:3] != s1[:3]:
                    rows.append(("LMT change\n(MIX 0→1)", d, "#e45756"))
                elif r1 != r0:
                    rows.append(("RF_DC_CAL only\nLMT frozen", d, "#4c78a8"))
                else:
                    rows.append(("LPF word only\nRF_DC frozen", d, "#72b7b2"))
    order = ["LPF word only\nRF_DC frozen", "RF_DC_CAL only\nLMT frozen",
             "LMT change\n(MIX 0→1)"]
    rng = np.random.default_rng(1)
    ax.axhspan(0, 0.36, color="0.85", alpha=0.6, zorder=0)
    ax.text(2.45, 0.19, "measurement floor", fontsize=6.2, ha="right", color="0.3")
    labels = []
    for k, name in enumerate(order):
        v = np.array([d for n, d, _ in rows if n == name])
        col = [c for n, _, c in rows if n == name][0]
        labels.append(f"{name}\n(n={v.size})")
        ax.scatter(k + rng.normal(0, 0.07, v.size), np.clip(v, 3e-3, None),
                   s=12, color=col, alpha=0.7, edgecolors="none")
        ax.hlines(np.median(v), k - 0.3, k + 0.3, color="k", lw=2, zorder=5)
        ax.text(k, np.median(v) * 1.6, f"{np.median(v):.2f}°", ha="center",
                fontsize=7.4, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 20)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("|ΔH| for a 1 dB step (deg)")
    ax.set_title("(b) re-running the RF-DC correction alone costs nothing;\n"
                 "     changing the mixer word costs 4.4°", fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.suptitle("The RF-DC / RF-state confound is already half-broken by data the "
                 "campaign discarded", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out)
    print("wrote", out)


def fig6(out="fig6_cross_radio.png"):
    res = build_H("A")
    a, b = res[S.TREATED], res[S.CONTROL]
    lo = a["lo"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharex=True, sharey=True)
    for ax, g in zip(axes, (45, 5)):
        x, y = np.degrees(a[f"H{g}"]), np.degrees(b[f"H{g}"])
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "-", color="0.5", lw=0.9)
        for bi in (0, 1, 2):
            m = (lo <= 1300e6) if bi == 0 else (
                (lo > 1300e6) & (lo <= 4000e6)) if bi == 1 else (lo > 4000e6)
            r = np.corrcoef(x[m], y[m])[0, 1]
            ax.scatter(x[m], y[m], s=14, color=BANDC[bi], alpha=0.7,
                       edgecolors="none",
                       label=f"{BANDN[bi]}  ρ={r:.3f}")
        ax.set_xlabel("H on R17 (deg)")
        ax.set_title(f"g = {g} dB vs the 26 dB reference\n"
                     f"overall ρ = {np.corrcoef(x, y)[0,1]:.3f}", fontsize=9)
        ax.legend(fontsize=6.6, loc="upper left")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("H on R18 (deg)")
    fig.suptitle("How universal is H really? Tight below 4 GHz where the effect is "
                 "large; scattered above it, where the ripple —\nand therefore each "
                 "unit's own harness termination — dominates",
                 fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)
    print("wrote", out)


def fig7(out="fig7_parameters.png"):
    f = FT.add_anchor(S.load_stages(["A", "F", "E_tx_0", "rate_pilot"]), ref=None)
    m = _rung("L26")
    d = build_design(f, m.terms)
    idx = np.arange(len(f))
    _p, _s, taus, _n = m.fit_eval(d, f.D, idx, idx[:1])
    X = d.matrix(taus)
    active = np.any(np.abs(d.S) > 0, axis=0)
    theta = m._solve(X[:, active], f.D)
    names = [n for n, a in zip(d.names, active) if a]

    fam = {"lna|const": [], "mixer|const": [], "tia|const": [], "lpf|const": [],
           "ripple1": [], "ripple2": []}
    for n, t in zip(names, np.degrees(theta)):
        field = n.split("=")[0]
        basis = n.split("|")[2]
        lvl = int(n.split("=")[1].split("|")[0])
        if basis == "const":
            fam[f"{field}|const"].append((lvl, t))
        elif basis in ("cos0", "sin0"):
            fam["ripple1"].append((lvl, basis, t))
        elif basis in ("cos1", "sin1"):
            fam["ripple2"].append((lvl, basis, t))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    ax = axes[0]
    for key, col, lbl in (("lna|const", "#e45756", "LNA index"),
                          ("mixer|const", "#f58518", "mixer index"),
                          ("tia|const", "#72b7b2", "TIA index")):
        v = sorted(fam[key])
        ax.plot([x for x, _ in v], [y for _, y in v], "o-", color=col, ms=5,
                lw=1.2, label=f"{lbl} ({len(v)} params)")
    ax.set_xlabel("audited state index")
    ax.set_ylabel("fitted lumped phase (deg)")
    ax.set_xticks(range(5))
    ax.axvspan(0.6, 1.4, color="#fde", alpha=0.5, zorder=0)
    ax.text(1.0, ax.get_ylim()[0] * 0.72, "LNA index 1\nnever measured\n(E-CAL2)",
            fontsize=6.2, ha="center", color="#a00", style="italic")
    ax.set_title("(a) the RF-state terms carry the signal", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper left")

    ax = axes[1]
    v = sorted(fam["lpf|const"])
    ax.plot([x for x, _ in v], [y for _, y in v], "o-", color="#4c78a8", ms=4,
            lw=1.0, label=f"LPF word ({len(v)} params)")
    ax.axhline(0, color="0.5", lw=0.8)
    sd = np.std([y for _, y in v])
    ax.axhspan(-sd, sd, color="0.85", alpha=0.6, zorder=0)
    ax.text(0.5, 0.04, f"scatter ±{sd:.2f}° about zero, no trend —\n"
                       "this is why deployment rule 5 exists",
            transform=ax.transAxes, ha="center", fontsize=6.8, color="0.3")
    ax.set_xlabel("baseband LPF gain word")
    ax.set_ylabel("fitted lumped phase (deg)")
    ax.set_title("(b) the baseband term fits noise", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper left")

    ax = axes[2]
    w = 0.35
    for k, (key, tau, col) in enumerate((("ripple1", taus[0], "#e45756"),
                                         ("ripple2", taus[1], "#4c78a8"))):
        by = {}
        for lvl, basis, t in fam[key]:
            by.setdefault(lvl, {})[basis] = t
        lv = sorted(by)
        amp = [np.hypot(by[l].get(f"cos{k}", 0), by[l].get(f"sin{k}", 0))
               for l in lv]
        ax.bar(np.array(lv) + (k - 0.5) * w, amp, w, color=col,
               label=f"τ{k+1} = {tau*1e9:.2f} ns")
    ax.set_xlabel("LNA index")
    ax.set_ylabel("ripple amplitude (deg)")
    ax.set_xticks([0, 1, 2, 3])
    ax.axvspan(0.6, 1.4, color="#fde", alpha=0.5, zorder=0)
    ax.text(1.0, ax.get_ylim()[1] * 0.5, "never\nmeasured", fontsize=6.2,
            ha="center", color="#a00", style="italic")
    ax.set_title("(c) ripple amplitude per LNA state\n     — the mechanism, fitted",
                 fontsize=9)
    ax.legend(fontsize=6.6)

    fig.suptitle(f"The recommended model's parameters, fitted on all {len(f)} "
                 f"pooled cells ({int(active.sum())} columns here; 27 on stage A "
                 f"alone, where fewer states occur)", fontsize=10.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig5()
    fig6()
    fig7()
    fig4()
