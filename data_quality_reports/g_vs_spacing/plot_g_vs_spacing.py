"""Per-frequency scatter of fitted g vs configured spacing + physics model fits.

Each dot = one wall dataset (r0 and r1 fits shown separately). Overlays:
  - g = 1 (config correct)
  - g = (lambda/2)/d  ("effective spacing pinned at half-wavelength")
  - two-element mutual-coupling model fit (see below)

Coupling model: each channel receives its own signal plus a coupled copy of the
neighbour's, V0 = 1 + C e^{j phi}, V1 = e^{j phi} + C, with free-space-like
coupling  C(d) = A e^{j(psi0 - k d)} / (k d),  k = 2 pi / lambda.
The measured inter-channel phase is arg(V1 V0*); the model's g is the ratio of
its phase swing to the ideal swing (slope at broadside). Two real params (A, psi0)
per band.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
C_LIGHT = 3e8


def model_g(d, lam, A, psi0):
    """Phase-swing gain of the coupled two-element array at spacing d."""
    k = 2 * np.pi / lam
    C = A * np.exp(1j * (psi0 - k * d)) / (k * d)
    eps = 1e-4  # small ideal phase; slope at broadside
    v0 = 1 + C * np.exp(1j * eps)
    v1 = np.exp(1j * eps) + C
    return np.angle(v1 * np.conj(v0)) / eps


def fit_band(dmed, gmed, wts, lam):
    def resid(p):
        A, psi0 = p
        return (model_g(dmed, lam, A, psi0) - gmed) * np.sqrt(wts)

    best = None
    for psi0 in np.linspace(-np.pi, np.pi, 13):
        for A in (0.2, 0.5, 1.0, 2.0):
            try:
                r = least_squares(resid, [A, psi0], bounds=([0, -2 * np.pi], [10, 2 * np.pi]))
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best.x, best.cost


def main():
    df = pd.read_csv(os.path.join(HERE, "../../pdf_scripts/dataset/metrics_v2.csv"))
    w = df[(df.platform == "wall") & df.r0_g.notna() & (df.rx_lo > 1e8)].copy()
    w["d_cfg"] = w.wavelength_spacing * C_LIGHT / w.rx_lo

    bands = [
        ("2.412 GHz", (2.410e9, 2.414e9)),
        ("2.464-2.467 GHz", (2.462e9, 2.469e9)),
        ("5.77-5.84 GHz", (5.70e9, 5.90e9)),
        ("915 MHz", (0.90e9, 0.93e9)),
        ("868 MHz", (0.85e9, 0.88e9)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fits = {}

    for ax, (name, (flo, fhi)) in zip(axes.flat, bands):
        b = w[(w.rx_lo >= flo) & (w.rx_lo <= fhi)]
        lam = C_LIGHT / b.rx_lo.median()
        bound = b.r0_g_at_bound.fillna(False).astype(bool) | b.r1_g_at_bound.fillna(
            False
        ).astype(bool)
        good, capped = b[~bound], b[bound]
        for rr, col in (("r0", "tab:blue"), ("r1", "tab:orange")):
            ax.scatter(good.d_cfg * 100, good[f"{rr}_g"], s=8, alpha=0.25, color=col,
                       label=f"{rr} (n={len(good)})", edgecolors="none")
        if len(capped):
            ax.scatter(capped.d_cfg * 100, capped.r0_g, s=10, alpha=0.4, marker="x",
                       color="gray", label=f"fit at bound (n={len(capped)})")

        med = good.groupby("d_cfg").agg(
            g=("r0_g", "median"), g1=("r1_g", "median"), n=("r0_g", "size")
        ).reset_index()
        gm = (med.g + med.g1) / 2
        ax.scatter(med.d_cfg * 100, gm, s=70, marker="s", facecolor="none",
                   edgecolor="black", linewidth=1.4, label="config median", zorder=5)

        dd = np.linspace(max(0.008, med.d_cfg.min() * 0.7), med.d_cfg.max() * 1.25, 300)
        ax.plot(dd * 100, np.ones_like(dd), color="gray", lw=0.8, ls=":")
        ax.plot(dd * 100, (lam / 2) / dd, color="tab:green", lw=1.2, ls="--",
                label=r"$g=(\lambda/2)/d$ (pinned at $\lambda/2$)")

        if len(med) >= 2:
            (A, psi0), cost = fit_band(med.d_cfg.values, gm.values, med.n.values, lam)
            gpred = model_g(med.d_cfg.values, lam, A, psi0)
            rmse = float(np.sqrt(np.average((gpred - gm) ** 2, weights=med.n)))
            ax.plot(dd * 100, model_g(dd, lam, A, psi0), color="crimson", lw=1.8,
                    label=f"coupling fit A={A:.2f}, $\\psi_0$={psi0:+.2f}\n(rmse {rmse:.03f})")
            fits[name] = (A, psi0, rmse, len(good))

        ax.axvline(lam / 2 * 100, color="tab:green", lw=0.8, alpha=0.5)
        ax.text(lam / 2 * 100, ax.get_ylim()[0] + 0.02, r" $\lambda/2$",
                color="tab:green", fontsize=8, va="bottom")
        ax.set_title(f"{name}   ($\\lambda$={lam*100:.1f} cm)", fontsize=11)
        ax.set_xlabel("configured spacing d (cm)")
        ax.set_ylabel("fitted g (effective/configured)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    # master panel: everything in d/lambda
    ax = axes.flat[5]
    bound = w.r0_g_at_bound.fillna(False).astype(bool) | w.r1_g_at_bound.fillna(False).astype(bool)
    good = w[~bound]
    sc = ax.scatter(good.wavelength_spacing, (good.r0_g + good.r1_g) / 2, s=8, alpha=0.3,
                    c=np.log10(good.rx_lo), cmap="viridis", edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("log10 carrier Hz")
    dl = np.linspace(0.03, 1.7, 300)
    ax.plot(dl, np.ones_like(dl), color="gray", lw=0.8, ls=":")
    ax.plot(dl, 0.5 / dl, color="tab:green", lw=1.2, ls="--", label=r"$g=0.5/(d/\lambda)$")
    ax.axvline(0.5, color="tab:green", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"configured spacing $d/\lambda$")
    ax.set_ylabel("fitted g")
    ax.set_title("all wall datasets, universal axis", fontsize=11)
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Fitted g (effective/configured spacing) vs configured spacing, per band — wall arrays",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "g_vs_spacing.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    for k, (A, psi0, rmse, n) in fits.items():
        print(f"{k:18s} A={A:.3f}  psi0={psi0:+.3f} rad  rmse={rmse:.3f}  n={n}")


if __name__ == "__main__":
    main()
