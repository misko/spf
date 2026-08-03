"""Characterise the common gain response H(f, g) = [D(g,ref) - D(ref,g)]/2.

Questions:
  - Is there a discontinuity at the AD9361 gain-table band edges (1300/4000 MHz)?
  - Is there a linear-in-frequency (pure delay) component?
  - Is there a periodic (reflection / standing-wave) component, and at what delay?
  - How much of H is shared between the two physical radios?
"""

from __future__ import annotations

import numpy as np

import spflib as S

REF = 26


def build_H(stage="A", ref=REF, gains=(5, 45)):
    f = S.load_stages([stage])
    agg = S.aggregate_cells(f)
    res = {}
    for serial in (S.TREATED, S.CONTROL):
        m = agg["serial"] == serial
        lo, g1, g2, ph = (
            agg["lo_hz"][m],
            agg["g1"][m],
            agg["g2"][m],
            agg["phase"][m],
        )
        los = np.unique(lo)
        idx = {(int(a), int(b), int(c)): d for a, b, c, d in zip(lo, g1, g2, ph)}
        out = {"lo": los}
        for g in gains:
            D1 = np.array(
                [
                    S.wrap(idx[(int(x), g, ref)] - idx[(int(x), ref, ref)])
                    for x in los
                ]
            )
            D2 = np.array(
                [
                    S.wrap(idx[(int(x), ref, g)] - idx[(int(x), ref, ref)])
                    for x in los
                ]
            )
            out[f"H{g}"] = S.wrap((D1 - D2) / 2.0)
            out[f"A{g}"] = S.wrap(D1 + D2)
        # equal-gain anchor curve too
        out["C"] = np.array([idx[(int(x), ref, ref)] for x in los])
        res[serial] = out
    return res


def band_edges(lo, H, label):
    """Discontinuity across the AD9361 gain-table band edges."""
    print(f"\n  {label}: band-edge steps (deg)")
    for edge, lo_side, hi_side in ((1300e6, 1300e6, 1301e6), (4000e6, 4000e6, 4001e6)):
        i = np.argmin(np.abs(lo - lo_side))
        j = np.argmin(np.abs(lo - hi_side))
        # local slope estimate from the 4 neighbours on each side
        below = (lo <= lo_side) & (lo >= lo_side - 200e6)
        above = (lo >= hi_side) & (lo <= hi_side + 200e6)
        pb = np.polyfit(lo[below] / 1e9, np.degrees(np.unwrap(H[below])), 1)
        pa = np.polyfit(lo[above] / 1e9, np.degrees(np.unwrap(H[above])), 1)
        extrap = np.polyval(pb, lo[j] / 1e9)
        jump_raw = np.degrees(S.wrap(H[j] - H[i]))
        jump_fit = np.degrees(S.wrap(np.radians(np.polyval(pa, lo[j] / 1e9) - extrap)))
        print(
            f"    {lo[i]/1e6:6.0f} -> {lo[j]/1e6:6.0f} MHz : "
            f"raw step {jump_raw:8.3f}   trend-corrected step {jump_fit:8.3f}"
        )


def delay_and_ripple(lo, H, label, band_mask=None):
    """Fit  H(f) ~= a + 2*pi*f*tau  (per band) then look for periodic residue."""
    if band_mask is None:
        band_mask = np.ones(len(lo), dtype=bool)
    x = lo[band_mask]
    y = np.degrees(np.unwrap(H[band_mask]))
    if len(x) < 6:
        return None
    p = np.polyfit(x, y, 1)
    slope_deg_per_hz = p[0]
    tau_ps = slope_deg_per_hz / 360.0 * 1e12
    resid = y - np.polyval(p, x)
    print(
        f"    {label:26s} n={len(x):3d}  intercept {p[1]:8.3f} deg  "
        f"delay {tau_ps:9.2f} ps ({tau_ps*1e-12*S.C_LIGHT*1e3:8.2f} mm)  "
        f"resid rms {resid.std():6.3f} deg"
    )
    return x, resid


def ripple_spectrum(x_hz, resid_deg, label, tau_max_ns=12.0, n=4000):
    """Lomb-style scan for a sinusoid in frequency -> a time delay."""
    taus = np.linspace(0.05e-9, tau_max_ns * 1e-9, n)
    best = []
    xc = x_hz - x_hz.mean()
    y = resid_deg - resid_deg.mean()
    power = np.zeros(n)
    for i, t in enumerate(taus):
        c = np.cos(2 * np.pi * xc * t)
        s = np.sin(2 * np.pi * xc * t)
        Adesign = np.column_stack([c, s, np.ones_like(c)])
        coef, *_ = np.linalg.lstsq(Adesign, y, rcond=None)
        pred = Adesign @ coef
        power[i] = 1.0 - np.sum((y - pred) ** 2) / np.sum(y**2)
        best.append(np.hypot(coef[0], coef[1]))
    best = np.array(best)
    order = np.argsort(power)[::-1]
    picked = []
    for k in order:
        if all(abs(taus[k] - taus[j]) > 0.4e-9 for j in picked):
            picked.append(k)
        if len(picked) == 3:
            break
    print(f"    {label} top ripple components:")
    for k in picked:
        print(
            f"       tau {taus[k]*1e9:6.3f} ns  amp {best[k]:6.3f} deg  "
            f"R2 {power[k]:5.3f}  period {1/taus[k]/1e6:8.1f} MHz  "
            f"one-way free-space {taus[k]*S.C_LIGHT*1e3:7.1f} mm"
        )
    return taus, power, best


def main():
    res = build_H("A")
    print("=" * 78)
    print("COMMON GAIN RESPONSE  H(f,g) = [D(g,26) - D(26,g)]/2   (stage A)")
    print("=" * 78)

    for serial in (S.TREATED, S.CONTROL):
        r = res[serial]
        lo = r["lo"]
        print(f"\n### {S.SHORT[serial]}")
        for g in (5, 45):
            H = r[f"H{g}"]
            print(
                f"\n H(g={g}): overall mean |H| {np.degrees(np.abs(H)).mean():.3f} deg,"
                f" sd {np.degrees(H).std():.3f} deg"
            )
            band_edges(lo, H, f"g={g}")
            print(f"  per-band linear (delay) fit, g={g}:")
            for bidx, bname, mask in (
                (0, "low   <=1300", lo <= 1300e6),
                (1, "middle 1301-4000", (lo > 1300e6) & (lo <= 4000e6)),
                (2, "high  >4000", lo > 4000e6),
            ):
                out = delay_and_ripple(lo, H, f"{bname}", mask)
                if out is not None and mask.sum() >= 12:
                    ripple_spectrum(out[0], out[1], f"      {bname}")

    # cross-radio sharing
    print("\n" + "=" * 78)
    print("HOW MUCH OF H IS SHARED BETWEEN THE TWO PHYSICAL RADIOS?")
    print("=" * 78)
    a, b = res[S.TREATED], res[S.CONTROL]
    lo = a["lo"]
    for g in (5, 45):
        Ha, Hb = a[f"H{g}"], b[f"H{g}"]
        d = S.wrap(Ha - Hb)
        common = S.wrap((Ha + Hb) / 2)
        print(
            f"\n g={g}: mean|H_R17| {np.degrees(np.abs(Ha)).mean():6.3f}  "
            f"mean|H_R18| {np.degrees(np.abs(Hb)).mean():6.3f}  "
            f"mean|H_R17-H_R18| {np.degrees(np.abs(d)).mean():6.3f} deg"
        )
        print(
            f"      corr(H_R17,H_R18) = "
            f"{np.corrcoef(np.degrees(Ha), np.degrees(Hb))[0,1]:6.3f};  "
            f"variance explained by using the OTHER radio's H: "
            f"{1 - np.var(np.degrees(d))/np.var(np.degrees(Ha)):6.3f}"
        )
        for bname, mask in (
            ("low", lo <= 1300e6),
            ("mid", (lo > 1300e6) & (lo <= 4000e6)),
            ("high", lo > 4000e6),
        ):
            print(
                f"      {bname:5s}: mean|diff| {np.degrees(np.abs(d[mask])).mean():6.3f} deg"
                f"   corr {np.corrcoef(np.degrees(Ha[mask]), np.degrees(Hb[mask]))[0,1]:6.3f}"
            )

    # save for later plotting
    np.savez(
        "H_stageA.npz",
        lo=lo,
        H5_t=a["H5"],
        H45_t=a["H45"],
        A5_t=a["A5"],
        A45_t=a["A45"],
        C_t=a["C"],
        H5_c=b["H5"],
        H45_c=b["H45"],
        A5_c=b["A5"],
        A45_c=b["A45"],
        C_c=b["C"],
    )


if __name__ == "__main__":
    main()
