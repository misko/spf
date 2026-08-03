"""Compute the numbers the pre-commit review requires.

R3  the RF_DC_CAL discriminator that already exists inside the excluded F_neg stage
R4  decompose the "RF state" 1 dB step group into mixer / TIA / LNA, with a
    cluster bootstrap on the ratio
R6  per-stage behaviour of the recommended model where the RF state is frozen
R9  the two claims the review found false as written
"""

from __future__ import annotations

import numpy as np

import features as FT
import spflib as S
from diag_gainsteps import common_H

RNG = np.random.default_rng(20260802)


def _steps(stage, ref, adjacent_only=True):
    """Yield (serial, lo, g_prev, g, dH_deg, state_prev, state, band)."""
    out = []
    res = common_H(stage, ref)
    for serial, tab in res.items():
        for f_hz, (gs, H, _A) in tab.items():
            b = int(S.gain_band(np.array([f_hz]))[0])
            for i in range(1, len(gs)):
                if adjacent_only and gs[i] - gs[i - 1] != 1:
                    continue
                s0 = FT.HW.state(b, int(gs[i - 1]))
                s1 = FT.HW.state(b, int(gs[i]))
                if s0 is None or s1 is None:
                    continue
                out.append((serial, f_hz, int(gs[i - 1]), int(gs[i]),
                            float(abs(np.degrees(S.wrap(H[i] - H[i - 1])))),
                            s0[:3], s1[:3], b))
    return out


def r4_decompose():
    print("=" * 96)
    print("R4  what the '2.089 deg RF state' group actually contains")
    print("=" * 96)
    rows = _steps("F", 5) + _steps("E_tx_0", 33)
    groups = {"mixer": [], "tia_only": [], "lna": [], "lpf_only": []}
    clusters = {"mixer": [], "tia_only": [], "lna": [], "lpf_only": []}
    for serial, f_hz, g0, g1, d, s0, s1, b in rows:
        dl, dm, dt = s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]
        if dl:
            k = "lna"
        elif dm:
            k = "mixer"
        elif dt:
            k = "tia_only"
        else:
            k = "lpf_only"
        groups[k].append(d)
        clusters[k].append((serial, int(f_hz)))
    for k, v in groups.items():
        a = np.array(v)
        nclu = len(set(clusters[k]))
        if a.size:
            print(f"  {k:9s} n={a.size:3d} ({nclu} (radio,LO) clusters)  "
                  f"median {np.median(a):6.3f}  mean {a.mean():6.3f}  "
                  f"p90 {np.percentile(a,90):6.3f}  max {a.max():6.3f}")
        else:
            print(f"  {k:9s} n=  0  -- NONE MEASURED")

    mix = np.array(groups["mixer"])
    lpf = np.array(groups["lpf_only"])
    tia = np.array(groups["tia_only"])
    print(f"\n  mixer / LPF-only median ratio = {np.median(mix)/np.median(lpf):.2f}x")
    print(f"  TIA-only / LPF-only median ratio = "
          f"{np.median(tia)/np.median(lpf):.2f}x  <- indistinguishable")

    # cluster bootstrap on the mixer/LPF median ratio
    mix_c = np.array([f"{s}|{f}" for (s, f), _ in
                      zip(clusters["mixer"], groups["mixer"])], dtype=object)
    lpf_c = np.array([f"{s}|{f}" for (s, f), _ in
                      zip(clusters["lpf_only"], groups["lpf_only"])], dtype=object)
    um, ul = np.unique(mix_c), np.unique(lpf_c)
    boot = []
    for _ in range(4000):
        cm = RNG.choice(um, len(um), replace=True)
        cl = RNG.choice(ul, len(ul), replace=True)
        a = np.concatenate([mix[mix_c == c] for c in cm])
        b = np.concatenate([lpf[lpf_c == c] for c in cl])
        if a.size and b.size and np.median(b) > 0:
            boot.append(np.median(a) / np.median(b))
    boot = np.array(boot)
    print(f"  cluster bootstrap 95% CI on the mixer/LPF ratio: "
          f"[{np.percentile(boot,2.5):.1f}, {np.percentile(boot,97.5):.1f}]  "
          f"({len(um)} mixer clusters, {len(ul)} LPF clusters)")

    # was any LNA transition measured at all, at any step size?
    allrows = _steps("F", 5, adjacent_only=False) + _steps("E_tx_0", 33, False)
    lna_any = [(r[0][-6:], r[1] / 1e6, r[2], r[3], r[4])
               for r in allrows if r[5][0] != r[6][0]]
    print(f"\n  LNA transitions at ANY step size: n={len(lna_any)}")
    for s, f_mhz, g0, g1, d in lna_any:
        print(f"    {s} {f_mhz:7.1f} MHz  {g0:3d}->{g1:3d} dB ({g1-g0} dB)  "
              f"|dH| {d:6.3f} deg")


def r3_rfdc():
    print("\n" + "=" * 96)
    print("R3  the RF_DC_CAL discriminator already present in the excluded F_neg stage")
    print("=" * 96)
    f = S.load_stages(["F_neg"])
    hi = f.band == 2
    print(f"  F_neg rows: {len(f)} total, {int(hi.sum())} in the high band")
    print(f"  high-band LOs: {np.unique(f.lo_hz[hi])/1e6}")
    print(f"  high-band gains: {sorted(set(f.g1[hi].tolist())|set(f.g2[hi].tolist()))}")

    rows = _steps("F_neg", 5)
    rows = [r for r in rows if r[7] == 2]
    tab = FT.HW.tab["high"]

    def rfdc(g):
        r = int(tab.row_for_gain(np.array(g)))
        return int(tab.rfdc[r]) if r >= 0 else -1

    rfdc_only, lpf_flat, lmt = [], [], []
    for serial, f_hz, g0, g1, d, s0, s1, b in rows:
        if s0 != s1:
            lmt.append(d)
        elif rfdc(g1) != rfdc(g0):
            rfdc_only.append((g0, g1, d))
        else:
            lpf_flat.append(d)
    ro = np.array([d for _, _, d in rfdc_only])
    rise = np.array([d for g0, g1, d in rfdc_only if rfdc(g1) == 1])
    lp = np.array(lpf_flat)
    lm = np.array(lmt)
    for name, a in (("RF_DC_CAL toggles, LMT frozen", ro),
                    ("  of which rising edge (entering a flagged row)", rise),
                    ("LPF-only, RF_DC_CAL frozen", lp),
                    ("LMT change (MIX 0->1) at the same LOs", lm)):
        if a.size:
            print(f"  {name:48s} n={a.size:3d} median {np.median(a):6.3f} "
                  f"max {a.max():6.3f} deg")
    try:
        from scipy.stats import mannwhitneyu
        if ro.size and lp.size:
            print(f"  Mann-Whitney RF_DC-only vs LPF-only: "
                  f"p={mannwhitneyu(ro, lp).pvalue:.3f}")
        if lm.size and (ro.size + lp.size):
            print(f"  Mann-Whitney LMT vs everything else:  "
                  f"p={mannwhitneyu(lm, np.concatenate([ro, lp])).pvalue:.2e}")
    except ImportError:
        pass

    # were rows 22/23/24 (8/9/10 dB, the other RF_DC-only edge) ever sampled?
    print("\n  Was the second RF_DC-only edge (high rows 22/23/24 = 8/9/10 dB) sampled?")
    for stage in ("A", "F", "F_neg", "E_tx_0", "G", "rate_pilot"):
        try:
            g = S.load_stages([stage])
        except Exception:
            continue
        h = g.band == 2
        gains = set(g.g1[h].tolist()) | set(g.g2[h].tolist())
        hit = sorted(gains & {8, 9, 10})
        print(f"    {stage:10s} high-band gains present from {{8,9,10}}: {hit}")


def r6_frozen_state():
    print("\n" + "=" * 96)
    print("R6  does the recommended model help where the RF state is frozen?")
    print("=" * 96)
    import ladder as LD
    from models import build_design
    from transfer import pooled

    fp = pooled(["A", "F", "E_tx_0", "rate_pilot"])
    lna1 = FT.HW.vec(fp.band, fp.g1, "lna")
    mix1 = FT.HW.vec(fp.band, fp.g1, "mixer")
    tia1 = FT.HW.vec(fp.band, fp.g1, "tia")
    lna2 = FT.HW.vec(fp.band, fp.g2, "lna")
    mix2 = FT.HW.vec(fp.band, fp.g2, "mixer")
    tia2 = FT.HW.vec(fp.band, fp.g2, "tia")
    frozen = (lna1 == lna2) & (mix1 == mix2) & (tia1 == tia2)
    uneq = fp.g1 != fp.g2
    print(f"  pooled rows {len(fp)}; unequal-gain {int(uneq.sum())}; "
          f"of those, RF words identical on both arms: "
          f"{int((frozen & uneq).sum())} ({(frozen&uneq).sum()/uneq.sum():.1%})")

    ladder = [m for m in LD.make_ladder()
              if m.name.split()[0] in {"L26", "L31", "L16", "L05", "L30"}]
    for m in ladder:
        d = build_design(fp, m.terms)
        preds = np.zeros(len(fp))
        sup = np.zeros(len(fp), dtype=bool)
        for _l, tr, te in LD.splits_leave_one_frequency_out(fp):
            tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
            p, sp, _t, _n = m.fit_eval(d, fp.D, tri, tei)
            preds[tei] = p
            sup[tei] = sp
        fc = np.where(sup, preds, 0.0)
        print(f"\n  {m.name}")
        for lbl, mask in (("all unequal-gain cells", uneq),
                          ("  RF state FROZEN (both arms same words)", frozen & uneq),
                          ("  RF state DIFFERS", (~frozen) & uneq)):
            base = S.cmae_deg(fp.D[mask])
            got = S.cmae_deg(S.wrap(fp.D[mask] - fc[mask]))
            verdict = "HELPS" if got < base else "HARMS"
            print(f"    {lbl:42s} n={int(mask.sum()):4d} "
                  f"baseline {base:6.3f} -> model {got:6.3f}  {verdict}")
        m2 = frozen & uneq
        worse = np.mean(np.abs(S.wrap(fp.D[m2] - fc[m2])) > np.abs(fp.D[m2]))
        print(f"    fraction of frozen-state cells made WORSE: {worse:.1%}; "
              f"mean injected |correction| "
              f"{np.degrees(np.abs(fc[m2])).mean():.3f} deg, "
              f"max {np.degrees(np.abs(fc[m2])).max():.3f} deg")

    # per-stage
    print("\n  per-stage (L26, pooled LOFO, unequal-gain cells):")
    m = [x for x in ladder if x.name.startswith("L26")][0]
    d = build_design(fp, m.terms)
    preds = np.zeros(len(fp))
    sup = np.zeros(len(fp), dtype=bool)
    for _l, tr, te in LD.splits_leave_one_frequency_out(fp):
        tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
        p, sp, _t, _n = m.fit_eval(d, fp.D, tri, tei)
        preds[tei] = p
        sup[tei] = sp
    fc = np.where(sup, preds, 0.0)
    for st in ("A", "F", "E_tx_0", "rate_pilot"):
        k = (fp.stage.astype(str) == st) & uneq
        if k.sum():
            print(f"    {st:11s} n={int(k.sum()):4d} "
                  f"baseline {S.cmae_deg(fp.D[k]):6.3f} -> "
                  f"{S.cmae_deg(S.wrap(fp.D[k]-fc[k])):6.3f}")
    # with the frozen-state guard applied
    guarded = np.where(frozen, 0.0, fc)
    for st in ("A", "F", "E_tx_0", "rate_pilot"):
        k = (fp.stage.astype(str) == st) & uneq
        if k.sum():
            print(f"    {st:11s} WITH GUARD -> "
                  f"{S.cmae_deg(S.wrap(fp.D[k]-guarded[k])):6.3f}")
    print(f"    ALL unequal cells: no guard "
          f"{S.cmae_deg(S.wrap(fp.D[uneq]-fc[uneq])):.3f} -> with guard "
          f"{S.cmae_deg(S.wrap(fp.D[uneq]-guarded[uneq])):.3f}")


def r9_claims():
    print("\n" + "=" * 96)
    print("R9  the two claims the review flagged as false as written")
    print("=" * 96)
    res = common_H("E_tx_0", 33)
    print("  claim: 'H stays inside +-0.6 deg' over 27-40 dB")
    for serial, tab in res.items():
        for f_hz, (gs, H, _A) in sorted(tab.items()):
            d = np.degrees(H)
            print(f"    {S.SHORT[serial]} {f_hz/1e6:7.1f} MHz  max|H| {np.abs(d).max():5.3f}  "
                  f"peak-to-peak {d.max()-d.min():5.3f} deg")
    print("\n  claim: 'One dB across the MIX 1->2 boundary at 5766 MHz costs 7.7 deg'")
    resF = common_H("F", 5)
    for serial, tab in resF.items():
        for f_hz in (5766000000, 5866000000):
            gs, H, _ = tab[f_hz]
            i = list(gs).index(10)
            j = list(gs).index(5)
            print(f"    {S.SHORT[serial]} {f_hz/1e6:7.1f} MHz  "
                  f"{gs[j]}->{gs[i]} dB is a {gs[i]-gs[j]} dB step, "
                  f"|dH| = {abs(np.degrees(S.wrap(H[i]-H[j]))):6.3f} deg")


if __name__ == "__main__":
    r4_decompose()
    r3_rfdc()
    r9_claims()
    r6_frozen_state()
