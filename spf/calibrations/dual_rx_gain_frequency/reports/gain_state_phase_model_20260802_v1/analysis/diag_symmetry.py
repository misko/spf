"""Model-free diagnostics.

1. Verify the AD9361 gain-table decode and the requested-dB -> row mapping.
2. Split the measured gain effect into
      common   H(g)  = [ D(g,ref) - D(ref,g) ] / 2      (shared between arms)
      asymmetry A(g) =   D(g,ref) + D(ref,g)            (arm-specific)
   where D(g1,g2) = phase(g1,g2) - phase(ref,ref) at the same LO.
3. Quantify baseline (no-correction) error under several definitions.
"""

from __future__ import annotations

import numpy as np

import spflib as S

np.set_printoptions(suppress=True, linewidth=200)


def show_gain_tables():
    tabs = S.load_gain_tables()
    ser = sorted(tabs)
    print("=" * 78)
    print("AD9361 FULL gain tables read off the hardware (passive audit)")
    print("=" * 78)
    t0 = tabs[ser[0]]
    t1 = tabs[ser[1]]
    for band in ("low", "middle", "high"):
        a, b = t0[band], t1[band]
        same = (
            np.array_equal(a.b0, b.b0)
            and np.array_equal(a.b1, b.b1)
            and np.array_equal(a.b2, b.b2)
            and np.array_equal(a.gain_db, b.gain_db)
        )
        print(
            f"\nband={band:6s} {a.start_hz/1e6:7.0f}-{a.end_hz/1e6:7.0f} MHz "
            f"rows={len(a.gain_db)} gain {a.gain_db.min()}..{a.gain_db.max()} dB "
            f"identical_between_radios={same}"
        )
        # row-for-gain mapping check
        for g in (-1, 0, 5, 26, 33, 45):
            r = int(a.row_for_gain(np.array(g)))
            if r >= 0:
                print(
                    f"   g={g:3d} dB -> row {r:2d}  lna={a.lna[r]} mixer={a.mixer[r]:2d}"
                    f" tia={a.tia[r]} lpf={a.lpf[r]:2d} dig=0x{a.dig[r]:02x}"
                )
    # state transitions in the dense 11-17 region and overall
    print("\n" + "=" * 78)
    print("Hardware-state transitions (row->row+1 changes in LNA/mixer/TIA)")
    print("=" * 78)
    for band in ("low", "middle", "high"):
        a = t0[band]
        print(f"\n{band}:")
        prev = None
        for g in range(-1, 50):
            r = int(a.row_for_gain(np.array(g)))
            if r < 0:
                continue
            st = (int(a.lna[r]), int(a.mixer[r]), int(a.tia[r]))
            if prev is not None and st != prev[1]:
                lbl = []
                if st[0] != prev[1][0]:
                    lbl.append(f"LNA {prev[1][0]}->{st[0]}")
                if st[1] != prev[1][1]:
                    lbl.append(f"MIX {prev[1][1]}->{st[1]}")
                if st[2] != prev[1][2]:
                    lbl.append(f"TIA {prev[1][2]}->{st[2]}")
                print(
                    f"   {prev[0]:3d}->{g:3d} dB : {', '.join(lbl):24s}"
                    f" b0 0x{a.b0[int(a.row_for_gain(np.array(prev[0])))]:02x}"
                    f"->0x{a.b0[r]:02x}  b1 0x{a.b1[int(a.row_for_gain(np.array(prev[0])))]:02x}"
                    f"->0x{a.b1[r]:02x}"
                )
            prev = (g, st)


def symmetry(stage="A", ref=26, gains=(5, 45)):
    print("\n" + "=" * 78)
    print(f"SYMMETRY DECOMPOSITION  stage={stage} reference gain={ref} dB")
    print("=" * 78)
    f = S.load_stages([stage])
    agg = S.aggregate_cells(f)
    out = {}
    for serial in (S.TREATED, S.CONTROL):
        m = agg["serial"] == serial
        lo = agg["lo_hz"][m]
        g1 = agg["g1"][m]
        g2 = agg["g2"][m]
        ph = agg["phase"][m]
        los = np.unique(lo)

        def cell(f_hz, a, b):
            k = (lo == f_hz) & (g1 == a) & (g2 == b)
            return ph[k][0] if k.sum() == 1 else np.nan

        rows = {}
        for g in gains:
            Dg = np.array(
                [S.wrap(cell(f_, g, ref) - cell(f_, ref, ref)) for f_ in los]
            )
            Dr = np.array(
                [S.wrap(cell(f_, ref, g) - cell(f_, ref, ref)) for f_ in los]
            )
            H = S.wrap((Dg - Dr) / 2.0)  # common gain response
            A = S.wrap(Dg + Dr)  # arm asymmetry
            rows[g] = dict(D_g_ref=Dg, D_ref_g=Dr, H=H, A=A)
            print(
                f"\n{S.SHORT[serial]}  g={g:3d} vs ref={ref}:"
                f"\n   |D(g,ref)|  mean {np.degrees(np.abs(Dg)).mean():7.3f}  "
                f"p95 {np.percentile(np.degrees(np.abs(Dg)),95):7.3f}  "
                f"max {np.degrees(np.abs(Dg)).max():7.3f} deg"
                f"\n   |D(ref,g)|  mean {np.degrees(np.abs(Dr)).mean():7.3f}  "
                f"p95 {np.percentile(np.degrees(np.abs(Dr)),95):7.3f}  "
                f"max {np.degrees(np.abs(Dr)).max():7.3f} deg"
                f"\n   COMMON  H   mean {np.degrees(np.abs(H)).mean():7.3f}  "
                f"sd {np.degrees(H).std():7.3f}  "
                f"circmean {np.degrees(S.cmean(H)):7.3f} deg"
                f"\n   ASYMM   A   mean {np.degrees(np.abs(A)).mean():7.3f}  "
                f"sd {np.degrees(A).std():7.3f}  "
                f"circmean {np.degrees(S.cmean(A)):7.3f} deg"
                f"\n   -> asymmetric energy fraction "
                f"{np.mean(np.degrees(A/2)**2)/np.mean(np.degrees(Dg)**2 + np.degrees(Dr)**2)*2:6.3f}"
            )
        out[serial] = (los, rows)
    return out


def baselines(stage="A", ref=26):
    print("\n" + "=" * 78)
    print(f"BASELINE (NO-CORRECTION) ERROR LEVELS  stage={stage}")
    print("=" * 78)
    f = S.load_stages([stage])
    for serial in (S.TREATED, S.CONTROL):
        m = f.serial == serial
        g = f.sel(m)
        lo = g.lo_hz
        ph = g.phase
        print(f"\n--- {S.SHORT[serial]}  n={len(g)}")
        # B0 raw
        print(f"  B0 predict 0 (raw phase)                : "
              f"MAE {S.cmae_deg(ph):8.3f}  P95 {S.cp95_deg(ph):8.3f}  "
              f"max {S.cmax_deg(ph):8.3f} deg")
        # B1 single global constant
        c = S.cmean(ph)
        print(f"  B1 one constant per radio               : "
              f"MAE {S.cmae_deg(ph-c):8.3f}  P95 {S.cp95_deg(ph-c):8.3f}  "
              f"max {S.cmax_deg(ph-c):8.3f} deg")
        # B2 per-frequency constant (gain-blind); i.e. calibrate freq, ignore gain
        pred = np.zeros_like(ph)
        for f_hz in np.unique(lo):
            k = lo == f_hz
            pred[k] = S.cmean(ph[k])
        print(f"  B2 per-frequency constant (gain-blind)  : "
              f"MAE {S.cmae_deg(ph-pred):8.3f}  P95 {S.cp95_deg(ph-pred):8.3f}  "
              f"max {S.cmax_deg(ph-pred):8.3f} deg")
        # B2b per-frequency anchored at the reference equal-gain cell
        pred2 = np.full_like(ph, np.nan)
        for f_hz in np.unique(lo):
            k = lo == f_hz
            kr = k & (g.g1 == ref) & (g.g2 == ref)
            if kr.sum():
                pred2[k] = S.cmean(ph[kr])
        ok = ~np.isnan(pred2)
        print(f"  B2b per-frequency equal-gain anchor     : "
              f"MAE {S.cmae_deg(ph[ok]-pred2[ok]):8.3f}  "
              f"P95 {S.cp95_deg(ph[ok]-pred2[ok]):8.3f}  "
              f"max {S.cmax_deg(ph[ok]-pred2[ok]):8.3f} deg   "
              f"<-- THE GAIN-EFFECT BASELINE")
        # restricted to unequal-gain cells only
        uneq = ok & (g.g1 != g.g2)
        print(f"      (unequal-gain cells only, n={uneq.sum():5d})    : "
              f"MAE {S.cmae_deg(ph[uneq]-pred2[uneq]):8.3f}  "
              f"P95 {S.cp95_deg(ph[uneq]-pred2[uneq]):8.3f}  "
              f"max {S.cmax_deg(ph[uneq]-pred2[uneq]):8.3f} deg")
        # B3 per-frequency per-gain-pair (= saturated LUT, epoch-averaged): noise floor
        pred3 = np.zeros_like(ph)
        for f_hz in np.unique(lo):
            for a in np.unique(g.g1):
                for b in np.unique(g.g2):
                    k = (lo == f_hz) & (g.g1 == a) & (g.g2 == b)
                    if k.sum():
                        pred3[k] = S.cmean(ph[k])
        print(f"  B3 saturated LUT (in-sample noise floor): "
              f"MAE {S.cmae_deg(ph-pred3):8.3f}  P95 {S.cp95_deg(ph-pred3):8.3f}  "
              f"max {S.cmax_deg(ph-pred3):8.3f} deg")


if __name__ == "__main__":
    show_gain_tables()
    baselines("A")
    symmetry("A")
