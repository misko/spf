"""Does the common gain response H(g) step exactly where the audited AD9361
gain table changes LNA / mixer / TIA state?

Stage F: 12 gains (-1..26 dB) at 6 LOs spanning all three gain-table bands.
Stage E: 14 gains (27..40 dB) at 5100 and 5766 MHz (high band only).
"""

from __future__ import annotations

import numpy as np

import features as FT
import spflib as S


def common_H(stage, ref, serials=(S.TREATED, S.CONTROL)):
    f = S.load_stages([stage], serials=serials)
    agg = S.aggregate_cells(f)
    out = {}
    for serial in serials:
        m = agg["serial"] == serial
        lo, g1, g2, ph = (agg["lo_hz"][m], agg["g1"][m], agg["g2"][m],
                          agg["phase"][m])
        idx = {(int(a), int(b), int(c)): d for a, b, c, d in zip(lo, g1, g2, ph)}
        los = np.unique(lo)
        gains = sorted(set(g1.tolist()) | set(g2.tolist()))
        tab = {}
        for f_hz in los:
            k = int(f_hz)
            if (k, ref, ref) not in idx:
                continue
            base = idx[(k, ref, ref)]
            H, A, gs = [], [], []
            for g in gains:
                if (k, g, ref) in idx and (k, ref, g) in idx:
                    D1 = S.wrap(idx[(k, g, ref)] - base)
                    D2 = S.wrap(idx[(k, ref, g)] - base)
                    H.append(S.wrap((D1 - D2) / 2))
                    A.append(S.wrap(D1 + D2))
                    gs.append(g)
            tab[k] = (np.array(gs), np.array(H), np.array(A))
        out[serial] = tab
    return out


def report(stage, ref, title):
    print("=" * 100)
    print(f"{title}   (stage {stage}, reference gain {ref} dB)")
    print("=" * 100)
    res = common_H(stage, ref)
    for serial in (S.TREATED, S.CONTROL):
        print(f"\n### {S.SHORT[serial]}")
        for f_hz, (gs, H, A) in sorted(res[serial].items()):
            b = int(S.gain_band(np.array([f_hz]))[0])
            bn = FT.BANDS[b]
            print(f"\n  LO {f_hz/1e6:7.1f} MHz  (gain table: {bn})")
            print("     g dB |  H deg  | step deg | audited state (lna,mix,tia,lpf)"
                  " | hw change")
            prev_state, prev_H = None, None
            for g, h in zip(gs, H):
                st = FT.HW.state(b, int(g))
                if st is None:
                    continue
                short = f"({st[0]},{st[1]:2d},{st[2]},{st[3]:2d})"
                step = "" if prev_H is None else f"{np.degrees(S.wrap(h-prev_H)):8.3f}"
                chg = ""
                if prev_state is not None:
                    names = []
                    if st[0] != prev_state[0]:
                        names.append(f"LNA{prev_state[0]}>{st[0]}")
                    if st[1] != prev_state[1]:
                        names.append(f"MIX{prev_state[1]}>{st[1]}")
                    if st[2] != prev_state[2]:
                        names.append(f"TIA{prev_state[2]}>{st[2]}")
                    chg = " ".join(names)
                print(f"    {g:4d}  | {np.degrees(h):7.3f} | {step:>8s} |"
                      f" {short:22s} | {chg}")
                prev_state, prev_H = st, h


def step_vs_hwchange(stage, ref):
    """Aggregate: are steps at hardware-state changes bigger than steps that
    only move the baseband LPF word?"""
    res = common_H(stage, ref)
    hw_steps, lpf_steps = [], []
    for serial, tab in res.items():
        for f_hz, (gs, H, A) in tab.items():
            b = int(S.gain_band(np.array([f_hz]))[0])
            for i in range(1, len(gs)):
                s0 = FT.HW.state(b, int(gs[i - 1]))
                s1 = FT.HW.state(b, int(gs[i]))
                if s0 is None or s1 is None:
                    continue
                if gs[i] - gs[i - 1] != 1:
                    continue  # only adjacent 1 dB steps
                d = abs(np.degrees(S.wrap(H[i] - H[i - 1])))
                rf_changed = (s0[0] != s1[0]) or (s0[1] != s1[1]) or (s0[2] != s1[2])
                (hw_steps if rf_changed else lpf_steps).append(d)
    hw_steps = np.array(hw_steps)
    lpf_steps = np.array(lpf_steps)
    print("\n" + "-" * 100)
    print("1 dB steps in the COMMON gain response H, split by what the audited "
          "gain table does")
    print("-" * 100)
    for name, arr in (("RF state change (LNA/mixer/TIA)", hw_steps),
                      ("baseband LPF word only        ", lpf_steps)):
        if arr.size:
            print(f"  {name}: n={arr.size:3d}  median {np.median(arr):6.3f} deg  "
                  f"mean {arr.mean():6.3f}  p90 {np.percentile(arr,90):6.3f}  "
                  f"max {arr.max():6.3f}")
    if hw_steps.size and lpf_steps.size:
        print(f"  ratio of medians (RF change / LPF only) = "
              f"{np.median(hw_steps)/np.median(lpf_steps):.2f}x")


if __name__ == "__main__":
    report("F", 5, "LOW-GAIN AND TIA BOUNDARY SWEEP")
    step_vs_hwchange("F", 5)
    print("\n\n")
    report("E_tx_0", 33, "HIGH-BAND 27-40 dB SWEEP (fixed TX 0 dB)")
    step_vs_hwchange("E_tx_0", 33)
