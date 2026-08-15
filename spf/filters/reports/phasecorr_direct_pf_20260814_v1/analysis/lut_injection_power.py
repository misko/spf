"""Does Addendum 3's estimator recover a REAL gain-state-shaped term? Inject one and see.

power_calibration.py measured sensitivity by adding an INDEPENDENT Gaussian d ~ N(0,A^2)
to the residual and fitting Delta(mean|e|) = 0.0101*A^2. A reviewer objected, correctly,
that this answers a narrower question than the report claims. A physical gain-phase term
is not i.i.d. noise: it is a DETERMINISTIC FUNCTION OF GAIN STATE, so it has

  * long temporal runs (gain is held across consecutive frames),
  * per-capture structure (each capture visits its own slice of the gain grid),
  * correlation with geometry/AGC (gain IS the AGC's response to received power, which
    depends on range and orientation),

and none of that is present in a Gaussian. The comment in power_calibration.sensitivity()
asserting that a correlated d "would only make the ceiling smaller" is NOT PROVEN: a term
aligned with sign(e) gives a LINEAR, not quadratic, response and a much LARGER effect.

This script does the honest experiment:

  1. loads the residuals exactly as gain_fixed_effects.py does (same KW, same per-stream
     circular centring, same dedup -- the dedup is a known defect, kept for comparability),
  2. takes the DEPLOYED donor LUT shape as the candidate correction shape and centres it
     to zero weighted mean over the rover frames (the constant is absorbed downstream),
  3. injects alpha * LUT(gain state of that frame) along each stream's ACTUAL gain
     sequence, so run structure, capture grouping and AGC/geometry coupling are preserved,
  4. re-runs the FULL estimator + 6-fold-by-capture CV from gain_fixed_effects.py, and
  5. compares against an i.i.d. Gaussian injection of identical rms, PAIRED on fold seed,
     over 8 fold seeds.

It also measures corr(LUT(gain_state), e) directly, which is the reviewer's specific
concern: if it is ~0 the independence assumption is defensible on this data.

Read-only. Reads a cached npz built by cache_residuals.py; touches nothing under /mnt.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

HERE = ("/tmp/claude-1000/-home-mouse9911-gits-spf/"
        "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/audit2")
IMPL = ("/tmp/claude-1000/-home-mouse9911-gits-spf/"
        "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/impl")
DONOR = (f"{IMPL}/spf/calibrations/models/gsc9_arm_lut_per_radio/"
         "1040007c4a94000211000b009186843ef2.json")

RMS_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
SEEDS = tuple(range(8))
FOLDS = 6
MIN_N = 8


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def cmean(x):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(x))))) if len(x) else 0.0


# ---------------------------------------------------------------- data + LUT

def load_cached():
    z = np.load(f"{HERE}/residuals.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    sid = d["stream_id"]
    # streams are contiguous blocks; record slices for per-stream ops
    bnd = np.flatnonzero(np.diff(sid)) + 1
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [len(sid)]])
    d["slices"] = list(zip(starts.tolist(), ends.tolist()))
    return d


def donor_lut(d):
    """L_raw[i] = rx1[f_i, g1_i] + rx2[f_i, g2_i], radians, deployed donor model.

    Formula and sign convention are taken verbatim from the model file:
        "formula": "phase = intercept[f] + RX1[f,g1] + RX2[f,g2]"
        "phase_convention": "RX1 minus RX2"
    """
    m = json.load(open(DONOR))
    gains = np.asarray(m["gains_db"], dtype=int)
    freqs = np.asarray(m["frequencies_hz"], dtype=float)
    c = m["coefficients_rad"]
    rx1 = np.stack([[c[f"frequency[{fi}].rx1_phase[{i}]"] for i in range(len(gains))]
                    for fi in range(len(freqs))])
    rx2 = np.stack([[c[f"frequency[{fi}].rx2_phase[{i}]"] for i in range(len(gains))]
                    for fi in range(len(freqs))])

    g1 = np.clip(d["g1"].astype(int), gains[0], gains[-1]) - gains[0]
    g2 = np.clip(d["g2"].astype(int), gains[0], gains[-1]) - gains[0]
    n_clip = int(((d["g1"] < gains[0]) | (d["g2"] < gains[0])
                  | (d["g1"] > gains[-1]) | (d["g2"] > gains[-1])).sum())
    fi = np.argmin(np.abs(d["lo"][:, None] - freqs[None, :]), axis=1)
    max_off = float(np.abs(d["lo"] - freqs[fi]).max())
    L = rx1[fi, g1] + rx2[fi, g2]
    return L, n_clip, max_off, m


# ---------------------------------------------------- vectorised estimator

def _codes(d, kind):
    fm = np.round(d["lo"] / 1e6).astype(int)
    g1 = d["g1"].astype(int)
    g2 = d["g2"].astype(int)
    if kind == "cell":
        raw = (g1 * 100 + g2) * 100000 + fm
        _, inv = np.unique(raw, return_inverse=True)
        return [(inv, +1)], int(inv.max()) + 1
    if kind == "arm":
        r1 = g1 * 100000 + fm
        r2 = g2 * 100000 + fm
        u = np.unique(np.concatenate([r1, r2]))          # shared arm-1/arm-2 label space
        i1 = np.searchsorted(u, r1)
        i2 = np.searchsorted(u, r2)
        # arm-1 and arm-2 keys are DISTINCT in gain_fixed_effects (prefixes a1_/a2_)
        return [(i1, +1), (i2 + len(u), -1)], 2 * len(u)
    if kind == "rfblock":
        sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/"
                           "dual_rx_gain_frequency/reports/"
                           "gain_state_phase_model_20260802_v1/analysis")
        import features
        hs = features.HardwareStates()
        st = {g: hs.state(2, int(g)) for g in np.unique(np.concatenate([g1, g2]))}
        # gain_fixed_effects keys: m1_/l1_ (+1) from g1, m2_/l2_ (-1) from g2,
        # where s[1] is the mixer word and s[0] the LNA word.
        raw = [np.array([st[g][1] for g in g1]) * 100000 + fm,   # m1
               np.array([st[g][0] for g in g1]) * 100000 + fm,   # l1
               np.array([st[g][1] for g in g2]) * 100000 + fm,   # m2
               np.array([st[g][0] for g in g2]) * 100000 + fm]   # l2
        uu = [np.unique(r) for r in raw]
        off, spec = 0, []
        for r, un, sg in zip(raw, uu, (+1, +1, -1, -1)):
            spec.append((np.searchsorted(un, r) + off, sg))
            off += len(un)
        return spec, off
    raise ValueError(kind)


def cv(e, d, code_spec, n_codes, seed, folds=FOLDS, min_n=MIN_N):
    """gain_fixed_effects.cv(), vectorised. Identical arithmetic, mask_name='all'.

    delta is a CIRCULAR MEAN of the signed residual per key, fitted on training
    captures only; the prediction is re-centred PER HELD-OUT STREAM before subtraction,
    exactly as in the original (cmean(p) is inside the per-stream loop there).
    """
    rx = sorted(set(d["stream_rx"].tolist()))
    order = np.random.default_rng(seed).permutation(len(rx))
    assign = {rx[i]: int(order[i] % folds) for i in range(len(rx))}
    stream_fold = np.array([assign[r] for r in d["stream_rx"].tolist()])

    ejw = np.exp(1j * e)
    before = after = 0.0
    n_tot = 0
    for f in range(folds):
        tr = np.repeat(stream_fold != f, [b - a for a, b in d["slices"]])
        delta = np.zeros(n_codes)
        have = np.zeros(n_codes, bool)
        for code, sgn in code_spec:
            ct = np.bincount(code[tr], minlength=n_codes)
            # circular mean of sgn*e  ==  angle(sum exp(i*sgn*e))
            v = ejw[tr] if sgn > 0 else np.conj(ejw[tr])
            s = (np.bincount(code[tr], weights=v.real, minlength=n_codes)
                 + 1j * np.bincount(code[tr], weights=v.imag, minlength=n_codes))
            ok = ct >= min_n
            delta[ok] = np.angle(s[ok])
            have |= ok
        delta[~have] = 0.0
        for si, (a, b) in enumerate(d["slices"]):
            if stream_fold[si] != f:
                continue
            p = np.zeros(b - a)
            for code, sgn in code_spec:
                p += sgn * delta[code[a:b]]
            es = e[a:b]
            before += np.abs(es).sum()
            after += np.abs(wrap(es - wrap(p - cmean(p)))).sum()
            n_tot += b - a
    return (np.degrees(before / n_tot), np.degrees(after / n_tot), n_tot)


def restream_centre(e, d):
    out = e.copy()
    for a, b in d["slices"]:
        out[a:b] = wrap(out[a:b] - cmean(out[a:b]))
    return out


# ------------------------------------------------------------------- main

def main():
    d = load_cached()
    e0 = d["e"]
    n = len(e0)
    sd = np.degrees(e0.std())
    print(f"{len(d['slices'])} receiver-streams, "
          f"{len(set(d['stream_rx'].tolist()))} unique RX captures, {n} frames")
    print(f"residual sd {sd:.3f} deg, mean|e| {np.degrees(np.abs(e0)).mean():.3f} deg")
    print("(dedup-by-RX-prefix reproduced from gain_fixed_effects.py -- a KNOWN DEFECT "
          "that\n drops 6 disjoint-in-time stores; kept so these numbers are comparable "
          "to the report)\n")

    L_raw, n_clip, max_off, meta = donor_lut(d)
    wmean = L_raw.mean()                    # frame-weighted mean == equal weights here
    L = L_raw - wmean
    rms_L = np.degrees(np.sqrt((L ** 2).mean()))
    Ls = restream_centre(L, d)              # part surviving per-stream centring
    rms_Ls = np.degrees(np.sqrt((Ls ** 2).mean()))

    print("=== 0. THE INJECTED SHAPE: deployed donor LUT ===")
    print(f"  {DONOR.split('/')[-1]}")
    print(f"  {meta['label']}")
    print(f"  formula {meta['formula']!r}, {meta['parameter_count']} params, "
          f"carriers {[int(f/1e6) for f in meta['frequencies_hz']]} MHz")
    print(f"  gains clamped to [{meta['gains_db'][0]},{meta['gains_db'][-1]}] dB on "
          f"{n_clip} of {n} frames ({100*n_clip/n:.4f}%)")
    print(f"  max |LO - LUT carrier| = {max_off/1e6:.3f} MHz (exact match, both carriers "
          f"in table)")
    print(f"  natural amplitude on this corpus: rms {rms_L:.3f} deg, "
          f"p2p {np.degrees(L.max()-L.min()):.3f} deg, "
          f"{len(np.unique(np.round(np.degrees(L),6)))} distinct values")
    print(f"  after per-stream circular centring: rms {rms_Ls:.3f} deg "
          f"({100*rms_Ls/rms_L:.1f}% of the injected rms survives; the rest is a "
          f"per-stream\n    constant the pipeline absorbs)\n")

    # ---- reviewer's concern: is the LUT term correlated with the residual?
    print("=== 1. IS THE LUT TERM CORRELATED WITH THE RESIDUAL? (reviewer's concern) ===")
    r = np.corrcoef(L, e0)[0, 1]
    rs = np.corrcoef(Ls, e0)[0, 1]
    # the quantity that governs a LINEAR response of mean|e|:  d/dA E|e+A*u| = E[sign(e)*u]
    u = L / np.sqrt((L ** 2).mean())                       # unit-rms shape
    us = Ls / np.sqrt((Ls ** 2).mean())
    align = float(np.mean(np.sign(e0) * u))
    align_s = float(np.mean(np.sign(e0) * us))
    # circular-linear correlation of e with the LUT value
    rc = np.sqrt(np.corrcoef(np.cos(e0), L)[0, 1] ** 2
                 + np.corrcoef(np.sin(e0), L)[0, 1] ** 2)
    print(f"  Pearson corr(L, e)               = {r:+.5f}")
    print(f"  Pearson corr(L_streamcentred, e) = {rs:+.5f}")
    print(f"  circular-linear corr(e; L)       = {rc:.5f}")
    print(f"  E[sign(e) * u]  (u unit rms)     = {align:+.5f}   "
          f"-> linear term in Delta(mean|e|) = {align:+.5f} * A deg")
    print(f"  E[sign(e) * u_streamcentred]     = {align_s:+.5f}")
    se = 1.0 / np.sqrt(n)
    print(f"  (sd of E[sign(e)*u] under independence ~ 1/sqrt(N) = {se:.5f}; "
          f"observed |align| = {abs(align)/se:.2f} sigma)")

    # ---- true response of mean|e| to the LUT shape, measured not assumed
    print("\n=== 2. TRUE Delta(mean|e|) FOR THIS SHAPE vs THE GAUSSIAN LAW ===")
    base = np.degrees(np.abs(e0)).mean()
    rng = np.random.default_rng(7)
    print(f"  {'rms A':>7}  {'LUT shape':>12}  {'i.i.d. Gaussian':>16}  {'ratio':>7}")
    k_lut = []
    for A in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        a_rad = np.radians(A)
        dl = np.degrees(np.abs(wrap(e0 + a_rad * u))).mean() - base
        dg = np.mean([np.degrees(np.abs(wrap(e0 + rng.normal(0, a_rad, n)))).mean() - base
                      for _ in range(8)])
        k_lut.append(dl / A ** 2)
        print(f"  {A:7.1f}  {dl:+12.5f}  {dg:+16.5f}  {dl/dg:7.3f}")
    print(f"  LUT quadratic coefficient: mean Delta/A^2 = {np.mean(k_lut):.5f} "
          f"(Gaussian law in the report: 0.0101)")

    # ---- the injection experiment
    print("\n=== 3. INJECTION EXPERIMENT: full estimator + 6-fold CV by capture ===")
    print(f"  folds={FOLDS}  min_n={MIN_N}  mask=all  seeds={list(SEEDS)}")
    print("  injection: e' = wrap(e + alpha*u(gain state)), then per-stream re-centred,")
    print("  i.e. along each stream's ACTUAL gain sequence -- runs, capture grouping and")
    print("  AGC/geometry coupling preserved.\n")

    results = defaultdict(dict)
    for kind in ("cell", "arm"):
        code_spec, n_codes = _codes(d, kind)
        base_ch = {}
        for s in SEEDS:
            b, a, _ = cv(e0, d, code_spec, n_codes, s)
            base_ch[s] = a - b
        results[kind]["baseline"] = base_ch
        print(f"  [{kind}] no injection, per seed: "
              f"{' '.join(f'{base_ch[s]:+.3f}' for s in SEEDS)}")
        print(f"  [{kind}] no injection: mean {np.mean(list(base_ch.values())):+.4f} "
              f"sd {np.std(list(base_ch.values())):.4f} deg\n")

        for label, shape in (("LUT", u), ("gauss", None)):
            for A in RMS_SWEEP:
                a_rad = np.radians(A)
                ch, dl = {}, {}
                for s in SEEDS:
                    if shape is None:
                        inj = np.random.default_rng(1000 + s).normal(0, a_rad, n)
                    else:
                        inj = a_rad * shape
                    e1 = restream_centre(wrap(e0 + inj), d)
                    b, aa, _ = cv(e1, d, code_spec, n_codes, s)
                    ch[s] = aa - b
                    dl[s] = ch[s] - base_ch[s]
                results[kind][(label, A)] = dict(change=ch, delta=dl)
                v = np.array([dl[s] for s in SEEDS])
                sem = v.std(ddof=1) / np.sqrt(len(v))
                ideal = -np.mean(k_lut) * A ** 2 if label == "LUT" else -0.0101 * A ** 2
                det = "DETECTED" if (v.mean() < 0 and abs(v.mean()) > 2 * sem) else "no"
                print(f"  [{kind}] {label:5s} rms {A:4.1f} deg: "
                      f"change {np.mean([ch[s] for s in SEEDS]):+.4f} "
                      f"| paired delta {v.mean():+.4f} +- {sem:.4f} (sd {v.std(ddof=1):.4f}) "
                      f"| {int((v<0).sum())}/{len(v)} seeds neg "
                      f"| ideal {ideal:+.4f} | recov {v.mean()/ideal:6.2f} | {det}")
            print()

    np.save(f"{HERE}/injection_results.npy", dict(results), allow_pickle=True)
    print(f"raw results -> {HERE}/injection_results.npy")


if __name__ == "__main__":
    sys.exit(main())
