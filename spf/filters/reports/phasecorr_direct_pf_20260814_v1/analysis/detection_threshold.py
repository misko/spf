"""Where exactly does the Addendum-3 procedure start to see a real gain-shaped term?

Follow-on to lut_injection_power.py. Four things that script left open:

  1. A FINE rms sweep so the threshold is interpolated between measured points, not
     between 3.0 and 5.0.
  2. Three DIFFERENT thresholds, because "detects" means three different things:
       (a) paired vs the same-seed no-injection run -- only possible in simulation,
       (b) the report's own decision rule, "held-out change goes negative",
       (c) that rule surviving the procedure's own fold-seed dispersion (2 sd).
  3. The ONE-PARAMETER PROJECTION onto the donor LUT shape, which power_calibration.py
     named as "the statistic to use" and then did not run. Cross-validated by capture,
     with a capture-level block bootstrap, plus an injection-recovery check proving the
     estimator returns the amplitude it was given.
  4. An honest CI on corr(LUT, e): the frames are autocorrelated and capture-grouped, so
     the naive 1/sqrt(N) used in script 1 overstates significance. Bootstrap the 42
     captures instead.

Read-only. Reads the npz cached by cache_residuals.py; touches nothing under /mnt.
"""

from __future__ import annotations

import sys

import numpy as np

HERE = ("/tmp/claude-1000/-home-mouse9911-gits-spf/"
        "fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/audit2")
sys.path.insert(0, HERE)

import lut_injection_power as lip  # noqa: E402

wrap, cmean = lip.wrap, lip.cmean
SEEDS = tuple(range(8))
FINE = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)

# published Gaussian-heuristic break-even amplitudes, power_calibration.py section 3
GAUSS_BREAKEVEN = {"cell": 4.83, "arm": 2.86, "rfblock": 1.87, "1-param": 0.27}


# --------------------------------------------------- one-parameter projection

def _stream_ids(d):
    return np.repeat(np.arange(len(d["slices"])), [b - a for a, b in d["slices"]])


def obj(beta_deg, e, u, sid, ns, sel):
    """mean|wrap(e - streamcentred(beta*u))| over the frames in `sel`, in degrees.

    The prediction is re-centred per stream exactly as gain_fixed_effects.cv does
    (wrap(p - cmean(p)) inside the per-stream loop), so this is the same objective
    the per-cell fit is scored on -- just with one free parameter instead of ~326.
    """
    p = np.radians(beta_deg) * u
    z = np.exp(1j * p)
    cr = np.bincount(sid, weights=z.real, minlength=ns)
    ci = np.bincount(sid, weights=z.imag, minlength=ns)
    cm = np.angle(cr + 1j * ci)
    r = wrap(e - wrap(p - cm[sid]))
    return float(np.degrees(np.abs(r[sel])).mean())


def fit_beta(e, u, sid, ns, sel, lo=-10.0, hi=10.0):
    g = np.linspace(lo, hi, 81)
    v = [obj(b, e, u, sid, ns, sel) for b in g]
    i = int(np.argmin(v))
    g2 = np.linspace(g[max(i - 1, 0)], g[min(i + 1, len(g) - 1)], 41)
    v2 = [obj(b, e, u, sid, ns, sel) for b in g2]
    return float(g2[int(np.argmin(v2))])


def cv_proj(e, d, u, seed, folds=6):
    """Held-out one-parameter projection. Returns (before, after, mean fitted beta)."""
    sid = _stream_ids(d)
    ns = len(d["slices"])
    rx = sorted(set(d["stream_rx"].tolist()))
    order = np.random.default_rng(seed).permutation(len(rx))
    assign = {rx[i]: int(order[i] % folds) for i in range(len(rx))}
    sf = np.array([assign[r] for r in d["stream_rx"].tolist()])
    frame_fold = sf[sid]
    before = after = 0.0
    n = 0
    betas = []
    for f in range(folds):
        tr, te = frame_fold != f, frame_fold == f
        b = fit_beta(e, u, sid, ns, tr)
        betas.append(b)
        before += np.degrees(np.abs(e[te])).sum()
        after += obj(b, e, u, sid, ns, te) * te.sum()
        n += int(te.sum())
    return before / n, after / n, float(np.mean(betas)), betas


# ------------------------------------------------------------------- main

def main():
    d = lip.load_cached()
    e0 = d["e"]
    n = len(e0)
    L_raw, _, _, _ = lip.donor_lut(d)
    L = L_raw - L_raw.mean()
    u = L / np.sqrt((L ** 2).mean())          # unit-rms donor shape
    sid = _stream_ids(d)
    ns = len(d["slices"])
    rx_of_frame = d["stream_rx"][sid]
    captures = sorted(set(d["stream_rx"].tolist()))

    print(f"{ns} streams, {len(captures)} captures, {n} frames; "
          f"donor shape rms {np.degrees(np.sqrt((L**2).mean())):.3f} deg\n")

    # ---- 1. response law for THIS shape: is it really quadratic?
    print("=== 1. RESPONSE LAW OF mean|e| TO THE DONOR SHAPE ===")
    base = np.degrees(np.abs(e0)).mean()
    A = np.array(FINE)
    D = np.array([np.degrees(np.abs(wrap(e0 + np.radians(a) * u))).mean() - base
                  for a in A])
    M = np.stack([A, A ** 2], 1)
    c1, c2 = np.linalg.lstsq(M, D, rcond=None)[0]
    q = np.linalg.lstsq(A[:, None] ** 2, D, rcond=None)[0][0]
    print(f"  {'A':>5} {'measured':>10} {'0.0101*A^2':>12} {'c1*A+c2*A^2':>13} {'ratio':>7}")
    for a, dd in zip(A, D):
        print(f"  {a:5.1f} {dd:+10.5f} {0.0101*a**2:+12.5f} "
              f"{c1*a+c2*a**2:+13.5f} {dd/(0.0101*a**2):7.2f}")
    print(f"\n  best pure-quadratic for this shape:  Delta = {q:.5f} * A^2")
    print(f"  best linear+quadratic:               Delta = {c1:+.5f}*A {c2:+.5f}*A^2")
    print(f"  the LINEAR term dominates below A = {c1/c2:.2f} deg. The report's law is")
    print("  0.0101*A^2 with NO linear term, from an independent Gaussian. For a real")
    print("  gain-shaped term the linear term is nonzero because the term is not")
    print("  independent of e -- and its sign here makes the ceiling BIGGER, not smaller,")
    print("  contradicting the comment in power_calibration.sensitivity().\n")

    # ---- 2. honest CI on the correlation, bootstrapped over captures
    print("=== 2. corr(LUT(gain state), e): CAPTURE-LEVEL BOOTSTRAP ===")
    idx_by_cap = {c: np.flatnonzero(rx_of_frame == c) for c in captures}
    rng = np.random.default_rng(11)
    r_obs = float(np.corrcoef(u, e0)[0, 1])
    a_obs = float(np.mean(np.sign(e0) * u))
    rb, ab = [], []
    for _ in range(400):
        pick = rng.choice(len(captures), len(captures), replace=True)
        ii = np.concatenate([idx_by_cap[captures[p]] for p in pick])
        rb.append(np.corrcoef(u[ii], e0[ii])[0, 1])
        ab.append(np.mean(np.sign(e0[ii]) * u[ii]))
    rb, ab = np.array(rb), np.array(ab)
    print(f"  corr(u, e)        = {r_obs:+.5f}   bootstrap sd {rb.std():.5f}  "
          f"95% CI [{np.percentile(rb,2.5):+.5f}, {np.percentile(rb,97.5):+.5f}]")
    print(f"  E[sign(e)*u]      = {a_obs:+.5f}   bootstrap sd {ab.std():.5f}  "
          f"95% CI [{np.percentile(ab,2.5):+.5f}, {np.percentile(ab,97.5):+.5f}]")
    print(f"  naive 1/sqrt(N) sd = {1/np.sqrt(n):.5f}; capture-level sd is "
          f"{ab.std()*np.sqrt(n):.1f}x larger.")
    print(f"  significance of alignment: {a_obs/ab.std():.2f} sigma "
          f"(script 1's naive figure was 2.47 sigma)\n")

    # ---- 3. one-parameter projection, cross-validated
    print("=== 3. ONE-PARAMETER PROJECTION ONTO THE DONOR SHAPE (held out by capture) ===")
    print("  beta is in deg rms of the donor shape; beta>0 means the shape is PRESENT")
    print("  in the rover residual with the donor's sign.")
    bb, aa, bm, bl = [], [], [], []
    for s in SEEDS:
        b_, a_, m_, l_ = cv_proj(e0, d, u, s)
        bb.append(b_); aa.append(a_); bm.append(m_); bl += l_
    bb, aa, bm = np.array(bb), np.array(aa), np.array(bm)
    ch = aa - bb
    print(f"  fitted beta per fold (48 fits): mean {np.mean(bl):+.3f} deg rms, "
          f"sd {np.std(bl):.3f}, min {np.min(bl):+.3f}, max {np.max(bl):+.3f}")
    print(f"  held-out change: mean {ch.mean():+.5f} deg, sd {ch.std(ddof=1):.5f}, "
          f"per seed {' '.join(f'{c:+.4f}' for c in ch)}")
    b_full = fit_beta(e0, u, sid, ns, np.ones(n, bool))
    print(f"  in-sample beta on all data: {b_full:+.3f} deg rms")
    print(f"  {'DETECTED' if ch.mean() < -2*ch.std(ddof=1)/np.sqrt(len(ch)) else 'NOT detected'}"
          f": a 1-param projection of the deployed donor shape onto the rover residual\n")

    # ---- 4. does the 1-param estimator recover an injected amplitude?
    print("=== 4. RECOVERY CHECK: inject known beta, see what the 1-param fit returns ===")
    print(f"  {'injected':>9} {'recovered beta':>16} {'held-out change':>17}")
    for a in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        e1 = lip.restream_centre(wrap(e0 + np.radians(a) * u), d)
        b_, a_, m_, _ = cv_proj(e1, d, u, 0)
        print(f"  {a:9.1f} {m_:>16.3f} {a_-b_:>+17.5f}")
    print("  (recovered beta should track injected + the ~0.6 deg already in the data)\n")

    # ---- 5. the three thresholds, from the fine sweep
    print("=== 5. DETECTION THRESHOLDS FOR THE ADDENDUM-3 PROCEDURE ===")
    for kind in ("cell", "arm"):
        spec, ncodes = lip._codes(d, kind)
        base_ch = np.array([(lambda t: t[1] - t[0])(lip.cv(e0, d, spec, ncodes, s)[:2])
                            for s in SEEDS])
        b_mean, b_sd = base_ch.mean(), base_ch.std(ddof=1)
        print(f"\n  --- {kind} --- no-injection change {b_mean:+.4f} +- {b_sd:.4f} "
              f"(sd over {len(SEEDS)} fold seeds)")
        print(f"  {'rms':>5} {'change':>9} {'paired delta':>14} {'sd':>8} "
              f"{'seeds neg':>10} {'paired det':>11}")
        rows = []
        for a in FINE:
            inj = np.radians(a) * u
            e1 = lip.restream_centre(wrap(e0 + inj), d)
            c = np.array([(lambda t: t[1] - t[0])(lip.cv(e1, d, spec, ncodes, s)[:2])
                          for s in SEEDS])
            dl = c - base_ch
            sem = dl.std(ddof=1) / np.sqrt(len(dl))
            det = dl.mean() < -2 * sem
            rows.append((a, c.mean(), dl.mean(), dl.std(ddof=1)))
            print(f"  {a:5.1f} {c.mean():+9.4f} {dl.mean():+14.4f} "
                  f"{dl.std(ddof=1):8.4f} {int((dl<0).sum()):>7}/{len(dl)} "
                  f"{('YES' if det else 'no'):>11}")

        arr = np.array(rows)

        def cross(target):
            """smallest injected rms whose mean paired delta reaches `target` (<0)."""
            y = arr[:, 2]
            for i in range(len(y) - 1):
                if y[i] > target >= y[i + 1]:
                    f = (y[i] - target) / (y[i] - y[i + 1])
                    return arr[i, 0] + f * (arr[i + 1, 0] - arr[i, 0])
            return float("nan")

        t_a = arr[0, 0] if arr[0, 2] < -2 * 0.001 else float("nan")
        t_b = cross(-b_mean)
        t_c = cross(-b_mean - 2 * b_sd)
        print(f"\n  (a) paired vs same-seed control : DETECTED at every rms tested, "
              f"down to {arr[0,0]:.1f} deg (threshold is BELOW the sweep)")
        print(f"  (b) report's rule, change < 0   : needs injected rms "
              f"{t_b:.2f} deg")
        print(f"  (c) (b) beyond 2 seed sd        : needs injected rms "
              f"{t_c:.2f} deg")
        print(f"  Gaussian-heuristic break-even in the report: "
              f"{GAUSS_BREAKEVEN[kind]:.2f} deg")
        print(f"  measured/(heuristic) for rule (b): {t_b/GAUSS_BREAKEVEN[kind]:.2f}x")


if __name__ == "__main__":
    main()
