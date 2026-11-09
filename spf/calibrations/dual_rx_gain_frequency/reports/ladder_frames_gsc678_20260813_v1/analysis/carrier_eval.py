"""Prospective error AT THE ROVER'S OWN CARRIERS -- the test the ladder splits don't run.

LOFO/LOBLOCK/LORO answer "does this rung generalise across the sweep". The rover
question is narrower and harder: fit on a bench session, then predict at 5766 or
5840 MHz on a DIFFERENT session. Three regimes, each with its own failure mode:

  EPOCH   train GSC8a -> test GSC8b at one carrier. Same carrier, same cabling,
          different session. This is "calibrate today, fly tomorrow".
  CARRIER train on every LO except the target -> test at the target. The model
          has never seen this carrier. This is "calibrate at 5766, fly at 5840".
  BOTH    train on everything except (target carrier AND the test session).

Every rung is scored against L00 -- the measured equal-gain anchor with no model
at all. A rung that does not beat L00 here must not ship, whatever it scores on
the bench, because L00 is what you get for free.

Errors are reported on UNEQUAL-GAIN rows only: the equal-gain cell has an
identically-zero residual by construction and dilutes any aggregate. Predictions
are fail-closed -- an unsupported gain state falls back to the anchor, never to
an extrapolated value.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency/"
                   "reports/gain_state_phase_model_20260802_v1/analysis")

import features as FT  # noqa: E402
import ladder as LD  # noqa: E402
import load_gsc  # noqa: E402
import spflib as S  # noqa: E402

CARRIERS = {"5766": 5_766_000_000.0, "5840": 5_840_000_000.0}


def score(f, y, design, m, tr, te, rng):
    """Fail-closed stats on unequal-gain test rows, plus the L00 baseline there."""
    tri, tei = np.nonzero(tr)[0], np.nonzero(te)[0]
    if len(tri) == 0 or len(tei) == 0:
        return None
    p, sp, _tt, nc = m.fit_eval(design, y, tri, tei, rng=rng)
    fail_closed = np.where(sp, p, 0.0)
    uneq = f.g1[tei] != f.g2[tei]
    if uneq.sum() == 0:
        return None
    err = S.wrap(y[tei][uneq] - fail_closed[uneq])
    base = S.wrap(y[tei][uneq])            # L00: predict zero, i.e. anchor only
    return {
        "n_test": int(uneq.sum()),
        "params": int(nc),
        "coverage": float(sp[uneq].mean()),
        "mae_deg": S.cmae_deg(err),
        "rmse_deg": S.crmse_deg(err),
        "p95_deg": S.cp95_deg(err),
        "l00_mae_deg": S.cmae_deg(base),
        "l00_rmse_deg": S.crmse_deg(base),
        "ratio_vs_l00": S.cmae_deg(base) / max(S.cmae_deg(err), 1e-9),
    }


def main():
    # The rover's own equal-gain frames are 83% (5766) / 96% (5840) at 62 dB, so
    # 62 is the anchor a deployed correction would actually have. 26 dB is the
    # published bench convention and is kept only for comparison.
    ref = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    tag = sys.argv[2] if len(sys.argv) > 2 else f"ref{ref}"
    f = FT.add_anchor(load_gsc.load(), ref=ref, per_epoch=True)
    y = f.D
    print(f"anchor ref={ref} dB, rows={len(f)}", flush=True)
    ladder = LD.make_ladder()
    designs = {m.name: LD.build_design(f, m.terms) for m in ladder}

    regimes = {}
    for cname, chz in CARRIERS.items():
        at = f.lo_hz == chz
        regimes[f"EPOCH@{cname}"] = ((f.stage == "GSC8a") & at, (f.stage == "GSC8b") & at)
        regimes[f"CARRIER@{cname}"] = (~at, at)
        regimes[f"BOTH@{cname}"] = (~at & (f.stage != "GSC8b"), at & (f.stage == "GSC8b"))

    out = {}
    for rname, (tr, te) in regimes.items():
        print(f"\n{rname}: train={tr.sum()} test={te.sum()}", flush=True)
        rows = []
        for m in ladder:
            r = score(f, y, designs[m.name], m, tr, te, np.random.default_rng(0))
            if r:
                rows.append({"model": m.name, "tier": m.tier, **r})
        rows.sort(key=lambda r: r["mae_deg"])
        out[rname] = rows
        for r in rows[:6]:
            print(f"   {r['model'][:52]:<52} {r['mae_deg']:6.3f} deg  "
                  f"(L00 {r['l00_mae_deg']:6.3f}, {r['ratio_vs_l00']:.2f}x)  "
                  f"cov={r['coverage']:.0%} p={r['params']}", flush=True)
        with open(f"carrier_eval_{tag}.json", "w") as fh:
            json.dump(out, fh, indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
