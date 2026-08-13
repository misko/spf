"""Two sensitivities the deployment rules in L10 demand be checked.

1. FROZEN DELAYS. One ripple slot in the R18-only fit rails at the grid minimum
   (0.10 ns), which is a degenerate "DC" slot. Refit with the delays frozen at
   the committed campaign values (2.54 / 0.94 ns) and see whether the holdout
   moves. The comb passes ``check_comb_conditioning`` (kappa 1.47), so freezing
   is permitted here -- E-GSP7's rule is that freezing on an ALIASED comb is
   catastrophic, not that freezing is always wrong.

2. WHAT E-GSC7 ACTUALLY BUYS. Score the rung on the E-GSC7 rows alone, held out
   by LO, so the single-frequency mixer 6..14 evidence is graded on its own.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitlib as FL  # noqa: E402
import union as U  # noqa: E402

CAMPAIGN_TAUS = (2.54e-9, 0.94e-9)   # l31_pooled_v1.json tau_seconds


def frozen_eval(frames, folds, taus, static_fields, n_ripples):
    grid = [np.array([t]) for t in taus]
    pred = np.zeros(len(frames["D"]))
    sup = np.zeros(len(frames["D"]), dtype=bool)
    seen = np.zeros(len(frames["D"]), dtype=bool)
    for _lbl, tr, te in folds:
        if not tr.sum() or not te.sum():
            continue
        # a one-element grid per slot pins the delay without searching
        m = FL.fit_rung(frames["lo_hz"][tr], frames["g1"][tr], frames["g2"][tr],
                        frames["D"][tr], rf_hz=frames["rf_hz"][tr],
                        static_fields=static_fields, n_ripples=n_ripples)
        m.tau_seconds = tuple(taus)
        # refit the linear part at the frozen delays
        m2 = _refit_linear(frames, tr, taus, static_fields, n_ripples, m)
        for i in np.nonzero(te)[0]:
            p = m2.predict(frames["lo_hz"][i], int(frames["g1"][i]),
                           int(frames["g2"][i]), rf_hz=frames["rf_hz"][i],
                           apply_rf_state_guard=False)
            pred[i] = p.residual_rad if p.supported else 0.0
            sup[i] = p.supported
            seen[i] = True
    err = FL.wrap(frames["D"][seen] - np.where(sup, pred, 0.0)[seen])
    return {"coverage": float(sup[seen].mean()), "all_cells": FL.circ_stats(err)}


def _refit_linear(frames, tr, taus, static_fields, n_ripples, template):
    """Least squares at fixed delays, reusing fit_rung's design by monkeypatching
    the tau grid to a single point per slot."""
    saved = FL.TAU_GRID
    try:
        FL.TAU_GRID = np.array(taus, dtype=float)
        m = FL.fit_rung(frames["lo_hz"][tr], frames["g1"][tr], frames["g2"][tr],
                        frames["D"][tr], rf_hz=frames["rf_hz"][tr],
                        static_fields=static_fields, n_ripples=n_ripples)
    finally:
        FL.TAU_GRID = saved
    return m


def main(out_path):
    out = {}
    for tag, radios in (("R18_only", ("R18",)), ("pooled", ("R17", "R18"))):
        f = U.build(include_gsc7=True, radios=radios)
        folds = list(FL.folds_leave_one_out(f, "lo_hz"))
        free = FL.evaluate(f, folds, static_fields=FL.L31_FIELDS, n_ripples=2)
        froz = frozen_eval(f, folds, CAMPAIGN_TAUS, FL.L31_FIELDS, 2)
        out[tag] = {
            "baseline": FL.circ_stats(f["D"]),
            "LOFO_delays_free": {k: v for k, v in free.items()
                                 if not k.startswith("_")},
            "LOFO_delays_frozen_at_campaign": froz,
            "campaign_taus_ns": [t * 1e9 for t in CAMPAIGN_TAUS],
        }
        print(f"{tag}: baseline {out[tag]['baseline']['mae_deg']:.4f}  "
              f"free {free['all_cells']['mae_deg']:.4f}  "
              f"frozen {froz['all_cells']['mae_deg']:.4f}")

        # E-GSC7 rows graded on their own, inside the LOFO run
        seen, err = free["_seen"], free["_err"]
        for src in ("E-GSC6", "E-GSC7"):
            m = f["source"][seen] == src
            if m.any():
                out[tag].setdefault("LOFO_by_source", {})[src] = \
                    FL.circ_stats(err[m])
                out[tag].setdefault("baseline_by_source", {})[src] = \
                    FL.circ_stats(f["D"][seen][m])
        print(f"  by source: {json.dumps(out[tag].get('LOFO_by_source'))}")

    Path(out_path).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
