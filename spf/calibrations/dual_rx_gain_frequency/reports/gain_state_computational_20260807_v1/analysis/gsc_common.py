"""Shared loading and scoring helpers for the E-GSC computational program.

Conventions enforced here, and never mixed:

  ANCHORED    every model predicts D = phi - measured equal-gain anchor at the
              same (serial, stage, LO, epoch). The anchor is an INPUT.
  uneq        error restricted to unequal-gain cells (g1 != g2), i.e. the cells
              a deployed correction actually acts on.
  all         error including the equal-gain anchor cell, whose residual is
              zero by construction. Reported alongside, never instead.

Fail-closed: an unsupported cell predicts 0 (= the anchor), never an
extrapolated value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import features as FT
import ladder as LD  # noqa: F401  (re-exported TAU_GRID users)
import spflib as S
from models import LadderModel, Term, build_design

SPF_ROOT = Path("/home/mouse9911/gits/spf")
if str(SPF_ROOT) not in sys.path:
    sys.path.insert(0, str(SPF_ROOT))

# The ten pre-registered E-CAL3 training LOs (MHz).
PREREG_10_MHZ = (400, 1000, 1600, 2200, 2800, 3400, 4100, 4700, 5300, 5900)

# Committed fleet delays (l26_stage_a_v1 / the report's section 8 model).
TAU_FLEET = (2.56e-9, 0.92e-9)


# ------------------------------------------------------------------- rungs ---
def rung_terms(name: str):
    if name == "L26":
        return [
            Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
            Term("lna", basis="cos0"), Term("lna", basis="sin0"),
            Term("lna", basis="cos1"), Term("lna", basis="sin1"),
        ]
    if name == "L30":
        return [Term("lna"), Term("mixer"), Term("tia")]
    if name == "L31":
        return [
            Term("lna"), Term("mixer"), Term("tia"),
            Term("lna", basis="cos0"), Term("lna", basis="sin0"),
            Term("lna", basis="cos1"), Term("lna", basis="sin1"),
        ]
    raise KeyError(name)


def rung_model(name: str, taus=None):
    """taus=None -> delays grid-searched on the training fold only.
    taus=(t1,t2) -> delays frozen (single-element grids, so no search runs)."""
    terms = rung_terms(name)
    has_ripple = any(t.basis.startswith(("cos", "sin")) for t in terms)
    if not has_ripple:
        grids = ()
    elif taus is None:
        grids = [LD.TAU_GRID, LD.TAU_GRID]
    else:
        grids = [np.array([taus[0]]), np.array([taus[1]])]
    return LadderModel(name, terms, tau_grids=grids)


# -------------------------------------------------------------------- data ---
def load_anchored(stages, ref=26, per_epoch=True, quality_only=True):
    return FT.add_anchor(
        S.load_stages(stages, quality_only=quality_only), ref=ref,
        per_epoch=per_epoch,
    )


def lo_mhz(f):
    return (np.asarray(f.lo_hz) / 1e6).round().astype(int)


# ------------------------------------------------------------------ scoring --
def score(D, pred, supported, mask, uneq_mask):
    """Fail-closed error statistics on `mask`, all-cell and unequal-gain."""
    fc = np.where(supported, pred, 0.0)
    m = mask
    mu = mask & uneq_mask
    out = {
        "n": int(m.sum()),
        "coverage": float(supported[m].mean()) if m.sum() else float("nan"),
        "baseline_mae_deg": S.cmae_deg(D[m]) if m.sum() else float("nan"),
        "mae_deg": S.cmae_deg(S.wrap(D[m] - fc[m])) if m.sum() else float("nan"),
        "p95_deg": S.cp95_deg(S.wrap(D[m] - fc[m])) if m.sum() else float("nan"),
        "max_deg": S.cmax_deg(S.wrap(D[m] - fc[m])) if m.sum() else float("nan"),
        "n_uneq": int(mu.sum()),
        "coverage_uneq": float(supported[mu].mean()) if mu.sum() else float("nan"),
        "baseline_uneq_mae_deg": S.cmae_deg(D[mu]) if mu.sum() else float("nan"),
        "uneq_mae_deg": (
            S.cmae_deg(S.wrap(D[mu] - fc[mu])) if mu.sum() else float("nan")
        ),
        "uneq_p95_deg": (
            S.cp95_deg(S.wrap(D[mu] - fc[mu])) if mu.sum() else float("nan")
        ),
    }
    return out


def committed_predictions(f, coeff_name, apply_rf_state_guard=True):
    """Predict D with a COMMITTED coefficient set from the shipped package.

    Returns (pred_rad, supported). Uses the measured RF tone frequency for the
    ripple basis, matching the source analysis (`design.f = frames.rf_hz`).
    """
    from spf.calibrations.gain_state_phase_model_v1.model import GainStatePhaseModel

    mdl = GainStatePhaseModel.load_named(coeff_name)
    n = len(f)
    pred = np.zeros(n)
    sup = np.zeros(n, dtype=bool)
    cache = {}
    for i in range(n):
        key = (float(f.lo_hz[i]), int(f.g1[i]), int(f.g2[i]), float(f.rf_hz[i]))
        if key not in cache:
            p = mdl.predict(
                key[0], key[1], key[2], rf_hz=key[3],
                apply_rf_state_guard=apply_rf_state_guard,
            )
            cache[key] = (p.residual_rad, bool(p.supported))
        pred[i], sup[i] = cache[key]
    return pred, sup


def fit_and_predict(f, model, train_mask, test_mask):
    """Fit `model` on train rows of `f`, predict test rows. Delays are searched
    on the TRAINING FOLD ONLY (models.LadderModel.fit_eval guarantees this).

    Returns (pred_full, supported_full, taus_ns, n_active_columns).
    """
    d = build_design(f, model.terms)
    tri = np.nonzero(train_mask)[0]
    tei = np.nonzero(test_mask)[0]
    p, sp, taus, ncol = model.fit_eval(d, f.D, tri, tei)
    pred = np.zeros(len(f))
    sup = np.zeros(len(f), dtype=bool)
    pred[tei] = p
    sup[tei] = sp
    return pred, sup, (np.asarray(taus) * 1e9).tolist(), int(ncol)
