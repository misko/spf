"""Rung-shaped fitting on anchored residuals, with a configurable static-field set.

``GainStatePhaseModel.fit`` always carries all four static families
(lna, mixer, tia, lpf), i.e. it can only fit an L26-shaped rung. L30/L31 are
defined by *dropping* the categorical LPF family, so this module reimplements
the same algebra with the family list as a parameter.

The reimplementation is validated bit-for-bit against the package's own ``fit``
whenever ``static_fields == STATIC_FIELDS`` -- see ``selfcheck()``. Everything
else (tau grid, ridge, coarse/fine search order, signed-indicator construction)
is copied from ``model.py`` so an L31 fit differs from an L26 fit in exactly one
respect: the absent family.
"""

from __future__ import annotations

import math

import numpy as np

from spf.calibrations.gain_state_phase_model_v1.gain_tables import default_tables
from spf.calibrations.gain_state_phase_model_v1.model import (
    STATIC_FIELDS,
    GainStatePhaseModel,
)

L26_FIELDS = ("lna", "mixer", "tia", "lpf")
L31_FIELDS = ("lna", "mixer", "tia")
L30_FIELDS = ("lna", "mixer", "tia")

TAU_GRID = np.concatenate(
    [np.arange(0.10, 4.0, 0.02), np.arange(4.0, 8.01, 0.05)]
) * 1e-9


def wrap(x):
    return (np.asarray(x) + math.pi) % (2 * math.pi) - math.pi


def circ_stats(err_rad) -> dict:
    e = np.abs(wrap(err_rad))
    if not len(e):
        return {"n": 0}
    return {
        "n": int(len(e)),
        "mae_deg": float(np.degrees(e.mean())),
        "rmse_deg": float(np.degrees(np.sqrt((e ** 2).mean()))),
        "p95_deg": float(np.degrees(np.percentile(e, 95))),
        "max_deg": float(np.degrees(e.max())),
    }


def fit_rung(
    lo_hz,
    g1,
    g2,
    D,
    *,
    rf_hz=None,
    static_fields=L26_FIELDS,
    n_ripples=2,
    ridge=1e-8,
    tables=None,
    name="L26",
    tau_search_rows=None,
    tau_search_seed=0,
):
    """Fit one rung. Returns a ``GainStatePhaseModel`` with the absent families empty.

    ``tau_search_rows`` reproduces the SOURCE ANALYSIS' behaviour
    (``models.LadderModel.TAU_SEARCH_ROWS = 1600``): the ripple delays are
    grid-searched on a random subsample of the training fold rather than on all
    of it, with ``np.random.default_rng(0)`` re-seeded per fold. The shipped
    package's ``GainStatePhaseModel.fit`` does NOT do this -- it searches on
    every training row -- which is why the two disagree on any split whose
    training fold exceeds 1600 rows. ``None`` = the package's behaviour.
    """
    tab = tables or default_tables()
    lo = np.asarray(lo_hz, dtype=float)
    f_rf = lo if rf_hz is None else np.asarray(rf_hz, dtype=float)
    g1 = np.asarray(g1, dtype=int)
    g2 = np.asarray(g2, dtype=int)
    y = np.asarray(D, dtype=float)

    states = [
        (tab.state_for_lo(a, x), tab.state_for_lo(a, z))
        for a, x, z in zip(lo, g1, g2)
    ]
    keep = np.array([s1 is not None and s2 is not None for s1, s2 in states])
    if not keep.all():
        states = [s for s, k in zip(states, keep) if k]
        f_rf, y = f_rf[keep], y[keep]

    levels = {
        fld: sorted(
            {getattr(s1, fld) for s1, _ in states}
            | {getattr(s2, fld) for _, s2 in states}
        )
        for fld in static_fields
    }
    lna_levels = sorted(
        {s1.lna for s1, _ in states} | {s2.lna for _, s2 in states}
    )

    static_cols, static_key = [], []
    for fld in STATIC_FIELDS:           # keep the package's canonical order
        if fld not in static_fields:
            continue
        for lv in levels[fld]:
            static_cols.append(
                np.array(
                    [(getattr(s1, fld) == lv) - (getattr(s2, fld) == lv)
                     for s1, s2 in states],
                    dtype=float,
                )
            )
            static_key.append((fld, lv))
    S_static = (np.column_stack(static_cols) if static_cols
                else np.zeros((len(states), 0)))
    sig_lna = np.column_stack(
        [np.array([(s1.lna == lv) - (s2.lna == lv) for s1, s2 in states],
                  dtype=float) for lv in lna_levels]
    ) if lna_levels else np.zeros((len(states), 0))

    grid = TAU_GRID

    def design(taus):
        blocks = [S_static]
        for t in taus:
            blocks.append(sig_lna * np.cos(2 * np.pi * f_rf * t)[:, None])
            blocks.append(sig_lna * np.sin(2 * np.pi * f_rf * t)[:, None])
        return np.column_stack(blocks)

    def solve(X):
        A = X.T @ X + ridge * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)

    if tau_search_rows is not None and len(y) > tau_search_rows:
        rng = np.random.default_rng(tau_search_seed)
        sub = rng.choice(np.arange(len(y)), tau_search_rows, replace=False)
    else:
        sub = np.arange(len(y))

    S_sub, sig_sub, f_sub, y_sub = (
        S_static[sub], sig_lna[sub], f_rf[sub], y[sub],
    )

    def design_sub(taus):
        blocks = [S_sub]
        for t in taus:
            blocks.append(sig_sub * np.cos(2 * np.pi * f_sub * t)[:, None])
            blocks.append(sig_sub * np.sin(2 * np.pi * f_sub * t)[:, None])
        return np.column_stack(blocks)

    def sse(taus):
        X = design_sub(taus)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        th = np.linalg.solve(A, X.T @ y_sub)
        r = y_sub - X @ th
        return float(r @ r)

    taus = [float(grid[len(grid) // 2])] * n_ripples
    for coarse in (True, False):
        for slot in range(n_ripples):
            if coarse:
                cand = grid[:: max(1, len(grid) // 80)]
            else:
                step = float(np.median(np.diff(grid)))
                cand = taus[slot] + np.arange(-10, 11) * step
                cand = cand[(cand >= grid[0]) & (cand <= grid[-1])]
            best = (math.inf, taus[slot])
            for t in cand:
                trial = list(taus)
                trial[slot] = float(t)
                s = sse(trial)
                if s < best[0]:
                    best = (s, float(t))
            taus[slot] = best[1]

    X = design(taus)
    theta = solve(X)
    h = {fld: {} for fld in STATIC_FIELDS}
    for (fld, lv), v in zip(static_key, theta[: len(static_key)]):
        h[fld][int(lv)] = float(v)
    rest = theta[len(static_key):]
    ripple = {int(lv): {"a": [], "b": []} for lv in lna_levels}
    n = len(lna_levels)
    for k in range(n_ripples):
        a_blk = rest[2 * k * n: (2 * k + 1) * n]
        b_blk = rest[(2 * k + 1) * n: (2 * k + 2) * n]
        for lv, a_v, b_v in zip(lna_levels, a_blk, b_blk):
            ripple[int(lv)]["a"].append(float(a_v))
            ripple[int(lv)]["b"].append(float(b_v))

    return GainStatePhaseModel(
        tau_seconds=tuple(taus), h=h, ripple=ripple, name=name, tables=tab,
        provenance={
            "n_rows": int(len(y)),
            "n_columns": int(X.shape[1]),
            "rank": int(np.linalg.matrix_rank(X)),
            "ridge": ridge,
            "static_fields": list(static_fields),
            "n_ripples": n_ripples,
        },
    )


def evaluate(frames, folds, *, static_fields, n_ripples=2, name="rung",
             tau_search_rows=None):
    """Fit per fold on the training rows only, score the held-out rows fail-closed."""
    D = frames["D"]
    pred = np.zeros(len(D))
    sup = np.zeros(len(D), dtype=bool)
    seen = np.zeros(len(D), dtype=bool)
    n_folds = 0
    for _lbl, tr, te in folds:
        if tr.sum() == 0 or te.sum() == 0:
            continue
        n_folds += 1
        m = fit_rung(
            frames["lo_hz"][tr], frames["g1"][tr], frames["g2"][tr], D[tr],
            rf_hz=frames["rf_hz"][tr], static_fields=static_fields,
            n_ripples=n_ripples, tau_search_rows=tau_search_rows,
        )
        for i in np.nonzero(te)[0]:
            p = m.predict(
                frames["lo_hz"][i], int(frames["g1"][i]), int(frames["g2"][i]),
                rf_hz=frames["rf_hz"][i], apply_rf_state_guard=False,
            )
            pred[i] = p.residual_rad if p.supported else 0.0
            sup[i] = p.supported
            seen[i] = True
    fail_closed = np.where(sup, pred, 0.0)
    err = wrap(D[seen] - fail_closed[seen])
    uneq = (frames["g1"] != frames["g2"])[seen]
    return {
        "n_folds": n_folds,
        "coverage": float(sup[seen].mean()),
        "all_cells": circ_stats(err),
        "unequal_gain_cells": circ_stats(err[uneq]),
        "baseline_no_correction": circ_stats(D[seen]),
        "_err": err,
        "_seen": seen,
        "_sup": sup,
        "_pred": fail_closed,
    }


# --------------------------------------------------------------------- splits
def folds_leave_one_out(frames, key):
    v = frames[key]
    for u in np.unique(v):
        te = v == u
        yield f"{key}={u}", ~te, te


def folds_leave_freq_block_out(frames, n_blocks=8):
    los = np.unique(frames["lo_hz"])
    for k, idx in enumerate(np.array_split(np.arange(len(los)), n_blocks)):
        block = set(los[idx].tolist())
        te = np.array([x in block for x in frames["lo_hz"]])
        yield f"block{k}({los[idx][0]/1e6:.0f}-{los[idx][-1]/1e6:.0f})", ~te, te


def selfcheck(frames, n=400, seed=0):
    """Prove fit_rung(L26_FIELDS) == GainStatePhaseModel.fit on the same rows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frames["D"]), size=min(n, len(frames["D"])), replace=False)
    kw = dict(
        lo_hz=frames["lo_hz"][idx], gain_rx1_db=frames["g1"][idx],
        gain_rx2_db=frames["g2"][idx], residual_rad=frames["D"][idx],
        rf_hz=frames["rf_hz"][idx],
    )
    a = GainStatePhaseModel.fit(**kw)
    b = fit_rung(frames["lo_hz"][idx], frames["g1"][idx], frames["g2"][idx],
                 frames["D"][idx], rf_hz=frames["rf_hz"][idx],
                 static_fields=L26_FIELDS)
    dev = max(
        abs(a.tau_seconds[0] - b.tau_seconds[0]),
        abs(a.tau_seconds[1] - b.tau_seconds[1]),
    )
    hdev = 0.0
    for fld in STATIC_FIELDS:
        assert set(a.h[fld]) == set(b.h[fld]), fld
        for lv in a.h[fld]:
            hdev = max(hdev, abs(a.h[fld][lv] - b.h[fld][lv]))
    rdev = 0.0
    for lv in a.ripple:
        for c in ("a", "b"):
            for x, y in zip(a.ripple[lv][c], b.ripple[lv][c]):
                rdev = max(rdev, abs(x - y))
    return {
        "n_rows": int(len(idx)),
        "max_tau_deviation_s": float(dev),
        "max_h_deviation_rad": float(hdev),
        "max_ripple_deviation_rad": float(rdev),
        "identical": bool(dev == 0.0 and hdev == 0.0 and rdev == 0.0),
    }
