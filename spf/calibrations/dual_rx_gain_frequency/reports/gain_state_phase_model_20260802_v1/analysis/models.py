"""Model ladder for the AD9361 dual-RX gain/frequency phase difference.

Every model predicts D = phase - measured equal-gain anchor, in radians.
Models are linear in their parameters given the (optional) ripple delays, which
are chosen by grid search on the TRAINING fold only.

Term algebra
------------
A term is (value_field, basis, groups, arm_specific):
  value_field  what indexes the gain effect
               'g'                        requested dB
               'lna','mixer','tia','lpf'  audited AD9361 hardware-state fields
               'row'                      audited gain-table row index
  basis        how it multiplies frequency
               'const'          1                     lumped / baseband phase
               'delay'          2*pi*f                pure time delay
               'cosK','sinK'    cos/sin(2*pi*f*tau_K) reflection ripple
               'lin','quad'     centred f, f^2        smooth nuisance
  groups       subset of ('serial','band','lo_hz') -> separate copies
  arm_specific False: D = H(g1) - H(g2)   (antisymmetric, arms share H)
               True : D = d1(g1) - d2(g2) (arm-specific)

Design construction is split into a static signed-indicator matrix and a
frequency-basis multiplier so that a ripple grid search only redoes cheap
vector products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import features as FT
import spflib as S

F_SCALE = 1e9


@dataclass(frozen=True)
class Term:
    value_field: str
    basis: str = "const"
    groups: tuple = ()
    arm_specific: bool = False


def _field_values(f: S.Frames, name: str):
    if name.endswith("_lin"):
        name = name[:-4]
    if name == "g":
        return f.g1, f.g2
    return FT.HW.vec(f.band, f.g1, name), FT.HW.vec(f.band, f.g2, name)


def _group_key(f: S.Frames, groups):
    if not groups:
        return np.zeros(len(f), dtype=object)
    out = None
    for g in groups:
        p = np.array([str(x) for x in getattr(f, g)], dtype=object)
        out = p if out is None else np.array([a + "|" + b for a, b in zip(out, p)],
                                             dtype=object)
    return out


class Design:
    """Static signed indicators + per-column basis kind.

    `I` marks which parameters a row *invokes* (its gain level is that column's
    level), independent of sign cancellation. A test row is unsupported only if
    it invokes a parameter that training could not estimate -- an equal-gain
    cell whose signed row cancels to zero is a genuine prediction, not a hole.
    """

    def __init__(self, Smat, Imat, basis_kinds, colnames, freq_hz):
        self.S = Smat  # (n, p) signed indicators
        self.I = Imat  # (n, p) invoked-parameter indicators
        self.basis = basis_kinds  # list[str] length p
        self.names = colnames
        self.f = freq_hz

    def matrix(self, taus):
        n, p = self.S.shape
        X = np.empty((n, p))
        cache = {}
        for j, kind in enumerate(self.basis):
            if kind not in cache:
                cache[kind] = self._basis(kind, taus)
            X[:, j] = self.S[:, j] * cache[kind]
        return X

    def _basis(self, kind, taus):
        fr = self.f
        if kind == "const":
            return np.ones_like(fr)
        if kind == "delay":
            return 2 * np.pi * fr / F_SCALE
        if kind == "lin":
            return fr / F_SCALE - 3.0
        if kind == "quad":
            return (fr / F_SCALE - 3.0) ** 2
        if kind.startswith("cos"):
            return np.cos(2 * np.pi * fr * taus[int(kind[3:])])
        if kind.startswith("sin"):
            return np.sin(2 * np.pi * fr * taus[int(kind[3:])])
        raise ValueError(kind)


def build_design(f: S.Frames, terms) -> Design:
    cols, inv, kinds, names = [], [], [], []
    for t in terms:
        v1, v2 = _field_values(f, t.value_field)
        gk = _group_key(f, t.groups)
        groups = sorted(set(np.unique(gk)))
        if t.value_field.endswith("_lin"):
            # one continuous slope column per group: D += k * (state1 - state2)
            for gg in groups:
                gm = (gk == gg).astype(float) if t.groups else 1.0
                # a row whose state is unknown (-999 sentinel) contributes no
                # slope and is marked uninvoked, so it fails closed
                ok = (v1 != -999) & (v2 != -999)
                d = np.where(ok, (v1 - v2), 0).astype(float)
                cols.append(d * gm)
                inv.append(ok.astype(float) * (gm if t.groups else 1.0))
                kinds.append(t.basis)
                names.append(f"{t.value_field}|{gg}|{t.basis}|slope")
            continue
        # NB parentheses: '-' binds tighter than '|', so the sentinel must be
        # removed from the union, not just from the second operand
        levels = sorted((set(np.unique(v1)) | set(np.unique(v2))) - {-999})
        for lv in levels:
            for gg in groups:
                gm = (gk == gg).astype(float) if t.groups else 1.0
                h1 = (v1 == lv).astype(float)
                h2 = (v2 == lv).astype(float)
                if t.arm_specific:
                    cols.append(h1 * gm)
                    inv.append(h1 * gm)
                    kinds.append(t.basis)
                    names.append(f"{t.value_field}={lv}|{gg}|{t.basis}|rx1")
                    cols.append(-h2 * gm)
                    inv.append(h2 * gm)
                    kinds.append(t.basis)
                    names.append(f"{t.value_field}={lv}|{gg}|{t.basis}|rx2")
                else:
                    cols.append((h1 - h2) * gm)
                    inv.append(np.maximum(h1, h2) * gm)
                    kinds.append(t.basis)
                    names.append(f"{t.value_field}={lv}|{gg}|{t.basis}|sym")
    Smat = np.column_stack(cols) if cols else np.zeros((len(f), 0))
    Imat = np.column_stack(inv) if inv else np.zeros((len(f), 0))
    return Design(Smat, Imat, kinds, names, f.rf_hz)


class LadderModel:
    """A rung of the ladder. `tau_grids` gives one search grid per ripple slot."""

    def __init__(self, name, terms, tau_grids=(), ridge=1e-8, note="", tier=0):
        self.name = name
        self.terms = list(terms)
        self.tau_grids = list(tau_grids)
        self.ridge = ridge
        self.note = note
        self.tier = tier

    # ---- fitting against a prebuilt Design and a row mask
    def _solve(self, X, y):
        if X.shape[1] == 0:
            return np.zeros(0)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)

    TAU_SEARCH_ROWS = 1600  # tau is one scalar; a subsample pins it precisely

    def _search_taus(self, design, y, train_idx, active, rng):
        """Coarse grid then local refinement, on a training subsample."""
        sub = train_idx
        if len(sub) > self.TAU_SEARCH_ROWS:
            sub = rng.choice(train_idx, self.TAU_SEARCH_ROWS, replace=False)
        Ssub = design.S[sub][:, active]
        fsub = design.f[sub]
        ysub = y[sub]
        kinds = [k for k, a in zip(design.basis, active) if a]
        scratch = Design(Ssub, Ssub, kinds, None, fsub)

        def sse(trial):
            X = scratch.matrix(trial)
            th = self._solve(X, ysub)
            r = ysub - X @ th
            return float(r @ r)

        taus = np.array([g[len(g) // 2] for g in self.tau_grids], dtype=float)
        for coarse in (True, False):
            for slot, grid in enumerate(self.tau_grids):
                if len(grid) == 1:
                    taus[slot] = grid[0]
                    continue
                if coarse:
                    cand = grid[:: max(1, len(grid) // 80)]
                else:
                    step = float(np.median(np.diff(grid)))
                    cand = taus[slot] + np.arange(-10, 11) * step
                    cand = cand[(cand >= grid[0]) & (cand <= grid[-1])]
                best = (np.inf, taus[slot])
                for t in cand:
                    trial = taus.copy()
                    trial[slot] = t
                    s = sse(trial)
                    if s < best[0]:
                        best = (s, t)
                taus[slot] = best[1]
        return taus

    def fit_eval(self, design: Design, y, train_idx, test_idx, rng=None):
        """Returns (pred_test, supported_test, taus, n_effective_params)."""
        Str = design.S[train_idx]
        active = np.any(np.abs(Str) > 0, axis=0)  # estimable in training
        if not self.tau_grids:
            taus = np.zeros(0)
            X = design.matrix(taus)
        else:
            rng = rng or np.random.default_rng(0)
            taus = self._search_taus(design, y, train_idx, active, rng)
            X = design.matrix(taus)
        Xtr = X[train_idx][:, active]
        theta = self._solve(Xtr, y[train_idx])
        Xte = X[test_idx][:, active]
        pred = Xte @ theta
        # unsupported == the row invokes a parameter training could not estimate
        needed = design.I[test_idx] > 0
        missing = needed & ~active[None, :]
        supported = ~np.any(missing, axis=1)
        return pred, supported, taus, int(active.sum())
