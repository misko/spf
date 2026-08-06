"""The L26 gain-state phase model for the AD9361 dual-RX pair.

Predicts the *gain-dependent* part of the ``angle(RX1) - angle(RX2)`` phase
offset, given a measured per-session equal-gain anchor at the operating LO::

    D(f, g1, g2) = H(s1) - H(s2)
                 + sum_k [ a_k(l1) - a_k(l2) ] * cos(2*pi*f*tau_k)
                        + [ b_k(l1) - b_k(l2) ] * sin(2*pi*f*tau_k)

    s = (LNA, MIXER, TIA, LPF) read from the audited gain table for (band, dB)
    l = the LNA index of that state
    H(s) = h_lna[LNA] + h_mix[MIXER] + h_tia[TIA] + h_lpf[LPF]

    corrected = wrap( measured_RX1_minus_RX2 - anchor(serial, LO, session) - D )

The anchor is a **measurement**, not a fitted parameter, and it is the only
radio-specific state the model needs. Everything in ``coefficients/`` is
universal across the audited radios.

Three properties of this parameterisation are load-bearing and are enforced here
rather than left to the caller:

1. **Fail closed.** A hardware state the fit never saw is refused, never
   extrapolated. ``UnsupportedGainState`` is raised (or ``None`` returned by the
   ``try_`` variants); no code path silently returns zero for an unknown cell.
2. **The RF-state guard (deployment rule 5).** Where the audited
   ``(LNA, MIXER, TIA)`` words are identical on both arms there is no phase
   resolvable by the source experiment, and the fitted baseband-LPF differences
   are noise. Applying the correction there *injects* a mean 1.36 deg and makes
   81% of such cells worse, so the guard returns exactly zero instead. It is on
   by default.
3. **Coefficients are not individually meaningful.** The signed-indicator design
   is rank-deficient by construction (adding a constant to any ``h`` family
   cancels in every prediction). Only signed differences are identified. Read
   ``h_lna[2] - h_lna[0]``, never ``h_lna[2]`` alone.

See ``README.md`` for the physical backing, the measured performance, and the
limitations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .gain_tables import GainTables, HardwareState, band_for_lo, default_tables

COEFFICIENT_DIR = Path(__file__).resolve().parent / "coefficients"

#: Fields whose levels index the static ``H`` term.
STATIC_FIELDS = ("lna", "mixer", "tia", "lpf")

#: The field whose level indexes the reflection-ripple amplitudes.
RIPPLE_FIELD = "lna"


class UnsupportedGainState(Exception):
    """Raised when a requested cell invokes a state the fit never estimated.

    Carries the offending arm and field so a caller can log precisely which
    hardware state is missing -- that is the information an operator needs to
    decide whether to schedule a fill-in capture (see E-CAL2 in ``README.md``).
    """

    def __init__(self, message: str, *, arm: int | None = None,
                 field_name: str | None = None, level: int | None = None):
        super().__init__(message)
        self.arm = arm
        self.field_name = field_name
        self.level = level


def wrap(x: float) -> float:
    """Wrap radians to (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Prediction:
    """The outcome of one prediction, with the reason it took the value it did."""

    residual_rad: float
    supported: bool
    guarded: bool
    state_rx1: HardwareState | None
    state_rx2: HardwareState | None
    reason: str = ""

    @property
    def residual_deg(self) -> float:
        return math.degrees(self.residual_rad)


@dataclass
class GainStatePhaseModel:
    """A fitted gain-state model.

    Parameters
    ----------
    tau_seconds
        The reflection-ripple delays, in seconds. Fitted by grid search on the
        training fold only. These are *effective electrical group delays*; they
        do not identify a specific cable, trace or filter.
    h
        ``{field: {level: coefficient_rad}}`` for the static term.
    ripple
        ``{lna_level: {"a": [...], "b": [...]}}`` -- cos and sin amplitudes in
        radians, one pair per delay slot.
    """

    tau_seconds: tuple[float, ...]
    h: dict[str, dict[int, float]]
    ripple: dict[int, dict[str, list[float]]]
    name: str = "L26"
    provenance: dict = field(default_factory=dict)
    tables: GainTables = field(default_factory=default_tables, repr=False)

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: str | Path) -> "GainStatePhaseModel":
        """Load a coefficient file. Degrees on disk, radians in memory."""
        doc = json.loads(Path(path).read_text())
        h = {
            fld: {int(k): math.radians(v) for k, v in doc["h_deg"].get(fld, {}).items()}
            for fld in STATIC_FIELDS
        }
        ripple = {
            int(k): {
                "a": [math.radians(x) for x in v["a_deg"]],
                "b": [math.radians(x) for x in v["b_deg"]],
            }
            for k, v in doc.get("ripple_deg", {}).items()
        }
        return cls(
            tau_seconds=tuple(doc["tau_seconds"]),
            h=h,
            ripple=ripple,
            name=doc.get("model", "L26"),
            provenance=doc.get("provenance", {}),
        )

    @classmethod
    def load_named(cls, name: str = "l26_pooled_v1") -> "GainStatePhaseModel":
        """Load one of the committed coefficient sets by name."""
        path = COEFFICIENT_DIR / f"{name}.json"
        if not path.exists():
            available = sorted(p.stem for p in COEFFICIENT_DIR.glob("*.json"))
            raise FileNotFoundError(
                f"no coefficient set {name!r}; available: {available}"
            )
        return cls.load(path)

    def save(self, path: str | Path, **provenance) -> None:
        doc = {
            "model": self.name,
            "tau_seconds": list(self.tau_seconds),
            "h_deg": {
                fld: {str(k): math.degrees(v) for k, v in sorted(self.h[fld].items())}
                for fld in STATIC_FIELDS
            },
            "ripple_deg": {
                str(k): {
                    "a_deg": [math.degrees(x) for x in v["a"]],
                    "b_deg": [math.degrees(x) for x in v["b"]],
                }
                for k, v in sorted(self.ripple.items())
            },
            "provenance": {**self.provenance, **provenance},
        }
        Path(path).write_text(json.dumps(doc, indent=1) + "\n")

    # -------------------------------------------------------------- support
    @property
    def families_used(self) -> list[str]:
        """Which term families this rung actually carries.

        An empty coefficient family means the model *omits* that term by design,
        not that it is unsupported there. ``L30``/``L31`` deliberately drop the
        categorical baseband-LPF family -- that omission is what raises unseen-
        requested-gain coverage from 48% to 90%, and it makes them immune to the
        rule-5 failure mode by construction.
        """
        out = [fld for fld in STATIC_FIELDS if self.h.get(fld)]
        if self.ripple:
            out.append(f"ripple_{RIPPLE_FIELD}")
        return out

    @property
    def supported_levels(self) -> dict[str, list[int]]:
        out = {fld: sorted(self.h.get(fld) or {}) for fld in STATIC_FIELDS}
        out["ripple_lna"] = sorted(self.ripple)
        return out

    def supported_gains_db(self, lo_hz: float) -> list[int]:
        """Every requested gain at this LO whose state the model can predict."""
        band = band_for_lo(lo_hz)
        lo_db, hi_db = self.tables.gain_range_db(band)
        ok = []
        for g in range(lo_db, hi_db + 1):
            st = self.tables.state(band, g)
            if st is not None and self._missing(st) is None:
                ok.append(g)
        return ok

    def _missing(self, st: HardwareState) -> tuple[str, int] | None:
        """First (field, level) this state invokes that the fit lacks.

        A family with no coefficients at all is omitted by the rung (see
        ``families_used``) and imposes no support requirement. A family that is
        present but lacks *this* level is a genuine coverage hole and fails
        closed -- that is how the unmeasured LNA index 1 is refused rather than
        extrapolated.
        """
        for fld in STATIC_FIELDS:
            table = self.h.get(fld) or {}
            if not table:
                continue
            level = getattr(st, fld)
            if level not in table:
                return (fld, level)
        if self.ripple and getattr(st, RIPPLE_FIELD) not in self.ripple:
            return (f"ripple_{RIPPLE_FIELD}", getattr(st, RIPPLE_FIELD))
        return None

    # ----------------------------------------------------------- prediction
    def _basis(self, rf_hz: float) -> list[tuple[float, float]]:
        return [
            (math.cos(2 * math.pi * rf_hz * t), math.sin(2 * math.pi * rf_hz * t))
            for t in self.tau_seconds
        ]

    def _h_of(self, st: HardwareState) -> float:
        return sum(
            self.h[fld][getattr(st, fld)]
            for fld in STATIC_FIELDS
            if self.h.get(fld)
        )

    def predict(
        self,
        lo_hz: float,
        gain_rx1_db: int,
        gain_rx2_db: int,
        *,
        rf_hz: float | None = None,
        apply_rf_state_guard: bool = True,
    ) -> Prediction:
        """Predict the gain-dependent residual ``D``, with full provenance.

        ``lo_hz`` selects the gain-table band. ``rf_hz`` drives the ripple basis
        and defaults to ``lo_hz``; pass the actual RF tone frequency when it is
        known (the difference is ~1e-3 rad for a 100 kHz offset and is kept only
        for exactness against the source analysis).
        """
        band = band_for_lo(lo_hz)
        s1 = self.tables.state(band, int(gain_rx1_db))
        s2 = self.tables.state(band, int(gain_rx2_db))
        for arm, st, g in ((1, s1, gain_rx1_db), (2, s2, gain_rx2_db)):
            if st is None:
                return Prediction(
                    0.0, False, False, s1, s2,
                    f"RX{arm} gain {g} dB is outside the {band} gain table",
                )
        for arm, st in ((1, s1), (2, s2)):
            miss = self._missing(st)
            if miss is not None:
                return Prediction(
                    0.0, False, False, s1, s2,
                    f"RX{arm} invokes {miss[0]}={miss[1]}, which the fit never estimated",
                )

        if apply_rf_state_guard and s1.rf_words == s2.rf_words:
            return Prediction(
                0.0, True, True, s1, s2,
                "RF words (LNA, MIXER, TIA) identical on both arms -- "
                "deployment rule 5: correction not applied",
            )

        d = self._h_of(s1) - self._h_of(s2)
        for k, (c, s) in enumerate(self._basis(lo_hz if rf_hz is None else rf_hz)):
            a1, b1 = self.ripple[s1.lna]["a"][k], self.ripple[s1.lna]["b"][k]
            a2, b2 = self.ripple[s2.lna]["a"][k], self.ripple[s2.lna]["b"][k]
            d += (a1 - a2) * c + (b1 - b2) * s
        return Prediction(d, True, False, s1, s2, "ok")

    def predict_residual_rad(self, lo_hz: float, gain_rx1_db: int,
                             gain_rx2_db: int, **kw) -> float:
        """``D`` in radians. Raises ``UnsupportedGainState`` if not supported."""
        p = self.predict(lo_hz, gain_rx1_db, gain_rx2_db, **kw)
        if not p.supported:
            raise UnsupportedGainState(p.reason)
        return p.residual_rad

    def correct_measured_phase(
        self,
        measured_phase_rad: float,
        anchor_phase_rad: float,
        lo_hz: float,
        gain_rx1_db: int,
        gain_rx2_db: int,
        **kw,
    ) -> float:
        """Apply the full two-term correction and wrap.

        ``anchor_phase_rad`` is the measured equal-gain cell for this
        ``(serial, exact LO, session)``. It must be re-measured after any
        re-mate, harness change, radio swap or unvalidated boot -- it is never
        transferred. Average three frames where the schedule allows.
        """
        d = self.predict_residual_rad(lo_hz, gain_rx1_db, gain_rx2_db, **kw)
        return wrap(measured_phase_rad - anchor_phase_rad - d)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        lo_hz: Sequence[float],
        gain_rx1_db: Sequence[int],
        gain_rx2_db: Sequence[int],
        residual_rad: Sequence[float],
        *,
        rf_hz: Sequence[float] | None = None,
        tau_grid_seconds: Iterable[float] | None = None,
        n_ripples: int = 2,
        ridge: float = 1e-8,
        tables: GainTables | None = None,
        name: str = "L26",
    ) -> "GainStatePhaseModel":
        """Fit from anchored residuals ``D = phase - measured equal-gain anchor``.

        ``D`` stays well inside +-90 deg, so ordinary least squares on the
        unwrapped value is exact and no circular machinery is needed.

        The delays are chosen by grid search. When fitting for a holdout
        evaluation, call this with the **training fold only** -- searching tau on
        data you then score against is the leakage this module's source analysis
        is careful to avoid.
        """
        import numpy as np

        tab = tables or default_tables()
        lo = np.asarray(lo_hz, dtype=float)
        f_rf = lo if rf_hz is None else np.asarray(rf_hz, dtype=float)
        g1 = np.asarray(gain_rx1_db, dtype=int)
        g2 = np.asarray(gain_rx2_db, dtype=int)
        y = np.asarray(residual_rad, dtype=float)

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
            for fld in STATIC_FIELDS
        }
        lna_levels = levels[RIPPLE_FIELD]

        # signed indicator blocks: column = (arm1 at level) - (arm2 at level)
        static_cols, static_key = [], []
        for fld in STATIC_FIELDS:
            for lv in levels[fld]:
                col = np.array(
                    [
                        (getattr(s1, fld) == lv) - (getattr(s2, fld) == lv)
                        for s1, s2 in states
                    ],
                    dtype=float,
                )
                static_cols.append(col)
                static_key.append((fld, lv))
        sig_lna = np.column_stack(
            [
                np.array([(s1.lna == lv) - (s2.lna == lv) for s1, s2 in states],
                         dtype=float)
                for lv in lna_levels
            ]
        ) if lna_levels else np.zeros((len(states), 0))
        S_static = (np.column_stack(static_cols) if static_cols
                    else np.zeros((len(states), 0)))

        if tau_grid_seconds is None:
            grid = np.concatenate(
                [np.arange(0.10, 4.0, 0.02), np.arange(4.0, 8.01, 0.05)]
            ) * 1e-9
        else:
            grid = np.asarray(list(tau_grid_seconds), dtype=float)

        def design(taus):
            blocks = [S_static]
            for t in taus:
                blocks.append(sig_lna * np.cos(2 * np.pi * f_rf * t)[:, None])
                blocks.append(sig_lna * np.sin(2 * np.pi * f_rf * t)[:, None])
            return np.column_stack(blocks)

        def solve(X):
            A = X.T @ X + ridge * np.eye(X.shape[1])
            return np.linalg.solve(A, X.T @ y)

        def sse(taus):
            X = design(taus)
            r = y - X @ solve(X)
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

        return cls(
            tau_seconds=tuple(taus), h=h, ripple=ripple, name=name,
            tables=tab,
            provenance={"n_rows": int(len(y)), "n_columns": int(X.shape[1]),
                        "rank": int(np.linalg.matrix_rank(X)), "ridge": ridge},
        )
