"""Standalone self-test for the gain-state phase model.

Runs without pytest and without any campaign data::

    python -m spf.calibrations.gain_state_phase_model_v1.selftest

It checks the structural invariants that the model's correctness rests on:

1.  the committed gain tables decode as the report says they do;
2.  the model is exactly antisymmetric in its two arms;
3.  an equal-gain cell predicts exactly zero (so the anchor is self-consistent);
4.  predictions are invariant to the gauge freedom of the rank-deficient design,
    which is why individual coefficients must never be read physically;
5.  unknown hardware states fail closed instead of extrapolating;
6.  the rule-5 RF-state guard fires exactly where it should, and the LPF-free
    rungs are neutral there by construction rather than by guard;
7.  fit -> save -> load -> predict is a faithful round trip.

A separate check against the source analysis pipeline lives in
``PROVENANCE.md``; that one needs the campaign extraction and is not run here.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from .gain_tables import BANDS, band_for_lo, default_tables
from .model import GainStatePhaseModel, UnsupportedGainState

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((PASS if ok else FAIL, name, detail))


# --------------------------------------------------------------------------
def test_gain_tables() -> None:
    t = default_tables()
    n_rows = sum(len(t._bands[b]["bytes"]) for b in BANDS)
    check("gain tables: 231 rows across 3 bands", n_rows == 231, f"got {n_rows}")
    check(
        "gain tables: digital gain identically zero (cannot contribute phase)",
        t.digital_gain_is_zero_everywhere(),
    )
    check(
        "gain tables: verified byte-identical across both audited serials",
        len(t.verified_identical_across_serials) == 2,
    )
    check(
        "band selection: 1300/4000 MHz edges",
        band_for_lo(1_300_000_000) == "low"
        and band_for_lo(1_300_000_001) == "middle"
        and band_for_lo(4_000_000_000) == "middle"
        and band_for_lo(4_000_000_001) == "high",
    )
    # The mechanism claim: the same requested dB is a different hardware state
    # in different bands. 26 dB is LNA 0 in low/middle but LNA 2 in high.
    lo26 = t.state("low", 26)
    hi26 = t.state("high", 26)
    check(
        "same requested dB is a different hardware state per band (26 dB)",
        lo26.lna == 0 and hi26.lna == 2,
        f"low LNA={lo26.lna}, high LNA={hi26.lna}",
    )
    # The 13 dB-with-frozen-state demonstration from the report's section 3.3.
    states = {g: t.state("high", g).rf_words for g in (27, 30, 33, 36, 40)}
    check(
        "high band 27->40 dB leaves the RF words frozen at (2, 4, 1)",
        set(states.values()) == {(2, 4, 1)},
        str(states),
    )
    # LNA index 1 is never reachable on the gains the campaign scheduled; the
    # hole that E-CAL2 exists to fill. It IS reachable in principle:
    reachable = {
        b: sorted({g for g in range(-10, 74) if (s := t.state(b, g)) and s.lna == 1})
        for b in BANDS
    }
    check(
        "LNA index 1 is reachable at 31-32 (low) / 30-31 (middle) / 23-25 dB (high)",
        reachable["low"] == [31, 32]
        and reachable["middle"] == [30, 31]
        and reachable["high"] == [23, 24, 25],
        str(reachable),
    )


def test_model_invariants() -> None:
    m = GainStatePhaseModel.load_named("l26_pooled_v1")
    lo = 2_412_000_000

    # 2. antisymmetry in the arms
    worst = 0.0
    for g1 in (5, 26, 40, 45):
        for g2 in (5, 26, 40, 45):
            a = m.predict(lo, g1, g2, apply_rf_state_guard=False)
            b = m.predict(lo, g2, g1, apply_rf_state_guard=False)
            if a.supported and b.supported:
                worst = max(worst, abs(a.residual_rad + b.residual_rad))
    check("antisymmetry: D(g1,g2) == -D(g2,g1)", worst < 1e-15, f"max dev {worst:.2e}")

    # 3. equal-gain cell is exactly zero -> consistent with the anchor definition
    worst = max(
        abs(m.predict(lo, g, g, apply_rf_state_guard=False).residual_rad)
        for g in (5, 26, 40, 45)
    )
    check("equal-gain cell predicts exactly zero", worst == 0.0, f"max {worst:.2e}")

    # 4. gauge invariance: the design is rank-deficient, so shifting a whole
    #    coefficient family must not move any prediction.
    shifted = GainStatePhaseModel(
        tau_seconds=m.tau_seconds,
        h={f: {k: v + 0.25 for k, v in tab.items()} if f == "lna" else dict(tab)
           for f, tab in m.h.items()},
        ripple={k: {"a": list(v["a"]), "b": list(v["b"])} for k, v in m.ripple.items()},
    )
    worst = 0.0
    for g1 in (5, 26, 45):
        for g2 in (5, 26, 45):
            a = m.predict(lo, g1, g2, apply_rf_state_guard=False)
            b = shifted.predict(lo, g1, g2, apply_rf_state_guard=False)
            if a.supported:
                worst = max(worst, abs(a.residual_rad - b.residual_rad))
    check(
        "gauge invariance: shifting a whole h family changes no prediction",
        worst < 1e-15,
        f"max dev {worst:.2e} -- individual coefficients are NOT identified",
    )

    # 5. fail closed
    p = m.predict(2_412_000_000, 26, 99)
    check("fail closed: out-of-table gain is refused", not p.supported, p.reason)
    raised = False
    try:
        m.predict_residual_rad(2_412_000_000, 26, 99)
    except UnsupportedGainState:
        raised = True
    check("fail closed: predict_residual_rad raises", raised)
    # LNA 1 was never measured -> must be refused, not interpolated
    p = m.predict(2_412_000_000, 30, 26)
    check(
        "fail closed: the unmeasured LNA index 1 is refused (30 dB, middle band)",
        not p.supported,
        p.reason,
    )


def test_rule5_guard() -> None:
    lo = 5_100_000_000  # high band; 27..40 dB share the RF words
    m26 = GainStatePhaseModel.load_named("l26_pooled_v1")
    on = m26.predict(lo, 40, 27, apply_rf_state_guard=True)
    off = m26.predict(lo, 40, 27, apply_rf_state_guard=False)
    check(
        "rule 5: guard fires when (LNA, MIXER, TIA) match on both arms",
        on.guarded and on.residual_rad == 0.0 and on.supported,
        on.reason,
    )
    check(
        "rule 5: without the guard L26 would inject a non-zero LPF-only term",
        abs(off.residual_rad) > 1e-6,
        f"{math.degrees(off.residual_rad):+.3f} deg would be injected",
    )
    for name in ("l30_pooled_v1", "l31_pooled_v1"):
        mm = GainStatePhaseModel.load_named(name)
        v = mm.predict(lo, 40, 27, apply_rf_state_guard=False)
        check(
            f"rule 5: {name} is neutral in this regime BY CONSTRUCTION (no guard needed)",
            v.supported and abs(v.residual_rad) < 1e-15,
            f"{math.degrees(v.residual_rad):+.2e} deg; families={mm.families_used}",
        )


def test_fit_round_trip() -> None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        check("fit round trip", False, "numpy unavailable")
        return

    truth = GainStatePhaseModel.load_named("l26_pooled_v1")
    rng = np.random.default_rng(0)
    los = np.linspace(400e6, 5900e6, 90)
    gains = [5, 26, 45]
    rows = [
        (lo, g1, g2)
        for lo in los
        for g1 in gains
        for g2 in gains
        if truth.predict(lo, g1, g2, apply_rf_state_guard=False).supported
    ]
    lo_a = np.array([r[0] for r in rows])
    g1_a = np.array([r[1] for r in rows])
    g2_a = np.array([r[2] for r in rows])
    y = np.array(
        [
            truth.predict(lo, g1, g2, apply_rf_state_guard=False).residual_rad
            for lo, g1, g2 in rows
        ]
    )
    y_noisy = y + rng.normal(0, math.radians(0.05), size=len(y))

    got = GainStatePhaseModel.fit(lo_a, g1_a, g2_a, y_noisy)
    pred = np.array(
        [
            got.predict(lo, g1, g2, apply_rf_state_guard=False).residual_rad
            for lo, g1, g2 in rows
        ]
    )
    mae = float(np.degrees(np.abs(pred - y)).mean())
    check(
        "fit round trip: recovers a known model from noisy synthetic data",
        mae < 0.05,
        f"MAE vs truth {mae:.4f} deg (0.05 deg noise injected)",
    )
    check(
        "fit round trip: recovers both ripple delays",
        all(
            min(abs(t - u) for u in truth.tau_seconds) < 0.05e-9
            for t in got.tau_seconds
        ),
        f"got {[round(t*1e9, 3) for t in got.tau_seconds]} ns, "
        f"truth {[round(t*1e9, 3) for t in truth.tau_seconds]} ns",
    )

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "rt.json"
        got.save(p)
        back = GainStatePhaseModel.load(p)
        worst = max(
            abs(
                got.predict(lo, g1, g2, apply_rf_state_guard=False).residual_rad
                - back.predict(lo, g1, g2, apply_rf_state_guard=False).residual_rad
            )
            for lo, g1, g2 in rows[:200]
        )
        check("save/load round trip is exact", worst < 1e-12, f"max dev {worst:.2e}")


def main() -> int:
    test_gain_tables()
    test_model_invariants()
    test_rule5_guard()
    test_fit_round_trip()

    width = max(len(n) for _, n, _ in _results)
    for status, name, detail in _results:
        line = f"[{status}] {name.ljust(width)}"
        if detail:
            line += f"   {detail}"
        print(line)
    failed = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
