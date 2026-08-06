"""Tests for the L26 gain-state phase model package.

These are structural/contract tests: they need no campaign data and no network.
The numerical agreement with the source analysis pipeline is a separate,
data-dependent check documented in
``spf/calibrations/gain_state_phase_model_v1/PROVENANCE.md``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from spf.calibrations.gain_state_phase_model_v1 import (
    GainStatePhaseModel,
    UnsupportedGainState,
    band_for_lo,
    default_tables,
)
from spf.calibrations.gain_state_phase_model_v1.model import COEFFICIENT_DIR

COEFFICIENT_SETS = ["l26_pooled_v1", "l26_stage_a_v1", "l30_pooled_v1", "l31_pooled_v1"]


@pytest.fixture(scope="module")
def tables():
    return default_tables()


@pytest.fixture(scope="module")
def l26():
    return GainStatePhaseModel.load_named("l26_pooled_v1")


# --------------------------------------------------------------- gain tables
def test_gain_tables_have_231_rows(tables):
    assert sum(len(tables._bands[b]["bytes"]) for b in ("low", "middle", "high")) == 231


def test_digital_gain_is_identically_zero(tables):
    """Digital gain cannot contribute phase -- the premise for reading byte 2
    bit 5 as RF_DC_CAL rather than as part of a digital gain field."""
    assert tables.digital_gain_is_zero_everywhere()


def test_tables_verified_identical_across_both_audited_serials(tables):
    assert len(tables.verified_identical_across_serials) == 2


@pytest.mark.parametrize(
    "lo_hz,band",
    [
        (400e6, "low"),
        (1_300_000_000, "low"),
        (1_300_000_001, "middle"),
        (4_000_000_000, "middle"),
        (4_000_000_001, "high"),
        (5_900e6, "high"),
    ],
)
def test_band_edges(lo_hz, band):
    assert band_for_lo(lo_hz) == band


def test_same_db_is_a_different_hardware_state_per_band(tables):
    """26 dB is LNA 0 in low/middle but LNA 2 in high. This is why a model
    keyed on requested dB cannot represent the band-edge steps."""
    assert tables.state("low", 26).lna == 0
    assert tables.state("middle", 26).lna == 0
    assert tables.state("high", 26).lna == 2


def test_rf_state_frozen_across_27_to_40_db_in_high_band(tables):
    """13 dB of requested gain with no RF-word change -- the cleanest single
    demonstration that phase tracks hardware state, not requested dB."""
    words = {tables.state("high", g).rf_words for g in range(27, 41)}
    assert words == {(2, 4, 1)}


def test_out_of_table_gain_returns_none(tables):
    assert tables.state("high", -50) is None
    assert tables.state("middle", 200) is None


def test_lna_index_1_is_reachable_but_was_never_measured(tables, l26):
    """E-CAL2's coverage hole: the states exist in the table, but no scheduled
    gain in the source campaign visited them, so the model must refuse them."""
    reachable = {
        b: sorted({g for g in range(-10, 74)
                   if (s := tables.state(b, g)) is not None and s.lna == 1})
        for b in ("low", "middle", "high")
    }
    assert reachable == {"low": [31, 32], "middle": [30, 31], "high": [23, 24, 25]}
    assert 1 not in l26.supported_levels["lna"]


# -------------------------------------------------------------- coefficients
@pytest.mark.parametrize("name", COEFFICIENT_SETS)
def test_every_coefficient_set_loads_and_carries_provenance(name):
    m = GainStatePhaseModel.load_named(name)
    assert m.provenance["spf_git_sha"]
    assert m.provenance["phase_convention"] == "angle(RX1) - angle(RX2), radians"
    assert m.provenance["rank"] <= m.provenance["n_columns"]


def test_coefficient_files_are_valid_json():
    for p in COEFFICIENT_DIR.glob("*.json"):
        json.loads(p.read_text())


def test_unknown_coefficient_set_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="l26_pooled_v1"):
        GainStatePhaseModel.load_named("does_not_exist")


# ----------------------------------------------------------------- invariants
@pytest.mark.parametrize("name", COEFFICIENT_SETS)
def test_antisymmetry_in_the_two_arms(name):
    m = GainStatePhaseModel.load_named(name)
    for g1 in (5, 26, 40, 45):
        for g2 in (5, 26, 40, 45):
            a = m.predict(2_412e6, g1, g2, apply_rf_state_guard=False)
            b = m.predict(2_412e6, g2, g1, apply_rf_state_guard=False)
            if a.supported and b.supported:
                assert a.residual_rad == pytest.approx(-b.residual_rad, abs=1e-15)


@pytest.mark.parametrize("name", COEFFICIENT_SETS)
def test_equal_gain_cell_predicts_exactly_zero(name):
    """The anchor is the equal-gain cell, so the model must predict zero there
    or the correction would double-count."""
    m = GainStatePhaseModel.load_named(name)
    for g in (5, 26, 45):
        p = m.predict(2_412e6, g, g, apply_rf_state_guard=False)
        if p.supported:
            assert p.residual_rad == 0.0


def test_predictions_are_gauge_invariant(l26):
    """The signed-indicator design is rank-deficient: only signed differences
    are identified. Shifting a whole coefficient family must change nothing."""
    shifted = GainStatePhaseModel(
        tau_seconds=l26.tau_seconds,
        h={f: ({k: v + 0.37 for k, v in tab.items()} if f == "mixer" else dict(tab))
           for f, tab in l26.h.items()},
        ripple={k: {"a": list(v["a"]), "b": list(v["b"])}
                for k, v in l26.ripple.items()},
    )
    for g1 in (5, 26, 45):
        for g2 in (5, 26, 45):
            a = l26.predict(2_412e6, g1, g2, apply_rf_state_guard=False)
            b = shifted.predict(2_412e6, g1, g2, apply_rf_state_guard=False)
            if a.supported:
                assert a.residual_rad == pytest.approx(b.residual_rad, abs=1e-15)


def test_rank_is_strictly_less_than_column_count(l26):
    """Documents that the parameter count is an upper bound, not estimable
    rank -- 38 columns, rank 29 on the pooled fit."""
    assert l26.provenance["rank"] < l26.provenance["n_columns"]


# ---------------------------------------------------------------- fail closed
def test_out_of_table_gain_fails_closed(l26):
    p = l26.predict(2_412e6, 26, 99)
    assert not p.supported
    assert p.residual_rad == 0.0
    assert "outside" in p.reason


def test_unmeasured_lna_state_fails_closed(l26):
    p = l26.predict(2_412e6, 30, 26)
    assert not p.supported
    assert "lna=1" in p.reason


def test_raising_api_raises_rather_than_returning_zero(l26):
    with pytest.raises(UnsupportedGainState):
        l26.predict_residual_rad(2_412e6, 30, 26)
    with pytest.raises(UnsupportedGainState):
        l26.correct_measured_phase(0.1, 0.0, 2_412e6, 30, 26)


def test_supported_gain_list_is_a_strict_subset_of_the_table(l26, tables):
    ok = l26.supported_gains_db(2_412e6)
    lo_db, hi_db = tables.gain_range_db("middle")
    assert 0 < len(ok) < (hi_db - lo_db + 1)
    assert 30 not in ok and 31 not in ok  # the LNA 1 hole


# ------------------------------------------------------------ rule 5 guard
def test_rule5_guard_fires_when_rf_words_match(l26):
    p = l26.predict(5_100e6, 40, 27, apply_rf_state_guard=True)
    assert p.supported and p.guarded and p.residual_rad == 0.0


def test_rule5_guard_is_what_suppresses_the_lpf_only_term(l26):
    """Without the guard L26 injects a baseband-LPF-only correction where the
    source experiment measures no phase at all."""
    off = l26.predict(5_100e6, 40, 27, apply_rf_state_guard=False)
    assert off.supported and not off.guarded
    assert abs(off.residual_rad) > 1e-6


@pytest.mark.parametrize("name", ["l30_pooled_v1", "l31_pooled_v1"])
def test_lpf_free_rungs_are_neutral_without_any_guard(name):
    """L30/L31 carry no categorical LPF term, so they are already zero where
    the RF words match -- neutral by construction, not by guard."""
    m = GainStatePhaseModel.load_named(name)
    p = m.predict(5_100e6, 40, 27, apply_rf_state_guard=False)
    assert p.supported and p.residual_rad == pytest.approx(0.0, abs=1e-15)
    assert "lpf" not in m.families_used


def test_guard_does_not_fire_when_rf_words_differ(l26):
    p = l26.predict(2_412e6, 45, 26, apply_rf_state_guard=True)
    assert p.supported and not p.guarded and p.residual_rad != 0.0


# ---------------------------------------------------------------- correction
def test_correction_subtracts_anchor_and_residual_then_wraps(l26):
    lo, g1, g2 = 2_412e6, 45, 26
    measured, anchor = math.radians(112.0), math.radians(95.4802)
    d = l26.predict_residual_rad(lo, g1, g2)
    got = l26.correct_measured_phase(measured, anchor, lo, g1, g2)
    assert got == pytest.approx(
        (measured - anchor - d + math.pi) % (2 * math.pi) - math.pi
    )


def test_correction_result_is_always_wrapped(l26):
    got = l26.correct_measured_phase(math.radians(179.0), math.radians(-179.0),
                                     2_412e6, 45, 26)
    assert -math.pi < got <= math.pi


# ----------------------------------------------------------------------- fit
def test_fit_recovers_a_known_model_and_its_delays():
    truth = GainStatePhaseModel.load_named("l26_pooled_v1")
    los = np.linspace(400e6, 5900e6, 70)
    rows = [(lo, a, b) for lo in los for a in (5, 26, 45) for b in (5, 26, 45)
            if truth.predict(lo, a, b, apply_rf_state_guard=False).supported]
    y = np.array([truth.predict(lo, a, b, apply_rf_state_guard=False).residual_rad
                  for lo, a, b in rows])
    got = GainStatePhaseModel.fit(
        np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]), y,
    )
    pred = np.array([got.predict(lo, a, b, apply_rf_state_guard=False).residual_rad
                     for lo, a, b in rows])
    assert np.degrees(np.abs(pred - y)).max() < 0.01
    for t in got.tau_seconds:
        assert min(abs(t - u) for u in truth.tau_seconds) < 0.05e-9


def test_save_load_round_trip_is_exact(tmp_path: Path):
    m = GainStatePhaseModel.load_named("l26_pooled_v1")
    p = tmp_path / "rt.json"
    m.save(p, note="round trip")
    back = GainStatePhaseModel.load(p)
    assert back.tau_seconds == m.tau_seconds
    for g1 in (5, 26, 45):
        for g2 in (5, 26, 45):
            a = m.predict(2_412e6, g1, g2, apply_rf_state_guard=False)
            b = back.predict(2_412e6, g1, g2, apply_rf_state_guard=False)
            assert a.residual_rad == pytest.approx(b.residual_rad, abs=1e-12)


def test_selftest_module_passes():
    from spf.calibrations.gain_state_phase_model_v1 import selftest

    selftest._results.clear()
    assert selftest.main() == 0
