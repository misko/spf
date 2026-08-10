"""Every theta filter must report whether its stated confidence means anything.

E-INF1 H3 asks whether the filters are overconfident (median ``std(z)`` > 1.5).
Stage 2 could not answer it: 26,112 runs recorded only ``mse_*`` and ``runtime``,
and calibration cannot be reconstructed from a stored MSE. These tests pin the
contract that makes it answerable, and the two ways it silently breaks:

* a non-scalar metric -- ``spf/evaluation/aggregate.py`` calls ``float()`` on
  every metric, so a list or array raises TypeError and removes the whole filter
  family from the report;
* a metric present on only some runs -- the aggregator's ``dropped_metrics`` path
  discards it, which in a report looks exactly like "the filter was fine".
"""

import numpy as np
import pytest
import torch

from spf.evaluation import calibration
from spf.filters.filters import (
    _calibration_metrics,
    dual_radio_mse_theta_metrics,
    single_radio_mse_theta_metrics,
    theta_sigma,
)

CALIB_KEYS = {"calib_std_z", "calib_cov1", "calib_cov2", "calib_cov3", "calib_n"}


def pf_trajectory(thetas, variances):
    """A particle filter's shape: P_theta set on every entry."""
    return [
        {"theta": np.float64(t), "craft_theta": np.float64(t), "P_theta": np.float64(v)}
        for t, v in zip(thetas, variances)
    ]


def ekf_trajectory(thetas, variances):
    """An EKF's shape, as built at ekf_single_radio_filter.py:153 and
    ekf_dualradio_filter.py:183: theta is the scalar ``x[0, 0]``, the state and
    the full covariance are the 2-D ``mu``/``var``, and P_theta appears only
    under debug=True. The off-diagonal 9.9 is theta_dot's variance -- picking it
    up instead of var[0,0] would be a silent wrong answer, so it differs."""
    return [
        {
            "mu": np.array([[t], [0.0]]),
            "theta": t,
            "craft_theta": t,
            "var": np.array([[v, 0.0], [0.0, 9.9]]),
        }
        for t, v in zip(thetas, variances)
    ]


# --------------------------------------------------------------- theta_sigma


def test_sigma_is_the_square_root_of_the_reported_variance():
    traj = pf_trajectory([0.0, 0.0, 0.0], [0.25, 4.0, 1.0])
    np.testing.assert_allclose(theta_sigma(traj), [0.5, 2.0, 1.0])


def test_ekf_sigma_is_recovered_from_var_without_debug():
    """The EKFs only set P_theta under debug=True, but var[0,0] is that number.

    Reading var means the sweep gets calibration without turning on a debug path
    that also allocates jacobians and observations on every timestep.
    """
    traj = ekf_trajectory([0.0, 0.0], [0.25, 4.0])
    np.testing.assert_allclose(theta_sigma(traj), [0.5, 2.0])


def test_p_theta_wins_over_var_when_both_are_present():
    traj = [{"P_theta": 0.25, "var": np.array([[100.0, 0.0], [0.0, 1.0]])}]
    np.testing.assert_allclose(theta_sigma(traj), [0.5])


def test_a_filter_reporting_no_uncertainty_raises_rather_than_scoring_nan():
    """A silent NaN column found after a 40-minute sweep is the failure mode
    this guards; the smoke test should see it instead."""
    with pytest.raises(KeyError, match="P_theta"):
        theta_sigma([{"theta": 0.1}])


# -------------------------------------------------- the calibration block


def test_calibration_recovers_an_injected_sigma():
    """Errors drawn at exactly sigma must score std(z) = 1 and nominal coverage."""
    rng = np.random.default_rng(0)
    n, sigma = 20000, 0.3
    truth = rng.uniform(-np.pi, np.pi, n)
    pred = truth + rng.normal(0.0, sigma, n)
    out = _calibration_metrics(pred, truth, pf_trajectory(pred, np.full(n, sigma**2)))
    assert np.isclose(out["calib_std_z"], 1.0, rtol=0.05)
    assert np.isclose(out["calib_cov1"], calibration.NOMINAL_COVERAGE[1], atol=0.02)
    assert np.isclose(out["calib_cov2"], calibration.NOMINAL_COVERAGE[2], atol=0.02)


def test_an_overconfident_filter_scores_above_one():
    """Report sigma 3x too small: std(z) ~ 3 and +-1 sigma coverage collapses."""
    rng = np.random.default_rng(1)
    n, true_sigma = 20000, 0.3
    truth = rng.uniform(-np.pi, np.pi, n)
    pred = truth + rng.normal(0.0, true_sigma, n)
    claimed = (true_sigma / 3.0) ** 2
    out = _calibration_metrics(pred, truth, pf_trajectory(pred, np.full(n, claimed)))
    assert np.isclose(out["calib_std_z"], 3.0, rtol=0.05)
    assert out["calib_cov1"] < 0.30


def test_calibration_uses_the_short_way_round_the_seam():
    """+179 deg predicted against -179 deg truth is a 2 deg error, not 358."""
    truth = np.full(64, np.pi - 0.01)
    pred = np.full(64, -np.pi + 0.01)
    out = _calibration_metrics(pred, truth, pf_trajectory(pred, np.full(64, 1.0)))
    assert out["calib_std_z"] < 0.1
    assert out["calib_cov1"] == 1.0


def test_nonpositive_sigma_is_dropped_from_the_denominator():
    """calib_n is not len(trajectory): a filter reporting zero variance on some
    steps cannot be scored on those steps, and counting them would understate
    every coverage number."""
    variances = [1.0, 1.0, 0.0, -1.0, 1.0]
    out = _calibration_metrics(
        np.zeros(5), np.zeros(5), pf_trajectory(np.zeros(5), variances)
    )
    assert out["calib_n"] == 3.0


# ------------------------------------------- the aggregator's two contracts


@pytest.mark.parametrize("make", [pf_trajectory, ekf_trajectory])
def test_single_radio_metrics_are_all_scalars(make):
    traj = make(np.linspace(-1.0, 1.0, 32), np.full(32, 0.04))
    out = single_radio_mse_theta_metrics(traj, torch.zeros(32))
    assert CALIB_KEYS <= set(out)
    for key, value in out.items():
        # aggregate.py does float(v) on every metric; anything else kills the
        # whole family's report with a TypeError
        assert float(value) == pytest.approx(float(value)), key
        assert np.isscalar(value) or np.asarray(value).ndim == 0, key


@pytest.mark.parametrize("make", [pf_trajectory, ekf_trajectory])
def test_dual_radio_metrics_are_all_scalars(make):
    traj = make(np.linspace(-1.0, 1.0, 32), np.full(32, 0.04))
    out = dual_radio_mse_theta_metrics(traj, torch.zeros(32))
    assert CALIB_KEYS <= set(out)
    for key, value in out.items():
        assert np.isscalar(value) or np.asarray(value).ndim == 0, key


def test_mse_is_unchanged_by_adding_calibration():
    """The calibration block is purely additive -- stage-2 numbers must still be
    reproducible from a re-run, or the committed report becomes incomparable."""
    thetas = np.linspace(-1.0, 1.0, 32)
    traj = pf_trajectory(thetas, np.full(32, 0.04))
    truth = torch.zeros(32)
    expected = float(((torch.as_tensor(thetas) - truth) ** 2).mean())
    assert dual_radio_mse_theta_metrics(traj, truth)["mse_craft_theta"] == pytest.approx(
        expected
    )


def test_both_families_emit_the_same_calibration_keys():
    """Different key sets per family would make the report's columns depend on
    which families a sweep happened to include."""
    traj = pf_trajectory(np.linspace(-1, 1, 16), np.full(16, 0.04))
    single = set(single_radio_mse_theta_metrics(traj, torch.zeros(16))) - {
        "mse_single_radio_theta"
    }
    dual = set(dual_radio_mse_theta_metrics(traj, torch.zeros(16))) - {"mse_craft_theta"}
    assert single == dual == CALIB_KEYS


def test_empty_trajectory_reports_no_metrics_for_either_family():
    """Previously the dual-radio path had no guard and np.hstack([]) killed the
    run; the single-radio sibling returned {}."""
    assert single_radio_mse_theta_metrics([], torch.zeros(4)) == {}
    assert dual_radio_mse_theta_metrics([], torch.zeros(4)) == {}
