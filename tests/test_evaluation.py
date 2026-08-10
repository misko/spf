"""Unit tests for spf.evaluation. Pure functions, no fixtures, fast."""

import numpy as np
import pytest

from spf.evaluation import aggregate as agg
from spf.evaluation import calibration, frames, metrics, posterior

D = np.pi / 180.0


# --------------------------------------------------------------- metrics


def test_angular_error_takes_the_short_way_round_the_seam():
    """+179 and -179 degrees are 2 degrees apart, not 358."""
    err = metrics.angular_error([179 * D], [-179 * D])
    assert np.isclose(abs(err[0]), 2 * D)


def test_angular_error_is_signed():
    assert metrics.angular_error([10 * D], [0.0])[0] > 0
    assert metrics.angular_error([-10 * D], [0.0])[0] < 0


def test_angular_error_truncates_to_the_shorter_input():
    # filters may stop early via steps=; score the overlap, do not raise
    assert metrics.angular_error(np.zeros(5), np.zeros(9)).shape == (5,)
    assert metrics.angular_error(np.zeros(9), np.zeros(5)).shape == (5,)


def test_mse_matches_the_filters_existing_metric():
    """Must reproduce dual_radio_mse_theta_metrics so old numbers stay comparable."""
    import torch

    from spf.filters.filters import dual_radio_mse_theta_metrics

    rng = np.random.default_rng(0)
    pred = rng.uniform(-np.pi, np.pi, 200)
    truth = rng.uniform(-np.pi, np.pi, 200)
    # P_theta is required: the filters' metrics now also score calibration, and
    # a trajectory carrying no variance is rejected rather than scored as NaN.
    # Every real theta filter sets it; see tests/test_filter_calibration_metrics.py.
    theirs = dual_radio_mse_theta_metrics(
        [{"craft_theta": np.float64(p), "P_theta": np.float64(1.0)} for p in pred],
        torch.tensor(truth),
    )["mse_craft_theta"]
    assert np.isclose(metrics.mse(pred, truth), theirs)


def test_circular_mean_does_not_collapse_at_the_seam():
    """Arithmetic mean of {+179, -179} is 0, which points the wrong way."""
    m = metrics.circular_mean([179 * D, -179 * D])
    assert np.isclose(abs(m), np.pi, atol=1e-6)


def test_circular_mean_matches_arithmetic_mean_away_from_the_seam():
    a = [0.1, 0.2, 0.3]
    assert np.isclose(metrics.circular_mean(a), np.mean(a), atol=1e-9)


def test_circular_std_is_zero_for_identical_angles():
    assert np.isclose(metrics.circular_std([0.7, 0.7, 0.7]), 0.0, atol=1e-9)


def test_circular_std_tracks_linear_std_for_a_tight_cluster():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.05, 20000)
    assert np.isclose(metrics.circular_std(a), 0.05, rtol=0.05)


def test_summarize_block():
    out = metrics.summarize([0.0, 0.0, 0.0], [0.1, -0.1, 0.1])
    assert out["n"] == 3
    assert np.isclose(out["rmse_rad"], 0.1)
    assert np.isclose(out["rmse_deg"], np.degrees(0.1))
    assert np.isclose(out["mae_rad"], 0.1)


# ------------------------------------------------------------- baselines
#
# Reported MSEs are meaningless without these. 2.6 rad^2 reads like a number
# until you know a coin flip scores 3.29.


def test_uniform_random_floor_is_what_random_guessing_actually_scores():
    """The analytic constant, checked against simulation rather than asserted."""
    rng = np.random.default_rng(0)
    truth = rng.uniform(-np.pi, np.pi, 200000)
    guess = rng.uniform(-np.pi, np.pi, 200000)
    assert np.isclose(metrics.mse(guess, truth), metrics.UNIFORM_RANDOM_MSE, rtol=0.02)


def test_best_constant_is_the_circular_mean_and_beats_every_other_constant():
    rng = np.random.default_rng(3)
    truth = metrics.wrap_to_pi(rng.normal(2.9, 0.7, 4000))  # straddles the seam
    best = metrics.baselines(truth)["best_constant"]
    assert np.isclose(best["bearing_rad"], metrics.circular_mean(truth), atol=1e-9)
    for other in np.linspace(-np.pi, np.pi, 73):
        assert best["mse"] <= metrics.mse(np.full_like(truth, other), truth) + 1e-9


def test_a_concentrated_truth_makes_the_constant_floor_far_below_random():
    """Why both floors are reported: on a folded frame the constant floor is the
    binding one, and a filter can beat uniform-random by a lot while still
    losing to a fixed bearing -- i.e. having learned nothing about time."""
    truth = np.full(500, 0.4) + np.random.default_rng(5).normal(0, 0.3, 500)
    base = metrics.baselines(truth)
    assert base["best_constant"]["mse"] < 0.2 * base["uniform_random"]["mse"]


def test_uniform_truth_makes_the_two_floors_agree():
    """The other extreme: no constant helps, so the floors collapse together."""
    truth = np.random.default_rng(7).uniform(-np.pi, np.pi, 100000)
    base = metrics.baselines(truth)
    assert np.isclose(
        base["best_constant"]["mse"], base["uniform_random"]["mse"], rtol=0.03
    )


def test_skill_vs_random_is_zero_at_the_floor_and_one_at_perfect():
    assert np.isclose(metrics.skill_vs_random(metrics.UNIFORM_RANDOM_MSE), 0.0)
    assert np.isclose(metrics.skill_vs_random(0.0), 1.0)
    assert metrics.skill_vs_random(2 * metrics.UNIFORM_RANDOM_MSE) < 0


# ----------------------------------------------------------- calibration


def test_coverage_recovers_the_nominal_bands_on_gaussian_errors():
    rng = np.random.default_rng(2)
    sigma = 0.05
    truth = np.zeros(200000)
    pred = rng.normal(0.0, sigma, truth.shape)
    rows = calibration.coverage(pred, truth, np.full(truth.shape, sigma))
    for row in rows:
        assert np.isclose(row["measured"], row["nominal"], atol=0.01), row


def test_coverage_detects_overconfidence():
    """Errors 3x larger than the claimed sigma must show far under nominal."""
    rng = np.random.default_rng(3)
    truth = np.zeros(50000)
    pred = rng.normal(0.0, 0.15, truth.shape)
    rows = calibration.coverage(pred, truth, np.full(truth.shape, 0.05))
    one_sigma = [r for r in rows if r["k"] == 1][0]
    assert one_sigma["measured"] < 0.3


def test_calibration_ratio_recovers_the_scale_factor():
    rng = np.random.default_rng(4)
    truth = np.zeros(100000)
    pred = rng.normal(0.0, 0.15, truth.shape)
    ratio = calibration.calibration_ratio(pred, truth, np.full(truth.shape, 0.05))
    assert np.isclose(ratio, 3.0, rtol=0.05)


def test_nonpositive_sigma_is_dropped_not_treated_as_zero():
    pred = np.array([0.0, 0.0, 0.0, 0.0])
    truth = np.array([0.1, 0.1, 0.1, 0.1])
    sigma = np.array([0.1, 0.0, np.nan, 0.1])
    assert calibration.z_scores(pred, truth, sigma).shape == (2,)


def test_reliability_curve_is_diagonal_when_calibrated():
    rng = np.random.default_rng(5)
    truth = np.zeros(100000)
    pred = rng.normal(0.0, 0.05, truth.shape)
    curve = calibration.reliability_curve(pred, truth, np.full(truth.shape, 0.05))
    for point in curve:
        assert np.isclose(point["measured"], point["nominal"], atol=0.02), point


# ------------------------------------------------------------- posterior


@pytest.mark.parametrize("ntheta", [7, 65])
def test_bin_convention_matches_the_filters(ntheta):
    """spf.evaluation must bin theta exactly as the NN filters index a posterior."""
    import torch

    from spf.filters.filters import theta_phi_to_bins

    thetas = np.linspace(-np.pi, np.pi, 501)[:-1]
    ours = posterior.theta_to_bin(thetas, ntheta)
    theirs = theta_phi_to_bins(torch.tensor(thetas), ntheta).numpy()
    np.testing.assert_array_equal(ours, theirs)


def test_bin_centers_round_trip_to_their_own_bins():
    ntheta = 65
    centers = posterior.bin_centers(ntheta)
    np.testing.assert_array_equal(
        posterior.theta_to_bin(centers, ntheta), np.arange(ntheta)
    )
    assert np.isclose(centers[ntheta // 2], 0.0)


def test_nll_is_lower_for_a_posterior_on_the_truth():
    ntheta = 65
    truth = np.array([0.0])
    sharp = np.zeros((1, ntheta))
    sharp[0, posterior.theta_to_bin(truth, ntheta)[0]] = 1.0
    uniform = np.full((1, ntheta), 1.0 / ntheta)
    assert posterior.nll(sharp, truth) < posterior.nll(uniform, truth)


def test_confidently_wrong_scores_worse_than_diffuse():
    """The property a point-estimate error cannot express."""
    ntheta = 65
    truth = np.array([0.0])
    wrong = np.full((1, ntheta), 1e-6)
    wrong[0, 0] = 1.0
    uniform = np.full((1, ntheta), 1.0 / ntheta)
    assert posterior.nll(wrong, truth) > posterior.nll(uniform, truth)


def test_peak_and_circular_mean_differ_on_a_bimodal_posterior():
    ntheta = 65
    p = np.full((1, ntheta), 1e-9)
    p[0, posterior.theta_to_bin(np.array([0.4]), ntheta)[0]] = 0.6
    p[0, posterior.theta_to_bin(np.array([-2.6]), ntheta)[0]] = 0.4
    assert not np.isclose(
        posterior.peak_theta(p)[0], posterior.posterior_circular_mean(p)[0], atol=0.1
    )


def test_entropy_bounds():
    ntheta = 65
    uniform = np.full((1, ntheta), 1.0 / ntheta)
    assert np.isclose(posterior.entropy(uniform)[0], np.log(ntheta), atol=1e-6)
    sharp = np.zeros((1, ntheta))
    sharp[0, 3] = 1.0
    assert posterior.entropy(sharp)[0] < 1e-6


def _gaussian_posterior(ntheta, sigma):
    centers = posterior.bin_centers(ntheta)
    p = np.exp(-((centers / sigma) ** 2) / 2)
    return centers, p / p.sum()


def _hdr_set_mass(p, mass):
    """Actual mass of the smallest bin set reaching `mass` -- >= mass on a grid."""
    ordered = np.sort(p)[::-1]
    cum = np.cumsum(ordered)
    return float(cum[np.searchsorted(cum, mass)])


@pytest.mark.parametrize("mass", [0.68, 0.95])
def test_hdr_coverage_equals_the_hdr_sets_own_mass(mass):
    """Truths drawn FROM the posterior land in the HDR set exactly as often as
    that set's mass -- which on a discrete grid exceeds the requested mass."""
    ntheta = 65
    rng = np.random.default_rng(6)
    centers, p = _gaussian_posterior(ntheta, 0.3)
    truths = centers[rng.choice(ntheta, size=40000, p=p)]
    posteriors = np.repeat(p[None, :], truths.shape[0], axis=0)

    measured = posterior.hdr_coverage(posteriors, truths, mass)
    assert np.isclose(measured, _hdr_set_mass(p, mass), atol=0.01)
    assert measured >= mass, "a well-specified posterior must never under-cover"


def test_hdr_overshoot_is_bounded_by_the_largest_bin_and_vanishes_when_fine():
    """Overshoot comes from bins being indivisible, so it is bounded by the mass
    of one bin -- and goes to zero as the grid gets fine. It is NOT monotonic in
    ntheta: at ntheta=17 a sigma=0.3 posterior occupies ~1 bin, and where the
    grid happens to fall can put the overshoot below a finer grid's."""
    rng = np.random.default_rng(7)
    for ntheta in (17, 65, 513):
        centers, p = _gaussian_posterior(ntheta, 0.3)
        truths = centers[rng.choice(ntheta, size=40000, p=p)]
        posteriors = np.repeat(p[None, :], truths.shape[0], axis=0)
        overshoot = posterior.hdr_coverage(posteriors, truths, 0.68) - 0.68
        assert 0 <= overshoot <= p.max() + 0.01, (ntheta, overshoot, p.max())
    # fine grid -> negligible overshoot
    centers, p = _gaussian_posterior(513, 0.3)
    truths = centers[rng.choice(513, size=40000, p=p)]
    posteriors = np.repeat(p[None, :], truths.shape[0], axis=0)
    assert posterior.hdr_coverage(posteriors, truths, 0.68) - 0.68 < 0.02


def test_hdr_coverage_detects_an_overconfident_posterior():
    """The diagnostic use: truths spread wider than the posterior claims."""
    ntheta = 65
    rng = np.random.default_rng(8)
    centers, claimed = _gaussian_posterior(ntheta, 0.1)
    _, actual = _gaussian_posterior(ntheta, 0.5)
    truths = centers[rng.choice(ntheta, size=40000, p=actual)]
    posteriors = np.repeat(claimed[None, :], truths.shape[0], axis=0)
    assert posterior.hdr_coverage(posteriors, truths, 0.68) < 0.4


def test_posterior_rejects_malformed_input():
    with pytest.raises(ValueError):
        posterior.nll(np.full((1, 5), -0.1), np.array([0.0]))
    with pytest.raises(ValueError):
        posterior.nll(np.zeros((1, 5)), np.array([0.0]))


# ---------------------------------------------------------------- frames


def test_require_same_frame_accepts_one_frame():
    assert (
        frames.require_same_frame([frames.CRAFT_RELATIVE] * 3) == frames.CRAFT_RELATIVE
    )


def test_require_same_frame_refuses_to_mix():
    with pytest.raises(frames.FrameMismatch):
        frames.require_same_frame([frames.CRAFT_RELATIVE, frames.ABSOLUTE_NORTH])


def test_unknown_frame_rejected():
    with pytest.raises(ValueError):
        frames.check_frame("whatever")


# ------------------------------------------------------------- aggregate


def _result(frame, N, mse_value):
    return {"frame": frame, "N": N, "metrics": {"mse": mse_value}}


def test_absolute_and_craft_relative_never_pool():
    """The concrete trap: same hyperparameters, two frames, must not average."""
    results = [
        _result(frames.CRAFT_RELATIVE, 4096, 1.89),
        _result(frames.ABSOLUTE_NORTH, 4096, 0.33),
    ]
    rows = agg.aggregate(results, ["N"])
    assert len(rows) == 2
    by_frame = {r["frame"]: r["mse_mean"] for r in rows}
    assert np.isclose(by_frame[frames.CRAFT_RELATIVE], 1.89)
    assert np.isclose(by_frame[frames.ABSOLUTE_NORTH], 0.33)


def test_missing_frame_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="frame"):
        agg.group_results([{"N": 1, "metrics": {"mse": 1.0}}], ["N"])


def test_aggregate_averages_within_a_group_and_reports_n():
    results = [_result(frames.CRAFT_RELATIVE, 4096, v) for v in (1.0, 2.0, 3.0)]
    rows = agg.aggregate(results, ["N"])
    assert len(rows) == 1
    assert rows[0]["n_runs"] == 3
    assert np.isclose(rows[0]["mse_mean"], 2.0)
    assert np.isclose(rows[0]["mse_std"], np.std([1.0, 2.0, 3.0]))


def test_metrics_missing_from_some_runs_are_dropped_visibly():
    results = [
        {"frame": frames.CRAFT_RELATIVE, "N": 1, "metrics": {"mse": 1.0, "extra": 5.0}},
        {"frame": frames.CRAFT_RELATIVE, "N": 1, "metrics": {"mse": 3.0}},
    ]
    row = agg.aggregate(results, ["N"])[0]
    assert np.isclose(row["mse_mean"], 2.0)
    assert "extra_mean" not in row
    assert row["dropped_metrics"] == ["extra"]


def test_rank_is_per_frame():
    results = [
        _result(frames.CRAFT_RELATIVE, 128, 2.0),
        _result(frames.CRAFT_RELATIVE, 4096, 1.0),
        _result(frames.ABSOLUTE_NORTH, 128, 0.5),
    ]
    ranked = agg.rank(agg.aggregate(results, ["N"]))
    assert set(ranked) == {frames.CRAFT_RELATIVE, frames.ABSOLUTE_NORTH}
    assert [r["N"] for r in ranked[frames.CRAFT_RELATIVE]] == [4096, 128]
