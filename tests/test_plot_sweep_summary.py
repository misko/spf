"""The statistics behind the sweep summary figures.

These are pure functions over the rows in ``results.json``. They are worth
testing on their own because a figure that computes the wrong statistic still
renders perfectly -- the first version of ``fig_by_spacing`` took a raw ``min``
over every (configuration, seed, dataset) row and produced a plausible-looking
curve that was 3-5x better than anything reproducible.
"""

import numpy as np
import pytest

from spf.evaluation.frames import CRAFT_RELATIVE
from spf.filters.plot_sweep_summary import (
    UNIFORM_RANDOM_MSE,
    best_per_spacing,
    datasets_per_spacing,
    per_config_across_seeds,
)


def _row(spacing, mse, seed, N, n_runs=1, _type="PF_single_theta_dual_radio"):
    return {
        "type": _type,
        "frame": CRAFT_RELATIVE,
        "rx_wavelength_spacing": spacing,
        "mse_craft_theta_mean": mse,
        "n_runs": n_runs,
        "seed": seed,
        "N": N,
    }


def test_uniform_random_floor_is_pi_squared_over_three():
    assert np.isclose(UNIFORM_RANDOM_MSE, np.pi**2 / 3.0)


def test_per_config_averages_datasets_weighted_then_collects_seeds():
    """Two spacings with unequal dataset counts must not be weighted equally."""
    rows = [
        _row(0.6, mse=1.0, seed=0, N=512, n_runs=9),
        _row(0.8, mse=2.0, seed=0, N=512, n_runs=1),
    ]
    (per_seed,) = per_config_across_seeds(rows).values()
    # 9 datasets at 1.0 and 1 at 2.0 -> 1.1, not the unweighted 1.5
    assert np.isclose(per_seed[0], 1.1)


def test_best_per_spacing_averages_seeds_before_minimising():
    """The bug this guards: minimising over raw rows picks the luckiest seed.

    Configuration A is genuinely better on average (0.50 vs 0.60) but B has one
    lucky seed at 0.10. Minimising first reports 0.10 -- a number no rerun will
    reproduce.
    """
    rows = []
    for seed, mse in enumerate([0.50, 0.50, 0.50]):
        rows.append(_row(0.8, mse=mse, seed=seed, N=512))
    for seed, mse in enumerate([0.10, 0.85, 0.85]):
        rows.append(_row(0.8, mse=mse, seed=seed, N=4096))

    best = best_per_spacing(rows)[0.8][("PF_single_theta_dual_radio", CRAFT_RELATIVE)]
    assert np.isclose(best, 0.50)
    assert best > 0.10


def test_best_per_spacing_keeps_families_apart():
    rows = [
        _row(0.8, mse=0.4, seed=0, N=512),
        _row(0.8, mse=2.9, seed=0, N=None, _type="EKF_single_theta_dual_radio"),
    ]
    per = best_per_spacing(rows)[0.8]
    assert np.isclose(per[("PF_single_theta_dual_radio", CRAFT_RELATIVE)], 0.4)
    assert np.isclose(per[("EKF_single_theta_dual_radio", CRAFT_RELATIVE)], 2.9)


def test_spacings_are_not_pooled():
    rows = [_row(0.67317, 0.3, 0, 512), _row(0.67318, 0.9, 0, 512)]
    # 5 decimal places is the resolution the empirical table keys on
    assert sorted(best_per_spacing(rows)) == [0.67317, 0.67318]


def test_datasets_per_spacing_reports_the_evidence_behind_each_point():
    """Half the spacings in the rover sweep rest on one capture; the figure has
    to be able to say so."""
    rows = [
        _row(0.67317, 0.3, seed=0, N=512, n_runs=5),
        _row(0.67317, 0.4, seed=1, N=512, n_runs=5),
        _row(0.90397, 0.8, seed=0, N=512, n_runs=1),
    ]
    assert datasets_per_spacing(rows) == {0.67317: 5, 0.90397: 1}


@pytest.mark.parametrize("missing", ["mse_craft_theta_mean"])
def test_rows_without_an_mse_are_skipped_not_counted_as_zero(missing):
    rows = [_row(0.8, 0.5, 0, 512), _row(0.8, 0.5, 1, 512)]
    del rows[1][missing]
    per = best_per_spacing(rows)[0.8][("PF_single_theta_dual_radio", CRAFT_RELATIVE)]
    assert np.isclose(per, 0.5)
