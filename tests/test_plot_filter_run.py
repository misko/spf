"""The unified filter visualiser.

Smoke-level on purpose: assert the figure has the panels it claims and that the
summary block is finite and frame-tagged. Pixel comparison would be brittle and
would not catch anything a human reading the figure wouldn't.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from spf.evaluation.frames import CRAFT_RELATIVE, RADIO_FOLDED
from spf.filters.plot_filter_run import (
    open_dataset,
    plot_filter_run,
    run_filter,
    summarize_run,
)
from spf.utils import SEGMENTATION_VERSION

# empirical families only: the NN ones need a trained checkpoint fixture, which
# costs ~11 minutes to build and exercises no plotting code the others don't.
EMPIRICAL_FILTERS = [
    ("pf_dual", {"N": 512, "theta_err": 0.01, "theta_dot_err": 0.01}, CRAFT_RELATIVE),
    ("pf_single", {"N": 512, "theta_err": 0.01, "theta_dot_err": 0.01}, RADIO_FOLDED),
    ("ekf_dual", {"phi_std": 10.0, "p": 5.0, "noise_std": 0.001}, CRAFT_RELATIVE),
    ("ekf_single", {"phi_std": 10.0, "p": 5.0, "noise_std": 0.001}, RADIO_FOLDED),
]


@pytest.fixture(scope="module")
def ds(noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    # the fixture segments at the default version; track the constant rather
    # than hardcode, so bumping SEGMENTATION_VERSION does not silently break this
    return open_dataset(
        ds_fn, dirname, empirical_pkl_fn, segmentation_version=SEGMENTATION_VERSION
    )


@pytest.mark.parametrize("name,params,expected_frame", EMPIRICAL_FILTERS)
def test_each_family_produces_a_figure_and_a_summary(ds, name, params, expected_frame):
    theta, sigma, gt, extras = run_filter(ds, name, dict(params))

    assert extras["frame"] == expected_frame
    assert theta.shape == sigma.shape == gt.shape
    assert np.isfinite(theta).all()

    fig = plot_filter_run(ds, theta, sigma, gt, extras, f"{name} test")
    # 4 panels without a posterior, 5 with
    assert len(fig.axes) == 4
    matplotlib.pyplot.close(fig)

    summary = summarize_run(theta, sigma, gt)
    assert summary["n"] == theta.shape[0]
    assert np.isfinite(summary["mse"]) and summary["mse"] >= 0
    assert np.isclose(summary["rmse_deg"], np.degrees(summary["rmse_rad"]))
    # coverage is REPORTED, not gated -- only that it is a well-formed fraction
    for row in summary["coverage"]:
        assert 0.0 <= row["measured"] <= 1.0
        assert row["k"] in (1, 2, 3)


def test_unknown_filter_name_is_rejected(ds):
    with pytest.raises(ValueError, match="unknown filter"):
        run_filter(ds, "kalman_deluxe", {})


def test_unused_params_are_rejected_not_silently_ignored(ds):
    """A typo'd hyperparameter must fail loudly, or a sweep silently runs defaults."""
    with pytest.raises(ValueError, match="unused params"):
        run_filter(ds, "pf_dual", {"N": 512, "theta_er": 0.01})


def test_seed_is_honoured_end_to_end(ds):
    a, _, _, _ = run_filter(ds, "pf_dual", {"N": 512, "seed": 0})
    b, _, _, _ = run_filter(ds, "pf_dual", {"N": 512, "seed": 0})
    c, _, _, _ = run_filter(ds, "pf_dual", {"N": 512, "seed": 1})
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
