"""Frame-aware scoring for SPF predictions.

Filter-agnostic on purpose: the same functions score an EKF track, a particle
filter track, a network posterior before any filtering, or the empirical
``P(theta | phi)`` baseline. That is what makes "how much of the error is the
model and how much is the tracker?" an answerable question.

    from spf.evaluation import summarize, coverage, CRAFT_RELATIVE

    summarize(pred_theta, ground_truth)          # mse / rmse / mae block
    coverage(pred_theta, ground_truth, sigma)    # is the reported sigma real?
    posterior.summarize(model_output, truth)     # score P(theta) with no filter
"""

from spf.evaluation import aggregate, calibration, frames, metrics, posterior
from spf.evaluation.aggregate import aggregate as aggregate_results
from spf.evaluation.aggregate import group_results, rank, summarize_group
from spf.evaluation.calibration import (
    NOMINAL_COVERAGE,
    calibration_ratio,
    coverage,
    reliability_curve,
    z_scores,
)
from spf.evaluation.frames import (
    ABSOLUTE_NORTH,
    CRAFT_RELATIVE,
    FRAMES,
    RADIO_FOLDED,
    FrameMismatch,
    check_frame,
    require_same_frame,
)
from spf.evaluation.metrics import (
    angular_error,
    circular_mean,
    circular_std,
    mae,
    median_absolute_error,
    mse,
    rmse,
    rmse_deg,
    summarize,
    wrap_to_pi,
)

__all__ = [
    "aggregate",
    "calibration",
    "frames",
    "metrics",
    "posterior",
    "aggregate_results",
    "group_results",
    "rank",
    "summarize_group",
    "NOMINAL_COVERAGE",
    "calibration_ratio",
    "coverage",
    "reliability_curve",
    "z_scores",
    "ABSOLUTE_NORTH",
    "CRAFT_RELATIVE",
    "FRAMES",
    "RADIO_FOLDED",
    "FrameMismatch",
    "check_frame",
    "require_same_frame",
    "angular_error",
    "circular_mean",
    "circular_std",
    "mae",
    "median_absolute_error",
    "mse",
    "rmse",
    "rmse_deg",
    "summarize",
    "wrap_to_pi",
]
