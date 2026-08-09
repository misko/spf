"""Does the stated confidence mean anything?

A filter that reports a variance nobody has checked is reporting decoration. The
first time these functions were pointed at a real run -- the NN dual-radio PF on
a 539-timestep rover capture -- the +-1 sigma band covered 25.6% of the actual
errors where 68.3% is nominal, and +-3 sigma covered 67% where 99.7% is nominal.
That filter is roughly 2.7x overconfident in sigma, which matters for anything
downstream that gates on the reported variance.

Report coverage before gating on it. A hard bound set before the cause is
understood either enshrines the bug or blocks the work.
"""

import numpy as np

from spf.evaluation.metrics import angular_error

# P(|z| <= k) for a standard normal
NOMINAL_COVERAGE = {1: 0.6826894921, 2: 0.9544997361, 3: 0.9973002039}


def z_scores(pred, truth, sigma):
    """Angular error divided by the filter's reported sigma, per timestep.

    Entries with a non-finite or non-positive sigma are dropped: a filter that
    reports no uncertainty cannot be scored on it, and silently treating that as
    sigma=0 would manufacture infinite z-scores.
    """
    err = angular_error(pred, truth)
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)[: err.shape[0]]
    err = err[: s.shape[0]]
    ok = np.isfinite(err) & np.isfinite(s) & (s > 0)
    return err[ok] / s[ok]


def coverage(pred, truth, sigma, ks=(1, 2, 3)):
    """Measured vs nominal coverage of the +-k sigma bands.

    Returns a list of ``{"k", "measured", "nominal", "n"}``. ``measured`` well
    below ``nominal`` means overconfident; well above means the filter is
    hedging and its variance is not informative either.
    """
    z = z_scores(pred, truth, sigma)
    rows = []
    for k in ks:
        rows.append(
            {
                "k": int(k),
                "measured": float((np.abs(z) <= k).mean()) if z.size else float("nan"),
                "nominal": NOMINAL_COVERAGE.get(int(k), float("nan")),
                "n": int(z.size),
            }
        )
    return rows


def calibration_ratio(pred, truth, sigma):
    """How many times too small the reported sigma is.

    ``std(z)``: 1.0 is calibrated, >1 overconfident (errors larger than sigma
    claims), <1 underconfident. A single number to track across a sweep, where
    the full coverage table is too wide to tabulate.
    """
    z = z_scores(pred, truth, sigma)
    if z.size < 2:
        return float("nan")
    return float(z.std())


def reliability_curve(pred, truth, sigma, quantiles=np.linspace(0.05, 0.95, 19)):
    """Empirical vs nominal coverage across a range of central intervals.

    The curve behind the +-1/2/3 sigma table: for each nominal central mass q,
    what fraction of errors actually land inside the interval a Gaussian with
    the reported sigma would assign that mass. Plot measured against nominal;
    the diagonal is perfect calibration.
    """
    from scipy.stats import norm

    z = z_scores(pred, truth, sigma)
    out = []
    for q in np.asarray(quantiles, dtype=np.float64):
        half_width = norm.ppf(0.5 + q / 2.0)
        out.append(
            {
                "nominal": float(q),
                "measured": (
                    float((np.abs(z) <= half_width).mean()) if z.size else float("nan")
                ),
            }
        )
    return out
