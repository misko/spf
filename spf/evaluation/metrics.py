"""Angular error metrics.

Deliberately free of any filter or model: these score a sequence of predicted
angles against ground truth, so the same code scores an EKF track, a particle
filter track, the empirical baseline, or a network's point estimate before any
filtering happens.

Every function here treats angles as points on a circle. Two failure modes that
arithmetic invites and these avoid:

* ``pred - truth`` for +179 deg and -179 deg is 358 deg, not 2 deg.
* The arithmetic mean of {+179 deg, -179 deg} is 0 deg, which is the opposite
  direction from both. Use ``circular_mean``.

Inputs are array-like (numpy arrays, torch CPU tensors, lists). Outputs are
plain floats or numpy arrays.
"""

import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def wrap_to_pi(angles):
    """Map angles to [-pi, pi)."""
    a = np.asarray(angles, dtype=np.float64)
    return (a + np.pi) % (2 * np.pi) - np.pi


def angular_error(pred, truth):
    """Signed shortest-arc error ``pred - truth``, wrapped to [-pi, pi).

    Truncates to the shorter of the two inputs: filters may stop early
    (``steps=``), and comparing a short track against a long ground truth should
    score the overlap rather than raise.
    """
    p, t = _as_array(pred), _as_array(truth)
    n = min(p.shape[0], t.shape[0])
    return wrap_to_pi(p[:n] - t[:n])


def mse(pred, truth):
    """Mean squared angular error, rad^2. Matches the filters' existing metric."""
    return float((angular_error(pred, truth) ** 2).mean())


def rmse(pred, truth):
    """Root mean squared angular error, radians."""
    return float(np.sqrt(mse(pred, truth)))


def rmse_deg(pred, truth):
    return float(np.degrees(rmse(pred, truth)))


def mae(pred, truth):
    """Mean absolute angular error, radians."""
    return float(np.abs(angular_error(pred, truth)).mean())


def median_absolute_error(pred, truth):
    """Median absolute angular error, radians.

    Worth reporting next to the mean on the O4 emitter: it is bursty, so a
    minority of frames carry no signal and inflate the mean without saying much
    about performance when the emitter is actually transmitting.
    """
    return float(np.median(np.abs(angular_error(pred, truth))))


def circular_mean(angles, weights=None):
    """Phasor mean of a set of angles, in [-pi, pi)."""
    a = _as_array(angles)
    if weights is None:
        w = np.ones_like(a)
    else:
        w = _as_array(weights)
        if w.shape != a.shape:
            raise ValueError(f"weights shape {w.shape} != angles shape {a.shape}")
    total = w.sum()
    if total == 0:
        raise ValueError("weights sum to zero")
    return float(
        np.arctan2((w * np.sin(a)).sum() / total, (w * np.cos(a)).sum() / total)
    )


def circular_std(angles, weights=None):
    """Circular standard deviation, radians.

    ``sqrt(-2 ln R)`` where R is the mean resultant length. Unbounded as the
    distribution approaches uniform, and ~equal to the linear std for tight
    clusters, which makes it comparable to a filter's reported sigma.
    """
    a = _as_array(angles)
    w = np.ones_like(a) if weights is None else _as_array(weights)
    total = w.sum()
    if total == 0:
        raise ValueError("weights sum to zero")
    r = np.hypot((w * np.sin(a)).sum() / total, (w * np.cos(a)).sum() / total)
    r = min(float(r), 1.0)
    if r <= 0.0:
        return float("inf")
    return float(np.sqrt(-2.0 * np.log(r)))


def summarize(pred, truth):
    """The standard metric block for one track."""
    err = angular_error(pred, truth)
    return {
        "n": int(err.shape[0]),
        "mse": float((err**2).mean()),
        "rmse_rad": float(np.sqrt((err**2).mean())),
        "rmse_deg": float(np.degrees(np.sqrt((err**2).mean()))),
        "mae_rad": float(np.abs(err).mean()),
        "median_abs_err_rad": float(np.median(np.abs(err))),
    }
