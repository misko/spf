"""Score a distribution over theta, with no filter in the loop.

This is why ``spf.evaluation`` is its own package rather than living under
``spf.filters``. A tracker's error confounds two things: how good the per-frame
network posterior is, and how well the filter fuses posteriors over time. To
attribute a result you need to score the posterior directly -- the same way you
would score the empirical ``P(theta | phi)`` table, or any future model.

Bin convention, verified against both consumers: bin ``i`` of ``n`` is centred at
``-pi + (i + 0.5) * 2*pi/n``. That matches ``theta_phi_to_bins`` in
``spf.filters.filters`` (which is how the NN filters index a posterior) and the
analytic target grid in ``target_from_scatter`` (which is what training fits).
For n=65 both place theta=0 at bin 32.
"""

import numpy as np

from spf.evaluation.metrics import angular_error, wrap_to_pi

_EPS = 1e-12


def bin_centers(ntheta):
    """Centre angle of each of ``ntheta`` bins spanning [-pi, pi)."""
    return -np.pi + (np.arange(ntheta) + 0.5) * (2 * np.pi / ntheta)


def theta_to_bin(theta, ntheta):
    """Bin index for an angle, matching spf.filters.filters.theta_phi_to_bins."""
    t = np.asarray(theta, dtype=np.float64)
    return (np.floor(ntheta * (t + np.pi) / (2 * np.pi)).astype(np.int64)) % ntheta


def _normalized(posteriors):
    """(T, ntheta) float array, each row L1-normalised and non-negative."""
    p = np.asarray(posteriors, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2:
        raise ValueError(f"expected (T, ntheta), got shape {p.shape}")
    if (p < 0).any():
        raise ValueError("posterior has negative mass")
    totals = p.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("posterior row sums to zero")
    return p / totals


def nll(posteriors, truths):
    """Mean negative log-likelihood of the truth under the posterior, nats.

    The proper scoring rule: rewards putting mass on the right bin AND being
    honest about uncertainty. A confidently-wrong posterior scores far worse
    than a diffuse one, which is exactly the distinction a point-estimate error
    cannot make.
    """
    p = _normalized(posteriors)
    t = np.asarray(truths, dtype=np.float64).reshape(-1)[: p.shape[0]]
    p = p[: t.shape[0]]
    idx = theta_to_bin(t, p.shape[1])
    return float(-np.log(p[np.arange(p.shape[0]), idx] + _EPS).mean())


def peak_theta(posteriors):
    """Angle of the highest-mass bin, per timestep (the MAP estimate)."""
    p = _normalized(posteriors)
    return bin_centers(p.shape[1])[p.argmax(axis=1)]


def posterior_circular_mean(posteriors):
    """Circular mean of each posterior, per timestep.

    Differs from ``peak_theta`` on a multimodal posterior -- and a 2-element
    array's posterior is routinely bimodal, so the gap between these two is
    itself a useful diagnostic.
    """
    p = _normalized(posteriors)
    c = bin_centers(p.shape[1])
    return np.arctan2((p * np.sin(c)).sum(axis=1), (p * np.cos(c)).sum(axis=1))


def entropy(posteriors):
    """Shannon entropy per timestep, nats. log(ntheta) is a uniform posterior."""
    p = _normalized(posteriors)
    return -(p * np.log(p + _EPS)).sum(axis=1)


def peak_error(posteriors, truths):
    """Signed error of the MAP estimate, wrapped."""
    return angular_error(peak_theta(posteriors), truths)


def hdr_coverage(posteriors, truths, mass=0.68):
    """Fraction of truths inside the highest-density region holding ``mass``.

    The posterior analogue of +-1 sigma coverage, and the honest one for a
    multimodal distribution: take bins in descending probability until the
    cumulative mass reaches ``mass``, then ask whether the truth's bin is in that
    set. Well below ``mass`` means overconfident.

    Note this is **conservative on a coarse grid**. Bins are indivisible, so the
    smallest set reaching ``mass`` generally carries more than ``mass`` -- at
    ntheta=65 a sigma=0.3 rad posterior yields ~0.75 for mass=0.68. So compare a
    measured value against the HDR set's own mass, or against the same figure
    from another model on the same grid; do not read the gap to ``mass`` itself
    as miscalibration.
    """
    p = _normalized(posteriors)
    t = np.asarray(truths, dtype=np.float64).reshape(-1)[: p.shape[0]]
    p = p[: t.shape[0]]
    order = np.argsort(-p, axis=1)
    ordered = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(ordered, axis=1)
    # include the bin that carries the cumulative sum past `mass`
    inside = cum - ordered < mass
    truth_bin = theta_to_bin(t, p.shape[1])
    rank = (order == truth_bin[:, None]).argmax(axis=1)
    hit = inside[np.arange(p.shape[0]), rank]
    return float(hit.mean())


def summarize(posteriors, truths):
    """The standard posterior-quality block, independent of any filter."""
    p = _normalized(posteriors)
    t = np.asarray(truths, dtype=np.float64).reshape(-1)[: p.shape[0]]
    err = peak_error(p, t)
    mean_err = angular_error(posterior_circular_mean(p), t)
    return {
        "n": int(err.shape[0]),
        "nll": nll(p, t),
        "peak_rmse_rad": float(np.sqrt((err**2).mean())),
        "peak_rmse_deg": float(np.degrees(np.sqrt((err**2).mean()))),
        "circmean_rmse_rad": float(np.sqrt((mean_err**2).mean())),
        "mean_entropy": float(entropy(p).mean()),
        "uniform_entropy": float(np.log(p.shape[1])),
        "hdr68_coverage": hdr_coverage(p, t, 0.68),
        "hdr95_coverage": hdr_coverage(p, t, 0.95),
    }
