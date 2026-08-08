"""Systematic resampling for the particle filters.

This replaces ``filterpy.monte_carlo.systematic_resample`` for two reasons, both
of which matter for the hyperparameter sweeps:

**Reproducibility.** filterpy draws its offset from ``numpy.random.random`` --
the process-global, unseeded RNG. The filters seed a ``torch.Generator`` for
particle initialisation and process noise, but that generator does not cover
the resampling draw, so two runs of an identical configuration on an identical
dataset return different answers. Measured on one 539-timestep rover capture,
eight repeats of the same configuration spread by 42% of the mean MSE for the
empirical dual-radio PF and by 106% for the NN dual-radio PF. Adjacent points in
the shipped sweep grid differ by far less than that, so "the best setting" was
substantially a property of the RNG. Taking the generator as a required argument
makes the omission impossible to repeat.

**Speed.** filterpy walks the cumulative sum in a Python ``while`` loop, i.e.
O(N) interpreter steps per resample. Profiling put that single function at
48-53% of particle-filter runtime. ``np.searchsorted`` performs the identical
walk in C, which measured 1.9-2.2x end-to-end on the PF families (more at larger
N: 1.18x at N=512 rising to 2.31x at N=32768).

The algorithm is unchanged: partition [0, 1) into N equal subdivisions and take
one shared random offset, so the resampled indices are deterministic given that
offset. Statistical equivalence to filterpy was checked over 20 independent runs
of three filter families (Welch p = 0.30-0.64, KS p = 0.34-0.83).
"""

import numpy as np


def systematic_resample(weights, rng):
    """Systematic resampling: draw N indices proportional to ``weights``.

    Args:
        weights: 1-D array of non-negative weights summing to 1. The particle
            filters normalise before every call.
        rng: a random generator exposing ``.random()`` -> float in [0, 1).
            Required, not defaulted: an implicit global generator is the bug
            this module exists to prevent.

    Returns:
        ``np.ndarray`` of dtype ``i`` and length ``len(weights)``, each entry an
        index into ``weights``. Index ``i`` appears either ``floor(N*w[i])`` or
        ``ceil(N*w[i])`` times -- the defining property of systematic resampling.
    """
    n = len(weights)
    # One shared offset for all N subdivisions: this is what makes the scheme
    # "systematic" rather than multinomial, and it is the only randomness used.
    positions = (rng.random() + np.arange(n)) / n

    # float64 explicitly: an integer weights array would make the sentinel below
    # raise a confusing OverflowError instead of just working.
    cumulative_sum = np.cumsum(weights, dtype=np.float64)
    # The last bin absorbs the tail. The caller normalises, but the accumulated
    # sum can land a few ulp below 1.0, and for an offset within an ulp of 1.0
    # the final position rounds to exactly 1.0 -- in which case a bounded top
    # bin makes the search return n and index out of bounds. (filterpy's loop
    # raises IndexError on the same input; returning a wrong index silently
    # would be worse.) An unbounded top bin cannot: every position is finite.
    # For any properly normalised weight vector this changes nothing, since
    # every position is already below 1.0.
    cumulative_sum[-1] = np.inf

    # side="right" gives the first j with cumulative_sum[j] > positions[i],
    # matching filterpy's `if positions[i] < cumulative_sum[j]` exactly.
    return np.searchsorted(cumulative_sum, positions, side="right").astype("i")
