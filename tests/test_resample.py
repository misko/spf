import numpy as np
import pytest

from spf.filters.resample import systematic_resample


class FixedOffsetRng:
    """Stub generator so a test can pin the single random draw."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def reference_systematic_resample(weights, offset):
    """filterpy's loop, verbatim, with the offset injected.

    Kept as the oracle: the vectorized implementation must agree with the
    original algorithm index-for-index whenever both see the same offset.
    """
    n = len(weights)
    positions = (offset + np.arange(n)) / n
    indexes = np.zeros(n, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def random_weights(rng, n):
    w = rng.random(n) + 1e-6
    return w / w.sum()


def test_matches_filterpy_algorithm_exactly():
    """Same offset in, same indices out -- against the original loop."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 512))
        w = random_weights(rng, n)
        offset = float(rng.random())
        got = systematic_resample(w, FixedOffsetRng(offset))
        expected = reference_systematic_resample(w, offset)
        np.testing.assert_array_equal(got, expected)


def test_selection_counts_within_one_of_expectation():
    """The defining property: index i is drawn floor(N*w_i) or ceil(N*w_i) times.

    This holds for every offset, so it is an exact oracle that needs no
    reference implementation and no RNG agreement.
    """
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 512))
        w = random_weights(rng, n)
        idx = systematic_resample(w, rng)
        counts = np.bincount(idx, minlength=n)
        expected = n * w
        assert np.all(counts >= np.floor(expected) - 1e-9)
        assert np.all(counts <= np.ceil(expected) + 1e-9)


def test_all_indices_in_range():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(2, 512))
        idx = systematic_resample(random_weights(rng, n), rng)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n


def test_uniform_weights_are_the_identity():
    """With w = 1/N every subdivision lands in its own bin, for any offset."""
    n = 64
    w = np.full(n, 1.0 / n)
    for offset in (0.0, 0.25, 0.5, 0.999999):
        idx = systematic_resample(w, FixedOffsetRng(offset))
        np.testing.assert_array_equal(idx, np.arange(n))


@pytest.mark.parametrize("winner", [0, 7, 63])
def test_degenerate_weights_collapse_to_the_survivor(winner):
    n = 64
    w = np.full(n, 1e-300)
    w[winner] = 1.0
    w = w / w.sum()
    idx = systematic_resample(w, np.random.default_rng(3))
    np.testing.assert_array_equal(idx, np.full(n, winner))


def test_offset_at_the_top_of_the_range_stays_in_bounds():
    """An offset within an ulp of 1.0 drives the final position to exactly 1.0.

    filterpy raises IndexError here; a bounded top bin would make searchsorted
    return n and silently index out of bounds, which is worse. The last bin must
    absorb it.
    """
    n = 4096
    w = np.full(n, 1.0 / n)
    w = w / w.sum()  # not exactly 1.0 in float64
    for offset in (0.9999999999, 1.0 - 1e-15):
        positions = (offset + np.arange(n)) / n
        idx = systematic_resample(w, FixedOffsetRng(offset))
        assert idx.max() < n, f"out of bounds with max position {positions.max()!r}"
        assert idx.min() >= 0


def test_same_seed_same_indices():
    w = random_weights(np.random.default_rng(4), 256)
    a = systematic_resample(w, np.random.default_rng(11))
    b = systematic_resample(w, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_generally_differ():
    w = random_weights(np.random.default_rng(5), 256)
    a = systematic_resample(w, np.random.default_rng(11))
    b = systematic_resample(w, np.random.default_rng(12))
    assert not np.array_equal(a, b)


def test_generator_is_required():
    """No implicit global RNG -- that omission is the bug this module prevents."""
    with pytest.raises(TypeError):
        systematic_resample(np.full(8, 1 / 8))  # type: ignore[call-arg]
