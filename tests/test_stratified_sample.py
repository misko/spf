"""Sampling must be balanced, reproducible, and honest about what it skipped."""

import json
import os
import tempfile

import pytest
import yaml

from spf.filters.stratified_sample import (
    DEFAULT_AXES,
    build_manifest,
    describe,
    stratified_sample,
    stratum_key,
)

C = 299792458.0


def _record(vehicle="rover", routine="bounce", d_lambda=0.827, day="2026_08_01", i=0):
    return {
        "prefix": f"/data/{vehicle}_{day}_{routine}_{d_lambda}_{i}.zarr",
        "vehicle": vehicle,
        "routine": routine,
        "d_lambda": d_lambda,
        "day": day,
        "carrier": 5766e6,
        "spacing_m": 0.043,
    }


def _population():
    """Deliberately imbalanced, like the real corpus: 19 / 12 / 6 / 3."""
    recs = []
    for dl, count in ((0.82703, 19), (0.67317, 12), (0.90397, 6), (0.91557, 3)):
        for i in range(count):
            recs.append(_record(d_lambda=dl, i=i))
    return recs


# ------------------------------------------------------------- balance


def test_every_stratum_is_represented_before_any_repeats():
    """The point of the exercise: a 3-store spacing must not be crowded out."""
    recs = _population()
    chosen, strata = stratified_sample(recs, 4, DEFAULT_AXES, seed=0)
    assert len(chosen) == 4
    assert len({r["d_lambda"] for r in chosen}) == 4, "one per stratum expected"


def test_rare_stratum_survives_a_larger_sample():
    recs = _population()
    chosen, _ = stratified_sample(recs, 16, DEFAULT_AXES, seed=0)
    counts = {}
    for r in chosen:
        counts[r["d_lambda"]] = counts.get(r["d_lambda"], 0) + 1
    assert set(counts) == {0.82703, 0.67317, 0.90397, 0.91557}
    # round-robin: no stratum runs more than one ahead of another until exhausted
    assert max(counts.values()) - min(counts.values()) <= 1 or counts[0.91557] == 3


def test_uniform_random_would_have_been_worse():
    """Documents why this exists: proportional sampling can miss a stratum."""
    recs = _population()
    chosen, _ = stratified_sample(recs, 4, DEFAULT_AXES, seed=0)
    assert len({r["d_lambda"] for r in chosen}) == 4
    # a proportional draw of 4 from 19/12/6/3 expects only ~0.3 of the rarest
    assert 3 / len(recs) * 4 < 1.0


def test_sample_never_exceeds_population():
    recs = _population()[:5]
    chosen, _ = stratified_sample(recs, 50, DEFAULT_AXES, seed=0)
    assert len(chosen) == 5
    assert len({r["prefix"] for r in chosen}) == 5, "no duplicates"


def test_no_duplicates_in_a_normal_sample():
    chosen, _ = stratified_sample(_population(), 16, DEFAULT_AXES, seed=0)
    assert len({r["prefix"] for r in chosen}) == len(chosen)


# ------------------------------------------------------- reproducibility


def test_same_seed_same_sample():
    recs = _population()
    a, _ = stratified_sample(recs, 12, DEFAULT_AXES, seed=7)
    b, _ = stratified_sample(recs, 12, DEFAULT_AXES, seed=7)
    assert [r["prefix"] for r in a] == [r["prefix"] for r in b]


def test_different_seed_different_sample():
    recs = _population()
    a, _ = stratified_sample(recs, 12, DEFAULT_AXES, seed=0)
    b, _ = stratified_sample(recs, 12, DEFAULT_AXES, seed=1)
    assert [r["prefix"] for r in a] != [r["prefix"] for r in b]


def test_input_order_does_not_change_the_sample():
    """Filesystem ordering must not leak into a scientific sample."""
    recs = _population()
    a, _ = stratified_sample(recs, 12, DEFAULT_AXES, seed=3)
    b, _ = stratified_sample(list(reversed(recs)), 12, DEFAULT_AXES, seed=3)
    assert sorted(r["prefix"] for r in a) == sorted(r["prefix"] for r in b)


def test_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        stratified_sample(_population(), 0, DEFAULT_AXES, seed=0)


# --------------------------------------------------------------- axes


def test_axes_change_the_strata():
    recs = [
        _record(routine="bounce", d_lambda=0.8),
        _record(routine="circle", d_lambda=0.8),
    ]
    _, by_routine = stratified_sample(recs, 2, ("routine",), seed=0)
    _, by_dl = stratified_sample(recs, 2, ("d_lambda",), seed=0)
    assert len(by_routine) == 2
    assert len(by_dl) == 1


def test_stratum_key_follows_axis_order():
    r = _record()
    assert stratum_key(r, ("routine", "d_lambda")) == (r["routine"], r["d_lambda"])


# ------------------------------------------------------------ describe


def _write_capture(d, name, spacing, carrier, routine):
    prefix = os.path.join(d, name)
    with open(prefix + ".yaml", "w") as f:
        yaml.safe_dump(
            {
                "routine": routine,
                "receivers": [
                    {"antenna-spacing-m": spacing, "f-carrier": carrier},
                    {"antenna-spacing-m": spacing, "f-carrier": carrier},
                ],
            },
            f,
        )
    return prefix + ".zarr"


def test_describe_reads_the_yaml_and_derives_d_lambda():
    with tempfile.TemporaryDirectory() as d:
        p = _write_capture(
            d, "rover_2026_08_01_19_31_21_nRX2_bounce", 0.047, 5766e6, "bounce"
        )
        r = describe(p)
        assert r["vehicle"] == "rover"
        assert r["routine"] == "bounce"
        assert r["day"] == "2026_08_01"
        assert abs(r["d_lambda"] - 0.90397) < 1e-5


def test_describe_prefers_the_recorded_routine_over_the_name():
    """A merged v7 name holds BOTH rovers' routines; only the yaml disambiguates."""
    with tempfile.TemporaryDirectory() as d:
        p = _write_capture(
            d,
            "rover_2026_08_01_19_31_21_nRX2_bounce_tag_RO3.rover_x_circle_tag_RO2",
            0.043,
            5766e6,
            "bounce",
        )
        assert describe(p)["routine"] == "bounce"


def test_carrier_changes_d_lambda_for_the_same_antennas():
    """The reason four table keys were missing: d/lambda depends on carrier too."""
    with tempfile.TemporaryDirectory() as d:
        a = describe(
            _write_capture(d, "rover_2026_08_01_00_00_00_a", 0.047, 5766e6, "bounce")
        )
        b = describe(
            _write_capture(d, "rover_2026_08_01_00_00_00_b", 0.047, 5840e6, "bounce")
        )
        assert a["spacing_m"] == b["spacing_m"]
        assert a["d_lambda"] != b["d_lambda"]


# ---------------------------------------------------------- manifest


def test_manifest_reports_population_and_sample_per_stratum():
    recs = _population()
    chosen, strata = stratified_sample(recs, 8, DEFAULT_AXES, seed=0)
    m = build_manifest(recs, chosen, strata, DEFAULT_AXES, 8, 0)
    assert m["population"]["datasets"] == len(recs)
    assert m["selected"] == 8
    assert sum(r["in_sample"] for r in m["strata"]) == 8
    assert sum(r["in_population"] for r in m["strata"]) == len(recs)
    assert len(m["selected_datasets"]) == 8
    json.dumps(m)  # must be serialisable for the report
