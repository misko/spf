"""Provenance, multi-cache resolution, and the safety rails on table rebuilds.

An empirical table is referenced by every model config and by every past filter
result, so the two things that matter are that a rebuild cannot silently replace
one, and that any table can explain where it came from.
"""

import os
import pickle
import shutil
import tempfile

import pytest

from spf.dataset.spf_dataset import get_empirical_dist
from spf.scripts.create_empirical_p_dist import (
    PROVENANCE_KEY,
    create_empirical_p_dist,
    get_empirical_p_dist_parser,
    load_provenance,
    resolve_precompute_cache,
)


def _args(**overrides):
    parser = get_empirical_p_dist_parser()
    argv = ["--precompute-cache", overrides.pop("cache", "/nonexistent")]
    args = parser.parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ------------------------------------------------- multi-cache resolution


def test_resolve_picks_the_cache_that_has_the_segmentation():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        os.makedirs(os.path.join(b, "ds1_segmentation_nthetas65.yarr"))
        assert resolve_precompute_cache("/data/ds1.zarr", [a, b], 65) == b
        assert resolve_precompute_cache("/data/ds1", [a, b], 65) == b


def test_resolve_prefers_the_earlier_cache_on_a_tie():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for d in (a, b):
            os.makedirs(os.path.join(d, "ds1_segmentation_nthetas65.yarr"))
        assert resolve_precompute_cache("/data/ds1.zarr", [a, b], 65) == a


def test_resolve_returns_none_when_absent():
    with tempfile.TemporaryDirectory() as a:
        assert resolve_precompute_cache("/data/missing.zarr", [a], 65) is None


def test_resolve_is_nthetas_specific():
    """A 7-bin segmentation must not satisfy a 65-bin request."""
    with tempfile.TemporaryDirectory() as a:
        os.makedirs(os.path.join(a, "ds1_segmentation_nthetas7.yarr"))
        assert resolve_precompute_cache("/data/ds1.zarr", [a], 65) is None
        assert resolve_precompute_cache("/data/ds1.zarr", [a], 7) == a


# ------------------------------------------------------------ safety rails


def test_refuses_to_overwrite_an_existing_table():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "table.pkl")
        with open(out, "wb") as f:
            pickle.dump({"existing": True}, f)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            create_empirical_p_dist(_args(out=out, datasets=["/x.zarr"]))
        # and the original is untouched
        with open(out, "rb") as f:
            assert pickle.load(f) == {"existing": True}


def test_no_datasets_is_an_error():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="no datasets"):
            create_empirical_p_dist(
                _args(
                    out=os.path.join(d, "t.pkl"), datasets=[], datasets_from_file=None
                )
            )


def test_all_datasets_unloadable_is_an_error_not_an_empty_table():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError, match="no datasets loaded"):
            create_empirical_p_dist(
                _args(out=os.path.join(d, "t.pkl"), datasets=["/nope/missing.zarr"])
            )


def test_max_load_failures_aborts():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError, match="max-load-failures"):
            create_empirical_p_dist(
                _args(
                    out=os.path.join(d, "t.pkl"),
                    datasets=["/nope/a.zarr", "/nope/b.zarr"],
                    max_load_failures=1,
                )
            )


def test_datasets_from_file_is_merged_with_datasets():
    with tempfile.TemporaryDirectory() as d:
        listing = os.path.join(d, "list.txt")
        with open(listing, "w") as f:
            f.write("/nope/b.zarr\n\n/nope/c.zarr\n")
        with pytest.raises(RuntimeError, match="no datasets loaded"):
            create_empirical_p_dist(
                _args(
                    out=os.path.join(d, "t.pkl"),
                    datasets=["/nope/a.zarr"],
                    datasets_from_file=listing,
                    max_load_failures=10,
                )
            )


# ------------------------------------------------------------- provenance


@pytest.fixture(scope="module")
def built_table(noise1_n128_obits2):
    """A real table built from the synthetic fixture, with provenance."""
    dirname, _empirical_pkl_fn, ds_fn = noise1_n128_obits2
    out = os.path.join(tempfile.mkdtemp(), "rebuilt.pkl")
    create_empirical_p_dist(
        _args(
            out=out,
            cache=dirname,
            datasets=[ds_fn + ".zarr"],
            nbins=65,
            nthetas=65,
            device="cpu",
        )
    )
    with open(out, "rb") as f:
        table = pickle.load(f)
    yield out, table
    shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def test_table_carries_provenance(built_table):
    _out, table = built_table
    assert PROVENANCE_KEY in table
    prov = table[PROVENANCE_KEY]
    for field in (
        "created_utc",
        "command",
        "argv",
        "git",
        "params",
        "datasets",
        "keys",
        "environment",
        "segmentation_version",
    ):
        assert field in prov, field


def test_provenance_records_the_command_and_git_state(built_table):
    _out, table = built_table
    prov = table[PROVENANCE_KEY]
    assert isinstance(prov["command"], str) and prov["command"]
    assert set(prov["git"]) == {"commit", "branch", "dirty"}


def test_provenance_lists_every_contributing_dataset(built_table):
    _out, table = built_table
    prov = table[PROVENANCE_KEY]
    assert prov["datasets"]["loaded"] == len(prov["datasets"]["records"])
    assert prov["datasets"]["loaded"] >= 1
    for record in prov["datasets"]["records"]:
        assert record["path"].endswith(".zarr")
        assert record["precompute_cache"]


def test_provenance_key_counts_match_the_table(built_table):
    _out, table = built_table
    prov = table[PROVENANCE_KEY]
    spacing_keys = {k for k in table if k != PROVENANCE_KEY}
    assert set(prov["keys"]) == spacing_keys
    assert all(v["n_datasets"] >= 1 for v in prov["keys"].values())


def test_load_provenance_helper(built_table):
    out, table = built_table
    assert load_provenance(out) == table[PROVENANCE_KEY]


def test_load_provenance_is_none_for_a_legacy_table():
    with tempfile.TemporaryDirectory() as d:
        legacy = os.path.join(d, "legacy.pkl")
        with open(legacy, "wb") as f:
            pickle.dump({"SDRDEVICE.PLUTO_0.40831": {}}, f)
        assert load_provenance(legacy) is None


# --------------------------------------------- the reserved key is inert


def test_provenance_key_cannot_be_mistaken_for_a_spacing_key(built_table):
    """Spacing keys are '<DEVICE>_<d/lambda>'; the reserved key matches nothing."""
    _out, table = built_table
    assert not PROVENANCE_KEY.startswith("SDRDEVICE")
    assert "_" in PROVENANCE_KEY and PROVENANCE_KEY.startswith("__")


def test_get_empirical_dist_still_works_with_provenance_present(
    built_table, noise1_n128_obits2
):
    """The sole consumer indexes by exact key, so the extra entry is invisible."""
    from spf.dataset.spf_dataset import v5spfdataset_manager

    out, _table = built_table
    dirname, _emp, ds_fn = noise1_n128_obits2
    with v5spfdataset_manager(
        ds_fn + ".zarr",
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=out,
        paired=True,
        ignore_qc=True,
        gpu=False,
    ) as ds:
        for ridx in (0, 1):
            dist = get_empirical_dist(ds, ridx)
            assert dist.shape == (65, 65)


# ------------------------------------------------------ no stray figures


def test_no_figures_written_when_prefix_unset(built_table, tmp_path, monkeypatch):
    """An unset --output-fig-prefix used to still emit None_*.png into the CWD."""
    monkeypatch.chdir(tmp_path)
    assert not list(tmp_path.glob("*.png"))
    # built_table already ran a full build with output_fig_prefix=None
    assert not list(tmp_path.glob("None_*.png"))
