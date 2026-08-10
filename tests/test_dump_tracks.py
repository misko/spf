"""Naming and skip logic for the track dumper.

The filter-running half is `plot_filter_run.run_filter`, already covered by
tests/test_plot_filter_run.py. What is new here is the bookkeeping, and the two
ways it goes wrong: a filename that collides across configurations (silently
overwriting a track with a different one) and a resume check that does not match
the name actually written (redoing everything, or skipping work never done).
"""

import json
import os
import tempfile

import pytest

from spf.evaluation.frames import ABSOLUTE_NORTH, CRAFT_RELATIVE, RADIO_FOLDED
from spf.filters.dump_tracks import DETERMINISTIC, get_parser, track_filename
from spf.filters.plot_trajectory_comparison import TYPE_TO_FILTER

DS = "rover_2026_08_01_22_57_45_nRX2_bounce_spacing0p043_tag_RO3.rover_x"


def test_every_sweep_family_maps_to_a_runnable_filter():
    """The dumper reads the sweep's result `type`; an unmapped one would raise
    only after the dataset was opened and the batch was already running."""
    from spf.filters.plot_filter_run import FILTERS

    for _type, name in TYPE_TO_FILTER.items():
        assert name in FILTERS, f"{_type} maps to unknown filter {name}"


def test_the_two_absolute_variants_do_not_collide():
    """PF_single_theta_dual_radio_NN won BOTH frames with different
    hyperparameters. They share a type, so the frame has to be in the name --
    otherwise one track silently overwrites the other."""
    a = track_filename("/o", DS, "PF_single_theta_dual_radio_NN", ABSOLUTE_NORTH, 0)
    b = track_filename("/o", DS, "PF_single_theta_dual_radio_NN", CRAFT_RELATIVE, 0)
    assert a != b


def test_seeds_do_not_collide():
    names = {
        track_filename("/o", DS, "PF_single_theta_dual_radio", CRAFT_RELATIVE, s)
        for s in range(5)
    }
    assert len(names) == 5


def test_datasets_do_not_collide():
    a = track_filename("/o", DS, "PF_single_theta_dual_radio", CRAFT_RELATIVE, 0)
    b = track_filename("/o", DS + "_other", "PF_single_theta_dual_radio",
                       CRAFT_RELATIVE, 0)
    assert a != b


def test_filenames_land_in_the_output_dir_and_are_npz():
    fn = track_filename("/out/dir", DS, "EKF_single_theta_dual_radio",
                        CRAFT_RELATIVE, 0)
    assert os.path.dirname(fn) == "/out/dir"
    assert fn.endswith(".npz")


def test_ekf_families_are_marked_deterministic():
    """A seed axis on an EKF writes five byte-identical files and implies a
    spread that does not exist -- the same trap the sweep config avoids."""
    ekf = [t for t in TYPE_TO_FILTER if t.startswith("EKF")]
    assert ekf
    for _type in ekf:
        assert _type.startswith(DETERMINISTIC)
    for _type in (t for t in TYPE_TO_FILTER if t.startswith("PF")):
        assert not _type.startswith(DETERMINISTIC)


def test_resume_is_on_by_default():
    """Batches are run a few datasets at a time and re-invoked; without resume
    every batch would redo all prior work."""
    args = get_parser().parse_args(
        ["--datasets", "a.zarr", "--configs", "c.json",
         "--precompute-cache", "p", "--empirical-pkl-fn", "e.pkl",
         "--output-dir", "o"]
    )
    assert args.resume is True
    assert args.seeds == [0]
    assert args.segmentation_version == 3.7


def test_resume_matches_the_name_that_is_written():
    """The skip check and the write must agree; if they drift, a rerun either
    redoes everything or skips work it never did."""
    with tempfile.TemporaryDirectory() as d:
        fn = track_filename(d, DS, "PF_single_theta_single_radio", RADIO_FOLDED, 3)
        assert not os.path.exists(fn)
        with open(fn, "wb") as f:
            f.write(b"")
        assert os.path.exists(
            track_filename(d, DS, "PF_single_theta_single_radio", RADIO_FOLDED, 3)
        )


def test_config_keys_split_into_type_and_frame():
    """The dumper parses '<TYPE>|<frame>' keys as written by the stage-2
    analysis; a type containing '|' would break the round trip."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "spf", "filters", "configs",
                     "rover2026_stage3_winners.yaml")
    ) as f:
        assert f.read()  # the stage-3 config exists alongside these winners
    for _type in TYPE_TO_FILTER:
        assert "|" not in _type
        key = f"{_type}|{CRAFT_RELATIVE}"
        assert key.split("|") == [_type, CRAFT_RELATIVE]


@pytest.mark.parametrize("bad", ["EKF_XY_dual_radio", "PF_XY_dual_radio"])
def test_xy_families_are_not_dumpable(bad):
    """Their `var` is over tx position, not theta -- a sigma read from them
    would be a confident wrong number rather than an error."""
    assert bad not in TYPE_TO_FILTER
