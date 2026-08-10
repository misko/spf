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


# ------------------------------------------------------- config loading


def test_load_configs_accepts_a_bare_mapping():
    from spf.filters.plot_trajectory_comparison import load_configs

    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "c.json")
        with open(fn, "w") as f:
            json.dump({"PF_single_theta_dual_radio|craft_relative": {"params": {}}}, f)
        assert list(load_configs(fn)) == ["PF_single_theta_dual_radio|craft_relative"]


def test_load_configs_unwraps_a_provenance_carrying_file():
    """The committed winners file records WHICH report it came from and that it
    is not the leaderboard's top row. That block must not be mistaken for a
    configuration."""
    from spf.filters.plot_trajectory_comparison import load_configs

    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "c.json")
        with open(fn, "w") as f:
            json.dump({
                "__provenance__": {"source": "some/report/results.json"},
                "configs": {"EKF_single_theta_dual_radio|craft_relative": {"params": {}}},
            }, f)
        assert list(load_configs(fn)) == ["EKF_single_theta_dual_radio|craft_relative"]


def test_load_configs_rejects_keys_missing_the_frame():
    """Without '|<frame>' the two `absolute` variants collapse into one key and
    one silently wins."""
    from spf.filters.plot_trajectory_comparison import load_configs

    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "c.json")
        with open(fn, "w") as f:
            json.dump({"PF_single_theta_dual_radio_NN": {"params": {}}}, f)
        with pytest.raises(ValueError, match="TYPE"):
            load_configs(fn)


def test_the_committed_winners_file_loads_and_covers_every_family():
    from spf.filters.plot_trajectory_comparison import load_configs

    fn = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "e_inf1_filter_sweep",
        "stage3_winners.json",
    )
    configs = load_configs(fn)
    assert len(configs) == 7, sorted(configs)
    types = {k.split("|")[0] for k in configs}
    assert types == set(TYPE_TO_FILTER), types
    # the NN dual-radio family won BOTH frames, with different hyperparameters
    nn = {k: v for k, v in configs.items() if k.startswith("PF_single_theta_dual_radio_NN")}
    assert len(nn) == 2
    assert len({tuple(sorted(v["params"].items())) for v in nn.values()}) == 2


# ------------------------------------------------------------- saving


def test_save_track_writes_exactly_the_requested_path():
    """np.savez_compressed appends '.npz' to any path lacking it. Given the
    temp name '<out>.npz.<pid>.tmp' it wrote '<out>.npz.<pid>.tmp.npz' and the
    rename failed on a source that never existed -- every one of 336 runs failed
    this way. Passing a file handle is what makes it obey."""
    import numpy as np

    from spf.filters.dump_tracks import save_track

    with tempfile.TemporaryDirectory() as d:
        out = track_filename(d, DS, "PF_single_theta_dual_radio", CRAFT_RELATIVE, 0)
        save_track(out, theta=np.arange(4.0), sigma=np.ones(4), gt=np.zeros(4))
        assert os.path.exists(out)
        # nothing else -- no stray .tmp, no doubled .npz.npz
        assert os.listdir(d) == [os.path.basename(out)]


def test_save_track_round_trips_the_arrays():
    import numpy as np

    from spf.filters.dump_tracks import save_track

    with tempfile.TemporaryDirectory() as d:
        out = track_filename(d, DS, "PF_single_theta_dual_radio", CRAFT_RELATIVE, 0)
        save_track(out, theta=np.arange(4, dtype=np.float32),
                   sigma=np.ones(4, dtype=np.float32), frame=CRAFT_RELATIVE)
        with np.load(out) as z:
            np.testing.assert_array_equal(z["theta"], np.arange(4))
            assert str(z["frame"]) == CRAFT_RELATIVE


def test_save_track_leaves_no_temp_behind_when_it_fails(monkeypatch):
    """A killed or failing dump must not strand a partial file that `--resume`
    would later mistake for finished work."""
    import numpy as np

    from spf.filters import dump_tracks

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dump_tracks.np, "savez_compressed", boom)
    with tempfile.TemporaryDirectory() as d:
        out = track_filename(d, DS, "PF_single_theta_dual_radio", CRAFT_RELATIVE, 0)
        with pytest.raises(RuntimeError, match="disk full"):
            dump_tracks.save_track(out, theta=np.zeros(4))
        assert not os.path.exists(out)
        assert os.listdir(d) == []
