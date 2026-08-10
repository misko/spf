"""Labels, the local results store, and the JSON/Markdown report."""

import json
import os
import tempfile

import pytest

from spf.evaluation.frames import ABSOLUTE_NORTH, CRAFT_RELATIVE, RADIO_FOLDED
from spf.filters.labels import ds_fn_to_movement, result_to_frame
from spf.filters.report import build_report, report_to_markdown, write_report
from spf.filters.results_store import LocalResultsStore, make_results_store

# A real merged v7 rover store name: <RX capture>.<TX capture>. The RX rover is
# doing `bounce`, the TX rover `circle`, so the name contains BOTH words.
MERGED_V7 = (
    "/mnt/qnap01/mouse9911/rovers_2026/merged/"
    "rover_2026_08_01_19_31_21_nRX2_bounce_spacing0p043_tag_RO3."
    "rover_2026_08_01_19_54_32_nRX1_circle_spacing0p05075_tag_RO2.zarr"
)
WALL = "/mnt/md2/cache/nosig_data/wallarrayv3_2024_06_03_00_30_00_nRX2_rx_circle.zarr"


# ------------------------------------------------------------------ labels


def test_merged_v7_uses_the_recorded_routine_not_the_filename():
    """The filename has both 'bounce' and 'circle'; only the yaml disambiguates."""
    assert ds_fn_to_movement(MERGED_V7, routine="bounce") == "rover_bounce"


def test_merged_v7_filename_alone_is_ambiguous():
    """Documents why the routine is preferred: substring order decides, wrongly.

    'bounce' is tested before 'circle', so the fallback happens to be right here
    -- but it is right by token ordering, not by knowing which rover is the RX.
    """
    assert ds_fn_to_movement(MERGED_V7) == "rover_bounce"


def test_wall_array_movement():
    assert ds_fn_to_movement(WALL, routine="circle") == "2dwallarray_circle"


def test_unknown_routine_falls_back_to_the_filename():
    assert ds_fn_to_movement(WALL, routine="unknown") == "2dwallarray_circle"


def test_random_circle_is_not_swallowed_by_circle():
    name = "/data/rover_2025_01_01_random_circle_tag_RO1.zarr"
    assert ds_fn_to_movement(name) == "rover_random_circle"


def test_undeterminable_movement_raises():
    with pytest.raises(ValueError):
        ds_fn_to_movement("/data/something_unlabelled.zarr")


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"type": "PF_single_theta_single_radio"}, RADIO_FOLDED),
        ({"type": "EKF_single_theta_single_radio"}, RADIO_FOLDED),
        ({"type": "PF_single_theta_dual_radio"}, CRAFT_RELATIVE),
        ({"type": "EKF_XY_dual_radio"}, CRAFT_RELATIVE),
        ({"type": "PF_single_theta_dual_radio_NN", "absolute": False}, CRAFT_RELATIVE),
        ({"type": "PF_single_theta_dual_radio_NN", "absolute": True}, ABSOLUTE_NORTH),
    ],
)
def test_frame_is_recovered_from_type_and_absolute(result, expected):
    assert result_to_frame(result) == expected


# ------------------------------------------------------------------- store


def _result(ds_fn, _type, mse, **extra):
    r = {
        "type": _type,
        "ds_fn": ds_fn,
        "routine": "bounce",
        "N": 4096,
        "metrics": {"mse_craft_theta": mse, "runtime": 0.5},
    }
    r.update(extra)
    return r


def test_local_store_round_trip():
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        assert store.already_processed() == set()
        key = "fn_run_PF/3.700/ds_x.zarr/_Nx4096_results.pkl"
        store.put(key, [_result(MERGED_V7, "PF_single_theta_dual_radio", 1.5)])
        assert store.already_processed() == {key}
        assert os.path.exists(os.path.join(d, key))


def test_local_store_leaves_no_partial_file_behind():
    """Results are renamed into place, so a killed sweep cannot strand a .tmp."""
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        store.put("a/b/r.pkl", [_result(WALL, "PF_single_theta_dual_radio", 1.0)])
        stragglers = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".tmp")]
        assert stragglers == []


def test_make_results_store_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_results_store("s3", "/tmp/x")


def test_local_backend_needs_no_credentials():
    """The whole point: a local sweep must not touch boto3 or the network."""
    assert isinstance(make_results_store("local", "/tmp/x"), LocalResultsStore)


# ------------------------------------------------------------------ report


def test_report_groups_across_datasets_and_counts_runs():
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        for i, ds in enumerate([MERGED_V7, MERGED_V7.replace("19_31_21", "20_31_21")]):
            store.put(
                f"fn_run_PF/3.700/ds_{i}/_Nx4096_results.pkl",
                [_result(ds, "PF_single_theta_dual_radio", 1.0 + i)],
            )
        report = build_report(d)
        assert report["n_results"] == 2
        # same hyperparameters on two datasets must land in ONE group
        assert report["n_groups"] == 1
        row = report["rows"][0]
        assert row["n_runs"] == 2
        assert row["frame"] == CRAFT_RELATIVE
        assert row["movement"] == "rover_bounce"
        assert abs(row["mse_craft_theta_mean"] - 1.5) < 1e-9


def test_absolute_and_craft_relative_get_separate_rows():
    """The trap: identical hyperparameters, different ground truth."""
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        store.put(
            "fn_run_NN/3.700/ds_a/_absT_results.pkl",
            [_result(MERGED_V7, "PF_single_theta_dual_radio_NN", 0.33, absolute=True)],
        )
        store.put(
            "fn_run_NN/3.700/ds_a/_absF_results.pkl",
            [_result(MERGED_V7, "PF_single_theta_dual_radio_NN", 1.89, absolute=False)],
        )
        report = build_report(d)
        assert report["n_groups"] == 2
        frames = {r["frame"] for r in report["rows"]}
        assert frames == {ABSOLUTE_NORTH, CRAFT_RELATIVE}


def test_write_report_emits_tracked_file_types():
    """CSV is gitignored repo-wide; results must land as json + md to be committable."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        LocalResultsStore(d).put(
            "fn_run_PF/3.700/ds_a/_r.pkl",
            [_result(MERGED_V7, "PF_single_theta_dual_radio", 1.0)],
        )
        json_fn, md_fn = write_report(d, out)
        assert json_fn.endswith("results.json") and os.path.exists(json_fn)
        assert md_fn.endswith("LEADERBOARD.md") and os.path.exists(md_fn)
        with open(json_fn) as f:
            assert json.load(f)["n_results"] == 1
        with open(md_fn) as f:
            body = f.read()
        assert "frame: `craft_relative`" in body


def test_rerunning_never_clobbers_a_hand_written_report():
    """The generated table and the human narrative share a directory. If this
    script wrote REPORT.md, regenerating figures after an edit would silently
    destroy the written-up findings."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        LocalResultsStore(d).put(
            "fn_run_PF/3.700/ds_a/_r.pkl",
            [_result(MERGED_V7, "PF_single_theta_dual_radio", 1.0)],
        )
        narrative = os.path.join(out, "REPORT.md")
        with open(narrative, "w") as f:
            f.write("# findings a human wrote")
        write_report(d, out)
        write_report(d, out)
        with open(narrative) as f:
            assert f.read() == "# findings a human wrote"


def test_markdown_gives_each_frame_its_own_table():
    rows = [
        {"frame": CRAFT_RELATIVE, "N": 128, "mse_mean": 2.0, "n_runs": 1},
        {"frame": ABSOLUTE_NORTH, "N": 128, "mse_mean": 0.5, "n_runs": 1},
    ]
    md = report_to_markdown(
        {"workdir": "w", "n_results": 2, "n_groups": 2, "rows": rows}
    )
    assert md.count("## frame: `") == 2


def test_empty_workdir_is_an_error_not_an_empty_report():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="no results"):
            build_report(d)


def test_hyperparameters_survive_when_families_are_mixed():
    """The whole-sweep intersection bug: EKF carries phi_std, PF carries N and
    seed, so intersecting across families leaves NO hyperparameter and every
    configuration collapses into one row. A real sweep produced 42 rows from
    26,112 results that way, with n_runs up to 1500."""
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        for i, N in enumerate([512, 4096]):
            for seed in (0, 1):
                r = _result(MERGED_V7, "PF_single_theta_dual_radio", 1.0 + i)
                r.update({"N": N, "theta_err": 0.01, "seed": seed})
                store.put(f"pf/{N}_{seed}.pkl", [r])
        for j, phi in enumerate([5.0, 10.0]):
            r = _result(MERGED_V7, "EKF_single_theta_dual_radio", 2.0 + j)
            r.pop("N")
            r.update({"phi_std": phi, "p": 5.0, "dynamic_R": 0.0})
            store.put(f"ekf/{phi}.pkl", [r])

        report = build_report(d)
        assert report["n_results"] == 6
        # 4 PF rows (2 N x 2 seeds) + 2 EKF rows -- NOT collapsed
        assert report["n_groups"] == 6, report["rows"]
        pf_rows = [r for r in report["rows"] if r["type"].startswith("PF")]
        assert {r["N"] for r in pf_rows} == {512, 4096}
        assert {r["seed"] for r in pf_rows} == {0, 1}
        ekf_rows = [r for r in report["rows"] if r["type"].startswith("EKF")]
        assert {r["phi_std"] for r in ekf_rows} == {5.0, 10.0}
        assert all(r["n_runs"] == 1 for r in report["rows"])


def test_families_are_counted_in_the_report():
    with tempfile.TemporaryDirectory() as d:
        store = LocalResultsStore(d)
        store.put("a.pkl", [_result(MERGED_V7, "PF_single_theta_dual_radio", 1.0)])
        store.put("b.pkl", [_result(MERGED_V7, "EKF_single_theta_dual_radio", 2.0)])
        report = build_report(d)
        assert report["families"] == {
            "PF_single_theta_dual_radio": 1,
            "EKF_single_theta_dual_radio": 1,
        }
