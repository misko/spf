"""Input validation for the v7 TX/RX merge.

On 2026-08-04 a set of captures was staged from the rovers to the NAS by copying
only the `.zarr` directories. v7_tx_rx_merge.py reads each RX capture's `.yaml`
sidecar, so the run scanned and paired every TX against every RX, then died on a
FileNotFoundError naming ONE file -- making the fix iterative: rerun, wait, learn
about the next missing sidecar.

These tests pin the two properties that stops it: every missing input is reported
at once, and the report happens BEFORE any pairing work.
"""

from __future__ import annotations

import pytest

from spf.scripts import v7_tx_rx_merge


# ---------------------------------------------------------------- sidecars ---


def test_config_path_maps_a_finalized_capture_to_its_yaml():
    assert v7_tx_rx_merge.config_path_for("/d/rover_x_tag_RO1.zarr") == (
        "/d/rover_x_tag_RO1.yaml"
    )


def test_config_path_keeps_the_tmp_suffix():
    """An unfinalized run is .zarr.tmp + .yaml.tmp; finalization renames both."""
    assert v7_tx_rx_merge.config_path_for("/d/rover_x_tag_RO1.zarr.tmp") == (
        "/d/rover_x_tag_RO1.yaml.tmp"
    )


def test_config_path_is_not_confused_by_zarr_in_a_directory_name():
    """`str.replace('.zarr', '.yaml')` would mangle the directory too."""
    assert v7_tx_rx_merge.config_path_for("/mnt/aug4.zarr_staging/cap.zarr") == (
        "/mnt/aug4.zarr_staging/cap.yaml"
    )


def test_config_path_refuses_a_non_zarr_path():
    with pytest.raises(ValueError):
        v7_tx_rx_merge.config_path_for("/d/rover_x_tag_RO1.yaml")


# ------------------------------------------------------------ check_inputs ---


def make_capture(directory, name, sidecar=True):
    zarr = directory / f"{name}.zarr"
    zarr.mkdir()
    (zarr / "data.mdb").write_bytes(b"")
    if sidecar:
        (directory / f"{name}.yaml").write_text("receivers:\n- theta-in-pis: 0.5\n")
    return str(zarr)


def test_complete_inputs_have_no_problems(tmp_path):
    tx = make_capture(tmp_path, "cap_tag_RO2")
    rx = make_capture(tmp_path, "cap_tag_RO1")
    assert v7_tx_rx_merge.check_inputs([tx], [rx]) == []


def test_a_tx_without_a_sidecar_is_fine(tmp_path):
    """The merge reads TX GPS from its zarr and nothing else.

    This is the thing nobody knew on 2026-08-04: an unreachable TX rover never
    blocks a merge, because only the RX config is ever loaded.
    """
    tx = make_capture(tmp_path, "cap_tag_RO2", sidecar=False)
    rx = make_capture(tmp_path, "cap_tag_RO1")
    assert v7_tx_rx_merge.check_inputs([tx], [rx]) == []


def test_every_missing_rx_sidecar_is_reported_at_once(tmp_path):
    """Not the first one via a traceback -- all of them, in one report."""
    tx = make_capture(tmp_path, "cap_tag_RO2")
    good = make_capture(tmp_path, "good_tag_RO1")
    bad1 = make_capture(tmp_path, "bad1_tag_RO1", sidecar=False)
    bad3 = make_capture(tmp_path, "bad3_tag_RO3", sidecar=False)

    report = "\n".join(v7_tx_rx_merge.check_inputs([tx], [good, bad1, bad3]))

    assert "bad1_tag_RO1" in report
    assert "bad3_tag_RO3" in report, "only the first missing sidecar was reported"
    assert "good_tag_RO1" not in report
    assert "2 RX capture(s) staged WITHOUT their .yaml sidecar" in report


def test_the_report_says_what_a_staged_capture_is_and_how_to_fix_it(tmp_path):
    tx = make_capture(tmp_path, "cap_tag_RO2")
    rx = make_capture(tmp_path, "bad_tag_RO1", sidecar=False)

    report = "\n".join(v7_tx_rx_merge.check_inputs([tx], [rx]))

    assert ".zarr AND its .yaml" in report
    assert "stage_captures.sh" in report
    assert "TX sidecars are NOT needed" in report


def test_an_unfinalized_capture_wants_a_tmp_sidecar(tmp_path):
    zarr = tmp_path / "cap_tag_RO1.zarr.tmp"
    zarr.mkdir()
    (tmp_path / "cap_tag_RO1.yaml").write_text("receivers: []\n")  # wrong one

    report = "\n".join(v7_tx_rx_merge.check_inputs([], [str(zarr)]))
    assert "cap_tag_RO1.yaml.tmp" in report


def test_missing_zarrs_are_reported_too(tmp_path):
    report = "\n".join(
        v7_tx_rx_merge.check_inputs(
            [str(tmp_path / "no_tx.zarr")], [str(tmp_path / "no_rx.zarr")]
        )
    )
    assert "TX capture is missing" in report
    assert "RX capture is missing" in report


# ------------------------------------------------- fail BEFORE pairing work ---


@pytest.fixture
def spy_merge(monkeypatch):
    """Record every merge_v7rx_v7tx call instead of doing one."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(v7_tx_rx_merge, "merge_v7rx_v7tx", fake)
    return calls


def test_a_missing_sidecar_stops_the_run_before_any_pairing(tmp_path, spy_merge):
    """The whole cost of the 2026-08-04 failure was doing this work first.

    Pairing every TX against every RX takes minutes on a field-sized staging
    directory, and the sidecar read used to happen inside it -- so the operator
    paid for the scan before learning a file was missing.
    """
    tx = make_capture(tmp_path, "cap_tag_RO2")
    good = make_capture(tmp_path, "good_tag_RO1")
    bad = make_capture(tmp_path, "bad_tag_RO1", sidecar=False)
    out = tmp_path / "merged"

    status = v7_tx_rx_merge.main(
        ["--txs", tx, "--rxs", good, bad, "--output", str(out)]
    )

    assert status != 0
    assert spy_merge == [], "pairing ran before the inputs were validated"


def test_complete_inputs_reach_the_merge(tmp_path, spy_merge):
    """The guard must not refuse a good staging directory."""
    tx = make_capture(tmp_path, "cap_tag_RO2")
    rx = make_capture(tmp_path, "cap_tag_RO1")
    out = tmp_path / "merged"

    status = v7_tx_rx_merge.main(["--txs", tx, "--rxs", rx, "--output", str(out)])

    assert status == 0
    assert len(spy_merge) == 1


# -------------------------------------------------------------- --dry-run ---
#
# --dry-run exists to produce the GPS-overlap map cheaply, and it is exactly when
# an operator is asking "is this staging directory good?". So it reports the same
# missing sidecars -- but it still produces the map first, and reports at the end,
# rather than sending the operator away with nothing.


def test_dry_run_reports_missing_sidecars(tmp_path, spy_merge, capsys):
    tx = make_capture(tmp_path, "cap_tag_RO2")
    good = make_capture(tmp_path, "good_tag_RO1")
    bad = make_capture(tmp_path, "bad_tag_RO1", sidecar=False)
    out = tmp_path / "merged"

    status = v7_tx_rx_merge.main(
        ["--txs", tx, "--rxs", good, bad, "--output", str(out), "--dry-run"]
    )

    assert status != 0, "a dry run over incomplete inputs must not look clean"
    assert "bad_tag_RO1" in capsys.readouterr().err


def test_dry_run_still_produces_the_overlap_map(tmp_path, spy_merge):
    """Reporting the problem must not cost the operator the map they asked for."""
    tx = make_capture(tmp_path, "cap_tag_RO2")
    good = make_capture(tmp_path, "good_tag_RO1")
    bad = make_capture(tmp_path, "bad_tag_RO1", sidecar=False)
    out = tmp_path / "merged"

    v7_tx_rx_merge.main(
        ["--txs", tx, "--rxs", good, bad, "--output", str(out), "--dry-run"]
    )

    paired = {kwargs["rx_fn"] for kwargs in spy_merge}
    assert paired == {good, bad}, "the dry run skipped pairs it could have mapped"
