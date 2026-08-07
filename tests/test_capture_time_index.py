"""A capture's NAME can be wrong; its `gps_timestamp` cannot.

`system_clock_is_plausible` let a rover keep an unsynced clock whenever it
already looked plausible, so captures were named from a stale Pi clock while the
MAVLink-derived time inside the store stayed correct. This index recovers the
true UTC start without touching the raw data.

The trap the tests exist for is the TIMEZONE. Capture names come from
`datetime.fromtimestamp(...)` with no tzinfo -- the rover's LOCAL clock -- and
the fleet runs Europe/London. Comparing a name against a UTC `gps_timestamp`
directly reports a one-hour skew on every healthy August capture, which would
have condemned the entire campaign.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from spf.scripts.capture_time_index import (
    EPOCH_FLOOR,
    FLEET_TIMEZONE,
    _first_gps_epoch,
    _name_epoch,
    _refuse_output_inside_sources,
    find_captures,
    main,
)


LONDON = ZoneInfo("Europe/London")


def _epoch(text: str, zone=timezone.utc) -> float:
    return datetime.strptime(text, "%Y_%m_%d_%H_%M_%S").replace(tzinfo=zone).timestamp()


# ------------------------------------------------------------- timezone ------


def test_a_correctly_named_summer_capture_is_not_reported_as_an_hour_off():
    """RED for the whole tool: Europe/London is UTC+1 in August.

    A name written at 23:31:21 local corresponds to 22:31:21 UTC. Reading the
    name as UTC would make every healthy capture in the campaign look 3600s
    wrong -- the exact failure that would make this index worse than useless.
    """
    name = Path("rover_2026_08_05_23_31_21_nRX2_bounce_spacing0p035_tag_RO1.zarr")
    stamp, name_epoch = _name_epoch(name, LONDON)

    assert stamp == "2026_08_05_23_31_21"
    assert name_epoch == pytest.approx(_epoch("2026_08_05_22_31_21"))


def test_the_fleet_timezone_matches_what_the_rovers_are_actually_set_to():
    """reconcile_rover_boot_units.sh sets this; a mismatch silently offsets
    every skew in the index by an hour."""
    defaults = (
        Path(__file__).resolve().parents[1]
        / "data_collection/rover/rover_v3.1/rover_env_defaults.sh"
    ).read_text()

    assert f"printf '{FLEET_TIMEZONE}'" in defaults


def test_a_winter_capture_uses_gmt_not_a_fixed_offset():
    """A fixed +1 would be wrong for half the year; the zone has to do the work."""
    name = Path("rover_2026_01_15_12_00_00_nRX2_bounce_spacing0p035_tag_RO1.zarr")
    _stamp, name_epoch = _name_epoch(name, LONDON)

    assert name_epoch == pytest.approx(_epoch("2026_01_15_12_00_00"))


def test_a_name_with_no_timestamp_is_reported_not_guessed():
    assert _name_epoch(Path("something_else.zarr"), LONDON) == (None, None)


def test_a_digit_bearing_tag_cannot_be_mistaken_for_the_timestamp():
    name = Path("rover_2026_08_05_23_31_21_nRX2_bounce_spacing0p035_tag_2020_01_02_03_04_05.zarr")
    stamp, _epoch_value = _name_epoch(name, LONDON)

    assert stamp == "2026_08_05_23_31_21"


# ------------------------------------------------- reading the truth out ------


class _FakeReceivers(dict):
    pass


def _store(**receivers):
    return {"receivers": _FakeReceivers({k: {"gps_timestamp": v} for k, v in receivers.items()})}


def test_the_earliest_real_gps_time_across_all_receivers_wins():
    """r0 may never have got a fix while r1 did; the truth is still recoverable."""
    later = _epoch("2026_08_05_22_35_00")
    earlier = _epoch("2026_08_05_22_31_21")
    store = _store(
        r0=np.array([0.0, 0.0, later]),
        r1=np.array([0.0, earlier, later]),
    )

    best, read = _first_gps_epoch(store)

    assert best == pytest.approx(earlier)
    assert read == 2


def test_a_fix_without_utc_is_not_treated_as_a_time():
    """RED: `> 0` accepts a pre-2025 epoch, which is a fix that has no UTC yet.

    Same floor drone_run.sh applies before it will set the system clock.
    """
    real = _epoch("2026_08_05_22_31_21")
    store = _store(r0=np.array([0.0, float(EPOCH_FLOOR - 5000), real]))

    best, _read = _first_gps_epoch(store)

    assert best == pytest.approx(real)


def test_a_capture_that_never_had_utc_reports_no_time_rather_than_a_wrong_one():
    store = _store(r0=np.array([0.0, 0.0, 0.0]))

    best, read = _first_gps_epoch(store)

    assert best is None
    assert read == 1


def test_nan_stamps_do_not_become_a_time():
    store = _store(r0=np.array([np.nan, 0.0]))

    assert _first_gps_epoch(store)[0] is None


# ----------------------------------------------------- raw data is immutable ---


def test_the_index_refuses_to_be_written_inside_the_tree_it_scanned(tmp_path):
    """Writing beside immutable raw data is how a read-only tool becomes the
    thing that modified a dataset."""
    source = tmp_path / "captures"
    source.mkdir()

    with pytest.raises(SystemExit) as failure:
        _refuse_output_inside_sources(source / "index.json", [source])
    assert "immutable" in str(failure.value)

    # Nested is refused too, not just the immediate directory.
    with pytest.raises(SystemExit):
        _refuse_output_inside_sources(source / "a" / "b" / "index.json", [source])


def test_an_output_outside_the_scanned_tree_is_allowed(tmp_path):
    source = tmp_path / "captures"
    source.mkdir()

    _refuse_output_inside_sources(tmp_path / "cache" / "index.json", [source])


def test_stores_are_opened_read_only():
    """LMDB will happily create or upgrade a store opened any other way."""
    import inspect

    from spf.scripts import capture_time_index

    source = inspect.getsource(capture_time_index.inspect_capture)
    assert 'mode="r"' in source


# ------------------------------------------------------------- discovery ------


def test_directories_and_explicit_captures_are_both_accepted(tmp_path):
    root = tmp_path / "aug5"
    (root / "a.zarr").mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "nested" / "b.zarr").mkdir()
    (root / "notes.txt").write_text("x")

    found = find_captures([root])

    assert [path.name for path in found] == ["a.zarr", "b.zarr"]


def test_overlapping_roots_do_not_read_the_same_store_twice(tmp_path):
    root = tmp_path / "aug5"
    (root / "a.zarr").mkdir(parents=True)

    assert len(find_captures([root, root / "a.zarr"])) == 1


# ------------------------------------------------------------- end to end ----


def test_an_unreadable_capture_is_recorded_rather_than_aborting_the_scan(tmp_path):
    """One corrupt store must not cost the index of every other capture."""
    root = tmp_path / "aug5"
    (root / "broken.zarr").mkdir(parents=True)
    output = tmp_path / "cache" / "index.json"

    status = main([str(root), "--output", str(output)])

    payload = json.loads(output.read_text())
    assert payload["counts"] == {"UNREADABLE": 1}
    assert payload["filename_timezone"] == FLEET_TIMEZONE
    assert status == 2, "a scan needing a human must not exit 0"


# ------------------------------------------- bench vs mission captures (S-5) ---
#
# A --fake-drone capture has no vehicle, so it has no GPS and never had a UTC to
# lose. Reporting that as NO_GPS_TIME beside a mission capture that LOST its
# time makes the alarming case indistinguishable from the expected one -- and
# the first campaign scan surfaced three of them, all bench runs.


class _AttrStore(dict):
    def __init__(self, attrs, receivers):
        super().__init__({"receivers": receivers})
        self.attrs = attrs


def _no_gps_store(attrs):
    return _AttrStore(attrs, {"r0": {"gps_timestamp": np.zeros(4)}})


def _inspect_with(monkeypatch, store, name="rover_2026_08_01_00_06_05_x.zarr"):
    from spf.scripts import capture_time_index

    monkeypatch.setattr(
        capture_time_index, "zarr_open_from_lmdb_store", lambda *a, **k: store
    )
    return capture_time_index.inspect_capture(Path(name), LONDON, 300.0)


def test_a_fake_drone_capture_is_not_reported_as_having_lost_its_time(monkeypatch):
    record = _inspect_with(monkeypatch, _no_gps_store({"vehicle_present": False}))

    assert record.verdict == "BENCH_NO_VEHICLE"
    assert record.vehicle_present is False


def test_a_mission_capture_with_no_gps_time_is_still_alarming(monkeypatch):
    record = _inspect_with(monkeypatch, _no_gps_store({"vehicle_present": True}))

    assert record.verdict == "NO_GPS_TIME"


def test_a_capture_written_before_the_attribute_existed_falls_back(monkeypatch):
    """Honest, not silent: an old bench capture still reports NO_GPS_TIME rather
    than being quietly excused."""
    record = _inspect_with(monkeypatch, _no_gps_store({}))

    assert record.verdict == "NO_GPS_TIME"
    assert record.vehicle_present is None


def test_a_scan_of_only_bench_captures_does_not_fail_the_pipeline(tmp_path):
    """BENCH_NO_VEHICLE is an expected outcome, so it must not exit non-zero."""
    from spf.scripts.capture_time_index import main

    import spf.scripts.capture_time_index as module

    root = tmp_path / "bench"
    (root / "a.zarr").mkdir(parents=True)
    output = tmp_path / "cache" / "index.json"

    original = module.zarr_open_from_lmdb_store
    module.zarr_open_from_lmdb_store = lambda *a, **k: _no_gps_store(
        {"vehicle_present": False}
    )
    try:
        status = main([str(root), "--output", str(output)])
    finally:
        module.zarr_open_from_lmdb_store = original

    assert json.loads(output.read_text())["counts"] == {"BENCH_NO_VEHICLE": 1}
    assert status == 0


def test_the_collector_records_whether_there_was_a_vehicle():
    """The attribute has to reach the artifact, or the tool can never use it."""
    import inspect

    from spf.data_collector import DataCollector

    source = inspect.getsource(DataCollector._mark_capture_state)
    assert "vehicle_present" in source

    collection = (
        Path(__file__).resolve().parents[1] / "spf/mavlink_radio_collection.py"
    ).read_text()
    assert "data_collector.vehicle_present = not args.fake_drone" in collection
