"""Capture-time accounting for frames a radio declined to send.

The sequence numbers already make a gap detectable, but nothing announced it:
a rover could shed frames for hours and report success throughout.  These
tests cover the announcement, not the detection.
"""

from __future__ import annotations

import types

from spf.data_collector import DataCollector


class _Tracker:
    """Just the state ``_track_dropped_frames`` touches."""

    _track_dropped_frames = DataCollector._track_dropped_frames

    def __init__(self, receivers: int = 2) -> None:
        self._last_buffer_sequence = [None] * receivers
        self.dropped_frames_by_receiver = [0] * receivers


def _record(buffer_sequence):
    return types.SimpleNamespace(buffer_sequence=buffer_sequence)


def test_contiguous_capture_reports_zero_dropped():
    tracker = _Tracker()
    for sequence in range(5):
        assert tracker._track_dropped_frames(0, _record(sequence)) == 0
    assert tracker.dropped_frames_by_receiver == [0, 0]


def test_holes_are_counted_as_they_appear():
    tracker = _Tracker()
    tracker._track_dropped_frames(0, _record(0))
    # Sequences 1 and 2 never arrived.
    assert tracker._track_dropped_frames(0, _record(3)) == 2
    assert tracker._track_dropped_frames(0, _record(4)) == 0
    assert tracker._track_dropped_frames(0, _record(9)) == 4
    assert tracker.dropped_frames_by_receiver[0] == 6


def test_receivers_are_counted_independently():
    tracker = _Tracker()
    tracker._track_dropped_frames(0, _record(0))
    tracker._track_dropped_frames(1, _record(0))
    tracker._track_dropped_frames(0, _record(2))
    assert tracker.dropped_frames_by_receiver == [1, 0]


def test_first_record_of_a_stream_is_never_a_gap():
    tracker = _Tracker()
    # A capture may legitimately begin at a non-zero sequence; there is no
    # predecessor to compare against, so nothing is missing yet.
    assert tracker._track_dropped_frames(0, _record(7)) == 0
    assert tracker.dropped_frames_by_receiver[0] == 0


def test_a_snapshot_without_a_sequence_is_ignored():
    tracker = _Tracker()
    assert tracker._track_dropped_frames(0, types.SimpleNamespace()) == 0
    assert tracker.dropped_frames_by_receiver[0] == 0
