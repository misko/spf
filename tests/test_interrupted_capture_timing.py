import pytest

from spf.scripts.interrupted_capture_timing import (
    interruption_progress_timeout_seconds,
)


def test_early_interruptions_keep_startup_timeout_floor():
    assert interruption_progress_timeout_seconds(1) == 90.0
    assert interruption_progress_timeout_seconds(32) == 90.0


def test_late_interruption_deadline_scales_with_requested_records():
    assert interruption_progress_timeout_seconds(192) == 222.0
    assert interruption_progress_timeout_seconds(256) == 286.0


def test_interruption_deadline_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="positive"):
        interruption_progress_timeout_seconds(0)
