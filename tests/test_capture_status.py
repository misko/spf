import json

import pytest

from spf.capture_status import CaptureStatusWriter, format_status, mark_failed


class _Clock:
    def __init__(self):
        self.wall = 1000.0
        self.monotonic = 50.0

    def wall_time(self):
        return self.wall

    def monotonic_time(self):
        return self.monotonic

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


def test_capture_status_is_atomic_throttled_and_reports_eta(tmp_path):
    clock = _Clock()
    path = tmp_path / "status.json"
    writer = CaptureStatusWriter(
        path,
        capture_name="capture.zarr.tmp",
        expected_records_per_receiver=100,
        receiver_count=2,
        minimum_write_interval_seconds=5,
        wall_time=clock.wall_time,
        monotonic_time=clock.monotonic_time,
    )

    assert writer.publish("collecting", [0, 0], force=True)
    clock.advance(2)
    assert not writer.publish("collecting", [4, 3])
    assert json.loads(path.read_text())["state"] == "collecting"

    clock.advance(3)
    assert writer.publish("collecting", [10, 9])
    status = json.loads(path.read_text())
    assert status["records_written_by_receiver"] == [10, 9]
    assert status["common_records_written"] == 9
    assert status["frames_per_second"] == 1.8
    assert status["estimated_remaining_seconds"] == 91 / 1.8
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_final_failure_bypasses_throttle_and_preserves_primary_error(tmp_path):
    clock = _Clock()
    path = tmp_path / "status.json"
    writer = CaptureStatusWriter(
        path,
        capture_name="capture.zarr.tmp",
        expected_records_per_receiver=100,
        receiver_count=1,
        minimum_write_interval_seconds=60,
        wall_time=clock.wall_time,
        monotonic_time=clock.monotonic_time,
    )
    writer.publish("collecting", [3], force=True)
    error = OSError(19, "radio disappeared")

    assert writer.publish("failed", [3], error=error)
    status = json.loads(path.read_text())
    assert status["state"] == "failed"
    assert status["error_type"] == "OSError"
    assert status["error_errno"] == 19
    assert status["records_written_by_receiver"] == [3]


def test_launcher_failure_marks_existing_status_without_hiding_error(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(
        json.dumps(
            {
                "state": "failed",
                "capture_name": "capture.zarr.tmp",
                "error_type": "OSError",
                "error_message": "USB transfer failed",
                "records_written_by_receiver": [42],
            }
        )
    )

    result = mark_failed(path, exit_code=1)

    assert result["state"] == "failed"
    assert result["launcher_exit_code"] == 1
    assert result["error_type"] == "OSError"
    assert result["error_message"] == "USB transfer failed"
    assert result["records_written_by_receiver"] == [42]


def test_slow_capture_sets_late_watchdog_and_warns_once(tmp_path, caplog):
    clock = _Clock()
    writer = CaptureStatusWriter(
        tmp_path / "status.json",
        capture_name="slow.zarr.tmp",
        expected_records_per_receiver=100,
        receiver_count=1,
        minimum_write_interval_seconds=0,
        minimum_expected_frames_per_second=2,
        late_multiplier=1.2,
        late_grace_seconds=30,
        wall_time=clock.wall_time,
        monotonic_time=clock.monotonic_time,
    )
    writer.publish("collecting", [0], force=True)
    clock.advance(60)

    writer.publish("collecting", [10])
    writer.publish("collecting", [10])

    status = writer.last_payload
    assert status["late"] is True
    assert status["expected_duration_seconds"] == 50
    assert status["projected_duration_seconds"] == pytest.approx(600)
    assert caplog.text.count("Capture is late") == 1


def test_status_formatter_is_one_line_and_includes_primary_failure():
    rendered = format_status(
        {
            "state": "failed",
            "capture_name": "capture.zarr.tmp",
            "records_written_by_receiver": [12, 11],
            "frames_per_second": 1.75,
            "estimated_remaining_seconds": 42.25,
            "late": True,
            "error_type": "OSError",
            "error_message": "radio missing",
        }
    )

    assert "state=failed" in rendered
    assert "records=12,11" in rendered
    assert "rate_hz=1.750" in rendered
    assert "late=true" in rendered
    assert "error=OSError: radio missing" in rendered
    assert "\n" not in rendered
