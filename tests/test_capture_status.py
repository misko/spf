import json

from spf.capture_status import CaptureStatusWriter, mark_failed


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

    assert writer.publish("starting", [0, 0], force=True)
    clock.advance(2)
    assert not writer.publish("collecting", [4, 3])
    assert json.loads(path.read_text())["state"] == "starting"

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
