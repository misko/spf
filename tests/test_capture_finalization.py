import logging
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from spf.data_collector import (
    CaptureInterrupted,
    DataCollector,
    ThreadedRX,
    capture_signal_handlers,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sdr_controller import PPlus


class _DisconnectedCollector(DataCollector):
    def __init__(self, data_filename, primary_error, status_writer=None):
        self.primary_error = primary_error
        super().__init__(
            yaml_config={"receivers": []},
            data_filename=str(data_filename),
            position_controller=None,
            thread_class=None,
            status_writer=status_writer,
        )

    def setup_record_matrix(self):
        self.zarr = zarr_open_from_lmdb_store(self.data_filename, mode="w")
        self.zarr.create_group("receivers")

    def run_inner_collector_thread(self):
        raise self.primary_error

    def close(self):
        zarr = getattr(self, "zarr", None)
        if zarr is None:
            return
        zarr.store.close()
        self.zarr = None


class _CleanupAlsoDisconnects:
    def __init__(self, cleanup_error):
        self.cleanup_error = cleanup_error
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise self.cleanup_error


class _StatusSpy:
    def __init__(self):
        self.events = []

    def publish(self, state, counts, **kwargs):
        self.events.append((state, list(counts), kwargs))
        return True


class _QueuedReadThread:
    def __init__(self, records):
        self.read_q = queue.Queue()
        for record in records:
            self.read_q.put(record)
        self.error = None


class _WriterConcurrencyProbe(DataCollector):
    def __init__(self, records, receiver_count=1):
        self._probe_lock = threading.Lock()
        self.active_by_receiver = [0] * receiver_count
        self.max_active_by_receiver = [0] * receiver_count
        self.active_global = 0
        self.max_active_global = 0
        self.first_record_barrier = (
            threading.Barrier(receiver_count) if receiver_count > 1 else None
        )
        self.write_order = []
        super().__init__(
            yaml_config={
                "receivers": [{} for _ in range(receiver_count)],
                "n-records-per-receiver": len(records),
            },
            data_filename=None,
            position_controller=None,
            thread_class=None,
        )
        self.read_threads = [_QueuedReadThread(records) for _ in range(receiver_count)]

    def setup_record_matrix(self):
        pass

    def write_to_record_matrix(self, thread_idx, record_idx, data):
        with self._probe_lock:
            self.active_by_receiver[thread_idx] += 1
            self.active_global += 1
            self.max_active_by_receiver[thread_idx] = max(
                self.max_active_by_receiver[thread_idx],
                self.active_by_receiver[thread_idx],
            )
            self.max_active_global = max(self.max_active_global, self.active_global)
        if record_idx == 0 and self.first_record_barrier is not None:
            self.first_record_barrier.wait(timeout=2)
        # Make overlap deterministic with the old shared two-worker executor.
        time.sleep(0.02)
        with self._probe_lock:
            self.write_order.append((thread_idx, record_idx, data))
            self.active_by_receiver[thread_idx] -= 1
            self.active_global -= 1


def test_records_for_one_receiver_are_written_serially_in_fifo_order():
    records = list(range(20))
    collector = _WriterConcurrencyProbe(records)

    collector.run_inner_collector_thread()

    assert collector.max_active_by_receiver == [1]
    assert collector.write_order == [(0, index, index) for index in records]
    assert collector.records_written_by_receiver == [len(records)]


def test_receivers_write_in_parallel_but_each_remains_fifo():
    records = list(range(10))
    collector = _WriterConcurrencyProbe(records, receiver_count=2)

    collector.run_inner_collector_thread()

    assert collector.max_active_by_receiver == [1, 1]
    assert collector.max_active_global == 2
    for receiver_idx in range(2):
        assert [
            record_idx
            for observed_receiver, record_idx, _data in collector.write_order
            if observed_receiver == receiver_idx
        ] == records
    assert collector.records_written_by_receiver == [len(records), len(records)]


def test_disconnect_preserves_primary_error_and_finalizes_partial_zarr(
    tmp_path, caplog
):
    """A cleanup failure must not hide the capture failure or strand LMDB."""

    partial_path = tmp_path / "capture.zarr.tmp"
    primary_error = OSError(19, "Pluto disappeared during bulk RX")
    cleanup_error = OSError(19, "TX mute failed because Pluto is gone")
    radio = _CleanupAlsoDisconnects(cleanup_error)
    collector = _DisconnectedCollector(partial_path, primary_error)
    collector.receiver_pplus = {"usb:1.4.5": radio}
    collector.read_threads = []
    collector.collector_thread = threading.Thread(target=collector.run_collector_thread)

    with caplog.at_level(logging.ERROR):
        collector.start()
        with pytest.raises(OSError) as raised:
            collector.done()

    assert raised.value is primary_error
    assert radio.close_calls == 1
    assert "TX mute failed because Pluto is gone" in caplog.text
    assert partial_path.is_dir()
    assert not (tmp_path / "capture.zarr").exists()

    partial = zarr_open_from_lmdb_store(str(partial_path), mode="r")
    try:
        assert partial.attrs["capture_status"] == "incomplete"
        assert partial.attrs["capture_error_type"] == "OSError"
        assert (
            "Pluto disappeared during bulk RX" in partial.attrs["capture_error_message"]
        )
        assert any(
            "TX mute failed because Pluto is gone" in failure
            for failure in partial.attrs["capture_cleanup_errors"]
        )
    finally:
        partial.store.close()


def test_disconnect_publishes_durable_failed_state_with_primary_error(tmp_path):
    primary_error = OSError(19, "Pluto disappeared during bulk RX")
    status = _StatusSpy()
    collector = _DisconnectedCollector(
        tmp_path / "capture.zarr.tmp", primary_error, status_writer=status
    )
    collector.receiver_pplus = {}
    collector.read_threads = []
    collector.collector_thread = threading.Thread(target=collector.run_collector_thread)

    collector.start()
    with pytest.raises(OSError) as raised:
        collector.done()

    assert raised.value is primary_error
    assert [event[0] for event in status.events] == [
        "preparing",
        "collecting",
        "failed",
    ]
    assert status.events[-1][2]["error"] is primary_error


def test_pluto_close_attempts_tx_mute_after_rx_cleanup_failure():
    """RX teardown must not prevent the independent TX safety teardown."""

    radio = object.__new__(PPlus)
    radio.uri = "usb:1.4.5"
    calls = []

    def close_rx():
        calls.append("rx")
        raise OSError(19, "RX transport is gone")

    def close_tx():
        calls.append("tx")
        raise OSError(19, "TX mute failed")

    radio.close_rx = close_rx
    radio.close_tx = close_tx

    with pytest.raises(RuntimeError) as raised:
        radio.close()

    assert calls == ["rx", "tx"]
    assert "RX transport is gone" in str(raised.value)
    assert "TX mute failed" in str(raised.value)


def test_pluto_close_explicitly_releases_iio_context():
    class FakeContext:
        def __init__(self):
            self._context = object()
            self.destroy_calls = 0

        def __del__(self):
            if self._context is not None:
                self.destroy_calls += 1

    class FakeSdr:
        def __init__(self):
            self._ctx = FakeContext()
            self.tx_enabled_channels = [0, 1]
            self.tx_cyclic_buffer = True

        def tx_destroy_buffer(self):
            pass

        def rx_destroy_buffer(self):
            pass

    radio = object.__new__(PPlus)
    radio.uri = "usb:1.2.5"
    radio.sdr = FakeSdr()
    original_sdr = radio.sdr
    original_context = original_sdr._ctx
    radio.direct_rx = None
    radio.rx_config = None
    radio.tx_config = None
    radio._last_direct_gains = None
    radio._last_direct_rssis = None
    radio._last_direct_metadata = None

    radio.close()
    radio.close()

    assert radio.sdr is None
    assert original_sdr._ctx is None
    assert original_context._context is None
    assert original_context.destroy_calls == 1


def test_signal_handler_requests_stop_and_restores_process_handlers(tmp_path):
    collector = _DisconnectedCollector(
        tmp_path / "capture.zarr.tmp", RuntimeError("unused")
    )
    collector.collection_error = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    with capture_signal_handlers(collector):
        os.kill(os.getpid(), signal.SIGTERM)
        assert collector.stop_requested.wait(timeout=1)
        assert isinstance(collector.collection_error, CaptureInterrupted)
        assert "SIGTERM" in str(collector.collection_error)

    assert signal.getsignal(signal.SIGTERM) is previous_sigterm
    collector.close()


def test_stop_request_preserves_a_receiver_error_that_won_the_race(tmp_path):
    collector = _DisconnectedCollector(
        tmp_path / "capture.zarr.tmp", RuntimeError("unused")
    )
    collector.collection_error = None
    usb_error = OSError(19, "radio disappeared before SIGTERM")
    collector.read_threads = [SimpleNamespace(error=usb_error, run=True)]

    collector.request_stop(CaptureInterrupted(signal.SIGTERM))

    assert collector.collection_error is usb_error
    assert collector.read_threads[0].run is False
    collector.close()


def test_reader_error_never_blocks_behind_a_full_result_queue(monkeypatch):
    reader = object.__new__(ThreadedRX)
    reader.pplus = SimpleNamespace(soft_reset_radio=lambda: None)
    reader.rx_config = SimpleNamespace(
        uri="test://radio", rx_pos=[[0.0, 0.0], [0.04, 0.0]], lo=2.4e9
    )
    reader.nthetas = 65
    reader.read_q = queue.Queue(maxsize=1)
    reader.read_q.put(object())
    reader.run = True
    reader.error = None
    reader.get_data = lambda: (_ for _ in ()).throw(OSError(19, "radio gone"))
    monkeypatch.setattr(
        "spf.data_collector.precompute_steering_vectors", lambda **_kwargs: None
    )

    reader.read_forever()

    assert isinstance(reader.error, OSError)
    assert reader.run is False
    assert reader.read_q.full()


def _start_interrupt_worker(path, repo_root):
    command = [
        sys.executable,
        str(repo_root / "tests/helpers/run_interruptible_capture.py"),
        str(path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), environment.get("PYTHONPATH", "")]
    )
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    readable, _, _ = select.select([process.stdout], [], [], 10)
    if not readable:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"interrupt worker did not become ready: {stdout=} {stderr=}")
    assert process.stdout.readline().strip() == "READY"
    return process


def _wait_for_committed_progress(path, minimum=2):
    deadline = time.monotonic() + 10
    last_error = None
    while time.monotonic() < deadline:
        try:
            capture = zarr_open_from_lmdb_store(str(path), mode="r")
            try:
                counts = capture.attrs["capture_records_written_by_receiver"]
                if counts[0] >= minimum:
                    return counts[0]
            finally:
                capture.store.close()
        except Exception as error:
            last_error = error
        time.sleep(0.05)
    raise AssertionError(f"capture made no committed progress: {last_error}")


@pytest.mark.parametrize(
    ("terminate", "expected_status", "expected_error_type"),
    [
        ("interrupt", "incomplete", "CaptureInterrupted"),
        ("terminate", "incomplete", "CaptureInterrupted"),
        ("kill", "in_progress", None),
    ],
)
def test_subprocess_interruption_is_fail_closed_and_partial_lmdb_is_readable(
    tmp_path, terminate, expected_status, expected_error_type
):
    repo_root = Path(__file__).resolve().parents[1]
    partial_path = tmp_path / f"{terminate}.zarr.tmp"
    process = _start_interrupt_worker(partial_path, repo_root)
    committed_before_interrupt = _wait_for_committed_progress(partial_path)

    if terminate == "interrupt":
        process.send_signal(signal.SIGINT)
    else:
        getattr(process, terminate)()
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode != 0, (stdout, stderr)
    assert partial_path.is_dir()
    assert not (tmp_path / f"{terminate}.zarr").exists()

    partial = zarr_open_from_lmdb_store(str(partial_path), mode="r")
    try:
        assert partial.attrs["capture_status"] == expected_status
        committed_after_interrupt = partial.attrs[
            "capture_records_written_by_receiver"
        ][0]
        assert committed_after_interrupt >= committed_before_interrupt
        if expected_error_type is not None:
            assert partial.attrs["capture_error_type"] == expected_error_type
            expected_signal = "SIGINT" if terminate == "interrupt" else "SIGTERM"
            assert expected_signal in partial.attrs["capture_error_message"]
        else:
            assert "capture_error_type" not in partial.attrs
    finally:
        partial.store.close()
