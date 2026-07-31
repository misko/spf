import logging
import threading

import pytest

from spf.data_collector import DataCollector
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sdr_controller import PPlus


class _DisconnectedCollector(DataCollector):
    def __init__(self, data_filename, primary_error):
        self.primary_error = primary_error
        super().__init__(
            yaml_config={"receivers": []},
            data_filename=str(data_filename),
            position_controller=None,
            thread_class=None,
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
