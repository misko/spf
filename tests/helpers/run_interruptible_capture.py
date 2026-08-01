"""Subprocess worker used to exercise real SIGTERM/SIGKILL capture semantics."""

from __future__ import annotations

import sys
import threading
import time

from spf.data_collector import (
    CaptureInterrupted,
    DataCollector,
    capture_signal_handlers,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


class SlowCapture(DataCollector):
    def __init__(self, path):
        super().__init__(
            yaml_config={"receivers": [{}], "n-records-per-receiver": 100_000},
            data_filename=path,
            position_controller=None,
            thread_class=None,
        )
        self.read_threads = []
        self.receiver_pplus = {}
        self.collector_thread = threading.Thread(
            target=self.run_collector_thread, daemon=True
        )

    def setup_record_matrix(self):
        self.zarr = zarr_open_from_lmdb_store(self.data_filename, mode="w")
        self.zarr.create_group("receivers")

    def write_to_record_matrix(self, _thread_idx, record_idx, _data):
        self.zarr.attrs["last_complete_record"] = record_idx

    def run_inner_collector_thread(self):
        for record_idx in range(self.yaml_config["n-records-per-receiver"]):
            if self.stop_requested.wait(timeout=0.02):
                assert self.collection_error is not None
                raise self.collection_error
            self._write_record_and_track(0, record_idx, None)

    def close(self):
        zarr = getattr(self, "zarr", None)
        if zarr is not None:
            self.zarr = None
            zarr.store.close()


def main(path):
    collector = SlowCapture(path)
    with capture_signal_handlers(collector):
        collector.start()
        print("READY", flush=True)
        while collector.is_collecting():
            time.sleep(0.05)
        try:
            collector.done()
        except CaptureInterrupted:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
