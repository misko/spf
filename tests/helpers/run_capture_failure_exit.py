"""Subprocess probe for bounded exit after an ordinary capture failure."""

from __future__ import annotations

import logging
import threading
import time

from spf.capture_failure import terminate_capture_process


def _linger_forever() -> None:
    while True:
        time.sleep(60)


logging.basicConfig(level=logging.INFO)
threading.Thread(target=_linger_forever, daemon=False).start()

try:
    raise RuntimeError("synthetic direct USB capture failure")
except RuntimeError as error:
    terminate_capture_process(
        error,
        incident_id="incident-test",
        error_source="receiver:test-radio",
    )
