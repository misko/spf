"""Minimal, dependency-free capture failure termination primitives."""

from __future__ import annotations

import logging
import os
import sys


def capture_failure_exit_code(error: BaseException) -> int:
    """Return a non-success process code for a fully finalized capture failure."""

    signal_number = getattr(error, "signal_number", None)
    if signal_number is not None:
        return 128 + int(signal_number)
    if isinstance(error, SystemExit):
        try:
            requested = int(error.code)
        except (TypeError, ValueError):
            requested = 1
        # A SystemExit raised while a capture is active cannot be reported as a
        # successful completed artifact: cleanup marks that store incomplete.
        return requested if requested != 0 else 1
    return 1


def terminate_capture_process(
    error: BaseException,
    *,
    incident_id: str | None,
    error_source: str | None,
) -> None:
    """Log one owning traceback, flush it, then bypass unsafe C finalizers.

    The pinned libiio/libusb stack can leave a native event thread alive after
    an otherwise complete radio teardown.  Ordinary capture failures therefore
    need the same bounded process-exit guarantee as SIGINT/SIGTERM.  Callers
    must finalize the temporary store and request vehicle HOLD before entering
    this function.
    """

    logging.getLogger("spf.capture").error(
        "Capture incident %s from %s terminated the capture process: %s: %s",
        incident_id or "unassigned",
        error_source or "unknown",
        type(error).__name__,
        error,
        exc_info=(type(error), error, error.__traceback__),
    )
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(capture_failure_exit_code(error))
