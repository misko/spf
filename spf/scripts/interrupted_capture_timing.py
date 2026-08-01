"""Timing policy for real-radio interrupted-capture fault injection."""

from __future__ import annotations


STARTUP_TIMEOUT_SECONDS = 90.0
FIXED_CAPTURE_OVERHEAD_SECONDS = 30.0
MINIMUM_EXPECTED_COMMIT_RATE_HZ = 1.0


def interruption_progress_timeout_seconds(minimum_records: int) -> float:
    """Return a conservative absolute deadline for a requested frame boundary.

    Production 524288-sample V7 collection is normally close to 2 Hz on the
    Pi. The gate allows it to fall to 1 Hz, includes explicit setup overhead,
    and retains the historical 90-second floor for early interruption cases.
    """

    if minimum_records < 1:
        raise ValueError("minimum_records must be positive")
    capture_seconds = minimum_records / MINIMUM_EXPECTED_COMMIT_RATE_HZ
    return max(
        STARTUP_TIMEOUT_SECONDS,
        FIXED_CAPTURE_OVERHEAD_SECONDS + capture_seconds,
    )
