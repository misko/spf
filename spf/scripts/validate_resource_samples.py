"""Validate that a production collector reaches a bounded host-memory plateau."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def validate(
    path: Path,
    *,
    minimum_samples: int = 5,
    warmup_fraction: float = 0.2,
    maximum_anon_range_mib: float = 128.0,
    minimum_available_mib: float = 128.0,
) -> dict:
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be at least two")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < minimum_samples:
        raise ValueError(
            f"{path}: found {len(rows)} resource samples, expected at least "
            f"{minimum_samples}"
        )
    required = {"timestamp_unix", "rss_anon_kib", "available_kib"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    try:
        timestamp = np.asarray([float(row["timestamp_unix"]) for row in rows])
        rss_anon_mib = np.asarray([float(row["rss_anon_kib"]) / 1024.0 for row in rows])
        available_mib = np.asarray(
            [float(row["available_kib"]) / 1024.0 for row in rows]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: non-numeric resource sample: {error}") from error
    if not np.isfinite(timestamp).all() or not np.isfinite(rss_anon_mib).all():
        raise ValueError(f"{path}: resource samples contain non-finite values")
    if not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")

    warmup_samples = min(len(rows) - 2, int(np.ceil(len(rows) * warmup_fraction)))
    plateau = rss_anon_mib[warmup_samples:]
    anon_range_mib = float(np.ptp(plateau))
    minimum_observed_available_mib = float(available_mib.min())
    elapsed_hours = float((timestamp[-1] - timestamp[warmup_samples]) / 3600.0)
    if elapsed_hours > 0:
        slope_mib_per_hour = float(
            np.polyfit(
                (timestamp[warmup_samples:] - timestamp[warmup_samples]) / 3600.0,
                plateau,
                1,
            )[0]
        )
    else:
        slope_mib_per_hour = 0.0

    failures = []
    if anon_range_mib > maximum_anon_range_mib:
        failures.append(
            f"post-warmup anonymous RSS range {anon_range_mib:.1f} MiB exceeds "
            f"{maximum_anon_range_mib:.1f} MiB"
        )
    if minimum_observed_available_mib < minimum_available_mib:
        failures.append(
            f"minimum host available memory {minimum_observed_available_mib:.1f} MiB "
            f"is below {minimum_available_mib:.1f} MiB"
        )
    return {
        "status": "fail" if failures else "pass",
        "sample_count": len(rows),
        "warmup_samples_ignored": warmup_samples,
        "elapsed_seconds": float(timestamp[-1] - timestamp[0]),
        "post_warmup_anon_range_mib": anon_range_mib,
        "post_warmup_anon_slope_mib_per_hour": slope_mib_per_hour,
        "minimum_available_mib": minimum_observed_available_mib,
        "limits": {
            "maximum_anon_range_mib": maximum_anon_range_mib,
            "minimum_available_mib": minimum_available_mib,
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-anon-range-mib", type=float, default=128.0)
    parser.add_argument("--minimum-available-mib", type=float, default=128.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(
            args.resources,
            minimum_samples=args.minimum_samples,
            warmup_fraction=args.warmup_fraction,
            maximum_anon_range_mib=args.maximum_anon_range_mib,
            minimum_available_mib=args.minimum_available_mib,
        )
    except (OSError, ValueError) as error:
        result = {"status": "fail", "failures": [str(error)]}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
