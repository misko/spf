"""Validate resource safety across a lifecycle-churning interruption soak."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _numeric_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "timestamp_unix",
        "rss_kib",
        "rss_anon_kib",
        "available_kib",
        "artifact_kib",
    }
    if not rows:
        raise ValueError(f"{path}: no resource samples")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    try:
        return [
            {name: float(row[name]) for name in required}
            for row in rows
        ]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: non-numeric resource sample: {error}") from error


def _completed_rounds(path: Path) -> list[dict[str, int]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    completed = []
    for expected, row in enumerate(rows, start=1):
        if int(row["round"]) != expected or int(row["status"]) != 0:
            raise ValueError(f"{path}: invalid completed round row {row}")
        completed.append(
            {
                "round": expected,
                "started_unix": int(row["started_unix"]),
                "finished_unix": int(row["finished_unix"]),
            }
        )
    return completed


def validate(
    resources: Path,
    rounds: Path,
    *,
    minimum_samples: int = 5,
    maximum_anon_mib: float = 1024.0,
    minimum_available_mib: float = 256.0,
    recovery_anon_mib: float = 384.0,
) -> dict:
    samples = _numeric_rows(resources)
    completed = _completed_rounds(rounds)
    if len(samples) < minimum_samples:
        raise ValueError(
            f"{resources}: found {len(samples)} samples, require {minimum_samples}"
        )
    timestamps = [row["timestamp_unix"] for row in samples]
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{resources}: timestamps are not strictly increasing")

    anon_mib = [row["rss_anon_kib"] / 1024.0 for row in samples]
    rss_mib = [row["rss_kib"] / 1024.0 for row in samples]
    available_mib = [row["available_kib"] / 1024.0 for row in samples]
    maximum_observed_anon = max(anon_mib)
    minimum_observed_anon = min(anon_mib)
    minimum_observed_available = min(available_mib)
    failures = []
    if maximum_observed_anon > maximum_anon_mib:
        failures.append(
            f"peak aggregate anonymous RSS {maximum_observed_anon:.1f} MiB "
            f"exceeds {maximum_anon_mib:.1f} MiB"
        )
    if minimum_observed_available < minimum_available_mib:
        failures.append(
            f"minimum host available memory {minimum_observed_available:.1f} MiB "
            f"is below {minimum_available_mib:.1f} MiB"
        )
    if minimum_observed_anon > recovery_anon_mib:
        failures.append(
            f"anonymous RSS never recovered below {recovery_anon_mib:.1f} MiB"
        )

    round_summaries = []
    for item in completed:
        selected = [
            row
            for row in samples
            if item["started_unix"] <= row["timestamp_unix"] <= item["finished_unix"]
        ]
        minimum_round_anon_mib = (
            min(row["rss_anon_kib"] for row in selected) / 1024.0
            if selected
            else None
        )
        if not selected:
            failures.append(f"round {item['round']} has no resource samples")
        elif minimum_round_anon_mib > recovery_anon_mib:
            failures.append(
                f"round {item['round']} anonymous RSS never recovered below "
                f"{recovery_anon_mib:.1f} MiB"
            )
        round_summaries.append(
            {
                "round": item["round"],
                "sample_count": len(selected),
                "peak_anon_mib": (
                    max(row["rss_anon_kib"] for row in selected) / 1024.0
                    if selected
                    else None
                ),
                "minimum_anon_mib": minimum_round_anon_mib,
                "minimum_available_mib": (
                    min(row["available_kib"] for row in selected) / 1024.0
                    if selected
                    else None
                ),
            }
        )

    return {
        "status": "fail" if failures else "pass",
        "sample_count": len(samples),
        "elapsed_seconds": timestamps[-1] - timestamps[0],
        "completed_rounds": len(completed),
        "peak_rss_mib": max(rss_mib),
        "peak_anon_mib": maximum_observed_anon,
        "minimum_anon_mib": minimum_observed_anon,
        "minimum_available_mib": minimum_observed_available,
        "maximum_artifact_mib": max(row["artifact_kib"] for row in samples) / 1024.0,
        "limits": {
            "maximum_anon_mib": maximum_anon_mib,
            "minimum_available_mib": minimum_available_mib,
            "recovery_anon_mib": recovery_anon_mib,
        },
        "rounds": round_summaries,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", type=Path)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--maximum-anon-mib", type=float, default=1024.0)
    parser.add_argument("--minimum-available-mib", type=float, default=256.0)
    parser.add_argument("--recovery-anon-mib", type=float, default=384.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(
            args.resources,
            args.rounds,
            minimum_samples=args.minimum_samples,
            maximum_anon_mib=args.maximum_anon_mib,
            minimum_available_mib=args.minimum_available_mib,
            recovery_anon_mib=args.recovery_anon_mib,
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
