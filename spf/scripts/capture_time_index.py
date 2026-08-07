"""Recover the true UTC start of captures whose FILENAME carries a wrong time.

Between 2026-07-30 and 2026-08-05, `system_clock_is_plausible` let a rover keep
an unsynced clock whenever it already looked "plausible", so 19 of 47 campaign
captures were NAMED from a stale Pi clock. The recorded `gps_timestamp` inside
each store came from MAVLink `SYSTEM_TIME` and is unaffected, so nothing was
lost -- but the name is what every human and half the tooling sorts by.

RAW DATA IS IMMUTABLE. This never renames, never opens a store for writing, and
refuses to place its output inside a directory it scanned. It emits a sidecar
index; correcting anything is a separate, deliberate act.

    python -m spf.scripts.capture_time_index /mnt/md2/rovers/... \
        --output /mnt/md2/cache/capture_time_index/aug.json

TIMEZONE, which is the whole reason this is not a two-line script. Capture names
come from `datetime.fromtimestamp(...)` with no tzinfo -- the rover's LOCAL wall
clock -- and the fleet runs Europe/London (`spf_fleet_timezone`). In August that
is UTC+1, so comparing a filename against the store's UTC `gps_timestamp`
directly reports a one-hour skew on every healthy capture in the campaign. The
local zone is therefore explicit and overridable, and `--timezone UTC` is the
right flag for data from a rover that never got its zone reconciled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

# The fleet default, from data_collection/rover/rover_v3.1/rover_env_defaults.sh
# (spf_fleet_timezone). Kept as a string so a mismatch is visible in the index.
FLEET_TIMEZONE = "Europe/London"

# `<craft>_YYYY_MM_DD_HH_MM_SS_...`. Anchored to the first match so a tag that
# happens to contain digits cannot be mistaken for the timestamp.
NAME_TIME = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")

# Below this, a difference is the ordinary lag between the process start that
# names the file and the first GPS sample that lands in it -- not a wrong clock.
DEFAULT_TOLERANCE_SECONDS = 300.0

# A GPS-derived UTC before 2025-01-01 is a fix without UTC, not a real time.
# Same floor drone_run.sh applies before it will set the system clock.
EPOCH_FLOOR = 1735689600


@dataclass
class CaptureTime:
    path: str
    name_time_local: str | None
    name_epoch: float | None
    true_epoch: float | None
    true_utc: str | None
    true_local: str | None
    skew_seconds: float | None
    receivers_read: int
    verdict: str
    detail: str = ""


def _first_gps_epoch(store) -> tuple[float | None, int]:
    """Earliest real GPS epoch across every receiver, and how many were read.

    Every receiver, not just r0: a capture whose r0 never got a fix still has
    the truth in r1, and taking the minimum matches how the name is generated
    (once, at process start) rather than per receiver.
    """
    receivers = store["receivers"]
    best = None
    read = 0
    for name in receivers:
        try:
            stamps = np.asarray(receivers[name]["gps_timestamp"][:], dtype=np.float64)
        except (KeyError, IndexError, TypeError):
            continue
        read += 1
        # >= the floor, not just > 0: an epoch-0 row and a "fix but no UTC" row
        # are both non-times, and the second one survives a > 0 test.
        usable = stamps[np.isfinite(stamps) & (stamps >= EPOCH_FLOOR)]
        if usable.size == 0:
            continue
        candidate = float(usable.min())
        best = candidate if best is None else min(best, candidate)
    return best, read


def _name_epoch(path: Path, local_zone: ZoneInfo) -> tuple[str | None, float | None]:
    match = NAME_TIME.search(path.name)
    if match is None:
        return None, None
    stamp = "_".join(match.groups())
    naive = datetime.strptime(stamp, "%Y_%m_%d_%H_%M_%S")
    # fold=0: at the autumn DST repeat one local time maps to two instants. The
    # earlier one is the honest default, and the ambiguity is why the skew is
    # reported rather than silently corrected.
    return stamp, naive.replace(tzinfo=local_zone, fold=0).timestamp()


def inspect_capture(
    path: Path, local_zone: ZoneInfo, tolerance_seconds: float
) -> CaptureTime:
    name_stamp, name_epoch = _name_epoch(path, local_zone)
    record = CaptureTime(
        path=str(path),
        name_time_local=name_stamp,
        name_epoch=name_epoch,
        true_epoch=None,
        true_utc=None,
        true_local=None,
        skew_seconds=None,
        receivers_read=0,
        verdict="UNREADABLE",
    )

    try:
        # mode="r": raw data is immutable, and LMDB will happily create or
        # upgrade a store that is opened any other way.
        store = zarr_open_from_lmdb_store(str(path), readahead=True, mode="r")
    except Exception as error:  # noqa: BLE001 - report, never abort the scan
        record.detail = f"{type(error).__name__}: {error}"
        return record

    true_epoch, receivers_read = _first_gps_epoch(store)
    record.receivers_read = receivers_read
    if true_epoch is None:
        record.verdict = "NO_GPS_TIME"
        record.detail = (
            "no gps_timestamp at or after the 2025 floor in any receiver; this "
            "capture cannot date itself"
        )
        return record

    record.true_epoch = true_epoch
    record.true_utc = datetime.fromtimestamp(true_epoch, tz=timezone.utc).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    record.true_local = datetime.fromtimestamp(true_epoch, tz=local_zone).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    if name_epoch is None:
        record.verdict = "NO_NAME_TIME"
        record.detail = "filename carries no YYYY_MM_DD_HH_MM_SS stamp"
        return record

    record.skew_seconds = name_epoch - true_epoch
    record.verdict = (
        "OK" if abs(record.skew_seconds) <= tolerance_seconds else "MISDATED"
    )
    if record.verdict == "MISDATED":
        record.detail = (
            f"name says {name_stamp} local; GPS says {record.true_local} local "
            f"({record.true_utc} UTC)"
        )
    return record


def find_captures(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.name.endswith(".zarr"):
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob("*.zarr")))
    # A .zarr is a directory; rglob inside one would find nothing, but dedupe
    # anyway so overlapping roots do not read the same store twice.
    return sorted(set(found))


def _refuse_output_inside_sources(output: Path, roots: list[Path]) -> None:
    """The index must not land in the tree it describes.

    Writing beside immutable raw data is how a "read-only" tool becomes the
    thing that modified a dataset.
    """
    resolved = output.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if root_resolved == resolved or root_resolved in resolved.parents:
            raise SystemExit(
                f"Refusing to write {output} inside scanned path {root}. Raw data "
                "is immutable; write the index to a new location "
                "(e.g. /mnt/md2/cache/capture_time_index/)."
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="captures or directories")
    parser.add_argument(
        "--output", type=Path, required=True, help="where to write the JSON index"
    )
    parser.add_argument(
        "--timezone",
        default=FLEET_TIMEZONE,
        help=(
            "zone the FILENAMES were written in (the rover's local clock). "
            f"Default {FLEET_TIMEZONE}; use UTC for a rover whose zone was "
            "never reconciled."
        ),
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help="difference below which a capture counts as correctly named",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    local_zone = ZoneInfo(args.timezone)

    _refuse_output_inside_sources(args.output, args.paths)

    captures = find_captures(args.paths)
    if not captures:
        logging.error("No .zarr captures found under: %s", args.paths)
        return 1
    logging.info("Reading %d capture(s), read-only", len(captures))

    records = []
    for path in captures:
        record = inspect_capture(path, local_zone, args.tolerance_seconds)
        records.append(record)
        logging.info(
            "%-9s %+8s  %s",
            record.verdict,
            "" if record.skew_seconds is None else f"{record.skew_seconds:.0f}s",
            path.name,
        )

    counts: dict[str, int] = {}
    for record in records:
        counts[record.verdict] = counts.get(record.verdict, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": [str(path) for path in args.paths],
        "filename_timezone": args.timezone,
        "tolerance_seconds": args.tolerance_seconds,
        "counts": counts,
        "captures": [asdict(record) for record in records],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logging.info("Wrote %s", args.output)
    for verdict in sorted(counts):
        logging.info("  %-11s %d", verdict, counts[verdict])

    # Non-zero when something needs a human, so a scan can gate a pipeline.
    return 0 if set(counts) <= {"OK"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
