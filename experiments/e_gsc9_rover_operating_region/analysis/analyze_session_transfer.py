#!/usr/bin/env python3
"""Compare E-GSC9 session-B cell means with the matching session-A cells."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.comparative_analysis import _load_radio
from spf.calibrations.dual_rx_gain_frequency.model_matrix import _metrics
from spf.calibrations.dual_rx_gain_frequency.runner import load_calibration_document


SCHEMA = "spf.experiment.e_gsc9.session_transfer"
SCHEMA_VERSION = 1
H6_THRESHOLD_DEG = 0.5
H6_MINIMUM_SEPARATION_SECONDS = 12 * 60 * 60
SERIALS = (
    "104000bac4950008230026001b440a003a",
    "1040007c4a94000211000b009186843ef2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _circular_mean(values: list[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if not vector.size:
        raise ValueError("cannot average an empty circular vector")
    return float(np.angle(np.mean(np.exp(1j * vector))))


def _cell_means(radio: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    selected = radio["quality_mask"] & np.isfinite(radio["phase"])
    for frequency, gain1, gain2, phase in zip(
        radio["frequency_hz"][selected],
        radio["gain1"][selected],
        radio["gain2"][selected],
        radio["phase"][selected],
    ):
        groups[(int(frequency), int(gain1), int(gain2))].append(float(phase))
    return {coordinate: _circular_mean(values) for coordinate, values in groups.items()}


def _coordinates(rows: set[tuple[int, int, int]]) -> list[list[int]]:
    return [list(row) for row in sorted(rows)]


def _summarize(
    residuals: np.ndarray, *, expected: int
) -> dict[str, float | int | None]:
    result = _metrics(residuals, expected=expected)
    absolute_deg = np.abs(np.degrees(residuals))
    result["circular_median_absolute_deg"] = (
        float(np.median(absolute_deg)) if absolute_deg.size else None
    )
    return result


def _validation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing strict validation: {path}")
    document = json.loads(path.read_text())
    if document.get("status") != "pass":
        raise ValueError(f"strict validation is not pass: {path}")
    return document


def _compare_radio(
    *,
    serial: str,
    primary: dict[str, Any],
    repeat: dict[str, Any],
    expected_by_frequency: dict[int, set[tuple[int, int, int]]],
) -> dict[str, Any]:
    if primary["serial"] != serial or repeat["serial"] != serial:
        raise ValueError(f"dataset serial mismatch for {serial}")
    primary_times = primary["timestamp"][primary["analysis_mask"]]
    repeat_times = repeat["timestamp"][repeat["analysis_mask"]]
    if not primary_times.size or not repeat_times.size:
        raise ValueError(f"missing complete-block timestamps for {serial}")
    primary_start = float(np.min(primary_times))
    primary_end = float(np.max(primary_times))
    repeat_start = float(np.min(repeat_times))
    repeat_end = float(np.max(repeat_times))
    separation_seconds = repeat_start - primary_end
    first = _cell_means(primary)
    second = _cell_means(repeat)
    frequency_rows = []
    all_residuals: list[float] = []
    for frequency, expected in sorted(expected_by_frequency.items()):
        primary_cells = set(first) & expected
        repeat_cells = set(second) & expected
        common = primary_cells & repeat_cells
        residuals = np.asarray(
            [wrap_phase(second[cell] - first[cell]) for cell in sorted(common)],
            dtype=np.float64,
        )
        all_residuals.extend(residuals.tolist())
        metrics = _summarize(residuals, expected=len(expected))
        frequency_rows.append(
            {
                "frequency_hz": frequency,
                "expected_cells": len(expected),
                "primary_quality_valid_cells": len(primary_cells),
                "repeat_quality_valid_cells": len(repeat_cells),
                "common_quality_valid_cells": len(common),
                "missing_primary_cells": _coordinates(expected - primary_cells),
                "missing_repeat_cells": _coordinates(expected - repeat_cells),
                **metrics,
                "h6_pass": bool(
                    len(common) == len(expected)
                    and metrics["circular_mae_deg"] is not None
                    and metrics["circular_mae_deg"] < H6_THRESHOLD_DEG
                ),
            }
        )
    overall = _summarize(
        np.asarray(all_residuals, dtype=np.float64),
        expected=sum(len(cells) for cells in expected_by_frequency.values()),
    )
    return {
        "serial": serial,
        "primary_analysis_input_sha256": primary["analysis_input_sha256"],
        "repeat_analysis_input_sha256": repeat["analysis_input_sha256"],
        "primary_capture_unix_seconds": {
            "first": primary_start,
            "last": primary_end,
        },
        "repeat_capture_unix_seconds": {
            "first": repeat_start,
            "last": repeat_end,
        },
        "a_end_to_b_start_seconds": separation_seconds,
        "minimum_separation_seconds": H6_MINIMUM_SEPARATION_SECONDS,
        "separation_pass": separation_seconds >= H6_MINIMUM_SEPARATION_SECONDS,
        "per_frequency": frequency_rows,
        "overall": overall,
        "h6_pass": bool(
            separation_seconds >= H6_MINIMUM_SEPARATION_SECONDS
            and all(row["h6_pass"] for row in frequency_rows)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-a-config", type=Path, required=True)
    parser.add_argument("--session-a-root", type=Path, required=True)
    parser.add_argument("--session-b-config", type=Path, required=True)
    parser.add_argument("--session-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, primary_config = load_calibration_document(args.session_a_config)
    _, repeat_config = load_calibration_document(args.session_b_config)
    if tuple(primary_config.frequencies_hz) != tuple(repeat_config.frequencies_hz):
        raise ValueError("session A and B frequencies differ")
    if not set(repeat_config.gains_db).issubset(primary_config.gains_db):
        raise ValueError("session B gains are not a subset of session A gains")

    expected_by_frequency = {
        int(frequency): {
            (int(frequency), int(gain1), int(gain2))
            for gain1, gain2 in repeat_config.gain_pairs
        }
        for frequency in repeat_config.frequencies_hz
    }
    rows = []
    validations = []
    for serial in SERIALS:
        primary_path = args.session_a_root / serial / "calibration.v7.zarr"
        repeat_path = args.session_b_root / serial / "calibration.v7.zarr"
        primary_validation_path = args.session_a_root / serial / "validation.json"
        repeat_validation_path = args.session_b_root / serial / "validation.json"
        validations.extend(
            [
                {
                    "path": str(primary_validation_path.resolve()),
                    "status": _validation(primary_validation_path)["status"],
                },
                {
                    "path": str(repeat_validation_path.resolve()),
                    "status": _validation(repeat_validation_path)["status"],
                },
            ]
        )
        rows.append(
            _compare_radio(
                serial=serial,
                primary=_load_radio(primary_path, config=primary_config),
                repeat=_load_radio(repeat_path, config=repeat_config),
                expected_by_frequency=expected_by_frequency,
            )
        )

    repo = Path(__file__).resolve().parents[3]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix_ns": time.time_ns(),
        "spf_git_sha": git_sha,
        "decision_rule": {
            "metric": "circular MAE of matching quality-valid cell means",
            "threshold_deg_exclusive": H6_THRESHOLD_DEG,
            "strata": "each physical radio and each LO independently",
            "coverage": "all session-B scheduled cells must be present in both sessions",
            "minimum_a_end_to_b_start_seconds": H6_MINIMUM_SEPARATION_SECONDS,
            "overall_pass": "every radio-by-LO stratum passes",
        },
        "session_a": {
            "config": str(args.session_a_config.resolve()),
            "config_sha256": _sha256(args.session_a_config),
            "root": str(args.session_a_root.resolve()),
        },
        "session_b": {
            "config": str(args.session_b_config.resolve()),
            "config_sha256": _sha256(args.session_b_config),
            "root": str(args.session_b_root.resolve()),
        },
        "validations": validations,
        "per_radio": rows,
        "h6_pass": all(row["h6_pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
