"""Audit a phase model against a read-only per-snapshot wall-array export."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phase import UnsupportedPhaseModelInput, load_model


REQUIRED_COLUMNS = {
    "capture",
    "receiver",
    "serial",
    "lo_hz",
    "gain_rx1_db",
    "gain_rx2_db",
    "phase_meas_rad",
    "phase_gt_rad",
    "theta_gt_rad",
    "tx_pos_x_mm",
    "tx_pos_y_mm",
    "rx_pos_x_mm",
    "rx_pos_y_mm",
    "rx_theta_in_pis",
    "rx_heading_in_pis",
    "d_over_lambda",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _phasor(value: float) -> complex:
    return complex(math.cos(value), math.sin(value))


@dataclass
class _Capture:
    rows: int = 0
    supported_rows: int = 0
    raw_all: complex = 0j
    raw_supported: complex = 0j
    corrected_subtract: complex = 0j
    corrected_add: complex = 0j
    gain1_sum: float = 0.0
    gain1_square_sum: float = 0.0

    def add_raw(self, raw_residual: float, gain1_db: int) -> None:
        self.rows += 1
        self.raw_all += _phasor(raw_residual)
        self.gain1_sum += gain1_db
        self.gain1_square_sum += gain1_db * gain1_db

    def add_supported(
        self,
        raw_residual: float,
        subtract_residual: float,
        add_residual: float,
    ) -> None:
        self.supported_rows += 1
        self.raw_supported += _phasor(raw_residual)
        self.corrected_subtract += _phasor(subtract_residual)
        self.corrected_add += _phasor(add_residual)


def _circular_stats(total: complex, count: int) -> tuple[float, float, float]:
    if count <= 0:
        raise ValueError("cannot summarize an empty circular sample")
    resultant = min(max(abs(total) / count, 1e-12), 1.0)
    return (
        math.atan2(total.imag, total.real),
        resultant,
        math.sqrt(-2.0 * math.log(resultant)),
    )


def _across_capture_bias(
    capture_rows: list[dict[str, Any]], key: str
) -> tuple[float, float]:
    total = sum(_phasor(float(row[key])) for row in capture_rows)
    return math.atan2(total.imag, total.real), abs(total) / len(capture_rows)


def _sign_z(improved: int, total: int) -> float:
    return (improved - total / 2.0) / math.sqrt(total / 4.0)


def validate_snapshot_export(
    *,
    csv_gz_path: Path,
    receiver: str,
    serial: str,
    frequency_hz: int,
    model_name: str = "complete_2p4_shared_gain_lut_per_radio",
    registry_root: Path | None = None,
    meta_path: Path | None = None,
) -> dict[str, Any]:
    """Stream one receiver's rows and independently recompute validation metrics."""

    csv_gz_path = Path(csv_gz_path).resolve()
    model = load_model(model_name, serial, registry_root=registry_root)
    prediction_cache: dict[tuple[int, int], float | None] = {}
    captures: dict[str, _Capture] = {}
    selected_rows = 0
    serial_present_rows = 0
    serial_blank_rows = 0
    maximum_theta_error = 0.0
    maximum_phase_error = 0.0
    geometry_rows = 0

    with gzip.open(csv_gz_path, "rt", newline="") as source:
        rows = csv.DictReader(source)
        if rows.fieldnames is None:
            raise ValueError("snapshot export has no header")
        missing = REQUIRED_COLUMNS - set(rows.fieldnames)
        if missing:
            raise ValueError(f"snapshot export is missing columns: {sorted(missing)}")
        for row in rows:
            if row["receiver"] != receiver:
                continue
            selected_rows += 1
            capture = captures.setdefault(row["capture"], _Capture())
            if int(row["lo_hz"]) != int(frequency_hz):
                raise ValueError(
                    f"{row['capture']}: LO {row['lo_hz']} does not match "
                    f"{frequency_hz}"
                )
            recorded_serial = row["serial"].strip()
            if recorded_serial:
                serial_present_rows += 1
                if recorded_serial != serial:
                    raise ValueError(
                        f"{row['capture']}: serial {recorded_serial} does not "
                        f"match asserted serial {serial}"
                    )
            else:
                serial_blank_rows += 1

            gain1_db = int(row["gain_rx1_db"])
            gain2_db = int(row["gain_rx2_db"])
            measured = float(row["phase_meas_rad"])
            ground_truth = float(row["phase_gt_rad"])
            raw_residual = _wrap(measured - ground_truth)
            capture.add_raw(raw_residual, gain1_db)

            gain_pair = (gain1_db, gain2_db)
            if gain_pair not in prediction_cache:
                try:
                    prediction_cache[gain_pair] = model.predict_phase_offset(
                        frequency_hz=frequency_hz,
                        gain_rx1_db=gain1_db,
                        gain_rx2_db=gain2_db,
                    )
                except UnsupportedPhaseModelInput:
                    prediction_cache[gain_pair] = None
            prediction = prediction_cache[gain_pair]
            if prediction is not None:
                capture.add_supported(
                    raw_residual,
                    _wrap(measured - prediction - ground_truth),
                    _wrap(measured + prediction - ground_truth),
                )

            tx_x = float(row["tx_pos_x_mm"])
            tx_y = float(row["tx_pos_y_mm"])
            rx_x = float(row["rx_pos_x_mm"])
            rx_y = float(row["rx_pos_y_mm"])
            if abs(tx_x - rx_x) + abs(tx_y - rx_y) > 0.0:
                theta = _wrap(
                    math.atan2(tx_x - rx_x, tx_y - rx_y)
                    - (float(row["rx_theta_in_pis"]) + float(row["rx_heading_in_pis"]))
                    * math.pi
                )
                stored_theta = float(row["theta_gt_rad"])
                maximum_theta_error = max(
                    maximum_theta_error, abs(_wrap(theta - stored_theta))
                )
                phase = _wrap(
                    -math.sin(theta) * float(row["d_over_lambda"]) * 2.0 * math.pi
                )
                maximum_phase_error = max(
                    maximum_phase_error, abs(_wrap(phase - ground_truth))
                )
                geometry_rows += 1

    if not selected_rows:
        raise ValueError(f"snapshot export has no rows for {receiver}")

    capture_metrics = []
    for capture_name, capture in captures.items():
        if capture.supported_rows != capture.rows:
            continue
        raw_bias, raw_resultant, raw_circstd = _circular_stats(
            capture.raw_all, capture.rows
        )
        subtract_bias, subtract_resultant, subtract_circstd = _circular_stats(
            capture.corrected_subtract, capture.supported_rows
        )
        add_bias, add_resultant, add_circstd = _circular_stats(
            capture.corrected_add, capture.supported_rows
        )
        gain1_mean = capture.gain1_sum / capture.rows
        gain1_std = math.sqrt(
            max(
                0.0,
                capture.gain1_square_sum / capture.rows - gain1_mean * gain1_mean,
            )
        )
        capture_metrics.append(
            {
                "capture": capture_name,
                "rows": capture.rows,
                "gain_rx1_std_db": gain1_std,
                "raw_bias_rad": raw_bias,
                "raw_resultant": raw_resultant,
                "raw_circstd_rad": raw_circstd,
                "subtract_bias_rad": subtract_bias,
                "subtract_resultant": subtract_resultant,
                "subtract_circstd_rad": subtract_circstd,
                "add_bias_rad": add_bias,
                "add_resultant": add_resultant,
                "add_circstd_rad": add_circstd,
            }
        )
    if not capture_metrics:
        raise ValueError("no capture has complete model coverage")

    raw_bias, raw_resultant = _across_capture_bias(capture_metrics, "raw_bias_rad")
    subtract_bias, subtract_resultant = _across_capture_bias(
        capture_metrics, "subtract_bias_rad"
    )
    add_bias, add_resultant = _across_capture_bias(capture_metrics, "add_bias_rad")
    subtract_std_improved = sum(
        row["subtract_circstd_rad"] < row["raw_circstd_rad"] for row in capture_metrics
    )
    add_std_improved = sum(
        row["add_circstd_rad"] < row["raw_circstd_rad"] for row in capture_metrics
    )
    subtract_bias_improved = sum(
        abs(row["subtract_bias_rad"]) < abs(row["raw_bias_rad"])
        for row in capture_metrics
    )
    add_bias_improved = sum(
        abs(row["add_bias_rad"]) < abs(row["raw_bias_rad"]) for row in capture_metrics
    )
    centered_subtract_bias = np.asarray(
        [_wrap(row["subtract_bias_rad"] - subtract_bias) for row in capture_metrics]
    )
    dose_response = []
    for lower, upper in ((0.01, 3.0), (3.0, 7.0), (7.0, 99.0)):
        selected = [
            row for row in capture_metrics if lower < row["gain_rx1_std_db"] <= upper
        ]
        deltas = np.asarray(
            [row["subtract_circstd_rad"] - row["raw_circstd_rad"] for row in selected]
        )
        dose_response.append(
            {
                "gain_rx1_std_db_gt": lower,
                "gain_rx1_std_db_lte": upper,
                "captures": len(selected),
                "median_delta_circstd_rad": (
                    float(np.median(deltas)) if len(deltas) else None
                ),
                "improved_captures": int(np.sum(deltas < 0.0)),
            }
        )

    supported_rows = sum(capture.supported_rows for capture in captures.values())
    result = {
        "schema": "spf.calibration.external_wall_validation",
        "schema_version": 1,
        "input": {
            "csv_gz_path": str(csv_gz_path),
            "csv_gz_sha256": _sha256(csv_gz_path),
            "meta_path": str(Path(meta_path).resolve()) if meta_path else None,
            "meta_sha256": _sha256(Path(meta_path)) if meta_path else None,
            "receiver": receiver,
            "asserted_serial": serial,
            "frequency_hz": int(frequency_hz),
            "model_name": model_name,
            "model_path": str(model.path),
            "model_sha256": _sha256(model.path),
            "recorded_serial_rows": serial_present_rows,
            "blank_serial_rows": serial_blank_rows,
        },
        "integrity": {
            "selected_rows": selected_rows,
            "captures": len(captures),
            "supported_rows": supported_rows,
            "coverage_fraction": supported_rows / selected_rows,
            "minimum_capture_coverage_fraction": min(
                capture.supported_rows / capture.rows for capture in captures.values()
            ),
            "geometry_rows_rederived": geometry_rows,
            "maximum_theta_error_rad": maximum_theta_error,
            "maximum_phase_error_rad": maximum_phase_error,
        },
        "summary": {
            "fully_supported_captures": len(capture_metrics),
            "raw_bias_deg": math.degrees(raw_bias),
            "raw_bias_resultant": raw_resultant,
            "subtract_bias_deg": math.degrees(subtract_bias),
            "subtract_bias_resultant": subtract_resultant,
            "add_bias_deg": math.degrees(add_bias),
            "add_bias_resultant": add_resultant,
            "median_raw_circstd_rad": float(
                np.median([row["raw_circstd_rad"] for row in capture_metrics])
            ),
            "median_subtract_circstd_rad": float(
                np.median([row["subtract_circstd_rad"] for row in capture_metrics])
            ),
            "median_add_circstd_rad": float(
                np.median([row["add_circstd_rad"] for row in capture_metrics])
            ),
            "subtract_circstd_improved_captures": subtract_std_improved,
            "subtract_circstd_improved_sign_z": _sign_z(
                subtract_std_improved, len(capture_metrics)
            ),
            "add_circstd_improved_captures": add_std_improved,
            "add_circstd_improved_sign_z": _sign_z(
                add_std_improved, len(capture_metrics)
            ),
            "subtract_absolute_bias_improved_captures": subtract_bias_improved,
            "subtract_absolute_bias_improved_sign_z": _sign_z(
                subtract_bias_improved, len(capture_metrics)
            ),
            "add_absolute_bias_improved_captures": add_bias_improved,
            "add_absolute_bias_improved_sign_z": _sign_z(
                add_bias_improved, len(capture_metrics)
            ),
            "subtract_bias_iqr_deg": math.degrees(
                float(
                    np.percentile(centered_subtract_bias, 75)
                    - np.percentile(centered_subtract_bias, 25)
                )
            ),
            "dose_response": dose_response,
        },
        "capture_metrics": capture_metrics,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-gz", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--frequency-hz", type=int, required=True)
    parser.add_argument(
        "--model",
        default="complete_2p4_shared_gain_lut_per_radio",
    )
    parser.add_argument("--registry-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_snapshot_export(
        csv_gz_path=args.csv_gz,
        receiver=args.receiver,
        serial=args.serial,
        frequency_hz=args.frequency_hz,
        model_name=args.model,
        registry_root=args.registry_root,
        meta_path=args.meta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
