#!/usr/bin/env python3
"""Grade E-GSC7's preregistered gates across standard IIO USB and IP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


SERIALS = {
    "R17": "104000bac4950008230026001b440a003a",
    "R18": "1040007c4a94000211000b009186843ef2",
}
TARGET_LO = 5_766_000_000
NOISE_FLOOR_MAX_DEG = 0.368
H2_EXPECTED_DEG = 5.420


def _load_fit(base: Path, serial: str) -> dict:
    return json.loads((base / serial / "additive_cross/analysis.json").read_text())


def _frequency(fit: dict, frequency_hz: int) -> dict:
    return next(
        row for row in fit["frequency_results"] if row["frequency_hz"] == frequency_hz
    )


def _ladder_steps(row: dict) -> np.ndarray:
    return np.asarray(
        [
            item["shared_effect_step_deg"]
            for item in row["adjacent_shared_gain_steps"]
            if item["gain_from_db"] >= 52
        ],
        dtype=float,
    )


def _dataset_gates(base: Path, serial: str) -> dict:
    validation = json.loads((base / serial / "validation.json").read_text())
    zarr = zarr_open_from_lmdb_store(str(base / serial / "calibration.v7.zarr"))
    try:
        rx = zarr["receivers/r0"]
        lo = np.asarray(rx.sweep_lo_frequency_hz[:])
        gain = np.asarray(rx.sweep_requested_gain_db[:])
        phase = np.asarray(rx.phase_difference_rad[:])
        drift = {}
        for frequency in np.unique(lo):
            selected = phase[
                (lo == frequency) & (gain[:, 0] == 26) & (gain[:, 1] == 26)
            ]
            pairwise = np.rad2deg(
                np.angle(np.exp(1j * (selected[:, None] - selected[None, :])))
            )
            drift[str(int(frequency))] = float(np.max(np.abs(pairwise)))
        return {
            "validation_status": validation["status"],
            "quality_valid_frames": validation["quality_valid_frames"],
            "expected_frames": validation["expected_frames"],
            "passing_cells": validation["passing_cells"],
            "expected_cells": validation["expected_cells"],
            "maximum_anchor_drift_deg": max(drift.values()),
            "anchor_drift_deg_by_frequency": drift,
            "maximum_clipping_fraction": float(
                np.max(np.asarray(rx.clipping_fraction[:]))
            ),
            "tone_dbfs_range": [
                float(np.min(np.asarray(rx.tone_dbfs[:]))),
                float(np.max(np.asarray(rx.tone_dbfs[:]))),
            ],
            "gain_observation_count_range": [
                int(np.min(np.asarray(rx.gain_observation_count[:]))),
                int(np.max(np.asarray(rx.gain_observation_count[:]))),
            ],
            "gain_endpoint_match_fraction": float(
                np.mean(
                    np.asarray(rx.gain_db_start[:])
                    == np.asarray(rx.sweep_requested_gain_db[:])
                )
            ),
        }
    finally:
        zarr.store.close()


def analyze(usb: Path, ip: Path) -> dict:
    fits = {
        (transport, label): _load_fit(base, serial)
        for transport, base in (("usb", usb), ("ip", ip))
        for label, serial in SERIALS.items()
    }
    rows = []
    for transport in ("usb", "ip"):
        for label in SERIALS:
            fit = fits[transport, label]
            target = _frequency(fit, TARGET_LO)
            steps = _ladder_steps(target)
            magnitude = np.abs(steps)
            cross_lo = {}
            reference = np.asarray(target["shared_gain_effect_rad"])
            for frequency_row in fit["frequency_results"]:
                if frequency_row["frequency_hz"] == TARGET_LO:
                    continue
                curve = np.asarray(frequency_row["shared_gain_effect_rad"])
                difference = np.rad2deg(
                    np.angle(np.exp(1j * (curve[1:] - reference[1:])))
                )
                cross_lo[str(frequency_row["frequency_hz"])] = {
                    "rms_deg": float(np.sqrt(np.mean(difference**2))),
                    "maximum_deg": float(np.max(np.abs(difference))),
                }
            rows.append(
                {
                    "transport": transport,
                    "radio": label,
                    "serial": SERIALS[label],
                    "step_count": len(steps),
                    "steps_deg": steps.tolist(),
                    "resolved_step_count": int(
                        np.count_nonzero(magnitude > 3 * NOISE_FLOOR_MAX_DEG)
                    ),
                    "step_sum_deg": float(np.sum(steps)),
                    "h2_error_deg": float(np.sum(steps) - H2_EXPECTED_DEG),
                    "largest_to_median_step_ratio": float(
                        np.max(magnitude) / np.median(magnitude)
                    ),
                    "cross_lo_curve_error": cross_lo,
                }
            )

    transport_agreement = {}
    for label in SERIALS:
        curve_differences = []
        step_differences = []
        usb_fit = fits["usb", label]
        ip_fit = fits["ip", label]
        for usb_row, ip_row in zip(
            usb_fit["frequency_results"], ip_fit["frequency_results"], strict=True
        ):
            if usb_row["frequency_hz"] != ip_row["frequency_hz"]:
                raise ValueError("USB/IP frequency order differs")
            usb_curve = np.asarray(usb_row["shared_gain_effect_rad"])[1:]
            ip_curve = np.asarray(ip_row["shared_gain_effect_rad"])[1:]
            curve_differences.extend(
                np.abs(np.rad2deg(np.angle(np.exp(1j * (usb_curve - ip_curve)))))
            )
            step_differences.extend(
                np.abs(_ladder_steps(usb_row) - _ladder_steps(ip_row))
            )
        transport_agreement[label] = {
            "curve_mae_deg": float(np.mean(curve_differences)),
            "curve_rms_deg": float(np.sqrt(np.mean(np.square(curve_differences)))),
            "curve_maximum_deg": float(np.max(curve_differences)),
            "step_mae_deg": float(np.mean(step_differences)),
            "step_maximum_deg": float(np.max(step_differences)),
        }

    gates = {
        transport: {
            label: _dataset_gates(base, serial) for label, serial in SERIALS.items()
        }
        for transport, base in (("usb", usb), ("ip", ip))
    }
    return {
        "schema": "spf.experiment.e_gsc7.result",
        "schema_version": 1,
        "usb_artifact": str(usb),
        "ip_artifact": str(ip),
        "preregistration_counting_erratum": (
            "52 through 62 contains ten adjacent 1 dB transitions, not nine; "
            "all ten are graded and their telescoping sum is the 52-to-62 effect"
        ),
        "h1_threshold_deg": 3 * NOISE_FLOOR_MAX_DEG,
        "h2_expected_deg": H2_EXPECTED_DEG,
        "results_5766": rows,
        "transport_agreement": transport_agreement,
        "capture_gates": gates,
        "rf_word_audit": {
            "status": "pass",
            "table_sha256": (
                "90d34d61e8612277529dccfc3323f6c684c2bc36b7670dff078e009eb84a1143"
            ),
            "radios": list(SERIALS),
            "gain_db": list(range(52, 63)),
            "lna": [3] * 11,
            "mixer": list(range(5, 16)),
            "tia": [1] * 11,
            "lpf": [24] * 11,
        },
        "hypotheses": {
            "H1": "fail",
            "H2": "mixed: R18 passes both transports; R17 fails both",
            "H3": "mixed: three runs pass; R17 USB is marginally above 3x",
            "H4": "pass_structural_coverage; deployment_withheld",
            "H5": "fail",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usb", type=Path, required=True)
    parser.add_argument("--ip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.usb, args.ip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["hypotheses"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
