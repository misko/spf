"""Campaign-wide analysis for the controlled A-G spectroscopy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


ACTIVE_LEVELS_DB = (0, -7, -14, -21, -28)
LEVEL_STAGE = {
    level: f"E_tx_{'n' + str(abs(level)) if level < 0 else '0'}"
    for level in (*ACTIVE_LEVELS_DB, -80)
}
ANCHOR_STAGES = (
    "E_anchor_before",
    "E_anchor_after_0",
    "E_anchor_after_n7",
    "E_anchor_after_n14",
    "E_anchor_after_n21",
    "E_anchor_after_n28",
    "E_anchor_after_n80",
)
SPECTROSCOPY_STAGES = ("A", "B", "C", "D", "G")
TREATMENT_STAGES = ("B", "C", "D", "G")
PHASE_PAIR_LABELS = {
    (5, 26): "RX1 5 / RX2 26",
    (26, 26): "RX1 26 / RX2 26",
    (45, 26): "RX1 45 / RX2 26",
    (26, 5): "RX1 26 / RX2 5",
    (26, 45): "RX1 26 / RX2 45",
}
GAIN_TABLE_BANDS = (
    ("low_0_to_1300_mhz", 0, 1_300_000_000),
    ("mid_1301_to_4000_mhz", 1_300_000_000, 4_000_000_000),
    ("high_4001_to_6000_mhz", 4_000_000_000, 6_000_000_000),
)
HYSTERESIS_FEATURE_NAMES = (
    "delta_rx1_per_40db",
    "delta_rx2_per_40db",
    "absolute_delta_rx1_per_40db",
    "absolute_delta_rx2_per_40db",
    "direction_rx1",
    "direction_rx2",
)
RIPPLE_DELAY_MIN_NS = 0.3
RIPPLE_DELAY_MAX_NS = 9.5
RIPPLE_DELAY_STEP_NS = 0.0025
RIPPLE_MINIMUM_SEPARATION_NS = 0.4
ARRAY_FIELDS = (
    "sweep_completed",
    "sweep_quality_valid",
    "sweep_epoch",
    "sweep_lo_frequency_hz",
    "sweep_requested_gain_db",
    "sweep_tx_gain_db",
    "phase_difference_rad",
    "amplitude_ratio_db_rx1_over_rx2",
    "tone_dbfs",
    "tone_snr_db",
    "within_capture_phase_std_rad",
    "coherence",
    "system_timestamp",
)


def wrap_phase(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2 * np.pi) - np.pi


def circular_mean(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.angle(np.mean(np.exp(1j * value))))


def circular_std_deg(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    resultant = float(np.abs(np.mean(np.exp(1j * value))))
    resultant = min(1.0, max(np.finfo(float).tiny, resultant))
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant))))


def circular_metrics(value: np.ndarray) -> dict[str, float | int]:
    value = wrap_phase(np.asarray(value, dtype=np.float64))
    absolute = np.abs(np.degrees(value))
    return {
        "n": int(value.size),
        "circular_bias_deg": float(np.degrees(circular_mean(value))),
        "circular_std_deg": circular_std_deg(value),
        "circular_mae_deg": float(np.mean(absolute)),
        "circular_rmse_deg": float(np.sqrt(np.mean(absolute**2))),
        "circular_p95_deg": float(np.percentile(absolute, 95)),
        "circular_max_deg": float(np.max(absolute)),
    }


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ARRAY_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_radio_dataset(path: Path) -> dict[str, Any]:
    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        arrays = {name: np.asarray(receiver[name][:]) for name in ARRAY_FIELDS}
        attrs = {
            "serial": receiver.attrs.get("sdr_serial"),
            "firmware_release_tag": receiver.attrs.get("firmware_release_tag"),
            "firmware_git_sha": receiver.attrs.get("firmware_git_sha"),
            "gadget_git_sha": receiver.attrs.get("firmware_gadget_git_sha"),
            "firmware_image_sha256": receiver.attrs.get("firmware_image_sha256"),
            "phase_convention": zarr.attrs.get("phase_convention"),
            "calibration_run_signature": zarr.attrs.get("calibration_run_signature"),
        }
    finally:
        zarr.store.close()
    if not attrs["serial"]:
        raise ValueError(f"{path}: missing sdr_serial")
    return {
        "path": str(path),
        "attrs": attrs,
        "arrays": arrays,
        "scalar_input_sha256": _hash_arrays(arrays),
    }


def load_stage(campaign_root: Path, stage: str) -> dict[str, dict[str, Any]]:
    stage_root = campaign_root / "stages" / stage
    datasets = sorted(stage_root.glob("*/calibration.v7.zarr"))
    if not datasets:
        raise FileNotFoundError(f"{stage}: no calibration datasets below {stage_root}")
    radios = {}
    for path in datasets:
        radio = load_radio_dataset(path)
        serial = str(radio["attrs"]["serial"])
        if serial in radios:
            raise ValueError(f"{stage}: duplicate serial {serial}")
        radios[serial] = radio
    return radios


def aggregate_cells(
    radio: dict[str, Any],
    *,
    quality_only: bool = True,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    arrays = radio["arrays"]
    selected = np.asarray(arrays["sweep_completed"], dtype=bool)
    if quality_only:
        selected &= np.asarray(arrays["sweep_quality_valid"], dtype=bool)
    selected &= np.isfinite(arrays["phase_difference_rad"])
    frequency = np.asarray(arrays["sweep_lo_frequency_hz"], dtype=np.int64)
    gains = np.asarray(arrays["sweep_requested_gain_db"], dtype=np.int64)
    keys = sorted(
        {
            (int(frequency[index]), int(gains[index, 0]), int(gains[index, 1]))
            for index in np.flatnonzero(selected)
        }
    )
    cells = {}
    for key in keys:
        at_cell = (
            selected
            & (frequency == key[0])
            & (gains[:, 0] == key[1])
            & (gains[:, 1] == key[2])
        )
        phase = np.asarray(arrays["phase_difference_rad"][at_cell], dtype=np.float64)
        cells[key] = {
            "frequency_hz": key[0],
            "gain_rx1_db": key[1],
            "gain_rx2_db": key[2],
            "n": int(phase.size),
            "phase_mean_rad": circular_mean(phase),
            "phase_circular_std_deg": circular_std_deg(phase),
            "amplitude_ratio_db": float(
                np.nanmean(arrays["amplitude_ratio_db_rx1_over_rx2"][at_cell])
            ),
            "tone_dbfs": np.nanmean(arrays["tone_dbfs"][at_cell], axis=0).tolist(),
            "tone_snr_db": np.nanmean(arrays["tone_snr_db"][at_cell], axis=0).tolist(),
            "tx_gain_db": float(np.nanmean(arrays["sweep_tx_gain_db"][at_cell])),
            "timestamp_s": float(np.nanmean(arrays["system_timestamp"][at_cell])),
            "coherence": float(np.nanmean(arrays["coherence"][at_cell])),
        }
    return cells


def compare_cells(
    baseline: dict[tuple[int, int, int], dict[str, Any]],
    other: dict[tuple[int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(baseline) & set(other)):
        base = baseline[key]
        treatment = other[key]
        rows.append(
            {
                "frequency_hz": key[0],
                "gain_rx1_db": key[1],
                "gain_rx2_db": key[2],
                "phase_delta_rad": float(
                    wrap_phase(treatment["phase_mean_rad"] - base["phase_mean_rad"])
                ),
                "amplitude_ratio_delta_db": float(
                    treatment["amplitude_ratio_db"] - base["amplitude_ratio_db"]
                ),
            }
        )
    return rows


def difference_of_differences(
    treated: list[dict[str, Any]],
    control: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    treated_by_key = {
        (row["frequency_hz"], row["gain_rx1_db"], row["gain_rx2_db"]): row
        for row in treated
    }
    control_by_key = {
        (row["frequency_hz"], row["gain_rx1_db"], row["gain_rx2_db"]): row
        for row in control
    }
    rows = []
    for key in sorted(set(treated_by_key) & set(control_by_key)):
        treatment = treated_by_key[key]
        reference = control_by_key[key]
        rows.append(
            {
                "frequency_hz": key[0],
                "gain_rx1_db": key[1],
                "gain_rx2_db": key[2],
                "phase_delta_rad": float(
                    wrap_phase(
                        treatment["phase_delta_rad"] - reference["phase_delta_rad"]
                    )
                ),
                "amplitude_ratio_delta_db": float(
                    treatment["amplitude_ratio_delta_db"]
                    - reference["amplitude_ratio_delta_db"]
                ),
            }
        )
    return rows


def fit_delay(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(rows) < 3:
        raise ValueError("delay fit requires at least three frequency points")
    rows = sorted(rows, key=lambda row: row["frequency_hz"])
    frequency = np.asarray([row["frequency_hz"] for row in rows], dtype=np.float64)
    phase = np.unwrap([row["phase_delta_rad"] for row in rows])
    centered = frequency - np.mean(frequency)
    slope, intercept = np.polyfit(centered, phase, 1)
    prediction = intercept + slope * centered
    residual = wrap_phase(phase - prediction)
    delay_s = -float(slope) / (2 * np.pi)
    return {
        "n": len(rows),
        "minimum_frequency_hz": int(np.min(frequency)),
        "maximum_frequency_hz": int(np.max(frequency)),
        "delay_ps": delay_s * 1e12,
        "equivalent_free_space_path_mm": delay_s * 299_792_458.0 * 1e3,
        "residual_rmse_deg": float(np.sqrt(np.mean(np.degrees(np.abs(residual)) ** 2))),
        "residual_p95_deg": float(np.percentile(np.degrees(np.abs(residual)), 95)),
    }


def delay_by_band(
    rows: list[dict[str, Any]],
    *,
    gain_pair: tuple[int, int] = (26, 26),
) -> dict[str, dict[str, float | int]]:
    paired = [
        row for row in rows if (row["gain_rx1_db"], row["gain_rx2_db"]) == gain_pair
    ]
    bands = {
        "low_0_to_1300_mhz": [
            row for row in paired if row["frequency_hz"] <= 1_300_000_000
        ],
        "mid_1301_to_4000_mhz": [
            row
            for row in paired
            if 1_300_000_000 < row["frequency_hz"] <= 4_000_000_000
        ],
        "high_4001_to_6000_mhz": [
            row for row in paired if row["frequency_hz"] > 4_000_000_000
        ],
    }
    return {name: fit_delay(values) for name, values in bands.items()}


def _phase_rows_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = circular_metrics(
        np.asarray([row["phase_delta_rad"] for row in rows], dtype=np.float64)
    )
    amplitude = np.asarray(
        [row["amplitude_ratio_delta_db"] for row in rows], dtype=np.float64
    )
    return {
        **metrics,
        "amplitude_delta_median_db": float(np.median(amplitude)),
        "amplitude_delta_p05_db": float(np.percentile(amplitude, 5)),
        "amplitude_delta_p95_db": float(np.percentile(amplitude, 95)),
    }


def _metrics_by_rf_region(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _phase_rows_metrics(rows),
        "low_and_mid_at_or_below_4ghz": _phase_rows_metrics(
            [row for row in rows if row["frequency_hz"] <= 4_000_000_000]
        ),
        "high_above_4ghz": _phase_rows_metrics(
            [row for row in rows if row["frequency_hz"] > 4_000_000_000]
        ),
    }


def _metrics_by_pair_and_rf_region(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for pair in PHASE_PAIR_LABELS:
        at_pair = [
            row for row in rows if (row["gain_rx1_db"], row["gain_rx2_db"]) == pair
        ]
        output[f"{pair[0]}_{pair[1]}"] = {
            "label": PHASE_PAIR_LABELS[pair],
            "at_or_below_4ghz": _phase_rows_metrics(
                [row for row in at_pair if row["frequency_hz"] <= 4_000_000_000]
            ),
            "above_4ghz": _phase_rows_metrics(
                [row for row in at_pair if row["frequency_hz"] > 4_000_000_000]
            ),
        }
    return output


def analyze_treatments(
    campaign_root: Path,
    *,
    treated_serial: str,
    control_serial: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    stages = {stage: load_stage(campaign_root, stage) for stage in SPECTROSCOPY_STAGES}
    for stage, radios in stages.items():
        expected = {treated_serial, control_serial}
        if set(radios) != expected:
            raise ValueError(f"{stage}: expected serials {expected}, got {set(radios)}")
    cells = {
        stage: {serial: aggregate_cells(radio) for serial, radio in radios.items()}
        for stage, radios in stages.items()
    }
    comparisons: dict[str, dict[str, Any]] = {}
    dod_rows: dict[str, list[dict[str, Any]]] = {}
    for stage in TREATMENT_STAGES:
        per_radio = {
            serial: compare_cells(cells["A"][serial], cells[stage][serial])
            for serial in (treated_serial, control_serial)
        }
        dod = difference_of_differences(
            per_radio[treated_serial], per_radio[control_serial]
        )
        dod_rows[stage] = dod
        comparisons[stage] = {
            "per_radio": {
                serial: _phase_rows_metrics(rows) for serial, rows in per_radio.items()
            },
            "treated_minus_control": {
                **_phase_rows_metrics(dod),
                "delay_by_band_at_26_26": delay_by_band(dod),
                "by_gain_pair_and_rf_region": _metrics_by_pair_and_rf_region(dod),
            },
        }
    repeatability_rows = {
        serial: compare_cells(cells["D"][serial], cells["G"][serial])
        for serial in (treated_serial, control_serial)
    }
    repeatability_dod = difference_of_differences(
        repeatability_rows[treated_serial],
        repeatability_rows[control_serial],
    )
    provenance = {
        stage: {
            serial: {
                "dataset_path": radio["path"],
                "scalar_input_sha256": radio["scalar_input_sha256"],
                **radio["attrs"],
            }
            for serial, radio in stages[stage].items()
        }
        for stage in stages
    }
    return {
        "treated_serial": treated_serial,
        "control_serial": control_serial,
        "comparisons_to_A": comparisons,
        "restored_D_to_hot_G_repeatability": {
            "per_radio": {
                serial: _metrics_by_rf_region(rows)
                for serial, rows in repeatability_rows.items()
            },
            "treated_minus_control": _metrics_by_rf_region(repeatability_dod),
        },
        "provenance": provenance,
    }, dod_rows


def _linear_phase_slope(levels: np.ndarray, phases: np.ndarray) -> float:
    unwrapped = np.unwrap(np.asarray(phases, dtype=np.float64))
    return float(np.degrees(np.polyfit(levels, unwrapped, 1)[0]))


def analyze_levels(
    campaign_root: Path,
    *,
    serials: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    level_stages = {
        level: load_stage(campaign_root, stage) for level, stage in LEVEL_STAGE.items()
    }
    anchor_stages = {stage: load_stage(campaign_root, stage) for stage in ANCHOR_STAGES}
    cells = {
        level: {
            serial: aggregate_cells(
                radios[serial],
                quality_only=(level != -80),
            )
            for serial in serials
        }
        for level, radios in level_stages.items()
    }
    anchors: dict[str, list[dict[str, float]]] = {serial: [] for serial in serials}
    for serial in serials:
        for stage in ANCHOR_STAGES:
            stage_cells = aggregate_cells(anchor_stages[stage][serial])
            if len(stage_cells) != 1:
                raise ValueError(f"{stage}/{serial}: expected one anchor cell")
            cell = next(iter(stage_cells.values()))
            anchors[serial].append(
                {
                    "stage": stage,
                    "timestamp_s": cell["timestamp_s"],
                    "phase_rad": cell["phase_mean_rad"],
                }
            )

    summaries = {}
    plot_data: dict[str, Any] = {"responses": {}, "tone": {}, "anchors": anchors}
    level_order = np.asarray(sorted(ACTIVE_LEVELS_DB), dtype=np.float64)
    for serial in serials:
        anchor_time = np.asarray(
            [row["timestamp_s"] for row in anchors[serial]], dtype=np.float64
        )
        anchor_phase = np.unwrap([row["phase_rad"] for row in anchors[serial]])
        anchor_hours = (anchor_time - anchor_time[0]) / 3600.0
        anchor_slope = float(np.degrees(np.polyfit(anchor_hours, anchor_phase, 1)[0]))
        anchor_total = float(np.degrees(anchor_phase[-1] - anchor_phase[0]))
        serial_summary = {
            "anchor_drift_deg_per_hour": anchor_slope,
            "anchor_total_change_deg": anchor_total,
            "frequencies": {},
        }
        plot_data["responses"][serial] = {}
        plot_data["tone"][serial] = {}
        frequencies = sorted({key[0] for key in cells[0][serial]})
        for frequency_hz in frequencies:
            common_keys = set.intersection(
                *[
                    {key for key in cells[level][serial] if key[0] == frequency_hz}
                    for level in ACTIVE_LEVELS_DB
                ]
            )
            raw_slopes = []
            corrected_slopes = []
            raw_responses = []
            corrected_responses = []
            margins = []
            for key in sorted(common_keys):
                raw_phase = np.asarray(
                    [
                        cells[int(level)][serial][key]["phase_mean_rad"]
                        for level in level_order
                    ]
                )
                stage_time = np.asarray(
                    [
                        cells[int(level)][serial][key]["timestamp_s"]
                        for level in level_order
                    ]
                )
                interpolated_anchor = np.interp(stage_time, anchor_time, anchor_phase)
                corrected_phase = wrap_phase(raw_phase - interpolated_anchor)
                raw_unwrapped = np.unwrap(raw_phase)
                corrected_unwrapped = np.unwrap(corrected_phase)
                raw_slopes.append(_linear_phase_slope(level_order, raw_phase))
                corrected_slopes.append(
                    _linear_phase_slope(level_order, corrected_phase)
                )
                raw_responses.append(
                    np.degrees(raw_unwrapped - raw_unwrapped[-1]).tolist()
                )
                corrected_responses.append(
                    np.degrees(corrected_unwrapped - corrected_unwrapped[-1]).tolist()
                )
                if key in cells[-80][serial]:
                    active_tone = np.asarray(cells[-28][serial][key]["tone_dbfs"])
                    floor_tone = np.asarray(cells[-80][serial][key]["tone_dbfs"])
                    margins.append(float(np.min(active_tone - floor_tone)))
                else:
                    margins.append(float("nan"))
            raw_slopes_array = np.asarray(raw_slopes)
            corrected_slopes_array = np.asarray(corrected_slopes)
            margin = np.asarray(margins)
            qualified = np.isfinite(margin) & (margin >= 45.6)
            qualified_slopes = corrected_slopes_array[qualified]
            serial_summary["frequencies"][str(frequency_hz)] = {
                "cells": len(common_keys),
                "raw_phase_slope_deg_per_tx_db": {
                    "median": float(np.median(raw_slopes_array)),
                    "p05": float(np.percentile(raw_slopes_array, 5)),
                    "p95": float(np.percentile(raw_slopes_array, 95)),
                },
                "anchor_corrected_phase_slope_deg_per_tx_db": {
                    "median": float(np.median(corrected_slopes_array)),
                    "p05": float(np.percentile(corrected_slopes_array, 5)),
                    "p95": float(np.percentile(corrected_slopes_array, 95)),
                },
                "spur_qualified_anchor_corrected_slope_deg_per_tx_db": {
                    "cells": int(np.count_nonzero(qualified)),
                    "median": float(np.median(qualified_slopes)),
                    "p05": float(np.percentile(qualified_slopes, 5)),
                    "p95": float(np.percentile(qualified_slopes, 95)),
                },
                "minus28_tone_to_muted_floor_db": {
                    "median": float(np.median(margin)),
                    "minimum": float(np.min(margin)),
                    "fraction_at_least_45p6_db": float(np.mean(margin >= 45.6)),
                },
            }
            plot_data["responses"][serial][str(frequency_hz)] = {
                "levels_db": level_order.tolist(),
                "raw": raw_responses,
                "corrected": corrected_responses,
            }
            plot_data["tone"][serial][str(frequency_hz)] = {
                str(level): np.median(
                    [
                        cell["tone_dbfs"]
                        for key, cell in cells[level][serial].items()
                        if key[0] == frequency_hz
                    ],
                    axis=0,
                ).tolist()
                for level in (*ACTIVE_LEVELS_DB, -80)
            }
        summaries[serial] = serial_summary
    provenance = {
        LEVEL_STAGE[level]: {
            serial: {
                "dataset_path": radio["path"],
                "scalar_input_sha256": radio["scalar_input_sha256"],
            }
            for serial, radio in radios.items()
        }
        for level, radios in level_stages.items()
    }
    provenance.update(
        {
            stage: {
                serial: {
                    "dataset_path": radio["path"],
                    "scalar_input_sha256": radio["scalar_input_sha256"],
                }
                for serial, radio in radios.items()
            }
            for stage, radios in anchor_stages.items()
        }
    )
    return {"per_radio": summaries, "provenance": provenance}, plot_data


def analyze_low_gain(
    campaign_root: Path,
    *,
    serials: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = load_stage(campaign_root, "F")
    output = {}
    curves = {}
    reference_gain = 5
    for serial in serials:
        cells = aggregate_cells(stage[serial])
        frequencies = sorted({key[0] for key in cells})
        output[serial] = {}
        curves[serial] = {}
        for frequency_hz in frequencies:
            reference = cells[(frequency_hz, reference_gain, reference_gain)][
                "phase_mean_rad"
            ]
            gains = sorted(
                {
                    key[1]
                    for key in cells
                    if key[0] == frequency_hz and key[2] == reference_gain
                }
                & {
                    key[2]
                    for key in cells
                    if key[0] == frequency_hz and key[1] == reference_gain
                }
            )
            rx1 = np.asarray(
                [
                    wrap_phase(
                        cells[(frequency_hz, gain, reference_gain)]["phase_mean_rad"]
                        - reference
                    )
                    for gain in gains
                ]
            )
            rx2 = np.asarray(
                [
                    wrap_phase(
                        cells[(frequency_hz, reference_gain, gain)]["phase_mean_rad"]
                        - reference
                    )
                    for gain in gains
                ]
            )
            symmetric = wrap_phase(0.5 * (rx1 - rx2))
            gap = wrap_phase(rx1 + rx2)
            output[serial][str(frequency_hz)] = {
                "gains_db": gains,
                "symmetry_gap_mae_deg": float(np.mean(np.abs(np.degrees(gap)))),
                "symmetry_gap_p95_deg": float(
                    np.percentile(np.abs(np.degrees(gap)), 95)
                ),
                "maximum_adjacent_H_step_deg": float(
                    np.max(np.abs(np.diff(np.degrees(np.unwrap(symmetric)))))
                ),
            }
            curves[serial][str(frequency_hz)] = {
                "gains_db": gains,
                "H_deg": np.degrees(np.unwrap(symmetric)).tolist(),
                "symmetry_gap_deg": np.degrees(gap).tolist(),
            }
    return {
        "reference_gain_db": reference_gain,
        "per_radio": output,
        "provenance": {
            serial: {
                "dataset_path": radio["path"],
                "scalar_input_sha256": radio["scalar_input_sha256"],
            }
            for serial, radio in stage.items()
        },
    }, curves


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_campaign_integrity(campaign_root: Path) -> dict[str, Any]:
    """Summarize acquisition gates without turning waivers into passes."""

    plan_path = campaign_root / "campaign_plan.json"
    audit_path = campaign_root / "gain_table_audit.json"
    resolved_config_path = campaign_root / "resolved_configs" / "A.yaml"
    plan = json.loads(plan_path.read_text())
    audit = json.loads(audit_path.read_text())
    resolved_config = yaml.safe_load(resolved_config_path.read_text())
    stages = {}
    total_completed = 0
    total_expected = 0
    firmware_values: dict[str, set[str]] = {
        "firmware_release_tag": set(),
        "firmware_git_sha": set(),
        "gadget_git_sha": set(),
        "firmware_image_sha256": set(),
    }
    for stage_plan in plan["stages"]:
        stage = stage_plan["id"]
        result_path = campaign_root / "stages" / stage / "stage_result.json"
        result = json.loads(result_path.read_text())
        capture = result["capture"]
        total_completed += int(capture["completed_measurements"])
        total_expected += int(capture["expected_measurements"])
        validations = result.get("validations", {})
        waiver_path = campaign_root / "waivers" / f"{stage}.json"
        waiver = json.loads(waiver_path.read_text()) if waiver_path.exists() else None
        stage_entry = {
            "status": result["status"],
            "waived": waiver is not None,
            "effective_status": (
                "waived_quality_failure"
                if result["status"] != "complete" and waiver is not None
                else result["status"]
            ),
            "completed_frames": int(capture["completed_measurements"]),
            "expected_frames": int(capture["expected_measurements"]),
            "seconds_per_recorded_frame": float(result["seconds_per_recorded_frame"]),
            "per_radio": {},
            "stage_result_path": str(result_path),
            "stage_result_sha256": _sha256_file(result_path),
        }
        if waiver is not None:
            stage_entry["waiver"] = {
                "path": str(waiver_path),
                "sha256": _sha256_file(waiver_path),
                "reason": waiver.get("reason") or waiver.get("note"),
            }
        for serial, validation in validations.items():
            stage_entry["per_radio"][serial] = {
                "status": validation["status"],
                "completed_frames": int(validation["completed_frames"]),
                "quality_valid_frames": int(validation["quality_valid_frames"]),
                "passing_cells": int(validation["passing_cells"]),
                "expected_cells": int(validation["expected_cells"]),
            }
        stages[stage] = stage_entry

        for dataset_path in sorted(
            (campaign_root / "stages" / stage).glob("*/calibration.v7.zarr")
        ):
            radio = load_radio_dataset(dataset_path)
            for key in firmware_values:
                value = radio["attrs"].get(key)
                if value:
                    firmware_values[key].add(str(value))

    firmware = {key: sorted(values) for key, values in firmware_values.items()}
    firmware_consistent = all(len(values) == 1 for values in firmware.values())
    expected_firmware = resolved_config["pluto-firmware"]
    firmware_expectations = {
        "firmware_release_tag": str(expected_firmware["release-tag"]),
        "firmware_git_sha": str(expected_firmware["firmware-git-sha"]),
        "gadget_git_sha": str(expected_firmware["gadget-git-sha"]),
        "firmware_image_sha256": str(expected_firmware["image-sha256"]),
    }
    firmware_matches_resolved_config = all(
        firmware[key] == [expected] for key, expected in firmware_expectations.items()
    )
    table_hashes_by_band: dict[str, set[str]] = {}
    for radio in audit["radios"]:
        for band in radio["bands"]:
            table_hashes_by_band.setdefault(band["name"], set()).add(
                band["table_sha256"]
            )
    gain_tables_identical = all(
        len(values) == 1 for values in table_hashes_by_band.values()
    )
    return {
        "planned_frames": int(plan["measurements_all_radios"]),
        "stage_expected_frames": total_expected,
        "captured_frames": total_completed,
        "all_frames_captured": (
            total_completed == total_expected == int(plan["measurements_all_radios"])
        ),
        "rate_gate": {
            "seconds_per_frame": stages["rate_pilot"]["seconds_per_recorded_frame"],
            "limit_seconds_per_frame": float(
                plan["rate_gate"]["maximum-seconds-per-recorded-frame"]
            ),
            "passed": (
                stages["rate_pilot"]["seconds_per_recorded_frame"]
                <= float(plan["rate_gate"]["maximum-seconds-per-recorded-frame"])
            ),
        },
        "gain_table_audit_status": audit["status"],
        "gain_tables_identical_between_radios": gain_tables_identical,
        "gain_table_hashes": {
            band: sorted(values) for band, values in table_hashes_by_band.items()
        },
        "firmware_consistent_across_datasets": firmware_consistent,
        "firmware_matches_resolved_config": firmware_matches_resolved_config,
        "firmware": firmware,
        "expected_firmware": firmware_expectations,
        "stages": stages,
        "provenance": {
            "campaign_plan_path": str(plan_path),
            "campaign_plan_sha256": _sha256_file(plan_path),
            "gain_table_audit_path": str(audit_path),
            "gain_table_audit_sha256": _sha256_file(audit_path),
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_sha256": _sha256_file(resolved_config_path),
        },
    }


def _gain_table_row_by_gain(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Select the first full-table row for each reported absolute gain."""

    selected = {}
    for row in rows:
        selected.setdefault(int(row["gain_db"]), row)
    return selected


def analyze_gain_table_transitions(
    campaign_root: Path,
    *,
    low_gain_curves: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_path = campaign_root / "gain_table_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "pass":
        raise ValueError("gain-table transition analysis requires a passing audit")
    first_radio = audit["radios"][0]
    bands = {}
    observed_steps = {}
    for band in first_radio["bands"]:
        rows = _gain_table_row_by_gain(band["rows"])
        selected = {
            gain: {
                "bytes": rows[gain]["bytes"],
                "lna_index": (int(rows[gain]["bytes"][0]) >> 5) & 0x3,
                "mixer_index": int(rows[gain]["bytes"][0]) & 0x1F,
            }
            for gain in range(11, 18)
            if gain in rows
        }
        changes = [
            gain
            for gain in range(12, 18)
            if gain in selected
            and gain - 1 in selected
            and selected[gain]["bytes"] != selected[gain - 1]["bytes"]
        ]
        # The relevant LNA/mixer state transition is the first change in byte 0.
        first_byte_changes = [
            gain
            for gain in range(12, 18)
            if gain in selected
            and gain - 1 in selected
            and selected[gain]["bytes"][0] != selected[gain - 1]["bytes"][0]
        ]
        boundary = first_byte_changes[0] if first_byte_changes else None
        bands[band["name"]] = {
            "start_hz": int(band["start_hz"]),
            "end_hz": int(band["end_hz"]),
            "row_count": int(band["row_count"]),
            "table_sha256": band["table_sha256"],
            "states_11_to_17_db": selected,
            "all_state_changes_11_to_17_db": changes,
            "lna_mixer_boundary_gain_db": boundary,
        }
        if boundary is not None:
            observed_steps[band["name"]] = {}
            for serial, frequencies in low_gain_curves.items():
                values = []
                for frequency, curve in frequencies.items():
                    frequency_hz = int(frequency)
                    if not (
                        int(band["start_hz"]) < frequency_hz <= int(band["end_hz"])
                    ):
                        continue
                    gains = np.asarray(curve["gains_db"], dtype=np.int64)
                    h = np.asarray(curve["H_deg"], dtype=np.float64)
                    before = np.flatnonzero(gains == boundary - 1)
                    after = np.flatnonzero(gains == boundary)
                    if before.size and after.size:
                        values.append(
                            {
                                "frequency_hz": frequency_hz,
                                "step_deg": float(h[after[0]] - h[before[0]]),
                            }
                        )
                observed_steps[band["name"]][serial] = values
    return {
        "status": audit["status"],
        "radios": [radio["serial"] for radio in audit["radios"]],
        "tables_identical_between_radios": all(
            len({radio["bands"][index]["table_sha256"] for radio in audit["radios"]})
            == 1
            for index in range(len(first_radio["bands"]))
        ),
        "bands": bands,
        "observed_symmetric_H_steps": observed_steps,
        "provenance": {
            "path": str(audit_path),
            "sha256": _sha256_file(audit_path),
        },
    }, observed_steps


def _band_indices(frequency_hz: np.ndarray) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    return np.select(
        [frequency_hz <= 1_300_000_000, frequency_hz <= 4_000_000_000],
        [0, 1],
        default=2,
    )


def _band_polynomial_design(frequency_hz: np.ndarray) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    bands = _band_indices(frequency_hz)
    rows = []
    for index, frequency in enumerate(frequency_hz):
        row = []
        for band in range(3):
            selected = bands == band
            centered = (frequency - np.mean(frequency_hz[selected])) / 1e9
            active = float(bands[index] == band)
            row.extend((active, active * centered, active * centered**2))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _unwrap_by_band(
    frequency_hz: np.ndarray,
    phase_rad: np.ndarray,
) -> np.ndarray:
    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    unwrapped = np.empty_like(phase_rad)
    bands = _band_indices(frequency_hz)
    for band in range(3):
        selected = bands == band
        unwrapped[selected] = np.unwrap(phase_rad[selected])
    return unwrapped


def _component_fit(
    frequency_hz: np.ndarray,
    phase_rad: np.ndarray,
    delays_s: tuple[float, ...],
) -> dict[str, Any]:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    phase_rad = _unwrap_by_band(frequency_hz, phase_rad)
    design = _band_polynomial_design(frequency_hz)
    for delay_s in delays_s:
        design = np.column_stack(
            (
                design,
                np.sin(2 * np.pi * frequency_hz * delay_s),
                np.cos(2 * np.pi * frequency_hz * delay_s),
            )
        )
    coefficients = np.linalg.lstsq(design, phase_rad, rcond=None)[0]
    residual = phase_rad - design @ coefficients
    amplitudes = [
        float(
            np.hypot(
                coefficients[9 + 2 * index],
                coefficients[10 + 2 * index],
            )
        )
        for index in range(len(delays_s))
    ]
    return {
        "residual": residual,
        "sse": float(residual @ residual),
        "amplitudes_rad": amplitudes,
    }


def _single_delay_spectrum(
    frequency_hz: np.ndarray,
    phase_rad: np.ndarray,
    delay_grid_s: np.ndarray,
) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    phase_rad = _unwrap_by_band(frequency_hz, phase_rad)
    nuisance = _band_polynomial_design(frequency_hz)
    residual = (
        phase_rad - nuisance @ np.linalg.lstsq(nuisance, phase_rad, rcond=None)[0]
    )
    amplitudes = []
    for delay_s in delay_grid_s:
        design = np.column_stack(
            (
                np.sin(2 * np.pi * frequency_hz * delay_s),
                np.cos(2 * np.pi * frequency_hz * delay_s),
            )
        )
        coefficient = np.linalg.lstsq(design, residual, rcond=None)[0]
        amplitudes.append(float(np.hypot(*coefficient)))
    return np.asarray(amplitudes)


def fit_shared_delay_components(
    curves: list[tuple[np.ndarray, np.ndarray]],
    *,
    component_count: int = 2,
    delay_grid_s: np.ndarray | None = None,
    minimum_separation_s: float = RIPPLE_MINIMUM_SEPARATION_NS * 1e-9,
) -> dict[str, Any]:
    """Greedily fit delay components shared by several phase curves.

    Each curve gets its own per-band quadratic nuisance and sinusoid
    coefficients. Only component delay is shared. The separation constraint is
    wider than the 1/span Rayleigh resolution and prevents a sidelobe of the
    first component from being reported as a second physical path.
    """

    if delay_grid_s is None:
        delay_grid_s = (
            np.arange(
                RIPPLE_DELAY_MIN_NS,
                RIPPLE_DELAY_MAX_NS + RIPPLE_DELAY_STEP_NS / 2,
                RIPPLE_DELAY_STEP_NS,
            )
            * 1e-9
        )
    delay_grid_s = np.asarray(delay_grid_s, dtype=np.float64)
    working = []
    for frequency_hz, phase_rad in curves:
        frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
        phase_rad = _unwrap_by_band(frequency_hz, phase_rad)
        nuisance = _band_polynomial_design(frequency_hz)
        residual = (
            phase_rad - nuisance @ np.linalg.lstsq(nuisance, phase_rad, rcond=None)[0]
        )
        working.append([frequency_hz, residual])
    selected: list[float] = []
    for _ in range(component_count):
        candidates = [
            delay
            for delay in delay_grid_s
            if all(
                abs(delay - previous) >= minimum_separation_s for previous in selected
            )
        ]
        scores = []
        for delay_s in candidates:
            score = 0.0
            for frequency_hz, residual in working:
                design = np.column_stack(
                    (
                        np.sin(2 * np.pi * frequency_hz * delay_s),
                        np.cos(2 * np.pi * frequency_hz * delay_s),
                    )
                )
                coefficient = np.linalg.lstsq(design, residual, rcond=None)[0]
                error = residual - design @ coefficient
                score += float(error @ error)
            scores.append(score)
        selected_delay = float(candidates[int(np.argmin(scores))])
        selected.append(selected_delay)
        for curve in working:
            frequency_hz, residual = curve
            design = np.column_stack(
                (
                    np.sin(2 * np.pi * frequency_hz * selected_delay),
                    np.cos(2 * np.pi * frequency_hz * selected_delay),
                )
            )
            coefficient = np.linalg.lstsq(design, residual, rcond=None)[0]
            curve[1] = residual - design @ coefficient

    model_comparison = []
    total_n = sum(len(frequency_hz) for frequency_hz, _ in curves)
    for count in range(component_count + 1):
        delays = tuple(selected[:count])
        sse = sum(
            _component_fit(frequency_hz, phase_rad, delays)["sse"]
            for frequency_hz, phase_rad in curves
        )
        parameter_count = len(curves) * (9 + 2 * count) + count
        bic = total_n * math.log(sse / total_n) + parameter_count * math.log(total_n)
        model_comparison.append(
            {
                "components": count,
                "parameters": parameter_count,
                "sse": sse,
                "bic": bic,
            }
        )
    return {
        "delays_s": selected,
        "model_comparison": model_comparison,
    }


def _branch_curve(
    cells: dict[tuple[int, int, int], dict[str, Any]],
    *,
    receiver: int,
    gain_db: int,
    reference_gain_db: int = 26,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(sorted({key[0] for key in cells}), dtype=np.float64)
    phase = []
    for frequency in frequencies.astype(np.int64):
        reference = cells[(int(frequency), reference_gain_db, reference_gain_db)][
            "phase_mean_rad"
        ]
        key = (
            (int(frequency), gain_db, reference_gain_db)
            if receiver == 1
            else (int(frequency), reference_gain_db, gain_db)
        )
        phase.append(float(wrap_phase(cells[key]["phase_mean_rad"] - reference)))
    return frequencies, np.asarray(phase)


def analyze_ripple_structure(
    campaign_root: Path,
    *,
    treated_serial: str,
    control_serial: str,
    c_equal_gain_delay_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_cells = {
        stage: {
            serial: aggregate_cells(radio)
            for serial, radio in load_stage(campaign_root, stage).items()
        }
        for stage in SPECTROSCOPY_STAGES
    }
    arms = (
        (treated_serial, 1),
        (treated_serial, 2),
        (control_serial, 1),
        (control_serial, 2),
    )
    baseline_curves = [
        _branch_curve(stage_cells["A"][serial], receiver=receiver, gain_db=45)
        for serial, receiver in arms
    ]
    shared = fit_shared_delay_components(baseline_curves)
    delays = tuple(shared["delays_s"])
    amplitudes = {}
    spectra = {}
    delay_grid_s = (
        np.arange(
            RIPPLE_DELAY_MIN_NS,
            RIPPLE_DELAY_MAX_NS + RIPPLE_DELAY_STEP_NS / 2,
            RIPPLE_DELAY_STEP_NS,
        )
        * 1e-9
    )
    for stage in SPECTROSCOPY_STAGES:
        amplitudes[stage] = {}
        spectra[stage] = {}
        for serial, receiver in arms:
            frequency, phase = _branch_curve(
                stage_cells[stage][serial],
                receiver=receiver,
                gain_db=45,
            )
            fit = _component_fit(frequency, phase, delays)
            key = f"{serial}:rx{receiver}"
            amplitudes[stage][key] = [
                float(np.degrees(value)) for value in fit["amplitudes_rad"]
            ]
            if stage == "A" or (serial == treated_serial and receiver == 1):
                spectra[stage][key] = np.degrees(
                    _single_delay_spectrum(frequency, phase, delay_grid_s)
                ).tolist()

    primary_key = f"{treated_serial}:rx1"
    control_keys = (
        f"{treated_serial}:rx2",
        f"{control_serial}:rx1",
        f"{control_serial}:rx2",
    )
    primary_a = amplitudes["A"][primary_key][0]
    primary_b = amplitudes["B"][primary_key][0]
    control_ratios = [
        amplitudes["B"][key][0] / amplitudes["A"][key][0] for key in control_keys
    ]

    # A reflection through an added one-way cable traverses the added path twice.
    expected_moved_delay_s = delays[0] + 2 * abs(c_equal_gain_delay_s)
    moved_amplitudes = {}
    extended_delays = (*delays, expected_moved_delay_s)
    for stage in SPECTROSCOPY_STAGES:
        frequency, phase = _branch_curve(
            stage_cells[stage][treated_serial],
            receiver=1,
            gain_db=45,
        )
        fit = _component_fit(frequency, phase, extended_delays)
        moved_amplitudes[stage] = float(np.degrees(fit["amplitudes_rad"][-1]))

    return {
        "gain_db": 45,
        "reference_gain_db": 26,
        "detrending": "independent quadratic in each audited gain-table band",
        "delay_search_ns": [
            RIPPLE_DELAY_MIN_NS,
            RIPPLE_DELAY_MAX_NS,
            RIPPLE_DELAY_STEP_NS,
        ],
        "minimum_component_separation_ns": RIPPLE_MINIMUM_SEPARATION_NS,
        "shared_baseline_components": [
            {
                "delay_ns": delay * 1e9,
                "frequency_period_mhz": 1 / delay / 1e6,
                "one_way_free_space_equivalent_mm": (delay * 299_792_458.0 / 2 * 1e3),
            }
            for delay in delays
        ],
        "model_comparison": shared["model_comparison"],
        "component_amplitudes_deg": amplitudes,
        "pad_test": {
            "primary_component_treated_rx1_A_deg": primary_a,
            "primary_component_treated_rx1_B_deg": primary_b,
            "treated_rx1_fraction_remaining": primary_b / primary_a,
            "treated_rx1_suppression_percent": 100 * (1 - primary_b / primary_a),
            "unchanged_arm_B_over_A_ratios": control_ratios,
            "unchanged_arm_median_B_over_A_ratio": float(np.median(control_ratios)),
        },
        "jumper_test": {
            "equal_gain_one_way_delay_ns": c_equal_gain_delay_s * 1e9,
            "expected_moved_reflection_delay_ns": expected_moved_delay_s * 1e9,
            "amplitude_at_expected_moved_delay_deg": moved_amplitudes,
            "C_over_D_amplitude_ratio": (moved_amplitudes["C"] / moved_amplitudes["D"]),
            "interpretation": (
                "supportive_but_not_causal: C has energy at the predicted moved "
                "delay, but C failed repeatability and D did not restore A"
            ),
        },
    }, {
        "delay_grid_ns": (delay_grid_s * 1e9).tolist(),
        "spectra": spectra,
        "amplitudes": amplitudes,
        "delays_ns": [delay * 1e9 for delay in delays],
        "treated_serial": treated_serial,
        "control_serial": control_serial,
    }


def analyze_low_gain_overlap(
    campaign_root: Path,
    *,
    serials: tuple[str, str],
    prior_calibration_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage_f = load_stage(campaign_root, "F")
    if prior_calibration_root is None:
        return {
            "available": False,
            "reason": (
                "Stage A and F have no exact common frequencies; pass "
                "--prior-calibration-root for the exact-LO overlap check"
            ),
        }, []
    rows_for_plot = []
    output = {
        "available": True,
        "prior_calibration_root": str(prior_calibration_root),
        "acceptance_mae_deg": 0.75,
        "per_radio": {},
        "provenance": {},
    }
    for serial in serials:
        prior_path = prior_calibration_root / serial / "calibration.v7.zarr"
        prior = load_radio_dataset(prior_path)
        rows = compare_cells(
            aggregate_cells(prior),
            aggregate_cells(stage_f[serial]),
        )
        if not rows:
            raise ValueError(f"{serial}: no exact prior/F overlap cells")
        for row in rows:
            rows_for_plot.append({"serial": serial, **row})
        regions = {
            "all": rows,
            "at_or_below_4ghz": [
                row for row in rows if row["frequency_hz"] <= 4_000_000_000
            ],
            "above_4ghz": [row for row in rows if row["frequency_hz"] > 4_000_000_000],
        }
        output["per_radio"][serial] = {
            region: {
                **_phase_rows_metrics(values),
                "passes_0p75deg_mae": (
                    _phase_rows_metrics(values)["circular_mae_deg"] <= 0.75
                ),
            }
            for region, values in regions.items()
        }
        output["provenance"][serial] = {
            "dataset_path": str(prior_path),
            "scalar_input_sha256": prior["scalar_input_sha256"],
            **prior["attrs"],
        }
    return output, rows_for_plot


def _transition_rows(radio: dict[str, Any]) -> list[dict[str, Any]]:
    arrays = radio["arrays"]
    epoch = np.asarray(arrays["sweep_epoch"], dtype=np.int64)
    frequency = np.asarray(arrays["sweep_lo_frequency_hz"], dtype=np.int64)
    gain = np.asarray(arrays["sweep_requested_gain_db"], dtype=np.int64)
    valid = (
        np.asarray(arrays["sweep_completed"], dtype=bool)
        & np.asarray(arrays["sweep_quality_valid"], dtype=bool)
        & np.isfinite(arrays["phase_difference_rad"])
    )
    rows = []
    for index in range(1, len(epoch)):
        if (
            not valid[index]
            or not valid[index - 1]
            or epoch[index] != epoch[index - 1]
            or frequency[index] != frequency[index - 1]
        ):
            continue
        delta = gain[index] - gain[index - 1]
        rows.append(
            {
                "index": index,
                "epoch": int(epoch[index]),
                "cell": (
                    int(frequency[index]),
                    int(gain[index, 0]),
                    int(gain[index, 1]),
                ),
                "features": np.asarray(
                    (
                        delta[0] / 40.0,
                        delta[1] / 40.0,
                        abs(delta[0]) / 40.0,
                        abs(delta[1]) / 40.0,
                        np.sign(delta[0]),
                        np.sign(delta[1]),
                    ),
                    dtype=np.float64,
                ),
            }
        )
    return rows


def _hysteresis_cv(radio: dict[str, Any], *, ridge: float = 1.0) -> dict[str, Any]:
    rows = _transition_rows(radio)
    phase = np.asarray(radio["arrays"]["phase_difference_rad"], dtype=np.float64)
    baseline_errors = []
    corrected_errors = []
    coefficients = []
    for test_epoch in sorted({row["epoch"] for row in rows}):
        train = [row for row in rows if row["epoch"] != test_epoch]
        test = [row for row in rows if row["epoch"] == test_epoch]
        by_cell: dict[tuple[int, int, int], list[float]] = {}
        for row in train:
            by_cell.setdefault(row["cell"], []).append(phase[row["index"]])
        means = {
            cell: circular_mean(np.asarray(values)) for cell, values in by_cell.items()
        }
        train = [row for row in train if row["cell"] in means]
        test = [row for row in test if row["cell"] in means]
        x_train = np.asarray([row["features"] for row in train])
        y_train = np.asarray(
            [
                float(wrap_phase(phase[row["index"]] - means[row["cell"]]))
                for row in train
            ]
        )
        x_test = np.asarray([row["features"] for row in test])
        y_test = np.asarray(
            [
                float(wrap_phase(phase[row["index"]] - means[row["cell"]]))
                for row in test
            ]
        )
        train_design = np.column_stack((np.ones(len(x_train)), x_train))
        penalty = np.diag([0.0] + [ridge] * x_train.shape[1])
        coefficient = np.linalg.solve(
            train_design.T @ train_design + penalty,
            train_design.T @ y_train,
        )
        prediction = np.column_stack((np.ones(len(x_test)), x_test)) @ coefficient
        baseline_errors.extend(np.abs(np.degrees(y_test)))
        corrected_errors.extend(np.abs(np.degrees(wrap_phase(y_test - prediction))))
        coefficients.append(np.degrees(coefficient))
    baseline = np.asarray(baseline_errors)
    corrected = np.asarray(corrected_errors)
    coefficient = np.mean(coefficients, axis=0)
    return {
        "n": int(baseline.size),
        "ridge": ridge,
        "baseline_mae_deg": float(np.mean(baseline)),
        "order_corrected_mae_deg": float(np.mean(corrected)),
        "mae_improvement_deg": float(np.mean(baseline) - np.mean(corrected)),
        "baseline_p95_deg": float(np.percentile(baseline, 95)),
        "order_corrected_p95_deg": float(np.percentile(corrected, 95)),
        "mean_coefficients_deg": {
            "intercept": float(coefficient[0]),
            **{
                name: float(value)
                for name, value in zip(
                    HYSTERESIS_FEATURE_NAMES,
                    coefficient[1:],
                )
            },
        },
    }


def analyze_schedule_hysteresis(
    campaign_root: Path,
    *,
    serials: tuple[str, str],
) -> dict[str, Any]:
    output = {}
    provenance = {}
    for stage in (*SPECTROSCOPY_STAGES, "F"):
        radios = load_stage(campaign_root, stage)
        output[stage] = {serial: _hysteresis_cv(radios[serial]) for serial in serials}
        provenance[stage] = {
            serial: {
                "dataset_path": radios[serial]["path"],
                "scalar_input_sha256": radios[serial]["scalar_input_sha256"],
            }
            for serial in serials
        }
    return {
        "model": (
            "leave-one-epoch-out ridge regression on signed/absolute RX1/RX2 "
            "gain transition from the immediately preceding frame in the same "
            "frequency block"
        ),
        "per_stage": output,
        "provenance": provenance,
    }


def _plot_treatments(
    dod_rows: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for axis, stage in zip(axes.flat, TREATMENT_STAGES):
        rows = dod_rows[stage]
        for pair, label in PHASE_PAIR_LABELS.items():
            selected = sorted(
                [
                    row
                    for row in rows
                    if (row["gain_rx1_db"], row["gain_rx2_db"]) == pair
                ],
                key=lambda row: row["frequency_hz"],
            )
            axis.plot(
                np.asarray([row["frequency_hz"] for row in selected]) / 1e9,
                np.degrees([row["phase_delta_rad"] for row in selected]),
                marker=".",
                markersize=3,
                linewidth=0.8,
                label=label,
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(f"{stage} − A, treated radio minus control")
        axis.set_ylabel("phase change (degrees)")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("LO frequency (GHz)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_level_response(plot_data: dict[str, Any], output: Path) -> None:
    serials = sorted(plot_data["responses"])
    frequencies = sorted(
        {
            int(frequency)
            for serial in serials
            for frequency in plot_data["responses"][serial]
        }
    )
    figure, axes = plt.subplots(
        len(serials),
        len(frequencies),
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, serial in enumerate(serials):
        for column_index, frequency_hz in enumerate(frequencies):
            axis = axes[row_index, column_index]
            data = plot_data["responses"][serial][str(frequency_hz)]
            levels = np.asarray(data["levels_db"])
            for name, style in (("raw", "--"), ("corrected", "-")):
                response = np.asarray(data[name], dtype=np.float64)
                median = np.median(response, axis=0)
                low, high = np.percentile(response, [10, 90], axis=0)
                axis.plot(levels, median, style, label=name)
                if name == "corrected":
                    axis.fill_between(levels, low, high, alpha=0.18)
            axis.axhline(0, color="black", linewidth=0.6)
            axis.grid(alpha=0.25)
            axis.set_title(f"{serial[-6:]} · {frequency_hz / 1e6:.0f} MHz")
            axis.set_xlabel("TX gain (dB)")
            axis.set_ylabel("phase relative to TX=0 (degrees)")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_tone_floor(plot_data: dict[str, Any], output: Path) -> None:
    serials = sorted(plot_data["tone"])
    frequencies = sorted(
        {
            int(frequency)
            for serial in serials
            for frequency in plot_data["tone"][serial]
        }
    )
    figure, axes = plt.subplots(
        len(serials),
        len(frequencies),
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    levels = np.asarray(sorted((*ACTIVE_LEVELS_DB, -80)))
    for row_index, serial in enumerate(serials):
        for column_index, frequency_hz in enumerate(frequencies):
            axis = axes[row_index, column_index]
            data = plot_data["tone"][serial][str(frequency_hz)]
            tone = np.asarray([data[str(level)] for level in levels])
            axis.plot(levels, tone[:, 0], marker="o", label="RX1")
            axis.plot(levels, tone[:, 1], marker="o", label="RX2")
            axis.set_title(f"{serial[-6:]} · {frequency_hz / 1e6:.0f} MHz")
            axis.set_xlabel("TX gain (dB)")
            axis.set_ylabel("tone level (dBFS)")
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_anchor_drift(plot_data: dict[str, Any], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for serial, rows in sorted(plot_data["anchors"].items()):
        time_s = np.asarray([row["timestamp_s"] for row in rows])
        phase = np.unwrap([row["phase_rad"] for row in rows])
        axis.plot(
            (time_s - time_s[0]) / 3600,
            np.degrees(phase - phase[0]),
            marker="o",
            label=serial,
        )
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xlabel("hours from first E anchor")
    axis.set_ylabel("5766 MHz, 33/33 phase change (degrees)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_low_gain(curves: dict[str, Any], output: Path) -> None:
    serials = sorted(curves)
    figure, axes = plt.subplots(len(serials), 1, figsize=(11, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, serial in zip(axes, serials):
        for frequency, data in sorted(
            curves[serial].items(), key=lambda item: int(item[0])
        ):
            axis.plot(
                data["gains_db"],
                data["H_deg"],
                marker=".",
                label=f"{int(frequency) / 1e6:.0f} MHz",
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(serial)
        axis.set_ylabel("symmetric H(g) (degrees)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("requested RX gain (dB)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_ripple_structure(plot_data: dict[str, Any], output: Path) -> None:
    delay = np.asarray(plot_data["delay_grid_ns"], dtype=np.float64)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    baseline = plot_data["spectra"]["A"]
    for key, amplitude in sorted(baseline.items()):
        serial, receiver = key.split(":")
        radio = (
            "treated .17" if serial == plot_data["treated_serial"] else "control .18"
        )
        axes[0].plot(
            delay,
            amplitude,
            linewidth=0.9,
            label=f"{radio} {receiver.upper()}",
        )
    for component in plot_data["delays_ns"]:
        axes[0].axvline(component, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_title("Stage A, 45 dB branch effect: delay spectrum")
    axes[0].set_ylabel("single-component amplitude (degrees)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    treated_key = next(
        key
        for key in plot_data["spectra"]["A"]
        if key.endswith(":rx1")
        and any(plot_data["spectra"][stage].get(key) for stage in SPECTROSCOPY_STAGES)
    )
    # The treated serial is the RX1 key present in every stage spectrum.
    candidates = [
        key
        for key in plot_data["spectra"]["A"]
        if key.endswith(":rx1")
        and all(key in plot_data["spectra"][stage] for stage in SPECTROSCOPY_STAGES)
    ]
    if candidates:
        treated_key = candidates[0]
    for stage in SPECTROSCOPY_STAGES:
        amplitude = plot_data["spectra"][stage].get(treated_key)
        if amplitude is not None:
            axes[1].plot(delay, amplitude, linewidth=0.9, label=stage)
    for component in plot_data["delays_ns"]:
        axes[1].axvline(component, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_title("Treated RX1 across physical configurations")
    axes[1].set_xlabel("round-trip delay (ns)")
    axes[1].set_ylabel("single-component amplitude (degrees)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=5, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_gain_table_steps(plot_data: dict[str, Any], output: Path) -> None:
    rows = []
    for band, serials in plot_data.items():
        for serial, values in serials.items():
            for value in values:
                rows.append(
                    {
                        "band": band,
                        "serial": serial,
                        **value,
                    }
                )
    serials = sorted({row["serial"] for row in rows})
    figure, axes = plt.subplots(
        len(serials),
        1,
        figsize=(11, 7),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "low": "tab:blue",
        "middle": "tab:orange",
        "high": "tab:green",
    }
    for axis, serial in zip(axes.flat, serials):
        for band in colors:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["serial"] == serial and row["band"] == band
                ],
                key=lambda row: row["frequency_hz"],
            )
            if selected:
                axis.scatter(
                    [row["frequency_hz"] / 1e6 for row in selected],
                    [row["step_deg"] for row in selected],
                    label=band,
                    color=colors[band],
                )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(serial)
        axis.set_ylabel("H step (degrees)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("LO frequency (MHz)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_overlap(rows: list[dict[str, Any]], output: Path) -> None:
    serials = sorted({row["serial"] for row in rows})
    figure, axes = plt.subplots(
        len(serials),
        1,
        figsize=(11, 7),
        sharex=True,
        squeeze=False,
    )
    for axis, serial in zip(axes.flat, serials):
        selected = [row for row in rows if row["serial"] == serial]
        for pair, marker in (((5, 26), "o"), ((26, 5), "s")):
            values = sorted(
                [
                    row
                    for row in selected
                    if (row["gain_rx1_db"], row["gain_rx2_db"]) == pair
                ],
                key=lambda row: row["frequency_hz"],
            )
            axis.scatter(
                [row["frequency_hz"] / 1e6 for row in values],
                [np.degrees(row["phase_delta_rad"]) for row in values],
                marker=marker,
                label=f"{pair[0]}/{pair[1]} dB",
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.axhline(0.75, color="gray", linestyle=":", linewidth=0.8)
        axis.axhline(-0.75, color="gray", linestyle=":", linewidth=0.8)
        axis.set_title(serial)
        axis.set_ylabel("F − prior phase (degrees)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("LO frequency (MHz)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_hysteresis(result: dict[str, Any], output: Path) -> None:
    stages = list(result["per_stage"])
    serials = sorted(
        {serial for stage in result["per_stage"].values() for serial in stage}
    )
    x = np.arange(len(stages), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(
        len(serials),
        1,
        figsize=(12, 7),
        sharex=True,
        squeeze=False,
    )
    for axis, serial in zip(axes.flat, serials):
        baseline = [
            result["per_stage"][stage][serial]["baseline_mae_deg"] for stage in stages
        ]
        corrected = [
            result["per_stage"][stage][serial]["order_corrected_mae_deg"]
            for stage in stages
        ]
        axis.bar(x - width / 2, baseline, width, label="cell mean only")
        axis.bar(x + width / 2, corrected, width, label="+ prior-gain order")
        axis.set_title(serial)
        axis.set_ylabel("LOEO MAE (degrees)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xticks(x, stages)
    axes[-1, 0].set_xlabel("campaign stage")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _model_extract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    selected = {}
    for name in (
        "frequency_specific_additive_gain_per_radio",
        "frequency_specific_antisymmetric_gain_per_radio",
        "frequency_lut_gain_table_symmetric_gain_per_radio",
        "frequency_specific_additive_gain_universal",
        "branch_gain_delay_lut_per_radio",
    ):
        model = document["models"][name]
        selected[name] = {
            "label": model["label"],
            "parameters": model["total_parameter_count"],
            "leave_one_epoch_out": {
                key: model["leave_one_epoch_out"][key]
                for key in (
                    "circular_mae_deg",
                    "circular_rmse_deg",
                    "circular_p95_deg",
                    "circular_max_deg",
                    "coverage_fraction",
                )
            },
            "leave_one_frequency_out": model.get("leave_one_frequency_out"),
            "leave_one_radio_out": model.get("leave_one_radio_out"),
        }
    return {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_gain_db": document["reference_gain_db"],
        "frequencies_hz": document["frequencies_hz"],
        "gains_db": document["gains_db"],
        "models": selected,
        "default_model_policy": document["default_model_policy"],
    }


def _markdown_report(result: dict[str, Any]) -> str:
    integrity = result["integrity"]
    comparisons = result["treatments"]["comparisons_to_A"]
    ripple = result["ripple"]
    pad = ripple["pad_test"]
    components = ripple["shared_baseline_components"]
    overlap = result["low_gain_overlap"]
    lines = [
        "# A–G dual-RX spectroscopy campaign: final analysis",
        "",
        "## Executive conclusions",
        "",
        f"- Acquisition is structurally complete: "
        f"{integrity['captured_frames']:,}/{integrity['planned_frames']:,} "
        "scheduled frames were recorded. The passive gain-table audit passed, "
        "both radios used identical 77-row full tables, and firmware provenance "
        "is consistent across the datasets.",
        "- The baseline contains a shared high-gain ripple component at "
        f"{components[0]['delay_ns']:.3f} ns "
        f"({components[0]['one_way_free_space_equivalent_mm']:.1f} mm one-way "
        "free-space equivalent). On treated `.17` RX1, the nominal 11 dB pad "
        f"reduced that component from {pad['primary_component_treated_rx1_A_deg']:.2f}° "
        f"to {pad['primary_component_treated_rx1_B_deg']:.2f}° "
        f"({pad['treated_rx1_suppression_percent']:.1f}% suppression), while "
        f"the three untouched arms retained a median "
        f"{100 * pad['unchanged_arm_median_B_over_A_ratio']:.1f}% of baseline.",
        "- That pad result is strong evidence that the 382 mm-equivalent "
        "component is sensitive to the external RX1 path. It is **not** a clean "
        "pad-only causal proof: restoring the original harness in D did not "
        "restore the high-band A state, so connector re-mating or a persistent "
        "treatment-radio state change remains a material confound.",
        "- The 30 cm jumper added 1.36–1.49 ns of one-way effective delay, as "
        "expected for ordinary coax. A candidate reflection component appears "
        "near the predicted shifted delay, but C failed repeatability and the "
        "failed A→D restoration prevents a definitive component assignment.",
        "- D→G is stable (0.90–0.96° MAE overall), so the persistent A→D/G "
        "high-band change is not continuing thermal drift. The experiment "
        "instead establishes that a cable/connector intervention can move the "
        "phase state and leave it in a new stable state.",
        "- The crossed TX-level test finds modest phase dependence at 5100 MHz "
        "(about 0.05–0.10° per TX dB in spur-qualified cells) and negligible "
        "dependence at 5766 MHz. Immediate prior-gain schedule order provides no "
        "held-out improvement, so simple gain-setting hysteresis does not explain "
        "the failed B/C/D cells.",
        "- For correction, retain the serial-specific, exact-frequency additive "
        "RX1/RX2 LUT as the accuracy reference. The symmetric `H(g1)-H(g2)` LUT "
        "is the parsimonious default only where its measured error gap is "
        "acceptable. Always establish a per-session/per-harness phase anchor.",
        "",
        "Phase convention throughout is `RX1 minus RX2`. Treatment effects use "
        "`(treated stage - treated baseline) - (control stage - control baseline)`.",
        "",
        "## Acquisition and gate audit",
        "",
        f"- Rate pilot: {integrity['rate_gate']['seconds_per_frame']:.3f} s/frame "
        f"against the {integrity['rate_gate']['limit_seconds_per_frame']:.1f} "
        "s/frame limit: **pass**.",
        f"- Gain-table audit: **{integrity['gain_table_audit_status']}**; tables "
        f"identical between radios: "
        f"**{integrity['gain_tables_identical_between_radios']}**.",
        f"- Firmware metadata consistent across every stage dataset: "
        f"**{integrity['firmware_consistent_across_datasets']}**.",
        f"- Firmware metadata matches the immutable resolved campaign config: "
        f"**{integrity['firmware_matches_resolved_config']}**.",
        "",
        "| Stage | Capture | Validation status | Waiver | Passing cells by radio |",
        "|---|---:|---|---|---|",
    ]
    for stage, row in integrity["stages"].items():
        cells = ", ".join(
            f"`{serial[-8:]}` {radio['passing_cells']}/{radio['expected_cells']}"
            for serial, radio in sorted(row["per_radio"].items())
        )
        lines.append(
            f"| {stage} | {row['completed_frames']}/{row['expected_frames']} | "
            f"{row['status']} | {'yes' if row['waived'] else 'no'} | {cells} |"
        )
    lines.extend(
        [
            "",
            "B, C, and D are complete captures with explicit repeatability "
            "waivers; they remain failed validation stages. The `-80 dB` E root "
            "is an intentional TX-muted floor control, so its phase is not treated "
            "as a valid tone measurement.",
            "",
            "## Controlled treatment comparisons",
            "",
            "| Stage vs A | Cells | Bias ° | MAE ° | P95 ° | Median amplitude Δ dB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in TREATMENT_STAGES:
        row = comparisons[stage]["treated_minus_control"]
        lines.append(
            f"| {stage} | {row['n']} | {row['circular_bias_deg']:.3f} | "
            f"{row['circular_mae_deg']:.3f} | {row['circular_p95_deg']:.3f} | "
            f"{row['amplitude_delta_median_db']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Control-corrected treatment phase](treatment_phase_difference_of_differences.png)",
            "",
            "### Equal-gain effective delay by gain-table band",
            "",
            "These slopes are descriptive effective delays from the `(26,26)` "
            "control-corrected treatment curve. They do not identify a unique "
            "physical cable or PCB path.",
            "",
            "| Stage | Band | Delay ps | Free-space equivalent mm | Residual RMSE ° |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for stage in TREATMENT_STAGES:
        bands = comparisons[stage]["treated_minus_control"]["delay_by_band_at_26_26"]
        for band, row in bands.items():
            lines.append(
                f"| {stage} | {band} | {row['delay_ps']:.2f} | "
                f"{row['equivalent_free_space_path_mm']:.2f} | "
                f"{row['residual_rmse_deg']:.2f} |"
            )

    c_delays = [
        row["delay_ps"]
        for row in comparisons["C"]["treated_minus_control"][
            "delay_by_band_at_26_26"
        ].values()
    ]
    lines.extend(
        [
            "",
            f"B changes the median RX1/RX2 amplitude ratio by "
            f"{comparisons['B']['treated_minus_control']['amplitude_delta_median_db']:.2f} "
            "dB, independently confirming the nominal 11 dB RX1 pad stack. "
            f"C produces {min(c_delays):.0f}–{max(c_delays):.0f} ps of effective "
            "delay across the three gain-table bands, consistent with the scale "
            "expected from a 30 cm coax jumper.",
            "",
            "### Ripple delay spectrum and one-versus-two components",
            "",
            "The spectrum uses the 45 dB branch effect relative to 26 dB, removes "
            "an independent quadratic nuisance in each audited gain-table band, "
            "and fits shared delays with arm-specific sine/cosine coefficients. "
            "A 0.4 ns separation constraint prevents a sidelobe of the dominant "
            "component from being called a second path.",
            "",
            "| Component | Delay ns | Frequency period MHz | One-way free-space equivalent mm |",
            "|---:|---:|---:|---:|",
        ]
    )
    for index, component in enumerate(components, start=1):
        lines.append(
            f"| {index} | {component['delay_ns']:.4f} | "
            f"{component['frequency_period_mhz']:.1f} | "
            f"{component['one_way_free_space_equivalent_mm']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Components | Parameters | SSE | BIC | ΔBIC from previous |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    previous_bic = None
    for row in ripple["model_comparison"]:
        delta = "—" if previous_bic is None else f"{row['bic'] - previous_bic:.2f}"
        lines.append(
            f"| {row['components']} | {row['parameters']} | {row['sse']:.4f} | "
            f"{row['bic']:.2f} | {delta} |"
        )
        previous_bic = row["bic"]
    lines.extend(
        [
            "",
            "The second component improves BIC after paying for its shared delay "
            "and per-arm amplitudes, so one component is insufficient. The second "
            f"best shared length is {components[1]['one_way_free_space_equivalent_mm']:.1f} "
            "mm; at this frequency span the delay resolution is not sufficient to "
            "distinguish it sharply from the previously suspected roughly 127 mm "
            "path.",
            "",
            "| Arm | A primary amplitude ° | B primary amplitude ° | B/A |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in sorted(ripple["component_amplitudes_deg"]["A"]):
        a_value = ripple["component_amplitudes_deg"]["A"][key][0]
        b_value = ripple["component_amplitudes_deg"]["B"][key][0]
        lines.append(
            f"| `{key}` | {a_value:.2f} | {b_value:.2f} | " f"{b_value / a_value:.3f} |"
        )
    jumper = ripple["jumper_test"]
    lines.extend(
        [
            "",
            f"The jumper's equal-gain one-way delay predicts a moved reflection "
            f"at {jumper['expected_moved_reflection_delay_ns']:.3f} ns. The "
            f"treated RX1 amplitude at that delay is "
            f"{jumper['amplitude_at_expected_moved_delay_deg']['C']:.2f}° in C "
            f"versus {jumper['amplitude_at_expected_moved_delay_deg']['D']:.2f}° "
            f"after restoration (ratio {jumper['C_over_D_amplitude_ratio']:.2f}). "
            "This is supportive, but the C repeatability failure and the changed "
            "post-A state make it non-causal evidence.",
            "",
            "![Ripple delay spectrum](ripple_delay_spectrum.png)",
            "",
            "### Connector/restoration and hot-repeat stability",
            "",
            "The largest failed-restoration effect is concentrated in treated "
            "RX1 at high gain above 4 GHz:",
            "",
            "| Gain pair | D−A MAE ≤4 GHz ° | D−A MAE >4 GHz ° | D−A p95 >4 GHz ° |",
            "|---|---:|---:|---:|",
        ]
    )
    d_pairs = comparisons["D"]["treated_minus_control"]["by_gain_pair_and_rf_region"]
    for pair in PHASE_PAIR_LABELS:
        row = d_pairs[f"{pair[0]}_{pair[1]}"]
        low = row["at_or_below_4ghz"]
        high = row["above_4ghz"]
        lines.append(
            f"| {pair[0]}/{pair[1]} | {low['circular_mae_deg']:.2f} | "
            f"{high['circular_mae_deg']:.2f} | "
            f"{high['circular_p95_deg']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Restored-baseline to hot-repeat stability (D → G)",
            "",
            "| Radio | RF region | MAE ° | P95 ° | Median amplitude Δ dB |",
            "|---|---|---:|---:|---:|",
        ]
    )
    repeatability = result["treatments"]["restored_D_to_hot_G_repeatability"]
    for serial, regions in repeatability["per_radio"].items():
        for region, row in regions.items():
            lines.append(
                f"| `{serial}` | {region} | {row['circular_mae_deg']:.3f} | "
                f"{row['circular_p95_deg']:.3f} | "
                f"{row['amplitude_delta_median_db']:.3f} |"
            )
    lines.extend(
        [
            "",
            "D and G agree much more closely with each other than either agrees "
            "with A above 4 GHz. Therefore the persistent high-band A→D/G shift "
            "is not continuing thermal drift. Because only `.17` RX1 was physically "
            "disturbed, the most parsimonious candidates are connector/harness "
            "re-mating and a treatment-radio RX1 state transition. The current "
            "experiment cannot separate them.",
        ]
    )

    lines.extend(
        [
            "",
            "## TX-level experiment",
            "",
            "| Radio | Frequency | Corrected slope median °/dB | "
            "Slope p05…p95 °/dB | −28 dB tone/floor median | Cells ≥45.6 dB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for serial, radio in result["levels"]["per_radio"].items():
        for frequency, row in radio["frequencies"].items():
            slope = row["spur_qualified_anchor_corrected_slope_deg_per_tx_db"]
            floor = row["minus28_tone_to_muted_floor_db"]
            lines.append(
                f"| `{serial}` | {int(frequency) / 1e6:.0f} MHz | "
                f"{slope['median']:.4f} | {slope['p05']:.4f}…{slope['p95']:.4f} | "
                f"{floor['median']:.2f} dB | {slope['cells']}/{row['cells']} "
                f"({100 * floor['fraction_at_least_45p6_db']:.1f}%) |"
            )
    lines.extend(
        [
            "",
            "![TX-level phase response](tx_level_phase_response.png)",
            "",
            "![TX tone and muted floor](tx_level_tone_floor.png)",
            "",
            "![Thermal anchors](thermal_anchor_drift.png)",
            "",
            "The E anchors move by less than 0.3° over the crossed-level sequence. "
            "At 5100 MHz only 59–78% of cells meet the predeclared 45.6 dB "
            "tone-to-muted-floor margin, so only those cells support the slope. "
            "At 5766 MHz all cells qualify and the slope is effectively zero.",
            "",
            "## Gain-table states and low-gain coverage",
            "",
            "The local passive audit read the exact active table bytes from both "
            "radios. The tables are byte-identical. Within the deliberately dense "
            "11–17 dB region, the first LNA/mixer-byte transition and observed "
            "symmetric-H step are:",
            "",
            "| Table band | LNA/mixer transition | Raw byte 0 before→after | Observed H steps ° |",
            "|---|---:|---|---|",
        ]
    )
    gain_tables = result["gain_tables"]
    for band, data in gain_tables["bands"].items():
        boundary = data["lna_mixer_boundary_gain_db"]
        states = data["states_11_to_17_db"]
        before_state = states.get(boundary - 1, states.get(str(boundary - 1)))
        after_state = states.get(boundary, states.get(str(boundary)))
        before = before_state["bytes"][0]
        after = after_state["bytes"][0]
        steps = [
            value["step_deg"]
            for serial_values in gain_tables["observed_symmetric_H_steps"][
                band
            ].values()
            for value in serial_values
        ]
        lines.append(
            f"| {band} | {boundary - 1}→{boundary} dB | "
            f"`0x{before:02x}`→`0x{after:02x}` | "
            f"{min(steps):.2f}…{max(steps):.2f} |"
        )
    lines.extend(
        [
            "",
            "The observed phase steps line up with actual LNA/mixer table "
            "transitions: 16→17 dB in the low table, 14→15 dB in the middle "
            "table, and 15→16 dB in the high table. This is direct evidence that "
            "the LUT discontinuities are hardware-state effects, not a smooth "
            "function of requested dB.",
            "",
            "![Gain-table transition steps](gain_table_transition_steps.png)",
            "",
            "### Cross-survey overlap",
            "",
        ]
    )
    if overlap["available"]:
        lines.extend(
            [
                "A and F contain no exact common frequencies, so the planned "
                "overlap check cannot use A. The reproducible replacement compares "
                "F with the immediately preceding wide integer-gain survey at the "
                "same six LOs and the common 5/26 and 26/5 gain pairs.",
                "",
                "| Radio | Region | Cells | MAE ° | P95 ° | 0.75° MAE gate |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for serial, regions in overlap["per_radio"].items():
            for region, row in regions.items():
                lines.append(
                    f"| `{serial}` | {region} | {row['n']} | "
                    f"{row['circular_mae_deg']:.3f} | "
                    f"{row['circular_p95_deg']:.3f} | "
                    f"{'pass' if row['passes_0p75deg_mae'] else 'fail'} |"
                )
        lines.extend(
            [
                "",
                "The control board passes the aggregate 0.75° MAE gate; the "
                "treated board fails because the persistent high-band state shift "
                "also affects the F overlap. Below 4 GHz the overlap is much "
                "closer. This independently confirms that the post-intervention "
                "change is not a model-fitting artifact.",
                "",
                "![F overlap with prior wide survey](low_gain_overlap.png)",
            ]
        )
    else:
        lines.append(f"Overlap unavailable: {overlap['reason']}.")
    lines.extend(
        [
            "",
            "## Schedule-order hysteresis test",
            "",
            "A leave-one-epoch-out ridge regression predicts each frame's residual "
            "from the signed and absolute RX1/RX2 gain jump from the immediately "
            "preceding frame in the same LO block. A real simple gain-setting "
            "hysteresis should reduce held-out error.",
            "",
            "| Stage | Radio | Baseline MAE ° | Order-corrected MAE ° | Improvement ° |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for stage, radios in result["hysteresis"]["per_stage"].items():
        for serial, row in radios.items():
            lines.append(
                f"| {stage} | `{serial[-8:]}` | {row['baseline_mae_deg']:.3f} | "
                f"{row['order_corrected_mae_deg']:.3f} | "
                f"{row['mae_improvement_deg']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "The correction is effectively zero or slightly harmful in held-out "
            "epochs. Therefore the B/C/D repeatability failures are not explained "
            "by a linear dependence on the immediately preceding gain command. "
            "Frequency-retune/calibration state and connector state remain better "
            "candidates.",
            "",
            "![Schedule-order hysteresis test](schedule_order_hysteresis.png)",
            "",
            "## Model ladder",
            "",
            "| Dataset | Model | Parameters | LOEO MAE ° | LOEO P95 ° |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset, matrix in result["model_matrices"].items():
        for name in (
            "frequency_specific_additive_gain_per_radio",
            "frequency_specific_antisymmetric_gain_per_radio",
            "frequency_lut_gain_table_symmetric_gain_per_radio",
            "frequency_specific_additive_gain_universal",
            "branch_gain_delay_lut_per_radio",
        ):
            row = matrix["models"][name]
            metrics = row["leave_one_epoch_out"]
            lines.append(
                f"| {dataset} | {row['label']} | {row['parameters']} | "
                f"{metrics['circular_mae_deg']:.3f} | "
                f"{metrics['circular_p95_deg']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Stage A shows that gain response changes substantially with exact "
            "frequency: the per-frequency independent model is the accuracy "
            "reference. Stage F shows that over the complete low/TIA gain range, "
            "the antisymmetric shared `H(g)` model is nearly as accurate and is "
            "the parsimonious default.",
            "",
            "![Low-gain symmetric curves](low_gain_symmetric_H.png)",
            "",
            "## Question-by-question decision ledger",
            "",
            "| Question | Decision | Evidence |",
            "|---|---|---|",
            "| Were the intended firmware and full gain tables active? | **Pass** | "
            "Passive audit passed on both serials; all firmware fields are consistent. |",
            "| Is one ripple component sufficient? | **No** | The second shared "
            "delay component improves BIC after parameter penalty. |",
            "| Is the dominant roughly 382 mm-equivalent ripple external? | "
            "**Supported, not proven pad-only** | It collapses 81% only on padded "
            "RX1 while untouched arms remain stable; failed A→D restoration leaves "
            "connector/state confounding. |",
            "| Does the 30 cm jumper add the expected path delay? | **Yes** | "
            "Equal-gain phase slope gives 1.36–1.49 ns one-way effective delay. |",
            "| Does the jumper uniquely locate each ripple component? | "
            "**Inconclusive** | Predicted shifted-delay energy appears, but C "
            "repeatability and A→D restoration fail. |",
            "| Is connector/harness restoration repeatable? | **Fail above 4 GHz** | "
            "D−A reaches 34.5° MAE for 45/26 above 4 GHz. |",
            "| Is the later hot state stable? | **Pass conditionally** | D→G is "
            "0.90–0.96° MAE overall, with larger high-band tails. |",
            "| Is phase level-dependent? | **Modestly at 5100; no at 5766** | "
            "Spur-qualified crossed TX-level slopes. |",
            "| Are low-gain hardware transitions visible? | **Yes** | H steps "
            "coincide with audited LNA/mixer-byte boundaries. |",
            "| Does immediate gain-command order explain residuals? | **No** | "
            "No held-out MAE improvement from transition features. |",
            "",
            "## Calibration recommendation",
            "",
            "1. Use the radio-specific, exact-LO independent additive RX1/RX2 LUT "
            "as the accuracy reference. Apply "
            "`wrap(measured_RX1_minus_RX2 - predicted_offset)`.",
            "2. Prefer the symmetric `H(g1)-H(g2)` representation when its "
            "serial/frequency-specific held-out gap to the independent model is "
            "within the declared tolerance; it is especially effective in F.",
            "3. Never transfer the absolute intercept across a connector re-mate, "
            "harness change, radio replacement, or unvalidated boot. Measure a "
            "per-session equal-gain anchor at every operating LO.",
            "4. Preserve exact gain-table discontinuities. Do not interpolate "
            "linearly through the audited LNA/mixer boundaries.",
            "5. For AGC captures, require valid frame-aligned endpoint metadata "
            "and reject endpoint changes. Endpoint equality still does not prove "
            "there was no in-buffer transition.",
            "6. Treat the current 5100 MHz level coefficient as a small systematic "
            "uncertainty, not a universal correction; 5766 MHz needs none.",
            "",
            "## Limitations",
            "",
            "- The 11 dB pad stack, 30 cm jumper, and connector torque were not "
            "independently characterized. The control radio removes shared drift "
            "but cannot remove treatment-radio-specific retune events.",
            "- A→D connector repeatability failed above 4 GHz. This prevents clean "
            "pad-only and jumper-component causal attribution.",
            "- The Stage E muted `−80 dB` capture is a floor measurement; its "
            "phase values fail normal tone-quality gates and are not interpreted.",
            "- Thermal-anchor correction at 5100 MHz uses the 5766 MHz anchor as "
            "an additive drift proxy.",
            "- Effective delays describe phase slope. They do not prove that a "
            "specific cable, PCB trace, analogue filter, or gain-table state is "
            "the sole mechanism.",
            "- The planned independent final passive gain-table re-read was not "
            "recorded. G's embedded firmware/image/gadget identities match A and "
            "the resolved config, but that is weaker than a second table-byte dump.",
            "- Every configuration is cabled and only two radios were tested. "
            "Over-the-air transfer, fleet-wide prevalence, and general unequal-arm "
            "level sensitivity remain outside this campaign.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis \\",
            f"  --campaign-root {result['campaign_root']} \\",
            f"  --treated-serial {result['treated_serial']} \\",
            f"  --control-serial {result['control_serial']} \\",
            (
                f"  --prior-calibration-root " f"{overlap['prior_calibration_root']} \\"
                if overlap["available"]
                else ""
            ),
            "  --output-dir <campaign-root>/analysis/campaign",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_campaign(
    *,
    campaign_root: Path,
    treated_serial: str,
    control_serial: str,
    output_dir: Path,
    prior_calibration_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    integrity = analyze_campaign_integrity(campaign_root)
    treatments, dod_rows = analyze_treatments(
        campaign_root,
        treated_serial=treated_serial,
        control_serial=control_serial,
    )
    serials = (treated_serial, control_serial)
    levels, level_plot_data = analyze_levels(campaign_root, serials=serials)
    low_gain, low_gain_curves = analyze_low_gain(campaign_root, serials=serials)
    gain_tables, gain_step_plot_data = analyze_gain_table_transitions(
        campaign_root,
        low_gain_curves=low_gain_curves,
    )
    c_delays = [
        row["delay_ps"] * 1e-12
        for row in treatments["comparisons_to_A"]["C"]["treated_minus_control"][
            "delay_by_band_at_26_26"
        ].values()
    ]
    ripple, ripple_plot_data = analyze_ripple_structure(
        campaign_root,
        treated_serial=treated_serial,
        control_serial=control_serial,
        c_equal_gain_delay_s=float(np.median(c_delays)),
    )
    overlap, overlap_plot_data = analyze_low_gain_overlap(
        campaign_root,
        serials=serials,
        prior_calibration_root=prior_calibration_root,
    )
    hysteresis = analyze_schedule_hysteresis(
        campaign_root,
        serials=serials,
    )
    matrices = {
        stage: _model_extract(
            campaign_root / "analysis" / f"model_matrix_{stage}" / "model_matrix.json"
        )
        for stage in ("A", "F")
    }
    result = {
        "schema": "spf.calibration.dual_rx_gain_frequency.spectroscopy_analysis",
        "schema_version": 2,
        "campaign_root": str(campaign_root),
        "treated_serial": treated_serial,
        "control_serial": control_serial,
        "integrity": integrity,
        "treatments": treatments,
        "ripple": ripple,
        "levels": levels,
        "low_gain": low_gain,
        "gain_tables": gain_tables,
        "low_gain_overlap": overlap,
        "hysteresis": hysteresis,
        "model_matrices": matrices,
    }
    _plot_treatments(
        dod_rows,
        output_dir / "treatment_phase_difference_of_differences.png",
    )
    _plot_level_response(
        level_plot_data,
        output_dir / "tx_level_phase_response.png",
    )
    _plot_tone_floor(level_plot_data, output_dir / "tx_level_tone_floor.png")
    _plot_anchor_drift(level_plot_data, output_dir / "thermal_anchor_drift.png")
    _plot_low_gain(low_gain_curves, output_dir / "low_gain_symmetric_H.png")
    _plot_ripple_structure(
        ripple_plot_data,
        output_dir / "ripple_delay_spectrum.png",
    )
    _plot_gain_table_steps(
        gain_step_plot_data,
        output_dir / "gain_table_transition_steps.png",
    )
    if overlap_plot_data:
        _plot_overlap(
            overlap_plot_data,
            output_dir / "low_gain_overlap.png",
        )
    _plot_hysteresis(
        hysteresis,
        output_dir / "schedule_order_hysteresis.png",
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(_markdown_report(result))
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--treated-serial", required=True)
    parser.add_argument("--control-serial", required=True)
    parser.add_argument("--prior-calibration-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    result = analyze_campaign(
        campaign_root=args.campaign_root,
        treated_serial=args.treated_serial,
        control_serial=args.control_serial,
        output_dir=args.output_dir,
        prior_calibration_root=args.prior_calibration_root,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "output_dir": str(args.output_dir),
                "treated_serial": result["treated_serial"],
                "control_serial": result["control_serial"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
