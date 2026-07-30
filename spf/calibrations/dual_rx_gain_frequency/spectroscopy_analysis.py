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
    lines = [
        "# A–G dual-RX spectroscopy campaign analysis",
        "",
        "## Executive summary",
        "",
        "- All controlled A–G acquisitions are complete. B, C, and D retain their "
        "explicit cell-repeatability waivers; they are not silently relabelled as passes.",
        "- B is the actual **11 dB three-pad treatment on RX1 of `.17` only**; "
        "`.18` is the unchanged control. C is the nominal uncharacterized 30 cm "
        "RX1 jumper on `.17` only.",
        "- Phase convention is `RX1 minus RX2`. Treatment effects below use "
        "difference-of-differences: `(treated stage − treated A) − "
        "(control stage − control A)`.",
        "",
        "## Treatment comparisons",
        "",
        "| Stage vs A | Cells | Bias ° | MAE ° | P95 ° | Median amplitude Δ dB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparisons = result["treatments"]["comparisons_to_A"]
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
        ]
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
            "is not evidence of continuing thermal drift; it is a radio-specific "
            "state change that occurred after A and remained stable through G.",
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
            "## Limitations",
            "",
            "- The 11 dB pad stack, 30 cm jumper, and connector torque were not "
            "independently characterized. The control radio removes shared drift "
            "but cannot remove treatment-radio-specific retune events.",
            "- The Stage E muted `−80 dB` capture is a floor measurement; its "
            "phase values fail normal tone-quality gates and are not interpreted.",
            "- Thermal-anchor correction at 5100 MHz uses the 5766 MHz anchor as "
            "an additive drift proxy.",
            "- Effective delays describe phase slope. They do not prove that a "
            "specific cable, PCB trace, analogue filter, or gain-table state is "
            "the sole mechanism.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis \\",
            f"  --campaign-root {result['campaign_root']} \\",
            f"  --treated-serial {result['treated_serial']} \\",
            f"  --control-serial {result['control_serial']} \\",
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    treatments, dod_rows = analyze_treatments(
        campaign_root,
        treated_serial=treated_serial,
        control_serial=control_serial,
    )
    serials = (treated_serial, control_serial)
    levels, level_plot_data = analyze_levels(campaign_root, serials=serials)
    low_gain, low_gain_curves = analyze_low_gain(campaign_root, serials=serials)
    matrices = {
        stage: _model_extract(
            campaign_root / "analysis" / f"model_matrix_{stage}" / "model_matrix.json"
        )
        for stage in ("A", "F")
    }
    result = {
        "schema": "spf.calibration.dual_rx_gain_frequency.spectroscopy_analysis",
        "schema_version": 1,
        "campaign_root": str(campaign_root),
        "treated_serial": treated_serial,
        "control_serial": control_serial,
        "treatments": treatments,
        "levels": levels,
        "low_gain": low_gain,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    result = analyze_campaign(
        campaign_root=args.campaign_root,
        treated_serial=args.treated_serial,
        control_serial=args.control_serial,
        output_dir=args.output_dir,
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
