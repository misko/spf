"""Focused regression tests for the preregistered E-GSC9 decision rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "experiments/e_gsc9_rover_operating_region/analysis"
SERIAL = "104000bac4950008230026001b440a003a"
FREQUENCY = 5_766_000_000
CELLS = ((FREQUENCY, 26, 26), (FREQUENCY, 27, 26))


def _load(name: str) -> ModuleType:
    path = ANALYSIS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = _load("analyze_session_transfer")
SESSION_C = _load("analyze_session_c")


def _radio(
    phases_deg: list[float],
    coordinates: list[tuple[int, int, int]],
    epochs: list[int],
    timestamps: list[float],
    *,
    quality: list[bool] | None = None,
) -> dict:
    count = len(phases_deg)
    assert count == len(coordinates) == len(epochs) == len(timestamps)
    return {
        "serial": SERIAL,
        "analysis_input_sha256": "synthetic",
        "quality_mask": np.asarray(
            quality if quality is not None else [True] * count, dtype=bool
        ),
        "analysis_mask": np.ones(count, dtype=bool),
        "phase": np.radians(np.asarray(phases_deg, dtype=np.float64)),
        "frequency_hz": np.asarray([row[0] for row in coordinates], dtype=np.int64),
        "gain1": np.asarray([row[1] for row in coordinates], dtype=np.int64),
        "gain2": np.asarray([row[2] for row in coordinates], dtype=np.int64),
        "epoch": np.asarray(epochs, dtype=np.int64),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
    }


def test_h6_accepts_wrapped_cell_means_and_exact_twelve_hour_gap() -> None:
    primary = _radio([179.9, -30.0], list(CELLS), [0, 0], [100.0, 101.0])
    repeat = _radio([-179.85, -29.75], list(CELLS), [0, 0], [43_301.0, 43_302.0])
    result = TRANSFER._compare_radio(
        serial=SERIAL,
        primary=primary,
        repeat=repeat,
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )

    assert result["separation_pass"]
    assert result["a_end_to_b_start_seconds"] == 43_200
    assert result["h6_pass"]
    row = result["per_frequency"][0]
    assert row["common_quality_valid_cells"] == 2
    assert row["circular_mae_deg"] == pytest.approx(0.25)


def test_h6_rejects_missing_cells_short_gap_and_threshold_excess() -> None:
    primary = _radio([0.0, 0.0], list(CELLS), [0, 0], [100.0, 101.0])

    short_gap = _radio([0.25, 0.25], list(CELLS), [0, 0], [43_300.0, 43_301.0])
    short_result = TRANSFER._compare_radio(
        serial=SERIAL,
        primary=primary,
        repeat=short_gap,
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )
    assert not short_result["separation_pass"]
    assert short_result["per_frequency"][0]["h6_pass"]
    assert not short_result["h6_pass"]

    missing = _radio(
        [0.25, 0.25],
        list(CELLS),
        [0, 0],
        [43_301.0, 43_302.0],
        quality=[True, False],
    )
    missing_result = TRANSFER._compare_radio(
        serial=SERIAL,
        primary=primary,
        repeat=missing,
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )
    assert missing_result["separation_pass"]
    assert not missing_result["h6_pass"]
    missing_row = missing_result["per_frequency"][0]
    assert missing_row["common_quality_valid_cells"] == 1
    assert not missing_row["h6_pass"]

    excessive = _radio([0.51, 0.51], list(CELLS), [0, 0], [43_301.0, 43_302.0])
    excessive_result = TRANSFER._compare_radio(
        serial=SERIAL,
        primary=primary,
        repeat=excessive,
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )
    assert excessive_result["separation_pass"]
    assert not excessive_result["h6_pass"]
    excessive_row = excessive_result["per_frequency"][0]
    assert excessive_row["common_quality_valid_cells"] == 2
    assert excessive_row["circular_mae_deg"] == pytest.approx(0.51)
    assert not excessive_row["h6_pass"]


def _three_epoch_radio(offset_deg: float) -> dict:
    coordinates = [cell for epoch in range(3) for cell in CELLS]
    epochs = [epoch for epoch in range(3) for _cell in CELLS]
    base = [179.8, 10.0, 179.9, 10.1, -179.9, 9.9]
    return _radio(
        [value + offset_deg for value in base],
        coordinates,
        epochs,
        [float(index) for index in range(len(base))],
    )


def test_h7_accepts_bounded_treatment_that_reverses() -> None:
    result = SESSION_C._compare_radio(
        serial=SERIAL,
        radios={
            "a": _three_epoch_radio(0.0),
            "b": _three_epoch_radio(1.0),
            "aprime": _three_epoch_radio(0.1),
        },
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )

    assert result["h7_pass"]
    row = result["per_frequency"][0]
    assert row["treatment_b_minus_a"]["circular_mae_deg"] == pytest.approx(1.0)
    assert row["restoration_aprime_minus_a"]["circular_mae_deg"] == pytest.approx(0.1)
    assert row["retained_treatment_aprime_minus_b"][
        "circular_mae_deg"
    ] == pytest.approx(0.9)
    assert row["coupling_bound_pass"]
    assert row["reversal_pass"]
    assert row["reversal_cell_fraction_closer_to_a_than_b"] == 1.0


def test_h7_rejects_nonreversal_and_coupling_bound_excess() -> None:
    nonreversal = SESSION_C._compare_radio(
        serial=SERIAL,
        radios={
            "a": _three_epoch_radio(0.0),
            "b": _three_epoch_radio(1.0),
            "aprime": _three_epoch_radio(1.0),
        },
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )
    assert not nonreversal["h7_pass"]
    assert not nonreversal["per_frequency"][0]["reversal_pass"]

    excessive = SESSION_C._compare_radio(
        serial=SERIAL,
        radios={
            "a": _three_epoch_radio(0.0),
            "b": _three_epoch_radio(9.0),
            "aprime": _three_epoch_radio(0.0),
        },
        expected_by_frequency={FREQUENCY: set(CELLS)},
    )
    assert not excessive["h7_pass"]
    assert not excessive["per_frequency"][0]["coupling_bound_pass"]
