import json
from pathlib import Path

import numpy as np

from spf.calibrations.dual_rx_gain_frequency.power_cycle import (
    PowerCycleThresholds,
    analyze_cell_maps,
    write_power_cycle_bundle,
)
from spf.calibrations.dual_rx_gain_frequency.runner import (
    load_calibration_document,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "spf/calibrations/dual_rx_gain_frequency/configs/power_cycle_subsample.yaml"
)
FREQUENCIES = (868_000_000, 2_412_000_000)
GAINS = (16, 26, 41)


def _cells():
    return {
        (frequency, gain1, gain2): np.radians(
            7.0 + frequency / 1e9 + 0.1 * gain1 - 0.15 * gain2
        )
        for frequency in FREQUENCIES
        for gain1 in GAINS
        for gain2 in GAINS
    }


def _shift(before, shift):
    return {
        key: float(np.angle(np.exp(1j * (value + shift(key)))))
        for key, value in before.items()
    }


def _analyze(after):
    before = _cells()
    return analyze_cell_maps(
        before,
        after,
        expected_cells=len(before),
        reference_gain_db=26,
        global_anchor_frequency_hz=2_412_000_000,
        thresholds=PowerCycleThresholds(
            maximum_mae_deg=2,
            maximum_p95_deg=5,
            minimum_common_cell_fraction=1,
        ),
    )


def test_power_cycle_subsample_uses_900_frames_per_radio():
    _, config = load_calibration_document(CONFIG)
    assert config.gains_db == (0, 16, 26, 41, 52)
    assert config.measurements_per_radio == 900


def test_power_cycle_direct_reuse_when_raw_drift_is_small():
    before = _cells()
    result = _analyze(_shift(before, lambda key: np.radians(1)))
    assert result["verdict"] == "reusable_without_session_calibration"


def test_power_cycle_global_anchor_removes_common_shift():
    before = _cells()
    result = _analyze(_shift(before, lambda key: np.radians(12)))
    assert result["verdict"] == "one_global_session_anchor_required"
    assert result["one_global_anchor_adjusted"]["circular_p95_deg"] < 1e-9


def test_power_cycle_per_frequency_anchors_remove_retune_shifts():
    before = _cells()
    offsets = {868_000_000: np.radians(-14), 2_412_000_000: np.radians(17)}
    result = _analyze(_shift(before, lambda key: offsets[key[0]]))
    assert result["verdict"] == "one_anchor_per_frequency_required"
    assert result["one_anchor_per_frequency_adjusted"]["circular_p95_deg"] < 1e-9


def test_power_cycle_gain_shape_change_requires_recalibration():
    before = _cells()
    result = _analyze(
        _shift(
            before,
            lambda key: np.radians(0.8 * (key[1] - 26) - 0.6 * (key[2] - 26)),
        )
    )
    assert result["verdict"] == "gain_dependent_recalibration_required"


def test_power_cycle_fails_closed_when_a_frequency_anchor_is_missing():
    before = _cells()
    after = _shift(before, lambda key: np.radians(1))
    before.pop((868_000_000, 26, 26))
    after.pop((868_000_000, 26, 26))
    result = analyze_cell_maps(
        before,
        after,
        expected_cells=len(_cells()),
        reference_gain_db=26,
        global_anchor_frequency_hz=2_412_000_000,
        thresholds=PowerCycleThresholds(minimum_common_cell_fraction=0.8),
        expected_frequencies_hz=FREQUENCIES,
    )
    assert result["verdict"] == "inconclusive_missing_reference_anchors"


def _write_run(root, phases):
    serial = "SERIAL-A"
    root.mkdir()
    (root / "automation_plan.json").write_text(
        json.dumps(
            {
                "radio_serials": [serial],
                "calibration_config_sha256": "a" * 64,
                "firmware": {"image-sha256": "b" * 64},
            }
        )
    )
    (root / "automation_result.json").write_text(json.dumps({"status": "complete"}))
    radio = root / serial
    radio.mkdir()
    cells = [
        {
            "frequency_hz": frequency,
            "gain_rx1_db": gain1,
            "gain_rx2_db": gain2,
            "phase_mean_rad": phase,
            "pass": True,
        }
        for (frequency, gain1, gain2), phase in phases.items()
    ]
    (radio / "validation.json").write_text(
        json.dumps(
            {
                "serial": serial,
                "status": "pass",
                "expected_cells": len(cells),
                "cells": cells,
            }
        )
    )


def test_power_cycle_bundle_validates_roots_and_writes_report(tmp_path):
    before = _cells()
    after = _shift(before, lambda key: np.radians(11))
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    output = tmp_path / "comparison"
    _write_run(before_root, before)
    _write_run(after_root, after)

    result = write_power_cycle_bundle(
        before_root=before_root,
        after_root=after_root,
        output_dir=output,
        reference_gain_db=26,
        global_anchor_frequency_hz=2_412_000_000,
        thresholds=PowerCycleThresholds(minimum_common_cell_fraction=1),
    )

    assert result["overall_verdict"] == "one_global_session_anchor_required"
    assert (output / "power_cycle_comparison.json").is_file()
    assert "one_global_session_anchor_required" in (output / "README.md").read_text()
