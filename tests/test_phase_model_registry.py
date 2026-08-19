import json
import csv
import gzip
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.model_matrix import _design
from spf.calibrations.dual_rx_gain_frequency.additive_cross import (
    COMPLETE_2P4_MODEL_NAME,
    export_complete_2p4_model,
)
from spf.calibrations.models import (
    UnsupportedPhaseModelInput,
    load_model,
)
from spf.calibrations.models.external_wall_validation import (
    validate_snapshot_export,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = REPO_ROOT / "spf/calibrations/models"
MATRIX_PATH = (
    REPO_ROOT / "spf/calibrations/dual_rx_gain_frequency/reports/"
    "six_radio_dense_20260729_v1/model_matrix.json"
)


def _registry_model_paths(root):
    registry = json.loads((root / "registry.json").read_text())
    for model_name, model_row in registry["models"].items():
        for serial in model_row["configs_by_serial"]:
            yield root / model_name / f"{serial}.json"


@pytest.fixture
def loadable_historical_registry(tmp_path):
    """Make archived models executable without modifying historical evidence.

    Shared support profiles changed after these model snapshots were exported,
    so their recorded hashes correctly fail closed. Functional tests reconcile
    only a temporary copy; the repository copy remains immutable and is tested
    separately below.
    """

    root = tmp_path / "models"
    shutil.copytree(REGISTRY_ROOT, root)
    for model_path in _registry_model_paths(root):
        document = json.loads(model_path.read_text())
        support = document["support_profile"]
        support_path = (model_path.parent / support["path"]).resolve()
        support["sha256"] = hashlib.sha256(support_path.read_bytes()).hexdigest()
        model_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return root


def test_historical_registry_rejects_stale_support_hashes():
    rejected = []
    for model_path in _registry_model_paths(REGISTRY_ROOT):
        with pytest.raises(ValueError, match="support profile hash mismatch"):
            load_model(
                model_path.parent.name,
                model_path.stem,
                registry_root=REGISTRY_ROOT,
            )
        rejected.append(model_path)
    assert len(rejected) == 56


def _source_prediction(matrix, model_name, radio_index, coordinate):
    frequency_hz, gain1_db, gain2_db = coordinate
    frequencies = matrix["frequencies_hz"]
    gains = matrix["gains_db"]
    reference_gain = gains.index(matrix["reference_gain_db"])
    data = {
        "phase": np.asarray([0.0]),
        "frequency_hz": np.asarray([frequency_hz], dtype=np.int64),
        "frequency": np.asarray([frequencies.index(frequency_hz)], dtype=np.int64),
        "gain1_db": np.asarray([gain1_db], dtype=np.int64),
        "gain2_db": np.asarray([gain2_db], dtype=np.int64),
        "gain1": np.asarray([gains.index(gain1_db)], dtype=np.int64),
        "gain2": np.asarray([gains.index(gain2_db)], dtype=np.int64),
    }
    source_model = matrix["models"][model_name]
    source_fit = next(
        fit for fit in source_model["fits"] if fit["radio_index"] == radio_index
    )
    design, names = _design(
        source_model["kind"],
        data,
        gain_count=len(gains),
        frequency_count=len(frequencies),
        reference_gain=reference_gain,
        reference_frequency_hz=matrix["reference_frequency_hz"],
    )
    beta = np.asarray(
        [source_fit["coefficients_rad"][name] for name in names],
        dtype=np.float64,
    )
    return float(wrap_phase((design @ beta)[0]))


def test_all_exported_models_load_and_match_source_matrix(
    loadable_historical_registry,
):
    registry = json.loads((REGISTRY_ROOT / "registry.json").read_text())
    matrix = json.loads(MATRIX_PATH.read_text())
    assert registry["recommended_model"] == (
        "frequency_specific_additive_gain_per_radio"
    )
    assert registry["complete_2p4_model"] == COMPLETE_2P4_MODEL_NAME
    assert len(registry["models"]) == 10
    assert len(registry["radio_serials"]) == 6

    provenance_by_serial = {
        row["serial"]: row["radio_index"] for row in matrix["provenance"]
    }
    for model_name, model_row in registry["models"].items():
        expected_serials = set(model_row["configs_by_serial"])
        if model_name in matrix["models"]:
            assert expected_serials == set(registry["radio_serials"])
        for serial in expected_serials:
            model = load_model(
                model_name,
                serial,
                registry_root=loadable_historical_registry,
            )
            coordinate = sorted(model.supported_cells)[0]
            actual = model.predict_phase_offset(
                frequency_hz=coordinate[0],
                gain_rx1_db=coordinate[1],
                gain_rx2_db=coordinate[2],
            )
            if model_name not in matrix["models"]:
                assert np.isfinite(actual)
                continue
            expected = _source_prediction(
                matrix,
                model_name,
                provenance_by_serial[serial],
                coordinate,
            )
            assert actual == pytest.approx(expected, abs=1e-12)


def test_strict_prediction_rejects_a_quality_unsupported_cell(
    loadable_historical_registry,
):
    model = load_model(
        "frequency_specific_additive_gain_per_radio",
        "104473b80a16000de6ff2000f8a6beca79",
        registry_root=loadable_historical_registry,
    )
    all_cells = {
        (frequency, gain1, gain2)
        for frequency in model.frequencies_hz
        for gain1 in model.gains_db
        for gain2 in model.gains_db
    }
    unsupported = sorted(all_cells - model.supported_cells)[0]
    with pytest.raises(UnsupportedPhaseModelInput, match="no validated support"):
        model.predict_phase_offset(
            frequency_hz=unsupported[0],
            gain_rx1_db=unsupported[1],
            gain_rx2_db=unsupported[2],
        )


def test_phase_correction_subtracts_and_wraps_prediction(
    loadable_historical_registry,
):
    model = load_model(
        "frequency_specific_additive_gain_per_radio",
        "104000707f0700120f001a0095f2dbee49",
        registry_root=loadable_historical_registry,
    )
    coordinate = (2_412_000_000, 26, 41)
    offset = model.predict_phase_offset(
        frequency_hz=coordinate[0],
        gain_rx1_db=coordinate[1],
        gain_rx2_db=coordinate[2],
    )
    corrected = model.correct_measured_phase(
        measured_phase_rad=3.1,
        frequency_hz=coordinate[0],
        gain_rx1_db=coordinate[1],
        gain_rx2_db=coordinate[2],
    )
    assert corrected == pytest.approx(float(wrap_phase(3.1 - offset)))


def test_complete_additive_cross_export_supports_full_gain_product(tmp_path):
    serial = "TEST-SERIAL"
    gains = [-1, 0, 1]
    frequencies = [2_411_950_000, 2_467_100_000]
    frequency_results = []
    for frequency_index, frequency_hz in enumerate(frequencies):
        frequency_results.append(
            {
                "frequency_hz": frequency_hz,
                "status": "fit",
                "intercept_rad": 0.2 + 0.1 * frequency_index,
                "shared_gain_effect_rad": [-0.1, 0.0, 0.15],
                "held_out_shared_gain_curve_metrics": {
                    "n_observations": 6,
                    "circular_mae_deg": 1.0,
                    "circular_rmse_deg": 1.2,
                    "circular_p95_deg": 2.0,
                    "circular_max_deg": 2.5,
                },
                "quality_valid_training_observations": 15,
                "quality_valid_held_out_observations": 6,
            }
        )
    validation_cells_by_frequency = {}
    for frequency_hz in frequencies:
        validation_cells = list(
            {
                "frequency_hz": frequency_hz,
                "gain_rx1_db": gain1,
                "gain_rx2_db": gain2,
                "role": "training",
                "pass": True,
            }
            for gain1, gain2 in (
                (-1, 0),
                (0, -1),
                (0, 0),
                (1, 0),
                (0, 1),
            )
        )
        validation_cells.append(
            {
                "frequency_hz": frequency_hz,
                "gain_rx1_db": -1,
                "gain_rx2_db": 1,
                "role": "held_out",
                "pass": True,
            }
        )
        validation_cells_by_frequency[frequency_hz] = validation_cells

    analysis_paths = []
    validation_paths = []
    for source_index, (frequency_hz, frequency_result) in enumerate(
        zip(frequencies, frequency_results)
    ):
        analysis = {
            "serial": serial,
            "schedule_design": "additive_cross",
            "phase_convention": "RX1 minus RX2",
            "reference_gain_db": 0,
            "gain_values_db": gains,
            "training_pairs_per_frequency": 5,
            "overall_held_out_shared_gain_curve_metrics": {
                "n_observations": 6,
                "circular_mae_deg": 1.0,
                "circular_rmse_deg": 1.2,
                "circular_p95_deg": 2.0,
                "circular_max_deg": 2.5,
            },
            "frequency_results": [frequency_result],
        }
        validation = {
            "serial": serial,
            "status": "pass",
            "cells": validation_cells_by_frequency[frequency_hz],
        }
        analysis_path = tmp_path / f"analysis-{source_index}.json"
        validation_path = tmp_path / f"validation-{source_index}.json"
        analysis_path.write_text(json.dumps(analysis))
        validation_path.write_text(json.dumps(validation))
        analysis_paths.append(analysis_path)
        validation_paths.append(validation_path)

    exported = export_complete_2p4_model(
        analysis_path=analysis_paths,
        validation_path=validation_paths,
        output_root=tmp_path / "models",
    )
    assert exported["supported_cell_count"] == 18
    model = load_model(
        COMPLETE_2P4_MODEL_NAME,
        serial,
        registry_root=tmp_path / "models",
    )
    assert len(model.supported_cells) == 18
    assert model.predict_phase_offset(
        frequency_hz=2_411_950_000,
        gain_rx1_db=1,
        gain_rx2_db=-1,
    ) == pytest.approx(0.45)
    with pytest.raises(UnsupportedPhaseModelInput):
        model.predict_phase_offset(
            frequency_hz=2_400_000_000,
            gain_rx1_db=1,
            gain_rx2_db=-1,
        )
    float32_alias = int(np.float32(2_467_100_000))
    assert float32_alias == 2_467_099_904
    with pytest.raises(UnsupportedPhaseModelInput):
        model.predict_phase_offset(
            frequency_hz=float32_alias,
            gain_rx1_db=1,
            gain_rx2_db=-1,
        )
    assert model.predict_phase_offset(
        frequency_hz=float32_alias,
        gain_rx1_db=1,
        gain_rx2_db=-1,
        allow_float32_frequency_alias=True,
    ) == pytest.approx(0.55)


def test_external_wall_validation_rederives_geometry_and_subtracts_model(
    tmp_path, loadable_historical_registry
):
    serial = "104000bac4950008230026001b440a003a"
    frequency_hz = 2_467_100_000
    model = load_model(
        COMPLETE_2P4_MODEL_NAME,
        serial,
        registry_root=loadable_historical_registry,
    )
    fieldnames = [
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
    ]
    rows = []
    for capture in ("capture-a", "capture-b"):
        for gain1, gain2, theta in ((26, 26, 0.2), (26, 41, -0.3)):
            phase_gt = float(wrap_phase(-np.sin(theta) * 0.5 * 2.0 * np.pi))
            prediction = model.predict_phase_offset(
                frequency_hz=frequency_hz,
                gain_rx1_db=gain1,
                gain_rx2_db=gain2,
            )
            rows.append(
                {
                    "capture": capture,
                    "receiver": "r0",
                    "serial": serial,
                    "lo_hz": frequency_hz,
                    "gain_rx1_db": gain1,
                    "gain_rx2_db": gain2,
                    "phase_meas_rad": float(wrap_phase(phase_gt + prediction)),
                    "phase_gt_rad": phase_gt,
                    "theta_gt_rad": theta,
                    "tx_pos_x_mm": np.sin(theta) * 1000.0,
                    "tx_pos_y_mm": np.cos(theta) * 1000.0,
                    "rx_pos_x_mm": 0.0,
                    "rx_pos_y_mm": 0.0,
                    "rx_theta_in_pis": 0.0,
                    "rx_heading_in_pis": 0.0,
                    "d_over_lambda": 0.5,
                }
            )
    export_path = tmp_path / "snapshots.csv.gz"
    with gzip.open(export_path, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = validate_snapshot_export(
        csv_gz_path=export_path,
        receiver="r0",
        serial=serial,
        frequency_hz=frequency_hz,
        registry_root=loadable_historical_registry,
    )

    assert result["integrity"]["selected_rows"] == 4
    assert result["integrity"]["coverage_fraction"] == 1.0
    assert result["integrity"]["maximum_theta_error_rad"] < 1e-12
    assert result["integrity"]["maximum_phase_error_rad"] < 1e-12
    assert result["summary"]["subtract_bias_deg"] == pytest.approx(0.0, abs=1e-12)
    assert result["summary"]["median_subtract_circstd_rad"] == pytest.approx(
        0.0, abs=1e-12
    )
