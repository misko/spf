import json
from pathlib import Path

import numpy as np
import pytest

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.model_matrix import _design
from spf.calibrations.models import (
    UnsupportedPhaseModelInput,
    load_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = REPO_ROOT / "spf/calibrations/models"
MATRIX_PATH = (
    REPO_ROOT / "spf/calibrations/dual_rx_gain_frequency/reports/"
    "six_radio_dense_20260729_v1/model_matrix.json"
)


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


def test_all_exported_models_load_and_match_source_matrix():
    registry = json.loads((REGISTRY_ROOT / "registry.json").read_text())
    matrix = json.loads(MATRIX_PATH.read_text())
    assert registry["recommended_model"] == (
        "frequency_specific_additive_gain_per_radio"
    )
    assert len(registry["models"]) == 9
    assert len(registry["radio_serials"]) == 6

    provenance_by_serial = {
        row["serial"]: row["radio_index"] for row in matrix["provenance"]
    }
    for model_name, model_row in registry["models"].items():
        assert len(model_row["configs_by_serial"]) == 6
        for serial in registry["radio_serials"]:
            model = load_model(model_name, serial, registry_root=REGISTRY_ROOT)
            coordinate = sorted(model.supported_cells)[0]
            actual = model.predict_phase_offset(
                frequency_hz=coordinate[0],
                gain_rx1_db=coordinate[1],
                gain_rx2_db=coordinate[2],
            )
            expected = _source_prediction(
                matrix,
                model_name,
                provenance_by_serial[serial],
                coordinate,
            )
            assert actual == pytest.approx(expected, abs=1e-12)


def test_strict_prediction_rejects_a_quality_unsupported_cell():
    model = load_model(
        "frequency_specific_additive_gain_per_radio",
        "104473b80a16000de6ff2000f8a6beca79",
        registry_root=REGISTRY_ROOT,
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


def test_phase_correction_subtracts_and_wraps_prediction():
    model = load_model(
        "frequency_specific_additive_gain_per_radio",
        "104000707f0700120f001a0095f2dbee49",
        registry_root=REGISTRY_ROOT,
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
