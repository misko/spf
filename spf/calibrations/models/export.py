"""Export runtime phase models from a reproducible model-matrix result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .phase import (
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    SUPPORT_SCHEMA,
    SUPPORT_SCHEMA_VERSION,
)


REGISTRY_SCHEMA = "spf.calibration.phase_model_registry"
REGISTRY_SCHEMA_VERSION = 1
SOURCE_MATRIX_SCHEMA = "spf.calibration.dual_rx_gain_frequency.model_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _portable_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _support_profile(
    *,
    provenance: dict[str, Any],
    frequencies_hz: list[int],
    gains_db: list[int],
) -> dict[str, Any]:
    dataset = Path(provenance["dataset_path"])
    validation_path = dataset.parent / "validation.json"
    validation = _read_json(validation_path)
    if validation.get("serial") != provenance["serial"]:
        raise ValueError(f"validation serial mismatch: {validation_path}")
    if validation.get("status") == "partial":
        raise ValueError(f"partial validation cannot define support: {validation_path}")
    cells = validation.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"validation has no cell rows: {validation_path}")
    supported_cells = sorted(
        [
            [
                int(row["frequency_hz"]),
                int(row["gain_rx1_db"]),
                int(row["gain_rx2_db"]),
            ]
            for row in cells
            if row.get("pass")
        ]
    )
    return {
        "schema": SUPPORT_SCHEMA,
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "radio_serial": provenance["serial"],
        "phase_convention": "RX1 minus RX2",
        "frequencies_hz": [int(value) for value in frequencies_hz],
        "gains_db": [int(value) for value in gains_db],
        "expected_cells": int(validation["expected_cells"]),
        "supported_cell_count": len(supported_cells),
        "supported_cells": supported_cells,
        "source": {
            "dataset_path": provenance["dataset_path"],
            "analysis_input_sha256": provenance["analysis_input_sha256"],
            "validation_path": _portable_path(validation_path),
            "validation_sha256": _sha256(validation_path),
            "validation_status": validation["status"],
        },
    }


def export_model_registry(
    *,
    matrix_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write serial-specific JSON configs for every per-radio model family."""

    matrix_path = Path(matrix_path).resolve()
    output_root = Path(output_root).resolve()
    matrix = _read_json(matrix_path)
    if matrix.get("schema") != SOURCE_MATRIX_SCHEMA:
        raise ValueError(f"unsupported source model matrix: {matrix_path}")
    provenance_by_index = {int(row["radio_index"]): row for row in matrix["provenance"]}
    frequencies_hz = [int(value) for value in matrix["frequencies_hz"]]
    gains_db = [int(value) for value in matrix["gains_db"]]
    matrix_sha256 = _sha256(matrix_path)

    support_by_serial = {}
    for provenance in matrix["provenance"]:
        serial = str(provenance["serial"])
        support_path = output_root / "radio_support" / f"{serial}.json"
        support = _support_profile(
            provenance=provenance,
            frequencies_hz=frequencies_hz,
            gains_db=gains_db,
        )
        _write_json(support_path, support)
        support_by_serial[serial] = {
            "path": support_path,
            "sha256": _sha256(support_path),
        }

    registry_models = {}
    for model_name, model in sorted(matrix["models"].items()):
        if model["scope"] != "per_radio":
            continue
        configs = {}
        for fit in model["fits"]:
            radio_index = int(fit["radio_index"])
            provenance = provenance_by_index[radio_index]
            serial = str(provenance["serial"])
            config_path = output_root / model_name / f"{serial}.json"
            support = support_by_serial[serial]
            support_relative = Path("..") / "radio_support" / f"{serial}.json"
            document = {
                "schema": MODEL_SCHEMA,
                "schema_version": MODEL_SCHEMA_VERSION,
                "model_name": model_name,
                "label": model["label"],
                "scope": model["scope"],
                "kind": model["kind"],
                "formula": model["formula"],
                "phase_convention": matrix["phase_convention"],
                "radio_serial": serial,
                "reference_frequency_hz": float(matrix["reference_frequency_hz"]),
                "reference_gain_db": int(matrix["reference_gain_db"]),
                "frequencies_hz": frequencies_hz,
                "gains_db": gains_db,
                "can_predict_unseen_frequency": bool(
                    model["can_predict_unseen_frequency"]
                ),
                "coefficients_rad": {
                    str(name): float(value)
                    for name, value in fit["coefficients_rad"].items()
                },
                "parameter_count": int(fit["parameter_count"]),
                "support_profile": {
                    "path": str(support_relative),
                    "sha256": support["sha256"],
                    "strict_prediction_default": True,
                },
                "evaluation": {
                    "training_metrics_all_radios": model["training_metrics"],
                    "leave_one_epoch_out_all_radios": model["leave_one_epoch_out"],
                    "leave_one_frequency_out_all_radios": model[
                        "leave_one_frequency_out"
                    ],
                },
                "source": {
                    "model_matrix_path": _portable_path(matrix_path),
                    "model_matrix_sha256": matrix_sha256,
                    "dataset_path": provenance["dataset_path"],
                    "analysis_input_sha256": provenance["analysis_input_sha256"],
                    "firmware": provenance["attrs"],
                },
            }
            _write_json(config_path, document)
            configs[serial] = str(config_path.relative_to(output_root))
        registry_models[model_name] = {
            "label": model["label"],
            "kind": model["kind"],
            "formula": model["formula"],
            "can_predict_unseen_frequency": model["can_predict_unseen_frequency"],
            "configs_by_serial": configs,
        }

    registry = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "phase_convention": matrix["phase_convention"],
        "recommended_model": "frequency_specific_additive_gain_per_radio",
        "source_model_matrix_path": _portable_path(matrix_path),
        "source_model_matrix_sha256": matrix_sha256,
        "radio_serials": sorted(str(row["serial"]) for row in matrix["provenance"]),
        "models": registry_models,
    }
    _write_json(output_root / "registry.json", registry)
    return registry
