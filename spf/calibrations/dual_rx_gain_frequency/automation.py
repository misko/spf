"""Automated firmware preparation, probe, V7 capture, and validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from spf.calibrations.dual_rx_gain_frequency.runner import (
    load_calibration_document,
    probe_loopback,
    run_calibration,
    serials_from_ready_manifest,
)
from spf.calibrations.dual_rx_gain_frequency.validate import (
    validate_dataset,
    write_validation_report,
)
from spf.scripts.rover_capture_config import resolve_capture_plan


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREPARATION_CONFIG = (
    REPO_ROOT
    / "data_collection/rover/rover_v3.1/capture_configs/rover1_production_v7.yaml"
)
DEFAULT_READY_MANIFEST = Path("/run/spf/direct_usb_ready.json")
PREPARE_SCRIPT = (
    REPO_ROOT / "data_collection/rover/rover_v3.1/prepare_direct_usb_boot.sh"
)
AUTOMATION_SCHEMA = "spf.calibration.dual_rx_gain_frequency.automation"
AUTOMATION_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(document, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _firmware_from_capture_plan(plan) -> dict[str, str]:
    return {
        "release-tag": plan.firmware_release_tag,
        "asset-name": plan.firmware_asset_name,
        "image-url": plan.firmware_image_url,
        "image-sha256": plan.firmware_image_sha256,
        "firmware-git-sha": plan.firmware_git_sha,
        "gadget-git-sha": plan.gadget_git_sha,
        "boot-mode": plan.firmware_boot_mode,
    }


def _validated_inputs(
    *,
    config_path: Path,
    preparation_config_path: Path,
    rover_id: int,
    expected_radios: int,
) -> tuple[dict[str, Any], Any]:
    calibration_document, calibration_config = load_calibration_document(config_path)
    preparation_plan = resolve_capture_plan(rover_id, preparation_config_path)
    if preparation_plan.expected_radios != expected_radios:
        raise ValueError(
            f"preparation config expects {preparation_plan.expected_radios} radios, "
            f"but --expected-radios is {expected_radios}"
        )
    calibration_firmware = calibration_document.get("pluto-firmware")
    preparation_firmware = _firmware_from_capture_plan(preparation_plan)
    if calibration_firmware != preparation_firmware:
        raise ValueError(
            "calibration and preparation configs pin different Pluto firmware"
        )
    return calibration_document, calibration_config


def _prepare_radios(
    *,
    preparation_config_path: Path,
    rover_id: int,
    python: Path,
    ready_manifest_path: Path,
) -> None:
    privilege_prefix = [] if os.geteuid() == 0 else ["sudo"]
    command = [
        *privilege_prefix,
        "env",
        f"SPF_ROVER_ID={rover_id}",
        f"SPF_CAPTURE_CONFIG={preparation_config_path}",
        f"SPF_PYTHON={python}",
        f"SPF_DIRECT_USB_READY_FILE={ready_manifest_path}",
        f"PYTHONPATH={REPO_ROOT}",
        "bash",
        str(PREPARE_SCRIPT),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _automation_plan(
    *,
    config_path: Path,
    preparation_config_path: Path,
    rover_id: int,
    expected_radios: int,
    serials: tuple[str, ...],
    firmware: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{AUTOMATION_SCHEMA}.plan",
        "schema_version": AUTOMATION_SCHEMA_VERSION,
        "calibration_config": str(config_path.resolve()),
        "calibration_config_sha256": _file_sha256(config_path),
        "preparation_config": str(preparation_config_path.resolve()),
        "preparation_config_sha256": _file_sha256(preparation_config_path),
        "rover_id": rover_id,
        "expected_radios": expected_radios,
        "radio_serials": list(serials),
        "firmware": firmware,
    }


def _require_matching_resume_plan(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        return
    try:
        actual = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"automation plan is invalid JSON: {path}") from error
    if actual != expected:
        raise ValueError(
            "existing output has a different automation plan; use a new run root"
        )


def run_automated_calibration(
    *,
    config_path: Path,
    output_dir: Path,
    expected_radios: int = 2,
    preparation_config_path: Path = DEFAULT_PREPARATION_CONFIG,
    rover_id: int = 1,
    ready_manifest_path: Path = DEFAULT_READY_MANIFEST,
    resume: bool = False,
    python: Path | None = None,
) -> dict[str, Any]:
    """Prepare all radios, probe each serial, capture, and validate stored IQ."""

    config_path = Path(config_path).resolve()
    preparation_config_path = Path(preparation_config_path).resolve()
    output_dir = Path(output_dir).resolve()
    ready_manifest_path = Path(ready_manifest_path).resolve()
    python = Path(sys.executable if python is None else python).resolve()
    if expected_radios <= 0:
        raise ValueError("expected_radios must be positive")
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"automation output exists; pass --resume or choose a new root: {output_dir}"
        )

    calibration_document, calibration_config = _validated_inputs(
        config_path=config_path,
        preparation_config_path=preparation_config_path,
        rover_id=rover_id,
        expected_radios=expected_radios,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "automation_result.json"
    stage = "prepare"
    serials: tuple[str, ...] = ()
    try:
        _prepare_radios(
            preparation_config_path=preparation_config_path,
            rover_id=rover_id,
            python=python,
            ready_manifest_path=ready_manifest_path,
        )
        serials = serials_from_ready_manifest(ready_manifest_path)
        if len(serials) != expected_radios:
            raise RuntimeError(
                f"ready manifest has {len(serials)} radios, expected {expected_radios}"
            )

        plan = _automation_plan(
            config_path=config_path,
            preparation_config_path=preparation_config_path,
            rover_id=rover_id,
            expected_radios=expected_radios,
            serials=serials,
            firmware=dict(calibration_document["pluto-firmware"]),
        )
        stage = "plan"
        plan_path = output_dir / "automation_plan.json"
        _require_matching_resume_plan(plan_path, plan)
        _write_json_atomic(plan_path, plan)

        stage = "probe"
        probes = {}
        for serial in serials:
            probe = probe_loopback(config_path=config_path, serial=serial)
            _write_json_atomic(output_dir / serial / "probe.json", probe)
            probes[serial] = probe
            if probe.get("status") != "pass":
                raise RuntimeError(f"{serial}: loopback probe did not pass")

        stage = "capture"
        capture = run_calibration(
            config_path=config_path,
            output_dir=output_dir,
            ready_manifest_path=ready_manifest_path,
            serials=serials,
        )
        if capture.get("status") != "complete":
            raise RuntimeError("calibration capture is partial")

        stage = "validate"
        validations = {}
        quality_review_required = False
        for serial in serials:
            dataset = output_dir / serial / "calibration.v7.zarr"
            validation = validate_dataset(
                dataset,
                config=calibration_config,
                expected_serial=serial,
                recompute_iq=True,
            )
            write_validation_report(output_dir / serial / "validation.json", validation)
            validations[serial] = {
                key: value for key, value in validation.items() if key != "cells"
            }
            if validation["status"] == "partial":
                raise RuntimeError(f"{serial}: stored dataset is partial")
            if validation["status"] == "fail_quality":
                quality_review_required = True

        result = {
            "schema": f"{AUTOMATION_SCHEMA}.result",
            "schema_version": AUTOMATION_SCHEMA_VERSION,
            "status": "complete",
            "quality_review_required": quality_review_required,
            "radio_serials": list(serials),
            "output_dir": str(output_dir),
            "probes": probes,
            "capture": capture,
            "validations": validations,
        }
    except Exception as error:
        result = {
            "schema": f"{AUTOMATION_SCHEMA}.result",
            "schema_version": AUTOMATION_SCHEMA_VERSION,
            "status": "failed",
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "radio_serials": list(serials),
            "output_dir": str(output_dir),
        }
    _write_json_atomic(result_path, result)
    return result
