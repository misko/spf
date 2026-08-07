"""Render, audit, and run the controlled A-G spectroscopy campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

import yaml

from spf.calibrations.dual_rx_gain_frequency.automation import (
    DEFAULT_PREPARATION_CONFIG,
    DEFAULT_READY_MANIFEST,
    _prepare_radios,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.calibrations.dual_rx_gain_frequency.runner import (
    load_calibration_document,
    run_calibration,
    serials_from_ready_manifest,
)
from spf.calibrations.dual_rx_gain_frequency.validate import (
    validate_dataset,
    write_validation_report,
)
from spf.scripts.pluto_ready_manifest import load_manifest
from spf.scripts.pluto_multi_firmware import MultiPlutoFirmwareManager


CAMPAIGN_SCHEMA = "spf.calibration.dual_rx_gain_frequency.spectroscopy_campaign"
CAMPAIGN_SCHEMA_VERSION = 1
RENDERED_SCHEMA = f"{CAMPAIGN_SCHEMA}.rendered"
AUDIT_SCHEMA = f"{CAMPAIGN_SCHEMA}.gain_table_audit"
STAGE_RESULT_SCHEMA = f"{CAMPAIGN_SCHEMA}.stage_result"
QUALITY_WAIVER_SCHEMA = f"{CAMPAIGN_SCHEMA}.quality_waiver"
REPO_ROOT = Path(__file__).resolve().parents[3]
_CAMPAIGN_ONLY_STAGE_KEYS = {
    "allow-quality-failure",
    "description",
    "frequency-set",
    "gain-set",
    "id",
    "minimum-hours-after-stage-start",
    "operator-checkpoint",
    "requires",
}
_GAIN_TABLE_HEADER = re.compile(
    r"<gaintable AD(?P<device>\d+) type=(?P<type>\w+) dest=(?P<dest>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+)>"
)
_GAIN_TABLE_ROW = re.compile(
    r"^\s*(?P<gain>-?\d+)\s*,\s*"
    r"0x(?P<byte0>[0-9A-Fa-f]{2})\s*,\s*"
    r"0x(?P<byte1>[0-9A-Fa-f]{2})\s*,\s*"
    r"0x(?P<byte2>[0-9A-Fa-f]{2})\s*$"
)


class CampaignError(RuntimeError):
    """A fail-closed campaign precondition or execution failure."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _write_yaml_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            yaml.safe_dump(document, destination, sort_keys=False)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_campaign_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("campaign manifest must be a YAML mapping")
    if document.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"unexpected campaign schema: {document.get('schema')!r}")
    if document.get("schema-version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("unsupported campaign schema version")
    base = document.get("base-config")
    if not isinstance(base, str) or not base:
        raise ValueError("campaign requires base-config")
    base_path = (path.parent / base).resolve()
    load_calibration_document(base_path)
    if int(document.get("expected-radios", 0)) <= 0:
        raise ValueError("expected-radios must be positive")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("campaign requires at least one stage")
    ids = [stage.get("id") for stage in stages if isinstance(stage, dict)]
    if len(ids) != len(stages) or any(
        not isinstance(stage_id, str) for stage_id in ids
    ):
        raise ValueError("every campaign stage requires a string id")
    if len(ids) != len(set(ids)):
        raise ValueError("campaign stage ids must be unique")
    seen: set[str] = set()
    for stage in stages:
        requirements = stage.get("requires", [])
        if not isinstance(requirements, list) or any(
            requirement not in seen for requirement in requirements
        ):
            raise ValueError(
                f"{stage['id']}: requirements must name earlier campaign stages"
            )
        seen.add(stage["id"])
    return document, base_path


def _inclusive_range(spec: dict[str, Any], *, label: str) -> list[int]:
    try:
        start = int(spec["start"])
        stop = int(spec["stop-inclusive"])
        step = int(spec.get("step", 1))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} range is malformed") from error
    if step <= 0 or stop < start or (stop - start) % step:
        raise ValueError(f"{label} range must end exactly on a positive step")
    return list(range(start, stop + 1, step))


def _expand_set(
    raw: Any,
    *,
    values_key: str,
    range_key: str,
    add_key: str,
    label: str,
) -> list[int]:
    if isinstance(raw, list):
        values = [int(value) for value in raw]
    elif isinstance(raw, dict):
        values = []
        if values_key in raw:
            values.extend(int(value) for value in raw[values_key])
        if range_key in raw:
            values.extend(_inclusive_range(raw[range_key], label=label))
        values.extend(int(value) for value in raw.get(add_key, []))
    else:
        raise ValueError(f"{label} must be a list or mapping")
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if len(values) != len(set(values)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        raise ValueError(f"{label} contains duplicate coordinates: {duplicates}")
    return sorted(values)


def _resolve_stage_document(
    campaign: dict[str, Any],
    base_document: dict[str, Any],
    stage: dict[str, Any],
) -> dict[str, Any]:
    frequency_name = stage.get("frequency-set")
    gain_name = stage.get("gain-set")
    try:
        frequency_spec = campaign["frequency-sets"][frequency_name]
        gain_spec = campaign["gain-sets"][gain_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{stage['id']}: unknown frequency or gain set") from error
    frequencies = _expand_set(
        frequency_spec,
        values_key="values-hz",
        range_key="range-hz",
        add_key="add-hz",
        label=f"{stage['id']} frequencies",
    )
    gains = _expand_set(
        gain_spec,
        values_key="values-db",
        range_key="range-db",
        add_key="add-db",
        label=f"{stage['id']} gains",
    )
    document = copy.deepcopy(base_document)
    calibration = document["calibration"]
    calibration["frequencies-hz"] = frequencies
    calibration["gains-db"] = gains
    calibration["schedule-design"] = "additive_cross"
    for key, value in stage.items():
        if key not in _CAMPAIGN_ONLY_STAGE_KEYS:
            calibration[key] = value
    calibration["notes"] = (
        f"{stage['description']}; " + str(calibration.get("notes", ""))
    ).strip("; ")
    return document


def render_campaign(
    manifest_path: Path,
    output_root: Path,
    *,
    seconds_per_frame: float | None = None,
) -> dict[str, Any]:
    campaign, base_path = load_campaign_manifest(manifest_path)
    base_document = yaml.safe_load(base_path.read_text())
    output_root = Path(output_root).resolve()
    existing_plan_path = output_root / "campaign_plan.json"
    if existing_plan_path.is_file():
        existing = json.loads(existing_plan_path.read_text())
        if existing.get("manifest_sha256") != _sha256_file(
            Path(manifest_path).resolve()
        ) or existing.get("base_config_sha256") != _sha256_file(base_path):
            raise CampaignError(
                "existing campaign root is bound to a different manifest or base config"
            )
        return existing
    config_root = output_root / "resolved_configs"
    rows = []
    total_per_radio = 0
    for stage in campaign["stages"]:
        document = _resolve_stage_document(campaign, base_document, stage)
        config_path = config_root / f"{stage['id']}.yaml"
        _write_yaml_atomic(config_path, document)
        _, config = load_calibration_document(config_path)
        measurements = config.measurements_per_radio
        total_per_radio += measurements
        row = {
            "id": stage["id"],
            "description": stage["description"],
            "requires": list(stage.get("requires", [])),
            "operator_checkpoint": stage.get("operator-checkpoint"),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "frequencies": len(config.frequencies_hz),
            "gains": len(config.gains_db),
            "gain_pairs": len(config.gain_pairs),
            "epochs": config.repetitions,
            "measurements_per_radio": measurements,
            "measurements_all_radios": measurements * int(campaign["expected-radios"]),
            "allow_quality_failure": bool(stage.get("allow-quality-failure", False)),
            "minimum_hours_after_stage_start": stage.get(
                "minimum-hours-after-stage-start"
            ),
        }
        if seconds_per_frame is not None:
            row["estimated_capture_seconds"] = (
                row["measurements_all_radios"] * seconds_per_frame
            )
        rows.append(row)
    result = {
        "schema": RENDERED_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256_file(base_path),
        "expected_radios": int(campaign["expected-radios"]),
        "rate_gate": campaign["rate-gate"],
        "stages": rows,
        "measurements_per_radio": total_per_radio,
        "measurements_all_radios": total_per_radio * int(campaign["expected-radios"]),
    }
    if "analysis-contract" in campaign:
        if not isinstance(campaign["analysis-contract"], dict):
            raise ValueError("analysis-contract must be a YAML mapping")
        result["analysis_contract"] = copy.deepcopy(campaign["analysis-contract"])
    if seconds_per_frame is not None:
        result["estimated_capture_seconds"] = (
            result["measurements_all_radios"] * seconds_per_frame
        )
    _write_json_atomic(output_root / "campaign_plan.json", result)
    return result


def parse_gain_table_config(text: str | bytes) -> dict[str, Any]:
    if isinstance(text, bytes):
        text = text.decode("ascii")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or lines[-1] != "</gaintable>":
        raise ValueError("gain table is truncated or missing its closing tag")
    header = _GAIN_TABLE_HEADER.fullmatch(lines[0])
    if header is None:
        raise ValueError("gain table header is malformed")
    rows = []
    for line in lines[1:-1]:
        match = _GAIN_TABLE_ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed gain table row: {line!r}")
        rows.append(
            {
                "gain_db": int(match.group("gain")),
                "bytes": [
                    int(match.group("byte0"), 16),
                    int(match.group("byte1"), 16),
                    int(match.group("byte2"), 16),
                ],
            }
        )
    table_bytes = bytes(value for row in rows for value in row["bytes"])
    return {
        "device": int(header.group("device")),
        "type": header.group("type"),
        "destination": int(header.group("dest")),
        "start_hz": int(header.group("start")),
        "end_hz": int(header.group("end")),
        "rows": rows,
        "row_count": len(rows),
        "table_sha256": _sha256_bytes(table_bytes),
    }


def _firmware_matches(campaign: dict[str, Any], ready: dict[str, Any]) -> None:
    audit = campaign["gain-table-audit"]
    ready_firmware = ready.get("firmware", {})
    for campaign_key, ready_key in (
        ("firmware-git-sha", "firmware_git_sha"),
        ("gadget-git-sha", "gadget_git_sha"),
        ("image-sha256", "image_sha256"),
    ):
        expected = audit[campaign_key]
        actual = ready_firmware.get(ready_key)
        if actual != expected:
            raise CampaignError(
                f"ready {campaign_key} {actual!r} does not match {expected!r}"
            )


def audit_gain_tables(
    manifest_path: Path,
    *,
    ready_manifest_path: Path = DEFAULT_READY_MANIFEST,
    output_path: Path,
    radio_factory: Callable[..., Any] = DirectUsbLoopbackRadio,
    table_reader: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    campaign, base_path = load_campaign_manifest(manifest_path)
    ready = load_manifest(ready_manifest_path)
    _firmware_matches(campaign, ready)
    serials = serials_from_ready_manifest(ready_manifest_path)
    if len(serials) != int(campaign["expected-radios"]):
        raise CampaignError(
            f"ready manifest has {len(serials)} radios; "
            f"expected {campaign['expected-radios']}"
        )
    _, base_config = load_calibration_document(base_path)
    if table_reader is None:
        if os.geteuid() != 0:
            raise CampaignError(
                "gain-table audit requires root/CAP_NET_ADMIN for the "
                "serial-isolated Pluto network namespaces; rerun the audit "
                "command with sudo -E"
            )
        manager = MultiPlutoFirmwareManager(
            image=Path("/dev/null"),
            image_sha256="0" * 64,
            ssh_config=(REPO_ROOT / "data_collection/rover/rover_v3.1/ssh_config"),
            ssh_password=os.environ.get("SPF_PLUTO_SSH_PASSWORD", "analog"),
            state_root=Path("/run/spf/passive-gain-table-audit-unused"),
            expected_count=len(serials),
        )
        remote_command = r"""
device=
for candidate in /sys/bus/iio/devices/iio:device*; do
    if test "$(cat "$candidate/name" 2>/dev/null)" = ad9361-phy; then
        device=$candidate
        break
    fi
done
test -n "$device"
dd if="$device/gain_table_config" bs=4096 count=1 2>/dev/null
"""

        def table_reader(serial: str) -> str:
            return manager._ssh(serial, remote_command, timeout=15).stdout

    radio_rows = []
    failures = []
    for serial in serials:
        bands = []
        with radio_factory(serial, base_config) as radio:
            radio.stop_tone()
            for expected in campaign["gain-table-audit"]["bands"]:
                radio.configure_frequency(
                    int(expected["probe-frequency-hz"]), start_tone=False
                )
                # The normal Python-libiio attribute accessor uses a 1 KiB
                # buffer and returns EIO for this roughly 1.8 KiB binary
                # attribute. Read the driver's bounded 4 KiB sysfs attribute
                # locally on the selected Pluto instead.
                actual = parse_gain_table_config(table_reader(serial))
                checks = {
                    "type": actual["type"] == expected["expected-type"],
                    "start_hz": actual["start_hz"] == expected["expected-start-hz"],
                    "end_hz": actual["end_hz"] == expected["expected-end-hz"],
                    "row_count": actual["row_count"] == expected["expected-rows"],
                    "table_sha256": (
                        actual["table_sha256"] == expected["expected-table-sha256"]
                    ),
                }
                if not all(checks.values()):
                    failures.append(
                        {
                            "serial": serial,
                            "band": expected["name"],
                            "failed_checks": [
                                key for key, passed in checks.items() if not passed
                            ],
                        }
                    )
                bands.append(
                    {
                        "name": expected["name"],
                        "probe_frequency_hz": expected["probe-frequency-hz"],
                        "type": actual["type"],
                        "start_hz": actual["start_hz"],
                        "end_hz": actual["end_hz"],
                        "row_count": actual["row_count"],
                        "table_sha256": actual["table_sha256"],
                        "rows": actual["rows"],
                        "checks": checks,
                    }
                )
        radio_rows.append({"serial": serial, "bands": bands})
    result = {
        "schema": AUDIT_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "status": "pass" if not failures else "fail",
        "passive_tx": True,
        "firmware_git_sha": ready["firmware"]["firmware_git_sha"],
        "ready_manifest_path": str(Path(ready_manifest_path).resolve()),
        "ready_manifest_sha256": _sha256_file(Path(ready_manifest_path).resolve()),
        "radios": radio_rows,
        "failures": failures,
        "created_at_unix_ns": time.time_ns(),
    }
    output_path = Path(output_path)
    _write_json_atomic(output_path, result)
    # The audit is normally run through sudo for network-namespace isolation.
    # Keep its non-secret provenance readable by the unprivileged campaign
    # runner that consumes it afterward.
    output_path.chmod(0o644)
    return result


def prepare_campaign_radios(
    manifest_path: Path,
    *,
    preparation_config_path: Path = DEFAULT_PREPARATION_CONFIG,
    rover_id: int = 1,
    ready_manifest_path: Path = DEFAULT_READY_MANIFEST,
    python: Path | None = None,
) -> dict[str, Any]:
    campaign, _ = load_campaign_manifest(manifest_path)
    import sys

    _prepare_radios(
        preparation_config_path=Path(preparation_config_path).resolve(),
        rover_id=rover_id,
        python=Path(sys.executable if python is None else python).absolute(),
        ready_manifest_path=Path(ready_manifest_path).resolve(),
    )
    ready = load_manifest(ready_manifest_path)
    _firmware_matches(campaign, ready)
    serials = serials_from_ready_manifest(ready_manifest_path)
    if len(serials) != int(campaign["expected-radios"]):
        raise CampaignError("prepared radio count does not match campaign")
    return {
        "status": "pass",
        "radio_serials": list(serials),
        "ready_manifest": str(Path(ready_manifest_path).resolve()),
    }


def _rendered_stage(output_root: Path, stage_id: str) -> tuple[dict, dict]:
    plan_path = Path(output_root) / "campaign_plan.json"
    if not plan_path.is_file():
        raise CampaignError("render the campaign before running a stage")
    plan = json.loads(plan_path.read_text())
    matches = [stage for stage in plan["stages"] if stage["id"] == stage_id]
    if len(matches) != 1:
        raise CampaignError(f"unknown campaign stage: {stage_id}")
    return plan, matches[0]


def approve_stage(
    manifest_path: Path,
    output_root: Path,
    *,
    stage_id: str,
    operator: str,
    note: str,
) -> dict[str, Any]:
    campaign, _ = load_campaign_manifest(manifest_path)
    stage = next((row for row in campaign["stages"] if row["id"] == stage_id), None)
    if stage is None:
        raise CampaignError(f"unknown stage: {stage_id}")
    checkpoint = stage.get("operator-checkpoint")
    if not checkpoint:
        raise CampaignError(f"{stage_id} has no operator checkpoint")
    if not operator.strip() or not note.strip():
        raise ValueError("operator and note must be non-empty")
    approval = {
        "stage": stage_id,
        "operator": operator.strip(),
        "note": note.strip(),
        "expected_checkpoint": checkpoint,
        "approved_at_unix_ns": time.time_ns(),
    }
    _write_json_atomic(Path(output_root) / "approvals" / f"{stage_id}.json", approval)
    return approval


def _quality_failure_is_waivable(result: dict[str, Any]) -> bool:
    if result.get("status") != "failed":
        return False
    if result.get("capture", {}).get("status") != "complete":
        return False
    validations = result.get("validations")
    if not isinstance(validations, dict) or not validations:
        return False
    statuses = [validation.get("status") for validation in validations.values()]
    return "fail_quality" in statuses and all(
        status in {"pass", "fail_quality"} for status in statuses
    )


def waive_stage_quality_failure(
    output_root: Path,
    *,
    stage_id: str,
    operator: str,
    note: str,
) -> dict[str, Any]:
    if not operator.strip() or not note.strip():
        raise ValueError("operator and note must be non-empty")
    _, stage = _rendered_stage(output_root, stage_id)
    result_path = Path(output_root) / "stages" / stage["id"] / "stage_result.json"
    if not result_path.is_file():
        raise CampaignError(f"{stage_id} has no completed stage result to waive")
    result = json.loads(result_path.read_text())
    if not _quality_failure_is_waivable(result):
        raise CampaignError(
            f"{stage_id} is not a complete capture with only quality failures"
        )
    waiver = {
        "schema": QUALITY_WAIVER_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "stage": stage_id,
        "operator": operator.strip(),
        "note": note.strip(),
        "stage_result_sha256": _sha256_file(result_path),
        "validation_statuses": {
            serial: validation["status"]
            for serial, validation in result["validations"].items()
        },
        "waived_at_unix_ns": time.time_ns(),
    }
    _write_json_atomic(Path(output_root) / "waivers" / f"{stage_id}.json", waiver)
    return waiver


def _quality_waiver_is_valid(
    output_root: Path, stage_id: str, result_path: Path
) -> bool:
    waiver_path = Path(output_root) / "waivers" / f"{stage_id}.json"
    if not waiver_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    waiver = json.loads(waiver_path.read_text())
    return bool(
        _quality_failure_is_waivable(result)
        and waiver.get("schema") == QUALITY_WAIVER_SCHEMA
        and waiver.get("stage") == stage_id
        and waiver.get("stage_result_sha256") == _sha256_file(result_path)
    )


def _require_stage_gates(
    manifest_path: Path,
    output_root: Path,
    plan: dict,
    stage: dict,
) -> None:
    campaign, _ = load_campaign_manifest(manifest_path)
    audit_path = Path(output_root) / "gain_table_audit.json"
    if (
        not audit_path.is_file()
        or json.loads(audit_path.read_text()).get("status") != "pass"
    ):
        raise CampaignError("a passing gain-table audit is required")
    for requirement in stage["requires"]:
        result_path = Path(output_root) / "stages" / requirement / "stage_result.json"
        if not result_path.is_file():
            raise CampaignError(f"{stage['id']} requires completed stage {requirement}")
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete" and not _quality_waiver_is_valid(
            output_root, requirement, result_path
        ):
            raise CampaignError(f"required stage {requirement} is not complete")
    if stage.get("operator_checkpoint"):
        approval = Path(output_root) / "approvals" / f"{stage['id']}.json"
        if not approval.is_file():
            raise CampaignError(f"{stage['id']} requires an operator approval")
    rate_gate = plan["rate_gate"]
    if stage["id"] != rate_gate["stage"]:
        pilot_path = (
            Path(output_root) / "stages" / rate_gate["stage"] / "stage_result.json"
        )
        if not pilot_path.is_file():
            raise CampaignError("rate pilot has not completed")
        pilot = json.loads(pilot_path.read_text())
        if not pilot.get("rate_gate_passed"):
            raise CampaignError("rate pilot failed; campaign stops before A")
    timing = stage.get("minimum_hours_after_stage_start")
    if timing:
        source_path = (
            Path(output_root) / "stages" / timing["stage"] / "stage_result.json"
        )
        source = json.loads(source_path.read_text())
        required_ns = int(float(timing["hours"]) * 3600 * 1e9)
        if time.time_ns() - int(source["started_at_unix_ns"]) < required_ns:
            raise CampaignError(
                f"{stage['id']} must wait {timing['hours']} h after "
                f"{timing['stage']} started"
            )


def run_campaign_stage(
    manifest_path: Path,
    output_root: Path,
    *,
    stage_id: str,
    ready_manifest_path: Path = DEFAULT_READY_MANIFEST,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    plan, stage = _rendered_stage(output_root, stage_id)
    _require_stage_gates(manifest_path, output_root, plan, stage)
    campaign, _ = load_campaign_manifest(manifest_path)
    ready = load_manifest(ready_manifest_path)
    _firmware_matches(campaign, ready)
    serials = serials_from_ready_manifest(ready_manifest_path)
    if len(serials) != plan["expected_radios"]:
        raise CampaignError("ready radio count changed after campaign rendering")
    config_path = Path(stage["config_path"])
    if _sha256_file(config_path) != stage["config_sha256"]:
        raise CampaignError(f"{stage_id}: rendered config changed on disk")
    _, config = load_calibration_document(config_path)
    stage_root = output_root / "stages" / stage_id
    started_ns = time.time_ns()
    started = time.monotonic()
    capture = run_calibration(
        config_path=config_path,
        output_dir=stage_root,
        ready_manifest_path=ready_manifest_path,
        serials=serials,
    )
    validations = {}
    validation_statuses = []
    for serial in serials:
        validation = validate_dataset(
            stage_root / serial / "calibration.v7.zarr",
            config=config,
            expected_serial=serial,
            recompute_iq=True,
        )
        write_validation_report(stage_root / serial / "validation.json", validation)
        validation_statuses.append(validation["status"])
        validations[serial] = {
            key: value for key, value in validation.items() if key != "cells"
        }
    elapsed = time.monotonic() - started
    completed = int(capture["completed_measurements"])
    status = "complete"
    if capture["status"] != "complete" or "partial" in validation_statuses:
        status = "failed"
    if (
        any(value == "fail_quality" for value in validation_statuses)
        and not stage["allow_quality_failure"]
    ):
        status = "failed"
    result = {
        "schema": STAGE_RESULT_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "stage": stage_id,
        "status": status,
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": time.time_ns(),
        "elapsed_seconds": elapsed,
        "seconds_per_recorded_frame": elapsed / completed if completed else None,
        "capture": capture,
        "validations": validations,
    }
    if stage_id == plan["rate_gate"]["stage"]:
        limit = float(plan["rate_gate"]["maximum-seconds-per-recorded-frame"])
        result["rate_gate_limit_seconds_per_frame"] = limit
        result["rate_gate_passed"] = bool(
            result["seconds_per_recorded_frame"] is not None
            and result["seconds_per_recorded_frame"] <= limit
        )
        if not result["rate_gate_passed"]:
            result["status"] = "failed_rate_gate"
    _write_json_atomic(stage_root / "stage_result.json", result)
    return result


def campaign_status(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    plan_path = Path(output_root) / "campaign_plan.json"
    plan = (
        json.loads(plan_path.read_text())
        if plan_path.is_file()
        else render_campaign(manifest_path, output_root)
    )
    audit_path = Path(output_root) / "gain_table_audit.json"
    audit_status = (
        json.loads(audit_path.read_text()).get("status")
        if audit_path.is_file()
        else "missing"
    )
    stages = []
    for stage in plan["stages"]:
        result_path = Path(output_root) / "stages" / stage["id"] / "stage_result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        approval = Path(output_root) / "approvals" / f"{stage['id']}.json"
        stages.append(
            {
                "id": stage["id"],
                "status": result.get("status", "pending"),
                "approved": approval.is_file()
                if stage["operator_checkpoint"]
                else None,
                "quality_waived": (
                    _quality_waiver_is_valid(output_root, stage["id"], result_path)
                    if result_path.is_file()
                    else False
                ),
                "seconds_per_recorded_frame": result.get("seconds_per_recorded_frame"),
            }
        )
    return {"gain_table_audit": audit_status, "stages": stages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--seconds-per-frame", type=float)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--preparation-config", type=Path, default=DEFAULT_PREPARATION_CONFIG
    )
    prepare.add_argument("--rover-id", type=int, default=1)
    prepare.add_argument("--ready-manifest", type=Path, default=DEFAULT_READY_MANIFEST)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--ready-manifest", type=Path, default=DEFAULT_READY_MANIFEST)
    audit.add_argument("--output", type=Path, required=True)

    approve = subparsers.add_parser("approve")
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--stage", required=True)
    approve.add_argument("--operator", required=True)
    approve.add_argument("--note", required=True)

    waive = subparsers.add_parser("waive-quality")
    waive.add_argument("--output", type=Path, required=True)
    waive.add_argument("--stage", required=True)
    waive.add_argument("--operator", required=True)
    waive.add_argument("--note", required=True)

    run = subparsers.add_parser("run-stage")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--ready-manifest", type=Path, default=DEFAULT_READY_MANIFEST)

    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "render":
            result = render_campaign(
                args.manifest, args.output, seconds_per_frame=args.seconds_per_frame
            )
        elif args.command == "prepare":
            result = prepare_campaign_radios(
                args.manifest,
                preparation_config_path=args.preparation_config,
                rover_id=args.rover_id,
                ready_manifest_path=args.ready_manifest,
            )
        elif args.command == "audit":
            result = audit_gain_tables(
                args.manifest,
                ready_manifest_path=args.ready_manifest,
                output_path=args.output,
            )
        elif args.command == "approve":
            result = approve_stage(
                args.manifest,
                args.output,
                stage_id=args.stage,
                operator=args.operator,
                note=args.note,
            )
        elif args.command == "waive-quality":
            result = waive_stage_quality_failure(
                args.output,
                stage_id=args.stage,
                operator=args.operator,
                note=args.note,
            )
        elif args.command == "run-stage":
            result = run_campaign_stage(
                args.manifest,
                args.output,
                stage_id=args.stage,
                ready_manifest_path=args.ready_manifest,
            )
        else:
            result = campaign_status(args.manifest, args.output)
    except CampaignError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1

    display = result
    if args.command == "audit":
        display = {
            "status": result["status"],
            "output": str(Path(args.output).resolve()),
            "radios": [
                {
                    "serial": radio["serial"],
                    "bands": [
                        {
                            "name": band["name"],
                            "row_count": band["row_count"],
                            "table_sha256": band["table_sha256"],
                            "passed": all(band["checks"].values()),
                        }
                        for band in radio["bands"]
                    ],
                }
                for radio in result["radios"]
            ],
        }
    print(json.dumps(display, indent=2, sort_keys=True))
    return (
        0
        if result.get("status", "pass") not in {"fail", "failed", "failed_rate_gate"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
