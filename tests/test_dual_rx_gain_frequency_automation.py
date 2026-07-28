import json
from pathlib import Path

import pytest
import yaml

from spf.calibrations.dual_rx_gain_frequency import automation


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_CONFIG = (
    REPO_ROOT / "spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml"
)
PREPARATION_CONFIG = (
    REPO_ROOT
    / "data_collection/rover/rover_v3.1/capture_configs/rover1_production_v7.yaml"
)


def _stub_successful_hardware(monkeypatch):
    serials = ("SERIAL-A", "SERIAL-B")
    monkeypatch.setattr(automation, "_prepare_radios", lambda **kwargs: None)
    monkeypatch.setattr(automation, "serials_from_ready_manifest", lambda path: serials)
    monkeypatch.setattr(
        automation,
        "probe_loopback",
        lambda **kwargs: {
            "status": "pass",
            "serial": kwargs["serial"],
            "on_off_delta_db": 42.0,
        },
    )
    monkeypatch.setattr(
        automation,
        "run_calibration",
        lambda **kwargs: {
            "status": "complete",
            "radio_serials": list(serials),
            "measurements_per_radio": 324,
            "expected_measurements": 648,
            "completed_measurements": 648,
            "output_dir": str(kwargs["output_dir"]),
        },
    )

    def validate(path, *, config, expected_serial, recompute_iq):
        assert recompute_iq is True
        return {
            "status": "fail_quality" if expected_serial == "SERIAL-B" else "pass",
            "serial": expected_serial,
            "expected_frames": 324,
            "completed_frames": 324,
            "quality_valid_frames": 300,
            "expected_cells": 108,
            "passing_cells": 100,
            "quality_reason_counts": {},
            "cells": [],
        }

    monkeypatch.setattr(automation, "validate_dataset", validate)
    return serials


def test_automate_prepares_probes_captures_validates_and_resumes(tmp_path, monkeypatch):
    serials = _stub_successful_hardware(monkeypatch)
    output = tmp_path / "pilot"
    ready = tmp_path / "ready.json"

    result = automation.run_automated_calibration(
        config_path=CALIBRATION_CONFIG,
        preparation_config_path=PREPARATION_CONFIG,
        output_dir=output,
        ready_manifest_path=ready,
        expected_radios=2,
    )

    assert result["status"] == "complete"
    assert result["quality_review_required"] is True
    assert result["radio_serials"] == list(serials)
    plan = json.loads((output / "automation_plan.json").read_text())
    assert plan["radio_serials"] == list(serials)
    assert plan["firmware"]["boot-mode"] == "ram"
    for serial in serials:
        assert json.loads((output / serial / "probe.json").read_text())["status"] == (
            "pass"
        )
        assert (output / serial / "validation.json").is_file()

    resumed = automation.run_automated_calibration(
        config_path=CALIBRATION_CONFIG,
        preparation_config_path=PREPARATION_CONFIG,
        output_dir=output,
        ready_manifest_path=ready,
        expected_radios=2,
        resume=True,
    )
    assert resumed["status"] == "complete"


def test_automate_refuses_existing_output_without_resume(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="pass --resume"):
        automation.run_automated_calibration(
            config_path=CALIBRATION_CONFIG,
            preparation_config_path=PREPARATION_CONFIG,
            output_dir=output,
        )


def test_automate_refuses_firmware_mismatch_before_preparation(tmp_path, monkeypatch):
    document = yaml.safe_load(CALIBRATION_CONFIG.read_text())
    document["pluto-firmware"]["image-sha256"] = "d" * 64
    mismatched = tmp_path / "mismatched.yaml"
    mismatched.write_text(yaml.safe_dump(document, sort_keys=False))
    monkeypatch.setattr(
        automation,
        "_prepare_radios",
        lambda **kwargs: pytest.fail("preparation must not run"),
    )

    with pytest.raises(ValueError, match="pin different Pluto firmware"):
        automation.run_automated_calibration(
            config_path=mismatched,
            preparation_config_path=PREPARATION_CONFIG,
            output_dir=tmp_path / "output",
        )


def test_automate_records_preparation_failure(tmp_path, monkeypatch):
    def fail_prepare(**kwargs):
        raise RuntimeError("firmware loader failed")

    monkeypatch.setattr(automation, "_prepare_radios", fail_prepare)
    output = tmp_path / "failed"
    result = automation.run_automated_calibration(
        config_path=CALIBRATION_CONFIG,
        preparation_config_path=PREPARATION_CONFIG,
        output_dir=output,
    )
    assert result["status"] == "failed"
    assert result["failed_stage"] == "prepare"
    assert result["error"] == "firmware loader failed"
    assert json.loads((output / "automation_result.json").read_text()) == result


def test_boot_preparation_accepts_explicit_rover_id_and_ready_path():
    script = (
        REPO_ROOT / "data_collection/rover/rover_v3.1/prepare_direct_usb_boot.sh"
    ).read_text()
    assert 'if [[ -n "${SPF_ROVER_ID:-}" ]]' in script
    assert "SPF_DIRECT_USB_READY_FILE:-/run/spf/direct_usb_ready.json" in script
