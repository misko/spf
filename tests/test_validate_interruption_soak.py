import csv
import json
from pathlib import Path

import pytest

from spf.scripts.validate_interruption_soak import validate_soak


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_round(root: Path, signal_name="sigkill", threshold=4):
    with (root / "rounds.tsv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            delimiter="\t",
            fieldnames=(
                "round",
                "cases",
                "started_unix",
                "finished_unix",
                "status",
                "artifact_kib",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "round": 1,
                "cases": f"{signal_name}:{threshold}",
                "started_unix": 1,
                "finished_unix": 2,
                "status": 0,
                "artifact_kib": 3,
            }
        )
    campaign = root / "round-001" / "date_rover2"
    campaign.mkdir(parents=True)
    (campaign / "PASS").write_text("PASS\n")
    case = campaign / f"case-01-{signal_name}-{threshold}"
    _write_json(
        case / "reports" / f"interruption-{signal_name}-{threshold}-records.json",
        {
            "status": "pass",
            "signal": signal_name,
            "minimum_records": threshold,
            "capture_status": "in_progress" if signal_name == "sigkill" else "incomplete",
            "return_code": {"sigkill": -9, "sigint": 130, "sigterm": 143}[signal_name],
            "committed_after_interrupt": [threshold],
            "exit_seconds": 0.2,
            "serials": ["radio-a"],
            "release_probe_sessions": {"radio-a": 1},
        },
    )
    (case / "dmesg-delta.txt").write_text("")
    _write_json(
        campaign / "clean-recovery" / "validation.json",
        {
            "status": "pass",
            "data_version": 7,
            "receiver_count": 1,
            "receivers": {"r0": {"serial": "radio-a", "frames": 100}},
        },
    )


def test_validate_soak_summarizes_complete_round(tmp_path):
    _make_round(tmp_path)
    (tmp_path / "PASS").write_text("PASS\n")
    (tmp_path / "result.env").write_text("rounds_completed=1\n")

    result = validate_soak(
        tmp_path, expected_receivers=1, minimum_rounds=1, require_complete=True
    )

    assert result["status"] == "pass"
    assert result["signals"] == {"sigint": 0, "sigterm": 0, "sigkill": 1}
    assert result["interruption_committed_frames"] == 4
    assert result["clean_recovery_frames"] == 100
    assert result["serials"] == ["radio-a"]
    assert result["maximum_release_probe_sessions"] == 1
    assert result["strictly_revalidated_clean_captures"] == 0


def test_validate_soak_strictly_revalidates_every_clean_capture(
    tmp_path, monkeypatch
):
    _make_round(tmp_path)
    (tmp_path / "PASS").write_text("PASS\n")
    (tmp_path / "result.env").write_text("rounds_completed=1\n")
    clean_root = next(tmp_path.glob("round-001/*_rover*/clean-recovery"))
    clean_zarr = clean_root / "capture.zarr"
    clean_zarr.mkdir()
    calls = []

    def strict_validate(path, *, expected_frames, expected_receivers):
        calls.append((path, expected_frames, expected_receivers))
        return {
            "status": "pass",
            "receivers": {"r0": {"serial": "radio-a"}},
        }

    monkeypatch.setattr(
        "spf.scripts.validate_interruption_soak._strict_validate_clean_capture",
        strict_validate,
    )

    result = validate_soak(
        tmp_path,
        expected_receivers=1,
        minimum_rounds=1,
        require_complete=True,
        revalidate_clean_zarrs=True,
    )

    assert calls == [(clean_zarr, 100, 1)]
    assert result["strictly_revalidated_clean_captures"] == 1


@pytest.mark.parametrize(
    "release_probe_sessions",
    (
        None,
        {},
        {"different-radio": 1},
        {"radio-a": 0},
        {"radio-a": 4},
        {"radio-a": True},
    ),
)
def test_validate_soak_rejects_invalid_release_probe_evidence(
    tmp_path, release_probe_sessions
):
    _make_round(tmp_path)
    report_path = next(tmp_path.glob("round-*/**/reports/*.json"))
    report = json.loads(report_path.read_text())
    if release_probe_sessions is None:
        report.pop("release_probe_sessions")
    else:
        report["release_probe_sessions"] = release_probe_sessions
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="release probe"):
        validate_soak(
            tmp_path, expected_receivers=1, minimum_rounds=1, require_complete=False
        )


def test_validate_soak_rejects_kernel_usb_error(tmp_path):
    _make_round(tmp_path)
    delta = next(tmp_path.glob("round-*/**/dmesg-delta.txt"))
    delta.write_text("usb 1-1: USB disconnect, device number 9\n")

    with pytest.raises(ValueError, match="kernel USB error"):
        validate_soak(
            tmp_path, expected_receivers=1, minimum_rounds=1, require_complete=False
        )


def test_validate_soak_reports_preserved_failure_before_missing_pass(tmp_path):
    _make_round(tmp_path)
    (tmp_path / "FAILED").write_text("ROUND_FAILED: round=2 status=1\n")

    with pytest.raises(ValueError, match="ROUND_FAILED: round=2 status=1"):
        validate_soak(
            tmp_path, expected_receivers=1, minimum_rounds=1, require_complete=True
        )


def test_validate_soak_reports_missing_pass_cleanly(tmp_path):
    _make_round(tmp_path)

    with pytest.raises(ValueError, match="lacks its exact PASS marker"):
        validate_soak(
            tmp_path, expected_receivers=1, minimum_rounds=1, require_complete=True
        )
