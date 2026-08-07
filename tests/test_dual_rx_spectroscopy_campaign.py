import json
from pathlib import Path

import pytest
import yaml

from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.gain_state_phase_model_v1 import default_tables
from spf.calibrations.dual_rx_gain_frequency.runner import load_calibration_document
from spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign import (
    CampaignError,
    _quality_waiver_is_valid,
    approve_stage,
    audit_gain_tables,
    parse_gain_table_config,
    render_campaign,
    waive_stage_quality_failure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPO_ROOT
    / "spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml"
)
FOLLOWUP_MANIFEST = MANIFEST.with_name("gain_state_followups.yaml")


def test_spectroscopy_campaign_renders_exact_normalized_design(tmp_path):
    plan = render_campaign(MANIFEST, tmp_path, seconds_per_frame=0.53)
    by_id = {stage["id"]: stage for stage in plan["stages"]}

    assert plan["measurements_per_radio"] == 9_918
    assert plan["measurements_all_radios"] == 19_836
    assert by_id["rate_pilot"]["measurements_per_radio"] == 50
    assert by_id["rate_pilot"]["epochs"] == 1
    assert by_id["A"]["frequencies"] == 113
    assert by_id["A"]["gain_pairs"] == 5
    assert by_id["A"]["measurements_per_radio"] == 1_695
    assert by_id["E_tx_0"]["measurements_per_radio"] == 162
    assert by_id["F"]["gains"] == 12
    assert by_id["F"]["gain_pairs"] == 23
    assert by_id["F"]["measurements_per_radio"] == 414
    assert plan["estimated_capture_seconds"] == pytest.approx(19_836 * 0.53)

    muted_path = Path(by_id["E_tx_n80"]["config_path"])
    _, muted = load_calibration_document(muted_path)
    assert muted.tx_gain_policy == "fixed"
    assert muted.tx_gain_db == -80
    assert muted.require_preflight_tone is False
    assert by_id["E_tx_n80"]["allow_quality_failure"] is True

    serialized = json.loads((tmp_path / "campaign_plan.json").read_text())
    assert serialized == plan


def test_gain_state_followups_render_preregistered_design(tmp_path):
    plan = render_campaign(FOLLOWUP_MANIFEST, tmp_path, seconds_per_frame=0.928)
    by_id = {stage["id"]: stage for stage in plan["stages"]}

    assert plan["measurements_per_radio"] == 2_117
    assert plan["measurements_all_radios"] == 4_234
    assert by_id["rate_pilot"]["measurements_per_radio"] == 50
    assert by_id["E_CAL3_PROSPECTIVE_DENSE"]["frequencies"] == 113
    assert by_id["E_CAL3_PROSPECTIVE_DENSE"]["measurements_per_radio"] == 1_695
    assert by_id["E_CAL3_TRAIN_REPEAT"]["frequencies"] == 10
    assert by_id["E_CAL3_TRAIN_REPEAT"]["measurements_per_radio"] == 150
    assert by_id["E_CAL2_LOW"]["measurements_per_radio"] == 117
    assert by_id["E_CAL2_MIDDLE"]["measurements_per_radio"] == 39
    assert by_id["E_CAL2_HIGH"]["measurements_per_radio"] == 66

    contract = plan["analysis_contract"]
    training = set(contract["e-cal3"]["training-frequencies-hz"])
    _, dense = load_calibration_document(
        Path(by_id["E_CAL3_PROSPECTIVE_DENSE"]["config_path"])
    )
    _, repeated = load_calibration_document(
        Path(by_id["E_CAL3_TRAIN_REPEAT"]["config_path"])
    )
    assert training == set(repeated.frequencies_hz)
    assert training < set(dense.frequencies_hz)
    assert len(set(dense.frequencies_hz) - training) == 103
    assert contract["e-cal2"]["expected-measurements-per-radio"] == 222


def test_e_cal2_gain_sets_bracket_every_lna_boundary(tmp_path):
    plan = render_campaign(FOLLOWUP_MANIFEST, tmp_path)
    by_id = {stage["id"]: stage for stage in plan["stages"]}
    tables = default_tables()

    for stage_id, band in (
        ("E_CAL2_LOW", "low"),
        ("E_CAL2_MIDDLE", "middle"),
        ("E_CAL2_HIGH", "high"),
    ):
        _, config = load_calibration_document(Path(by_id[stage_id]["config_path"]))
        scheduled = set(config.gains_db)
        low, high = tables.gain_range_db(band)
        boundaries = []
        previous = tables.state(band, low)
        for gain in range(low + 1, high + 1):
            current = tables.state(band, gain)
            if current.lna != previous.lna:
                boundaries.append((gain - 1, gain))
            previous = current
        assert boundaries
        assert all({left, right} <= scheduled for left, right in boundaries)


def test_campaign_frequency_expansion_fails_on_duplicate_coordinates(
    tmp_path,
):
    document = yaml.safe_load(MANIFEST.read_text())
    document["base-config"] = str(MANIFEST.with_name("spectroscopy_campaign_base.yaml"))
    document["frequency-sets"]["spectroscopy"]["add-hz"].append(1_300_000_000)
    manifest = tmp_path / "campaign.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False))

    with pytest.raises(ValueError, match="duplicate coordinates"):
        render_campaign(manifest, tmp_path / "output")


def test_parse_gain_table_config_hashes_only_three_hardware_bytes_per_row():
    text = """\
<gaintable AD9361 type=FULL dest=3 start=0 end=1300000000>
-1, 0x00, 0x00, 0x20
0, 0x01, 0x02, 0x03
</gaintable>
"""
    parsed = parse_gain_table_config(text)

    assert parsed["device"] == 9361
    assert parsed["type"] == "FULL"
    assert parsed["row_count"] == 2
    assert parsed["rows"][0]["gain_db"] == -1
    # SHA256 of 00 00 20 01 02 03. The gain labels are deliberately excluded.
    assert (
        parsed["table_sha256"]
        == "7d834570045725ae3638789c1c16a40857dd2b47136cd8310471cb3acf1c21f9"
    )


def test_gain_table_parser_rejects_malformed_or_truncated_input():
    with pytest.raises(ValueError, match="closing tag"):
        parse_gain_table_config("<gaintable AD9361 type=FULL dest=3 start=0 end=1>")
    with pytest.raises(ValueError, match="malformed gain table row"):
        parse_gain_table_config(
            "<gaintable AD9361 type=FULL dest=3 start=0 end=1>\n"
            "0, 0x00, 0x00\n"
            "</gaintable>\n"
        )


def test_gain_table_audit_uses_bounded_local_reader(tmp_path, monkeypatch):
    table_text = """\
<gaintable AD9361 type=FULL dest=3 start=0 end=1300000000>
-1, 0x00, 0x00, 0x20
0, 0x01, 0x02, 0x03
</gaintable>
"""
    document = yaml.safe_load(MANIFEST.read_text())
    document["base-config"] = str(MANIFEST.with_name("spectroscopy_campaign_base.yaml"))
    document["expected-radios"] = 1
    document["gain-table-audit"]["bands"] = [
        {
            "name": "low",
            "probe-frequency-hz": 1_300_000_000,
            "expected-start-hz": 0,
            "expected-end-hz": 1_300_000_000,
            "expected-type": "FULL",
            "expected-rows": 2,
            "expected-table-sha256": (
                "7d834570045725ae3638789c1c16a40857dd2b47136cd8310471cb3acf1c21f9"
            ),
        }
    ]
    manifest = tmp_path / "campaign.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False))
    ready = {
        "firmware": {
            "firmware_git_sha": document["gain-table-audit"]["firmware-git-sha"],
            "gadget_git_sha": document["gain-table-audit"]["gadget-git-sha"],
            "image_sha256": document["gain-table-audit"]["image-sha256"],
        }
    }
    ready_manifest = tmp_path / "ready.json"
    ready_manifest.write_text(json.dumps(ready))
    monkeypatch.setattr(
        "spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign.load_manifest",
        lambda path: ready,
    )
    monkeypatch.setattr(
        "spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign."
        "serials_from_ready_manifest",
        lambda path: ("SERIAL-A",),
    )

    frequencies = []

    class FakeRadio:
        def __init__(self, serial, config):
            assert serial == "SERIAL-A"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stop_tone(self):
            return None

        def configure_frequency(self, frequency, *, start_tone):
            assert start_tone is False
            frequencies.append(frequency)

    output = tmp_path / "audit.json"
    result = audit_gain_tables(
        manifest,
        ready_manifest_path=ready_manifest,
        output_path=output,
        radio_factory=FakeRadio,
        table_reader=lambda serial: table_text,
    )

    assert result["status"] == "pass"
    assert result["passive_tx"] is True
    assert frequencies == [1_300_000_000]
    assert len(result["radios"][0]["bands"][0]["rows"]) == 2
    assert json.loads(output.read_text()) == result


def test_stage_approval_records_expected_checkpoint(tmp_path):
    approval = approve_stage(
        MANIFEST,
        tmp_path,
        stage_id="B",
        operator="operator-a",
        note="2+3+6 dB pad stack on .17 RX1; .17 RX2 and .18 untouched; 0.9 Nm",
    )

    assert approval["stage"] == "B"
    assert "2+3+6 dB pad stack" in approval["expected_checkpoint"]
    assert json.loads((tmp_path / "approvals/B.json").read_text()) == approval


def test_quality_waiver_is_hash_bound_and_only_accepts_complete_quality_failure(
    tmp_path,
):
    render_campaign(MANIFEST, tmp_path)
    result_path = tmp_path / "stages/B/stage_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "capture": {"status": "complete"},
                "validations": {
                    "treated": {"status": "fail_quality"},
                    "control": {"status": "pass"},
                },
            }
        )
    )

    waiver = waive_stage_quality_failure(
        tmp_path,
        stage_id="B",
        operator="operator-a",
        note="Complete treatment dataset retained despite repeatability failure.",
    )

    assert waiver["validation_statuses"] == {
        "treated": "fail_quality",
        "control": "pass",
    }
    assert _quality_waiver_is_valid(tmp_path, "B", result_path)

    result_path.write_text(result_path.read_text() + "\n")
    assert not _quality_waiver_is_valid(tmp_path, "B", result_path)


@pytest.mark.parametrize(
    "result",
    [
        {
            "status": "failed",
            "capture": {"status": "partial"},
            "validations": {"radio": {"status": "fail_quality"}},
        },
        {
            "status": "failed",
            "capture": {"status": "complete"},
            "validations": {"radio": {"status": "partial"}},
        },
        {
            "status": "complete",
            "capture": {"status": "complete"},
            "validations": {"radio": {"status": "pass"}},
        },
    ],
)
def test_quality_waiver_rejects_non_quality_or_incomplete_results(tmp_path, result):
    render_campaign(MANIFEST, tmp_path)
    result_path = tmp_path / "stages/B/stage_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))

    with pytest.raises(CampaignError, match="only quality failures"):
        waive_stage_quality_failure(
            tmp_path,
            stage_id="B",
            operator="operator-a",
            note="must not be accepted",
        )


def test_single_epoch_rate_pilot_is_valid_but_zero_epochs_are_not():
    CalibrationConfig(
        frequencies_hz=(2_400_000_000,),
        gains_db=(5, 26, 45),
        schedule_design="additive_cross",
        schedule_reference_gain_db=26,
        repetitions=1,
        min_quality_valid_per_cell=1,
    ).validate()

    with pytest.raises(ValueError, match="must be positive"):
        CalibrationConfig(
            frequencies_hz=(2_400_000_000,),
            gains_db=(26,),
            repetitions=0,
        ).validate()
