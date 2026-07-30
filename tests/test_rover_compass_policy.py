import math
import json

from spf.mavlink.compass_policy import (
    DEFAULT_COMPASS_CAL_FIT,
    EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
    evaluate_compass_policy,
    main,
    parse_parameter_file,
)


def _healthy_params(*, external_slot=1):
    params = {
        "COMPASS_ENABLE": 1,
        "COMPASS_CAL_FIT": DEFAULT_COMPASS_CAL_FIT,
        "COMPASS_DISBLMSK": 0,
        "COMPASS_PRIO1_ID": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
        "COMPASS_PRIO2_ID": 131594,
        "COMPASS_PRIO3_ID": 0,
    }
    for slot in (1, 2, 3):
        suffix = "" if slot == 1 else str(slot)
        external_key = "COMPASS_EXTERNAL" if slot == 1 else f"COMPASS_EXTERN{slot}"
        params.update(
            {
                f"COMPASS_DEV_ID{suffix}": 0,
                f"COMPASS_USE{suffix}": 0,
                external_key: 0,
                f"COMPASS_OFS{suffix}_X": 0,
                f"COMPASS_OFS{suffix}_Y": 0,
                f"COMPASS_OFS{suffix}_Z": 0,
            }
        )

    suffix = "" if external_slot == 1 else str(external_slot)
    external_key = (
        "COMPASS_EXTERNAL" if external_slot == 1 else f"COMPASS_EXTERN{external_slot}"
    )
    params.update(
        {
            f"COMPASS_DEV_ID{suffix}": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
            f"COMPASS_USE{suffix}": 1,
            external_key: 1,
            f"COMPASS_OFS{suffix}_X": -19,
            f"COMPASS_OFS{suffix}_Y": 156,
            f"COMPASS_OFS{suffix}_Z": -82,
        }
    )

    internal_slot = 2 if external_slot != 2 else 1
    suffix = "" if internal_slot == 1 else str(internal_slot)
    params[f"COMPASS_DEV_ID{suffix}"] = 131594
    return params


def test_accepts_external_compass_in_any_slot():
    for slot in (1, 2, 3):
        report = evaluate_compass_policy(_healthy_params(external_slot=slot))

        assert report.ok, report.errors
        assert report.external_compass.slot == slot
        assert math.isclose(report.external_compass.offset_norm_mg, 177.26, abs_tol=0.1)


def test_rejects_empty_enabled_slot_and_disabled_external_compass():
    params = _healthy_params(external_slot=3)
    params["COMPASS_USE"] = 1
    params["COMPASS_USE3"] = 0

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert "empty compass slot 1 is enabled for yaw" in report.errors
    assert "external compass slot 3 is disabled for yaw" in report.errors


def test_rejects_large_external_offsets():
    params = _healthy_params()
    params.update(
        {
            "COMPASS_OFS_X": -147.5,
            "COMPASS_OFS_Y": -501.7,
            "COMPASS_OFS_Z": 639.4,
        }
    )

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert report.external_compass.offset_norm_mg > 800
    assert any("exceeds 500.0 mG" in error for error in report.errors)


def test_rejects_permissive_calibration_and_duplicate_priorities():
    params = _healthy_params()
    params["COMPASS_CAL_FIT"] = 100
    params["COMPASS_PRIO2_ID"] = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any("COMPASS_CAL_FIT=100" in error for error in report.errors)
    assert any("priority IDs are not unique" in error for error in report.errors)


def test_rejects_driver_mask_and_wrong_primary_priority():
    params = _healthy_params()
    params["COMPASS_DISBLMSK"] = 1 << 7
    params["COMPASS_PRIO1_ID"] = 131594

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any("COMPASS_DISBLMSK" in error for error in report.errors)
    assert any("priority 1" in error for error in report.errors)


def test_parse_parameter_file_supports_mavproxy_and_csv(tmp_path):
    path = tmp_path / "rover.params"
    path.write_text(
        """
# comment
COMPASS_ENABLE 1
COMPASS_CAL_FIT,16
COMPASS_OFS_X -19.5
""".lstrip()
    )

    assert parse_parameter_file(path) == {
        "COMPASS_ENABLE": 1.0,
        "COMPASS_CAL_FIT": 16.0,
        "COMPASS_OFS_X": -19.5,
    }


def test_cli_writes_machine_readable_report(tmp_path, capsys):
    params_path = tmp_path / "healthy.params"
    params_path.write_text(
        "\n".join(f"{key} {value}" for key, value in _healthy_params().items())
    )
    report_path = tmp_path / "report.json"

    assert main([str(params_path), "--json-output", str(report_path)]) == 0
    report = json.loads(report_path.read_text())

    assert report["ok"] is True
    assert report["external_compass"]["device_id"] == 658953
    assert "PASS: external GPS compass" in capsys.readouterr().out
