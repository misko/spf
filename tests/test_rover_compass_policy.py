import math
import json

import pytest

from spf.mavlink.compass_policy import (
    DEFAULT_COMPASS_CAL_FIT,
    EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
    UnsafeCompassRepairError,
    decode_compass_device_id,
    evaluate_compass_policy,
    format_compass_inventory,
    main,
    parse_parameter_file,
    plan_external_compass_repairs,
)


def _healthy_params(*, external_slot=1):
    params = {
        "COMPASS_ENABLE": 1,
        "COMPASS_CAL_FIT": DEFAULT_COMPASS_CAL_FIT,
        "COMPASS_DISBLMSK": 0,
        "COMPASS_OFFS_MAX": 1800,
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


def _rover2_params_with_stale_internal_priority():
    """Match Rover 2 when its disabled LSM303D is not probed this boot."""
    params = _healthy_params()
    params.update(
        {
            "COMPASS_DEV_ID2": 658945,
            "COMPASS_DEV_ID3": 0,
            "COMPASS_EXTERNAL": 1,
            "COMPASS_EXTERN2": 0,
            "COMPASS_EXTERN3": 0,
            "COMPASS_USE": 1,
            "COMPASS_USE2": 0,
            "COMPASS_USE3": 0,
            "COMPASS_PRIO1_ID": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
            "COMPASS_PRIO2_ID": 658945,
            "COMPASS_PRIO3_ID": 131594,
        }
    )
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


def test_warns_for_large_external_offsets_within_configured_limit():
    params = _healthy_params()
    params.update(
        {
            "COMPASS_OFS_X": -147.5,
            "COMPASS_OFS_Y": -501.7,
            "COMPASS_OFS_Z": 639.4,
        }
    )

    report = evaluate_compass_policy(params)

    assert report.ok, report.errors
    assert report.external_compass.offset_norm_mg > 800
    assert report.external_offset_limit_mg == 1800
    assert any(
        "exceeds the preferred 500.0 mG" in warning for warning in report.warnings
    )


def test_rejects_external_offsets_above_configured_limit():
    params = _healthy_params()
    params.update(
        {
            "COMPASS_OFFS_MAX": 1000,
            "COMPASS_OFS_X": 800,
            "COMPASS_OFS_Y": 700,
            "COMPASS_OFS_Z": 600,
        }
    )

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any(
        "exceeds configured COMPASS_OFFS_MAX=1000.0 mG" in error
        for error in report.errors
    )


@pytest.mark.parametrize("configured_limit", [499, 3001])
def test_rejects_invalid_configured_offset_limit(configured_limit):
    params = _healthy_params()
    params["COMPASS_OFFS_MAX"] = configured_limit

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any(
        "expected an ArduPilot limit from 500 to 3000 mG" in error
        for error in report.errors
    )


def test_rejects_permissive_calibration_and_duplicate_priorities():
    params = _healthy_params()
    params["COMPASS_CAL_FIT"] = 100
    params["COMPASS_PRIO2_ID"] = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any("COMPASS_CAL_FIT=100" in error for error in report.errors)
    assert any("priority IDs are not unique" in error for error in report.errors)


def test_rejects_duplicate_detected_full_device_ids():
    params = _healthy_params()
    params["COMPASS_DEV_ID2"] = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any("appears in multiple slots: [1, 2]" in error for error in report.errors)


def test_warns_for_stale_disabled_secondary_priority():
    params = _rover2_params_with_stale_internal_priority()

    report = evaluate_compass_policy(params)

    assert report.ok, report.errors
    assert any(
        "compass priority 3 device ID 131594 is not detected this boot" in warning
        for warning in report.warnings
    )


def test_warns_for_stale_disabled_second_priority():
    params = _rover2_params_with_stale_internal_priority()
    params.update(
        {
            "COMPASS_PRIO2_ID": 131594,
            "COMPASS_PRIO3_ID": 658945,
        }
    )

    report = evaluate_compass_policy(params)

    assert report.ok, report.errors
    assert any(
        "compass priority 2 device ID 131594 is not detected this boot" in warning
        for warning in report.warnings
    )


def test_rejects_stale_primary_priority():
    params = _rover2_params_with_stale_internal_priority()
    params.update(
        {
            "COMPASS_PRIO1_ID": 424242,
            "COMPASS_PRIO2_ID": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
        }
    )

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert any(
        "compass priority 1 device ID 424242 is not detected this boot" in error
        for error in report.errors
    )


def test_device_id_decode_and_inventory_include_bus_and_priorities():
    assert decode_compass_device_id(658953) == {
        "bus_type": 1,
        "bus": 1,
        "address": 14,
        "device_type": 10,
    }
    report = evaluate_compass_policy(_healthy_params())

    inventory = format_compass_inventory(report)

    assert inventory[0] == "Compass priorities: PRIO1=658953 PRIO2=131594 PRIO3=0"
    assert "device_id=658953" in inventory[1]
    assert "external=True use_for_yaw=True" in inventory[1]
    assert "bus_type=1 bus=1 address=14 device_type=10" in inventory[1]


def test_repair_plan_prioritizes_and_exclusively_uses_external_compass():
    params = _healthy_params(external_slot=2)
    params.update(
        {
            "COMPASS_PRIO1_ID": 131594,
            "COMPASS_PRIO2_ID": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
            "COMPASS_USE": 1,
            "COMPASS_USE2": 0,
        }
    )

    assert plan_external_compass_repairs(params) == {
        "COMPASS_PRIO1_ID": EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
        "COMPASS_PRIO2_ID": 131594,
        "COMPASS_USE": 0,
        "COMPASS_USE2": 1,
    }


def test_repair_plan_refuses_duplicate_detected_ids():
    params = _healthy_params()
    params["COMPASS_DEV_ID2"] = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID

    with pytest.raises(UnsafeCompassRepairError, match="duplicate detected"):
        plan_external_compass_repairs(params)


def test_repair_plan_preserves_stale_disabled_secondary_priority():
    params = _rover2_params_with_stale_internal_priority()

    assert plan_external_compass_repairs(params) == {}


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


def test_cli_accepts_rover2_stale_secondary_and_records_warning(tmp_path, capsys):
    params_path = tmp_path / "rover2.params"
    params_path.write_text(
        "\n".join(
            f"{key} {value}"
            for key, value in _rover2_params_with_stale_internal_priority().items()
        )
    )
    report_path = tmp_path / "report.json"

    assert main([str(params_path), "--json-output", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    output = capsys.readouterr().out

    assert report["ok"] is True
    assert any(
        "priority 3 device ID 131594" in warning for warning in report["warnings"]
    )
    assert "WARNING: compass priority 3 device ID 131594" in output
    assert "PASS: external GPS compass" in output


# ------------------------------------------------- retryable classification ---
#
# Rover 4, 2026-08-06: the external IST8310 on the GPS mast intermittently fails
# to appear on the FC's I2C bus. A reboot re-probes and it usually returns, so
# drone_run.sh retries that case. It must NEVER retry a misconfiguration, which
# reads identically after any number of reboots.


def _external_absent_params():
    """The external compass vanished from the bus; its slot is still enabled."""
    params = _healthy_params()
    params["COMPASS_DEV_ID"] = 0  # not probed this boot
    return params


def test_a_healthy_vehicle_is_not_retryable_because_it_has_not_failed():
    report = evaluate_compass_policy(_healthy_params())
    assert report.ok
    assert not report.retryable


def test_a_missing_external_compass_is_retryable():
    report = evaluate_compass_policy(_external_absent_params())

    assert not report.ok
    assert report.retryable, report.errors
    # every error must be an absence error, or the classification is unsound
    assert len(report.absence_errors) == len(report.errors)
    assert any("was not detected as external" in e for e in report.absence_errors)


def test_a_misconfiguration_is_never_retryable():
    """A nonzero driver mask survives every reboot; retrying would bury it."""
    params = _healthy_params()
    params["COMPASS_DISBLMSK"] = 4

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert not report.retryable


def test_an_absent_compass_plus_a_misconfiguration_is_not_retryable():
    """Fail closed: one non-absence error disqualifies the whole verdict."""
    params = _external_absent_params()
    params["COMPASS_DISBLMSK"] = 4

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert report.absence_errors, "the absence should still be recognised"
    assert not report.retryable, "but a config error must veto the retry"


def test_an_empty_enabled_slot_is_a_config_error_while_the_external_is_present():
    """Only the external's OWN empty slot is an absence symptom."""
    params = _healthy_params()
    params["COMPASS_USE3"] = 1  # slot 3 enabled for a compass never fitted

    report = evaluate_compass_policy(params)

    assert not report.ok
    assert not report.retryable, report.errors


def test_retryable_is_exposed_in_the_json_the_boot_path_reads():
    report = evaluate_compass_policy(_external_absent_params())
    payload = report.to_dict()

    assert payload["retryable"] is True
    assert payload["absence_errors"]
    assert evaluate_compass_policy(_healthy_params()).to_dict()["retryable"] is False
