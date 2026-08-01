#!/usr/bin/env python3
"""Guarded ArduPilot inspection and compass-calibration CLI.

Exit codes:
    0  Query completed and the requested health/policy check passed.
    1  Query completed and reported an unhealthy/failed state.
    2  Usage, transport, ownership, timeout, or safety failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# When invoked as ``python path/to/ardu_cli.py``, prefer the checkout containing
# this script over any older editable SPF installation in the environment.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymavlink import mavutil

from spf.mavlink.check_prearm import (
    PrearmResult,
    resolve_default_master,
    run_prearm_checks,
)
from spf.mavlink.compass_policy import (
    EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
    UnsafeCompassRepairError,
    evaluate_compass_policy,
    format_compass_inventory,
    parse_parameter_file,
    plan_external_compass_repairs,
)


SERVICE_NAME = "mavlink_controller.service"
SOURCE_SYSTEM = 254

STATUS_MESSAGE_IDS = {
    "SYS_STATUS": mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
    "GPS_RAW_INT": mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
    "EKF_STATUS_REPORT": mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
}

EKF_FLAG_NAMES = {
    1: "attitude",
    2: "velocity_horiz",
    4: "velocity_vert",
    8: "pos_horiz_rel",
    16: "pos_horiz_abs",
    32: "pos_vert_abs",
    64: "pos_vert_agl",
    128: "const_pos_mode",
    256: "pred_pos_horiz_rel",
    512: "pred_pos_horiz_abs",
    1024: "uninitialized",
}

GPS_FIX_NAMES = {
    0: "NO_GPS",
    1: "NO_FIX",
    2: "2D_FIX",
    3: "3D_FIX",
    4: "DGPS",
    5: "RTK_FLOAT",
    6: "RTK_FIXED",
    7: "STATIC",
    8: "PPP",
}

MAGCAL_COMMANDS = {
    "start": mavutil.mavlink.MAV_CMD_DO_START_MAG_CAL,
    "accept": mavutil.mavlink.MAV_CMD_DO_ACCEPT_MAG_CAL,
    "cancel": mavutil.mavlink.MAV_CMD_DO_CANCEL_MAG_CAL,
}


class CliError(RuntimeError):
    """Expected operational error that should produce exit status 2."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(report: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _is_direct_serial(master: str | None) -> bool:
    if master is None:
        return True
    lowered = master.lower()
    return master.startswith("/dev/") or lowered.startswith("serial:")


def _service_is_active() -> bool:
    try:
        return (
            subprocess.run(
                ["systemctl", "is-active", "--quiet", SERVICE_NAME],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except OSError:
        return False


def _connect(args: argparse.Namespace):
    if (
        _is_direct_serial(args.master)
        and not args.allow_active_service
        and _service_is_active()
    ):
        raise CliError(
            f"{SERVICE_NAME} is active and may own the ArduPilot serial link; "
            f"stop it first or use a MAVLink network fan-out. "
            f"--allow-active-service is an explicit expert override."
        )

    try:
        master = args.master or resolve_default_master()
    except RuntimeError as error:
        raise CliError(str(error)) from error

    try:
        connection = mavutil.mavlink_connection(
            master,
            baud=args.baud,
            source_system=SOURCE_SYSTEM,
            dialect="ardupilotmega",
        )
        heartbeat = connection.wait_heartbeat(timeout=args.heartbeat_timeout)
    except Exception as error:
        raise CliError(f"MAVLink connection to {master} failed: {error}") from error
    if heartbeat is None:
        raise CliError(f"no ArduPilot heartbeat received from {master}")
    return connection, heartbeat, master


def _armed(heartbeat) -> bool:
    return bool(int(heartbeat.base_mode) & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def _enum_name(enum_name: str, value: int) -> str:
    enum = mavutil.mavlink.enums.get(enum_name, {})
    entry = enum.get(int(value))
    return entry.name if entry is not None else f"UNKNOWN_{value}"


def _mav_result_name(result: int | None) -> str:
    if result is None:
        return "NO_ACK"
    return _enum_name("MAV_RESULT", result)


def _message_dict(message) -> dict[str, Any]:
    if hasattr(message, "to_dict"):
        return _json_safe(message.to_dict())
    return {
        key: _json_safe(value)
        for key, value in vars(message).items()
        if not key.startswith("_")
    }


def _request_message(connection, message_id: int, interval_us: int = 500_000) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def collect_status(connection, heartbeat, timeout_s: float) -> dict[str, Any]:
    """Collect one current heartbeat/GPS/EKF/sensor-health snapshot."""
    for message_id in STATUS_MESSAGE_IDS.values():
        _request_message(connection, message_id)

    received: dict[str, Any] = {}
    deadline = time.monotonic() + timeout_s
    wanted = set(STATUS_MESSAGE_IDS)
    while wanted and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type=list(wanted), blocking=True, timeout=min(0.5, remaining)
        )
        if message is None:
            continue
        message_type = message.get_type()
        received[message_type] = message
        wanted.discard(message_type)

    heartbeat_dict = _message_dict(heartbeat)
    report: dict[str, Any] = {
        "complete": not wanted,
        "missing_messages": sorted(wanted),
        "armed": _armed(heartbeat),
        "mode": mavutil.mode_string_v10(heartbeat),
        "system_status": _enum_name("MAV_STATE", int(heartbeat.system_status)),
        "heartbeat": heartbeat_dict,
        "gps": None,
        "ekf": None,
        "sensors": None,
    }

    gps = received.get("GPS_RAW_INT")
    if gps is not None:
        gps_dict = _message_dict(gps)
        fix_type = int(gps.fix_type)
        report["gps"] = {
            **gps_dict,
            "fix_type_name": GPS_FIX_NAMES.get(fix_type, f"UNKNOWN_{fix_type}"),
            "latitude_deg": int(gps.lat) / 1e7,
            "longitude_deg": int(gps.lon) / 1e7,
            "altitude_m": int(gps.alt) / 1000.0,
        }

    ekf = received.get("EKF_STATUS_REPORT")
    if ekf is not None:
        flags = int(ekf.flags)
        report["ekf"] = {
            **_message_dict(ekf),
            "flag_names": [name for bit, name in EKF_FLAG_NAMES.items() if flags & bit],
        }

    sensors = received.get("SYS_STATUS")
    if sensors is not None:
        present = int(sensors.onboard_control_sensors_present)
        enabled = int(sensors.onboard_control_sensors_enabled)
        healthy = int(sensors.onboard_control_sensors_health)
        report["sensors"] = {
            **_message_dict(sensors),
            "compass_present": bool(
                present & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
            ),
            "compass_enabled": bool(
                enabled & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
            ),
            "compass_healthy": bool(
                healthy & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
            ),
            "gps_healthy": bool(healthy & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS),
            "prearm_healthy": bool(
                healthy & mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
            ),
        }
    return report


def _decode_param_id(param_id: Any) -> str:
    if isinstance(param_id, bytes):
        param_id = param_id.decode("ascii", errors="replace")
    return str(param_id).rstrip("\x00")


def download_parameters(connection, timeout_s: float) -> tuple[dict[str, float], bool]:
    """Download a complete parameter snapshot without writing FC state."""
    while connection.recv_match(blocking=False) is not None:
        pass
    connection.mav.param_request_list_send(
        connection.target_system, connection.target_component
    )

    params: dict[str, float] = {}
    indexes: set[int] = set()
    expected_count: int | None = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type="PARAM_VALUE", blocking=True, timeout=min(0.5, remaining)
        )
        if message is None:
            continue
        params[_decode_param_id(message.param_id)] = float(message.param_value)
        indexes.add(int(message.param_index))
        expected_count = int(message.param_count)
        if expected_count > 0 and len(indexes) >= expected_count:
            return params, True
    return params, False


def _print_status(report: dict[str, Any]) -> None:
    print(
        f"Vehicle: armed={report['armed']} mode={report['mode']} "
        f"state={report['system_status']}"
    )
    gps = report["gps"]
    if gps is None:
        print("GPS: unavailable")
    else:
        print(
            f"GPS: {gps['fix_type_name']} satellites={gps.get('satellites_visible')} "
            f"lat={gps['latitude_deg']:.7f} lon={gps['longitude_deg']:.7f}"
        )
    ekf = report["ekf"]
    if ekf is None:
        print("EKF: unavailable")
    else:
        print(f"EKF: flags={ekf['flags']} ({', '.join(ekf['flag_names'])})")
    sensors = report["sensors"]
    if sensors is None:
        print("Sensors: unavailable")
    else:
        print(
            "Sensors: "
            f"compass_present={sensors['compass_present']} "
            f"compass_enabled={sensors['compass_enabled']} "
            f"compass_healthy={sensors['compass_healthy']} "
            f"gps_healthy={sensors['gps_healthy']} "
            f"prearm_healthy={sensors['prearm_healthy']}"
        )
    if report["missing_messages"]:
        print("UNKNOWN: missing " + ", ".join(report["missing_messages"]))


def command_status(args: argparse.Namespace) -> int:
    connection, heartbeat, master = _connect(args)
    report = collect_status(connection, heartbeat, args.message_timeout)
    report["master"] = master
    if args.json or args.json_output:
        _write_json(report, args.json_output)
    else:
        _print_status(report)
    return 0 if report["complete"] else 2


def _compass_report(params: dict[str, float], expected_device_id: int) -> dict:
    policy = evaluate_compass_policy(
        params, expected_external_device_id=expected_device_id
    )
    return {
        "parameters": {
            key: value
            for key, value in sorted(params.items())
            if key.startswith("COMPASS")
        },
        "policy": policy.to_dict(),
    }


def _print_compass(report: dict[str, Any]) -> None:
    for key, value in report["parameters"].items():
        print(f"{key:<20} {value:g}")
    policy = report["policy"]
    for line in report.get("inventory", []):
        print(line)
    external = policy["external_compass"]
    if external is not None:
        print(
            "External compass: "
            f"slot={external['slot']} device_id={external['device_id']} "
            f"offset_norm_mG={external['offset_norm_mg']:.1f}"
        )
    for warning in policy["warnings"]:
        print(f"WARNING: {warning}")
    for error in policy["errors"]:
        print(f"FAIL: {error}")
    repair = report.get("repair")
    runtime_verification_pending = bool(
        repair is not None and repair.get("reboot_required")
    )
    if policy["ok"] and not runtime_verification_pending:
        print("PASS: external GPS compass is the sole fleet-approved yaw source")
    elif policy["ok"]:
        print("PENDING: stored policy is correct; reboot and re-run compass check")
    if repair is not None:
        if repair["changes"]:
            print("Applied compass priority/use repairs:")
            for key, change in repair["changes"].items():
                print(f"  {key}: {change['before']:g} -> {change['after']:g}")
            if repair["reboot_required"]:
                print("REBOOT REQUIRED: priority changes take effect after FC reboot")
            else:
                print("PASS: yaw-use changes were acknowledged by ArduPilot")
        else:
            print("No compass priority/use repair was necessary")


def apply_parameter_changes(
    connection, params: dict[str, float], changes: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Write and acknowledge a small, prevalidated set of ArduPilot parameters."""
    applied: dict[str, dict[str, float]] = {}
    for name, desired_value in changes.items():
        before = float(params[name])
        acknowledged_value = None
        for _attempt in range(3):
            connection.param_set_send(name.upper(), float(desired_value))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                ack = connection.recv_match(
                    type="PARAM_VALUE",
                    blocking=True,
                    timeout=min(0.25, remaining),
                )
                if (
                    ack is None
                    or _decode_param_id(ack.param_id).upper() != name.upper()
                ):
                    continue
                acknowledged_value = float(ack.param_value)
                break
            if (
                acknowledged_value is not None
                and abs(acknowledged_value - float(desired_value)) <= 1e-6
            ):
                break
        if (
            acknowledged_value is None
            or abs(acknowledged_value - float(desired_value)) > 1e-6
        ):
            raise CliError(f"ArduPilot did not acknowledge {name}={desired_value}")
        params[name] = acknowledged_value
        applied[name] = {"before": before, "after": acknowledged_value}
    return applied


def command_compass(args: argparse.Namespace) -> int:
    if args.repair and not args.yes:
        raise CliError(
            "compass --repair writes persistent priority/use parameters; "
            "repeat with --yes after making the disarmed rover safe"
        )
    if args.params:
        if args.repair:
            raise CliError("--repair requires a live vehicle, not --params")
        try:
            params = parse_parameter_file(args.params)
        except (OSError, ValueError) as error:
            raise CliError(str(error)) from error
        complete = True
        master = None
    else:
        connection, heartbeat, master = _connect(args)
        params, complete = download_parameters(connection, args.parameter_timeout)
        if not params:
            raise CliError("no ArduPilot parameters were received")

    repair = None
    if args.repair:
        if not complete:
            raise CliError("cannot repair from an incomplete parameter snapshot")
        if _armed(heartbeat):
            raise CliError("vehicle is armed; refusing compass repair")
        try:
            planned_changes = plan_external_compass_repairs(
                params,
                expected_external_device_id=args.expected_device_id,
            )
        except UnsafeCompassRepairError as error:
            raise CliError(f"compass repair is unsafe: {error}") from error
        applied_changes = apply_parameter_changes(connection, params, planned_changes)
        reboot_required = any(key.startswith("COMPASS_PRIO") for key in applied_changes)
        repair = {
            "changes": applied_changes,
            "reboot_required": reboot_required,
            "verification": (
                "parameter writes acknowledged; "
                + (
                    "reboot and live re-read required"
                    if reboot_required
                    else "live values acknowledged"
                )
            ),
        }

    report = _compass_report(params, args.expected_device_id)
    policy_object = evaluate_compass_policy(
        params, expected_external_device_id=args.expected_device_id
    )
    report.update(
        {
            "master": master,
            "parameter_download_complete": complete,
            "parameter_count": len(params),
            "inventory": list(format_compass_inventory(policy_object)),
            "repair": repair,
        }
    )
    if args.json or args.json_output:
        _write_json(report, args.json_output)
    else:
        _print_compass(report)
        if not complete:
            print("UNKNOWN: complete parameter snapshot was not received")
    if not complete:
        return 2
    return 0 if report["policy"]["ok"] else 1


def _prearm_to_dict(result: PrearmResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "conclusive": result.conclusive,
        "command_result": result.command_result,
        "command_result_name": _mav_result_name(result.command_result),
        "healthy": result.healthy,
        "messages": list(result.messages),
    }


def command_prearm(args: argparse.Namespace) -> int:
    connection, heartbeat, master = _connect(args)
    if _armed(heartbeat):
        raise CliError(
            "vehicle is armed; refusing to treat pre-arm status as meaningful"
        )
    result = run_prearm_checks(connection, timeout_s=args.result_timeout)
    report = _prearm_to_dict(result)
    report["master"] = master
    report["armed"] = False
    if args.json or args.json_output:
        _write_json(report, args.json_output)
    else:
        print(f"Command result: {report['command_result_name']}")
        print(
            "Pre-arm health: "
            + (
                "PASS"
                if result.healthy is True
                else "FAIL"
                if result.healthy is False
                else "UNKNOWN"
            )
        )
        for message in result.messages:
            print(f"  - {message}")
    if result.passed:
        return 0
    return 1 if result.conclusive else 2


def _wait_command_ack(connection, command: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type="COMMAND_ACK", blocking=True, timeout=min(0.5, remaining)
        )
        if message is not None and int(message.command) == command:
            return message
    return None


def send_magcal_command(
    connection,
    action: str,
    *,
    mask: int = 0,
    retry: bool = False,
    autosave: bool = True,
) -> None:
    command = MAGCAL_COMMANDS[action]
    if action == "start":
        params = (mask, int(retry), int(autosave), 0, 0, 0, 0)
    else:
        # Match MAVProxy's accept/cancel wire convention.
        params = (mask, 0, 1, 0, 0, 0, 0)
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        command,
        0,
        *params,
    )


def monitor_magcal(connection, timeout_s: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type=["MAG_CAL_PROGRESS", "MAG_CAL_REPORT"],
            blocking=True,
            timeout=min(0.5, remaining),
        )
        if message is None:
            continue
        event = _message_dict(message)
        event["message_type"] = message.get_type()
        if "cal_status" in event:
            event["cal_status_name"] = _enum_name(
                "MAG_CAL_STATUS", int(event["cal_status"])
            )
        events.append(event)
    return events


def _print_magcal_events(events: Iterable[dict[str, Any]]) -> None:
    for event in events:
        status = event.get("cal_status_name", event.get("cal_status", "unknown"))
        if event["message_type"] == "MAG_CAL_PROGRESS":
            print(
                f"Compass {event.get('compass_id')}: {status}, "
                f"{event.get('completion_pct')}%"
            )
        else:
            print(
                f"Compass {event.get('compass_id')}: {status}, "
                f"fitness={event.get('fitness')} autosaved={event.get('autosaved')}"
            )


def command_magcal(args: argparse.Namespace) -> int:
    action = args.magcal_action
    mutating = action in MAGCAL_COMMANDS
    if mutating and not args.yes:
        raise CliError(
            f"magcal {action} changes flight-controller calibration state; "
            "repeat with --yes after making the disarmed rover physically safe"
        )

    connection, heartbeat, master = _connect(args)
    if mutating and _armed(heartbeat):
        raise CliError(f"vehicle is armed; refusing magcal {action}")

    ack_report = None
    if mutating:
        send_magcal_command(
            connection,
            action,
            mask=args.mask,
            retry=getattr(args, "retry", False),
            autosave=getattr(args, "autosave", True),
        )
        command = MAGCAL_COMMANDS[action]
        ack = _wait_command_ack(connection, command, args.command_timeout)
        if ack is None:
            raise CliError(f"no COMMAND_ACK received for magcal {action}")
        result = int(ack.result)
        ack_report = {
            "command": command,
            "result": result,
            "result_name": _mav_result_name(result),
        }
        if result not in (
            mavutil.mavlink.MAV_RESULT_ACCEPTED,
            mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
        ):
            report = {
                "action": action,
                "master": master,
                "ack": ack_report,
                "events": [],
            }
            if args.json or args.json_output:
                _write_json(report, args.json_output)
            else:
                print(f"magcal {action}: {ack_report['result_name']}")
            return 1

    monitor_seconds = args.timeout if action == "monitor" else args.monitor_seconds
    events = monitor_magcal(connection, monitor_seconds) if monitor_seconds > 0 else []
    report = {"action": action, "master": master, "ack": ack_report, "events": events}
    if args.json or args.json_output:
        _write_json(report, args.json_output)
    else:
        if ack_report is not None:
            print(f"magcal {action}: {ack_report['result_name']}")
        _print_magcal_events(events)
        if action == "monitor" and not events:
            print("No magnetometer calibration progress/report messages observed")
    return 0 if (action != "monitor" or events) else 1


def _add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--master",
        help=(
            "MAVLink endpoint. Defaults to the single "
            "/dev/serial/by-id/usb-ArduPilot* device."
        ),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--heartbeat-timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-active-service",
        action="store_true",
        help="Expert override for direct-link ownership protection.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument("--json-output", help="Write JSON to this file.")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an SPF rover's ArduPilot and manage compass calibration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status", help="Read arm, mode, GPS, EKF, and sensor-health state."
    )
    _add_connection_options(status)
    status.add_argument("--message-timeout", type=float, default=8.0)
    status.set_defaults(handler=command_status)

    compass = subparsers.add_parser(
        "compass", help="Read COMPASS_* parameters and evaluate fleet policy."
    )
    _add_connection_options(compass)
    compass.add_argument(
        "--params", help="Evaluate a saved parameter file instead of a live vehicle."
    )
    compass.add_argument("--parameter-timeout", type=float, default=30.0)
    compass.add_argument(
        "--expected-device-id",
        type=int,
        default=EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
    )
    compass.add_argument(
        "--repair",
        action="store_true",
        help=(
            "repair priority order and COMPASS_USE* only when exactly one "
            "known external compass is identified"
        ),
    )
    compass.add_argument(
        "--yes",
        action="store_true",
        help="Confirm persistent writes requested by --repair.",
    )
    compass.set_defaults(handler=command_compass)

    prearm = subparsers.add_parser(
        "prearm", help="Run ArduPilot pre-arm checks without attempting to arm."
    )
    _add_connection_options(prearm)
    prearm.add_argument("--result-timeout", type=float, default=8.0)
    prearm.set_defaults(handler=command_prearm)

    magcal = subparsers.add_parser(
        "magcal", help="Start, monitor, accept, or cancel compass calibration."
    )
    magcal_subparsers = magcal.add_subparsers(dest="magcal_action", required=True)
    for action in ("start", "accept", "cancel"):
        action_parser = magcal_subparsers.add_parser(action)
        _add_connection_options(action_parser)
        action_parser.add_argument("--yes", action="store_true")
        action_parser.add_argument(
            "--mask", type=lambda value: int(value, 0), default=0
        )
        action_parser.add_argument("--command-timeout", type=float, default=8.0)
        action_parser.add_argument("--monitor-seconds", type=float, default=0.0)
        if action == "start":
            action_parser.add_argument("--retry", action="store_true")
            action_parser.add_argument(
                "--no-autosave", dest="autosave", action="store_false"
            )
            action_parser.set_defaults(autosave=True)
        action_parser.set_defaults(handler=command_magcal)
    monitor = magcal_subparsers.add_parser("monitor")
    _add_connection_options(monitor)
    monitor.add_argument("--timeout", type=float, default=30.0)
    monitor.set_defaults(handler=command_magcal)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except CliError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
