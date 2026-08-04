#!/usr/bin/env python3
"""Guarded ArduPilot inspection and calibration CLI.

Exit codes:
    0  Query completed and the requested health/policy check passed.
    1  Query completed and reported an unhealthy/failed state.
    2  Usage, transport, ownership, timeout, or safety failure.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import termios
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

RC_MAX_CHANNELS = 18

# The rover control map, so `rc` names what moved instead of only numbering it.
# CH1-6 are ArduPilot's (rover3_rc_servo_parameters.params); CH7/9/10/12 carry
# RCx_OPTION 0 and are read raw by the Pi in handle_RC_CHANNELS, so they are
# invisible to ArduPilot itself but still visible in this stream.
# Source of truth: ROVER_RUNBOOK.md 3.5.2.
RC_CHANNEL_ROLES = {
    1: "roll/ail   (RCMAP_ROLL)",
    2: "pitch/ele  (RCMAP_PITCH)",
    3: "throttle   (RCMAP_THROTTLE)",
    4: "yaw/rud    (RCMAP_YAW)",
    5: "ARM/DISARM (RC5_OPTION 153, switch SF)",
    6: "scripting1 (RC6_OPTION 300 - inert, no Lua in-tree)",
    7: "reboot FC / kill collector (Pi, switch SD)",
    8: "FLIGHT MODE (MODE_CH 8, switch SA: Manual/RTL/Guided)",
    9: "shutdown Pi (Pi, switch SH)",
    10: "compass calibration (Pi, switch SC)",
    12: "ultrasonic enable/disable (Pi)",
}

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

ACCELCAL_START_COMMAND = mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION
ACCELCAL_POSITION_COMMAND = mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS
ACCELCAL_ACCEPTED_RESULTS = {
    mavutil.mavlink.MAV_RESULT_ACCEPTED,
    mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
}
ACCELCAL_POSES = {
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_LEVEL: (
        "LEVEL",
        "Place the rover normally on a level surface.",
    ),
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_LEFT: (
        "LEFT SIDE",
        "Place the rover on its left side.",
    ),
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_RIGHT: (
        "RIGHT SIDE",
        "Place the rover on its right side.",
    ),
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_NOSEDOWN: (
        "NOSE DOWN",
        "Place the rover nose down.",
    ),
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_NOSEUP: (
        "NOSE UP",
        "Place the rover nose up.",
    ),
    mavutil.mavlink.ACCELCAL_VEHICLE_POS_BACK: (
        "UPSIDE DOWN",
        "Place the rover upside down, on its back.",
    ),
}
ACCELCAL_SUCCESS = mavutil.mavlink.ACCELCAL_VEHICLE_POS_SUCCESS
ACCELCAL_FAILED = mavutil.mavlink.ACCELCAL_VEHICLE_POS_FAILED

CLI_CHEATSHEET = """quick reference:
  # Direct serial has one owner; stop production before using this CLI.
  sudo systemctl stop mavlink_controller.service

  # Read-only inspection.
  python -m spf.ardupilot.ardu_cli status
  python -m spf.ardupilot.ardu_cli compass --parameter-timeout 60
  python -m spf.ardupilot.ardu_cli prearm

  # Repair only unambiguous compass priority/yaw-use settings.
  python -m spf.ardupilot.ardu_cli compass --repair --yes --parameter-timeout 60

  # Calibrate the fleet external compass in slot 1. Progress prints live;
  # 300 seconds is a ceiling and a terminal report exits early.
  python -m spf.ardupilot.ardu_cli magcal start --yes --mask 1 --retry --monitor-seconds 300
  python -m spf.ardupilot.ardu_cli magcal monitor --timeout 10
  python -m spf.ardupilot.ardu_cli magcal cancel --yes --mask 1

  # Full six-position accelerometer calibration. The CLI prints every pose and
  # waits for Enter only after the assembled, disarmed rover is motionless.
  python -m spf.ardupilot.ardu_cli accelcal start --yes

  # Restore production only after compass/prearm checks pass.
  sudo systemctl restart mavlink_controller.service
  journalctl -fu mavlink_controller.service

exit codes:
  0  completed and healthy/passed
  1  completed with an unhealthy/failed result
  2  usage, transport, ownership, timeout, or safety failure
"""


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


def _claim_serial_exclusive(connection, master: str) -> None:
    """Prevent another local process from opening this direct serial TTY."""
    if not _is_direct_serial(master) or not hasattr(connection, "port"):
        return
    try:
        fcntl.ioctl(connection.port.fileno(), termios.TIOCEXCL)
    except Exception as error:
        try:
            connection.close()
        except Exception:
            pass
        raise CliError(
            f"could not claim exclusive MAVLink ownership of {master}: {error}. "
            f"Stop {SERVICE_NAME}, MAVProxy, and other serial readers, then retry."
        ) from error


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
        _claim_serial_exclusive(connection, master)
        heartbeat = connection.wait_heartbeat(timeout=args.heartbeat_timeout)
    except CliError:
        raise
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


def monitor_magcal(
    connection,
    timeout_s: float,
    *,
    on_event=None,
    exit_on_report: bool = True,
) -> list[dict[str, Any]]:
    """Monitor calibration, optionally streaming events as they arrive.

    ``timeout_s`` is a safety ceiling, not a minimum runtime.  A
    ``MAG_CAL_REPORT`` is ArduPilot's terminal calibration result, so the
    normal CLI path returns immediately after receiving it instead of making
    an operator wait for the remainder of a long monitoring window.
    """
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
        if on_event is not None:
            on_event(event)
        if exit_on_report and event["message_type"] == "MAG_CAL_REPORT":
            break
    return events


def _print_magcal_event(event: dict[str, Any]) -> None:
    status = event.get("cal_status_name", event.get("cal_status", "unknown"))
    if event["message_type"] == "MAG_CAL_PROGRESS":
        print(
            f"Compass {event.get('compass_id')}: {status}, "
            f"{event.get('completion_pct')}%",
            flush=True,
        )
    else:
        print(
            f"Compass {event.get('compass_id')}: {status}, "
            f"fitness={event.get('fitness')} autosaved={event.get('autosaved')}",
            flush=True,
        )


def _print_magcal_events(events: Iterable[dict[str, Any]]) -> None:
    for event in events:
        _print_magcal_event(event)


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

    emit_json = bool(args.json or args.json_output)
    if ack_report is not None and not emit_json:
        print(f"magcal {action}: {ack_report['result_name']}", flush=True)

    monitor_seconds = args.timeout if action == "monitor" else args.monitor_seconds
    events = (
        monitor_magcal(
            connection,
            monitor_seconds,
            on_event=None if emit_json else _print_magcal_event,
        )
        if monitor_seconds > 0
        else []
    )
    report = {"action": action, "master": master, "ack": ack_report, "events": events}
    if emit_json:
        _write_json(report, args.json_output)
    else:
        if action == "monitor" and not events:
            print("No magnetometer calibration progress/report messages observed")
    return 0 if (action != "monitor" or events) else 1


def send_accelcal_start(connection) -> None:
    """Start ArduPilot's full six-position accelerometer calibration."""
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        ACCELCAL_START_COMMAND,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    )


def send_accelcal_position(connection, position: int) -> None:
    """Confirm that the vehicle is motionless in the requested position."""
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        ACCELCAL_POSITION_COMMAND,
        0,
        position,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _accelcal_terminal_from_message(message) -> bool | None:
    """Return terminal success/failure, or ``None`` for a non-terminal event."""
    if message.get_type() == "COMMAND_LONG":
        if int(message.command) != ACCELCAL_POSITION_COMMAND:
            return None
        position = int(round(float(message.param1)))
        if position == ACCELCAL_SUCCESS:
            return True
        if position == ACCELCAL_FAILED:
            return False
        return None

    text = str(getattr(message, "text", "")).lower()
    if "calibration successful" in text:
        return True
    if "calibration failed" in text or "calibration cancelled" in text:
        return False
    return None


def _wait_accelcal_event(connection, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type=["COMMAND_LONG", "STATUSTEXT"],
            blocking=True,
            timeout=min(0.5, remaining),
        )
        if message is None:
            continue
        terminal = _accelcal_terminal_from_message(message)
        if terminal is not None:
            return ("terminal", terminal, message)
        if (
            message.get_type() == "COMMAND_LONG"
            and int(message.command) == ACCELCAL_POSITION_COMMAND
        ):
            position = int(round(float(message.param1)))
            if position in ACCELCAL_POSES:
                return ("pose", position, message)
    return None


def run_accelcal(
    connection,
    *,
    command_timeout_s: float,
    pose_timeout_s: float,
    result_timeout_s: float,
    input_fn=input,
    output_fn=print,
) -> dict[str, Any]:
    """Run the complete interactive accelerometer-calibration state machine."""
    send_accelcal_start(connection)
    ack = _wait_command_ack(connection, ACCELCAL_START_COMMAND, command_timeout_s)
    if ack is None:
        raise CliError("no COMMAND_ACK received when starting accelcal")
    start_result = int(ack.result)
    if start_result not in ACCELCAL_ACCEPTED_RESULTS:
        return {
            "success": False,
            "start_result": start_result,
            "start_result_name": _mav_result_name(start_result),
            "poses_completed": [],
            "failure": "start command rejected",
        }

    completed: list[int] = []
    expected_position = min(ACCELCAL_POSES)
    while expected_position <= max(ACCELCAL_POSES):
        event = _wait_accelcal_event(connection, pose_timeout_s)
        if event is None:
            raise CliError(
                f"timed out waiting for accelerometer pose {expected_position}/6"
            )
        event_type, value, _message = event
        if event_type == "terminal":
            return {
                "success": bool(value),
                "start_result": start_result,
                "start_result_name": _mav_result_name(start_result),
                "poses_completed": completed,
                "failure": None if value else "flight controller reported failure",
            }

        position = int(value)
        if position < expected_position:
            # ArduPilot periodically repeats its current request. Ignore a stale
            # request already acknowledged by this process.
            continue
        if position != expected_position:
            raise CliError(
                f"flight controller requested pose {position}, expected "
                f"{expected_position}; refusing to guess calibration state"
            )

        title, instruction = ACCELCAL_POSES[position]
        output_fn(f"Step {position}/6 — {title}")
        output_fn(f"  {instruction}")
        output_fn(
            "  Hold the complete rover still; do not support it by a moving part."
        )
        try:
            input_fn("  Press Enter when stable (Ctrl-C aborts): ")
        except EOFError as error:
            raise CliError(
                "accelcal requires an interactive terminal for pose confirmation"
            ) from error

        send_accelcal_position(connection, position)
        # Do not require the per-pose COMMAND_ACK. ArduPilot periodically
        # publishes its current requested pose and terminal result, which are
        # the authoritative calibration state. On real Pixhawk USB links an
        # individual ACK can be lost even though the sample was accepted and
        # the flight controller advanced. The next loop still fails closed if
        # no next-pose/terminal progress arrives or if a pose is skipped.
        completed.append(position)
        expected_position += 1

    event = _wait_accelcal_event(connection, result_timeout_s)
    if event is None or event[0] != "terminal":
        raise CliError(
            "all poses were accepted but no terminal calibration result was received"
        )
    success = bool(event[1])
    return {
        "success": success,
        "start_result": start_result,
        "start_result_name": _mav_result_name(start_result),
        "poses_completed": completed,
        "failure": None if success else "flight controller reported failure",
    }


def _print_accelcal_pose_plan() -> None:
    print("Full accelerometer calibration requires these six poses:")
    for position, (title, instruction) in ACCELCAL_POSES.items():
        print(f"  {position}. {title}: {instruction}")


def command_accelcal(args: argparse.Namespace) -> int:
    if not args.yes:
        raise CliError(
            "accelcal changes flight-controller calibration state; repeat with "
            "--yes after making the disarmed rover physically safe"
        )
    if args.json or args.json_output:
        raise CliError("interactive accelcal does not support JSON output")

    connection, heartbeat, master = _connect(args)
    if _armed(heartbeat):
        raise CliError("vehicle is armed; refusing accelcal")

    print(f"Connected to {master}")
    print(
        "Keep the assembled rover disarmed and motionless while each sample is taken."
    )
    print("The first command also calibrates the gyros; do not move it until prompted.")
    _print_accelcal_pose_plan()
    report = run_accelcal(
        connection,
        command_timeout_s=args.command_timeout,
        pose_timeout_s=args.pose_timeout,
        result_timeout_s=args.result_timeout,
    )
    if report["success"]:
        print("PASS: accelerometer calibration saved successfully.")
        print("Reboot ArduPilot, then run: python -m spf.ardupilot.ardu_cli prearm")
        return 0
    print(f"FAIL: accelerometer calibration failed: {report['failure']}")
    return 1


def _rc_snapshot(message) -> dict[int, int]:
    """Channel -> raw microseconds for every channel the receiver reports."""
    count = int(getattr(message, "chancount", 0) or 0)
    values: dict[int, int] = {}
    for channel in range(1, RC_MAX_CHANNELS + 1):
        raw = getattr(message, f"chan{channel}_raw", None)
        if raw is None:
            continue
        # ArduPilot pads unpopulated channels with 0 past chancount; keeping them
        # would report phantom "movement" the moment a receiver changes protocol.
        if channel > count and not raw:
            continue
        values[channel] = int(raw)
    return values


def _rc_role(channel: int) -> str:
    return RC_CHANNEL_ROLES.get(channel, "")


def _format_rc_row(values: dict[int, int]) -> str:
    return " ".join(f"{channel}:{value}" for channel, value in sorted(values.items()))


def command_rc(args: argparse.Namespace) -> int:
    """Echo RC_CHANNELS so an operator can see what the FC receives from the radio."""
    connection, _heartbeat, master = _connect(args)
    print(f"Connected to {master}")

    interval_us = max(1, int(args.rate_hz and 1_000_000 / args.rate_hz))
    _request_message(
        connection, mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, interval_us
    )

    deadline = time.monotonic() + args.duration
    started = time.monotonic()
    last: dict[int, int] = {}
    seen: dict[int, dict[str, int]] = {}
    frames = 0
    last_print = started
    last_rssi: int | None = None

    print(
        f"Listening for RC_CHANNELS for {args.duration:.0f}s "
        f"(threshold {args.threshold} us). Move one switch at a time; Ctrl-C to stop.\n"
    )
    try:
        while time.monotonic() < deadline:
            message = connection.recv_match(
                type="RC_CHANNELS", blocking=True, timeout=0.5
            )
            now = time.monotonic()
            if message is None:
                # Silence is the single most informative outcome here, so say so
                # while waiting rather than only in the summary.
                if now - last_print >= args.quiet_tick:
                    elapsed = now - started
                    state = "no RC_CHANNELS yet" if not frames else "no change"
                    print(f"  t+{elapsed:6.1f}s  {state}")
                    last_print = now
                continue

            frames += 1
            values = _rc_snapshot(message)
            rssi = int(getattr(message, "rssi", 255) or 0)
            for channel, value in values.items():
                record = seen.setdefault(channel, {"min": value, "max": value})
                record["min"] = min(record["min"], value)
                record["max"] = max(record["max"], value)

            if not last:
                print(
                    f"  baseline  {len(values)} channels, rssi {rssi}\n"
                    f"            {_format_rc_row(values)}\n"
                )
                last = values
                last_print = now
                last_rssi = rssi
                continue

            changed = [
                (channel, last.get(channel), value)
                for channel, value in values.items()
                if last.get(channel) is not None
                and abs(value - last[channel]) >= args.threshold
            ]
            if args.all:
                print(f"  t+{now - started:6.1f}s  {_format_rc_row(values)}")
                last_print = now
            for channel, before, value in changed:
                role = _rc_role(channel)
                suffix = f"   {role}" if role else ""
                print(
                    f"  t+{now - started:6.1f}s  CH{channel:<2} "
                    f"{before:>4} -> {value:<4}{suffix}"
                )
                last_print = now
            if changed:
                last = values
            else:
                # Track the latest values without redrawing, so a slow drift
                # still eventually crosses the threshold against a fresh base.
                last.update(
                    {
                        channel: value
                        for channel, value in values.items()
                        if channel not in last
                    }
                )
            if rssi != last_rssi and abs(rssi - (last_rssi or 0)) >= args.threshold:
                print(f"  t+{now - started:6.1f}s  rssi {last_rssi} -> {rssi}")
                last_rssi = rssi
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        try:
            connection.close()
        except Exception:
            pass

    elapsed = max(time.monotonic() - started, 1e-6)
    moved = sorted(
        channel
        for channel, record in seen.items()
        if record["max"] - record["min"] >= args.threshold
    )
    report = {
        "master": master,
        "frames": frames,
        "elapsed_s": round(elapsed, 2),
        "rate_hz": round(frames / elapsed, 2),
        "channels": {
            str(channel): {
                "min": record["min"],
                "max": record["max"],
                "moved": (record["max"] - record["min"]) >= args.threshold,
                "role": _rc_role(channel),
            }
            for channel, record in sorted(seen.items())
        },
        "moved": moved,
        "rc_received": frames > 0,
    }
    if args.json or args.json_output:
        _write_json(report, args.json_output)
        return 0 if frames else 1

    print(f"\n  {frames} RC_CHANNELS frames in {elapsed:.1f}s ({report['rate_hz']}/s)")
    if not frames:
        print(
            "\nFAIL: the flight controller received no RC frames at all.\n"
            "  The transmitter/receiver bind is not the only suspect here — the FC\n"
            "  never saw a single frame, so look between the receiver and the FC:\n"
            "    - receiver powered, and its bind LED showing a live link?\n"
            "    - SBUS/signal wire in the FC's RC input, right pin and orientation?\n"
            "    - RC_PROTOCOLS allows the receiver's protocol "
            "(rover3_rc_servo_parameters.params sets 1)\n"
            "    - serial RC receivers need the FC rebooted after rewiring"
        )
        return 1

    print("\n  channel        min    max   moved   role")
    for channel, record in sorted(seen.items()):
        did_move = (record["max"] - record["min"]) >= args.threshold
        print(
            f"  CH{channel:<2}         {record['min']:>5}  {record['max']:>5}   "
            f"{'yes' if did_move else ' - '}     {_rc_role(channel)}"
        )
    if not moved:
        print(
            "\nFAIL: RC frames are arriving but nothing moved.\n"
            "  The receiver is talking to the flight controller, so the wiring is\n"
            "  fine and the problem is upstream of it: transmitter off, wrong model\n"
            "  selected, RxNum mismatch, or the switches are not mapped to channels\n"
            "  in the transmitter's mixer."
        )
        return 1
    print(f"\nPASS: {frames} frames, channels moved: {', '.join(f'CH{c}' for c in moved)}")
    return 0


def send_fc_reboot(connection) -> None:
    """Ask the flight controller to reboot (param1=1, autopilot reboot)."""
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def reboot_flight_controller(
    args: argparse.Namespace,
    connection,
    *,
    settle_s: float,
    timeout_s: float,
    output_fn=print,
    sleep_fn=time.sleep,
    connect_fn=None,
):
    """Reboot the FC and return a fresh, heartbeat-verified connection.

    Accel offsets do not take effect until the flight controller restarts, so a
    magcal run before this reboot fits against the pre-calibration attitude --
    exactly the error that calibrating accel-before-compass exists to avoid.

    Fails closed. If the FC does not come back within ``timeout_s`` this raises
    rather than returning a connection, because the alternative is running a
    calibration against a half-booted vehicle.
    """
    connect_fn = connect_fn or _connect
    output_fn("  rebooting the flight controller...")
    send_fc_reboot(connection)
    # No ACK is expected: a rebooting FC stops talking mid-command.
    try:
        connection.close()
    except Exception:
        pass

    sleep_fn(settle_s)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            new_connection, heartbeat, _master = connect_fn(args)
        except Exception as error:  # transport not back yet
            last_error = error
            sleep_fn(1.0)
            continue
        if heartbeat is not None:
            output_fn("  flight controller is back")
            return new_connection
        try:
            new_connection.close()
        except Exception:
            pass
        sleep_fn(1.0)
    raise CliError(
        f"flight controller did not return within {timeout_s:.0f}s of the reboot"
        + (f" (last error: {last_error})" if last_error else "")
        + ". Refusing to continue calibrating against an unverified vehicle."
    )


def command_reboot(args: argparse.Namespace) -> int:
    if not args.yes:
        raise CliError(
            "reboot restarts the flight controller; repeat with --yes after "
            "making the disarmed rover physically safe"
        )
    connection, heartbeat, master = _connect(args)
    if _armed(heartbeat):
        raise CliError("vehicle is armed; refusing to reboot the flight controller")
    print(f"Connected to {master}")
    connection = reboot_flight_controller(
        args, connection, settle_s=args.settle, timeout_s=args.reboot_timeout
    )
    try:
        connection.close()
    except Exception:
        pass
    print("PASS: flight controller rebooted and is responding.")
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    """The full mission-required calibration, in the only order that is correct.

    accel (+gyro) -> reboot -> compass -> prearm verify.

    Base parameters and SYSID_THISMAV are deliberately absent: drone_run.sh
    force-loads them on every boot, so calibrating them here would be redundant.
    RC calibration is absent for a stronger reason -- the same boot sync
    force-writes RC1-16 MIN/MAX/TRIM from rover3_rc_servo_parameters.params, so
    any RC calibration is reverted at the next boot.
    """
    if not args.yes:
        raise CliError(
            "calibrate changes flight-controller calibration state and reboots "
            "it; repeat with --yes once the disarmed rover is physically safe "
            "and you can pick it up and rotate it"
        )
    if args.json or args.json_output:
        raise CliError("interactive calibration does not support JSON output")

    connection, heartbeat, master = _connect(args)
    if _armed(heartbeat):
        raise CliError("vehicle is armed; refusing to calibrate")

    print(f"Connected to {master}")
    print(
        "\nThis runs the full mission-required calibration:\n"
        "  1. accelerometer + gyro  (six poses, you will be prompted)\n"
        "  2. flight-controller reboot  (accel offsets take effect here)\n"
        "  3. compass / magcal  (rotate the rover through all axes)\n"
        "  4. pre-arm verification\n"
    )

    print("== 1/4  accelerometer + gyro ==")
    _print_accelcal_pose_plan()
    accel = run_accelcal(
        connection,
        command_timeout_s=args.command_timeout,
        pose_timeout_s=args.pose_timeout,
        result_timeout_s=args.result_timeout,
    )
    if not accel["success"]:
        print(f"FAIL: accelerometer calibration failed: {accel['failure']}")
        return 1
    print("  accelerometer calibration saved\n")

    print("== 2/4  reboot ==")
    connection = reboot_flight_controller(
        args, connection, settle_s=args.settle, timeout_s=args.reboot_timeout
    )
    print()

    print("== 3/4  compass / magcal ==")
    print(
        "  Rotate the rover slowly through every axis: level spin, nose up,\n"
        "  nose down, on each side, and inverted. Keep away from metal.\n"
    )
    send_magcal_command(connection, "start", mask=0, retry=False, autosave=True)
    ack = _wait_command_ack(
        connection, MAGCAL_COMMANDS["start"], args.command_timeout
    )
    if ack is None:
        raise CliError("no COMMAND_ACK received for magcal start")
    if int(ack.result) not in ACCELCAL_ACCEPTED_RESULTS:
        print(f"FAIL: magcal start rejected: {_mav_result_name(int(ack.result))}")
        return 1

    events = monitor_magcal(
        connection,
        args.magcal_timeout,
        on_event=_print_magcal_event,
    )
    reports = [
        event for event in events if event.get("message_type") == "MAG_CAL_REPORT"
    ]
    if not reports:
        print(
            f"\nFAIL: no MAG_CAL_REPORT within {args.magcal_timeout:.0f}s. "
            "The compass is NOT calibrated."
        )
        return 1
    failed = [
        report
        for report in reports
        if report.get("cal_status_name") != "MAG_CAL_SUCCESS"
    ]
    if failed:
        statuses = ", ".join(
            str(report.get("cal_status_name")) for report in failed
        )
        print(f"\nFAIL: compass calibration did not succeed: {statuses}")
        return 1
    print("  compass calibration saved (autosave)\n")

    print("== 4/4  pre-arm verification ==")
    connection = reboot_flight_controller(
        args, connection, settle_s=args.settle, timeout_s=args.reboot_timeout
    )
    result = run_prearm_checks(connection, timeout_s=args.prearm_timeout)
    try:
        connection.close()
    except Exception:
        pass
    if not result.passed:
        print("\nFAIL: calibration completed but the rover does not pass pre-arm.")
        print("  Inspect with: rover ardupilot prearm")
        return 1
    print("\nPASS: accelerometer, gyro and compass calibrated; pre-arm is clean.")
    return 0


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
        description="Inspect an SPF rover's ArduPilot and manage sensor calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_CHEATSHEET,
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

    reboot = subparsers.add_parser(
        "reboot", help="Reboot the flight controller and wait for it to return."
    )
    _add_connection_options(reboot)
    reboot.add_argument("--yes", action="store_true")
    reboot.add_argument("--settle", type=float, default=2.0)
    reboot.add_argument("--reboot-timeout", type=float, default=60.0)
    reboot.set_defaults(handler=command_reboot)

    calibrate = subparsers.add_parser(
        "calibrate",
        help=(
            "Full mission calibration in the correct order: "
            "accel+gyro, reboot, compass, pre-arm verify."
        ),
    )
    _add_connection_options(calibrate)
    calibrate.add_argument("--yes", action="store_true")
    calibrate.add_argument("--command-timeout", type=float, default=10.0)
    calibrate.add_argument("--pose-timeout", type=float, default=120.0)
    calibrate.add_argument("--result-timeout", type=float, default=60.0)
    calibrate.add_argument("--settle", type=float, default=2.0)
    calibrate.add_argument("--reboot-timeout", type=float, default=60.0)
    calibrate.add_argument(
        "--magcal-timeout",
        type=float,
        default=300.0,
        help="Ceiling on compass calibration; a terminal report exits early.",
    )
    calibrate.add_argument("--prearm-timeout", type=float, default=30.0)
    calibrate.set_defaults(handler=command_calibrate)

    rc = subparsers.add_parser(
        "rc",
        help="Echo RC_CHANNELS live: what the flight controller hears from the radio.",
    )
    _add_connection_options(rc)
    rc.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Seconds to listen before printing the summary.",
    )
    rc.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Microseconds a channel must move before it is reported as a change.",
    )
    rc.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Requested RC_CHANNELS stream rate.",
    )
    rc.add_argument(
        "--all",
        action="store_true",
        help="Print every frame, not only changes.",
    )
    rc.add_argument(
        "--quiet-tick",
        type=float,
        default=5.0,
        help="Seconds of no change before printing a keepalive line.",
    )
    rc.set_defaults(handler=command_rc)

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
        action_parser.add_argument(
            "--monitor-seconds",
            type=float,
            default=0.0,
            help=(
                "maximum time to monitor; progress prints live and a terminal "
                "MAG_CAL_REPORT ends monitoring early"
            ),
        )
        if action == "start":
            action_parser.add_argument("--retry", action="store_true")
            action_parser.add_argument(
                "--no-autosave", dest="autosave", action="store_false"
            )
            action_parser.set_defaults(autosave=True)
        action_parser.set_defaults(handler=command_magcal)
    monitor = magcal_subparsers.add_parser("monitor")
    _add_connection_options(monitor)
    monitor.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help=(
            "maximum time to monitor; progress prints live and a terminal "
            "MAG_CAL_REPORT ends monitoring early"
        ),
    )
    monitor.set_defaults(handler=command_magcal)

    accelcal = subparsers.add_parser(
        "accelcal",
        help="Run guarded, interactive six-position accelerometer calibration.",
    )
    accelcal_subparsers = accelcal.add_subparsers(dest="accelcal_action", required=True)
    accelcal_start = accelcal_subparsers.add_parser(
        "start", help="Start and complete all six accelerometer poses."
    )
    _add_connection_options(accelcal_start)
    accelcal_start.add_argument(
        "--yes",
        action="store_true",
        help="Confirm persistent accelerometer calibration changes.",
    )
    accelcal_start.add_argument("--command-timeout", type=float, default=10.0)
    accelcal_start.add_argument(
        "--pose-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for each pose request.",
    )
    accelcal_start.add_argument(
        "--result-timeout",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for the saved terminal result.",
    )
    accelcal_start.set_defaults(handler=command_accelcal)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
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
