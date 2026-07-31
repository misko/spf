#!/usr/bin/env python3
"""Run ArduPilot pre-arm checks without attempting to arm the vehicle."""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import dataclass

from pymavlink import mavutil


PREARM_BIT = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
PREARM_COMMAND = mavutil.mavlink.MAV_CMD_RUN_PREARM_CHECKS


@dataclass(frozen=True)
class PrearmResult:
    """Result reported by ArduPilot after running its pre-arm checks."""

    command_result: int | None
    healthy: bool | None
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.command_result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            and self.healthy is True
        )

    @property
    def conclusive(self) -> bool:
        return (
            self.command_result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            and self.healthy is not None
        )


def _message_text(message) -> str:
    text = message.text
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return str(text).rstrip("\x00").strip()


def run_prearm_checks(connection, timeout_s: float = 8.0) -> PrearmResult:
    """Request checks and collect the resulting ACK, health bit, and failures."""
    while connection.recv_match(blocking=False) is not None:
        pass

    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        PREARM_COMMAND,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    command_result = None
    healthy = None
    messages: list[str] = []
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        message = connection.recv_match(
            type=["COMMAND_ACK", "STATUSTEXT", "SYS_STATUS"],
            blocking=True,
            timeout=min(0.5, remaining),
        )
        if message is None:
            continue

        message_type = message.get_type()
        if message_type == "COMMAND_ACK" and message.command == PREARM_COMMAND:
            command_result = int(message.result)
        elif message_type == "SYS_STATUS":
            healthy = bool(message.onboard_control_sensors_health & PREARM_BIT)
        elif message_type == "STATUSTEXT":
            text = _message_text(message)
            if text.lower().startswith("prearm:") and text not in messages:
                messages.append(text)

    return PrearmResult(
        command_result=command_result,
        healthy=healthy,
        messages=tuple(messages),
    )


def resolve_default_master() -> str:
    """Find the one ArduPilot USB serial device used by a rover."""
    devices = sorted(glob.glob("/dev/serial/by-id/usb-ArduPilot*"))
    if len(devices) == 1:
        return devices[0]
    if len(devices) > 1:
        raise RuntimeError(
            "Multiple ArduPilot serial devices found; select one with --master"
        )
    raise RuntimeError(
        "No ArduPilot USB serial device found; connect it or pass --master"
    )


def _result_name(result: int | None) -> str:
    if result is None:
        return "NO_ACK"
    entry = mavutil.mavlink.enums["MAV_RESULT"].get(result)
    return entry.name if entry is not None else f"UNKNOWN_{result}"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask ArduPilot to run its pre-arm checks, print every reported "
            "failure, and exit without attempting to arm."
        )
    )
    parser.add_argument(
        "--master",
        help=(
            "MAVLink connection string. Defaults to the single "
            "/dev/serial/by-id/usb-ArduPilot* device."
        ),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--heartbeat-timeout", type=float, default=10.0)
    parser.add_argument("--result-timeout", type=float, default=8.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)

    try:
        master = args.master or resolve_default_master()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 2

    print(f"Connecting to {master} ...")
    try:
        connection = mavutil.mavlink_connection(
            master,
            baud=args.baud,
            source_system=254,
            dialect="ardupilotmega",
        )
        heartbeat = connection.wait_heartbeat(timeout=args.heartbeat_timeout)
    except Exception as error:
        print(f"ERROR: MAVLink connection failed: {error}")
        return 2

    if heartbeat is None:
        print("ERROR: No ArduPilot heartbeat received")
        return 2
    if heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
        print("ERROR: Vehicle is already armed; pre-arm status is not meaningful")
        return 2

    print(
        "Heartbeat received from "
        f"system {connection.target_system}, component {connection.target_component}"
    )
    result = run_prearm_checks(connection, timeout_s=args.result_timeout)

    print(f"Command result: {_result_name(result.command_result)}")
    if result.healthy is True:
        print("Pre-arm health: PASS")
    elif result.healthy is False:
        print("Pre-arm health: FAIL")
    else:
        print("Pre-arm health: UNKNOWN")

    if result.messages:
        print("Reported failures:")
        for message in result.messages:
            print(f"  - {message}")
    elif result.healthy is False:
        print(
            "No individual PreArm message was received; retry with a longer "
            "--result-timeout."
        )

    if result.passed:
        return 0
    if result.conclusive:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
