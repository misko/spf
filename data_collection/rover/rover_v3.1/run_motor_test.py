#!/usr/bin/env python3
"""Spin the rover's motors at a ladder of throttle levels and ask the operator
to confirm each one.

    THE WHEELS MUST BE OFF THE GROUND. A rover on its wheels WILL drive away.

Uses MAV_CMD_DO_MOTOR_TEST. Two things about Rover that are easy to get wrong,
both found the hard way on rover 2:

  * The vehicle must be ARMED. Rover accepts DO_MOTOR_TEST while disarmed,
    replies MAV_RESULT_ACCEPTED, says "Throttle disarmed" and moves nothing.
    Pass --arm; the script disarms again on every exit path.
  * Motor instances 3 and 4 are the throttle outputs. Instances 1 and 2 are
    steering and move nothing on a skid-steer rover.

Because an ACK is not evidence, every burst is verified by watching
SERVO_OUTPUT_RAW and reporting the PWM swing.

Safety, in the order the checks fire:

  1. --wheels-raised is mandatory. There is no default.
  0. The vehicle is disarmed on every exit path, including Ctrl-C.
  2. An interactive confirmation you must type in full (unless --dry-run).
  3. Refuses to run while mavlink_controller.service holds the serial link.
  4. Claims the TTY with TIOCEXCL so nothing else can open it mid-test.
  5. Every burst carries ArduPilot's own timeout, so the motors stop even if
     this process is killed.
  6. Ctrl-C, any exception, and normal exit all send an explicit stop.
  7. Levels run low to high, and any answer other than "y" aborts the run.

Usage:
    sudo systemctl stop mavlink_controller.service

    # rehearse the whole flow, send nothing:
    ./run_motor_test.py --dry-run

    # the real thing (--arm is required for anything to actually spin):
    ./run_motor_test.py --wheels-raised --arm
    ./run_motor_test.py --wheels-raised --arm --levels 10,25 --seconds 1.5
    ./run_motor_test.py --wheels-raised --arm --json /tmp/motor_test.json

Exit status: 0 all confirmed, 1 operator reported a fault or aborted,
2 setup/connection error.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import termios
import time
from datetime import datetime, timezone

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - environment problem, not logic
    print("ERROR: pymavlink is missing. Use the SPF virtualenv:", file=sys.stderr)
    print("  /home/pi/spf-virtualenv/bin/python3 run_motor_test.py ...", file=sys.stderr)
    raise SystemExit(2)

SERVICE_NAME = "mavlink_controller.service"
SOURCE_SYSTEM = 250
DEFAULT_LEVELS = (10, 25, 50, 100)
# ArduPilot Rover skid-steer: SERVO1_FUNCTION 74 (throttle right),
# SERVO3_FUNCTION 73 (throttle left). Motor test addresses them by instance.
# ArduPilot Rover maps DO_MOTOR_TEST instances 1/2 to STEERING and 3/4 to the
# throttle outputs. Determined empirically on rover 2 by sweeping instances 1-4
# and watching SERVO_OUTPUT_RAW: instances 1 and 2 moved nothing, instance 3
# drove servo3 1500->2042 us and instance 4 drove servo1 1500->958 us. Testing
# instances 1/2 is why an earlier version reported success while nothing spun.
MOTORS = {
    3: "throttle left (SERVO3, function 73)",
    4: "throttle right (SERVO1, function 74)",
}
MOTOR_TEST_THROTTLE_PERCENT = 0

CONFIRM_PHRASE = "WHEELS UP"


class TestError(Exception):
    pass


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


def _resolve_master(explicit: str | None) -> str:
    """Reuse the repo's resolver: /dev/serial/by-id/usb-ArduPilot*.

    Globbing /dev/ttyACM* is wrong on a rover — the Pluto CDC gadget also
    presents ttyACM nodes, so the first one is frequently not the flight
    controller. The by-id path names the device by what it actually is.
    """
    if explicit:
        return explicit
    try:
        from spf.mavlink.check_prearm import resolve_default_master
    except ImportError as error:
        raise TestError(
            f"cannot import the SPF resolver ({error}); run with the SPF "
            "virtualenv or pass --master"
        ) from error
    try:
        return resolve_default_master()
    except RuntimeError as error:
        raise TestError(str(error)) from error


def _connect(master: str, baud: int, timeout: float):
    connection = mavutil.mavlink_connection(
        master, baud=baud, source_system=SOURCE_SYSTEM, dialect="ardupilotmega"
    )
    # Same exclusive claim ardu_cli uses: stop anything else opening this TTY
    # while the motors are live.
    if master.startswith("/dev/") and hasattr(connection, "port"):
        try:
            fcntl.ioctl(connection.port.fileno(), termios.TIOCEXCL)
        except Exception as error:
            connection.close()
            raise TestError(f"could not claim exclusive ownership of {master}: {error}")
    heartbeat = connection.wait_heartbeat(timeout=timeout)
    if heartbeat is None:
        raise TestError(f"no ArduPilot heartbeat from {master}")
    return connection


def _stop_all(connection, motors) -> None:
    """Best-effort: 0% for a moment on every motor. Never raises."""
    for instance in motors:
        try:
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
                0,
                float(instance),
                float(MOTOR_TEST_THROTTLE_PERCENT),
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
            )
        except Exception:
            pass


def _servo_raw(connection, timeout: float = 3.0):
    """Current PWM on outputs 1-4, or None."""
    message = connection.recv_match(
        type="SERVO_OUTPUT_RAW", blocking=True, timeout=timeout
    )
    if message is None:
        return None
    return (
        message.servo1_raw,
        message.servo2_raw,
        message.servo3_raw,
        message.servo4_raw,
    )


def _request_servo_stream(connection) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW),
        100000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _get_param(connection, name: str, timeout: float = 5.0):
    connection.mav.param_request_read_send(
        connection.target_system, connection.target_component, name.encode(), -1
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = connection.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if message is None:
            continue
        if message.param_id.rstrip("\x00") == name:
            return message.param_value
    return None


def _set_param(connection, name: str, value: float, timeout: float = 5.0) -> bool:
    """Set a parameter and confirm by reading it back."""
    connection.mav.param_set_send(
        connection.target_system,
        connection.target_component,
        name.encode(),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        actual = _get_param(connection, name, timeout=1.5)
        if actual is not None and abs(actual - value) < 1e-6:
            return True
    return False


def _is_armed(connection) -> bool:
    heartbeat = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
    if heartbeat is None:
        return False
    return bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def _set_arm(connection, arm: bool, timeout: float = 10.0) -> str | None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0 if arm else 0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    ack = connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
    if ack is None:
        return None
    return mavutil.mavlink.enums["MAV_RESULT"][ack.result].name


def _spin(connection, instance: int, percent: int, seconds: float) -> dict:
    """Run one motor at one level.

    Returns the ACK *and* the observed PWM swing. The ACK alone is not evidence:
    ArduPilot Rover replies MAV_RESULT_ACCEPTED to DO_MOTOR_TEST while disarmed,
    emits "Throttle disarmed", and never moves the output. Watching
    SERVO_OUTPUT_RAW is what actually proves the motor was driven.
    """
    before = _servo_raw(connection)
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        float(instance),
        float(MOTOR_TEST_THROTTLE_PERCENT),
        float(percent),
        # ArduPilot stops the motor itself after this many seconds, so a crash
        # or a killed process cannot leave a motor running.
        float(seconds),
        0.0,
        0.0,
        0.0,
    )
    ack_message = connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
    ack = (
        None
        if ack_message is None
        else mavutil.mavlink.enums["MAV_RESULT"][ack_message.result].name
    )

    peak = before
    statustexts: list[str] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        sample = _servo_raw(connection, timeout=0.5)
        if sample is not None and before is not None:
            if max(abs(a - b) for a, b in zip(sample, before)) > max(
                abs(a - b) for a, b in zip(peak or before, before)
            ):
                peak = sample
        text = connection.recv_match(type="STATUSTEXT", blocking=False)
        if text is not None:
            statustexts.append(text.text)

    swing = (
        None
        if before is None or peak is None
        else max(abs(a - b) for a, b in zip(peak, before))
    )
    return {
        "ack": ack,
        "pwm_before": before,
        "pwm_peak": peak,
        "pwm_swing": swing,
        "statustext": statustexts,
    }


def _ask(prompt: str) -> bool:
    try:
        answer = input(f"    {prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wheels-raised",
        action="store_true",
        help="REQUIRED. Assert the wheels are off the ground.",
    )
    parser.add_argument("--master", help="MAVLink endpoint (default: first /dev/ttyACM*)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    parser.add_argument(
        "--levels",
        default=",".join(str(v) for v in DEFAULT_LEVELS),
        help=f"throttle percentages, low to high (default: {','.join(str(v) for v in DEFAULT_LEVELS)})",
    )
    parser.add_argument(
        "--motors", default="3,4", help="motor instances to test (default: 3,4)"
    )
    parser.add_argument(
        "--seconds", type=float, default=2.0, help="burst length per level (default: 2)"
    )
    parser.add_argument("--json", help="write a machine-readable report here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rehearse the sequence and prompts; send no MAVLink commands",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help=(
            "arm the vehicle for the duration of the test. REQUIRED on Rover: "
            "DO_MOTOR_TEST is accepted while disarmed but never drives the "
            "outputs (the FC replies 'Throttle disarmed'). The vehicle is "
            "disarmed again on every exit path."
        ),
    )
    parser.add_argument(
        "--bypass-arming-checks",
        action="store_true",
        help=(
            "temporarily set ARMING_CHECK=0 so a bench test can run past an "
            "unrelated pre-arm failure (e.g. 'Compass not healthy'). The "
            "original value is recorded and RESTORED on every exit path. "
            "ARMING_CHECK is NOT in the fleet params files, so nothing else "
            "would ever put it back."
        ),
    )
    parser.add_argument(
        "--allow-active-service",
        action="store_true",
        help="expert override; the service may fight you for the serial link",
    )
    args = parser.parse_args()

    try:
        levels = [int(v) for v in args.levels.split(",") if v.strip()]
        motors = [int(v) for v in args.motors.split(",") if v.strip()]
    except ValueError:
        print("ERROR: --levels and --motors take comma-separated integers", file=sys.stderr)
        return 2
    if not levels or not motors:
        print("ERROR: --levels and --motors must be non-empty", file=sys.stderr)
        return 2
    if any(not 0 < v <= 100 for v in levels):
        print("ERROR: levels must be within 1..100", file=sys.stderr)
        return 2
    if levels != sorted(levels):
        print("ERROR: levels must ascend; refusing to start above the lowest", file=sys.stderr)
        return 2
    if any(m not in MOTORS for m in motors):
        print(f"ERROR: motors must be within {sorted(MOTORS)}", file=sys.stderr)
        return 2
    if args.seconds <= 0 or args.seconds > 10:
        print("ERROR: --seconds must be within 0..10", file=sys.stderr)
        return 2

    if not args.wheels_raised and not args.dry_run:
        print(
            "REFUSING: --wheels-raised is mandatory.\n\n"
            "  This drives the motors directly and bypasses the arming checks.\n"
            "  A rover on its wheels WILL drive away. Put it on stilts or a stand,\n"
            "  confirm both wheels spin freely, then pass --wheels-raised.\n"
            "  Use --dry-run to rehearse without sending anything.",
            file=sys.stderr,
        )
        return 2

    print(f"\n=== rover motor test ===")
    print(f"  levels  : {levels} %")
    print(f"  motors  : {[MOTORS[m] for m in motors]}")
    print(f"  burst   : {args.seconds}s (ArduPilot stops each burst itself)")
    if args.dry_run:
        print("  MODE    : DRY RUN — no MAVLink commands will be sent\n")
    else:
        print()
        print("  !! THE WHEELS MUST BE OFF THE GROUND !!")
        try:
            typed = input(f'  Type "{CONFIRM_PHRASE}" to continue: ').strip()
        except EOFError:
            typed = ""
        if typed != CONFIRM_PHRASE:
            print("Aborted: confirmation not given.", file=sys.stderr)
            return 1

    if _service_is_active() and not args.allow_active_service and not args.dry_run:
        print(
            f"\nREFUSING: {SERVICE_NAME} is active and owns the ArduPilot serial link.\n"
            f"  sudo systemctl stop {SERVICE_NAME}\n"
            f"  ...then re-run. Restart it when you are done.",
            file=sys.stderr,
        )
        return 2

    connection = None
    results: list[dict] = []
    aborted = False
    armed_by_us = False
    original_arming_check = None
    try:
        if not args.dry_run:
            master = _resolve_master(args.master)
            print(f"  connecting to {master} ...")
            connection = _connect(master, args.baud, args.heartbeat_timeout)
            print(f"  heartbeat from system {connection.target_system}")
            _request_servo_stream(connection)

            if args.bypass_arming_checks:
                original_arming_check = _get_param(connection, "ARMING_CHECK")
                if original_arming_check is None:
                    raise TestError("could not read ARMING_CHECK; refusing to bypass")
                print(
                    f"  ARMING_CHECK {original_arming_check:g} -> 0 "
                    f"(will be restored to {original_arming_check:g})"
                )
                if not _set_param(connection, "ARMING_CHECK", 0.0):
                    raise TestError("could not set ARMING_CHECK=0")

            if args.arm:
                print("  arming (wheels MUST be raised) ...")
                result = _set_arm(connection, True)
                time.sleep(2)
                if not _is_armed(connection):
                    raise TestError(
                        f"arming failed (ack={result}). Check pre-arm health:\n"
                        f"    sudo python3 -m spf.mavlink.check_prearm"
                    )
                armed_by_us = True
                print("  ARMED\n")
            else:
                print(
                    "\n  NOTE: --arm not given. On Rover, DO_MOTOR_TEST is accepted\n"
                    "  while disarmed but never drives the outputs. Expect no movement.\n"
                )

        for percent in levels:
            print(f"--- {percent}% ---")
            for instance in motors:
                label = MOTORS[instance]
                print(f"  spinning {label} at {percent}% for {args.seconds}s ...")
                telemetry: dict = {}
                if not args.dry_run:
                    telemetry = _spin(connection, instance, percent, args.seconds)
                    time.sleep(0.5)
                    ack = telemetry.get("ack")
                    if ack is not None and ack != "MAV_RESULT_ACCEPTED":
                        print(f"    WARNING: flight controller replied {ack}")
                    for text in telemetry.get("statustext", []):
                        print(f"    FC says: {text}")
                    swing = telemetry.get("pwm_swing")
                    if swing is None:
                        print("    WARNING: no SERVO_OUTPUT_RAW; cannot verify PWM moved")
                    elif swing < 10:
                        # The failure mode that made an ACK look like success.
                        print(
                            f"    NO PWM MOVEMENT (swing {swing} us). The command was "
                            f"accepted but the output never left {telemetry['pwm_before']}."
                        )
                        print(
                            "    On Rover, DO_MOTOR_TEST only drives outputs while ARMED "
                            "— re-run with --arm."
                        )
                    else:
                        print(
                            f"    PWM {telemetry['pwm_before']} -> {telemetry['pwm_peak']} "
                            f"(swing {swing} us)"
                        )
                ok = True if args.dry_run else _ask(f"did {label} spin smoothly at {percent}%?")
                results.append(
                    {
                        "percent": percent,
                        "motor": instance,
                        "label": label,
                        "operator_confirmed": ok,
                        "dry_run": args.dry_run,
                        **telemetry,
                    }
                )
                if not ok:
                    print(f"\n  ABORTING at {percent}% on {label} (operator reported a fault).")
                    aborted = True
                    break
            if aborted:
                break
            print()
    except KeyboardInterrupt:
        print("\n  interrupted — stopping motors", file=sys.stderr)
        aborted = True
    except TestError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 2
    finally:
        if connection is not None:
            _stop_all(connection, motors)
            if armed_by_us:
                # Every exit path disarms: normal finish, operator abort, Ctrl-C,
                # and any exception. Never leave a rover armed.
                for attempt in range(3):
                    _set_arm(connection, False)
                    time.sleep(1.0)
                    if not _is_armed(connection):
                        print("  DISARMED")
                        break
                else:
                    print(
                        "  WARNING: could not confirm disarm — DISARM MANUALLY "
                        "before going near the vehicle",
                        file=sys.stderr,
                    )
            if original_arming_check is not None:
                # Same discipline as disarm: restore on every exit path. Nothing
                # else restores this - ARMING_CHECK is not in the params files.
                for _ in range(3):
                    if _set_param(connection, "ARMING_CHECK", original_arming_check):
                        print(f"  ARMING_CHECK restored to {original_arming_check:g}")
                        break
                else:
                    print(
                        f"  WARNING: could not restore ARMING_CHECK to "
                        f"{original_arming_check:g} - SET IT BACK BY HAND before "
                        f"this rover goes anywhere",
                        file=sys.stderr,
                    )
            try:
                connection.close()
            except Exception:
                pass

    confirmed = [r for r in results if r["operator_confirmed"]]
    print("=== summary ===")
    for r in results:
        mark = "ok " if r["operator_confirmed"] else "FAULT"
        print(f"  {mark}  {r['percent']:>3}%  {r['label']}")
    print(f"\n  {len(confirmed)}/{len(results)} confirmed")

    if args.json:
        report = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "levels": levels,
            "motors": motors,
            "seconds": args.seconds,
            "dry_run": args.dry_run,
            "aborted": aborted,
            "results": results,
        }
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"  report: {args.json}")

    if not args.dry_run:
        print(f"\n  Remember: sudo systemctl start {SERVICE_NAME}")

    return 1 if aborted or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
