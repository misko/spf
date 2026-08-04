#!/usr/bin/env python3
"""Spin the rover's motors at a ladder of throttle levels and ask the operator
to confirm each one.

    THE WHEELS MUST BE OFF THE GROUND. This drives the motors directly,
    bypassing the arming checks, exactly as ArduPilot's bench motor test does.
    A rover on its wheels WILL drive away.

Uses MAV_CMD_DO_MOTOR_TEST, so it needs neither arming nor a GPS fix nor a
calibrated compass — which is the point: it isolates "do the motors and ESCs
work" from every other pre-arm gate.

Safety, in the order the checks fire:

  1. --wheels-raised is mandatory. There is no default.
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

    # the real thing:
    ./run_motor_test.py --wheels-raised
    ./run_motor_test.py --wheels-raised --levels 10,25 --seconds 1.5
    ./run_motor_test.py --wheels-raised --json /tmp/motor_test.json

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
MOTORS = {1: "motor 1 (throttle right, SERVO1)", 2: "motor 2 (throttle left, SERVO3)"}
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


def _spin(connection, instance: int, percent: int, seconds: float) -> str | None:
    """Run one motor at one level. Returns a MAV_RESULT name, or None on timeout."""
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
    ack = connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
    if ack is None:
        return None
    return mavutil.mavlink.enums["MAV_RESULT"][ack.result].name


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
        "--motors", default="1,2", help="motor instances to test (default: 1,2)"
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
    try:
        if not args.dry_run:
            master = _resolve_master(args.master)
            print(f"  connecting to {master} ...")
            connection = _connect(master, args.baud, args.heartbeat_timeout)
            print(f"  heartbeat from system {connection.target_system}\n")

        for percent in levels:
            print(f"--- {percent}% ---")
            for instance in motors:
                label = MOTORS[instance]
                print(f"  spinning {label} at {percent}% for {args.seconds}s ...")
                ack = None
                if not args.dry_run:
                    ack = _spin(connection, instance, percent, args.seconds)
                    time.sleep(args.seconds + 0.5)
                    if ack is not None and ack != "MAV_RESULT_ACCEPTED":
                        print(f"    WARNING: flight controller replied {ack}")
                ok = True if args.dry_run else _ask(f"did {label} spin smoothly at {percent}%?")
                results.append(
                    {
                        "percent": percent,
                        "motor": instance,
                        "label": label,
                        "ack": ack,
                        "operator_confirmed": ok,
                        "dry_run": args.dry_run,
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
