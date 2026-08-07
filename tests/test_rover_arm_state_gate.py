"""vehicle_arm_state_gate must never be the reason a rover fails to boot.

drone_run.sh runs on every rover at every boot and rovers pull origin/main
before running it, so a defect here lands on the whole fleet unattended. Twice
on 2026-08-06 that happened: `--print-plan` died on an unbound variable under
`set -u`, and a flag guard refused the `=0` that is the production default.

Both were invisible to text assertions. These tests EXECUTE the function, under
`set -euo pipefail`, with the MAVLink call stubbed -- because the failure mode
that matters is "the boot stopped", and only running it can show that.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "data_collection/rover/rover_v3.1/drone_run.sh"

# Enough of the script's preamble for the function to run standalone. Kept
# minimal on purpose: anything more and the test starts passing for reasons
# unrelated to the gate.
HARNESS = """
set -euo pipefail
die() { printf 'ERROR: %s\\n' "$*" >&2; exit 1; }
is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off|"") return 1 ;;
        *) die "Invalid boolean value: $1" ;;
    esac
}
REQUIRE_DISARMED_FOR_PARAM_SYNC="${REQUIRE_DISARMED_FOR_PARAM_SYNC:-0}"
PYTHON="$FAKE_PYTHON"
MAVLINK_CONTROLLER=/dev/null
__GATE__
vehicle_arm_state_gate
echo "GATE_RETURNED"
"""

# The real call is `"$PYTHON" "$MAVLINK_CONTROLLER" --status-json "$file"`, so
# the flag is $2 and the destination is $3.
# Token replacement, not %-formatting: both the harness and the gate contain
# printf format specifiers of their own.
STUB = """#!/bin/bash
if [[ "${2:-}" == "--status-json" ]]; then
__BODY__
fi
exec __PYTHON__ "$@"
"""


def _gate_source() -> str:
    text = LAUNCHER.read_text()
    body = text.split("vehicle_arm_state_gate() {", 1)[1].split("\n}\n", 1)[0]
    return "vehicle_arm_state_gate() {" + body + "\n}\n"


def _run(tmp_path, stub_body, env=None):
    stub = tmp_path / "fake_python"
    stub.write_text(
        STUB.replace("__BODY__", stub_body).replace(
            "__PYTHON__", os.sys.executable
        )
    )
    stub.chmod(0o755)

    script = tmp_path / "harness.sh"
    script.write_text(HARNESS.replace("__GATE__", _gate_source()))

    environment = dict(os.environ, FAKE_PYTHON=str(stub))
    environment.update(env or {})
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


def _emit(payload):
    return f"""  printf '{payload}\\n' > "$3"; exit 0"""


DISARMED = _emit('{"armed": false, "mav_mode": "ROVER_MODE_HOLD"}')
ARMED = _emit('{"armed": true, "mav_mode": "ROVER_MODE_MANUAL"}')


def test_a_disarmed_vehicle_passes_and_the_boot_continues(tmp_path):
    result = _run(tmp_path, DISARMED)

    assert result.returncode == 0
    assert "GATE_RETURNED" in result.stdout
    assert "PASS" in result.stdout
    assert "DISARMED" in result.stdout


def test_an_armed_vehicle_warns_and_the_boot_continues(tmp_path):
    """The production default. Dying here would strand a rover for the case
    that is not actually unsafe -- no write happens when parameters match, and
    when they differ prepare_vehicle_parameters refuses one layer down."""
    result = _run(tmp_path, ARMED)

    assert result.returncode == 0, "warn-by-default must not stop the boot"
    assert "GATE_RETURNED" in result.stdout
    assert "WARNING: vehicle is ARMED" in result.stderr
    assert "SPF_REQUIRE_DISARMED_FOR_PARAM_SYNC=1" in result.stderr


def test_the_knob_makes_an_armed_vehicle_fatal(tmp_path):
    result = _run(
        tmp_path, ARMED, env={"REQUIRE_DISARMED_FOR_PARAM_SYNC": "1"}
    )

    assert result.returncode != 0
    assert "GATE_RETURNED" not in result.stdout, "the boot must not continue"
    assert "ARMED before parameter sync" in result.stderr


@pytest.mark.parametrize(
    "stub_body, reason",
    [
        ("  exit 1", "the controller exited non-zero"),
        ("""  printf 'not json\\n' > "$3"; exit 0""", "the payload was unparseable"),
        ("""  : > "$3"; exit 0""", "the payload was empty"),
    ],
)
def test_an_unreadable_status_never_stops_the_boot(tmp_path, stub_body, reason):
    """Inferring "armed" from silence would strand rovers for exactly the reason
    warn-by-default exists. Under `set -e` a bare failing call would also abort
    the launcher outright."""
    result = _run(tmp_path, stub_body)

    assert result.returncode == 0, f"boot stopped because {reason}"
    assert "GATE_RETURNED" in result.stdout
    assert "WARNING" in result.stderr


def test_an_unreadable_status_is_not_fatal_even_with_the_knob_on(tmp_path):
    """The knob promises "refuse to boot an ARMED rover", not "refuse to boot a
    rover whose arm state could not be read"."""
    result = _run(
        tmp_path, "  exit 1", env={"REQUIRE_DISARMED_FOR_PARAM_SYNC": "1"}
    )

    assert result.returncode == 0
    assert "GATE_RETURNED" in result.stdout
