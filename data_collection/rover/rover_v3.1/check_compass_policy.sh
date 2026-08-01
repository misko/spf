#!/usr/bin/env bash
set -euo pipefail

readonly SERVICE_NAME="mavlink_controller.service"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly DEFAULT_PYTHON="/home/pi/spf-virtualenv/bin/python"
readonly PYTHON_BIN="${SPF_PYTHON:-${DEFAULT_PYTHON}}"
readonly ARDU_CLI="${REPO_ROOT}/spf/ardupilot/ardu_cli.py"

usage() {
    cat <<'EOF'
Usage:
  check_compass_policy.sh [checker options]
  check_compass_policy.sh --params SAVED_PARAMS [checker options]

Without --params, download a fresh parameter snapshot over the rover's one
ArduPilot serial link and evaluate it without writing flight-controller state.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set SPF_PYTHON to the SPF virtualenv's Python executable." >&2
    exit 2
fi

if [[ "${1:-}" == "--params" ]]; then
    [[ $# -ge 2 ]] || {
        echo "ERROR: --params requires a parameter file." >&2
        exit 2
    }
    readonly PARAMETER_FILE="$2"
    shift 2
    exec "${PYTHON_BIN}" "${ARDU_CLI}" compass \
        --params "${PARAMETER_FILE}" "$@"
fi

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "ERROR: ${SERVICE_NAME} is active and may own the ArduPilot link." >&2
    echo "Stop it first: sudo systemctl stop ${SERVICE_NAME}" >&2
    exit 2
fi

exec "${PYTHON_BIN}" "${ARDU_CLI}" compass "$@"
