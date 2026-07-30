#!/usr/bin/env bash
set -euo pipefail

readonly SERVICE_NAME="mavlink_controller.service"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly DEFAULT_PYTHON="/home/pi/spf-virtualenv/bin/python"
readonly PYTHON_BIN="${SPF_PYTHON:-${DEFAULT_PYTHON}}"

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "ERROR: ${SERVICE_NAME} is active and may own the ArduPilot link." >&2
    echo "Stop it first: sudo systemctl stop ${SERVICE_NAME}" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set SPF_PYTHON to the SPF virtualenv's Python executable." >&2
    exit 2
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/spf/mavlink/check_prearm.py" "$@"
