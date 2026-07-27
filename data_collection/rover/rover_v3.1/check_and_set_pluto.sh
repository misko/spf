#!/usr/bin/env bash
#
# Compatibility entry point for explicit, one-time Pluto AD9361/2R2T
# provisioning. Normal Rover boot uses the read-only check-config-all gate.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
export SPF_FIRMWARE_STATE_DIR="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: sudo $(basename "$0") (--dry-run | --apply) [EXPECTED_PLUTOS]

This is an explicit persistent provisioning operation. It addresses each Pluto
by USB serial/path, backs up its U-Boot environment, and verifies actual dual
RX after any required reboot.

Normal collection boot never invokes this command and never writes U-Boot.
EOF
}

[[ "${EUID}" -eq 0 ]] || die "Run this provisioning wrapper with sudo."
[[ "$#" -ge 1 && "$#" -le 2 ]] || {
    usage >&2
    exit 2
}

case "$1" in
    --dry-run) loader_option=(--dry-run) ;;
    --apply) loader_option=() ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "Choose exactly one of --dry-run or --apply."
        ;;
esac

if [[ "$#" -eq 2 ]]; then
    expected_count="$2"
else
    [[ -f /home/pi/rover_id ]] || die "Missing /home/pi/rover_id."
    rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
    expected_count="$(
        "$PYTHON" -c \
            'import sys; from spf.scripts.rover_capture_config import resolve_capture_plan; print(resolve_capture_plan(int(sys.argv[1])).expected_radios)' \
            "$rover_id"
    )"
fi

[[ "$expected_count" =~ ^[1-9][0-9]*$ ]] ||
    die "Expected Pluto count must be a positive integer."

printf '%s\n' \
    "Pluto provisioning target: ${expected_count} radio(s)." \
    "Required persistent state: attr_name=compatible, attr_val=ad9361," \
    "compatible=ad9361, mode=2r2t."
if [[ "$1" == "--apply" ]]; then
    printf '%s\n' \
        "Persistent U-Boot changes are authorized by explicit --apply."
fi

exec bash "$LOADER" provision-config-all "$expected_count" "${loader_option[@]}"
