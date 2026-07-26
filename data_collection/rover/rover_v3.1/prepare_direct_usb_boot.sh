#!/usr/bin/env bash
#
# Root-only boot preparation for the recoverable multi-Pluto direct-USB path.
# This script never writes Pluto QSPI.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly MAPPING_SCRIPT="${SCRIPT_DIR}/device_mapping.sh"
readonly READY_DIR="/run/spf"
readonly READY_FILE="${READY_DIR}/direct_usb_ready"

EXPECTED_RADIOS="${SPF_DIRECT_USB_EXPECTED_RADIOS:-}"
FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "${EUID}" -eq 0 ]] || die "prepare_direct_usb_boot.sh must run as root."
[[ -f /home/pi/rover_id ]] || die "Missing /home/pi/rover_id."

rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
case "$rover_id" in
    1|3)
        default_expected=2
        ;;
    2)
        default_expected=1
        ;;
    *)
        die "Unsupported rover_id: ${rover_id}"
        ;;
esac
EXPECTED_RADIOS="${EXPECTED_RADIOS:-$default_expected}"
[[ "$EXPECTED_RADIOS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_DIRECT_USB_EXPECTED_RADIOS must be a positive integer."
[[ "$EXPECTED_RADIOS" -eq "$default_expected" ]] ||
    die "Rover ${rover_id} expects ${default_expected} radios, not ${EXPECTED_RADIOS}."

for command in bash iio_info; do
    command -v "$command" >/dev/null 2>&1 ||
        die "Required command is missing: ${command}"
done

rm -f -- "$READY_FILE"

# Boot must not rewrite U-Boot or reset a radio opportunistically. Verify the
# settings established during Rover provisioning and fail closed on drift.
SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
    bash "$LOADER" check-config-all "$EXPECTED_RADIOS"

SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
    bash "$LOADER" load-all "$EXPECTED_RADIOS"

# USB-IIO URIs contain the post-enumeration USB device address, so mapping must
# be regenerated only after every radio reaches its final RAM-booted state.
mapping_tmp="$(mktemp /run/spf-device-mapping.XXXXXX)"
trap 'rm -f -- "${mapping_tmp:-}"' EXIT
bash "$MAPPING_SCRIPT" >"$mapping_tmp"

mapfile -t mapping_lines < <(awk 'NF { print }' "$mapping_tmp")
[[ "${#mapping_lines[@]}" -eq "$EXPECTED_RADIOS" ]] ||
    die "Expected ${EXPECTED_RADIOS} mapping rows; found ${#mapping_lines[@]}."

declare -A seen_ports=()
declare -A seen_addresses=()
for line in "${mapping_lines[@]}"; do
    read -r receiver_port usb_address extra <<<"$line"
    [[ -z "${extra:-}" && "$receiver_port" =~ ^[0-9]+$ &&
        "$usb_address" =~ ^[0-9]+$ ]] ||
        die "Invalid device mapping row: ${line}"
    [[ -z "${seen_ports[$receiver_port]:-}" ]] ||
        die "Duplicate receiver port in device mapping: ${receiver_port}"
    [[ -z "${seen_addresses[$usb_address]:-}" ]] ||
        die "Duplicate USB address in device mapping: ${usb_address}"
    seen_ports["$receiver_port"]=1
    seen_addresses["$usb_address"]=1
    iio_info -u "usb:1.${usb_address}.5" >/dev/null 2>&1 ||
        die "Mapped USB-IIO context usb:1.${usb_address}.5 is unavailable."
done

install -o pi -g pi -m 0644 "$mapping_tmp" /home/pi/device_mapping

SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
    bash "$LOADER" verify-all "$EXPECTED_RADIOS"

spf_git_sha="$(
    git -c "safe.directory=${REPO_ROOT}" \
        -C "$REPO_ROOT" rev-parse --verify HEAD
)" || die "Could not determine the SPF Git commit."
[[ "$spf_git_sha" =~ ^[0-9a-f]{40}$ ]] ||
    die "Invalid SPF Git commit: ${spf_git_sha}"

mkdir -p "$READY_DIR"
{
    printf 'spf_git_sha=%s\n' "$spf_git_sha"
    printf 'rover_id=%s\n' "$rover_id"
    printf 'expected_radios=%s\n' "$EXPECTED_RADIOS"
    printf 'firmware_image_sha256=%s\n' \
        "f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4"
    printf '%s\n' "--- device_mapping ---"
    cat /home/pi/device_mapping
    printf '%s\n' "--- loader_status ---"
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
    SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
        bash "$LOADER" status-all "$EXPECTED_RADIOS"
} >"$READY_FILE"

trap - EXIT
rm -f -- "$mapping_tmp"
printf 'PASS: direct-USB boot preparation is complete.\n'
cat "$READY_FILE"
