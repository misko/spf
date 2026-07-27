#!/usr/bin/env bash
#
# Root-only boot preparation for the recoverable multi-Pluto direct-USB path.
# This script never writes Pluto QSPI.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly MAPPING_SCRIPT="${SCRIPT_DIR}/device_mapping.sh"
readonly READY_DIR="/run/spf"
readonly READY_FILE="${READY_DIR}/direct_usb_ready.json"

FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"
DISABLE_DIRECT_USB="${SPF_DIRECT_USB_DISABLE:-0}"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off|"") return 1 ;;
        *) die "Invalid SPF_DIRECT_USB_DISABLE value: $1" ;;
    esac
}

[[ "${EUID}" -eq 0 ]] || die "prepare_direct_usb_boot.sh must run as root."

if is_true "$DISABLE_DIRECT_USB"; then
    rm -f -- "$READY_FILE"
    printf '%s\n' \
        "Direct-USB RAM loading was explicitly disabled by" \
        "SPF_DIRECT_USB_DISABLE=${DISABLE_DIRECT_USB}."
    exit 0
fi

[[ -f /home/pi/rover_id ]] || die "Missing /home/pi/rover_id."
rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
[[ "$rover_id" =~ ^[1-3]$ ]] || die "Unsupported rover_id: ${rover_id}"

resolver_args=(
    --rover-id "$rover_id"
    --format null
)
if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
    resolver_args+=(--config "$SPF_CAPTURE_CONFIG")
fi
mapfile -d '' -t config_values < <(
    "$PYTHON" -m spf.scripts.rover_capture_config "${resolver_args[@]}"
)
[[ "${#config_values[@]}" -eq 15 ]] ||
    die "Capture config resolver returned ${#config_values[@]} fields, expected 15."
configured_radios="${config_values[5]}"
firmware_release_tag="${config_values[8]}"
firmware_asset_name="${config_values[9]}"
firmware_image_url="${config_values[10]}"
firmware_image_sha256="${config_values[11]}"

for command in bash iio_info "$PYTHON"; do
    command -v "$command" >/dev/null 2>&1 ||
        die "Required command is missing: ${command}"
done

rm -f -- "$READY_FILE"

run_loader() {
    SPF_FIRMWARE_RELEASE_TAG="$firmware_release_tag" \
    SPF_FIRMWARE_ASSET_NAME="$firmware_asset_name" \
    SPF_FIRMWARE_IMAGE_URL="$firmware_image_url" \
    SPF_FIRMWARE_IMAGE_SHA256="$firmware_image_sha256" \
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
    SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
        bash "$LOADER" "$@"
}

attached_radios="$(run_loader discover-count)"
[[ "$attached_radios" =~ ^[0-9]+$ ]] ||
    die "Could not determine the attached Pluto count: ${attached_radios}"
[[ "$attached_radios" -gt 0 ]] || die "No runtime Pluto radios are attached."
[[ "$attached_radios" -eq "$configured_radios" ]] ||
    die "Config has ${configured_radios} receivers but ${attached_radios} Plutos are attached."

# Boot must not rewrite U-Boot. Verify the persistent settings established
# during Rover provisioning without treating the active runtime version as
# QSPI identity. Then always load the exact configured image into RAM, including
# after a Pi-only reboot that left an older RAM image powered.
run_loader check-config-all "$attached_radios"

run_loader load-all "$attached_radios"

# USB-IIO URIs contain the post-enumeration USB device address, so mapping must
# be regenerated only after every radio reaches its final RAM-booted state.
mapping_tmp="$(mktemp /run/spf-device-mapping.XXXXXX)"
trap 'rm -f -- "${mapping_tmp:-}"' EXIT
bash "$MAPPING_SCRIPT" >"$mapping_tmp"

mapfile -t mapping_lines < <(awk 'NF { print }' "$mapping_tmp")
[[ "${#mapping_lines[@]}" -eq "$attached_radios" ]] ||
    die "Expected ${attached_radios} mapping rows; found ${#mapping_lines[@]}."

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

run_loader verify-all "$attached_radios"

mkdir -p "$READY_DIR"
manifest_args=(
    write
    --rover-id "$rover_id"
    --output "$READY_FILE"
    --device-mapping /home/pi/device_mapping
)
if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
    manifest_args+=(--config "$SPF_CAPTURE_CONFIG")
fi
"$PYTHON" -m spf.scripts.pluto_ready_manifest "${manifest_args[@]}"

trap - EXIT
rm -f -- "$mapping_tmp"
printf 'PASS: direct-USB boot preparation is complete.\n'
cat "$READY_FILE"
