#!/usr/bin/env bash
#
# Root-only boot preparation for the recoverable multi-Pluto direct-USB path.
# This script never writes Pluto QSPI.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly MAPPING_SCRIPT="${SCRIPT_DIR}/device_mapping.sh"
readonly READY_FILE="${SPF_DIRECT_USB_READY_FILE:-/run/spf/direct_usb_ready.json}"
readonly READY_DIR="$(dirname -- "$READY_FILE")"

FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"
DISABLE_DIRECT_USB="${SPF_DIRECT_USB_DISABLE:-0}"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
# Firmware delivery: default is to persistently flash the gain/RSSI image to the
# Pluto QSPI once and, on every boot, only re-flash when the running version does
# not match EXPECTED_DEVICE_FW (fast steady-state boot -- no per-boot RAM load).
# Set SPF_PLUTO_RAM_LOAD=1 to fall back to the legacy volatile RAM load.
RAM_LOAD="${SPF_PLUTO_RAM_LOAD:-0}"
# Expected running /opt/VERSIONS device-fw for the pinned direct-USB image. Tied
# to the image the loader downloads; bump both together (overridable via
# /etc/spf/*.env).
EXPECTED_DEVICE_FW="${SPF_PLUTO_EXPECTED_DEVICE_FW:-v0.38_plutoplus_with_timestamping-9-g7b02}"

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

# Readiness is session-bound. Invalidate it before any operation that can fail,
# including configuration parsing, so stale firmware/fingerprint state can
# never authorize capture after a reboot or interrupted RAM load.
rm -f -- "$READY_FILE"

if is_true "$DISABLE_DIRECT_USB"; then
    printf '%s\n' \
        "Direct-USB RAM loading was explicitly disabled by" \
        "SPF_DIRECT_USB_DISABLE=${DISABLE_DIRECT_USB}."
    exit 0
fi

if [[ -n "${SPF_ROVER_ID:-}" ]]; then
    rover_id="${SPF_ROVER_ID}"
else
    [[ -f /home/pi/rover_id ]] || die \
        "Missing /home/pi/rover_id and SPF_ROVER_ID was not provided."
    rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
fi
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

run_loader() {
    SPF_FIRMWARE_RELEASE_TAG="$firmware_release_tag" \
    SPF_FIRMWARE_ASSET_NAME="$firmware_asset_name" \
    SPF_FIRMWARE_IMAGE_URL="$firmware_image_url" \
    SPF_FIRMWARE_IMAGE_SHA256="$firmware_image_sha256" \
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
    SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
        bash "$LOADER" "$@"
}

# Wait (bounded) for every configured Pluto to USB-enumerate before counting.
# Previously the unit's network-online.target dependency incidentally delayed
# this step until the USB tree had settled; now that the loader no longer waits
# on the network, poll for the radios directly so a slightly-late enumeration
# does not fail the whole boot. No LAN/internet involved (USB only).
PLUTO_DISCOVER_TIMEOUT="${SPF_PLUTO_DISCOVER_TIMEOUT:-30}"
attached_radios=0
discover_deadline=$((SECONDS + PLUTO_DISCOVER_TIMEOUT))
while true; do
    attached_radios="$(run_loader discover-count)"
    [[ "$attached_radios" =~ ^[0-9]+$ ]] ||
        die "Could not determine the attached Pluto count: ${attached_radios}"
    [[ "$attached_radios" -eq "$configured_radios" ]] && break
    (( SECONDS < discover_deadline )) || break
    printf 'Waiting for Plutos to enumerate: found %s of %s.\n' \
        "$attached_radios" "$configured_radios"
    sleep 1
done
[[ "$attached_radios" -gt 0 ]] || die "No runtime Pluto radios are attached."
[[ "$attached_radios" -eq "$configured_radios" ]] ||
    die "Config has ${configured_radios} receivers but ${attached_radios} Plutos are attached."

# Boot must not rewrite U-Boot. Verify the persistent AD9361/2r2t settings
# established during Rover provisioning without treating the active runtime
# version as QSPI identity.
run_loader check-config-all "$attached_radios"

if is_true "$RAM_LOAD"; then
    # Legacy: RAM-load the exact configured image every boot (volatile, ~30s/radio).
    run_loader load-all "$attached_radios"
else
    # Default: ensure the gain/RSSI image is persistently in QSPI. Downloads/verifies
    # the pinned image into the local cache (no network if already cached), then
    # flashes pluto.frm (mtd3 only) to any radio whose running firmware != expected.
    # Radios already on the expected build are skipped -- no DFU dance, no reboot.
    run_loader download
    SPF_PLUTO_EXPECTED_DEVICE_FW="$EXPECTED_DEVICE_FW" \
    SPF_FIRMWARE_DFU="${FIRMWARE_CACHE}/${firmware_asset_name}" \
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
        bash "${SCRIPT_DIR}/ensure_pluto_qspi.sh" "$attached_radios"
fi

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
