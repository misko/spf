#!/usr/bin/env bash
#
# Root-only boot preparation for the recoverable multi-Pluto direct-USB path.
# The normal matching path is read-only. An explicit active-firmware mismatch
# may update only the Pluto firmware partition in QSPI via pluto.frm.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly MAPPING_SCRIPT="${SCRIPT_DIR}/device_mapping.sh"
readonly RADIO_MISSING_HANDLER="${SCRIPT_DIR}/radio_missing_shutdown.sh"
readonly READY_FILE="${SPF_DIRECT_USB_READY_FILE:-/run/spf/direct_usb_ready.json}"
readonly READY_DIR="$(dirname -- "$READY_FILE")"

FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"
DISABLE_DIRECT_USB="${SPF_DIRECT_USB_DISABLE:-0}"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
# Firmware delivery is selected by pluto-firmware.boot-mode in the canonical
# capture config. SPF_PLUTO_RAM_LOAD remains an explicit operational override:
# 1 forces a volatile RAM load, while 0 forces persistent QSPI verification.
RAM_LOAD_OVERRIDE="${SPF_PLUTO_RAM_LOAD:-}"
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off|"") return 1 ;;
        *) die "Invalid boolean value: $1" ;;
    esac
}

[[ "${EUID}" -eq 0 ]] || die "prepare_direct_usb_boot.sh must run as root."

# Readiness is session-bound. Invalidate it before any operation that can fail,
# including configuration parsing, so stale firmware/fingerprint state can
# never authorize capture after a reboot or interrupted firmware operation.
rm -f -- "$READY_FILE"

if is_true "$DISABLE_DIRECT_USB"; then
    printf '%s\n' \
        "Direct-USB firmware preparation was explicitly disabled by" \
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
firmware_gadget_git_sha="${config_values[13]}"
firmware_boot_mode="${config_values[14]}"
# The v2 image publishes its release tag as /opt/VERSIONS device-fw. Derive the
# expected running value from the same config pin instead of maintaining a
# second hard-coded version that can silently lag a firmware rollout.
expected_device_fw="${SPF_PLUTO_EXPECTED_DEVICE_FW:-$firmware_release_tag}"

if [[ -n "$RAM_LOAD_OVERRIDE" ]]; then
    if is_true "$RAM_LOAD_OVERRIDE"; then
        ram_load=1
    else
        ram_load=0
    fi
elif [[ "$firmware_boot_mode" == "ram" ]]; then
    ram_load=1
elif [[ "$firmware_boot_mode" == "qspi" ]]; then
    ram_load=0
else
    die "Unsupported configured firmware boot mode: ${firmware_boot_mode}"
fi

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
if [[ "$attached_radios" -lt "$configured_radios" ]]; then
    "$RADIO_MISSING_HANDLER" "$configured_radios" "$attached_radios"
    die "Config has ${configured_radios} receivers but only ${attached_radios} Plutos are attached."
elif [[ "$attached_radios" -gt "$configured_radios" ]]; then
    die "Config has ${configured_radios} receivers but ${attached_radios} Plutos are attached."
fi

# The persistent AD9361/2r2t configuration is established once during Rover
# provisioning (check_and_set_pluto.sh); it is NOT re-checked per boot. That
# check needs ssh to the Pluto's shared 192.168.2.1 and raced dropbear at boot
# (the radios may not be ssh-ready yet). A wrong config is still caught below by
# verify-all's dual-RX check, which is ssh-free.

if is_true "$ram_load"; then
    # Legacy: RAM-load the exact configured image every boot (volatile, ~30s/radio).
    run_loader load-all "$attached_radios"
else
    # Default: ensure the gain/RSSI image is persistently in QSPI. Downloads/verifies
    # the pinned image into the local cache (no network if already cached), then
    # reads each active version over USB-IIO. A matching radio never mounts its
    # updater volume. Only an explicit mismatch opens that serial's mass-storage
    # device to flash pluto.frm (mtd3 only). No ssh/shared-192.168.2.1 race.
    run_loader download
    SPF_PLUTO_EXPECTED_DEVICE_FW="$expected_device_fw" \
    SPF_PLUTO_EXPECTED_GADGET_SHA="$firmware_gadget_git_sha" \
    SPF_FIRMWARE_DFU="${FIRMWARE_CACHE}/${firmware_asset_name}" \
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
        bash "${SCRIPT_DIR}/ensure_pluto_qspi.sh" "$attached_radios"
    # Keep the per-boot verify below ssh-free too: interface-6 + USB-IIO + dual-RX
    # (all by USB serial) already prove the radio is collector-ready; the ssh
    # daemon-pid check is a provisioning-time concern and would reintroduce the
    # shared-IP ssh race.
    export SPF_PLUTO_SKIP_SSH_VERIFY=1
fi

# USB-IIO URIs contain the post-enumeration USB device address, so mapping must
# be regenerated only after every radio reaches its final booted state.
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
