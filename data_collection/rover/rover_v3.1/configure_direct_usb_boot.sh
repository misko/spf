#!/usr/bin/env bash
#
# Install one mutually exclusive Rover 3.1 boot workflow.
#
# None of these commands starts the motion-capable production collector.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DIRECT_ENV_SOURCE="${SCRIPT_DIR}/direct_usb_boot.env.example"
readonly COLLECTION_ENV_SOURCE="${SCRIPT_DIR}/rover_collection.env.example"
readonly FIRMWARE_LOADER="${SCRIPT_DIR}/load_direct_usb_firmware.sh"
readonly ENV_DIR="/etc/spf"
readonly DIRECT_ENV_DEST="${ENV_DIR}/direct_usb_boot.env"
readonly COLLECTION_ENV_DEST="${ENV_DIR}/rover_collection.env"
readonly LOADER_UNIT="spf-pluto-direct-usb.service"
readonly PREFLIGHT_UNIT="spf-direct-usb-preflight.service"
readonly PRODUCTION_UNIT="mavlink_controller.service"
readonly SYSTEMD_DIR="/etc/systemd/system"
readonly DROPIN_DIR="${SYSTEMD_DIR}/${PRODUCTION_UNIT}.d"
readonly DROPIN="${DROPIN_DIR}/10-direct-usb.conf"
readonly READY_FILE="/run/spf/direct_usb_ready"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "$1 must run as root."
}

unit_state() {
    local operation="$1"
    local unit="$2"
    local state
    state="$(systemctl "$operation" "$unit" 2>/dev/null || true)"
    printf '%s\n' "${state:-unknown}"
}

collection_profile() {
    if [[ -f "$COLLECTION_ENV_DEST" ]]; then
        sed -n 's/^SPF_CAPTURE_PROFILE=//p' "$COLLECTION_ENV_DEST" | tail -1
    else
        printf 'legacy_iio_v4\n'
    fi
}

ram_load_setting() {
    local value
    if [[ -f "$DIRECT_ENV_DEST" ]]; then
        value="$(
            sed -n 's/^SPF_DIRECT_USB_DISABLE=//p' "$DIRECT_ENV_DEST" | tail -1
        )"
        printf '%s\n' "${value:-0}"
    else
        printf '0\n'
    fi
}

show_status() {
    printf 'capture_profile=%s\n' "$(collection_profile)"
    printf 'ram_load_disabled=%s\n' "$(ram_load_setting)"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$PRODUCTION_UNIT" \
        "$(unit_state is-enabled "$PRODUCTION_UNIT")" \
        "$(unit_state is-active "$PRODUCTION_UNIT")"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$LOADER_UNIT" \
        "$(unit_state is-enabled "$LOADER_UNIT")" \
        "$(unit_state is-active "$LOADER_UNIT")"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$PREFLIGHT_UNIT" \
        "$(unit_state is-enabled "$PREFLIGHT_UNIT")" \
        "$(unit_state is-active "$PREFLIGHT_UNIT")"
    printf 'firmware_before_mavlink=base-unit\n'
    [[ ! -f "$DROPIN" ]] || printf 'obsolete_direct_usb_dropin=installed\n'
    if [[ -f "$READY_FILE" ]]; then
        printf 'ready_stamp=%s\n' "$READY_FILE"
    else
        printf 'ready_stamp=absent\n'
    fi
}

install_units_and_environments() {
    [[ -f "$DIRECT_ENV_SOURCE" ]] ||
        die "Missing environment template: ${DIRECT_ENV_SOURCE}"
    [[ -f "$COLLECTION_ENV_SOURCE" ]] ||
        die "Missing environment template: ${COLLECTION_ENV_SOURCE}"
    for unit in "$LOADER_UNIT" "$PREFLIGHT_UNIT" "$PRODUCTION_UNIT"; do
        [[ -f "${SCRIPT_DIR}/${unit}" ]] ||
            die "Missing unit source: ${SCRIPT_DIR}/${unit}"
    done

    install -d -m 0755 "$ENV_DIR"
    if [[ ! -f "$DIRECT_ENV_DEST" ]]; then
        install -m 0644 "$DIRECT_ENV_SOURCE" "$DIRECT_ENV_DEST"
        printf 'Installed environment template: %s\n' "$DIRECT_ENV_DEST"
    else
        printf 'Preserving existing environment: %s\n' "$DIRECT_ENV_DEST"
    fi
    if [[ ! -f "$COLLECTION_ENV_DEST" ]]; then
        install -m 0644 "$COLLECTION_ENV_SOURCE" "$COLLECTION_ENV_DEST"
        printf 'Installed environment template: %s\n' "$COLLECTION_ENV_DEST"
    else
        printf 'Preserving existing environment: %s\n' "$COLLECTION_ENV_DEST"
    fi
    install -m 0644 "${SCRIPT_DIR}/${LOADER_UNIT}" \
        "${SYSTEMD_DIR}/${LOADER_UNIT}"
    install -m 0644 "${SCRIPT_DIR}/${PREFLIGHT_UNIT}" \
        "${SYSTEMD_DIR}/${PREFLIGHT_UNIT}"
    install -m 0644 "${SCRIPT_DIR}/${PRODUCTION_UNIT}" \
        "${SYSTEMD_DIR}/${PRODUCTION_UNIT}"
}

set_ram_load_disabled() {
    local disabled="$1"
    if grep -q '^SPF_DIRECT_USB_DISABLE=' "$DIRECT_ENV_DEST"; then
        sed -i "s/^SPF_DIRECT_USB_DISABLE=.*/SPF_DIRECT_USB_DISABLE=${disabled}/" \
            "$DIRECT_ENV_DEST"
    else
        printf 'SPF_DIRECT_USB_DISABLE=%s\n' "$disabled" >>"$DIRECT_ENV_DEST"
    fi
}

cache_firmware_image() {
    local cache_dir
    cache_dir="$(
        sed -n 's/^SPF_FIRMWARE_CACHE_DIR=//p' "$DIRECT_ENV_DEST" | tail -1
    )"
    cache_dir="${cache_dir:-/home/pi/.cache/spf/firmware}"
    SPF_FIRMWARE_CACHE_DIR="$cache_dir" bash "$FIRMWARE_LOADER" download
}

set_profile() {
    local profile="$1"
    if grep -q '^SPF_CAPTURE_PROFILE=' "$COLLECTION_ENV_DEST"; then
        sed -i "s/^SPF_CAPTURE_PROFILE=.*/SPF_CAPTURE_PROFILE=${profile}/" \
            "$COLLECTION_ENV_DEST"
    else
        printf 'SPF_CAPTURE_PROFILE=%s\n' "$profile" >>"$COLLECTION_ENV_DEST"
    fi
}

stop_all() {
    local unit
    for unit in "$PRODUCTION_UNIT" "$PREFLIGHT_UNIT" "$LOADER_UNIT"; do
        systemctl stop "$unit" 2>/dev/null || true
    done
}

enable_qualification() {
    require_root "qualify"
    stop_all
    install_units_and_environments
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PRODUCTION_UNIT"
    systemctl enable "$LOADER_UNIT" "$PREFLIGHT_UNIT"
    printf '%s\n' \
        "PASS: 100-frame motion-free direct-USB qualification is enabled." \
        "Production collection remains disabled."
    show_status
}

enable_default_production() {
    require_root "production-default"
    stop_all
    install_units_and_environments
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl enable "$LOADER_UNIT" "$PRODUCTION_UNIT"
    printf '%s\n' \
        "PASS: RAM firmware preparation before MAVLink is enabled." \
        "The existing capture profile is preserved." \
        "No service was started."
    show_status
}

enable_production() {
    local profile="$1"
    require_root "production"
    stop_all
    install_units_and_environments
    set_profile "$profile"
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl enable "$LOADER_UNIT" "$PRODUCTION_UNIT"
    printf '%s\n' \
        "PASS: ${profile} production boot is enabled." \
        "No service was started. Review ${COLLECTION_ENV_DEST}; use" \
        "SPF_BOOT_VALIDATE_ONLY=1 for the first reboot."
    show_status
}

restore_legacy() {
    require_root "restore-legacy"
    stop_all
    install_units_and_environments
    set_profile legacy_iio_v4
    set_ram_load_disabled 1
    rm -f -- "$DROPIN" "$READY_FILE"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl enable "$LOADER_UNIT" "$PRODUCTION_UNIT"
    printf '%s\n' \
        "PASS: explicit stock-QSPI legacy IIO boot policy is enabled." \
        "The firmware prerequisite still runs before MAVLink but exits without" \
        "RAM-loading because SPF_DIRECT_USB_DISABLE=1." \
        "Reset each Pluto into QSPI with rollback-all (or a full power cycle)" \
        "before starting collection; this command does not reset radios." \
        "No service was started."
    show_status
}

usage() {
    cat <<EOF
Usage: sudo $(basename "$0") COMMAND

Commands:
  qualify        Enable loader + one 100-frame fake-drone qualification.
  production-default
                 RAM-load before MAVLink; preserve the current capture profile.
  production-v4 Enable original production loop through direct USB, v4 Zarr.
  production-v7 Enable original production loop through direct USB, v7 Zarr.
  restore-legacy
                 Explicitly opt out of RAM loading and use legacy IIO/v4.
  status         Show profile, unit exclusivity, ordering, and ready stamp.

The historical 'enable' command remains an alias for 'qualify'.
No command starts the motion-capable production service immediately.
EOF
}

main() {
    case "${1:-}" in
        enable|qualify)
            [[ "$#" -eq 1 ]] || die "$1 takes no arguments."
            enable_qualification
            ;;
        production-default)
            [[ "$#" -eq 1 ]] || die "production-default takes no arguments."
            enable_default_production
            ;;
        production-v4)
            [[ "$#" -eq 1 ]] || die "production-v4 takes no arguments."
            enable_production direct_usb_v4
            ;;
        production-v7)
            [[ "$#" -eq 1 ]] || die "production-v7 takes no arguments."
            enable_production direct_usb_v7
            ;;
        restore-legacy)
            [[ "$#" -eq 1 ]] || die "restore-legacy takes no arguments."
            restore_legacy
            ;;
        status)
            [[ "$#" -eq 1 ]] || die "status takes no arguments."
            show_status
            ;;
        -h|--help|help)
            usage
            ;;
        "")
            usage >&2
            exit 2
            ;;
        *)
            usage >&2
            die "Unknown command: ${1}"
            ;;
    esac
}

main "$@"
