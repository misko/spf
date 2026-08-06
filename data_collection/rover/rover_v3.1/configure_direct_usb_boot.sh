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
readonly READY_FILE="/run/spf/direct_usb_ready.json"
readonly UPDATE_UNIT="spf-rover-update.service"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "$1 must run as root."
}

require_rover_identity() {
    local rover_id
    [[ -f /home/pi/rover_id ]] ||
        die "Missing /home/pi/rover_id; refusing to select a production config."
    rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
    [[ "$rover_id" =~ ^[1-4]$ ]] ||
        die "Unsupported rover_id: ${rover_id}"
}

unit_state() {
    local operation="$1"
    local unit="$2"
    local state
    state="$(systemctl "$operation" "$unit" 2>/dev/null || true)"
    printf '%s\n' "${state:-unknown}"
}

collection_config() {
    if [[ -f "$COLLECTION_ENV_DEST" ]]; then
        value="$(
            sed -n 's/^SPF_CAPTURE_CONFIG=//p' "$COLLECTION_ENV_DEST" | tail -1
        )"
        printf '%s\n' "${value:-canonical-by-rover-id}"
    else
        printf 'canonical-by-rover-id\n'
    fi
}

direct_usb_disabled_setting() {
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
    printf 'capture_config=%s\n' "$(collection_config)"
    printf 'direct_usb_disabled=%s\n' "$(direct_usb_disabled_setting)"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$UPDATE_UNIT" \
        "$(unit_state is-enabled "$UPDATE_UNIT")" \
        "$(unit_state is-active "$UPDATE_UNIT")"
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
    for unit in "$UPDATE_UNIT" "$LOADER_UNIT" "$PREFLIGHT_UNIT" "$PRODUCTION_UNIT"; do
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
    install -m 0644 "${SCRIPT_DIR}/${UPDATE_UNIT}" \
        "${SYSTEMD_DIR}/${UPDATE_UNIT}"
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
    local cache_dir config_override rover_id
    local -a resolver_args config_values
    cache_dir="$(
        sed -n 's/^SPF_FIRMWARE_CACHE_DIR=//p' "$DIRECT_ENV_DEST" | tail -1
    )"
    cache_dir="${cache_dir:-/home/pi/.cache/spf/firmware}"
    rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
    resolver_args=(--rover-id "$rover_id" --format null)
    config_override="$(collection_config)"
    if [[ "$config_override" != "canonical-by-rover-id" ]]; then
        resolver_args+=(--config "$config_override")
    fi
    mapfile -d '' -t config_values < <(
        /home/pi/spf-virtualenv/bin/python3 \
            -m spf.scripts.rover_capture_config "${resolver_args[@]}"
    )
    # -ge, not -eq: the resolver gained a 16th field (device-fw) in d79fa576 and only
    # one of four call sites was updated, silently breaking the other three. This
    # script indexes fields 0-14 only, so any resolver emitting at least 16 is
    # compatible; a future field addition cannot break it again.
    [[ "${#config_values[@]}" -ge 16 ]] ||
        die "Capture config resolver returned the wrong firmware plan."
    SPF_FIRMWARE_RELEASE_TAG="${config_values[8]}" \
    SPF_FIRMWARE_ASSET_NAME="${config_values[9]}" \
    SPF_FIRMWARE_IMAGE_URL="${config_values[10]}" \
    SPF_FIRMWARE_IMAGE_SHA256="${config_values[11]}" \
    SPF_FIRMWARE_CACHE_DIR="$cache_dir" \
        bash "$FIRMWARE_LOADER" download
}

use_canonical_config() {
    sed -i \
        '/^SPF_CAPTURE_PROFILE=/d; /^SPF_CAPTURE_CONFIG=/d' \
        "$COLLECTION_ENV_DEST"
}

stop_all() {
    local unit
    for unit in "$PRODUCTION_UNIT" "$PREFLIGHT_UNIT" "$LOADER_UNIT" "$UPDATE_UNIT"; do
        systemctl stop "$unit" 2>/dev/null || true
    done
}

enable_qualification() {
    require_root "qualify"
    require_rover_identity
    stop_all
    install_units_and_environments
    use_canonical_config
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PRODUCTION_UNIT"
    systemctl enable "$UPDATE_UNIT" "$LOADER_UNIT" "$PREFLIGHT_UNIT"
    printf '%s\n' \
        "PASS: 100-frame motion-free direct-USB qualification is enabled." \
        "Production collection remains disabled."
    show_status
}

enable_default_production() {
    require_root "production-default"
    require_rover_identity
    stop_all
    install_units_and_environments
    use_canonical_config
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl enable "$UPDATE_UNIT" "$LOADER_UNIT" "$PRODUCTION_UNIT"
    printf '%s\n' \
        "PASS: config-selected firmware preparation before MAVLink is enabled." \
        "The canonical Rover V7 capture config is selected by rover_id." \
        "No service was started."
    show_status
}

enable_production() {
    require_root "production"
    require_rover_identity
    stop_all
    install_units_and_environments
    use_canonical_config
    set_ram_load_disabled 0
    cache_firmware_image
    rm -f -- "$DROPIN"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl enable "$UPDATE_UNIT" "$LOADER_UNIT" "$PRODUCTION_UNIT"
    printf '%s\n' \
        "PASS: canonical V7 production boot is enabled." \
        "No service was started. Review ${COLLECTION_ENV_DEST}. To bring the" \
        "rover up without letting it move, use 'qualify' instead: it disables" \
        "the mission unit outright."
    show_status
}

restore_legacy() {
    require_root "restore-legacy"
    stop_all
    install_units_and_environments
    use_canonical_config
    set_ram_load_disabled 1
    rm -f -- "$DROPIN" "$READY_FILE"
    systemctl daemon-reload
    systemctl disable "$PREFLIGHT_UNIT"
    systemctl disable "$PRODUCTION_UNIT"
    systemctl enable "$UPDATE_UNIT" "$LOADER_UNIT"
    printf '%s\n' \
        "PASS: explicit stock-QSPI recovery policy is enabled." \
        "The firmware prerequisite still runs before MAVLink but exits without" \
        "direct-USB preparation because SPF_DIRECT_USB_DISABLE=1." \
        "This command does not reset radios or replace the QSPI firmware." \
        "Automatic production collection is disabled."
    show_status
}

usage() {
    cat <<EOF
Usage: sudo $(basename "$0") COMMAND

Commands:
  qualify        Enable loader + one 100-frame fake-drone qualification.
  production-default
                 Prepare the YAML-selected firmware before MAVLink and use the
                 canonical Rover V7 config (currently persistent QSPI).
  production-v7 Alias for production-default.
  restore-legacy
                 Explicitly opt out of direct-USB preparation for legacy recovery.
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
            die "production-v4 was retired; production uses the canonical V7 config"
            ;;
        production-v7)
            [[ "$#" -eq 1 ]] || die "production-v7 takes no arguments."
            enable_production
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
