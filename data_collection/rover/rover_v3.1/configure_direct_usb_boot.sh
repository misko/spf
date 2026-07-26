#!/usr/bin/env bash
#
# Install or remove the opt-in Rover direct-USB boot workflow.
#
# Enabling this workflow disables the legacy mission service so a motion-capable
# production collector cannot race the motion-free boot qualification capture.
# Restoring the legacy service never starts it immediately; it takes effect at
# the next boot or after an explicit operator start.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_SOURCE="${SCRIPT_DIR}/direct_usb_boot.env.example"
readonly ENV_DIR="/etc/spf"
readonly ENV_DEST="${ENV_DIR}/direct_usb_boot.env"
readonly LOADER_UNIT="spf-pluto-direct-usb.service"
readonly PREFLIGHT_UNIT="spf-direct-usb-preflight.service"
readonly LEGACY_UNIT="mavlink_controller.service"
readonly SYSTEMD_DIR="/etc/systemd/system"
readonly READY_FILE="/run/spf/direct_usb_ready"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

unit_state() {
    local operation="$1"
    local unit="$2"
    local state
    state="$(systemctl "$operation" "$unit" 2>/dev/null || true)"
    printf '%s\n' "${state:-unknown}"
}

show_status() {
    printf '%-34s enabled=%-10s active=%s\n' \
        "$LEGACY_UNIT" \
        "$(unit_state is-enabled "$LEGACY_UNIT")" \
        "$(unit_state is-active "$LEGACY_UNIT")"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$LOADER_UNIT" \
        "$(unit_state is-enabled "$LOADER_UNIT")" \
        "$(unit_state is-active "$LOADER_UNIT")"
    printf '%-34s enabled=%-10s active=%s\n' \
        "$PREFLIGHT_UNIT" \
        "$(unit_state is-enabled "$PREFLIGHT_UNIT")" \
        "$(unit_state is-active "$PREFLIGHT_UNIT")"
    if [[ -f "$READY_FILE" ]]; then
        printf 'ready_stamp=%s\n' "$READY_FILE"
    else
        printf 'ready_stamp=absent\n'
    fi
}

enable_direct_usb() {
    [[ "${EUID}" -eq 0 ]] ||
        die "enable must run as root."
    [[ -f "$ENV_SOURCE" ]] || die "Missing environment template: ${ENV_SOURCE}"
    [[ -f "${SCRIPT_DIR}/${LOADER_UNIT}" ]] ||
        die "Missing unit source: ${SCRIPT_DIR}/${LOADER_UNIT}"
    [[ -f "${SCRIPT_DIR}/${PREFLIGHT_UNIT}" ]] ||
        die "Missing unit source: ${SCRIPT_DIR}/${PREFLIGHT_UNIT}"

    systemctl stop "$LEGACY_UNIT"
    systemctl disable "$LEGACY_UNIT"

    install -d -m 0755 "$ENV_DIR"
    if [[ -f "$ENV_DEST" ]]; then
        printf 'Preserving existing environment: %s\n' "$ENV_DEST"
    else
        install -m 0644 "$ENV_SOURCE" "$ENV_DEST"
        printf 'Installed environment template: %s\n' "$ENV_DEST"
    fi
    install -m 0644 \
        "${SCRIPT_DIR}/${LOADER_UNIT}" \
        "${SYSTEMD_DIR}/${LOADER_UNIT}"
    install -m 0644 \
        "${SCRIPT_DIR}/${PREFLIGHT_UNIT}" \
        "${SYSTEMD_DIR}/${PREFLIGHT_UNIT}"
    systemctl daemon-reload
    systemctl enable "$LOADER_UNIT" "$PREFLIGHT_UNIT"

    printf '%s\n' \
        "PASS: direct-USB boot workflow enabled for the next boot." \
        "Review ${ENV_DEST}, then reboot or start ${PREFLIGHT_UNIT} explicitly."
    show_status
}

restore_legacy() {
    [[ "${EUID}" -eq 0 ]] ||
        die "restore-legacy must run as root."
    systemctl stop "$PREFLIGHT_UNIT" "$LOADER_UNIT"
    systemctl disable "$PREFLIGHT_UNIT" "$LOADER_UNIT"
    rm -f -- "$READY_FILE"
    systemctl enable "$LEGACY_UNIT"

    printf '%s\n' \
        "PASS: legacy mission service enabled for the next boot." \
        "The mission service was not started. Roll back RAM firmware separately" \
        "before an IIO capture, then start the service only when motion is safe."
    show_status
}

usage() {
    cat <<EOF
Usage: sudo $(basename "$0") COMMAND

Commands:
  enable          Install and enable the direct-USB loader and 100-frame
                  motion-free boot preflight; disable the legacy mission unit.
  restore-legacy  Stop and disable direct-USB boot units; enable (but do not
                  start) the legacy mission unit.
  status          Show enabled/active state and ready-stamp presence.
EOF
}

main() {
    case "${1:-}" in
        enable)
            [[ "$#" -eq 1 ]] || die "enable takes no arguments."
            enable_direct_usb
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
