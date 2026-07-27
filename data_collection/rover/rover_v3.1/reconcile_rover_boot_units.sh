#!/usr/bin/env bash
#
# Converge the root-managed Rover systemd units on the files in this checkout.
#
# Exit status:
#   0  already converged
#   10 units installed and enabled; one reboot is required
#   75 this exact desired state already requested a reboot but still drifted
#   any other non-zero value is an installation or verification failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly STATE_FILE_NAME="boot-unit-reconcile-attempt"

SYSTEMD_DIR="/etc/systemd/system"
STATE_DIR="/var/lib/spf"
SYSTEMCTL="systemctl"
USE_SUDO=1

readonly -a MANAGED_UNITS=(
    spf-pluto-direct-usb.service
    spf-direct-usb-preflight.service
    mavlink_controller.service
)
readonly -a ENABLED_UNITS=(
    spf-pluto-direct-usb.service
    mavlink_controller.service
)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [TEST OPTIONS]

Production use takes no arguments. The options below isolate regression tests:
  --systemd-dir DIR
  --state-dir DIR
  --systemctl COMMAND
  --unprivileged
EOF
}

run_privileged() {
    if [[ "$USE_SUDO" -eq 1 ]]; then
        sudo -n "$@"
    else
        "$@"
    fi
}

unit_files_match() {
    local unit
    for unit in "${MANAGED_UNITS[@]}"; do
        [[ -f "${SCRIPT_DIR}/${unit}" ]] ||
            die "Missing managed unit source: ${SCRIPT_DIR}/${unit}"
        cmp -s -- "${SCRIPT_DIR}/${unit}" "${SYSTEMD_DIR}/${unit}" ||
            return 1
    done
}

required_units_enabled() {
    local unit
    for unit in "${ENABLED_UNITS[@]}"; do
        run_privileged "$SYSTEMCTL" is-enabled --quiet "$unit" ||
            return 1
    done
}

desired_state_record() {
    local git_commit unit unit_hash
    git_commit="$(
        git -C "${SCRIPT_DIR}/../../.." rev-parse --verify HEAD
    )" || die "Cannot resolve the SPF Git commit."
    printf 'git_commit=%s\n' "$git_commit"
    for unit in "${MANAGED_UNITS[@]}"; do
        unit_hash="$(sha256sum "${SCRIPT_DIR}/${unit}" | awk '{print $1}')"
        printf 'unit_sha256[%s]=%s\n' "$unit" "$unit_hash"
    done
}

install_and_verify() {
    local marker_file marker_source marker_target_tmp
    local state_record desired_state previous_state unit unit_target_tmp
    local -a temporary_targets=()

    marker_file="${STATE_DIR}/${STATE_FILE_NAME}"
    state_record="$(desired_state_record)"
    desired_state="$(
        printf '%s\n' "$state_record" | sha256sum | awk '{print $1}'
    )"

    if unit_files_match && required_units_enabled; then
        if [[ -e "$marker_file" ]]; then
            run_privileged rm -f -- "$marker_file"
        fi
        printf 'PASS: Rover boot units already match Git state %s.\n' \
            "$desired_state"
        return 0
    fi

    previous_state=""
    if [[ -f "$marker_file" ]]; then
        previous_state="$(
            sed -n 's/^desired_state=//p' "$marker_file" | head -1
        )"
    fi
    if [[ "$previous_state" == "$desired_state" ]]; then
        printf '%s\n' \
            "ERROR: Rover boot-unit drift remains after the one permitted" \
            "reconciliation reboot for desired state ${desired_state}." \
            "Refusing another reboot; inspect ${marker_file} and systemd units." \
            >&2
        return 75
    fi

    cleanup_targets() {
        local target
        trap - EXIT
        for target in "${temporary_targets[@]}"; do
            run_privileged rm -f -- "$target" >/dev/null 2>&1 || true
        done
        [[ -z "${marker_source:-}" ]] ||
            rm -f -- "$marker_source" >/dev/null 2>&1 || true
    }
    trap cleanup_targets EXIT

    run_privileged install -d -m 0755 "$SYSTEMD_DIR" "$STATE_DIR"
    for unit in "${MANAGED_UNITS[@]}"; do
        unit_target_tmp="${SYSTEMD_DIR}/.spf-${unit}.new.$$"
        temporary_targets+=("$unit_target_tmp")
        run_privileged install -m 0644 \
            "${SCRIPT_DIR}/${unit}" "$unit_target_tmp"
        cmp -s -- "${SCRIPT_DIR}/${unit}" "$unit_target_tmp" ||
            die "Temporary unit verification failed: ${unit}"
        run_privileged mv -f -- \
            "$unit_target_tmp" "${SYSTEMD_DIR}/${unit}"
    done

    unit_files_match ||
        die "Installed Rover unit files failed post-install verification."
    run_privileged "$SYSTEMCTL" daemon-reload
    run_privileged "$SYSTEMCTL" enable "${ENABLED_UNITS[@]}"
    required_units_enabled ||
        die "Required Rover units were not enabled after installation."
    unit_files_match ||
        die "Rover unit files changed during systemd reconciliation."

    marker_source="$(mktemp /tmp/spf-boot-unit-state.XXXXXX)"
    {
        printf 'desired_state=%s\n' "$desired_state"
        printf '%s\n' "$state_record"
    } >"$marker_source"
    marker_target_tmp="${STATE_DIR}/.${STATE_FILE_NAME}.new.$$"
    temporary_targets+=("$marker_target_tmp")
    run_privileged install -m 0644 "$marker_source" "$marker_target_tmp"
    run_privileged mv -f -- "$marker_target_tmp" "$marker_file"

    printf '%s\n' \
        "PASS: installed and verified Rover boot units for ${desired_state}." \
        "A single reconciliation reboot is now required."
    cleanup_targets
    return 10
}

main() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --systemd-dir)
                [[ "$#" -ge 2 ]] || die "--systemd-dir requires a value."
                SYSTEMD_DIR="$2"
                shift 2
                ;;
            --state-dir)
                [[ "$#" -ge 2 ]] || die "--state-dir requires a value."
                STATE_DIR="$2"
                shift 2
                ;;
            --systemctl)
                [[ "$#" -ge 2 ]] || die "--systemctl requires a value."
                SYSTEMCTL="$2"
                shift 2
                ;;
            --unprivileged)
                USE_SUDO=0
                shift
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done

    install_and_verify
}

main "$@"
