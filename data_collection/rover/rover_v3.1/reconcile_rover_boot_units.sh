#!/usr/bin/env bash
#
# Converge the root-managed Rover systemd units, and the /usr/local/bin/rover
# CLI symlink, on the files in this checkout.
#
# Exit status (units only -- CLI symlink reconciliation is advisory and never
# changes the status):
#   0  already converged
#   10 units installed and enabled; one reboot is required
#   75 this exact desired state already requested a reboot but still drifted
#   any other non-zero value is an installation or verification failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly STATE_FILE_NAME="boot-unit-reconcile-attempt"

# Shared fleet defaults (timezone, capture env keys). Sourced, never executed
# for side effects -- the file is documented as side-effect free.
# shellcheck source=rover_env_defaults.sh
source "${SCRIPT_DIR}/rover_env_defaults.sh"

SYSTEMD_DIR="/etc/systemd/system"
STATE_DIR="/var/lib/spf"
SYSTEMCTL="systemctl"
CLI_LINK="/usr/local/bin/rover"
FLEET_TIMEZONE="${SPF_ROVER_TIMEZONE:-}"
USE_SUDO=1

readonly -a MANAGED_UNITS=(
    spf-rover-update.service
    spf-pluto-direct-usb.service
    spf-direct-usb-preflight.service
    mavlink_controller.service
)
readonly -a ENABLED_UNITS=(
    spf-rover-update.service
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

# The `rover` CLI is a root-owned symlink into this checkout -- the same kind of
# artifact as the units below, and it drifts the same way. Its usual failure is
# simply never having been created: `rover install` joined provision_rover.sh's
# base stage (afbbbfd, 2026-08-03 19:58) hours after Rover 4 had already run
# that stage, and nothing re-runs a provisioning stage. Converge it here so a
# rover provisioned before any base-stage change still self-heals.
#
# Deliberately advisory, and deliberately outside install_and_verify: a missing
# CLI is a convenience failure, never a reason to fail a boot or to request the
# reconciliation reboot that a drifted unit earns. This function returns 0 on
# every path so it cannot perturb the 0/10/75 contract above.
reconcile_cli_symlink() {
    local want="${SCRIPT_DIR}/rover" want_real have

    if [[ ! -x "$want" ]]; then
        printf 'WARN: rover CLI source is missing or not executable: %s\n' \
            "$want" >&2
        return 0
    fi
    want_real="$(readlink -f -- "$want")"

    if [[ -L "$CLI_LINK" ]]; then
        have="$(readlink -f -- "$CLI_LINK" 2>/dev/null || true)"
        if [[ "$have" == "$want_real" ]]; then
            printf 'PASS: rover CLI symlink is current.\n'
            return 0
        fi
    elif [[ -e "$CLI_LINK" ]]; then
        # A real file here was put there by a person. Replacing it would destroy
        # their work; `rover install` refuses for the same reason.
        printf 'WARN: %s exists and is not a symlink; leaving it untouched.\n' \
            "$CLI_LINK" >&2
        return 0
    fi

    if run_privileged ln -sfn -- "$want" "$CLI_LINK" 2>/dev/null; then
        printf 'PASS: reconciled rover CLI symlink -> %s\n' "$want"
    else
        printf 'WARN: could not reconcile %s; run: sudo %s install\n' \
            "$CLI_LINK" "$want" >&2
    fi
    return 0
}

# Capture filenames are stamped in LOCAL time, so a rover whose timezone drifts
# from the fleet silently names its captures hours -- and possibly a calendar
# day -- away from a sibling rover recording the same emitter pass. Rover 4 did
# exactly this on 2026-08-04: America/Los_Angeles against the fleet's
# Europe/London, NTP-correct throughout, nothing erroring anywhere.
#
# Converged here for the same reason as the units and the CLI symlink: a rover
# provisioned at any point in the past must end up in the current desired state
# without anyone remembering to intervene.
#
# Advisory, like the CLI symlink: a wrong timezone misnames data, it does not
# stop a mission, so it must never fail a boot or earn a reboot. Returns 0 on
# every path.
reconcile_timezone() {
    local want current
    want="$FLEET_TIMEZONE"
    if [[ -z "$want" ]]; then
        want="$(spf_fleet_timezone 2>/dev/null || true)"
    fi
    [[ -n "$want" ]] || return 0

    current="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
    if [[ "$current" == "$want" ]]; then
        printf 'PASS: timezone is %s.\n' "$want"
        return 0
    fi

    if run_privileged timedatectl set-timezone "$want" 2>/dev/null; then
        printf 'PASS: timezone %s -> %s (capture filenames use local time).\n' \
            "${current:-unknown}" "$want"
    else
        printf 'WARN: could not set timezone to %s (is %s); capture filenames will not match the fleet.\n' \
            "$want" "${current:-unknown}" >&2
    fi
    return 0
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
            --cli-link)
                [[ "$#" -ge 2 ]] || die "--cli-link requires a value."
                CLI_LINK="$2"
                shift 2
                ;;
            --timezone)
                [[ "$#" -ge 2 ]] || die "--timezone requires a value."
                FLEET_TIMEZONE="$2"
                shift 2
                ;;
            --no-timezone)
                FLEET_TIMEZONE=""
                shift
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

    # Before the units, and never able to change the status they return.
    reconcile_timezone
    reconcile_cli_symlink
    install_and_verify
}

main "$@"
