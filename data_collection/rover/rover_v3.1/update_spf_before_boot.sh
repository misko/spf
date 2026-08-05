#!/usr/bin/env bash
#
# Update the SPF checkout before any radio discovery, firmware work, or mission
# process starts. Remote access is bounded so an offline field rover can still
# boot from its verified local checkout.

set -euo pipefail
export GIT_TERMINAL_PROMPT=0

readonly REPO_ROOT="${SPF_UPDATE_REPO_ROOT:-/home/pi/spf}"
readonly SCRIPT_DIR="${REPO_ROOT}/data_collection/rover/rover_v3.1"
readonly PROFILE_ENV="${SPF_UPDATE_PROFILE_ENV:-/etc/spf/rover_collection.env}"

if [[ -f "$PROFILE_ENV" ]]; then
    # Root-managed file containing shell-style assignments only.
    # shellcheck disable=SC1090
    source "$PROFILE_ENV"
fi

PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
BOOT_UNIT_RECONCILER="${SPF_BOOT_UNIT_RECONCILER:-${SCRIPT_DIR}/reconcile_rover_boot_units.sh}"
INSTALL_DEPS="${SPF_INSTALL_DEPS:-${SCRIPT_DIR}/install_deps.sh}"
CONSTRAINTS="${SPF_PIP_CONSTRAINTS:-${SCRIPT_DIR}/rover_constraints.txt}"
REMOTE_WAIT_SECONDS="${SPF_UPDATE_REMOTE_WAIT_SECONDS:-12}"
GIT_TIMEOUT_SECONDS="${SPF_UPDATE_GIT_TIMEOUT_SECONDS:-4}"
REBOOT_DELAY_SECONDS="${SPF_UPDATE_REBOOT_DELAY_SECONDS:-15}"
SKIP_SELF_UPDATE="${SPF_SKIP_SELF_UPDATE:-0}"

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

require_nonnegative_integer() {
    [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer."
}

require_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer."
}

request_reboot() {
    printf 'Rebooting in %s seconds so the verified update becomes boot state.\n' \
        "$REBOOT_DELAY_SECONDS"
    sleep "$REBOOT_DELAY_SECONDS"
    if [[ -n "${SPF_TEST_REBOOT_COMMAND:-}" ]]; then
        "$SPF_TEST_REBOOT_COMMAND"
    else
        sudo -n systemctl reboot
    fi
}

reconcile_boot_units() {
    local reconcile_rc=0
    "$BOOT_UNIT_RECONCILER" || reconcile_rc=$?
    case "$reconcile_rc" in
        0) return 0 ;;
        10) return 10 ;;
        75)
            die "Boot-unit reconciliation reboot limit reached; refusing a reboot loop."
            ;;
        *) die "Boot-unit reconciliation failed (status ${reconcile_rc})." ;;
    esac
}

fetch_main_bounded() {
    local attempt_timeout deadline remaining
    deadline=$((SECONDS + REMOTE_WAIT_SECONDS))
    while (( SECONDS < deadline )); do
        remaining=$((deadline - SECONDS))
        attempt_timeout="$GIT_TIMEOUT_SECONDS"
        (( attempt_timeout <= remaining )) || attempt_timeout="$remaining"
        if timeout "${attempt_timeout}s" \
            git -C "$REPO_ROOT" fetch --quiet origin main; then
            return 0
        fi
        (( SECONDS < deadline )) || return 1
        sleep 1
    done
    return 1
}

main() {
    local current_hash remote_hash reconcile_rc=0 updated=0

    require_positive_integer SPF_UPDATE_REMOTE_WAIT_SECONDS "$REMOTE_WAIT_SECONDS"
    require_positive_integer SPF_UPDATE_GIT_TIMEOUT_SECONDS "$GIT_TIMEOUT_SECONDS"
    require_nonnegative_integer SPF_UPDATE_REBOOT_DELAY_SECONDS "$REBOOT_DELAY_SECONDS"
    git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 ||
        die "Not an SPF Git checkout: ${REPO_ROOT}"
    [[ -x "$BOOT_UNIT_RECONCILER" ]] ||
        die "Boot-unit reconciler is unavailable: ${BOOT_UNIT_RECONCILER}"

    current_hash="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
    git -C "$REPO_ROOT" diff --quiet -- ||
        die "Tracked working-tree changes prevent a safe boot update."
    git -C "$REPO_ROOT" diff --cached --quiet -- ||
        die "Staged changes prevent a safe boot update."
    if is_true "$SKIP_SELF_UPDATE"; then
        printf 'Repository update explicitly disabled; using %s.\n' "$current_hash"
    elif ! fetch_main_bounded; then
        printf '%s\n' \
            "Git remote unavailable after ${REMOTE_WAIT_SECONDS}s;" \
            "continuing with checked-out code ${current_hash}."
    else
        remote_hash="$(git -C "$REPO_ROOT" rev-parse --verify refs/remotes/origin/main)"
        if [[ "$current_hash" == "$remote_hash" ]]; then
            printf 'Repository is unchanged at %s.\n' "$current_hash"
        else
            git -C "$REPO_ROOT" merge-base --is-ancestor \
                "$current_hash" "$remote_hash" ||
                die "Local HEAD does not fast-forward to origin/main; refusing mutation."
            printf 'Fast-forwarding SPF from %s to %s before radio startup.\n' \
                "$current_hash" "$remote_hash"
            git -C "$REPO_ROOT" merge --ff-only "$remote_hash"
            [[ "$(git -C "$REPO_ROOT" rev-parse --verify HEAD)" == "$remote_hash" ]] ||
                die "Repository HEAD does not match the fetched main commit."
            updated=1

            [[ -f "$INSTALL_DEPS" ]] ||
                die "Dependency installer is unavailable: ${INSTALL_DEPS}"
            [[ -x "$PYTHON" ]] || die "Python is unavailable: ${PYTHON}"
            printf 'Repository updated; refreshing dependencies and editable install.\n'
            bash "$INSTALL_DEPS"
            # Constrain the resolve to the fleet reference versions. Without
            # this, each rover installs whatever PyPI serves that day: on
            # 2026-08-04 the fleet ran three zarr versions and Rover 4 could
            # not capture at all, because a fresh resolve paired numcodecs
            # 0.16.5 with zarr 2.x and every capture died at import.
            #
            # Falls back to an unconstrained install rather than dying. A
            # rover that boots with drifted versions is recoverable and
            # visible in `rover doctor`; a rover whose boot chain died over a
            # dependency constraint is neither.
            # Bounded. The constraints are chosen so nothing should need
            # building, but a boot must never be able to sit in a compiler:
            # spf-pluto-direct-usb.service Requires= this unit, so a stuck pip
            # blocks the radios and every capture behind them.
            if [[ -f "$CONSTRAINTS" ]]; then
                if timeout "${SPF_PIP_TIMEOUT_SECONDS:-300}" \
                    "$PYTHON" -m pip install -e "$REPO_ROOT" -c "$CONSTRAINTS"; then
                    printf 'Dependencies pinned to the fleet reference.\n'
                else
                    printf '%s\n' \
                        "WARNING: constrained install failed or timed out;" \
                        "retrying without constraints. Versions may drift from" \
                        "the fleet -- check with: rover doctor" >&2
                    "$PYTHON" -m pip install -e "$REPO_ROOT"
                fi
            else
                "$PYTHON" -m pip install -e "$REPO_ROOT"
            fi
        fi
    fi

    reconcile_boot_units || reconcile_rc=$?
    if [[ "$updated" -eq 1 || "$reconcile_rc" -eq 10 ]]; then
        request_reboot
        return 0
    fi
    printf 'PASS: repository and root-managed boot units are ready.\n'
}

main "$@"
