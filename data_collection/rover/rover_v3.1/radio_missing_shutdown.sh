#!/usr/bin/env bash
#
# Audible, fail-closed response when fewer Plutos are attached than configured.
# Radios are powered outside the Pi, so retrying the collector cannot recover
# this condition. Give an SSH operator a bounded grace period to cancel before
# requesting a clean shutdown.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly MAVLINK_CONTROLLER="${SCRIPT_DIR}/../../../spf/mavlink/mavlink_controller.py"

PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
ACTION="${SPF_RADIO_MISSING_ACTION:-poweroff}"
GRACE_SECONDS="${SPF_RADIO_MISSING_GRACE_SECONDS:-45}"
TONE_GAP_SECONDS="${SPF_RADIO_MISSING_TONE_GAP_SECONDS:-1}"
TONE_TIMEOUT_SECONDS="${SPF_RADIO_MISSING_TONE_TIMEOUT_SECONDS:-10}"
CANCEL_FILE="${SPF_RADIO_MISSING_CANCEL_FILE:-/run/spf/cancel_radio_missing_shutdown}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

expected_radios="${1:-}"
found_radios="${2:-}"
[[ "$expected_radios" =~ ^[1-9][0-9]*$ ]] ||
    die "expected radio count must be a positive integer"
[[ "$found_radios" =~ ^[0-9]+$ ]] ||
    die "found radio count must be a non-negative integer"
(( found_radios < expected_radios )) ||
    die "radio-missing handler requires found < expected"
[[ "$GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_RADIO_MISSING_GRACE_SECONDS must be a positive integer"
[[ "$TONE_GAP_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "SPF_RADIO_MISSING_TONE_GAP_SECONDS must be non-negative"
[[ "$TONE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_RADIO_MISSING_TONE_TIMEOUT_SECONDS must be a positive integer"
case "$ACTION" in
    poweroff|log-only) ;;
    *) die "SPF_RADIO_MISSING_ACTION must be poweroff or log-only" ;;
esac

printf 'CRITICAL: found %s of %s configured Pluto radios.\n' \
    "$found_radios" "$expected_radios" >&2
printf 'The radio-missing alarm will repeat for %s seconds before poweroff.\n' \
    "$GRACE_SECONDS" >&2
printf 'To cancel from SSH: sudo touch %s\n' "$CANCEL_FILE" >&2

mkdir -p -- "$(dirname -- "$CANCEL_FILE")"
rm -f -- "$CANCEL_FILE"
trap 'rm -f -- "$CANCEL_FILE"' EXIT

deadline=$((SECONDS + GRACE_SECONDS))
attempt=0
while (( SECONDS < deadline )); do
    if [[ -e "$CANCEL_FILE" ]]; then
        printf '%s\n' \
            'Operator cancelled missing-radio poweroff; radio preparation remains failed.'
        exit 0
    fi

    attempt=$((attempt + 1))
    elapsed=$((SECONDS + GRACE_SECONDS - deadline))
    remaining=$((deadline - SECONDS))
    tone_timeout="$TONE_TIMEOUT_SECONDS"
    (( tone_timeout <= remaining )) || tone_timeout="$remaining"
    printf 'Playing radio-missing alarm %s at %ss/%ss.\n' \
        "$attempt" "$elapsed" "$GRACE_SECONDS"
    if ! timeout --foreground "$tone_timeout" \
        "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer radio-missing; then
        printf 'WARNING: radio-missing alarm %s could not be played.\n' \
            "$attempt" >&2
    fi

    if [[ -e "$CANCEL_FILE" ]]; then
        printf '%s\n' \
            'Operator cancelled missing-radio poweroff; radio preparation remains failed.'
        exit 0
    fi
    (( SECONDS < deadline )) && sleep "$TONE_GAP_SECONDS"
done

if [[ "$ACTION" == "log-only" ]]; then
    printf 'TEST MODE: system poweroff inhibited.\n'
    exit 0
fi

if [[ -e "$CANCEL_FILE" ]]; then
    printf '%s\n' \
        'Operator cancelled missing-radio poweroff; radio preparation remains failed.'
    exit 0
fi

printf 'The %s-second missing-radio grace period completed; requesting clean system poweroff.\n' \
    "$GRACE_SECONDS" \
    >&2
sync
systemctl --no-block poweroff
