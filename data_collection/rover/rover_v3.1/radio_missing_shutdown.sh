#!/usr/bin/env bash
#
# Audible, fail-closed response when fewer Plutos are attached than configured.
# Radios are powered outside the Pi, so retrying the collector cannot recover
# this condition. The operator must restore the radio and power-cycle the rover.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly MAVLINK_CONTROLLER="${SCRIPT_DIR}/../../../spf/mavlink/mavlink_controller.py"
readonly TONE_REPEATS=3

PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
ACTION="${SPF_RADIO_MISSING_ACTION:-poweroff}"
TONE_GAP_SECONDS="${SPF_RADIO_MISSING_TONE_GAP_SECONDS:-1}"
TONE_TIMEOUT_SECONDS="${SPF_RADIO_MISSING_TONE_TIMEOUT_SECONDS:-10}"

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
printf '%s\n' \
    'Recovery requires restoring the externally powered radio and power-cycling the rover.' \
    >&2

for attempt in $(seq 1 "$TONE_REPEATS"); do
    printf 'Playing radio-missing alarm %s/%s.\n' "$attempt" "$TONE_REPEATS"
    if ! timeout --foreground "$TONE_TIMEOUT_SECONDS" \
        "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer radio-missing; then
        printf 'WARNING: radio-missing alarm %s/%s could not be played.\n' \
            "$attempt" "$TONE_REPEATS" >&2
    fi
    if (( attempt < TONE_REPEATS )); then
        sleep "$TONE_GAP_SECONDS"
    fi
done

if [[ "$ACTION" == "log-only" ]]; then
    printf 'TEST MODE: system poweroff inhibited.\n'
    exit 0
fi

printf 'The three radio-missing alarms completed; requesting clean system poweroff.\n' \
    >&2
sync
systemctl --no-block poweroff
