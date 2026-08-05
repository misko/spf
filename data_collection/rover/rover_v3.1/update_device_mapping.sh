#!/usr/bin/env bash
#
# Regenerate ~/device_mapping from what is on the USB bus right now.
#
# The collector turns this file into the URIs it opens:
#
#     "<port> <dev>"  ->  pluto://usb:1.<dev>.5     (mavlink_radio_collection.py)
#
# Device numbers are assigned at enumeration and change across reboots and
# re-enumerations, so a stale file makes the collector open a URI that no longer
# exists. The symptom is `Exception: No device found` -- which names neither the
# mapping nor the port, and looks identical to a radio that has genuinely died.
#
# It is normally written by a line in ~/.bashrc, which means it is only ever
# refreshed when a human logs in interactively. Anything non-interactive -- a
# script, an ssh command, a service -- uses whatever the last login left behind.
# That is what this command exists to fix without requiring a login.
#
# Usage:
#   update_device_mapping.sh            # show the diff, then write
#   update_device_mapping.sh --check    # report drift, write nothing (exit 1 if stale)

set -euo pipefail

MAPPING="${SPF_DEVICE_MAPPING:-/home/pi/device_mapping}"
SERVICE_NAME="mavlink_controller.service"
CHECK_ONLY=0

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

# Exactly the derivation ~/.bashrc uses, so this cannot disagree with it.
current="$(lsusb -t | grep usb-storage |
    sed 's/.*Port \([0-9]*\): Dev \([0-9]*\),.*/\1 \2/g' || true)"

if [[ -z "$current" ]]; then
    die "No radios present a usb-storage interface.
  A Pluto that has half-dropped still answers 'lsusb' while no longer
  appearing here, so this is not necessarily an empty bus.
  Check:  lsusb ; dmesg -T | grep -i 'usb disconnect'
  Refusing to write an empty mapping over a good one."
fi

existing=""
[[ -r "$MAPPING" ]] && existing="$(cat "$MAPPING")"

if [[ "$existing" == "$current" ]]; then
    printf 'PASS: %s already matches the bus.\n' "$MAPPING"
    printf '%s\n' "$current" | sed 's/^/    /'
    exit 0
fi

printf 'device mapping has drifted (%s)\n\n' "$MAPPING"
printf '  stored:\n'; printf '%s\n' "${existing:-    (absent)}" | sed 's/^/    /'
printf '  bus now:\n'; printf '%s\n' "$current" | sed 's/^/    /'
printf '\n'

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    printf 'STALE: run `rover radio remap` to update.\n'
    exit 1
fi

# The collector reads this once at startup. Rewriting it underneath a running
# capture cannot corrupt that capture, but the next one would silently use a
# different mapping than the operator just looked at, so say so.
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    printf 'NOTE: %s is running; it keeps the mapping it started with.\n' \
        "$SERVICE_NAME"
    printf '      Restart it to pick this up.\n\n'
fi

printf '%s\n' "$current" >"${MAPPING}.new.$$"
mv -f -- "${MAPPING}.new.$$" "$MAPPING"
printf 'PASS: wrote %s\n' "$MAPPING"
printf '%s\n' "$current" | sed 's/^/    /'
printf '\nCheck which receiver each one is:  rover radio map\n'
