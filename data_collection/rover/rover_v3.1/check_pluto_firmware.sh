#!/usr/bin/env bash
#
# Report the running firmware (device-fw) of every attached Pluto, per radio.
#
# Both Plutos share the USB-gadget IP 192.168.2.1, so this does NOT use ssh/IP.
# Each radio is identified by its USB serial and read from its own mass-storage
# updater volume (img/version.js, regenerated with the running device-fw on every
# Pluto boot). Read-only and non-disruptive. Run with sudo (needs to mount).
#
# Usage:  sudo check_pluto_firmware.sh

set -euo pipefail

MNT="$(mktemp -d)"
cleanup() { umount "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true; }
trap cleanup EXIT

# Serials of attached runtime Plutos (USB 0456:b673).
pluto_serials=" $(
    for dev in /sys/bus/usb/devices/*/; do
        [[ "$(cat "${dev}idVendor" 2>/dev/null)" == "0456" &&
           "$(cat "${dev}idProduct" 2>/dev/null)" == "b673" ]] || continue
        cat "${dev}serial" 2>/dev/null || true
    done | tr '\n' ' '
) "

found=0
for d in /dev/sd?; do
    [[ -b "${d}1" ]] || continue
    ser="$(udevadm info --query=property --name="$d" 2>/dev/null |
        sed -n 's/^ID_SERIAL_SHORT=//p' | head -1)"
    [[ -n "$ser" && "$pluto_serials" == *" $ser "* ]] || continue
    found=$((found + 1))
    if mount -o ro "${d}1" "$MNT" 2>/dev/null; then
        fw="$(grep -oE 'VerLocal = "[^"]+"' "$MNT/img/version.js" 2>/dev/null |
            grep -oE 'v[0-9][^"]*' | head -1)"
        umount "$MNT" 2>/dev/null || true
        printf '%-9s serial=%s  firmware=%s\n' "$d" "$ser" "${fw:-unknown}"
    else
        printf '%-9s serial=%s  firmware=UNREADABLE (mass-storage not ready)\n' \
            "$d" "$ser"
    fi
done

[[ "$found" -gt 0 ]] || { echo "No attached Pluto mass-storage volumes found."; exit 1; }
