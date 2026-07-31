#!/usr/bin/env bash
#
# Version-conditional persistent QSPI firmware for the direct-USB gain/RSSI build.
#
# This replaces the per-boot RAM load. For each attached Pluto it reads the
# firmware the radio is *currently running* (which, with RAM loading disabled,
# is exactly what is in QSPI) and:
#   - if it already matches the expected build  -> SKIP (the fast steady state)
#   - otherwise                                 -> flash pluto.frm to QSPI and
#     wait for the radio to reboot into the expected build.
#
# The flash uses the on-device mass-storage updater (copy pluto.frm, eject),
# which writes ONLY the firmware partition (/dev/mtdblock3). It never writes the
# FSBL/U-Boot bootloader (mtdblock0) -- see PLUTO_QSPI_FLASH.md. pluto.frm is
# derived from the published .dfu by make_pluto_frm.sh.
#
# Env:
#   SPF_PLUTO_EXPECTED_DEVICE_FW  required: expected /opt/VERSIONS device-fw string
#   SPF_FIRMWARE_DFU              required: path to the published .dfu
#   SPF_PLUTO_FRM                 optional: prebuilt/cached pluto.frm path
#   SPF_PLUTO_FLASH_TIMEOUT       optional: per-radio reboot wait, default 180s
# Arg 1: expected Pluto count.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
EXPECTED_FW="${SPF_PLUTO_EXPECTED_DEVICE_FW:?SPF_PLUTO_EXPECTED_DEVICE_FW is required}"
DFU="${SPF_FIRMWARE_DFU:?SPF_FIRMWARE_DFU is required}"
FRM="${SPF_PLUTO_FRM:-/home/pi/.cache/spf/firmware/pluto.frm}"
FLASH_TIMEOUT="${SPF_PLUTO_FLASH_TIMEOUT:-180}"
# How long to wait for a radio's mass-storage FAT to become mountable. The Pluto
# exposes its updater volume a second or two AFTER the USB device enumerates, so
# a naive mount right at boot fails even on a perfectly healthy radio.
MSD_READY_TIMEOUT="${SPF_PLUTO_MSD_TIMEOUT:-45}"
EXPECTED_COUNT="${1:?expected Pluto count required}"
MNT="/run/spf-pluto-msd"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]] || die "bad expected count: ${EXPECTED_COUNT}"
for cmd in udevadm mount umount eject sync; do
    command -v "$cmd" >/dev/null 2>&1 || die "missing required command: ${cmd}"
done

serial_of() {  # $1=/dev/sdX -> ID_SERIAL_SHORT
    udevadm info --query=property --name="$1" 2>/dev/null |
        sed -n 's/^ID_SERIAL_SHORT=//p' | head -1
}

pluto_usb_serials() {  # serials of attached runtime Plutos (USB 0456:b673)
    local dev v p
    for dev in /sys/bus/usb/devices/*/; do
        v="$(cat "${dev}idVendor" 2>/dev/null || true)"
        p="$(cat "${dev}idProduct" 2>/dev/null || true)"
        [[ "$v" == "0456" && "$p" == "b673" ]] || continue
        cat "${dev}serial" 2>/dev/null || true
    done
}

blkdev_for_serial() {  # $1=serial -> /dev/sdX (or fail)
    local d
    for d in /dev/sd?; do
        [[ -b "$d" ]] || continue
        [[ "$(serial_of "$d")" == "$1" ]] && { printf '%s' "$d"; return 0; }
    done
    return 1
}

# Mount a radio's mass-storage FAT at $MNT, retrying until the volume is ready
# (present + info.html readable) or MSD_READY_TIMEOUT elapses. $2 = "ro"|"rw".
mount_msd() {  # $1=/dev/sdX ; $2 mode ; 0 mounted, 1 never became ready
    local d="$1" mode="${2:-ro}" deadline=$((SECONDS + MSD_READY_TIMEOUT))
    mkdir -p "$MNT"
    while (( SECONDS < deadline )); do
        if [[ -b "${d}1" ]] && mount -o "$mode" "${d}1" "$MNT" 2>/dev/null; then
            if [[ -f "$MNT/info.html" ]]; then
                return 0
            fi
            umount "$MNT" 2>/dev/null || true
        fi
        sleep 1
    done
    return 1
}

# A radio is "on expected" iff its mass-storage info page -- regenerated on every
# Pluto boot with the running device-fw -- CONTAINS the exact expected device-fw
# string. A substring presence test is more robust than parsing: info.html also
# lists u-boot, IIO and template versions, so picking "the version" by position
# is unreliable (that read the template "v0.15" instead of the build).
# Returns: 0 = on expected, 1 = readable but wrong firmware, 2 = not readable
# (radio not enumerating stably) -- the caller must NOT treat 2 as "flash it".
radio_on_expected() {  # $1=/dev/sdX
    local d="$1" rc=1
    mount_msd "$d" ro || return 2
    grep -qF "$EXPECTED_FW" "$MNT/info.html" 2>/dev/null && rc=0
    umount "$MNT" 2>/dev/null || true
    return "$rc"
}

build_frm() {
    if [[ -f "$FRM" && "$FRM" -nt "$DFU" ]]; then
        return 0
    fi
    mkdir -p "$(dirname -- "$FRM")"
    bash "${SCRIPT_DIR}/make_pluto_frm.sh" "$DFU" "$FRM" >&2
}

flash_radio() {  # $1=serial $2=/dev/sdX ; writes pluto.frm to mtd3, waits for reboot
    local ser="$1" dev="$2" deadline back
    printf '%s: flashing pluto.frm to QSPI (mtd3) via %s\n' "$ser" "$dev" >&2
    mount_msd "$dev" rw || die "${ser}: mass-storage did not become mountable to flash"
    cp -- "$FRM" "$MNT/pluto.frm"
    sync
    umount "$MNT" 2>/dev/null || true
    eject "$dev"

    # The updater resets the radio: wait for it to drop off, then re-enumerate.
    deadline=$((SECONDS + FLASH_TIMEOUT))
    while (( SECONDS < deadline )); do
        blkdev_for_serial "$ser" >/dev/null 2>&1 || break
        sleep 1
    done
    while (( SECONDS < deadline )); do
        if back="$(blkdev_for_serial "$ser" 2>/dev/null)" && [[ -b "${back}1" ]]; then
            sleep 2   # let the FAT settle
            printf '%s: re-enumerated at %s after flash\n' "$ser" "$back" >&2
            return 0
        fi
        sleep 2
    done
    die "${ser}: did not re-enumerate within ${FLASH_TIMEOUT}s after flash"
}

main() {
    build_frm

    # Snapshot the attached Pluto serials up front (device nodes shuffle on reset).
    # A block device is a Pluto MSD iff its ID_SERIAL_SHORT matches an attached
    # runtime Pluto (0456:b673) -- the MSD's ID_MODEL is the generic
    # "File-Stor_Gadget", so identify by USB VID/serial instead.
    local pluto_set d ser
    pluto_set=" $(pluto_usb_serials | tr '\n' ' ') "
    local serials=()
    for d in /dev/sd?; do
        [[ -b "$d" ]] || continue
        ser="$(serial_of "$d")"
        [[ -n "$ser" && "$pluto_set" == *" $ser "* ]] || continue
        serials+=("$ser")
    done
    [[ "${#serials[@]}" -eq "$EXPECTED_COUNT" ]] ||
        die "expected ${EXPECTED_COUNT} Pluto mass-storage disks, found ${#serials[@]}"

    local ser d state
    for ser in "${serials[@]}"; do
        d="$(blkdev_for_serial "$ser")" || die "${ser}: block device vanished"
        state=0
        radio_on_expected "$d" || state=$?
        case "$state" in
            0)
                printf '%s: already on expected firmware (%s); skip\n' \
                    "$ser" "$EXPECTED_FW"
                continue
                ;;
            2)
                die "${ser}: mass-storage not readable within ${MSD_READY_TIMEOUT}s;" \
                    "radio is not enumerating stably (hardware) -- refusing to flash blind"
                ;;
        esac
        # state 1: readable but running a different firmware -> flash it.
        printf '%s: not on expected firmware "%s" -> flashing\n' "$ser" "$EXPECTED_FW"
        flash_radio "$ser" "$d"
        d="$(blkdev_for_serial "$ser")" || die "${ser}: gone after flash"
        state=0
        radio_on_expected "$d" || state=$?
        [[ "$state" -eq 0 ]] ||
            die "${ser}: after flash still not on expected firmware" \
                "'${EXPECTED_FW}' (state ${state})"
        printf '%s: now booting expected firmware from QSPI (%s)\n' "$ser" "$EXPECTED_FW"
    done
    printf 'PASS: %s Pluto(s) on expected QSPI firmware %s\n' "$EXPECTED_COUNT" "$EXPECTED_FW"
}

main
