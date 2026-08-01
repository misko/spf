#!/usr/bin/env bash
#
# Build a mass-storage-flashable pluto.frm from a PlutoSDR firmware .dfu.
#
# WHY THIS WORKS (verified against plutosdr-fw Makefile + the on-device
# /sbin/update.sh flasher): plutosdr-fw builds BOTH pluto.frm and pluto.dfu from
# the same FIT image (pluto.itb = kernel + FPGA bitstream + rootfs):
#     pluto.dfu = pluto.itb + 16-byte DFU suffix   (dfu-suffix -a)
#     pluto.frm = pluto.itb + md5(pluto.itb) trailer ("cat itb md5 > frm")
# So we recover the .itb by stripping the DFU suffix, then append the md5 trailer
# exactly as the Makefile does. No Vivado/toolchain rebuild is needed.
#
# SAFETY: writing pluto.frm to the Pluto USB mass storage triggers
# handle_frimware_frm() in /sbin/update.sh, which `dd`s ONLY /dev/mtdblock3
# (the "qspi-linux" firmware partition). It never touches /dev/mtdblock0
# ("qspi-fsbl-uboot", the FSBL+U-Boot bootloader). The historical PlutoPlus
# bricks came from flashing a full *-fw-*.zip or boot.frm, which rewrites
# mtdblock0/1 (handle_boot_frm). Flashing pluto.frm ALONE keeps the known-good
# v0.37 bootloader that already loads this exact FIT every boot in RAM mode.
#
# Usage:
#   make_pluto_frm.sh <input.dfu> [output.frm]
# Default output: alongside the input, named pluto.frm

set -euo pipefail

FRM_MAGIC="ITB PlutoSDR (ADALM-PLUTO)"   # from the Pluto /etc/device_config
FIT_MAGIC_HEX="d00dfeed"                 # devicetree/FIT magic (big-endian)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -ge 1 ]] || die "usage: $(basename "$0") <input.dfu> [output.frm]"
in_dfu="$1"
out_frm="${2:-$(dirname -- "$in_dfu")/pluto.frm}"
[[ -f "$in_dfu" ]] || die "input not found: $in_dfu"
command -v dfu-suffix >/dev/null 2>&1 || die "dfu-suffix (dfu-util) is required"
command -v md5sum >/dev/null 2>&1 || die "md5sum is required"

# 1) sanity: input must start with the FIT magic and carry a valid DFU suffix
head_hex="$(od -An -tx1 -N4 "$in_dfu" | tr -d ' \n')"
[[ "$head_hex" == "$FIT_MAGIC_HEX" ]] ||
    die "input does not start with FIT magic ${FIT_MAGIC_HEX} (got ${head_hex}); not a pluto .dfu"
dfu-suffix -c "$in_dfu" >/dev/null 2>&1 ||
    die "input has no valid DFU suffix; is this really a .dfu?"

work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
itb="${work}/pluto.itb"

# 2) recover the raw FIT (.itb) by removing the DFU suffix
cp -- "$in_dfu" "$itb"
dfu-suffix -D "$itb" >/dev/null

# 3) append the md5 trailer exactly as plutosdr-fw's Makefile does
md5="$(md5sum "$itb" | cut -d ' ' -f 1)"
printf '%s\n' "$md5" >"${work}/md5"
cat "$itb" "${work}/md5" >"${work}/pluto.frm"

# 4) verify the result the same way the on-device flasher will:
#    - md5 of body (everything but the last 33 bytes) equals the trailer
#    - the FRM_MAGIC string is present (update.sh greps for it before dd)
body_md5="$(head -c -33 "${work}/pluto.frm" | md5sum | cut -d ' ' -f 1)"
trailer="$(tail -c 33 "${work}/pluto.frm" | tr -d '\n')"
[[ "$body_md5" == "$trailer" ]] || die "internal: md5 trailer mismatch"
grep -aq "$FRM_MAGIC" "${work}/pluto.frm" ||
    die "FRM_MAGIC '${FRM_MAGIC}' absent; the Pluto flasher would reject this image"

mv -- "${work}/pluto.frm" "$out_frm"
printf 'OK: wrote %s (%s bytes)\n' "$out_frm" "$(stat -c %s "$out_frm")"
printf '  FIT (.itb) md5   : %s\n' "$md5"
printf '  FRM_MAGIC present: yes\n'
printf '  Flash target     : /dev/mtdblock3 (qspi-linux firmware) — bootloader untouched\n'
