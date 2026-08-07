#!/usr/bin/env bash
#
# Emit a machine-readable fingerprint of this rover. READ-ONLY: no writes, no
# service changes, no mounts. Safe to run on a provisioned field rover.
#
# Output is stable `key=value` lines, sorted by key, so two rovers can be
# diffed directly. See compare_rovers.sh.
#
# Fields that are EXPECTED to differ between rovers (id, hostname, address,
# rest offset, tag, sysid) are prefixed `identity.`; everything else should
# match across the fleet and is prefixed `fleet.`. compare_rovers.sh uses that
# split to separate intended differences from drift.
#
# Usage:
#   ./audit_rover.sh                 # on the rover
#   ssh pi@192.168.1.44 'bash /home/pi/spf/data_collection/rover/rover_v3.1/audit_rover.sh'
#
set -uo pipefail

REPO_ROOT="${SPF_REPO_ROOT:-/home/pi/spf}"
emit() { printf '%s=%s\n' "$1" "${2:-}"; }
q() { "$@" 2>/dev/null || true; }

# --- identity (expected to differ per rover) --------------------------------
rover_id="$(tr -d '[:space:]' </home/pi/rover_id 2>/dev/null || true)"
emit identity.rover_id "${rover_id:-UNSET}"
emit identity.hostname "$(q hostname)"
emit identity.eth0_addr "$(q ip -brief addr show eth0 | awk '{print $3}')"

if [[ -n "${rover_id:-}" && -r "${REPO_ROOT}/spf/scripts/rover_capture_config.py" ]]; then
    plan="$(q /home/pi/spf-virtualenv/bin/python3 -c "
import sys; sys.path.insert(0,'${REPO_ROOT}')
from spf.scripts.rover_capture_config import resolve_capture_plan
import yaml, pathlib
p = resolve_capture_plan(${rover_id})
c = yaml.safe_load(pathlib.Path(p.config_path).read_text())
print(pathlib.Path(p.config_path).name, p.expected_radios, p.data_version,
      p.firmware_boot_mode, p.firmware_release_tag, c.get('rest-offset-m'), sep='|')
")"
    IFS='|' read -r cfg radios dver bootmode fwtag rest <<<"${plan:-|||||}"
    emit identity.capture_config "${cfg:-UNRESOLVED}"
    emit identity.rest_offset "${rest:-UNKNOWN}"
    emit identity.tag "RO${rover_id}"
    emit fleet.expected_radios "${radios:-UNKNOWN}"
    emit fleet.data_version "${dver:-UNKNOWN}"
    emit fleet.firmware_boot_mode "${bootmode:-UNKNOWN}"
    emit fleet.firmware_release_tag "${fwtag:-UNKNOWN}"
fi

# --- fleet (should match across rovers) -------------------------------------
emit fleet.model "$(q tr -d '\0' </proc/device-tree/model)"
emit fleet.os "$(q bash -c '. /etc/os-release; echo $PRETTY_NAME')"
emit fleet.image_stage "$(q awk '/pi-gen/{print $NF}' /etc/rpi-issue)"
emit fleet.arch "$(q uname -m)/$(q dpkg --print-architecture)"
emit fleet.python "$(q python3 --version | awk '{print $2}')"
emit fleet.venv_python "$(q /home/pi/spf-virtualenv/bin/python3 --version | awk '{print $2}')"
emit fleet.boot_target "$(q systemctl get-default)"

emit fleet.git_branch "$(q git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
emit fleet.git_head "$(q git -C "$REPO_ROOT" rev-parse --short HEAD)"
# A dirty checkout stops update_spf_before_boot.sh from self-updating.
if q git -C "$REPO_ROOT" diff --quiet -- && q git -C "$REPO_ROOT" diff --cached --quiet --; then
    emit fleet.git_clean yes
else
    emit fleet.git_clean NO
fi

emit fleet.wlan0 "$(ip link show wlan0 >/dev/null 2>&1 && echo present || echo absent)"
# Bluetooth is a live radio on every rover and nothing in the checklist mentions
# it (S-8, 2026-08-06). 2.4 GHz is well out of band at 5.8 GHz, so this is
# reported rather than acted on -- but "nobody had looked" is not the same as
# "it is fine", and an audit line is what turns one into the other.
#
# sysfs, not hciconfig: the tool is deprecated and absent from newer images,
# where its absence would read as "no bluetooth radio" -- the reassuring answer,
# which is the one a check must never give by accident. fleet.stock.bluetooth
# above reports the systemd unit; this reports the radio itself, and they
# disagree whenever the unit is disabled but the overlay was never applied.
emit fleet.hci0 "$([[ -e /sys/class/bluetooth/hci0 ]] && echo present || echo absent)"
# NOT `command -v ifup`: /sbin is absent from the pi user's PATH over ssh, so
# that reports "absent" on rover 1 where the package is installed.
if [[ "$(q dpkg-query -W -f='${Status}' ifupdown)" == "install ok installed" ]]; then
    emit fleet.ifupdown present
else
    emit fleet.ifupdown absent
fi
if [[ -L /boot/config.txt ]]; then
    emit fleet.boot_config_layout symlink
elif [[ -f /boot/firmware/config.txt ]]; then
    emit fleet.boot_config_layout split
else
    emit fleet.boot_config_layout legacy
fi

emit fleet.pluto_count "$(q bash -c 'lsusb | grep -c PLUTO')"
# The FMU's USB id changes once ArduPilot is flashed, so this doubles as a
# flash-state signal: 26ac:0011 = stock 3DR/PX4 bootloader (NOT yet flashed),
# 1209:5741 = ArduPilot fmuv3 running (flashed, as on rover 1).
# NOTE: do not wrap these in q() - q() ends in `|| true` and always returns 0,
# which would make the first branch always fire.
_lsusb="$(lsusb 2>/dev/null || true)"
if grep -q "1209:5741" <<<"$_lsusb"; then
    emit fleet.fmu ardupilot-fmuv3
elif grep -qiE "26ac:0011|PX4 FMU|3D Robotics" <<<"$_lsusb"; then
    emit fleet.fmu stock-bootloader-UNFLASHED
else
    emit fleet.fmu absent
fi
emit fleet.device_mapping_lines "$(q bash -c 'wc -l </home/pi/device_mapping')"

if [[ -r /run/spf/direct_usb_ready.json ]]; then
    emit fleet.ready_manifest_version "$(q /home/pi/spf-virtualenv/bin/python3 -c \
        'import json;print(json.load(open("/run/spf/direct_usb_ready.json")).get("ready_manifest_version"))')"
else
    emit fleet.ready_manifest_version absent
fi

for unit in mavlink_controller spf-pluto-direct-usb spf-rover-update spf-direct-usb-preflight; do
    emit "fleet.unit.${unit}" "$(q systemctl is-enabled "${unit}.service")"
done
for unit in ModemManager bluetooth rpi-eeprom-update e2scrub_reap NetworkManager-wait-online; do
    emit "fleet.stock.${unit}" "$(q systemctl is-enabled "${unit}.service")"
done

emit fleet.throttled "$(q vcgencmd get_throttled)"
