#!/usr/bin/env bash
#
# Provision an SPF rover. Replaces setup.sh, which is deprecated and unsafe on
# current Raspberry Pi OS images (see the comment block at the top of setup.sh).
#
# Runs ON the rover, from its own checkout. Every stage is idempotent, so a
# re-run after a failure or a reboot is always safe.
#
#   sudo ./provision_rover.sh <ROVER_ID> --stage <stage>
#
# STAGES, in order. The split exists because two of them end in a reboot and one
# is deliberately gated on a human check.
#
#   identity   write ~/rover_id, verify the checkout resolves a capture plan
#   network    hostname + static eth0 + DNS.  WIFI IS LEFT UP.
#              --> then PROVE the static address from another machine <--
#   wifi-off   disable wifi (refuses unless the static address answers), reboot
#   base       apt deps, venv, editable install, ~/.bashrc, udev, device_mapping
#   firmware   ArduPilot flash, Pluto provisioning, direct-USB production boot
#              --> reboot to let the boot chain run <--
#   audit      print this rover's fingerprint for comparison against another
#
#   all        identity + network + base  (stops BEFORE wifi-off and firmware,
#              because those need a human gate and a reboot respectively)
#
# BOOTSTRAP. This script lives in the repo, so the repo must exist first. On a
# fresh image that is two commands, and they cannot be scripted from here:
#
#   sudo apt-get update && sudo apt-get install -y git
#   cd /home/pi && git clone https://github.com/misko/spf.git
#
# The static address is 192.168.1.(40 + rover_id), matching the historical
# setup.sh formula. Reserve it in the router's DHCP pool first — the rover
# addresses sit inside the pool and IP squatting has been observed
# (ROVER_RUNBOOK.md section 2).
#
set -uo pipefail

REPO_ROOT="${SPF_REPO_ROOT:-/home/pi/spf}"
ROVER_DIR="${REPO_ROOT}/data_collection/rover/rover_v3.1"
VENV="${SPF_VENV:-/home/pi/spf-virtualenv}"
PI_HOME="${SPF_PI_HOME:-/home/pi}"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }
head_() { printf '\n== %s ==\n' "$*"; }
need_root() { [[ $EUID -eq 0 ]] || die "stage '$1' must run as root (use sudo)"; }

usage() {
    sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 1
case "$1" in -h|--help|help) usage 0 ;; esac

rover_id="$1"; shift
[[ "$rover_id" =~ ^[1-4]$ ]] ||
    die "Unsupported rover_id: ${rover_id} (add it to CANONICAL_CONFIGS and the
       ^[1-4]\$ guards in drone_run.sh, prepare_direct_usb_boot.sh and
       configure_direct_usb_boot.sh before provisioning a new rover)"

stage=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage="${2:-}"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ -n "$stage" ]] || die "--stage is required (see --help)"

address="192.168.1.$((40 + rover_id))"
[[ -d "$ROVER_DIR" ]] || die "checkout not found at ${REPO_ROOT} — see BOOTSTRAP in --help"

# --------------------------------------------------------------------------
stage_identity() {
    head_ "identity"
    local current=""
    [[ -f "${PI_HOME}/rover_id" ]] && current="$(tr -d '[:space:]' <"${PI_HOME}/rover_id")"
    if [[ "$current" == "$rover_id" ]]; then
        note "~/rover_id already ${rover_id}"
    else
        printf '%s\n' "$rover_id" >"${PI_HOME}/rover_id"
        chown "$(stat -c '%U:%G' "$PI_HOME")" "${PI_HOME}/rover_id" 2>/dev/null || true
        note "~/rover_id ${current:-<unset>} -> ${rover_id}"
    fi

    # Fail here rather than deep inside the firmware stage if the repo does not
    # know about this rover yet.
    if [[ -x "${VENV}/bin/python3" ]]; then
        local plan
        plan="$(cd "$REPO_ROOT" && "${VENV}/bin/python3" -c "
from spf.scripts.rover_capture_config import resolve_capture_plan as r
p = r(${rover_id})
print(p.expected_radios, p.data_version, p.firmware_boot_mode)
" 2>&1)" || die "capture plan does not resolve for rover ${rover_id}: ${plan}"
        note "capture plan: radios/version/boot = ${plan}"
    else
        note "venv absent — capture plan will be verified in the 'base' stage"
    fi
}

# --------------------------------------------------------------------------
stage_network() {
    head_ "network (wifi stays UP)"
    bash "${ROVER_DIR}/configure_rover_network.sh" "$rover_id" --stage static-only ||
        die "network configuration failed"
    cat <<EOF

  NEXT — do this before 'wifi-off', from ANOTHER machine:

    ssh pi@${address} 'hostname; ip -brief addr show eth0'

  wifi is still up, so a failure here is recoverable. The wifi-off stage
  refuses to run until ${address} answers.
EOF
}

stage_wifi_off() {
    head_ "wifi-off"
    bash "${ROVER_DIR}/configure_rover_network.sh" "$rover_id" --stage disable-wifi ||
        die "refused or failed — is ${address} reachable?"
    note "reboot now, then confirm ${address} returns and wlan0 is gone:"
    note "  sudo reboot"
}

# --------------------------------------------------------------------------
stage_base() {
    head_ "base"

    note "apt dependencies"
    bash "${ROVER_DIR}/install_deps.sh" >/dev/null 2>&1 || die "install_deps.sh failed"

    if [[ -x "${VENV}/bin/python3" ]]; then
        note "venv present at ${VENV}"
    else
        note "creating venv at ${VENV}"
        sudo -u "$(stat -c '%U' "$PI_HOME")" python3 -m venv "$VENV" || die "venv creation failed"
    fi
    note "editable install (idempotent, slow on a Pi)"
    sudo -u "$(stat -c '%U' "$PI_HOME")" "${VENV}/bin/pip" -q install -e "$REPO_ROOT" ||
        die "pip install -e failed"
    sudo -u "$(stat -c '%U' "$PI_HOME")" "${VENV}/bin/pip" -q install RPi.GPIO ||
        die "pip install RPi.GPIO failed"

    # Rebuilt from scratch each run so re-provisioning cannot duplicate lines.
    note "~/.bashrc SPF lines"
    local rc="${PI_HOME}/.bashrc"
    grep -v spf "$rc" | grep -v lsusb >"${rc}.spf.tmp" && mv "${rc}.spf.tmp" "$rc"
    {
        echo 'export PYTHONPATH=/home/pi/spf'
        echo 'test -z "$VIRTUAL_ENV" && source ~/spf-virtualenv/bin/activate'
        # Pi-4 mass-storage form, matching rover 1. The Pi-5 'lsusb | grep PLUTO'
        # variant in the old setup.sh is deliberately not used.
        printf '%s\n' "lsusb -t | grep usb-storage | sed 's/.*Port \\([0-9]*\\): Dev \\([0-9]*\\),.*/\\1 \\2/g' > ~/device_mapping"
    } >>"$rc"
    chown "$(stat -c '%U:%G' "$PI_HOME")" "$rc" 2>/dev/null || true

    note "udev usb_device rule"
    local rules=/etc/udev/rules.d/99-com.rules
    if [[ -f "$rules" ]]; then
        grep -v usb_device "$rules" >"${rules}.tmp" && mv "${rules}.tmp" "$rules"
    fi
    printf 'SUBSYSTEM=="usb_device", MODE="0664", GROUP="usb"\n' >>"$rules"

    # .bashrc only regenerates this on an INTERACTIVE login, so a provisioning
    # run over ssh would otherwise leave it missing.
    note "device_mapping"
    bash "${ROVER_DIR}/device_mapping.sh" >"${PI_HOME}/device_mapping"
    chown "$(stat -c '%U:%G' "$PI_HOME")" "${PI_HOME}/device_mapping" 2>/dev/null || true
    local rows; rows="$(wc -l <"${PI_HOME}/device_mapping")"
    note "device_mapping rows: ${rows}"
    [[ "$rows" -gt 0 ]] || die "device_mapping is empty — are the Plutos attached and enumerated?"

    stage_identity
}

# --------------------------------------------------------------------------
stage_firmware() {
    head_ "firmware"

    note "ArduPilot (Rover 4.5.0 fmuv3)"
    (cd "$PI_HOME" && bash "${ROVER_DIR}/flash_ardupilot.sh") || die "ArduPilot flash failed"

    note "Pluto persistent provisioning (ad9361/2r2t)"
    # Refuses on a radio already running the direct-USB image, or on a QSPI
    # version outside SPF_APPROVED_QSPI_DEVICE_FW. Both are interlocks, not
    # errors: provisioning belongs on stock firmware, before direct-USB. A
    # refusal on an already-provisioned radio is expected and non-fatal here.
    bash "${ROVER_DIR}/check_and_set_pluto.sh" --apply ||
        note "WARNING: provisioning refused for at least one radio (see above)."

    note "direct-USB production boot"
    bash "${ROVER_DIR}/configure_direct_usb_boot.sh" production-default ||
        die "configure_direct_usb_boot.sh failed"

    cat <<'EOF'

  NEXT: reboot to let the boot chain run. First boot is slow — it flashes any
  radio whose QSPI firmware differs from the pinned image.

    sudo reboot

  Then confirm: 2 Plutos, ready manifest v2, and all four units in the same
  state as the reference rover:

    ./provision_rover.sh <ID> --stage audit
EOF
}

# --------------------------------------------------------------------------
stage_audit() {
    head_ "audit"
    bash "${ROVER_DIR}/audit_rover.sh"
    cat <<'EOF'

  Compare against a known-good rover:
    ./compare_rovers.sh 192.168.1.41 192.168.1.4N
EOF
}

# --------------------------------------------------------------------------
case "$stage" in
    identity) need_root identity; stage_identity ;;
    network)  need_root network;  stage_network ;;
    wifi-off) need_root wifi-off; stage_wifi_off ;;
    base)     need_root base;     stage_base ;;
    firmware) need_root firmware; stage_firmware ;;
    audit)    stage_audit ;;
    all)
        need_root all
        stage_identity
        stage_network
        stage_base
        cat <<EOF

  'all' stops here on purpose. Remaining, in order:
    1. prove ${address} from another machine
    2. sudo ./provision_rover.sh ${rover_id} --stage wifi-off   (then reboot)
    3. sudo ./provision_rover.sh ${rover_id} --stage firmware   (then reboot)
    4. ./provision_rover.sh ${rover_id} --stage audit
EOF
        ;;
    *) die "unknown stage: ${stage} (see --help)" ;;
esac
