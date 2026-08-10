#!/usr/bin/env bash
#
# Download, verify, and RAM-boot the hardware-tested PlutoPlus direct-USB
# gain/RSSI firmware. This script never writes the image to QSPI.

set -euo pipefail

readonly RELEASE_TAG="${SPF_FIRMWARE_RELEASE_TAG:-v0.38-plutoplus-spf-gain-series-v4-rc16}"
readonly ASSET_NAME="${SPF_FIRMWARE_ASSET_NAME:-plutoplus-spf-main-867e18542311-pluto.dfu}"
readonly IMAGE_SHA256="${SPF_FIRMWARE_IMAGE_SHA256:-27aca40915fd75fbcabfadef88fee96ff422c6058f83fab8a57a09b8d1eae911}"
readonly IMAGE_URL="${SPF_FIRMWARE_IMAGE_URL:-https://github.com/misko/plutosdr-fw/releases/download/${RELEASE_TAG}/${ASSET_NAME}}"
readonly PLUTO_RUNTIME_ID="0456:b673"
readonly PLUTO_DFU_ID="0456:b674"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly CACHE_ROOT="${SPF_FIRMWARE_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/spf/firmware}"
readonly STATE_ROOT="${SPF_FIRMWARE_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/spf/pluto-firmware}"
readonly APPROVED_QSPI_DEVICE_FW="${SPF_APPROVED_QSPI_DEVICE_FW:-v0.37-dirty}"
readonly IMAGE_PATH="${CACHE_ROOT}/${ASSET_NAME}"
readonly SSH_HOST="${SPF_PLUTO_SSH_HOST:-192.168.2.1}"
readonly SSH_PASSWORD="${SPF_PLUTO_SSH_PASSWORD:-analog}"
readonly SSH_CONFIG="${SCRIPT_DIR}/ssh_config"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly REPO_ROOT
readonly MULTI_LOADER="${REPO_ROOT}/spf/scripts/pluto_multi_firmware.py"

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND

Commands:
  download   Download the published DFU image and verify its SHA-256.
  status     Show USB enumeration and Pluto firmware/process status.
  load       Back up device state, then load the verified image into RAM.
  verify     Require the direct-USB firmware, interface, and daemons.
  rollback   Reboot into the unchanged firmware stored in QSPI.
  load-all N      RAM-load and verify exactly N attached Plutos by serial/path.
  verify-all N    Verify exactly N RAM-loaded direct-USB Plutos.
  check-config-all N  Verify persistent ad9361/2r2t settings on N Plutos.
  provision-config-all N [--dry-run]
                      Explicitly provision and verify ad9361/2r2t on N Plutos.
  rollback-all N  Reset exactly N Plutos into unchanged QSPI firmware.
  status-all N    Show serial/path/direct-USB status for N expected Plutos.
  discover-count  Print the number of attached runtime Plutos.

Safety:
  load and rollback require exactly one attached Pluto. The image is loaded
  into RAM with dfu-util; this script never flashes QSPI. A power cycle also
  restores the installed QSPI firmware.
  Multi-radio mutation commands require root so each duplicate USB-network
  address can be isolated in a temporary network namespace.

Overrides:
  SPF_PLUTO_SSH_HOST          Pluto USB-network address (default: 192.168.2.1)
  SPF_PLUTO_SSH_PASSWORD      Pluto root password (default: factory 'analog')
  SPF_FIRMWARE_CACHE_DIR      Download cache
  SPF_FIRMWARE_STATE_DIR      Pre-load backup directory
  SPF_APPROVED_QSPI_DEVICE_FW Approved persistent device-fw (default: v0.37-dirty)
EOF
}

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

usb_count() {
    local usb_id="$1"
    { lsusb -d "$usb_id" 2>/dev/null || true; } |
        awk 'END { print NR + 0 }'
}

require_one_usb_device() {
    local usb_id="$1"
    local description="$2"
    local count
    count="$(usb_count "$usb_id")"
    if [[ "$count" -ne 1 ]]; then
        die "Expected exactly one ${description} (${usb_id}); found ${count}. Disconnect every other Pluto and retry."
    fi
}

wait_for_usb_count() {
    local usb_id="$1"
    local expected="$2"
    local timeout_seconds="$3"
    local description="$4"
    local deadline=$((SECONDS + timeout_seconds))
    local count

    while (( SECONDS < deadline )); do
        count="$(usb_count "$usb_id")"
        if [[ "$count" -eq "$expected" ]]; then
            return 0
        fi
        sleep 0.5
    done

    count="$(usb_count "$usb_id")"
    die "Timed out waiting for ${description}: expected ${expected}, found ${count} (${usb_id})."
}

ssh_pluto() {
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh \
        -F "$SSH_CONFIG" \
        -o ConnectTimeout=5 \
        -o LogLevel=ERROR \
        -o ServerAliveInterval=2 \
        -o ServerAliveCountMax=2 \
        "root@${SSH_HOST}" \
        "$@"
}

wait_for_ssh() {
    local timeout_seconds="$1"
    local deadline=$((SECONDS + timeout_seconds))

    while (( SECONDS < deadline )); do
        if ssh_pluto true >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    die "Pluto did not become reachable by SSH at ${SSH_HOST} within ${timeout_seconds} seconds."
}

wait_for_stock_runtime() {
    local timeout_seconds="$1"
    local deadline=$((SECONDS + timeout_seconds))

    while (( SECONDS < deadline )); do
        if [[ "$(usb_count "$PLUTO_RUNTIME_ID")" -eq 1 ]] &&
            ! has_vendor_interface_six &&
            ssh_pluto true >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    die "Pluto did not return as reachable stock firmware within ${timeout_seconds} seconds."
}

verify_image() {
    local image="$1"
    [[ -f "$image" ]] || die "Firmware image is missing: ${image}"
    printf '%s  %s\n' "$IMAGE_SHA256" "$image" | sha256sum --check --status ||
        die "SHA-256 mismatch for ${image}; refusing to load it."
}

download_image() {
    local temporary_image=""

    require_command curl
    require_command sha256sum
    mkdir -p "$CACHE_ROOT"

    if [[ -f "$IMAGE_PATH" ]]; then
        if printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE_PATH" | sha256sum --check --status; then
            log "Using verified cached image: ${IMAGE_PATH}"
            log "SHA-256: ${IMAGE_SHA256}"
            return 0
        fi
        die "Cached image has the wrong SHA-256: ${IMAGE_PATH}. Remove that file explicitly, then retry."
    fi

    temporary_image="$(mktemp "${CACHE_ROOT}/.${ASSET_NAME}.download.XXXXXX")"
    trap 'rm -f -- "${temporary_image:-}"' EXIT
    log "Downloading ${IMAGE_URL}"
    curl --fail --location --retry 3 --output "$temporary_image" "$IMAGE_URL"
    verify_image "$temporary_image"
    mv -- "$temporary_image" "$IMAGE_PATH"
    trap - EXIT

    log "Downloaded and verified: ${IMAGE_PATH}"
    log "SHA-256: ${IMAGE_SHA256}"
}

has_vendor_interface_six() {
    lsusb -d "$PLUTO_RUNTIME_ID" -v 2>/dev/null |
        awk '
            /bInterfaceNumber/ { interface_number = $2 }
            /bInterfaceClass/ {
                if (interface_number == 6 && $2 == 255) {
                    found = 1
                }
            }
            END { exit(found ? 0 : 1) }
        '
}

has_usb_iio_context() {
    iio_info -s 2>&1 | grep -Eq '\[usb:[^]]+\]'
}

show_status() {
    require_command lsusb
    log "Runtime USB devices (${PLUTO_RUNTIME_ID}):"
    lsusb -d "$PLUTO_RUNTIME_ID" || true
    log "DFU USB devices (${PLUTO_DFU_ID}):"
    lsusb -d "$PLUTO_DFU_ID" || true

    if [[ "$(usb_count "$PLUTO_RUNTIME_ID")" -ne 1 ]]; then
        log "Detailed status unavailable: exactly one runtime Pluto is required."
        return 0
    fi

    if has_vendor_interface_six; then
        log "Vendor-specific direct-USB interface 6: present"
    else
        log "Vendor-specific direct-USB interface 6: absent"
    fi

    if command -v iio_info >/dev/null 2>&1 && has_usb_iio_context; then
        log "Standard USB-IIO context: present"
    else
        log "Standard USB-IIO context: absent or iio_info unavailable"
    fi

    if ssh_pluto '
        printf "%s\n" "--- /opt/VERSIONS ---"
        cat /opt/VERSIONS
        printf "%s\n" "--- processes ---"
        ps | grep "[i]iod" || true
        ps | grep "[s]dr_usb_gadget" || true
    '; then
        return 0
    fi

    log "Pluto SSH status unavailable at ${SSH_HOST}."
}

verify_direct_firmware() {
    require_command iio_info
    require_command lsusb
    require_command ssh
    require_command sshpass
    require_one_usb_device "$PLUTO_RUNTIME_ID" "runtime Pluto"

    has_vendor_interface_six ||
        die "Vendor-specific direct-USB interface 6 is absent."
    has_usb_iio_context ||
        die "Standard USB-IIO context is absent."
    ssh_pluto '
        pidof iiod >/dev/null &&
        pidof sdr_usb_gadget >/dev/null
    ' || die "Required Pluto processes iiod and sdr_usb_gadget are not both running."

    log "PASS: standard USB-IIO, vendor interface 6, iiod, and sdr_usb_gadget are present."
    log "Firmware image SHA-256: ${IMAGE_SHA256}"
}

back_up_pluto_state() {
    local timestamp
    local backup_path

    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$STATE_ROOT"
    backup_path="${STATE_ROOT}/${timestamp}-before-direct-usb-ram-boot.txt"

    ssh_pluto '
        printf "%s\n" "--- /opt/VERSIONS ---"
        cat /opt/VERSIONS
        printf "%s\n" "--- fw_printenv ---"
        fw_printenv
    ' >"$backup_path" || die "Could not back up Pluto version and U-Boot environment."

    log "Saved pre-load Pluto state: ${backup_path}"
}

load_firmware() {
    local reboot_result=0

    require_command curl
    require_command dfu-util
    require_command iio_info
    require_command lsusb
    require_command sha256sum
    require_command ssh
    require_command sshpass

    download_image
    verify_image "$IMAGE_PATH"
    require_one_usb_device "$PLUTO_RUNTIME_ID" "runtime Pluto"
    [[ "$(usb_count "$PLUTO_DFU_ID")" -eq 0 ]] ||
        die "A DFU-mode Pluto is already present. Restore it to runtime mode before loading."
    ssh_pluto true >/dev/null ||
        die "Cannot reach the Pluto by SSH at ${SSH_HOST}."
    back_up_pluto_state

    log "Requesting volatile RAM-boot mode. QSPI will not be modified."
    ssh_pluto /usr/sbin/device_reboot ram || reboot_result=$?
    if [[ "$reboot_result" -ne 0 ]]; then
        log "SSH disconnected with status ${reboot_result}; checking whether the Pluto entered DFU mode."
    fi

    wait_for_usb_count "$PLUTO_DFU_ID" 1 30 "one Pluto in DFU mode"
    wait_for_usb_count "$PLUTO_RUNTIME_ID" 0 10 "the runtime Pluto to disconnect"

    log "Loading verified image into Pluto RAM..."
    dfu-util \
        -d "${PLUTO_RUNTIME_ID},${PLUTO_DFU_ID}" \
        -a firmware.dfu \
        -D "$IMAGE_PATH"
    dfu-util \
        -d "${PLUTO_RUNTIME_ID},${PLUTO_DFU_ID}" \
        -a firmware.dfu \
        -e

    wait_for_usb_count "$PLUTO_RUNTIME_ID" 1 60 "the RAM-booted Pluto to enumerate"
    wait_for_usb_count "$PLUTO_DFU_ID" 0 10 "the DFU device to detach"
    wait_for_ssh 60
    verify_direct_firmware

    log "RAM boot complete. This firmware is lost on reset or power cycle."
}

rollback_firmware() {
    local reboot_result=0

    require_command iio_info
    require_command lsusb
    require_command ssh
    require_command sshpass
    require_one_usb_device "$PLUTO_RUNTIME_ID" "runtime Pluto"
    ssh_pluto true >/dev/null ||
        die "Cannot reach the Pluto by SSH at ${SSH_HOST}."

    log "Rebooting into the unchanged firmware stored in QSPI..."
    ssh_pluto /usr/sbin/device_reboot reset || reboot_result=$?
    if [[ "$reboot_result" -ne 0 ]]; then
        log "SSH disconnected with status ${reboot_result}; waiting for the Pluto to return."
    fi

    wait_for_stock_runtime 90

    if has_vendor_interface_six; then
        die "Vendor interface 6 is still present after rollback; inspect the installed QSPI image."
    fi
    has_usb_iio_context ||
        die "Standard USB-IIO context is absent after rollback."
    if ssh_pluto 'pidof sdr_usb_gadget >/dev/null'; then
        die "sdr_usb_gadget is still running after rollback; inspect the installed QSPI image."
    fi

    log "PASS: stock QSPI firmware is running and direct-USB interface 6 is absent."
    show_status
}

run_multi_loader() {
    local command="$1"
    local expected_count="$2"
    shift 2

    [[ "$expected_count" =~ ^[1-9][0-9]*$ ]] ||
        die "Expected Pluto count must be a positive integer."
    require_command python3
    [[ -f "$MULTI_LOADER" ]] ||
        die "Multi-radio loader is missing: ${MULTI_LOADER}"
    if [[ "$command" == "load-all" ]]; then
        download_image
        verify_image "$IMAGE_PATH"
    fi
    python3 "$MULTI_LOADER" \
        "$command" \
        --image "$IMAGE_PATH" \
        --image-sha256 "$IMAGE_SHA256" \
        --ssh-config "$SSH_CONFIG" \
        --ssh-password "$SSH_PASSWORD" \
        --state-root "$STATE_ROOT" \
        --expected-count "$expected_count" \
        --approved-qspi-version "$APPROVED_QSPI_DEVICE_FW" \
        "$@"
}

discover_count() {
    require_command python3
    python3 "$MULTI_LOADER" discover-count
}

main() {
    local command="${1:-}"

    case "$command" in
        download)
            [[ "$#" -eq 1 ]] || die "download takes no arguments."
            download_image
            ;;
        status)
            [[ "$#" -eq 1 ]] || die "status takes no arguments."
            show_status
            ;;
        load)
            [[ "$#" -eq 1 ]] || die "load takes no arguments."
            load_firmware
            ;;
        verify)
            [[ "$#" -eq 1 ]] || die "verify takes no arguments."
            verify_direct_firmware
            ;;
        rollback)
            [[ "$#" -eq 1 ]] || die "rollback takes no arguments."
            rollback_firmware
            ;;
        load-all|verify-all|check-config-all|rollback-all|status-all)
            [[ "$#" -eq 2 ]] || die "${command} requires the expected Pluto count."
            run_multi_loader "$command" "$2"
            ;;
        provision-config-all)
            [[ "$#" -eq 2 || "$#" -eq 3 ]] ||
                die "provision-config-all requires N and optional --dry-run."
            if [[ "$#" -eq 3 && "$3" != "--dry-run" ]]; then
                die "Unknown provision-config-all option: $3"
            fi
            if [[ "$#" -eq 3 ]]; then
                run_multi_loader "$command" "$2" "$3"
            else
                run_multi_loader "$command" "$2"
            fi
            ;;
        discover-count)
            [[ "$#" -eq 1 ]] || die "discover-count takes no arguments."
            discover_count
            ;;
        -h|--help|help)
            usage
            ;;
        "")
            usage >&2
            exit 2
            ;;
        *)
            usage >&2
            die "Unknown command: ${command}"
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
