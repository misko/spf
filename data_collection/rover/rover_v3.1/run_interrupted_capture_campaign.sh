#!/usr/bin/env bash
# Exercise graceful and abrupt production-collector interruption semantics on
# every configured Pluto, then prove that a clean V7 capture still succeeds.
# Receive-only: all collector invocations use --fake-drone and never enable TX.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
readonly ROVER_ID="${SPF_ROVER_ID:-$(tr -d '[:space:]' </home/pi/rover_id)}"
readonly CONFIG="${SPF_CAPTURE_CONFIG:-}"
readonly OUTPUT_ROOT="${SPF_INTERRUPT_OUTPUT_ROOT:-/home/pi/preflight/interrupted_capture}"
readonly CASES="${SPF_INTERRUPT_CASES:-sigterm:2 sigint:10 sigkill:25 sigterm:100}"
readonly CLEAN_RECORDS="${SPF_INTERRUPT_CLEAN_RECORDS:-100}"
readonly FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
readonly FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"
readonly PREPARE_SCRIPT="${SPF_PREPARE_DIRECT_USB_BOOT:-${SCRIPT_DIR}/prepare_direct_usb_boot.sh}"
readonly DEVICE_MAPPING="${SPF_DEVICE_MAPPING:-/home/pi/device_mapping}"
readonly DMESG_BIN="${SPF_DMESG_BIN:-dmesg}"
readonly SUDO_BIN="${SPF_SUDO:-sudo}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "${EUID}" -ne 0 ]] || die "run as the rover user; sudo is used only for boot preparation"
[[ -x "$PYTHON" ]] || die "Python is unavailable: ${PYTHON}"
[[ -x "$PREPARE_SCRIPT" ]] || die "boot preparation is unavailable: ${PREPARE_SCRIPT}"
command -v "$DMESG_BIN" >/dev/null || die "dmesg command is unavailable: ${DMESG_BIN}"
command -v "$SUDO_BIN" >/dev/null || die "sudo command is unavailable: ${SUDO_BIN}"
[[ "$ROVER_ID" =~ ^[1-3]$ ]] || die "unsupported rover ID: ${ROVER_ID}"
[[ "$CLEAN_RECORDS" =~ ^[1-9][0-9]*$ ]] || die "clean record count must be positive"

resolver_args=(--rover-id "$ROVER_ID" --format null)
if [[ -n "$CONFIG" ]]; then
    resolver_args+=(--config "$CONFIG")
fi
mapfile -d '' -t plan < <(
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.rover_capture_config "${resolver_args[@]}"
)
[[ "${#plan[@]}" -eq 15 ]] || die "capture resolver returned ${#plan[@]} fields"
readonly RESOLVED_CONFIG="${plan[1]}"
readonly ROUTINE="${plan[3]}"
readonly EXPECTED_RADIOS="${plan[5]}"

run_id="$(date -u +%Y%m%dT%H%M%SZ)_rover${ROVER_ID}"
run_root="${OUTPUT_ROOT}/${run_id}"
ready_file="${run_root}/ready.json"
mkdir -p "$run_root"

"$SUDO_BIN" env \
    PYTHONPATH="$REPO_ROOT" \
    SPF_ROVER_ID="$ROVER_ID" \
    SPF_CAPTURE_CONFIG="$RESOLVED_CONFIG" \
    SPF_DIRECT_USB_READY_FILE="$ready_file" \
    SPF_PYTHON="$PYTHON" \
    SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
    SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
    bash "$PREPARE_SCRIPT" \
    >"${run_root}/prepare.log" 2>&1
[[ -s "$ready_file" ]] || die "boot preparation returned without a readiness manifest"

case_index=0
for specification in $CASES; do
    case_index=$((case_index + 1))
    interrupt_signal="${specification%%:*}"
    minimum_records="${specification#*:}"
    [[ "$interrupt_signal" =~ ^sig(int|term|kill)$ ]] ||
        die "invalid interruption signal in case: ${specification}"
    [[ "$minimum_records" =~ ^[1-9][0-9]*$ ]] ||
        die "invalid committed-record threshold in case: ${specification}"

    case_root="${run_root}/case-$(printf '%02d' "$case_index")-${interrupt_signal}-${minimum_records}"
    mkdir -p "$case_root/reports"
    "$DMESG_BIN" >"${case_root}/dmesg-before.txt" ||
        die "could not snapshot kernel log before case ${specification}"
    before_lines="$(wc -l <"${case_root}/dmesg-before.txt")"

    set +e
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m pytest -q \
        "${REPO_ROOT}/tests/radio_hardware/test_interrupted_collection_hardware.py" \
        --radio-hardware \
        --radio-interrupt \
        --radio-interrupt-signal="$interrupt_signal" \
        --radio-interrupt-min-records="$minimum_records" \
        --radio-expected-count="$EXPECTED_RADIOS" \
        --radio-capture-config="$RESOLVED_CONFIG" \
        --radio-device-mapping="$DEVICE_MAPPING" \
        --radio-ready-manifest="$ready_file" \
        --radio-report-dir="${case_root}/reports" \
        --basetemp="${case_root}/pytest-temp" \
        --junitxml="${case_root}/junit.xml" \
        >"${case_root}/pytest.log" 2>&1
    pytest_status=$?

    "$DMESG_BIN" >"${case_root}/dmesg-after.txt"
    dmesg_status=$?
    set -e
    kernel_usb_error=0
    if (( dmesg_status == 0 )); then
        tail -n "+$((before_lines + 1))" "${case_root}/dmesg-after.txt" \
            >"${case_root}/dmesg-delta.txt"
        if grep -Eqi 'USB disconnect|error -71|device descriptor read|xhci.*error|I/O error' \
            "${case_root}/dmesg-delta.txt"; then
            kernel_usb_error=1
        fi
    else
        : >"${case_root}/dmesg-delta.txt"
    fi
    printf '%s\n' \
        "pytest_status=${pytest_status}" \
        "dmesg_status=${dmesg_status}" \
        "kernel_usb_error=${kernel_usb_error}" \
        >"${case_root}/case-status.env"

    if (( dmesg_status != 0 )); then
        die "could not snapshot kernel log after case ${specification}"
    fi
    if (( kernel_usb_error != 0 )); then
        die "kernel USB error appeared during interruption case ${specification}"
    fi
    if (( pytest_status != 0 )); then
        printf 'ERROR: interruption case %s failed with status %s\n' \
            "$specification" "$pytest_status" >&2
        exit "$pytest_status"
    fi
done

# A matrix can prove fail-closed behavior while still leaving a latent recovery
# problem. End with an ordinary production-sized capture and strict validator.
clean_root="${run_root}/clean-recovery"
mkdir -p "$clean_root"
PYTHONPATH="$REPO_ROOT" SPF_DIRECT_USB_READY_FILE="$ready_file" \
    "$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
    --fake-drone --no-ultrasonic \
    --yaml-config "$RESOLVED_CONFIG" \
    --device-mapping "$DEVICE_MAPPING" \
    --routine "$ROUTINE" \
    --records-per-receiver "$CLEAN_RECORDS" \
    --temp "$clean_root" \
    --tag "INTERRUPT_RECOVERY_RO${ROVER_ID}" \
    >"${clean_root}/collector.log" 2>&1

mapfile -t clean_stores < <(find "$clean_root" -maxdepth 1 -name '*.zarr' -print)
[[ "${#clean_stores[@]}" -eq 1 ]] ||
    die "clean recovery produced ${#clean_stores[@]} final Zarr stores"
PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.validate_direct_usb_v7_zarr \
    "${clean_stores[0]}" \
    --expected-frames "$CLEAN_RECORDS" \
    --expected-receivers "$EXPECTED_RADIOS" \
    --output "${clean_root}/validation.json" >/dev/null

printf 'PASS\n' >"${run_root}/PASS"
printf 'PASS: interrupted-capture campaign completed: %s\n' "$run_root"
