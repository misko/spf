#!/usr/bin/env bash
#
# Run one production-shaped, motion-free direct-USB capture after boot.

set -euo pipefail

readonly REPO_ROOT="/home/pi/spf"
readonly READY_FILE="/run/spf/direct_usb_ready.json"
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
readonly RECORDS="${SPF_DIRECT_USB_RECORDS:-100}"
readonly OUTPUT_ROOT="${SPF_DIRECT_USB_OUTPUT_ROOT:-/home/pi/preflight/boot_direct_usb}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f "$READY_FILE" ]] || die "Direct-USB firmware preparation did not pass."
[[ -x "$PYTHON" ]] || die "Python environment is unavailable: ${PYTHON}"
rover_id="$(tr -d '[:space:]' </home/pi/rover_id)"
resolver_args=(--rover-id "$rover_id" --format null)
if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
    resolver_args+=(--config "$SPF_CAPTURE_CONFIG")
fi
mapfile -d '' -t config_values < <(
    "$PYTHON" -m spf.scripts.rover_capture_config "${resolver_args[@]}"
)
[[ "${#config_values[@]}" -eq 15 ]] ||
    die "Capture config resolver returned ${#config_values[@]} fields, expected 15."
CONFIG="${config_values[1]}"
EXPECTED_RADIOS="${config_values[5]}"
[[ -f "$CONFIG" ]] || die "Direct-USB capture config is unavailable: ${CONFIG}"
[[ "$RECORDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_DIRECT_USB_RECORDS must be a positive integer."
[[ "$EXPECTED_RADIOS" =~ ^[1-9][0-9]*$ ]] ||
    die "Canonical config receiver count must be a positive integer."

export PYTHONBREAKPOINT=0
run_dir="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)_rover${rover_id}"
mkdir -p "$run_dir"

cd "$REPO_ROOT"
"$PYTHON" spf/mavlink_radio_collection.py \
    --fake-drone \
    --no-ultrasonic \
    --yaml-config "$CONFIG" \
    --device-mapping /home/pi/device_mapping \
    --routine center \
    --records-per-receiver "$RECORDS" \
    --temp "$run_dir" \
    --tag "BOOT_DIRECT_V7_RO${rover_id}" \
    2>&1 | tee "$run_dir/console.txt"

mapfile -t zarr_paths < <(find "$run_dir" -maxdepth 1 -name '*.zarr' -print)
[[ "${#zarr_paths[@]}" -eq 1 ]] ||
    die "Expected one final Zarr in ${run_dir}; found ${#zarr_paths[@]}."

"$PYTHON" -m spf.scripts.validate_direct_usb_v7_zarr \
    "${zarr_paths[0]}" \
    --expected-frames "$RECORDS" \
    --expected-receivers "$EXPECTED_RADIOS" \
    --output "$run_dir/validation.json"

printf 'PASS\n' >"$run_dir/PASS"
printf 'PASS: boot direct-USB preflight completed: %s\n' "$run_dir"
