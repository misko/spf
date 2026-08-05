#!/usr/bin/env bash
#
# Capture a short motion-free v7 store on every configured radio, then report
# per-channel receive health.
#
# This is the loop for diagnosing an antenna: swap it, run this, read the table.
# It uses the production capture path -- same collector, same config, same v7
# store -- so what it measures is what a mission would record. The only
# differences are --fake-drone (no vehicle motion) and a small record count.
#
# Pass an existing .zarr to skip the capture and only report on that store.
#
# Usage:
#   run_rx_signal_check.sh                 # capture 100 frames/receiver, report
#   run_rx_signal_check.sh --records 250   # longer capture
#   run_rx_signal_check.sh /path/to.zarr   # report on an existing store

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly SERVICE_NAME="mavlink_controller.service"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
RECORDS="${SPF_RX_CHECK_RECORDS:-100}"
OUTPUT_ROOT="${SPF_RX_CHECK_OUTPUT_ROOT:-/home/pi/preflight/rx_signal_check}"
KEEP=0
EXISTING=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --records) [[ "$#" -ge 2 ]] || die "--records requires a value."
                   RECORDS="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help)
            sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)         EXISTING="$1"; shift ;;
    esac
done

if [[ -n "$EXISTING" ]]; then
    [[ -e "$EXISTING" ]] || die "No such store: ${EXISTING}"
    cd "$REPO_ROOT"
    exec "$PYTHON" -m spf.scripts.rx_signal_metrics "$EXISTING"
fi

[[ -x "$PYTHON" ]] || die "Python environment is unavailable: ${PYTHON}"
[[ "$RECORDS" =~ ^[1-9][0-9]*$ ]] || die "--records must be a positive integer."

# The collector owns the radios in production. Refuse rather than fight it for
# USB, which produces confusing partial captures rather than a clean error.
if systemctl is-active --quiet "$SERVICE_NAME"; then
    die "${SERVICE_NAME} is active and owns the radios.
  Stop it first: sudo systemctl stop ${SERVICE_NAME}"
fi

rover_id="$(tr -d '[:space:]' </home/pi/rover_id 2>/dev/null || true)"
[[ -n "$rover_id" ]] || die "Missing /home/pi/rover_id; this must run on a rover."

resolver_args=(--rover-id "$rover_id" --format null)
[[ -z "${SPF_CAPTURE_CONFIG:-}" ]] || resolver_args+=(--config "$SPF_CAPTURE_CONFIG")
mapfile -d '' -t config_values < <(
    "$PYTHON" -m spf.scripts.rover_capture_config "${resolver_args[@]}"
)
# -ge for the same reason as run_direct_usb_boot_preflight.sh: the resolver has
# gained fields before and indexing a fixed count broke callers silently.
[[ "${#config_values[@]}" -ge 16 ]] ||
    die "Capture config resolver returned ${#config_values[@]} fields."
CONFIG="${config_values[1]}"
[[ -f "$CONFIG" ]] || die "Capture config is unavailable: ${CONFIG}"

run_dir="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)_rover${rover_id}"
mkdir -p "$run_dir"

printf '\n== capturing %s frames per receiver (motion-free) ==\n' "$RECORDS"
printf '   config : %s\n   output : %s\n\n' "$CONFIG" "$run_dir"

export PYTHONBREAKPOINT=0
cd "$REPO_ROOT"
"$PYTHON" spf/mavlink_radio_collection.py \
    --fake-drone \
    --no-ultrasonic \
    --yaml-config "$CONFIG" \
    --device-mapping /home/pi/device_mapping \
    --routine center \
    --records-per-receiver "$RECORDS" \
    --temp "$run_dir" \
    --tag "RXCHECK_RO${rover_id}" \
    >"$run_dir/console.txt" 2>&1 ||
    { tail -20 "$run_dir/console.txt" >&2; die "Capture failed; see ${run_dir}/console.txt"; }

mapfile -t zarr_paths < <(find "$run_dir" -maxdepth 1 -name '*.zarr' -print)
[[ "${#zarr_paths[@]}" -eq 1 ]] ||
    die "Expected one final Zarr in ${run_dir}; found ${#zarr_paths[@]}. See console.txt"

status=0
"$PYTHON" -m spf.scripts.rx_signal_metrics "${zarr_paths[0]}" \
    --json "$run_dir/rx_signal_metrics.json" || status=$?

if [[ "$KEEP" -eq 0 ]]; then
    printf '  store: %s\n' "${zarr_paths[0]}"
    printf '  (kept; remove %s when done)\n' "$run_dir"
fi
exit "$status"
