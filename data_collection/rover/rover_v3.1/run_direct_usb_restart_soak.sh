#!/usr/bin/env bash
# Restart every configured Pluto between production V7 captures and prove at
# least the requested aggregate on-air capture duration. Receive-only: the
# collector uses --fake-drone and never enables a Pluto transmitter.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
readonly ROVER_ID="${SPF_ROVER_ID:-$(tr -d '[:space:]' </home/pi/rover_id)}"
readonly CONFIG="${SPF_CAPTURE_CONFIG:-}"
readonly OUTPUT_ROOT="${SPF_RESTART_SOAK_OUTPUT_ROOT:-/home/pi/preflight/direct_usb_restart_soak}"
readonly RECORDS="${SPF_RESTART_SOAK_RECORDS:-3500}"
readonly MIN_SESSIONS="${SPF_RESTART_SOAK_MIN_SESSIONS:-2}"
readonly MAX_SESSIONS="${SPF_RESTART_SOAK_MAX_SESSIONS:-4}"
readonly MIN_CAPTURE_SECONDS="${SPF_RESTART_SOAK_MIN_CAPTURE_SECONDS:-3600}"
readonly FIRMWARE_CACHE="${SPF_FIRMWARE_CACHE_DIR:-/home/pi/.cache/spf/firmware}"
readonly FIRMWARE_STATE="${SPF_FIRMWARE_STATE_DIR:-/var/lib/spf/pluto-firmware}"
readonly SSH_CONFIG="${SCRIPT_DIR}/ssh_config"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "${EUID}" -ne 0 ]] || die "run as the rover user; the script invokes sudo only for hardware preparation"
[[ -x "$PYTHON" ]] || die "Python is unavailable: ${PYTHON}"
[[ "$ROVER_ID" =~ ^[1-3]$ ]] || die "unsupported rover ID: ${ROVER_ID}"
for value in "$RECORDS" "$MIN_SESSIONS" "$MAX_SESSIONS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "record/session counts must be positive integers"
done
[[ "$MIN_CAPTURE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "SPF_RESTART_SOAK_MIN_CAPTURE_SECONDS must be non-negative"
(( MIN_SESSIONS <= MAX_SESSIONS )) || die "minimum sessions exceeds maximum sessions"

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
readonly FIRMWARE_ASSET="${plan[9]}"
readonly FIRMWARE_SHA256="${plan[11]}"
readonly FIRMWARE_IMAGE="${FIRMWARE_CACHE}/${FIRMWARE_ASSET}"

run_id="$(date -u +%Y%m%dT%H%M%SZ)_rover${ROVER_ID}"
run_root="${OUTPUT_ROOT}/${run_id}"
mkdir -p "$run_root"
zarr_list="${run_root}/zarr_paths.txt"
: >"$zarr_list"

prepare() {
    local ready_file="$1"
    local log_file="$2"
    sudo env \
        PYTHONPATH="$REPO_ROOT" \
        SPF_ROVER_ID="$ROVER_ID" \
        SPF_CAPTURE_CONFIG="$RESOLVED_CONFIG" \
        SPF_DIRECT_USB_READY_FILE="$ready_file" \
        SPF_PYTHON="$PYTHON" \
        SPF_FIRMWARE_CACHE_DIR="$FIRMWARE_CACHE" \
        SPF_FIRMWARE_STATE_DIR="$FIRMWARE_STATE" \
        bash "${SCRIPT_DIR}/prepare_direct_usb_boot.sh" >"$log_file" 2>&1
    [[ -s "$ready_file" ]] || die "boot preparation returned without a readiness manifest"
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.pluto_ready_manifest verify \
        --rover-id "$ROVER_ID" \
        --config "$RESOLVED_CONFIG" \
        --output "$ready_file" \
        --device-mapping /home/pi/device_mapping >"${ready_file}.verified.json"
}

# Establish the pinned image/config before asking the restart-only command to
# preserve it. This matching path performs no firmware write.
prepare "${run_root}/initial-ready.json" "${run_root}/initial-prepare.log"

for session_index in $(seq 1 "$MAX_SESSIONS"); do
    session_root="${run_root}/session-$(printf '%02d' "$session_index")"
    mkdir -p "$session_root"

    sudo env PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.pluto_multi_firmware \
        restart-all \
        --image "$FIRMWARE_IMAGE" \
        --image-sha256 "$FIRMWARE_SHA256" \
        --ssh-config "$SSH_CONFIG" \
        --state-root "$FIRMWARE_STATE" \
        --expected-count "$EXPECTED_RADIOS" \
        >"${session_root}/restart.log" 2>&1

    ready_file="${session_root}/ready.json"
    prepare "$ready_file" "${session_root}/prepare.log"
    dmesg >"${session_root}/dmesg-before.txt"
    before_lines="$(wc -l <"${session_root}/dmesg-before.txt")"
    start_ns="$(date +%s%N)"

    PYTHONPATH="$REPO_ROOT" SPF_DIRECT_USB_READY_FILE="$ready_file" \
        "$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
        --fake-drone --no-ultrasonic \
        --yaml-config "$RESOLVED_CONFIG" \
        --device-mapping /home/pi/device_mapping \
        --routine "$ROUTINE" \
        --records-per-receiver "$RECORDS" \
        --temp "$session_root" \
        --tag "RESTART_SOAK_S${session_index}_RO${ROVER_ID}" \
        >"${session_root}/collector.log" 2>&1 &
    collector_pid=$!
    printf 'timestamp_unix,pid,rss_kib,rss_anon_kib,rss_file_kib,vmsize_kib,available_kib,artifact_kib\n' \
        >"${session_root}/resources.csv"
    (
        while kill -0 "$collector_pid" 2>/dev/null; do
            timestamp="$(date +%s)"
            rss="$(awk '/VmRSS:/ {print $2}' "/proc/${collector_pid}/status" 2>/dev/null || true)"
            rss_anon="$(awk '/RssAnon:/ {print $2}' "/proc/${collector_pid}/status" 2>/dev/null || true)"
            rss_file="$(awk '/RssFile:/ {print $2}' "/proc/${collector_pid}/status" 2>/dev/null || true)"
            vmsize="$(awk '/VmSize:/ {print $2}' "/proc/${collector_pid}/status" 2>/dev/null || true)"
            available="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
            artifact="$(du -sk "$session_root" 2>/dev/null | awk '{print $1}')"
            printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$timestamp" "$collector_pid" "$rss" "$rss_anon" "$rss_file" \
                "$vmsize" "$available" "$artifact"
            sleep 30
        done
    ) >>"${session_root}/resources.csv" 2>"${session_root}/resource-monitor.log" &
    monitor_pid=$!
    set +e
    wait "$collector_pid"
    collector_status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    [[ "$collector_status" -eq 0 ]] ||
        die "collector failed in session ${session_index} with status ${collector_status}"
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.validate_resource_samples \
        "${session_root}/resources.csv" \
        --output "${session_root}/resource-validation.json" >/dev/null

    end_ns="$(date +%s%N)"
    awk -v start="$start_ns" -v end="$end_ns" \
        'BEGIN { printf "%.3f\n", (end-start)/1000000000 }' \
        >"${session_root}/collector_wall_seconds"
    dmesg >"${session_root}/dmesg-after.txt"
    tail -n "+$((before_lines + 1))" "${session_root}/dmesg-after.txt" \
        >"${session_root}/dmesg-delta.txt"
    if grep -Eqi 'USB disconnect|error -71|device descriptor read|xhci.*error|I/O error' \
        "${session_root}/dmesg-delta.txt"; then
        die "kernel USB error appeared during session ${session_index}"
    fi

    mapfile -t stores < <(find "$session_root" -maxdepth 1 -name '*.zarr' -print)
    [[ "${#stores[@]}" -eq 1 ]] ||
        die "session ${session_index} produced ${#stores[@]} final Zarr stores"
    zarr_path="${stores[0]}"
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.validate_direct_usb_v7_zarr \
        "$zarr_path" \
        --expected-frames "$RECORDS" \
        --expected-receivers "$EXPECTED_RADIOS" \
        --output "${session_root}/validation.json" >/dev/null
    printf '%s\n' "$zarr_path" >>"$zarr_list"

    aggregate_json="${run_root}/aggregate.json"
    set +e
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.validate_restart_soak \
        --zarr-list "$zarr_list" \
        --minimum-sessions "$MIN_SESSIONS" \
        --minimum-capture-seconds "$MIN_CAPTURE_SECONDS" \
        --output "$aggregate_json"
    aggregate_status=$?
    set -e
    if [[ "$aggregate_status" -eq 0 ]]; then
        printf 'PASS\n' >"${run_root}/PASS"
        printf 'PASS: restart soak completed: %s\n' "$run_root"
        exit 0
    fi
    [[ "$aggregate_status" -eq 3 ]] || die "aggregate validation failed"
done

die "maximum sessions reached before the capture-duration gate passed"
