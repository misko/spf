#!/usr/bin/env bash
#
# Rover 3.1 production boot launcher.
#
# The canonical Rover YAML is the source of truth for dataset schema, firmware,
# radio count, motion routine, frame count, geometry, and RF configuration.

set -euo pipefail

readonly REPO_ROOT="/home/pi/spf"
readonly SCRIPT_DIR="${REPO_ROOT}/data_collection/rover/rover_v3.1"
readonly PROFILE_ENV="/etc/spf/rover_collection.env"
readonly READY_FILE="/run/spf/direct_usb_ready.json"
readonly DEVICE_MAPPING="/home/pi/device_mapping"
readonly MAVLINK_CONTROLLER="${REPO_ROOT}/spf/mavlink/mavlink_controller.py"
readonly PARAMS_FILE="/home/pi/this_rover.params"
readonly TIME_FILE="/home/pi/time"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off|"") return 1 ;;
        *) die "Invalid boolean value: $1" ;;
    esac
}

if [[ -f "$PROFILE_ENV" ]]; then
    # The file is root-managed and contains only shell-style assignments.
    # shellcheck disable=SC1090
    source "$PROFILE_ENV"
fi

PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
SKIP_PARAMETER_SYNC="${SPF_SKIP_PARAMETER_SYNC:-0}"
BOOT_VALIDATE_ONLY="${SPF_BOOT_VALIDATE_ONLY:-0}"
RUN_ONCE="${SPF_RUN_ONCE:-0}"
OUTPUT_ROOT="${SPF_OUTPUT_ROOT:-/home/pi/temp}"
CAPTURE_STATUS_FILE="${SPF_CAPTURE_STATUS_FILE:-/home/pi/preflight/capture_status.json}"
RADIO_WAIT_SECONDS="${SPF_RADIO_WAIT_SECONDS:-600}"
ROVER_ID_FILE="${SPF_ROVER_ID_FILE:-/home/pi/rover_id}"

if [[ "$PYTHON" == */* ]]; then
    [[ -x "$PYTHON" ]] || die "Python environment is unavailable: ${PYTHON}"
else
    command -v "$PYTHON" >/dev/null 2>&1 ||
        die "Python environment is unavailable: ${PYTHON}"
fi
[[ -f "$ROVER_ID_FILE" ]] || die "Missing ${ROVER_ID_FILE}."
rover_id="$(tr -d '[:space:]' <"$ROVER_ID_FILE")"
[[ "$rover_id" =~ ^[1-3]$ ]] || die "Unsupported rover_id: ${rover_id}"

resolver_args=(
    --rover-id "$rover_id"
    --format null
)
if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
    resolver_args+=(--config "$SPF_CAPTURE_CONFIG")
fi
mapfile -d '' -t config_values < <(
    "$PYTHON" -m spf.scripts.rover_capture_config "${resolver_args[@]}"
)
[[ "${#config_values[@]}" -eq 15 ]] ||
    die "Capture config resolver returned ${#config_values[@]} fields, expected 15."
config="${config_values[1]}"
config_sha256="${config_values[2]}"
routine="${config_values[3]}"
records_per_receiver="${config_values[4]}"
expected_radios="${config_values[5]}"
rx_transport="${config_values[6]}"
data_version="${config_values[7]}"
firmware_release_tag="${config_values[8]}"
firmware_image_sha256="${config_values[11]}"

[[ "$RADIO_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_RADIO_WAIT_SECONDS must be a positive integer."

print_plan() {
    printf '%s\n' \
        "rover_id=${rover_id}" \
        "config=${config}" \
        "config_sha256=${config_sha256}" \
        "routine=${routine}" \
        "records_per_receiver=${records_per_receiver}" \
        "expected_radios=${expected_radios}" \
        "rx_transport=${rx_transport}" \
        "data_version=${data_version}" \
        "firmware_release_tag=${firmware_release_tag}" \
        "firmware_image_sha256=${firmware_image_sha256}" \
        "boot_validate_only=${BOOT_VALIDATE_ONLY}" \
        "run_once=${RUN_ONCE}" \
        "output_root=${OUTPUT_ROOT}" \
        "capture_status_file=${CAPTURE_STATUS_FILE}"
}

case "${1:-}" in
    --print-plan)
        [[ "$#" -eq 1 ]] || die "--print-plan takes no arguments."
        print_plan
        exit 0
        ;;
    --boot-validate-only)
        [[ "$#" -eq 1 ]] || die "--boot-validate-only takes no arguments."
        BOOT_VALIDATE_ONLY=1
        ;;
    --once)
        [[ "$#" -eq 1 ]] || die "--once takes no arguments."
        RUN_ONCE=1
        ;;
    "")
        ;;
    *)
        die "Unknown argument: $1"
        ;;
esac

export PYTHONPATH="$REPO_ROOT"
export PYTHONBREAKPOINT=0

wait_for_radios() {
    local deadline found_radios
    deadline=$((SECONDS + RADIO_WAIT_SECONDS))
    while (( SECONDS < deadline )); do
        found_radios="$(lsusb | grep -c ADALM || true)"
        if [[ "$found_radios" -eq "$expected_radios" ]]; then
            return 0
        fi
        printf 'Expected %s Pluto radios but found %s; retrying.\n' \
            "$expected_radios" "$found_radios"
        "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer failure || true
        sleep 5
    done
    die "Timed out waiting for ${expected_radios} Pluto radios."
}

verify_direct_ready() {
    [[ -f "$READY_FILE" ]] ||
        die "Direct-USB preparation did not produce ${READY_FILE}."
    [[ -f "$DEVICE_MAPPING" ]] || die "Missing ${DEVICE_MAPPING}."
    manifest_args=(
        verify
        --rover-id "$rover_id"
        --output "$READY_FILE"
        --device-mapping "$DEVICE_MAPPING"
    )
    if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
        manifest_args+=(--config "$SPF_CAPTURE_CONFIG")
    fi
    "$PYTHON" -m spf.scripts.pluto_ready_manifest \
        "${manifest_args[@]}" >/dev/null
}

sync_vehicle_configuration() {
    is_true "$SKIP_PARAMETER_SYNC" && return 0
    sed "s/__ROVER_ID__/${rover_id}/g" \
        < <(cat \
            "${SCRIPT_DIR}/rover3_base_parameters.params" \
            "${SCRIPT_DIR}/rover3_rc_servo_parameters.params") \
        >"$PARAMS_FILE"
    # Verify-first: the committed params already match on a normal boot, so skip
    # the write (and its full ~1300-param FC fetch over 115200 serial) when the
    # diff is already clean. --diff-params exits with the diff count (0 == match).
    if "$PYTHON" "$MAVLINK_CONTROLLER" --diff-params "$PARAMS_FILE"; then
        return 0
    fi
    "$PYTHON" "$MAVLINK_CONTROLLER" --load-params "$PARAMS_FILE"
    if ! "$PYTHON" "$MAVLINK_CONTROLLER" --diff-params "$PARAMS_FILE"; then
        die "Vehicle parameter verification failed after loading parameters."
    fi
}

system_clock_is_plausible() {
    # Raspberry Pi 5 has an RTC and may also have NTP. A plausible clock is
    # sufficient for boot filenames; GPS UTC is refreshed after a completed
    # capture, when the vehicle necessarily has a usable navigation solution.
    [[ "$(date +%s)" -ge 1735689600 ]]  # 2025-01-01T00:00:00Z
}

sync_gps_time() {
    local phase="${1:-capture}"
    if [[ "$phase" == "boot" ]] && system_clock_is_plausible; then
        printf '%s\n' \
            'System clock is plausible; deferring GPS UTC refresh until after capture.'
        return 0
    fi
    # --get-time blocks until the FC has GPS UTC time. Bound the first attempt so
    # a no-sky / cold-TTFF boot cannot hang forever before the governor and
    # capture loop; the loop below keeps re-syncing and sets the clock the moment
    # GPS locks. Time comes from GPS via MAVLink (no internet/NTP).
    if timeout "${SPF_GPS_TIME_TIMEOUT:-180}" \
        "$PYTHON" "$MAVLINK_CONTROLLER" --get-time "$TIME_FILE"; then
        gps_time="$(cat "$TIME_FILE")"
        # --get-time can return a 1970 timestamp if it exits on a 3D fix before
        # UTC arrives; never set the system clock to epoch-0 (it would corrupt
        # every subsequent data/log filename until the next successful sync).
        if [[ "$gps_time" == 1970-* ]]; then
            printf 'GPS reported a 1970 time (fix without UTC yet); not setting clock.\n'
        else
            sudo date -s "$gps_time"
        fi
    else
        printf 'No GPS time within %ss; continuing (capture loop keeps retrying --get-time).\n' \
            "${SPF_GPS_TIME_TIMEOUT:-180}"
    fi
}

read_only_vehicle_gate() {
    local status_file
    status_file="$(mktemp /tmp/spf-mavlink-status.XXXXXX)"
    trap 'rm -f -- "${status_file:-}"' RETURN
    "$PYTHON" "$MAVLINK_CONTROLLER" --status-json "$status_file"
    "$PYTHON" -c \
        'import json,sys; s=json.load(open(sys.argv[1])); '\
'assert s["armed"] is False, "vehicle is armed"; '\
'print("PASS: real MAVLink heartbeat received; vehicle is disarmed; mode="+s["mav_mode"])' \
        "$status_file"
    rm -f -- "$status_file"
    trap - RETURN
}

run_capture() {
    local capture_status
    mkdir -p "$OUTPUT_ROOT" "$(dirname "$CAPTURE_STATUS_FILE")"
    set +e
    "$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
        --yaml-config "$config" \
        --device-mapping "$DEVICE_MAPPING" \
        --tag "RO${rover_id}" \
        --temp "$OUTPUT_ROOT" \
        --status-file "$CAPTURE_STATUS_FILE"
    capture_status=$?
    set -e
    if [[ "$capture_status" -ne 0 ]]; then
        "$PYTHON" -m spf.capture_status mark-failed \
            --path "$CAPTURE_STATUS_FILE" \
            --exit-code "$capture_status" || true
        for _attempt in 1 2 3; do
            "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer failure || true
            sleep 1
        done
        return "$capture_status"
    fi
}

main() {
    print_plan
    wait_for_radios

    if [[ "$rx_transport" == "direct_usb" ]]; then
        verify_direct_ready
    else
        printf 'Read-only verification of legacy IIO Pluto radios.\n'
        sudo -n bash "${SCRIPT_DIR}/load_direct_usb_firmware.sh" \
            check-config-all "$expected_radios"
        bash "${SCRIPT_DIR}/device_mapping.sh" >"$DEVICE_MAPPING"
    fi

    if is_true "$BOOT_VALIDATE_ONLY"; then
        read_only_vehicle_gate
        printf 'PASS: boot validation stopped before parameter writes or collection.\n'
        return 0
    fi

    sync_vehicle_configuration
    sync_gps_time boot
    printf 'performance\n' | sudo tee \
        /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null

    while true; do
        run_capture
        is_true "$RUN_ONCE" && break
        sleep 8
        sync_gps_time capture
        sleep 2
    done
}

main
