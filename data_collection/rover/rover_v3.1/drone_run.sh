#!/usr/bin/env bash
#
# Rover 3.1 production boot launcher.
#
# With no overrides this preserves the historical legacy-IIO capture loop.
# /etc/spf/rover_collection.env selects a qualified direct-USB profile without
# changing the motion, frame count, radio geometry, or repeat cadence.

set -euo pipefail

readonly REPO_ROOT="/home/pi/spf"
readonly SCRIPT_DIR="${REPO_ROOT}/data_collection/rover/rover_v3.1"
readonly PROFILE_ENV="/etc/spf/rover_collection.env"
readonly READY_FILE="/run/spf/direct_usb_ready"
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

CAPTURE_PROFILE="${SPF_CAPTURE_PROFILE:-legacy_iio_v4}"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
SKIP_SELF_UPDATE="${SPF_SKIP_SELF_UPDATE:-0}"
SKIP_PARAMETER_SYNC="${SPF_SKIP_PARAMETER_SYNC:-0}"
BOOT_VALIDATE_ONLY="${SPF_BOOT_VALIDATE_ONLY:-0}"
RUN_ONCE="${SPF_RUN_ONCE:-0}"
OUTPUT_ROOT="${SPF_OUTPUT_ROOT:-/home/pi/temp}"
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

mapfile -d '' -t profile_values < <(
    "$PYTHON" -m spf.scripts.rover_capture_profile \
        --profile "$CAPTURE_PROFILE" \
        --rover-id "$rover_id" \
        --format null
)
[[ "${#profile_values[@]}" -eq 6 ]] ||
    die "Capture profile resolver returned ${#profile_values[@]} fields, expected 6."
config="${profile_values[0]}"
routine="${profile_values[1]}"
records_per_receiver="${profile_values[2]}"
expected_radios="${profile_values[3]}"
rx_transport="${profile_values[4]}"
data_version="${profile_values[5]}"

if [[ -n "${SPF_RECORDS_PER_RECEIVER:-}" ]]; then
    records_per_receiver="$SPF_RECORDS_PER_RECEIVER"
fi
[[ "$records_per_receiver" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_RECORDS_PER_RECEIVER must be a positive integer."
[[ "$RADIO_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_RADIO_WAIT_SECONDS must be a positive integer."

print_plan() {
    printf '%s\n' \
        "rover_id=${rover_id}" \
        "capture_profile=${CAPTURE_PROFILE}" \
        "config=${config}" \
        "routine=${routine}" \
        "records_per_receiver=${records_per_receiver}" \
        "expected_radios=${expected_radios}" \
        "rx_transport=${rx_transport}" \
        "data_version=${data_version}" \
        "boot_validate_only=${BOOT_VALIDATE_ONLY}" \
        "run_once=${RUN_ONCE}" \
        "output_root=${OUTPUT_ROOT}"
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

maybe_self_update() {
    is_true "$SKIP_SELF_UPDATE" && return 0
    sleep 10
    if ! ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
        printf 'No internet connectivity; continuing with checked-out code.\n'
        return 0
    fi

    "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer git
    printf 'Checking for repository updates.\n'
    bash "${SCRIPT_DIR}/install_deps.sh"
    current_hash="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
    git -C "$REPO_ROOT" pull --ff-only
    new_hash="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
    if [[ "$current_hash" != "$new_hash" ]]; then
        printf 'Repository updated; installing current unit and rebooting.\n'
        sleep 15
        sudo install -m 0644 \
            "${SCRIPT_DIR}/spf-pluto-direct-usb.service" \
            /etc/systemd/system/spf-pluto-direct-usb.service
        sudo install -m 0644 \
            "${SCRIPT_DIR}/mavlink_controller.service" \
            /etc/systemd/system/mavlink_controller.service
        sudo systemctl daemon-reload
        sudo systemctl enable \
            spf-pluto-direct-usb.service mavlink_controller.service
        sudo reboot
        exit 0
    fi
    "$PYTHON" -m pip install -e "$REPO_ROOT"
}

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
        sleep 15
    done
    die "Timed out waiting for ${expected_radios} Pluto radios."
}

verify_direct_ready() {
    [[ -f "$READY_FILE" ]] ||
        die "Direct-USB preparation did not produce ${READY_FILE}."
    grep -qx "rover_id=${rover_id}" "$READY_FILE" ||
        die "Direct-USB ready stamp belongs to a different rover."
    grep -qx "expected_radios=${expected_radios}" "$READY_FILE" ||
        die "Direct-USB ready stamp has the wrong radio count."
    [[ -f "$DEVICE_MAPPING" ]] || die "Missing ${DEVICE_MAPPING}."
    mapping_rows="$(awk 'NF { count++ } END { print count + 0 }' "$DEVICE_MAPPING")"
    [[ "$mapping_rows" -eq "$expected_radios" ]] ||
        die "Device mapping has ${mapping_rows} rows, expected ${expected_radios}."
}

sync_vehicle_configuration() {
    is_true "$SKIP_PARAMETER_SYNC" && return 0
    sed "s/__ROVER_ID__/${rover_id}/g" \
        < <(cat \
            "${SCRIPT_DIR}/rover3_base_parameters.params" \
            "${SCRIPT_DIR}/rover3_rc_servo_parameters.params") \
        >"$PARAMS_FILE"
    "$PYTHON" "$MAVLINK_CONTROLLER" --load-params "$PARAMS_FILE"
    if ! "$PYTHON" "$MAVLINK_CONTROLLER" --diff-params "$PARAMS_FILE"; then
        die "Vehicle parameter verification failed after loading parameters."
    fi
}

sync_gps_time() {
    "$PYTHON" "$MAVLINK_CONTROLLER" --get-time "$TIME_FILE"
    sudo date -s "$(cat "$TIME_FILE")"
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
    mkdir -p "$OUTPUT_ROOT"
    "$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
        --yaml-config "$config" \
        --device-mapping "$DEVICE_MAPPING" \
        --routine "$routine" \
        --tag "RO${rover_id}" \
        --records-per-receiver "$records_per_receiver" \
        --temp "$OUTPUT_ROOT"
}

main() {
    print_plan
    maybe_self_update
    wait_for_radios

    if [[ "$rx_transport" == "direct_usb" ]]; then
        verify_direct_ready
    else
        printf 'Checking and configuring legacy IIO Pluto radios.\n'
        bash "${SCRIPT_DIR}/check_and_set_pluto.sh"
    fi

    if is_true "$BOOT_VALIDATE_ONLY"; then
        read_only_vehicle_gate
        printf 'PASS: boot validation stopped before parameter writes or collection.\n'
        return 0
    fi

    sync_vehicle_configuration
    sync_gps_time
    printf 'performance\n' | sudo tee \
        /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null

    while true; do
        run_capture
        is_true "$RUN_ONCE" && break
        sleep 8
        sync_gps_time
        sleep 2
    done
}

main
