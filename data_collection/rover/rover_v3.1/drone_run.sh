#!/usr/bin/env bash
#
# Rover 3.1 production boot launcher.
#
# The canonical Rover YAML is the source of truth for dataset schema, firmware,
# radio count, motion routine, frame count, geometry, and RF configuration.

set -euo pipefail

readonly REPO_ROOT="/home/pi/spf"
readonly SCRIPT_DIR="${REPO_ROOT}/data_collection/rover/rover_v3.1"
# Overridable so tests can point the profile at a tempdir instead of /etc,
# mirroring the SPF_ROVER_CONFIG seam the `rover` CLI already provides.
readonly PROFILE_ENV="${SPF_PROFILE_ENV:-/etc/spf/rover_collection.env}"
readonly READY_FILE="/run/spf/direct_usb_ready.json"
readonly DEVICE_MAPPING="/home/pi/device_mapping"
readonly MAVLINK_CONTROLLER="${REPO_ROOT}/spf/mavlink/mavlink_controller.py"
readonly PARAMS_FILE="/home/pi/this_rover.params"
readonly COMPASS_READY_FILE="/home/pi/compass_ready.json"
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
CAPTURE_WATCHDOG_FILE="${SPF_CAPTURE_WATCHDOG_FILE:-/home/pi/preflight/capture_watchdog.jsonl}"
CAPTURE_WATCHDOG_INTERVAL_SECONDS="${SPF_CAPTURE_WATCHDOG_INTERVAL_SECONDS:-1}"
CAPTURE_WATCHDOG_MAXIMUM_BYTES="${SPF_CAPTURE_WATCHDOG_MAXIMUM_BYTES:-16777216}"
CAPTURE_RESTART_ATTEMPTS="${SPF_CAPTURE_RESTART_ATTEMPTS:-1}"
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
[[ "$rover_id" =~ ^[1-4]$ ]] || die "Unsupported rover_id: ${rover_id}"

# Stall handling. Defaults live in rover_env_defaults.sh so this script and the
# `rover` CLI cannot disagree about them; both source that one file.
# shellcheck source=data_collection/rover/rover_v3.1/rover_env_defaults.sh
source "${SCRIPT_DIR}/rover_env_defaults.sh"
CRASH_DETECT="${SPF_CRASH_DETECT:-$(spf_default_crash_detect)}"
CRASH_RECOVERY="${SPF_CRASH_RECOVERY:-$(spf_default_crash_recovery "$rover_id")}"
ULTRASONIC="${SPF_ULTRASONIC:-$(spf_default_ultrasonic)}"
is_true "$CRASH_DETECT" && crash_detect_flag=--crash-detect ||
    crash_detect_flag=--no-crash-detect
is_true "$CRASH_RECOVERY" && crash_recovery_flag=--crash-recovery ||
    crash_recovery_flag=--no-crash-recovery
is_true "$ULTRASONIC" && ultrasonic_flag=--ultrasonic ||
    ultrasonic_flag=--no-ultrasonic

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
[[ "${#config_values[@]}" -eq 16 ]] ||
    die "Capture config resolver returned ${#config_values[@]} fields, expected 16."
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
[[ "$CAPTURE_RESTART_ATTEMPTS" =~ ^[0-9]+$ ]] ||
    die "SPF_CAPTURE_RESTART_ATTEMPTS must be a non-negative integer."
[[ "$CAPTURE_WATCHDOG_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_CAPTURE_WATCHDOG_INTERVAL_SECONDS must be a positive integer."
[[ "$CAPTURE_WATCHDOG_MAXIMUM_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_CAPTURE_WATCHDOG_MAXIMUM_BYTES must be a positive integer."

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
        "capture_status_file=${CAPTURE_STATUS_FILE}" \
        "capture_watchdog_file=${CAPTURE_WATCHDOG_FILE}" \
        "capture_restart_attempts=${CAPTURE_RESTART_ATTEMPTS}" \
        "crash_detect=${CRASH_DETECT}" \
        "ultrasonic=${ULTRASONIC}" \
        "crash_recovery=${CRASH_RECOVERY}" \
        "compass_gate_retries=${COMPASS_GATE_RETRIES}" \
        "compass_reboot_enable=${COMPASS_REBOOT_ENABLE}" \
        "compass_reboot_delay_s=${COMPASS_REBOOT_DELAY_S}"
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

revalidate_radios_after_capture_failure() {
    local mapping_candidate
    wait_for_radios
    mapping_candidate="$(mktemp /tmp/spf-device-mapping.XXXXXX)"
    if ! bash "${SCRIPT_DIR}/device_mapping.sh" >"$mapping_candidate"; then
        rm -f -- "$mapping_candidate"
        return 1
    fi
    # Replace the transient bus/address mapping only after complete discovery.
    mv -- "$mapping_candidate" "$DEVICE_MAPPING"

    if [[ "$rx_transport" == "direct_usb" ]]; then
        local manifest_args=(
            --rover-id "$rover_id"
            --output "$READY_FILE"
            --device-mapping "$DEVICE_MAPPING"
        )
        if [[ -n "${SPF_CAPTURE_CONFIG:-}" ]]; then
            manifest_args+=(--config "$SPF_CAPTURE_CONFIG")
        fi
        # This is a read-only identity/config provenance refresh: no firmware
        # flash and no IQ/TX operation. Refresh refuses to overwrite the prior
        # manifest unless serial, physical path, stable hardware identity,
        # config and firmware are unchanged; only transient attachment facts
        # such as USB address may change. The next process reapplies and
        # verifies the full RX configuration before creating a fresh Zarr.
        sudo -n "$PYTHON" -m spf.scripts.pluto_ready_manifest \
            refresh "${manifest_args[@]}" >/dev/null
        sudo -n "$PYTHON" -m spf.scripts.pluto_ready_manifest \
            verify "${manifest_args[@]}" >/dev/null
    else
        sudo -n bash "${SCRIPT_DIR}/load_direct_usb_firmware.sh" \
            check-config-all "$expected_radios"
    fi
}

ensure_vehicle_hold_after_capture_failure() {
    # The failed collector tries HOLD for two seconds before its bounded hard
    # exit. Reconnect independently and verify HOLD before any radio work or
    # new artifact, covering a MAVLink outage that outlasted that deadline.
    "$PYTHON" "$MAVLINK_CONTROLLER" \
        --mode HOLD \
        --connect-attempts 3 \
        --heartbeat-timeout 3
}

notify_capture_failure() {
    for _attempt in 1 2 3; do
        "$PYTHON" "$MAVLINK_CONTROLLER" --buzzer failure || true
        sleep 1
    done
}

# The external IST8310 on the GPS mast intermittently fails to appear on the
# flight controller's I2C bus. ArduPilot probes compasses once at boot and never
# rescans, so the two recoveries available are: re-read the parameters (the gate
# reads COMPASS_DEV_ID*, which the FC populates during that probe, so a check run
# too soon after an FC restart can report a healthy compass as absent), and
# reboot the flight controller for a fresh probe.
#
# Only an absence is retried. `retryable` in the policy JSON is true when EVERY
# error is "the external compass is not on the bus"; a misconfiguration reads
# identically after any number of reboots, so retrying it would bury a real
# defect behind a delay. Fail closed on anything else.
readonly COMPASS_GATE_RETRIES="${SPF_COMPASS_GATE_RETRIES:-3}"

compass_gate_failure_is_retryable() {
    [[ -s "$COMPASS_READY_FILE" ]] || return 1
    "$PYTHON" - "$COMPASS_READY_FILE" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as handle:
        report = json.load(handle)
except Exception:
    sys.exit(1)          # unreadable verdict is not a licence to retry
sys.exit(0 if report.get("retryable") is True else 1)
PYEOF
}

# Randomised so a retry cannot land in lockstep with whatever periodic or
# thermal condition caused the miss, and so four rovers restarting together do
# not retry in phase. Delay grows with the retry number.
compass_gate_backoff_seconds() {
    local retry="$1"
    case "$retry" in
        1) printf '%s\n' "$(( 5 + RANDOM % 6 ))" ;;    #  5-10s, re-read only
        2) printf '%s\n' "$(( 15 + RANDOM % 11 ))" ;;  # 15-25s, after a reboot
        *) printf '%s\n' "$(( 30 + RANDOM % 16 ))" ;;  # 30-45s, after a reboot
    esac
}

# Retry 1 settles and re-reads only; every later retry reboots the FC first.
# The cheap one goes first because the gate reads COMPASS_DEV_ID*, which the FC
# writes during its boot probe -- a check run soon after an FC restart can call
# a healthy compass absent, and ruling that out costs seconds instead of ~40s.
compass_gate_retry_reboots_fc() {
    [[ "$1" -ge 2 ]]
}

# Reboot via ardu_cli, not MAVLINK_CONTROLLER --reboot: ardu_cli refuses to
# reboot an armed vehicle and its reconnect loop tolerates the fmuv3 enumerating
# USB twice. --allow-active-service is correct here and only here: this script
# IS mavlink_controller.service, and no capture holds the port during boot sync.
reboot_flight_controller_for_compass() {
    "$PYTHON" "${REPO_ROOT}/spf/ardupilot/ardu_cli.py" reboot \
        --yes --allow-active-service
}

run_compass_gate() {
    # "$@" is the mode-specific mavlink_controller invocation.
    rm -f -- "$COMPASS_READY_FILE"
    "$PYTHON" "$MAVLINK_CONTROLLER" "$@" \
        --compass-policy-json "$COMPASS_READY_FILE"
}

# One initial check, then COMPASS_GATE_RETRIES recovery attempts:
#   retry 1 -> settle  5-10s, re-read parameters only (no reboot)
#   retry 2 -> FC reboot, then 15-25s
#   retry 3 -> FC reboot, then 30-45s
# Returns: 0 pass
#          1 failed for a reason no reboot can fix -- caller must die
#          2 external compass absent and every FC-level recovery is spent --
#            caller may escalate to a full rover reboot
compass_gate_with_retries() {
    local what="$1"; shift
    local retry delay
    for (( retry = 0; retry <= COMPASS_GATE_RETRIES; retry++ )); do
        if [[ "$retry" -gt 0 ]]; then
            delay="$(compass_gate_backoff_seconds "$retry")"
            if compass_gate_retry_reboots_fc "$retry"; then
                printf 'Compass gate (%s): external compass still absent; rebooting the flight controller, then waiting %ss (retry %s/%s).\n' \
                    "$what" "$delay" "$retry" "$COMPASS_GATE_RETRIES"
                if ! reboot_flight_controller_for_compass; then
                    printf 'Compass gate: flight-controller reboot failed; not retrying.\n' >&2
                    return 1
                fi
            else
                printf 'Compass gate (%s): external compass absent; settling %ss and re-reading parameters (retry %s/%s).\n' \
                    "$what" "$delay" "$retry" "$COMPASS_GATE_RETRIES"
            fi
            sleep "$delay"
        fi

        if run_compass_gate "$@"; then
            if [[ "$retry" -gt 0 ]]; then
                printf 'Compass gate passed on retry %s of %s.\n' \
                    "$retry" "$COMPASS_GATE_RETRIES"
            fi
            clear_compass_reboot_count
            return 0
        fi
        if ! compass_gate_failure_is_retryable; then
            printf 'Compass gate failed for a reason a reboot cannot fix; not retrying.\n' >&2
            return 1
        fi
    done
    printf 'Compass gate: external compass still absent after %s retries.\n' \
        "$COMPASS_GATE_RETRIES" >&2
    return 2
}

# How many times THIS rover has rebooted itself chasing an absent compass.
# Persisted outside /run so it survives the reboot it is counting; cleared the
# moment the gate passes, so the number always describes one unbroken episode.
readonly COMPASS_REBOOT_COUNT_FILE="/home/pi/compass_reboot_attempts"
readonly COMPASS_REBOOT_DELAY_S="${SPF_COMPASS_REBOOT_DELAY_S:-20}"
# The escalation is deliberately unbounded, so it needs an off switch that does
# not require a code change or catching the 20s window. Set
# SPF_COMPASS_REBOOT_ENABLE=0 in /etc/spf/rover_collection.env to park a rover
# with a known-bad compass lead: it then fails closed like before instead of
# cycling. Per-rover, survives reboots, and is visible in the boot plan.
readonly COMPASS_REBOOT_ENABLE="${SPF_COMPASS_REBOOT_ENABLE:-1}"

clear_compass_reboot_count() {
    rm -f -- "$COMPASS_REBOOT_COUNT_FILE"
}

# Escalation of last resort for an ABSENT external compass: a full rover reboot
# power-cycles more of the system than an FC reset and re-runs the whole boot
# sequence. Deliberately unbounded -- a rover that can recover on the ninth boot
# is worth more than one parked overnight -- but every cycle announces itself and
# then waits, so an operator watching the journal can SSH in or cut power and
# stop the loop. Raise SPF_COMPASS_REBOOT_DELAY_S for a wider window.
#
# Never reached for a misconfiguration: compass_gate_with_retries returns 1 for
# those and the caller dies instead.
reboot_rover_for_absent_compass() {
    if ! is_true "$COMPASS_REBOOT_ENABLE"; then
        printf 'Compass gate: external compass absent, but rover reboot is disabled\n' >&2
        printf '  (SPF_COMPASS_REBOOT_ENABLE=0). Failing closed instead of cycling.\n' >&2
        die "Compass policy verification failed; refusing collection and motion."
    fi
    local count=0
    [[ -r "$COMPASS_REBOOT_COUNT_FILE" ]] && \
        count="$(tr -cd '0-9' <"$COMPASS_REBOOT_COUNT_FILE")"
    count="$(( ${count:-0} + 1 ))"
    printf '%s\n' "$count" >"$COMPASS_REBOOT_COUNT_FILE" 2>/dev/null || true

    printf '\n'
    printf '=== COMPASS ABSENT: REBOOTING THE ROVER (reboot #%s for this fault) ===\n' "$count" >&2
    printf 'The external GPS compass did not appear after %s flight-controller retries.\n' \
        "$COMPASS_GATE_RETRIES" >&2
    printf 'If this keeps repeating the compass is not coming back on its own:\n' >&2
    printf '  reseat the GPS/compass connector at BOTH ends, and check the mast lead.\n' >&2
    printf 'Rebooting in %ss -- Ctrl-C, `systemctl stop mavlink_controller`, or power off to intervene.\n' \
        "$COMPASS_REBOOT_DELAY_S" >&2
    sleep "$COMPASS_REBOOT_DELAY_S"
    printf 'Rebooting now.\n' >&2
    sudo -n systemctl reboot || sudo -n reboot
    # systemctl reboot returns immediately; block so nothing downstream runs
    # against a vehicle whose yaw source was never verified.
    sleep 300
    exit 1
}

sync_vehicle_configuration() {
    if is_true "$SKIP_PARAMETER_SYNC"; then
        verify_compass_policy_read_only
        return 0
    fi
    sed "s/__ROVER_ID__/${rover_id}/g" \
        < <(cat \
            "${SCRIPT_DIR}/rover3_base_parameters.params" \
            "${SCRIPT_DIR}/rover3_rc_servo_parameters.params") \
        >"$PARAMS_FILE"
    # A normal boot uses one complete download for managed-parameter verification,
    # compass inventory logging, and policy. Changed parameters get one additional
    # full readback after their acknowledged writes.
    # `|| status=$?` and not a bare call: under `set -e` a bare command
    # returning 2 would exit the script before the case could read it.
    local status=0
    compass_gate_with_retries "parameter sync" \
        --prepare-vehicle-params "$PARAMS_FILE" || status="$?"
    case "$status" in
        0) return 0 ;;
        2) reboot_rover_for_absent_compass ;;  # does not return
        *) die "Vehicle parameter or compass policy verification failed." ;;
    esac
}

verify_compass_policy_read_only() {
    local status=0
    compass_gate_with_retries "read-only" --check-compass-policy || status="$?"
    if [[ "$status" -eq 0 ]]; then
        return 0
    fi
    # BOOT_VALIDATE_ONLY exists to inspect a rover without side effects, so it
    # reports and stops. Every other caller is a real boot and may escalate.
    if [[ "$status" -eq 2 ]] && ! is_true "$BOOT_VALIDATE_ONLY"; then
        reboot_rover_for_absent_compass  # does not return
    fi
    die "Compass policy verification failed; refusing collection and motion."
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
    local capture_pid capture_status watchdog_pid watchdog_status
    mkdir -p "$OUTPUT_ROOT" "$(dirname "$CAPTURE_STATUS_FILE")"
    set +e
    "$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
        --yaml-config "$config" \
        --device-mapping "$DEVICE_MAPPING" \
        --tag "RO${rover_id}" \
        --temp "$OUTPUT_ROOT" \
        --status-file "$CAPTURE_STATUS_FILE" \
        "$crash_detect_flag" \
        "$crash_recovery_flag" \
        "$ultrasonic_flag" &
    capture_pid=$!
    "$PYTHON" -m spf.capture_watchdog monitor \
        --pid "$capture_pid" \
        --status-file "$CAPTURE_STATUS_FILE" \
        --storage-path "$OUTPUT_ROOT" \
        --output "$CAPTURE_WATCHDOG_FILE" \
        --expected-plutos "$expected_radios" \
        --interval-seconds "$CAPTURE_WATCHDOG_INTERVAL_SECONDS" \
        --maximum-bytes "$CAPTURE_WATCHDOG_MAXIMUM_BYTES" &
    watchdog_pid=$!
    wait "$capture_pid"
    capture_status=$?
    wait "$watchdog_pid"
    watchdog_status=$?
    set -e
    if [[ "$watchdog_status" -ne 0 ]]; then
        printf 'WARNING: capture watchdog exited with status %s.\n' \
            "$watchdog_status" >&2
    fi
    if [[ "$capture_status" -ne 0 ]]; then
        "$PYTHON" -m spf.capture_status mark-failed \
            --path "$CAPTURE_STATUS_FILE" \
            --exit-code "$capture_status" || true
        return "$capture_status"
    fi
    return 0
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
        verify_compass_policy_read_only
        printf 'PASS: boot validation stopped before parameter writes or collection.\n'
        return 0
    fi

    sync_vehicle_configuration
    sync_gps_time boot
    printf 'performance\n' | sudo tee \
        /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null

    consecutive_capture_failures=0
    while true; do
        if run_capture; then
            consecutive_capture_failures=0
            is_true "$RUN_ONCE" && break
            sleep 8
            sync_gps_time capture
            sleep 2
            continue
        else
            capture_status=$?
        fi

        consecutive_capture_failures=$((consecutive_capture_failures + 1))
        if ! ensure_vehicle_hold_after_capture_failure; then
            notify_capture_failure
            die "Capture failed and vehicle HOLD could not be confirmed."
        fi
        notify_capture_failure
        if is_true "$RUN_ONCE" ||
            (( consecutive_capture_failures > CAPTURE_RESTART_ATTEMPTS )); then
            printf 'Capture failed %s consecutive time(s); no automatic restart.\n' \
                "$consecutive_capture_failures" >&2
            return "$capture_status"
        fi
        printf 'Capture failed; re-attesting radios before a new artifact (%s/%s).\n' \
            "$consecutive_capture_failures" "$CAPTURE_RESTART_ATTEMPTS" >&2
        if ! revalidate_radios_after_capture_failure; then
            die "Radio re-attestation failed after capture incident."
        fi
        sleep 2
    done
}

main
