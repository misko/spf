#!/usr/bin/env bash
#
# Rover 3.1 production boot launcher.
#
# The canonical Rover YAML is the source of truth for dataset schema, firmware,
# radio count, motion routine, frame count, geometry, and RF configuration.

set -euo pipefail

# Overridable for the same reason PROFILE_ENV and ROVER_ID_FILE are: the script
# sources its own siblings through SCRIPT_DIR, so with this hardcoded it can only
# run on a rover. test_boot_launcher_prints_canonical_v7_plan_without_hardware
# invokes `--print-plan` from a checkout and died on
#   line 110: /home/pi/spf/.../rover_env_defaults.sh: No such file or directory
# which left CI red for 20+ consecutive runs -- long enough that it masked a real
# unbound-variable defect in print_plan that reached every rover (37f2a00).
# Same seam update_spf_before_boot.sh already uses as SPF_UPDATE_REPO_ROOT.
readonly REPO_ROOT="${SPF_REPO_ROOT:-/home/pi/spf}"
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
# SPF_BOOT_VALIDATE_ONLY and SPF_RUN_ONCE are REMOVED, not merely unread.
#
# BOOT_VALIDATE_ONLY was a weaker, in-band duplicate of what
# `configure_direct_usb_boot.sh qualify` already does properly: qualify
# systemctl-disables the motion-capable unit, so the mission loop cannot start
# at all, rather than relying on an env var read from inside it. RUN_ONCE was a
# bench convenience with no runbook procedure attached.
#
# Fail closed rather than ignore: a rover whose /etc/spf/rover_collection.env
# still says SPF_BOOT_VALIDATE_ONLY=1 was deliberately parked so it could not
# move. Silently ignoring that line would put it into the motion-capable mission
# loop on its next boot -- and rovers pull origin/main at boot, so it would
# happen unattended. Refuse to start until a human removes the line.
# Refuse ONLY if the flag was actually engaged. `=0` is the harmless production
# default and is present on every rover today -- dying on that would take the
# whole fleet down at the next boot for no safety benefit.
if [[ -n "${SPF_BOOT_VALIDATE_ONLY:-}" ]]; then
    if is_true "${SPF_BOOT_VALIDATE_ONLY}"; then
        die "SPF_BOOT_VALIDATE_ONLY=1 is set, but the flag is removed. This rover
  was parked so it could not move; starting now would run the MOTION-CAPABLE
  mission loop. Use 'configure_direct_usb_boot.sh qualify' to disable the
  mission unit instead, then delete the line from ${PROFILE_ENV}."
    fi
    printf 'NOTE: SPF_BOOT_VALIDATE_ONLY is removed and ignored; delete it from %s.\n' \
        "$PROFILE_ENV" >&2
fi
if [[ -n "${SPF_RUN_ONCE:-}" ]]; then
    if is_true "${SPF_RUN_ONCE}"; then
        die "SPF_RUN_ONCE=1 is set, but the flag is removed and the launcher now
  always runs the mission loop. Delete the line from ${PROFILE_ENV}."
    fi
    printf 'NOTE: SPF_RUN_ONCE is removed and ignored; delete it from %s.\n' \
        "$PROFILE_ENV" >&2
fi
OUTPUT_ROOT="${SPF_OUTPUT_ROOT:-/home/pi/temp}"
CAPTURE_STATUS_FILE="${SPF_CAPTURE_STATUS_FILE:-/home/pi/preflight/capture_status.json}"
CAPTURE_WATCHDOG_FILE="${SPF_CAPTURE_WATCHDOG_FILE:-/home/pi/preflight/capture_watchdog.jsonl}"
CAPTURE_WATCHDOG_INTERVAL_SECONDS="${SPF_CAPTURE_WATCHDOG_INTERVAL_SECONDS:-1}"
CAPTURE_WATCHDOG_MAXIMUM_BYTES="${SPF_CAPTURE_WATCHDOG_MAXIMUM_BYTES:-16777216}"
CAPTURE_RESTART_ATTEMPTS="${SPF_CAPTURE_RESTART_ATTEMPTS:-1}"
RADIO_WAIT_SECONDS="${SPF_RADIO_WAIT_SECONDS:-600}"
ROVER_ID_FILE="${SPF_ROVER_ID_FILE:-/home/pi/rover_id}"

# Compass-gate and GPS-clock tunables live here, with every other env-derived
# knob, because print_plan() reads them and `--print-plan` exits long before the
# functions that use them. Defining them next to those functions made
# `--print-plan` die on an unbound variable under `set -u`.
COMPASS_GATE_RETRIES="${SPF_COMPASS_GATE_RETRIES:-3}"
COMPASS_REBOOT_COUNT_FILE="/home/pi/compass_reboot_attempts"
COMPASS_REBOOT_DELAY_S="${SPF_COMPASS_REBOOT_DELAY_S:-20}"
COMPASS_REBOOT_ENABLE="${SPF_COMPASS_REBOOT_ENABLE:-1}"
GPS_TIME_SYNC_ATTEMPTS="${SPF_GPS_TIME_SYNC_ATTEMPTS:-3}"
GPS_TIME_TIMEOUT_S="${SPF_GPS_TIME_TIMEOUT:-180}"
# Stall watchdog tuning. Empty means "whatever the collector's own defaults
# are", so the numbers keep exactly one home; setting a knob passes the flag
# through explicitly. mavlink_radio_collection.py has accepted these three
# flags all along, but run_capture() built a fixed argument list that omitted
# them -- so retuning the watchdog in the field meant editing this script on
# four rovers, and the 0.3 m/s implied speed floor was not adjustable at all.
STALL_DETECT_SECONDS="${SPF_STALL_DETECT_SECONDS:-}"
STALL_MANUAL_SECONDS="${SPF_STALL_MANUAL_SECONDS:-}"
STALL_PROGRESS_RADIUS_M="${SPF_STALL_PROGRESS_RADIUS_M:-}"
# Warn by default; see vehicle_arm_state_gate for why this is not fatal yet.
REQUIRE_DISARMED_FOR_PARAM_SYNC="${SPF_REQUIRE_DISARMED_FOR_PARAM_SYNC:-0}"

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
# Either of these can silently reinstate the defect just removed, from one line
# in /etc/spf/rover_collection.env with no code change: ATTEMPTS=0 makes the
# retry loop run zero times, and GNU `timeout 0` DISABLES the timeout rather
# than meaning "do not wait".
[[ "$GPS_TIME_SYNC_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_GPS_TIME_SYNC_ATTEMPTS must be a positive integer, got: ${GPS_TIME_SYNC_ATTEMPTS}"
[[ "$GPS_TIME_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_GPS_TIME_TIMEOUT must be a positive integer of seconds, got: ${GPS_TIME_TIMEOUT_S}"
[[ "$COMPASS_GATE_RETRIES" =~ ^[0-9]+$ ]] ||
    die "SPF_COMPASS_GATE_RETRIES must be a non-negative integer."
[[ "$COMPASS_REBOOT_DELAY_S" =~ ^[0-9]+$ ]] ||
    die "SPF_COMPASS_REBOOT_DELAY_S must be a non-negative integer."
# Positive decimals, and only when set. A stall knob at 0 is worse than absent:
# `--stall-detect-seconds 0` makes every waypoint stall instantly and the rover
# spends the session escaping jams that are not there.
for stall_knob in \
    "SPF_STALL_DETECT_SECONDS:${STALL_DETECT_SECONDS}" \
    "SPF_STALL_MANUAL_SECONDS:${STALL_MANUAL_SECONDS}" \
    "SPF_STALL_PROGRESS_RADIUS_M:${STALL_PROGRESS_RADIUS_M}"; do
    stall_knob_value="${stall_knob#*:}"
    # `if`, not `[[ ... ]] && continue`: under `set -e` that compound returns
    # nonzero for every knob that IS set, and aborts the boot.
    if [[ -n "$stall_knob_value" ]]; then
        # A decimal, with at least one nonzero digit -- which rules out 0, 0.0
        # and 0.00 without shelling out to arithmetic.
        [[ "$stall_knob_value" =~ ^[0-9]+(\.[0-9]+)?$ &&
            "$stall_knob_value" =~ [1-9] ]] ||
            die "${stall_knob%%:*} must be a positive number, got: ${stall_knob_value}"
    fi
done
unset stall_knob stall_knob_value

stall_args=()
if [[ -n "$STALL_DETECT_SECONDS" ]]; then
    stall_args+=(--stall-detect-seconds "$STALL_DETECT_SECONDS")
fi
if [[ -n "$STALL_MANUAL_SECONDS" ]]; then
    stall_args+=(--stall-manual-seconds "$STALL_MANUAL_SECONDS")
fi
if [[ -n "$STALL_PROGRESS_RADIUS_M" ]]; then
    stall_args+=(--stall-progress-radius-m "$STALL_PROGRESS_RADIUS_M")
fi

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
        "output_root=${OUTPUT_ROOT}" \
        "capture_status_file=${CAPTURE_STATUS_FILE}" \
        "capture_watchdog_file=${CAPTURE_WATCHDOG_FILE}" \
        "capture_restart_attempts=${CAPTURE_RESTART_ATTEMPTS}" \
        "crash_detect=${CRASH_DETECT}" \
        "ultrasonic=${ULTRASONIC}" \
        "crash_recovery=${CRASH_RECOVERY}" \
        "compass_gate_retries=${COMPASS_GATE_RETRIES}" \
        "compass_reboot_enable=${COMPASS_REBOOT_ENABLE}" \
        "compass_reboot_delay_s=${COMPASS_REBOOT_DELAY_S}" \
        "gps_time_sync_attempts=${GPS_TIME_SYNC_ATTEMPTS}" \
        "gps_time_timeout_s=${GPS_TIME_TIMEOUT_S}" \
        "stall_detect_seconds=${STALL_DETECT_SECONDS:-<collector default>}" \
        "stall_manual_seconds=${STALL_MANUAL_SECONDS:-<collector default>}" \
        "stall_progress_radius_m=${STALL_PROGRESS_RADIUS_M:-<collector default>}" \
        "require_disarmed_for_param_sync=${REQUIRE_DISARMED_FOR_PARAM_SYNC}"
}

case "${1:-}" in
    --print-plan)
        [[ "$#" -eq 1 ]] || die "--print-plan takes no arguments."
        print_plan
        exit 0
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
# The escalation is deliberately unbounded, so it needs an off switch that does
# not require a code change or catching the 20s window. Set
# SPF_COMPASS_REBOOT_ENABLE=0 in /etc/spf/rover_collection.env to park a rover
# with a known-bad compass lead: it then fails closed like before instead of
# cycling. Per-rover, survives reboots, and is visible in the boot plan.

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
    vehicle_arm_state_gate
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
    if [[ "$status" -eq 2 ]]; then
        reboot_rover_for_absent_compass  # does not return
    fi
    die "Compass policy verification failed; refusing collection and motion."
}

# EVERY boot sets the clock from GPS before anything is named.
#
# There is no battery-backed RTC on this Pi, so a network-less field boot
# restores a timestamp saved at the last shutdown -- journald says "System clock
# time unset or jumped backwards, restoring from recorded timestamp". The
# capture filename is stamped from that clock at process start
# (mavlink_radio_collection.py:254), MINUTES before the first sample is
# recorded, so a stale clock names the artifact wrongly and nothing downstream
# notices until the merge.
#
# 19 of 47 finalised campaign captures were misdated this way, by up to 4h28m.
# The cause was a former `system_clock_is_plausible` guard that skipped the boot
# sync whenever the clock read later than 2025-01-01 -- which a clock restored
# to four hours ago clears by nineteen months. It is deliberately gone.
#
# Blocking here costs nothing real: the capture already waits for the planner to
# take control, which requires a GPS fix, so GPS UTC is available before any
# data is written. The name was simply being stamped too early.
# Both are validated, because either can silently reinstate the exact defect
# that was just removed, from an /etc/spf/rover_collection.env line with no code
# change: ATTEMPTS=0 makes the retry loop run zero times, and GNU `timeout 0`
# DISABLES the timeout rather than meaning "do not wait".

gps_time_sync_once() {
    # --get-time blocks until the FC reports GPS UTC. Bounded so a no-sky /
    # cold-TTFF boot cannot hang forever. Time comes from GPS via MAVLink; the
    # rover has no internet and does not use NTP in the field.
    local gps_time
    if ! timeout "$GPS_TIME_TIMEOUT_S" \
        "$PYTHON" "$MAVLINK_CONTROLLER" --get-time "$TIME_FILE"; then
        return 1
    fi
    gps_time="$(cat "$TIME_FILE")"

    # Validate as an EPOCH, never by string prefix. --get-time writes naive
    # local time, so epoch 0 renders "1970-01-01 01:00:00" in Europe/London but
    # "1969-12-31 16:00:00" west of UTC -- a "1970-*" test misses it there and
    # the clock really does get set to epoch 0. An epoch floor is
    # timezone-independent.
    local epoch
    # Explicit, because `date -d ""` does NOT fail: it parses to today at
    # 00:00:00 local, which sails past any sanity floor while being up to 24h
    # wrong. An empty file is what a timeout-killed attempt used to leave.
    if [[ -z "${gps_time//[[:space:]]/}" ]]; then
        printf 'GPS time file is empty; not setting clock.\n'
        return 1
    fi
    if ! epoch="$(date -d "$gps_time" +%s 2>/dev/null)" || [[ -z "$epoch" ]]; then
        printf 'GPS time %q is unparseable; not setting clock.\n' "$gps_time"
        return 1
    fi
    # A GPS-derived UTC before 2025 is a fix-without-UTC, not a real time.
    if [[ "$epoch" -lt 1735689600 ]]; then
        printf 'GPS reported %s (epoch %s) — no UTC yet; not setting clock.\n' \
            "$gps_time" "$epoch"
        return 1
    fi
    # @epoch, so the value we validated is the value we set, independent of the
    # zone. sudo -n like every other privileged call here: a prompting sudo
    # would hang the boot.
    sudo -n date -s "@$epoch"
}

# Set by the first successful GPS sync of this boot. Until it is 1, no capture
# should be named: the filename is stamped from the wall clock when the capture
# process starts, so an unverified clock produces a misdated artifact.
CLOCK_VERIFIED_FROM_GPS=0

# "boot" and "pre-capture" are naming-critical -- a wrong clock at either point
# misdates an artifact -- so they get the full retry budget. The post-capture
# sync is opportunistic housekeeping and gets one try.
gps_time_sync_is_naming_critical() {
    [[ "$1" == "boot" || "$1" == "pre-capture" ]]
}

sync_gps_time() {
    local phase="${1:-capture}"
    local attempts=1 attempt
    if gps_time_sync_is_naming_critical "$phase"; then
        attempts="$GPS_TIME_SYNC_ATTEMPTS"
    fi
    for (( attempt = 1; attempt <= attempts; attempt++ )); do
        if gps_time_sync_once; then
            CLOCK_VERIFIED_FROM_GPS=1
            printf 'System clock set from GPS UTC (%s, attempt %s/%s).\n' \
                "$phase" "$attempt" "$attempts"
            return 0
        fi
        if [[ "$attempt" -lt "$attempts" ]]; then
            printf 'No GPS UTC yet (%s, attempt %s/%s); retrying.\n' \
                "$phase" "$attempt" "$attempts"
        fi
    done
    if gps_time_sync_is_naming_critical "$phase"; then
        printf '\n' >&2
        printf 'WARNING: no GPS UTC after %s attempts of up to %ss each (%s).\n' \
            "$attempts" "$GPS_TIME_TIMEOUT_S" "$phase" >&2
        printf '  The system clock is UNVERIFIED and may be hours stale (no RTC).\n' >&2
        printf '  Any capture named before the next successful sync carries a WRONG\n' >&2
        printf '  timestamp -- order and pair on gps_timestamp, never on the filename.\n' >&2
    else
        printf 'No GPS time within %ss; continuing (capture loop keeps retrying --get-time).\n' \
            "$GPS_TIME_TIMEOUT_S"
    fi
    return 1
}

# The capture filename is stamped from the wall clock when
# mavlink_radio_collection.py starts, so the clock must be correct BEFORE
# run_capture, not after it. Syncing only afterwards is what left exactly one
# capture per session misnamed whenever the boot sync had failed.
#
# Costs nothing in the normal case: once the clock is verified this returns
# immediately, and when it is not, the capture was about to block waiting for a
# GPS fix anyway -- the mission cannot start without one.
ensure_clock_verified_before_capture() {
    if [[ "$CLOCK_VERIFIED_FROM_GPS" -eq 1 ]]; then
        return 0
    fi
    if sync_gps_time pre-capture; then
        return 0
    fi
    printf 'WARNING: starting a capture with an UNVERIFIED clock; its FILENAME may\n' >&2
    printf '  carry the wrong time. gps_timestamp inside the store is still correct.\n' >&2
    return 0
}

# An armed-state check before parameter sync.
#
# NOT, as the removed read_only_vehicle_gate's comment claimed, the last line of
# defence: prepare_vehicle_parameters() has refused writes to an armed vehicle
# since 5ec2c0e (2026-08-01) and survived the SPF_BOOT_VALIDATE_ONLY removal
# untouched. That refusal raises, and sync_vehicle_configuration turns it into a
# die. So an armed rover whose parameters DIFFER already stops the boot today.
#
# What was actually unreported is the narrower case: an armed rover whose
# parameters already MATCH. No write happens, so nothing is unsafe and nothing
# said anything -- the rover simply drove off with its arm switch already on and
# no record of it. This warns about that, and gives the armed-and-differing case
# a diagnosis by name instead of a generic verification failure.
#
# It WARNS rather than dying, deliberately. The dangerous case -- an armed
# vehicle about to be WRITTEN to -- is already fatal one layer down, so dying
# here would only add a new way to strand a rover for the harmless case, on an
# operator error the boot path has tolerated silently for its whole life. That
# trade should be made on evidence: the journal is correctly stamped now (the
# clock sync moved above the compass gate), so how often this fires is about to
# become knowable. Flip SPF_REQUIRE_DISARMED_FOR_PARAM_SYNC=1 once it is.
#
# Failing to READ the status is not treated as armed. A missing heartbeat here
# is a different fault with its own handling downstream, and inferring "armed"
# from silence would strand rovers for the one reason this is trying to avoid.
vehicle_arm_state_gate() {
    local status_file state
    status_file="$(mktemp /tmp/spf-mavlink-status.XXXXXX)"
    trap 'rm -f -- "${status_file:-}"' RETURN
    if ! "$PYTHON" "$MAVLINK_CONTROLLER" --status-json "$status_file"; then
        printf 'WARNING: no MAVLink status before parameter sync; arm state unknown.\n' >&2
        return 0
    fi
    if ! state="$("$PYTHON" -c \
        'import json,sys; s=json.load(open(sys.argv[1])); '\
'print(("ARMED" if s["armed"] else "DISARMED")+" mode="+str(s["mav_mode"]))' \
        "$status_file")"; then
        printf 'WARNING: could not parse vehicle status before parameter sync.\n' >&2
        return 0
    fi
    if [[ "$state" == ARMED* ]]; then
        if is_true "$REQUIRE_DISARMED_FOR_PARAM_SYNC"; then
            die "Vehicle is ARMED before parameter sync (${state}); disarm and reboot."
        fi
        printf 'WARNING: vehicle is ARMED before parameter sync (%s).\n' "$state" >&2
        printf '  Parameters are about to be written to an armed rover; check the RC\n' >&2
        printf '  arm switch. SPF_REQUIRE_DISARMED_FOR_PARAM_SYNC=1 makes this fatal.\n' >&2
        return 0
    fi
    printf 'PASS: real MAVLink heartbeat received; vehicle is disarmed; %s\n' "$state"
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
        "$ultrasonic_flag" \
        ${stall_args[@]+"${stall_args[@]}"} &
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

    # BEFORE the compass gate, not after it. reboot_rover_for_absent_compass
    # never returns, so a rover cycling on an absent compass used to reboot
    # without ever reaching the clock sync -- every journal line from every
    # iteration stamped with the stale Pi clock, and that journal is exactly
    # what the loop has to be diagnosed from.
    #
    # Nothing here depends on parameter sync: --get-time needs only a heartbeat
    # and a GPS fix. Ordinary boots take the same total time either way, since
    # both steps run regardless. A no-sky boot now spends its GPS timeout
    # before the gate rather than after, which is the price of a correctly
    # stamped log.
    #
    # Guarded, not bare: sync_gps_time now reports failure, and under `set -e` a
    # bare call would abort the boot. An unverified clock is loud but not fatal
    # -- the capture still cannot start until the planner has GPS anyway, and
    # ensure_clock_verified_before_capture still gates the naming.
    sync_gps_time boot || printf 'Continuing with an unverified system clock.\n' >&2
    sync_vehicle_configuration
    printf 'performance\n' | sudo tee \
        /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null

    consecutive_capture_failures=0
    while true; do
        ensure_clock_verified_before_capture
        if run_capture; then
            consecutive_capture_failures=0
            sleep 8
            # Opportunistic: keeps a long session's clock fresh. The naming
            # guarantee comes from ensure_clock_verified_before_capture above.
            sync_gps_time capture || true
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
        if (( consecutive_capture_failures > CAPTURE_RESTART_ATTEMPTS )); then
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
