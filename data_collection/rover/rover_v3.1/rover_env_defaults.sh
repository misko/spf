# shellcheck shell=bash
#
# Defaults for the capture-service settings held in
# /etc/spf/rover_collection.env -- the file systemd hands to every unit via
# EnvironmentFile=, and the only place a setting can reach a running capture.
#
# This file exists so the `rover` CLI and drone_run.sh cannot disagree about
# what a setting defaults to. A per-rover default computed independently in two
# scripts is exactly the drift that produced the ARMING_CHECK divergence fixed
# in b4fa14a: the value looked right wherever you asked, and was wrong on the
# vehicle.
#
# It is SOURCED, so it must contain only function and constant definitions --
# no side effects, no output, no `set` changes.

# Settings that reach the capture services, in display order. `rover env` and
# `rover config` both read this, so a new knob is added here once.
SPF_CAPTURE_ENV_KEYS=(SPF_CRASH_DETECT SPF_CRASH_RECOVERY SPF_ULTRASONIC)

# Stall detection is safe everywhere: its worst outcome is handing a working
# rover to its operator. On by default across the fleet.
spf_default_crash_detect() {
    printf '1'
}

# Crash RECOVERY drives the rover autonomously -- reversing out of a jam and
# stepping off the axis it backed out along. That is a real motion risk, so it
# is opt-in per rover and enabled only where it has been exercised.
#
# Hand-set to rover 4 for now. Revisit once the fleet has field evidence; the
# expectation is that this becomes `printf '1'` for everyone, at which point
# this function collapses like spf_default_crash_detect above.
spf_default_crash_recovery() {
    local rover_id="${1:-}"
    if [[ "$rover_id" == "4" ]]; then printf '1'; else printf '0'; fi
}

# The ultrasonic obstacle stop. On by default -- it exists to stop the rover
# hitting things. Taranis CH12 toggles it live, but that path needs a working
# RC link and a receiver the flight controller can hear; when the switch does
# not reach the Pi there was previously no way to disable the sensor short of
# editing drone_run.sh on the rover, which dirties the checkout and silently
# stops it self-updating. This knob is the supported way to turn it off.
spf_default_ultrasonic() {
    printf '1'
}

# Resolve one key's built-in default for a given rover id.
spf_capture_env_default() {
    local key="$1" rover_id="${2:-}"
    case "$key" in
        SPF_CRASH_DETECT)   spf_default_crash_detect ;;
        SPF_CRASH_RECOVERY) spf_default_crash_recovery "$rover_id" ;;
        SPF_ULTRASONIC)     spf_default_ultrasonic ;;
        *) printf '' ;;
    esac
}

# Human-readable note explaining WHY a default is what it is, so `rover env`
# can answer "and why is that?" without the operator reading this file.
spf_capture_env_default_note() {
    local key="$1" rover_id="${2:-}"
    case "$key" in
        SPF_CRASH_DETECT)
            printf 'fleet-wide' ;;
        SPF_ULTRASONIC)
            printf 'fleet-wide; CH12 toggles it live when RC reaches the Pi' ;;
        SPF_CRASH_RECOVERY)
            if [[ "$rover_id" == "4" ]]; then
                printf 'rover 4 only'
            else
                printf 'rover 4 only; this is rover %s' "${rover_id:-unknown}"
            fi ;;
        *) printf '' ;;
    esac
}

# Shared truthiness, matching drone_run.sh's is_true so a value that boots the
# capture cannot read differently in the CLI.
spf_env_is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}
