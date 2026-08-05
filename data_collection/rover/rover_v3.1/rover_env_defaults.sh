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

# The ultrasonic obstacle stop. OFF fleet-wide as of 2026-08-05.
#
# It was on by default, and Taranis CH12 could toggle it live. Both changed:
# the sensor is off unless a rover explicitly enables it, and when it is off the
# transmitter cannot bring it back.
#
# THE ENV SETTING IS AUTHORITATIVE, NOT THE TRANSMITTER. With ultrasonic off the
# collector runs --no-ultrasonic, which never constructs the distance finder at
# all; handle_RC_CHANNELS then returns on `distance_finder is None` before it
# looks at CH12. So a stray, held, or failsafe-frozen CH12 cannot re-enable a
# sensor the operator disabled -- which matters, because an R9 SX in failsafe
# holds its last channel values, and CH12 sat high on rovers 2 and 3 during
# 2026-08-04. Pinned by
# tests/test_mavlink_rc_safety.py::test_ultrasonic_rc_is_ignored_when_capture_disabled_the_sensor.
#
# What is given up: the obstacle stop. Stall detection (SPF_CRASH_DETECT, on
# fleet-wide) remains the backstop -- a rover that stops covering ground while
# armed and commanded is handed to the operator in MANUAL.
spf_default_ultrasonic() {
    printf '0'
}

# The fleet's time base. Capture filenames are stamped in LOCAL time, so a
# rover on a different timezone produces filenames that disagree with its
# siblings by the UTC offset -- Rover 4 sat on America/Los_Angeles and stamped a
# capture taken minutes after Rover 1's as eight hours and one calendar day
# earlier. Clocks were NTP-correct throughout; only the naming diverged, and
# nothing errors when it does.
spf_fleet_timezone() {
    printf 'Europe/London'
}

# --------------------------------------------------------------- journald ---
#
# Rovers 1 and 4 ran journald with Storage=volatile: the journal lives only in
# /run, so every reboot destroys it. On 2026-08-04 Rover 4 recorded a capture
# with one receive channel 24 dB below its sibling and the AGC railed on 99% of
# frames; by the time anyone looked, the journal for that boot no longer
# existed, and the .log sidecar was zero bytes. There was no record at all.
#
# The desired state is defined HERE, once, so the boot reconciler, the
# provisioner and `rover doctor` cannot disagree about what "persistent" means
# -- the same reason the capture-env defaults live in this file.
#
# The knobs, and why these numbers:
#   Storage=persistent   unconditional, rather than `auto`. `auto` only writes
#                        to disk if /var/log/journal already exists, which is
#                        exactly the state that silently failed to hold.
#   SystemMaxUse=1G      the rovers have 470 GB SD cards at 8% used, so space
#                        is not tight, but an uncapped journal is still an
#                        unbounded write amplifier on flash. 1 GB is months of
#                        rover logs and is trimmed by journald itself, oldest
#                        first, with no operator involvement.
#   SystemMaxFileSize=64M  keeps rotation granular, so hitting the cap discards
#                        an old 64 MB slice rather than a large fraction of the
#                        retained history.
#
# It is a DROP-IN, never an edit to /etc/systemd/journald.conf: drop-ins take
# precedence over the main file, so this overrides the distro's Storage=volatile
# without modifying a packaged file (and is removable in one `rm`).
SPF_JOURNALD_DROPIN_PATH_DEFAULT="/etc/systemd/journald.conf.d/10-spf-persistent.conf"
SPF_JOURNAL_DIR_DEFAULT="/var/log/journal"

spf_journald_dropin_content() {
    cat <<'EOF'
# Managed by SPF: data_collection/rover/rover_v3.1/reconcile_rover_boot_units.sh
# Edits are overwritten at the next boot. Change rover_env_defaults.sh instead.
#
# Volatile journals cost us the diagnosis of Rover 4's 24 dB channel imbalance
# on 2026-08-04: the boot that recorded it was gone before anyone could look.
[Journal]
Storage=persistent
SystemMaxUse=1G
SystemMaxFileSize=64M
EOF
}

# Is the journal ACTUALLY landing on disk right now? Configuration is not the
# same as effect -- journald reads Storage at start, so a rover that has just
# been converged is still volatile until it reboots. Both the reconciler and
# doctor have to be able to tell those two states apart, so the check lives
# here with the desired state it is checked against.
spf_journal_is_persistent_now() {
    local dir="${1:-}" entry
    [[ -n "$dir" ]] || dir="$SPF_JOURNAL_DIR_DEFAULT"
    [[ -d "$dir" ]] || return 1
    for entry in "$dir"/*/*.journal "$dir"/*.journal; do
        if [[ -f "$entry" ]]; then
            return 0
        fi
    done
    return 1
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
            printf 'off fleet-wide since 2026-08-05; CH12 cannot re-enable it' ;;
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
