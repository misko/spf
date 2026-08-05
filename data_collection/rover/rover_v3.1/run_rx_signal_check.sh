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
#
# RF overrides (applied to a COPY of the config; production YAML is untouched):
#   --freq HZ          carrier, e.g. --freq 5.8e9      (fleet default 5.766e9)
#   --bandwidth HZ     e.g. --bandwidth 5e6            (default 3e6)
#   --sampling HZ      sample rate, e.g. --sampling 20e6  (default 30e6)
#   --rx-gain DB       e.g. --rx-gain 20               (default -3)
#   --gain-mode MODE   slow_attack | fast_attack | manual  (default slow_attack)
#
# A capture at the wrong frequency looks exactly like a dead antenna, so the
# effective RF settings are printed above the metrics table every run.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly SERVICE_NAME="mavlink_controller.service"
PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
RECORDS="${SPF_RX_CHECK_RECORDS:-100}"
OUTPUT_ROOT="${SPF_RX_CHECK_OUTPUT_ROOT:-/home/pi/preflight/rx_signal_check}"
RADIO_WAIT_SECONDS="${SPF_RX_CHECK_RADIO_WAIT_SECONDS:-45}"
KEEP=0
EXISTING=""
declare -a OVERRIDES=()

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --records) [[ "$#" -ge 2 ]] || die "--records requires a value."
                   RECORDS="$2"; shift 2 ;;
        # RF overrides. These do not edit the production config -- an effective
        # copy is written into the run directory and captured with instead, so
        # the canonical YAML stays exactly as the fleet flies it.
        --freq|--f-carrier)
                   [[ "$#" -ge 2 ]] || die "$1 requires a value."
                   OVERRIDES+=("f-carrier=$2"); shift 2 ;;
        --rx-gain) [[ "$#" -ge 2 ]] || die "--rx-gain requires a value."
                   OVERRIDES+=("rx-gain=$2"); shift 2 ;;
        --gain-mode)
                   [[ "$#" -ge 2 ]] || die "--gain-mode requires a value."
                   OVERRIDES+=("rx-gain-mode=$2"); shift 2 ;;
        --bandwidth)
                   [[ "$#" -ge 2 ]] || die "--bandwidth requires a value."
                   OVERRIDES+=("bandwidth=$2"); shift 2 ;;
        --sampling|--f-sampling)
                   [[ "$#" -ge 2 ]] || die "$1 requires a value."
                   OVERRIDES+=("f-sampling=$2"); shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help)
            sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# `systemctl stop` returns when the unit is inactive, NOT when its USB handles
# are released. Running immediately after a stop loses the race and the Pluto
# library reports the unhelpful "No device found" -- two of three back-to-back
# runs failed exactly this way on Rover 1. Wait for the radios to actually be
# free instead of trusting the unit state.
wait_for_radios() {
    local deadline=$((SECONDS + RADIO_WAIT_SECONDS)) holder
    while (( SECONDS < deadline )); do
        holder="$(pgrep -f "mavlink_radio_collection|drone_run.sh" | grep -v "^$$\$" || true)"
        if [[ -z "$holder" ]]; then
            return 0
        fi
        printf '\r   waiting for the radios to be released (pid %s)... %ss  ' \
            "$(tr "\n" "," <<<"$holder" | sed "s/,\$//")" "$((deadline - SECONDS))"
        sleep 1
    done
    printf '\n'
    die "Radios still held after ${RADIO_WAIT_SECONDS}s by:
$(pgrep -af "mavlink_radio_collection|drone_run.sh" || true)
  Stop those before retrying."
}
wait_for_radios
[[ -t 1 ]] && printf '\r%*s\r' 70 '' || true

# A V7 capture refuses to record without boot-verified Pluto firmware, and it
# only discovers that AFTER bringing the radios online -- about five minutes in,
# ending in a RuntimeError deep in data_collector. The manifest that proves it
# lives in /run (tmpfs), so it is absent on any boot where
# spf-pluto-direct-usb.service did not complete. Check it up front.
READY_FILE="${SPF_DIRECT_USB_READY_FILE:-/run/spf/direct_usb_ready.json}"
[[ -f "$READY_FILE" ]] || die "Pluto firmware is not boot-verified: ${READY_FILE} is absent.
  A V7 capture will refuse to record without it.
  Fix:    sudo systemctl start spf-pluto-direct-usb.service
  Then:   ls -l ${READY_FILE}
  If that service fails, see: journalctl -u spf-pluto-direct-usb.service -b"

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

# Overrides are applied to a COPY. The production config is immutable: it is
# what the fleet flies, and a diagnostic must never be able to change it.
if [[ "${#OVERRIDES[@]}" -gt 0 ]]; then
    effective="${run_dir}/effective_config.yaml"
    "$PYTHON" - "$CONFIG" "$effective" "${OVERRIDES[@]}" <<'PYEOF' || die "Could not apply overrides."
import sys, yaml
source, target, *pairs = sys.argv[1:]
config = yaml.safe_load(open(source))
overrides = dict(pair.split("=", 1) for pair in pairs)
for key, raw in overrides.items():
    try:
        value = float(raw) if key != "rx-gain-mode" else raw
        if key != "rx-gain-mode" and value.is_integer():
            value = int(value)
    except ValueError:
        value = raw
    for receiver in config["receivers"]:
        before = receiver.get(key)
        receiver[key] = value
        print(f"   override  {key}: {before} -> {value}")
with open(target, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PYEOF
    CONFIG="$effective"
fi

printf '\n== capturing %s frames per receiver (motion-free) ==\n' "$RECORDS"
printf '   config : %s\n   output : %s\n' "$CONFIG" "$run_dir"
# Print the RF the capture will actually use. A sweep that finds nothing at the
# wrong frequency looks identical to a dead antenna, so the settings must be on
# screen next to the numbers they produced.
"$PYTHON" - "$CONFIG" <<'PYEOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
rx = config["receivers"][0]
print(f"   rf     : carrier {rx['f-carrier']/1e9:.6g} GHz  "
      f"bw {rx['bandwidth']/1e6:.4g} MHz  fs {rx['f-sampling']/1e6:.4g} MS/s  "
      f"gain {rx['rx-gain']} ({rx['rx-gain-mode']})")
print()
PYEOF

readonly STATUS_FILE="${run_dir}/capture_status.json"

# Progress comes from the collector's own status file, the same one the capture
# watchdog polls. CaptureStatus.publish throttles to one write every ~5 s, so
# the counts step rather than stream; the elapsed clock ticks every second so
# the display still shows liveness between updates.
status_counts() {
    [[ -r "$STATUS_FILE" ]] || return 1
    sed -n 's/.*"records_written_by_receiver": \[\([^]]*\)\].*/\1/p' \
        <(tr -d '\n' <"$STATUS_FILE") | tr -d ' '
}

draw_bar() {
    local done="$1" total="$2" elapsed="$3" per_rx="$4" width=30 filled pct bar pad
    (( total > 0 )) || total=1
    pct=$(( done * 100 / total ))
    filled=$(( done * width / total ))
    (( filled > width )) && filled=$width
    printf -v bar '%*s' "$filled" ''
    printf -v pad '%*s' "$(( width - filled ))" ''
    printf '\r   [%s%s] %3d%%  %s/%s  [%s]  %s  ' \
        "${bar// /=}" "${pad// /·}" "$pct" "$done" "$total" "${per_rx:-…}" "$elapsed"
}

export PYTHONBREAKPOINT=0
# Run from the run directory, not the checkout. mavlink_controller.py calls
# logging.basicConfig(filename="logs.log") with a RELATIVE path, so the log
# lands in whatever the working directory is. Pointing that at the repo means
# a single earlier `sudo` run leaves a root-owned logs.log that makes every
# later non-root capture die with PermissionError before it starts. Here the
# log lands beside the capture it belongs to, and the checkout is never
# written to.
cd "$run_dir"
"$PYTHON" "${REPO_ROOT}/spf/mavlink_radio_collection.py" \
    --fake-drone \
    --no-ultrasonic \
    --yaml-config "$CONFIG" \
    --device-mapping /home/pi/device_mapping \
    --routine center \
    --records-per-receiver "$RECORDS" \
    --temp "$run_dir" \
    --status-file "$STATUS_FILE" \
    --tag "RXCHECK_RO${rover_id}" \
    >"$run_dir/console.txt" 2>&1 &
capture_pid=$!

# A bar is only meaningful on a terminal. Redirected to a file or a log, print
# a plain line per update instead of thousands of carriage returns.
started=$SECONDS
last_line=""
while kill -0 "$capture_pid" 2>/dev/null; do
    counts="$(status_counts || true)"
    done_frames=0
    if [[ -n "$counts" ]]; then
        # The laggard receiver is the honest progress: the capture is not
        # finished until every receiver has its frames.
        done_frames="$(tr ',' '\n' <<<"$counts" | sort -n | head -1)"
        [[ "$done_frames" =~ ^[0-9]+$ ]] || done_frames=0
    fi
    elapsed=$(printf '%d:%02d' $(( (SECONDS-started)/60 )) $(( (SECONDS-started)%60 )))
    if [[ -t 1 ]]; then
        draw_bar "$done_frames" "$RECORDS" "$elapsed" "${counts:-}"
    elif [[ "$done_frames/$RECORDS" != "$last_line" ]]; then
        printf '   %s/%s frames  [%s]  %s\n' \
            "$done_frames" "$RECORDS" "${counts:-…}" "$elapsed"
        last_line="$done_frames/$RECORDS"
    fi
    sleep 1
done

wait "$capture_pid" || capture_status=$?
[[ -t 1 ]] && printf '\r%*s\r' 90 ''
if [[ "${capture_status:-0}" -ne 0 ]]; then
    tail -20 "$run_dir/console.txt" >&2
    die "Capture failed (status ${capture_status}); see ${run_dir}/console.txt"
fi
printf '   captured in %s\n' "$(printf '%d:%02d' $(( (SECONDS-started)/60 )) $(( (SECONDS-started)%60 )))"

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
