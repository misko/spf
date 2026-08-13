#!/usr/bin/env bash
# Capture exactly one operator-declared E-GSC9 pad-discriminator leg.

set -euo pipefail

readonly REPO=/home/pi/spf
readonly PYTHON=/home/pi/spf-virtualenv/bin/python3
readonly BASE_CONFIG="$REPO/experiments/e_gsc9_rover_operating_region/configs/e_gsc9_pad_discriminator.yaml"
readonly BASE_CONFIG_SHA256=9b84c68aaba0b6cd81a732b7e98dd389f14cf048aeb6482db32fc3e1cefa9874
readonly PREPARE="$REPO/experiments/e_gsc9_rover_operating_region/analysis/prepare_session_c_leg.py"
readonly STAGE=/home/pi/gsc9_staging
readonly QNAP=/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency
readonly SERIAL_R17=104000bac4950008230026001b440a003a
readonly SERIAL_R18=1040007c4a94000211000b009186843ef2

run_date="${GSC9_C_RUN_DATE:-$(date +%Y%m%d)}"
[[ "$run_date" =~ ^[0-9]{8}$ ]] || {
    echo "ABORT: GSC9_C_RUN_DATE must be YYYYMMDD" >&2
    exit 2
}

leg="${1:-}"
case "$leg" in
    a)
        expected_state=no_pads
        prerequisite=e_gsc9_session_b_20260814_v1
        run="e_gsc9_session_c_a_${run_date}_v1"
        ;;
    b)
        expected_state=pads_installed
        prerequisite="e_gsc9_session_c_a_${run_date}_v1"
        run="e_gsc9_session_c_b_${run_date}_v1"
        ;;
    aprime)
        expected_state=pads_removed
        prerequisite="e_gsc9_session_c_b_${run_date}_v1"
        run="e_gsc9_session_c_aprime_${run_date}_v1"
        ;;
    *)
        echo "Usage: GSC9_C_PHYSICAL_STATE=... GSC9_C_OPERATOR_NOTE=... $0 {a|b|aprime}" >&2
        exit 2
        ;;
esac

physical_state="${GSC9_C_PHYSICAL_STATE:-}"
operator_note="${GSC9_C_OPERATOR_NOTE:-}"
[[ "$physical_state" == "$expected_state" ]] || {
    echo "ABORT: leg $leg requires GSC9_C_PHYSICAL_STATE=$expected_state" >&2
    exit 1
}
[[ -n "${operator_note//[[:space:]]/}" ]] || {
    echo "ABORT: GSC9_C_OPERATOR_NOTE must describe the observed hardware state" >&2
    exit 1
}

mountpoint -q /mnt/qnap01 || { echo "ABORT: QNAP is not mounted"; exit 1; }
[[ -w "$QNAP" ]] || { echo "ABORT: QNAP destination is not writable"; exit 1; }
[[ "$(sha256sum "$BASE_CONFIG" | awk '{print $1}')" == "$BASE_CONFIG_SHA256" ]] || {
    echo "ABORT: base session-C config changed"
    exit 1
}
if pgrep -f 'spf.calibrations.dual_rx_gain_frequency (run|automate)' >/dev/null; then
    echo "ABORT: another calibration capture is active"
    exit 1
fi

for serial in "$SERIAL_R17" "$SERIAL_R18"; do
    validation="$QNAP/$prerequisite/$serial/validation.json"
    [[ -f "$validation" ]] || { echo "ABORT: missing prerequisite $validation"; exit 1; }
    "$PYTHON" - "$validation" <<'PY'
import json, sys
document = json.load(open(sys.argv[1]))
if document.get("status") != "pass":
    raise SystemExit(f"prerequisite validation is not pass: {sys.argv[1]}")
PY
done

readonly OUTPUT="$STAGE/$run"
readonly CONFIG_DIR="$STAGE/e_gsc9_session_c_configs"
readonly CONFIG="$CONFIG_DIR/$run.yaml"
readonly LOG="$QNAP/_logs/$run.log"
mkdir -p "$OUTPUT" "$CONFIG_DIR" "$QNAP/_logs"
exec > >(tee -a "$LOG") 2>&1
exec 9>/run/lock/e_gsc9_session_c.lock
flock -n 9 || { echo "ABORT: another E-GSC9 session-C runner owns the lock"; exit 1; }

echo "START $(date --iso-8601=seconds) leg=$leg state=$physical_state"
"$PYTHON" "$PREPARE" \
    --base "$BASE_CONFIG" --leg "$leg" --physical-state "$physical_state" \
    --operator-note "$operator_note" --config-output "$CONFIG" \
    --state-output "$OUTPUT/physical_state.json"
cp -- "$CONFIG" "$OUTPUT/capture_config.yaml"

cd "$REPO"
export LD_LIBRARY_PATH=/usr/local/lib
for attempt in $(seq 1 12); do
    "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency run \
        --transport iio-usb --config "$CONFIG" --output "$OUTPUT" \
        --serial "$SERIAL_R17" --serial "$SERIAL_R18" && break
    [[ "$attempt" -lt 12 ]] || { echo "ABORT: capture retries exhausted"; exit 1; }
    sleep 30
done

for serial in "$SERIAL_R17" "$SERIAL_R18"; do
    "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency validate \
        --transport iio-usb --config "$CONFIG" \
        --dataset "$OUTPUT/$serial/calibration.v7.zarr" --serial "$serial" \
        --output "$OUTPUT/$serial/validation.json"
done

tar --sparse -C "$STAGE" -cf - "$run" | \
    tar --sparse --no-same-owner -C "$QNAP" -xf -
for serial in "$SERIAL_R17" "$SERIAL_R18"; do
    "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency validate \
        --transport iio-usb --config "$CONFIG" \
        --dataset "$QNAP/$run/$serial/calibration.v7.zarr" --serial "$serial" \
        --output "$QNAP/$run/$serial/validation.json"
done
cmp -- "$OUTPUT/physical_state.json" "$QNAP/$run/physical_state.json"
cmp -- "$OUTPUT/capture_config.yaml" "$QNAP/$run/capture_config.yaml"

if [[ "$leg" == aprime ]]; then
    "$PYTHON" \
        "$REPO/experiments/e_gsc9_rover_operating_region/analysis/analyze_session_c.py" \
        --leg-a-root "$QNAP/e_gsc9_session_c_a_${run_date}_v1" \
        --leg-b-root "$QNAP/e_gsc9_session_c_b_${run_date}_v1" \
        --leg-aprime-root "$QNAP/$run" \
        --output "$QNAP/$run/pad_discriminator_analysis.json"
    echo "PASS $(date --iso-8601=seconds): $run validated locally and from QNAP and H7 analyzed"
else
    echo "PASS $(date --iso-8601=seconds): $run validated locally and from QNAP"
fi
