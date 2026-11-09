#!/usr/bin/env bash
# Run the time-separated E-GSC9 session-B control on the bench Pi.

set -euo pipefail

readonly REPO=/home/pi/spf
readonly PYTHON=/home/pi/spf-virtualenv/bin/python3
readonly CONFIG="$REPO/experiments/e_gsc9_rover_operating_region/configs/e_gsc9_session_transfer.yaml"
readonly PREP_CONFIG="$REPO/experiments/e_gsc9_rover_operating_region/configs/e_gsc9_bench_prepare.yaml"
readonly CONFIG_SHA256=395b1348ff72fc2a2358d3231ae941c53466082a030db4f4216e5ba9c1e1bb08
readonly EARLIEST_UNIX=1786694220
readonly RUN=e_gsc9_session_b_20260814_v1
readonly STAGE=/home/pi/gsc9_staging
readonly QNAP=/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency
readonly LOG="$QNAP/_logs/${RUN}.log"
readonly SERIAL_R17=104000bac4950008230026001b440a003a
readonly SERIAL_R18=1040007c4a94000211000b009186843ef2

mkdir -p "$QNAP/_logs" "$STAGE"
exec > >(tee -a "$LOG") 2>&1
exec 9>/run/lock/e_gsc9_session_b.lock
flock -n 9 || { echo "ABORT: another E-GSC9 session-B runner owns the lock"; exit 1; }

echo "START $(date --iso-8601=seconds)"
[[ "$(date +%s)" -ge "$EARLIEST_UNIX" ]] || {
    echo "ABORT: session B may not start before 2026-08-14T08:57:00+01:00"
    exit 1
}
session_start="$(date +%s)"
mountpoint -q /mnt/qnap01 || { echo "ABORT: QNAP is not mounted"; exit 1; }
[[ -w "$QNAP" ]] || { echo "ABORT: QNAP destination is not writable"; exit 1; }
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || {
    echo "ABORT: session-B config changed after scheduling"
    exit 1
}
if pgrep -f 'spf.calibrations.dual_rx_gain_frequency (run|automate)' >/dev/null; then
    echo "ABORT: another calibration capture is active"
    exit 1
fi

# Resolve the exact physical targets before removing power. These are the two
# downstream ports of hub 1-1; RF connectors are never handled.
hub_status="$(sudo -n uhubctl -l 1-1)"
grep -F "Port 1:" <<<"$hub_status" | grep -F "$SERIAL_R17" >/dev/null || {
    echo "ABORT: R17 is not on expected powered hub port 1"
    exit 1
}
grep -F "Port 2:" <<<"$hub_status" | grep -F "$SERIAL_R18" >/dev/null || {
    echo "ABORT: R18 is not on expected powered hub port 2"
    exit 1
}

echo "Power-cycling only hub 1-1 ports 1 and 2"
power_is_off=1
restore_power() {
    if [[ "$power_is_off" -eq 1 ]]; then
        sudo -n uhubctl -l 1-1 -p 1 -a on || true
        sudo -n uhubctl -l 1-1 -p 2 -a on || true
    fi
}
trap restore_power EXIT
sudo -n uhubctl -l 1-1 -p 1 -a off
sudo -n uhubctl -l 1-1 -p 2 -a off
sleep 10
sudo -n uhubctl -l 1-1 -p 1 -a on
sudo -n uhubctl -l 1-1 -p 2 -a on
power_is_off=0
trap - EXIT

deadline=$((SECONDS + 90))
while true; do
    scan="$(LD_LIBRARY_PATH=/usr/local/lib iio_info -s 2>&1 || true)"
    if grep -F "$SERIAL_R17" <<<"$scan" >/dev/null &&
       grep -F "$SERIAL_R18" <<<"$scan" >/dev/null; then
        break
    fi
    (( SECONDS < deadline )) || { echo "ABORT: radios did not re-enumerate"; exit 1; }
    sleep 2
done

cd "$REPO"
export LD_LIBRARY_PATH=/usr/local/lib
sudo -n env \
    SPF_ROVER_ID=1 \
    SPF_CAPTURE_CONFIG="$PREP_CONFIG" \
    SPF_PYTHON="$PYTHON" \
    SPF_PLUTO_RAM_LOAD=0 \
    LD_LIBRARY_PATH=/usr/local/lib \
    bash data_collection/rover/rover_v3.1/prepare_direct_usb_boot.sh

manifest_mtime="$(stat -c %Y /run/spf/direct_usb_ready.json)"
[[ "$manifest_mtime" -ge "$session_start" ]] || {
    echo "ABORT: ready manifest is not fresh for session B"
    exit 1
}
mkdir -p "$STAGE/$RUN"
cp -- /run/spf/direct_usb_ready.json "$STAGE/$RUN/direct_usb_ready.json"

run_capture() {
    local attempt
    for attempt in $(seq 1 12); do
        "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency run \
            --transport iio-usb --config "$CONFIG" --output "$STAGE/$RUN" \
            --serial "$SERIAL_R17" --serial "$SERIAL_R18" && return 0
        echo "Capture attempt $attempt failed; resumable retry follows"
        sleep 30
    done
    return 1
}
run_capture

for serial in "$SERIAL_R17" "$SERIAL_R18"; do
    "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency validate \
        --transport iio-usb --config "$CONFIG" \
        --dataset "$STAGE/$RUN/$serial/calibration.v7.zarr" \
        --serial "$serial" --output "$STAGE/$RUN/$serial/validation.json"
done

# Preserve sparse LMDB allocation and avoid NFS ownership changes. This is a
# copy, not a move: local evidence remains until the experiment is complete.
tar --sparse -C "$STAGE" -cf - "$RUN" | \
    tar --sparse --no-same-owner -C "$QNAP" -xf -

# A second full IQ recomputation from QNAP proves that the copied stores are
# readable and semantically complete.
for serial in "$SERIAL_R17" "$SERIAL_R18"; do
    "$PYTHON" -m spf.calibrations.dual_rx_gain_frequency validate \
        --transport iio-usb --config "$CONFIG" \
        --dataset "$QNAP/$RUN/$serial/calibration.v7.zarr" \
        --serial "$serial" --output "$QNAP/$RUN/$serial/validation.json"
done

"$PYTHON" \
    "$REPO/experiments/e_gsc9_rover_operating_region/analysis/analyze_session_transfer.py" \
    --session-a-config \
        "$REPO/experiments/e_gsc9_rover_operating_region/configs/e_gsc9_rover_region_grid.yaml" \
    --session-a-root "$QNAP/e_gsc9_session_a_20260813_v1" \
    --session-b-config "$CONFIG" --session-b-root "$QNAP/$RUN" \
    --output "$QNAP/$RUN/session_transfer_vs_a.json"

echo "PASS $(date --iso-8601=seconds): session B captured, validated locally and from QNAP, and H6 analyzed"
