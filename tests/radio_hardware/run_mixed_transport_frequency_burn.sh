#!/usr/bin/env bash
# Long, opt-in two-radio USB/IP + LO + AGC transition burn-in.

set -euo pipefail

[[ "$#" -ge 2 && "$#" -le 3 ]] || {
    printf 'usage: %s HOST_A HOST_B [REPORT_DIR]\n' "$0" >&2
    exit 2
}

readonly HOST_A="$1"
readonly HOST_B="$2"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
readonly REPORT_DIR="${3:-/tmp/spf-mixed-transport-burn-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly FREQUENCIES="${SPF_BURN_FREQUENCIES:-868M,915M,1280M,1300M,1301M,2412M,2467.1M,4000M,4001M,5766M,5804M,5866M}"
readonly EPOCHS="${SPF_BURN_EPOCHS:-3}"
readonly FRAMES="${SPF_BURN_FRAMES_PER_SESSION:-4}"
readonly SAMPLES="${SPF_BURN_SAMPLES_PER_CHANNEL:-131072}"
readonly ATTENUATION_DB="${SPF_BURN_ATTENUATION_DB:-30}"
ORIGINAL_RMEM="$(sysctl -n net.core.rmem_max)"
readonly ORIGINAL_RMEM
readonly BURN_RMEM_BYTES="$((32 * 1024 * 1024))"

mkdir -p "$REPORT_DIR"

cleanup() {
    sudo sysctl -q -w "net.core.rmem_max=${ORIGINAL_RMEM}" || true
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.mute_pluto_tx \
        --expected-count 2 --output "$REPORT_DIR/final-mute.json" >/dev/null || true
}
trap cleanup EXIT
cleanup
sudo sysctl -q -w "net.core.rmem_max=${BURN_RMEM_BYTES}"

printf 'hosts=%s,%s\nfrequencies=%s\nepochs=%s\nframes_per_session=%s\nsamples_per_channel=%s\nattenuation_db=%s\n' \
    "$HOST_A" "$HOST_B" "$FREQUENCIES" "$EPOCHS" "$FRAMES" "$SAMPLES" \
    "$ATTENUATION_DB" >"$REPORT_DIR/run.conf"
printf 'rmem_max=%s\n' "$BURN_RMEM_BYTES" >>"$REPORT_DIR/run.conf"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" "$PYTHON" -m pytest -q \
    tests/radio_hardware/test_mixed_transport_frequency_burn_hardware.py \
    --radio-hardware --radio-expected-count=2 --radio-gain-series-v3 \
    --radio-direct-ip --radio-soak --radio-tx-loopback \
    --radio-tx-loopback-attenuation-db="$ATTENUATION_DB" \
    --radio-direct-ip-ladder-host="$HOST_A" \
    --radio-direct-ip-ladder-host="$HOST_B" \
    --radio-burn-frequencies="$FREQUENCIES" --radio-cycles="$EPOCHS" \
    --radio-frames-per-request="$FRAMES" --radio-samples="$SAMPLES" \
    --radio-tx-sample-rate=3000000 --radio-tx-bandwidth=3000000 \
    --radio-tx-tone-hz=100000 --radio-tx-gain-db=-10 \
    --radio-report-dir="$REPORT_DIR" | tee "$REPORT_DIR/pytest.log"
