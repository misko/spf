#!/usr/bin/env bash
# Run the bounded two-radio direct-IP ladder and restore host/radio safety state.

set -euo pipefail

[[ "$#" -ge 2 && "$#" -le 3 ]] || {
    printf 'usage: %s HOST_A HOST_B [REPORT_DIR]\n' "$0" >&2
    exit 2
}

readonly HOST_A="$1"
readonly HOST_B="$2"
[[ "$HOST_A" != "$HOST_B" ]] || {
    printf 'ERROR: the two direct-IP hosts must be unique\n' >&2
    exit 2
}
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly PYTHON="${SPF_PYTHON:-/home/pi/spf-virtualenv/bin/python3}"
readonly REPORT_DIR="${3:-/tmp/spf-direct-ip-parallel-ladder-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly ORIGINAL_RMEM="$(sysctl -n net.core.rmem_max)"
readonly RATES="${SPF_IP_LADDER_RATES:-1M,1.25M,1.5M,2M,3M,6M,10M,15M,20M,25M,30M}"
readonly CYCLES="${SPF_IP_LADDER_CYCLES:-3}"
readonly REQUIRED_RATE="${SPF_IP_LADDER_REQUIRED_RATE:-3000000}"
readonly INTERFACE="${SPF_IP_LADDER_INTERFACE:-eth0}"
readonly FRAMES="${SPF_IP_LADDER_FRAMES_PER_REQUEST:-4}"
readonly CONTINUE_AFTER_FAILURE="${SPF_IP_LADDER_CONTINUE_AFTER_FAILURE:-1}"
[[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] && (( FRAMES <= 16 )) || {
    printf 'ERROR: SPF_IP_LADDER_FRAMES_PER_REQUEST must be in [1, 16]\n' >&2
    exit 2
}
# Linux reports SO_RCVBUF as twice net.core.rmem_max. At four 4 MiB frames,
# 32 MiB rmem_max gives each radio a 64 MiB effective socket while keeping two
# simultaneous sockets below this host's default global UDP memory ceiling.
readonly LADDER_RMEM_BYTES="$((FRAMES * 8 * 1024 * 1024))"
readonly MIN_RECEIVE_BUFFER_MIB="$((FRAMES * 16))"

mkdir -p "$REPORT_DIR"

cleanup() {
    sudo sysctl -q -w "net.core.rmem_max=${ORIGINAL_RMEM}" || true
    PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.mute_pluto_tx \
        --expected-count 2 --output "$REPORT_DIR/final-mute.json" >/dev/null || true
}
trap cleanup EXIT

PYTHONPATH="$REPO_ROOT" "$PYTHON" -m spf.scripts.mute_pluto_tx \
    --expected-count 2 --output "$REPORT_DIR/initial-mute.json" >/dev/null
sudo sysctl -q -w "net.core.rmem_max=${LADDER_RMEM_BYTES}"

printf 'hosts=%s,%s\nrates=%s\ncycles=%s\nrequired_rate=%s\ninterface=%s\nframes_per_request=%s\nrmem_max=%s\neffective_receive_buffer_mib=%s\n' \
    "$HOST_A" "$HOST_B" "$RATES" "$CYCLES" "$REQUIRED_RATE" "$INTERFACE" \
    "$FRAMES" "$LADDER_RMEM_BYTES" "$MIN_RECEIVE_BUFFER_MIB" \
    >"$REPORT_DIR/run.conf"

cd "$REPO_ROOT"
pytest_args=(
    -m pytest -q
    tests/radio_hardware/test_direct_ip_parallel_ladder_hardware.py
    --radio-hardware --radio-gain-series-v3 --radio-direct-ip-ladder
    --radio-direct-ip-ladder-host="$HOST_A"
    --radio-direct-ip-ladder-host="$HOST_B"
    --radio-direct-ip-ladder-rates="$RATES"
    --radio-direct-ip-ladder-cycles="$CYCLES"
    --radio-direct-ip-ladder-required-rate="$REQUIRED_RATE"
    --radio-direct-ip-ladder-interface="$INTERFACE"
    --radio-samples=524288 --radio-frames-per-request="$FRAMES"
    --radio-gain-observation-interval=2048
    --radio-gain-observation-capacity=256
    --radio-direct-ip-min-receive-buffer-mib="$MIN_RECEIVE_BUFFER_MIB"
    --radio-report-dir="$REPORT_DIR"
)
if [[ "$CONTINUE_AFTER_FAILURE" == "1" ]]; then
    pytest_args+=(--radio-direct-ip-ladder-continue-after-failure)
fi
set +e
PYTHONPATH="$REPO_ROOT" "$PYTHON" "${pytest_args[@]}" | tee "$REPORT_DIR/pytest.txt"
readonly PYTEST_STATUS="${PIPESTATUS[0]}"
set -e

if (( PYTEST_STATUS == 0 )); then
    printf 'PASS: report: %s/direct_ip_parallel_sample_rate_ladder.json\n' "$REPORT_DIR"
else
    printf 'FAIL: preserved report: %s/direct_ip_parallel_sample_rate_ladder.json\n' \
        "$REPORT_DIR" >&2
fi
exit "$PYTEST_STATUS"
