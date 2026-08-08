#!/usr/bin/env bash
# Receive-only RAM-boot acceptance campaign for an unreleased protocol-v3 DFU.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT
readonly MULTI_LOADER="${ROOT}/spf/scripts/pluto_multi_firmware.py"
readonly FIRMWARE_LOADER="${ROOT}/data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh"
readonly SSH_CONFIG="${ROOT}/data_collection/rover/rover_v3.1/ssh_config"
readonly TEST_FILE="${ROOT}/tests/radio_hardware/test_gain_series_v3_hardware.py"
readonly BASELINE_TEST="${ROOT}/tests/radio_hardware/test_direct_usb_hardware.py"

usage() {
    cat <<'EOF'
Usage: run_gain_series_v3_candidate.sh IMAGE.dfu [DIRECT_IP_HOST]

RAM-load an unreleased protocol-v3 image on exactly two attached Plutos and
run the receive-only promotion gates. DIRECT_IP_HOST is optional; when given,
its standard IIO serial is checked against the attached USB radios before the
direct-IP parity gate runs.

Environment:
  SPF_V3_EXPECTED_RADIOS       Exact radio count (default: 2)
  SPF_V3_IMAGE_SHA256          Expected image SHA-256 (default: computed)
  SPF_V3_PYTHON                Python with SPF test dependencies
                               (default: /home/pi/spf-virtualenv/bin/python)
  SPF_V3_PRODUCTION_RECORDS    V7 records per radio (default: 100)
  SPF_V3_REPORT_ROOT           Artifact directory (default: timestamped /tmp)

The script loads only volatile RAM. It never writes QSPI and never enables TX.
On failure it leaves the candidate running for inspection and prints the
explicit rollback command.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || {
    usage
    exit 0
}
[[ "$#" -eq 1 || "$#" -eq 2 ]] || {
    usage >&2
    exit 2
}

IMAGE="$(realpath -- "$1")"
readonly IMAGE
readonly DIRECT_IP_HOST="${2:-}"
readonly EXPECTED_RADIOS="${SPF_V3_EXPECTED_RADIOS:-2}"
readonly PYTHON="${SPF_V3_PYTHON:-/home/pi/spf-virtualenv/bin/python}"
readonly PRODUCTION_RECORDS="${SPF_V3_PRODUCTION_RECORDS:-100}"
readonly REPORT_ROOT="${SPF_V3_REPORT_ROOT:-/tmp/spf-gain-series-v3-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly STATE_ROOT="${REPORT_ROOT}/firmware-state"

[[ -f "$IMAGE" ]] || die "candidate image not found: $IMAGE"
[[ "$EXPECTED_RADIOS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_EXPECTED_RADIOS must be positive"
[[ "$PRODUCTION_RECORDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_PRODUCTION_RECORDS must be positive"
[[ -x "$PYTHON" ]] || die "test Python is not executable: $PYTHON"
[[ -f "$MULTI_LOADER" ]] || die "multi-radio loader is missing"
[[ -f "$TEST_FILE" ]] || die "protocol-v3 hardware tests are missing"

for command_name in iio_attr iio_info lsusb sha256sum sudo tee; do
    require_command "$command_name"
done
sudo -n true 2>/dev/null ||
    die "passwordless sudo is required; run sudo -v before starting"

mkdir -p "$REPORT_ROOT" "$STATE_ROOT"
actual_sha="$(sha256sum "$IMAGE" | awk '{print $1}')"
expected_sha="${SPF_V3_IMAGE_SHA256:-$actual_sha}"
[[ "$actual_sha" == "$expected_sha" ]] ||
    die "candidate SHA-256 mismatch: expected $expected_sha, got $actual_sha"
readonly actual_sha expected_sha

run_logged() {
    local name="$1"
    shift
    printf '\n===== %s =====\n' "$name"
    "$@" 2>&1 | tee "${REPORT_ROOT}/${name}.log"
}

rollback_hint() {
    local rc=$?
    printf '\nCampaign stopped with status %d. Candidate remains in RAM.\n' "$rc" >&2
    printf 'Inspect %s, then roll back explicitly with:\n' "$REPORT_ROOT" >&2
    printf '  sudo %q rollback-all %q\n' "$FIRMWARE_LOADER" "$EXPECTED_RADIOS" >&2
    exit "$rc"
}
trap rollback_hint ERR

common_loader_args=(
    --image "$IMAGE"
    --image-sha256 "$actual_sha"
    --ssh-config "$SSH_CONFIG"
    --ssh-password analog
    --state-root "$STATE_ROOT"
    --expected-count "$EXPECTED_RADIOS"
)

printf 'image=%s\nsha256=%s\nexpected_radios=%s\nreport_root=%s\n' \
    "$IMAGE" "$actual_sha" "$EXPECTED_RADIOS" "$REPORT_ROOT"
run_logged iio-before iio_info -s

run_logged baseline-v2 \
    "$PYTHON" -m pytest -q "$BASELINE_TEST" \
    --radio-hardware \
    --radio-expected-count="$EXPECTED_RADIOS" \
    --radio-samples=16384 \
    --radio-cycles=2 \
    --radio-frames-per-request=2 \
    --radio-report-dir="${REPORT_ROOT}/baseline-v2-report"

run_logged persistent-config \
    sudo -n "$PYTHON" "$MULTI_LOADER" check-config-all \
    "${common_loader_args[@]}"
run_logged ram-load \
    sudo -n "$PYTHON" "$MULTI_LOADER" load-all \
    "${common_loader_args[@]}"
run_logged iio-after-ram-load iio_info -s

# Protocol v3 must remain backwards compatible with the promoted v2 host path.
run_logged candidate-v2-compatibility \
    "$PYTHON" -m pytest -q "$BASELINE_TEST" \
    --radio-hardware \
    --radio-expected-count="$EXPECTED_RADIOS" \
    --radio-samples=524288 \
    --radio-cycles=2 \
    --radio-frames-per-request=3 \
    --radio-report-dir="${REPORT_ROOT}/candidate-v2-report"

run_logged candidate-v3-usb-smoke \
    "$PYTHON" -m pytest -q \
    "${TEST_FILE}::test_v3_usb_gain_observations" \
    --radio-hardware \
    --radio-gain-series-v3 \
    --radio-expected-count="$EXPECTED_RADIOS" \
    --radio-samples=32768 \
    --radio-frames-per-request=3 \
    --radio-gain-observation-interval=2048 \
    --radio-gain-observation-capacity=256 \
    --radio-report-dir="${REPORT_ROOT}/candidate-v3-smoke-report"

run_logged candidate-v3-production-zarr \
    "$PYTHON" -m pytest -q \
    "${TEST_FILE}::test_v3_gain_series_round_trips_through_v7_zarr" \
    --radio-hardware \
    --radio-gain-series-v3 \
    --radio-zarr \
    --radio-expected-count="$EXPECTED_RADIOS" \
    --radio-samples=524288 \
    --radio-zarr-frames="$PRODUCTION_RECORDS" \
    --radio-gain-observation-interval=2048 \
    --radio-gain-observation-capacity=256 \
    --radio-report-dir="${REPORT_ROOT}/candidate-v3-zarr-report"

if [[ -n "$DIRECT_IP_HOST" ]]; then
    ip_serial="$(iio_attr -u "ip:${DIRECT_IP_HOST}" -C hw_serial | awk -F': ' '/^hw_serial:/ {print $2}')"
    [[ -n "$ip_serial" ]] ||
        die "could not read hw_serial from ip:${DIRECT_IP_HOST}"
    lsusb -d 0456:b673 -v 2>/dev/null | grep -Fq "$ip_serial" ||
        die "direct-IP serial $ip_serial is not one of the attached USB radios"
    printf 'direct_ip_host=%s direct_ip_serial=%s\n' "$DIRECT_IP_HOST" "$ip_serial"
    run_logged candidate-v3-direct-ip \
        "$PYTHON" -m pytest -q \
        "${TEST_FILE}::test_v3_direct_ip_uses_the_same_inner_frame" \
        --radio-hardware \
        --radio-gain-series-v3 \
        --radio-direct-ip \
        --radio-direct-ip-host="$DIRECT_IP_HOST" \
        --radio-samples=524288 \
        --radio-gain-observation-interval=2048 \
        --radio-gain-observation-capacity=256 \
        --radio-report-dir="${REPORT_ROOT}/candidate-v3-ip-report"
fi

run_logged final-status \
    sudo -n "$PYTHON" "$MULTI_LOADER" status-all \
    "${common_loader_args[@]}"
printf '\nPASS: protocol-v3 RAM candidate completed every requested gate.\n'
printf 'Artifacts: %s\n' "$REPORT_ROOT"
printf 'QSPI was not modified.\n'
