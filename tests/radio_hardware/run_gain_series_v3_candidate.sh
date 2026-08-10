#!/usr/bin/env bash
# RAM-boot acceptance campaign for an unreleased protocol-v3 DFU.  TX remains
# fail-closed unless the operator explicitly enables the attenuated loopback.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ROOT
readonly MULTI_LOADER="${ROOT}/spf/scripts/pluto_multi_firmware.py"
readonly SSH_CONFIG="${ROOT}/data_collection/rover/rover_v3.1/ssh_config"
readonly TEST_FILE="${ROOT}/tests/radio_hardware/test_gain_series_v3_hardware.py"
readonly TX_TEST_FILE="${ROOT}/tests/radio_hardware/test_gain_series_v3_tx_loopback_hardware.py"
readonly BASELINE_TEST="${ROOT}/tests/radio_hardware/test_direct_usb_hardware.py"

usage() {
    cat <<'EOF'
Usage: run_gain_series_v3_candidate.sh [OPTIONS] IMAGE.dfu [DIRECT_IP_HOST]

RAM-load an unreleased protocol-v3 image on exactly two attached Plutos and
run the promotion gates. DIRECT_IP_HOST is optional; when given,
its standard IIO serial is checked against the attached USB radios before the
direct-IP parity gate runs.

Options:
  --with-tx-loopback
      Enable the cabled TX2 -> attenuator/splitter -> RX1/RX2 gate.
  --loopback-attenuation-db DB
      Declare the physical attenuation to each RX input. Required with TX;
      the hardware test rejects values below 30 dB.

Environment:
  SPF_V3_EXPECTED_RADIOS       Exact radio count (default: 2)
  SPF_V3_IMAGE_SHA256          Expected image SHA-256 (default: computed)
  SPF_V3_PYTHON                Python with SPF test dependencies
                               (default: /home/pi/spf-virtualenv/bin/python)
  SPF_V3_PRODUCTION_RECORDS    V7 records per radio (default: 100)
  SPF_V3_TX_BOOT_EPOCHS        Independent RAM-boot TX epochs (default: 3)
  SPF_V3_STARTUP_STRESS_CYCLES Fresh v3 STARTs per radio (default: 100)
  SPF_V3_IP_BURN_IN_CYCLES     Maximum-size buffered IP bursts (default: 20)
  SPF_V3_REPORT_ROOT           Artifact directory (default: timestamped /tmp)

The script loads only volatile RAM and never writes QSPI. TX remains disabled
unless --with-tx-loopback is present. The TX fixture and the outer runner both
mute TX1/TX2 on exit. On failure the candidate remains in RAM for inspection
and the script prints the explicit rollback command.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

with_tx_loopback=0
loopback_attenuation_db=""
positionals=()
while (( "$#" )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --with-tx-loopback)
            with_tx_loopback=1
            ;;
        --loopback-attenuation-db)
            shift
            [[ "$#" -gt 0 ]] || die "--loopback-attenuation-db requires a value"
            loopback_attenuation_db="$1"
            ;;
        --loopback-attenuation-db=*)
            loopback_attenuation_db="${1#*=}"
            ;;
        --*)
            die "unknown option: $1"
            ;;
        *)
            positionals+=("$1")
            ;;
    esac
    shift
done
set -- "${positionals[@]}"

[[ "$#" -eq 1 || "$#" -eq 2 ]] || {
    usage >&2
    exit 2
}
if [[ "$with_tx_loopback" -eq 1 && -z "$loopback_attenuation_db" ]]; then
    die "--with-tx-loopback requires --loopback-attenuation-db"
fi

IMAGE="$(realpath -- "$1")"
readonly IMAGE
readonly DIRECT_IP_HOST="${2:-}"
readonly EXPECTED_RADIOS="${SPF_V3_EXPECTED_RADIOS:-2}"
readonly PYTHON="${SPF_V3_PYTHON:-/home/pi/spf-virtualenv/bin/python}"
readonly PRODUCTION_RECORDS="${SPF_V3_PRODUCTION_RECORDS:-100}"
readonly TX_BOOT_EPOCHS="${SPF_V3_TX_BOOT_EPOCHS:-3}"
readonly STARTUP_STRESS_CYCLES="${SPF_V3_STARTUP_STRESS_CYCLES:-100}"
readonly IP_BURN_IN_CYCLES="${SPF_V3_IP_BURN_IN_CYCLES:-20}"
readonly REPORT_ROOT="${SPF_V3_REPORT_ROOT:-/tmp/spf-gain-series-v3-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly STATE_ROOT="${REPORT_ROOT}/firmware-state"
readonly WITH_TX_LOOPBACK="$with_tx_loopback"
readonly LOOPBACK_ATTENUATION_DB="$loopback_attenuation_db"

[[ -f "$IMAGE" ]] || die "candidate image not found: $IMAGE"
[[ "$EXPECTED_RADIOS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_EXPECTED_RADIOS must be positive"
[[ "$PRODUCTION_RECORDS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_PRODUCTION_RECORDS must be positive"
[[ "$TX_BOOT_EPOCHS" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_TX_BOOT_EPOCHS must be positive"
[[ "$STARTUP_STRESS_CYCLES" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_STARTUP_STRESS_CYCLES must be positive"
[[ "$IP_BURN_IN_CYCLES" =~ ^[1-9][0-9]*$ ]] ||
    die "SPF_V3_IP_BURN_IN_CYCLES must be positive"
[[ -x "$PYTHON" ]] || die "test Python is not executable: $PYTHON"
[[ -f "$MULTI_LOADER" ]] || die "multi-radio loader is missing"
[[ -f "$TEST_FILE" ]] || die "protocol-v3 hardware tests are missing"
[[ -f "$TX_TEST_FILE" ]] || die "protocol-v3 TX hardware tests are missing"

for command_name in iio_attr iio_info lsusb sha256sum sudo sysctl tee; do
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

direct_ip_serial=""
original_rmem_max=""
if [[ -n "$DIRECT_IP_HOST" ]]; then
    direct_ip_serial="$(
        iio_attr -u "ip:${DIRECT_IP_HOST}" -C hw_serial |
            awk -F': ' '/^hw_serial:/ {print $2}'
    )"
    [[ -n "$direct_ip_serial" ]] ||
        die "could not read hw_serial from initial ip:${DIRECT_IP_HOST}"
    original_rmem_max="$(sysctl -n net.core.rmem_max)"
    target_rmem_max="$original_rmem_max"
    if (( target_rmem_max < 67108864 )); then
        target_rmem_max=67108864
    fi
    sudo -n sysctl -q -w "net.core.rmem_max=${target_rmem_max}" |
        tee "${REPORT_ROOT}/direct-ip-sysctl.log"
fi
readonly direct_ip_serial original_rmem_max

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
    printf '  sudo %q %q rollback-all' "$PYTHON" "$MULTI_LOADER" >&2
    printf ' %q' "${common_loader_args[@]}" >&2
    printf '\n' >&2
    exit "$rc"
}
trap rollback_hint ERR

mute_tx_on_exit() {
    local rc=$?
    set +e
    "$PYTHON" -m spf.scripts.mute_pluto_tx \
        --expected-count "$EXPECTED_RADIOS" \
        --output "${REPORT_ROOT}/tx-mute-on-exit.json" \
        >>"${REPORT_ROOT}/tx-mute-on-exit.log" 2>&1
    local mute_rc=$?
    if [[ "$mute_rc" -ne 0 ]]; then
        printf 'ERROR: final TX mute verification failed; inspect %s\n' \
            "${REPORT_ROOT}/tx-mute-on-exit.log" >&2
        [[ "$rc" -ne 0 ]] || rc="$mute_rc"
    fi
    if [[ -n "$original_rmem_max" ]]; then
        sudo -n sysctl -q -w "net.core.rmem_max=${original_rmem_max}" \
            >>"${REPORT_ROOT}/direct-ip-sysctl.log" 2>&1
        local sysctl_rc=$?
        if [[ "$sysctl_rc" -ne 0 ]]; then
            printf 'ERROR: failed to restore net.core.rmem_max=%s\n' \
                "$original_rmem_max" >&2
            [[ "$rc" -ne 0 ]] || rc="$sysctl_rc"
        fi
    fi
    trap - EXIT
    exit "$rc"
}
trap mute_tx_on_exit EXIT

common_loader_args=(
    --image "$IMAGE"
    --image-sha256 "$actual_sha"
    --ssh-config "$SSH_CONFIG"
    --ssh-password analog
    --state-root "$STATE_ROOT"
    --expected-count "$EXPECTED_RADIOS"
)

printf 'image=%s\nsha256=%s\nexpected_radios=%s\nreport_root=%s\nwith_tx_loopback=%s\nloopback_attenuation_db=%s\ntx_boot_epochs=%s\nstartup_stress_cycles=%s\nip_burn_in_cycles=%s\n' \
    "$IMAGE" "$actual_sha" "$EXPECTED_RADIOS" "$REPORT_ROOT" \
    "$WITH_TX_LOOPBACK" "$LOOPBACK_ATTENUATION_DB" "$TX_BOOT_EPOCHS" \
    "$STARTUP_STRESS_CYCLES" "$IP_BURN_IN_CYCLES"
run_logged iio-before iio_info -s

run_logged pre-campaign-tx-mute \
    "$PYTHON" -m spf.scripts.mute_pluto_tx \
    --expected-count "$EXPECTED_RADIOS" \
    --output "${REPORT_ROOT}/pre-campaign-tx-mute.json"

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
if [[ "$WITH_TX_LOOPBACK" -eq 1 ]]; then
    # TX health is the mandatory promotion gate and has exhibited boot-order
    # failures. Exercise independent FPGA boots before spending time on the
    # receive, IP, and Zarr stages. The final epoch remains loaded for them.
    for ((epoch = 1; epoch <= TX_BOOT_EPOCHS; epoch++)); do
        run_logged "ram-load-epoch-${epoch}" \
            sudo -n "$PYTHON" "$MULTI_LOADER" load-all \
            "${common_loader_args[@]}"
        run_logged "post-load-tx-mute-epoch-${epoch}" \
            "$PYTHON" -m spf.scripts.mute_pluto_tx \
            --expected-count "$EXPECTED_RADIOS" \
            --output "${REPORT_ROOT}/post-load-tx-mute-epoch-${epoch}.json"
        run_logged "iio-after-ram-load-epoch-${epoch}" iio_info -s
        run_logged "candidate-v3-tx2-loopback-epoch-${epoch}" \
            "$PYTHON" -m pytest -q "$TX_TEST_FILE" \
            --radio-hardware \
            --radio-gain-series-v3 \
            --radio-tx-loopback \
            --radio-tx-loopback-attenuation-db="$LOOPBACK_ATTENUATION_DB" \
            --radio-expected-count="$EXPECTED_RADIOS" \
            --radio-gain-observation-interval=2048 \
            --radio-gain-observation-capacity=256 \
            --radio-report-dir="${REPORT_ROOT}/candidate-v3-tx-report-epoch-${epoch}"
        run_logged "post-tx-mute-epoch-${epoch}" \
            "$PYTHON" -m spf.scripts.mute_pluto_tx \
            --expected-count "$EXPECTED_RADIOS" \
            --output "${REPORT_ROOT}/post-tx-mute-epoch-${epoch}.json"
    done
else
    run_logged ram-load \
        sudo -n "$PYTHON" "$MULTI_LOADER" load-all \
        "${common_loader_args[@]}"
    run_logged post-load-tx-mute \
        "$PYTHON" -m spf.scripts.mute_pluto_tx \
        --expected-count "$EXPECTED_RADIOS" \
        --output "${REPORT_ROOT}/post-load-tx-mute.json"
    run_logged iio-after-ram-load iio_info -s
fi

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
    "${TEST_FILE}::test_v3_repeated_fresh_usb_starts" \
    "${TEST_FILE}::test_v3_simultaneous_usb_streams" \
    --radio-hardware \
    --radio-gain-series-v3 \
    --radio-expected-count="$EXPECTED_RADIOS" \
    --radio-samples=32768 \
    --radio-frames-per-request=3 \
    --radio-cycles="$STARTUP_STRESS_CYCLES" \
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
    resolved_direct_ip_host="$(
        "$PYTHON" -m spf.scripts.resolve_pluto_ip \
            --serial "$direct_ip_serial" \
            --preferred-host "$DIRECT_IP_HOST"
    )"
    # Do not use grep -q here.  With pipefail enabled it closes the pipe after
    # the first match, lsusb receives SIGPIPE, and a genuine match becomes a
    # false failure (status 141).
    lsusb -d 0456:b673 -v 2>/dev/null | grep -F "$direct_ip_serial" >/dev/null ||
        die "direct-IP serial $direct_ip_serial is not an attached USB radio"
    printf 'direct_ip_host_requested=%s direct_ip_host_resolved=%s direct_ip_serial=%s\n' \
        "$DIRECT_IP_HOST" "$resolved_direct_ip_host" "$direct_ip_serial"
    run_logged candidate-v3-direct-ip \
        "$PYTHON" -m pytest -q \
        "${TEST_FILE}::test_v3_direct_ip_survives_malformed_control_datagrams" \
        "${TEST_FILE}::test_v3_direct_ip_uses_the_same_inner_frame" \
        --radio-hardware \
        --radio-gain-series-v3 \
        --radio-direct-ip \
        --radio-direct-ip-host="$resolved_direct_ip_host" \
        --radio-samples=524288 \
        --radio-gain-observation-interval=2048 \
        --radio-gain-observation-capacity=256 \
        --radio-report-dir="${REPORT_ROOT}/candidate-v3-ip-report"
    run_logged candidate-v3-direct-ip-buffered-burst \
        "$PYTHON" -m pytest -q \
        "${TEST_FILE}::test_v3_direct_ip_buffers_a_maximum_finite_burst" \
        --radio-hardware \
        --radio-gain-series-v3 \
        --radio-direct-ip \
        --radio-direct-ip-host="$resolved_direct_ip_host" \
        --radio-samples=524288 \
        --radio-frames-per-request=16 \
        --radio-cycles="$IP_BURN_IN_CYCLES" \
        --radio-gain-observation-interval=2048 \
        --radio-gain-observation-capacity=256 \
        --radio-direct-ip-min-payload-mibps=20 \
        --radio-direct-ip-min-receive-buffer-mib=64 \
        --radio-report-dir="${REPORT_ROOT}/candidate-v3-ip-burst-report"
fi

run_logged final-status \
    sudo -n "$PYTHON" "$MULTI_LOADER" status-all \
    "${common_loader_args[@]}"
printf '\nPASS: protocol-v3 RAM candidate completed every requested gate.\n'
printf 'Artifacts: %s\n' "$REPORT_ROOT"
printf 'QSPI was not modified.\n'
