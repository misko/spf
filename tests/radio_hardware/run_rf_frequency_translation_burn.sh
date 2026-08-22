#!/usr/bin/env bash
# USB-only fixed-emitter/swept-RX-LO burn-in with fail-closed TX cleanup.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT
readonly PYTHON="${SPF_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
readonly DURATION_SECONDS="${SPF_RF_TRANSLATION_DURATION_SECONDS:-3600}"
readonly EXPECTED_COUNT="${SPF_RF_TRANSLATION_EXPECTED_RADIOS:-4}"
readonly REPORT_DIR="${1:-/tmp/spf-rf-frequency-translation-$(date -u +%Y%m%dT%H%M%SZ)}"

LIBIIO_SOURCE="${SPF_LIBIIO_SOURCE:-${REPO_ROOT}/../libiio}"
LIBIIO_BUILD="${SPF_LIBIIO_BUILD:-${LIBIIO_SOURCE}/build-tandem-host}"
if [[ -f "${LIBIIO_SOURCE}/bindings/python/iio.py" && -f "${LIBIIO_BUILD}/libiio.so" ]]; then
    export PYTHONPATH="${LIBIIO_SOURCE}/bindings/python${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${LIBIIO_BUILD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "$REPORT_DIR"
cd "$REPO_ROOT"

"$PYTHON" -m spf.scripts.rf_frequency_translation_burn \
    --expected-count "$EXPECTED_COUNT" \
    --duration-seconds "$DURATION_SECONDS" \
    --physical-attenuation-db "${SPF_RF_TRANSLATION_ATTENUATION_DB:-0}" \
    --tx-gain-db "${SPF_RF_TRANSLATION_TX_GAIN_DB:--30}" \
    --report "$REPORT_DIR/rf-frequency-translation-burn.json" \
    2>&1 | tee "$REPORT_DIR/run.log"
