#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
python_bin="${SPF_PYTHON:-${repo_root}/../spf-virtualenv/bin/python}"
artifact_base="${SPF_CALIBRATION_ARTIFACT_ROOT:-${repo_root}/artifacts/dual_rx_gain_frequency/power_cycle}"
config="${script_dir}/configs/power_cycle_subsample.yaml"
expected_radios="${SPF_EXPECTED_RADIOS:-2}"

usage() {
  command_name="$(basename -- "$0")"
  echo "Usage:"
  echo "  ${command_name} capture-before EXPERIMENT_ID"
  echo "  ${command_name} capture-after  EXPERIMENT_ID CYCLE --confirmed-power-cycle"
  echo "  ${command_name} compare        EXPERIMENT_ID CYCLE"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

mode="$1"
experiment_id="$2"
if [[ ! "${experiment_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXPERIMENT_ID may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter is not executable: ${python_bin}" >&2
  exit 2
fi

experiment_root="${artifact_base}/${experiment_id}"

case "${mode}" in
  capture-before)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    "${python_bin}" -m spf.calibrations.dual_rx_gain_frequency automate \
      --config "${config}" \
      --output "${experiment_root}/before" \
      --expected-radios "${expected_radios}"
    echo
    echo "Before capture complete."
    echo "Keep all RF cables fixed. Remove power from every Pluto for at least 10 seconds."
    ;;
  capture-after)
    if [[ $# -ne 4 || "$4" != "--confirmed-power-cycle" ]]; then
      usage
      echo "The after label requires an explicit cold-power-cycle confirmation." >&2
      exit 2
    fi
    cycle="$3"
    if [[ ! "${cycle}" =~ ^[1-9][0-9]*$ ]]; then
      echo "CYCLE must be a positive integer" >&2
      exit 2
    fi
    if [[ ! -f "${experiment_root}/before/automation_result.json" ]]; then
      echo "The before capture is missing for ${experiment_id}" >&2
      exit 1
    fi
    if ! grep -q '"status": "complete"' "${experiment_root}/before/automation_result.json"; then
      echo "The before capture is not complete for ${experiment_id}" >&2
      exit 1
    fi
    "${python_bin}" -m spf.calibrations.dual_rx_gain_frequency automate \
      --config "${config}" \
      --output "${experiment_root}/after_cycle_${cycle}" \
      --expected-radios "${expected_radios}"
    ;;
  compare)
    if [[ $# -ne 3 ]]; then
      usage
      exit 2
    fi
    cycle="$3"
    if [[ ! "${cycle}" =~ ^[1-9][0-9]*$ ]]; then
      echo "CYCLE must be a positive integer" >&2
      exit 2
    fi
    "${python_bin}" -m spf.calibrations.dual_rx_gain_frequency \
      compare-power-cycles \
      --before-root "${experiment_root}/before" \
      --after-root "${experiment_root}/after_cycle_${cycle}" \
      --output-dir "${experiment_root}/comparison_cycle_${cycle}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
