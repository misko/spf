#!/usr/bin/env bash
# Repeatedly run the receive-only interrupted-capture campaign for a bounded
# unattended interval. Each round remains independently inspectable and ends
# with a strict clean-recovery capture.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly CAMPAIGN="${SPF_INTERRUPT_CAMPAIGN:-${SCRIPT_DIR}/run_interrupted_capture_campaign.sh}"
readonly DURATION_SECONDS="${SPF_INTERRUPT_SOAK_SECONDS:-43200}"
readonly MAX_ROUNDS="${SPF_INTERRUPT_SOAK_MAX_ROUNDS:-1000}"
readonly MIN_FREE_GIB="${SPF_INTERRUPT_SOAK_MIN_FREE_GIB:-25}"
readonly OUTPUT_ROOT="${SPF_INTERRUPT_SOAK_OUTPUT_ROOT:-/home/pi/preflight/interrupted_capture_soak}"
readonly CLEAN_RECORDS="${SPF_INTERRUPT_SOAK_CLEAN_RECORDS:-100}"
readonly DRY_RUN="${SPF_INTERRUPT_SOAK_DRY_RUN:-0}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for value in "$DURATION_SECONDS" "$MAX_ROUNDS" "$MIN_FREE_GIB" "$CLEAN_RECORDS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "numeric controls must be non-negative integers"
done
(( MAX_ROUNDS > 0 )) || die "SPF_INTERRUPT_SOAK_MAX_ROUNDS must be positive"
(( CLEAN_RECORDS > 0 )) || die "SPF_INTERRUPT_SOAK_CLEAN_RECORDS must be positive"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] ||
    die "SPF_INTERRUPT_SOAK_DRY_RUN must be 0 or 1"
[[ -x "$CAMPAIGN" ]] || die "interruption campaign is not executable: ${CAMPAIGN}"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${OUTPUT_ROOT}/${run_id}"
stop_file="${run_root}/STOP"
summary="${run_root}/rounds.tsv"
mkdir -p "$run_root"
printf 'round\tcases\tstarted_unix\tfinished_unix\tstatus\tartifact_kib\n' >"$summary"
printf '%s\n' \
    "duration_seconds=${DURATION_SECONDS}" \
    "max_rounds=${MAX_ROUNDS}" \
    "minimum_free_gib=${MIN_FREE_GIB}" \
    "clean_records=${CLEAN_RECORDS}" \
    "stop_file=${stop_file}" \
    "dry_run=${DRY_RUN}" >"${run_root}/settings.env"

on_signal() {
    printf 'STOPPED_BY_SIGNAL\n' >"${run_root}/STOPPED"
    exit 128
}
trap on_signal INT TERM HUP

matrices=(
    'sigterm:1 sigint:8 sigkill:32 sigterm:128'
    'sigkill:2 sigterm:16 sigint:64 sigkill:192'
    'sigint:3 sigkill:24 sigterm:96 sigint:224'
    'sigterm:5 sigkill:40 sigint:160 sigterm:256'
)

started_epoch="$(date +%s)"
deadline_epoch=$((started_epoch + DURATION_SECONDS))
round=0
while (( round < MAX_ROUNDS )); do
    now="$(date +%s)"
    if (( round > 0 && now >= deadline_epoch )); then
        break
    fi
    if [[ -e "$stop_file" ]]; then
        printf 'STOPPED_BY_FILE\n' >"${run_root}/STOPPED"
        break
    fi

    available_kib="$(df -Pk "$run_root" | awk 'NR == 2 {print $4}')"
    required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    if (( available_kib < required_kib )); then
        printf 'LOW_DISK: available_kib=%s required_kib=%s\n' \
            "$available_kib" "$required_kib" >"${run_root}/FAILED"
        die "free disk fell below ${MIN_FREE_GIB} GiB"
    fi

    round=$((round + 1))
    cases="${matrices[$(((round - 1) % ${#matrices[@]}))]}"
    round_root="${run_root}/round-$(printf '%03d' "$round")"
    mkdir -p "$round_root"
    round_started="$(date +%s)"

    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%s\n' "$cases" >"${round_root}/planned-cases.txt"
        status=0
    else
        set +e
        SPF_INTERRUPT_OUTPUT_ROOT="$round_root" \
        SPF_INTERRUPT_CASES="$cases" \
        SPF_INTERRUPT_CLEAN_RECORDS="$CLEAN_RECORDS" \
            "$CAMPAIGN" >"${round_root}/campaign.log" 2>&1
        status=$?
        set -e
    fi

    round_finished="$(date +%s)"
    artifact_kib="$(du -sk "$round_root" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$round" "$cases" "$round_started" "$round_finished" "$status" \
        "$artifact_kib" >>"$summary"

    if (( status != 0 )); then
        printf 'ROUND_FAILED: round=%s status=%s\n' "$round" "$status" \
            >"${run_root}/FAILED"
        exit "$status"
    fi
done

finished_epoch="$(date +%s)"
printf '%s\n' \
    "rounds_completed=${round}" \
    "started_unix=${started_epoch}" \
    "finished_unix=${finished_epoch}" \
    "elapsed_seconds=$((finished_epoch - started_epoch))" \
    >"${run_root}/result.env"
printf 'PASS\n' >"${run_root}/PASS"
printf 'PASS: unattended interruption soak completed: %s\n' "$run_root"
