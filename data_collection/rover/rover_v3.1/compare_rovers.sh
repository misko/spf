#!/usr/bin/env bash
#
# Diff two rovers' fingerprints. READ-ONLY on both.
#
# Runs audit_rover.sh over ssh on each host and splits the result:
#
#   identity.*  expected to differ (rover_id, hostname, address, rest offset,
#               tag, capture config) - reported, never flagged
#   fleet.*     should match - any difference is DRIFT and is flagged
#
# Exit status: 0 if no fleet drift, 1 if drift found. Suitable for a gate.
#
# Usage:
#   ./compare_rovers.sh 192.168.1.41 192.168.1.44
#   ./compare_rovers.sh 192.168.1.41 192.168.1.44 --allow fleet.stock.ModemManager
#
set -uo pipefail

die() { printf 'compare_rovers: %s\n' "$*" >&2; exit 2; }

[[ $# -ge 2 ]] || die "usage: $0 HOST_A HOST_B [--allow KEY]..."
host_a="$1"; host_b="$2"; shift 2

allow=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow) allow+=("${2:-}"); shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done

REMOTE_SCRIPT="/home/pi/spf/data_collection/rover/rover_v3.1/audit_rover.sh"
tmp_a="$(mktemp)"; tmp_b="$(mktemp)"
trap 'rm -f "$tmp_a" "$tmp_b"' EXIT

fetch() {
    local host="$1" out="$2"
    timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "pi@${host}" \
        "bash ${REMOTE_SCRIPT}" >"$out" 2>/dev/null \
        || die "could not audit ${host} (unreachable, or ${REMOTE_SCRIPT} missing - git pull on that rover)"
    [[ -s "$out" ]] || die "empty fingerprint from ${host}"
}

fetch "$host_a" "$tmp_a"
fetch "$host_b" "$tmp_b"

is_allowed() {
    local key="$1"
    for a in ${allow[@]+"${allow[@]}"}; do [[ "$key" == "$a" ]] && return 0; done
    return 1
}

printf '=== identity (expected to differ) ===\n'
printf '%-34s %-30s %s\n' "key" "$host_a" "$host_b"
while IFS='=' read -r key val_a; do
    [[ "$key" == identity.* ]] || continue
    val_b="$(grep -m1 "^${key}=" "$tmp_b" | cut -d= -f2-)"
    printf '%-34s %-30s %s\n' "$key" "${val_a:-<none>}" "${val_b:-<none>}"
done <"$tmp_a"

printf '\n=== fleet (must match) ===\n'
drift=0
while IFS='=' read -r key val_a; do
    [[ "$key" == fleet.* ]] || continue
    val_b="$(grep -m1 "^${key}=" "$tmp_b" | cut -d= -f2-)"
    if [[ "$val_a" == "$val_b" ]]; then
        continue
    elif is_allowed "$key"; then
        printf '  ALLOWED  %-30s %-28s %s\n' "$key" "$val_a" "$val_b"
    else
        printf '  DRIFT    %-30s %-28s %s\n' "$key" "${val_a:-<none>}" "${val_b:-<none>}"
        drift=$((drift + 1))
    fi
done <"$tmp_a"

# Keys present on B but absent on A are drift too.
while IFS='=' read -r key val_b; do
    [[ "$key" == fleet.* ]] || continue
    grep -q "^${key}=" "$tmp_a" && continue
    is_allowed "$key" && continue
    printf '  DRIFT    %-30s %-28s %s\n' "$key" "<absent>" "$val_b"
    drift=$((drift + 1))
done <"$tmp_b"

printf '\n'
if [[ $drift -eq 0 ]]; then
    printf 'PASS: no fleet drift between %s and %s\n' "$host_a" "$host_b"
    exit 0
fi
printf 'FAIL: %d fleet field(s) differ between %s and %s\n' "$drift" "$host_a" "$host_b"
exit 1
