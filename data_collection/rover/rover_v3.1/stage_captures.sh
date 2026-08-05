#!/usr/bin/env bash
#
# stage_captures.sh — copy captures off a rover WITH their sidecars.
#
# A recorded session is not one file. It is a family:
#
#     <capture>.zarr    the data (a directory: an LMDB store)
#     <capture>.yaml    the capture config — routine, antenna spacing, tag
#     <capture>.log     the run log
#
# and while a run is unfinalized all three carry a `.tmp` suffix, renamed
# together when it completes (ROVER_RUNBOOK §12.2).
#
# WHY THIS SCRIPT EXISTS. On 2026-08-04 a staging copy took only the `.zarr`
# directories. v7_tx_rx_merge.py reads the RX capture's `.yaml` for the
# receiver/antenna config, so the merge died — after it had already paired every
# TX against every RX — on a FileNotFoundError naming a single sidecar. The
# sidecars were still on the rovers and had to be fetched by hand. `cp *.zarr`
# is the natural thing to type and it is wrong; this is the thing to type
# instead, and it cannot leave a family member behind.
#
# The copy runs from the BASE STATION and pulls: there is no rsync or scp on a
# rover (ROVER_RUNBOOK §12.2), so offload is never driven from the Pi.
#
# The source is opened READ-ONLY. Nothing is ever written back to a rover or into
# a recorded capture; --delete is deliberately not offered.
#
# usage:
#   stage_captures.sh --from <SRC> --to <DEST> [--match GLOB] [--dry-run]
#
#   --from   pi@roverpi1:temp        a rover (anything rsync/ssh accepts), or
#            /mnt/qnap01/.../aug4    a local/mounted directory
#   --to     staging directory, created if absent
#   --match  glob over CAPTURE NAMES, default '*'. Note this matches the name
#            WITHOUT the .zarr/.yaml/.log extension, so '*tag_RO1*' selects the
#            whole family of every RO1 capture.
#   --dry-run  list the families and what would be copied; copy nothing.
#
# examples:
#   stage_captures.sh --from pi@roverpi1:temp --to /mnt/qnap01/.../aug4
#   stage_captures.sh --from pi@roverpi3:temp --to "$STAGE" --match '*2026_08_04*'
#
# Exit status is non-zero if any staged capture is missing its `.zarr` or `.yaml`.
# A capture with no `.yaml` can still serve as a TX (emitter) — the merge reads
# only the RX sidecar, never the TX one — but it is data lost either way, so it
# is reported rather than passed over.

set -uo pipefail

SRC=""
DEST=""
MATCH='*'
DRY_RUN=0

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

usage() {
    # The header comment IS the usage text; it ends at the line before `set -u`.
    sed -n "3,$(($(grep -n '^set -uo' "$0" | cut -d: -f1) - 2))p" "$0" |
        sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) SRC="${2:-}"; shift 2 ;;
        --to)   DEST="${2:-}"; shift 2 ;;
        --match) MATCH="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        *) printf '\nERROR: unknown option: %s\n' "$1" >&2; usage 2 ;;
    esac
done

[[ -n "$SRC" ]] || { printf '\nERROR: --from is required\n' >&2; usage 2; }
[[ -n "$DEST" ]] || { printf '\nERROR: --to is required\n' >&2; usage 2; }

# A remote source is host:path. Match a colon that is not part of a path, so
# ./weird:dir and absolute paths stay local.
REMOTE_HOST=""
REMOTE_DIR=""
if [[ "$SRC" == *:* && "$SRC" != /* && "$SRC" != .* ]]; then
    REMOTE_HOST="${SRC%%:*}"
    REMOTE_DIR="${SRC#*:}"
    [[ -n "$REMOTE_DIR" ]] || REMOTE_DIR="."
fi

# List the names in the source directory, one per line. Read-only in both modes.
list_source() {
    if [[ -n "$REMOTE_HOST" ]]; then
        ssh -o BatchMode=yes "$REMOTE_HOST" "ls -1 -- '${REMOTE_DIR}'" ||
            die "cannot list ${SRC} (ssh failed)"
    else
        [[ -d "$SRC" ]] || die "no such source directory: ${SRC}"
        ls -1 -- "$SRC"
    fi
}

# --------------------------------------------------------------- families ---
#
# Group names into capture families. The key carries the .tmp state as well as
# the capture name, because an unfinalized .zarr.tmp belongs with .yaml.tmp and
# NOT with a stray finalized .yaml -- pairing those would silently attach the
# wrong config to a capture.

declare -A FAMILY_BASE=()   # key -> capture name, no extension
declare -A FAMILY_TMP=()    # key -> "" or ".tmp"
declare -A FAMILY_HAS=()    # "key ext" -> 1

record_member() {
    local name="$1" base tmp ext stem
    case "$name" in
        *.zarr.tmp|*.yaml.tmp|*.log.tmp)
            tmp=".tmp"; stem="${name%.tmp}" ;;
        *.zarr|*.yaml|*.log)
            tmp=""; stem="$name" ;;
        *) return 0 ;;          # not part of a capture family
    esac
    ext="${stem##*.}"
    base="${stem%.*}"
    local key="${base}${tmp}"
    FAMILY_BASE["$key"]="$base"
    FAMILY_TMP["$key"]="$tmp"
    FAMILY_HAS["$key $ext"]=1
}

printf '\n== source ==\n'
note "from : ${SRC}"
note "to   : ${DEST}"
note "match: ${MATCH}"

# Substitution, not a pipe or a process substitution: `die` inside list_source
# runs in a subshell, so its exit status has to be checked here or an
# unreachable rover would look like an empty directory and stage nothing
# "successfully".
if ! SOURCE_LISTING="$(list_source)"; then
    exit 1
fi
while IFS= read -r name; do
    [[ -n "$name" ]] && record_member "$name"
done <<<"$SOURCE_LISTING"

# Selected families, in a stable order.
SELECTED=()
for key in "${!FAMILY_BASE[@]}"; do
    # shellcheck disable=SC2053  # the glob on the right is intentional
    [[ "${FAMILY_BASE[$key]}" == $MATCH ]] && SELECTED+=("$key")
done
if [[ ${#SELECTED[@]} -gt 0 ]]; then
    mapfile -t SELECTED < <(printf '%s\n' "${SELECTED[@]}" | sort)
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    printf '\n'
    warn "no captures in ${SRC} matched '${MATCH}' — nothing to stage"
    printf '\n'
    exit 0
fi

# ------------------------------------------------------------- what to copy ---

FILES=()
INCOMPLETE_AT_SOURCE=0
printf '\n== %d capture(s) ==\n' "${#SELECTED[@]}"
for key in "${SELECTED[@]}"; do
    base="${FAMILY_BASE[$key]}"
    tmp="${FAMILY_TMP[$key]}"
    present=()
    absent=()
    for ext in zarr yaml log; do
        if [[ -n "${FAMILY_HAS["$key $ext"]:-}" ]]; then
            present+=(".${ext}")
            FILES+=("${base}.${ext}${tmp}")
        else
            absent+=(".${ext}")
        fi
    done
    if [[ ${#absent[@]} -eq 0 ]]; then
        ok "${base}${tmp}  (.zarr .yaml .log)"
    else
        INCOMPLETE_AT_SOURCE=1
        # A member absent at the SOURCE cannot be staged -- say so here rather
        # than let it look like a copy failure below.
        if [[ " ${absent[*]} " == *" .yaml "* ]]; then
            bad "${base}${tmp}  has ${present[*]} but NO .yaml at the source"
            note "    only usable as a TX (emitter); no merge can read it as an RX"
        else
            warn "${base}${tmp}  has ${present[*]}, missing ${absent[*]} at the source"
        fi
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '\n== dry run: %d file(s) would be copied ==\n' "${#FILES[@]}"
    printf '  %s\n' "${FILES[@]}"
    printf '\n'
    exit 0
fi

# --------------------------------------------------------------------- copy ---

mkdir -p "$DEST" || die "cannot create staging directory: ${DEST}"

printf '\n== copying %d file(s) ==\n' "${#FILES[@]}"
# --files-from keeps the whole family in ONE rsync, so a partial copy is visible
# as a single non-zero status instead of per-file guesswork. No --delete: the
# staging directory may already hold captures from another rover.
#
# -r is passed EXPLICITLY. rsync documents that --files-from cancels the -r that
# -a would normally imply, and a .zarr is a directory: without this the store
# arrives as an empty directory and the failure only shows up much later, when
# something tries to open it. That is the same class of bug this script exists
# to prevent, so it is verified at the destination below.
rsync_src="$SRC"
[[ "$rsync_src" == */ ]] || rsync_src="${rsync_src}/"
if ! printf '%s\n' "${FILES[@]}" |
        rsync -a -r --info=stats1 --files-from=- "$rsync_src" "$DEST/"; then
    die "rsync failed copying ${SRC} -> ${DEST}"
fi

# ------------------------------------------------------------------- verify ---
#
# Trusting rsync's exit status is not enough: the whole point of this script is
# that the family arrives complete, so check it at the destination.

printf '\n== staged ==\n'
staging_failed=0
for key in "${SELECTED[@]}"; do
    base="${FAMILY_BASE[$key]}"
    tmp="${FAMILY_TMP[$key]}"
    absent=()
    for ext in zarr yaml log; do
        [[ -e "${DEST}/${base}.${ext}${tmp}" ]] || absent+=(".${ext}")
    done
    fatal=0
    # Report the .yaml first: it is the member the merge cannot run without, and
    # the one the 2026-08-04 staging copy dropped.
    if [[ " ${absent[*]} " == *" .yaml "* ]]; then
        bad "${base}${tmp}  NO .yaml — usable as a TX (emitter) only;"
        note "    v7_tx_rx_merge.py reads the RX .yaml and cannot use this as an RX"
        fatal=1
    fi
    if [[ " ${absent[*]} " == *" .zarr "* ]]; then
        bad "${base}${tmp}  NO .zarr — nothing was staged"
        fatal=1
    # An empty .zarr at the destination means the store directory was created
    # but not recursed into; see the -r note above.
    elif [[ -z "$(ls -A "${DEST}/${base}.zarr${tmp}" 2>/dev/null)" ]]; then
        bad "${base}${tmp}  .zarr staged EMPTY — rerun; do not merge this"
        fatal=1
    fi
    if [[ " ${absent[*]} " == *" .log "* ]]; then
        warn "${base}${tmp}  no .log — the run log is lost, but the merge still works"
    fi
    if [[ "$fatal" -ne 0 ]]; then
        staging_failed=1
    elif [[ ${#absent[@]} -eq 0 ]]; then
        ok "${base}${tmp}"
    fi
done

printf '\n'
if [[ "$staging_failed" -ne 0 ]]; then
    bad "some captures arrived incomplete (see above); fetch the rest from the rover.
      A capture with no .yaml can still serve as a TX -- the merge reads only the
      RX sidecar -- so this is not always a blocker, but it is always data lost."
    printf '\n'
    exit 1
fi
if [[ "$INCOMPLETE_AT_SOURCE" -ne 0 ]]; then
    warn "every .zarr/.yaml is present, but something was incomplete at the source"
fi
ok "every staged capture has its .zarr and .yaml"
printf '\n'
