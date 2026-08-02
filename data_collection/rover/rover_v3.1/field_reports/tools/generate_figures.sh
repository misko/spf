#!/usr/bin/env bash
# Regenerate the figures for a rover field report.
#
# Reads the captures READ-ONLY (raw data is immutable). Produces, per capture, a
# 6-panel summary; and per merged dataset, a direction-finding figure (builds the
# segmentation/beamformer precompute cache on first run).
#
# Env overrides:
#   PY      python from the spf virtualenv       (default: python)
#   DATE    report date -> picks the figure dir  (default: 2026_07_31)
#   RAW     dir holding the raw captures         (default: /mnt/qnap01/mouse9911/rovers_july_2026)
#   MERGED  dir holding the merged datasets      (default: $RAW/merged)
#   FIG     output figure dir                    (default: <tools>/../${DATE}_figures)
#   CACHE   segmentation precompute cache dir    (default: $MERGED/../precompute_cache_v7)
#
# Examples:
#   # July 2026 (defaults)
#   tools/generate_figures.sh
#
#   # August 2026 (raw lives in an aug1/ subdir; shared version-scoped cache)
#   DATE=2026_08_01 \
#   RAW=/mnt/qnap01/mouse9911/rovers_august_2026/aug1 \
#   MERGED=/mnt/qnap01/mouse9911/rovers_august_2026/merged \
#   CACHE=/mnt/qnap01/mouse9911/precomputed/precompute_cache_3p7 \
#   tools/generate_figures.sh
set -uo pipefail

PY=${PY:-python}
DATE=${DATE:-2026_07_31}
RAW=${RAW:-/mnt/qnap01/mouse9911/rovers_july_2026}
MERGED=${MERGED:-$RAW/merged}
HERE="$(cd "$(dirname "$0")" && pwd)"
FIG=${FIG:-$HERE/../${DATE}_figures}
CACHE=${CACHE:-$MERGED/../precompute_cache_v7}
mkdir -p "$FIG"

echo "== raw=$RAW"
echo "== merged=$MERGED"
echo "== fig=$FIG"
echo "== cache=$CACHE"

echo "== per-zarr 6-panel summaries (raw + merged) =="
for f in "$RAW"/*.zarr "$RAW"/*.zarr.tmp "$MERGED"/*.zarr; do
    [ -d "$f" ] || continue
    case "$f" in *precompute*) continue ;; esac
    s=$(basename "$f" | sed 's/\.zarr\(\.tmp\)\?$//')
    $PY "$HERE/zarr_summary_figure.py" "$f" "$FIG/$s.png" "/tmp/$s.json" || true
done

echo "== direction-finding figures (merged datasets; builds cache if missing) =="
for f in "$MERGED"/*.zarr; do
    [ -d "$f" ] || continue
    p="${f%.zarr}"; s=$(basename "$p")
    $PY "$HERE/merged_df_figure.py" "$p" "$FIG/df_$s.png" "$CACHE" || true
done

echo "== done -> $FIG =="
