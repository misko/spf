#!/usr/bin/env bash
# Regenerate every figure in the 2026-07-31 field report.
#
# Reads the captures READ-ONLY (raw data is immutable). Produces, per capture, a
# 6-panel summary; and per merged dataset, a direction-finding figure (builds the
# segmentation/beamformer precompute cache on first run).
#
# Env overrides:
#   PY     python from the spf virtualenv        (default: python)
#   DATA   captures dir                          (default: /mnt/qnap01/mouse9911/rovers_july_2026)
#   CACHE  segmentation precompute cache dir      (default: $DATA/precompute_cache_v7)
set -uo pipefail

PY=${PY:-python}
DATA=${DATA:-/mnt/qnap01/mouse9911/rovers_july_2026}
HERE="$(cd "$(dirname "$0")" && pwd)"
FIG="$HERE/../2026_07_31_figures"
CACHE=${CACHE:-$DATA/precompute_cache_v7}
mkdir -p "$FIG"

echo "== per-zarr 6-panel summaries (raw + merged) =="
for f in "$DATA"/*.zarr "$DATA"/*.zarr.tmp "$DATA"/merged/*.zarr; do
    [ -d "$f" ] || continue
    case "$f" in *precompute*) continue ;; esac
    s=$(basename "$f" | sed 's/\.zarr\(\.tmp\)\?$//')
    $PY "$HERE/zarr_summary_figure.py" "$f" "$FIG/$s.png" "/tmp/$s.json" || true
done

echo "== direction-finding figures (merged datasets; builds cache if missing) =="
for f in "$DATA"/merged/*.zarr; do
    [ -d "$f" ] || continue
    p="${f%.zarr}"; s=$(basename "$p")
    $PY "$HERE/merged_df_figure.py" "$p" "$FIG/df_$s.png" "$CACHE" || true
done

echo "== done -> $FIG =="
