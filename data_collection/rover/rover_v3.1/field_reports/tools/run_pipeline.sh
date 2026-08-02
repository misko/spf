#!/usr/bin/env bash
# End-to-end rover field-data pipeline: merge -> precompute -> figures -> metrics.
#
# This is the exact sequence used to produce the 2026-08-01 report appendix.
# Every step reads the recorded captures READ-ONLY; nothing is ever written back
# into the raw capture directory.
#
# ---------------------------------------------------------------------------
# WHERE EACH STEP'S FILES LIVE
#
#   step 0  raw captures (immutable, input)
#           $RAW/*.zarr[.tmp]            one dir per collection launch
#           $RAW/*.yaml[.tmp]            the capture config for each
#
#   step 1  merged TX/RX datasets        <- v7_tx_rx_merge.py
#           $MERGED/<rx>.<tx>.zarr       one per overlapping emitter/receiver pair
#           $MERGED/<rx>.<tx>.yaml       config copied from the RX capture
#
#   step 2  segmentation / beamformer    <- segment_zarr.py
#           $CACHE/<name>_segmentation_nthetas65.pkl    per-window segmentation
#           $CACHE/<name>_segmentation_nthetas65.yarr   windowed_beamformer,
#                                                       mean_phase, window stats
#           NOTE: cache filenames encode dataset + nthetas but NOT the
#           segmentation version, so $CACHE must be version-scoped (…_3p7).
#
#   step 3  figures                      <- zarr_summary_figure.py, merged_df_figure.py
#           $FIG/<capture>.png           6-panel per-capture summary
#           $FIG/df_<merged>.png         beamformer + estimated theta vs ground truth
#
#   step 4  metrics                      <- df_metrics.py
#           $METRICS                     per-dataset DF quality / signal strength JSON
#           (the markdown table in the report is printed by --markdown)
# ---------------------------------------------------------------------------
#
# Usage (defaults reproduce the 2026-08-01 appendix):
#   tools/run_pipeline.sh                 # all steps
#   STEPS=1,2 tools/run_pipeline.sh       # only merge + precompute
set -uo pipefail

PY=${PY:-/home/mouse9911/virtual-envs/spf/bin/python3}
REPO=${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
DATE=${DATE:-2026_08_01}
BASE=${BASE:-/mnt/qnap01/mouse9911/rovers_august_2026}
RAW=${RAW:-$BASE/aug1}
MERGED=${MERGED:-$BASE/merged}
CACHE=${CACHE:-/mnt/qnap01/mouse9911/precomputed/precompute_cache_3p7}
HERE="$(cd "$(dirname "$0")" && pwd)"
FIG=${FIG:-$HERE/../${DATE}_figures}
METRICS=${METRICS:-$HERE/../${DATE}_metrics.json}
MIN_TIMESTEPS=${MIN_TIMESTEPS:-300}
STEPS=${STEPS:-1,2,3,4}

echo "repo=$REPO"
echo "raw=$RAW  merged=$MERGED"
echo "cache=$CACHE"
echo "fig=$FIG  metrics=$METRICS"
has() { [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"

# --- step 1: merge TX (emitter) x RX (receivers) -----------------------------
# Pairs every emitter capture against every receiver capture. Pairs that do not
# overlap in GPS time, or that cannot be read, are skipped in ~2 s each, so this
# doubles as the overlap map. Add --dry-run to get the map without copying IQ.
if has 1; then
  echo "=== [1] merge -> $MERGED ==="
  mkdir -p "$MERGED"
  LOGLEVEL=INFO $PY -m spf.scripts.v7_tx_rx_merge \
      --txs "$RAW"/*_tag_RO2.zarr \
      --rxs "$RAW"/*_tag_RO1.zarr* "$RAW"/*_tag_RO3.zarr* \
      --output "$MERGED" --min-timesteps "$MIN_TIMESTEPS"
  echo "    merged datasets: $(ls -d "$MERGED"/*.zarr 2>/dev/null | wc -l)"
fi

# --- step 2: segmentation + beamformer precompute ---------------------------
# Computing to a LOCAL dir first is much faster and avoids writing LMDB over NFS;
# set CACHE directly to the network path to skip the staging copy.
if has 2; then
  echo "=== [2] precompute -> $CACHE ==="
  STAGE=${STAGE:-$HOME/spf_cache/$(basename "$CACHE")}
  mkdir -p "$STAGE"
  $PY spf/scripts/segment_zarr.py -i "$MERGED"/*.zarr -c "$STAGE" --gpu -p 8
  mkdir -p "$CACHE"
  rsync -a "$STAGE"/ "$CACHE"/
  echo "    cached datasets: $(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)  ($(du -sh "$CACHE" | cut -f1))"
fi

# --- step 3: figures --------------------------------------------------------
if has 3; then
  echo "=== [3] figures -> $FIG ==="
  PY=$PY DATE=$DATE RAW="$RAW" MERGED="$MERGED" CACHE="$CACHE" FIG="$FIG" \
      bash "$HERE/generate_figures.sh"
  echo "    figures: $(ls "$FIG"/*.png 2>/dev/null | wc -l)"
fi

# --- step 4: DF quality / signal-strength metrics ---------------------------
if has 4; then
  echo "=== [4] metrics -> $METRICS ==="
  $PY "$HERE/df_metrics.py" "$MERGED" "$CACHE" "$METRICS" --markdown
fi

echo "=== pipeline done ==="
