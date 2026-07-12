#!/bin/bash
# Rebuild the dataset audit PDF.
#   ./rebuild.sh            fast: pinned metrics_v2.csv -> out/quality_report.pdf
#   ./rebuild.sh --rescan   slow: re-scan the fleet first (hours), then build
set -e
cd "$(dirname "$0")"
PY=/home/mouse9911/virtual-envs/spf/bin/python
mkdir -p out

CSV=metrics_v2.csv
if [ "$1" == "--rescan" ]; then
  $PY dataset_quality_scan.py \
      --splits /mnt/md2/splits/apr17_train_nosig_noroverbounce_noblade.txt \
               /mnt/md2/splits/apr17_val_nosig_noroverbounce.txt \
      --precompute-cache /mnt/md2/cache/precompute_cache_3p7 \
      --output-dir out/rescan
  CSV=out/rescan/metrics.csv
fi

$PY dataset_quality_report_pdf.py --csv "$CSV" --out out/quality_report.pdf
echo "wrote out/quality_report.pdf"
