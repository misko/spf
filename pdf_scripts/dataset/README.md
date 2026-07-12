# Dataset audit PDF — rebuild kit

Everything needed to rebuild `quality_report.pdf` (the full fleet dataset-quality audit,
~64 pages: wall/rover breakdowns, bad-months / stale-positions / NaN sections, per-file
appendix).

## Contents

| File | Role |
|---|---|
| `dataset_quality_scan.py` | Fleet scanner (scan_v=2). Read-only over all zarr datasets; emits one row per dataset into `metrics.csv` (forward-model phase residuals, 3-param systematic fit (g, Δθ, c), circular stats, NaN/frozen-tail/timestamp gates, status OK/FLAG/QUARANTINE/ERROR + reasons). |
| `dataset_quality_report_pdf.py` | Report generator: `metrics.csv` → `quality_report.pdf` (matplotlib, no LaTeX needed). Sections 1–14 + per-file appendix. |
| `metrics_v2.csv` | Pinned scan output used for the shipped report (2,250 datasets, scan of 2026-07-12). Rebuilding the PDF from this file reproduces the report **exactly**, no data access needed. |
| `make_v2_splits.py` | Downstream consumer: turns `metrics.csv` into the v2scan train/val split manifests (documented here because the report and the splits must stay in sync). |
| `rebuild.sh` | One-command rebuild (fast path + full-rescan path). |

Canonical copies of the scripts live in `spf/scripts/`; the ones here are pinned with the
report. If you change the canonical ones, re-copy and rebuild.

## Rebuild

Fast (from the pinned metrics, ~2 min, no dataset access):

```bash
./rebuild.sh                      # -> out/quality_report.pdf
```

Full re-scan (hours; walks every zarr in /mnt/md2/cache/nosig_data):

```bash
./rebuild.sh --rescan             # regenerates metrics.csv first
```

## Verification

The report generator is verified for occlusion by exporting pages to PNG
(`pdftoppm -png -r 60`) and inspecting — footers, annotations, and appendix pagination
were all bug-fixed against rendered output, so re-verify after any layout change.

## Environment

`/home/mouse9911/virtual-envs/spf/bin/python` (needs pandas, numpy, matplotlib; the
rescan path additionally needs the spf package + zarr datasets mounted).
