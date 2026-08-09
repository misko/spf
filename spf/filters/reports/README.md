# Filter sweep reports

Small, reviewable outputs from `spf/filters/report.py`. Follows the append-only
convention already used by `spf/calibrations/*/reports/`: one directory per run,
named `<name>_<YYYYMMDD>_v1`, never edited once written — a re-run gets a new
directory.

Each directory holds:

| File | Contents |
|---|---|
| `REPORT.md` | per-frame leaderboard, the reviewable summary |
| `results.json` | every aggregated row, machine-readable |
| `inputs_manifest.json` | dataset list, config, checkpoint + config md5s, git commit |

## Why not CSV

`.gitignore` matches `**/*.csv` repo-wide (only the JLC fab files are excepted),
so `run_filters_report.py`'s CSV output can never be committed. That is the
reason `report.py` exists and writes JSON + Markdown instead.

Bulk output — the per-job `results.pkl` under the work dir, per-dataset PNGs —
stays out of git entirely. Put it on qnap01. Note `.gitignore` already reserves
`**/filter_reports` and `**/run_on_filters_results`, so those names are
always ignored if you reach for them.

## Reading a report

Rows are grouped by every non-metric field **and by angular frame**, and each
frame gets its own table. A `craft_relative` row and an `absolute_north` row of
the same hyperparameters are answers to different questions — see
`spf/evaluation/frames.py`. Never rank across the two.

`n_runs` is the denominator behind every averaged number on a row. A row with
`n_runs = 1` on a particle filter is one draw from a distribution whose
per-dataset spread was measured at 42–106% of the mean; treat it accordingly.

## Index

_(none yet — the first sweep lands here)_
