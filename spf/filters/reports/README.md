# Filter sweep reports

Small, reviewable outputs from `spf/filters/report.py`. Follows the append-only
convention already used by `spf/calibrations/*/reports/`: one directory per run,
named `<name>_<YYYYMMDD>_v1`, never edited once written — a re-run gets a new
directory.

Each directory holds:

| File | Contents | Written by |
|---|---|---|
| `REPORT.md` | the findings, written by hand, with figures | a person |
| `LEADERBOARD.md` | per-frame leaderboard of every group | `report.py` |
| `results.json` | every aggregated row, machine-readable | `report.py` |
| `figures/` | PNGs referenced by `REPORT.md` | `plot_sweep_summary.py`, `plot_trajectory_comparison.py` |
| `inputs_manifest.json` | dataset list, config, checkpoint + config md5s, git commit | the sweep |

`report.py` deliberately does **not** write `REPORT.md`: the generated table and
the hand-written narrative share a directory, and regenerating the table after an
edit would otherwise destroy the write-up.

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

## Reading a figure

Every MSE panel is drawn against the **uniform-random floor**, π²/3 = 3.290 rad²
— what a uniform guess scores on the circle. Without it an MSE of 2.6 reads as a
number rather than "21% of the way from guessing to perfect".

The per-capture trajectory figure adds a second floor, the **best constant**: the
single fixed bearing with the lowest MSE on that capture. A filter that does not
beat it has learned nothing about *time*. On a folded frame that floor is far
below the random one, and filters have been observed between the two — so
skill-vs-random alone is not enough.

Both live in `spf/evaluation/metrics.py` (`baselines`, `skill_vs_random`).

## Index

| Report | Corpus | Headline |
|---|---|---|
| [`e_inf1_rover_coarse_20260809_v1`](e_inf1_rover_coarse_20260809_v1/REPORT.md) | 16 merged-v7 rover captures | H1 supported (NN beats empirical by 20–44%); EKF dual-radio near-uninformative; H3 unanswerable as run |
