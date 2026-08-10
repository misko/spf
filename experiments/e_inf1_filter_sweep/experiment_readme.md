# E-INF1 — how do our best models track, locally, across three corpora?

**Status:** designed 2026-08-08, not yet run. Tooling landed in changes 1–4
(`spf/filters/resample.py`, `spf/evaluation/`, `spf/filters/report.py`,
`spf/filters/plot_filter_run.py`, `spf/filters/configs/`).
**Results:** [`RESULTS.md`](RESULTS.md) _(written when it runs)_
**Est. compute:** ~8 h wall on the 24-core box. **No cloud.**

This is a computational experiment, so the `experiments/README.md` "hardware
setup" section does not apply — there is no bench, no radio, no schematic. The
inputs are recorded datasets and trained checkpoints, all read-only.

---

## 1. Purpose

We have never measured how the current models plus the EKF/PF trackers perform
on the **2026 rover corpus**, and the historical numbers we do have came from a
sweep that was (a) run on cloud infrastructure we are no longer using and (b)
built on a particle filter whose results were not reproducible.

Three things are unknown:

1. **Which filter family and hyperparameters actually win**, per corpus, now
   that runs are seeded and repeatable.
2. **How far the 2026 rover data is out of distribution.** Its arrays run at
   d/λ = 0.673 / 0.827 / 0.904 — all past the λ/2 unambiguous limit, so the
   arrays are spatially aliased. The models take d/λ as an input, but whether
   they extrapolate there is untested.
3. **Whether the reported confidence means anything.** A single pilot run put
   ±1σ coverage at 25.6% against 68.3% nominal. If that holds across the sweep,
   nothing downstream can gate on filter variance.

## 2. Hypotheses (pre-registered)

- **H1 — NN beats empirical on rover.** The NN dual-radio PF has lower craft-relative
  MSE than the empirical dual-radio PF on the rover corpus, because the empirical
  `P(θ|φ)` table was built from wall-array data at different spacings.
- **H2 — the rover corpus is materially harder.** Best-config craft-relative MSE
  on rover is at least 2× the best on the frozen val set.
- **H3 — filters are overconfident everywhere, not just in the pilot.** Median
  `std(z)` across the best 10 configs is > 1.5 on every corpus.
- **H4 — the aliased spacings are the problem.** Within the rover corpus, MSE is
  worse at d/λ = 0.904 than at 0.673, for the same filter and hyperparameters.

## 3. Approach

Four stages. Each is a gate: do not start the next until the previous passes.

| Stage | What | Cost |
|---|---|---|
| 0 | segment the rover corpus at v3.7 (`segment_zarr.py`, **unchanged**) | ~2 h GPU, ~1.3 GB |
| 1 | build inference caches on **GPU** (`create_inference_cache.py --device cuda`) | rover 160 s; val 2.1 h |
| 2 | coarse grid (348 configs) on a **stratified 16-dataset sample** per corpus, **5 seeds** | ~1.3 h wall per corpus |
| 3 | top ~4 configs per family on the **full** corpora, 1 seed | ~1 h val, ~15 min rover |

Stage 2 uses seeds because a 16-dataset average only cuts the 42–106% per-dataset
PF spread by √16 = 4×. Stage 3 does not, because 565 datasets cut it by √565 ≈ 24×,
leaving ~1.7% residual — below any effect worth acting on.

**Stratification for the sample:** by (routine × d/λ × capture day) for rover, and
by (vehicle × routine × band × d/λ) for the frozen val set. The sample list is
written to the report directory so the stage-2 result is reproducible.

### Controls

- **Seeded runs.** Since change 1, `seed` fully determines a PF run; the sweep
  records it.
- **Frames never pooled.** `absolute=True` and `absolute=False` are scored against
  different ground truth and are kept in separate frames end to end.
- **Posterior scored separately from the tracker.** `spf/evaluation/posterior.py`
  scores the network's `P(θ)` with no filter, so H1/H2 can be attributed to the
  model or to the tracker rather than confounded.
- **The frozen val set is untouched.** It is read-only; no new val view is created.

## 4. Software setup

```bash
# stage 0 -- segmentation code is NOT modified; it is used as-is
python spf/scripts/segment_zarr.py -i <merged>.zarr \
    -c /mnt/qnap01/mouse9911/rovers_2026/precompute --gpu --workers 2

# stage 1 -- GPU, batch >= 64. CPU at batch 16 is 42x slower (13 vs 543 sess/s)
python spf/scripts/create_inference_cache.py --device cuda --parallel 2 \
    --config-fn <ckpt>/config.yml --checkpoint-fn <ckpt>/best.pth \
    --inference-cache /mnt/qnap01/.../inference_cache \
    --precompute-cache <precompute> --segmentation-version 3.7 -d <datasets>

# stage 2/3 -- 24 workers, not 30: throughput peaks at 24 and regresses at 30
python spf/filters/run_filters_on_data.py -d <datasets> \
    --empirical-pkl-fn empirical_dists/full_20260809_v1.pkl \
    --work-dir /mnt/qnap01/.../filter_runs/<stage> \
    --config spf/filters/configs/<corpus>_coarse.yaml \
    --results-backend local --parallel 24

python spf/filters/report.py --work-dir <workdir> \
    --output-dir spf/filters/reports/e_inf1_<corpus>_<date>_v1
```

Model: `/mnt/md0/checkpoints/jun26_2026/paired_3p7_thin_noblade/best.pth`
(the newest paired checkpoint, 2026-07-04). Environment:
`~/virtual-envs/spf/bin/python3`.

## 5. Outputs and acceptance gates

| Artifact | Location | Gate |
|---|---|---|
| rover precompute | `rovers_2026/precompute/` | all 48 stores segment without error |
| inference caches | `rovers_2026/inference_cache/` | one `.npz` per dataset per model |
| stage-2 reports | `spf/filters/reports/e_inf1_<corpus>_coarse_<date>_v1/` | every PF config appears at all 5 seeds; per config+seed, `n_runs` sums to 16 across spacing groups ‡ |
| stage-3 reports | `spf/filters/reports/e_inf1_<corpus>_full_<date>_v1/` | same, summing to 48 ‡ |
| per-dataset figures | qnap01 (gitignored) | ≥1 figure per corpus visually checked per CLAUDE.md |
| `RESULTS.md` | this directory | states H1–H4 outcomes with numbers |

‡ **Amended 2026-08-10.** These two gates originally read "every config has
`n_runs = 5`" and "`n_runs` equals the corpus size". Both are unmeetable as
written, and the error is in the measurement definition, not in the results:
`n_runs` counts the **datasets averaged into a group**, while `rx_wavelength_spacing`
and `seed` are themselves grouping keys. So `n_runs` is the dataset count at one
spacing — on the stage-2 corpus that is 1, 2, 5 or 6, never 5-meaning-seeds and
never 16 or 48. Measured over the committed stage-2 report:
`Counter({1: 4896, 2: 1632, 5: 1632, 6: 1632})` across 9,792 rows.

The amended gates test the two things the originals were reaching for — that no
seed is missing, and that every dataset in the corpus contributed — using
quantities the report actually carries. The stage-2 report satisfies the amended
gates. Recorded here rather than silently corrected because this file is the
pre-registration.

## 6. Decision rules (pre-registered)

- **H1** — if NN craft-relative MSE is not below empirical on rover, the NN
  advantage does not survive the domain shift and retraining with 2026 rover data
  becomes the priority over further filter tuning.
- **H2** — if rover MSE is within 2× of val, the 2026 corpus is usable as-is for
  evaluation. If it is far worse, diagnose with the posterior scores first
  (stage 3 emits them): a bad posterior means the model, a good posterior with a
  bad track means the tracker.
- **H3** — if `std(z) > 1.5` holds, **no downstream component may gate on filter
  variance** until the cause is found, and that becomes its own work item.
- **H4** — if d/λ = 0.904 is materially worse **after conditioning on frame yield**
  (see Risks: RO4 yields 36.9% against 65–76%), the RO4 rovers need re-spacing
  before further capture, and the 9 RO4 stores — 6 at d/λ 0.90397 plus 3 at
  0.91557 — are excluded from training. A raw MSE gap alone does not settle it.

## 7. Risks

| Risk | Catch |
|---|---|
| ~~RO4 stores have no empirical-table entry → `KeyError`~~ | ✅ **RESOLVED** (`2a07ae0`). It was **17 of 48** stores, not 3: d/λ derives from spacing **and** carrier, and the 2026 fleet added both 5840 MHz and 0.047 m. `empirical_dists/full_20260809_v1.pkl` covers all 48, verified live per key. **Use that table, not `full.pkl`.** |
| **H4 is confounded with RO4 signal quality** | RO4 yields **36.9%** valid (θ,φ) frames against 65–76% for the other spacings, so "d/λ = 0.904 is worse" and "RO4 captures half the signal" are not separable by MSE alone. Report per-key **yield alongside** MSE; treat H4 as unresolved unless the gap survives conditioning on yield. |
| Two empirical keys changed under the new table | `PLUTO_0.82703` (24→43 datasets, corr 0.937 vs the old table) and `PLUTO_0.67317` (21→33, corr 0.989) now include 2026 rover data. Empirical-filter numbers at those spacings are **not** comparable to any result predating `2a07ae0`. |
| The O4 emitter is bursty (60–70% NaN is healthy) so mean-over-all-frames metrics measure the silence | report `median_abs_err` alongside the mean; both are in `metrics.summarize` |
| Stage 2 sample is unrepresentative | stratify, and record the sample list in the report |
| Another session is using the box | timings are not a gate here; only stage-2/3 wall clock is affected |
| Filenames are unreliable for ordering/attribution (19 captures carry a restored-clock timestamp) | group by `routine` and `gps_timestamp`, never by filename — `spf/filters/labels.py` already prefers the recorded routine |
