# Reproduce the PDF's plots and accuracy locally

Date: 2026-11-08

Status: Historical reproduction has been investigated. The user has authorized the first service milestone below. Core behavior changes still require discussion before editing.

## Current service milestone

Add a local historical inference job service and a viewer for saved results under `spf/inference_service/`. Reuse `plot_filter_run.open_dataset`, `run_filter`, and `summarize_run`. Save predictions and scores from the same execution. Start with dual-radio EKF, empirical PF, and neural PF under an explicitly labeled current-code profile.

Run scientific jobs in isolated subprocesses with separate outputs and prediction caches. Persist requests, effective parameters, seeds, input provenance, progress, and terminal job state. Let the browser submit, inspect, cancel, and play back completed results. Playback is a display of saved results, not evidence of streaming inference equivalence.

Verify the HTTP path against the six original center recordings. The reference aggregate EKF MSE is `1.6775933362061464` radians squared, using equal weight per recording. Also exercise empirical and neural PF on the original diamond recording, browser playback, cancellation, and recovery after restart.

The earlier historical tests reproduced all 120 EKF heatmap cells to four decimal places. PF scores vary with calibration and resampling state; the exact original calibration identity and NumPy state remain unresolved. Preserve those limits when displaying historical comparisons.

After this milestone, evaluate warm-model execution and feature replay through existing interfaces. Reliable raw replay and live filtering remain gated on separately discussed pairing and incremental-filter changes. Raspberry Pi deployment remains outside scope.

## Goal and scope

Recreate the bearing-distribution plots, temporal tracking plots, and accuracy comparisons presented in the March 2025 SPF PDF on `kalman`, using existing code and artifacts wherever possible. Establish what can be reproduced exactly, explain discrepancies, and then compare recent checkpoints using the same evaluation procedure.

Source document: `Droning on 2__An update of sorts (2).pdf`, supplied in this chat. The local attachment is `/home/mouse9911/.codex/attachments/89e955a3-934b-4475-a897-485274beff16/Droning on 2__An update of sorts (2).pdf`.

Raspberry Pi 5 deployment, live collection integration, streaming repairs, and real-time performance work are outside the current scope. A new inference framework is not a prerequisite. Use existing offline inference, filter runners, and plotting code first.

Truly held-out ROVER evaluation remains a later extension. Historical reproduction may require recordings already used in development. Identify that use honestly rather than blocking reproduction or describing those recordings as a new holdout.

## Aggregate development evaluation

Use the frozen rosters in `experiments/inference_eval_v1/cohorts.json` for repeatable aggregate comparisons. `rover_tune16_v1` balances the three array spacings across five training days. `rover_devcheck11_v1` contains all eleven August 7 checkpoint-validation windows. `rover_all48_v1` supports descriptive reporting across the release after settings are selected.

All three are development cohorts. Neural training or checkpoint selection and empirical calibration overlap these recordings. The train/check groups have no shared raw source captures, but windows within each group can share sources. Preserve that distinction when reporting score spread.

The local service supports named-cohort evaluations and two-axis sweeps through the same existing filter and scoring functions. Require every recording, seed, and configuration to finish with finite scores in one angular frame before publishing a heatmap. Average recordings equally within each seed, then average seeds. Report capture spread and seed spread separately, without claiming independent-session confidence intervals.

Use craft-relative neural predictions for comparisons with the dual-radio empirical PF and EKF. Keep other hyperparameters fixed within a heatmap. Start with the supplied small verification grids, then expand the wider presets on tuning data. Check selected settings separately on development-check data. No filter, preprocessing, model, or core metric implementation change is required for this workflow.

## Preserve existing behavior

Treat existing preprocessing, model inference, postprocessing, and filter calculations as frozen experimental behavior. A bug fix can change an experiment. Discuss any required core change with the user before editing it.

Do not copy or reimplement segmentation, normalization, model loading, coordinate transforms, or filter mathematics. Do not refactor shared code to simplify plotting. First try existing entry points and supported options. Add a small separate orchestration or plotting script only if existing tools cannot produce the required artifact, and explain its necessity before implementation.

Preserve datasets, checkpoints, empirical tables, and existing caches. Write future results to a separate explicit destination. Inspect loader side effects before execution because some loaders regenerate artifacts or delete damaged files. Record the code revision and relevant local modifications; a commit hash alone does not identify the working tree.

## 1. Map each target figure to its evidence

- [ ] Inventory the target panels and captions in the PDF.
- [ ] Locate original figures, notebooks, plotting scripts, result files, configurations, checkpoints, and recording lists.
- [ ] Record each figure's source path, capture identities, frame range, model, calibration table, parameters, seed, units, coordinate convention, and metric definition.
- [ ] Mark unavailable evidence as unknown. Do not infer exact settings from a plot's appearance.

Prioritize these figure families:

| PDF pages | Target | Evidence needed |
| --- | --- | --- |
| 20–21 | Beamformer, single-radio NN, and paired-radio bearing distributions | Exact examples, angular bins, normalization, model, and coordinate conventions |
| 23–24 | Bearing estimates over a rover trajectory, with uncertainty | Recordings, frame or time axis, ground truth, filter settings, and definition of displayed uncertainty |
| 25–26 | Neural particle-filter, empirical particle-filter, and EKF error heatmaps | Sweep axes, effective parameters, capture cohort, score aggregation, seed handling, and metric units |

Page 26 describes rover diamond while the EKF plot labels rover center. Resolve that mismatch from original artifacts if possible. Otherwise preserve the uncertainty and avoid treating the displayed panels as a matched algorithm comparison.

Deliverable: a figure inventory that labels each target as recoverable, partially recoverable, or missing required evidence.

## 2. Reproduce one figure before scaling up

- [ ] Choose the target with the most complete provenance and a small recording or frame range.
- [ ] Trace its existing offline loading, preprocessing, inference, postprocessing, scoring, and plotting calls.
- [ ] Record checkpoint and configuration hashes, empirical-table lineage, preprocessing version, dependencies, device, precision, and seeds.
- [ ] Invoke the existing path with its original effective settings and preserve its outputs.
- [ ] Repeat as needed to distinguish expected stochastic variation from unexplained differences.

Start with `spf/dataset/spf_dataset.py`, `spf/dataset/spf_nn_dataset_wrapper.py`, `spf/filters/run_filters_on_data.py`, and `spf/evaluation/metrics.py`. Locate and reuse the actual plotting entry points during the audit. Do not assume the current filter runner matches the historical plotting path.

Capture intermediate values only through existing interfaces. If a diagnostic requires a core edit, discuss it first. Reference outputs establish what the implementation does, not whether it is correct.

Deliverable: one regenerated figure, its underlying numerical output, a reproducible invocation, and a comparison against the PDF or recovered original artifact.

Completion condition: explain the agreement or discrepancy before expanding to other figures. Do not tune parameters merely to reach a number shown in the PDF.

## 3. Recreate the remaining plots and reported scores

- [ ] Reuse the proven invocation pattern for the remaining distribution and trajectory plots.
- [ ] Recover original sweep settings before launching the heatmap runs.
- [ ] Record requested and effective parameter values separately where they differ.
- [ ] Keep capture identities and frame correspondence explicit. Report missing or mismatched outputs instead of silently truncating, resampling, or repairing them.
- [ ] Save underlying arrays and score tables beside every figure, along with the invocation and provenance.
- [ ] Match axes, units, angle wrapping, uncertainty definitions, and aggregation before comparing numerical results.

Preserve fixed time steps and other historical assumptions for reproduction. Separate folded single-radio angles from full-circle bearings. Neural distribution loss is not angular mean squared error. If reporting angular RMSE in degrees alongside MSE in radians squared, document the conversion and keep the original metric visible.

For stochastic particle filters, record seeds and repeated-run variation. Recover the historical aggregation before reproducing it; report additional per-session summaries separately. Do not replace a historical metric silently with a preferred one.

Deliverable: the recreated figure set and a table containing the PDF value, reproduced value, metric definition, data cohort, and explanation of any mismatch.

## 4. Label the strength of each reproduction claim

Use one of these labels for every result:

- **Historical reproduction:** the original inputs, model, calibration, effective settings, and scoring are recovered. Document any numerical tolerance and its basis.
- **Comparable reconstruction:** some original artifacts are unavailable. State each substitution and compare only the aspects that remain equivalent.
- **New experiment:** a newer model, different cohort, changed algorithm behavior, or different metric is used. A similar-looking plot does not make this a historical reproduction.

Do not set an invented accuracy threshold before recovering the original metric and its variability. A successful investigation can establish that an exact number is not reproducible with available evidence. Explain that limitation instead of modifying behavior until the number matches.

## 5. Compare recent models after establishing the reference

- [ ] Evaluate recent checkpoints on the same recovered cohort and with the same supported plotting and scoring procedure.
- [ ] Keep these results separate from the historical reproduction.
- [ ] Identify which artifacts changed and which stayed fixed.
- [ ] Compare unfiltered neural predictions, empirical methods, EKF, and neural or empirical particle filters only where their current implementations support a valid comparison.

The reviewed filter report uses a June checkpoint, while newer models are from August. This model difference must be explicit. Preserve existing effective behavior; do not bundle filter fixes into a model comparison.

Deliverable: side-by-side historical and recent-model figures with traceable scores and clear limits on any improvement claim.

## 6. Extend to a genuine holdout afterward

Audit candidate recordings against training, checkpoint selection, empirical calibration, and filter tuning. Trace merged recordings to raw captures. Freeze whole sessions and evaluation choices before examining final performance.

The newer validation recordings already influenced checkpoint selection. The reviewed empirical-table build included the 48 merged ROVER captures. Those facts prevent treating those captures as an independent final test for the affected artifacts.

If no existing cohort qualifies, record the need for new data as a follow-up. It does not block historical reproduction. Reuse the established figure and scoring procedure when an eligible cohort becomes available.

## Discuss necessary core changes before editing

For each proposed change, show the blocked figure or comparison, the exact function, the smallest edit, and why existing interfaces are insufficient. Explain which intermediate values and scores could change and how historical behavior will remain reproducible. Wait for the user's agreement before editing core behavior.

Known evaluation limitations remain explicit:

- The single-radio EKF runner accepts a noise setting but passes a fixed value. Record the effective value. Do not present this as a functioning sweep dimension or silently repair it.
- Filter runners use fixed time steps. Preserve them for historical reproduction. Any acquisition-time extension is a separate experiment.
- The XY EKF initializes from transmitter truth and scores a position component as an angle. Keep it out of bearing-score comparisons and deployable localization claims.
- Previously observed streaming pairing and collection issues are deferred. They do not justify changes to reproduce offline figures.

Do not conceal fixes inside adapters, plotting code, or score correction. Preserve suspicious historical results as historical evidence with limitations; do not promote them into valid accuracy claims.

## First milestone and completion evidence

The first milestone is the figure inventory and a concrete proposal for reproducing one well-supported panel through existing code. No new framework is required to reach this milestone.

The reproduction work is complete when every selected figure has a regenerated artifact and underlying scores, or a documented evidence gap. Every reported number must identify its data, model, calibration, effective settings, metric, and reproduction label. Recent-model comparisons follow the established reference and remain separately labeled.

This revision changes the plan only. It does not claim that any plots or accuracy results have been reproduced yet.
