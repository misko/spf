# E-INF2 — exhaustive hyperparameter survey

**Status: DESIGNED, NOT RUN.** Hypotheses and decision rules are recorded here
before any of it executes.

## Why

[E-INF1](../e_inf1_filter_sweep/experiment_readme.md) tuned on a grid that
**truncated six of seven families**. Measured on its own committed results, taking
each parameter's marginal profile (best achievable MSE at each value, minimising
over the others):

| family [frame] | param | winner | edge | slope to neighbour |
|---|---|---|---|---|
| `PF single radio NN` [folded] | `theta_dot_err` | 0.1 | **high** | still −20% |
| `PF dual radio` [craft] | `theta_dot_err` | 0.1 | **high** | still −8% |
| `PF dual radio NN` [craft] | `theta_dot_err` | 0.1 | **high** | still −7% |
| `PF single radio` [folded] | `theta_dot_err` | 0.1 | **high** | still −7% |
| `EKF single radio` [folded] | `phi_std` | 1.0 | **low** | still −33% |
| `EKF dual radio` [craft] | `phi_std` | 0.0 | **low** | still −9% |
| `PF single radio (NN)` [folded] | `N` | 1024 | **low** | flat |

MSE is still falling steeply into the wall on four particle-filter families, so
the available gain is plausibly comparable to the effect E-INF1's H1 measured
(NN beats the empirical table by 20–44%). **Tuning is not a second-order concern
on this corpus.**

Two structural problems beyond the boundaries:

1. **The grid is a resolution regression.** The March 2025 "Droning on 2 (SPF)"
   deck swept **13** `theta_dot_err` values against **8** particle counts.
   `rover2026_coarse.yaml` sweeps **4** against 2–3. That deck's surface is *not*
   monotonic — a broad optimum at 0.001–0.005 with a sharp cliff below — so a
   four-point axis can step over a minimum entirely.
2. **`run_EKF_single_theta_single_radio` was never offered the `dynamic_R`
   form.** The shipped configs pair `phi_std>0` with `dynamic_R=0` and
   `phi_std=0` with `dynamic_R>0`; the dual-radio EKF gets both blocks and its
   winner uses the second. The single-radio EKF only ever saw the first. That is
   a missing region, not a boundary, and is the most likely reason that family
   looks mediocre.

## Design

Three phases. Phase A maps the surface widely and cheaply; phase B spends seeds
only where it matters; phase C confirms on the full corpus.

| phase | what | grid | seeds | datasets | jobs | est. |
|---|---|---|---|---|---|---|
| **A — survey** | map the surface, produce the heatmaps | `N` ×9, `theta_dot_err` ×12, `theta_err` ×3; EKF full factorial `phi_std` ×9 × `noise_std` ×6 × `p` ×3 × `dynamic_R` ×3 | 2 | 8 | 33,696 | ~55 min |
| **B — refine** | dense local grid around each family's phase-A optimum | 5×5×5 per family | 5 | 16 | ~50,000 | ~80 min |
| **C — confirm** | phase-B winners on the whole corpus | 1 config/family | 5 | 48 | ~1,300 | ~4 min |

Total ≈ **2.5 h** at the quiet-box rate measured this session (10.3 jobs/s at
`--parallel 22`; peak 14.5).

**Phase B's grid cannot be written in advance** — it is centred on what phase A
finds. That is the point of splitting them, and the config for B is a deliverable
*of* A.

### Why 2 seeds and 8 datasets in phase A

Phase A ranks *regions*; it does not pick a winner. Seed-to-seed spread on the
corpus mean was measured at 0.2–3.4% per configuration, far below the 7–20% gaps
being resolved. Phase B re-runs the surviving region at 5 seeds on all 16
datasets, which is where statistical care belongs. Halving both axes in A buys a
4× wider grid for the same cost.

## Pre-registered hypotheses

| id | prediction | decision rule if supported |
|---|---|---|
| **H1** | Every PF family's optimum is **interior** in the phase-A grid | The extended ranges are sufficient; phase B refines locally. If any optimum is still on an edge, extend that axis again before phase B — do not refine against a wall. |
| **H2** | At least one PF family improves **≥10%** on its E-INF1 corpus-mean MSE | E-INF1's stage-2/3 leaderboards are superseded as *tuning* results; the H1 (NN vs empirical) comparison must be re-decided at matched tuning, since it was measured on a truncated grid. |
| **H3** | Optimal `theta_dot_err` differs by **≥5×** between `craft_relative` and `absolute_north` for the NN dual-radio filter | Confirms the frame difference is a dynamics/process-model effect, not a bug — and that per-frame tuning is mandatory, not optional. |
| **H4** | `EKF single radio` with the `dynamic_R` form beats its E-INF1 winner (1.022) | The family was mis-specified rather than weak; its E-INF1 ranking is not evidence about EKFs. |
| **H5** | The `phi_std>0 ⇄ dynamic_R=0` pairing convention is **not** required — some crossed cell wins | The shipped configs encode a convention nobody tested; drop it. |

**Falsifiers.** H1 fails if any PF optimum sits on an edge. H2 fails if every
family is within 10% of its E-INF1 number — in which case the truncation was real
but harmless, and that is worth recording as much as the alternative. H3 fails if
the two frames' optima are within 5×, which would undercut the dynamics
explanation in the [stage-3 report](../../spf/filters/reports/e_inf1_rover_full_20260810_v1/REPORT.md).

## Acceptance gates

| artifact | gate |
|---|---|
| phase-A results | every PF config present at both seeds; `n_runs` sums to 8 per config+seed |
| heatmaps | one per family; **every edge-optimum reported in the tool's truncation summary** |
| phase-B results | every config at all 5 seeds; `n_runs` sums to 16 per config+seed |
| phase-C results | 48 datasets, and per-store paired stats vs the E-INF1 winner — **not** a bare corpus mean |
| `RESULTS.md` | states H1–H5 with numbers, including any falsified |

⚠️ The phase-C gate exists because of a mistake made in E-INF1: a 0.039 corpus-mean
difference was reported as a ranking when paired testing showed **p = 0.780** and
92% of it came from one capture. **Any claim that A beats B needs the paired test,
not the mean.**

## Risks

| risk | mitigation |
|---|---|
| The optimum is routine-dependent, so these numbers do not transfer | Recorded as a finding, not a bug: `theta_dot_err` 0.1 is *worst* on `rover_diamond` (2025 deck) and *best* on our `rover_bounce`. Any new routine must re-tune this axis. Phase C reports per-spacing and per-routine, never a single number. |
| Larger `theta_dot_err` wins by flattening the posterior rather than modelling motion | Phase B records the calibration block (`calib_std_z`, coverage) alongside MSE. A configuration that improves MSE while `std(z)` worsens is buying accuracy with dishonesty and must be flagged. |
| Cost blows out | Phase A is bounded at 33,696 jobs. Run in batches with `--resume`; each batch is independently useful. |
| Work dir collision destroys committed results | Every phase gets its own new work dir. `--resume` defaults to true, so an existing dir silently skips; `--no-resume` would overwrite by key. |

## Inputs

| | |
|---|---|
| datasets | phase A: first 8 of [`stage2_rover_sample_n16.txt`](../e_inf1_filter_sweep/stage2_rover_sample_n16.txt); B: all 16; C: all 48 |
| empirical table | `empirical_dists/full_20260809_v1.pkl` |
| model | `/mnt/md0/checkpoints/jun26_2026/paired_3p7_thin_noblade/best.pth` |
| segmentation | 3.7, precompute `/mnt/qnap01/mouse9911/rovers_2026/precompute` |
| phase-A config | [`rover2026_survey.yaml`](../../spf/filters/configs/rover2026_survey.yaml) |
| heatmaps | [`plot_hyperparam_heatmap.py`](../../spf/filters/plot_hyperparam_heatmap.py) |

Nothing reads or writes `/mnt/md2`, and no historical/validation data is involved
— consistent with the decision recorded in
[E-INF1 RESULTS](../e_inf1_filter_sweep/RESULTS.md) to evaluate on rover 2026 only.
