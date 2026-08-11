# E-INF2 — results

**Status: PHASE A COMPLETE** (2026-08-10). Phases B and C not run. Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | every PF family's optimum is interior in the phase-A grid | ✅ **SUPPORTED** — zero truncated axes, down from eight |
| H2 | ≥1 PF family improves ≥10% on its E-INF1 corpus mean | ❌ **FALSIFIED** — best consistent gain 3.0% (8/8); the one −10.1% is better on 2/8 |
| H3 | optimal `theta_dot_err` differs ≥5× between craft_relative and absolute_north | ✅ **SUPPORTED** — 0.1 vs 0.005, a 20× ratio |
| H4 | `EKF single radio` with `dynamic_R` beats its E-INF1 winner (1.022) | ❌ **REFUTED** — both EKF optima use `dynamic_R = 0` |
| ~~H5~~ | ~~the `phi_std>0 ⇄ dynamic_R=0` pairing convention is not required~~ | ❌ **REFUTED before running** — the convention is a guard: (0,0) gives a singular update, and dynamic_R>0 ignores phi_std entirely |

## Baselines to beat (E-INF1, corpus mean over 16 stores, 5 seeds)

| family | frame | MSE | source |
|---|---|---|---|
| `PF_single_theta_single_radio_NN` | `radio_folded` | 0.301 | [stage 2](../../spf/filters/reports/e_inf1_rover_coarse_20260809_v1/REPORT.md) |
| `PF_single_theta_dual_radio_NN` | `absolute_north` | 0.534 | " |
| `PF_single_theta_single_radio` | `radio_folded` | 0.541 | " |
| `PF_single_theta_dual_radio_NN` | `craft_relative` | 0.667 | " |
| `PF_single_theta_dual_radio` | `craft_relative` | 0.838 | " |
| `EKF_single_theta_single_radio` | `radio_folded` | 1.022 | " |
| `EKF_single_theta_dual_radio` | `craft_relative` | 2.583 | " |
| _uniform random_ | — | _3.290_ | π²/3 |

Every one of these was measured on a grid now known to be truncated for its
family, so they are upper bounds on what the filter can do, not measurements of
the filter.

## Method note carried forward from E-INF1

Any claim that one configuration or frame beats another must be supported by a
**paired per-store test**, not a corpus-mean difference. E-INF1 reported a 0.039
gap between two frames as a ranking; paired testing later gave Wilcoxon
**p = 0.780**, with 92% of the gap coming from a single capture out of 48. That
correction is recorded in the
[stage-3 report](../../spf/filters/reports/e_inf1_rover_full_20260810_v1/REPORT.md).

## Phase A outcome (2026-08-10)

Full write-up:
[`spf/filters/reports/e_inf2_survey_20260810_v1/REPORT.md`](../../spf/filters/reports/e_inf2_survey_20260810_v1/REPORT.md).
28,800 runs, 3,600 configurations, 8 captures, 2 seeds, zero failures, ~75 min.

**The tuning gain is cost, not accuracy.** Matched to the same datasets and
seeds, six of seven families move by less than 1%. What the wide grid found is
that particle count is over-provisioned: `PF dual radio` matches its E-INF1
winner at **N = 4096 instead of 16384 — 2.7× faster** (0.834 vs 0.861, better on
5/8 datasets). Dropping further to 256 may also be free but the per-dataset split
runs against the mean, so it is recorded as ambiguous.

H2's falsification was pre-registered as a real outcome and is treated as one:
boundary-hitting optima were a genuine defect, and fixing them showed the
truncation was harmless. An optimum on an edge is worth checking and usually is
not worth much.

⚠️ **Phase A's grid was not a superset of E-INF1's**, though it was designed as
though it were: `theta_err` was narrowed from five values to three to buy width
elsewhere. The judgement came from a *marginal* profile, which is minimised over
the other axes and therefore hides the interactions a 2-D heatmap exists to
show — the error the tool was built to prevent. Phase B restores full coverage.
