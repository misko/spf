# E-INF2 — results

**Status: NOT RUN.** Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | every PF family's optimum is interior in the phase-A grid | _pending_ |
| H2 | ≥1 PF family improves ≥10% on its E-INF1 corpus mean | _pending_ |
| H3 | optimal `theta_dot_err` differs ≥5× between craft_relative and absolute_north | _pending_ |
| H4 | `EKF single radio` with `dynamic_R` beats its E-INF1 winner (1.022) | _pending_ |
| H5 | the `phi_std>0 ⇄ dynamic_R=0` pairing convention is not required | _pending_ |

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
