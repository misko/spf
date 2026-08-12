# E-INF2 — results

**Status: COMPLETE** (2026-08-11). Phases A and B run; phase C judged not worth running (see below). Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | every PF family's optimum is interior in the phase-A grid | ✅ **SUPPORTED** — zero truncated axes in both phases |
| H2 | ≥1 PF family improves ≥10% on its E-INF1 corpus mean | ❌ **FALSIFIED** — matched on 16 captures × 5 seeds, max change **1.6%**; four families reproduce E-INF1 exactly |
| H3 | optimal `theta_dot_err` differs ≥5× between craft_relative and absolute_north | ✅ **SUPPORTED** — 0.1 vs 0.005, a 20× ratio |
| H4 | `EKF single radio` with `dynamic_R` beats its E-INF1 winner (1.022) | ❌ **REFUTED** — the static form wins and the family lands 1.0% *worse* (1.032) |
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

## Phase B outcome — E-INF2 COMPLETE (2026-08-11)

Full write-up:
[`spf/filters/reports/e_inf2_refine_20260811_v1/REPORT.md`](../../spf/filters/reports/e_inf2_refine_20260811_v1/REPORT.md).
50,864 runs, 16 captures, 5 seeds, zero failures, ~1h48m.

Matched to E-INF1 on the same captures and seeds, paired per store with a
Wilcoxon test, over a grid that is a genuine superset this time:

| family | E-INF1 | phase B | Δ | wilcoxon |
|---|---|---|---|---|
| `PF single radio NN` | 0.301 | 0.299 | −0.5% | 0.940 |
| `PF dual radio NN` [abs] | 0.534 | 0.534 | +0.0% | 0.416 |
| `PF single radio` | 0.541 | 0.541 | +0.0% | 0.037 |
| `PF dual radio NN` [craft] | 0.667 | 0.673 | +0.9% | 0.029 |
| `PF dual radio` | 0.838 | 0.838 | −0.0% | 0.131 |
| `EKF single radio` | 1.022 | 1.032 | +1.0% | 0.782 |
| `EKF dual radio` | 2.583 | 2.625 | +1.6% | 0.782 |

**E-INF1's hyperparameters were already right.** Four of seven reproduce exactly.
Six of seven optima had sat on a grid boundary, four with MSE apparently still
falling 7–20% into the wall — and none of it was worth anything, because the
surface past the boundary is a plateau rather than a slope.

**One actionable finding:** `PF dual radio NN` [craft] holds its accuracy at
**N = 1024 instead of 16384** — 16× fewer particles for +0.9% MSE (significant at
p = 0.029, negligible in magnitude). Measured 2.67× faster, though runtime across
runs is confounded by box load: families whose N did not change read 0.79–0.94×,
so the true figure is nearer 3.1×.

**Phase C is not worth running as designed.** Its purpose was confirming phase B's
winners on 48 stores, but with one exception those are the configurations E-INF1
already confirmed there. Only the N=1024 change is new, and that is a
single-family run.

### The durable lesson

An optimum on a grid boundary looks alarming and usually is not. Extending six
truncated axes cost ~4 h of compute and bought at most 1.6%. Check the boundary,
but check it with a heatmap — the plateau is visible immediately, and the marginal
profile that prompted this whole experiment is what hid it.

## Addendum — the March 2025 deck's grids, reproduced (2026-08-11)

Full write-up:
[`spf/filters/reports/e_inf2_deck2025_20260811_v1/REPORT.md`](../../spf/filters/reports/e_inf2_deck2025_20260811_v1/REPORT.md).
133,760 runs, 50,160 configurations, 16 captures, 5 seeds.

Checking the "Droning on 2 (SPF)" deck's axes against every grid run here found
four we had never tried: `theta_err` 0.1; `theta_dot_err` 0.075/0.09/0.12/0.15;
`phi_std` 8/12/14/16/18; and six `noise_std` intermediates. All were run.

**None of it helps.** Matched to E-INF1 on the same captures and seeds, paired per
store, every family is within ±3%. The one candidate — `PF dual radio NN`
[absolute] at −2.9% using the never-before-tried `theta_err` = 0.1 — is better on
8 of 16 captures at Wilcoxon **p = 0.860**. `EKF single radio` reproduces E-INF1's
winner exactly.

**The `phi_std` 8–18 region is worse, not better.** The deck's EKF optimum sits at
`phi_std` = 12; on our corpus every one of those five values is worse than
`phi_std` = 1 for the single-radio EKF. It does not transfer.

**What does transfer is the surface shape, and only in one frame.** Our
`absolute_north` panel reproduces the deck's — optimum at `theta_dot_err`
0.001–0.002, degrading upward — across different hardware, carrier, d/λ and
routine. Our `craft_relative` panel is the deck's **mirror image**: best at 0.09,
worst at 0.0001, where the deck is best at 0.001–0.005 and worst at 0.1–0.2.

⚠️ **This corrects the routine-dependence note in the Risks table above.** It is
narrower than recorded: the inversion is confined to `craft_relative`. In
`absolute_north` the two corpora agree. A frame and a routine were being
conflated. The mechanism is that `rover_diamond` runs straight legs with occasional
corners (heading mostly constant → slow craft-relative bearing → small process
noise), while `rover_bounce` changes direction constantly. The absolute frame is
insensitive because the emitter's world-frame bearing drifts slowly regardless.

It also reproduces **H3** on the deck's own axis resolution: craft-relative
optimises at `theta_dot_err` 0.09–0.12 and absolute-north at 0.001–0.002, a ~50×
ratio against the 20× measured on our grid.
