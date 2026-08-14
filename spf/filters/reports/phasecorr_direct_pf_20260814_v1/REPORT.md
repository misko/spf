# Gain-phase correction on the direct PF filters — a clean negative, and why

**Run 2026-08-14.** 1,920 runs · 2 direct (non-NN) PF families · 4 arms · 5 seeds · 48 rover
captures · paired per capture · zero failures. Read-only: **the rover data was not modified**;
the correction is applied to an in-memory copy of `mean_phase` and nothing under
`/mnt/qnap01/mouse9911/rovers_2026` was opened for writing.

## Result

**Every correction arm is significantly WORSE than no correction**, and the negative control
is worse by a comparable margin.

### `PF single radio` [folded frame] — uniform floor 1.645 rad²

| arm | MSE | RMSE | vs random | std(z) | cov@68 | ΔMSE vs none | better on | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **none** | **0.5143** | **41.1°** | **3.20×** | **1.84** | 0.618 | — | — | — |
| shuffled *(control)* | 0.5297 | 41.7° | 3.11× | 1.86 | 0.605 | +0.0154 | 6/48 | 0.0000 |
| arm_lut | 0.5397 | 42.1° | 3.05× | 1.93 | 0.599 | +0.0254 | 5/48 | 0.0000 |
| constant | 0.5412 | 42.2° | 3.04× | 1.89 | 0.598 | +0.0269 | 4/48 | 0.0000 |

### `PF dual radio` [craft-relative] — uniform floor 3.290 rad²

| arm | MSE | RMSE | vs random | std(z) | cov@68 | ΔMSE vs none | better on | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **none** | **0.9411** | **55.6°** | **3.50×** | **1.57** | 0.705 | — | — | — |
| shuffled *(control)* | 1.0043 | 57.4° | 3.28× | 1.71 | 0.692 | +0.0632 | 7/48 | 0.0000 |
| constant | 1.0181 | 57.8° | 3.23× | 1.71 | 0.689 | +0.0769 | 5/48 | 0.0000 |
| arm_lut | 1.0281 | 58.1° | 3.20× | 1.68 | 0.686 | +0.0870 | 6/48 | 0.0000 |

**Calibration degrades too.** `std(z)` rises in every arm (paired p ≤ 0.0072 throughout), and
68% coverage falls. The correction did not merely fail to help accuracy — it made the
filters' uncertainty *less* honest as well.

## The controls are what make this interpretable

Without them the headline would read "gain-phase correction hurts", and that would be wrong.

**`shuffled` — same correction values, permuted across gain states — also degrades**, by
+0.0632 (dual) and +0.0154 (single). And `arm_lut` is **not better than `shuffled`**; it is
*worse*, on both families (Δ = +0.0238, p = 0.0145; Δ = +0.0100, p = 0.0001).

If the correction carried real information about the frames it was applied to, it would have
to beat a version of itself with the gain mapping destroyed. It does not. **What is being
measured is the magnitude of the perturbation, not its content.**

**`constant` is statistically indistinguishable from `arm_lut`** (p = 0.101 dual, p = 0.485
single). Almost all the damage is done by the ~7° constant offset, not by the gain-dependent
variation — which is the entire quantity the model exists to predict.

## Why: the empirical table was fitted on uncorrected φ

This was pre-registered as the risk to watch, and it is what happened. The empirical table
maps φ → p(θ) and was built from **uncorrected** data, so the gain-phase offset is already
baked into it. Subtracting ~7° from φ at inference moves the input *away* from the
distribution the table was fitted on. The correction and the table disagree, and the table
wins because it is what converts φ into a bearing.

That also explains the ordering: the median correction is ≈ 6.97°, essentially all of it
constant (measured earlier: weighted mean D = 6.32° of a 6.37° mean |D|), and `constant`
reproduces nearly all of the harm on its own.

**So the constant is absorbed — by the empirical table itself.** That answers the open
question directly, and it means the deployable prize was never the 6–10°; it is only the
1.0–1.9° of variation, of which a held-out donor table removes 22–41%, i.e. ~0.3–0.7° of
phase ≈ 0.12–0.28° of bearing, against filter RMSEs of 41–56°.

## What this does and does not rule out

**Ruled out:** applying a gain-phase correction at inference against the frozen empirical
table. It is harmful, reproducibly, on 48 captures at p < 0.0001.

**NOT ruled out:** that gain-phase correction helps when the empirical table is *rebuilt from
corrected φ*. This experiment cannot distinguish "the correction is worthless" from "the
correction is inconsistent with the table". The controls say the second is more likely, since
a random perturbation of the same size does comparable damage.

The honest next step, if anyone wants to settle it, is the fork recorded in the plan: rebuild
the empirical table from corrected φ and re-run. That touches a frozen artifact and needs its
own named table, so it is a larger change than this one — and given the effect size above
(~0.2° of bearing inside a 41–56° RMSE), I would not spend the bench or compute time on it
without a specific reason.

## Caveat on the donor

The rover's radios are **held out** — they are not the fitted serial and have no calibration
of their own. This measures the held-out-donor case, which is what the rover would get today.
A same-radio table would remove 82–86% of the variation rather than 22–41%, but the failure
mode identified above is about the table mismatch, not the donor, so a same-radio table would
not obviously change the sign of this result.

## Reproducing

```bash
P=~/virtual-envs/spf/bin/python3
PYTHONPATH=$PWD $P spf/filters/run_filters_on_data.py \
  -d $(cat experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt) \
  --empirical-pkl-fn empirical_dists/full_20260809_v1.pkl \
  --work-dir /mnt/qnap01/mouse9911/rovers_2026/filter_runs/phasecorr \
  --config spf/filters/configs/rover2026_phasecorr.yaml \
  --results-backend local --parallel 16
$P analysis/analyze_phasecorr.py
```

⚠️ `PYTHONPATH` matters: launching `run_filters_on_data.py` as a script puts `spf/filters/` on
`sys.path`, not the repo root, so without it `import spf` resolves to the installed package
and you can silently run a mixture of two checkouts.

**Baselines.** The single-radio family scores a folded half-circle, whose uniform-random floor
is **π²/6 = 1.645 rad²**, not the π²/3 that `spf/evaluation/metrics.py` defines. Scoring it
against π²/3 overstates skill by roughly 2×.
