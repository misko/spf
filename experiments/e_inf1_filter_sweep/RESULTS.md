# E-INF1 — results

**Status: NOT RUN.** Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | NN dual-radio PF beats empirical on the rover corpus | _pending_ |
| H2 | best rover MSE ≥ 2× best frozen-val MSE | _pending_ |
| H3 | median `std(z)` > 1.5 on every corpus (filters overconfident) | _pending_ |
| H4 | MSE worse at d/λ = 0.904 than at 0.673 | _pending_ |

## Blocking item — ✅ RESOLVED 2026-08-09 (`2a07ae0`)

**Use `empirical_dists/full_20260809_v1.pkl`. Not `full.pkl`.**

The gap was larger than first recorded: **17 of 48** merged rover stores, not 3.
d/λ is derived from antenna spacing **and** carrier frequency, and the 2026 fleet
changed both — a new carrier (5840 MHz) and a new spacing (0.047 m, RO4) — giving
3 × 2 = 6 combinations where the old table held only the 2 that existed when it
was built.

| d/λ | spacing | carrier | in `full.pkl` | stores |
|---|---|---|---|---|
| 0.67317 | 0.035 m | 5766 MHz | ✅ | 12 |
| 0.68181 | 0.035 m | 5840 MHz | ❌ | 3 |
| 0.82703 | 0.043 m | 5766 MHz | ✅ | 19 |
| 0.83765 | 0.043 m | 5840 MHz | ❌ | 5 |
| 0.90397 | 0.047 m | 5766 MHz | ❌ | 6 |
| 0.91557 | 0.047 m | 5840 MHz | ❌ | 3 |

`full_20260809_v1.pkl` (48 keys, 2,445 datasets) covers **48/48**, verified live
for every distinct key. Report:
[`spf/calibrations/empirical_p_dist/reports/empirical_rebuild_20260809_v1/`](../../spf/calibrations/empirical_p_dist/reports/empirical_rebuild_20260809_v1/REPORT.md).

⚠️ **Two existing keys changed.** `PLUTO_0.82703` (24→43 datasets, corr 0.937 vs
the old table) and `PLUTO_0.67317` (21→33, corr 0.989) now fold in 2026 rover
data. Empirical-filter results at those spacings are not comparable to anything
predating `2a07ae0` — including the pilot numbers below, which used `full.pkl`.

## Pilot observations (single dataset, 2026-08-07)

Not results — these motivated the experiment and are recorded so the stage-2
numbers can be sanity-checked against something.

On `rover_2026_08_01_19_31_21…RO3` (539 timesteps, d/λ = 0.827), one seed each:

| filter | MSE (rad²) | frame |
|---|---|---|
| PF dual, NN, `absolute=True` | 0.32–0.76 across 8 seeds | absolute_north |
| PF dual, NN, `absolute=False` | 1.86 | craft_relative |
| PF dual, empirical | 1.63–2.44 across 8 seeds | craft_relative |
| PF single, empirical (r0/r1) | 0.32 / 0.58 | radio_folded |
| EKF dual | 2.57 | craft_relative |

±1σ coverage for the NN dual PF was **25.6%** against 68.3% nominal.

⚠️ The two NN rows are **not** comparable — different ground truth. And the
per-seed spread (42% empirical, 106% NN) is why stage 2 runs 5 seeds.
