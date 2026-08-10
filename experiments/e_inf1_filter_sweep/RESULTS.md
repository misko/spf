# E-INF1 — results

**Status: STAGE 2 COMPLETE** (rover corpus, 2026-08-10). Stage 3 (all 48 rover
stores, then the frozen val corpus) not started. Design and decision rules are
pre-registered in [`experiment_readme.md`](experiment_readme.md); this file
exists so the hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | NN dual-radio PF beats empirical on the rover corpus | ✅ **SUPPORTED** — −20.4% craft-relative, −44.4% radio-folded |
| H2 | best rover MSE ≥ 2× best frozen-val MSE | _pending — needs the val corpus (stage 3)_ |
| H3 | median `std(z)` > 1.5 on every corpus (filters overconfident) | ⚠️ **UNANSWERABLE AS RUN** — the sweep records no calibration metric |
| H4 | MSE worse at d/λ = 0.904 than at 0.673 | ⚠️ **NOT RESOLVED** — 3 of 6 spacings have one capture each |

Full write-up with figures:
[`spf/filters/reports/e_inf1_rover_coarse_20260809_v1/REPORT.md`](../../spf/filters/reports/e_inf1_rover_coarse_20260809_v1/REPORT.md).

## Stage 2 — 26,112 runs, 9,792 configurations, 16 captures

Corpus means across 5 seeds; ± is the seed standard deviation. **Frames are not
comparable** — compare down a frame, never across.

| family | frame | best MSE (rad²) | ±1σ | skill vs random (π²/3) |
|---|---|---|---|---|
| `PF_single_theta_single_radio_NN` | `radio_folded` | 0.301 | 0.002 | +90.9% |
| `PF_single_theta_dual_radio_NN` | `absolute_north` | 0.534 | 0.077 | +83.8% |
| `PF_single_theta_single_radio` | `radio_folded` | 0.541 | 0.004 | +83.6% |
| `PF_single_theta_dual_radio_NN` | `craft_relative` | 0.667 | 0.003 | +79.7% |
| `PF_single_theta_dual_radio` | `craft_relative` | 0.838 | 0.006 | +74.5% |
| `EKF_single_theta_single_radio` | `radio_folded` | 1.022 | 0.000 | +68.9% |
| `EKF_single_theta_dual_radio` | `craft_relative` | 2.583 | 0.000 | +21.5% |

**H1 — supported.** Both matched comparisons favour the network, against the
*rebuilt* empirical table (`full_20260809_v1.pkl`), so this is the empirical
baseline at its strongest. Seed spreads are ~30× smaller than the gaps.

**H3 — unanswerable, not falsified.** The result dicts carry only `mse_*` and
`runtime`; `spf/evaluation/calibration.py` was never wired into the run wrappers.
On one capture every filter but the empirical dual-radio PF was overconfident
(`std(z)` 1.34–31.21), which is suggestive and not a test. Wiring the calibration
block into the 5 PF and 2 EKF wrappers and re-running stage 2 (~40 min) settles it.

**H4 — not resolved.** No family is monotonic in d/λ, but d/λ 0.83765, 0.90397
and 0.91557 each rest on a single capture. The two well-supported spacings
(0.67317, n=5 and 0.82703, n=6) are within 2% of each other despite differing by
23% in d/λ. This needs a spacing-balanced capture matrix, not more seeds.

### Unplanned finding — the EKF dual-radio filter is near-uninformative

Not a pre-registered hypothesis; recorded because the random floor exposed it.
Its best configuration reaches 2.583 rad² against a uniform-random floor of
3.290, its median configuration 2.908, and its **worst configuration, 3.332, is
worse than guessing**. On the one capture examined in depth the single-radio EKF
(1.148) also lost to a fixed bearing (0.654) — i.e. it had learned nothing about
*time* there. Skill-vs-random alone rated that same filter at +65%.

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
