# E-INF1 — results

**Status: STAGE 2 COMPLETE** (rover corpus, 2026-08-10). Stage 3 (all 48 rover
stores, then the frozen val corpus) not started. Design and decision rules are
pre-registered in [`experiment_readme.md`](experiment_readme.md); this file
exists so the hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | NN dual-radio PF beats empirical on the rover corpus | ✅ **SUPPORTED** — −20.4% craft-relative, −44.4% radio-folded |
| H2 | best rover MSE ≥ 2× best frozen-val MSE | ❌ **NOT TESTED — withdrawn 2026-08-10 by decision.** See below. |
| H3 | median `std(z)` > 1.5 on every corpus (filters overconfident) | ✅ **SUPPORTED** — corpus median 4.43; 92.6% of 9,792 configurations exceed 1.5 |
| H4 | MSE worse at d/λ = 0.904 than at 0.673 | ⚠️ **NOT RESOLVED** — 3 of 6 spacings have one capture each |

## H3 — SUPPORTED. The filters are overconfident, and it is not marginal (2026-08-10)

Re-ran the identical 16-store sweep with calibration scored inside every theta
filter (`ed8b054`): 26,112 results, 9,792 configurations, both amended acceptance
gates passing. Report:
[`spf/filters/reports/e_inf1_rover_coarse_calib_20260810_v1/`](../../spf/filters/reports/e_inf1_rover_coarse_calib_20260810_v1/).

**Corpus median `std(z)` = 4.43. 92.6% of configurations exceed the 1.5
threshold.** Per family, `std(z)` averaged across seeds and datasets:

| family | frame | configs | median `std(z)` | > 1.5 |
|---|---|---|---|---|
| `PF_single_theta_dual_radio_NN` | `craft_relative` | 1800 | **6.41** | **100%** |
| `EKF_single_theta_dual_radio` | `craft_relative` | 504 | 5.28 | 83% |
| `PF_single_theta_single_radio_NN` | `radio_folded` | 1800 | 4.56 | **100%** |
| `PF_single_theta_dual_radio_NN` | `absolute_north` | 1800 | 4.45 | **100%** |
| `PF_single_theta_dual_radio` | `craft_relative` | 1800 | 4.42 | 86% |
| `PF_single_theta_single_radio` | `radio_folded` | 1800 | 3.27 | 90% |
| `EKF_single_theta_single_radio` | `radio_folded` | 288 | **0.75** | 25% |

Read: the reported σ is typically **4–6× too small**. All three NN particle-filter
families have **no** honestly-calibrated configuration anywhere in the grid.

**Accuracy and honesty are separate axes.** The most accurate family
(`PF dual radio NN [abs]`, best MSE 0.050 = 12.8° RMSE) is among the worst
calibrated. `EKF_single_theta_single_radio` is the only family whose σ is
conservative — it *overstates* uncertainty (median 0.75, ±1σ coverage 0.81
against 0.683 nominal) — and it is also mediocre in accuracy.

⚠️ **`std(z)` cannot rank filters by skill, by construction.** A uniform-random
guesser that honestly reports σ = π/√3 = 1.814 rad scores `std(z)` = 1.00 —
better calibrated than every filter here. It is a scale-free self-consistency
check and must always be read alongside MSE against the π²/3 floor.

### Pre-registered consequence, now in force

> **H3** — if `std(z) > 1.5` holds, **no downstream component may gate on filter
> variance** until the cause is found, and that becomes its own work item.

Any code of the form "act on the bearing only when the filter's variance is low"
is not doing what it appears to. It would fire most confidently on the NN dual
radio PF — the family with zero honest configurations.

### The re-run is comparable to the original

MSE reproduces the committed stage-2 report exactly: all 9,792 configuration keys
matched, max |ΔMSE| = 8.9e-16, 9,250 of 9,792 rows bit-identical. The remaining
sub-ulp drift is summation order from `report.py`'s unsorted file walk, not a
behavioural change. So the calibration wiring is confirmed purely additive, and
H1's numbers stand unchanged.

## Not a hypothesis, but recorded: absolute_north vs craft_relative is a NULL result

The stage-3 table lists `PF_single_theta_dual_radio_NN` twice — 0.588 in
`absolute_north`, 0.627 in `craft_relative` — because it won both frames in
stage 2. **That 0.039 is not distinguishable from zero**, and an earlier revision
of the stage-3 report presented it as a ranked difference. Corrected.

Paired over 48 stores × 5 seeds: absolute wins **120/240** (a coin flip); per
store absolute 23 / craft 25; **median Δ = −0.0087**, i.e. craft is better at the
median. **92.1% of the mean gap comes from one capture**
(`rover_2026_08_07_01_27_43…RO3`, craft 2.730 vs absolute 1.020); dropping it the
gap is +0.0031, dropping two it reverses to −0.0111. Bootstrap 95% CI
[−0.046, +0.141]; Wilcoxon p = 0.780.

The comparison is also **unfair to craft-relative**: the two rows run at their own
stage-2 winning hyperparameters (so it is frame × tuning × N, not frame), and
craft's optimum is **pinned at the swept grid boundary** — all its best
configurations sit at `theta_dot_err = 0.1`, the largest value the grid tries,
while absolute's winner is interior. 0.627 is an upper bound.

**No frame bug.** Adversarially audited: `craft_ground_truth_thetas ==
pi_norm(absolute_thetas[0] − rx_heading[0])` is an algebraic identity (max
violation 1.05e-4 deg corpus-wide); rotation sign, per-radio heading pairing,
single application and cache immutability all verified; the inference cache
stores unrotated posteriors; `absolute_thetas` is pure geometry from tx/rx
positions, so the absolute metric is **not** self-referential. Two real defects
found, both of which *handicap* absolute: `rotate_dist`'s interpolation smear
(absolute-only; ~1.2% of the per-capture gap at the tuned point) and the target
being the circular mean over both radios while the observation is radio 0's
(0.0100 rad² measured by rescoring).

What *is* established is per-capture and mechanistic, not a corpus ranking: at
identical hyperparameters and seed, each frame's optimal process noise lands where
its measured angular rate predicts (craft p95 |dθ/dt| 18.26°/step → θ̇_err 0.1;
absolute 5.21°/step → 0.02), because the constant-angular-velocity model is
applied in the frame the filter runs in. Deciding which frame is better needs
craft re-tuned on an extended grid with both arms at matched N.

## H2 withdrawn — the frozen val corpus cannot answer it (2026-08-10)

H2 compares the best rover MSE against the best frozen-val MSE, to decide whether
the 2026 rover corpus is trustworthy enough to evaluate on. Checking what the
frozen val list actually contains shows the comparison does not carry that meaning.

`/mnt/md2/splits/apr17_val_nosig_noroverbounce.txt` is 565 datasets:

| | count |
|---|---|
| 2D wall array (static array in a room, emitter moved around it) | 544 |
| rover | 21 |
| …of which rover running the `bounce` routine | **4** |

The 2026 corpus is entirely `rover_bounce`. So a val comparison is almost entirely
against a **different platform** with different motion and geometry, and a gap
would conflate "the 2026 corpus is bad" with "rovers are harder than a bench
array" — precisely the distinction H2 exists to make.

The only like-for-like is the 4 rover-bounce captures (2025-04-05, spacings 0.035
and 0.043 → d/λ 0.67317 and 0.82703, both present in the empirical table). All
four are in `val_degraded_v2`, which [`docs/learnings.md` L1](../../docs/learnings.md)
says is "reported but never optimized toward".

**Decision: evaluate on the 2026 rover corpus only; do not run the historical
data.** H2 is withdrawn rather than answered with a number whose meaning is
ambiguous. Consequences, recorded so this is reversible:

- No inference caches are built for val; `/mnt/md2` is not read or written at all.
- The "is the 2026 corpus trustworthy?" question is now **unanswered**, not
  answered negatively. If it needs answering later, the honest instrument is a
  matched 2025-vs-2026 rover-bounce comparison at the same d/λ — not the frozen
  val set — and that is a capture/curation question, not a filter-sweep one.
- H3's pre-registered wording says "on every corpus"; with only one corpus in
  scope it is judged on the 2026 rover corpus alone.

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
