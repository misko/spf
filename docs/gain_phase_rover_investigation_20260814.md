# Gain-phase correction and the rover: what we did, what we found, and where it ends

**2026-08-11 → 2026-08-14.** A synthesis of one investigation, from "which gain-phase model
should the rover use" to "do not deploy it, and we cannot yet measure whether a better one
would help". Includes **two retractions** — of this document's original closing argument, and
then of the analysis that was supposed to replace it.

Working branch **`gainphase-rover-investigation-20260814`**, kept in step with `main`; the
final commit of the investigation is the tip of both. (An earlier version of this line pinned
`f91a2ba`, which stopped being the tip three commits later.)

---

## FINAL CONCLUSION

Five statements, each with its evidence and its status.

| # | conclusion | status |
|---|---|---|
| 1 | **Do not deploy the gain-phase correction we have.** The R18-derived held-out donor changes the geometry-conditioned rover residual by **+0.017°, 95% CI [−0.020, +0.061]** (42 unique captures, 84 streams) — indistinguishable from zero — and costs a small but reproducible accuracy penalty end to end in both direct particle filters. | **Supported. Act on this.** |
| 2 | **The physical question is NOT closed, and this investigation cannot close it.** Gain-state fixed effects fitted on the rover data itself return +0.018° to +0.087° on held-out captures. That was reported as an upper bound. It is not one: the statistic is **quadratically insensitive** to a small phase term — a *perfect* correction of the 1.0–1.9° at stake could move it by only **+0.010° to +0.036°**, i.e. no larger than one standard deviation of the statistic's own fold-seed noise (sd **0.033°**, observed range +0.002° to +0.111° over 8 seeds) — and the sign flips with a nuisance parameter (`min_n` 8 → 25 gives −0.013°). The experiment had no power to detect what it claimed to bound. | **Overclaimed twice. Retracted twice. See [Retraction 2](#retraction-2--the-upper-bound-was-not-one).** |
| 3 | **Keep the bench model.** `mixer + LNA`, 28 parameters per radio-carrier, is the most parsimonious description of the E-GSC9 Session-A measurements and localises R17's fault to a single coefficient (−77.09° on RX1's LNA switch against −18.10° on its own RX2). It is valuable for bench work and hardware diagnostics. | **Supported.** |
| 4 | **Decline a same-radio bench campaign on cost/benefit, not on physics.** Even a *perfect* correction removes at most the 1.0–1.9° of phase variation ≈ 0.4–0.76° of bearing, against dual-filter RMSEs of 41–56°. That is a ≤1.4% ceiling, and it justifies declining a 2.6 h/radio campaign. It does **not** justify the claim that such a campaign would find nothing. | **Rescoped.** The earlier "it cannot do better" was withdrawn, reinstated on bad evidence, and is now withdrawn again. |
| 5 | **The remaining work is the 35–37° residual itself** — multipath, GPS/heading, segmentation — which is a different and larger investigation. **That is where rover bearing accuracy actually lives.** | **Supported, and unaffected by either retraction.** |

**In one sentence:** the correction we built does not help the rover and should not ship, the
model behind it is good bench physics worth keeping, and whether gain state carries *any*
usable phase information on these radios **remains unmeasured** — twice asserted here, twice
retracted, and the honest statement is "we have not measured it", not "it is zero".

### What changed, and why you should trust this version less than its confidence suggests

**Two closure claims have now been retracted from this document.**

The first conditioned on `rx_theta_in_pis`, believing it was ground-truth bearing. It is the
array mount orientation and is **constant per receiver per capture**, so the analysis compared
a 1.75° correction against the entire trajectory's phase motion and concluded gain-phase was a
2% term.

The second — added to *fix* the first — fitted gain-state effects on the rover data itself and
read a positive held-out change as proof of absence. It was underpowered by construction: the
metric cannot resolve the effect size at issue. **Both were caught in external review, not by
me, after being committed and reported.**

The pattern is the same both times: a statistic was quoted without asking what value it would
take if the hypothesis were true. Withdrawn numbers are listed in [Retraction 1](#retraction-1--the-wrong-angle-variable)
and [Retraction 2](#retraction-2--the-upper-bound-was-not-one).

---

## The question, and the answer

**Question:** which gain-phase model should correct the rover's data, and how much does it buy?

**Answer for the model we have:** do not deploy it. The R18-derived **held-out donor**
correction produces no detectable improvement end to end, and a small accuracy penalty in the
particle filters.

**Answer for gain-phase in general: unmeasured.** Fixed effects fitted on the rover data
itself, cross-validated across physical captures, come out slightly *worse* on held-out
captures (+0.018° to +0.087°). That was published here as an upper bound. **It is not one,**
and the reason is arithmetic rather than subtle:

| what | value |
|---|---:|
| gain-phase term at stake (bench-measured sd) | **1.0–1.9°** |
| ceiling on Δ(mean \|e\|) from removing it *perfectly* | **+0.010 to +0.038°** |
| fold-seed sd of the published statistic (8 seeds, 6-fold CV) | **0.033°** (range +0.002 … +0.111) |
| same statistic at `min_n` 25 instead of 8 | **−0.013°** (sign flips) |

The measured law is **Δ(mean \|e\|) = 0.0101 · A²** degrees for a term of rms amplitude *A*
degrees, on the actual 124,950 residuals — quadratic, because a small offset buried in a 49.2°
residual barely moves a mean-absolute statistic. **A perfect oracle correction would have been
invisible to this experiment.** So the positive numbers are consistent with a real 1–2° effect,
with no effect at all, and with anything in between.

⚠️ **This document has now asserted closure twice and retracted it twice.** See
[Retraction 1](#retraction-1--the-wrong-angle-variable) and
[Retraction 2](#retraction-2--the-upper-bound-was-not-one). What survives is the engineering
decision, which rests on the end-to-end sweep and on the ≤1.4%-of-RMSE ceiling — not on either
withdrawn argument.

---

## The chain of experiments

| # | experiment / report | what it settled |
|---|---|---|
| 1 | [`ladder_frames_gsc678_20260813_v1`](../spf/calibrations/dual_rx_gain_frequency/reports/ladder_frames_gsc678_20260813_v1/REPORT.md) | First frame-level ladder fit. The shipped mechanistic family is the **wrong shape**; the anchor gain, not the radio, decides whether the model's core assumption holds. |
| 2 | [`e_gsc9_rover_operating_region`](../experiments/e_gsc9_rover_operating_region/experiment_readme.md) | Preregistered capture of the rover's own operating cells. Designed as a 1,600-cell `[23,62]²` grid; the measured-level fallback **executed 1,369 cells over `[26,62]²`**, covering **99.9829%** (5766) and **100%** (5840) where prior campaigns covered **0**. |
| 3 | [`e_gsc9` RESULTS](../experiments/e_gsc9_rover_operating_region/RESULTS.md) | Session A executed: 27,380 frames, all quality-valid. H2 falsified on the damaged unit, H3 localised its defect, two gates failed and were retained. |
| 4 | [`rover_model_gsc9_20260814_v1`](../spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/REPORT.md) | Refit on the measured cells. **`mixer + LNA`, 28 parameters per radio-carrier**, is the most parsimonious description of the bench data. Coefficients committed. |
| 5 | [`phasecorr_direct_pf_20260814_v1`](../spf/filters/reports/phasecorr_direct_pf_20260814_v1/REPORT.md) | Applied it to the direct PF filters, 1,920 runs. **Significantly worse**, and the negative control degraded similarly. |
| 6 | same report, addendum | Rebuilt the empirical table from corrected φ. Consistency **halved the accuracy penalty and flipped the calibration sign** — but accuracy stayed worse. |
| 7 | `analysis/why_null.py` | ⚠️ **WITHDRAWN — conditioned on the wrong angle.** |
| 8 | `analysis/geometry_conditioned.py` | Corrected: the donor correction changes the geometry-conditioned residual by **+0.017°, 95% CI [−0.020, +0.061]** over 42 unique captures. A null **for the donor**, not for the physics. |
| 9 | `analysis/gain_fixed_effects.py` | ⚠️ **NOT an upper bound — underpowered by construction.** Gain-state fixed effects fitted on rover data, 6-fold CV by physical capture, give +0.018° to +0.087° on held-out captures. The statistic's ceiling for the effect at issue is +0.010–0.038°, no more than one sd of its own 0.033° fold-seed noise. **Non-informative, in either direction.** |
| 10 | `analysis/power_calibration.py` | The sensitivity law that should have been computed *before* #9: **Δ(mean \|e\|) = 0.0101·A²**, and the break-even amplitude at which a free *k*-parameter fit's signal exceeds its parameter cost — **4.83° (cell), 2.86° (arm), 1.87° (rfblock)** against 1.0–1.9° at stake. |

---

## What was actually learned

### The model itself is good, and physically legible

`mixer + LNA` — per radio, per arm, **28 parameters per radio-carrier** (the committed
two-carrier artifact holds 56) — ties a 74-parameter LUT and an 80-parameter four-word model on
bench cells, and **matches the LUT's cross-carrier transfer** at far fewer, physically
interpretable parameters (0.276° vs 0.277° predicting 5840 from 5766). It does *not*
demonstrate a transfer capability the LUT lacks; the earlier wording claimed that and was
wrong. The physics is clean: over 26→62 dB the
AD9361 moves only the baseband LPF, then one LNA step at 40→41, then the mixer. The LPF sits
after the mixer and contributes nothing measurable; the RF-side blocks carry all the phase.
Smooth functions of dB fail outright because the response is a **staircase over discrete
hardware states**.

Its coefficients are diagnostic: they localise the damaged unit's fault to **one number** —
its RX1 LNA switch carries −77.09° where its own RX2 carries −18.10°, a −59.00° arm asymmetry,
independently reproducing E-GSC9's H3 (−59.49°) from a different fit.

### The donor correction has no measurable effect on rover data

Conditioning on geometry **exactly** (`e = wrap(mean_phase − ground_truth_phi)`, centred per
stream, 42 unique RX captures after deduplication):

| quantity | value |
|---|---:|
| mean \|e\| without correction | **36.728°** |
| mean \|e\| with correction | 36.746° |
| change | **+0.017°, 95% CI [−0.020, +0.061]** |
| better on | 45/84 streams |
| corr(correction, residual) | +0.0138, r² = 0.019% |

**This bounds the donor, not the physics.** A mismatched predictor is attenuated toward zero
correlation even when the underlying term is real, so it says nothing about a same-radio or
sample-weighted correction. The bench-measured gain term (1.0–1.9° sd) is small against a
36.7° residual, which is *suggestive* — but that argument assumes the rover's radios behave
like the two bench units, which is exactly what has not been shown.

### Three real defects were found and fixed or recorded

- **A support-rule defect** in the published ladder pipeline: it refused rows for needing a
  parameter that could not affect them, so **every mechanistic rung failed closed on exactly
  the single-carrier fit a rover would run**. No published number described that case.
- **An `eval()` guard** in the sweep config expander that caught only `SyntaxError`, so any
  string-valued config axis (`"none"`) raised `NameError` out of the run. Fixed.
- **A silent table/inference mismatch** — a correction applied at inference against a table
  fitted without it is a ~7° error and nothing detected it. Now an assert.

---

## Corrections I made to my own earlier claims

Recorded because the intermediate numbers circulated before they were right.

1. **"~28° uncorrected, 39–58× gain"** → measured on bench cells with a 36 dB arm split; the
   rover runs 13 dB. On its actual cells: **6.4–10.5°, 33–43×**.
2. **"33–43×"** → that baseline is almost entirely a *constant*, which the empirical table
   already absorbs. The removable part is the **1.0–1.9° of variation**, → 2.2–7×.
3. **"No consumer exists for these coefficients"** → wrong. `PhaseOffsetModel` with 11 model
   families and a fail-closed support profile already existed; our fit ships in it unchanged.
4. **"A third of a histogram bin, so the table cannot move"** → the centroid shift is indeed
   0.002–0.057° against a 5.54° bin, but the conditionals move by **TV 0.11–0.25**: a sub-bin
   shift near a sharp ridge flips frames across bin edges.
5. **"The table has zero rover captures"** → an artifact of reading a field that was `None`.
   It has **48**, the same ones used for evaluation.
6. **"A same-radio capture per rover unit is the remaining option"** → I retired this on the
   strength of the withdrawn 2% argument. It is not that the option is wrong; **the argument
   used to rule it out was wrong.** I then reinstated the retirement on the strength of
   `gain_fixed_effects.py`, which does not support it either (correction 9). The campaign is
   now declined **on cost/benefit** — a ≤1.4%-of-RMSE ceiling — which is a different and much
   weaker statement than "it would find nothing".
7. **"The correction explains 0.060% of rover phase variance"** and everything built on it →
   withdrawn; see [Retraction 1](#retraction-1--the-wrong-angle-variable). The correct figure is r² = 0.019% against a 36.7°
   geometry-conditioned residual, and it bounds the **donor**, not the physics.
8. **A second silent-stratification bug**, caught before reporting: the first run of
   `gain_fixed_effects.py` read `gain_endpoints_equal` from `cached_keys`, where it does not
   exist, so it silently defaulted to all-True — making the "stable gain" stratum identical to
   "all" and leaving "unstable" empty. It surfaced only because the empty stratum divided by
   zero. **This is the same class of error as the `rx_theta_in_pis` bug** — reading a field
   that is not what its name suggests — and it was caught by a crash rather than by design.
9. **"The physical question is now CLOSED, on evidence"** and **"no imported model can beat
   this upper bound"** → **withdrawn in full.** See
   [Retraction 2](#retraction-2--the-upper-bound-was-not-one). The estimator had no power to
   detect the effect it purported to bound, the "upper bound" argument rested on a nesting
   premise that is factually false (radio identity is absent from the fitted feature space),
   and the reported sign is an artifact of two nuisance choices. This is the **second**
   unearned closure in this document, one addendum after the first.

---

## Where this ends

**Pause dedicated rover gain-phase work.** The model is the best parsimonious description of
the Session-A bench measurements, committed and documented, and worth keeping for bench and
hardware-diagnostic use — it localised a real fault in R17. The currently tested donor
correction is not worth deploying.

It is **not** established that no correction could matter, and this document has twice claimed
otherwise in error. Sessions B and C are **terminated by decision for rover deployment**, but
they remain required before any broader temporal-transfer or physical-discriminator claim; G8
and G9 failed and stand.

The blockers that remain are recorded rather than solved, and none is now worth solving *for
this purpose*: the anchor cannot be measured in flight; 66% of rover frames change gain
mid-buffer unguarded (69% at 5766 MHz, 45% at 5840); cross-radio transfer is 1.16×, i.e. none.

**Why the work stops here is now a budget argument, not a physics one.** The ceiling on any
gain-phase correction is the 1.0–1.9° of removable phase variation ≈ 0.4–0.76° of bearing,
against dual-filter RMSEs of 41–56°: **at most ~1.4%**. That is a sound reason to stop. It is
not a finding that the effect is absent, and it should not be quoted as one.

⚠️ Two numbers now carry that entire argument and **neither has been independently audited**:
the 1.0–1.9° correctable variation, and the °-bearing-per-°-phase conversion. If either is
wrong by a factor of two, the priority call changes. Auditing them is cheap and is the first
item in [what would actually settle this](#what-would-actually-settle-it).

The geometry-conditioned residual is **35–37°**, and identifying what composes it — multipath,
GPS/heading error, segmentation, oscillator effects — is a different and larger investigation
than gain tables. **That is where rover bearing accuracy actually lives.**

### What would actually settle it

Not "nothing further is warranted" — rather, nothing further is *warranted at this priority*.
If the question is reopened, the low-cost route needs **no new capture**:

1. **Audit the two load-bearing numbers** above. Hours.
2. **Replace the primary statistic.** A one-parameter circular projection α̂ of the residual
   onto a *hypothesised* LUT shape, capture-clustered bootstrap. A k=1 statistic has a
   break-even amplitude of **0.27°** against the free fit's 1.87–4.83°, so unlike everything in
   this investigation it can actually see a 1–2° term. A prototype exists and separates
   injected-effect from no-effect at disjoint 95% CIs.
3. **Add the controls `gain_fixed_effects.py` lacked** — per-capture CI, seed spread, `min_n`
   sweep, and a **run-preserving** null (circular shift of the gain-key sequence within each
   stream; a plain within-stream shuffle is *not* adequate, because gain is held for long runs
   and the residual has lag-1 autocorrelation 0.573).
4. **Fix the two known code defects** before any re-run: the dedup drops 6 merged stores /
   9,274 frames / 12 streams (6.9%) that are strictly *disjoint in time*, not duplicates; and
   `arm`/`rfblock` sum marginal conditional means rather than fitting jointly.

Only a **same-radio bench LUT** can test the per-radio hypothesis directly — the free fit
provably cannot reach it — and only **protocol-v3 firmware** would make a sample-weighted
trajectory model computable at all. Both require new capture. Decline them on the ≤1.4%
ceiling if you decline them; do not decline them on `gain_fixed_effects.py`.

---

## Retraction 1 — the wrong angle variable

An external methodological review found that `why_null.py` — the analysis supplying this
document's original closing argument — **conditioned on `rx_theta_in_pis`, which is the array
mount orientation, not ground-truth bearing.** It is constant per receiver per capture
(verified: 1.0 on r0, 0.5 on r1), so every frame fell into a single "bearing bin" and the
81.98° denominator contained the real geometric signal.

Withdrawn: the 81.98° residual, r = −0.0245, r² = 0.060%, the 2.1% share, the "other 98%", and
the claim that the physical question is closed. The review also correctly noted that the 2.1%
was a standard-deviation ratio presented next to an r² and then treated as an additive
decomposition — three different quantities.

Also corrected here: the executed grid was 1,369 cells not 1,600; cross-carrier transfer
matches the LUT rather than exceeding it; "28 parameters" is per radio-carrier; and "the model
is correct" is softened to what the evidence supports. Further caveats the review raised and
this document now honours: the 48 merged stores are not 48 independent captures (42 unique RX
recordings), frames are not independent observations, and the matched-table experiment is a
pipeline-consistency test rather than a clean generalisation estimate.

**The engineering decision is unchanged. The scientific closure is retracted.**

---

## Retraction 2 — the "upper bound" was not one

A second external review, on 2026-08-14, rejected the replacement argument. It is right.
`gain_fixed_effects.py` was added *because* Retraction 1 left the physical question open, and
it does not close it either.

**Withdrawn:** "the physical question is now CLOSED, on evidence"; "because the effect is
fitted on the target data, this is an upper bound no imported model can beat"; "every cell is
positive" as evidence of absence; and the reinstated "no same-radio bench campaign is
justified" *as a physics claim*.

Three independent reasons, in order of force:

**1. The experiment had no power.** The reported statistic is the change in mean |wrap(e)|,
which responds *quadratically* to a small phase term: measured on the real 124,950 residuals,
**Δ = 0.0101·A²** degrees for an rms amplitude *A*. At the 1.0–1.9° at issue the ceiling is
**+0.010 to +0.038°** — against a seed-to-seed sd of **0.033°** and a `min_n` sensitivity that
flips the sign (+0.042° at 8, −0.013° at 25). Restated as break-even amplitude, a free fit
needs a **4.83° (cell) / 2.86° (arm) / 1.87° (rfblock)** effect before its signal exceeds its own
parameter cost. **Every parameterisation was guaranteed to return a positive number whether or
not the effect exists.** This is the fatal one, and neither I nor the reviewer stated it — it
came out of adjudicating the review.

**2. The nesting premise is factually false.** "A same-radio model is a constrained version of
what was just fitted freely" requires radio identity to be *in* the fitted feature space. It is
not: `state_keys()` keys on `(g1, g2, LO)` only, while **6 distinct physical Plutos across 3
rover units** are pooled into one accumulator — and `sdr_serial` is present in all 84 streams,
so this was one lookup away. A same-radio model is not a sub-model; it is a *different* model.

**3. The sign is an artifact.** +0.042° at `min_n=8` becomes −0.013° at 25 and −0.006° at 50.
The published table reports one fold seed with no dispersion, and the whole band is
statistically indistinguishable from randomised gain labels.

**Two further defects, real but not fatal:** `arm` and `rfblock` sum *marginal* conditional
means instead of fitting jointly (an exact 2× overcount when g₁ = g₂; measured inflation on
this corpus is 1.02×, so it is an efficiency loss, not the reported result's cause), and the
dedup silently drops 6 merged stores — 9,274 frames, 12 streams, 6.9% — that are **disjoint in
time**, not duplicates.

**Where the review itself over-reached,** recorded so the next reader is not misled by it
either: its illustrative mechanism (radio A +2°, radio B −2°) is a *constant* per-radio offset,
which line 123's per-stream circular centring already removes; the "2× doubling" is conditional
on g₁ = g₂ and measures 1.02× here; "the CI may be too narrow" is refuted — a capture-clustered
bootstrap is 1.009× the per-stream one; the mean-vs-median gap depends on *skew*, not on the
49° spread, and substituting a circular median makes held-out error **worse** at every `min_n`;
and its proposed sample-weighted correction is **not computable from this corpus at all** —
`gain_observation_*`, `gain_event_*` and `sample_counter_end_exclusive` are absent from 0/96
receiver groups, and `first_gain_change_sample` is the −1 sentinel in 383,686 of 383,688 cells.
Those are v7 *schema* fields that this firmware never wrote.

---

**The engineering decision is again unchanged. The second closure is retracted too.** The
correct statement is **"we have not measured it"**, not "it is zero" — and the reason to stop
is the ≤1.4%-of-RMSE ceiling, which is a budget argument that stands on its own.

## Artifacts

**Code** — new files: `spf/dataset/phase_corrected_dataset.py` ·
`spf/calibrations/models/gsc9_arm_lut_per_radio/` ·
`spf/filters/configs/rover2026_phasecorr.yaml`, `rover2026_tbl_{none,arm_lut}.yaml`.
Changes to pre-existing files: **49 insertions, 2 deletions** across
`spf/filters/run_filters_on_data.py` and `spf/scripts/create_empirical_p_dist.py`.
*(An earlier version of this line said "32 inserted lines … 0 modified, 0 deleted". Both halves
were wrong: the table-rebuild commit added 13 more, and widening an `except SyntaxError` to
`except (SyntaxError, NameError)` is a modification, not an insertion.)*

**Analysis** (`spf/filters/reports/phasecorr_direct_pf_20260814_v1/analysis/`):
`why_null.py` ⚠️ *withdrawn, retained* · `geometry_conditioned.py` · `gain_fixed_effects.py`
⚠️ *conclusion withdrawn, code retained* · `power_calibration.py` *(new — the sensitivity law
and break-even table)* · plus the 4 figures under `figures/`.

**Coefficients:** `spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/coefficients/`
— `rfblock/` (28-param physical), `luts62/`, `luts56/`

**Read-only discipline:** no rover capture was modified at any point; the correction is applied
to an in-memory copy of `mean_phase`, verified by assertion. No file was deleted.
