# Gain-phase correction and the rover: what we did, what we found, and where it ends

**2026-08-11 → 2026-08-14.** A synthesis of one investigation, from "which gain-phase model
should the rover use" to "do not deploy it, and we cannot yet measure whether a better one
would help". Includes **three retractions** — of this document's original closing argument, of
the analysis that was supposed to replace it, and of the cost/benefit ceiling that was left
carrying the recommendation.

Working branch **`gainphase-rover-investigation-20260814`**, kept in step with `main`; the
final commit of the investigation is the tip of both. (An earlier version of this line pinned
`f91a2ba`, which stopped being the tip three commits later.)

---

## FINAL CONCLUSION

Five statements, each with its evidence and its status.

| # | conclusion | status |
|---|---|---|
| 1 | **Do not deploy the gain-phase correction we have.** The R18-derived held-out donor changes the geometry-conditioned rover residual by **+0.009°, capture-clustered 95% CI [−0.024, +0.044]** (42 physical captures, 84 streams, 134,224 frames) — indistinguishable from zero — and costs a small but reproducible accuracy penalty end to end in both direct particle filters. | **Supported. Act on this.** |
| 2 | **The physical question is NOT closed, and this investigation cannot close it.** Gain-state fixed effects fitted on the rover data itself return +0.018° to +0.087° on held-out captures. That was reported as an upper bound. It is not one: the statistic is **quadratically insensitive** to a small phase term — a *perfect* correction of the deployed donor's true **1.78–2.70° rms** could move it by only **+0.032° to +0.074°**, and the sign flips with a nuisance parameter (`min_n` 8 → 25 gives −0.013°) against a fold-seed sd of 0.033°. Measured directly by injecting the real per-cell correction: **+0.045°**. The experiment was **under-powered**, not blind — an earlier version of this row said the ceiling was "no larger than one sd", which was itself computed from a MAD mistaken for an rms. | **Overclaimed twice. Retracted twice. See [Retraction 2](#retraction-2--the-upper-bound-was-not-one).** |
| 3 | **Keep the bench model.** `mixer + LNA`, 28 parameters per radio-carrier, is the most parsimonious description of the E-GSC9 Session-A measurements and localises R17's fault to a single coefficient (−77.09° on RX1's LNA switch against −18.10° on its own RX2). It is valuable for bench work and hardware diagnostics. | **Supported.** |
| 4 | **Decline a same-radio bench campaign on cost/benefit, not on physics.** ⚠️ The **"≤1.4% ceiling" is WITHDRAWN** — it was wrong four ways at once (see [Retraction 3](#retraction-3--the-14-ceiling-was-not-a-ceiling)). What replaces it needs no conversion at all: `arm_lut` minus `constant` **is** the gain-dependent term, and it has already been run end to end. Across 1,920 runs its entire effect was **−0.059° single-radio (p = 0.485)** and **+0.283° dual-radio (p = 0.101)** — under **0.6%** of filter RMSE, **opposite signs, neither significant.** That justifies declining a 2.6 h/radio campaign. It is **not** a bound, and it does **not** say such a campaign would find nothing. | **Rescoped and re-founded** on a measurement instead of a derivation. |
| 5 | **The remaining work is the 35–37° residual itself** — multipath, GPS/heading, segmentation — which is a different and larger investigation. **That is where rover bearing accuracy actually lives.** | **Supported, and unaffected by any of the three retractions.** |

**In one sentence:** the correction we built does not help the rover and should not ship, the
model behind it is good bench physics worth keeping, and whether gain state carries *any*
usable phase information on these radios **remains unmeasured** — twice asserted here, twice
retracted, and the honest statement is "we have not measured it", not "it is zero".

### What changed, and why you should trust this version less than its confidence suggests

**Two closure claims and one cost/benefit ceiling have now been retracted from this document.**

The first conditioned on `rx_theta_in_pis`, believing it was ground-truth bearing. It is the
array mount orientation and is **constant per receiver per capture**, so the analysis compared
a 1.75° correction against the entire trajectory's phase motion and concluded gain-phase was a
2% term.

The second — added to *fix* the first — fitted gain-state effects on the rover data itself and
read a positive held-out change as proof of absence. It was underpowered by construction: the
metric cannot resolve the effect size at issue. **Both were caught in external review, not by
me, after being committed and reported.**

The third was the ceiling left carrying the recommendation once the physics claim was gone: a
"≤1.4% of filter RMSE" figure that mistook a MAD for an rms, divided the largest numerator by
the largest denominator, used a fixed phase→bearing factor belonging to a *different array*,
and put an L1 numerator over an L2 denominator.

The pattern is the same all three times: **a number was quoted without tracing what it actually
measured.** First the wrong variable, then the wrong power, then the wrong moment. Withdrawn
numbers are listed in [Retraction 1](#retraction-1--the-wrong-angle-variable),
[Retraction 2](#retraction-2--the-upper-bound-was-not-one) and
[Retraction 3](#retraction-3--the-14-ceiling-was-not-a-ceiling).

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
| gain-phase term at stake — **MAD** about the weighted mean | 1.0–1.9° |
| the same term as an **rms**, which is what this law takes | **1.78–5.65°** (R18, the deployed donor: **1.78–2.70°**) |
| ceiling on Δ(mean \|e\|) from removing it *perfectly* | **+0.032 to +0.074°** (R18); +0.032 to +0.322° over all four |
| fold-seed sd of the published statistic (8 seeds, 6-fold CV) | **0.033°** (range +0.002 … +0.111) |
| same statistic at `min_n` 25 instead of 8 | **−0.013°** (sign flips) |

The measured law is **Δ(mean \|e\|) = 0.0101 · A²** degrees for a term of rms amplitude *A*,
on the actual 124,950 residuals — nearly quadratic, because a small offset buried in a 49.2°
residual barely moves a mean-absolute statistic. *(For a real gain-shaped term it is not purely
quadratic: measured Δ = 0.00692·A + 0.00761·A², and the linear part dominates below 0.91°.)*
At the corrected amplitude a perfect correction is worth **+0.032 to +0.074°** against a
fold-seed sd of 0.033° and a run-to-run range of +0.002 to +0.111°. **So the positive numbers
are consistent with a real 2° effect, with no effect at all, and with anything in between.**

⚠️ **This document has asserted closure twice, retracted it twice, and has since withdrawn the
cost/benefit ceiling that replaced it.** See
[Retraction 1](#retraction-1--the-wrong-angle-variable) and
[Retraction 2](#retraction-2--the-upper-bound-was-not-one). What survives is the engineering
decision, which rests on the end-to-end sweep and on the direct measurement in
[Retraction 3](#retraction-3--the-14-ceiling-was-not-a-ceiling) — not on either withdrawn argument.

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
| 8 | `analysis/geometry_conditioned.py` | Corrected the angle variable: a null **for the donor**, not for the physics. ⚠️ Its numbers are **superseded by #13** — its keep-first dedup discarded 9,274 frames that were disjoint in time. |
| 9 | `analysis/gain_fixed_effects.py` | ⚠️ **NOT an upper bound — underpowered by construction.** Gain-state fixed effects fitted on rover data, 6-fold CV by physical capture, give +0.018° to +0.087° on held-out captures. The statistic's ceiling for the effect at issue is +0.010–0.038°, no more than one sd of its own 0.033° fold-seed noise. **Non-informative, in either direction.** |
| 10 | `analysis/power_calibration.py` | The sensitivity law that should have been computed *before* #9: **Δ(mean \|e\|) = 0.0101·A²**. ⚠️ Its parameter-counting break-even estimates are **withdrawn** — see #11. |
| 11 | `analysis/lut_injection_power.py`, `detection_threshold.py` | **Measured** detection thresholds, by injecting the deployed LUT along each stream's real gain sequence: **3.08° (cell), 3.41° (arm), 2.91° (rfblock), 0.29° (1-param)** against the donor's true **2.08° rms**. The parameter-counting heuristic predicted a 2.58× spread; the measured spread is **1.17×** and the ordering *inverts*. Also: an i.i.d. Gaussian is **orthogonal** to the estimator — recovery 35–153% for a gain-shaped term, **0%** for a Gaussian of identical rms. |
| 12 | `analysis/true_amplitude.py` | The amplitude audit. "1.0–1.9°" is a **MAD**; the rms about the same weighted mean is **1.78 / 2.70 / 4.18 / 5.65°**. Observed rms/MAD = 1.81–3.02, so even a Gaussian 1.253 conversion would have been wrong. |
| 13 | `analysis/geometry_conditioned_v2.py` | The canonical donor number, with the dedup fixed: **+0.009°, capture-clustered 95% CI [−0.024, +0.044]** over **134,224** frames (the old keep-first rule discarded 9,274 frames that were disjoint in time). |

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
| physical captures / streams / frames | 42 / 84 / **134,224** |
| mean \|e\| without correction | **36.712°** |
| mean \|e\| with correction | 36.721° |
| change | **+0.009°, capture-clustered 95% CI [−0.024, +0.044]** |
| better on | 45/84 streams |
| corr(correction, residual) | +0.0132, r² = 0.017% |

*(Canonical values, from `analysis/geometry_conditioned_v2.py`. The first published version read
124,950 frames / 36.728° / +0.017° [−0.020, +0.061]; its keep-first dedup discarded 9,274 frames
that were disjoint in time, not duplicated. The correction moves the result further toward the
null. The capture-clustered bootstrap turned out immaterial — 0.9995× the per-stream width.)*

**This bounds the donor, not the physics.** A mismatched predictor is attenuated toward zero
correlation even when the underlying term is real, so it says nothing about a same-radio or
sample-weighted correction. The bench-measured gain term (1.0–1.9° MAD, 1.78–5.65° rms) is small against a
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
   now declined **on cost/benefit** — the term applied end to end moves bearing RMSE by −0.06°
   (single, p = 0.485) / +0.28° (dual, p = 0.101) — which is a different and much weaker
   statement than "it would find nothing".
7. **"The correction explains 0.060% of rover phase variance"** and everything built on it →
   withdrawn; see [Retraction 1](#retraction-1--the-wrong-angle-variable). The correct figure is r² = 0.019% against a 36.7°
   geometry-conditioned residual, and it bounds the **donor**, not the physics.
8. **A second silent-stratification bug**, caught before reporting: the first run of
   `gain_fixed_effects.py` read `gain_endpoints_equal` from `cached_keys`, where it does not
   exist, so it silently defaulted to all-True — making the "stable gain" stratum identical to
   "all" and leaving "unstable" empty. It surfaced only because the empty stratum divided by
   zero. **This is the same class of error as the `rx_theta_in_pis` bug** — reading a field
   that is not what its name suggests — and it was caught by a crash rather than by design.
10. **"1.0–1.9° (bench-measured sd)"** → it is a **MAD about the weighted mean**, correctly
   labelled as such where it is defined and mislabelled everywhere it was consumed. The
   sensitivity law takes a *second* moment, so the mistake was squared. The rms about the same
   weighted circular mean is **1.78 / 2.70 / 4.18 / 5.65°**; observed rms/MAD is **1.81–3.02**,
   so even a Gaussian 1.253 conversion would have been wrong by 1.4–2.4×.
11. **"A prototype separates an injected 1.4° effect (α̂ = 0.814, CI [+0.595, +1.018]) from none
   (α̂ = −0.186, CI [−0.412, +0.063]) at disjoint CIs"** → **withdrawn.** That prototype exists
   in **no committed file**; I published a confidence interval that cannot be reproduced from
   anything in the repo. Implemented properly, the real-data result is **inconclusive**:
   β̂ = +1.18° rms, CI [−0.675, +2.433]. The injection-recovery check does pass (slope 0.99).
12. **"E-GSC9 Sessions B and C were terminated by decision; the re-calibration interval is
   unmeasured"** → **false for Session B.** It ran to completion and **H6 is falsified on all
   four strata**. The artifacts were on disk before the report that says otherwise was written.
   The engineering consequence is smaller than the MAE suggests: |bias| ≈ MAE on all four
   strata, so the drift is almost entirely a per-session **constant**, which the empirical table
   absorbs. The sd about it is **0.199° / 0.294°** (clean unit) and 0.723° / 0.578° (damaged) —
   1.6–4.2% of the removable variance on a healthy radio.
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
otherwise in error. **Session C is terminated by decision** and H7 remains unrun. ⚠️ **Session B
was NOT terminated — it ran to completion on 2026-08-14 and H6 is FALSIFIED on all four
radio×carrier strata** (circular MAE 3.220° / 1.115° on the clean unit, 2.781° / 5.090° on the
damaged one, against a preregistered 0.5° gate, at 273/273 cell coverage and a valid 12.2–12.8 h
separation, both radios `validation: pass`). An earlier version of this section said it never
ran; see correction 12. G8 and G9 failed and stand.

The blockers that remain are recorded rather than solved, and none is now worth solving *for
this purpose*: the anchor cannot be measured in flight; 66% of rover frames change gain
mid-buffer unguarded (69% at 5766 MHz, 45% at 5840); cross-radio transfer is 1.16×, i.e. none.

**Why the work stops here is a budget argument, not a physics one.** ⚠️ The derived "≤1.4% of
filter RMSE" ceiling that used to sit here is **WITHDRAWN** — it was wrong four ways at once
(see [Retraction 3](#retraction-3--the-14-ceiling-was-not-a-ceiling)). What replaces it needs no
conversion and no functional form, because the experiment was already run:

> **`arm_lut` minus `constant` *is* the gain-dependent term.** Both arms apply the same ~6.3°
> per-receiver constant; only `arm_lut` adds the gain-dependent variation. Across the committed
> 1,920 runs that difference is **−0.059° single-radio (p = 0.485)** and **+0.283° dual-radio
> (p = 0.101)** — under **0.6%** of filter RMSE, **opposite in sign between the two families,
> and neither significant.**

That is a sound reason to stop. It is not a finding that the effect is absent, and it should not
be quoted as one. Two caveats travel with it: the measurement sits at an **out-of-distribution
operating point** (φ already shifted ~6.3° away from where the frozen empirical table was
fitted), and **nothing committed propagates a phase term of known amplitude through the actual
particle filter.** Until that is done this is a measurement of limited power, not a bound.

The geometry-conditioned residual is **35–37°**, and identifying what composes it — multipath,
GPS/heading error, segmentation, oscillator effects — is a different and larger investigation
than gain tables. **That is where rover bearing accuracy actually lives.**

### What would actually settle it

Not "nothing further is warranted" — rather, nothing further is *warranted at this priority*.
If the question is reopened, the low-cost route needs **no new capture**:

1. **Audit the two load-bearing numbers** above. Hours.
2. **Replace the primary statistic.** A one-parameter circular projection of the residual onto
   a *hypothesised* LUT shape, capture-clustered bootstrap. Its **measured** break-even is
   **0.29°** against the free fit's 2.91–3.41°, so unlike everything else here it can see a 2°
   term. ⚠️ It has now been run for the first time, and it is **inconclusive**: β̂ = **+1.18° rms,
   capture-bootstrap 95% CI [−0.675, +2.433]**, 88% of resamples positive, injection-recovery
   slope 0.99 (unbiased). The deployed table's own 2.08° amplitude sits *inside* that CI — so
   the effect is neither detected nor excluded. The CI is **capture-limited, not frame-limited**;
   doubling 42 → ~84 captures would roughly halve it. *(An earlier version of this item cited
   "α̂ = 0.814, CI [+0.595, +1.018]" from a prototype that was never committed. Those numbers are
   withdrawn — see correction 11.)*
3. **Add the controls `gain_fixed_effects.py` lacked** — per-capture CI, seed spread, `min_n`
   sweep, and a **run-preserving** null (circular shift of the gain-key sequence within each
   stream; a plain within-stream shuffle is *not* adequate, because gain is held for long runs
   and the residual has lag-1 autocorrelation 0.573).
4. **Fix the two known code defects** before any re-run: the dedup drops 6 merged stores /
   9,274 frames / 12 streams (6.9%) that are strictly *disjoint in time*, not duplicates; and
   `arm`/`rfblock` sum marginal conditional means rather than fitting jointly.

Only a **same-radio bench LUT** can test the per-radio hypothesis directly — the free fit
provably cannot reach it — and only **protocol-v3 firmware** would make a sample-weighted
trajectory model computable at all. Both require new capture. Decline them on the measured
end-to-end effect if you decline them; not on `gain_fixed_effects.py`, and not on the
withdrawn ≤1.4% ceiling.

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
**+0.032 to +0.074°** (using the corrected rms, not the MAD) — against a seed-to-seed sd of
**0.033°**, a run-to-run range of +0.002 to +0.111°, and a `min_n` sensitivity that
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
correct statement is **"we have not measured it"**, not "it is zero" — and the reason to stop is
a budget argument. ⚠️ *At the time this was written that budget argument was the "≤1.4%-of-RMSE
ceiling", which has itself since been withdrawn; see
[Retraction 3](#retraction-3--the-14-ceiling-was-not-a-ceiling). What now carries it is the
direct end-to-end measurement: **−0.059° single, +0.283° dual, neither significant.***

---

## Retraction 3 — the "≤1.4% ceiling" was not a ceiling

A third review, on 2026-08-14, accepted the two scientific retractions but found that the
overstatement had **moved into the engineering quantification**. It is right. The ceiling that
was left carrying the whole recommendation was wrong in four independent ways, and the
corrections do not all point the same direction.

**1. The amplitude was the wrong moment.** "1.0–1.9°" is a **MAD**; the law it was fed to takes
an rms. The true rms is **1.78–5.65°** (1.78–2.70° for the only donor actually deployed).
Observed rms/MAD is 1.81–3.02, so the Gaussian 1.253 the reviewer proposed would also have been
wrong. **This makes the ceiling bigger**: +0.032 to +0.074° for the deployed donor, against the
0.033° fold-seed sd. The sentence "no larger than one sd of the statistic's own fold-seed noise"
is **withdrawn**; the honest word is *under-powered*, not *blind*.

**2. The arithmetic did not follow from its own inputs.** 0.76° ÷ 41.09° (the *smaller* filter
RMSE) is **1.85%**, not 1.37%. The published figure paired the largest numerator with the
largest denominator.

**3. The conversion is not a constant.** For a 2-element array \|dθ/dφ\| = 1/(2π(d/λ)·\|cos θ\|)
— a function of both spacing and bearing that **diverges at endfire**. Measured framewise over
177,410 frames: median **0.296**, IQR [0.237, 0.545], P90 **1.333**, P95 **2.573**; **34.6% of
frames exceed 0.40**, and the mean and rms both diverge. The 0.40 corresponds to d/λ = 0.398 —
the *wall array* — while every rover capture is **d/λ = 0.673–0.916** and spatially aliased.
*(In fairness: as a mean-absolute conversion 0.40 turns out to be defensible — re-inverting
through the repo's own `phase_diff_to_theta` gives 0.41–0.49 °/°. It is the use of it, not the
value, that fails.)*

**4. An L1 numerator over an L2 denominator.** 1.0–1.9° is mean-absolute; 41–56° is an RMSE.
Like-for-like the rms displacement is 3.50 °/° at A = 1.78°, 12× the mean-absolute figure. And
if the removed term is approximately *independent* of the remaining error, the reduction is
quadratic, not linear — 0.017% rather than 1.4%, a factor of ~100 the other way.

Corrected, the derived band spans **0.01% to 2.9%** depending on two choices nobody has
measured. **A quantity with a 300× range is not a ceiling.**

**5. The deeper problem: the pipeline never inverts a sine.** φ is quantised into 65 bins of
5.5385° and read into a multimodal empirical p(θ\|φ). A 1.0° correction leaves the likelihood
**bit-identical on 82.1% of frames**, and the median shift in the table's circular-mean θ is
**exactly 0.000°**. A local derivative — corrected or not — is the wrong instrument entirely.

### What replaces it

The measurement that needs no conversion, and which was sitting in the committed sweep the
whole time: **`arm_lut` minus `constant` is exactly the gain-dependent term.** Over 1,920 runs,
**−0.059° single-radio (p = 0.485)** and **+0.283° dual-radio (p = 0.101)** — under 0.6% of
filter RMSE, opposite signs, neither significant.

**The recommendation is unchanged. It is now founded on a measurement rather than a
derivation** — and it is a measurement of limited power at an out-of-distribution operating
point, not a bound.

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
`why_null.py` ⚠️ *withdrawn, retained* · `geometry_conditioned.py` ⚠️ *superseded* ·
`gain_fixed_effects.py` ⚠️ *conclusion withdrawn, code retained* · `power_calibration.py`
⚠️ *amplitude amended, break-even section withdrawn* · **`geometry_conditioned_v2.py`** (the
canonical donor number, dedup fixed) · **`true_amplitude.py`** (MAD vs rms) ·
**`lut_injection_power.py`**, **`detection_threshold.py`** (measured detection thresholds and
the one-parameter projection) · **`jacobian_audit.py`** (the phase→bearing audit) ·
**`verify_disjoint.py`** · plus the 4 figures under `figures/`.
Session-B verification: `experiments/e_gsc9_rover_operating_region/analysis/audit_session_b_h6.py`.

⚠️ **Every number quoted in this document is now produced by a committed script.** That was not
true of the α̂ projection figures, which is why they were withdrawn (correction 11).

**Coefficients:** `spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/coefficients/`
— `rfblock/` (28-param physical), `luts62/`, `luts56/`

**Read-only discipline:** no rover capture was modified at any point; the correction is applied
to an in-memory copy of `mean_phase`, verified by assertion. No file was deleted.
