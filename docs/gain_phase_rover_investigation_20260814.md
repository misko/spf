# Gain-phase correction and the rover: what we did, what we found, and where it ends

**2026-08-11 → 2026-08-14.** A synthesis of one investigation, from "which gain-phase model
should the rover use" to "the donor model we have does not help, and the physical question is
still open". Includes a **retraction** of this document's original closing argument.

Working branch preserved at **`gainphase-rover-investigation-20260814`** (`f91a2ba`); every
commit is also on `main`.

---

## The question, and the answer

**Question:** which gain-phase model should correct the rover's data, and how much does it buy?

**Answer for the model we have:** do not deploy it. The R18-derived **held-out donor**
correction produces no detectable improvement end to end, and a small accuracy penalty in the
particle filters.

**What is NOT established:** that gain-phase is physically negligible. An earlier version of
this document claimed the correction explained 0.060% of rover phase variance and that "even a
perfect correction cannot matter". **That analysis was wrong and is retracted** — see the
retraction section below. A null on a held-out donor does not bound a same-radio or
sample-weighted correction.

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
   strength of the withdrawn 2% argument. **That retirement is itself withdrawn.** It is not
   justified *yet* — the no-new-capture check below should come first — but the reasoning I
   used to rule it out was wrong.
7. **"The correction explains 0.060% of rover phase variance"** and everything built on it →
   withdrawn; see the Retraction section. The correct figure is r² = 0.019% against a 36.7°
   geometry-conditioned residual, and it bounds the **donor**, not the physics.

---

## Where this ends

**Pause dedicated rover gain-phase work.** The model is the best parsimonious description of
the Session-A bench measurements, committed and documented, and worth keeping for bench and
hardware-diagnostic use — it localised a real fault in R17. The currently tested donor
correction is not worth deploying.

It is **not** established that no correction could matter. Sessions B and C are **terminated by
decision for rover deployment**, but they remain required before any broader temporal-transfer
or physical-discriminator claim; G8 and G9 failed and stand.

The blockers that remain are recorded rather than solved, and none is now worth solving *for
this purpose*: the anchor cannot be measured in flight; 69% of rover frames change gain
mid-buffer unguarded; cross-radio transfer is 1.16×, i.e. none.

The geometry-conditioned residual is **36.7°**, and identifying what composes it — multipath,
GPS/heading error, segmentation, oscillator effects — is a different and larger investigation
than gain tables. The one **low-cost, no-new-capture** check that remains for gain-phase is to
fit gain-state fixed effects directly to that residual, with capture-level cross-validation and
no donor model at all. That would bound the physical question properly, which this work did
not.

---

## Retraction

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

## Artifacts

**Code** (all additive; 32 inserted lines across two existing files, 0 modified, 0 deleted):
`spf/dataset/phase_corrected_dataset.py` · `spf/calibrations/models/gsc9_arm_lut_per_radio/`
· `spf/filters/configs/rover2026_phasecorr.yaml`, `rover2026_tbl_{none,arm_lut}.yaml`

**Coefficients:** `spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/coefficients/`
— `rfblock/` (28-param physical), `luts62/`, `luts56/`

**Read-only discipline:** no rover capture was modified at any point; the correction is applied
to an in-memory copy of `mean_phase`, verified by assertion. No file was deleted.
