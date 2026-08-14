# Gain-phase correction and the rover: what we did, what we found, and where it ends

**2026-08-11 → 2026-08-14.** A synthesis of one investigation, from "which gain-phase model
should the rover use" to "gain-phase is a 2% term and the answer is to stop".

Working branch preserved at **`gainphase-rover-investigation-20260814`** (`f91a2ba`); every
commit is also on `main`.

---

## The question, and the answer

**Question:** which gain-phase model should correct the rover's data, and how much does it buy?

**Answer:** none of them, and essentially nothing. On rover frames the model's predicted
correction explains **0.060% of the phase variance** — 1.75° of predicted correction against an
81.98° phase residual at fixed bearing. Applying it makes the particle filters slightly but
significantly *worse*.

That is a negative, but it was expensive to establish honestly and it retires a line of work
that had been running for weeks.

---

## The chain of experiments

| # | experiment / report | what it settled |
|---|---|---|
| 1 | [`ladder_frames_gsc678_20260813_v1`](../spf/calibrations/dual_rx_gain_frequency/reports/ladder_frames_gsc678_20260813_v1/REPORT.md) | First frame-level ladder fit. The shipped mechanistic family is the **wrong shape**; the anchor gain, not the radio, decides whether the model's core assumption holds. |
| 2 | [`e_gsc9_rover_operating_region`](../experiments/e_gsc9_rover_operating_region/experiment_readme.md) | Preregistered capture of the rover's own operating cells — a 1,600-cell grid covering **100%** of rover frames where prior campaigns covered **0**. |
| 3 | [`e_gsc9` RESULTS](../experiments/e_gsc9_rover_operating_region/RESULTS.md) | Session A executed: 27,380 frames, all quality-valid. H2 falsified on the damaged unit, H3 localised its defect, two gates failed and were retained. |
| 4 | [`rover_model_gsc9_20260814_v1`](../spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/REPORT.md) | Refit on the measured cells. **`mixer + LNA`, 28 parameters**, is the best and most physical model. Coefficients committed. |
| 5 | [`phasecorr_direct_pf_20260814_v1`](../spf/filters/reports/phasecorr_direct_pf_20260814_v1/REPORT.md) | Applied it to the direct PF filters, 1,920 runs. **Significantly worse**, and the negative control degraded similarly. |
| 6 | same report, addendum | Rebuilt the empirical table from corrected φ. Consistency **halved the accuracy penalty and flipped the calibration sign** — but accuracy stayed worse. |
| 7 | `analysis/why_null.py` | **The correction explains 0.060% of rover phase variance.** The question closes. |

---

## What was actually learned

### The model itself is good, and physically legible

`mixer + LNA` — per radio, per arm, 28 parameters — ties a 74-parameter LUT and an
80-parameter four-word model on bench cells, and **transfers across carriers where a LUT
cannot** (0.276° predicting 5840 from 5766 alone). The physics is clean: over 26→62 dB the
AD9361 moves only the baseband LPF, then one LNA step at 40→41, then the mixer. The LPF sits
after the mixer and contributes nothing measurable; the RF-side blocks carry all the phase.
Smooth functions of dB fail outright because the response is a **staircase over discrete
hardware states**.

Its coefficients are diagnostic: they localise the damaged unit's fault to **one number** —
its RX1 LNA switch carries −77.09° where its own RX2 carries −18.10°, a −59.00° arm asymmetry,
independently reproducing E-GSC9's H3 (−59.49°) from a different fit.

### But the term it models is 2% of the rover's phase budget

| quantity | value |
|---|---:|
| φ residual within a 2° bearing bin, rover | **81.98°** |
| predicted correction, sd within a bin | 1.75° |
| **share** | **2.1%** |
| variance of φ explained | **0.060%** |

Even a *perfect* gain-phase correction would address 2% of what moves φ on a flying rover. The
other 98% — multipath, geometry, segmentation — was never in scope of this work.

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
6. **"A same-radio capture per rover unit is the remaining option"** → retired. It would
   recover more of a term that is 2% of the problem.

---

## Where this ends

**Stop here on gain-phase for the rover.** The model is correct, committed, documented, and
worth keeping for bench and hardware-diagnostic use — it found a real fault in R17. It is not
worth deploying, and no further capture changes that.

The blockers that remain are recorded rather than solved, and none is now worth solving *for
this purpose*: the anchor cannot be measured in flight; 69% of rover frames change gain
mid-buffer unguarded; cross-radio transfer is 1.16×, i.e. none.

**What would move rover bearing accuracy is the 81.98° residual**, not the 1.75° inside it.
That is where a next experiment belongs, and it is a different investigation — segmentation,
multipath, and geometry, not gain tables.

---

## Artifacts

**Code** (all additive; 32 inserted lines across two existing files, 0 modified, 0 deleted):
`spf/dataset/phase_corrected_dataset.py` · `spf/calibrations/models/gsc9_arm_lut_per_radio/`
· `spf/filters/configs/rover2026_phasecorr.yaml`, `rover2026_tbl_{none,arm_lut}.yaml`

**Coefficients:** `spf/calibrations/dual_rx_gain_frequency/reports/rover_model_gsc9_20260814_v1/coefficients/`
— `rfblock/` (28-param physical), `luts62/`, `luts56/`

**Read-only discipline:** no rover capture was modified at any point; the correction is applied
to an in-memory copy of `mean_phase`, verified by assertion. No file was deleted.
