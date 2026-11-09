# E-CAL5 — results (2026-08-07)

**Full report and code hashes:**
[`reports/e_cal5_positive_control_20260807_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal5_positive_control_20260807_v1/REPORT.md)
· raw capture `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_cal5_positive_control_20260807/` (gitignored)

Session `e_cal5_positive_control_20260807`, SPF `5fa45b0` clean,
**1,050/1,050 frames**, harness untouched since E-CAL1 arm 1.

---

## Answer: yes, we would have seen it

| | Measured |
|---|---|
| Known mixer step, 5 → 6 dB (rows 19→20, MIX 1→2) | **7.434°** (median \|·\| 7.118, sem **0.097**) |
| LPF-only floor, per 1 dB, same capture | **0.440°** (median \|·\| 0.172) |
| **Ratio** | **16.89×** |
| Cluster-robust 95% CI on mean \|step\| | **[6.788°, 8.173°]**, 6 clusters |

All three pre-registered gates cleared (≥5× floor, ≥1.5°, sem < 0.35°) →
**`sensitivity_demonstrated`**, the first branch of the decision rule.

**Consequence for both E-CAL1 arms:**

| | RF-DC excess measured | An H₁-sized 2.664° effect would have shown at |
|---|---|---|
| Arm 1 (tracking ON) | +0.069° ± 0.077 | **34.5σ** |
| Arm 2 (tracking OFF) | +0.019° ± 0.082 | **32.4σ** |

Both nulls upgrade from *"we saw nothing"* to **"we saw nothing, and we would have
seen it."** The RF-DC machinery is quiet — not invisible to this chain.

## Resolved in every cell, including the weak one

| Radio | LO (MHz) | step | sem |
|---|---|---|---|
| R18 `843ef2` | 4001 / 5100 / 5766 | +6.621 / +6.641 / +8.112 | 0.047 / 0.351 / 0.194 |
| R17 `0a003a` | 4001 / 5100 / 5766 | +6.906 / +9.021 / +7.242 | 0.034 / 0.040 / 0.097 |

**R18 @ 5100 MHz — marginal or failing in both E-CAL1 arms — resolves this step at
19× its own local floor.** Its weakness costs precision on a 0.07° effect; it is
nowhere near costing detection of a degrees-scale one.

## Honest reading of the magnitude

The step is **2.8× the campaign's 2.664° median** for "1 dB step that changes the
mixer word". The audited table explains it rather than contradicting it: rows
19→20 move the **LPF word four places** (12→8) as well as the mixer, worth roughly
1.8° at the floor measured here, and `RF_DC_CAL` toggles (worth the ~0.02–0.07°
E-CAL1 measured — negligible).

**No purified mixer coefficient is claimed.** A positive control needs a step of
known class at or above the H₁ magnitude, detected with margin. That is what this is.

## Caveats

- Demonstrates detection at ~7.4° against a 0.44°/dB floor. "We would have seen
  2.664°" follows from those two numbers plus the estimator's sem — not from a
  measurement made at 2.664° itself.
- Two radios, one harness topology, three high-band LOs.
- Says nothing about the row-11 (−3 dB) `RF_DC_CAL` edge, still unsampled.

## Gate status

**Both radios pass outright** (21/21 cells each) — the only one of the three
captures on 2026-08-07 where that is true.

## A bug this experiment caught

E-CAL1's `load_epoch_h` hardcoded its gain list `(5, 8, 9, 10)`, so gain 6 was
never read and the mixer series came back empty — as a `KeyError`, not a wrong
number. It is now a `gain_set` parameter defaulting to E-CAL1's values;
**arms 1 and 2 were re-scored and reproduce bit-identically**, so the change is
provably inert for the published results.
