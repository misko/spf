# E-CAL5 — yes, we would have seen it

**Session:** `e_cal5_positive_control_20260807` · captured 2026-08-07 · SPF `5fa45b0`, clean
**Pre-registration:** [`experiments/e_cal5_positive_control/`](../../../../../experiments/e_cal5_positive_control/experiment_readme.md)
— committed, with its falsifying branch, before the capture
**Closes the stated limitation of:** [E-CAL1 arm 1](../e_cal1_rfdc_20260807_v1/REPORT.md) · [arm 2](../e_cal1_arm2_rfdc_tracking_20260807_v1/REPORT.md)

---

## 1. Result

Both E-CAL1 arms returned null, and both reports named the same thing they could
not settle: whether the chain could have seen an RF-DC effect at all. A null is
only worth the sensitivity behind it, and that sensitivity was *inferred* from a
noise floor rather than *demonstrated*.

**It is now demonstrated.**

| | Measured |
|---|---|
| Mixer step 5 → 6 dB (rows 19→20, MIX 1→2) | **7.434°** (median \|·\| 7.118°, sem **0.097°**) |
| LPF-only floor, per 1 dB, same capture | **0.440°** (median \|·\| 0.172°) |
| **Ratio** | **16.89×** |
| Cluster-robust 95% CI on mean \|step\| | **[6.788°, 8.173°]** over 6 clusters |
| Pre-registered gates (≥5× floor, ≥1.5°, sem < 0.35°) | **all cleared** |

**Verdict: `sensitivity_demonstrated`** — the first branch of the pre-registered
decision rule.

**Therefore both E-CAL1 nulls upgrade** from "we saw nothing" to **"we saw
nothing, and we would have seen it"**:

| | RF-DC excess measured | An H₁-sized 2.664° effect would have appeared at |
|---|---|---|
| Arm 1 (tracking ON) | +0.069° ± 0.077 | **34.5σ** |
| Arm 2 (tracking OFF) | +0.019° ± 0.082 | **32.4σ** |

The RF-DC machinery is quiet. It is not that this harness cannot see.

## 2. The measurement is consistent across every cell

Unlike the RF-DC nulls, this step is large enough to resolve in **every**
(radio, LO) cluster individually, including the weak ones:

| Radio | LO (MHz) | Mixer step 5→6 | sem | LPF floor /dB |
|---|---|---|---|---|
| R18 `843ef2` | 4001 | +6.621° | 0.047 | 0.041° |
| R18 `843ef2` | 5100 | +6.641° | 0.351 | 0.561° |
| R18 `843ef2` | 5766 | +8.112° | 0.194 | 0.733° |
| R17 `0a003a` | 4001 | +6.906° | 0.034 | 0.046° |
| R17 `0a003a` | 5100 | +9.021° | 0.040 | 0.077° |
| R17 `0a003a` | 5766 | +7.242° | 0.097 | 0.555° |

Six clusters, same sign, 6.6–9.0°, every sem ≤ 0.351°. Notably **R18 @ 5100 MHz —
the cell that was marginal or failing in both E-CAL1 arms — resolves this step at
+6.641° ± 0.351**, i.e. 19× its own local floor. Its weakness costs precision on a
0.07° effect; it is nowhere near costing detection of a degrees-scale one.

## 3. The step is larger than the campaign median, and that is expected

The campaign's reference for "1 dB step that changes the mixer word" is a **median
of 2.664° over 12 heterogeneous steps** at various LOs and gains. This one measures
7.434°, about 2.8× that.

That is not a discrepancy to explain away — it follows from the audited table. The
5→6 transition moves **more than the mixer**: the LPF word drops four places
(12 → 8) at the same time, and `RF_DC_CAL` toggles.

```
   5   19    LNA 0  MIX 1  TIA 0  LPF 12  RF_DC_CAL 0
   6   20    LNA 0  MIX 2  TIA 0  LPF  8  RF_DC_CAL 1
```

At the ~0.44°/dB floor measured here, four LPF words are worth roughly 1.8°, and
the `RF_DC_CAL` contribution is the ~0.02–0.07° that E-CAL1 measured — negligible.
So the mixer term itself is the bulk of it, and this remains a step *of the class*
the campaign characterised, at a scale independently established to be
degrees-not-tenths.

**For the purpose of this control the exact decomposition does not matter.** What
matters is that a step of this class, at ≥ the H₁ magnitude, is recovered at 16.9×
the floor with a 0.097° standard error. No purified mixer coefficient is claimed
here.

## 4. Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| Completeness | 525/525 per radio | **pass** — 1050/1050 |
| Quality | ≥20 of 25 epochs per cell, circstd ≤ 5° | **pass both radios** — 21/21 cells each |
| Frames | — | R17 525/525 valid; R18 520/525 valid |
| Gain tables | pre/post audits identical | **pass** — all 6 byte-identical, high = `90d34d61…` |
| Harness | untouched since E-CAL1 arm 1 | **pass** |
| Provenance | git SHA + clean flag | **pass** — `5fa45b0`, dirty = False |

This is the only capture of the three today where **both** radios pass the strict
gate outright.

## 5. A bug this experiment caught

The analysis reuses E-CAL1's estimator so the floors are comparable. On first run
it failed: `load_epoch_h` **hardcoded** its gain list as `(5, 8, 9, 10)`, so gain 6
was never read and the mixer series came back empty — surfacing as a `KeyError`
rather than a wrong number, which is the good failure mode.

The list is now a `gain_set` parameter defaulting to E-CAL1's values. **Arm 1 and
arm 2 were re-scored after the change and both reproduce bit-identically**, so the
parameterisation is provably inert for the published results. (A first attempt
named the parameter `gains`, which silently shadowed a local array of the same
name; that is why it is `gain_set`.)

The hash of `analyze.py` in the arm-1 report's `inputs_manifest.json` changes
accordingly, and is updated there with this note.

## 6. What this does and does not license

**Does:** both E-CAL1 arms' nulls are now informative about the physics. The chain
detects a phase step of the size H₁ predicted at >30σ, so the RF-DC machinery's
silence is a property of the machinery, not of the measurement.

**Does not:**

- It does not establish sensitivity at *arbitrary* magnitudes — it demonstrates
  detection at ~7.4° and characterises a 0.44°/dB floor. The claim "we would have
  seen 2.664°" follows from those two numbers plus the estimator's sem, not from a
  measurement made at 2.664° itself.
- It does not purify the mixer coefficient (§3).
- Two radios, one harness topology, three high-band LOs, unchanged since E-CAL1
  arm 1.
- It says nothing about the row-11 (−3 dB) `RF_DC_CAL` edge, still unsampled.

## 7. Reproduce

```bash
python experiments/e_cal5_positive_control/analyze.py \
  artifacts/dual_rx_gain_frequency/e_cal5_positive_control_20260807 results.json
```
