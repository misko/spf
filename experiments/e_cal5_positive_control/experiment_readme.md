# E-CAL5 — positive control: would we have seen it?

**Status:** pre-registered 2026-08-07, to run on the unchanged E-CAL1 harness.
**Est. bench time:** ~25 min for both radios (1,050 frames).
**Results:** [`RESULTS.md`](RESULTS.md)
**Closes:** the stated limitation of
[E-CAL1 arm 1](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260807_v1/REPORT.md)
and [arm 2](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_arm2_rfdc_tracking_20260807_v1/REPORT.md).

---

## 1. Purpose

Both E-CAL1 arms returned null. Arm 1 measured the RF-DC contribution at
+0.069° ± 0.077 and arm 2, with the tracking loop pinned off, at +0.019° ± 0.082.
Neither can distinguish

> **"the RF-DC machinery is quiet"** from **"this harness cannot see RF-DC effects at all."**

That is a real gap, and it is the one both reports name as the thing they cannot
settle. A null is only as strong as the demonstrated sensitivity behind it. Right
now that sensitivity is *inferred* from the noise floor, not *demonstrated*.

This experiment demonstrates it, by measuring a phase step of independently
established magnitude on the same harness, in the same session, with the same
estimator.

## 2. The control step

From the audited high table — verified on both radios in this session:

```
  dB  row  LNA  MIX  TIA  LPF  RF_DC_CAL
   5   19    0    1    0   12      0
   6   20    0    2    0    8      1     <- 1 dB step, MIXER 1 -> 2
   8   22    0    2    0   10      0
  10   24    0    2    0   12      0     <- 8->10 is LPF-only, the floor
```

**5 → 6 dB is a 1 dB step that changes the mixer word.** That is exactly the class
of step the source campaign measured at a median of **2.664°** (n = 12), and that
E-GSC4 independently replicated at **6.04× the same-dataset LPF floor**
(CI [5.11, 7.51]) from a different session. Like those statistics, it carries
whatever LPF change accompanies it — so the comparison is apples-to-apples rather
than a purified mixer term.

**And 2.664° is precisely the H₁ magnitude E-CAL1 pre-registered.** So this asks
the exact question that matters: *if the RF-DC machinery had injected phase at the
size H₁ predicted, would this chain have seen it?*

One capture spans the whole dynamic range:

| Step | What moves | Expected |
|---|---|---|
| 5 → 6 | **MIXER** 1→2 (+ LPF, RF_DC_CAL) | **large** — campaign median 2.664° |
| 8 → 10 | LPF only, per 1 dB | **floor** — 0.197–0.275° measured in the E-CAL1 arms |
| 8 → 9 (from E-CAL1) | RF_DC_CAL edge | **null** — +0.019…+0.069° |

## 3. Approach

Additive cross about a **5 dB reference**, gains {5, 6, 8, 10}, at the same three
high-band LOs (4001 / 5100 / 5766 MHz), 25 epochs — the E-CAL1 design with one
gain added. Same estimator, same `H(g) = [D(g,5) − D(5,g)]/2`, same analysis code.

Everything except the gain set and the seed is copied from
`e_cal1_rfdc_discriminator.yaml`, so the floor measured here is directly
comparable to the floor measured in both arms.

## 4. Hardware setup

**Unchanged from the E-CAL1 arm 1 / arm 2 session — no connector touched.** Two
Plutos, R17 `104000bac495…` and R18 `1040007c4a94…`, each

```text
TX2 o--->[ 30 dB attenuator ]--->[ two-way splitter ]---> RX1
                                          \-----------> RX2
```

Firmware `v0.38-plutoplus-spf-gain-rssi-fingerprint-v3`, QSPI, 2R2T, direct-USB.

## 5. Decision rule

Pre-registered. Do not renegotiate after seeing the data.

| Measured \|ΔH(5→6)\| | Conclusion |
|---|---|
| **≥ 5× the same-session LPF floor** and **≥ 1.5°**, sem < 0.35° | **Sensitivity demonstrated.** The chain detects an RF-state step of the magnitude H₁ predicted for RF-DC. Both E-CAL1 nulls upgrade from "we saw nothing" to "we saw nothing, and we would have seen it." |
| between 2× and 5× the floor | Partial. The chain sees the step but with less margin than the campaign implies; restate both nulls with the demonstrated margin rather than the assumed one. |
| **at the floor** (< 2×) | **Falsifying.** The chain cannot see a known mixer step, so it could not have seen an RF-DC step either, and **both E-CAL1 arms' nulls are uninformative** about the physics. The harness or the estimator would then be the problem, not the RF-DC machinery. |

The floor is the **8 → 10 LPF-only step measured in this same capture**, never
imported from another session.

## 6. Risks

| Risk | Handling |
|---|---|
| The 5→6 step also moves LPF by −4 words, which may partly cancel the mixer term | The campaign's own 2.664° statistic has the same property, so the reference is comparable. Report the LPF-only floor beside it and do not claim a purified mixer coefficient. |
| Low-gain SNR at 5–6 dB, high band | Identical to E-CAL1, which passed; R18 @ 5100 MHz remains the weak cell and is expected to be the marginal one again. |
| A large step could wrap | 2.664° is nowhere near ±180°; wrapping is handled in the estimator regardless. |

## 7. Outputs

`artifacts/dual_rx_gain_frequency/e_cal5_positive_control_20260807/` with pre- and
post-run gain-table audits and per-radio validation, and a committed report
stating the measured step, the floor, their ratio, and which decision-rule branch
was reached.
