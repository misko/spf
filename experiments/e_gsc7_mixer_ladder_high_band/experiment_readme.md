# E-GSC7 — the high-band mixer ladder, 52→62 dB

**Status: DESIGNED, NOT RUN.** Hypotheses and decision rules recorded before any
data is taken. Needs two radios on a bench harness.

> ## ⚠️ Correction (2026-08-12): this preregistration contained a factual error
>
> It states that 5766 MHz is the rover's only carrier. **It is not.** The 2026
> corpus carries **5766 MHz (76.2% of frames) and 5840 MHz (23.8%)**. The error
> propagated into the design: the transferability LOs chosen below are
> 5000/5300/5500/5900, and **5840 — the only other carrier actually in use — was
> omitted.**
>
> This does not affect E-GSC7's execution or its graded outcomes, which stand.
> It does mean H5's failure (5766 → 5300 at 9.06° RMS) leaves the operationally
> important question — whether 5766 transfers the 74 MHz to 5840 — untested.
> [E-GSC8](../e_gsc8_carrier_transfer_5840/experiment_readme.md) exists to close
> that hole. The mistake was mine, in the design brief.

## Why

The committed gain-state phase model
([`spf/calibrations/gain_state_phase_model_v1/`](../../spf/calibrations/gain_state_phase_model_v1/))
**cannot correct a single frame of the 2026 rover corpus.** Measured directly:
0 of 400 frames supported, because the model fitted mixer levels `{0, 1, 2, 4}`
and the rover invokes `{4, 5, 6, …, 15}`.

The cause is structural, not a bug. The 2026-07-30 spectroscopy campaign anchored
at **26 dB** and swept a low-gain regime; the rover AGC runs **46–62 dB** because
it tracks a weak signal at range. Calibration and deployment occupy disjoint
regions of the gain table.

Measured on 50,252 frames from 12 rover captures at 5766 MHz:

| | |
|---|---|
| frames with unequal arm gain | **99.2%** (median \|g1−g2\| = 13 dB) |
| frames needing a correction (RF words differ) | **93.5%** |
| modal mixer pair (arm1, arm2) | **(15, 4) — 68.3% of all frames** |
| gain metadata valid | 100% |

**E-GSC6 already closed part of this.** Its gain list reaches mixer `{1, 2, 4, 5, 15}`,
which would cover **76.0%** of the frames needing correction — that is a *refit*,
not a capture, and should be done first regardless of this experiment.

This experiment closes the remaining **24%**, and does something E-GSC6 could not:
resolve *how* the effect distributes across the mixer ladder.

### The gap is nine integers, and they are confound-free

The audited high-band table across this span:

| dB | LNA | **mixer** | TIA | LPF | RF_DC | what moves |
|---|---|---|---|---|---|---|
| 51 | 3 | 4 | 1 | 24 | 0 | — |
| 52 | 3 | **5** | 1 | 24 | 1 | mixer, rf_dc |
| 53 | 3 | **6** | 1 | 24 | 1 | **mixer only** |
| 54–61 | 3 | **7…14** | 1 | 24 | 1 | **mixer only** |
| 62 | 3 | **15** | 1 | 24 | 1 | **mixer only** |

**From 52 to 62 dB, LNA, TIA, LPF and RF_DC_CAL are all frozen and only the mixer
word advances, one index per dB.** This is the first opportunity in the project to
measure adjacent 1 dB mixer transitions with *every* other RF word held constant.
L10 finding 2 measured the mixer step at 2.664° median but against an LPF-confounded
schedule; E-CAL5's positive control (7.434°) also moved the LPF. Here nothing else
moves.

### And there is a falsifiable prediction waiting

E-GSC6 measured, at 5766 MHz, the shared-effect steps:

```
51 → 52 dB   +0.330°
52 → 62 dB   +5.420°     ← a single jump over NINE unmeasured mixer words
```

If the mixer effect is additive across the ladder, **the nine 1 dB steps this
experiment measures must sum to 5.420°.** That is a strong consistency check
against data already on disk, and it costs nothing extra to evaluate.

## Design

`additive_cross` against the established 26 dB reference, so the results compose
with every prior campaign.

| | |
|---|---|
| gains (dB) | **26** (reference), **52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62** |
| cross pairs | 2 × 11 + 1 = **23** |
| held-out diagonal cells | **11** — `(g, g)` for each new gain, per E-GSC6's method |
| LOs | **5766 MHz required** (the rover's only carrier). Then 5000, 5300, 5500, 5900 for transferability |
| epochs | 3 (anchor-drift gate, per E-GSC6) |
| frames | 34 pairs × 5 LOs × 3 epochs ≈ **510 per radio, ~1,020 for the pair** |

That is **~12% of E-GSC6's 8,784 frames**. If bench time is tight, 5766 MHz alone
is ~204 frames for the pair and still answers H1, H2 and H4.

Config to model on: [`e_gsc6_equal_gain_diagonal.yaml`](../../spf/calibrations/dual_rx_gain_frequency/configs/e_gsc6_equal_gain_diagonal.yaml).
Change `gains-db` and `frequencies-hz`; keep `schedule-design: additive_cross`,
`schedule-reference-gain-db: 26`, and the `held-out-gain-pairs` diagonal pattern.

## Pre-registered hypotheses

| id | prediction | decision rule |
|---|---|---|
| **H1** | Each of the nine 1 dB steps 52→62 produces a **resolvable** mixer step, i.e. median \|Δ\| > 3× the 0.355–0.368° frame-level noise floor | The mixer ladder is real and per-index. Fit `h_mixer` at levels 6–14 and ship. |
| **H2** | The nine steps **sum to 5.420° ± 1°** at 5766 MHz, matching E-GSC6's single 52→62 jump | Additivity holds across the ladder; the model's `H(s1) − H(s2)` form is valid here. **If the sum disagrees, the discrete-state parameterisation is wrong in this region and the model must not be extended by fitting — that is the more important outcome.** |
| **H3** | Step size is **roughly uniform** across mixer 5→15 (no single index dominating by >3×) | A per-index table is warranted. If one index dominates, a two-parameter (threshold) form may be preferable and cheaper to identify. |
| **H4** | Refitting with 53–61 raises rover-corpus coverage from **76.0% to 100%** of frames needing correction | Deploy. Coverage is arithmetic from the gain table, so this is a check that the fit converged, not a discovery. |
| **H5** | `h_mixer(6…14)` measured at 5766 MHz predicts the same indices at 5000/5300/5500/5900 MHz to within the ripple term | The mixer table is frequency-portable within the high band, as L10 finding 1 implies. Failure means per-LO tables are needed and the "universal" claim narrows. |

**Falsifiers.** H1 fails if steps sit at the noise floor — in which case the
5.420° jump is *not* mixer-driven and something else in the 52→62 span is
responsible, which would be a significant finding about the audited table. H2
fails on any material disagreement with E-GSC6; treat that as evidence against
additivity, **not** as a reason to re-measure until it agrees.

## Acceptance gates

| artifact | gate |
|---|---|
| capture | ≥95% quality-valid frames per (radio, LO, gain-pair) cell, per E-GSC6's standard |
| **railing** | `railed_fraction` within its normal band at **every** gain up to 62 dB — see Risks |
| RF words | readback confirms LNA=3, TIA=1, LPF=24 frozen across 52–62 on **both** radios; any drift invalidates the confound-free claim |
| anchor drift | interleaved (26,26) cell agrees across all three epochs, worst single drift < 4° (E-GSC6's observed worst) |
| H2 check | the nine-step sum reported against E-GSC6's 5.420° **before** any refit is attempted |
| `RESULTS.md` | states H1–H5 with numbers, including any falsified |

## Risks

| risk | mitigation |
|---|---|
| **Front-end rails at 53–62 dB on a bench.** The rover uses high gain because the signal is weak at range; a cabled tee with a strong reference tone may saturate. This is the main reason the experiment could fail outright. | E-GSC6 demonstrably captured 62 dB at these LOs, so it is achievable — replicate its TX drive and pad configuration. Pad down the TX rather than reducing RX gain: **the gain state is the independent variable and must not be changed to fix a level problem.** Check `railed_fraction` at 62 dB *first*, before spending time on the ladder. |
| AGC hunting instead of the commanded gain | Manual gain mode, and verify `gain_endpoints_equal` per frame. In the rover corpus **88.9%** of frames change gain mid-buffer; a calibration capture must not. |
| Interpreting 52 dB as a clean mixer step | It is not — 51→52 moves `RF_DC_CAL` as well as the mixer. E-CAL1 showed RF-DC injects no resolvable phase (+0.069° ± 0.077), so it should be benign, but 52 is the one rung with a second word moving. Report it separately. |
| Two units is not a distribution | Same caveat as L-GSC6. Report per-radio; do not pool without showing agreement. Prefer the untouched control unit for absolute numbers — L-GSC6 found R17's harness inflated its high band 2–5.5×. |
| Fitting to a wrong-shaped model | H2 exists precisely to catch this. Run the sum check before fitting. |

## Inputs

| | |
|---|---|
| harness | two radios, bench, same TX/pad configuration as E-GSC6 |
| firmware | E-GSC6's pinned build (`v0.38-plutoplus-spf-gain-series-v4-rc17`) or newer with recorded provenance |
| prior data to compare against | [`equal_gain_diagonal_20260811_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/equal_gain_diagonal_20260811_v1/) — the 5.420° figure for H2 |
| model to extend | [`gain_state_phase_model_v1/coefficients/l26_pooled_v1.json`](../../spf/calibrations/gain_state_phase_model_v1/coefficients/) |
| deployment target | the 2026 rover corpus, 5766 MHz, `/mnt/qnap01/mouse9911/rovers_2026/merged/` (read-only) |

## Do this first, before any bench time

**Refit the existing model to include mixer 5 and 15 from E-GSC6's data.** That is
a computation on data already on disk and takes rover coverage from **0% to ~76%**.
It also de-risks this experiment: if the refit's mixer-15 coefficient does not
reproduce E-GSC6's own held-out metrics, the problem is in the fitting path rather
than in missing capture, and no amount of bench time will fix it.
