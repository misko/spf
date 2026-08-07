# E-GSP2 — harness pad sweep: is the ripple actually a reflection?

**Status:** needs parts (a characterised pad set and a second splitter) and one
free computational prerequisite. See [Blockers](#blockers).
**Est. bench time:** ~2 hours of capture across the ABABA sequence, plus
characterisation.
**Queue entry:** [`docs/future_experiments.md` → E-GSP2](../../docs/future_experiments.md)

---

## 1. Purpose

The frequency half of the gain-state model rests entirely on one hypothesis:

> An LNA state change alters the receiver input impedance. Against a mismatched
> source, the round trip produces a standing wave whose phase contribution is
> periodic in frequency with period `1/τ`.

The supporting evidence is real but indirect: two fitted delays (2.54 and
0.92 ns) that agree across radios, an amplitude ordering that inverts across the
4 GHz band edge exactly as the gain tables predict, a retrodiction of the
433/600 MHz anticorrelation, and one pad experiment. **Nothing has ever measured
a reflection coefficient, and nothing has varied the mismatch.**

This experiment varies the mismatch and asks whether the ripple responds the way
a reflection must.

## 2. Hypothesis

For a standing wave, the pad is either **inside** the reflecting loop (the wave
traverses it twice, `n = 2`) or **outside** it (`n = 0`). **No single geometry
gives `n = 1`.** A fitted non-integer `n` on one resolved peak therefore means an
unresolved *mixture* of both paths, which predicts a **saturating**, not linear,
amplitude curve:

```text
A(L) = | A_in · 10^(−L/10)  +  A_out |
```

where `L` is the pad's insertion loss in dB, `A_in` is the component inside the
loop, `A_out` the component outside it.

**H₁ (reflection):** `A(L)` saturates as above, with `A_in` dominant. Ripple
amplitude for a new harness then becomes **predictable from insertion loss
alone** — the practical prize.

**H₀ (not a reflection):** `A(L)` is flat. Pad attenuation does not change the
ripple at all, the "reflection" language must be retired from every document,
and the ripple term reverts to a purely empirical basis.

### What the one existing data point actually says

Stage B's pad (measured amplitude change −10.49 dB) took the treated arm's
2.5475 ns component to 0.99°. But the **unpadded reference is not a single
number** — it is 5.34° in stage A, 10.80° in D and 10.40° in G, all nominally the
same harness state:

| unpadded reference | amplitude | suppression | implied `n` |
|---|---:|---:|---:|
| A | 5.34° | 14.65 dB | 1.40 |
| D (restored) | 10.80° | 20.76 dB | **1.98** |
| G (12 h, hot) | 10.40° | 20.43 dB | **1.95** |

**Round-trip (`n = 2`) is entirely consistent with existing data** — matched by
both D and G. Anchoring on stage A alone gives 1.40 and appears to exclude it,
but that is an artifact of the documented failed A→D restoration. This is exactly
why the unpadded reference must be re-measured immediately before **and** after
each pad state rather than taken once at the start.

## 3. Approach

Two arms, each an **ABABA sequence** with restoration checks. Treated radio only;
control radio untouched throughout, providing the common-mode reference that
removes shared drift.

**Arm (a) — pad sweep.** At least four values spanning 3 / 6 / 11 / 20 dB on the
treated arm's RX1. Every pad **VNA-characterised for actual insertion loss** over
400–6000 MHz, not trusted at its nominal value — stage B's "11 dB" measured
10.49 dB. Fit `A_in`, `A_out` per delay component.

**Arm (b) — splitter swap.** A different unit, and a deliberately better-matched
unit, to change `Γ_s` without changing electrical length. This separates
"mismatch magnitude" from "path length".

Sequence per pad state, with the reference re-measured either side:

```text
  [ unpadded ] → [ pad L ] → [ unpadded ] → [ pad L ] → [ unpadded ]
       ^              ^            ^
       |              |            +-- restoration check: must return within
       |              |                unchanged-harness repeatability
       |              +--------------- treatment
       +------------------------------ reference for THIS pad state
```

E-CAL4's characterised-length insertion is the natural third arm and stays where
it is designed.

## 4. Hardware setup

### 4.1 Radios

| | |
|---|---|
| Count | **2 × PlutoSDR (Pluto+)** |
| **Treated** | R17 `104000bac4950008230026001b440a003a` — RX1 arm is modified |
| **Control** | R18 `1040007c4a94000211000b009186843ef2` — **never touched** for the whole sequence |
| Provisioning | 2R2T, direct-USB gain/RSSI firmware, RAM-loaded, QSPI boot |
| Firmware release | `v0.38-plutoplus-spf-gain-rssi-fingerprint-v3` |
| TX | one radio's TX2 active at a time |

Treatment effects are computed as a difference of differences:

```text
effect = (treated_stage − treated_baseline) − (control_stage − control_baseline)
```

so anything that drifts on both radios cancels. **The control radio's harness
must not be disturbed at any point**, including when swapping the treated pad.

### 4.2 Physical schematic

```text
+============== PLUTO A : R17  — TREATED ======================================+
|                                                                              |
|   TX2 o---[a]--->  [ 30 dB attenuator ]  ---[b]--->  [ two-way splitter ]     |
|                                                          |         |         |
|                                                         [c]       [d]        |
|                                                          |         |         |
|                                              ####################  |         |
|                                              #  PAD UNDER TEST  #  |         |
|                                              #  3 / 6 / 11 / 20 #  |         |
|                                              #   dB, or NONE    #  |         |
|                                              ####################  |         |
|                                                          |         |         |
|   RX1 o<-------------------------------------------------+         |         |
|   RX2 o<-----------------------------------------------------------+         |
|                                                                              |
|          ^^^ ONLY this arm changes. RX2 is never disturbed.                  |
+==============================================================================+

+-------------- PLUTO B : R18  — CONTROL, UNTOUCHED ---------------------------+
|                                                                              |
|   TX2 o---[e]--->  [ 30 dB attenuator ]  ---[f]--->  [ two-way splitter ]     |
|                                                          |         |         |
|                                                         [g]       [h]        |
|   RX1 o<-------------------------------------------------+         |         |
|   RX2 o<-----------------------------------------------------------+         |
+------------------------------------------------------------------------------+

  Arm (b), splitter swap: replace the treated radio's splitter only,
  keeping cables [c] and [d] and the pad state fixed.
```

### 4.3 Passive parts and adapters

| Ref | Item | Requirement | Part / serial | Measured IL (dB) |
|---|---|---|---|---|
| — | Pad 3 dB | SMA, DC–6 GHz | | |
| — | Pad 6 dB | SMA, DC–6 GHz | | |
| — | Pad 11 dB | SMA, DC–6 GHz | | |
| — | Pad 20 dB | SMA, DC–6 GHz | | |
| — | Splitter #1 (baseline) | two-way, DC–6 GHz, SMA | | — |
| — | Splitter #2 (swap) | two-way, **different match** | | — |
| — | 30 dB attenuator ×2 | one per radio, unchanged | | — |
| a–h | Cables | SMA, phase-stable; c/d and g/h matched pairs | | — |

**Measured insertion loss is mandatory, not optional.** `n` is computed as
`suppression_dB / L_dB`; a nominal-vs-actual error of 0.5 dB moves `n` by ~5%,
and the whole question is whether `n` is nearer 1 or 2. Characterise each pad on
the VNA over the full 400–6000 MHz span and record the frequency dependence, not
a single number.

**Connector discipline.** Every pad change is a connector operation, which is the
exact failure mode that made stage C inconclusive. Hence ABABA: each treatment is
bracketed by a restoration that must return within unchanged-harness
repeatability, and each pad state carries its own immediately-adjacent unpadded
reference.

## 5. Software setup

| | |
|---|---|
| Repo / env | `/home/pi/spf`, `/home/pi/spf-virtualenv` |
| Config | to be written — model on `configs/spectroscopy_campaign.yaml` stage A, restricted to the gain pairs that drive the ripple |
| Gain set | the `(45, 26)` and `(26, 45)` pair drives the largest ripple (ΔLNA ≠ 0); include `(5, 26)`/`(26, 5)` as the ΔLNA = 0 control |
| LO set | full 400–5900 MHz comb at 50 MHz — **already alias-free**, 7.85 samples per 392.5 MHz period, delay ceiling 10 ns |
| Epochs | 3 per stage is sufficient; the precision here comes from the pad axis, not repeats |

The campaign harness supports operator checkpoints between stages
(`operator-checkpoint:` in the manifest), which is the mechanism for the ABABA
connector operations. Use it — it records the physical action in the run result.

## 6. Outputs

### 6.1 Raw capture (gitignored)

```
artifacts/dual_rx_gain_frequency/e_gsp2_SESSION/
├── gain_table_audit.json / gain_table_audit_final.json
├── stages/
│   ├── baseline_0/  pad_03_a/  baseline_1/  pad_03_b/  baseline_2/
│   ├── pad_06_a/ ... pad_20_b/ ...
│   └── splitter_swap_a/ splitter_swap_b/
├── <serial>/calibration.v7.zarr        per stage, per radio
└── pad_characterisation.json           VNA insertion loss vs frequency, per pad
```

`pad_characterisation.json` is a **required input**, not an afterthought — the
analysis cannot produce `n` without it.

**Acceptance gates:**

| Gate | Requirement |
|---|---|
| Completeness | all scheduled frames per stage, per radio |
| Restoration | every unpadded stage returns within unchanged-harness repeatability (the campaign's D→G bound, 0.90–0.96° MAE, is the reference) |
| Control | the control radio's ripple amplitude is unchanged across the whole sequence — if it moves, the difference-of-differences is compromised |
| Untouched arms | treated RX2 retains its baseline ripple amplitude |
| Gain tables | pre/post audits byte-identical |

### 6.2 Committed analysis

```
spf/calibrations/dual_rx_gain_frequency/reports/e_gsp2_pad_sweep_2026MMDD_v1/
├── REPORT.md
├── results.json
├── inputs_manifest.json
└── figures/
    ├── amplitude_vs_pad.png        A(L) per component, with the fitted mixture
    └── ripple_spectrum_by_stage.png
```

`results.json` must contain, **per delay component** (2.5475 ns and 1.0075 ns)
and per arm:

| Field | Meaning |
|---|---|
| `amplitude_deg` | fitted ripple amplitude at each pad state, with its immediately-adjacent unpadded reference |
| `insertion_loss_db` | measured, per pad, at the component's frequency |
| `A_in`, `A_out` | fitted mixture parameters, with confidence intervals |
| `implied_n` | `suppression_dB / L_dB` per pad, **reported per reference stage** so the stage-A vs D/G ambiguity is visible |
| `delay_per_stage_ns` | **per-stage** delay search, not the shared baseline (see blockers) |
| `control_amplitude_deg` | the control radio, to demonstrate it did not move |
| `restoration_residual_deg` | each unpadded stage vs the previous one |

### 6.3 Downstream updates

- `docs/learnings.md` — the ripple mechanism claim in L10.
- The gain-state model package README §3.3, which currently presents the
  reflection mechanism with the pad experiment as its main physical support.
- `docs/future_experiments.md` — mark E-GSP2, and update E-GSP1 (the VNA
  experiment) since a falsification here would make it pointless.

## 7. Decision rule

Pre-registered. The primary output is the **mixture fit**, not a pass/fail.

| Result | Conclusion |
|---|---|
| `A(L)` saturates, `A_out` small | Reflection model holds quantitatively. Ripple amplitude for a new harness becomes predictable from insertion loss — proceed to E-GSP1 to close it from first principles. |
| `A_out` comparable to `A_in` | A second reflecting interface sits outside the pad and must be localised. The splitter swap and E-CAL4's length insertion are the tools. |
| **`A(L)` flat — no pad dependence** | **The strongest available falsification of the reflection mechanism.** Retire the reflection language; the ripple term becomes an empirical basis with no physical claim. E-GSP1 should not be run. |
| Splitter swap moves the **delay**, not just amplitude | The splitter is part of the resonant path; the harness model needs another element. |

Note that **linearity in `L_dB` is not the prediction** — treating it as such is
the mistake the stage-A-only `n ≈ 1.40` reading invites.

## 8. Risks

| Risk | Why it matters | Check |
|---|---|---|
| **Unstable unpadded reference** | Already demonstrated: 5.34 / 10.80 / 10.40° across nominally identical states. This single fact makes a one-baseline design worthless. | ABABA with a reference immediately adjacent to every pad state. |
| **Connector operations dominate the effect** | Stage C failed for exactly this reason. | Restoration checks between every treatment; abort if a restoration fails. |
| **Nominal vs actual pad loss** | `n` is a ratio against `L_dB`. | VNA characterisation of every pad across the band, committed as an input. |
| **Pad insertion delay confounds amplitude** | See blockers — a displaced sinusoid read at the old delay loses amplitude. | Per-stage delay search, relaxed component separation. |
| **Control radio drifts** | The difference-of-differences assumes it does not. | Report control amplitude per stage; treat movement as a failed run. |

## Blockers

**One free computational prerequisite, and it gates the interpretation of every
number here.** Stage B's pad adds its own insertion delay — the campaign measured
+349 / +314 / −213 ps of equal-gain delay change across the three bands — but
every stage's ripple amplitude was read at the **shared** baseline delays
`[2.5475, 1.0075]` ns. A displaced sinusoid read at the old delay loses amplitude
purely geometrically, so part of the existing 14.64 dB may be **delay
displacement rather than attenuation**.

The campaign performed exactly this check for the 30 cm jumper — energy at the
predicted shifted delay of 5.469 ns was 5.82° in C against 0.13° in A — and never
for the pad.

**Before this experiment is designed in detail:** re-fit stage B with a
per-stage delay search (the fitter already sweeps 0.3–9.5 ns at 2.5 ps
resolution) and relax `minimum_component_separation_ns` below its 0.4 default.
Until that is done, `81.5%`, `14.64 dB` and every `n` above are uninterpretable.
It costs no bench time.

**Parts required:** four characterised pads and a second splitter, roughly $100.
