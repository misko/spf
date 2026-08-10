# E-GSC6 — does the equal-gain anchor move with gain index?

**Status:** designed 2026-08-10, not yet run.
**Blocks:** the acceptance threshold for the tandem-AGC firmware phase campaign.
**Cost:** one capture session on the existing dual-RX bench. No new parts, no code
change — a configuration change only.

---

## 1. Purpose

A planned PlutoSDR firmware feature ("tandem AGC") gives the FPGA ownership of the
AD9361 `CTRL_IN` pins so it can step **both** receive gain indices together, keeping
RX1 and RX2 at one common index at all times. The premise is that holding the two
arms at equal gain removes the dominant gain-state-dependent differential phase term.

Every dual-RX gain-phase result we have is built on the `additive_cross` schedule:

```python
[(gain, reference) for gain in gains_db] +
[(reference, gain) for gain in gains_db if gain != reference]
```

On the (RX1, RX2) gain plane that traces a cross centred on the reference. It visits
exactly **one** equal-gain cell — `(reference, reference)`, which *is* the anchor that
`D` is defined against. So `D(ref, ref) ≡ 0` by construction, and the anchor's own
index dependence is structurally unobservable in every campaign run to date.

```
                    RX2 gain →
             g₁    g₂   [ref]   g₃    g₄
      g₁     ·     ·     ✕      ·     ·
      g₂     ·     ·     ✕      ·     ·
RX1  [ref]   ✕     ✕     ✕      ✕     ✕      ← the training cross
      g₃     ·     ·     ✕      ·     ·
      g₄     ·     ·     ✕      ·     ·

      ✕ measured        · never visited
      the diagonal (g₁,g₁) … (g₄,g₄) is entirely in the `·` region
```

**Tandem AGC operates only on that diagonal.** Its entire operating regime is the one
line through the gain plane that no campaign has sampled away from the anchor.

What is unknown is narrow and specific: writing `D(g1,g2) = P1(g1) − P2(g2) + C(g1,g2)`
with the interaction term `C` absorbed to zero on every measured cell, the tandem
quantity is `D(g,g) = A(g) + C(g,g)`. The claim "equal gain leaves only the arm-specific
residual `A`" is exactly the assumption `C ≡ 0`. The cross cannot test it, and the
committed coefficient sets cannot either — they are antisymmetric by construction, so
they return `D(g,g) ≡ 0` as an identity of the model form rather than as a result.

This is not a wild extrapolation. The nearest measured approach to the diagonal is
|Δg| = 1 in the wide survey's held-out set, where a separable model scores 0.805° MAE
against its own 0.514° in-sample floor. But one index step is still one step, and the
number that comes out of this experiment is what the firmware campaign will be graded
against.

## 2. Hypothesis

Pre-registered before any diagonal data exists.

**H1 (separability holds).** `|D(g,g)|` is bounded by the arm-specific residual `A`
already published in `gain_state_phase_model_20260802_v1/REPORT.md` §3.2:

| Band | predicted mean \|D(g,g)\| | p95 | max |
|---|---:|---:|---:|
| low ≤1300 MHz | 0.73° | 2.55° | 4.24° |
| middle 1301–4000 MHz | 1.24° | 4.22° | 6.23° |
| high >4000 MHz | 3.72° | 10.84° | 23.71° |

and is materially below the anchored unequal-gain baseline of **6.65° MAE**
(8.31° counting only cells a correction acts on).

**H2 (index dependence).** `|D(g,g)|` is larger at common indices where the audited
gain table changes the **LNA** word than where the `(LNA, MIXER, TIA)` triple is
frozen. From L10: adjacent 1 dB LNA steps median 7.983°, mixer 2.664°, TIA/LPF-only
at the 0.355–0.368° noise floor. High-band mean |A| by LNA index runs
2.91 / 1.92 / 0.68 / 2.13°.

**H0 (the falsifier).** `|D(g,g)|` materially exceeds `A` — a real interaction term.
Tandem still helps but by less than projected, and the residual model stays mandatory
rather than becoming a fallback.

Note that H1 and H0 are both *useful* outcomes. This experiment is not a go/no-go on
the firmware feature; the projected 6.0× / 5.3× / 3.4× per-band improvement is worth
having across a wide range of outcomes. What the experiment buys is a **measured**
acceptance threshold in place of an extrapolated one.

## 3. Approach

Add the diagonal cells to an otherwise ordinary additive-cross run as **held-out
pairs**. This exploits an existing property of the schedule rather than needing new
tooling:

- `gain_pairs = training_gain_pairs + held_out_gain_pairs`, so held-out cells are
  captured at every frequency and every epoch;
- `is_held_out_pair()` marks them, so they are excluded from the fit.

The same session therefore yields the usual model *and* an unbiased diagonal
measurement, with the diagonal scored on a model that never saw it.

Validation in `config.py` permits `(g, g)` provided `g` is in `gains_db` and `g` is
**not** the schedule reference — `(ref, ref)` is rejected as overlapping the training
cross, which is correct, since that cell is the anchor.

Controls:

- **Interleaved equal-gain anchors**, as the existing campaigns do, so anchor drift
  and harness disturbance are visible in-run rather than inferred afterwards.
- **A frozen-word control index.** Include at least one common index where the audited
  table holds `(LNA, MIXER, TIA)` constant across its neighbours. H2 predicts this is
  the quietest cell in its band; if it is not, the RF-word parameterisation is not
  driving the residual and the analysis needs rethinking.
- **Both radios**, because the residual is unit-specific — see §8.

## 4. Hardware setup

### 4.1 Radios

Two Pluto+ units, the same pair used by the 2026-07-30 campaign if available, so the
result can be compared against their published per-band `A`. Record each unit's serial
and resolve by serial, never by IP.

Per-radio configuration is whatever the standard `dual_rx_gain_frequency` capture
path sets; this experiment changes only the gain-pair schedule and the LO list.

### 4.2 Physical schematic

Reuse the bench in [`docs/dual_rx_gain_phase_sweep.md`](../../docs/dual_rx_gain_phase_sweep.md)
unchanged, including its `phase_difference = angle(RX1) − angle(RX2)` convention.

```
              signal generator
                     │
                     │   (set level for the HIGHEST common gain in the set —
                     │    see §8, clipping risk)
              ┌──────┴──────┐
              │  attenuator │   if required
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │  2-way 50 Ω │   a real splitter, NOT a tee
              │   splitter  │
              └──┬───────┬──┘
                 │       │
          coax A │       │ coax B      equal type, equal length
                 │       │
              ┌──┴───┐ ┌─┴────┐
              │ RX1  │ │ RX2  │        one Pluto+, both receive ports
              └──────┘ └──────┘
```

### 4.3 Passive parts and adapters

Record every part, and its position in the chain, in the run metadata. The residual
under test is a property of this exact assembly — see §8.

**Do not disturb any connector between the anchor baseline and the last diagonal
cell.** If a connector is touched for any reason, the run is void from that point and
must be re-baselined. This is not a precaution; it is the observed behaviour (§8).

## 5. Software setup

Firmware: whatever the standard capture path uses. This experiment does **not** need
the tandem-AGC firmware — it measures the premise that feature rests on, using
ordinary per-channel manual gain writes.

### 5.1 Configuration

Derive the common-gain set from the audited tables rather than assuming, because the
index→dB mapping is band-dependent:

1. Read `spf/calibrations/gain_state_phase_model_v1/gain_tables_audited.json` — three
   77-row AD9361 FULL RX tables, 231 rows, per-band byte hashes.
2. For the band under test, locate the rows where the **LNA** word changes. In the
   audited tables the LNA transitions sit at indices **8, 20 and 30**.
3. Convert those indices to the dB values the capture path uses. One full-table index
   step is exactly 1 dB.
4. Choose common gains that **bracket** each transition (one below, one above), plus
   at least one frozen-word control index, plus the extremes of the operating range.
   The range actually in use is roughly **27–73 dB**.

Then set them as held-out pairs:

```python
CalibrationConfig(
    schedule_design="additive_cross",
    schedule_reference_gain_db=REF,          # unchanged from the campaign default
    gains_db=(...),                          # must contain every diagonal gain
    held_out_gain_pairs=(
        (g, g) for g in DIAGONAL_GAINS       # every g != REF
    ),
    frequencies_hz=(...),                    # see below
    ...
)
```

Constraints the validator enforces, and which will reject the config if violated:
every gain in a held-out pair must appear in `gains_db`; pairs must be unique; and no
held-out pair may contain `schedule_reference_gain_db`.

**LO selection.** The frequency dependence is a reflection standing wave with fitted
delays of 2.54 ns and 0.88–0.92 ns — ripple periods of roughly 392.5 MHz and
1.1 GHz. Sample enough LOs per band to average over the shorter period rather than
landing on an arbitrary phase of it: at least 8 per band, spanning at least two full
392.5 MHz periods. Cover all three bands — low ≤1300, middle 1301–4000, high >4000 —
because the residual is not a smooth function of frequency across the 4 GHz gain-table
edge.

Run the existing capture entry point for `dual_rx_gain_frequency` with this config.
Randomised cell order, checkpointing, and the standard quality gates all apply
unchanged.

## 6. Outputs

### 6.1 Raw capture (gitignored)

`artifacts/dual_rx_gain_frequency/<run>/` per the directory's convention. Never
committed.

### 6.2 Committed analysis

`spf/calibrations/dual_rx_gain_frequency/reports/equal_gain_diagonal_<date>_v1/`,
append-only per that directory's convention, containing at minimum:

| Artifact | Content | Acceptance gate |
|---|---|---|
| `REPORT.md` | narrative, method, input hashes | reproduces its own numbers from committed JSON |
| per-band table | mean / p95 / max \|D(g,g)\| for low, middle, high | all three bands populated, ≥8 LOs each |
| per-index table | \|D(g,g)\| by common index, LNA/MIXER/TIA words annotated from the audited table | every diagonal gain present, both radios |
| per-radio table | the same split by unit | both radios, or an explicit statement that only one was available |
| comparison | measured \|D(g,g)\| against published `A`, and against the 6.65° anchored unequal-gain baseline | stated per band, not pooled |
| anchor drift trace | interleaved equal-gain anchor vs wall-clock | no unexplained step; see §8 |

### 6.3 Downstream updates the result requires

- The per-band measured numbers become the **acceptance threshold for the tandem-AGC
  hardware phase campaign**, replacing the currently extrapolated values.
- `docs/learnings.md` gains an entry recording whether separability held on the
  diagonal.
- The tandem-AGC firmware plan's phase section is updated to cite measurement rather
  than projection.

## 7. Decision rule

Pre-registered. Judged **per band**, never pooled.

| Measured mean \|D(g,g)\| vs published `A` | Reading | Consequence |
|---|---|---|
| within noise of `A` | separability holds; H1 | adopt the measured values as the campaign threshold; the residual model becomes a genuine fallback rather than a requirement |
| above `A` but below ~2× | a small interaction term exists | adopt the measured values; keep the residual model in the pipeline for the affected band |
| above ~2× `A`, or above the 6.65° anchored baseline in any band | H0 — the interaction term dominates | tandem's phase benefit is materially smaller than projected in that band; re-scope the firmware plan's phase claims and keep the residual model mandatory. The exact-gain-event half of the feature is unaffected and still worth building |

Independently, on H2: if the frozen-word control index is **not** among the quietest
cells in its band, the `(LNA, MIXER, TIA)` parameterisation is not what drives the
residual, and both the index-clamp recommendation and the Campaign C index selection
need revisiting.

## 8. Risks

| Risk | Why it matters | Check that catches it |
|---|---|---|
| **Harness disturbance mid-run** | The dominant observed effect. During the 2026-07-30 campaign, connector work drove one radio's high-band mean \|A\| from 3.49° to 29.41° — and it did **not** recover when the harness was restored, still 29.01° eight hours later, while the untouched control radio stayed flat at 3.83–4.23°. An 8× inflation that survives restoration will swamp everything here | interleaved equal-gain anchors; inspect the drift trace for a step before trusting any aggregate. Void the run from the disturbance onward |
| **Clipping at high common gain** | The diagonal puts *both* arms at high gain simultaneously, which the cross never does — the cross always holds one arm at the reference. A source level fine for the cross may overload at the top of the diagonal | set the level from the highest common gain in the set, and check the standard quality gates reject nothing above −3 dBFS. Per-index level adjustment is acceptable if recorded |
| **dB / index confusion** | The config works in dB; the audited tables are indexed. One full-table index is 1 dB, but the offset is band-dependent | derive the dB set from the audited table per band (§5.1), and record both index and dB for every cell |
| **Only one radio available** | The residual is unit-specific in *every* band — cross-radio correlation of `A` is +0.50 / +0.59 / −0.23 low/middle/high, and above 4 GHz the mean inter-unit difference of 4.83° exceeds the residual itself. The often-quoted ρ≈0.99 describes `H`, not `A` | if only one unit is available, say so explicitly and scope the threshold to that unit. Do not generalise |
| **LO set too sparse** | With ripple periods of ~392.5 MHz and ~1.1 GHz, a handful of LOs can land on an arbitrary ripple phase and produce a confidently wrong band mean | ≥8 LOs per band spanning ≥2 short-period cycles; report the per-LO spread alongside the mean |
| **Reference gain in the diagonal set** | `(ref, ref)` is the anchor and is rejected by the validator | intentional — the anchor is already measured. Exclude `ref` from the diagonal gain list |
| **Comparing against the wrong baseline** | The 14.2–14.8° figure is raw uncorrected; 6.65° is anchored. Quoting the wrong one over- or under-states the win by 2× | compare against the anchored 6.65° MAE, and state which baseline every claim uses |
