# E-GSC6 — does the equal-gain anchor move with gain index?

**Status:** designed 2026-08-10, not yet run.
**Revised 2026-08-10** after a bench/code audit against the committed capture path.
The revision corrects the harness schematic, the gain-table indices, the operating
gain range, the clipping analysis and the decision rule; see §9 for the changelog.
**Blocks:** the acceptance threshold for the tandem-AGC firmware phase campaign.
**Cost:** one capture session on the existing dual-RX bench — **≈2 h wall clock for
both radios** (§5.4). No new parts, no code change: a configuration change only.

---

## 1. Purpose

A planned PlutoSDR firmware feature ("tandem AGC") gives the FPGA ownership of the
AD9361 `CTRL_IN` pins so it can step **both** receive gain indices together, keeping
RX1 and RX2 at one common index at all times. The premise is that holding the two
arms at equal gain removes the dominant gain-state-dependent differential phase term.

Every dual-RX gain-phase result we have is built on the `additive_cross` schedule
(`spf/calibrations/dual_rx_gain_frequency/config.py:181-184`):

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
This is verifiable in the analysis code: at `g1 == g2` the shared-symmetric prediction
in `additive_cross.py:277-281` collapses to the bare intercept.

This is not a wild extrapolation. The nearest measured approach to the diagonal is the
seven `|Δg| = 1` pairs in the wide survey's 48-pair held-out set — `(5,6)`, `(6,5)`,
`(18,19)`, `(21,22)`, `(32,33)`, `(46,47)`, `(51,52)`, two of which straddle low-band
LNA transitions. But one index step is still one step, and the number that comes out of
this experiment is what the firmware campaign will be graded against.

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
gain table changes the **LNA** word than at the frozen-word control indices where the
`(LNA, MIXER, TIA)` triple is constant across both neighbours. From L10 and E-GSC4:
adjacent 1 dB LNA steps median **7.983°** (against a 0.180° same-dataset fitted-curve
LPF floor, wide survey), the mixer step **2.664°** (against a 0.343° LPF floor, A–G
campaign — a different session, **never pooled** with the LNA figure), and TIA/LPF-only
steps at the 0.355–0.368° measured frame-level noise floor. High-band mean \|A\| by LNA
index runs 2.91 / 1.92 / 0.68 / 2.13°.

**H0 (the falsifier).** `|D(g,g)|` materially exceeds `A` — a real interaction term.
Tandem still helps but by less than projected, and the residual model stays mandatory
rather than becoming a fallback.

Note that H1 and H0 are both *useful* outcomes. This experiment is not a go/no-go on
the firmware feature; the projected 6.0× / 5.3× / 3.4× per-band improvement is worth
having across a wide range of outcomes. What the experiment buys is a **measured**
acceptance threshold in place of an extrapolated one.

## 3. Approach

Add the diagonal cells to an otherwise ordinary additive-cross run as **held-out
pairs**. This exploits existing properties of the schedule rather than needing new
tooling:

- `gain_pairs = training_gain_pairs + held_out_gain_pairs` (`config.py:186-190`), so
  held-out cells are captured at every frequency and every epoch;
- `is_held_out_pair()` (`config.py:192`) marks them, so they are excluded from the fit;
- `analyze_additive_cross_dataset` (`additive_cross.py:173`) fits **only** the axis
  cross and scores every held-out cell against it.

The same session therefore yields the usual model *and* an unbiased diagonal
measurement, with the diagonal scored on a model that never saw it.

Validation in `config.py:113-123` permits `(g, g)` provided `g` is in `gains_db` and
`g` is **not** the schedule reference — any pair containing the reference is rejected
as overlapping the training cross, which is correct, since `(ref, ref)` is the anchor.

### 3.1 The existing analysis already emits both quantities under test

This is worth stating explicitly, because it is what makes "no code change" true and
it fixes which numbers the decision rule should use. The per-frequency fit anchors
`rx1_effect[ref] = rx2_effect[ref] = 0` and folds the anchor into `intercept`
(`additive_cross.py:256-268`). At a diagonal held-out cell `(g,g)`:

| Emitted field | Reduces to | Interpretation |
|---|---|---|
| `shared_residual` → `held_out_shared_gain_curve_metrics` | `phase − intercept` | **`D(g,g)`** — the tandem quantity itself |
| `independent_residual` → `held_out_independent_rx_metrics` | `phase − intercept − rx1_effect[g] − rx2_effect[g]` | **`C(g,g)`** — the interaction term, i.e. `D(g,g) − A(g)` |

The second row is the falsifier, measured directly. Because
`rx1_effect[g] + rx2_effect[g]` is the fitted **same-session** `A(g)` — `D(g,ref)` and
`D(ref,g)` are exactly the two cross arms L10 defines `A` from — the residual is
already differenced against this run's own harness rather than against a three-week-old
published number. (It is the least-squares-smoothed `A`; the model-free
`A = D(g,ref) + D(ref,g)` is also recoverable per cell from the same cross cells, and
both should be reported.)

Controls:

- **Interleaved equal-gain anchors**, as the existing campaigns do, so anchor drift
  and harness disturbance are visible in-run rather than inferred afterwards.
- **Frozen-word control indices.** Include common indices where the audited table
  holds `(LNA, MIXER, TIA)` constant across both neighbours. H2 predicts these are the
  quietest cells in their band; if they are not, the RF-word parameterisation is not
  driving the residual and the analysis needs rethinking. §5.2 names three that are
  frozen in **all three** bands, so the control is the same cell everywhere.
- **Both radios**, because the residual is unit-specific — see §8.

## 4. Hardware setup

### 4.1 Radios

Two Pluto+ units, the same pair used by the 2026-07-30 campaign so the result can be
compared against their published per-band `A`:

| Label | Serial | USB port |
|---|---|---|
| R17 | `104000bac4950008230026001b440a003a` | `1-1.1` |
| R18 | `1040007c4a94000211000b009186843ef2` | `1-1.2` |

Resolve by serial, never by IP. Both units expose a USB-gadget interface on a
duplicate `192.168.2.10/24`, and `iio_info -s` will mis-attribute one `192.168.2.1`
context to the wrong serial — resolving by IP silently swaps the radios.

**Both radios stay attached for the whole run.** `runner.py:282-286` drives them
**interleaved, one at a time, per frequency block**, alternating which radio leads on
each block. There is no point at which one radio is disconnected, so no connector work
is required between units — but it does mean each radio needs its **own** harness,
attached simultaneously (§4.2), and that wall-clock time adds across radios (§5.4).

Per-radio configuration is whatever the standard `dual_rx_gain_frequency` capture
path sets; this experiment changes only the gain-pair schedule and the LO list.

### 4.2 Physical schematic

The `dual_rx_gain_frequency` capture path generates its own tone: `config.py:158`
rejects any `tx_source` other than `fpga_dds`, and the runner drives TX and requires a
clean preflight tone before every block (`runner.py:135-156`). There is **no external
signal generator** — the tone is the radio's own FPGA DDS on TX2, looped back into its
own two receive ports. This is the chain the 2026-07-30 campaign recorded
(`wide_integer_gain_cross_band.yaml`: `TX2 -> 30 dB attenuator -> two-way splitter ->
RX1/RX2`), and it is what the published `A` was measured on.

One such chain **per radio**, both live at once:

```
        ┌───────── PLUTO R17 (104000bac495…) ─────────┐
        │                                             │
        │   TX2 ──► 30 dB attenuator                  │
        │                  │                          │
        │           ┌──────┴──────┐                   │
        │           │  2-way 50 Ω │  a real splitter, │
        │           │   splitter  │  NOT a tee        │
        │           └──┬───────┬──┘                   │
        │       coax A │       │ coax B               │
        │      (equal type, equal length)             │
        │           ┌──┴───┐ ┌─┴────┐                 │
        │           │ RX1  │ │ RX2  │                 │
        │           └──────┘ └──────┘                 │
        └─────────────────────────────────────────────┘

        ┌───────── PLUTO R18 (1040007c4a94…) ─────────┐
        │   identical, independent chain               │
        │   TX2 ──► 30 dB att ──► splitter ──► RX1/RX2 │
        └─────────────────────────────────────────────┘
```

Phase convention, unchanged from `docs/dual_rx_gain_phase_sweep.md` and
`spf.rf.get_phase_diff`:

```text
phase_difference = angle(RX1) - angle(RX2)
```

> **Do not** wire the bench as `docs/dual_rx_gain_phase_sweep.md` draws it. That
> document describes a *different* experiment (`spf.scripts.dual_rx_phase_sweep`),
> driven by an external sine generator, and instructs you to leave the Pluto TX
> disconnected and terminated. Built that way, no tone reaches either receiver and
> `require_preflight_tone` aborts the run at the first block.

### 4.3 Passive parts and adapters

Record every part, and its position in the chain, in the run metadata — **for both
chains separately**. The residual under test is a property of these exact assemblies,
and it is unit-specific (§8). Two attenuators, two splitters, four coax cables, plus
any adapters.

**Do not disturb any connector between the anchor baseline and the last diagonal
cell**, on either chain. If a connector is touched for any reason, that radio's run is
void from that point and must be re-baselined. This is not a precaution; it is the
observed behaviour (§8).

## 5. Software setup

### 5.1 Firmware — pin it, and verify it

The published `A` this experiment is graded against was captured on
`device-fw v0.38-plutoplus-spf-gain-series-v4-rc12-9-g867e1`, release tag
`v0.38-plutoplus-spf-gain-series-v4-rc16`, `boot-mode: qspi`. Reuse that pin, in the
`pluto-firmware` block, copied verbatim from `wide_integer_gain_cross_band.yaml`.

This experiment does **not** need the tandem-AGC firmware — it measures the premise
that feature rests on, using ordinary per-channel manual gain writes. It also should
not silently run on something newer:

- As of 2026-08-10 both radios are on a **volatile RC17 candidate load** reporting
  `device-fw v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe` (`git describe` ran
  before the rc17 tag existed). A power cycle reverts them to the persistent QSPI
  image.
- `/run/spf/direct_usb_ready.json` is **stale** (2026-08-07) and describes yet a third
  build, `…gain-rssi-fingerprint-v2-8-gf53d`. `runner.py:82-83` refuses to start when
  the manifest reports an unverified radio, but a stale manifest that says
  `firmware_verified: true` satisfies that gate while describing firmware that is not
  on the hardware, and `data_collector.py:145-155` would then record the *config's*
  firmware strings with `firmware_verified: false`.

**Preflight, in order:** power-cycle or reflash both radios to the pinned QSPI image,
regenerate the ready manifest, confirm `/opt/VERSIONS` `device-fw` on each unit equals
the pin, and only then start the capture. Whatever firmware is actually used, record
it — and if it is not the pinned build, say so in the report and treat the comparison
against published `A` as cross-firmware.

### 5.2 Configuration

Derive the common-gain set from the audited tables rather than assuming, because the
index→dB mapping is band-dependent. Read
`spf/calibrations/gain_state_phase_model_v1/gain_tables_audited.json` — three 77-row
AD9361 FULL RX tables, 231 rows, per-band byte hashes.

**LNA transitions, decoded from that file.** They are at different rows *and* different
dB values in each band, so no single triple of indices describes them:

| Band | Table rows | dB step (lower→upper) | LNA word | co-moving LPF |
|---|---|---|---|---|
| low ≤1300 | 34, 36, 55 | 30→31, 32→33, 51→52 | 0→1, 1→2, 2→3 | 11→0, 1→0, 18→14 |
| middle 1301–4000 | 35, 37, 55 | 29→30, 31→32, 49→50 | 0→1, 1→2, 2→3 | 11→1, 2→0, 17→14 |
| high >4000 | 37, 40, 55 | 22→23, 25→26, 40→41 | 0→1, 1→2, 2→3 | 13→0, 2→0, **14→14** |

These agree with `gain_state_computational_20260807_v1/REPORT.md` §5.3. The high-band
**40→41 dB** row is the only transition with `(MIXER, TIA, LPF)` all frozen — the
cleanest attribution in the corpus, and the highest-value diagonal cell here.

**Operating range.** The band-common integer range the campaigns actually use is
**−1 … 62 dB** (`wide_integer_gain_cross_band.yaml`, `survey_cross_band.yaml`). The
audited tables top out at 73 / 71 / 62 dB for low / middle / high, so anything above
62 dB is unreachable in the high band. This matters twice:

- `_validate_frame_gain` (`runner.py:93-108`) raises when the gain read back from the
  frame metadata differs from the requested value. A clamped out-of-range request is a
  **hard run failure**, not a silent substitution.
- Two of the three high-band LNA transitions sit at **22→23 and 25→26 dB**. A gain set
  floored anywhere above 22 dB misses them, which would leave H2 untestable in the
  band with the largest residual and the greatest need for the threshold.

**Frozen-word controls.** Three dB values hold `(LNA, MIXER, TIA)` constant across both
neighbours in **all three** bands: **8, 20 and 45 dB**. Using these makes the control
the same physical cell in every band.

**The reference gain is 26 dB**, unchanged from the campaign default so the anchor
matches the published `A`. Note that 26 dB is itself the upper side of the high-band
25→26 LNA transition, and `(26,26)` is the anchor, which the validator rejects. Straddle
that transition with `(25,25)` and `(27,27)` instead, and say so in the report.

The resulting set — 3 LNA brackets per band, 3 frozen-word controls, both range
extremes, 21 gains, 20 diagonal cells:

```python
REF = 26

GAINS_DB = (
    -1,                                  # bottom of the band-common range
    8, 20,                               # frozen-word controls (all bands)
    22, 23,                              # high-band LNA 0→1
    25, 26, 27,                          # high-band LNA 1→2, straddled around REF
    29, 30, 31, 32, 33,                  # middle 0→1, 1→2; low 0→1, 1→2
    40, 41,                              # high-band LNA 2→3 — the clean row
    45,                                  # frozen-word control (all bands)
    49, 50, 51, 52,                      # middle 2→3; low 2→3
    62,                                  # top of the band-common range
)

CalibrationConfig(
    schedule_design="additive_cross",
    schedule_reference_gain_db=REF,
    gains_db=GAINS_DB,
    held_out_gain_pairs=tuple((g, g) for g in GAINS_DB if g != REF),
    frequencies_hz=FREQUENCIES_HZ,       # §5.3
    repetitions=3,                       # campaign standard; min_quality_valid_per_cell=2
    tx_gain_policy="adaptive_max_rx_gain",
    tx_reference_rx_gain_db=REF,
    ...                                  # all other knobs at campaign defaults
)
```

Constraints the validator enforces, and which will reject the config if violated:
every gain in a held-out pair must appear in `gains_db`; pairs must be unique; and no
held-out pair may contain `schedule_reference_gain_db`.

That is 41 training pairs (`2N − 1`) plus 20 held-out diagonal cells = **61 pairs per
frequency per epoch**.

### 5.3 LO selection

The frequency dependence is a reflection standing wave with fitted delays of 2.54 ns
and 0.88–0.92 ns — ripple periods of roughly 392.5 MHz and 1.1 GHz. Sample enough LOs
per band to average over the shorter period rather than landing on an arbitrary phase
of it: at least 8 per band, spanning at least two full 392.5 MHz periods (≥785 MHz).
Cover all three bands, because the residual is not a smooth function of frequency
across the 4 GHz gain-table edge.

```python
FREQUENCIES_HZ = tuple(mhz * 1_000_000 for mhz in (
    # low ≤1300 — span 867 MHz = 2.2 ripple periods
    433, 550, 700, 850, 950, 1050, 1180, 1300,
    # middle 1301–4000 — span 2650 MHz
    1350, 1700, 2100, 2437, 2800, 3200, 3600, 4000,
    # high >4000 — span 1800 MHz
    4100, 4400, 4700, 5000, 5300, 5500, 5766, 5900,
))
```

2437 and 5766 MHz are retained from prior campaigns for continuity. Report the per-LO
spread alongside every band mean.

Run the existing capture entry point for `dual_rx_gain_frequency` with this config.
Randomised cell order, checkpointing, and the standard quality gates all apply
unchanged.

### 5.4 Cost and duration

Measured from the 2026-07-30 overnight wide survey's own per-frame records
(`observations.jsonl`, 27,825 frames per radio, identical `sample_rate_hz`,
`buffer_size`, `settle_seconds`, `repetitions` and RF-DC policy to this config):

| Quantity | Measured |
|---|---:|
| Mean per-cell measurement | 485 ms (R17) / 507 ms (R18) |
| Median per-cell | 444 ms / 468 ms |
| Per-radio measurement time, 27,825 cells | 3.75 h / 3.92 h |
| Wall clock, both radios interleaved | ≈8.85 h |
| Residual per-block overhead (radio open, RF-DC cal, preflight tone, LO settle) | ≈13.5 s per block per radio |

Applying that to this config — 61 pairs × 24 LOs × 3 epochs = **4,392 frames per
radio, 8,784 total**, and 24 × 3 × 2 = 144 block-open events:

| Term | Estimate |
|---|---:|
| Measurement, 8,784 cells at ~496 ms | 1.21 h |
| Block overhead, 144 events at ~13.5 s | 0.54 h |
| **Capture total, both radios** | **≈1.8 h** |
| With ~10% retry/requalification margin | **≈2 h** |
| Post-capture analysis + full validation (scaled from the survey's ~3 h for 55,650 frames) | ≈0.5 h |

So **one session of roughly two hours of capture** for both radios, against the wide
survey's ~8.9 h — the "one capture session" claim holds comfortably. Because the
runner interleaves rather than parallelises, this is a **sum** across the two radios;
capturing one radio only would be ≈1 h, not ≈2 h.

Sensitivity to LO count, if a denser ripple sample is wanted:

| LOs per band | Frames (both radios) | Capture wall clock |
|---:|---:|---:|
| 8 (24 total, as specified) | 8,784 | ≈1.8 h |
| 12 (36 total) | 13,176 | ≈2.6 h |
| 16 (48 total) | 17,568 | ≈3.5 h |

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
| firmware provenance | `device_fw`, release tag, image sha256, `boot_mode`, and the recorded `firmware_verified` flag, per radio | states whether the run matched the pinned build of §5.1; an unverified run is reported as cross-firmware, not silently compared |
| per-band table | mean / p95 / max \|D(g,g)\| for low, middle, high, from `held_out_shared_gain_curve_metrics` | all three bands populated, ≥8 LOs each, per-LO spread reported |
| interaction table | mean / p95 / max \|C(g,g)\| from `held_out_independent_rx_metrics` — the falsifier | per band, and per common index |
| per-index table | \|D(g,g)\| and \|C(g,g)\| by common index, LNA/MIXER/TIA/LPF words annotated from the audited table, **both row index and dB recorded** | every diagonal gain present, both radios |
| per-radio table | the same split by unit | both radios, or an explicit statement that only one was available |
| comparison | measured \|D(g,g)\| against **same-session** `A` (fitted and model-free), then against published `A`, then against the 6.65° anchored unequal-gain baseline | stated per band, not pooled; each claim names which `A` and which baseline it uses |
| frozen-word control result | \|D(g,g)\| at 8 / 20 / 45 dB versus the LNA-bracket cells, per band | H2 resolved explicitly |
| anchor drift trace | interleaved equal-gain anchor vs wall-clock, per radio | no unexplained step; see §8 |

### 6.3 Downstream updates the result requires

- The per-band measured numbers become the **acceptance threshold for the tandem-AGC
  hardware phase campaign**, replacing the currently extrapolated values.
- `docs/learnings.md` gains an entry recording whether separability held on the
  diagonal.
- `docs/future_experiments.md` is marked with the outcome.
- The tandem-AGC firmware plan's phase section is updated to cite measurement rather
  than projection.

## 7. Decision rule

Pre-registered. Judged **per band**, never pooled.

The primary statistic is `|C(g,g)|` — the held-out residual against the
**same-session** `A` (§3.1, `held_out_independent_rx_metrics`). This is preferred over
a comparison with published `A` because `A` is a property of the specific harness
assembly and is not stable across connector work (§8): the 2026-07-30 campaign saw one
radio's high-band mean \|A\| go from 3.49° to 29.41° and *not* recover. The published
`A` comparison is reported as a secondary, cross-session sanity check.

| Measured mean \|C(g,g)\| relative to same-session \|A\| | Reading | Consequence |
|---|---|---|
| within the measured per-step noise floor (0.355–0.368°) | separability holds; H1 | adopt the measured values as the campaign threshold; the residual model becomes a genuine fallback rather than a requirement |
| above the noise floor but below \|A\| | a small interaction term exists | adopt the measured values; keep the residual model in the pipeline for the affected band |
| comparable to or above \|A\|, or \|D(g,g)\| above the 6.65° anchored baseline in any band | H0 — the interaction term dominates | tandem's phase benefit is materially smaller than projected in that band; re-scope the firmware plan's phase claims and keep the residual model mandatory. The exact-gain-event half of the feature is unaffected and still worth building |

Independently, on H2: if the frozen-word control indices (8 / 20 / 45 dB) are **not**
among the quietest cells in their band, the `(LNA, MIXER, TIA)` parameterisation is not
what drives the residual, and both the index-clamp recommendation and the Campaign C
index selection need revisiting. Report the high-band 40→41 dB pair separately — it is
the only LNA transition without a co-moving LPF change, so it is the one cell where an
LNA attribution needs no confound argument.

## 8. Risks

| Risk | Why it matters | Check that catches it |
|---|---|---|
| **Harness disturbance mid-run** | The dominant observed effect. During the 2026-07-30 campaign, connector work drove one radio's high-band mean \|A\| from 3.49° to 29.41° — and it did **not** recover when the harness was restored, still 29.01° eight hours later, while the untouched control radio stayed flat at 3.83–4.23°. An 8× inflation that survives restoration will swamp everything here | interleaved equal-gain anchors; inspect the per-radio drift trace for a step before trusting any aggregate. Void that radio's run from the disturbance onward. Both chains stay connected for the whole session (§4.1), so no mid-run connector work is *required* |
| **Wrong bench built from a stale reference** | `docs/dual_rx_gain_phase_sweep.md` describes an external generator and a terminated TX, which is a different experiment. Wired that way no tone reaches the receivers | build the §4.2 self-loopback chain; `require_preflight_tone` fails closed on the first block if the tone is absent |
| **Out-of-range gain request** | The high band tops out at 62 dB, middle at 71, low at 73. A request above the band ceiling is clamped by the driver and then rejected by `_validate_frame_gain`, failing the run | keep the whole gain set inside the band-common **−1 … 62 dB** range (§5.2) |
| **dB / index confusion** | The config works in dB; the audited tables are indexed, and the index→dB offset is band-dependent (−1 / −3 / −10 at row 0). One full-table index step is 1 dB above the duplicated floor rows | derive the dB set from the audited table per band (§5.2), and record both row index and dB for every cell |
| **Missing the high-band LNA transitions** | Two of three sit at 22→23 and 25→26 dB. Flooring the gain set above 22 dB silently removes H2's evidence in the band with the largest residual | the §5.2 set includes 22, 23, 25, 27; verify all three high-band brackets are present before launching |
| **Firmware drift from the comparison baseline** | Published `A` was measured on `…rc12-9-g867e1`; the radios currently hold a volatile RC17 candidate and the ready manifest is stale (§5.1) | pin the firmware, regenerate the ready manifest, verify `/opt/VERSIONS` per unit, and gate the report on the recorded `firmware_verified` flag |
| **Only one radio available** | The residual is unit-specific in *every* band — cross-radio correlation of `A` is +0.50 / +0.59 / −0.23 low/middle/high, and above 4 GHz the mean inter-unit difference of 4.83° exceeds the residual itself. The often-quoted ρ≈0.99 describes `H`, not `A` | if only one unit is available, say so explicitly and scope the threshold to that unit. Do not generalise. Note this halves the session to ≈1 h (§5.4) |
| **LO set too sparse** | With ripple periods of ~392.5 MHz and ~1.1 GHz, a handful of LOs can land on an arbitrary ripple phase and produce a confidently wrong band mean | ≥8 LOs per band spanning ≥785 MHz; report the per-LO spread alongside the mean |
| **Reference gain in the diagonal set** | `(ref, ref)` is the anchor and any pair containing `ref` is rejected by the validator | intentional — the anchor is already measured. Exclude 26 dB from the diagonal list, and straddle the high-band 25→26 transition with `(25,25)`/`(27,27)` |
| **Comparing against the wrong baseline** | The 14.2–14.8° figure is raw uncorrected; 6.65° is anchored. Quoting the wrong one over- or under-states the win by 2× | compare against the anchored 6.65° MAE, and state which baseline every claim uses |

### 8.1 What is *not* a risk here: clipping on the diagonal

An earlier revision of this document listed simultaneous high gain on both arms as a
clipping risk, and prescribed setting the source level from the highest common gain,
with per-index level adjustment permitted. That analysis was wrong, and following it
would have degraded the run. Recorded here so it is not reintroduced:

- ADC headroom is a **per-arm** property, and the additive cross already drives every
  gain in `gains_db` on each arm individually — `(g, ref)` sweeps RX1 across the whole
  set and `(ref, g)` sweeps RX2. The diagonal introduces no per-arm level the cross has
  not already visited.
- `tx_gain_for` (`config.py:211-218`) keys the adaptive source level on
  `max(gain_rx1, gain_rx2)`. For any `g`, the diagonal cell `(g,g)` therefore receives
  **exactly** the TX level that cross cell `(g, ref)` receives.
- The diagonal is in fact the better-conditioned case: on the cross at large gain
  mismatch the weak arm is underdriven and loses SNR — the wide survey's own quality
  rejections cluster on its extreme mismatched held-out cell `(RX1=13, RX2=62)`. On the
  diagonal both arms sit at the target level.
- Per-index source-level adjustment would introduce a level covariate **perfectly
  confounded** with gain index, which is the one thing this experiment cannot tolerate.
  Do not do it. Leave `tx_gain_policy="adaptive_max_rx_gain"` at the campaign setting
  and let the standard quality gates run.

## 9. Revision log

**2026-08-10, revision 1** — bench and code audit against the committed capture path,
before any data exists. Changes that alter what an executor would do:

1. **§4.2 schematic replaced.** Was an external signal generator feeding a splitter,
   reusing `docs/dual_rx_gain_phase_sweep.md`. The `dual_rx_gain_frequency` path
   requires `tx_source="fpga_dds"`; the real chain is per-radio self-loopback
   `TX2 → 30 dB pad → splitter → RX1/RX2`. The old wiring would fail preflight.
2. **§5.2 LNA transitions corrected.** Was "indices 8, 20 and 30", which matched
   nothing in the audited tables; 8 and 20 are in fact *frozen-word* dB values. Real
   transitions are band-dependent: rows 34/36/55, 35/37/55, 37/40/55.
3. **§5.2 operating range corrected.** Was "roughly 27–73 dB". The band-common range
   is −1…62 dB; 73 dB is unreachable above 4 GHz and would hard-fail
   `_validate_frame_gain`, and a 27 dB floor would have excluded two of three
   high-band LNA transitions.
4. **§7 decision rule re-based** on the same-session `A` via
   `held_out_independent_rx_metrics`, which is exactly `C(g,g)`, rather than on
   published `A` — consistent with §8's own finding that `A` is harness-specific.
5. **§8.1 added**, retracting the clipping risk and the per-index level adjustment it
   licensed.
6. **§5.1 added**, pinning firmware and recording that the attached radios currently
   run a volatile RC17 candidate while the ready manifest is stale.
7. **§5.4 added**, with a duration estimate measured from the wide survey's own
   per-frame records rather than asserted.
8. **§2 provenance fixed.** The "0.805° vs 0.514°" pairing conflated the wide survey's
   held-out MAE pooled over all 48 off-axis pairs (|Δg| from 1 to 49) with an
   in-sample additive-fit residual from a different report. Replaced with the actual
   seven `|Δg| = 1` held-out pairs.
9. **§4.1/§4.3 corrected** on multi-radio operation: the runner interleaves both
   radios per frequency block, so both need simultaneously-attached independent
   harnesses, and wall clock sums across units.
