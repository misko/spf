# Provenance

Everything in this directory derives from one source analysis and one hardware
audit. This file records exactly what, how it was verified, and where the
reference implementation deliberately differs from the source pipeline.

## Sources

| Item | Location |
|---|---|
| Source analysis (model ladder, mechanism, holdouts) | [`../dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1/`](../dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1/) |
| Source campaign (A–G controlled spectroscopy, 2026-07-30) | [`../dual_rx_gain_frequency/reports/spectroscopy_campaign_20260730_v1/`](../dual_rx_gain_frequency/reports/spectroscopy_campaign_20260730_v1/) |
| Distilled conclusion | [`docs/learnings.md`](../../../docs/learnings.md) entry **L10** |
| Queued follow-ups | [`docs/future_experiments.md`](../../../docs/future_experiments.md) **E-CAL1**–**E-CAL4** |
| Raw V7/LMDB stores (read-only, outside Git) | `/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full{,_r2}` |
| Gain-table audit (source of `gain_tables_audited.json`) | `.../spectroscopy_20260730_full_r2/gain_table_audit.json` |

Repository state when the coefficients were fitted:
`a4f562d58d5b9769c2a9c4f3a03888c73f2336b8` (also recorded in each coefficient
file's `spf_git_sha`). This package lands on a later HEAD — three unrelated
rover/ardupilot commits followed — but `git diff a4f562d..HEAD` over the source
analysis directory is empty, so every fit input is byte-identical to the state
the coefficients record.

The raw stores were opened **read-only** (`zarr.LMDBStore(..., readonly=True, lock=False)`)
by the source analysis' `extract.py`, which writes only to an output directory
given on its command line. No campaign data was modified at any point.

## Committed gain tables

`gain_tables_audited.json` carries the three 77-row AD9361 FULL RX gain tables
(231 rows total) read off the hardware during the campaign's gain-table audit.

- Source audit JSON SHA-256:
  `214ba7a9caba7f26aea8851cdac1be88a07049ff347fc29b59cf7054deb94ef4`
  (also recorded inside the file as `source_sha256`).
- Per-band table-byte SHA-256, pinning the 231 bytes each band contributes:
  low `b3a66c34e11db2fd1aa25fa4910d56fb644b5a355d3fac39d027edb698d77b00`,
  middle `8644770dc52f4fbef2b6c6bc5e2677f3678ea00d5c7abb51495b705820faaa74`,
  high `90d34d61e8612277529dccfc3323f6c684c2bc36b7670dff078e009eb84a1143`.
- The two audited serials
  (`104000bac4950008230026001b440a003a`, `1040007c4a94000211000b009186843ef2`)
  were verified **byte-identical** across all three bands at export time; the
  export asserts this and fails rather than silently picking one radio.
- Digital gain (byte 2 bits 4:0) was verified **identically zero on all 231
  rows**, which is the premise for reading byte 2 bit 5 as `RF_DC_CAL`. This is
  re-checked by `selftest.py` and by `tests/test_gain_state_phase_model.py`.

Committing these tables is what makes the package self-contained: before this,
the `(band, requested dB) → (LNA, MIXER, TIA, LPF)` map existed only on the
QNAP share. They are chip data, not fitted data.

## Coefficient sets

All four were fitted with the **source analysis' own pipeline** (its `ladder.py`
/ `models.py`, unmodified, run from a scratch copy) and then re-serialised into
this package's format. Each records its own provenance block.

| Set | Rung | Stages | Rows | LOs | Columns | Rank |
|---|---|---|---:|---:|---:|---:|
| `l26_pooled_v1` | L26 | A, F, E_tx_0, rate_pilot | 4,641 | 119 | 38 | 29 |
| `l26_stage_a_v1` | L26 | A | 3,389 | 113 | 27 | 14 |
| `l30_pooled_v1` | L30 | A, F, E_tx_0, rate_pilot | 4,641 | 119 | 9 | 6 |
| `l31_pooled_v1` | L31 | A, F, E_tx_0, rate_pilot | 4,641 | 119 | 21 | 14 |

`l26_pooled_v1` is the default: it spans 27 distinct requested gains and 17 LPF
levels, against stage A's 3 gains and 7 LPF levels, so it refuses far fewer
cells in deployment. `l26_stage_a_v1` is the set that reproduces the published
stage-A numbers exactly.

Rank is strictly below the column count in every set. The signed-indicator
design is rank-deficient by construction; only signed differences are
identified. **Never read an individual coefficient as a physical quantity.**

## Verification performed

Every claim below was executed, not asserted.

### 1. The extraction reproduces the published dataset

| Quantity | Reproduced | Published |
|---|---:|---:|
| Stage A rows | 3,389 | 3,389 |
| Baseline `D` MAE | 6.6472° | 6.647° |
| Baseline `D` P95 | 18.382° | 18.38° |
| Baseline `D` max | 41.556° | 41.6° |
| Pooled rows / LOs | 4,641 / 119 | 4,641 / 119 |
| Pooled baseline MAE / P95 | 5.5559° / 17.861° | 5.56° / 17.86° |

### 2. Every published L26 holdout number reproduces

| Split | Reproduced MAE | Published | Reproduced P95 | Unequal-gain |
|---|---:|---:|---:|---:|
| LOEO leave-one-epoch-out | 2.0779 | 2.08 | 7.039 | 2.5975 (pub. 2.60) |
| LOFO leave-one-frequency-out | 2.2620 | 2.26 | 7.541 | 2.8277 (pub. 2.83) |
| LOBLK leave-frequency-block-out | 2.4730 | 2.47 | 7.622 | 3.0915 |
| LORO leave-one-radio-out | 2.2190 | 2.22 | 7.307 | 2.7740 |
| LOBAND leave-one-band-out | 6.6472 (coverage 0.000) | fails closed | 18.382 | 8.3096 |

Pooled leave-one-frequency-out: L26 **2.1087** (pub. 2.11), L30 **2.9853**
(pub. 2.99), L31 **2.2606** (pub. 2.26).

### 3. The reference implementation agrees with the source pipeline to machine precision

`model.py` is an independent implementation — its own design construction, its
own basis evaluation, its own support logic. Comparing its prediction against
the source pipeline's `X @ theta` on every row of the fitted set:

| Set | max abs deviation | Coverage |
|---|---:|---:|
| `l26_pooled_v1` | 1.110e-16 rad (6.4e-15°) | 1.0000 |
| `l26_stage_a_v1` | 1.110e-16 rad (6.4e-15°) | 1.0000 |
| `l30_pooled_v1` | 2.776e-17 rad (1.6e-15°) | 1.0000 |
| `l31_pooled_v1` | 1.110e-16 rad (6.4e-15°) | 1.0000 |

### 4. The independent refit path reproduces the pipeline bit-for-bit

`fit_from_extracted.py` implements its own anchor attachment, design matrix,
tau grid search and holdout loop. Run on stage A with epoch holdout it returns
`2.077875486167299°` against the source pipeline's `2.0778754861672994°`, with
27 columns and rank 14 — identical to the last representable digit.

### 5. The rule-5 numbers reproduce, once the mask is stated exactly

The report's rule-5 figures are computed on **frozen-RF ∧ unequal-gain** cells
using **held-out (LOFO)** predictions. Under that exact mask:

| Quantity | Reproduced | Published |
|---|---:|---:|
| Cells | 672 | 672 |
| Mean injected \|D\| | 1.362° | 1.36° |
| Max injected \|D\| | 4.716° | 4.72° |
| Fraction made worse | 81.4% | 81% |

Two masks that do **not** reproduce them, recorded so nobody re-derives the
confusion: including equal-gain cells gives 1,418 cells / 0.645° / 38.6%, and
using full-fit rather than held-out predictions gives 1.184° / 79.3%.

L30 and L31 inject exactly `0.0000°` on all 1,418 frozen cells and make 0% of
them worse — the report's "neutral by construction" claim, confirmed.

### 6. Structural invariants

`selftest.py` — 20 checks, all passing — and
`tests/test_gain_state_phase_model.py` — 43 tests, all passing — cover
antisymmetry, exact zero at the equal-gain cell, gauge invariance, fail-closed
behaviour on both out-of-table and unmeasured-state requests, the rule-5 guard,
and fit/save/load round trips.

## Where this package departs from the source report's text

Three claims in the source report were checked here and did not survive
unchanged. None of them changes a single published performance number; all three
are recorded because a reader who trusts the report's wording would be misled.

### 1. `h_tia` is not "separately identified" on stage A

`REPORT.md` §8 states `h_tia` "is separately identified but fits to
−0.20 ± 0.42°". Decoding the audited tables shows that on stage A's gain set
`{5, 26, 45}`, **TIA 0 occurs only at 5 dB, which is also the only MIXER 1
cell**, in all three bands — so the TIA and MIXER-1 families are *perfectly
collinear* and cannot be separately estimated. The fitted coefficients show
exactly the 50/50 ridge split this implies:

```text
l26_stage_a_v1 :  h_tia = {0: −0.9388, 1: +0.9388}   h_mixer[1] = −0.9388
l26_pooled_v1  :  h_tia = {0: −0.6469, 1: +0.6469}
```

The identified TIA *difference* is 1.88° (stage A) and 1.29° (pooled); neither
matches −0.20 ± 0.42°, and that quoted value could not be reconciled with any
committed artefact. Nothing downstream depends on it — TIA moves every holdout
by ≤0.01°, and the 1 dB TIA step is measured at the noise floor — but `h_tia`
should be described as **unidentified**, not as a small measured quantity.

### 2. "LNA index 1 was never measured at any frequency" is campaign-scoped

True of the A–G campaign. **Not true of the repository.** The 2.4 GHz
integer-gain experiments swept every integer gain from −3 to 71 dB on both axes
at 2412/2467 MHz on these same two radios, which visits LNA index 1 at 30–31 dB
and brackets all three middle-band LNA boundaries at 1 dB.

### 3. "No adjacent-1 dB LNA transition was measured anywhere" is likewise scoped

Decoding the audited middle table against the steps those reports published:

| Step | Words that change | Published step |
|---|---|---:|
| 14→15 dB | MIXER 1→2, LPF | +4.1° to +4.2° |
| 24→25 dB | MIXER 2→4, LPF | +2.5° to +2.9° |
| 31→32 dB | **LNA 1→2**, LPF | −2.6° to −4.4° |
| 49→50 dB | **LNA 2→3**, LPF | −14.3° to −16.7° |

These are a different experiment on different dates and are not poolable with
the campaign's `H` statistics as-is, and each LNA step comes bundled with an LPF
move and an `RF_DC_CAL` edge. They are nonetheless direct adjacent-1 dB LNA
evidence, and they support rather than contradict the mechanism: the LNA steps
are 6–50× the 0.343° LPF-only floor.

## Deliberate difference from the source pipeline

`GainTables.row_for_gain` returns `-1` for a request **below** the band's table
minimum. The source analysis' `spflib.row_for_gain` clamps such a request to
row 0 (its docstring says it returns -1, but the implementation does not — the
`gain_db >= g` scan matches every row when the request is below the minimum).

This package refuses instead, because the driver would clamp and the realized
gain would then not be the gain the caller asked for — a correction keyed on
the requested value would be describing a cell the hardware never visited.

**This cannot change any published number.** No scheduled gain in the source
campaign falls below its band minimum (band minima are −1 low, −3 middle, −10
high; the pooled gain set is −1, 0, 5, 10–17, 26–40, 45). Verification 3 above
was re-run after the change and remained bit-identical.

## File hashes

| File | Bytes | SHA-256 |
|---|---:|---|
SHA-256 truncated to 32 hex chars for legibility; recompute in full with
`sha256sum`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `__init__.py` | 606 | `7aa039127ca55004e8003321d8c1e750…` |
| `coefficients/l26_pooled_v1.json` | 2,239 | `8768d7057e26c429a34fa06036c104ac…` |
| `coefficients/l26_stage_a_v1.json` | 1,723 | `3d6e30bf0bfb15424fbfac7d7e069bb4…` |
| `coefficients/l30_pooled_v1.json` | 1,249 | `834445a748b35c7674638a879a37cd88…` |
| `coefficients/l31_pooled_v1.json` | 1,754 | `7f0a90a7b187798bccddbc15128e83e7…` |
| `demo.py` | 7,934 | `3b89540559213c29269b9bc0c09319d2…` |
| `figures/fig1_data.png` | 237,269 | `469192cba1e6980401adc029735c011b…` |
| `figures/fig2_mechanism.png` | 85,429 | `b887efc1fc1b37ad88827702dca316cf…` |
| `figures/fig3_ladder.png` | 115,471 | `68a1cf2202eefbe10fa803592acb5f79…` |
| `figures/fig4_fit.png` | 99,048 | `b5272f68c56c8be9901d5974a83333ad…` |
| `figures/fig5_error.png` | 94,124 | `a7cabb5754f5621162607f86718a599f…` |
| `figures/fig6_coverage.png` | 160,490 | `59b61ce8b44c80e340db83066f424694…` |
| `figures/fig7_calibration_cost.png` | 99,412 | `d909d4492249aea877baa05fd22607b8…` |
| `fit_from_extracted.py` | 8,601 | `ca5793f93320ead64ced8962fa963cf9…` |
| `gain_tables.py` | 7,278 | `fe20b4a7be3549c01399d46705cf2d7d…` |
| `gain_tables_audited.json` | 12,183 | `a86afda990d8643747eea03e31cb33fe…` |
| `make_figures.py` | 21,946 | `8fc990ed6349adad30c9745731de2e47…` |
| `model.py` | 17,724 | `91f609fc5d51fdfa9023a428c049de48…` |
| `selftest.py` | 9,695 | `1c515f65b60a6072bd1aef2924dcdec0…` |

Hashes cover the files as committed; `README.md` and this file are excluded
because they reference the hashes.

## Figures

The seven figures in `figures/` are regenerated end-to-end by `make_figures.py`
from the extracted campaign scalars, the committed coefficients, and the source
analysis' committed result JSONs. Nothing in them is schematic or hand-drawn.

Two panels plot **published** values rather than recomputing them, and say so in
the source: `fig2` panels (a) and (c) use the report's symmetry-decomposition and
fitted-ripple-amplitude tables, because both need machinery this script does not
reproduce — the symmetry split needs the paired additive-cross cells, and each
ripple amplitude is read at its own band's best-fit delay. An earlier draft used
a variance proxy instead; it overstated the ΔLNA = 0 cells by roughly 10× (1.1°
and 3.2° against the published 0.11–0.36°) and was discarded.

All seven were visually inspected for occlusion and label collisions, then
reviewed a second time by an agent given the images and no explanatory prose,
specifically to catch claims the figures do not support. That pass found a
truncated tick label, a legend drawn over data, two titles that overclaimed
against their own data, and one factually wrong caption (see below); all were
fixed and re-inspected.

**The wrong caption is worth recording.** `fig6` originally said the refused
regions were "the unmeasured LNA states". They are not. Per band, 11 gains are
refused for an unestimated MIXER level and 8-11 for an unestimated LPF level,
against only 2-3 for the LNA. The practical consequence is the opposite of what
the caption implied: E-CAL2's LNA fill would not by itself open up much of the
gain grid; widening the requested-gain set at the operating LOs would.

## Reproducing the coefficients

The source analysis needs `numpy<2`, `zarr<=2.18.4`, `numcodecs<0.16`, `lmdb`
(the repo's `~/virtual-envs/spf` satisfies all four).

```bash
# 1. read-only scalar extraction from the campaign stores (~3 min, 44 .npz, 2.4 MB)
cd spf/calibrations/dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1/analysis
python -u extract.py /path/to/scratch/extracted

# 2. refit with this package's independent implementation and score a holdout
cd /home/mouse9911/gits/spf
python -m spf.calibrations.gain_state_phase_model_v1.fit_from_extracted \
    --extracted /path/to/scratch/extracted \
    --stage spectroscopy_20260730_full/A \
    --stage spectroscopy_20260730_full_r2/F \
    --stage spectroscopy_20260730_full/E_tx_0 \
    --stage spectroscopy_20260730_full/rate_pilot \
    --holdout frequency \
    --out /tmp/l26_refit.json
```

Verifications 1–5 above additionally require the source analysis' own modules,
run from a scratch copy of `analysis/` with `extracted/` symlinked to the
extraction output. Do not run them in place: steps 3 and 4 of the source
analysis overwrite the committed result JSONs.
