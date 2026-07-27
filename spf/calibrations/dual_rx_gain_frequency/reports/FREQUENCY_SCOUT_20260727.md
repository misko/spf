# Cross-band frequency scout — paused two-epoch analysis

Date: 2026-07-27

This is a deliberately provisional analysis of the paused cross-band scout.
The datasets remain resumable. They contain two complete randomized epochs at
every configured frequency and nine complete frequency blocks from the third
epoch.

## Captured evidence

- Artifact root:
  `artifacts/dual_rx_gain_frequency/frequency_scout_cross_band_20260727_v1`
- Radio A: `104000f6ad020002fdff3a00bba2f096a1`
- Radio B: `104000707f0700120f001a0095f2dbee49`
- V7 frames: 927/1,269 durable frames per radio
- Frequencies: 47 points from 433 MHz through 5.9 GHz
- Ordered gain grid at every frequency: `[-1, 26, 62]²`
- Complete randomized epochs: two
- Strict stored-IQ validation: all 927 completed frames readable on both radios
- Phase-supported cells: 368/423 on each radio
- Quality-valid frames: 804 Radio A; 803 Radio B

The unsupported cells are retained in V7. They fail closed because one channel
is too weak, its tone SNR/coherence is inadequate, or its segmented phase is
unstable. They are not silently included in fitting.

Generated-evidence SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Radio A validation | `46abadf81d7c05fb422fa6e2a3d5f056b9da6db98f51aab6dc61a4941fed20b3` |
| Radio A model | `b689778553059b307d4bee4e6ccdfac6bf617445fc9b94514a2ea1f295dc7d9c` |
| Radio B validation | `d5495f836940125840ca3d438edb3fed3477f04a7643c975f22313d819b71f06` |
| Radio B model | `ae023fb9648002e0ad84a05f22f737e277a93ec09eecdd12a3c47aa4c68dee29` |
| Cross-radio analysis | `91121b108689b31d7b9027aa8419776fa4d7be13a26b9a2ef15ea2820a4db617` |

## Parsimonious model comparison

All quoted prediction errors are paired leave-one-epoch-out errors on the same
eligible observations.

| Model | Radio A MAE | Radio B MAE | Interpretation |
|---|---:|---:|---|
| Per-frequency ordered additive | 1.04° | 0.94° | Best supported simple model |
| Per-frequency additive plus cell interaction | 1.03° | 0.95° | Within the predeclared 0.1° equivalence margin; reject extra interaction |
| Gain-difference-only | 1.22° | 1.35° | Worse; ordered RX1/RX2 effects matter |
| One gain curve shared globally | 3.90° | 3.83° | Too coarse |
| One gain curve per AD9361 gain-table band | 3.30° | 3.33° | Better than global sharing, but still materially worse |
| Constant-plus-linear-delay frequency baseline | 10.16° | 8.97° | Strongly rejected |

At a minimum 10 dB tone SNR in both channels, the selected additive model
improves to 0.87° MAE for Radio A and 0.83° for Radio B.

The selected measured-frequency model is:

```text
phase_r(f, g1, g2)
    = intercept_r(f)
    + RX1_effect_r(f, g1)
    + RX2_effect_r(f, g2)
    + residual
```

With three gains, this uses five identifiable parameters per measured
frequency: one intercept and two non-reference effects for each receiver.
Across 47 frequencies that is 235 nominal parameters. Sharing gain effects
only within the three hardware gain-table bands reduces this to 59 nominal
parameters, but adds 2.26° and 2.39° held-out MAE for Radios A and B
respectively. That reduction is not justified for phase correction.

The interaction comparison shows that a nine-cell lookup table is unnecessary
at this scout resolution. RX1 and RX2 contributions are additive to within the
measurement repeatability.

## Frequency and physical-length interpretation

A linear phase-versus-frequency slope is an effective differential delay. It
contains PCB, cables, analogue filters, LO retunes, and calibration state; it
is not automatically a physical trace measurement.

Individual global descriptive fits are:

| Radio | Effective delay | Free-space-equivalent path | Fit residual MAE |
|---|---:|---:|---:|
| Radio A | 27.80 ps | 8.33 mm | 10.67° |
| Radio B | 17.01 ps | 5.10 mm | 9.87° |

The direct Radio-A-minus-Radio-B fit is 10.80 ps, or 3.24 mm
free-space-equivalent, but its residual is 9.85° MAE and 90.49° maximum.
Band-local results are inconsistent with one physical length:

| Region | A-minus-B delay | Free-space equivalent | Residual MAE |
|---|---:|---:|---:|
| ≤1.3 GHz full table | -10.91 ps | -3.27 mm | 1.77° |
| 1.3–4.0 GHz full table | 3.27 ps | 0.98 mm | 4.42° |
| >4.0 GHz full table | 37.17 ps | 11.14 mm | 18.38° |
| 5.7–5.9 GHz only | 510.40 ps | 153.01 mm | 0.69° |

The locally precise but physically implausible 153 mm result demonstrates why
a short frequency interval can make radio-specific analogue behavior look like
trace length. A small real path mismatch may still contribute, but this
experiment does not isolate it.

## Operational recommendation

1. For a seen radio at a measured frequency, use the radio-specific
   per-frequency ordered additive model.
2. Apply corrections only to gain pairs whose cells pass the stored-IQ quality
   and repeatability gates.
3. Do not add an RX1×RX2 interaction term at this resolution.
4. Do not use one shared gain curve across frequencies or gain-table bands.
5. Do not interpolate the baseline to an unseen frequency from a simple delay
   or low-order polynomial. Measure a same-session equal-gain baseline or add
   direct calibration data at the target frequency.
6. Resume the third epoch before promoting these provisional models to final
   calibration artifacts.
7. To isolate PCB trace length, repeat with a common external source and swap
   the RX cables. A radio-internal term should remain with the radio; an
   external path term should follow or reverse with the cable swap.

## Reproduction

```bash
python -m spf.calibrations.dual_rx_gain_frequency validate \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/frequency_scout_cross_band.yaml \
  --dataset ARTIFACT_ROOT/SERIAL/calibration.v7.zarr \
  --serial SERIAL \
  --output ARTIFACT_ROOT/SERIAL/validation.json

python -m spf.calibrations.dual_rx_gain_frequency fit \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/frequency_scout_cross_band.yaml \
  --dataset ARTIFACT_ROOT/SERIAL/calibration.v7.zarr \
  --output ARTIFACT_ROOT/SERIAL/model.json

python -m spf.calibrations.dual_rx_gain_frequency report \
  --validation ARTIFACT_ROOT/SERIAL/validation.json \
  --model ARTIFACT_ROOT/SERIAL/model.json \
  --output-dir ARTIFACT_ROOT/SERIAL/analysis

python -m spf.calibrations.dual_rx_gain_frequency compare-radios \
  --model-a ARTIFACT_ROOT/104000f6ad020002fdff3a00bba2f096a1/model.json \
  --model-b ARTIFACT_ROOT/104000707f0700120f001a0095f2dbee49/model.json \
  --output-dir ARTIFACT_ROOT/cross_radio
```

Set `ARTIFACT_ROOT` to:

```text
artifacts/dual_rx_gain_frequency/frequency_scout_cross_band_20260727_v1
```
