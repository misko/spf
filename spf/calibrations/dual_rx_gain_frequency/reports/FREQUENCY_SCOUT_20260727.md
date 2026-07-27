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
| Reproducible comparative analysis | `244335b0a18ebea1f3387490e63eac0b6b6af68d0d48d0800727ebf8362e86ca` |

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

## Transfer between radios

The reproducible comparison also transfers the complete three-gain
same-frequency shape from one radio to the other. It then obtains the target
intercept from a quality-valid 26/26 dB frame in each epoch and excludes those
anchors from scoring:

| Source → target | Frames | All-frequency MAE / RMSE / max | 5.7–5.9 GHz MAE / RMSE / max |
|---|---:|---:|---:|
| Radio A → Radio B | 700 | 2.05° / 3.10° / 15.46° | 4.62° / 6.61° / 15.46° |
| Radio B → Radio A | 701 | 2.02° / 3.11° / 15.97° | 4.56° / 6.58° / 15.97° |

Thus, the other radio's gain shape plus a same-session intercept is a useful
fallback prior, but it is about twice as inaccurate overall as the
radio-specific model and materially worse in the 5.8 GHz operating region.
It must not replace per-radio calibration. The whole-run and per-epoch anchor
policies have nearly identical aggregate errors here; an epoch anchor removes
only intercept drift and cannot repair radio-specific gain-shape differences.

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

## Dense-capture decision and immutable gates

The scout supports proceeding with
`configs/survey_cross_band.yaml`, rather than a sparse or one-dimensional gain
design. The dense schedule is:

| Quantity | Per radio | Both radios |
|---|---:|---:|
| Frequencies | 12 | 12 |
| Ordered gain pairs per frequency | 289 | 578 frames per epoch |
| Randomized epochs | 3 | 3 |
| V7 frames | 10,404 | 20,808 |

The 17 gains are the union of representative endpoints and gain-table stage
transitions. The complete Cartesian grid is retained because additivity has
only been established at the scout's three gains; the dense run must test it
through the intermediate stage transitions.

The dense run uses the scout-qualified 0 dB adaptive-TX reference. This is
not a promise that every 63 dB-asymmetric pair will contain a measurable tone.
The stronger receive channel determines safe TX attenuation, so the weaker
channel can legitimately fall below the phase-quality threshold.

Pass/fail conditions are fixed before starting the dense artifact:

1. **Scout completion:** both serials reach 1,269/1,269 durable frames; strict
   V7/full-IQ validation passes; every configured cell has three completed
   attempts. A weak frame may fail phase quality but may not be missing.
2. **Dense structural capture:** both serials reach 10,404/10,404 durable
   frames with the configured firmware hashes, protocol v2, serial, schedule,
   shape, and gain/RSSI provenance. Any missing, corrupt, or mismatched frame
   fails the run.
3. **Cell support:** a phase correction is emitted only when at least two of
   three frames pass the stored-IQ quality gates and their circular phase
   standard deviation is at most 5°. Unsupported cells remain explicit.
4. **Signal-conditioned model accuracy:** on frames where both channel tones
   have at least 10 dB SNR, leave-one-epoch-out MAE must be at most 2° and p95
   at most 5° for a radio-specific correction model to pass.
5. **Parsimony:** retain the ordered additive model unless an interaction model
   improves paired held-out MAE by more than the predeclared 0.1° practical
   margin. A simpler gain-difference, shared-frequency, or delay model must
   meet the same paired criterion before replacing it.
6. **Deployment support:** corrections are valid only for the exact radio
   serial, LO frequency, and ordered gain pair that passed the cell gate.
   Nearby committed centres such as 2.4671 GHz or 5.839 GHz require their own
   capture unless deployment is standardized on a calibrated anchor.

Collection remains paused at the durable scout checkpoint while this analysis
is reviewed. Resumption must use the unchanged scout configuration and output
directory; the dense survey must use a new output directory.

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

python -m spf.calibrations.dual_rx_gain_frequency compare-models \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/frequency_scout_cross_band.yaml \
  --artifact-root ARTIFACT_ROOT \
  --output-dir ARTIFACT_ROOT/comparative
```

Set `ARTIFACT_ROOT` to:

```text
artifacts/dual_rx_gain_frequency/frequency_scout_cross_band_20260727_v1
```
