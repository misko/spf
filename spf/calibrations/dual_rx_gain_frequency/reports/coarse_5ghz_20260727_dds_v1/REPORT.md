# Dual-RX gain/frequency phase model comparison

> Status: preliminary engineering evidence from the deliberately paused epoch-0 capture. These coefficients are not yet production calibration.

## Reproduce

From the SPF repository root, with the source V7 artifact present:

```bash
python -m spf.calibrations.dual_rx_gain_frequency compare-models \
  --config spf/calibrations/dual_rx_gain_frequency/configs/coarse_5ghz.yaml \
  --artifact-root artifacts/dual_rx_gain_frequency/coarse_5ghz_20260727_dds_v1 \
  --output-dir spf/calibrations/dual_rx_gain_frequency/reports/coarse_5ghz_20260727_dds_v1
```

The command reads no IQ into the optimiser and makes no radio calls. It hashes every V7 scalar array used by the analysis. Full-IQ verification is a separate mandatory `validate` step.

## Input checkpoint

| Pluto serial | Completed frames | Scalar-input SHA-256 | Complete blocks used |
|---|---:|---|---|
| `104000707f0700120f001a0095f2dbee49` | 10658 | `bfb6217518420aa16bdadf2190d2b1833c32d707f1667b74edb8efe00048d024` | epoch 0 @ 5804 MHz, epoch 0 @ 5866 MHz |
| `104000f6ad020002fdff3a00bba2f096a1` | 13024 | `496f4575f7f6292ca4742258a36b217ae1d11a8424acadb97b8dae564e274564` | epoch 0 @ 5804 MHz, epoch 0 @ 5866 MHz |

## Errors explained by competing models

Every table below uses five deterministic folds that hold out complete ordered gain-pair cells. A repeated cell is never split across train and test. “Reduction” is the held-out MAE reduction relative to one constant phase at the same radio and frequency.

### …0095f2dbee49 at 5804 MHz

| Model | Parameters | Held-out MAE | p95 | MAE reduction |
|---|---:|---:|---:|---:|
| Constant baseline | 1 | 10.67° | 24.78° | 0.0% |
| Linear gain difference | 2 | 10.71° | 23.44° | -0.4% |
| Separate linear RX1/RX2 | 3 | 10.01° | 23.62° | 6.2% |
| Shared signed stage curve | 8 | 3.80° | 8.25° | 64.4% |
| Ordered stage-boundary model | 15 | 1.43° | 3.79° | 86.6% |
| Shared signed categorical curve | 73 | 3.87° | 7.93° | 63.7% |
| Ordered categorical additive | 145 | 1.16° | 3.26° | 89.1% |

A harder test removes an entire low/high RX1-by-RX2 quadrant while leaving each gain state represented elsewhere:

| Model | Held-out quadrant MAE | p95 |
|---|---:|---:|
| Shared signed stage curve | 5.76° | 9.58° |
| Ordered stage-boundary model | 1.82° | 3.90° |
| Ordered categorical additive | 1.97° | 4.37° |

Residual linear drift was -0.03°/hour; the report does not interpret the small rank correlations as temperature because no temperature channel or repeated epoch is available.

### …0095f2dbee49 at 5866 MHz

| Model | Parameters | Held-out MAE | p95 | MAE reduction |
|---|---:|---:|---:|---:|
| Constant baseline | 1 | 10.49° | 27.00° | 0.0% |
| Linear gain difference | 2 | 10.42° | 23.75° | 0.6% |
| Separate linear RX1/RX2 | 3 | 10.03° | 24.08° | 4.3% |
| Shared signed stage curve | 8 | 2.70° | 6.13° | 74.2% |
| Ordered stage-boundary model | 15 | 1.45° | 3.92° | 86.1% |
| Shared signed categorical curve | 73 | 2.73° | 6.09° | 74.0% |
| Ordered categorical additive | 145 | 1.27° | 3.48° | 87.9% |

A harder test removes an entire low/high RX1-by-RX2 quadrant while leaving each gain state represented elsewhere:

| Model | Held-out quadrant MAE | p95 |
|---|---:|---:|
| Shared signed stage curve | 4.19° | 7.63° |
| Ordered stage-boundary model | 1.85° | 4.25° |
| Ordered categorical additive | 2.03° | 4.61° |

Residual linear drift was +0.02°/hour; the report does not interpret the small rank correlations as temperature because no temperature channel or repeated epoch is available.

### …00bba2f096a1 at 5804 MHz

| Model | Parameters | Held-out MAE | p95 | MAE reduction |
|---|---:|---:|---:|---:|
| Constant baseline | 1 | 9.00° | 23.06° | 0.0% |
| Linear gain difference | 2 | 9.27° | 22.16° | -3.0% |
| Separate linear RX1/RX2 | 3 | 9.19° | 22.03° | -2.1% |
| Shared signed stage curve | 8 | 2.02° | 4.86° | 77.5% |
| Ordered stage-boundary model | 15 | 1.58° | 4.29° | 82.5% |
| Shared signed categorical curve | 73 | 1.92° | 4.76° | 78.6% |
| Ordered categorical additive | 145 | 1.41° | 3.99° | 84.3% |

A harder test removes an entire low/high RX1-by-RX2 quadrant while leaving each gain state represented elsewhere:

| Model | Held-out quadrant MAE | p95 |
|---|---:|---:|
| Shared signed stage curve | 2.37° | 5.41° |
| Ordered stage-boundary model | 1.62° | 4.47° |
| Ordered categorical additive | 1.70° | 4.58° |

Residual linear drift was +0.33°/hour; the report does not interpret the small rank correlations as temperature because no temperature channel or repeated epoch is available.

### …00bba2f096a1 at 5866 MHz

| Model | Parameters | Held-out MAE | p95 | MAE reduction |
|---|---:|---:|---:|---:|
| Constant baseline | 1 | 8.97° | 24.42° | 0.0% |
| Linear gain difference | 2 | 8.99° | 21.34° | -0.1% |
| Separate linear RX1/RX2 | 3 | 8.99° | 21.48° | -0.2% |
| Shared signed stage curve | 8 | 2.05° | 5.20° | 77.2% |
| Ordered stage-boundary model | 15 | 1.52° | 4.00° | 83.1% |
| Shared signed categorical curve | 73 | 1.87° | 4.90° | 79.1% |
| Ordered categorical additive | 145 | 1.49° | 4.01° | 83.4% |

A harder test removes an entire low/high RX1-by-RX2 quadrant while leaving each gain state represented elsewhere:

| Model | Held-out quadrant MAE | p95 |
|---|---:|---:|
| Shared signed stage curve | 2.08° | 5.25° |
| Ordered stage-boundary model | 1.63° | 4.00° |
| Ordered categorical additive | 1.78° | 4.46° |

Residual linear drift was -0.15°/hour; the report does not interpret the small rank correlations as temperature because no temperature channel or repeated epoch is available.

## Parsimonious interpretation

The 15-parameter ordered stage-boundary model is the smallest model that retains nearly all observed predictive accuracy:

```text
phase(f,g1,g2) = intercept(f)
                 + linear_RX1(f)*g1 + stage_steps_RX1(f,g1)
                 + linear_RX2(f)*g2 + stage_steps_RX2(f,g2)
                 + residual
```

Its six boundaries (`-6, 6, 16, 23, 26, 41 dB`) are derived from starts of LNA/mixer-byte plateaus lasting at least three requested gain states in `drivers/iio/adc/ad9361.c` at Linux commit `d798b0d821b85ebd51ecffbfa68d8e4d69b77132`. The final 52–62 dB one-index-per-dB mixer ramp is represented by the linear term rather than eleven one-point dummies. The shared signed models are materially worse, so RX1 and RX2 require separate effects. The 145-parameter categorical model remains a useful exact-grid reference but gains little over the compact model.

Important: this compact basis was developed during epoch-0 exploration. The held-out-cell results test missing combinations inside that epoch, but the untouched repeat epochs must provide confirmatory model-selection evidence.

## Preliminary calibration artifacts

One machine-readable JSON calibration is emitted per serial. Each contains both the compact stage model and exact categorical effects, plus a fail-closed list of production-supported ordered pairs.

| Pluto serial | Frequencies fitted | Production-supported pairs | Status |
|---|---|---:|---|
| `104000707f0700120f001a0095f2dbee49` | 5804 MHz, 5866 MHz | 0 | `preliminary_single_epoch` |
| `104000f6ad020002fdff3a00bba2f096a1` | 5804 MHz, 5866 MHz | 0 | `preliminary_single_epoch` |

### Compact coefficients by radio and frequency

Slopes and steps are shown in degrees. The exact-radian values and the 73-state categorical curves are in the linked JSON artifacts.

| Radio / calibration | Frequency | Intercept | RX1 / RX2 slope | RX1 stage steps (boundary:value) | RX2 stage steps |
|---|---:|---:|---:|---|---|
| […0095f2dbee49](calibrations/104000707f0700120f001a0095f2dbee49.json) | 5804 MHz | -37.44° | +0.113 / -0.105°/dB | -6:+6.57°, 6:+6.07°, 16:+3.24°, 23:+1.10°, 26:-13.78°, 41:-18.08° | -6:-5.89°, 6:-8.02°, 16:-4.85°, 23:-10.31°, 26:+13.74°, 41:+19.25° |
| […0095f2dbee49](calibrations/104000707f0700120f001a0095f2dbee49.json) | 5866 MHz | -49.53° | +0.102 / -0.121°/dB | -6:+6.18°, 6:+6.06°, 16:+3.29°, 23:-0.48°, 26:-13.63°, 41:-17.30° | -6:-5.77°, 6:-7.97°, 16:-4.28°, 23:-6.20°, 26:+14.76°, 41:+20.38° |
| […00bba2f096a1](calibrations/104000f6ad020002fdff3a00bba2f096a1.json) | 5804 MHz | -58.58° | +0.117 / -0.071°/dB | -6:+4.64°, 6:+3.19°, 16:+5.51°, 23:+3.34°, 26:-13.08°, 41:-18.28° | -6:-5.72°, 6:-5.89°, 16:-1.94°, 23:-10.53°, 26:+14.03°, 41:+19.72° |
| […00bba2f096a1](calibrations/104000f6ad020002fdff3a00bba2f096a1.json) | 5866 MHz | -85.75° | +0.071 / +0.011°/dB | -6:+4.50°, 6:+3.56°, 16:+4.28°, 23:+4.78°, 26:-12.75°, 41:-17.08° | -6:-5.97°, 6:-6.13°, 16:-2.16°, 23:-6.81°, 26:+14.28°, 41:+19.89° |

No current pair is marked production-supported because the configured gate requires at least two quality-valid repeats and at most 5° repeat circular standard deviation. The paused artifact has only one complete epoch at the fitted frequencies.

## Transfer to another frequency or radio

The following figures transfer a complete categorical gain shape. “Optimal” uses all target observations to align the intercept and is only a descriptive lower bound. Equal-gain anchors are operationally possible but do not replace validation.

| Transfer | Unanchored MAE | One-anchor MAE | Five-anchor MAE | Optimal-intercept MAE |
|---|---:|---:|---:|---:|
| `104000707f0700120f001a0095f2dbee49:5804000000->5866000000` | 10.43° | 2.85° | 2.70° | 2.65° |
| `104000707f0700120f001a0095f2dbee49:5866000000->5804000000` | 10.41° | 3.16° | 2.67° | 2.66° |
| `104000f6ad020002fdff3a00bba2f096a1:5804000000->5866000000` | 24.74° | 3.20° | 3.01° | 2.99° |
| `104000f6ad020002fdff3a00bba2f096a1:5866000000->5804000000` | 24.72° | 3.45° | 3.07° | 3.04° |
| `104000707f0700120f001a0095f2dbee49->104000f6ad020002fdff3a00bba2f096a1:5804000000` | 17.87° | 4.05° | 3.29° | 3.31° |
| `104000707f0700120f001a0095f2dbee49->104000f6ad020002fdff3a00bba2f096a1:5866000000` | 32.18° | 3.93° | 3.95° | 3.92° |
| `104000f6ad020002fdff3a00bba2f096a1->104000707f0700120f001a0095f2dbee49:5804000000` | 17.74° | 3.23° | 3.18° | 3.21° |
| `104000f6ad020002fdff3a00bba2f096a1->104000707f0700120f001a0095f2dbee49:5866000000` | 32.17° | 7.56° | 3.86° | 3.89° |

## Recommendations

### Previously calibrated (“seen”) radio

After all three epochs and full-IQ validation pass, use that serial’s exact-frequency, ordered RX1/RX2 calibration. Apply `wrap(measured_RX1_minus_RX2 - predicted_phase)`. Fail closed for a weak/clipped live frame, invalid gain metadata, an unvalidated gain pair, a different frequency, or a gain-change event. Keep the compact stage model as the preferred explanation and smoothing diagnostic; the exact-grid additive model is the conservative operational table.

A reboot or materially different temperature is not yet a “seen” condition. Begin a session with several equal-gain anchors spanning the stage boundaries and reject the stored table if their residuals are inconsistent. The planned repeated epochs and reboot/temperature checks must quantify the threshold.

### New (“unseen”) radio with anchor measurements

Do not silently label a transferred model as calibrated. The current two-radio transfer shows that a source-radio gain shape plus anchors is useful but leaves several degrees of error. Five distributed equal-gain anchors are preferable to a single 26/26 dB anchor because they average frame noise and expose gain-state-specific disagreement. Use the transferred result only with an explicit lower-confidence flag and collect that serial’s full calibration when precision matters.

### New radio without any anchor

Do not apply another serial’s absolute phase correction. Radio-to-radio baseline shifts are large enough to dominate the residual. At most, use the population/stage shape as a prior for experiment design; report the phase as uncalibrated until at least an intercept anchor is measured.

### New RF frequency

The nearby-frequency transfer is better than a constant phase but still roughly two to three times worse than a same-frequency model. Measure the requested frequency. A linear effective-delay baseline may reduce the number of frequency anchors only after the full four-frequency, three-epoch dataset validates it.

## Required next evidence

1. Diagnose the RX2 high-gain DC/rail condition with matched TX2-on and TX2-off captures before spending the remaining exhaustive epochs.
2. Resume this checkpoint only if preparation semantics remain unchanged; otherwise start a clean V7 artifact.
3. Complete three epochs at all four frequencies for both serials.
4. Recompute every stored phase and quality decision from full IQ.
5. Run leave-one-epoch-out, reboot, and temperature/anchor validation.
6. Promote only exact serial/frequency/gain cells that pass the support policy into a production correction artifact.
