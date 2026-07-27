# Dual-RX gain/frequency calibration report

- Pluto serial: `104000f6ad020002fdff3a00bba2f096a1`
- Frames: 1269/1269 complete; 1105 phase-valid (87.1%)
- Cells: 369/423 pass the three-epoch criterion
- Validation status: `fail_quality`

## Frequency coverage

| Frequency (MHz) | Passing cells | Coverage | Median repeat std | Model observations | Train RMSE | CV MAE / p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 433.000 | 9/9 | 100.0% | 0.17° | 27 | 0.55° | 0.46° / 1.36° |
| 600.000 | 9/9 | 100.0% | 0.97° | 27 | 1.38° | 1.34° / 3.77° |
| 700.000 | 9/9 | 100.0% | 1.19° | 27 | 1.47° | 1.60° / 3.72° |
| 800.000 | 9/9 | 100.0% | 0.90° | 27 | 1.06° | 1.27° / 2.99° |
| 868.000 | 9/9 | 100.0% | 1.54° | 27 | 1.56° | 1.56° / 3.54° |
| 915.000 | 9/9 | 100.0% | 0.76° | 27 | 1.54° | 1.56° / 3.64° |
| 1000.000 | 9/9 | 100.0% | 0.02° | 27 | 0.53° | 0.47° / 1.46° |
| 1100.000 | 9/9 | 100.0% | 1.24° | 27 | 1.40° | 1.52° / 3.48° |
| 1200.000 | 9/9 | 100.0% | 0.31° | 27 | 0.53° | 0.57° / 1.26° |
| 1250.000 | 9/9 | 100.0% | 0.38° | 27 | 1.18° | 1.02° / 3.59° |
| 1280.000 | 9/9 | 100.0% | 0.93° | 27 | 1.51° | 1.53° / 4.22° |
| 1290.000 | 9/9 | 100.0% | 0.17° | 27 | 0.39° | 0.33° / 1.04° |
| 1299.000 | 9/9 | 100.0% | 0.71° | 27 | 0.75° | 0.96° / 1.79° |
| 1300.000 | 9/9 | 100.0% | 0.23° | 27 | 1.53° | 1.22° / 3.57° |
| 1301.000 | 9/9 | 100.0% | 1.05° | 27 | 1.48° | 1.66° / 3.03° |
| 1310.000 | 9/9 | 100.0% | 1.26° | 27 | 1.67° | 1.85° / 4.70° |
| 1320.000 | 9/9 | 100.0% | 0.06° | 27 | 0.38° | 0.32° / 1.00° |
| 1500.000 | 9/9 | 100.0% | 0.47° | 27 | 0.76° | 0.81° / 1.86° |
| 1800.000 | 9/9 | 100.0% | 0.14° | 27 | 0.78° | 0.54° / 0.98° |
| 2100.000 | 9/9 | 100.0% | 0.28° | 27 | 0.69° | 0.58° / 2.20° |
| 2300.000 | 7/9 | 77.8% | 0.17° | 21 | 0.43° | 0.45° / 1.24° |
| 2400.000 | 7/9 | 77.8% | 0.07° | 21 | 0.37° | 0.29° / 0.67° |
| 2412.000 | 7/9 | 77.8% | 0.15° | 21 | 0.25° | 0.28° / 0.54° |
| 2437.000 | 7/9 | 77.8% | 0.25° | 21 | 0.75° | 0.63° / 1.57° |
| 2467.000 | 7/9 | 77.8% | 0.20° | 21 | 0.27° | 0.33° / 0.65° |
| 2700.000 | 7/9 | 77.8% | 0.16° | 21 | 0.24° | 0.25° / 0.57° |
| 3000.000 | 7/9 | 77.8% | 0.28° | 21 | 0.36° | 0.34° / 0.73° |
| 3300.000 | 7/9 | 77.8% | 0.26° | 21 | 0.44° | 0.44° / 1.02° |
| 3600.000 | 7/9 | 77.8% | 1.21° | 21 | 1.59° | 1.51° / 2.86° |
| 3900.000 | 7/9 | 77.8% | 0.52° | 21 | 0.63° | 0.72° / 1.69° |
| 3990.000 | 7/9 | 77.8% | 0.60° | 21 | 0.79° | 0.87° / 2.08° |
| 3999.000 | 7/9 | 77.8% | 0.32° | 21 | 0.78° | 0.69° / 1.27° |
| 4000.000 | 7/9 | 77.8% | 0.30° | 21 | 0.68° | 0.60° / 2.55° |
| 4001.000 | 7/9 | 77.8% | 0.37° | 21 | 0.76° | 0.75° / 2.30° |
| 4010.000 | 7/9 | 77.8% | 1.05° | 21 | 1.33° | 1.52° / 3.56° |
| 4200.000 | 7/9 | 77.8% | 0.08° | 21 | 0.69° | 0.60° / 2.30° |
| 4500.000 | 7/9 | 77.8% | 0.32° | 21 | 1.33° | 1.29° / 2.95° |
| 4800.000 | 7/9 | 77.8% | 0.50° | 20 | 0.95° | 1.02° / 2.60° |
| 5100.000 | 7/9 | 77.8% | 1.85° | 21 | 2.48° | 2.80° / 4.67° |
| 5400.000 | 7/9 | 77.8% | 0.25° | 21 | 1.51° | 1.37° / 3.80° |
| 5600.000 | 7/9 | 77.8% | 1.01° | 21 | 1.48° | 1.52° / 4.61° |
| 5700.000 | 7/9 | 77.8% | 0.90° | 21 | 1.56° | 1.54° / 4.43° |
| 5766.000 | 7/9 | 77.8% | 0.73° | 21 | 1.17° | 1.11° / 3.67° |
| 5804.000 | 7/9 | 77.8% | 1.62° | 20 | 1.60° | 2.01° / 4.21° |
| 5838.000 | 7/9 | 77.8% | 0.33° | 21 | 0.49° | 0.56° / 1.18° |
| 5866.000 | 7/9 | 77.8% | 0.52° | 21 | 0.84° | 0.88° / 1.97° |
| 5900.000 | 7/9 | 77.8% | 0.54° | 21 | 0.64° | 0.70° / 1.82° |

## Gain-mismatch coverage

| Absolute RX gain mismatch | Passing cells | Cell coverage | Valid frames | Frame coverage |
|---:|---:|---:|---:|---:|
| 0–5 dB | 141/141 | 100.0% | 423/423 | 100.0% |
| 6–10 dB | 0/0 | n/a | 0/0 | n/a |
| 11–20 dB | 0/0 | n/a | 0/0 | n/a |
| 21–30 dB | 94/94 | 100.0% | 280/282 | 99.3% |
| 31–40 dB | 94/94 | 100.0% | 282/282 | 100.0% |
| 41–50 dB | 0/0 | n/a | 0/0 | n/a |
| 51–60 dB | 0/0 | n/a | 0/0 | n/a |
| 61–72 dB | 40/94 | 42.6% | 120/282 | 42.6% |

## Model diagnostics

- Leave-one-epoch-out circular MAE: 0.99°
- Leave-one-epoch-out circular RMSE: 1.47°
- Leave-one-epoch-out circular p95: 3.44°
- Leave-one-epoch-out maximum error: 8.14°

### Paired model comparisons

| Comparison | Held-out frames | First MAE / p95 | Second MAE / p95 | Recommended |
|---|---:|---:|---:|---|
| Ordered additive vs gain difference | 1105 | 0.99° / 3.44° | 1.18° / 3.72° | `additive` |
| Additive vs cell interaction | 1105 | 0.99° / 3.44° | 0.99° / 3.58° | `additive` |
| Frequency-specific vs shared curves | 1105 | 0.99° / 3.44° | 3.90° / 12.80° | `frequency_specific_gain_curves` |
| Frequency-specific vs gain-table-shared curves | 1105 | 0.99° / 3.44° | 3.27° / 10.65° | `frequency_specific_gain_curves` |
| Per-frequency vs constant-plus-delay baseline | 1105 | 0.99° / 3.44° | 10.04° / 33.95° | `frequency_specific_intercepts` |
| Unanchored vs one-frame anchor | 964 | 1.03° / 3.47° | 1.17° / 3.89° | `unanchored` |

Every row uses an identical held-out observation mask. Differences within the declared 0.1° MAE equivalence margin select the predeclared simpler operational model.

### Effective differential-delay description

- Common reference gain: 26 dB on RX1 and RX2
- Phase slope: -1.00° per 100 MHz
- Effective differential delay: 0.028 ns
- Free-space-equivalent signed path difference: 0.83 cm
- Linear-fit residual RMSE / maximum: 14.38° / 47.41°

Descriptive only: cables, PCB and analogue paths, LO retunes, and calibration state can all contribute. It is not asserted to be literal cable length.

### Signal-confidence tiers

| Minimum SNR in both channels | Eligible frames | Held-out frames | MAE | RMSE | p95 |
|---:|---:|---:|---:|---:|---:|
| -10 dB | 1105 | 1105 | 0.99° | 1.47° | 3.44° |
| 0 dB | 981 | 981 | 0.88° | 1.37° | 3.26° |
| 10 dB | 861 | 859 | 0.85° | 1.35° | 3.17° |

## Model fit plots

In the per-frequency diagnostics, solid lines are additive-model predictions and circular markers are passing three-epoch cell means. Error bars show repeat circular standard deviation. Failed or unsupported cells are not drawn. Phase is placed on the branch nearest the fitted frequency intercept so wrap-around does not create false jumps.

### Fitted gain effects across frequency

![Fitted RX1 and RX2 gain effects, overview page 1](fitted_gain_effects_01.png)

![Fitted RX1 and RX2 gain effects, overview page 2](fitted_gain_effects_02.png)

![Fitted RX1 and RX2 gain effects, overview page 3](fitted_gain_effects_03.png)

![Fitted RX1 and RX2 gain effects, overview page 4](fitted_gain_effects_04.png)

### Frequency baseline and delay description

![Per-frequency baseline and constant-plus-delay description](frequency_intercept_delay.png)

### Per-frequency data versus fit

#### 433.000 MHz

![Gain sweeps and observed-versus-fitted phase at 433.000 MHz](model_fit_433000000.png)

#### 600.000 MHz

![Gain sweeps and observed-versus-fitted phase at 600.000 MHz](model_fit_600000000.png)

#### 700.000 MHz

![Gain sweeps and observed-versus-fitted phase at 700.000 MHz](model_fit_700000000.png)

#### 800.000 MHz

![Gain sweeps and observed-versus-fitted phase at 800.000 MHz](model_fit_800000000.png)

#### 868.000 MHz

![Gain sweeps and observed-versus-fitted phase at 868.000 MHz](model_fit_868000000.png)

#### 915.000 MHz

![Gain sweeps and observed-versus-fitted phase at 915.000 MHz](model_fit_915000000.png)

#### 1000.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1000.000 MHz](model_fit_1000000000.png)

#### 1100.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1100.000 MHz](model_fit_1100000000.png)

#### 1200.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1200.000 MHz](model_fit_1200000000.png)

#### 1250.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1250.000 MHz](model_fit_1250000000.png)

#### 1280.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1280.000 MHz](model_fit_1280000000.png)

#### 1290.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1290.000 MHz](model_fit_1290000000.png)

#### 1299.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1299.000 MHz](model_fit_1299000000.png)

#### 1300.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1300.000 MHz](model_fit_1300000000.png)

#### 1301.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1301.000 MHz](model_fit_1301000000.png)

#### 1310.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1310.000 MHz](model_fit_1310000000.png)

#### 1320.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1320.000 MHz](model_fit_1320000000.png)

#### 1500.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1500.000 MHz](model_fit_1500000000.png)

#### 1800.000 MHz

![Gain sweeps and observed-versus-fitted phase at 1800.000 MHz](model_fit_1800000000.png)

#### 2100.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2100.000 MHz](model_fit_2100000000.png)

#### 2300.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2300.000 MHz](model_fit_2300000000.png)

#### 2400.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2400.000 MHz](model_fit_2400000000.png)

#### 2412.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2412.000 MHz](model_fit_2412000000.png)

#### 2437.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2437.000 MHz](model_fit_2437000000.png)

#### 2467.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2467.000 MHz](model_fit_2467000000.png)

#### 2700.000 MHz

![Gain sweeps and observed-versus-fitted phase at 2700.000 MHz](model_fit_2700000000.png)

#### 3000.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3000.000 MHz](model_fit_3000000000.png)

#### 3300.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3300.000 MHz](model_fit_3300000000.png)

#### 3600.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3600.000 MHz](model_fit_3600000000.png)

#### 3900.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3900.000 MHz](model_fit_3900000000.png)

#### 3990.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3990.000 MHz](model_fit_3990000000.png)

#### 3999.000 MHz

![Gain sweeps and observed-versus-fitted phase at 3999.000 MHz](model_fit_3999000000.png)

#### 4000.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4000.000 MHz](model_fit_4000000000.png)

#### 4001.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4001.000 MHz](model_fit_4001000000.png)

#### 4010.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4010.000 MHz](model_fit_4010000000.png)

#### 4200.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4200.000 MHz](model_fit_4200000000.png)

#### 4500.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4500.000 MHz](model_fit_4500000000.png)

#### 4800.000 MHz

![Gain sweeps and observed-versus-fitted phase at 4800.000 MHz](model_fit_4800000000.png)

#### 5100.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5100.000 MHz](model_fit_5100000000.png)

#### 5400.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5400.000 MHz](model_fit_5400000000.png)

#### 5600.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5600.000 MHz](model_fit_5600000000.png)

#### 5700.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5700.000 MHz](model_fit_5700000000.png)

#### 5766.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5766.000 MHz](model_fit_5766000000.png)

#### 5804.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5804.000 MHz](model_fit_5804000000.png)

#### 5838.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5838.000 MHz](model_fit_5838000000.png)

#### 5866.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5866.000 MHz](model_fit_5866000000.png)

#### 5900.000 MHz

![Gain sweeps and observed-versus-fitted phase at 5900.000 MHz](model_fit_5900000000.png)
