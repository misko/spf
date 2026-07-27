# Dual-RX gain/frequency calibration report

- Pluto serial: `104000707f0700120f001a0095f2dbee49`
- Frames: 1269/1269 complete; 1105 phase-valid (87.1%)
- Cells: 368/423 pass the three-epoch criterion
- Validation status: `fail_quality`

## Frequency coverage

| Frequency (MHz) | Passing cells | Coverage | Median repeat std | Model observations | Train RMSE | CV MAE / p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 433.000 | 9/9 | 100.0% | 0.40° | 27 | 1.19° | 1.04° / 3.78° |
| 600.000 | 9/9 | 100.0% | 0.27° | 27 | 0.66° | 0.58° / 2.21° |
| 700.000 | 9/9 | 100.0% | 0.44° | 27 | 1.11° | 1.03° / 3.46° |
| 800.000 | 9/9 | 100.0% | 0.61° | 27 | 0.80° | 0.92° / 2.10° |
| 868.000 | 9/9 | 100.0% | 1.35° | 27 | 1.58° | 1.56° / 4.08° |
| 915.000 | 9/9 | 100.0% | 0.42° | 27 | 0.84° | 0.81° / 2.08° |
| 1000.000 | 9/9 | 100.0% | 0.02° | 27 | 0.49° | 0.46° / 1.08° |
| 1100.000 | 9/9 | 100.0% | 0.42° | 27 | 1.25° | 1.19° / 3.39° |
| 1200.000 | 9/9 | 100.0% | 0.63° | 27 | 0.87° | 0.96° / 2.63° |
| 1250.000 | 9/9 | 100.0% | 0.90° | 27 | 1.35° | 1.24° / 3.26° |
| 1280.000 | 9/9 | 100.0% | 0.11° | 27 | 1.00° | 0.91° / 2.81° |
| 1290.000 | 9/9 | 100.0% | 1.10° | 27 | 1.42° | 1.66° / 4.15° |
| 1299.000 | 9/9 | 100.0% | 0.69° | 27 | 0.80° | 0.99° / 1.70° |
| 1300.000 | 9/9 | 100.0% | 1.21° | 27 | 1.59° | 1.87° / 4.78° |
| 1301.000 | 9/9 | 100.0% | 0.90° | 27 | 0.90° | 1.06° / 2.49° |
| 1310.000 | 9/9 | 100.0% | 0.48° | 27 | 1.35° | 1.33° / 3.29° |
| 1320.000 | 9/9 | 100.0% | 1.05° | 27 | 1.50° | 1.57° / 4.37° |
| 1500.000 | 8/9 | 88.9% | 0.96° | 25 | 1.12° | 1.25° / 2.95° |
| 1800.000 | 9/9 | 100.0% | 0.09° | 27 | 0.22° | 0.22° / 0.65° |
| 2100.000 | 9/9 | 100.0% | 0.17° | 27 | 0.34° | 0.30° / 0.83° |
| 2300.000 | 7/9 | 77.8% | 0.25° | 21 | 0.78° | 0.73° / 2.03° |
| 2400.000 | 7/9 | 77.8% | 0.07° | 21 | 0.34° | 0.25° / 0.93° |
| 2412.000 | 7/9 | 77.8% | 0.29° | 21 | 0.51° | 0.46° / 1.12° |
| 2437.000 | 7/9 | 77.8% | 0.14° | 21 | 0.50° | 0.42° / 1.56° |
| 2467.000 | 7/9 | 77.8% | 0.38° | 21 | 0.88° | 0.82° / 2.01° |
| 2700.000 | 7/9 | 77.8% | 0.24° | 21 | 0.35° | 0.29° / 0.90° |
| 3000.000 | 7/9 | 77.8% | 0.14° | 21 | 0.16° | 0.18° / 0.47° |
| 3300.000 | 7/9 | 77.8% | 0.14° | 21 | 0.98° | 0.67° / 1.54° |
| 3600.000 | 7/9 | 77.8% | 1.08° | 21 | 1.68° | 1.91° / 5.62° |
| 3900.000 | 7/9 | 77.8% | 0.20° | 21 | 0.65° | 0.60° / 2.08° |
| 3990.000 | 7/9 | 77.8% | 0.29° | 21 | 0.55° | 0.57° / 1.39° |
| 3999.000 | 7/9 | 77.8% | 0.67° | 21 | 1.11° | 0.99° / 3.00° |
| 4000.000 | 7/9 | 77.8% | 0.17° | 21 | 0.38° | 0.35° / 0.94° |
| 4001.000 | 7/9 | 77.8% | 0.47° | 21 | 1.00° | 0.91° / 2.37° |
| 4010.000 | 7/9 | 77.8% | 0.23° | 21 | 1.00° | 0.93° / 2.76° |
| 4200.000 | 7/9 | 77.8% | 0.09° | 21 | 0.53° | 0.39° / 2.01° |
| 4500.000 | 7/9 | 77.8% | 0.17° | 21 | 0.26° | 0.29° / 0.60° |
| 4800.000 | 7/9 | 77.8% | 0.42° | 21 | 0.78° | 0.75° / 1.44° |
| 5100.000 | 7/9 | 77.8% | 0.93° | 21 | 1.29° | 1.54° / 3.36° |
| 5400.000 | 7/9 | 77.8% | 0.33° | 21 | 0.52° | 0.54° / 1.38° |
| 5600.000 | 7/9 | 77.8% | 0.68° | 21 | 1.58° | 1.35° / 2.61° |
| 5700.000 | 7/9 | 77.8% | 0.20° | 21 | 0.38° | 0.38° / 1.00° |
| 5766.000 | 7/9 | 77.8% | 0.35° | 21 | 0.48° | 0.47° / 1.18° |
| 5804.000 | 7/9 | 77.8% | 0.31° | 21 | 0.85° | 0.78° / 1.74° |
| 5838.000 | 7/9 | 77.8% | 0.73° | 21 | 1.05° | 1.04° / 3.07° |
| 5866.000 | 7/9 | 77.8% | 0.19° | 21 | 0.68° | 0.63° / 1.43° |
| 5900.000 | 7/9 | 77.8% | 0.70° | 21 | 0.97° | 1.03° / 2.90° |

## Gain-mismatch coverage

| Absolute RX gain mismatch | Passing cells | Cell coverage | Valid frames | Frame coverage |
|---:|---:|---:|---:|---:|
| 0–5 dB | 141/141 | 100.0% | 423/423 | 100.0% |
| 6–10 dB | 0/0 | n/a | 0/0 | n/a |
| 11–20 dB | 0/0 | n/a | 0/0 | n/a |
| 21–30 dB | 94/94 | 100.0% | 282/282 | 100.0% |
| 31–40 dB | 94/94 | 100.0% | 282/282 | 100.0% |
| 41–50 dB | 0/0 | n/a | 0/0 | n/a |
| 51–60 dB | 0/0 | n/a | 0/0 | n/a |
| 61–72 dB | 39/94 | 41.5% | 118/282 | 41.8% |

## Model diagnostics

- Leave-one-epoch-out circular MAE: 0.88°
- Leave-one-epoch-out circular RMSE: 1.31°
- Leave-one-epoch-out circular p95: 2.94°
- Leave-one-epoch-out maximum error: 6.65°

### Paired model comparisons

| Comparison | Held-out frames | First MAE / p95 | Second MAE / p95 | Recommended |
|---|---:|---:|---:|---|
| Ordered additive vs gain difference | 1104 | 0.88° / 2.94° | 1.28° / 3.99° | `additive` |
| Additive vs cell interaction | 1104 | 0.88° / 2.94° | 0.88° / 2.98° | `additive` |
| Frequency-specific vs shared curves | 1105 | 0.88° / 2.94° | 3.83° / 11.04° | `frequency_specific_gain_curves` |
| Frequency-specific vs gain-table-shared curves | 1105 | 0.88° / 2.94° | 3.28° / 10.22° | `frequency_specific_gain_curves` |
| Per-frequency vs constant-plus-delay baseline | 1105 | 0.88° / 2.94° | 9.35° / 27.08° | `frequency_specific_intercepts` |
| Unanchored vs one-frame anchor | 964 | 0.88° / 2.91° | 1.11° / 3.65° | `unanchored` |

Every row uses an identical held-out observation mask. Differences within the declared 0.1° MAE equivalence margin select the predeclared simpler operational model.

### Effective differential-delay description

- Common reference gain: 26 dB on RX1 and RX2
- Phase slope: -0.61° per 100 MHz
- Effective differential delay: 0.017 ns
- Free-space-equivalent signed path difference: 0.51 cm
- Linear-fit residual RMSE / maximum: 18.40° / 104.70°

Descriptive only: cables, PCB and analogue paths, LO retunes, and calibration state can all contribute. It is not asserted to be literal cable length.

### Signal-confidence tiers

| Minimum SNR in both channels | Eligible frames | Held-out frames | MAE | RMSE | p95 |
|---:|---:|---:|---:|---:|---:|
| -10 dB | 1105 | 1105 | 0.88° | 1.31° | 2.94° |
| 0 dB | 984 | 984 | 0.82° | 1.26° | 2.90° |
| 10 dB | 872 | 872 | 0.78° | 1.23° | 2.71° |

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
