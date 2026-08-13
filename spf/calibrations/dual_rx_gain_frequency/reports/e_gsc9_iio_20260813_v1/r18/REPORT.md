# Integer-gain additive-cross analysis

- Pluto serial: `1040007c4a94000211000b009186843ef2`
- Phase convention: `angle(rx1) - angle(rx2)`
- Reference gain: 62 dB
- Design per frequency: 73 training-axis pairs + 1296 off-axis held-out pairs

The fit uses only `(gain, reference)` and `(reference, gain)` frames. Off-axis pairs are never used to estimate the gain curves.

## Held-out prediction

| Frequency | Valid held-out frames | Independent RX curves MAE / p95 | Shared H(g) MAE / p95 | RX1 vs -RX2 correlation | Curve disagreement RMS |
|---:|---:|---:|---:|---:|---:|
| 5766.000 MHz | 6480 | 0.36° / 0.90° | 2.17° / 3.28° | 0.9979 | 2.34° |
| 5840.000 MHz | 6480 | 0.40° / 1.10° | 3.24° / 5.30° | 0.9918 | 3.77° |

## Overall result

- Independent RX1/RX2 curves: 0.38° MAE, 1.00° p95.
- Shared antisymmetric H(g) curve: 2.70° MAE, 5.07° p95.

## Legacy 17-gain grid on the same held-out cells

| Frequency | Dense integer MAE / p95 | Sparse linear MAE / p95 | Sparse nearest MAE / p95 |
|---:|---:|---:|---:|
| 5766.000 MHz | 2.16° / 3.07° | 4.05° / 12.91° | 4.48° / 18.05° |
| 5840.000 MHz | 3.23° / 5.23° | 4.84° / 12.62° | 5.29° / 17.68° |

Linear interpolation and nearest-neighbour use only the previously published 17 stage-focused gains. Their errors are scored against the same off-axis cells as the dense integer curve.

## Largest adjacent integer-gain steps

| Frequency | Gain transition | Absolute phase step | Signed phase step |
|---:|---:|---:|---:|
| 5766.000 MHz | 40→41 dB | 16.54° | -16.54° |
| 5766.000 MHz | 54→55 dB | 0.92° | +0.92° |
| 5766.000 MHz | 52→53 dB | 0.73° | +0.73° |
| 5766.000 MHz | 57→58 dB | 0.59° | +0.59° |
| 5766.000 MHz | 60→61 dB | 0.54° | +0.54° |
| 5766.000 MHz | 53→54 dB | 0.52° | +0.52° |
| 5766.000 MHz | 56→57 dB | 0.51° | +0.51° |
| 5766.000 MHz | 59→60 dB | 0.43° | +0.43° |
| 5766.000 MHz | 61→62 dB | 0.42° | +0.42° |
| 5766.000 MHz | 55→56 dB | 0.33° | +0.33° |
| 5840.000 MHz | 40→41 dB | 16.18° | -16.18° |
| 5840.000 MHz | 52→53 dB | 0.70° | +0.70° |
| 5840.000 MHz | 55→56 dB | 0.66° | +0.66° |
| 5840.000 MHz | 60→61 dB | 0.66° | +0.66° |
| 5840.000 MHz | 53→54 dB | 0.57° | +0.57° |
| 5840.000 MHz | 56→57 dB | 0.56° | +0.56° |
| 5840.000 MHz | 38→39 dB | 0.51° | -0.51° |
| 5840.000 MHz | 39→40 dB | 0.49° | +0.49° |
| 5840.000 MHz | 54→55 dB | 0.36° | +0.36° |
| 5840.000 MHz | 61→62 dB | 0.34° | +0.34° |

Equal manual and AGC-reported dB states are treated as the same gain state for this experiment. The result tests gain-table structure; it does not replay historical capture timing.
