# Integer-gain additive-cross analysis

- Pluto serial: `104000bac4950008230026001b440a003a`
- Phase convention: `angle(rx1) - angle(rx2)`
- Reference gain: 62 dB
- Design per frequency: 73 training-axis pairs + 1296 off-axis held-out pairs

The fit uses only `(gain, reference)` and `(reference, gain)` frames. Off-axis pairs are never used to estimate the gain curves.

## Held-out prediction

| Frequency | Valid held-out frames | Independent RX curves MAE / p95 | Shared H(g) MAE / p95 | RX1 vs -RX2 correlation | Curve disagreement RMS |
|---:|---:|---:|---:|---:|---:|
| 5766.000 MHz | 6480 | 1.00° / 2.54° | 26.82° / 62.26° | 0.9780 | 39.17° |
| 5840.000 MHz | 6480 | 1.35° / 2.92° | 30.90° / 67.92° | 0.9693 | 44.16° |

## Overall result

- Independent RX1/RX2 curves: 1.17° MAE, 2.75° p95.
- Shared antisymmetric H(g) curve: 28.86° MAE, 67.47° p95.

## Legacy 17-gain grid on the same held-out cells

| Frequency | Dense integer MAE / p95 | Sparse linear MAE / p95 | Sparse nearest MAE / p95 |
|---:|---:|---:|---:|
| 5766.000 MHz | 26.80° / 61.78° | 27.15° / 72.51° | 28.46° / 79.47° |
| 5840.000 MHz | 30.90° / 67.79° | 31.11° / 79.16° | 32.26° / 86.38° |

Linear interpolation and nearest-neighbour use only the previously published 17 stage-focused gains. Their errors are scored against the same off-axis cells as the dense integer curve.

## Largest adjacent integer-gain steps

| Frequency | Gain transition | Absolute phase step | Signed phase step |
|---:|---:|---:|---:|
| 5766.000 MHz | 40→41 dB | 47.23° | -47.23° |
| 5766.000 MHz | 54→55 dB | 1.82° | +1.82° |
| 5766.000 MHz | 61→62 dB | 1.40° | +1.40° |
| 5766.000 MHz | 58→59 dB | 1.27° | +1.27° |
| 5766.000 MHz | 50→51 dB | 0.84° | +0.84° |
| 5766.000 MHz | 52→53 dB | 0.83° | +0.83° |
| 5766.000 MHz | 47→48 dB | 0.81° | +0.81° |
| 5766.000 MHz | 57→58 dB | 0.66° | +0.66° |
| 5766.000 MHz | 43→44 dB | 0.64° | -0.64° |
| 5766.000 MHz | 49→50 dB | 0.62° | -0.62° |
| 5840.000 MHz | 40→41 dB | 50.19° | -50.19° |
| 5840.000 MHz | 61→62 dB | 2.19° | +2.19° |
| 5840.000 MHz | 54→55 dB | 1.17° | +1.17° |
| 5840.000 MHz | 50→51 dB | 1.06° | -1.06° |
| 5840.000 MHz | 57→58 dB | 1.01° | +1.01° |
| 5840.000 MHz | 43→44 dB | 0.91° | -0.91° |
| 5840.000 MHz | 52→53 dB | 0.73° | +0.73° |
| 5840.000 MHz | 55→56 dB | 0.73° | +0.73° |
| 5840.000 MHz | 60→61 dB | 0.64° | +0.64° |
| 5840.000 MHz | 49→50 dB | 0.60° | +0.60° |

Equal manual and AGC-reported dB states are treated as the same gain state for this experiment. The result tests gain-table structure; it does not replay historical capture timing.
