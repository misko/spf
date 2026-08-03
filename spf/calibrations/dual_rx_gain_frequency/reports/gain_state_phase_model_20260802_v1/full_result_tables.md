| Model | Params | LOEO MAE / P95 | LOFO MAE / P95 | LOBLK MAE | LORO MAE | cov (LOFO) |
|---|---:|---:|---:|---:|---:|---:|
| L00 zero (per-session anchor only) | 0 | 6.65 / 18.38 | 6.65 / 18.38 | 6.65 | 6.65 | 1.00 |
| L01 sym H(g), universal | 3 | 5.12 / 16.00 | 5.16 / 16.14 | 5.64 | 5.13 | 1.00 |
| L02 sym H(radio,g) | 6 | 5.11 / 16.02 | 5.16 / 16.14 | 5.64 | **unsupported** | 1.00 |
| L03 arm d1(g),d2(g) universal | 6 | 5.12 / 16.00 | 5.16 / 16.13 | 5.64 | 5.13 | 1.00 |
| L04 arm d1,d2 per radio | 12 | 5.12 / 16.00 | 5.16 / 16.16 | 5.65 | **unsupported** | 1.00 |
| L05 sym H(lna,mixer,tia,lpf) universal | 15 | 3.21 / 11.54 | 3.29 / 11.92 | 3.38 | 3.25 | 1.00 |
| L06 sym H(gain-table row) universal | 9 | 3.21 / 11.54 | 3.29 / 11.92 | 3.38 | 3.25 | 1.00 |
| L07 sym H(lna,mixer,tia,lpf) per radio | 30 | 3.20 / 11.63 | 3.28 / 12.02 | 3.38 | **unsupported** | 1.00 |
| L08 sym H(band,g) universal | 9 | 3.21 / 11.54 | 3.29 / 11.92 | 3.38 | 3.25 | 1.00 |
| L09 sym H(radio,band,g) | 18 | 3.20 / 11.63 | 3.28 / 12.02 | 3.38 | **unsupported** | 1.00 |
| L10 arm d(radio,band,g) | 36 | 3.20 / 11.57 | 3.28 / 11.91 | 3.39 | **unsupported** | 1.00 |
| L11 sym H(band,g) + delay(g) universal | 12 | 2.99 / 11.54 | 3.08 / 12.04 | 3.14 | 3.05 | 1.00 |
| L12 sym H(band,g) + delay(g) per radio | 24 | 2.98 / 11.62 | 3.08 / 12.13 | 3.15 | **unsupported** | 1.00 |
| L13 sym H(state) + delay(lna) universal | 18 | 3.21 / 11.51 | 3.31 / 12.07 | 3.50 | 3.26 | 1.00 |
| L14 sym H(band,g) + 1 ripple, amp per g, universal | 15 | 2.85 / 8.88 | 2.99 / 9.34 | 3.25 | 2.90 | 1.00 |
| L15 sym H(radio,band,g) + 1 ripple per radio | 30 | 2.84 / 9.04 | 2.96 / 9.48 | 3.24 | **unsupported** | 1.00 |
| L16 MECHANISTIC: H(state) + ripple amp per LNA state | 21 | 2.42 / 8.55 | 2.50 / 8.92 | 2.70 | 2.49 | 1.00 |
| L17 MECHANISTIC + mixer ripple | 27 | 2.45 / 8.25 | 2.57 / 8.77 | 2.74 | 2.52 | 1.00 |
| L18 sym H(band,g) + 2 ripples, amp per g, universal | 21 | 2.54 / 8.07 | 2.70 / 8.60 | 3.49 | 2.71 | 1.00 |
| L19 sym H(band,g)+2 ripples per radio + delay | 48 | 2.42 / 8.31 | 2.56 / 8.94 | 3.30 | **unsupported** | 1.00 |
| L20 L18 + arm-specific ripple correction | 45 | 2.59 / 8.16 | 2.72 / 8.60 | 3.51 | **unsupported** | 1.00 |
| L21 sym H(band,g) + quad(f) per band-gain, per radio | 54 | 2.80 / 10.86 | 3.00 / 11.56 | 10.38 | **unsupported** | 1.00 |
| L22 L19 + quad(f) per band-gain | 78 | 2.15 / 7.69 | 2.41 / 8.70 | 9.56 | **unsupported** | 1.00 |
| L23 sym H(radio,f,g)  per-frequency antisym LUT | 678 | 0.99 / 3.85 | **unsupported** | **unsupported** | **unsupported** | 0.00 |
| L24 arm d(radio,f,g)  per-frequency additive LUT | 1356 | 0.62 / 2.82 | **unsupported** | **unsupported** | **unsupported** | 0.00 |
| L25 MECH: H(state) + ripple amp per (band,LNA) | 29 | 2.39 / 7.97 | 2.55 / 8.70 | 2.75 | 2.46 | 1.00 |
| L26 MECH: H(state) + 2 ripples per LNA state | 27 | 2.08 / 7.04 | 2.26 / 7.54 | 2.47 | 2.22 | 1.00 |
| L27 MECH: + delay(state) + 2 ripples per (band,LNA) | 49 | 1.68 / 6.19 | 1.85 / 6.93 | 3.52 | 1.91 | 1.00 |
| L28 MECH L27 + per-radio ripple amplitudes | 77 | 1.70 / 6.23 | 1.89 / 7.09 | 3.54 | **unsupported** | 1.00 |
| L29 AGNOSTIC: H(band,g) + 6 fixed-delay Fourier terms per g | 45 | 2.75 / 8.59 | 3.10 / 9.83 | 4.03 | 2.92 | 1.00 |
| L30 MIN: H(lna,mixer,tia) only, universal | 8 | 3.49 / 12.10 | 3.54 / 12.27 | 3.66 | 3.52 | 1.00 |
| L31 MIN + 2 ripples per LNA state, universal | 20 | 2.45 / 7.39 | 2.58 / 8.12 | 2.79 | 2.54 | 1.00 |
| L32 MIN + 2 ripples per (band,LNA) + delay(lna,mixer) | 42 | 2.12 / 6.86 | 2.33 / 7.71 | 5.04 | 2.28 | 1.00 |
| L33 L32 + linear LPF slope (1 param) | 43 | 1.81 / 6.38 | 1.99 / 7.28 | 3.58 | 2.00 | 1.00 |
| L34 L33 + ripple amps per (band,LNA,radio) | 71 | 1.82 / 6.36 | 2.02 / 7.25 | 3.64 | **unsupported** | 1.00 |


### Minimal models, pooled A+F+E+pilot leave-one-frequency-out

| Model | Params | MAE | P95 | coverage |
|---|---:|---:|---:|---:|
| L00 zero (per-session anchor only) | 0 | 5.56 | 17.86 | 1.00 |
| L01 sym H(g), universal | 27 | 4.54 | 15.00 | 1.00 |
| L05 sym H(lna,mixer,tia,lpf) universal | 26 | 2.91 | 11.22 | 1.00 |
| L06 sym H(gain-table row) universal | 40 | 2.86 | 11.15 | 0.99 |
| L16 MECHANISTIC: H(state) + ripple amp per LNA state | 32 | 2.33 | 8.51 | 1.00 |
| L26 MECH: H(state) + 2 ripples per LNA state | 38 | 2.11 | 6.69 | 1.00 |
| L30 MIN: H(lna,mixer,tia) only, universal | 9 | 2.99 | 11.39 | 1.00 |
| L31 MIN + 2 ripples per LNA state, universal | 21 | 2.26 | 7.36 | 1.00 |
| L32 MIN + 2 ripples per (band,LNA) + delay(lna,mixer) | 44 | 1.97 | 7.12 | 1.00 |
| L33 L32 + linear LPF slope (1 param) | 45 | 1.93 | 6.58 | 1.00 |
| L34 L33 + ripple amps per (band,LNA,radio) | 73 | 1.96 | 6.79 | 1.00 |
| L24 arm d(radio,f,g)  per-frequency additive LUT | 1748 | 5.56 | 17.86 | 0.00 |


### Leave-one-PROBE-GAIN-out, pooled (can the model predict an unmeasured requested gain?)

| Model | Params | coverage | fail-closed MAE | supported MAE | supported P95 |
|---|---:|---:|---:|---:|---:|
| L00 zero (per-session anchor only) | 0 | 1.000 | 5.56 | 5.56 | 17.86 |
| L01 sym H(g), universal | 26 | 0.477 | 4.88 | 3.00 | 10.45 |
| L05 sym H(lna,mixer,tia,lpf) universal | 26 | 0.885 | 4.64 | 3.29 | 13.19 |
| L06 sym H(gain-table row) universal | 39 | 0.782 | 4.99 | 3.77 | 15.10 |
| L16 MECHANISTIC: H(state) + ripple amp per LNA state | 32 | 0.885 | 4.76 | 3.42 | 14.20 |
| L26 MECH: H(state) + 2 ripples per LNA state | 38 | 0.885 | 4.54 | 3.17 | 11.94 |
| L30 MIN: H(lna,mixer,tia) only, universal | 9 | 0.896 | 4.67 | 3.32 | 12.26 |
| L31 MIN + 2 ripples per LNA state, universal | 21 | 0.896 | 4.47 | 3.09 | 11.28 |
| L32 MIN + 2 ripples per (band,LNA) + delay(lna,mixer) | 44 | 0.699 | 4.51 | 2.25 | 8.20 |
| L33 L32 + linear LPF slope (1 param) | 45 | 0.699 | 4.63 | 2.43 | 8.05 |
| L34 L33 + ripple amps per (band,LNA,radio) | 73 | 0.699 | 4.64 | 2.43 | 8.02 |
| L24 arm d(radio,f,g)  per-frequency additive LUT | 1752 | 0.164 | 5.68 | 0.93 | 2.93 |


### Pooled leave-one-gain-table-band-out (extrapolation across a whole band)

| Model | Params | coverage | fail-closed MAE | P95 |
|---|---:|---:|---:|---:|
| L00 zero (per-session anchor only) | 0 | 1.00 | 5.56 | 17.86 |
| L01 sym H(g), universal | 27 | 0.93 | 6.37 | 20.54 |
| L05 sym H(lna,mixer,tia,lpf) universal | 26 | 0.81 | 5.31 | 18.53 |
| L06 sym H(gain-table row) universal | 35 | 0.32 | 6.53 | 19.34 |
| L08 sym H(band,g) universal | 40 | 0.00 | 5.56 | 17.86 |
| L11 sym H(band,g) + delay(g) universal | 67 | 0.00 | 5.56 | 17.86 |
| L14 sym H(band,g) + 1 ripple, amp per g, universal | 94 | 0.00 | 5.56 | 17.86 |
| L16 MECHANISTIC: H(state) + ripple amp per LNA state | 32 | 0.81 | 5.09 | 17.74 |
| L26 MECH: H(state) + 2 ripples per LNA state | 38 | 0.81 | 5.36 | 18.25 |
| L30 MIN: H(lna,mixer,tia) only, universal | 9 | 0.90 | 5.09 | 18.11 |
| L31 MIN + 2 ripples per LNA state, universal | 21 | 0.90 | 5.96 | 19.26 |
| L32 MIN + 2 ripples per (band,LNA) + delay(lna,mixer) | 36 | 0.00 | 5.56 | 17.86 |
| L33 L32 + linear LPF slope (1 param) | 37 | 0.00 | 5.56 | 17.86 |
| L29 AGNOSTIC: H(band,g) + 6 fixed-delay Fourier terms per g | 364 | 0.00 | 5.56 | 17.86 |
| L24 arm d(radio,f,g)  per-frequency additive LUT | 1080 | 0.00 | 5.56 | 17.86 |


### Calibration comb spacing (stage A, fail-closed MAE, deg)

| Model | 2750 MHz gap | 1375 MHz gap | 688 MHz gap | 344 MHz gap | 172 MHz gap | 96 MHz gap |
|---|---:|---:|---:|---:|---:|---:|
| L00 anchor only | 6.65 | 6.65 | 6.65 | 6.65 | 6.65 | 6.65 |
| L08 sym H(band,g) | 5.96 | 3.43 | 3.38 | 3.41 | 3.42 | 3.35 |
| L16 MECH H(state)+ripple/LNA | 5.50 | 2.90 | 2.70 | 2.75 | 2.63 | 2.60 |
| L26 MECH H(state)+2 ripples/LNA | 5.48 | 2.88 | 2.47 | 2.40 | 2.35 | 2.25 |
| L27 MECH +delay +2 ripples/(band,LNA) | 4.92 | 7.55 | 3.52 | 2.17 | 2.17 | 1.98 |
| L24 per-frequency additive LUT | 6.65 | 6.65 | 6.65 | 6.65 | 6.65 | 6.65 |


### Which parameters must be radio-specific?

| Variant | Params | LOEO MAE | LORO MAE | LORO coverage |
|---|---:|---:|---:|---:|
| all universal | 43 | 1.810 | 2.003 | 1.00 |
| + per-radio static H | 52 | 1.808 | 6.647 | 0.00 |
| + per-radio delay | 49 | 1.822 | 6.647 | 0.00 |
| + per-radio ripple amplitudes | 71 | 1.824 | 6.647 | 0.00 |
| + everything per-radio | 86 | 1.816 | 6.647 | 0.00 |
