# Two-radio integer-gain curve comparison

- Radios: `104000bac4950008230026001b440a003a`, `1040007c4a94000211000b009186843ef2`
- Frequencies: 433.000 MHz, 600.000 MHz, 700.000 MHz, 800.000 MHz, 868.000 MHz, 915.000 MHz, 1000.000 MHz, 1100.000 MHz, 1200.000 MHz, 1250.000 MHz, 1280.000 MHz, 1290.000 MHz, 1299.000 MHz, 1300.000 MHz, 1301.000 MHz, 1310.000 MHz, 1320.000 MHz, 1500.000 MHz, 1800.000 MHz, 2100.000 MHz, 2300.000 MHz, 2400.000 MHz, 2411.000 MHz, 2411.950 MHz, 2412.000 MHz, 2437.000 MHz, 2464.000 MHz, 2467.000 MHz, 2467.100 MHz, 2700.000 MHz, 3000.000 MHz, 3300.000 MHz, 3600.000 MHz, 3900.000 MHz, 3990.000 MHz, 3999.000 MHz, 4000.000 MHz, 4001.000 MHz, 4010.000 MHz, 4200.000 MHz, 4500.000 MHz, 4800.000 MHz, 5100.000 MHz, 5400.000 MHz, 5600.000 MHz, 5700.000 MHz, 5766.000 MHz, 5770.000 MHz, 5804.000 MHz, 5838.000 MHz, 5839.000 MHz, 5866.000 MHz, 5900.000 MHz
- Reference gain: 26 dB

## Does the gain curve transfer between radios?

| Frequency | Curve correlation | RMS difference | Maximum difference | Shared-curve held-out MAE / p95 |
|---:|---:|---:|---:|---:|
| 433.000 MHz | 0.9977 | 0.68° | 1.78° | 1.13° / 2.10° |
| 600.000 MHz | 0.9822 | 0.50° | 1.65° | 0.61° / 1.46° |
| 700.000 MHz | 0.9197 | 1.48° | 3.38° | 0.97° / 2.44° |
| 800.000 MHz | 0.9932 | 0.91° | 2.60° | 1.19° / 2.26° |
| 868.000 MHz | 0.9852 | 0.92° | 2.14° | 0.61° / 1.49° |
| 915.000 MHz | 0.9677 | 0.59° | 1.69° | 0.71° / 1.79° |
| 1000.000 MHz | 0.9993 | 0.88° | 1.95° | 1.21° / 2.42° |
| 1100.000 MHz | 0.9830 | 0.77° | 1.82° | 0.50° / 1.40° |
| 1200.000 MHz | 0.9920 | 0.50° | 1.43° | 0.72° / 1.41° |
| 1250.000 MHz | 0.9927 | 0.59° | 1.79° | 1.07° / 2.20° |
| 1280.000 MHz | 0.9957 | 0.53° | 1.33° | 0.58° / 1.18° |
| 1290.000 MHz | 0.8808 | 1.59° | 3.47° | 0.83° / 1.76° |
| 1299.000 MHz | 0.9836 | 0.51° | 1.26° | 0.60° / 1.27° |
| 1300.000 MHz | 0.9890 | 0.40° | 1.03° | 0.68° / 1.51° |
| 1301.000 MHz | 0.9697 | 0.74° | 1.92° | 0.69° / 1.56° |
| 1310.000 MHz | 0.9792 | 0.69° | 2.03° | 0.86° / 1.87° |
| 1320.000 MHz | 0.9928 | 0.69° | 1.62° | 1.07° / 2.54° |
| 1500.000 MHz | 0.9933 | 0.56° | 1.61° | 1.08° / 1.98° |
| 1800.000 MHz | 0.9981 | 0.59° | 1.37° | 2.11° / 4.59° |
| 2100.000 MHz | 0.9996 | 0.32° | 0.79° | 0.83° / 2.33° |
| 2300.000 MHz | 0.9966 | 0.67° | 1.60° | 1.76° / 3.88° |
| 2400.000 MHz | 0.9989 | 0.32° | 1.00° | 1.75° / 2.59° |
| 2411.000 MHz | 0.9983 | 0.40° | 1.26° | 1.66° / 2.56° |
| 2411.950 MHz | 0.9979 | 0.45° | 1.63° | 1.70° / 2.81° |
| 2412.000 MHz | 0.9983 | 0.39° | 0.95° | 1.66° / 2.87° |
| 2437.000 MHz | 0.9986 | 0.37° | 0.76° | 1.39° / 2.27° |
| 2464.000 MHz | 0.9991 | 0.39° | 0.94° | 0.87° / 1.55° |
| 2467.000 MHz | 0.9981 | 0.67° | 1.64° | 0.91° / 1.71° |
| 2467.100 MHz | 0.9992 | 0.30° | 0.69° | 0.93° / 1.63° |
| 2700.000 MHz | 0.9983 | 0.44° | 1.45° | 1.65° / 2.51° |
| 3000.000 MHz | 0.9983 | 0.93° | 1.88° | 0.86° / 1.88° |
| 3300.000 MHz | 0.9939 | 0.99° | 2.75° | 1.03° / 2.87° |
| 3600.000 MHz | 0.9986 | 0.44° | 1.19° | 2.04° / 3.15° |
| 3900.000 MHz | 0.9993 | 0.39° | 1.14° | 2.56° / 3.60° |
| 3990.000 MHz | 0.9991 | 0.36° | 1.10° | 1.88° / 3.05° |
| 3999.000 MHz | 0.9990 | 0.37° | 1.34° | 1.76° / 2.83° |
| 4000.000 MHz | 0.9989 | 0.40° | 1.16° | 1.64° / 2.98° |
| 4001.000 MHz | 0.9985 | 0.51° | 1.60° | 1.34° / 2.67° |
| 4010.000 MHz | 0.9990 | 0.40° | 1.28° | 1.50° / 3.50° |
| 4200.000 MHz | 0.9952 | 0.83° | 1.89° | 3.08° / 5.97° |
| 4500.000 MHz | 0.9971 | 0.58° | 2.07° | 1.05° / 2.35° |
| 4800.000 MHz | 0.9948 | 1.02° | 3.73° | 1.59° / 4.66° |
| 5100.000 MHz | 0.7584 | 6.05° | 12.14° | 5.37° / 15.18° |
| 5400.000 MHz | 0.9072 | 3.53° | 8.65° | 4.24° / 11.63° |
| 5600.000 MHz | 0.9967 | 1.15° | 2.95° | 1.74° / 4.19° |
| 5700.000 MHz | 0.9848 | 1.59° | 4.55° | 1.47° / 4.04° |
| 5766.000 MHz | 0.9918 | 1.51° | 4.73° | 1.52° / 3.99° |
| 5770.000 MHz | 0.9927 | 1.46° | 3.96° | 1.53° / 4.58° |
| 5804.000 MHz | 0.9955 | 1.17° | 3.21° | 1.51° / 4.19° |
| 5838.000 MHz | 0.9953 | 1.39° | 4.48° | 1.49° / 4.04° |
| 5839.000 MHz | 0.9959 | 1.30° | 3.65° | 1.67° / 3.60° |
| 5866.000 MHz | 0.9946 | 1.36° | 3.87° | 1.83° / 4.10° |
| 5900.000 MHz | 0.9932 | 1.35° | 4.26° | 2.28° / 5.37° |

Using one radio-shared curve per exact frequency gives 1.46° MAE and 3.67° p95 on held-out cell means.
Using one curve across all measured frequencies gives 5.01° MAE and 14.94° p95.

Each radio/frequency retains its own intercept in these transfer tests. Only the gain-dependent curve is shared.

## Directional transfer to the other radio

| Source gain curve | Target radio | Held-out MAE / p95 | Maximum |
|---|---|---:|---:|
| `104000bac4950008230026001b440a003a` | `1040007c4a94000211000b009186843ef2` | 1.27° / 4.38° | 20.95° |
| `1040007c4a94000211000b009186843ef2` | `104000bac4950008230026001b440a003a` | 1.25° / 3.74° | 14.65° |

This is a stricter transfer test than the averaged curve: the source gain curve never uses the target radio. The target anchor is the circular mean of its quality-valid `(reference, reference)` cell at each frequency (three frames in this experiment). This directly tests a low-cost per-frequency onboarding measurement.

## Frequency sensitivity of the gain curve

| Radio | First frequency | Second frequency | Curve correlation | RMS difference | Maximum difference |
|---|---:|---:|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | 433.000 MHz | 600.000 MHz | -0.2223 | 10.08° | 21.13° |
| `1040007c4a94000211000b009186843ef2` | 433.000 MHz | 600.000 MHz | -0.1585 | 9.33° | 19.01° |

## Interpretation

The radio-specific intercept remains necessary. The comparison above tests whether the much cheaper gain-dependent term can be shared. Held-out cells, not training-axis cells, determine the reported transfer error.

![Cross-radio curve transfer by frequency](curve_transfer_by_frequency.png)

![Directional transfer by frequency](directional_transfer_by_frequency.png)
