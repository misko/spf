# Dual-RX phase model matrix

## Scope and evaluation

- Phase convention: `RX1 minus RX2`.
- Reference gain: 26 dB.
- Reference frequency for delay fits: 2.975831 GHz.
- Differential phase identifies only RX1 delay minus RX2 delay. Reported branch-delay LUT terms are relative contributions under the stated reference constraints, not absolute physical delays.

## Input checkpoint

| Pluto serial | Completed | Quality-valid | Validation | Scalar-input SHA-256 |
|---|---:|---:|---|---|
| `104000bac4950008230026001b440a003a` | 27825 | 27800 | `fail_quality` | `5b6e5fca896d7a9bc15f715f359f5df00640dea83f18e6f05a795e8385519d6e` |
| `1040007c4a94000211000b009186843ef2` | 27825 | 27802 | `fail_quality` | `b67eb443c732b816a88113a185d04bb524424af79dac21702256308c1b39c96c` |

All datasets are structurally complete. `fail_quality` means quality-rejected frames remain explicit in the V7 dataset; only quality-valid observations enter these fits.

## Evaluation design

The main table uses leave-one-epoch-out prediction: two repeats train the model and the third is unseen. This is the fair operational test for lookup tables whose deployment support is the measured grid.

## Known-cell prediction

| Model | Scope | Parameters | MAE ° | RMSE ° | P95 ° | Max ° | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per-frequency additive gain LUT per radio | per_radio | 13462 | 0.713 | 1.118 | 2.528 | 11.772 | 100.00% |
| Full frequency/gain-pair LUT per radio | per_radio | 18535 | 0.723 | 1.151 | 2.597 | 11.437 | 100.00% |
| Per-frequency antisymmetric gain LUT per radio | per_radio | 6784 | 0.912 | 1.386 | 2.963 | 13.147 | 100.00% |
| Frequency LUT + gain-table-specific f-scaled symmetric gain LUT per radio | per_radio | 484 | 2.665 | 4.239 | 10.368 | 23.253 | 100.00% |
| Frequency LUT + gain-table-specific symmetric gain LUT per radio | per_radio | 484 | 2.991 | 4.580 | 10.845 | 23.621 | 100.00% |
| Strict universal per-frequency additive LUT | universal | 6731 | 3.239 | 6.796 | 16.051 | 41.972 | 100.00% |
| Strict universal full cell LUT | universal | 9268 | 3.245 | 6.799 | 16.050 | 42.378 | 100.00% |
| Frequency LUT + f-scaled symmetric gain LUT per radio | per_radio | 232 | 4.517 | 5.919 | 11.923 | 26.679 | 100.00% |
| Frequency LUT + additive gain LUT per radio | per_radio | 358 | 4.827 | 6.480 | 13.448 | 29.134 | 100.00% |
| Frequency LUT + symmetric gain LUT per radio | per_radio | 232 | 4.828 | 6.484 | 13.502 | 28.679 | 100.00% |
| Frequency LUT + linear gains per radio | per_radio | 110 | 6.268 | 8.230 | 16.239 | 35.628 | 100.00% |
| Strict universal frequency + gain LUT | universal | 179 | 6.303 | 9.303 | 19.590 | 48.177 | 100.00% |
| Gain-dependent branch-delay LUT per radio | per_radio | 508 | 10.899 | 16.876 | 37.160 | 96.613 | 100.00% |
| Strict universal gain-dependent delay LUT | universal | 254 | 11.023 | 17.304 | 34.978 | 102.257 | 100.00% |
| One delay + additive gain LUT per radio | per_radio | 256 | 11.222 | 17.269 | 38.254 | 99.127 | 100.00% |
| Additive gain LUT per radio | per_radio | 254 | 13.958 | 20.291 | 48.493 | 89.423 | 100.00% |
| Linear gains per radio | per_radio | 6 | 14.779 | 20.916 | 49.051 | 89.630 | 100.00% |
| Constant per radio | per_radio | 2 | 14.810 | 21.166 | 49.754 | 90.281 | 100.00% |

![Known-cell model comparison](known_cell_model_comparison.png)

### Best model by radio

| Pluto serial | MAE ° | RMSE ° | P95 ° | Max ° |
|---|---:|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | 0.719 | 1.112 | 2.531 | 11.772 |
| `1040007c4a94000211000b009186843ef2` | 0.706 | 1.124 | 2.525 | 9.097 |

### Symmetric default versus independent accuracy reference

The parsimonious default assumes RX1 and RX2 share one physical gain-state response. Under the RX1-minus-RX2 phase convention, that response enters with opposite signs:

```text
independent:   phase = C[f] + A[f,g1] + B[f,g2]
antisymmetric: phase = C[f] + H[f,g1] - H[f,g2]
```

| Model | Parameters per radio/frequency | Per radio | Fleet total | LOEO MAE | LOEO p95 |
|---|---:|---:|---:|---:|---:|
| Independent A/B | 1 + 63 + 63 = 127 | 6731 | 13462 | 0.713° | 2.528° |
| Shared antisymmetric H | 1 + 63 = 64 | 3392 | 6784 | 0.912° | 2.963° |

The fleet totals above cover 2 radios. The reference gain (26 dB) is fixed to zero in each gain LUT, so it is not an additional fitted coefficient.

Relative to independent A/B, symmetric H changes LOEO MAE by +0.200°, RMSE by +0.268°, p95 by +0.435°, and maximum error by +1.375°. It removes 6678 parameters (49.6%).

Both models are always fitted. Symmetric H is the default structural model; independent A/B remains the accuracy reference, and the H-minus-A/B gap is a required output.

#### Gap by radio

| Pluto serial | Symmetric MAE | Independent MAE | MAE gap | Symmetric p95 | Independent p95 | P95 gap |
|---|---:|---:|---:|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | 0.898° | 0.719° | +0.179° | 2.785° | 2.531° | +0.254° |
| `1040007c4a94000211000b009186843ef2` | 0.926° | 0.706° | +0.220° | 3.178° | 2.525° | +0.653° |

### Can one gain LUT be scaled across frequency?

The direct test is deliberately stricter than the epoch holdout above. It fits only the additive-cross axes and predicts 15,216 quality-valid observations from 48 RX1/RX2 gain pairs that were never used for fitting.

```text
frequency-scaled: phase = C(r,f) + (f/GHz) * [G(r,g1) - G(r,g2)]
```

`C(r,f)` is still an exact-frequency equal-gain anchor. A separate scalar `k` and learned `G` are not identifiable, so `k` is absorbed into the LUT coefficients.

| Gain model | Parameters | Held-out MAE | Held-out p95 |
|---|---:|---:|---:|
| Frequency LUT + symmetric gain LUT per radio | 232 | 4.995° | 14.830° |
| Frequency LUT + f-scaled symmetric gain LUT per radio | 232 | 4.551° | 12.931° |
| Frequency LUT + gain-table-specific symmetric gain LUT per radio | 484 | 2.743° | 9.656° |
| Frequency LUT + gain-table-specific f-scaled symmetric gain LUT per radio | 484 | 2.496° | 8.744° |
| Per-frequency antisymmetric gain LUT per radio | 6784 | 1.074° | 3.485° |
| Per-frequency additive gain LUT per radio | 13462 | 0.805° | 2.699° |

Forcing one global LUT to scale with frequency improves MAE from 4.995° to 4.551°. Separating the three AD936x gain-table bands matters much more; adding the frequency factor to those three LUTs improves MAE from 2.743° to 2.496°.

![Frequency-scaled model held-out comparison](frequency_scaled_gain_model_comparison.png)

The exact fitted `H(r,f,g)` curves provide a second, descriptive low-rank test. A best rank-1 approximation may learn any scale at each frequency; the forced-frequency column specifically requires the scale to be proportional to `f`.

| Frequency scope | Forced f scaling | Best rank 1 | Best rank 2 |
|---|---:|---:|---:|
| All frequencies | 57.3% | 70.3% | 89.5% |
| Low table, ≤1.3 GHz | 28.0% | 76.8% | 96.2% |
| Middle table, 1.3–4.0 GHz | 79.3% | 87.3% | 99.2% |
| High table, >4.0 GHz | 84.6% | 86.8% | 99.4% |

![Gain LUT low-rank structure](gain_lut_low_rank_structure.png)

Conclusion: proportional-to-frequency scaling is real but only a coarse compression. It cannot move gain-stage discontinuities or represent frequency-dependent analogue dispersion. One scaled LUT per hardware gain-table band is a useful fallback; exact-frequency H remains the parsimonious precision default, with independent A/B as the accuracy reference. Rank two is a promising future compressed model, but it is not recommended until its phase prediction is evaluated on held-out data.

### What H(r,f,g) looks like across gain

`H` is a circular phase correction LUT in degrees, not a gain value. At fixed radio and frequency it is anchored at `H(26 dB) = 0`. Across all fitted curves, the median absolute 1 dB step is 0.304°, p95 is 4.189°, and p99 is 14.444°. The low median and large tail describe a mostly flat or gently varying staircase with a few hardware gain-stage jumps.

| Radio | Frequency | H range | Span | Largest adjacent step |
|---|---:|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | 915.000 MHz | -3.43° to 3.39° | 6.82° | 32→33 dB: +5.72° |
| `104000bac4950008230026001b440a003a` | 2412.000 MHz | -19.92° to 1.38° | 21.29° | 49→50 dB: -15.87° |
| `104000bac4950008230026001b440a003a` | 2467.100 MHz | -16.73° to 2.13° | 18.85° | 49→50 dB: -13.99° |
| `104000bac4950008230026001b440a003a` | 4000.000 MHz | -24.03° to 3.99° | 28.02° | 49→50 dB: -17.42° |
| `104000bac4950008230026001b440a003a` | 4001.000 MHz | -17.76° to 9.98° | 27.73° | 40→41 dB: -17.16° |
| `104000bac4950008230026001b440a003a` | 5804.000 MHz | -17.77° to 12.97° | 30.75° | 40→41 dB: -16.17° |
| `1040007c4a94000211000b009186843ef2` | 915.000 MHz | -4.84° to 2.43° | 7.27° | 32→33 dB: +6.41° |
| `1040007c4a94000211000b009186843ef2` | 2412.000 MHz | -19.62° to 1.04° | 20.65° | 49→50 dB: -15.84° |
| `1040007c4a94000211000b009186843ef2` | 2467.100 MHz | -16.12° to 2.04° | 18.16° | 49→50 dB: -14.27° |
| `1040007c4a94000211000b009186843ef2` | 4000.000 MHz | -24.37° to 3.40° | 27.77° | 49→50 dB: -17.38° |
| `1040007c4a94000211000b009186843ef2` | 4001.000 MHz | -17.57° to 9.49° | 27.07° | 40→41 dB: -17.13° |
| `1040007c4a94000211000b009186843ef2` | 5804.000 MHz | -18.13° to 12.94° | 31.07° | 40→41 dB: -16.29° |

![Symmetric gain-response LUT slices](symmetric_gain_lut_slices.png)

## Unseen-frequency test

| Model | Scope | MAE ° | RMSE ° | P95 ° | Coverage |
|---|---:|---:|---:|---:|---:|
| Gain-dependent branch-delay LUT per radio | per_radio | 11.430 | 17.802 | 39.518 | 100.00% |
| Strict universal gain-dependent delay LUT | universal | 11.526 | 18.115 | 37.213 | 100.00% |
| One delay + additive gain LUT per radio | per_radio | 11.724 | 18.160 | 40.719 | 100.00% |
| Additive gain LUT per radio | per_radio | 14.225 | 20.680 | 49.393 | 100.00% |
| Linear gains per radio | per_radio | 15.009 | 21.268 | 49.927 | 100.00% |
| Constant per radio | per_radio | 15.029 | 21.501 | 50.607 | 100.00% |

![Unseen-frequency model comparison](unseen_frequency_model_comparison.png)

### Fitted effective differential delays

| Model | Pluto serial | Base delay ps | Free-space equivalent mm |
|---|---|---:|---:|
| One delay + additive gain LUT per radio | `104000bac4950008230026001b440a003a` | 19.967 | 5.986 |
| One delay + additive gain LUT per radio | `1040007c4a94000211000b009186843ef2` | 12.644 | 3.791 |
| Gain-dependent branch-delay LUT per radio | `104000bac4950008230026001b440a003a` | 19.897 | 5.965 |
| Gain-dependent branch-delay LUT per radio | `1040007c4a94000211000b009186843ef2` | 12.521 | 3.754 |

## Strict universal transfer to an unseen radio

| Model | MAE ° | RMSE ° | P95 ° | Max ° | Coverage |
|---|---:|---:|---:|---:|---:|
| Strict universal per-frequency additive LUT | 6.213 | 13.475 | 32.538 | 73.181 | 100.00% |
| Strict universal full cell LUT | 6.217 | 13.477 | 32.530 | 73.181 | 100.00% |
| Strict universal frequency + gain LUT | 8.552 | 14.844 | 32.513 | 78.033 | 100.00% |
| Strict universal gain-dependent delay LUT | 11.681 | 18.529 | 38.351 | 107.733 | 100.00% |

## Physical interpretation

A delay-only explanation is supported only if the delay models are competitive on the unseen-frequency test. A good fit on the measured frequencies alone is insufficient: a frequency lookup can absorb arbitrary retune- or band-dependent phase offsets.

The gain-dependent branch-delay model assigns relative delay terms to RX1 and RX2 gain states. Because only RX1−RX2 phase is observed, adding the same absolute delay to both paths is unobservable; the individual physical path lengths cannot be recovered from this experiment.

## Interpretation and recommendation

The recommended correction on this measured grid is the per-radio, per-frequency additive lookup. The full cell lookup has many more parameters and no held-out benefit, so an RX1-by-RX2 interaction table is not justified by these data.

Do not use the strict universal model as a precision correction for an unseen radio. Its leave-one-radio-out error measures the cost of omitting radio-specific calibration.

The delay models are useful descriptions of a broad phase slope, but their unseen-frequency errors reject path imbalance as the only mechanism. Retune state, frequency-band analogue response, and frequency-dependent gain-table effects still require explicit frequency calibration or denser within-band characterization.

## Reproduction

```bash
python -m spf.calibrations.dual_rx_gain_frequency model-matrix \
  --config spf/calibrations/dual_rx_gain_frequency/configs/wide_integer_gain_cross_band.yaml \
  --dataset /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/overnight_wide_integer_gain_cross_20260730_special_17_18_v1/104000bac4950008230026001b440a003a/calibration.v7.zarr \
  --dataset /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/overnight_wide_integer_gain_cross_20260730_special_17_18_v1/1040007c4a94000211000b009186843ef2/calibration.v7.zarr \
  --output-dir artifacts/dual_rx_gain_frequency/model_matrix
```
