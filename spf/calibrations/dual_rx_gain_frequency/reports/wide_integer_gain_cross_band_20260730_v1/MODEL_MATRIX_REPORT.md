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
| Strict universal per-frequency additive LUT | universal | 6731 | 3.239 | 6.796 | 16.051 | 41.972 | 100.00% |
| Strict universal full cell LUT | universal | 9268 | 3.245 | 6.799 | 16.050 | 42.378 | 100.00% |
| Frequency LUT + additive gain LUT per radio | per_radio | 358 | 4.827 | 6.480 | 13.448 | 29.134 | 100.00% |
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
  --dataset artifacts/dual_rx_gain_frequency/overnight_wide_integer_gain_cross_20260730_special_17_18_v1/104000bac4950008230026001b440a003a/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/overnight_wide_integer_gain_cross_20260730_special_17_18_v1/1040007c4a94000211000b009186843ef2/calibration.v7.zarr \
  --output-dir artifacts/dual_rx_gain_frequency/model_matrix
```
