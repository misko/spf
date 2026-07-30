# A–G dual-RX spectroscopy campaign analysis

## Executive summary

- All controlled A–G acquisitions are complete. B, C, and D retain their explicit cell-repeatability waivers; they are not silently relabelled as passes.
- B is the actual **11 dB three-pad treatment on RX1 of `.17` only**; `.18` is the unchanged control. C is the nominal uncharacterized 30 cm RX1 jumper on `.17` only.
- Phase convention is `RX1 minus RX2`. Treatment effects below use difference-of-differences: `(treated stage − treated A) − (control stage − control A)`.

## Treatment comparisons

| Stage vs A | Cells | Bias ° | MAE ° | P95 ° | Median amplitude Δ dB |
|---|---:|---:|---:|---:|---:|
| B | 565 | -100.114 | 91.612 | 166.389 | -10.493 |
| C | 565 | -75.179 | 88.860 | 171.998 | -0.344 |
| D | 565 | -0.637 | 5.843 | 42.661 | 0.054 |
| G | 565 | -1.601 | 5.968 | 42.543 | 0.039 |

![Control-corrected treatment phase](treatment_phase_difference_of_differences.png)

### Equal-gain effective delay by gain-table band

These slopes are descriptive effective delays from the `(26,26)` control-corrected treatment curve. They do not identify a unique physical cable or PCB path.

| Stage | Band | Delay ps | Free-space equivalent mm | Residual RMSE ° |
|---|---|---:|---:|---:|
| B | low_0_to_1300_mhz | 349.37 | 104.74 | 15.06 |
| B | mid_1301_to_4000_mhz | 314.06 | 94.15 | 17.37 |
| B | high_4001_to_6000_mhz | -213.18 | -63.91 | 72.57 |
| C | low_0_to_1300_mhz | 1493.78 | 447.82 | 21.18 |
| C | mid_1301_to_4000_mhz | 1460.73 | 437.92 | 21.83 |
| C | high_4001_to_6000_mhz | 1355.87 | 406.48 | 55.35 |
| D | low_0_to_1300_mhz | -2.14 | -0.64 | 0.65 |
| D | mid_1301_to_4000_mhz | -0.40 | -0.12 | 0.77 |
| D | high_4001_to_6000_mhz | 16.65 | 4.99 | 17.90 |
| G | low_0_to_1300_mhz | -1.36 | -0.41 | 0.74 |
| G | mid_1301_to_4000_mhz | 0.05 | 0.02 | 0.77 |
| G | high_4001_to_6000_mhz | 24.33 | 7.29 | 20.34 |

B changes the median RX1/RX2 amplitude ratio by -10.49 dB, independently confirming the nominal 11 dB RX1 pad stack. C produces 1356–1494 ps of effective delay across the three gain-table bands, consistent with the scale expected from a 30 cm coax jumper.

### Restored-baseline to hot-repeat stability (D → G)

| Radio | RF region | MAE ° | P95 ° | Median amplitude Δ dB |
|---|---|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | all | 0.957 | 3.664 | 0.007 |
| `104000bac4950008230026001b440a003a` | low_and_mid_at_or_below_4ghz | 0.373 | 1.244 | 0.010 |
| `104000bac4950008230026001b440a003a` | high_above_4ghz | 2.067 | 6.312 | 0.007 |
| `1040007c4a94000211000b009186843ef2` | all | 0.898 | 3.301 | -0.004 |
| `1040007c4a94000211000b009186843ef2` | low_and_mid_at_or_below_4ghz | 0.389 | 1.293 | 0.000 |
| `1040007c4a94000211000b009186843ef2` | high_above_4ghz | 1.862 | 9.382 | -0.020 |

D and G agree much more closely with each other than either agrees with A above 4 GHz. Therefore the persistent high-band A→D/G shift is not evidence of continuing thermal drift; it is a radio-specific state change that occurred after A and remained stable through G.

## TX-level experiment

| Radio | Frequency | Corrected slope median °/dB | Slope p05…p95 °/dB | −28 dB tone/floor median | Cells ≥45.6 dB |
|---|---:|---:|---:|---:|---:|
| `104000bac4950008230026001b440a003a` | 5100 MHz | 0.0961 | 0.0842…0.1109 | 46.36 dB | 16/27 (59.3%) |
| `104000bac4950008230026001b440a003a` | 5766 MHz | -0.0068 | -0.0374…-0.0015 | 51.83 dB | 27/27 (100.0%) |
| `1040007c4a94000211000b009186843ef2` | 5100 MHz | 0.0548 | -0.0159…0.1223 | 46.80 dB | 21/27 (77.8%) |
| `1040007c4a94000211000b009186843ef2` | 5766 MHz | -0.0080 | -0.0481…0.0062 | 51.68 dB | 27/27 (100.0%) |

![TX-level phase response](tx_level_phase_response.png)

![TX tone and muted floor](tx_level_tone_floor.png)

![Thermal anchors](thermal_anchor_drift.png)

## Model ladder

| Dataset | Model | Parameters | LOEO MAE ° | LOEO P95 ° |
|---|---|---:|---:|---:|
| A | Per-frequency additive gain LUT per radio | 1130 | 0.632 | 2.424 |
| A | Per-frequency antisymmetric gain LUT per radio | 678 | 1.002 | 3.234 |
| A | Frequency LUT + gain-table-specific symmetric gain LUT per radio | 238 | 3.319 | 11.455 |
| A | Strict universal per-frequency additive LUT | 565 | 6.795 | 33.275 |
| A | Gain-dependent branch-delay LUT per radio | 20 | 12.517 | 42.612 |
| F | Per-frequency additive gain LUT per radio | 276 | 0.899 | 3.272 |
| F | Per-frequency antisymmetric gain LUT per radio | 144 | 0.936 | 3.212 |
| F | Frequency LUT + gain-table-specific symmetric gain LUT per radio | 78 | 0.970 | 3.585 |
| F | Strict universal per-frequency additive LUT | 138 | 1.157 | 3.489 |
| F | Gain-dependent branch-delay LUT per radio | 92 | 6.954 | 15.294 |

Stage A shows that gain response changes substantially with exact frequency: the per-frequency independent model is the accuracy reference. Stage F shows that over the complete low/TIA gain range, the antisymmetric shared `H(g)` model is nearly as accurate and is the parsimonious default.

![Low-gain symmetric curves](low_gain_symmetric_H.png)

## Limitations

- The 11 dB pad stack, 30 cm jumper, and connector torque were not independently characterized. The control radio removes shared drift but cannot remove treatment-radio-specific retune events.
- The Stage E muted `−80 dB` capture is a floor measurement; its phase values fail normal tone-quality gates and are not interpreted.
- Thermal-anchor correction at 5100 MHz uses the 5766 MHz anchor as an additive drift proxy.
- Effective delays describe phase slope. They do not prove that a specific cable, PCB trace, analogue filter, or gain-table state is the sole mechanism.

## Reproduction

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis \
  --campaign-root /home/pi/spf-campaigns/spectroscopy_20260730_full_r2 \
  --treated-serial 104000bac4950008230026001b440a003a \
  --control-serial 1040007c4a94000211000b009186843ef2 \
  --output-dir <campaign-root>/analysis/campaign
```
