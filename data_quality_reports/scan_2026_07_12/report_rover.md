# Dataset quality scan — rover (139 datasets)

## Group summary (device x d/lambda)

| device | d/lam | n | med circstd raw r0 | med corr r0 | med g r0 | med dtheta r0 | med heading | QUAR | FLAG |
|---|---|---|---|---|---|---|---|---|---|
| PLUTO | 0.41762 | 16 | 0.600 | 0.490 | 0.960 | -0.180 | -0.165 | 2 | 9 |
| PLUTO | 0.41764 | 16 | 0.801 | 0.540 | 1.070 | -0.210 | -0.165 | 5 | 10 |
| PLUTO | 0.67317 | 21 | 1.038 | 1.008 | 0.900 | 0.050 | 0.070 | 0 | 21 |
| PLUTO | 0.68484 | 19 | 0.630 | 0.552 | 1.060 | -0.010 | 0.020 | 7 | 11 |
| PLUTO | 0.82703 | 24 | 1.159 | 0.977 | 0.930 | -0.080 | -0.105 | 0 | 19 |
| PLUTO | 0.84138 | 28 | 0.842 | 0.370 | 0.950 | 0.080 | 0.090 | 1 | 15 |
| PLUTO | 0.91964 | 7 | 0.706 | 0.569 | 0.900 | 0.070 | 0.090 | 0 | 7 |
| PLUTO | 0.99302 | 8 | 0.965 | 0.773 | 0.910 | -0.100 | -0.160 | 0 | 7 |

## Worst 25 (by corrected circstd r0)

| dataset | status | reasons | nan% r0/r1 | circstd raw->corr r0 | g r0 | dtheta r0 |
|---|---|---|---|---|---|---|
| rover_2025_04_05_21_55_51_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 57/51 | 2.00->1.83 | 0.92 | -0.35 |
| rover_2025_04_05_20_52_12_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 50/48 | 1.99->1.80 | 0.94 | -0.35 |
| jan30_mission1_rover1 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 88/77 | 1.79->1.75 | 1.10 | +0.15 |
| rover_2025_04_05_19_28_49_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=-0.28;FLAG:r0_noisy;FLAG:r1 | 76/61 | 1.82->1.74 | 1.02 | -0.35 |
| rover_2025_04_05_21_27_13_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=-0.35;FLAG:r0_noisy;FLAG:r1 | 60/46 | 2.26->1.64 | 0.94 | -0.35 |
| rover_2025_04_05_22_23_44_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 50/45 | 1.73->1.63 | 0.98 | +0.35 |
| dec18_mission1_rover1 | FLAG | FLAG:r0_noisy;FLAG:fit_at_bound | 67/81 | 1.66->1.62 | 0.90 | +0.03 |
| rover_2025_04_05_21_48_21_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 80/62 | 1.62->1.48 | 0.90 | -0.05 |
| jan30_mission2_rover1 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 87/78 | 1.37->1.37 | 1.04 | +0.05 |
| rover_2025_04_05_20_56_16_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy | 63/60 | 1.32->1.32 | 0.96 | +0.01 |
| rover_2025_04_05_19_30_17_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 62/32 | 1.60->1.30 | 0.90 | +0.29 |
| rover_2025_04_05_22_18_11_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=-0.30;FLAG:r0_noisy;FLAG:r1 | 73/45 | 1.74->1.30 | 0.90 | -0.35 |
| rover_2025_04_05_21_48_21_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 83/66 | 1.40->1.24 | 0.90 | -0.15 |
| rover_2025_04_05_22_18_11_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=-0.35;FLAG:r0_noisy;FLAG:r1 | 72/49 | 1.54->1.20 | 0.90 | -0.35 |
| rover_2025_04_05_22_52_51_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=0.35;FLAG:r0_noisy;FLAG:r1_ | 54/55 | 1.92->1.19 | 0.90 | +0.35 |
| rover_2025_04_05_20_03_50_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:ts_nonm | 57/41 | 1.28->1.19 | 0.90 | +0.15 |
| rover_2025_04_05_16_54_58_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy | 55/38 | 1.26->1.18 | 0.98 | +0.15 |
| rover_2025_04_05_18_58_22_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy | 56/35 | 1.19->1.18 | 0.98 | +0.05 |
| rover_2025_04_05_20_52_12_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 46/45 | 1.34->1.18 | 0.90 | +0.09 |
| rover_2025_04_05_22_46_03_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 76/51 | 1.22->1.15 | 0.92 | -0.15 |
| rover_2025_04_05_17_55_14_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:fit_at_ | 75/76 | 1.21->1.11 | 0.90 | -0.09 |
| dec18_mission2_rover1 | FLAG | FLAG:heading=-0.28;FLAG:r0_noisy;FLAG:ts | 89/70 | 1.29->1.10 | 0.92 | -0.33 |
| rover_2025_04_05_21_20_32_nRX2_bounce_spacing0p0 | FLAG | FLAG:heading=-0.27;FLAG:r0_noisy;FLAG:r1 | 81/56 | 1.54->1.09 | 0.98 | -0.35 |
| rover_2025_04_05_18_58_30_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy;FLAG:ts_nonm | 73/74 | 1.16->1.07 | 0.90 | -0.09 |
| rover_2025_04_05_17_31_17_nRX2_bounce_spacing0p0 | FLAG | FLAG:r0_noisy;FLAG:r1_noisy | 28/28 | 1.04->1.01 | 0.94 | +0.05 |

## Errors (23)

- rover_2025_02_23_23_53_30_nRX2_center_spacing0p043_tag_RO3.rover_2025_02_23_23_45_29_nRX1_circle_spacing0p05075_tag_RO2: AssertionError: Too many mismatches in rx_spacing 3000.0
- wallarrayv3_2025_01_31_21_20_40_nRX2_bounce_spacing0p07: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_05_06_38_25_nRX2_bounce_spacing0p07: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_05_06_38_25_nRX2_bounce_spacing0p07.zarr: No such file or directory
- wallarrayv3_2025_02_10_22_49_58_nRX2_rx_random_circle_spacing0p07: AssertionError: Too many mismatches in rx_spacing 4997.0
- wallarrayv3_2025_02_11_00_10_15_nRX2_bounce_spacing0p07: ValueError: Segmentation file does not exist for /mnt/md2/cache/precompute_cache_3p7/wallarrayv3_2025_02_11_00_10_15_nRX
- wallarrayv3_2025_02_11_17_45_04_nRX2_bounce_spacing0p065: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_11_17_45_04_nRX2_bounce_spacing0p065.zarr: No such file or director
- wallarrayv3_2025_02_13_21_07_06_nRX2_bounce_spacing0p065: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_13_21_07_06_nRX2_bounce_spacing0p065.zarr: No such file or director
- wallarrayv3_2025_02_13_22_46_06_nRX2_bounce_spacing0p065: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_13_22_46_06_nRX2_bounce_spacing0p065.zarr: No such file or director
- wallarrayv3_2025_02_13_04_18_52_nRX2_bounce_spacing0p065: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_13_04_18_52_nRX2_bounce_spacing0p065.zarr: No such file or director
- wallarrayv3_2025_02_14_01_16_13_nRX2_bounce_spacing0p065: Error: /mnt/md2/cache/nosig_data/wallarrayv3_2025_02_14_01_16_13_nRX2_bounce_spacing0p065.zarr: No such file or director
- wallarrayv3_2025_02_19_10_25_30_nRX2_bounce_spacing0p065: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_19_10_33_22_nRX2_bounce_spacing0p065: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_19_12_11_09_nRX2_bounce_spacing0p065: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_19_15_51_16_nRX2_bounce_spacing0p065: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_19_18_02_04_nRX2_bounce_spacing0p065: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_23_11_55_09_nRX2_rx_random_circle_spacing0p025: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_02_26_00_58_26_nRX2_bounce_spacing0p025: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_03_10_13_57_16_nRX2_bounce_spacing0p08: AssertionError: 0.000000 -0.250000
- wallarrayv3_2025_03_30_19_29_02_nRX2_bounce_spacing0p08: ValueError: Segmentation file does not exist for /mnt/md2/cache/precompute_cache_3p7/wallarrayv3_2025_03_30_19_29_02_nRX
- wallarrayv3_2025_03_31_17_02_19_nRX2_bounce_spacing0p08: ValueError: Segmentation file does not exist for /mnt/md2/cache/precompute_cache_3p7/wallarrayv3_2025_03_31_17_02_19_nRX
