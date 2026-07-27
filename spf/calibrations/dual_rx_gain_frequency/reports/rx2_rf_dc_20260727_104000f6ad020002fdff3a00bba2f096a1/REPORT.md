# RX2 RF-DC failure and recovery evidence

- Pluto serial: `104000f6ad020002fdff3a00bba2f096a1`
- LO: `5866000000` Hz
- RX2 gains: `[45, 48, 50, 51, 52, 55, 60, 62]` dB
- RF-only initialization duration: `80.05` ms
- Post-recovery valid TX-on frames: `24/24`

## Result

Before recovery, the failed RX2 gains were `[48, 50, 51, 52, 55, 60, 62]` dB. The symptom was present in fresh TX2-off contexts, so TX2 transmission was not required for the observed failure. The Linux driver's RF-DC initialization left failed gains `[]` and the matched post-recovery capture passed the declared recovery condition.

| RX2 gain dB | pre TX-off DC dBFS | pre TX-off max clip | post TX-off DC dBFS | post TX-off max clip | post valid TX-on |
|---:|---:|---:|---:|---:|---:|
| 45 | -79.48 | 0.00% | -77.48 | 0.00% | 3 |
| 48 | -9.11 | 2.39% | -75.86 | 0.00% | 3 |
| 50 | -4.94 | 10.63% | -70.67 | 0.00% | 3 |
| 51 | -2.07 | 21.42% | -72.42 | 0.00% | 3 |
| 52 | -8.92 | 2.89% | -73.78 | 0.00% | 3 |
| 55 | -6.92 | 7.18% | -75.22 | 0.00% | 3 |
| 60 | -2.65 | 21.48% | -69.67 | 0.00% | 3 |
| 62 | -3.44 | 18.18% | -67.94 | 0.00% | 3 |

## RF correction words (input A bank)

| RX2 gain dB | I before | I after | Q before | Q after |
|---:|---:|---:|---:|---:|
| 45 | -511 | 11 | 485 | -132 |
| 48 | -511 | 12 | -512 | -133 |
| 50 | -511 | 12 | 474 | -133 |
| 51 | -511 | 13 | -512 | -133 |
| 52 | -511 | 10 | -512 | -114 |
| 55 | -511 | 12 | -512 | -114 |
| 60 | -511 | 16 | 488 | -113 |
| 62 | -511 | 21 | 488 | -112 |

The RF-only operation is not presented as the complete ADI recovery procedure: the Linux `calib_mode=rf_dc_offs` interface does not rerun the separate BB-DC initialization. ADI recommends isolating the input and running both initial calibrations for the complete procedure. See [ADI's AD936x DC-offset issue note](https://ez.analog.com/rf/wide-band-rf-transceivers/design-support/w/documents/10060/ad936x_5f00_dcoffset_5f00_issue).

## Reproducibility and policy

The adjacent `evidence.json` records SHA-256 manifests for every diagnostic file, including every full-IQ `.npy` frame, plus the recovery snapshot hash. The large source evidence remains under `artifacts/` and is not committed.

New calibration runs initialize RF-DC with TX2 stopped before every radio/frequency block, then require a direct-USB tone preflight. A failed initialization, preflight, metadata check, clipping check, or phase-quality check fails closed. The earlier paused exhaustive scan must not be resumed because it predates this preparation policy.

This result is scoped to the identified radio and test grid. Repeat the same before/recovery/after test on each radio before treating the fleet as characterized.
