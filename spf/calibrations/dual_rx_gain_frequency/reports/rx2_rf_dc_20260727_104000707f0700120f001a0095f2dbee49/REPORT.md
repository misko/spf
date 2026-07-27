# RX2 RF-DC failure and recovery evidence

- Pluto serial: `104000707f0700120f001a0095f2dbee49`
- LO: `5866000000` Hz
- RX2 gains: `[45, 48, 50, 51, 52]` dB
- RF-only initialization duration: `78.94` ms
- Post-recovery valid TX-on frames: `15/15`

## Result

Before recovery, the failed RX2 gains were `[48, 50, 51, 52]` dB. The symptom was present in fresh TX2-off contexts, so TX2 transmission was not required for the observed failure. The Linux driver's RF-DC initialization left failed gains `[]` and the matched post-recovery capture passed the declared recovery condition.

| RX2 gain dB | pre TX-off DC dBFS | pre TX-off max clip | post TX-off DC dBFS | post TX-off max clip | post valid TX-on |
|---:|---:|---:|---:|---:|---:|
| 45 | -81.32 | 0.00% | -81.63 | 0.00% | 3 |
| 48 | -11.76 | 0.40% | -73.59 | 0.00% | 3 |
| 50 | -4.64 | 16.58% | -71.40 | 0.00% | 3 |
| 51 | -3.74 | 17.71% | -71.78 | 0.00% | 3 |
| 52 | -69.29 | 0.00% | -74.28 | 0.00% | 3 |

## RF correction words (input A bank)

| RX2 gain dB | I before | I after | Q before | Q after |
|---:|---:|---:|---:|---:|
| 45 | 492 | -144 | -512 | -24 |
| 48 | -512 | -144 | 461 | -25 |
| 50 | 472 | -143 | -512 | -25 |
| 51 | -512 | -143 | 423 | -26 |
| 52 | -512 | -121 | -512 | -26 |

The RF-only operation is not presented as the complete ADI recovery procedure: the Linux `calib_mode=rf_dc_offs` interface does not rerun the separate BB-DC initialization. ADI recommends isolating the input and running both initial calibrations for the complete procedure. See [ADI's AD936x DC-offset issue note](https://ez.analog.com/rf/wide-band-rf-transceivers/design-support/w/documents/10060/ad936x_5f00_dcoffset_5f00_issue).

## Reproducibility and policy

The adjacent `evidence.json` records SHA-256 manifests for every diagnostic file, including every full-IQ `.npy` frame, plus the recovery snapshot hash. The large source evidence remains under `artifacts/` and is not committed.

New calibration runs initialize RF-DC with TX2 stopped before every radio/frequency block, then require a direct-USB tone preflight. A failed initialization, preflight, metadata check, clipping check, or phase-quality check fails closed. The earlier paused exhaustive scan must not be resumed because it predates this preparation policy.

This result is scoped to the identified radio and test grid. Repeat the same before/recovery/after test on each radio before treating the fleet as characterized.
