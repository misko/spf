# External wall-array validation at 2467.100 MHz

Date: 2026-07-29

## Verdict

An independent historical wall-array corpus confirms the `.17` radio model's
gain-dependent term. The absolute bench intercept does not transfer unchanged
to that installation.

For `.17`, applying the documented correction

```text
corrected = wrap(measured_RX1_minus_RX2 - predicted_radio_offset)
```

reduces within-capture circular dispersion in 96 of 102 captures. Coverage is
100 percent, so this result has no gain-grid selection confound. The remaining
post-model bias is a highly reproducible installation-specific constant.

This supports a two-layer operational correction:

```text
corrected = wrap(
    measured_RX1_minus_RX2
    - radio_model(serial, exact_frequency, gain_rx1, gain_rx2)
    - deployment_anchor(installation, serial, exact_frequency)
)
```

At this exact setup and frequency, the measured `.17` deployment anchor after
the radio correction is `+95.4802°`.

## Independent input

The model consumer supplied:

| File | Bytes | SHA-256 |
|---|---:|---|
| `VALIDATION_RESULT_2467.md` | 6,407 | `5d2ae6492e4b01631ec3bdd555ad96153017570c90ff3fe56d9c9bac3c39c674` |
| `spf_2467p1_r17_snapshots.csv.gz` | 66,313,737 | `cefa15fea7b2f8ff86b83aa0c387571f283d5a1b1f3561775a2c2e51acb38c2c` |
| `spf_2467p1_r17_snapshots.csv.gz.meta.json` | 886 | `7be67adaf2c43ac7c1eda9a59cb14c2e044c543a625e68bcbb845a1cb917c455` |

The compressed export contains 1,730,000 rows: 865,000 snapshots for each of
two wall-array receivers across 102 captures. Source Zarrs were not modified.

Historical files do not contain Pluto serial numbers. The mapping from
`192.168.1.17` to `104000bac4950008230026001b440a003a` is an operator-supplied
assertion. The gain-term result supports that mapping indirectly but cannot
prove that no board was swapped during the capture period.

## Recomputed `.17` result

The committed audit code independently streamed the supplied export, loaded
the model from commit `c389d88`, and recalculated every prediction.

Integrity results:

| Check | Result |
|---|---:|
| Selected `.17` rows | 865,000 |
| Captures | 102 |
| Exact LO | 2,467,100,000 Hz |
| Supported rows | 865,000 |
| Overall/minimum per-capture coverage | 100% / 100% |
| Geometry rows independently re-derived | 854,989 |
| Maximum re-derived angle error | below `5.1e-9` rad |
| Maximum re-derived phase error | below `8.1e-9` rad |

The geometry check uses:

```text
theta = wrap(
    atan2(tx_x - rx_x, tx_y - rx_y)
    - pi * (rx_theta_in_pis + rx_heading_in_pis)
)

phase_ground_truth = wrap(-sin(theta) * d_over_lambda * 2*pi)
```

Model results:

| Metric | Raw | Subtract model | Add model |
|---|---:|---:|---:|
| Across-capture bias | +90.8696° | +95.4802° | +86.2568° |
| Bias resultant | 0.983727 | 0.984537 | 0.982437 |
| Median capture circstd | 0.461062 rad | 0.451813 rad | 0.474824 rad |
| Captures with lower circstd | — | 96/102 | 3/102 |
| Circstd sign-test z | — | +8.91 | -9.51 |
| Captures with lower absolute bias | — | 0/102 | 101/102 |

The wrong-sign addition makes dispersion materially worse. Absolute bias is
not a valid sign test here because the much larger installation constant
dominates it. The dispersion dose-response selects subtraction unambiguously
and agrees with the SPF convention: signal matrix row 0 is RX1, row 1 is RX2,
and `get_avg_phase_fast2()` computes RX1 minus RX2.

The supplied dose-response also reproduces exactly:

| RX1 gain standard deviation | Captures | Median Δcircstd | Improved |
|---|---:|---:|---:|
| 0.01–3 dB | 25 | -0.015132 rad | 25/25 |
| 3–7 dB | 49 | -0.007511 rad | 45/49 |
| 7–99 dB | 28 | -0.006825 rad | 26/28 |

Because the model intercept is constant within each exact frequency,
dispersion improvement can only come from its gain-dependent term.

## What the intercept represents

The bench experiment did not measure the bare Pluto in isolation. Its path
was:

```text
TX2 -> 30 dB attenuator -> two-way splitter -> matched branch cables -> RX1/RX2
```

The differential phase therefore includes the radio and any mismatch between
the two bench fixture branches. It excludes the wall installation's antenna
feed cables, connectors, antennas, and mounting paths. Transfer of the gain
shape is expected; transfer of the complete intercept is not guaranteed.

The approximately `+4.61°` shift from raw to corrected wall bias is expected:
the average predicted `.17` radio offset over the realized AGC states is
negative, so subtracting it moves the residual positive. It is not evidence of
a second sign error.

The raw wall bias changes from approximately `+8.20°` at 2412 MHz to
`+90.87°` at 2467.1 MHz. The smallest unwrapped interpretation is a 4.17 ns
differential delay, or 1.25 m free-space equivalent. For coax with velocity
factor 0.66–0.85, that corresponds to roughly 0.82–1.06 m of physical length
difference.

This is plausible but not a root-cause proof. Two frequencies leave a
`360°` phase-wrap ambiguity; each additional wrap adds about 18.15 ns. A
multi-frequency installation sweep or direct cable measurement is needed
before assigning the offset solely to cable length.

## Second receiver (`.18`) diagnostic

The export also contains 865,000 r1 rows at historical IP `192.168.1.18`.
SPF's operator inventory maps that IP to serial
`1040007c4a94000211000b009186843ef2`, although the historical Zarrs do not
record that serial.

Applying `.18`'s exact-frequency model gives:

| Metric | Raw | Subtract model | Add model |
|---|---:|---:|---:|
| Across-capture bias | -13.5413° | -11.9034° | -15.1804° |
| Bias resultant | 0.997157 | 0.997066 | 0.997179 |
| Median capture circstd | 0.480793 rad | 0.489753 rad | 0.473299 rad |
| Captures with lower circstd | — | 14/102 | 79/102 |
| Captures with lower absolute bias | — | 100/102 | 1/102 |

This does not reproduce `.17`'s sign-clean gain validation. The most likely
classes of explanation are:

1. historical r1 RX1/RX2 gain fields or IQ channels are ordered oppositely;
2. the historical `192.168.1.18` board is not the currently inventoried
   serial;
3. the historical r1 capture/configuration path had a polarity or channel
   mapping difference.

Do not promote an `.18` deployment correction from these rows until channel
ordering and historical identity are independently resolved. The `.18`
result is a useful diagnostic, not a failed replication of the `.17` result.

## Reproduction

The audit implementation is
`spf/calibrations/models/external_wall_validation.py`. From the repository
root:

```bash
python -m spf.calibrations.models.external_wall_validation \
  --csv-gz /mnt/qnap01/mouse9911/share/spf_2467p1_r17_snapshots.csv.gz \
  --meta /mnt/qnap01/mouse9911/share/spf_2467p1_r17_snapshots.csv.gz.meta.json \
  --receiver r0 \
  --serial 104000bac4950008230026001b440a003a \
  --frequency-hz 2467100000 \
  --output /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/external_wall_validation_2467p1_20260729/r0_17.json
```

Repeat with `--receiver r1` and serial
`1040007c4a94000211000b009186843ef2` for the diagnostic second-radio result.

Recomputed JSON hashes:

| Result | SHA-256 |
|---|---|
| `.17` / r0 | `eea8ecfeba656c89bde3e5f38557eda8af2e3861c4d792c4545f9304d2aecd7e` |
| asserted `.18` / r1 | `7e312a925205ce704baf08fb417217fc90354176d978f59c947a4feac31cc5c5` |

Large input and output artifacts remain outside Git. This report, the model,
and the streaming audit code are sufficient to reproduce the result when the
hash-pinned handoff is mounted.
