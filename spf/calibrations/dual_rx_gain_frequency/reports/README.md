# Dual-RX gain/frequency reports

This directory contains the small, reviewable outputs from the reproducible
calibration and hardware-diagnostic commands. Large V7/LMDB datasets and
full-IQ diagnostic frames remain under the gitignored `artifacts/` tree. Each
committed report records SHA-256 hashes of the exact inputs it used.

## Current report set

| Report | Scope | Status |
|---|---|---|
| [Replacement-radio cross-band pilot](pilot_cross_band_20260728_new_radios_v2/README.md) | Two new radios, 12 frequencies, complete 3×3 ordered gain grid, three randomized epochs | 648/648 frames complete; all 72 observed-tone preflights passed; extreme 63 dB mismatch explicitly unsupported |
| [Replacement Radio `…beca79`](pilot_cross_band_20260728_new_radios_v2/104473b80a16000de6ff2000f8a6beca79/REPORT.md) | Per-frequency data-versus-fit figures and protocol-v2 metadata audit | 221/324 phase-valid; additive held-out MAE 1.45° |
| [Replacement Radio `…5222f`](pilot_cross_band_20260728_new_radios_v2/104000b299050013f4ff0700255e35222f/REPORT.md) | Per-frequency data-versus-fit figures and protocol-v2 metadata audit | 249/324 phase-valid; additive held-out MAE 1.03° |
| [Replacement-radio comparison](pilot_cross_band_20260728_new_radios_v2/comparison/CROSS_RADIO_REPORT.md) | Board-specific frequency baselines and descriptive delay fits | One global path-delay explanation rejected by 14.77° residual MAE |
| [Dense model matrix](survey_cross_band_20260727_v1/MODEL_MATRIX_REPORT.md) | Two radios, 12 frequencies, complete 17×17 ordered gain grid, three randomized epochs | Capture structurally complete; 20,409/20,808 frames quality-valid |
| [Completed cross-band scout](FREQUENCY_SCOUT_20260727.md) | Two radios, 47 frequencies, three gains per receiver, three randomized epochs | Structurally complete; weak cells remain explicitly unsupported |
| [Plotted Radio A model](frequency_scout_cross_band_20260727_v1/104000f6ad020002fdff3a00bba2f096a1/REPORT.md) | 47 per-frequency data-versus-fit figures plus overview plots | Complete three-epoch model |
| [Plotted Radio B model](frequency_scout_cross_band_20260727_v1/104000707f0700120f001a0095f2dbee49/REPORT.md) | 47 per-frequency data-versus-fit figures plus overview plots | Complete three-epoch model |
| [Phase model comparison](coarse_5ghz_20260727_dds_v1/REPORT.md) | Two radios, complete epoch-0 blocks at 5804 and 5866 MHz | Preliminary model-shape evidence only |
| [RF-DC recovery: …00bba2f096a1](rx2_rf_dc_20260727_104000f6ad020002fdff3a00bba2f096a1/REPORT.md) | 5866 MHz, RX2 45–62 dB | Recovery passed, 24/24 post-recovery TX-on frames valid |
| [RF-DC recovery: …0095f2dbee49](rx2_rf_dc_20260727_104000707f0700120f001a0095f2dbee49/REPORT.md) | 5866 MHz, RX2 45–52 dB | Recovery passed, 15/15 post-recovery TX-on frames valid |

## Combined findings

The completed dense run contains 20,808/20,808 scheduled frames. Of those,
20,409 pass the stored-IQ quality gates and enter the model matrix. The best
known-cell correction is a radio-specific, per-frequency additive lookup:

```text
phase = intercept[radio, frequency]
      + RX1_effect[radio, frequency, gain1]
      + RX2_effect[radio, frequency, gain2]
```

Its leave-one-epoch-out MAE is 0.830° on
`…0095f2dbee49` and 0.917° on `…00bba2f096a1`, with an aggregate p95 of
3.053°. A full frequency-by-RX1-by-RX2 cell table uses 6,936 parameters and is
slightly worse at 0.912° MAE, so the data do not justify a gain-pair
interaction table. Frequency-independent gain lookup and linear-gain models
are inadequate at 20.4–20.9° MAE.

A gain-dependent differential-delay model improves the unseen-frequency
baseline from 22.7° to 10.9° MAE, but remains far worse than explicit
per-frequency calibration. Its reference-gain effective delays are about
23.3 ps and 36.6 ps for the two radios (7.0 mm and 11.0 mm free-space
equivalents). These are descriptive RX1−RX2 delays, not literal PCB trace
lengths. Path imbalance explains a broad trend but not the band/retune and
gain-frequency structure.

A strict universal per-frequency additive lookup gives 5.30° known-cell MAE.
When trained on one radio and applied unchanged to the other, it gives 10.51°
MAE. Radio identity and serial-specific calibration therefore remain material.
The replacement-radio pilot independently reaches the same conclusion: its
two radios have 1.03–1.45° serial-specific additive-model MAE, while their
cross-radio frequency-baseline difference cannot be represented by one path
delay without 14.77° residual MAE.

Both radios also exhibited a severe RX2 RF-DC correction failure at high gain.
It was present in fresh TX2-off captures, so current TX2 transmission was not
required for the symptom. Driver-supported RF-DC initialization removed the
observed stuck correction words, DC rail condition, and clipping on both
tested radios.

This changes the status of the earlier partial V7 run: it remains useful for
model development but is superseded by the complete post-recovery dense run.
The dense coefficients are high-quality candidates on their exact measured
grid; reboot, temperature, and deployment-session anchor validation remain
before they should be treated as stable production calibration.

## Correction recommendation

For a previously calibrated (“seen”) radio:

1. Identify it by Pluto serial and require the exact calibrated LO and ordered
   `(RX1 gain, RX2 gain)` pair.
2. Require valid direct-USB endpoint metadata, equal endpoints, adequate tone
   or signal quality, no clipping, and no gain-event warning.
3. Apply
   `wrap(measured_angle_RX1_minus_RX2 - predicted_phase_offset)`.
4. Use the radio-specific, per-frequency additive lookup; do not substitute a
   frequency-independent gain curve or the larger gain-pair cell table.
5. At every boot/session, measure distributed equal-gain anchors across gain
   stages. Reject the stored calibration if anchor residuals exceed the
   threshold established by the clean repeated dataset.
6. Treat a materially different temperature as unvalidated until the planned
   temperature/reboot repeats quantify it.

For a new (“unseen”) radio:

1. Never copy another radio’s absolute phase correction without an anchor. A
   strict universal dense lookup leaves 10.51° leave-one-radio-out MAE.
2. A transferred shape plus distributed equal-gain anchors remains a useful
   temporary lower-confidence estimate, but it is not validated by the strict
   universal result as a precision correction.
3. Collect a full serial-specific calibration when phase precision matters.
   Until then, mark the estimate as transferred/uncalibrated and fail closed
   outside exact measured frequency and gain support.

## Next acceptance gate

The clean V7 dense capture and leave-one-epoch-out model gate are complete.
The remaining acceptance work is to repeat distributed anchor cells across
radio reboot/RAM firmware reload and controlled temperature states, establish
session rejection thresholds, and validate the correction in the real
receive/beamforming path. Only exact serial/frequency/ordered-gain cells that
also pass those session checks should be promoted into production
calibration files.
