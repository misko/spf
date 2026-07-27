# Dual-RX gain/frequency reports

This directory contains the small, reviewable outputs from the reproducible
calibration and hardware-diagnostic commands. Large V7/LMDB datasets and
full-IQ diagnostic frames remain under the gitignored `artifacts/` tree. Each
committed report records SHA-256 hashes of the exact inputs it used.

## Current report set

| Report | Scope | Status |
|---|---|---|
| [Completed cross-band scout](FREQUENCY_SCOUT_20260727.md) | Two radios, 47 frequencies, three gains per receiver, three randomized epochs | Structurally complete; weak cells remain explicitly unsupported |
| [Plotted Radio A model](frequency_scout_cross_band_20260727_v1/104000f6ad020002fdff3a00bba2f096a1/REPORT.md) | 47 per-frequency data-versus-fit figures plus overview plots | Complete three-epoch model |
| [Plotted Radio B model](frequency_scout_cross_band_20260727_v1/104000707f0700120f001a0095f2dbee49/REPORT.md) | 47 per-frequency data-versus-fit figures plus overview plots | Complete three-epoch model |
| [Phase model comparison](coarse_5ghz_20260727_dds_v1/REPORT.md) | Two radios, complete epoch-0 blocks at 5804 and 5866 MHz | Preliminary model-shape evidence only |
| [RF-DC recovery: …00bba2f096a1](rx2_rf_dc_20260727_104000f6ad020002fdff3a00bba2f096a1/REPORT.md) | 5866 MHz, RX2 45–62 dB | Recovery passed, 24/24 post-recovery TX-on frames valid |
| [RF-DC recovery: …0095f2dbee49](rx2_rf_dc_20260727_104000707f0700120f001a0095f2dbee49/REPORT.md) | 5866 MHz, RX2 45–52 dB | Recovery passed, 15/15 post-recovery TX-on frames valid |

## Combined findings

The preliminary phase data strongly rejects a single constant or a simple
linear gain-difference correction. Their held-out MAE is roughly 9–11°. A
15-parameter ordered stage-boundary model reduces held-out cell MAE to
1.43–1.58° and held-out quadrant MAE to 1.62–1.85°. A 145-parameter ordered
categorical model is slightly better on held-out cells (1.16–1.49°) but not
consistently better on the harder quadrant test. The compact stage model is
therefore the parsimonious explanation; the exact ordered-pair table remains
the conservative operational representation once repeatability is established.

Both radios also exhibited a severe RX2 RF-DC correction failure at high gain.
It was present in fresh TX2-off captures, so current TX2 transmission was not
required for the symptom. Driver-supported RF-DC initialization removed the
observed stuck correction words, DC rail condition, and clipping on both
tested radios.

This changes the status of the earlier partial V7 run: it is useful for
developing and comparing model forms, but it is superseded as a source of
production calibration coefficients. It must not be resumed because it
predates the RF-DC-before-frequency-block preparation policy. No calibration
in this report set is currently deployable.

## Correction recommendation

For a previously calibrated (“seen”) radio:

1. Identify it by Pluto serial and require the exact calibrated LO and ordered
   `(RX1 gain, RX2 gain)` pair.
2. Require valid direct-USB endpoint metadata, equal endpoints, adequate tone
   or signal quality, no clipping, and no gain-event warning.
3. Apply
   `wrap(measured_angle_RX1_minus_RX2 - predicted_phase_offset)`.
4. At every boot/session, measure distributed equal-gain anchors across gain
   stages. Reject the stored calibration if anchor residuals exceed the
   threshold established by the clean repeated dataset.
5. Treat a materially different temperature as unvalidated until the planned
   temperature/reboot repeats quantify it.

For a new (“unseen”) radio:

1. Never copy another radio’s absolute phase correction without an anchor.
   The measured unanchored cross-radio MAE is 17.7–32.2°.
2. A transferred stage shape plus five distributed equal-gain anchors is a
   useful temporary lower-confidence estimate; the current two-radio evidence
   leaves about 3.2–4.0° MAE.
3. Collect a full serial-specific calibration when phase precision matters.
   Until then, mark the estimate as transferred/uncalibrated and fail closed
   outside exact measured frequency and gain support.

## Next acceptance gate

A new, clean V7 run must complete all three independently randomized epochs at
all four coarse frequencies for both radios. Every scheduled frame must pass
structural validation or retain an explicit quality failure; scalar metrics
must be recomputed from stored IQ; and the final analysis must include
leave-one-epoch-out, reboot, anchor, and temperature checks. Only exact
serial/frequency/ordered-gain cells meeting the repeatability policy may be
promoted into production calibration files.
