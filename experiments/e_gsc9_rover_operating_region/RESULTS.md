# E-GSC9 results

**Status:** IN PROGRESS. Session A and same-session controls were captured on
2026-08-13. Session B and session C remain outstanding.

**Radios:** R17 `104000bac4950008230026001b440a003a` and R18
`1040007c4a94000211000b009186843ef2`, both persistently booted from
`v0.38-plutoplus-spf-libiio-metadata-v5` and captured with request-driven
IIO-over-USB.

**Raw data:**
`/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc9_*_20260813_v1/`.
The local staging copies remain in `/home/pi/gsc9_staging/`. All 14 radio LMDB
stores and all non-LMDB sidecars were compared after the copy; every key and
value matched.

## Captures and validation

The measured level ladder forced the preregistered fallback to fixed TX
attenuation -35 dB and gains 26..62. This preserves 99.9829% of the 5766 MHz
rover frames and 100.0000% of the 5840 MHz frames.

| Capture | Result |
|---|---|
| TX ladders -23/-29/-35 dB | Complete, 72 frames/radio/leg |
| Session A, 37x37 ordered grid, two LOs, five epochs | Complete, 13,690/13,690 frames on each radio |
| Session A strict validation | PASS on both radios; 27,380/27,380 quality-valid frames |
| A2 transition bridge | Complete; R17 PASS, R18 414/420 quality-valid |
| A3 16384 amplitude | PASS on both radios, 168/168 frames each |
| A3 8192 amplitude | PASS on both radios, 168/168 frames each |

R18's six A2 rejects are all `rx1_tone_snr_low` at 5840 MHz and gains 16..22.
Their coherence is 0.9950..0.9991, within-frame phase standard deviation is
0.38..0.67 degrees, gain metadata is valid, and frame counts are complete. This
is a narrow low-SNR quality limitation, not capture corruption; the evidence is
retained rather than silently replaced.

## Preregistered hypotheses

These are provisional until sessions B and C finish.

| ID | Current result | Evidence |
|---|---|---|
| H1 | **PASS with the preregistered fallback** | The captured 26..62 grid covers 134,351/134,374 (99.9829%) rover frames at 5766 MHz and 43,036/43,036 (100%) at 5840 MHz. |
| H2 | **FALSIFIED on R17 at 5840 MHz** | Rover-cell residual median/P95: R17 0.462/1.451 degrees at 5766 and 1.213/2.419 at 5840; R18 0.139/0.444 and 0.231/0.799. The preregistered median limit is 1.0 degree. |
| H3 | **PASS for the damaged R17 unit** | The 40->41 dB equal-gain transition carries 82.2% and 75.3% of the sum of measured absolute 1 dB steps at 5766 and 5840 MHz. Its signed steps are -59.49 and -62.49 degrees. |
| H4 | **FALSIFIED as a radio-general claim** | R18 selects anchor 56 at both carriers. R17 selects 33 at 5766 and 38 at 5840 under the preregistered S2/S3 rule, outside the predicted 52..58 interval. |
| H5 | **PASS at the median-effect criterion** | Halving DDS amplitude produces median absolute cell-mean phase shifts of 0.192 degrees on R17 and 0.133 degrees on R18, below 3x each radio's session-A median anchor drift. Measured per-arm level changes are -6.01/-6.02 dB. |
| H6 | **PENDING** | Requires session B after the 12-hour separation and a power cycle. |
| H7 | **PENDING** | Requires the physical no-pad/pads/pads-removed A/B/A sequence. |

The ordinary axis-only additive fit over every held-out grid frame, rather than
only rover-operating cells, gives MAE/P95 1.173/2.747 degrees on R17 and
0.381/0.998 degrees on R18. Reproducible fit artifacts are under
`spf/calibrations/dual_rx_gain_frequency/reports/e_gsc9_iio_20260813_v1/`.

## Acceptance gates

| Gate | Status | Evidence or remaining action |
|---|---|---|
| G1 | PASS against frozen fallback | Five epochs x 1,369 cells x two carriers are present per radio; all cells have n=5. |
| G2 | PASS | Session-A tone range is -58.67..-15.77 dBFS and maximum clipping fraction is zero. |
| G3 | PASS for stored-frame evidence | Gain endpoints are equal on 27,380/27,380 frames; the runner commands manual gain and strict validation confirms requested/start/end agreement. |
| G4 | PASS | A passive post-session-A readback is stored beside the QNAP raw data. Both radios have identical 77-row tables in all three bands; high-table SHA-256 is `90d34d61e8612277529dccfc3323f6c684c2bc36b7670dff078e009eb84a1143`. |
| G5 | ACTIVE REPORTING RULE | Effects are compared with same-run median anchor drift before being stated as resolved. |
| G6 | ACTIVE REPORTING RULE | Unresolved effects will be reported as bounds. |
| G7 | PENDING | The required same-LO control must be reported before a cross-carrier claim. |
| G8 | **FAIL** | `/run/spf/direct_usb_ready.json` was created at 15:49:52, before session A began at 19:51. Per-frame firmware/metadata validation still passed, but the literal freshness gate did not. |
| G9 | **FAIL on R17** | Worst pairwise equal-gain across-epoch drift is 4.468 degrees on R17 and 2.476 degrees on R18; the gate is strictly below 4 degrees. |
| G10 | PENDING | This report must be finalized after H6 and H7. |

## Remaining execution

Session B has been conformed, before capture, to the measured session-A fallback:
gains 26..62 and fixed TX attenuation -35 dB. It contains 273 cells per LO and
1,638 frames/radio. It must begin no earlier than 2026-08-14 08:57 BST, after a
real power cycle of both radios, without disturbing an RF connector.

Session C still requires operator handling: no pads, insert 10 dB pads on both
arms, then remove them and repeat. The final analysis must retain the failed H2,
G8, and G9 outcomes rather than recapturing until they disappear.

The live table artifact is
`e_gsc9_session_a_20260813_v1/gain_table_audit_post_session_a.json` under the
QNAP raw-data root. It records the v5 firmware identity, every decoded table
row, and every expected-value check for both serials.
