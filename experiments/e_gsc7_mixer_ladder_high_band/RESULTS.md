# E-GSC7 results — standard libiio over USB and IP

**Run 2026-08-12** on both v5 radios. Two independent 1,020-frame passes were
captured through libiio `MetadataBuffer`: first USB, then IP. Both transports
have 1,020/1,020 complete and quality-valid frames, no clipping, valid endpoint
gain/RSSI metadata, and request-driven capture/sample sequences.

## Answer

**The smooth per-index mixer-ladder hypothesis is falsified.** The clean radio
(R18) reproduces E-GSC6's aggregate 52→62 effect, but most individual 1 dB
steps are below the preregistered resolution threshold and the curve is not
portable at 5300 MHz. Do not ship mixer 6…14 as a universal high-band table from
this experiment.

| Hypothesis | Outcome |
|---|---|
| H1 — every adjacent step >1.104° | **Fail.** Resolved: R18 USB 1/10, IP 3/10; R17 USB/IP 2/10. |
| H2 — 52→62 sum = 5.420° ±1° | **R18 passes both:** USB 5.919°, IP 5.405°. **R17 fails both:** 7.026°, 8.004°. |
| H3 — no step dominates by >3× median | Mixed: R18 USB 2.52×, IP 2.71×; R17 IP 2.98× pass, R17 USB 3.04× marginally fails. |
| H4 — represented mixer-state coverage 76%→100% | **Structural pass:** fits now contain mixer 5…15. Deployment is withheld because H1/H5 fail. |
| H5 — 5766 MHz curve transfers within high band | **Fail.** R18's 5300 MHz curve differs by 9.06° RMS (USB), 8.88° (IP); the same failure repeats across transports. |

## Mandatory H2 check, before refitting

The preregistration says “nine steps,” but inclusive 52→62 contains **ten**
adjacent 1 dB transitions (mixer 5→15). All ten are graded; their telescoping
sum is exactly the 52→62 effect that E-GSC6 measured.

| Radio | USB | IP | E-GSC6 target | Verdict |
|---|---:|---:|---:|---|
| R18, untouched control | **5.919°** | **5.405°** | 5.420° ±1° | pass / pass |
| R17, connector-damaged | 7.026° | 8.004° | 5.420° ±1° | fail / fail |

The clean unit therefore supports additivity only at the aggregate 52→62
scale. It does not support the stronger claim that every intermediate index has
a separately resolvable phase step.

## Capture gates

| Gate | USB | IP |
|---|---:|---:|
| complete / quality-valid | 1,020 / 1,020 | 1,020 / 1,020 |
| passing cells | 340 / 340 | 340 / 340 |
| clipping fraction, maximum | 0 | 0 |
| gain endpoint match | 100% | 100% |
| gain observations per frame | 1 | 1 |
| RF words 52…62 | pass both radios | pass both radios |
| worst anchor drift, R17 | 0.762° | 1.892° |
| worst anchor drift, R18 | **4.691° fail** at 5300 MHz | 3.598° pass |

The USB capture therefore misses the preregistered `<4°` anchor gate on one
R18/5300 epoch pair, even though its per-frame quality gates pass. That failure
is not hidden: it reinforces H5's 5300 MHz portability failure. At the required
5766 MHz, worst anchor drift is only 0.385° over USB and 1.203° over IP.

The preregistered `railed_fraction` gate is ill-posed for this manual-gain
experiment: by definition it reports 100% when 62 dB is deliberately commanded.
The actual saturation evidence is clean: zero clipped samples in both passes,
and the channel commanded to 62 dB remains between −32.38 and −13.68 dBFS.

The live gain-table readback is byte-identical on both radios (SHA-256
`90d34d61…a1143`). Across 52…62, LNA=3, TIA=1 and LPF=24 are frozen while mixer
advances exactly 5…15. The confound-free premise therefore holds.

## USB versus IP

The transport repeat is strong evidence that the unexpected scientific result
is in the RF/session behavior, not a transport artifact.

| Radio | USB/IP curve MAE | RMS | maximum | adjacent-step MAE |
|---|---:|---:|---:|---:|
| R17 | 0.237° | 0.366° | 1.356° | 0.345° |
| R18 | 0.306° | 0.515° | 2.424° | 0.464° |

R18's 5300 MHz anomaly appears independently in USB and IP. R17's much larger
5300/5500 behavior likewise repeats and remains consistent with E-GSC6's warning
that its damaged harness must not define absolute coefficients.

## Artifacts

- USB validated data: `artifacts/dual_rx_gain_frequency/e_gsc7_iio_usb_20260812_v2/`
- IP validated data: `artifacts/dual_rx_gain_frequency/e_gsc7_iio_ip_20260812_v1/`
- Machine-readable result: `spf/calibrations/dual_rx_gain_frequency/reports/e_gsc7_iio_20260812_v1/analysis.json`
- Analysis entry point: [`analyze.py`](analyze.py)

The earlier USB `v1` directory is diagnostic-only: strict validation found that
the calibration writer had omitted optional V7 gain-series/sample-time arrays.
The writer was fixed and the entire USB pass was recaptured as `v2`; no `v1`
measurement is used above.
