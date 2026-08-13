# E-GSC8 results — 5766 MHz serves 5840 MHz on the control radio

**Run date:** 2026-08-13<br>
**Transport:** request-driven standard libiio over USB<br>
**Firmware:** `v0.38-plutoplus-spf-libiio-metadata-v5`, persistent QSPI

## Outcome

The primary session captured and strictly validated **816/816 frames** and
**272/272 cells** across the two radios. H3, the mandatory 5766 MHz same-LO
control, is graded before H1/H2.

| radio | H3: new 5766 sum | H3 gate | H1: 5766→5840 RMS (paired-state 95% CI) | H1 | H2: prior 5766→5300 RMS | H2 |
|---|---:|---|---:|---|---:|---|
| R18, untouched control | 5.269° | pass | **0.451° (0.329–0.554°)** | **pass** | 9.059° | **pass** |
| R17, damaged connector | 6.678° | pass | 2.842° (2.626–3.026°) | fail: upper bound exceeds 3° | 79.842° | pass |

The deployment-relevant answer is therefore yes for the clean R18 control: a
5766 MHz gain-phase curve predicts the rover's 5840 MHz carrier with ample
margin under the preregistered 3° bound. R17's point estimate is also below 3°,
but its paired-state confidence interval narrowly crosses the bound; its known
connector damage remains a reason not to define absolute deployment
coefficients from that unit.

H3 uses GSC7's observed **full USB/IP difference** as the transport-repeatability
tolerance around the USB/IP midpoint. This is the literal full repeatability
tolerance; treating the previous min/max interval itself as the gate would halve
that tolerance. The primary run passes this gate on both radios.

## Independent repeat

A complete second 816-frame session was captured and validated. R17 again
passes H3 and H1 (2.674°, CI 2.462–2.861°). R18's 5766 sum is 4.957°, outside
its H3 acceptance band, so its attractive 0.372° 5840 error is deliberately
marked **not interpretable**. This is why the primary session, not a pooled
average, is the reported experiment result.

## Capture and fit checks

- Primary: each radio passed 408/408 frames, 136/136 cells, and all 44/44
  held-out cells; no quality reasons were recorded.
- Repeat: the same strict validation counts passed on each radio.
- Primary independent-arm held-out RMSE was 1.717° on R17 and 1.919° on R18.
- Comparisons use paired gain states. The CI is a 20,000-resample paired-state
  nonparametric bootstrap, and H1 requires its upper bound to be at most 3°.
- The optional 5700 and 5900 MHz bracket carriers were captured and are retained
  in the machine-readable diagnostic output.

## Artifacts

- Primary raw and fitted data:
  `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc8_iio_usb_20260813_v1/`
- Independent repeat:
  `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc8_iio_usb_20260813_v2/`
- Primary grading:
  `spf/calibrations/dual_rx_gain_frequency/reports/e_gsc8_iio_20260813_v1/analysis.json`
- Repeat grading:
  `spf/calibrations/dual_rx_gain_frequency/reports/e_gsc8_iio_20260813_v2/analysis.json`
- Reproduction entry point: [`analyze.py`](analyze.py)

## Addendum — R17's H1 failure is an offset, not a shape error

Added 2026-08-13 from this experiment's own `analysis.json`, after E-GSC8 was used to revisit
the [L31 refit](../../spf/calibrations/dual_rx_gain_frequency/reports/l31_gsc6_gsc7_union_20260812_v1/REPORT.md)'s
deployment verdict. Decomposing the 11 paired-state differences into a constant term and the
rest:

| radio | raw RMS | mean offset | de-meaned RMS | range |
|---|---:|---:|---:|---:|
| R17 | 2.842° | **−2.819°** | **0.360°** | 1.183° |
| R18 | 0.451° | −0.069° | 0.445° | 1.339° |

R17's curve is displaced almost rigidly between the two LOs; its **shape** transfers slightly
*better* than R18's. The gain-state phase model consumes only differences between gain states,
`D(f, g1, g2) = H(s1) − H(s2)`, so a per-LO constant cancels exactly. On the axis that model
uses, R17's transfer is therefore not a failure.

This does not amend the grading above. H1 is an absolute-curve test with a preregistered 3°
bound and R17's CI does cross it; that grade stands as recorded. The point is narrower: a
consumer of these numbers should apply the absolute grade only to an absolute use, and R17's
result should not be read as evidence that its gain-state curve fails to transfer.

The two offsets are themselves per-LO anchor terms — the same free absolute reference the
gain-phase work has flagged as its top open question, here visible in a bench capture with
every other variable pinned.
