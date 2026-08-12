# E-GSC8 — does a 5766 MHz gain-phase fit serve 5840 MHz?

**Status: DESIGNED, NOT RUN.** Needs two radios on a bench harness.
Small: one added carrier, ~200–600 frames.

## Why this exists — and a correction to E-GSC7

**The 2026 rover corpus has two carriers, not one.**

| carrier | frames | share |
|---|---|---|
| 5766 MHz | 146,208 | **76.2%** |
| **5840 MHz** | **45,636** | **23.8%** |

[E-GSC7's preregistration](../e_gsc7_mixer_ladder_high_band/experiment_readme.md)
asserted that "5766 MHz is the only carrier in the 2026 corpus". **That was
wrong**, and the error propagated into E-GSC7's design: its transferability LOs
were 5000/5300/5500/5900 and **5840 — the one carrier that actually matters — was
omitted**. This experiment closes that hole. The mistake was in the design brief,
not in E-GSC7's execution, which was clean on both transports.

It matters because **E-GSC7's H5 failed**: the 5766 MHz curve did not transfer to
5300 MHz, at **9.06° RMS (USB) / 8.88° (IP)**, reproducing across transports so it
is an RF effect rather than a link artefact. That is 4× L26's entire pooled LOFO
error.

5840 is only **74 MHz** from 5766, against 5300's 466 MHz. But L10 finding 3 puts
the reflection ripple period at **~392.5 MHz**, so 74 MHz is **0.19 of a period** —
roughly 68° of ripple phase. Near enough that transfer is plausible; far enough
that assuming it, after H5 already failed once, is not justified.

**Both carriers sit in the same high band and require the identical mixer set
{4…15}** (verified against the audited table), so this is purely a question about
the frequency-dependent ripple term, not about state coverage.

## Design

Reuse [`e_gsc7_mixer_ladder_high_band.yaml`](../../spf/calibrations/dual_rx_gain_frequency/configs/e_gsc7_mixer_ladder_high_band.yaml)
unchanged except for `frequencies-hz`.

| | |
|---|---|
| carriers | **5766** (repeat, as the same-LO control) and **5840** (the target). Add **5700** and **5900** if cheap — they bracket 5840 and give the ripple two more points over one period |
| gains | E-GSC7's ladder unchanged: 26 (ref), 52…62 |
| transport | one is sufficient; E-GSC7 showed USB vs IP is a null (p = 0.32/0.70) |
| frames | ~200 per carrier per radio |

**The 5766 repeat is not optional.** E-GSC7's own analysis graded cross-LO
transfer against a *same-LO transport repeat*, and that control is what made the
H5 failure interpretable rather than ambiguous. Keep it.

## Pre-registered hypotheses

| id | prediction | decision rule |
|---|---|---|
| **H1** | The 5766 MHz per-state curve predicts 5840 MHz to within **3°** RMS — materially better than the 9.06° H5 measured at 5300 MHz | A single high-band fit serves 76.2% + 23.8% = the whole corpus. Refit once, deploy once. |
| **H2** | The 5766→5840 error is **smaller than 5766→5300** on the same radio and gains | Transfer degrades with frequency separation, as the ripple model predicts. Supports a local-in-frequency fit rather than per-LO tables. |
| **H3** | Repeating 5766 reproduces E-GSC7's own 52→62 aggregate to within its transport repeatability (R18: 5.919 USB / 5.405 IP) | The harness has not drifted between sessions and the comparison is sound. **If this fails, nothing else in the run is interpretable.** |

**Falsifiers.** H1 fails at >3° RMS — in which case **5840 needs its own fit**, and
the deployable model becomes per-LO rather than universal-in-band. That is a
worse outcome but a decisive one, and it is the answer this experiment exists to
get. H3 failing invalidates the run.

## Acceptance gates

| artifact | gate |
|---|---|
| capture | ≥95% quality-valid frames per (radio, LO, gain-pair) |
| railing | `railed_fraction` within its normal band at every gain to 62 dB — pad the **TX** down, never the RX gain |
| control | H3 evaluated and reported **before** H1 or H2 |
| statistics | H1/H2 by **paired** comparison per gain state, not a difference of means |
| `RESULTS.md` | states H1–H3 with numbers, including any falsified |

## Risks

| risk | mitigation |
|---|---|
| Session-to-session harness drift confounds the cross-session comparison | That is exactly what H3 controls for, and why the 5766 repeat is mandatory. |
| Reading a null as "transfer works" | H1 needs a *measured* bound under 3°, not merely absence of evidence. Report the RMS with its CI. |
| Two units is not a distribution | Report per radio. Prefer the untouched control (R18); E-GSC7 found R17's connector damage inflated its numbers and failed H2 on both transports. |

## Inputs

| | |
|---|---|
| config to fork | [`e_gsc7_mixer_ladder_high_band.yaml`](../../spf/calibrations/dual_rx_gain_frequency/configs/e_gsc7_mixer_ladder_high_band.yaml) |
| prior run to compare against | [`e_gsc7_iio_20260812_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_gsc7_iio_20260812_v1/) |
| why it matters | [`rover_applicability_ladder_20260812_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/rover_applicability_ladder_20260812_v1/) |

## This is second in line

The **L31-shaped refit over the E-GSC6 ∪ E-GSC7 union comes first** — it needs no
bench time and takes RF-state coverage from 0.51% to 100%. If that refit cannot
reproduce E-GSC6's own held-out metrics, the problem is in the fitting path and
no capture at any carrier will help.
