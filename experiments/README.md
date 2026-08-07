# Experiments

One folder per experiment. Each folder **must** contain an `experiment_readme.md`
covering:

| Section | What it must state |
|---|---|
| Purpose | why the experiment exists, and what is currently unknown |
| Hypothesis | the falsifiable prediction, written before the data exists |
| Approach | the measurement, the controls, and the analysis that follows |
| Hardware setup | radio count, per-radio configuration, every adapter and passive part, **and a schematic of the physical setup** |
| Software setup | firmware, config file, exact commands, environment |
| Outputs | every artifact the run must produce, where it goes, and its acceptance gate |
| Decision rule | what result changes what belief — pre-registered |
| Risks | what could invalidate the run, and the check that catches it |

Code is optional. An experiment may ship its own analysis scripts, or leave that
to whoever executes it. The **outputs** are not optional — they must be defined
before the run so the result is interpretable by someone who was not there.

Raw captures never live here. They go to `artifacts/dual_rx_gain_frequency/<run>`
(gitignored) or a campaign root, and the committed analysis goes to
`spf/calibrations/dual_rx_gain_frequency/reports/<name>/` per that directory's
append-only convention.

## Index

| Experiment | Question | Status |
|---|---|---|
| [e_cal1_rfdc_discriminator](e_cal1_rfdc_discriminator/experiment_readme.md) | Does the RF-DC calibration machinery inject phase on its own, or is the step entirely the LNA/mixer/TIA network? | ✅ arm 1 run 2026-08-07 — entirely the LNA/mixer/TIA network; RF-DC is +0.069° ± 0.077 (95% CI [−0.168, +0.392]) vs the 2.664° mixer step. Arm 2 unblocked in code, ready but unrun and low priority |
| [e_gsp2_pad_sweep](e_gsp2_pad_sweep/experiment_readme.md) | Is the frequency ripple actually a harness reflection? | needs parts |

Designs and decision rules for the wider programme are in
[`docs/future_experiments.md`](../docs/future_experiments.md).
