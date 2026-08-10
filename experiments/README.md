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
| [e_cal1_rfdc_discriminator](e_cal1_rfdc_discriminator/experiment_readme.md) | Does the RF-DC calibration machinery inject phase on its own, or is the step entirely the LNA/mixer/TIA network? | ✅ arm 1 run 2026-08-07 — entirely the LNA/mixer/TIA network; RF-DC is +0.069° ± 0.077 (95% CI [−0.168, +0.392]) vs the 2.664° mixer step. Arm 2 also run 2026-08-07 with the tracking loop pinned off: also null (+0.019° ± 0.082) |
| [e_cal5_positive_control](e_cal5_positive_control/experiment_readme.md) | Both E-CAL1 arms are null — but would this chain have seen an RF-DC effect if there were one? | ✅ run 2026-08-07 — **yes**: a known mixer step measures 7.43° ± 0.10 against a 0.44°/dB floor (16.9×), so an H₁-sized 2.664° effect would have shown at >30σ |
| [e_gsp7_conditioned_comb](e_gsp7_conditioned_comb/experiment_readme.md) | Does a 10-LO comb *chosen by conditioning* calibrate, where E-CAL3's uniform 10-LO comb failed at 11.61°? | ✅ run 2026-08-07 — yes, but only if the delays are ALSO frozen: chosen-10 frozen 4.95° vs the E-CAL3 comb's 23.50°, same session, same coverage |
| [e_lnk1_transport_sample_rate](e_lnk1_transport_sample_rate/experiment_readme.md) | How do direct-USB, libiio/USB, IIO-over-RNDIS and IIO-over-Ethernet compare across the sample-rate range on `.18` — in throughput, integrity, metadata and measured phase? | ✅ throughput run 2026-08-07 — Ethernet holds up **exactly** as well as USB and no better: both wall at ~23 MB/s (2.9 MS/s), so the limit is the radio, not the link. Ethernet's apparent 28% lead at production buffer size is a buffer-tuning artifact. Phase/CPU metrics not yet run |
| [e_agc1_pin_and_detector_bringup](e_agc1_pin_and_detector_bringup/experiment_readme.md) | Do the AD9361 gain-control pins and detector outputs behave the way the tandem-AGC design contract assumes? The mapping is currently verified only by joining a schematic to a constraints file | designed 2026-08-10, not yet run — one radio, stock RC17 firmware, userspace only: no FPGA change, no bitstream, no RAM boot |
| [e_gsc6_equal_gain_diagonal](e_gsc6_equal_gain_diagonal/experiment_readme.md) | Does the equal-gain anchor move with gain index? Every campaign to date measures exactly one equal-gain cell — the anchor itself — but the planned tandem-AGC firmware operates entirely on that diagonal | designed 2026-08-10, not yet run — config change only, rides along on an ordinary additive-cross run as held-out pairs |
| [e_gsp2_pad_sweep](e_gsp2_pad_sweep/experiment_readme.md) | Is the frequency ripple actually a harness reflection? | needs parts |
| [e_inf1_filter_sweep](e_inf1_filter_sweep/experiment_readme.md) | How do the current models plus the EKF/PF trackers perform on the 2026 rover corpus versus the frozen val set, and does the reported confidence mean anything? | designed 2026-08-08, tooling landed, not yet run; blocked on the d/λ = 0.904 empirical-table gap |

Designs and decision rules for the wider programme are in
[`docs/future_experiments.md`](../docs/future_experiments.md).
