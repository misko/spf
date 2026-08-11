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
| [e_lnk1_transport_sample_rate](e_lnk1_transport_sample_rate/experiment_readme.md) | How do direct-USB, libiio/USB, IIO-over-RNDIS and IIO-over-Ethernet compare across the sample-rate range on `.18` — in throughput, integrity, metadata and measured phase? | ✅ throughput run 2026-08-07 — Ethernet holds up **exactly** as well as USB and no better: both wall at ~23 MB/s (2.9 MS/s), so the limit is the radio, not the link. Ethernet's apparent 28% lead at production buffer size is a buffer-tuning artifact. **Metric 5 (H3) run 2026-08-11 for USB vs Ethernet: PASS** — 0.089° between arms against 1.34° fixture repeatability, so Ethernet is not disqualified. RNDIS blocked on the duplicate-IP hazard; direct-usb needs a matched-rate run; metric 4 (CPU) outstanding |
| [e_agc1_pin_and_detector_bringup](e_agc1_pin_and_detector_bringup/experiment_readme.md) | Do the AD9361 gain-control pins and detector outputs behave the way the tandem-AGC design contract assumes? The mapping is currently verified only by joining a schematic to a constraints file | ✅ session 1 run 2026-08-10 on **both radios** (steps 1–4, 6) — **H1 PASS**: pin map is the identity, **40/40 trials**, other channel never moved, radios agree exactly, so RTL is unblocked. **H2 PASS**. **H6: edges are NOT honoured outside RX** — a real change to the contract. Unplanned: arming takes gain ownership from software, silently (rc=0). **H3 PASS** with zero cross-channel leakage; **hold band 22 dB** closes O-2 and the oscillation rule did not fire; `0x114` identified at 0.5 dB/LSB. **H4's latch CONFIRMED** on both radios (survives total signal removal, clears only on a gain change) — O-3 half closed, only the blank duration outstanding. All 8 CTRL_OUT bits characterised across the pair. 3.5 of 4 open items |
| [e_gsc6_equal_gain_diagonal](e_gsc6_equal_gain_diagonal/experiment_readme.md) | Does the equal-gain anchor move with gain index? Every campaign to date measures exactly one equal-gain cell — the anchor itself — but the planned tandem-AGC firmware operates entirely on that diagonal | config written and validated 2026-08-11 (21 gains × 24 LOs × 3 epochs = 8,784 frames, ≈1.75 h), both radios preflight-probed OK — but **run BLOCKED**: `dataset.py` requires a boot-verified firmware attestation, which the volatile RC17 image does not carry and which cannot be hand-written. Needs the `automate` RAM-load flow or a reflash to the pinned image |
| [e_hcp1_cross_arm_coupling](e_hcp1_cross_arm_coupling/experiment_readme.md) | Can the bare-tee harness account for the arm-specific residual `A`, which is the doubt hanging over the whole dual-RX phase programme? | ✅ run 2026-08-11 — **no**: coupling is ≤1.25 dB (median 0.50, n=24) and **frequency-flat** across 12 LOs, while `A` rises ~5× into the high band. Bounds phase coupling at ≲8.9° worst case, ~3° typical. **R18 reproduces R17 exactly**, and coupling is not unit-specific while `A` is — a second argument. E-GSC6 unblocked; E-GSP2 still the definitive A/B but no longer urgent |
| [e_gsp2_pad_sweep](e_gsp2_pad_sweep/experiment_readme.md) | Is the frequency ripple actually a harness reflection? | needs parts |
| [e_inf1_filter_sweep](e_inf1_filter_sweep/experiment_readme.md) | How do the current models plus the EKF/PF trackers perform on the 2026 rover corpus versus the frozen val set, and does the reported confidence mean anything? | designed 2026-08-08, tooling landed, not yet run; blocked on the d/λ = 0.904 empirical-table gap |

Designs and decision rules for the wider programme are in
[`docs/future_experiments.md`](../docs/future_experiments.md).
