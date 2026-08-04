# Project learnings

Durable, hard-won conclusions. Read this before making decisions about data quality,
training-set curation, or hardware/capture changes. Each entry states the finding, the
evidence, and what to do (or not do) because of it. Newest first.

## L10 — dual-RX gain phase tracks the AD9361 RF state, not the requested dB (2026-08-02)

Re-analysis of the 2026-07-30 A–G spectroscopy campaign (2 radios, 113 LOs 400–5900 MHz;
18,202 of the campaign's 19,836 frames used). Full report, code and input hashes in
`spf/calibrations/dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1/`.

**Baseline, recorded for the first time.** With a per-frequency equal-gain anchor already
applied, changing the RX gain pair still costs **6.65° MAE / 18.4° P95 / 41.6° max** —
**8.31° MAE** counting only the unequal-gain cells a correction actually acts on. Raw
uncorrected `RX1−RX2` is 14.2–14.8° MAE. (The previously quoted ~14° `constant_per_radio`
figure is the *anchored-per-radio* number, not a no-correction number.)

**Three findings that constrain every future model:**

1. **The gain effect is 94–99% antisymmetric between the arms.** Writing
   `D(g1,g2) = H(g1) − H(g2)` with one shared `H` leaves 1.3–6.0% of the energy in the
   arm-specific residual — though that residual concentrates above 4 GHz (mean 3.72°,
   max 23.7° there, against 0.73° mean below 1300 MHz). Measured model-free from the additive-cross schedule via
   `H = [D(g,26) − D(26,g)]/2`, `A = D(g,26) + D(26,g)` — no fit involved. `H` is largely
   shared *between radios* for the large-`H` case (ρ = 0.985 at g=45; 0.996 in the low and
   middle bands) but much less so above 4 GHz and for small `H` (ρ ≈ 0.45–0.48).
2. **Phase steps where the audited gain table changes an RF word — and the measured
   driver is the mixer.** A 1 dB step changing the **mixer** word moves `H` by a median
   **2.664°** against **0.343°** for a baseband-LPF-only step (7.8×, cluster-bootstrap 95%
   CI [5.1, 16.3], n = 12 vs 132 over 12 and 14 (radio, LO) clusters). The four measured
   **TIA** 0→1 steps sit at 0.339°, indistinguishable from the LPF floor (p = 0.995) — and
   both sit at the *measured* per-step noise floor of 0.355–0.368°, so neither is resolvable
   by this experiment, while the mixer step is 7.4× that floor. Critically,
   **no adjacent-1 dB LNA transition was measured anywhere in the campaign** — the LNA
   evidence is four 9 dB steps (5.4–10.0°) plus the ripple in finding 3. Across 27→40 dB
   at 5100/5766 MHz the audited state is frozen and 13 dB of gain costs <1° on three of
   four curves (1.8° on the fourth). **Corollary: parameterise by the audited
   `(LNA, MIXER, TIA)` words, not by requested dB.** Doing so raises the fraction of
   *unseen* requested gains that are predictable at all from 48% to 90%.
3. **The frequency dependence is a reflection standing wave modulated by the LNA state.**
   Fitted delays 2.54 ns and 0.88–0.92 ns, shared by both radios. Where the gain table
   changes the LNA index the ripple amplitude is 1.1–10.7°; where it does not, 0.11–0.36°.
   The ordering *inverts* across the 4 GHz band edge exactly as the tables predict. This
   also explains the previously anomalous 433/600 MHz gain-curve anticorrelation
   (ρ = −0.22): 167 MHz is 0.43 of the 392.5 MHz ripple period ≈ 153° of ripple phase.

**Gain-table byte 2 bit 5 is `RF_DC_CAL`, not digital gain.** Digital gain is identically
zero on all 231 rows of all three tables, so it cannot contribute phase. The flag is set on
exactly the rows that begin a new LNA/mixer/TIA state, which confounds an RF-state phase
step with an RF-DC-recalibration step. The excluded `F_neg` stage bounds any RF-DC-only
step at **≲0.7°** (n = 24, median 0.722°) against a 4.36° LMT step at the same LOs, but at
n = 4 rising edges it does not resolve the attribution to the 0.35° level. Read finding 2
as "the RF-state transition, including any RF-DC correction it triggers"; E-CAL1 closes it.

**What to do.** Given a measured equal-gain anchor at the operating LO, a **27-parameter
universal** model (`H(state)` + two LNA-state-indexed ripples) predicts an unmeasured
frequency at 2.26° MAE (2.83° on unequal-gain cells) and an *unmeasured radio* at 2.22°,
against the 6.65° baseline. **No parameter needs to be radio-specific** — promoting any
family to per-radio changes same-radio error by ≤0.014° (inside the predeclared 0.1°
margin) and destroys transfer. The minimal radio-specific state is one measured anchor per
(serial, exact LO, session). **This is a two-radio result on one harness topology**; the
fifth-radio pre-registered test in `reports/four_radio_dense_20260728_v1/README.md` is the
condition for promoting it to a fleet claim.

**What NOT to do.** (a) Do not fit smooth polynomials in frequency: they look fine when
neighbouring LOs are retained and blow up to 9.6–10.4° MAE across a real 690 MHz gap (the
reported 173.6° max is wrap-saturated — the true excursion is larger). (b) Do not
extrapolate across a gain-table band — train on two bands, predict the third, and no model
beats baseline by more than 8%. The model interpolates within a measured span; it does not
extrapolate. (c) Do not apply the correction when the audited `(LNA, MIXER, TIA)` words are
identical on both arms: there the fitted baseband-LPF differences are noise, and the model
injects a mean 1.36° and makes 81% of those cells worse. (d) Do not expect a stored gain
model to match a fresh calibration: across a 12-hour session boundary even the
1356-parameter per-frequency LUT degrades from 0.62° to 2.74°, so there is real session
drift in the *gain-dependent* term, not only in the intercept.

**Calibration cost.** Held-out error is flat for frequency gaps from 96 MHz to ~690 MHz,
so a ~10-point comb over 400–5900 MHz recovers essentially all of the 113-point comb's
benefit for the gain term. Beyond ~1.4 GHz gaps it degrades.

## L9 — sub-GHz post-processing recovery: mostly NOT recoverable for phase (2026-07-12)

The wall array no longer exists, so historical sub-GHz data can only be salvaged by
post-processing. All three candidate levers were tested empirically (read-only, 32/12/32
datasets across Oct/Nov/Dec/Jan; scripts in /tmp session logs, results recorded here):
1. **Slow-drift detrend** (sliding circular mean of the residual across snapshots):
   PARTIAL — receiver agreement ρ −0.46 → +0.26, |Δg| 0.64 → 0.53, circstd 1.03 → 0.91.
   The only lever that moved anything.
2. **DC-excision** (recompute per-snapshot phase from raw-IQ cross-spectrum with
   |f| < 0.002·fs removed): NULL — ρ ≈ 0 before and after; the corruption rides ON the
   tone, not beside it (loop bandwidths evidently exceed the tone offset).
3. **Gain-conditioned offsets** (per recorded AGC gain-state circular offsets, fine and
   3 dB bins): NULL — recorded per-snapshot gain doesn't index the corruption. Caveat:
   fast-attack gain moves within buffers and is undersampled (~1 Hz), so this doesn't
   fully exonerate gain events — it exonerates the recorded gain as a correction key.
Ceiling: circstd ~0.9 / ρ ~0.3 vs the 2.4 GHz benchmark (0.5 / 0.97) — per-dataset
phase/g stays unusable. What remains usable: per-config MEDIANS after detrend (for the
coupling curve), amplitude/RSSI tasks, duty statistics, and — since GT position labels
are intact — the band is INPUT-DEGRADED, not label-corrupt: keeping it in training is an
A/B question (exactly what the running base/r1(keeps it)/r2(drops it) ladder measures).
**Cross-receiver raw-sample fusion (tested):** coherent 4-channel processing is
INFEASIBLE — the two boards' LO relative phase wanders 2.6 rad within one 8 ms buffer
(measured; independent crystal phase noise), vs 0.007-0.12 rad intra-pair. But that
probe found the KEY structural fact: **within-buffer intra-pair phase is nearly perfect
(r1: 0.007 rad, r0: 0.12 rad); the corruption is buffer-to-buffer**, smooth-ish
(residual autocorr 0.70-0.99 at lag 1, decorrelating over ~10-100 snapshots). The
information survives capture; recovery = separating a smooth per-receiver process from
geometry. Naive two-step detrend locks in the initial (corrupted) g; a self-referencing
sliding-trend joint fit degenerates (rewards over-subtraction). A properly regularized
joint estimator (parametric spline least-squares with cross-validated knot spacing,
gapped/leave-window-out trend) on jump-trajectory datasets (rx_random_circle: sin θ
jumps, nuisance smooth → separable) is the one remaining credible route — designed as
E-REC2. **REC2b (GT-free) was then tested and FAILED structurally** (48 datasets,
all routines): the gantry moves continuously, so geometry is as smooth as the nuisance
at snapshot timescales — a label-blind trend absorbed 36-54% of the geometry and every
fit collapsed to the amplitude floor. There is no label-free timescale contrast on this
corpus; **sub-GHz recovery for training purposes is closed**. Only REC2a (GT-using,
metrology-only, audit outputs) remains viable.
Re-collection does NOT need the wall array: a bench rig (two Plutos + emitter on a
measured arc/turntable) with fs/16 IF suffices for the E-IF1 diagnostic and for a
sub-GHz calibration set; rovers with GPS truth are the field alternative.

## L8 — "bad months" = the sub-GHz campaign; NaN spike = transmit sparsity, not decay (2026-07-12)

The Oct/Nov-2024 "bad months" were a band confound, and the NaN ratio has a different
cause than the phase quality:
- Oct 2024: the SAME rig concurrently produced 96 normal 2.4 GHz datasets (quar 1,
  NaN 0.00, circstd 0.62) and 85/85 quarantined sub-GHz ones. Nov 2024 is 100% sub-GHz.
- Window probe: Oct/Nov sub-GHz duty = 16–19%, Jan 2025 = 58% (emitter switched from
  bursty LoRa-style packets to near-continuous tone ~Dec; lab log "tone blaster").
  NaN tracks duty (0.26–0.34 → 0.02): **smaller/fewer packets do drive up NaN** — a
  benign experiment property, exactly as hypothesized.
- But corrected circstd ≈ 1.0–1.1 in ALL sub-GHz months including low-NaN January: the
  phase-quality failure is band-driven (IF-at-DC, L4; deep coupling) and month-invariant.
- Oct/Nov "noise" windows carry amplitude 40–110 (vs 1.9 in Jan): fast-attack AGC pumps
  gain up between sparse bursts.
- Feb 2025 is NOT a bad month in scan v2 (8/272 quarantined); its lab-log debugging era
  surfaces as the 5 unreadable ERROR files instead.
- **Do:** treat NaN as a duty descriptor, not damage (rovers already handled this way);
  judge sub-GHz by phase metrics. **Don't:** attribute quality to calendar time without
  splitting by band/config first — era and band were confounded in this corpus.

## L7 — IF policy: off-center fs/16, all tracking on; the BBDC question is moot (2026-07-12)

- **Off-center IF is completely free for the measurement:** both RX channels share one
  LO, so the IF rotation e^{j2π·f_IF·t} cancels exactly in x₁·x₀* — measured phase
  difference, amplitude stats, and segmentation are identical at any IF. Moving
  off-center only removes pathologies (tracking notch, offsets, LO leakage, 1/f skirt,
  and the quadrature image, which lands ON the tone at IF=0). Never 0-center.
- **Window:** |IF| ≥ max(10× crystal wander, ~0.01·fs) and ≤ passband/2 − signal
  bandwidth. f_IF = fs/16 satisfies this at every SPF band/rate; wideband signals
  (20 MHz Wi-Fi at 30 MS/s) are the only case needing thought.
- **BBDC-off adds no noise** — it trades the loop's unlogged time-varying correction for
  a quasi-static, recorded, post-removable DC spur. Bias on the phase product
  ~(offset/signal)² (~0.003 rad for a near-rails signal and −30 dBFS spur). Caveats:
  spur is gain-dependent (largest at high AGC gain = weak signal, the rover regime);
  amplitude-bit cost is log2(2048/(2048−|c|)) — negligible below ~20% FS; a big spur
  also steals AGC headroom. With the fs/16 policy none of this is exercised.
- Full Q&A: report §6b "IF policy" page; experiment: E-IF1 (demoted to diagnostic).

## L6 — gain-change phase inversion was FIXED in code in Jan 2024 (found 2026-07-12)

Commit 9d00b7b (2024-01-26, "Fix phase inversion Rx1; Fix gain phase inversion") applies
two mitigations on every Pluto capture (`sdr_controller.py:709-721`, also
`test_throughput.py`; explored in `notebooks/iio_sdr_interface_tests.ipynb`):
1. `adi,rx1-rx2-phase-inversion-enable = 1` — compensates the ADI-documented RX2-inverted-
   relative-to-RX1 behavior.
2. reg 0x22 |= (1<<6) = INVERT_BYPASSED_LNA_POLARITY — keeps polarity consistent when AGC
   crosses the LNA-bypass gain boundary (the "phase flips on gain change" failure).
All wall v2/v3 fleet data (May 2024+) was captured with both active — so do NOT cite
LNA-bypass polarity flips as a live mechanism for heavy-tail outliers in fleet data;
residual heavy tails (e.g. 28% outlier fraction at 2.412 GHz small-spacing) are so far
unattributed (AGC amplitude transients, interference, multipath remain candidates).
DC-offset tracking (L4) was NOT part of this fix and remains unaddressed in code.

## L5 — `F:gain` flags one physical fact, not 1,268 problems (2026-07-12)

`F:r{0,1}_gain=X` in the quality scan is **not receiver/AGC gain**. It is the fitted
parameter g = effective/configured antenna spacing (the amplitude of
φ ≈ φ₀ + g·(−2π d/λ)·sin(θ−Δθ)), flagged when |g−1| > 0.25.

- It fires on 1,268/2,250 datasets (all wall) because it flags **config families, not
  captures**: every capture taken on a small-spacing config inherits the same rig-level
  fact. Wall 2.4 GHz medians: configured 2.5 cm → g=2.14 (effective 5.4 cm), 3.5 → 1.78
  (6.2 cm), 5.075 → 1.38 (7.0 cm), ≥6.5 cm → ≈1.0. Effective spacing floors near λ/2.
- It is real physics, not mislabeling: g varies smoothly with configured spacing, and the
  two independent receivers of each session agree at ρ=+0.97 (2.412 GHz), +0.91 (2.464),
  +0.85 (5.8 GHz).
- A 2-parameter mutual-coupling model, C(d) = A·e^{j(ψ₀−kd)}/(kd) with
  g = (1−|C|²)/(1+2ReC+|C|²), fits the 2.4/5.8 GHz medians with rmse 0.04–0.12 and
  physical amplitudes (A = 0.2–0.5 ⇒ |C| < 0.5). See report §5b + full for/against
  discussion there; decisive validation would be a bench VNA S21-vs-distance sweep.
- **Do:** treat gain-flagged data as *feature-mislabeled* (usable phase, wrong
  rx_spacing input); keep it in training (r1/r2 regimes only drop QUAR/nan>20%/noisy).
  Candidate improvement: effective-spacing sidecar from the fitted g(d, band).
- **Don't:** read the flag count as a data-quality crisis, or drop data on it.
- Scanner v3: gate on deviation from the config-family median (|g − median_g(config)| >
  0.15) instead of |g−1| > 0.25 — silences the known floor, catches true per-capture
  anomalies (like the rover 43↔47 mm mislabel that data surgery fixed).

## L4 — sub-GHz (868/915) per-dataset g is unusable; root cause: IF margin < crystal wander (2026-07-12)

The g scatter at 868/915 MHz is enormous (per-config IQR 0.2–0.5, receiver agreement
ρ ≈ 0.0, median |g_r0 − g_r1| = 0.58). Investigation chain
(`data_quality_reports/g_vs_spacing/probe_subghz_windows.py`):

- Not Gaussian noise: Monte-Carlo of the scanner's fit is unbiased with sd ≤ 0.1 even at
  σ_φ = 1.2. Not coverage (all bands fill 12/12 angle bins). Not segmentation quality.
- **Configured IF was 100 kHz, not zero** (`f-intermediate: 100000` in every capture
  config, all bands) — but crystal wander scales with carrier (ppm × f_c; Pluto
  `xo_correction` is an uncalibrated 40.000000 MHz) and at 915 MHz reaches 50–140 kHz,
  the size of the offset itself: nominal +100 kHz measured as +48 kHz (r0) and −39 kHz
  (r1) — the tone wandered THROUGH DC. At 2.412 GHz ~10% of combos cancelled to ~0; at
  5.8 GHz the 2.4×-larger wander threw the tone clear and saved the band. Rule: f_IF
  must dominate wander (≥10×), i.e. scale with carrier; also calibrate xo_correction
  per board (one-time trim, shrinks wander ~10×).
- **The failure at ~0 Hz effective IF:** Raw IQ shows a continuous strong
  carrier (rms ~1200, near rails) with 74% of power within 0.2%·fs of DC on r0; r1 has
  the same tone at −0.0034·fs. Crystals are independent per BOARD; within a board the
  RX LO is shared — both antennas show identical tone offsets (verified +47.9/+47.9 kHz
  on r0, −38.8/−38.1 on r1) — so crystal error is common-mode and cancels in the phase
  difference. The damage path is PER CHANNEL: each RX chain has its own DC offsets and
  correction state (gain-indexed RF-DC words + BBDC tracking loop), so a near-DC tone is
  perturbed differently per channel, producing the slow differential drift (2× drift
  span, corrected circstd ≈ 1.0) that aliases into the fitted amplitude. Small geometric swing (±0.9–1.4 rad at d/λ = 0.12–0.23) makes the aliasing
  ~3× worse than at 2.4 GHz. Matches the lab-log note "the issue might be that the IF is
  0hz?" (Jan 2025); all sub-GHz data is from the Oct-2024–Jan-2025 degraded era.
- Within an on-snapshot, "signal" and "noise" windows are identical (same rms/spectrum):
  the carrier never stops, so the phase-stability window selection is arbitrary there.
- **Scope limit (IF sample, n=160):** sub-GHz datasets where BOTH receivers clear the
  notch still show circstd ≈ 1.0 — the DC collision explains the g-fit
  IRREPRODUCIBILITY (independent per-receiver drift aliasing), but the band's overall
  phase quality also reflects the degraded Oct-24–Jan-25 era and the wider near-DC 1/f
  skirt. Run a small controlled IF A/B capture before commissioning a full sub-GHz
  re-capture (report §6b, R3).
- **Config state of record:** no capture parameter for DC tracking exists —
  `bb_dc_offset_tracking_en` / `rf_dc_offset_tracking_en` appear nowhere in
  `sdr_controller.py` or any capture yaml, so every historical Pluto dataset ran with the
  driver default (ENABLED; confirmed by the Feb-2025 live iio_attr dump in the lab log).
  There is no enabled-vs-disabled data to compare. The natural insertion point for a knob
  is `PPlus.setup_rx_config` (it already sets debug attrs / raw registers there).
- **Do (future capture):** set IF a few hundred kHz off zero (`--fi`), or disable BBDC
  tracking for sub-GHz sessions. **Don't:** use historical sub-GHz per-dataset g, or
  build a sidecar from sub-GHz medians (they remain bias-suspect; report §5b draws that
  band's fit dashed-grey "NOT trusted").

## L3 — scalar φ̂ vs beamformer: same per-window information, different aggregation (2026-07-12)

For 2 elements, the windowed beamformer is a deterministic transform of
(φ̂, |Σz|, P₀, P₁) — per window they are equivalent. The beamformer wins at aggregation:
likelihood curves average correctly across windows (multimodality preserved), scalars do
not; the 65-bin grid unifies configs with d/λ 0.21–1.55; junk windows degrade to flat "no
opinion". Scanner-v3 candidate: score datasets on the cached `weighted_beamformer` —
offset-corrected GT-bin percentile (alignment) + entropy (informativeness). Naive
(uncorrected) version already separates healthy (0.52–0.55) from DC-corrupted (0.42–0.47)
but confounds offset with informativeness — must fit the offset first, same lesson as g.

## L2 — NaN snapshots are a "no signal detected" marker, not corruption (2026-07-12)

`mean_phase = NaN` is written by segmentation when zero windows pass the phase-stability
gate (`segmentation.py:91`). Raw IQ is intact. Expected on rovers (bursty beacon, 60%+
NaN with clean valid remainder); pathological on wall (continuous emitter ⇒ ~0%), where
high NaN marks broken sessions (Nov 2024, Feb 2025 hardware-debug eras) — and the valid
remainder of those eras is also noisier. Hence platform-specific two-tier gates and the
R2 regime dropping only >20%-NaN datasets.

## L1 — the validation set is frozen; new insight becomes a named subset (2026-07)

Never edit the historical val list; add named `val_subset_groups` (must be subsets,
hard-asserted at load). Decision metric for quality experiments is
`val_clean/single_loss`; `val/single_loss` is the historical-continuity metric;
`val_degraded` is reported but never optimized toward. Eval-only runs MUST pass a scratch
`--output` (val-and-exit executes the save-best path). Bit-compat baseline for the jun26
single checkpoint: val/single_loss = 0.09735 via --val-and-exit.

## E-DATA1 stage 1 (2026-07-14): dropping degraded captures helps; label-clean subset does not

At 250k steps on identical schedules (frozen val set, 1991 batches,
`val_clean/single_loss`): base(full train) 0.10211, r1(label-clean 1630 files)
0.10330 (+1.2%), r2(no-degraded 1217 files) 0.10125 (−0.8%). Read: the
degraded-flagged captures are mildly toxic (removing them wins despite 25% less
data), while the label-clean filter cuts data that was actually useful — its
smaller train set hurts more than its cleaner labels help at this scale. Not yet
decision-grade: gaps are ~1%, and the ladder's 500k checkpoint (tighter 1.5%
kill) is the real test. Ops note recorded the hard way, twice: the trainer
freezes silently (no crash) when / hits ENOSPC mid-checkpoint — the runner disk
guard (<10G refuse) and WANDB_DIR on /mnt/md1 are load-bearing, and any
"trainer alive but log frozen" watchdog alert should be read first as a
disk-full or clean-exit signature, not a GPU hang.

**Full-val-set caveat (same 250k evals):** r2's win exists ONLY on val_clean.
On val_degraded it is +16.9% vs base (0.13058 vs 0.11169), on val_band915
+18.5% (0.13365 vs 0.11277), and on full historical val +4.6% (0.10363 vs
0.09912) — the signature of lost robustness to conditions it never trains on
(and it cycles its smaller train set 5 epochs vs base's 3 in the same steps).
r1 is uniformly ~+1% everywhere, catastrophic nowhere. If deployment includes
degraded-like RF or the 915MHz band, do NOT promote r2 on val_clean alone —
judge the 500k gate on the full four-set table.

## Power board (2026-07-14): KiCad 10 upgrade audit found D8 (USB3 ESD) missing from the board

KiCad 10.0.4's new `kicad-cli pcb drc --schematic-parity` caught a component
that was in the schematic, netlist, and BOM-intent but never on the board:
D8, the USBLC6 ESD array for USB port 3 (J13) — port 3's data lines ran
J14→J13 unprotected while ports 1/2 had D2/D3. Root cause: the 3rd-port
change (c0e1eb9) added D8's `place()` call but not its entry in
generate_schematic.py's per-ref footprint map (`REF_FP`), so the netlist
carried D8 with an empty footprint and generate_board.py's
`SKIP: no footprint` console print dropped it silently. Every gate passed
because DRC/audit/routing-completeness check only board-internal
consistency, and kicad-happy reviews the schematic side (where D8 looked
fine). Fixes shipped: missing footprint is now a hard error in
generate_board.py, `--schematic-parity` is in the gate list, and D8 was
retrofitted onto the routed board (verified copper, DRC 0/0, audit PASS,
v4.5). Standing rule: any tool "skip" on generated artifacts must be a
build failure — a printed warning on a generation console is invisible by
the next session. Parity noise classes on this board (stable, not bugs):
134 lib-prefix footprint_symbol_mismatch, 18 merged-drain-pad MOSFET
net_conflicts (Q2/Q3/QA*/QB*), 4 mounting-hole extra_footprint.

## Power board (2026-07-14): D5/D4 polarity reversed — symbol-pin vs footprint-pad convention

The JLC order review caught a bug class invisible to EVERY electrical check
(DRC, audit, parity, netlist — all self-consistent): KiCad diode/LED
footprints put the CATHODE on pad 1 (verified programmatically: the
asymmetric silk/fab marker graphics cluster at pad 1 on D_SMB, D_SOD-123,
LED_0805). Our generator's generic 2-pin symbols let the author wire pin 1
to either net; D5 (reverse-battery schottky) got pin1=VBATT_F → cathode on
the battery side → the always-on 3.3V supervisor rail could NEVER power
(dead MCU on every board), and D4's LED was reversed (never lights). The
TVS diodes and electrolytics happened to be wired pad1-correct. Fix: pin1
carries the cathode net in the generator, parts rotated 180 on the routed
board with pad nets rebound (copper untouched), v4.7. Standing rule: for
every 2-pad polarized part, EXPLICITLY verify pad1's net is the cathode/+
per the footprint's marker — connectivity checks can never catch this.

## Power board (2026-07-14): XT60 battery connector polarity was REVERSED (third
## instance of the symbol-pin vs footprint-pad convention bug in one day)

KiCad's AMASS_XT60PW-M footprint puts pad 1 at the "-" blade (its own silk
says so; JLC/EasyEDA part data agrees: + is the north blade with the
opening west). Our XT60 symbol had pin1="+" -> pad1, so a correctly-wired
battery would have put + into GND: the front-end reverse protection blocks
it and every board reads dead-on-arrival. Fixed in v4.8 (symbol pins
renamed, J1 net map + board pad nets swapped, VBATT_RAW re-run to the
north blade, GND via pours). Deterministic verification method that found
it: fetch JLC's footprint per LCSC code from the EasyEDA API
(easyeda.com/api/products/<code>/components — blocked from curl, works
via WebFetch) and compare pad frames against the KiCad footprint; also
cross-check the footprint's own polarity silk against pad nets. THE RULE,
final form: for EVERY polarized 2-pad part — diodes, LEDs, electrolytics,
AND connectors — verify pad 1's net against the footprint's own polarity
marker. Symbol pin names are vibes; footprint pads are physical.

## Power board (2026-07-14): 6A switch nodes were routed at 0.15mm — current
## capacity is invisible to every gate

A user question ("is the copper to LA1 good for the current?") exposed that
BOTH buck switch nodes (SW_A/SW_B: FET half-bridge -> inductor, ~6A) were
routed entirely in 0.15mm thin-pass tracks — instant trace burnout at
load. The KRT thin reconciliation pass routed them and nothing objected:
DRC checks clearance not ampacity, the audit checks placement, netclasses
were never width-assigned per net. Also found: 5VB_PRE (buck B output ->
pi filter, 6A) had no plane and only 1.2mm necks. Fix (v4.9): F.Cu pour
patches (priority 3 over the GND pour) SW_A 83mm2 / SW_B 57mm2 / 5V_A
126mm2 / 5VB_PRE 171mm2 / 5V_B lily at L4 + plane-stitch vias; DRC 0/0.
STANDING RULE: before fab, walk every net that carries >1A (buck SW nodes,
inductor outputs, pre-filter rails, battery input, VBUS feeds) and check
minimum copper cross-section along the whole path — assign netclass
widths OR pour patches. No tool does this for you.

## Power board (2026-07-14, v4.10): netclass ampacity framework — the guardrail
## that should have existed from day one

"Are nets being used correctly?" — they weren't: one Default netclass, zero
assignments, so nothing distinguished a 6A trunk from a sense tap. Now:
current-tiered netclasses (SWITCH_NODE / PWR_RAIL / VBUS / USB_DATA) with
.kicad_dru minimum-width rules per class — undersized power copper is a
hard DRC violation forever. Design decisions encoded: trunk current rides
pours/planes (floors are backstops); PWR_RAIL floor 0.25 permits mA sense
taps; SWITCH_NODE taps (gate-drive returns at 0.15) are exempted via NAMED
RULE AREAS (SW_TAP_A/B) so the exemption is visible on the board, not
tribal knowledge. Cleanup this surfaced: HO_A gate trace rerouted around
the SW_A pour (it was slicing the pour in half); SW_B pour islands bridged
with 2x1.0mm B.Cu hops + via pairs under HO_B; 6 tracks at 249800nm
(0.2um below floor, prints as "0.25" — KRT import artifact) bumped to
250000. Empirics: KiCad dru width rules compare EXACT nanometers; zone
fills can silently split into islands over a crossing trace (the
unconnected-items check catches island-to-island, read it); A* at >=0.35mm
cannot thread corridors built for 0.15 — reroute the blocker, not the
victim (and when the blocker is unmovable, bridge under it on B.Cu).

## Rover (2026-07-23): Taranis CH8 mode order is Manual/RTL/Guided — three docs
## disagreed; ground truth is what boots actually load

The README's Taranis section says SA = [Manual, Guided, RTL] and the Jun-2024
rover3_idX param dumps agree (MODE4=15/MODE6=11) — but drone_run.sh enforces
rover3_base_parameters.params at EVERY boot, and that file (Apr 2025) sets
MODE4=11/MODE6=15, i.e. switch pos 1/4/6 = Manual/RTL(11)/Guided(15). The
operator field-guide slide (Slides deck linked from project_spf.pdf p.57)
confirms Manual/RTL/Guided. ardupilot_setup.md is older still (MODE4=10=Auto).
Mid-position is RTL, not Guided — flipping "to Guided" one notch drives the
rover home instead of handing control to the Pi. Rule: answer RC-function
questions from rover3_base_parameters.params + rover3_rc_servo_parameters.params
(boot-loaded) + mavlink_controller.py handle_RC_CHANNELS (Pi-side CH7/9/10/12),
not from the README transcript. Full map: ROVER_RUNBOOK.md §3.5 +
taranis_q_controls.png (generated by make_taranis_map.py).

## Rover (2026-08-03): the 100%-throttle stall is integrator windup, and the
## existing "stall" heuristic detected the exact opposite failure

Symptom: the rover jams going forward and then sits at full throttle until
someone pulls the battery. Mechanism: `AR_AttitudeControl::get_throttle_out_speed()`
sums `throttle_base = _desired_speed * (cruise_throttle / cruise_speed)` with
the `ATC_SPEED_*` PID. `rover3_base_parameters.params` sets CRUISE_THROTTLE 10
against CRUISE_SPEED 1.5, so the feed-forward contributes ~10% and the
integrator does the rest; with ATC_SPEED_P/I 0.2 and zero measured speed the
output saturates in about two seconds. ArduPilot's own anti-windup DOES work
(the saturation flags are passed into `update_all`), so I settles near 0.6
rather than IMAX — the output is pinned regardless, and it unwinds once the
rover moves. **The danger was never the magnitude, it was that nothing bounded
the duration.** Corollary for any recovery maneuver: bound it in time and the
same windup is harmless.

Two things that were actively misleading before this:

- `handle_SERVO_OUTPUT_RAW` defines `motor_active` as *false iff servo1 and
  servo3 are both 1500*. The only stall heuristic in `move_to_point` was
  `elif self.armed and not self.motor_active`, i.e. it fires when the throttle
  is asleep at neutral. A rover pinned at full forward has servos at 800/2200,
  so it could never fire on this failure. It was also inside the
  `if self.distance_finder is not None:` guard, so `--no-ultrasonic` disabled
  it entirely.
- `FS_CRASH_CHECK` is 0, and its only actions are Hold/Disarm — it cannot hand
  control to the operator, so the MANUAL handback is necessarily Pi-side.

Detection must be **displacement from an anchor**, never distance-to-target:
WP_PIVOT_ANGLE is 0 and TURN_RADIUS is 5.0, so a turn is an arc and a healthy
rover legitimately moves away from its waypoint for twenty seconds at a time.

## Rover (2026-08-03): DO_SET_REVERSE is a COMMAND, not a parameter, and it
## self-clears on every mode change

`MAV_CMD_DO_SET_REVERSE` (183) routes through `Rover/GCS_Mavlink.cpp` to
`Mode::set_reversed()` → `g2.wp_nav.set_reversed(value)`: a runtime flag inside
AR_WPNav with no AP_Param backing. It therefore never appears in a param list
and **cannot go in rover3_base_parameters.params** — a line for it would simply
fail to set. There is nothing to enforce at boot and no drift path to guard,
unlike ARMING_CHECK (b4fa14a): `Mode::enter()` calls `set_reversed(false)`
unconditionally on every successful mode entry, so the MANUAL→GUIDED startup
handshake alone guarantees it is clear before the planner issues a waypoint.

Within a single mode it does persist, so anything that sets it must clear it on
every exit path or the next leg is driven backwards.

Related: Rover never chooses reverse on its own. `_reversed` is only ever *read*
in AR_WPNav — a GUIDED target behind the rover makes it turn around, never back
up. Reverse toward a waypoint has to be commanded explicitly. Verified against
Rover-4.5.7 source; the fleet runs 4.5.0, which is why
tests/test_in_simulator_crash.py asserts reverse PWM at the servos rather than
trusting the command to have worked.

## Rover (2026-08-03): two SITL settings silently make a stall un-triggerable

Found while building `tests/test_in_simulator_crash.py`. Both produce a
motionless simulated rover with no stall detected, which reads like a broken
watchdog and is not one. Recorded because the obvious choice is wrong in both
cases.

**Do not jam a SITL rover with `MOT_THR_MAX`.** Measured: `MOT_THR_MAX <= 5`
does stop the rover (0.03 m/s) but collapses both throttle outputs to exactly
1500 for ~90% of samples. `motor_active` is defined as *not* both servos at
1500, so it flickers false, the detector's gate opens and closes, and the anchor
never accumulates. Clamp `SERVO1/3_MIN/MAX` to 1480/1520 instead: ArduPilot pegs
the output at the clamp — clearly off neutral, `motor_active` solidly true —
while the physics sees ~2% throttle. Restore to 1000/2000 (SITL defaults; the
sim never loads `rover3_rc_servo_parameters.params`, which uses 800/2200).

| clamp | off-neutral | sim speed |
|---|---|---|
| 1490-1510 | 49% | 0.11 m/s |
| 1480-1520 | 100% | 0.26 m/s |
| 1470-1530 | 100% | 0.41 m/s |
| 1000-2000 (free) | 100% | 3.9 m/s |

**Run the stall suite at `-S 1`, not `-S 5`.** The stall clock is host
`time.monotonic()` while the vehicle moves in SIM time, so any speedup
multiplies the ground covered per wall-second and defeats the 3 m progress
radius. At `-S 5` the clamped rover above still covered 15.8 m per 12 s of wall
time — five times the radius, so the anchor kept resetting. `-S 5` remains fine
for `tests/test_in_simulator.py`, which has no wall-clock/sim-time coupling.

General rule this is an instance of: whenever a host-side timer is compared
against vehicle-side motion, sim speedup is not a free accelerant — it changes
the quantity under test.

## Rover (2026-08-04): the RC link is R9M ACCESS + R9 SX, not XJT D16 + X8R —
## and rover identity lives in RxNum, not the receiver slot

`README.md`'s "Taranis Q setup" transcript (Jun-2024) describes the **internal XJT
module on D16 binding an X8R**, and `make_schematic.py:100` still draws a
"FrSky X8R RC" block. Neither is current. Confirmed from the transmitter
2026-08-04: all three rover models run the **R9M in the external bay on ACCESS**
with an **R9 SX** receiver, `Ch Range CH1-16`.

Per-rover: **Rover1 RxNum `01`** (slot 2), **Rover2 RxNum `05`** (slot 3),
**Rover3 RxNum `00`** (slot 1). The occupied receiver slot differs per rover for
no functional reason — RxNum is what the bind writes into the receiver and what
model-match checks, so **never infer the rover from the slot index**, and never
re-slot to make them tidy (that costs a re-bind). Uniqueness of RxNum across
models is the real invariant, the same discipline as the SiK NetIDs.

Two consequences worth carrying:
- **Rover 3 on RxNum `00`** is the bottom of the range and the value an untouched
  model can carry, so a newly created/restored model left at its default can
  command Rover 3. Set RxNum before binding anything new on this transmitter.
- **`FS_THR_ENABLE 0`** (`rover3_base_parameters.params`) was survivable when the
  link was a short-range X8R — link loss implied the rover was near. On a 900 MHz
  R9 the rover can be well past visual range when the link drops and ArduPilot
  takes no failsafe action. Open item, not a settled decision.

The general rule, same as the CH8 mode-order finding above: **`README.md`'s
Taranis section is a 2024 transcript, not ground truth.** Answer RC questions
from the boot-loaded params, `handle_RC_CHANNELS`, and `ROVER_RUNBOOK.md` §3.5.

## Rover (2026-08-04): a GUIDED reposition target inside WP_RADIUS is a no-op —
## the stall escape maneuver silently did nothing

The stall recovery drives an escape leg by sending DO_REPOSITION to a point a
fixed distance away. At 3 m against `rover3_base_parameters.params`'
`WP_RADIUS 5.0`, ArduPilot judged every escape target ALREADY REACHED the moment
it was issued and never drove the leg. The maneuver logged "reversing out",
cleared its reverse flag, held to settle, logged "stepping off that axis" — and
moved the rover nowhere. Stage 1 of the recovery was a no-op on the whole fleet;
a jammed rover would have ridden the clock to MANUAL every time without ever
attempting to free itself.

Rule: **any commanded GUIDED destination must be further away than WP_RADIUS**,
or arrival is immediate and the vehicle never moves. `STALL_ESCAPE_DISTANCE_M`
is now 8.0 m and `tests/test_crash_detection.py` reads WP_RADIUS out of the
params file and asserts the legs exceed it, so neither value can drift into the
other.

How it was caught, and why it was nearly missed: in SITL the escape produced
`servo1=1502, servo3=1487` — near neutral AND unequal, i.e. steering trim rather
than the pegged reverse output expected. Log-level assertions all passed, because
the Pi did everything it was supposed to; only servo telemetry showed the vehicle
ignoring it. This is the concrete payoff of asserting on vehicle telemetry rather
than on the collector's own log lines.

Second-order lesson from the same fix: two geometry unit tests asserted a
hardcoded `3.0` m leg length, which is precisely what let this hide. Tests that
restate a constant instead of importing it cannot catch a bad constant.

## Rover (2026-08-04): a recovery maneuver that moves the rover defers the
## escalation it is supposed to lead to — the clock alone cannot cap attempts

The stall watchdog was built around one rule: the progress anchor resets only on
real motion, never merely on attempting a recovery. That rule is what makes
"stuck in reverse too" safe — a maneuver that moves the rover nowhere leaves the
clock running and escalation proceeds.

It is not sufficient, and assuming it was produced a contract violation. An
escape that DOES shift the rover a metre resets the anchor, which restarts the
clock, which defers the hand-over to the operator. A rover creeping ~1 m per
attempt therefore escapes **forever** and never reaches a human — while
ROVER_RUNBOOK §17.2 promised "three back-out attempts, then MANUAL at 40 s".

Fix: `STALL_MAX_ESCAPES = 3`, counted since the last waypoint actually REACHED
(or an operator hand-back), not since the last anchor reset. Counting anchor
resets would be circular, because the escape is what moves the anchor.

Two general lessons:

- **A self-referential termination condition is not a termination condition.**
  Any "give up after N tries" whose counter can be cleared by the try itself is
  unbounded. Ask what clears the counter and whether the action under test can
  cause it.
- The bug was only reachable once an EARLIER bug was fixed. Before the escape
  restored GUIDED, the vehicle sat in HOLD and never moved, so the clock did cap
  attempts and the test failed for a different reason. Two defects in series can
  present as one symptom, and fixing the first changes the diagnosis of the
  second — so re-diagnose after every fix rather than carrying the old theory
  forward.

## Rover (2026-08-04): a successful FrSky bind proves TX↔RX, not RX→flight
## controller — the R9 SX ships with its SBUS pin emitting PWM

Rover 4 was bound, registered, wired correctly into the FC's RCIN, and
completely deaf. Cause: the R9 SX ships with **all six pins as PWM channels
CH1–CH6**, and the port silkscreened `CH6/SBUS OUT` keeps emitting PWM channel 6
until it is explicitly switched. Binding does not change it. Fix is on the
transmitter: the **receiver line** (the row showing `R9SX1`) → ENTER → Options →
`REC OPTIONS R9SX` → **Pin6 = SBUS** (Pin5 = S.PORT for telemetry).

Three traps around it, each of which cost time:

- **Module options ≠ receiver options.** The External RF row's Options is the
  R9M's own (RF power, telemetry) and shows only a power setting. The pin map
  lives one level down, on the receiver row.
- **The transmitter reads those options over the air**, so the receiver must be
  powered and linked before the screen populates. On a bench that needs a flight
  battery: the servo/RCIN rail is not powered by the Pi's USB, so a USB-only
  rover has a dead receiver, an empty RC stream, and a menu that will not load.
- **Rovers 1–3 only worked because someone set this years ago and never wrote it
  down.** Same shape as `rover install` and MAVProxy: knowledge living in a
  person, invisible until Rover 4 was built purely from what is in the repo.

Diagnostic rule this produced: **`RC_CHANNELS` message count is not evidence of
an RC link.** ArduPilot streams that message at whatever rate you request whether
or not a receiver exists, filling `chancount=0`. Only populated channels prove
receiver→FC. `rover ardupilot rc` originally got this backwards and blamed the
transmitter for a receiver-side fault; it now distinguishes no-frames /
frames-without-channels / real-channels. `0 channels, rssi 255` means no RC
input, full stop.

Also worth keeping: **CH16 carries RSSI** on the R9 SX ("6 PWM / 16 SBUS (CH16
outputs RSSI)"), so a CH16 that moves on its own is signal strength, not a stray
control. And the receiver's default failsafe is **Hold** — combined with the
fleet's `FS_THR_ENABLE 0`, a rover that loses its 900 MHz link keeps executing
its last command with ArduPilot taking no action. Still open.

Procedure: `rover4_setup.md` §14.5. Manual:
https://www.frsky-rc.com/wp-content/uploads/Downloads/Manual/R9%20SX/R9%20SX-Manual.pdf
