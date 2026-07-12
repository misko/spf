# Project learnings

Durable, hard-won conclusions. Read this before making decisions about data quality,
training-set curation, or hardware/capture changes. Each entry states the finding, the
evidence, and what to do (or not do) because of it. Newest first.

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
