# Future experiments

Queued, concrete experiments with motivation, design, and decision rules. Read together
with `docs/learnings.md` (the findings that motivate these). When an experiment runs,
record the outcome in `learnings.md` and mark it here.

## E-IF1 — 2×2 IF / BBDC-tracking capture matrix  (highest value per hour)

**POLICY ANSWER (decided 2026-07-12, no experiment needed):** production captures use
off-center IF = fs/16 with ALL tracking loops ON (defaults). Never 0-centered — DC hosts
the tracking notch, offsets, LO leakage, 1/f, and the quadrature image (which lands ON
the tone at IF=0). Off-center is free for the measurement: the shared-LO IF rotation
cancels exactly in x1·x0*, so phase/amplitude/segmentation are unaffected. With proper
IF, the BBDC question is MOOT for policy; the matrix below is now DIAGNOSTIC — {IF=0}
cells causally confirm the historical sub-GHz mechanism, BBDC-off cells + gain sweep are
optional science. Only constraint window: |IF| >= max(10x crystal wander, ~0.01*fs) and
<= passband/2 − signal bandwidth (watch wideband signals like 20 MHz Wi-Fi at 30 MS/s).

- **Motivation:** learnings L4/L6 — the tone-at-DC failure is observational; no data
  exists with BBDC tracking disabled (no config knob has ever existed), and the sub-GHz
  scope-limit means we can't prove IF placement alone rescues the band.
- **AMENDED 2026-07-12: the wall array no longer exists.** Run the matrix on a bench
  rig instead — two Plutos + emitter on a measured arc/turntable (tape-measure geometry
  is sufficient for circstd/ρ comparisons; no GRBL needed). Same cells, same decision
  rule. Post-processing recovery of historical sub-GHz was tested and is a dead end for
  per-dataset phase (learnings L9: detrend partial, DC-excision null, gain-conditioning
  null).
- **Design (original, for reference):** wall array, one afternoon, same rig/emitter/era. Four capture sessions at
  915 MHz (and optionally a 2.412 GHz control pair):
  {IF = 0, IF = fs/16} × {BBDC tracking on, off}. Few hundred snapshots each.
- **Prerequisite (small code change):** add a `bb-dc-tracking: true|false` receiver key,
  wired in `PPlus.setup_rx_config` (`sdr_controller.py:709-721` — the block that already
  sets `adi,rx1-rx2-phase-inversion-enable` and reg 0x22): set
  `bb_dc_offset_tracking_en` on voltage0/voltage1 and log the setting into the capture
  yaml (report §6b R6). Also set `--fi` explicitly per R1 (f_IF ≥ max(10× crystal ppm
  error, 0.01·fs); fs/16 default).
- **Expected outcome (prediction to test):** BBDC-off adds NO noise — it trades the
  loop's time-varying notch for a quasi-static DC spur. With IF=fs/16 the spur's bias on
  the phase product is ~(offset/signal)^2 ≈ 0.3% (~0.003 rad) — invisible; offsets step
  at AGC gain changes (discrete, not drift) and are removable in post from recorded IQ
  (unlike the loop's unlogged correction). At IF=0 with BBDC off, expect a STATIC bias
  (absorbed by the φ₀ fit) instead of drift — better than tracking-on but not clean.
  RISK TO QUANTIFY: the residual spur grows with RX gain (LO self-mixing is amplified
  with the signal), so it is largest exactly when the signal is weakest; at max gain it
  could come within 10-20 dB of a weak signal, where it both biases the phase product
  and steals AGC headroom (AGC regulates total power incl. spur → signal share of the
  12-bit range shrinks). Amplitude-bit cost is negligible below ~20% FS offset
  (log2(2048/(2048−|c|)) ≈ 0.07 bits at 5% FS).
- **Add to the protocol:** a manual-gain sweep (min→max gain index) with BBDC on/off,
  recording DC spur magnitude (dBFS) per gain — converts the unknown offset-vs-gain
  curve into data; decides whether BBDC-off is safe for weak-signal (rover) captures or
  only for strong-signal wall sessions.
- **Decision rule:** run the quality scanner on the four cells. If {IF=fs/16} recovers
  corrected circstd to ≈0.4–0.5 (2.46x-like) regardless of BBDC → IF placement is
  sufficient, commission full sub-GHz re-capture with R1. If only {IF=0, BBDC off}
  improves → tracking loop confirmed as the mechanism; both knobs become policy. If
  nothing improves → residual sub-GHz problem is era/hardware; go to bench (E-HW1).

## E-HW1 — bench VNA S21-vs-distance sweep of the antenna mounts

- **Motivation:** learnings L5 — the mutual-coupling model for g(d) fits 2.4/5.8 GHz
  medians (rmse 0.04–0.12) but A, ψ₀ are lumped and never bench-validated; competing
  mechanisms (phase-center shift, mount scattering) can't be excluded from fleet data.
- **Design:** VNA S21 between the two elements on the actual mounts, sweeping physical
  spacing at 2.4 GHz (and 5.8/915 if time permits). One afternoon.
- **Decision rule:** if measured C(d) matches the fitted A·e^{j(ψ₀−kd)}/(kd), the
  effective-spacing sidecar can be trusted fleet-wide including for spacings never
  collected; if not, the sidecar stays a per-config lookup table (still valid).

## E-TR1 — effective-spacing sidecar training experiment

- **Motivation:** learnings L5 — `rx_spacing_input` is nominal, wrong by up to 2.1× for
  small-spacing 2.4 GHz configs. The network learns around it, but a physical input may
  help generalization across configs.
- **Design:** after the Stage-1 r1/r2 ladder concludes: one run identical to the winner
  but with rx_spacing replaced by effective spacing (config → median g × d from the
  scan; 2.4/5.8 GHz only, sub-GHz excluded per L4). Same steps/schedule, compare on
  `val_clean/single_loss` (+ per-spacing val groups).
- **Non-destructive:** new config + a sidecar table checked into the repo; no dataset
  or split edits.

## E-SC3 — scanner v3 metric upgrades (no new capture needed)

1. **Per-config-family g gate:** |g − median_g(config)| > 0.15 instead of |g−1| > 0.25
   (silences the known coupling floor, catches true per-capture anomalies). (L5)
2. **`QUAR:tone_at_dc` gate:** measure IF per dataset (one FFT of one snapshot, ~free)
   and quarantine |IF| < 0.002·fs. Stronger, physical predictor of drift failure than
   any downstream statistic. (report §6b R5)
3. **Phase-first status rule:** NaN becomes a pure duty DESCRIPTOR (never a quarantine
   cause); QUARANTINE gates on valid-part phase quality (wall: mean circstd_corr > 0.85
   or n_valid < 100; rover: > 1.1). Validated on v2 data: flips only ~6 wrongly-NaN-
   condemned datasets to keep and ~175 low-NaN phase-junk (mostly Jan-25 sub-GHz) to
   quarantine; outcome-equivalent to v2 for 98% of the fleet but causally correct. (L8)
4. **Beamformer-based metrics:** offset-corrected GT-bin percentile (alignment) +
   entropy (informativeness) from the cached `weighted_beamformer` — scores datasets on
   the representation the NN actually consumes; works where scalar g fits fail. Must fit
   the per-dataset offset (bin shift) first, else offset confounds informativeness. (L3)

## E-CAP1 — capture-metadata hygiene (with the next capture campaign)

- **hw-serial recording:** Pluto never records its serial (BladeRF configs do, unverified).
  One-line runtime read (`self.sdr._ctx.attrs["hw_serial"]` in `PPlus.__init__`,
  `sdr_controller.py:610`) + inject `receiver["hw-serial"]` into yaml_config before
  dataset creation — flows into the zarr config blob and sidecar with no schema change.
  Enables attributing per-unit systematics (g, φ₀) to physical radios instead of IPs.
- **Log configured f_IF** alongside rx_lo (report §6b R6) so nominal-vs-measured IF is
  auditable.
- **Gain-in-IQ (bigger project):** embed the real-time gain index (CTRL_OUT bits) into
  IQ LSBs via custom firmware — the pgreenland v0.38 timestamp fork proves the build/
  flash path works in-house. 80/20 alternative: manual gain per capture session.

## E-DATA1 — staged data-quality training ladder  (RUNNING as of 2026-07-12)

Stage 1: base / r1(label-clean, 1630) / r2(no-degraded, 1217) × 250k steps, sequential;
kill >3% behind on `val_clean/single_loss` at 250k, >1.5% at 500k; resume survivors
+250k per stage to 1M. Baselines (jun26 checkpoint): val 0.09735 / val_clean 0.1014 /
val_degraded 0.10902 / val_915 0.11003. Runner: `checkpoints/jul12_2026/stage1_runner.sh`.
