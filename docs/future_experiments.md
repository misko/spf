# Future experiments

Queued, concrete experiments with motivation, design, and decision rules. Read together
with `docs/learnings.md` (the findings that motivate these). When an experiment runs,
record the outcome in `learnings.md` and mark it here.

## E-GSC10 — can we detect the rover gain-phase term *at all*? — 📋 **DESIGNED, NOT RUN (2026-08-14)**

**Why this exists.** `docs/gain_phase_rover_investigation_20260814.md` twice claimed the
rover gain-phase question was closed and twice retracted it. The second attempt
(`gain_fixed_effects.py`) was underpowered by construction: its statistic moves by at most
+0.032–0.074° for the deployed donor's true 1.78–2.70° rms (the "1.0–1.9°" it was given is a
MAD, not an rms), against a seed sd of 0.033°. **The open
question is not "is there an effect" but "can any statistic on existing data see one".**

**No new capture is required for stages 1–3.**

**Stage 1 — audit the two load-bearing numbers (hours).** The entire surviving cost/benefit
argument rested on (a) the 1.0–1.9° of correctable gain-phase variation and (b) the
°-bearing-per-°-phase conversion. Neither has been independently checked. *If (a) is wrong
by 2×, the priority call changes.* **Both were audited on 2026-08-14 — see Retraction 3 in
the phase-correction report. (a) is a MAD, not an rms (true rms 1.78–5.65°); (b) is not a
constant at all. The derived "≤1.4%" ceiling is withdrawn and replaced by the direct
end-to-end measurement. This stage is DONE.**

**Stage 2 — replace the statistic (the actual experiment).** A **one-parameter circular
projection** α̂ of the geometry-conditioned residual onto a *hypothesised* LUT shape, with a
**capture-clustered** bootstrap. Break-even amplitude **0.27°** versus the free fit's
1.87–4.83°, so it is the first instrument in this investigation able to see a 1–2° term.
- β̂ = the rover's gain-phase amplitude along that shape, in degrees rms
- **⚠️ RUN 2026-08-14, and INCONCLUSIVE.** β̂ = **+1.18° rms, capture-bootstrap 95% CI
  [−0.675, +2.433]**, 88% of resamples positive, injection-recovery slope 0.99 (unbiased),
  held-out change +0.0027 ± 0.0046° — not a detection. The deployed table's own **2.08° rms**
  sits *inside* that CI, so the effect is neither detected nor excluded. The CI is
  **capture-limited, not frame-limited** (the naive 1/√N sd understates the true sd by 3.1×);
  doubling 42 → ~84 captures would roughly halve it. *(An earlier version of this entry cited
  "α̂ = 0.814, CI [+0.595, +1.018]" from a prototype that was never committed. Withdrawn.)*

**Stage 3 — the controls Addendum 3 lacked.** Per-capture CI, fold-seed spread, `min_n`
sweep, and a **run-preserving null**: circular shift of the gain-key sequence within each
stream. A plain within-stream shuffle is *not* adequate — gain is held for long runs and the
residual has lag-1 autocorrelation 0.573, so shuffling destroys the run structure and
inflates significance.

**Fix first, before any re-run** (both are confirmed defects, neither caused the result):
- the dedup in `gain_fixed_effects.py` / `geometry_conditioned.py` drops 6 merged stores —
  9,274 frames, 12 streams, 6.9% — that are **disjoint in time**, not duplicates. Concatenate
  per RX capture instead of keep-first.
- `arm` and `rfblock` sum *marginal* conditional means rather than fitting jointly (exact 2×
  overcount at g₁ = g₂; 1.02× inflation measured here). `rfblock` also double-counts within
  one arm — mixer and LNA words are both functions of the same gain index, both signed +1.

**Decision rule.** If Stage 2 gives β̂ with a CI excluding 0, the gain term is real on rover
radios and E-GSC10b (same-radio bench LUT, **requires new capture**) becomes justified on
physics. If the CI covers 0 *and* is tight enough to exclude the deployed 2.08°, the donor shape is
excluded on rover data — a genuine null, unlike anything obtained so far. If the CI is wide,
report that the corpus cannot answer it and stop **on power**, not on a claimed absence.

**Out of scope without new capture:** any sample-weighted gain-trajectory model.
`gain_observation_*`, `gain_event_*` and `sample_counter_end_exclusive` are v7 *schema*
fields that this firmware never wrote — they appear in **0 of 96** receiver groups, and
`first_gain_change_sample` is the −1 sentinel in 383,686 of 383,688 cells. That needs
protocol-v3 firmware.

---

## E-CAL1 — resolve the RF-DC vs RF-state confound — ✅ **BOTH ARMS COMPLETE (2026-08-07)**

**Outcome: H₀ — the RF-DC machinery injects no resolvable phase.** The row-23 (+9 dB)
edge was sampled at 2 radios × 3 high-band LOs × 25 epochs (1050 frames, all gates
passed). With the LPF ramp differenced out pairwise, `H(9) − mid[H(8), H(10)]` =
**+0.069° ± 0.077**, cluster-robust 95% CI [−0.168°, +0.392°], against the 2.664°
median mixer step. The like-for-like check is starker still: the step that *raises*
`RF_DC_CAL` (median |ΔH| 0.320°) is *smaller* than the one that *lowers* it (0.446°).
The attribution closes to the LNA/mixer/TIA network and the hedge is retired
everywhere. Report, code and hashes:
`spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260807_v1/`;
learning recorded in `docs/learnings.md` L10.

**Arm 2 — RUN 2026-08-07. Also null: +0.019° ± 0.082**, CI [−0.258°, +0.250°]; arm 1
minus arm 2 is +0.050° ± 0.113 (t = 0.44), so the continuous tracking loop is not the
source either. Report: `reports/e_cal1_arm2_rfdc_tracking_20260807_v1/` (R18 misses the
strict gate on one marginal cell). The code blocker was fixed first: `rf_dc_offset_tracking_en` is now a tri-state config knob with a **verified**
write path (the driver can accept that attribute write without applying it, so it is
read back and a mismatch aborts the capture), `never` joins the calibration-policy
enum, both states are recorded in V7, and the config is
`configs/e_cal1_arm2_rfdc_tracking.yaml` (arm 1's schedule byte-for-byte). Covered by
12 unit tests plus 4 hardware tests (`--radio-hardware --radio-rf-dc-tracking`), the
latter run green against both radios. Existing run signatures are unchanged.

As predicted, it returned "also zero" — it partitioned a quantity arm 1 had already
measured as indistinguishable from zero.

**The positive control (E-CAL5) ran the same day and closes the gap.** ✅ A 1 dB MIXER step
of independently established magnitude measures **7.434° ± 0.097** against a **0.440°/dB**
floor from the same capture — **16.9×**, resolved in all six (radio, LO) cells. An H₁-sized
2.664° effect would therefore have appeared at **34.5σ** (arm 1) / **32.4σ** (arm 2).
**Both nulls are now informative about the physics**: the RF-DC machinery is quiet, not
invisible to this chain. Report: `reports/e_cal5_positive_control_20260807_v1/`; design in
`experiments/e_cal5_positive_control/`.

Also unrun: the row-11 (−3 dB) edge, still resting on the ≲0.7° `F_neg` bound.

**Design (as run):** [`experiments/e_cal1_rfdc_discriminator/`](../experiments/e_cal1_rfdc_discriminator/experiment_readme.md)
— purpose, hypothesis, schematic, parts list, commands, outputs and gates. Config at
`spf/calibrations/dual_rx_gain_frequency/configs/e_cal1_rfdc_discriminator.yaml`
(validated: 525 frames per radio; actual bench time ~25 min for both radios).
See the report's §6 for four runbook defects found while executing it.

**Motivation (L10).** Gain-table byte 2 bit 5 is `RF_DC_CAL`, and it is set on exactly the
rows that begin a new LNA/mixer/TIA state. So "the LMT words changed" and "the RF-DC
correction was re-run" are confounded in nearly every capture.

**What the campaign already shows.** The high table has two rows where `RF_DC_CAL` toggles
with the LMT words frozen (row 11 = −3 dB, row 23 = +9 dB). The **row-11 edge was in fact
sampled**, inside the stage excluded elsewhere as an abandoned attempt
(`F_unsupported_negative_gain_attempt_20260730`), which is complete and quality-valid in
the high band at 5766/5866 MHz over −10…26 dB. Measured there:

| 1 dB step at 5766/5866 MHz | n | median &#124;ΔH&#124; |
|---|---:|---:|
| `RF_DC_CAL` toggles, LMT frozen | 24 | 0.722° |
| — rising edge only (entering row 11) | 4 | 0.333° |
| LPF word only, `RF_DC_CAL` frozen | 32 | 0.473° |
| LMT change (`MIX 0→1`), same LOs | 12 | **4.364°** |

Mann-Whitney: RF-DC-only vs LPF-only p = 0.849 (indistinguishable); LMT vs the rest
p = 1.0e-5. **So an RF-DC-only step is already bounded at ≲0.7°** against a 4.36° LMT step.

**What is still missing.** At n = 4 rising edges against a ~0.5° per-step floor this cannot
reach a 0.35° decision rule, and the second, higher-SNR edge is genuinely unsampled: gains
**8 and 9 dB appear at no high-band LO in any stage of either campaign** (only 10 dB does).

**Design.** Additive-cross around a 5 dB reference, gains {8, 9, 10} — the row-23 edge, at
much better SNR than row 11 — at 4001 / 5100 / 5766 MHz, high table only. Use **≥16
epochs**, not 3: the measured per-step standard error at 3 epochs is 0.54–0.81°, so a 0.35°
rule is unreachable without ~16–25 repeats. Second arm: repeat with
`rf_dc_offset_tracking_en = 0` to A/B the tracking loop directly.

**Decision rule.** With the sem driven under 0.35°: a step at +9 dB comparable to the
2.664° median mixer step means the RF-DC machinery injects phase on its own and the model
needs an `RF_DC_CAL`-indexed term. A step at or below 0.35° closes the attribution to the
LNA/mixer/TIA network. Report the sem alongside the estimate so the power is auditable.

## E-AGC1 — AD9361 gain-pin and detector bring-up — ✅ **3.5 of 4 OPEN ITEMS CLOSED (2026-08-11)**

Design and results: `experiments/e_agc1_pin_and_detector_bringup/`. Both radios, stock RC17,
userspace only — no FPGA change, no bitstream, no QSPI write. ~4 min per radio.

- **O-5 pin mapping CLOSED.** The map is the identity, 40/40 trials across two radios, other
  channel never moved once. RTL work unblocked.
- **O-1 ENSM CLOSED, and it changes the contract.** `CTRL_IN` edges are honoured in `fdd`
  and **ignored in `alert` and `sleep`** (0/3 each), so the enable sequence must guarantee RX
  is active before arming. Incidentally `wait` is advertised but unreachable.
- **O-2 hold band CLOSED at 21–22 dB** across four arms — 22× wider than the 1 dB at which
  the pre-registered rule said the policy would oscillate. That predicted change did not
  happen.
- **O-3 HALF closed.** H4's *latch* is confirmed on both radios: with the gain held and the
  tone **muted entirely** the overload bits stay asserted, so the bit holds until a gain
  change. The blank *duration* is bounded under 322 µs and unmeasured, and H5 has no dither
  to time at all. **Both need the FPGA stage — do not re-attempt from userspace.**
- **Unplanned, and it matters for SPF:** arming `0x0FB` takes gain ownership away from
  software *silently* — a `hardwaregain` write returns rc=0 and is dropped. Any host
  `set_gains()` during tandem operation is a no-op.
- All 8 `CTRL_OUT` bits characterised across the pair; `0x114` identified at 0.5 dB/LSB, while
  `0x107`/`0x108` do not resolve because the ADC saturates within ~1 dB.

**Still blocked:** `TANDEM_AGC_V1_DESIGN.md` is not in this repository, so O-1/O-2/O-5 cannot
be formally marked closed there. CH1's large-LMT bit on R17 needs ~20 dB more drive (a 10 dB
pad instead of 30 dB).

## E-HCP1 — can the bare-tee harness account for the arm-specific residual `A`? — ✅ **NO (2026-08-11)**

Design and results: `experiments/e_hcp1_cross_arm_coupling/`. Created because the 2026-08-10
harness finding (the "splitter" is a bare SMA tee, ~0 dB port-to-port isolation) put the whole
dual-RX phase programme in doubt, and E-GSP2 — the definitive A/B — needs parts.

Hold one arm's gain fixed, sweep the other, watch the **fixed** arm's input-referred RSSI:
coupling is **≤1.25 dB, median 0.50 dB** over 12 LOs from 433 MHz to 5900 MHz, and
**frequency-flat**, while `A` rises ~5× into the high band. **R18 reproduces R17 exactly**, so
the coupling is *not* unit-specific whereas `A` is — a second independent argument. Bounds
phase coupling at ≲8.9° worst case, ~3° typical.

**Consequences:** E-GSC6 was unblocked and run on this harness. **E-GSP2 remains the
definitive tee-versus-divider A/B but is no longer urgent and blocks nothing.** A first
first-order estimate of 17–39° in `learnings.md` was too pessimistic and has been corrected.
The high band carries the weakest evidence, since Pluto TX rolls off above 4 GHz — a 10 dB
pad would firm it up.

## E-GSC6 — does the equal-gain anchor move with gain index? — ✅ **RUN (2026-08-11)**

Design and results: `experiments/e_gsc6_equal_gain_diagonal/`; full report in
`spf/calibrations/dual_rx_gain_frequency/reports/equal_gain_diagonal_20260811_v1/`. Both
radios on RC17 via the `automate` flow, 8,784 frames, **100.00% quality-valid**.

- **Separability HOLDS.** `C(g,g)` ≈ 0.5° per cell on both radios (n=480 each) with no
  LNA-state structure — at the 0.355–0.368° noise floor. No material interaction term.
- **But tandem does not null the differential phase.** `D(g,g)` is not zero and *is*
  LNA-state structured, so tandem leaves `D(g,g) = A(g)`. **The residual model stays
  required, not a fallback** — though `C ≈ 0.5°` means a per-arm term indexed by LNA state
  captures it.
- **Measured acceptance threshold** (from R18, the untouched control), replacing the
  extrapolated projection: `D(g,g)` **≥7.2× / 8.6× / 2.3×** against the 6.65° anchored
  baseline for low/middle/high, versus 6.0× / 5.3× / 3.4× projected. **Worse than projected
  above 4 GHz**, the band that matters most. Low band is a **bound** — it sits below its own
  median anchor drift.
- **Harness health dominates.** R17 is 5–7× worse in every band while its `C` matches R18's;
  its high band is **0.3×**, worse than doing nothing. Independent evidence that R17's
  assembly is degraded.
- **H2 supported, restated.** `|D(g,g)|` is flat within an audited LNA state and steps
  between states. The original "transition vs frozen-word control" phrasing was untestable —
  a control gain sits inside whichever plateau its neighbours occupy.
- Anchor-drift gate **PASSED**; it cannot explain R17 (0.63° median vs 20.4°).

**Follow-ups this created:** re-terminate R17's harness and re-run the diagonal as a
before/after; test whether the diagonal's frequency structure is a reflection (a free partial
**E-CAL4** from data in hand, no VNA needed); and `TANDEM_AGC_V1_DESIGN.md` must exist before
the threshold can be handed off.

## E-CAL2 — fill the unmeasured LNA states, then retest band portability

**Status (2026-08-07): completed; precision gate failed.** The 444-frame targeted
capture filled the missing states. L26 coverage rose to 91.50% but its augmented
leave-one-band-out MAE was 5.58°; L30 reached 100% coverage at 4.83°. Missing
state coverage is no longer the main explanation: every required gain-table band
must be sampled directly for precision correction. See §8.1 of the
gain-state-phase-model report.

**Motivation (L10).** Band portability failed: train on two gain-table bands, predict the
third, and no model beats baseline by more than 8%. Part of that is genuine frequency
extrapolation, but part is a campaign coverage hole — **LNA index 1 was never measured
in the A–G campaign**, and LNA index 3 was measured there only in the high band. The
separate 2.4 GHz integer-gain experiments reach LNA index 1 at two middle-band LOs, but
cannot make the frequency-spanning model band-portable.

**Second motivation.** The A–G campaign's 1 dB-step statistic contains **zero LNA
transitions** — its only LNA changes are four 9 dB steps. Existing 2.4 GHz integer-gain
data already show adjacent LNA 1→2 and 2→3 steps of 2.6–16.7°, so the claim is not
untested repository-wide. Reanalyse that committed data first; the new capture is needed
to span all three gain-table bands and separate coverage from frequency extrapolation.

**Design.** Use band-specific probe gains so every transition is actually bracketed:

- low-band LOs: {30,31,32,33,51,52};
- middle-band LOs: {29,30,31,32,49,50};
- high-band LOs: {22,23,25,26,40,41}.

Run these on the existing 6-LO operating set, additive-cross around 26 dB, 3 epochs.
This is 222 frames per radio: 74 cells per epoch across the three low-, one middle-,
and two high-band LOs. Then re-run the pooled leave-one-gain-table-band-out in
`reports/gain_state_phase_model_20260802_v1/analysis/run_band.py`.

**Decision rule.** If leave-one-band-out drops below ~3° MAE at ≥90% coverage, the
hardware-state parameterisation is genuinely band-portable and a single fleet model can
cover 400–5900 MHz. If it stays near baseline, band portability is an extrapolation limit,
not a coverage limit — and every operating band must be sampled directly.

## E-CAL3 — prospective coarse-comb confirmation

**Status (2026-08-07): completed; ten-LO claim rejected.** A fit using exactly
the ten pre-registered training LOs gave 11.61° MAE on the other 103 LOs, worse
than the 9.06° anchor-only baseline. The committed dense-fit L26 coefficients
gave 4.79–4.80° MAE. A pre-reboot-only analysis reproduced the failure at
11.57°, so the mid-run TX2/DDS recovery reboot is not its cause. The earlier
comb analysis held out blocks while training on most remaining dense LOs; it was
not an exact ten-frequency calibration simulation.

**Motivation (L10).** Subsampling the 113-point comb shows held-out error is flat for gaps
from 96 MHz to ~690 MHz, implying a ~10-point comb suffices for the gain-dependent term.
That is a retrospective subsample of one dense capture, not a prospective test.

**Design.** In one uninterrupted, randomized session, capture the full 113-LO stage-A
comb but pre-register ≈{400, 1000, 1600, 2200, 2800, 3400, 4100, 4700, 5300, 5900} MHz
as the only ten training LOs. Fit the 27-column stage-A model using those ten LOs and
score it only on the other 103 LOs from that same session. Interleave equal-gain anchors
and repeat the ten training LOs at the end so early-to-late drift is measured separately.
This one prospective validation still pays the dense-capture cost; if it passes, later
radio calibrations need only the ten-point comb.

**Decision rule.** Held-out unequal-gain MAE ≤3° at 100% stage-A coverage, with the
early/end training-comb drift inside the unchanged-harness repeatability bound, confirms
the ~12× calibration-time reduction. Anything above ~4° after accounting for measured
drift means the retrospective subsample was optimistic. Do not test against an older
session: §4.6 already shows that would confound comb sparsity with session drift.

## E-CAL4 — is the arm asymmetry a cable-length difference?

**Motivation (L10).** The gain response is 94–99% antisymmetric; the residual 1.3–6.0% is
arm-specific. The reflection mechanism predicts that residual is itself a ripple whose delay
equals the RX1/RX2 external path-length difference.

**Design.** Use a VNA-characterised length (e.g. 15 cm, with measured group delay over
the whole band) on treated-radio RX1 only. Run an ABABA sequence without changing the
untreated arms: original baseline → jumper → restored baseline → jumper → restored
baseline. Record connector torque and pre-register whether the spectral prediction uses
one-way or round-trip delay. Then run the separate RX1↔RX2 cable-swap discriminator.
Predict: `A(f,g) = D(g,26) + D(26,g)` gains the same treatment-specific ripple component
during both jumper insertions, while `H(f,g)` and the control arms remain comparatively
unchanged.

**Decision rule.** Both jumper stages must show the pre-registered component, both
restorations must return within unchanged-harness repeatability, and the component must
be absent from the untreated arms. The cable-swap result must follow/reverse with the
external path rather than remain attached to the radio. Any failed restoration leaves
the physical attribution inconclusive. This is the controlled version of the cable-swap
test that `FREQUENCY_SCOUT_20260727.md` proposed and never ran.

# Gain-state model: two parallel programs

E-CAL2 and E-CAL3 both completed on 2026-08-07 and both failed their precision gates.
What survived is narrow and worth stating exactly, because the two programs below are
scoped by it:

- The **committed** L26 coefficients transfer. On a fresh 103-LO holdout they improved a
  9.06° anchor-only baseline to 4.79–4.80°. That is real, prospective, and the only
  validated claim the model currently has.
- **Refitting** L26 from ten uniformly spaced LOs does not work — 11.61°, worse than no
  model. The nonlinear frequency basis is not identifiable from ten points.
- Band portability failed **even after** the missing gain states were filled, so it is a
  frequency-extrapolation limit, not a coverage hole.

The **computational program (E-GSC1–5)** runs entirely on data already captured and
should be exhausted before booking bench time — it can resolve identifiability, band
portability, `h_tia`, the LNA attribution and the choice of shipped default without a
single new frame. The **physical program (E-GSP1–6)** targets what no re-analysis can
reach: whether the reflection mechanism is real, whether it survives a different harness,
and what drives the >4 GHz degradation.

Run E-GSC1 and E-GSC4 first. They need no new captures — they run on committed fitted
coefficients, not on raw IQ, which is a real limitation and is scoped in E-GSC1 — and
their outcomes change what the bench experiments should be.

~~**Before any of it, update `docs/learnings.md` L10.**~~ ✅ **Done 2026-08-07.** L10 now
carries: the refuted sparse-comb claim struck out with E-CAL3's 11.61° and E-GSC2/3's
diagnosis (uniform 600 MHz spacing aliased the ripple delays; N\* = 16 free / 8 frozen;
a sparse protocol still needs a *prospective* conditioning-chosen capture); the
2.26°/2.22° headline corrected to the prospective 4.79–4.80° and restated as a
convention-invariant ratio (~1.9× transfer, ~2.9× same-session refit); E-GSC4's 318
adjacent-1 dB LNA transitions at 44.46× the LPF floor; `h_tia` marked for dropping; and
band non-portability confirmed as a real extrapolation limit rather than a coverage hole.

---

## E-GSP7 — conditioning-chosen sparse comb — ✅ **COMPLETE (2026-08-07)**

**Answers the E-GSC ledger row "Is a sparse protocol now recommendable?", which said
"Not yet — it needs a prospective sparse capture with the comb chosen by conditioning."**
That capture ran: 111 LOs, 3,330 frames, one unchanged-harness session, combs
pre-registered before the first frame, every arm at 100% state coverage.

**Answer: yes, but only as a conjunction.** A ten-LO comb calibrates **only** when it is
chosen by conditioning **and** fitted with the ripple delays frozen at fleet values:

| ten-LO comb | free-delay | frozen-delay |
|---|---|---|
| chosen by conditioning (cond 1.09) | 8.35° ✗ | **4.95° ✓** (baseline 7.48°) |
| the actual E-CAL3 comb (cond 17.92) | 9.37° ✗ | **23.50° ✗✗** |

Drop either condition and it is worse than applying no model. If the delays must be
fitted, the floor is **N\* = 16** — and chosen-16 is the only arm that recovers the true
delays, (2.50, 0.98) ns against fleet (2.56, 0.92).

**Unpredicted and operationally important: freezing the delays on a badly-conditioned
comb is actively dangerous.** It helps the good comb (8.35 → 4.95) and wrecks the aliased
one (9.37 → 23.50), because with free delays the fit escapes by moving τ somewhere less
collinear and freezing removes that escape. Never freeze without checking conditioning.

**Prefer the committed coefficients anyway.** With no refit they score 3.86° (1.93×) on
this same fresh capture — better than the best ten-LO refit. The sparse refit earns its
keep only where a genuinely local fit is required.

Four prior predictions confirmed prospectively: E-CAL3's failure reproduces at the same
ratio (1.287× vs 1.281× worse than baseline); E-GSC's 73.4% frozen-delay recovery measures
70.4%; N\* = 16 free; and the "~1.9× on transfer" headline measures 1.93×.

**Still open:** evaluation was within-session, so this establishes identifiability and
conditioning, not session-to-session robustness. A cross-session repeat is the remaining
step before a sparse protocol is deployable. Report:
`spf/calibrations/dual_rx_gain_frequency/reports/e_gsp7_conditioned_comb_20260807_v1/`;
design in `experiments/e_gsp7_conditioned_comb/`.

---

# Computational program — existing data only, no new captures

## E-GSC1 — mine the wide 53-LO integer-gain survey with the gain-state parameterisation

**Motivation.** Every gain-state result to date comes from the A–G campaign, whose stage A
carries only three requested gains. The wide integer-gain survey
(`wide_integer_gain_cross_band_20260730_v1`) is 55,650 frames: 53 exact LOs from 433 MHz
to 5.9 GHz, **every common integer gain from −1 to 62 dB on both receiver axes**, 48
off-axis held-out pairs per frequency, 3 separated epochs, both radios. No gain-state
analysis script has ever referenced it. It is the richest gain-axis data in the
repository.

**Its raw stores are gone; its fitted coefficients are not.** Verified 2026-08-07:
`artifacts/dual_rx_gain_frequency/` does not exist, and the only `calibration.v7.zarr`
stores anywhere under `/mnt` are the two spectroscopy campaigns plus the 2026-08-07
follow-up. The survey's raw IQ was last recorded on a Raspberry Pi. **But**
`reports/wide_integer_gain_cross_band_20260730_v1/model_matrix.json` is committed (3.6 MB)
and holds, per radio, `models.frequency_specific_additive_gain_per_radio.fits[i].
coefficients_rad` — **6,731 fitted coefficients** over 53 LOs (433–5900 MHz, all three
bands) and 64 integer gains (−1…62 dB), plus `identifiability` and
`frequency_scaling_structure` blocks.

So this entry runs on **fitted curves, not raw observations**, and must be scoped
accordingly: no leave-one-epoch-out (epoch structure is not in the file), no re-anchoring,
no re-derived quality masks, and every conclusion inherits the source fit's assumptions.
Within those limits it still covers every integer gain in all three bands — which is what
breaks the `h_tia`/MIXER collinearity, reaches LNA index 1, and supplies adjacent-1 dB LNA
steps at 53 LOs rather than the campaign's zero. Restoring the raw stores from a Pi backup
would additionally re-enable epoch holdout and the committed report's own reproduction
command, which currently needs `--prior-calibration-root` pointing at the missing survey.

**Design.** Extend the extraction to the survey stores (read-only). Attach the equal-gain
anchor per (radio, LO, epoch). Refit the ladder rungs L00/L01/L05/L06/L16/L26/L27/L30/
L31/L33 plus the per-frequency LUT reference. Report LOEO / LOFO / LOBLK / LORO / LOBAND
with coverage beside every error, and — the cleanest test available anywhere in this
corpus — the **48 genuinely off-axis pairs per frequency**, which were excluded from
fitting by design. Report the symmetric-minus-independent error gap, as that report
mandates for every analysis.

**Decision rule.**
- **`h_tia`:** dense gains break the stage-A collinearity (TIA 0 occurs only at 5 dB,
  which is also the only MIXER 1 cell). If `h_tia` becomes separately identifiable and its
  magnitude sits at or below the 0.355–0.368° measured noise floor, drop it and re-declare
  L26 with one fewer family.
- **Band portability:** if leave-one-band-out still exceeds ~3° at ≥90% coverage with the
  survey's full state coverage, band non-portability is confirmed as a frequency-
  extrapolation limit independent of any campaign coverage hole, and "sample every
  operating band directly" becomes permanent policy rather than a provisional finding.
- **Off-axis:** if the hardware-state rungs approach the 1.31–1.36° independent-curve
  off-axis reference, the parameterisation is capturing the gain structure and not just
  the ripple.

## E-GSC2 — the identifiability curve: how many LOs does L26 actually need?

**Motivation.** E-CAL3 gives exactly two points: 10 LOs fails at 11.61°, 113 LOs works.
Nothing in between, and no diagnosis of *which* parameters fail first. The known failure
mode is specific — the ten-LO refit put the delays at 4.15 ns and 0.16 ns instead of
2.56 and 0.92 — which points at the nonlinear terms rather than the linear ones.

**Design.** On both the wide survey (53 LOs) and the A–G dense set (113 LOs), refit from
N ∈ {6, 8, 10, 12, 16, 20, 24, 32, 48} LOs and score on all held-out LOs. Two variants
per N: **(a) delays free**, **(b) delays frozen at the fleet values**. Repeat each N over
many random LO subsets to get a distribution, and separately over a D-optimal subset
chosen for identifiability. Plot MAE vs N with subset spread, and report the fitted
delays per fit so the failure mode is visible rather than inferred.

**Decision rule.** Define N\* as the smallest N at which the free-delay refit beats
anchor-only in ≥90% of random subsets.
- If frozen-delay fitting reaches the committed-coefficient level (~4.8°) at N far below
  N\*, the "learn the basis once fleet-wide, fit only linear terms per unit" architecture
  is validated and the achievable bench-time saving is quantified. That becomes the sparse
  calibration protocol, to be confirmed prospectively before any claim is made.
- If even frozen-delay fits need large N, sparse calibration is closed — say so
  explicitly, the way sub-GHz recovery was closed in L9, rather than leaving it open.

## E-GSC3 — decompose the 2.26° → 4.8° gap between cross-validation and prospect

**Motivation.** The retrospective cross-validation said 2.26°; the prospective test said
4.79–4.80°. That factor of two now governs how the model is advertised, and nothing
explains it. Note the baselines moved too — 6.65° retrospective against 9.06°
prospective — so the new session may simply have been harder.

**Design.** All existing data. (a) Score the committed coefficients on the campaign's own
A→G, A→D, A→B and A→C transfers to get a session-drift distribution. (b) Compare the
prospective 9.06° anchor-only baseline against the campaign's per-session baselines.
(c) Re-score the prospective data restricted to the cells the retrospective folds also
covered, so the comparison is paired.

**Decision rule.** If the 9.06 → 4.8 improvement ratio matches the 6.65 → 2.26 ratio
within fold noise, the model is behaving as designed and only session difficulty changed
— the headline should then be stated as a **ratio**, not an absolute degree figure. If
the ratio itself degraded, there is genuine transfer loss, and the next question is
whether it tracks the reboot, the harness, or elapsed time.

## E-GSC4 — the adjacent-1 dB LNA discriminator, from committed 2.4 GHz data

**Motivation.** L10 finding 2 rests on 12 mixer steps and **zero** LNA steps; the LNA's
role currently rests on four 9 dB steps and on the ripple. But three adjacent-1 dB LNA
transitions are already captured at 2412/2467 MHz by the integer-gain runs, which swept
every integer gain: LNA 0→1 at 29→30 dB, 1→2 at 31→32, 2→3 at 49→50. They were reported
as raw phase steps and never decomposed into `H`.

**Design.** Recompute `H = [D(g,26) − D(26,g)]/2` from the additive-cross axes of
`ALL_GAIN_CROSS_2P4_20260729` and `INTEGER_GAIN_CROSS_2P4_20260729`, then run the §3.3
discriminator, classifying every adjacent step by which audited word moves. Each LNA step
there is bundled with an LPF move and an `RF_DC_CAL` edge, so derive the **LPF-only floor
from the same dataset** as the control rather than importing the campaign's. Report
cluster-bootstrap CIs over (radio, LO) as §3.3 does. These are different sessions from the
campaign, so report them as a separate corroboration — do **not** pool them into the
campaign's `H` statistics.

**Decision rule.** If |ΔH| for the LNA steps exceeds the same-dataset LPF-only floor by
≥5×, the LNA claim is established at 1 dB resolution and L10 finding 2 should be rewritten
to say so. If it is comparable to the floor, the LNA's role genuinely rests on the ripple
and the 9 dB steps alone, and every document should say that plainly.

## E-GSC5 — should L30 be the shipped default instead of L26?

**Motivation.** E-CAL2 found L30 (8 parameters) reaching 100% coverage at 4.83°
augmented leave-one-band-out against L26's 5.58° at 91.50%. L30 also carries no
categorical baseband-LPF family, so it is immune by construction to the rule-5 failure
mode that makes L26 net-harmful where both arms share RF words. The package currently
ships L26 as default on the strength of the retrospective numbers.

**Design.** Score L26, L30 and L31 on **identical masks** across every dataset now
available: A–G stage A, the pooled set, the wide survey (E-GSC1), and the prospective
103-LO capture. Report coverage beside error everywhere, and separately report the
unequal-gain subset.

**The straight swap is already decided — do not re-litigate it.** L26 is 2.262° against
L30's 3.539° on stage-A leave-one-frequency-out, and 2.11° against 2.99° pooled. Any
"within 0.1°" clause fails by 0.88–1.28°, and a conjunctive rule built on it can never
reach the band-transfer evidence at all. (The 0.1° figure is also borrowed from the
*per-radio-family* practical-equivalence margin, against paired per-fold noise of
±0.209–0.237° — it is inside fold noise for this comparison anyway.)

**The real question is whether the default should be band-conditional.** L26 is clearly
better *within* a measured band; L30 is better *across* an unmeasured one (4.83° at 100%
coverage vs 5.58° at 91.50%) and is immune by construction to the rule-5 failure mode.

**Decision rule.** Pre-register on retrospective and survey data only; write down the
predicted winner and margin; then score the prospective 103-LO capture **once**, as
confirmation, never as the selection criterion — that capture is the only clean external
test set the model has, and selecting on it would convert 4.79–4.80° into a selected
minimum and invalidate the one validated claim in the corpus.
- Ship **both** rungs and make the choice band-conditional if L26 wins within-band by
  >0.5° while L30 wins cross-band by >0.5°.
- Require any cross-band winner to beat the **5.71° augmented anchor-only baseline** by a
  stated margin. Both candidates currently sit close to it, so a rule comparing only the
  two candidates could promote a model that is barely better than no model at all.
- Weight L30's rule-5 immunity explicitly rather than leaving it as prose: a model that
  needs no guard is operationally cheaper and cannot be misapplied.

---

# Physical program — new captures

## E-GSP1 — measure `Γ_RX` per gain state with a VNA, and predict the ripple instead of fitting it

**Motivation.** The entire frequency half of the model rests on one hypothesis: an LNA
state change alters the receiver input impedance, and the round trip against a mismatched
source produces a standing wave periodic in frequency. The support for that is a fitted
delay that is consistent across radios, an 11 dB pad that suppressed the 2.548 ns
component by 81.5% on the treated arm only, and an amplitude ordering that inverts across
4 GHz as the gain tables predict. **None of it measures a reflection coefficient.** This
experiment would convert a fitted mechanism into a measured one.

**Design.** With a calibrated 2-port VNA, measure `S11` at the RX1 and RX2 SMA inputs over
400–6000 MHz with the radio powered, tuned to each LO, and its gain forced to a chosen
audited row, for **each LNA index** (and each mixer word reachable without changing LNA).
Separately measure the source-side `Γ_s` looking back into the harness (30 dB pad →
splitter → cable).

**Pre-register the predictor in full, with quadrature.** The report's fitting basis is
written `Re{ρ·e^{−j2πfτ}}` with free `a, b`, which is harmless when both quadratures are
fitted — but E-GSP1 *predicts from a measured complex `Γ`*, where a 90° rotation swaps
`a ↔ b` and flips sign, and the comparison then fails for a purely bookkeeping reason.
For a small reflection the **phase** perturbation is the imaginary part; the real part is
the amplitude ripple. The model also fits a **state difference**, and `H` carries a factor
of one half. So the quantity to compare against the fitted `a_k`, `b_k` is:

```text
ΔΦ_pred(f; l1, l2) = ½ · Im{ [ Γ_RX(l1) − Γ_RX(l2) ] · Γ_s · e^{−j2πfτ_k} }
```

Pre-register the one-way versus round-trip convention for `τ` before unblinding — the
campaign's own jumper test already used round-trip (`2.5475 + 2 × 1.4607 = 5.469 ns`), and
a slip here is a factor of two against a 10% acceptance gate.

Note the committed coefficients contain **no LNA index 1** (`h_deg.lna` has keys 0, 2, 3),
so "each LNA index" has no fitted counterpart at index 1 — either restrict the comparison
to 0/2/3 or pair this with the E-CAL2 fill.

*Safety and validity notes:* use a DC block, keep VNA drive well below the RX damage
threshold and below the AGC/clipping level, verify the forced gain row via the audited
table read-back rather than the requested dB, and re-audit the gain tables before and
after. A powered, tuned receiver is not a passive one-port — record whether `S11` depends
on LO tune state as well as gain state, because that is itself a finding.

**Decision rule.** If the predicted ripple amplitude matches the fitted amplitude within
~30% and the predicted delay within ~10%, the reflection mechanism is confirmed from
first principles, and the ripple term becomes **computable for a new harness** rather
than something that must be fitted per installation — which would be the single largest
practical win available. If the predicted amplitude is far too small to explain the
observed 1.1–10.7°, a different mechanism dominates, the "reflection" language must be
retired from every document, and the ripple term reverts to an empirical basis.

## E-GSP2 — harness-parameter sweep: does the ripple move the way a reflection must?

**Written up:** [`experiments/e_gsp2_pad_sweep/`](../experiments/e_gsp2_pad_sweep/experiment_readme.md)
— purpose, hypothesis, schematic, parts list, outputs and gates. Needs ~$100 of
characterised pads plus a second splitter, and one free computational prerequisite
(re-fit stage B with a per-stage delay search) before the numbers are interpretable.

**Motivation.** One harness, one pad value, one splitter, throughout. The reflection model
makes sharp, falsifiable predictions about what happens when the mismatch or the
electrical length changes — and the second fitted delay (0.92 ns ≈ 151 mm) has never been
attributed to any physical path.

**The existing data does not pin `n` down, because the unpadded reference is unstable.**
Stage B's pad (measured amplitude change −10.49 dB) took the treated arm's 2.5475 ns
component to 0.99°. But the *unpadded* amplitude on that same arm is not one number — it
is 5.34° in stage A, 10.80° in D and 10.40° in G, all nominally the same harness state.
So the implied traversal count `n = suppression_dB / 10.49` spans:

| unpadded reference | amplitude | suppression | `n` |
|---|---:|---:|---:|
| A | 5.34° | 14.65 dB | 1.40 |
| D (restored) | 10.80° | 20.76 dB | **1.98** |
| G (12 h, hot) | 10.40° | 20.43 dB | **1.95** |

**Round-trip (`n = 2`) is therefore entirely consistent with the existing data** — it is
matched by both D and G. Anchoring on stage A alone gives 1.40 and appears to exclude it,
but that is an artifact of the failed A→D restoration, which is exactly why the unpadded
reference must be re-measured immediately before and after *each* pad state rather than
taken once at the start. Measuring `n` properly is the point of the sweep.

**A free prerequisite, in the computational stream.** Stage B's pad also adds its own
insertion delay — the campaign measured +349 / +314 / −213 ps of equal-gain delay change
across the three bands — but every stage's ripple amplitude was read at the *shared*
baseline delays. A displaced sinusoid read at the old delay loses amplitude, so part of
the 14.64 dB may be delay displacement rather than attenuation. Re-fit stage B with a
per-stage delay search before interpreting any `n`. The campaign already did exactly this
check for the 30 cm jumper and never for the pad.

**Design.** Two arms, each an ABABA sequence with restoration checks and recorded
connector torque, on the treated radio only, control radio untouched throughout.
**(a) Pad sweep:** at least four values spanning 3 / 6 / 11 / 20 dB on the treated arm,
each VNA-characterised for actual insertion loss rather than trusted at its nominal value
(stage B's "11 dB" measured 10.49 dB). Fit `suppression_dB = n · L_dB` across the sweep
and report `n` with a confidence interval, **separately for each of the two delay
components**. **(b) Splitter swap:** a different unit, and a deliberately better-matched
unit, to change `Γ_s` without changing length. E-CAL4's characterised-length insertion is
the third arm and stays where it is designed.

**Decision rule.** Pre-register the **two-path mixture**, not a single traversal count.
For a standing wave the pad is either inside the reflecting loop (`n = 2`) or outside it
(`n = 0`); no single geometry gives `n = 1`, so a fitted non-integer `n` on one resolved
peak means an unresolved *mixture* of both. The reflection model therefore predicts a
**saturating** amplitude curve

```text
A(L) = | A_in · 10^(−L/10)  +  A_out |
```

which is curved in `L_dB`, not linear. Fit `A_in` and `A_out` per component.

- If `A(L)` saturates as above with `A_out` small, the reflection model holds
  quantitatively and ripple amplitude for a new harness becomes predictable from insertion
  loss — the practical prize.
- If `A_out` is comparable to `A_in`, a second reflecting interface sits outside the pad
  and must be localised (the splitter swap and E-CAL4's length insertion are the tools).
- **The falsification is `A(L)` flat — no pad dependence at all.** Linearity in `L_dB` is
  *not* the prediction, and treating it as such would be the mistake that the stage-A-only
  `n ≈ 1.40` reading already invites.
- The splitter swap must move amplitude without moving delay. If it moves the delay, the
  splitter is part of the resonant path and the harness model needs another element.

## E-GSP3 — thermal dependence of the ripple delay and of the anchor

**Motivation.** Every result to date shares one temperature history. Session drift is real
and unexplained: across a 12-hour boundary even the 1356-parameter LUT degrades from 0.62°
to 2.74°, so something in the *gain-dependent* term moves, not only the intercept.
Electrical length and reflection coefficients are temperature-dependent in a way that
makes a quantitative prediction.

**Do not measure this as a shift in the fitted delay — that cannot work.** At τ = 2.54 ns,
a typical coax thermal coefficient of 50 ppm/°C over a 40 °C span moves τ by **5.1 ps**,
against a **20 ps** grid step in the delay search. Even a pessimistic 150 ppm/°C over
40 °C gives only 15 ps. The fitted `τ` is the wrong observable by roughly an order of
magnitude.

**The right observable is the ripple phase at the top of the band**, which accumulates as
`2πfΔτ` and is comfortably measurable:

| | Δτ | ripple phase @ 2.4 GHz | @ 5.9 GHz |
|---|---:|---:|---:|
| 50 ppm/°C, ΔT = 20 °C | 2.5 ps | 2.2° | **5.4°** |
| 50 ppm/°C, ΔT = 40 °C | 5.1 ps | 4.4° | **10.8°** |
| 150 ppm/°C, ΔT = 40 °C | 15.2 ps | 13.2° | **32.4°** |

Against a per-step noise floor of 0.355–0.368° and ripple amplitudes up to 10.7°, a 5–30°
shift in ripple phase is a large, unambiguous signal.

**Design.** Repeat stage A at 3–4 controlled enclosure temperatures spanning the operating
range, with a settle period at each and logged die temperature alongside ambient. Weight
the LO set toward **4–6 GHz**, where the effect is 2.4× larger than at 2.4 GHz and 6×
larger than at 1 GHz. Hold τ fixed at the fleet value and fit only the ripple phase per
temperature; report the phase shift versus frequency and check it is linear in `f` as a
delay change requires. Include one return-to-start point to separate reversible thermal
response from monotonic drift, and record whether the *anchor* moves with temperature
independently of the ripple.

**Decision rule.** Fit Δτ from the slope of ripple-phase shift versus frequency.
- If the implied thermal coefficient lands in the 20–200 ppm/°C range that coax and PCB
  dielectrics occupy, **and** the shift is linear in `f`, the reflection attribution gains
  independent physical support and part of the session drift becomes predictable —
  potentially allowing a temperature-compensated anchor rather than a re-measured one,
  which would relax the strictest operational constraint the model has.
- If the ripple phase is temperature-invariant while the anchor still drifts, the drift
  lives elsewhere (LO, calibration state, connector seating) and should be chased there
  instead of being modelled.
- If the shift is not linear in `f`, it is not a delay change and the mechanism is
  something else.

## E-GSP4 — E-CAL1 — ✅ **CLOSED 2026-08-07**

This was the programme's top attribution gap: every statement of the form "the mixer word
moves the phase" had to be read as "the RF-state transition, *including any RF-DC
correction it triggers*". **It is no longer a hole.** E-CAL1 arm 1 measured the RF-DC
contribution at +0.069° ± 0.077 (see the E-CAL1 entry above), superseding the ≲0.7°
n=4 bound. Statements about the mixer word are now plain RF-state statements.

## E-GSP5 — fleet breadth, with harness build as a deliberate variable

**Motivation.** "Nothing needs to be radio-specific" is a two-unit leave-one-out result,
and **both units shared a harness topology**. That means a harness-common effect and a
die-common effect are currently indistinguishable — which matters enormously, because the
ripple is hypothesised to be a harness reflection.

**Design.** At least three additional radios, and critically at least **two different
harness builds** (different cable lengths, different splitter units, ideally different
pad stacks), arranged so that radio identity and harness identity are crossed rather than
confounded. Score the **committed coefficients with no refit** as the primary measure,
plus leave-one-radio-out and leave-one-harness-out.

**Decision rule.** If the committed coefficients hold near 4.8° on new radios with a
*different* harness build, the model is a die property and can ship fleet-wide as-is. If
they hold on new radios sharing the harness build but degrade on a new build, the ripple
is harness-specific: it must then be characterised per harness type — which is exactly
what E-GSP1 would make computable rather than requiring a capture per build.

## E-GSP6 — a focused >4 GHz campaign

**Motivation.** Everything degrades above 4 GHz, and it is not yet known whether that is
one cause or four: arm asymmetry rises from 0.73° to 3.72°, cross-radio correlation of `H`
collapses from ρ≈0.99 to ρ≈0.45, augmented leave-one-band-out reaches 5.59–7.69° in the
high band, and an A→D connector re-mate moved that band by 12–34°. Production capture has
moved to 5.840 GHz, so this is now the operating band, not an edge case.

**Nothing is aliased at the existing spacing — do not justify this capture that way.**
Stage A sampled 400–5900 MHz on a 50 MHz grid: 7.85 samples per 392.5 MHz ripple period,
alias-free to a delay of 1/(2·50 MHz) = **10 ns**, against fitted delays of 2.55 and
0.92 ns. Finer spacing is only warranted if you name a **longer** delay you want to
newly resolve (>10 ns ⇒ Δf < 50 MHz), or if you justify it on amplitude-estimation
precision in the 4–6 GHz window — which holds only ~40 LOs today — with a stated target
standard error.

**Design.** A stage-A-style capture restricted to 4000–6000 MHz, on ≥3 radios, structured
as a **factorial that separates the candidate causes** rather than one dense sweep:
arm asymmetry (via the `A(f,g)` split), connector re-mate repeatability, and
session/thermal drift, each as a deliberate factor with restoration checks. Include the
equal-gain anchor at every LO and every session so the anchor's own repeatability in this
band is measured rather than assumed.

**Budget it before scheduling it.** At the measured 0.9 s/frame and ~175 pairs per LO per
epoch, a 25 MHz grid over 4–6 GHz × 3 epochs × 3 radios is ≈128k frames ≈ 32 h, and the
re-mate/session repeats multiply that to several days — an order of magnitude beyond the
largest capture this project has run. State the frame count in the config and cut the
factorial to fit before the first frame is taken.

**Decision rule.** Determine whether a single mechanism explains the high-band
degradation. If a ripple at a different delay accounts for it, extend the model and keep
the antisymmetric form. If the **arm asymmetry** dominates, an arm-specific term is
required above 4 GHz and the antisymmetric assumption must be formally scoped to ≤4 GHz
in every document — it is currently stated globally with the asymmetry growth as a
footnote. If connector repeatability dominates, the problem is mechanical and no model
change will fix it.

---

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

## E-REC2 — regularized joint recovery of sub-GHz phase (algorithm task, no capture)

- **Motivation:** L9 upgraded — corruption is buffer-to-buffer, smooth (autocorr 0.7-0.99
  @lag1, τ~10-100 snapshots), while within-buffer phase is near-perfect. Separable from
  geometry wherever the trajectory out-jumps the nuisance (rx_random_circle; ~7+9
  random/circle sub-GHz datasets sampled).
- **Design:** fit jointly per receiver: phase = g·k·sin(θ_gt−Δθ) + φ₀ + spline_t(knots
  every ~10-15 snapshots), by circular least squares. Crucial details learned the hard
  way: (a) NOT two-step (initial g gets locked in); (b) NOT self-referencing sliding
  means (objective degenerates — rewards over-subtraction; observed g pinned at grid
  bounds with artifact circstd 0.08); use leave-window-out/gapped trend or parametric
  spline with cross-validated knot count.
- **Decision metric:** receiver agreement ρ(g_r0, g_r1) on ≥30 random/circle datasets.
  Success = ρ ≥ 0.6 (2.4 GHz benchmark 0.97; pre-recovery ≈ 0). Secondary: corrected
  circstd of the CV-held-out residual (not the fit residual).
- **TWO VARIANTS — leakage rule is hard:**
  - **REC2a (metrological, GT-using):** the joint fit above. Output may ONLY feed the
    audit (g medians, coupling curve, per-rig systematics). GT-corrected phase must
    NEVER become a training or validation input — the spline is estimated from
    residuals against the GT model, so subtracting it injects label information
    (bounded by spline smoothness, but nonzero).
  - **REC2b — TESTED 2026-07-12: FAILED, structurally.** 48 datasets (12 random / 12
    circle / 24 bounce), gapped-window GT-free trend (W=15, guard=3; leakage check
    passed — correction has no θ argument). The trend absorbed 36-54% of the geometry
    in EVERY routine and all post-correction g collapsed to the 0.5 grid floor.
    Root cause: the GRBL gantry moves continuously — θ(t) is as smooth as the nuisance
    at snapshot timescales, so there is NO label-free timescale contrast on this
    corpus; the separating information IS the label structure. GT-free recovery is
    closed. (data_quality_reports/rec2/rec2b_prototype.py + rec2b_eval.csv)
  - **REC2b original design (for reference):** estimate the trend from the RAW measured
    phase alone — robust sliding circular trend over ~15 snapshots. On jump
    trajectories (rx_random_circle) the window-mean of g·k·sinθ is ≈ constant (folds
    into φ₀), so the trend captures δ(t) without seeing labels; geometry is preserved.
    Same identifiability condition as REC2a; invalid for smooth (bounce) trajectories.
    GT is used only to EVALUATE (ρ improvement), never to construct.
- **STAGED PLAN (each step gates the next):**
  1. Estimator built right (spline circular-LS, leave-block-out trend, CV knot count),
     validated on SEMI-SYNTHETIC truth: real θ(t) trajectories + synthetic δ(t) matched
     to measured autocorr, known g. Gate: unbiased g on jumpy trajectories AND honest
     refusal (wide CI) on bounce. CPU, hours.
  2. REC2a on ~30 real random/circle datasets. Gate: ρ(g_r0,g_r1) ≥ 0.6. Fail ⇒ band
     unrecoverable, close L9.
  3. ~~REC2b~~ CLOSED (failed structurally, see above). Steps 4-5 are void for
     training; REC2a (metrology-only) is all that remains.
  4. (metrology only) Materialize sidecar /mnt/md2/cache/subghz_rec2b_v1/ (provenance: variant, params,
     source hashes; raw untouched); scanner re-scores corrected phase.
  5. Training A/B (one 250k ladder slot): r2+recovered vs r2 on val_clean/single_loss.
     Only GPU step; runs after the current Stage-1 ladder.
- **If it works:** sub-GHz metrology restored (coupling curve third band) and, if step 5
  wins, the band re-enters training via the sidecar; otherwise sub-GHz stays
  input-degraded and only the medians are salvaged.

## E-HW1 — bench VNA S21-vs-distance sweep of the antenna mounts

- **Motivation:** learnings L5 — the mutual-coupling model for g(d) fits 2.4/5.8 GHz
  medians (rmse 0.04–0.12) but A, ψ₀ are lumped and never bench-validated; competing
  mechanisms (phase-center shift, mount scattering) can't be excluded from fleet data.
- **Design:** VNA S21 between the two elements on the actual mounts, sweeping physical
  spacing at 2.4 GHz (and 5.8/915 if time permits). One afternoon.
- **Decision rule:** if measured C(d) matches the fitted A·e^{j(ψ₀−kd)}/(kd), the
  effective-spacing sidecar can be trusted fleet-wide including for spacings never
  collected; if not, the sidecar stays a per-config lookup table (still valid).

## E-HW2 — rover power board v1 (PCB)

Design spec + block diagram: data_collection/rover/rover_v3.1/power_board_v1/.
Replaces the failure-prone mechanical switch (solid-state high-side w/ soft-start —
root-causes the Apr-2025 switch deaths), the 0.1V-hysteresis LPD (10.2/11.7 V + 10 s
qualifier + 60 s Pi shutdown handshake), and the loose bucks (2 rails: Pi 5.1V/6A,
radios+aux 5.1V/5A, <10 mVpp at radio ports). Adds INA226 battery telemetry over I2C
(closes the no-BATT_*-monitor gap) and per-radio USB load switches for software
power-cycling hung Plutos. Next: KiCad schematic capture per DESIGN.md; bring-up plan
included in the doc.

## E-HW3 — always-on RC listener: MAVLink router + dedicated service

- **Motivation:** learnings 2026-08-07 — the CH9 shutdown (and CH7 reboot, CH10 magcal)
  live inside `handle_RC_CHANNELS`, which only runs while a `Drone` message loop exists.
  That means the capture process. So the switch is inert during `wait_for_radios` (up to
  600 s), between capture iterations, and permanently after `drone_run.sh` returns —
  `mavlink_controller.service` carries no `Restart=`. The level-trigger fix makes the
  switch work *whenever a loop runs*; it does not make a loop run. An operator standing
  at an idle or failed rover still has no way to power it down cleanly, which is the
  root of D1/D2 (32 unclean shutdowns) and S25 (unfinalised captures).
- **Root constraint:** the FC serial port has exactly one owner — `ardu_cli` refuses to
  open it while `mavlink_controller.service` is up. Every software fix is really "how do
  we get a second consumer of `RC_CHANNELS`".
- **Design:** `mavlink-routerd` (or mavproxy) owns `/dev/serial/by-id/usb-ArduPilot*`
  and fans out to local UDP. The capture connects to UDP (`drone-uri` already accepts
  `udpin:`/`tcp:`, so this is config, not code), and a small `spf-rc-actions.service`
  with `Restart=always` subscribes alongside it and owns the destructive channels.
  Secondary win: `rover ardupilot rc` becomes runnable *during* a capture, removing the
  catch-22 where diagnosing the switch requires stopping the service that implements it.
- **Cost:** new daemon + unit on every rover, boot ordering, `drone-uri` churn across
  capture configs, and rework (mostly deletion) of the service-ownership guard in
  `ardu_cli`. Provisioning, `rover doctor`, and `rover audit` all need to know about it.
- **Alternatives considered and rejected for now:** a long-lived Python supervisor
  (loses capture crash isolation, very high cost); systemd-arbitrated exclusive access
  (races at every capture start/stop); receiver PWM → Pi GPIO and FC relay → Pi GPIO
  (both survive more failures, including a wedged FC, but need a hardware rollout across
  four rovers). A physical shutdown button remains worth doing independently — it is the
  only option that survives loss of the RC link, and E-HW2's power board already
  specifies the GPIO handshake for it.
- **Decision rule:** if the level-trigger fix proves the switch reliable whenever a
  capture is running, do this next; if CH9 turns out to be unreliable even inside a live
  loop, diagnose that first — the fault would be below MAVLink and a router will not
  help.
- **Also queued with it:** add `SR*_RC_CHAN` to the enforced parameter set. `Drone` now
  requests its own `RC_CHANNELS` interval, so this is defence in depth rather than a
  dependency.

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

## E-DATA1 — staged data-quality training ladder  (stage 2 RUNNING as of 2026-07-14)

Stage 1: base / r1(label-clean, 1630) / r2(no-degraded, 1217) × 250k steps, sequential;
kill >3% behind on `val_clean/single_loss` at 250k, >1.5% at 500k; resume survivors
+250k per stage to 1M. Baselines (jun26 checkpoint): val 0.09735 / val_clean 0.1014 /
val_degraded 0.10902 / val_915 0.11003. Runner: `checkpoints/jul12_2026/stage1_runner.sh`.

**Stage 1 RESULT (2026-07-14, 250k, val_clean/single_loss on frozen set, 1991
batches):** base 0.10211 / r1 0.10330 (+1.2%) / r2 0.10125 (−0.8%). No kill
(all under 3%); r2 (no-degraded) best, r1 (label-clean) slightly behind base.
All three resumed to 500k via `checkpoints/jul12_2026/stage2_runner.sh`;
decision point at 500k uses the tighter 1.5% rule (r1 is the kill candidate
if its gap holds). See docs/learnings.md E-DATA1 entry.
