# Dataset reliability metrics & low-dimensional corrections — analysis + plan

Read-only analysis of the production datasets (2,256 zarr datasets: 1,691 train + 565 val,
nosig cache + segmentation v3.7 cache) and a brainstormed plan for (a) ground-truth-based
reliability/correctness metrics and (b) low-dimensional (1–3 D) corrections for systematic
errors. **No data or code was modified.** Probe evidence at the bottom was computed live on
4 datasets.

## The core idea: the forward model is the free lunch

Ground truth gives θ_gt per snapshot (from tx/rx positions). Physics predicts the observable:

```
φ_pred = −2π · (d/λ) · sin(θ_gt − θ_mount)        (the pipeline's −sin convention)
δφ     = pi_norm(φ_meas − φ_pred)                  (wrapped residual, φ_meas = mean_phase)
```

Everything below builds on δφ. Crucially, the **forward** direction is immune to the two
inverse-problem pathologies (front/back ambiguity, d/λ>0.5 aliasing): predicting φ from θ_gt is
single-valued even when θ from φ is not. Residual statistics must be **circular** (wrap at ±π).

## 1. Error-source taxonomy (what can be wrong, and its signature in δφ)

### A. Ground-truth (label) errors
| Source | Signature | Typical scale |
|---|---|---|
| GPS position error (rover) | zero-mean bearing noise ∝ 1/distance | 2–5 m @ 9–13 m range → **0.15–0.5 rad** — likely the rover noise floor |
| **Heading/compass bias (rover)** | **constant θ shift, SAME on both receivers** | probe: −0.12…−0.16 rad, repeatable across missions |
| Heading noise (EKF/mag) | zero-mean θ noise | mission-dependent |
| tx↔rx timestamp misalignment (v4→v5 interpolation) | residual correlated with **target angular velocity**; fixable by time-lag Δt | unknown — measurable |
| GRBL frozen position (**#42**) | δφ diverges while positions constant; detectable directly | silent, entire tail of a run |
| Wall-array calibration-origin offset | smooth residual vs position; 2-D (dx,dy) model | unknown |
| Lever arm (GPS antenna ≠ array center) | distance-dependent bearing bias | ~0.1–0.3 m |

### B. Measurement (φ) errors
| Source | Signature |
|---|---|
| **Per-receiver phase offset** (LO/RX1-RX2 calibration residual, `phi_drift`) | constant δφ bias per (dataset × receiver); probe: −0.19/+0.12 rad, opposite signs — matches the ±drift model |
| Thermal phase drift | slowly time-varying bias → fit c(t)=c₀+c₁t |
| Sign/cabling swap (which antenna is element 0) | δφ ≈ −φ_pred·2 pattern; detected by sign-flip fit |
| Segmentation corruption (**#45**, NaN mean_phase, few windows) | NaN or high-χ residuals; gate on window count/stddev channel |
| Multipath / reflections | **angle-dependent** structured bias (residual vs θ_gt heatmap) |
| f16 cache quantization | small, bounded |

### C. Geometry/config errors
| Source | Signature |
|---|---|
| **Wrong effective d/λ** (spacing config error, wrong `rx_lo`, or **mutual coupling** altering the array manifold) | **gain error**: φ_meas swings g× more/less than predicted. Probe: wall array g≈1.6 on both rigs' receivers — effective d/λ ≈ 0.5 vs configured 0.322. Repeatable ⇒ systematic |
| **Array mount-angle error** (`rx_theta_in_pis` wrong) | horizontal shift: φ ∝ sin(θ−Δ), per receiver. Probe: wall r0 +0.17 rad consistently, r1 ~0 |
| Near-field/plane-wave violation | computed: at 0.9 m wall-array range the curvature term is ~0.01 rad — **negligible**; ruled out as the g≈1.6 cause |

### D. Physical/expected (not errors — metrics must tolerate)
Front/back ambiguity (sinθ two-to-one) · d/λ>0.5 aliasing (0.122–1.549 across the fleet) ·
emitter duty cycle (rover NaN 64–68% is bursty-emitter reality, not necessarily corruption).

## 1b. Platform separation — wall array and rover are different REGIMES

The two platforms must never share thresholds, error models, correction menus, or pooled fits.
The deep reason: **who owns the residual differs.**

| Property | Wall array (v5) | Rover (v4) |
|---|---|---|
| Ground truth | GRBL steps → **sub-mm**; bearing error ≈ 0 | GPS (2–5 m) + compass heading; bearing noise ≈ **0.15–0.5 rad** at 9–13 m |
| ⇒ residual measures | **the RF chain** (calibration, coupling, multipath) | mostly **label noise**; systematics only visible as *biases* under it |
| Heading concept | none (fixed mounts; `rx_heading_in_pis`=0) | compass/EKF heading, **degrees in v4**, converted at v4→v5 |
| tx/rx clocks | same GRBL process → inherently aligned | **two crafts, two GPS clocks**; tx interpolated at rx timestamps (v4→v5) → Δt lag is rover-only |
| Emitter | continuous SDR blaster → NaN ≈ 0% | bursty WiFi/o4 → **NaN 60–70% is normal** |
| Range | ~0.9–3 m | ~9–15+ m |
| Range consequence | lever-arm/reference-point errors LARGE (5 cm @ 0.9 m ≈ 0.055 rad); indoor **static multipath** (repeatable per rig) | lever arm negligible (5 cm @ 13 m ≈ 0.004 rad); outdoor, variable multipath |
| Platform-specific failure | **#42 frozen GRBL position** | GPS jumps/dropouts, compass bias, inter-craft time sync |
| Observed systematic (probe) | **g ≈ 1.6** + r0 mount +0.17 rad | **heading bias −0.14 rad** (common-mode), g ≈ 1 |

**Consequences enforced throughout this plan:**

1. **Thresholds per platform** (Tier 1): NaN% gate — wall fail at >5%, rover fail only at >90%
   (60–70% is its healthy baseline); position-liveness uses the frozen-run-length detector on
   wall vs GPS-jump/HDOP-style checks on rover; velocity bounds differ (gantry vs rover max).
2. **Correction menus per platform**:
   - **Wall**: C1 offset, **C3 gain** (the observed systematic), C4 per-receiver mount,
     C8 (dx,dy) origin. *Excluded*: C2 heading (no heading exists), C6 Δt (single clock).
   - **Rover**: C1 offset, **C2 common-mode heading** (the observed systematic), C6 Δt
     (interpolation lag), C7 drift (long missions). *Excluded*: C8; C3 only as a diagnostic
     (g≈1 observed — a rover g≠1 signals config error, not a fit target).
3. **Fit protocol differs**: rover fits must be **distance-weighted** (bearing noise ∝ 1/range —
   weight δφ by range or by predicted bearing-noise variance) and are only identifiable at all
   because bias survives averaging over GPS noise; wall fits can use every snapshot unweighted
   and can afford the richer models (C3+C4+C8) because the GT is exact.
4. **Interpretation differs**: wall residual σ after correction ≈ RF-chain quality (multipath
   map-able as residual-vs-position, since the room is static); rover residual σ after
   correction ≈ GPS quality — compare it against the *predicted* GPS-induced floor
   (σ_gps/range) and flag only the excess.
5. **Never pool across platforms** (and within platform, stratify by `sdr_device_type` ×
   `rx_spacing` group — the same grouping the empirical tables already use). Fleet-scan
   reports, baselines, and regression thresholds are all per-(platform × device × spacing).
6. The NN already conditions on `vehicle_type`/`sdr_device_type` — the metrics and sidecar must
   preserve that same separation so corrections can eventually feed it consistently.

## 2. Metric suite (per dataset × receiver, cheap: needs only cached keys + mean_phase)

**Tier 1 — validity gates (no physics needed)**
- **M1 NaN fraction** of mean_phase (probe: rover 64–68% vs wall 0.2% — huge spread; set per-platform thresholds).
- **M2 Window health**: mean signal-windows/snapshot, stddev-channel distribution, median |sig| (f16 saturation check).
- **M3 Position liveness**: consecutive identical rx/tx positions run-length (detects **#42 frozen GRBL**); velocity plausibility (≤ gantry/rover max speed); timestamp monotonicity + cadence vs `seconds-per-sample`.
- **M4 RF sanity**: gains railed at limits (AGC saturation), rssis outliers, rx_lo/spacing/heading fields constant and plausible.

**Tier 2 — forward-model residual (the core)**
- **M5 Bias**: circular mean of δφ (per receiver). Nonzero ⇒ phase-offset / heading bias.
- **M6 Noise**: circular stddev of δφ. The dataset's effective label+measurement quality in one number.
- **M7 Outlier fraction**: |δφ| > τ (e.g. 1 rad) — multipath bursts, GT glitches.
- **M8 Gain**: fitted g (φ_meas vs φ_pred slope). g≠1 ⇒ effective-d/λ error (config/coupling).
- **M9 Mount shift**: fitted Δθ per receiver; on the rover, the **common** component of Δθ across receivers = heading bias, the **differential** component = per-array mount error. (This decomposition is the cleanest disambiguator we have.)
- **M10 Time-lag**: argmax over Δt of residual concentration — detects tx/rx clock misalignment; also `gps_timestamp − system_timestamp` drift where present.
- **M11 Drift**: c(t) linear fit slope (rad/hour) — thermal drift.
- **M12 Structure**: residual-vs-θ_gt profile (binned circular means). Flat ⇒ clean; sinusoidal ⇒ geometry error; localized bumps ⇒ multipath from a specific direction.

**Tier 3 — cross-checks**
- **M13 Cross-receiver consistency**: both receivers observe the same emitter → their δφ should be independent after their own corrections; correlated residuals ⇒ shared GT error (heading/GPS), uncorrelated ⇒ per-radio effects. (Same decomposition idea as M9.)
- **M14 Beamformer-peak agreement**: argmax of `weighted_beamformer` vs θ_gt (mod front/back mirror) — independent of mean_phase, catches segmentation-vs-beamformer disagreement.
- **M15 Empirical-table outlier score**: per-dataset P(θ|φ) heatmap vs the fleet-pooled table (KL/EMD) — one number for "this dataset doesn't look like its device+spacing group".
- **M16 Coverage**: θ_gt histogram entropy + distance distribution (training-bias metric, not correctness).
- **M17 Model-based flag** (optional): per-dataset val loss of the trained single model (already logged per-dataset in wandb) as a learned anomaly detector — correlates with M6 but catches non-physics issues too.

## 3. Low-dimensional correction candidates (1–3 D each)

All fit per **(dataset × receiver)** by robust circular regression (von Mises MLE / grid +
analytic offset; trim or weight by segmentation confidence). Ordered by evidence:

| # | Params | Model | Compensates | Evidence |
|---|---|---|---|---|
| **C1** | 1: c | φ→pi_norm(φ−c) | per-radio phase offset (calibration/LO) | probe biases −0.19/+0.12 rad |
| **C2** | 1: Δh (shared) | θ_gt→θ_gt−Δh both receivers | rover heading/compass bias | probe: −0.12…−0.16 rad, both receivers, both missions |
| **C3** | 1: g | φ_pred→g·φ_pred (i.e. effective d/λ) | spacing/λ config error, mutual coupling | wall g≈1.56–1.70, all rigs/receivers |
| **C4** | 1: Δθ_r (per receiver) | mount-angle correction | `rx_theta_in_pis` error | wall r0 +0.17 rad repeatable |
| **C5** | 1: s∈{±1} | sign flip | cabling swap | none seen yet; cheap to test |
| **C6** | 1: Δt | shift GT trajectory in time | tx/rx timestamp misalignment | untested (M10 measures it) |
| **C7** | 2: c₀,c₁ | c(t)=c₀+c₁t | thermal drift | untested (M11) |
| **C8** | 2: dx,dy | wall-array origin offset | calibration-origin error | untested |
| **C9** | 3: (c, g, Δθ) | φ≈c+g·(−2π(d/λ))sin(θ−Δθ) | the combined standard model | probe: wall σ 1.00→0.51–0.57 (−45%), rover 0.54→0.44–0.47 (−12%) |

**Where applied**: as a **calibration sidecar** (per-dataset yaml/parquet, never rewriting zarr —
destructive-script lesson). Consumers opt in: (a) empirical-table construction, (b) filter
observation models, (c) training targets `y_phi`/`y_rad` (label correction), (d) a per-dataset
`(c,g,Δθ)` conditioning input to the NN. Correcting the **measurement** (C1/C3/C5/C7) vs
correcting the **label** (C2/C4/C6/C8) is a real distinction — keep them separate in the sidecar.

## 4. Guards against fitting away real signal (the main risk)

1. **Stability split**: fit on the first half of a dataset, validate on the second (time split,
   not random). A true systematic transfers; overfit noise doesn't.
2. **Cross-rig consistency**: parameters should cluster by rig/day (as the probe shows). A
   dataset whose fit disagrees with its rig-mates is flagged, not corrected.
3. **Parameter priors/bounds**: |c| < π, g∈[0.7,2], |Δθ| < 0.35 rad, |Δt| < 2 s; fits at bounds
   are diagnostics ("something else is wrong"), never applied.
4. **Improvement threshold**: apply a correction only if it reduces held-out circstd by >X%
   (e.g. 10%) — otherwise record the metric, skip the correction.
5. **Identifiability**: g and Δθ are near-degenerate when angular coverage is narrow — require
   M16 coverage above a floor before fitting >1 parameter.
6. **Never correct what a bug explains**: run M3 (frozen positions) and M1/M2 gates first; a
   dataset failing Tier 1 gets quarantined, not calibrated.

## 5. Rollout plan

- **Phase 1 — fleet scan (read-only)**: one script computing M1–M12 for all 2,256 datasets
  (cheap: cached keys + mean_phase only; the 4-dataset probe took seconds each). The scanner
  branches per platform at the top (§1b): platform-specific gates, correction menu, and
  distance weighting; output is one parquet + **two ranked reports (wall / rover),
  sub-stratified by device × spacing** — never a pooled ranking. Immediately answers: how
  widespread are the wall g≈1.6 and rover heading-bias findings? is the heading bias per-day
  (declination drift) or permanent (installation)? how many wall datasets have
  frozen-position stretches?
- **Phase 2 — root-cause the two confirmed systematics** before correcting: (a) wall-array
  g≈1.6 — check the capture configs' `antenna-spacing-m` vs the physical rig, and test the
  mutual-coupling hypothesis on a raw-IQ dataset; (b) rover −0.14 rad — compare against compass
  declination/installation records.
- **Phase 3 — calibration sidecar + opt-in consumers**, with the §4 guards, starting with the
  empirical tables and filters (lowest risk), then training labels (A/B a retrain).
- **Phase 4 — CI gate**: Tier-1 metrics run at collection time (`data_collector` writes them to
  the run log) so bad captures are caught in the field, not months later.

## Appendix 0 — FLEET SCAN RESULTS (2026-07-12, all 2,250 datasets)

Scanner: `spf/scripts/dataset_quality_scan.py` → `data_quality_reports/scan_2026_07_12/`
(metrics.csv + report_wall.md + report_rover.md). Status: 547 OK · 1,216 FLAG ·
464 QUARANTINE · 23 ERROR. Wall 2,088 / rover 139 datasets. Headlines:

1. **The wall "spacing sweep" is largely mislabeled (biggest finding).** Effective d/λ
   (= fitted g × configured) clusters at **0.50–0.65 for every configured value in 0.28–0.49**
   (1,644 receiver-fits), and g≈1.0 only for configured ≥0.56. I.e. datasets configured at
   0.28/0.32/0.41/0.48 λ all behave as if the antennas sit at ~0.5–0.6 λ (~6.5–7.5 cm at
   2.4 GHz). Most consistent hypotheses: a **physical spacing floor** (antenna body width —
   sub-7 cm configs never physically realized) and/or **mutual coupling** inflating the
   effective manifold at close spacing. Consequence: `rx_spacing_input` to the NN and the
   spacing keys of the empirical tables are wrong for roughly **half the wall fleet**.
   (The cfg 0.20 group hit the g=2.0 grid bound — its true effective is underestimated.)
2. **Rover heading bias is era-dependent, stable within era**: Dec–Feb missions consistently
   −0.14…−0.33 rad (median ≈ −0.15); the later "rover*"-named era centers ≈ +0.04. Supports a
   per-day/era 1-D C2 correction; not a single permanent constant.
3. **#42 frozen-tail found in the wild: 21 wall datasets** in the training pool carry the
   frozen-position signature (positions freeze and stay frozen to the end) — quarantine or
   truncate before any retrain.
4. **431 wall datasets quarantined at >5% NaN** mean_phase (threshold may need tuning for
   bursty-emitter wall runs) — review the gate before acting.
5. **23 ERRORs are themselves integrity findings** — mostly "Too many mismatches in
   rx_spacing" asserts (the recorded per-snapshot spacing array is not constant within the
   dataset) and one missing segmentation cache.
6. **Correction payoff fleet-wide**: median circstd 0.878→0.573 (wall), 0.844→0.572 (rover)
   with the ≤3-param model — consistent with the 4-dataset probe.

### Appendix 0b — deeper investigation of the 1,216 FLAG / 464 QUAR / 23 ERROR (same day)

The statuses decompose into **six distinct populations** — most are NOT "bad data", and the
initial "half the fleet mislabeled" headline is refined to a per-band story:

1. **~980 wall FLAG:gain — real, tight, CORRECTABLE systematic (keep + sidecar, don't exclude).**
   g is not fit noise: independent receivers agree (corr 0.64, median |Δg|=0.08) and per-group
   IQRs are razor-thin (e.g. cfg 0.32λ → IQR 1.56–1.66). Effective spacing **in meters**:
   - **2.4 GHz**: floor ≈ **6.2–7.3 cm** — cfg 3.5/4.0/5.07 cm never realized (all → 6.2–6.9 cm);
     cfg ≥ 6 cm reads correct. The grid-pinned cfg-2.5 cm group re-fit wide: eff 4.6–6.1 cm.
   - **5.8 GHz**: cfg ≥ 4.3 cm all correct; cfg 2.5 cm → floor ≈ **3.2–3.4 cm**.
   - **915 MHz**: **inverted** — small cfg correct, large cfg (7/7.5 cm) reads 1.3–1.6× LARGER
     (9–12 cm). A floor cannot do that → mutual coupling and/or actually-larger mounting.
   Floors ≈ 0.5–0.6 λ in both upper bands (antenna bodies scale with λ), so "physical floor"
   vs "coupling" remains to be separated at the rig — but the sidecar remedy is identical.
2. **431 QUAR:nan>5% — a genuine capture-quality ERA, not a gate artifact**: month-clustered
   (Nov 2024: 210/210 = 100%, Oct 2024: 47%, Feb 2025: 35%, vs 0/855 Jun–Sep 2024), NaN is
   5–50% (not dead), but the valid part is ALSO noisy (median corrected circstd 0.96). Something
   degraded the rig/emitter those months. Quarantine deserved; the 142 datasets in the 5–20%
   NaN sub-bucket are borderline/salvageable.
3. **~460 FLAG:noisy** — largely overlaps the same bad months.
4. **172 FLAG:ts_nonmonotonic-only — benign**: median 0.02% of timestamp pairs out of order
   (clock jitter). Gate too sensitive; v2 should flag only >1% (max seen 18% — a few real).
5. **21 QUAR frozen_tail** — real #42 casualties; exclude/truncate.
6. **23 ERROR — all genuine integrity failures**: 9 yaml-vs-zarr mount-theta mismatch asserts,
   6 unreadable/corrupt zarrs, 6 missing segmentation caches, 2 rx_spacing-not-constant.
7. **Rover fit_at_bound (73/139) was a scanner limitation** (grid too narrow). Wide re-fit
   splits them into (a) larger heading biases (common −0.28…−0.34) and (b) **big differential
   mount anomalies** — feb7_mission1_rover1 r1 = −0.72 rad (−41°!), feb8 r1 = +0.54 —
   miswired/rotated-array sessions, genuinely suspect; exclude or investigate individually.

**Scanner v2 tweaks**: wall g grid → 3.0; rover Δθ grid → ±0.9; ts gate at >1%; wall NaN
two-tier (5–20% = FLAG, >20% = QUAR); report common vs differential Δθ for wall too.

## Appendix — live probe evidence (4 datasets, read-only)

| Dataset | d/λ cfg | NaN% | raw bias r0/r1 | raw circstd | best (g, Δθ) | corrected σ |
|---|---|---|---|---|---|---|
| wall 06-03 00:30 rx_circle | 0.322 | 0.1/0.3 | −0.08 / −0.22 | 1.00 / 1.02 | (1.56,+0.18) / (1.56,+0.02) | 0.57 / 0.57 |
| wall 06-03 00:56 rx_circle | 0.322 | ~0.2 | — | 1.05 / 1.24 | (1.62,+0.16) / (1.70,+0.02) | 0.51 / 0.56 |
| dec28_mission3_rover1 | 0.418 | 64/68 | −0.19 / +0.12 | 0.54 / 0.55 | (0.98,−0.16) / (0.96,−0.16) | 0.46 / 0.46 |
| dec28_mission4_rover1 | 0.418 | ~ | — | 0.52 / 0.51 | (0.94,−0.12) / (1.00,−0.14) | 0.47 / 0.44 |

Distances: wall median 0.9 m (near-field phase term ~0.01 rad — negligible, ruled out);
rover median ~13 m (3 m GPS error → ~0.23 rad bearing noise — plausibly the rover floor).
Consistency of (g, Δθ) across independent datasets from the same rig is the strongest evidence
these are true systematics, not fit noise.

## Appendix L — lab-log cross-reference (`~/gits/spf/log.pdf`, reviewed 2026-07-12)

The project log (103 pp, reverse-chronological, Jan 2024 → Jul 2025) independently confirms or
root-causes several scanner populations. Page refs are approximate (log has no page numbers).

**Confirmed root causes**
- **Feb-23-2025 rover spacing surgery (ERROR population).** Log shows live "data surgery":
  rover1 Feb-23 yamls relabeled 0.043→0.047 (`sed -i`) and `zarr_fix_rx_spacing.py` run over
  `rover_2025_02_23*.zarr` (one file printed `0.047 0.0 0.043` — a partial fix). March-15 entry:
  "Fix yaml and zarr for previously recorded files that were 43mm but were actually 47mm."
  → the 2 `rx_spacing not constant` ERRORs are surgery leftovers; physical-array-vs-config
  mislabels are a *documented real failure mode*, supporting the g-fit interpretation.
- **Heading-bias eras (rover Δθ_common −0.15 Dec–Feb → +0.04 spring).** Log ~Apr 4 2025:
  "Rover 2 compass calibration through mavproxy / Magcal start and accept" — compass was
  recalibrated right before the spring sessions. Dec–Feb data predates the magcal.
- **Feb-2025 bad month (wall+rover).** Feb 1: "Issues identified with rover gain settings";
  Feb 6: "Signal is weakish… raise emitter? How to get to 5.8ghz?"; Feb 15: "Debugging radios /
  New USB cables / TURN OF WIFI!!! / Make sure data on correct ports!!! / Flashed radio that
  was wonky" — the rig was actively broken/debugged through February.
- **Dec-2024/Jan-2025 era.** Dec 31–Jan 2: "DC offset correction is an issue!! On RX1!",
  "the issue might be that the IF is 0hz? And part of the DC component" (tone at DC collides
  with LO leakage/BBDC tracking); detrend rework happened here (sawtooth phase ramps in
  dec31_mission1_rover3 plots).
- **Rover-bounce split leak.** `bounce_rover.txt` was generated **March 12 2025** — the Apr-5
  rover bounce session postdates the exclusion list, explaining the 29-file leak.
- **Phase inversion on gain change (Jan 24–25 2024, "Investigating random noise in wall array
  v2 estimates").** AD9361 rx1/rx2 polarity inversion, "ALSO INVERTS AT LNA BYPASS!",
  `INVERT_BYPASSED_LNA_POLARITY`, `rx1rx2_phase_inversion_en` — documented device behavior
  where gain-index changes flip RX polarity (π phase jump). NOTE (2026-07-12): commit
  9d00b7b (Jan 26 2024) mitigates this in production — rx1-rx2-phase-inversion-enable +
  INVERT_BYPASSED_LNA_POLARITY are set on every Pluto capture (sdr_controller.py:709-721),
  predating the fleet; heavy tails in fleet data need another explanation (learnings L6).

**Corrections to earlier analysis**
- **AGC is `fast_attack`, not slow_attack** — all `v5_configs` use `rx-gain-mode: fast_attack`;
  log (Mar 24 2024) shows an explicit bench comparison: "Fast attack vs slow attack. Fast
  attack seems much better!". Gain can therefore step *within* a capture buffer.
- **Duplicate lines in split txts can be intentional upweighting** — log (Mar 12 2025) built
  `march11_train_nosig_noroverbounce_5xrover.txt` by echoing rover lines 5×. The 5 wall dupes
  in apr17 may be the same pattern, not an accident; do not "fix" without checking intent.

**Other facts worth keeping**
- Physical wall antenna mounts: 40 mm, 50.75 mm, 65 mm STLs (Jun 2024) — a 65 mm mount exists,
  consistent with the ~6.2–7.3 cm effective-spacing floor at 2.4 GHz being partly *physical*.
- Emitters: ESP32 "tone blaster" wifi CW ch 12 (flashed Dec 20 2024); 5.8 GHz devices in the
  environment: O4 5.766/5.804/5.838, Runcam 5.839, Bee 5.77 — in-band interferer candidates.
- BladeRF rig (Mar 16 2025) was a **desk test setup** (192.168.1.17/18, "side facing the wall")
  — the blade OOD gap is partly a different physical environment, not just a device effect.
- Prior experiment (Mar 24 2025): "Try half data in training only to see affect on val
  (unchanged)" — dropping ~50% of data did not move val; sets expectations for regime R2.
- Custom Pluto firmware precedent: pgreenland `v0.38_plutoplus_timestamp` fork built+flashed
  in-house (buildroot, DFU); "newer timestamp 0.38 firmware does not work on some PLUTOPLUS".
  Gain-in-IQ embedding would follow the same toolchain.
- Log page 1 TODO (undated, recent): "DO VAL BY FILE! FIND OUTLIERS!!!!" — the per-file val +
  outlier program is the user's own top-of-log priority.

## Note — FLAG:gain volume explained (2026-07-12)

`F:r{0,1}_gain=X` = fitted g (effective/configured spacing) with |g-1|>0.25. Fires on
1,268/2,250 datasets (all wall) because it flags CONFIG FAMILIES, not captures: wall
2.4 GHz medians by configured spacing — 2.5cm→g=2.14 (eff 5.4cm), 3.5cm→1.78 (6.2cm),
5.075cm→1.38 (7.0cm), >=6.5cm→~1.0. Effective spacing floors at ~lambda/2; receiver
agreement rho=0.64 => physical, stable. 767 FLAG datasets have gain as their ONLY reason.
Damage class: feature-mislabeled (usable data, wrong rx_spacing input) — NOT dropped by
r1/r2 regimes (which act only on QUAR/nan>20%/noisy). Gain-flagged configs are also
intrinsically harder (corrected circstd 0.64 vs 0.48; outliers 33% vs 6%) due to coupling.

Scanner v3: replace |g-1|>0.25 with per-config-family deviation |g - median_g(config)|>0.15
(silences expected flags, catches true per-capture anomalies like the 43/47mm surgery).
Follow-up experiment: per-config effective-spacing sidecar (config -> median g) as an
alternative rx_spacing input.

## Note — sub-GHz g spread ROOT CAUSE: beacon parked at ~0 Hz IF (2026-07-12 probe)

Raw-IQ probe (probe_subghz_windows.py) of the worst rx-disagreement 915 dataset vs a 2.4
control:
- 915: the buffer is a CONTINUOUS strong carrier (rms ~1200, near rails), 74% of r0 power
  within 0.2%*fs of DC; r1 sees the same tone at -0.0034*fs. The two receivers' independent
  LO offsets park the beacon within tens of kHz of DC — the "IF=0" problem from the lab log
  (Jan 2025). The AD9361 RF-DC/BBDC tracking loops treat a near-DC tone as offset to null:
  each receiver pair slowly rotates/attenuates it INDEPENDENTLY -> slow per-receiver phase
  drift -> g fits wander independently (rho~0, |dg| median 0.58). Segmentation duty (17%) is
  arbitrary there: on-snapshot windows are identical (SIG/NOI windows same rms/spectrum).
- 2.4 control: tone at -150 kHz IF (safely off DC), bursty (SIG rms 161 vs NOI 1.1),
  clean identity phase-vs-geometry track.
=> Sub-GHz g spread is NOT noise, NOT segmentation quality, NOT antenna physics variance:
   it is DC-collision + independent DC-tracking loops. Fix for future sub-GHz capture: set
   IF >= a few hundred kHz (fi param), or disable BBDC tracking. Historical sub-GHz g is
   unusable per-dataset; medians remain biased-suspect.

## Scanner v3 candidate — beamformer-based metrics (replaces fixed-number g where scalar fails)

weighted_beamformer (65-bin) is already in the precompute cache; scanner can score, per
snapshot, the percentile rank of the GT bin (folded front-back). Demo: 2.4-control mean
rank 0.52-0.55, sub-GHz 0.42-0.47 (below 0.5 = systematically misaligned). Naive version
confounds systematic offset with informativeness (control only 0.55) — v3 version must
offset-compensate first (allow one global bin-shift/alias per dataset, analogous to
fitting c), then report (a) offset-corrected GT-percentile = alignment, (b) beamformer
entropy = informativeness. Cost ~zero (cache read).
