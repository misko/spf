# E-GSC9 — capture the rover's operating region, and everything the model needs around it

**Status: IN PROGRESS.** Session A and the same-session A2/A3 controls were captured on
2026-08-13. See [`RESULTS.md`](RESULTS.md) for the immutable failures as well as the passes.
Session B and the physical A/B/A pad discriminator remain outstanding.

**Cost after the measured fallback: one completed 2.64 h session, about 30 min
for ladders/A2/A3/B, and about 51 min for the three physical C legs.** Destination for raw output:
`/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc9_<session>_<date>_v1/`.

---

## 0. Why this experiment exists

[`ladder_frames_gsc678_20260813_v1`](../../spf/calibrations/dual_rx_gain_frequency/reports/ladder_frames_gsc678_20260813_v1/REPORT.md)
(`7b88a6e`) established four things on frames, and this experiment is the direct consequence
of all four:

1. **The rover's operating region has never been measured.** The rover pins RX1 near 62 dB
   and runs RX2 at 46–51 dB; median `|g1−g2|` = 13 dB. E-GSC6/7/8 swept `(26,g)`, `(g,26)`
   and `(g,g)` for g = 52…62 — that is `|g1−g2| ∈ {0} ∪ {26…36}`. **Not one rover operating
   cell was ever measured**, and gains 46/47/48 were never measured on either arm at either
   carrier. The best model reaches 37.2% of rover frames at 5766 MHz and **1.1% at 5840**.
2. **The model is already at the measurement noise floor**, so accuracy is not the target.
   `L24` prospective MAE 0.716° / 0.511°; single-frame repeatability 0.682° / 0.435°;
   cell-mean repeatability 0.109° / 0.149°. **The target is coverage and validity.**
3. **The anchor gain is the dominant design choice.** At a 26 dB anchor the damaged unit
   violates `D(g,g) = 0` by 53.8–65.5°; at 55–58 dB both units sit at 0.65–1.7°. The step is
   located only to "somewhere in 27…51", because no equal-gain cell exists there.
4. **69% of rover frames at 5766 MHz change gain mid-buffer** and nothing guards it.

---

## 1. Approach, and the arithmetic that justifies it

**Measure `D(g1, g2)` for every ordered pair in `[23,62]²`, at 5766 and 5840 MHz, on both
radios, five times, at one fixed source level.**

`[23,62]²` = **1,600 cells**. The full band-2 cross is 73² = 5,329. The grid is 30% of the
cells and 27% of the wall clock, and it is **operationally identical**:

| carrier | rover arm-pair frames | inside `[23,62]²` | coverage | distinct cells | cells inside |
|---|---:|---:|---:|---:|---:|
| 5766 MHz | 134,374 | 134,374 | **100.0000%** | 524 | **524** |
| 5840 MHz | 43,036 | 43,036 | **100.0000%** | 337 | **337** |
| **total** | **177,410** | **177,410** | **100.0000%** | | |

*Measured, not asserted* — exhaustive read-only census of **all 42 distinct RX captures** in
`/mnt/qnap01/mouse9911/rovers_2026/merged`, deduplicated on the RX-capture prefix. Verified
twice independently (design agent and judge), then a third time before this document was
committed.

**Why the grid stops where it does.** The floor is 23 dB because that is the rover's measured
minimum (`g1` min 23, `g2` min 25 at 5766; 29/30 at 5840). The ceiling is 62 because that is
the top of the band-2 table. Gains −10…22 and cells with `|g1−g2| > 39` carry **0.0000% of
rover frames**.

> ⚠️ **They are excluded on coverage grounds, not because they are unmeasurable.** An earlier
> claim in this project's chat — and in one of the three candidate designs — asserted that
> roughly a third of the full cross is physically unmeasurable at any TX setting. **That is
> false and was falsified during review.** Those cells are reachable; reaching them merely
> requires a *second TX stratum* plus a stitching control, because a single fixed source
> level cannot hold both arms in range across a >39 dB split. The honest sentence is: they
> are excluded because they carry no operational mass and would cost a second level regime
> to acquire. If a future rover routine widens the split, this decision must be revisited.

---

## 2. The fixed-TX requirement — the spine of the design

**TX drive is fixed for the whole of session A. This is a hard requirement, not a
convenience.** The harness's default `adaptive_max_rx_gain` policy sets
`tx = reference − max(g1,g2)`, which is fatal to the thing being measured:

> The held-out cell `(48,47)` would be captured at `tx = −9`, while the two training cells
> that predict it, `(48,62)` and `(62,47)`, would be captured at `tx = −23`. **A 14 dB source
> level difference lands directly on the additivity residual**, perfectly correlated with
> gain index. Any measured departure from additivity would be unattributable.

Under a single fixed TX, every cell in the grid sees the same source, so the level model
`tone_dbfs = K + 0.999·g + 0.983·tx` (fitted on 27,406 archived frames, residual sd
0.71–0.91 dB) means level differences across the grid are *exactly* the gain differences
being studied — which is the point.

**Level ladder, run first, 2.7 min.** Three configs at TX ∈ {23, 29, 35} over
`{23,35,45,49,56,62}²`, both LOs, permissive quality thresholds so nothing is discarded.

> **Preregistered selection rule.** `tx-gain-db` = the largest integer `T` such that
> `max` over (radio, LO, arm) of measured `tone_dbfs` at g = 62 is ≤ **−12.0 dBFS**, subject
> to `min` over (radio, LO, arm) at g = 23 being ≥ **−58.0 dBFS**. If both cannot hold, satisfy
> the upper bound and **raise the gain floor**, regenerate with `analysis/gen_configs.py`, and
> restate the coverage claim at the new floor. The fallback is nearly free: at floor 26 coverage
> is still 99.98% / 100.00% of frames (508/524 + 337/337 cells); at floor 40 it collapses to
> 295/524. **The operator therefore has ~3 dB of headroom and should know it before committing.**

The chosen `T` is written into the config **once, before the first session-A frame**, and the
config is then frozen — `dataset.py` hashes the whole document into
`calibration_run_signature`, so any later edit (including `notes:`) invalidates resume.

**Measured precision floor, for interpreting the cold corner.** Binning within-capture phase
std against the weak arm's own level over 95,944 archived frames: p95 = 3.109° at −70…−60 dBFS,
**0.220° at −40…−30 dBFS**. The design's coldest corner sits near −58 dBFS.

---

## 3. Sessions

| # | config | cells | reps | wall clock | purpose |
|---|---|---:|---:|---:|---|
| 0 | `level_ladder_tx{23,29,35}` | 36 ea | 1 | **2.7 min** | choose the fixed TX; **hard STOP for a human read** |
| A | `rover_region_grid` | **1,369** | 5 | completed | measured fallback grid, gains 26..62, both carriers |
| A2 | `t2_transitions_bridge` | 70 | — | 4.7 min | brackets the RF-word transitions the grid cannot reach |
| A3 | `t3a_ampm_16384` + `t3b_ampm_8192` | 28 + 28 | — | 4.4 min | AM-PM control: identical cells, −6.02 dB source |
| B | `session_transfer` | 273 | — | ~17 min | repeat ≥12 h later after a power cycle, nothing re-cabled |
| C | `pad_discriminator` | 273/leg | 3 | ~17 min/leg | 10 dB pads, **with an A/B/A reversal leg** |

**Total ≈ 4.0 h of capture; the longest single run is 2.64 h**, inside the four
proven-clean 3.1–3.4 h sessions this rig has already completed.

**Order within session A: epoch-outer.** Each of 5 epochs contains the complete
1,369-cell fallback grid, so the run is *epoch-complete* — stopping early costs
epochs, never coverage. All 37 captured equal-gain cells appear in every block,
≤48 s apart.

**Degradation ladder** — what each stopping point buys:

| stop after | wall clock | what you have |
|---|---:|---|
| epoch 1 | 32 min | full grid, n=1, no across-epoch variance |
| epoch 3 | 1.6 h | full grid, n=3, **minimum for any published number** |
| epoch 5 | 2.64 h | full grid, n=5, design power |

---

## 4. Anchor strategy — chosen from data, not convention

**During capture there is no privileged anchor.** All 37 fallback equal-gain
cells `(g,g)`, g ∈ 26…62, are measured in every block: 37 anchor observations
per block, 5 across-epoch observations of each.

**The anchor is chosen afterwards** by this preregistered score, computed per
(radio, carrier) over all 37 captured equal-gain candidates:

- **S1(a)** — the `|D(a,a)|` bias the model's forced `D(a,a) = 0` injects at the anchor.
- **S2(a)** — across-epoch circular std of `D(a)`, i.e. how stable the anchor itself is.
- **S3(a)** — **fit `H` from the training cross referenced at `a`, then score held-out cells.**
  (This replaces an earlier draft's S3, which scored a constant predictor — a baseline, not
  the model. Correction applied after judging.)

The winner is the `a` minimising S3 subject to S2 below the session's median drift. This is
the first time the anchor gain becomes a measurement rather than a convention.

---

## 5. Pre-registered hypotheses

| id | prediction | falsifier |
|---|---|---|
| **H1** | The grid covers **100.000%** of rover arm-pair frames and 524/524 + 337/337 distinct cells at the two carriers | Any rover cell outside `[23,62]²`. *(Already verified on the corpus; H1 is a check that the capture matches the census, not a discovery.)* |
| **H2** | Additivity holds at the rover's own cells: median \|residual\| of `D(g1,g2) − d1(g1) + d2(g2)` ≤ **1.0°** over the 524/337 rover cells | Median > 1.0°, or P95 > 3.0°. **This is the hypothesis the whole gain-phase programme rests on and it has never been tested off-axis.** |
| **H3** | The equal-gain step in the damaged unit is **localised to a single 1 dB transition** in 23…62, or shown to lie below 23 | No single transition carries ≥50% of the total step. A2 brackets 13…49 so the LNA 0→1 step at 22→23 and the TIA step at 13→14 are both covered. |
| **H4** | The best anchor by S3 is **not 26 dB and not 62 dB**, but lies in 52…58 | The optimum is 26 or 62, which would mean the convention was right and §0(3) misread the bench data. |
| **H5** | Differential phase is **independent of source level**: A3's two legs agree within the session's anchor drift | Legs differ by more than 3× median anchor drift ⇒ an AM-PM term exists and every `D` in this project carries a level covariate. |
| **H6** | A 12 h gap with a power cycle costs **< 0.5°** on cell means | ≥ 0.5°, which would put a re-calibration interval on the deployed table. |
| **H7** | Inserting 10 dB pads shifts `D` by less than the harness-coupling bound, **and reverses on removal** | Non-reversal under the A/B/A leg ⇒ the connector work itself, not the pad, moved the measurement. |

**Falsifier discipline.** H2 is the load-bearing one. If it fails, the additive form is wrong
off-axis and *every* rung of the ladder above L00 is invalid at the rover's cells — a more
important result than a successful capture, and it must be reported as such rather than
re-measured until it agrees.

**H6 operational definition, frozen before session B:** compute a circular mean
for every quality-valid cell in each session, then compute the circular MAE of
the matching A-to-B cell-mean shifts independently for each radio and each LO.
H6 passes only if all scheduled B cells are present in both sessions and all
four radio-by-LO MAEs are strictly below 0.5 degrees. Median, P95, maximum, and
circular bias are retained as supporting diagnostics but do not replace the
predeclared MAE decision metric.

**H7 operational definition, frozen before session C:** E-HCP1's conservative
worst-case phase-coupling upper bound is 8.9 degrees. For every radio and LO,
all scheduled cells must be quality-valid in all three legs, the circular MAE
of B-minus-A cell means must be strictly below 8.9 degrees, and the circular
MAE of A-prime-minus-A must be smaller than A-prime-minus-B. All four strata
must pass. Separately, G5/G6 permit the treatment to be reported as a resolved
effect only when its MAE is at least 3 times the largest median within-leg
equal-gain anchor drift; otherwise it is reported only as an upper bound.
The no-pad/pads/pads-removed legs use TX gains -35/-25/-35 dB respectively,
so the two 10 dB pads are compensated at the transmitter and nominal received
level is matched across the physical treatment.

---

## 6. Statistical power

Each cell is measured **n = 5** times (once per epoch). The additivity residual
`e = D(g1,g2) − D(g1,a) − D(a,g2) + D(a,a)` is a combination of **four independent cell
means**, so `Var(e) = 4σ²/n`, not `σ²/n`.

> **This corrects an error in the winning draft**, which used `σ²/n` and therefore reported a
> standard error 2× too small. With the measured single-frame σ ≈ 0.68° at 5766 and n = 5,
> `SE(e) = 2σ/√5 = 0.61°`. H2's 1.0° median threshold is therefore ~1.6 SE — detectable, but
> **not** the comfortable margin the draft implied. At 5840 (σ ≈ 0.44°) `SE(e) = 0.39°`.

Cell means themselves are far better determined: cell-mean repeatability was measured at
0.109° / 0.149° across sessions, so the LUT that this experiment produces is limited by the
*residual test*, not by the table.

---

## 7. Acceptance gates, each justified as well-posed

E-GSC7's `railed_fraction` gate reported 100% whenever 62 dB was deliberately commanded — it
could not fail, and so it measured nothing. Every gate below is checked against that failure
mode: it must be possible for it to fail while the run is otherwise healthy.

| id | gate | why it is well-posed |
|---|---|---|
| **G1** | 5 epochs × 1,369 fallback cells present per radio per carrier; `n ≥ 3` for any published cell | Counts, not thresholds. Fails on a truncated run. |
| **G2** | Every kept frame within the level envelope: `tone_dbfs ∈ [−65, −6]`, `clipping_fraction = 0` | The envelope is **narrower than the config's capture thresholds**, so it can fail on kept data. *(An earlier draft set analysis-time bounds looser than capture-time ones, which made the gate unfailable — corrected.)* |
| **G3** | `gain_endpoints_equal` true on 100% of frames; manual gain mode confirmed by readback | The rover corpus shows 69% instability; a calibration that inherits it is worthless. Readback catches silent AGC re-entry. |
| **G4** | Live gain-table readback SHA-256 identical on both radios and to E-GSC7's `90d34d61…a1143` | Any firmware or table drift invalidates the state decomposition. |
| **G5** | **Resolved-margin:** no effect is *stated* unless `effect / median same-run anchor drift ≥ 3` | Ties every claim to that session's own noise. Directly generalises the L-GSC lesson. |
| **G6** | Any `|D|` failing G5 is published as a **BOUND**, never as a value | Prevents a drift-sized number entering a table as a measurement. |
| **G7** | The same-LO repeat is computed and reported **before** any 5766-vs-5840 claim | E-GSC8's precedent: R18's attractive 0.372° transfer was correctly voided because its same-LO control failed. Ordering is the gate. |
| **G8** | `/run/spf/direct_usb_ready.json` mtime **postdates session start** | The only mechanical catch for the documented stale-manifest failure. |
| **G9** | Anchor drift: worst single across-epoch drift < 4° (E-GSC6's observed worst) | Empirical, and it has failed before (E-GSC7 hit 4.691° at 5300 MHz). |
| **G10** | `RESULTS.md` states H1–H7 with numbers, including any falsified | House convention. |

---

## 8. Runbook

Full operational detail is in [`NOTES_FOR_BENCH_AGENT.md`](NOTES_FOR_BENCH_AGENT.md). The two
non-obvious points:

**A hard STOP after the level ladder.** Read `tone_dbfs` at g = 62 and g = 23 on both arms and
both LOs, apply the §2 selection rule, and only then freeze the config. Do not begin the 2.64 h
commit on an unread ladder.

**Wrap session A in a restart loop.** `_open_preflight_radio` has a bare `finally:` with no
`except:`, so an exhausted handoff-prime sequence propagates and kills the run. The capture is
resumable by `calibration_run_signature`, so:

```bash
for i in $(seq 1 12); do <capture command>; sleep 30; done
```

---

## 9. What this design deliberately does NOT cover

- **Gains −10…22 and `|g1−g2| > 39`.** 0.0000% of rover frames; excluded on coverage grounds,
  **not** because they are unmeasurable (§1).
- **Bands 0 and 1.** The rover is band 2 only. A future carrier outside 5 GHz needs its own
  capture; nothing here interpolates in frequency.
- **A third radio.** Cross-radio transfer was measured at 1.16× — i.e. none — so the per-radio
  result rests on n = 2, one of which is damaged. **This is the largest un-addressed weakness**
  and it needs a third unit, not a longer sweep.
- **Mid-buffer gain instability.** 69% of rover frames change gain mid-buffer. **A bench
  capture with manual gain cannot reproduce this and this experiment does not try.** The fix is
  rover-side: either a stability guard in `predict()`, or a sample-weighted correction using
  the `gain_observation_db` / `gain_observation_sample_bounds` arrays that the stores already
  carry. Recorded here so it is not mistaken for something E-GSC9 delivers.
- **The anchor on a moving platform.** `φ(f,g,g)` still contains the bearing. E-GSC9 makes the
  anchor *choosable*; it does not make it *measurable in flight*. That remains the top blocker.
- **The bench-level vs rover-level gap.** Every bench cell is measured at a level set by the
  bench, while the rover's arms sit ~16 dB away at the same commanded gain. If phase depends on
  level as well as gain state, H5 is what detects it — and if H5 fails, this gap becomes the
  next experiment.

---

## 10. Provenance

Designed by a 5-scout / 3-design / 2-judge panel against `main` @ `7b88a6e`; both judges
independently selected the same design. Corrections applied after judging, each recorded above
at the point it bites: the `σ²/n` power error (§6), the unfailable level gate (G2), the
anchor score that scored a baseline (§4), the missing A/B/A reversal leg (§3 session C), and
the false "physically unmeasurable" claim (§1). Grafts from the two losing designs: the
transitions bridge (A2), the AM-PM control (A3), the resolved-margin and ordering gates
(G5–G7), the restart wrapper (§8), the degradation ladder (§3) and the measured precision
floor (§2).

All 11 configs validate against `main`. Read-only throughout; no file was deleted.
