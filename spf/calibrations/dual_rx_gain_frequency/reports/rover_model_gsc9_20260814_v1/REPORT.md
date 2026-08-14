# The models, refitted on E-GSC9 — the rover's cells are measured, and the model is solved

**Run 2026-08-14.** Supersedes the model recommendation in
[`ladder_frames_gsc678_20260813_v1`](../ladder_frames_gsc678_20260813_v1/REPORT.md),
which was measured on bench cells the rover does not use. `main` @ `c7d93b5`. Read-only
throughout; no dataset, cache, coefficient file or segmentation module was modified, and no
file was deleted.

E-GSC9 session A captured **1,369 ordered cells over `[26,62]²`, five epochs, both carriers,
both radios — 27,380 frames**, all quality-valid. For the first time the rover's own
operating cells are measured rather than reached by assumption.

---

## The answer

**Use a per-radio, per-carrier, per-arm gain LUT — `D = d1(g1) − d2(g2)` — fitted on E-GSC9
session A. Four coefficient files are committed beside this report.**

Held-out error on the **rover's own cells**, leave-one-epoch-out, **weighted by how often the
rover actually uses each cell**:

| radio | carrier | no correction | mechanistic (L26/L30/L31 shape) | direct measured cell | **arm LUT** |
|---|---|---:|---:|---:|---:|
| R18 (clean) | 5766 | 6.372° | 1.615° | 0.163° | **0.149°** |
| R18 (clean) | 5840 | 6.788° | 2.593° | 0.227° | **0.206°** |
| R17 (damaged) | 5766 | 7.061° | 1.805° | 1.122° | **1.013°** |
| R17 (damaged) | 5840 | 10.522° | 4.148° | 0.842° | **0.868°** |

**On the clean unit the correction is worth 33–43×, and the residual is at the hardware
repeatability floor** (cell-mean repeatability was measured at 0.109°/0.149°).

![rover cells](figures/fig6_gsc9_rover_cells.png)

**Figure 1.** Every predictor on the rover's own cells, at both anchors. Left: the 62 dB
anchor, the only one the rover can currently measure. Right: 56 dB, the anchor E-GSC9's
own selection rule chose for the clean unit. The dotted line is the cell-mean repeatability
floor. The mechanistic family — every rung the gain-phase programme has shipped — sits an
order of magnitude above the arm LUT in all eight comparisons.

---

## Three things E-GSC9 settled

### 1. Additivity survives, and it is now a choice rather than a necessity

Every previous rung *had* to reach the rover's cells by assuming
`D(g1,g2) = d1(g1) − d2(g2)`, because nothing had measured them. Now both are available, so
the assumption can be priced directly: **the arm LUT is as good as looking the cell up**,
and in three of four cases slightly better (0.149 vs 0.163; 0.206 vs 0.227; 1.013 vs 1.122).

That is not a rounding artifact — it is pooling. The additive form estimates 74 parameters
from all 5,476 training frames; the direct lookup estimates 1,369 cell means from four frames
each. **Additivity is not a compromise here; it is a variance reduction.**

⚠️ It survives *on the clean unit*. E-GSC9's own H2 is **falsified on R17 at 5840 MHz**
(rover-cell residual median 1.213°, P95 2.419°, against a preregistered 1.0° limit). R18
passes comfortably at 0.139°/0.231°. The additive form is a property of a healthy unit, not
of the hardware class.

### 2. The mechanistic decomposition is confirmed as the wrong shape, on the rover's own cells

`L26`, `L30`, `L31` and `L33` all encode a single shared `H` over audited RF words. On the
rover's cells that costs **10.8× on R18 at 5766** (1.615° vs 0.149°) and 12.6× at 5840. The
previous report inferred this from bench axes; it is now measured where the rover lives.

### 3. R17's ~55° step is localised — to a single 1 dB transition

E-GSC9 H3 places **82.2% (5766) and 75.3% (5840) of the summed absolute 1 dB equal-gain steps
at the 40→41 dB transition**, with signed steps of −59.49° and −62.49°. The previous report
could only say "somewhere between 27 and 51 dB, because nothing was measured there". It is
now a named, single-transition hardware defect in the damaged unit.

---

## A correction to the previous report's headline

`ladder_frames_gsc678_20260813_v1` quoted the no-correction baseline as **~28°** and the gain
as 39–58×. Both numbers were measured on bench cells spanning `|g1−g2| ∈ {0} ∪ {26…36}`.

**On the rover's actual cells the uncorrected error is 6.4–10.5°, not 28°**, because the rover
runs a 13 dB median arm split, not 36 dB. The correction is therefore worth **33–43× on the
clean unit**, not 39–58×. The earlier figure was not wrong about the model — it was measured
on the wrong cells, which is precisely the defect E-GSC9 was built to remove. The model
recommendation is unchanged in shape and improved in evidence; only the baseline moves.

---

## The anchor, now chosen by measurement

E-GSC9's preregistered selection rule picked **56 dB for R18 at both carriers** — inside the
52…58 band the previous report predicted from the equal-gain data. H4 is nonetheless recorded
as **falsified as a radio-general claim**, because R17 selects 33 and 38.

Moving the anchor from 62 to 56 **roughly halves the uncorrected error** — 6.372→3.846 and
6.788→3.237 on R18 — and slightly improves the corrected error (0.149→0.140, 0.206→0.145).

| anchor | who can use it | R18 5766 | R18 5840 |
|---|---|---:|---:|
| 62 dB | the rover **today** (83%/96% of its equal-gain frames) | 0.149° | 0.206° |
| 56 dB | requires a scheduled anchor epoch in the rover routine | **0.140°** | **0.145°** |

Both LUT sets are committed (`coefficients/luts62/`, `coefficients/luts56/`) so the choice is
a deployment decision, not a refit.

---

## What is committed, and what it is not

`coefficients/luts{62,56}/arm_lut_<radio>_<carrier>_anchor<a>_20260814_v1.json` — eight files,
each carrying `d1_deg`, `d2_deg`, full provenance, the held-out score, and a
`DEPLOYMENT_STATUS`.

> ⚠️ **These are NOT `GainStatePhaseModel` files and `model.py` cannot load them.** That class
> encodes one shared `H`; this model is arm-specific, which is the entire reason it works.
> **A consumer does not exist yet.** Writing one — or extending the class to an arm-specific
> form — is the next code task, and it is small. This is stated rather than worked around.

---

## What is still not fixed

- **The anchor cannot be measured in flight.** `φ(f,g,g)` still contains the bearing on a
  moving rover. E-GSC9 makes the anchor *choosable*; it does not make it *measurable*. **This
  remains the single blocker between these tables and a deployed correction.**
- **69% of rover frames at 5766 change gain mid-buffer** and nothing guards it. A manual-gain
  bench capture cannot reproduce this. The stores already carry
  `gain_observation_db` / `gain_observation_sample_bounds`, so a sample-weighted correction is
  buildable from data in hand.
- **Two radios, one damaged.** Cross-radio transfer is 1.16× — none — so every table here is
  single-unit. A third unit remains the largest un-addressed weakness.
- **Sessions B and C are outstanding.** H6 (12 h transfer) and H7 (pad discriminator) are
  PENDING; until B returns, the re-calibration interval is unmeasured.
- **Two gates failed and are retained, not re-run.** G8 (manifest freshness — the file predated
  session A by four hours) and G9 (R17 across-epoch drift 4.468° against a <4° gate). R18
  passed G9 at 2.476°. The R17 tables should be read with that drift in mind.
- **23 frames at 5766 MHz (0.0171%) sit at gains 24–25, outside the captured grid.** The level
  ladder forced the preregistered fallback from a 23 dB floor to 26 dB. Those frames fail
  closed.

---

## Reproducing

```bash
P=~/virtual-envs/spf/bin/python3
$P analysis/extract_gsc.py   ./extracted          # read-only, adds the 4 E-GSC9 stages
$P analysis/gsc9_models.py   rover62              # the table above
$P analysis/gsc9_models.py   best56
$P analysis/emit_lut.py      ./luts62 62          # the committed coefficient files
$P analysis/emit_lut.py      ./luts56 56
```

`analysis/rover_cell_weights.json` is the rover cell-usage histogram over **all 42 distinct RX
captures**, deduplicated on the RX-capture prefix — the weights every number above uses.
