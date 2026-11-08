# Resume state — dual-RX gain/phase/frequency model investigation

**Last worked:** 2026-08-02
**Subject:** what makes RX1−RX2 differential phase change when you change the receive gain
setting on an AD9361/AD9363 (Pluto), and how small a physics-motivated model can be.
**Status:** analysis only. Nothing committed. Several earlier conclusions were **corrected
mid-investigation** — read §6 before trusting any artifact produced before that point.

---

## 1. Where everything is

### Source data (committed, read-only)

| Dataset | LOs | Gains | Radios | Range | Role |
|---|---:|---:|---:|---|---|
| `wide_integer_gain_cross_band_20260730_v1` | 53 | 64 | 2 | 433–5900 MHz | **all fitting and all numbers** |
| `six_radio_dense_20260729_v1` | 12 | 17 | 6 | 868–5866 MHz | structural corroboration only |
| `four_radio_dense_20260728_v1` | 12 | 17 | 4 | 868–5866 MHz | unused (subset of the six) |

All under `spf/calibrations/dual_rx_gain_frequency/reports/<name>/model_matrix.json`.
Raw IQ lives in gitignored `artifacts/` and is **absent from this checkout** — everything
below was computed from committed fitted coefficients, not raw frames.

User instruction (2026-08-02): **focus fitting on the wide integer-gain cross-band survey.**
The dense sets may be used for structure/evidence but not for fitted numbers.

### Working scripts (NOT committed, in `~/spf_logs/`)

| File | What it does |
|---|---|
| `ladder.py` | loads H, defines the step basis, fits the model ladder with held-out folds |
| `stage_model.py` | earlier exploratory version — **superseded**, kept only for reference |
| `gain_vs_gain.py` | generates the (g1,g2) heat maps: 3 band maps, contact sheet, 53 per-LO |
| `gain_phase_report_pdf.py` | builds the 9-page PDF report |
| `plot_frontier.py`, `plot_frontier2.py` | parameter/accuracy frontier plots |
| `gain_vs_gain/` | output PNGs (`per_lo/` has one per LO) |
| `gain_phase_model_report.pdf` | the 9-page report |

Python: `/home/mouse9911/virtual-envs/spf/bin/python3`

---

## 2. THE EXTRACTION TRAP — read this first

The coefficient dictionary in `model_matrix.json` has **two key shapes**:

```
frequency[N].gain_phase[N]     3339 keys   (53 LOs x 63 gains)
frequency[N].intercept           53 keys   (one per LO)   <-- ONE bracket pair only
```

A naive parser that takes the *last* bracket pair as the gain index writes each intercept
into `H[r, i, i]`, corrupting 106 diagonal cells. Also **gain index 27 (= 26 dB, the
reference) is absent** and must be re-inserted as exactly 0, not left NaN.

Correct extraction:

```python
for k, v in fit["coefficients_rad"].items():
    if "gain_phase" not in k:      # skip the intercepts
        continue
    i, j = map(int, re.findall(r"\[(\d+)\]", k))
    H[r, i, j] = v
H[:, :, list(gains).index(26)] = 0.0     # reference gain pinned at 0
H = np.degrees(H)
```

**Validation that the extraction is right** — reproduce the file's own committed block
`symmetric_gain_structure.adjacent_one_db_step_distribution` exactly:

```
n_steps 6678 | median 0.303721 | p90 1.654995 | p95 4.189408 | p99 14.443869 | max 26.472736
```

All five match to 6 decimals only with the correct parse. **Always run this check first.**

The bug inflated sd 6.93 → 7.37°, max |H| 19.2 → 67.4°, and depressed the fits badly
(Regime B held-out gain 99.2% → 84.2%). Every number produced before it was found is wrong.

### Band convention

The repo uses `frequencies <= 1_300_000_000` → low band
([`cross_radio.py:99`](spf/calibrations/dual_rx_gain_frequency/cross_radio.py:99)).
`np.digitize(freqs, [1300e6, 4000e6])` puts 1300 MHz in *mid* — off by one LO. Use:

```python
band = np.where(freqs <= 1300e6, 0, np.where(freqs <= 4000e6, 1, 2))
```

---

## 3. What H is

```
phase(r, f, g1, g2) = C[r,f] + H[r,f,g1] − H[r,f,g2]
```

`C[r,f]` = per-(radio, LO) intercept (the frequency LUT). `H` = the gain-response curve.
H is identified only up to an additive constant per (radio, LO), which C absorbs, so
**centring each curve across gain is the correct gauge**, not a convenience.

Corrected statistics for H (2 radios × 53 LOs × 64 gains):

| Quantity | Value |
|---|---|
| sd (centred) | **6.93°** |
| mean \|H\| = cost of no correction | **5.44°** |
| p95 \|H\| | 13.85° |
| max \|H\| | 19.19° |
| median peak-to-peak across gain | 21.5° |
| noise floor (source model's own LOEO MAE) | **0.91°** |

Context: frequency dominates total phase error, not gain. In the source ladder,
constant-per-radio = 14.81° MAE; frequency LUT + gain LUT = 4.83°. Perfecting the gain
term moves total phase MAE by ~2°, not ~15°.

---

## 4. Established findings

### 4.1 The steps are gain-table switch events

Decoded from `comparative_analysis.py` (`HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN`, transcribed
from Linux `ad9361.c` @ SHA `d798b0d`; byte0 = LNA in bits 6:5, mixer-Gm in bits 4:0):

- **high-band LNA switches: 23, 26, 41 dB**
- **high-band mixer-Gm switches: −6, 6, 16 dB**, then a 1-index-per-dB ramp 52→62

Measured high-band step profile fires at **6, 16, 23, 26, 41** — every in-range driver
transition. (−6 is below the measured range; 26 is the reference gain and only shows once
the reference is correctly re-inserted as 0.)

**Only the high-band table is transcribed in the repo.** Transcribing the low and mid
tables from `ad9361.c` is the single highest-value next step — it would make every
boundary an a-priori hardware constant instead of something read off the data.

### 4.2 Per-band step locations (measured, >3× band median)

| Band | Steps | Largest |
|---|---|---|
| low ≤1300 | 5, 17, 27, 31, 33, 52 | **31 dB (10.4°)** — a *spike*, see 4.3 |
| mid 1300–4000 | 3, 15, 25, 26, 27, 30, 31, 32, 50 | **50 dB (13.6°)** |
| high >4000 | 6, 16, 23, 26, 41, 42–52 | **41 dB (15.7°)** = LNA 2→3 |

Cross-band profile correlations ≈ 0 (low-mid +0.01, low-high −0.08, mid-high +0.11).
**Bands must be modelled separately.**

### 4.3 Step vs spike

Comparing plateau level 4 dB below vs above each gain:

| Band | Feature | Persistent | Transient | Character |
|---|---|---:|---:|---|
| low | 32 dB | 2.52 | **6.98** | **SPIKE (returns)** |
| mid | 50 dB | **14.37** | 7.09 | STEP (state change) |
| high | 41 dB | **16.23** | 8.07 | STEP (state change) |

The low band's dominant feature is a narrow transient, not a state change. Unexplained —
open question.

### 4.4 Cross-radio structure (six-radio dense set — evidence only, no fitting)

Step-profile correlation across 6 radios: **median r = 0.980** (min 0.839, 15 pairs).
Locations are shared; amplitudes vary by unit (5→6 dB step ranges 2.43–4.37°).
This licenses treating boundaries as fixed hardware structure and fitting only amplitudes.

### 4.5 Band edges (wide survey, 1–2 MHz resolution)

| Comparison | Separation | mean \|ΔH\| | r |
|---|---:|---:|---:|
| **3999 vs 4001** (across 4000) | 2 MHz | **5.90°** | +0.587 |
| 3990 vs 3999 (within mid) | 9 MHz | 0.28° | +0.999 |
| 4001 vs 4010 (within high) | 9 MHz | 0.26° | +0.999 |
| **1299 vs 1301** (across 1300) | 2 MHz | 0.86° | +0.773 |
| 1290 vs 1299 (within low) | 9 MHz | 1.00° | +0.905 |
| 1310 vs 1320 (within mid) | 10 MHz | 1.69° | +0.875 |

**The 4000 MHz edge is real and sharp** (21× bigger change across a 4.5× smaller gap).
**The 1300 MHz edge is NOT detected** — across it is no larger than within-band variation.

---

## 5. The models

Shared skeleton — `s_k(g)` fixed by the gain table, `A_k(f)` fitted:

```
H(f, g) = Σ_k  s_k(g) · A_k(f)
s_k(g) = 1[g ≥ boundary_k]  plus a constant and a max(0, g−52) mixer ramp
```

### Regime A — predict an unmeasured LO (5-fold random LO holdout)

| A_k(f) | Params | EV | MAE | Contiguous-block holdout |
|---|---:|---:|---:|---|
| **flat (no f dependence)** | **24** | 74.1% | 2.69° | **70.2% / 2.91°** ✅ only survivor |
| `a + m·f` | 48 | 76.7% | 2.59° | −931% / 11.93° |
| ripple τ=0.96 ns | 72 | 78.1% | 2.34° | −365969% / 132° |
| **ripple τ=2.55 ns** | 72 | **85.6%** | **1.93°** | −4924% / 16.58° |

### Regime B — per-LO amplitudes (exact-LO anchored; the deployment case)

| Boundaries/band | Params | In-sample | **Held-out gain settings** |
|---:|---:|---:|---:|
| 6 | 848 | 99.4% / 0.30° | **98.4% / 0.41°** |
| **8** | **1060** | 99.6% / 0.27° | **99.2% / 0.35°** |
| 10 | 1272 | 99.6% / 0.26° | 98.3% / 0.42° |
| free 64-pt LUT per LO (rung 6) | 6784 | 100% (tautology) | **0% / 5.44°** — no entry for an unseen gain |

**The headline result:** a free lookup table is *no better than no correction* at a gain
setting it never measured; the 10-coefficient physics basis gets 0.35°. The value of the
stage model is not compression — it is covering ground a LUT structurally cannot.

Baseline for every row: **no correction = 5.44°**.

---

## 6. CORRECTIONS — earlier claims that are now WRONG

These were stated confidently before being refuted. Anything produced earlier carries them.

1. **"Each stage contributes `a + m·f` (insertion phase + RF group delay)."** ❌ **Refuted.**
   The 41 dB LNA step is flat at −17.3° across 4.0–5.9 GHz (linear R² = 0.005); a pure
   delay would grow −17 → −25°. The frequency dependence is a **ripple**, not a delay.
2. **"50 dB is missing from the driver boundary list."** ❌ Wrong attribution. The list is
   *high-band specific* and correct there; 50 dB is a **mid-band** transition. The original
   claim came from averaging the step profile across all three gain tables.
3. **"The 1300 MHz band edge is the sharpest evidence (r = 0.11)."** ❌ Wrong. That came
   from the dense set's 1280 vs 1320 pair — 40 MHz apart, so mostly ordinary frequency
   variation. At the wide survey's 2 MHz resolution the 1300 edge is **not detected**.
   The **4000 MHz** edge is the real evidence.
4. **"A bounded sinusoid will fix extrapolation."** ❌ Refuted. Bounded *basis* ≠ bounded
   *coefficient*; ill-conditioning over a gap produced 132° errors.
5. **"A4s (linear, 48 params) is the recommended Regime-A model."** ❌ Dominated. It loses
   to the ripple on interpolation and to flat amplitudes on extrapolation.
6. **The 9-page PDF (`gain_phase_model_report.pdf`) is STALE** — its §3 physics and §6
   model both assert the `a + m·f` form, and its §5 leads with the 1300 MHz edge. Both
   need rewriting per items 1 and 3 before the document is shown to anyone.

---

## 7. Open questions

1. **Is τ = 2.55 ns a fixture constant or a radio constant?** This is the blocking
   question — see §8. If fixture, calibration is bench-specific and cannot transfer to the
   wall array or a rover without re-measurement.
2. **Why is the low band's dominant feature a spike, not a step?** (§4.3, unexplained)
3. **Why is the 1300 MHz edge undetectable while 4000 MHz is sharp?** Possibly the low and
   mid tables share more state than mid and high do — untested.
4. **What are the low- and mid-band gain tables?** Only high band is transcribed.
5. **High-band steps at 39–51 dB** are not explained by the transcribed table. Likely TIA /
   LPF / digital-gain transitions, which live in gain-table bytes beyond byte 0 and were
   never transcribed. Untested inference.
6. Does any of this survive against **raw IQ**? Everything here is fitted against H, itself
   a fitted quantity with a 0.91° error. Regime B's 0.35° residual sits below that floor,
   so B8 and rung 6 are currently indistinguishable on this evidence.

---

## 8. The experiment that would unblock this

**A cable-length / cable-swap experiment has never been run.** It exists only as
recommendation 7 in
[`FREQUENCY_SCOUT_20260727.md:143`](spf/calibrations/dual_rx_gain_frequency/reports/FREQUENCY_SCOUT_20260727.md:143).
All 10 committed experiments use the identical fixture
(`TX2 → 30 dB attenuator → two-way splitter → RX1/RX2`), and
`POWER_CYCLE_REPRODUCIBILITY.md` explicitly forbids moving cables.

The ripple hypothesis says a standing wave between splitter and RX input produces
frequency-periodic phase with period 1/τ. Fitted τ = **2.55 ns** (global optimum over a
0.2–6.0 ns scan; sharp peak — 85.6% at 2.55, 64% at 2.00, 70% at 3.00). This independently
reproduces a value of 2.55 ns obtained by a separate agent from the full H surface.
τ = 2.55 ns → **38.3 cm free space / 26.8 cm of VF=0.7 coax**, one way.

**Variant A — add a known length.** Insert ΔL ≈ 30 cm of VF=0.7 coax in the RX1 branch only.
Prediction: τ shifts by 2ΔL/(0.7c) = +2.86 ns, so 2.55 → 5.41 ns, ripple period halves
392 → 185 MHz. If the period is unchanged, the ripple is internal to the IC/package.

**Variant B — swap the two branch cables.** An external path term follows the cable (sign
flips in the differential); a radio-internal term stays with the radio.

**Variant C — cheapest, no hardware change.** Sweep ~20 LOs across a single 800 MHz window
in the high band. The predicted 392 MHz period would show two full cycles; the wide
survey's 66 MHz median spacing easily resolves it. Confirms or kills the periodicity
without touching the fixture.

**Caution from the corpus:** the frequency scout already showed that fitting a delay over a
narrow band produced a "locally precise but physically implausible 153 mm" result. My τ is
a ripple period rather than a phase slope, so it isn't the same quantity — but it deserves
the same skepticism until a cable measurement confirms it.

---

## 9. Suggested next actions, in order

1. Re-run the §2 validation check before touching anything.
2. Transcribe the **low- and mid-band gain tables** from `ad9361.c` → removes the last
   boundary-selection effect and probably explains §4.3 and §7.5.
3. Run experiment **Variant C** (no hardware change) to confirm the ripple periodicity.
4. Decide τ's nature with **Variant A or B**.
5. Rewrite the PDF per §6 — wide-survey-only, ripple physics, 4000 MHz edge evidence,
   flat-amplitude recommendation for Regime A.
6. Only then consider committing anything to
   `spf/calibrations/dual_rx_gain_frequency/reports/`.
