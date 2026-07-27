# QC metrics — definitions, formulas, and gate logic

**Authoritative reference for what every quality metric means and exactly how it is computed.**
Companion to [`DATA_OVERVIEW.md`](./DATA_OVERVIEW.md) (what data we have, with per-group metric
tables) and [`../04_training_inference/TRAIN_VAL_SPLITS.md`](../04_training_inference/TRAIN_VAL_SPLITS.md)
(how these metrics select train/val partitions).

- **Source of truth:** `spf/scripts/dataset_quality_scan.py` (byte-identical archived copy at
  `pdf_scripts/dataset/dataset_quality_scan.py`, sha256 `566d5749…`).
- **Canonical output:** `data_quality_reports/scan_2026_07_12_v2/metrics.csv` — 2,250 rows × 48
  columns, `scan_v = 2`. Byte-identical archived copy at `pdf_scripts/dataset/metrics_v2.csv`
  (sha256 `4f00219b…`).
- `data_quality_reports/scan_2026_07_12/` (no `_v2`) is **superseded** — different NaN taxonomy and
  a narrower rover fit grid. Do not read numbers from it.
- **Verification:** the gate logic below was replayed against all 2,227 non-ERROR rows —
  **0 status mismatches, 0 `reasons`-string mismatches**. Every formula and threshold here is
  from the code, not from prose.

> Where this document and `data_quality_plan.md` disagree, **this document wins**.
> `data_quality_plan.md` is the design record (metric rationale M1–M17, correction menu C1–C9,
> root-cause investigations); its scan counts and gate descriptions are stale.

---

## 1. Inputs

The scanner opens each dataset read-only (`dataset_quality_scan.py:147-154`):

```python
ds = v5spfdataset(zarr_fn, nthetas=65, ignore_qc=True,
                  skip_fields=set(["signal_matrix"]),
                  precompute_cache=precompute_cache, paired=False)
```

`ignore_qc=True` **bypasses the loader's own QC** (`spf_dataset.py:1153-1168`), so the scan sees
every dataset — including ones the loader would refuse. It never writes:
`segment_if_not_exist` defaults to `False`, which is why 6 of the 23 ERRORs are "segmentation file
does not exist" rather than a cache write.

| Symbol | Meaning | Source |
|---|---|---|
| `phi_m` | measured phase difference, per snapshot | `ds.mean_phase[f"r{r}"]` — `:201` |
| `theta` | ground-truth bearing in the **receiver-array frame** | `ds.ground_truth_thetas[r]` — `:202` |
| `k` | array scale constant, `-2π·(d/λ)` | `:199` |
| `ranges_m` | tx↔rx separation, metres | `:166-173` |

**`phi_m` provenance.** Read from the precompute cache, never recomputed: per snapshot, every
`simple_segmentation` region of `type == "signal"` contributes its circular-mean phase, weighted
`w = (end−start) · abs_signal_median / (stddev + 1e-6)` (`segmentation.py:85-88`), combined by
`mean_phase_mean` (`rf.py:415-420`). **If no signal region is found the snapshot is `NaN`**
(`segmentation.py:90-91`) — this is the sole origin of `nan_frac`.

**`theta` provenance.** `arctan2(dx, dy)` of tx−rx, minus `(rx_theta_in_pis + rx_heading_in_pis)·π`
(`spf_dataset.py:1590-1608`); note 0 rad = **+y**. When `rx_heading_in_pis` is absent from the zarr
it is filled with zeros (`spf_dataset.py:1210-1224`) — wall captures have no compass, which is why
`heading_common` is a rover-only diagnostic.

**`k`.** `d/λ ∈ [0.12208, 1.5488]` in this scan ⇒ `k ∈ [−9.731, −0.767]`. One `k` per dataset,
shared by both receivers.

---

## 2. Math primitives

### `circ_mean_std(d, w=None)` — `:37-41`

```python
z  = np.exp(1j * d)
zb = z.mean() if w is None else (z * w).sum() / w.sum()
R  = min(max(np.abs(zb), 1e-9), 1.0)
return float(np.angle(zb)), float(np.sqrt(-2 * np.log(R)))
```

**circstd is the Mardia/Fisher circular SD, `s = √(−2·ln R)`** — *not* `sqrt(1−R²)` and *not*
`scipy.stats.circstd` with a range argument. `R` is clipped to `[1e-9, 1]`, so
**circstd ∈ [0, 6.4379] rad**. Weights need not be normalised.

### `wrap(x)` — `:44-45`
`np.angle(np.exp(1j·x))` → (−π, π]. (Note `pi_norm`/`torch_pi_norm` in `rf.py` use
`((x+π) % 2π) − π`, which is **[−π, π)** — half-open at the *lower* end.)

---

## 3. The systematic model

```
phi_meas  ≈  c  +  g · k · sin(theta_gt − dtheta) ,        k = −2π·(d/λ)
```

Three free parameters **per receiver**:

| Param | Meaning | How obtained |
|---|---|---|
| `g` | gain / effective-spacing scale (1.0 = nominal d/λ) | grid search |
| `dtheta` | mount / boresight angular shift (rad) | grid search |
| `c` (`offset_c`) | constant phase offset (rad) | **analytic** — circular mean of the residual |

**Fit** (`fit_systematics`, `:51-64`): exhaustive brute-force 2-D grid search — no gradient, no
refinement, no interpolation. Objective is the **weighted circular SD of the residual**; ties go to
the first-seen (lowest `g`, then lowest `dθ`). `c` is exactly the circular mean of the residual at
the winning grid point, so `circstd_corr` is the dispersion *about* that offset.

### 3.1 Grids are platform-specific — `:96-101`

| Platform | `g_grid` | n | `d_grid` (rad) | n | Grid size |
|---|---|---|---|---|---|
| **wall** | `arange(0.70, 3.01, 0.02)` | 116 | `arange(−0.35, 0.351, 0.02)` | 36 | 4,176 |
| **rover** | `arange(0.90, 1.11, 0.02)` | 11 | `arange(−0.90, 0.901, 0.02)` | 91 | 1,001 |

The code comment at `:99` is important: for rovers **"g is a diagnostic, not a fit target"** — the
narrow g range makes the rover fit effectively a mount/heading fit.

### 3.2 Distance weighting — rover only — `:78-81`

```python
w = np.clip(rng, 1.0, None) ** 2 ;  w /= w.sum()      # rover
w = None                                              # wall (uniform)
```

Weight ∝ range², floored at 1 m ("bearing-noise variance ~ 1/range²") — this **up-weights distant
samples**. It is used only inside `circ_mean_std`, so it affects `bias`, `circstd_raw`, `offset_c`,
`circstd_corr`, `g`, `dtheta` — **not** `outlier_frac`, `drift_span`, or `structure`. (Indirectly it
does touch `outlier_frac`, whose *centre* is the weighted `bias` even though its count is unweighted.)

### 3.3 Subsampling — `:88-93`

Above `max_fit_points = 4000` valid samples, a **deterministic uniform-stride** subsample is taken
(`np.linspace(0, n−1, 4000).astype(int)`). The fit, the offset-only fallback and the **coverage
histogram** use the subsample; `bias`, `circstd_raw`, `outlier_frac`, `drift_span`, `structure` use
the full arrays.

### 3.4 Coverage-aware identifiability guard (new in v2) — `:102-117`

θ is histogrammed into 12 equal bins of π/6; `coverage_bins` counts bins holding ≥20 samples.
If `coverage_bins < 4` the fit is **skipped** and only the offset is estimated, forcing
`g = 1.0, dtheta = 0.0, g_at_bound = False`.

> **Consequence:** a low-coverage receiver can *never* raise `FLAG:r*_gain` (|1.0−1| = 0) and
> *never* raise `FLAG:fit_at_bound`. Low coverage **suppresses** those two flags — always read
> `INFO:low_coverage` before trusting the absence of a gain flag.

---

## 4. Column reference (48 columns)

Scope: `DS` = one per dataset, `RX` = emitted twice with `r0_`/`r1_` prefixes.
The 23 blanks in nearly every column are the 23 ERROR rows.

### 4.1 Identity / configuration (7, `DS`)

| Column | Meaning | Computation | Observed |
|---|---|---|---|
| `dataset` | name | basename minus `.zarr` — `:145` | 2,250 unique |
| `device` | SDR model | from receiver URI — `:159` | PLUTO 2,179 · BLADERF2 48 |
| `platform` | wall vs rover | `"wall"` if `"wall"` in name or vehicle_type — `:156` | wall 2,088 · rover 139 |
| `wavelength_spacing` | d/λ | `rx_spacing / wavelength` — `:160` | 0.12208 … 1.5488, med 0.40831 |
| `rx_lo` | LO frequency, **first snapshot of r0 only** | `:161` | 16 distinct; **59 rows are 0 Hz** (31 rover, 28 wall) — a defect, not gated |
| `n_snapshots` | snapshots in the zarr | `:162` | 75 … 10,000 |
| `scan_v` | schema version | literal `2` — `:214` | 2 |

### 4.2 Dataset-level geometry / liveness / timing (7, `DS`)

| Column | Meaning | Computation | Observed |
|---|---|---|---|
| `range_med_m` | median tx↔rx separation | `:166-173` | **wall 0.06–2.04 (med 1.02); rover 0.01–58.9 (med 29.0)** — cleanest platform discriminator |
| `ts_med_dt` | median inter-snapshot interval | `:176-179` | wall med 0.995 s; rover med 0.489 s |
| `ts_nonmono_frac` | fraction of non-increasing timestamp pairs | `(np.diff(ts) <= 0).mean()` — `:178`; ties count | 0 … 0.179, med 0 |
| `rx_speed_p99` | 99th-pct receiver ground speed | `:194-196` | wall ~0.004 m/s; rover med 1.95 m/s. **Computed but never gated** |
| `frozen_max_frac` | longest "no movement" run / total | `:180-188`, L1 norm, **both** rx and tx must be static | wall max 0.893; **rover max 0.0018** |
| `frozen_tail` | does the longest frozen run *end* the dataset | `runs[-1] > max(20, 0.05·n)` — `:189` | True on 21 rows, **all wall** |
| `frozen_tail_start` | index where the terminal freeze begins (salvage cut point) | `:190-192`, emitted only when `frozen_tail` | present on those 21 rows |

### 4.3 Per-receiver metrics (14 × 2 = 28, `RX`)

Computed by `receiver_metrics(...)` — `:67-136`. **Early-exit guard `:72-73`:** if `n_valid < 50`
only `nan_frac`/`n_valid` are set. No row hit this path (min n_valid: r0 71, r1 68).

| Column | Meaning | Computation | Observed |
|---|---|---|---|
| `r{i}_nan_frac` | fraction of snapshots with no usable phase **or** no ground truth | `:69-70` | r0 med 0.0028; r1 med 0.0020 |
| `r{i}_n_valid` | snapshots passing that mask | `:71` | r0 med 9,727 |
| `r{i}_bias` | circular mean of the **uncorrected** residual | `:83-84`, weighted for rover | r0 med 0.044 |
| `r{i}_circstd_raw` | circular SD before any correction | `:84` | r0 med 0.875 |
| `r{i}_outlier_frac` | fraction >1 rad from `bias` | `:85`, count unweighted, on the raw residual | r0 med 0.225 |
| `r{i}_coverage_bins` | θ bins (of 12) holding ≥20 samples | `:104-105`, subsampled | wall med 12; **rover med 8 (r0), min 0** |
| `r{i}_low_coverage` | `coverage_bins < 4` | `:106` | 25 receiver-instances across **16 datasets** (4 wall, 12 rover) |
| `r{i}_g` | fitted effective-spacing scale | grid search `:112`, or forced 1.0 | wall [0.70, 3.00]; rover [0.90, 1.10] |
| `r{i}_dtheta` | fitted mount/boresight shift | grid search `:112`, or forced 0.0 | wall ±0.35; rover ±0.90 |
| `r{i}_offset_c` | fitted constant offset `c` | analytic — `:61-63` / `:108` | r0 med 0.009 |
| `r{i}_circstd_corr` | **headline noise metric** — circular SD after (g, dθ, c) | `:112` / `:108` | r0 med 0.572; r1 med 0.628 |
| `r{i}_g_at_bound` | fit landed on a grid edge | `:114-117`; `abs(dθ)` catches **both** dθ bounds | 273 receiver-instances (124 via g, 168 via \|dθ\|) |
| `r{i}_drift_span` | spread of post-fit residual mean across 4 time quarters | `:119-123`, unweighted, full arrays | r0 med 0.210. **Values >π indicate wrap contamination** |
| `r{i}_structure` | dispersion of per-θ-bin residual means | `:125-135`, needs ≥4 usable bins else NaN | r0 med 0.283, NaN on 9 rows |

### 4.4 Cross-receiver decomposition (4, `DS`) — `:208-213`, rounded to 4 dp

| Column | Meaning | Computation |
|---|---|---|
| `dtheta_common` | common-mode mount shift (both receivers share it) | `(r0_dtheta + r1_dtheta)/2` |
| `dtheta_diff` | differential shift; **sign convention r0 − r1** | `(r0_dtheta − r1_dtheta)/2` |
| `heading_common` | **rover-only alias** of `dtheta_common` — compass bias | `:211-212`, blank on all wall rows |
| `mount_diff` | **rover-only alias** of `dtheta_diff` | `:213`, blank on all wall rows |

### 4.5 Verdict (2, `DS`)

| Column | Values |
|---|---|
| `status` | `OK` 727 · `FLAG` 1,176 · `QUARANTINE` 324 · `ERROR` 23 |
| `reasons` | `;`-joined tags (726 rows empty), or the exception text for ERROR rows |

---

## 5. Gate logic — the complete tag vocabulary

`gate(row)` — `:224-263`. Tags append in source order, so `reasons` strings are deterministic.
**`gate()` reads only 16 of the 48 columns**; the other 30 are diagnostic-only.

| Tag (exact string) | Platform | Trigger | Line | Count |
|---|---|---|---|---|
| `QUAR:nan>20%` | **wall only** | `max(nan0, nan1) > 0.20` | `:231-232` | 289 |
| `FLAG:nan5-20%` | **wall only** | `elif max(nan0, nan1) > 0.05` (mutually exclusive with above) | `:233-234` | 142 |
| `QUAR:frozen_tail(#42)` | **wall only** | `frozen_tail` is True | `:235-236` | 21 |
| `FLAG:r{0,1}_gain=<g>` | **wall only** | `abs(g − 1.0) > 0.25` | `:237-239` | r0 863, r1 1,251 |
| `FLAG:r{0,1}_noisy` | **both, different thresholds** | wall `circstd_corr > 0.8`; **rover `> 0.7`** | `:240-241`, `:250-251` | r0 493, r1 547 |
| `QUAR:no_signal` | **rover only** | `max(nan0, nan1) > 0.90` **OR** `min(n_valid) < 100` | `:243-246` | 15 |
| `FLAG:heading=<v>` | **rover only** | `abs(heading_common) > 0.25` | `:247-248` | 29 |
| `FLAG:ts_nonmonotonic` | both | `ts_nonmono_frac > 0.01` | `:252-253` | 3 |
| `FLAG:fit_at_bound` | both | `r0_g_at_bound or r1_g_at_bound` | `:254-255` | 222 |
| `INFO:low_coverage` | both | `r0_low_coverage or r1_low_coverage` | `:256-257` | 16 |
| `recovered_on_serial_retry` | n/a | appended outside `gate()` when a parallel-pass ERROR succeeded on serial re-scan | `:345` | 0 |

**Status derivation** (`:258-262`): any `QUAR:` tag ⇒ `QUARANTINE`; else any `FLAG:` tag ⇒ `FLAG`;
else `OK`. `INFO:` never changes status. `ERROR` is set outside `gate()` (`:218`), with
`" (persistent)"` appended when the serial re-check also failed — all 23 in this scan are persistent.

### 5.1 Traps when matching tags

1. The frozen-tail tag is **`QUAR:frozen_tail(#42)`**, not `QUAR:frozen_tail`.
2. `FLAG:r0_gain` / `FLAG:r1_gain` / `FLAG:heading` **always carry `=<value>`** (`%.2f`). Tag-matching
   code must prefix-match or split on `=`. Observed heading suffixes span **−0.81 … +0.39**.
3. The gate's `else` branch is **"rover OR unknown platform"**, not strictly rover.
4. **The two platforms use different quarantine tags.** Wall quarantines are `QUAR:nan>20%` /
   `QUAR:frozen_tail(#42)`; rover quarantines are `QUAR:no_signal`. Code that filters on a *tag
   string* rather than on `status == "QUARANTINE"` will silently miss one platform — this is exactly
   the live bug in `make_v2_splits.py` documented in
   [`TRAIN_VAL_SPLITS.md` §3](../04_training_inference/TRAIN_VAL_SPLITS.md#3-the-r2-quarantine-leak).

### 5.2 Platform threshold differences at a glance

| Concern | Wall | Rover |
|---|---|---|
| NaN quarantine | `> 20%` | `> 90%` **or** `n_valid < 100` |
| NaN flag | `5–20%` | *(none)* |
| Noisy threshold | `circstd_corr > 0.8` | `circstd_corr > 0.7` |
| Gain flag | `\|g−1\| > 0.25` | *(none — g is a diagnostic)* |
| Heading flag | *(none — no compass)* | `\|heading_common\| > 0.25` |
| Frozen tail | quarantines | *(none — rovers never freeze)* |
| Fit weighting | uniform | range², floored at 1 m |

**Rover NaN of 46–70% is normal** (bursty beacon), not damage — only `no_signal` and heading bias
are true rover failures. See `docs/learnings.md` L2.

---

## 6. Caveats to carry into any analysis

1. **`drift_span > π` is wrap contamination**, not real drift — the quarter means are circular but
   their max−min is linear.
2. **`FLAG:r*_gain` is not receiver/AGC gain.** It is the fitted effective/configured spacing ratio,
   and it flags *config families*, not individual captures (`learnings.md` L5).
3. **Low coverage silently suppresses** gain and at-bound flags (§3.4).
4. **`rx_lo` is sampled from one snapshot of one receiver** and is 0 Hz on 59 rows.
5. **`rx_speed_p99` and `frozen_max_frac` are computed but never gated.**
6. The scan is **read-only**; ERROR rows are diagnostics, not failures to fix in place.

## 7. Regenerating

```bash
python spf/scripts/dataset_quality_scan.py --splits <split1.txt> [...] \
  --precompute-cache /mnt/md2/cache/precompute_cache_3p7 \
  --output-dir data_quality_reports/scan_<YYYY_MM_DD> --parallel 12
```

Outputs `metrics.csv` + `report_wall.md` + `report_rover.md`. The PDF is built separately by
`pdf_scripts/dataset/rebuild.sh` → `dataset_quality_report_pdf.py`.
