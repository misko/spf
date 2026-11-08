# `v5spfdataset` — primary on-disk RF dataset · `spf/dataset/spf_dataset.py:818`

Reference doc produced by a Pass 1→3 deep dive per
[`DOCUMENTATION_GUIDE.md`](../../../DOCUMENTATION_GUIDE.md). Read-only audit; line numbers verified
against source at writing time (`spf/dataset/spf_dataset.py`). Units/frames link
[`conventions.md`](../../00_concepts/conventions.md). Status tags: ✅ Active · ⚠️ quirk · 🔴 broken · 🧪 unused.

`v5spfdataset(Dataset)` is the central read path for collected RF sessions: it opens a v5 (or
upgraded-v4) zarr + yaml, computes ground-truth angles from TX/RX positions, loads or generates a
segmentation/beamforming precompute cache, runs quality control, and serves per-(receiver,session)
feature dicts via `__getitem__`. The two module-level collate fns turn those dicts into training
batches.

---

## The angle frames (read first — #1 bug source)

Four θ frames coexist; all in **radians, π-normed to [-π, π]** via `torch_pi_norm` (`spf/rf.py:77`)
unless noted. Orientation fields `*_in_pis` are in **multiples of π** (`rx_theta_in_pis=0.5` ⇒ π/2).

| Key | Frame | Source |
|---|---|---|
| `ground_truth_theta` / `y_rad` | **array-relative** θ (emitter relative to antenna boresight) | `get_ground_truth_thetas` `:1573` |
| `absolute_theta` | **world frame** (raw `arctan2(dx,dy)` of TX−RX) | `get_ground_truth_thetas` `:1586` |
| `craft_ground_truth_theta` / `craft_y_rad` | **craft/vehicle-heading-relative** θ | `get_craft_ground_truth_thetas` `:1545` |
| `ground_truth_phi` / `y_phi` | phase-difference target (rad) derived from array-relative θ | `get_ground_truth_phis` `:1508` |

Relations (verified): `absolute = pi_norm(arctan2(dx,dy))` where `d = tx_pos_mm − rx_pos_mm`
(`:1581-1586`); `ground_truth_theta = pi_norm(absolute − (rx_theta_in_pis + rx_heading_in_pis)·π)`
(`:1587-1591`); `craft = pi_norm(ground_truth_theta + rx_theta_in_pis·π)` (heading deliberately *not*
added — keeps it craft-relative, `:1545-1553`); `phi = pi_norm(−sin(θ_array)·(d/λ)·2π)` (`:1508-1527`).
⚠️ The `−sin` sign is acknowledged as ambiguous in a long in-code comment (`:1518-1521`).

---

### `v5spfdataset.__init__(...)`  ·  `spf/dataset/spf_dataset.py:819`  ·  ✅ Active

**Summary.** Opens the zarr/yaml, derives geometry (wavelengths, `rx_wavelength_spacing = d/λ`,
steering vectors), caches per-receiver scalar/vector fields into tensors via `refresh()`, computes
all GT angle arrays, loads/generates the segmentation precompute cache via `get_segmentation()`, then
runs quality control. Establishes nearly every invariant the rest of the class depends on.

**Signature (key args)**
- Inputs (≈35 params; the load-bearing ones):
  - `prefix` (str) — path **without** `.zarr`/`.yaml` (it strips `.zarr`, `:906`). `{prefix}.zarr` and
    `{prefix}.yaml` must exist; `temp_file=True` appends `temp_file_suffix` (`.tmp`) to both (`:909-911`).
  - `nthetas` (int) — beamforming discretization; also names the precompute cache file (`results_fn` `:1636`).
  - `precompute_cache` (str, dir) — where `<prefix>_segmentation_nthetas{nthetas}.pkl/.yarr` live.
  - `paired` (bool, default False) — if True `__getitem__` returns **all receivers at once** (list len
    `n_receivers`); else one receiver per index.
  - `skip_fields` (annotated `List[str]=[]`, default `[]`) — fields to omit. ⚠️ **Callers pass a `set`**
    (conftest `:54`, test_fake_data `:71`), not a list. Stored as-is on `self.skip_fields` (`:887`);
    membership tests (`in`) and `set(...)` ops work for both, so the annotation is wrong but harmless
    here. (Contrast `v5inferencedataset.__init__:506` which does `+=` on its `[]` default → mutable-default
    bug; KNOWN_ISSUES #11. **`v5spfdataset` does NOT mutate its default** — confirmed `:887` is a plain bind.)
  - `gpu`, `segment_if_not_exist`, `n_parallel`, `skip_detrend` — forwarded to `mp_segment_zarr` via
    `get_segmentation`.
  - `ignore_qc` (bool) — bypass the QC `raise`s (`:1138`).
  - `temp_file` (bool) — in-progress recording mode; forces `snapshots_per_session==1` (`:912`), skips the
    `__init__`-time segmentation (defers to `render_session`), and asserts incompatible with QC (`:1141`).
  - `phi_drift_max=0.2`, `min_mean_windows=10` — QC thresholds.
  - `empirical_data_fn` (str|None), `empirical_individual_radio`, `empirical_symmetry` — optional learned
    φ→θ dist (pickle loaded `:1159`).
  - `v4` (bool) — treat receiver zarrs as v4 and upgrade on the fly (`v4_to_v5`).
  - `flip`/`double_flip` — augmentation flags consumed in `render_session`.
  - `distance_normalization=1000` — mm→m divisor for `*_pos_xy`.
  - `target_ntheta` (int|None) — bin count for `y_rad_binned` (defaults to `nthetas`, `:868`).
  - `segmentation_version` (float, default `SEGMENTATION_VERSION` = **3.7** at `spf/utils.py:14` — the
    inventory's "3.5" is stale).
- Returns: None (constructor).
- Side effects: **zarr/LMDB open** (×2: `:915` then again in `refresh` `:1170`, `map_size=2**32`); yaml read;
  optional empirical pickle read; **`get_segmentation` → `mp_segment_zarr`** (writes `.pkl`+`.yarr` cache,
  multiprocessing Pool, GPU) when cache missing and `segment_if_not_exist`; **raises `ValueError`** on QC
  failure (`:1143/1147/1151`) and many `assert`s (receiver-config consistency `:957-968`, `rx_theta_in_pis`
  median vs yaml `:1056-1064`, `n_antennas==2` `:997`).

**Called from** (Pass 2)
- `v5spfdataset_manager` (`:357`/`:364`) — context manager (handles `b2://` cache download); used by
  filters (`run_filters_on_data.py:122`), `segmentation_metrics.py`, `test_particle_filter.py`,
  `test_ekf.py`.
- Training: `train_single_point.py:59,78` (`preload_dataset` / `load_single_dataset_using_configs`).
- Inference: `single_point_networks_inference.py:59`.
- Scripts: `ls_ds.py`, `check_phi_error.py`, `segment_zarr.py`, `create_empirical_p_dist.py`,
  `fake_dataset.py:156`, `open_partial_ds.py`, `benchmark_dataset*.py`.
- Tests: 9 fixtures in `tests/conftest.py` (`:48,57,81,104,122,140,159,177,195`);
  `test_fake_data.py:18,39,65,129`; `test_partial.py:13`; `test_v5inference.py:40,79,138`;
  `test_zarr_tools.py:17,24`; `test_paired_rotate_dist.py:13`.
- Fan-in: very high (central class).

**Contract** (Pass 2)
- *Preconditions:* `{prefix}.zarr`+`.yaml` exist and parse; yaml has ≥2 receivers with identical
  `antenna-spacing-m`, wavelength, bandwidth (asserted `:957-968`); `r0.gains` is `(n_snapshots, 2)`;
  if `signal_matrix` not skipped, its shape is `(n_snapshots, 2, session_length)` (`:1024-1030`); for
  non-temp/non-v4 data, `rx_theta_in_pis` median must equal yaml `theta-in-pis` (`:1056`).
- *Postconditions/invariants established:* `n_antennas_per_receiver==2`; `len(rx_spacing mismatches)<30`
  between r0/r1 (`:1254`); `self.ground_truth_thetas/absolute_thetas/ground_truth_phis` are
  `(n_receivers, N)`, `craft_ground_truth_thetas` is `(N,)`; `self.mean_phase[f"r{i}"]` populated
  (real cache, or all-`inf` for temp `:1131`); QC guarantees (unless `ignore_qc`) phi-drift ≤ `phi_drift_max`,
  avg windows ≥ `min_mean_windows`, ≥20% sessions with >2 segments.
- *Hard requirement:* exactly **2 receivers** is assumed throughout (asserted in `get_ground_truth_thetas:1595`).

**Tested by** (Pass 3)
- `test_fake_data.py::test_dataset_load` / `::test_dataset_load_drift` — construct & read `phi_drifts`.
- `test_fake_data.py::test_fake_data_array_orientation` (`:65`) — the strongest contract test: asserts
  `ground_truth_phis ≈ mean_phase` (φ correctness, atol 1e-3) and
  `craft_ground_truth_theta == arctan2(dx,dy)` (`:76-96`). (Uses `n_parallel=0`.)
- QC `raise` paths exercised indirectly by `test_dataset_load_drift`.

**Analysis** (Pass 3)
- *Correctness:* construction order is sound — geometry → `refresh()` (caches keys + GT angles) →
  `get_segmentation()` → drift/QC. The double zarr open (ctor `:915` and `refresh` `:1170`) is redundant
  but harmless (refresh re-binds `self.z`).
- *Issues:* (1) `skip_fields` type annotation (`List[str]`) contradicts every real call site (a `set`);
  not a bug, but misleading. (2) The `assert ... rx_theta_in_pis.median()==yaml theta-in-pis` (`:1056`)
  uses exact float `==` on a `.median()` — brittle if data ever stores non-identical values; fine for
  synthetic data where all rows equal. (3) Mutable default `skip_fields=[]` is harmless here (not mutated)
  but is a footgun — see `v5inferencedataset` for the live version (KNOWN_ISSUES #11). (4) QC raising inside
  `__init__` means a bad dataset can't be opened even read-only without `ignore_qc=True`.
- *Missing coverage:* no test asserts the QC `ValueError` messages fire on a deliberately-bad dataset; no
  test covers `temp_file=True` construction at this level (covered via `v5spfdataset_manager` in `test_ekf`).

**Open questions.** The `−sin` sign convention in `get_ground_truth_phis` is validated empirically
(`test_fake_data` φ≈mean_phase) but its *physical* justification stays a TODO comment.

---

### `render_session(receiver_idx, session_idx, double_flip=False)`  ·  `:1354`  ·  ✅ Active

**Summary.** Builds the full feature dict for one receiver of one session: raw cached fields, GT angles
(with optional flip augmentation), precomputed segmentation/beamformer slices, mean-phase, empirical
dist, and binned targets. This is the body of `__getitem__`. **This is the authoritative source of the
output-dict contract** (see table below).

**Signature**
- Inputs: `receiver_idx` (int, 0..n_receivers−1); `session_idx` (int, 0..n_sessions−1); `double_flip`
  (bool) — extra augmentation (passed down from `__getitem__`).
- Returns: `data` (dict[str, Tensor|...]) — see **output contract** below.
- Side effects: calls `reinit()` (re-opens zarr + segmentation in worker processes); **randomness**
  (`torch.rand` for `flip_left_right` when `self.flip`, `:1357`); for `temp_file`, may call
  `get_segmentation(...precompute_to_idx=session_idx...)` → heavy compute (`:1420-1425`); dtype casts to
  `self.target_dtype` (`:1488-1494`).

**Called from:** `__getitem__` (`:1613,:1621`). Fan-in: 1 (but transitively every consumer of `ds[idx]`).

**Output dict contract** (verified against `render_session` body, `:1362-1506`).
Per-snapshot leading dim is **1** (unsqueezed); `S = snapshots_per_session` (default 1). "frame" column
applies to θ keys.

| Key | Shape (S=snapshots) | Units / frame | Source line |
|---|---|---|---|
| `<v5rx_f64_keys…>` e.g. `rx_theta_in_pis`, `rx_spacing`, `rx_lo`, `rx_bandwidth`, `rx_heading_in_pis`, `avg_phase_diff`, `rssis`, `tx_pos_x_mm`,`tx_pos_y_mm`,`rx_pos_x_mm`,`rx_pos_y_mm` | `(S,)` (`avg_phase_diff` `(S,2)`) | as stored (pis / m / Hz / mm) | `:1362-1367` (`get_values_at_key`) |
| `rx_wavelength_spacing` | `(S,)` | d/λ dimensionless | cached `:1237` |
| `gains` | `(S,1,2)` | dB | `:1369` |
| `signal_matrix` (unless skipped) | `(1,2,session_length)` | complex64 | `:1364` |
| `abs_signal_and_phase_diff` (if signal kept) | `(1,2,session_length+1)` | abs amplitude ++ φ (rad) | `:1415` |
| `receiver_idx` | `(1,S)` int32 | — | `:1370` |
| `ground_truth_theta` / `y_rad` | `(S,)` | rad, **array-relative** | `:1371,:1406` |
| `absolute_theta` | `(S,)` | rad, **world** | `:1374` |
| `ground_truth_phi` / `y_phi` | `(S,)` | rad, φ target | `:1375,:1407` |
| `craft_ground_truth_theta` / `craft_y_rad` | `(S,)` | rad, **craft-relative** | `:1376,:1408` (recomputed `:1398`) |
| `vehicle_type` | `(1,)` | −0.5 wallarray / 0.5 rover | `:1377` |
| `sdr_device_type` | `(1,)` | `SDRDEVICE` enum value | `:1380` |
| `windowed_beamformer` (unless skipped) | `(1,S,≤windows_per_snapshot,nthetas)` | float (beamformer power) | `data_from_precomputed:281` |
| `weighted_beamformer` | `(1,S,nthetas)` | float | `:291` |
| `all_windows_stats` | `(1,S,3,≤windows_per_snapshot)` | ch0=mean φ, ch1=stddev φ, ch2=median abs | `:297` |
| `weighted_windows_stats` | `(1,S,3)` | as above | `:307` |
| `downsampled_segmentation_mask` | `(1,S,1,n_windows)` | bool/float mask | `:311` |
| `simple_segmentations` (unless skipped) | list[S] of segment-dict lists | sample indices | `:320` |
| `mean_phase_segmentation` | `(1,S)` | rad (NaN where no signal) | `:1441` |
| `empirical` (if empirical_data) | `(1,S,emp_nbins)` | prob dist over θ bins | `:1474` |
| `rx_pos_xy` | `(1,S,2)` | **meters** (mm/`distance_normalization`) | `:1457` |
| `tx_pos_xy` | `(1,S,2)` | **meters** | `:1462` |
| `y_rad_binned` | `(1,S)` long | bin idx into `target_ntheta` over [-π,π] | `:1480` |
| `craft_y_rad_binned` | `(1,S)` long | bin idx | `:1483` |

**Flip/augmentation semantics** (Pass 3, `:1357,1387-1505`):
- `flip_left_right` (random, when `self.flip`) **or** `double_flip` ⇒ negate `ground_truth_phi`,
  `ground_truth_theta` (`:1388-1389`), negate `all_windows_stats[:,:,0]` (mean φ) and
  `mean_phase_segmentation` (`:1454-1455`), flip `weighted_beamformer`/`windowed_beamformer` along θ
  axis (`:1500-1503`). `craft_ground_truth_theta` is then recomputed from the (possibly flipped) θ +
  the original `craft_offset` (`:1398`).
- `double_flip` additionally maps θ → `sign(θ)·π − θ` (back-of-array reflection, `:1392-1395`) and negates
  `rx_pos_xy`/`tx_pos_xy` (`:1467-1469`).
- ⚠️ `:1411` comment "WARNING this does not respect flipping!" — `abs_signal_and_phase_diff` is computed
  from the raw signal *before* any flip and is **not** flipped; consumers using both flipped GT and this
  raw key would see inconsistency. Low severity (key only built when `signal_matrix` kept; training paths
  skip signal).

**Issues:** (a) `data_from_precomputed` randomly subsamples windows when
`shape > windows_per_snapshot` (`:285-288,301-304`, self-flagged "hacky and not right") → **reads are
nondeterministic** in that regime. (b) `mean_phase_segmentation` can be NaN; only the `empirical` path
guards it (`:1477`); a TODO (`:1445`) notes NaN isn't handled elsewhere. (c) For empirical, `to_bin`
clamps θ to [-π, π−1e-5] (`spf/utils.py:251`) and maps NaN→bin 0 then overwrites with uniform — fine.

**Tested by:** `test_fake_data_array_orientation` (reads `tx_pos_*`, `rx_pos_*`, `craft_ground_truth_theta`
per `ds[idx][0]`); `test_partial.py` compares rendered dicts across partial datasets;
`test_v5inference.py::test_preprocessing_equal` compares 20 rendered samples to the inference path.
No test exercises `flip`/`double_flip` augmentation paths.

---

### `__getitem__(idx)`  ·  `:1604`  ·  ✅ Active

**Summary.** Maps a flat index to rendered session(s). Paired mode returns
`[render_session(r, idx) for r in receivers]` (one per receiver, same session); unpaired returns a
1-element list for the single receiver `idx % n_receivers`.

**Signature**
- Input: `idx` (int).
- Returns: `list[dict]` — length `n_receivers` if `paired` else 1; **or `None`** for `temp_file` when the
  requested index exceeds `valid_entries` (`:1609,:1619`).
- Side effects: `double_flip = self.double_flip and rand()>0.5` (paired only, `:1611`) → randomness;
  asserts `idx < n_snapshots` (paired) / `< n_snapshots*n_receivers` (unpaired) (`:1606,:1616`).

**Contract:** paired index space is `[0, n_snapshots)`; unpaired is `[0, n_snapshots*n_receivers)`.
⚠️ Note `__getitem__` asserts against `n_snapshots`, while `__len__` returns `n_sessions`
(`:1284`). When `snapshots_per_session>1` or non-unit stride, `n_sessions < n_snapshots`, so the
assert is looser than `__len__` — callers must index within `len(ds)`, not up to `n_snapshots`.
Returned list is consumed by collate fns (each element is "one receiver's dict").

**Tested by:** every test that does `ds[idx]` / `ds[idx][0]` / `ds[idx][r_idx]`
(`test_fake_data`, `test_partial`, `test_v5inference`, `test_paired_rotate_dist`).
**Missing:** no test asserts the `None` temp-file return path nor the unpaired index arithmetic.

---

### `__len__()`  ·  `:1284`  ·  ✅ Active

Returns `n_sessions` (paired) or `n_sessions * n_receivers` (unpaired). `n_sessions` is derived in
`__init__` from tiling mode (`:1004-1014`). No side effects. ⚠️ Subtle mismatch with `__getitem__`'s
`n_snapshots`-based assert (above). Tested indirectly via `range(len(ds))` loops in `test_fake_data`.

---

### `get_ground_truth_thetas()`  ·  `:1573`  ·  ✅ Active

**Summary.** From cached `tx_pos_mm`/`rx_pos_mm`, computes per-receiver **absolute** θ
(`arctan2(dx, dy)` of TX−RX, π-normed) and **array-relative** θ (absolute minus
`(rx_theta_in_pis + rx_heading_in_pis)·π`). Returns both stacked.

**Signature** — Inputs: none (reads `self.cached_keys`). Returns: `(ground_truth_thetas, absolute_thetas)`,
each `(n_receivers, N)` rad. Side effects: `assert n_receivers==2` (`:1595`).

**Called from:** `refresh()` `:1243`. **Contract:** requires `cached_keys[ridx]` populated with
`tx_pos_mm`, `rx_pos_mm` (built in `refresh` `:1223-1235`), `rx_theta_in_pis`, `rx_heading_in_pis`.
Note `d[0]=Δx`, `d[1]=Δy`, so `arctan2(Δx, Δy)` — x-from-y convention (matches `test_fake_data:96`).

**Analysis.** ✅ Verified against `test_fake_data_array_orientation:96` (`arctan2(dx,dy)` equals the
derived craft theta). Hard 2-receiver assert. No wraparound risk (`torch_pi_norm` applied). The "pick the
front candidate" comment (`:1592-1594`) is aspirational — **no candidate-resolution code is present**;
GT is taken directly from positions, so front/back ambiguity is resolved by *knowing* the true TX
position, not by the array. Issue: low — naming may mislead readers into thinking ambiguity logic exists here.

---

### `get_ground_truth_phis()`  ·  `:1508`  ·  ✅ Active

**Summary.** Converts array-relative GT θ into the expected phase difference:
`φ = pi_norm(−sin(θ_array) · (d/λ) · 2π)`.

**Signature** — Inputs: none (uses `self.ground_truth_thetas`, `self.rx_wavelength_spacing`). Returns:
`(n_receivers, N)` rad, π-normed. Side effects: none.

**Called from:** `refresh()` `:1246`; result feeds `get_all_phi_drifts` (QC) and `y_phi`/`ground_truth_phi`.
**Contract:** must run after `ground_truth_thetas` is set. **Tested by:** `test_fake_data_array_orientation:76`
asserts `ground_truth_phis ≈ mean_phase` (atol 1e-3) — strong validation that the sign/scale are correct
for the synthetic generator. **Issue:** the `−sin` sign is the documented ambiguity (`:1518-1521`); correct
empirically but physically unexplained. Confidence high that it matches the rest of the pipeline.

---

### `get_craft_ground_truth_thetas()`  ·  `:1545`  ·  ✅ Active

**Summary.** Rotates array-relative GT θ into the craft/vehicle frame by adding `rx_theta_in_pis·π`
(the receiver's mounting offset). Heading is **deliberately not** added (a commented line `:1550`
notes adding it would yield the absolute frame instead). Cross-checks all receivers agree.

**Signature** — Inputs: none. Returns: `(N,)` rad (from receiver 0; others only validated). Side effects:
`logging.warning` if receiver-1's craft θ differs from receiver-0's by >0.2 mean abs (`:1566-1569`).

**Called from:** `refresh()` `:1247`. **Contract:** `cached_keys[ridx]["rx_theta_in_pis"]` present;
GT thetas set. **Tested by:** `test_fake_data_array_orientation` (the `craft_theta == arctan2` assert),
`test_paired_rotate_dist`. **Issue:** returns only receiver-0's value; receiver disagreement only *warns*,
never raises — a genuinely misaligned craft would silently use r0. Severity low (synthetic data agrees).

---

### `get_segmentation(version, segment_if_not_exist, precompute_to_idx=-1)`  ·  `:1652`  ·  ✅ Active

**Summary.** Loads the segmentation `.pkl` + beamforming `.yarr` precompute cache for this dataset,
generating it via `mp_segment_zarr` if missing (and allowed), verifying the version, and loading
`mean_phase`. Idempotent for non-temp files (returns the cached `self.segmentation` if already set, `:1672`).

**Signature**
- Inputs: `version` (float) — expected seg version, compared with `np.isclose` (`:1764`);
  `segment_if_not_exist` (bool) — generate if `.pkl` absent; `precompute_to_idx` (int, −1=all) — temp-file
  partial precompute.
- Returns: `self.segmentation` (dict; `{}` when `simple_segmentations` skipped, `:1724`).
- Side effects: **heavy** — `mp_segment_zarr` (writes `.pkl`+`.yarr`, multiprocessing Pool, GPU when
  `self.gpu`) `:1706`; pickle read `:1722`; zarr open `:1732`; sets `self.segmentation`,
  `self.precomputed_zarr`, `self.precomputed_entries` (temp), and `self.mean_phase` (`get_mean_phase`
  `:1775`). **Raises `ValueError`** if cache missing and not allowed (`:1688`) or version mismatch (`:1765`).
  **On `pickle.UnpicklingError`: `os.remove(results_fn)` then recurses** (`:1753-1760`).

**Called from:** `__init__:1080`, `reinit:1316`, `render_session:1421` (temp). Fan-in: 3 (all internal,
all reachable). **Contract:** for non-temp, the cache must end up matching `version` (else raises);
populates `mean_phase` used by drift QC and `mean_phase_segmentation`.

**Tested by:** every fixture with `segment_if_not_exist=True` exercises the *generate* path
(conftest, `test_fake_data`, `test_paired_rotate_dist`). The *load-existing* and version-mismatch paths are
exercised by repeated opens of the same cache within a session (conftest opens nthetas=65 then 7).

**Analysis (Pass 3) — confirms KNOWN_ISSUES #8.**
- 🔴 **Destructive recursion on corrupt cache** (`:1753-1760`): on `UnpicklingError`, it `os.remove`s the
  `.pkl` and recurses with the same args. If `segment_if_not_exist=False`, the recursion then hits the
  "file does not exist" `raise ValueError` (`:1688`) — so it deletes the user's cache *and* fails. If
  `segment_if_not_exist=True`, it regenerates (heavy) — acceptable but silent data loss of the old cache.
  Recursion is bounded (regeneration writes a valid file, or the not-exist branch raises) — so unbounded
  recursion is **unlikely** in practice, refining the inventory's "risks unbounded recursion": the second
  pass either regenerates or raises. Still: **deletes a file on a parse error**, P0-worthy.
- ⚠️ The `.yarr` open (`:1732`) is **not** wrapped by the `try/except UnpicklingError`, so a corrupt `.yarr`
  (vs `.pkl`) raises uncaught — only `.pkl` corruption triggers the delete-and-retry.
- The "version mismatch" `ValueError` message (`:1766`) is misleading ("Expected to generate…") even when no
  generation was attempted.

**Missing coverage:** no test feeds a deliberately-corrupt `.pkl` (the delete-and-recurse path is untested);
no test asserts the version-mismatch raise.

---

## Module-level collate functions (the training handoff)

### `v5_collate_keys_fast(keys, batch)`  ·  `:113`  ·  ✅ Active

**Summary.** The primary training collate. `batch` is a list over batch items, each a list over receivers
(2 when paired), each a `dict[str,Tensor]` (one `render_session` output). For each requested `key`,
`vstack`s that key across **all (item, receiver)** dicts into one tensor; casts `windowed_beamformer` /
`all_windows_stats` to float32 (`:119`). Returns a `TensorDict` with `batch_size = d["y_rad"].shape`.

**Signature** — Inputs: `keys` (List[str]); `batch` (List[List[Dict[str,Tensor]]]). Returns:
`TensorDict`. Side effects: none.

**Called from:** `train_single_point.py:329` (via `partial`, keys = `global_config_to_keys_used(cfg)` +
`"empirical"`); `benchmark_dataset*.py:72/78` (static key list incl. `all_windows_stats, rx_pos_xy,
downsampled_segmentation_mask, rx_spacing, y_rad, craft_y_rad, y_phi`);
`single_point_networks_inference.py:48,73` and `particle_single_radio_nn_filter.py:81` and
`spf_nn_dataset_wrapper.py:117` (single-sample inference). Fan-in: ~6.

**Contract:** every requested key must exist in every dict (KeyError otherwise) → callers must request only
keys `render_session` actually produces given the `skip_fields`. `"y_rad"` must be present (used for
`batch_size`). All dicts must share compatible shapes for `vstack`. **Postcondition:** returns a TensorDict
keyed exactly by `keys`. **Issue:** `batch_size=d["y_rad"].shape` — if `y_rad` not requested, KeyError;
silent dependency. **Tested by:** indirectly through training/inference integration (`test_v5inference`,
`test_particle_filter`); no dedicated unit test.

### `v5_collate_beamsegnet(batch)`  ·  `:140`  ·  ✅ Active

**Summary.** Builds a fixed-key training dict for the beam-seg net: vstacks `y_rad, y_phi, rx_pos_xy,
receiver_idx (expanded to y_rad shape), craft_y_rad, rx_spacing (reshaped (-1,1))`, and the numpy-backed
`all_windows_stats`/`windowed_beamformer` (via `np.vstack` → `torch.from_numpy`), plus
`downsampled_segmentation_mask`. If `"x"` present in the first dict, also adds full-length `x` and a
`v5_segmentation_mask(x)` per sample (`:171-173,188-190`).

**Signature** — Inputs: `batch` (paired list-of-list-of-dict). Returns: `dict` (~9–11 keys). Side effects: none.

**Called from:** only `simple_train.py:198` (**commented out** — `v5_collate_all_fast` is used instead).
So this is **🧪 effectively unused in active code paths** (the only live caller is a commented line in a
notebook script). Refines the inventory's "✅ Active": the function is correct but currently dormant.

**Contract:** requires `all_windows_stats` (numpy, `:141` reads `.shape[1]`), `y_rad`, `y_phi`, `rx_pos_xy`,
`craft_y_rad`, `rx_spacing`, `windowed_beamformer`, `downsampled_segmentation_mask`, `receiver_idx` in every
dict. `all_windows_stats`/`windowed_beamformer` must be **numpy** arrays (uses `np.vstack`) — but
`render_session` returns them as **torch tensors**; `np.vstack` on torch tensors works via array protocol,
producing an ndarray, so `torch.from_numpy` succeeds. ⚠️ This implicit torch→numpy round-trip is fragile and
untested. **Tested by:** none (no live caller, no test).

### `v5_collate_all_fast(batch)`  ·  `:131`  ·  ✅ Active (notebook only)

Vstacks **every** key found in `batch[0][0]` across all (item,receiver) dicts; returns a plain dict.
Carries a commented-out `@torch.jit.script` (`:130`). Only caller: `simple_train.py:197`. No tests.
**Contract:** all dicts must share the same key set and vstack-compatible shapes; will choke on list-valued
keys like `simple_segmentations` if present (it only handles Tensors via `vstack`). Low risk since notebook
controls the skip_fields.

---

## Minor helpers (documented briefly)

- `refresh()` `:1166` — re-opens zarr, (re)loads per-receiver scalar/vector keys into `self.cached_keys`
  as float32 tensors, builds `rx_pos_mm`/`tx_pos_mm`/`rx_wavelength_spacing`, then recomputes all GT angle
  arrays. Returns True if it reloaded, False if `valid_entries` unchanged. Asserts <30 rx_spacing
  mismatches (`:1254`). Incremental-update branch (`old_n>0`, `:1214`) appends only new rows — but the
  `rx_pos_mm` etc. are rebuilt every time. **Issue:** the incremental path mutates in place while the
  GT arrays are fully recomputed; only the full-reload path (`old_n==0`) is exercised by tests.
- `reinit()` `:1305` — DataLoader-worker support: if `self.z is None` (post-fork via `close()`/worker
  init), re-opens zarr and re-loads segmentation. Called at the top of `render_session`.
- `v4_to_v5()` `:1289` — wraps each receiver zarr in `ZarrWrapper` and synthesizes v5 keys
  (`rx_heading_in_pis = heading/360/2`, zero-fills missing f64 keys). No-op unless `v4`.
- `get_mean_phase()` `:1598` — loads per-receiver `mean_phase` from the precompute `.yarr`.
- `get_all_phi_drifts()` `:1529` — per-receiver `pi_norm(mean_phase − ground_truth_phi)`; feeds QC.
- `estimate_phi(data)` `:1276` — seg-mask-weighted mean of `all_windows_stats` (`+0.001` denom guard).
- `get_values_at_key` `:1339` — zarr read for `signal_matrix`, cached tensor otherwise.
- `close()` `:1265` — closes zarr stores, nulls data/segmentation refs (enables worker re-open).
- `get_estimated_thetas()` `:1623` — per-receiver θ from `mean_phase` via `phase_diff_to_theta`
  (`large_phase_goes_right=False`). ⚠️unverified callers.

---

## Module cluster summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `__init__` | ✅ Active | very high | partial (load only) | QC raises in-ctor; `skip_fields` annotated List but callers pass `set`; mutable default present but unmutated |
| `render_session` | ✅ Active | 1 (via `__getitem__`) | indirect | Authoritative output-dict contract; `abs_signal_and_phase_diff` not flipped (`:1411`); nondeterministic window subsample |
| `__getitem__` | ✅ Active | very high | indirect | Asserts vs `n_snapshots` while `__len__`=`n_sessions`; `None` temp path untested |
| `__len__` | ✅ Active | high | indirect | Index-space mismatch with `__getitem__` |
| `get_ground_truth_thetas` | ✅ Active | 1 (`refresh`) | ✅ (`test_fake_data:96`) | Hard 2-rx assert; "front candidate" comment is aspirational (no resolution code) |
| `get_ground_truth_phis` | ✅ Active | 1 (`refresh`) | ✅ (`test_fake_data:76`) | `−sin` sign empirically correct, physically unexplained |
| `get_craft_ground_truth_thetas` | ✅ Active | 1 (`refresh`) | ✅ | Returns r0 only; receiver disagreement only warns |
| `get_segmentation` | ✅ Active | 3 | generate-path only | 🔴 deletes `.pkl` on UnpicklingError then recurses (KNOWN_ISSUES #8); only `.pkl` corruption caught |
| `v5_collate_keys_fast` | ✅ Active | ~6 | indirect | Silent `y_rad` dependency for `batch_size`; no unit test |
| `v5_collate_beamsegnet` | 🧪 dormant | 1 (commented) | none | Only live caller is commented out; torch→numpy via `np.vstack` is fragile |
| `v5_collate_all_fast` | ✅ Active (notebook) | 1 | none | Tensor-only; would break on list keys |

**Cross-cutting themes.** (1) **Angle frames** (array / craft / absolute / φ) are the correctness core; the
two GT-angle tests in `test_fake_data` are the only direct validators and they only cover the
synthetic/no-flip path — flip/double_flip augmentation and the φ sign convention are **untested**.
(2) **Hardcoded 2-receiver** assumption pervades (`assert n_receivers==2`, `[0,1]` literals at `:1121`).
(3) **Caching side effects** dominate: `get_segmentation` is destructive on corrupt caches (#8), and reads
are **nondeterministic** when windows exceed `windows_per_snapshot`. (4) The `skip_fields` parameter is
typed `List[str]` but universally called with a `set`; harmless but a documentation/typing defect, adjacent
to the genuine mutable-default bug in `v5inferencedataset` (#11). (5) `__getitem__`/`__len__` use different
index denominators (`n_snapshots` vs `n_sessions`) — safe only if callers stay within `len(ds)`.
