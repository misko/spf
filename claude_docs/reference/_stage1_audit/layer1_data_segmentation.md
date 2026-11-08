# Stage-1 LAYER 1 audit — data → batch (segmentation + v5spfdataset + collate)

READ-ONLY audit. Config traced: `model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`
(identical to the dec15 single config). Launch: `python spf/scripts/train_single_point.py -c <cfg>`.
Venv torch = 2.9.1. Evidence pinned to current `main` (commit `574c454`).
Confidence: ✅ verified (read deciding lines / ran a check) · 🔍 analysis (reasoned, not executed).

---

## 0. Config flags that shape the batch (verified)

| Flag | Value | Effect on batch |
|---|---|---|
| `global.windowed_beamformer_input` | **true** | `windowed_beamformer` IS a batch key (not skipped) |
| `global.beamformer_input` | false | `weighted_beamformer` loaded into `data` but **not collated** (not in keys) |
| `global.signal_matrix_input` | false | `signal_matrix` skipped; `abs_signal_and_phase_diff` never built |
| `global.phase_input` | false | `weighted_windows_stats` **not** in batch |
| `global.empirical_input` | false | …but see Finding F1 — `empirical` STILL lands in the batch |
| `global.gains_input` | true | `gains` in batch |
| `global.vehicle_type_input` | true | `vehicle_type` in batch |
| `global.sdr_device_type_input` | true | `sdr_device_type` in batch |
| `global.rx_spacing_input` / `frequency_input` | true | carried via `rx_wavelength_spacing` / `rx_lo` (always in keys regardless of flag) |
| `global.n_radios` | 2 | `paired=True` ⇒ `__getitem__` returns BOTH receivers ⇒ collate doubles rows |
| `global.nthetas` | 65 | beamformer θ bins; `target_ntheta`=65 (model `output_ntheta` defaults to nthetas) |
| `datasets.flip` | **true** | per-row random left-right flip (see Finding F2) |
| `datasets.double_flip` | false | back-of-array reflection path OFF |
| `datasets.train_snapshots_per_session` | 1 | **S = 1** |
| `datasets.segmentation_version` | 3.7 | matches `SEGMENTATION_VERSION` (`spf/utils.py:14`) ✅ |
| `model.name` | beamformer | single-radio per-row model (`train…:1078` asserts flip⇒beamformer) |
| `windows_per_snapshot` | 256 (default, not in cfg) | == n_windows for this data ⇒ **no subsample** (Finding F4) |

Keys requested by collate = `global_config_to_keys_used(global)` **+ `"empirical"`** (appended
unconditionally at `train_single_point.py:323-325`).
Resolved key list: `all_windows_stats, rx_lo, rx_wavelength_spacing, y_rad, y_phi, craft_y_rad,
system_timestamp, y_rad_binned, craft_y_rad_binned, rx_pos_xy, tx_pos_xy, rx_theta_in_pis,
rx_heading_in_pis, windowed_beamformer, gains, vehicle_type, sdr_device_type, empirical`.

---

## 1. THE BATCH CONTRACT (verified) — the boundary the model layer depends on

`__getitem__` is paired ⇒ returns `[r0_dict, r1_dict]` per index. `v5_collate_keys_fast`
(`spf_dataset.py:113`) `vstack`s each key over **all (item, receiver)** dicts. With
`batch_size=256` and 2 receivers, effective **B = 256 × 2 = 512 rows**. `S = 1`, `n_windows = 256`,
`nthetas = 65`. `batch_size` of the returned `TensorDict` = `d["y_rad"].shape` = `(512,)`.

Per-receiver `render_session` shapes are listed; collated tensor = per-receiver with leading dim
replaced by B (vstack concatenates the leading singleton/`S` axis across rows).

| Batch key | Collated shape | dtype | Frame / meaning | Source line |
|---|---|---|---|---|
| `windowed_beamformer` | `(B, 1, 256, 65)` | **float32** (cast in collate `:119-120`; cache is f16) | per-window delay-and-sum beam power, θ axis = last | `data_from_precomputed:281`, collate `:119` |
| `all_windows_stats` | `(B, 1, 3, 256)` | **float32** (cast in collate) | ch0=trimmed circular mean φ (rad), ch1=stddev φ, ch2=median |abs| | `:296`, collate `:119` |
| `empirical` | `(B, 1, 65)` | target_dtype (float32) | prob dist over 65 θ bins, indexed by mean-phase bin | `render_session:1474` |
| `y_rad` | `(B, 1)` | float32 | **array-relative** θ (rad), flip-negated | `:1406` |
| `y_phi` | `(B, 1)` | float32 | φ target (rad), flip-negated | `:1407` |
| `craft_y_rad` | `(B, 1)` | float32 | **craft-relative** θ (rad), recomputed from flipped θ | `:1408,:1398` |
| `y_rad_binned` | `(B, 1, 1)` | long | bin idx of `y_rad` into 65 over [-π,π] | `:1480` |
| `craft_y_rad_binned` | `(B, 1, 1)` | long | bin idx of `craft_y_rad` | `:1483` |
| `rx_pos_xy` | `(B, 1, 2)` | float32 | meters (mm/1000) | `:1457` |
| `tx_pos_xy` | `(B, 1, 2)` | float32 | meters | `:1462` |
| `rx_theta_in_pis` | `(B, 1)` | float32 | receiver mount offset (multiples of π) | `get_values_at_key` |
| `rx_heading_in_pis` | `(B, 1)` | float32 | heading (multiples of π) | `get_values_at_key` |
| `rx_wavelength_spacing` | `(B, 1)` | float32 | d/λ dimensionless | cached `:587` |
| `rx_lo` | `(B, 1)` | float32 | LO freq (Hz) | `get_values_at_key` |
| `system_timestamp` | `(B, 1)` | float32 | seconds | `get_values_at_key` |
| `gains` | `(B, 1, 2)` | float32 | dB (note `[:,None]` at `:1369` ⇒ per-row `(1,1,2)`) | `:1369` |
| `vehicle_type` | `(B, 1)` | float32 | −0.5 wallarray / 0.5 rover | `:1377` |
| `sdr_device_type` | `(B, 1)` | float32 | SDRDEVICE enum value (0/2/3) | `:1380` |

Notes on the contract:
- **θ axis convention for `windowed_beamformer` is the LAST axis (dim 3)** — the flip flips `dims=(3,)`
  (`:1502`). The model layer must treat dim 3 as the 65 steering directions. ✅ verified.
- `all_windows_stats` θ-mean (ch0) is the only channel flipped (negated) under flip; stddev/amplitude
  are flip-invariant. ✅ verified `:1454`.
- Everything float16-in-cache is upcast: `windowed_beamformer`/`all_windows_stats` to f32 in collate;
  others to `target_dtype=torch.float32` in `render_session:1488-1494`.

---

## 2. Findings

### F1 — `empirical` is in the batch despite `empirical_input: false` — ✅ verified — affects THIS run: YES (benign)
`global_config_to_keys_used` does NOT add `empirical` (because `empirical_input` false,
`train_utils.py:91-92`). BUT `params_for_ds` force-appends it:
`if "empirical" not in keys_to_get: keys_to_get.append("empirical")` (`train_single_point.py:324-325`).
And the skip-fields line that would drop it is **commented out** (`train_single_point.py:126-127`).
Since `empirical_data_fn=full.pkl` is set, `render_session:1471` builds `data["empirical"]`
shape `(1,1,65)`. **Consequence:** `empirical` is delivered in every batch and is consumed by the
loss/target machinery **only as a shape reference** (`batch["empirical"].shape[:2]` in
`target_from_scatter`/`uniform_preds`, `train…:531,544,586`). It is **not** a model input feature
(the beamformer model has `empirical_input` false). So `empirical_symmetry: true` selects the `'sym'`
(65,65) table but the only thing the run uses from `empirical` is its first two dims `(B,1)`.
Severity P3 (works; just surprising — the model could equally use any `(B,1,*)` key for the shape).
🔍 If `full.pkl` ever lacked a needed `SDRDEVICE_spacing` key, `get_empirical_dist` would `KeyError`
inside the worker (reachable, since empirical is always built here).

### F2 — `flip: true` augmentation is **internally consistent**, but flips are drawn PER (receiver,row) — ✅ verified — affects THIS run: YES (acceptable)
`render_session` draws `flip_left_right = self.flip and rand()>0.5` **independently** each call
(`:1357`). Under one flip the following are all consistently transformed:
- `ground_truth_theta`(=`y_rad`) ← negated `:1389`; `ground_truth_phi`(=`y_phi`) ← negated `:1388`.
- `craft_y_rad` recomputed from the flipped θ + original `craft_offset` `:1398` (so it tracks the flip).
- `all_windows_stats[:,:,0]` (mean φ) ← negated `:1454`; `mean_phase_segmentation` ← negated `:1455`.
- `weighted_beamformer` ← `flip(dims=(2,))` `:1500`; `windowed_beamformer` ← `flip(dims=(3,))` `:1502`.
- `y_rad_binned`/`craft_y_rad_binned` are derived **after** the flip from flipped θ `:1480-1485`. ✅
**No field is flipped inconsistently** for THIS config's batch keys. The one documented inconsistency
(`abs_signal_and_phase_diff` "does not respect flipping", `:1411`) is **not reachable here** —
`signal_matrix` is skipped so that key is never built. ✅ verified.
**Caveat (🔍, not a bug for stage 1):** because the flip is per-call, r0 and r1 of the *same* session
can have different flip states. The "beamformer" model treats each (item,receiver) as an independent
training row (paired only loads both, doesn't fuse them), so divergent r0/r1 flips are fine. It WOULD
be a latent hazard for any paired/multi model that consumes both receivers jointly — see cross-layer
assertion A3. Also note `weighted_beamformer` is loaded (not in skip_fields) and flipped at `:1500`
but then **dropped at collate** (not a requested key) — no `KeyError`, just wasted work. ✅ verified.

### F3 — KNOWN_ISSUES #45 (`keep_signal_surrounded_by_noise` drops abutting signal) does **NOT** affect this config's `windowed_beamformer` — ✅ verified — affects THIS run: NO (for the model's input)
`keep_signal_surrounded_by_noise` (`segmentation.py:801-829`) drops signal windows adjacent to other
signal windows, running *after* `combine_windows`. It mutates only the **segmentation list / mask /
weighted stats / mean_phase**. The **`windowed_beamformer` is computed independently** as a
delay-and-sum over **all** windows (`segmentation.py:504-527`, `.reshape(nthetas,-1,window_size).mean(2).T`)
and is **never masked by segmentation**. The model in THIS config consumes `windowed_beamformer`
(unmasked, all 256 windows) and `all_windows_stats` (also all 256 windows — F4) and does **NOT**
receive `downsampled_segmentation_mask` (it is commented out of `global_config_to_keys_used`,
`train_utils.py:71`). Therefore #45 cannot corrupt the model's beamformer/stats input here.
**Where #45 still bites this run (indirectly, 🔍):** it affects `mean_phase`/`weighted_*`, which feed
(a) the QC gate in `__init__` (`average_windows_in_segmentation`, drift) and (b) the `empirical` index
via `mean_phase_segmentation`. But `skip_qc: true` (`ignore_qc=True`) disables the QC raises
(`:1138` guard), and `empirical` is shape-only (F1). Net effect on training signal: negligible. Refutes
the implicit worry that #45 degrades THIS config's trained features.

### F4 — `windows_per_snapshot` subsample is NOT triggered ⇒ reads are deterministic — ✅ verified — affects THIS run: NO (good)
`buffer-size = 524288` (verified in `dec18_mission1_rover1.yaml`) ⇒ `n_windows = 524288 // 2048 = 256`.
Default `windows_per_snapshot = 256` (`spf_dataset.py:855`, not overridden by cfg). The nondeterministic
subsample branches fire only when `shape > windows_per_snapshot` (`data_from_precomputed:285,301`);
`256 > 256` is false, so **no random `subsample_tensor`** — the documented "nondeterministic window
read" hazard does **not** apply to this config. ✅ verified. (Would silently activate if any dataset in
the split had a larger buffer; all apr17 nosig captures use 524288.)

### F5 — Train/val device asymmetry: train is `noblade`, val is NOT — ✅ verified — affects THIS run: YES (eval-only)
`train_paths = apr17_train_nosig_noroverbounce_**noblade**.txt` (1691 lines) vs
`val_paths = apr17_val_nosig_noroverbounce.txt` (565 lines). **No `…_val_…_noblade.txt` exists**
(verified `ls /mnt/md2/splits/apr17*`). So the val set may include BLADERF2 captures absent from
training → a covariate-shift in the *main* `val/single_loss` (the `save_on` metric). Mitigation already
present: per-device `alternate_val` dataloaders keyed on `sdr_device_type` (`train…:275-281`) surface
bladeRF separately. Severity P2 for metric interpretation (best-checkpoint selection on `val/single_loss`
is computed partly on an out-of-train-distribution device). Not a crash. 🔍 on magnitude.

### F6 — KNOWN_ISSUES #11 (mutable-default `skip_fields`) — ✅ verified — affects THIS run: NO
`v5spfdataset.__init__` binds `self.skip_fields = skip_fields` (`:887`) with no `+=`/mutation; the
caller passes a fresh `set` each time (`load_dataloaders` builds `skip_fields = set([...])`,
`train…:121`). The mutable-default footgun is real only in `v5inferencedataset` (`:506`, the `+=` path),
which is **not on the stage-1 training path**. ✅ verified — #11 does not affect this run.

### F7 — KNOWN_ISSUES #34 / #8 (destructive recursion on corrupt `.pkl` in `get_segmentation`) — ✅ verified — affects THIS run: conditionally
`get_segmentation` (`:1652`) is called with `segment_if_not_exist=False` for both train and val
(`train…:199,242,264`). On a `pickle.UnpicklingError` it `os.remove`s the `.pkl` then recurses
(`:1753-1760`); the recursion hits the not-exist `raise ValueError` (`:1688`) ⇒ **the cache is deleted
AND the load fails** (the dataset is then dropped by the `try/except` in `load_*_dataset`, silently
reducing the dataset count). A corrupt `.yarr` (vs `.pkl`) is **not** caught and raises uncaught.
Severity P1 *iff* a cache file is corrupt; with intact `precompute_cache_3p7` it never fires.
Affects THIS run: only on pre-existing cache corruption (not observed). 🔍 reachability.

### F8 — `skip_qc: true` consequences — ✅ verified — affects THIS run: YES (intended)
`ignore_qc=True` (`datasets.skip_qc`) routes around the three QC `raise`s (phi-drift, min-windows,
≥20% multi-segment) at `:1142-1153` (guarded by `if not self.ignore_qc`, `:1138`). **Consequence:**
datasets with high phi-drift, almost-no-signal windows, or degenerate segmentation are **admitted** to
training. Combined with F3, a session whose segmentation collapsed (e.g. #45 dropping all signal
windows) still contributes its (unmasked) `windowed_beamformer`/`all_windows_stats` and a possibly-NaN
`mean_phase_segmentation`. `mean_phase_segmentation` is **not a batch key here** (only `empirical`
consumes it, and NaN→uniform is handled `:1477-1478`), so NaN does not leak into the loss. The model's
inputs (`windowed_beamformer`, `all_windows_stats` ch0-2) are finite (f16 cast, no NaN injected on the
beamformer path). ✅ verified no-NaN-in-batch for this config's keys. Severity P2 (label/feature quality,
not a crash).

### F9 (NEW) — `all_windows_stats` amplitude channel is float16 in cache → saturates above 65504 — 🔍 analysis — affects THIS run: low
`all_windows_stats` is cast f16 before caching (`segmentation.py:570`) and re-upcast to f32 at collate.
Channel 2 (median |abs| amplitude) for strong captures can exceed f16 range (65504→inf) or lose integer
precision >2048. This feeds the model as a raw feature here (the model consumes `all_windows_stats`).
For the rover nosig captures amplitudes are typically well under 2048 so impact is minor, but it is a
real precision boundary on the model's input. P2/P3. (Same root as segmentation.md Issue S1.)

### F10 (NEW, minor) — collate `batch_size` silently depends on `y_rad` — ✅ verified — affects THIS run: NO (y_rad always present)
`v5_collate_keys_fast` sets `TensorDict(batch_size=d["y_rad"].shape)` (`:127`). `y_rad` is always in the
key list, so safe. Flagged only as a fragility for any future config that drops `y_rad`.

---

## 3. Reference-doc cross-check (independently verified)

| Doc claim | Verdict |
|---|---|
| `segmentation.md`: window_size=stride=2048, mask `//window_size` correct only when stride==window_size | ✅ confirmed (`DEFAULT_SEGMENT_ARGS:40-41`); holds here ⇒ mask indexing correct (mask not in batch anyway) |
| `segmentation.md` S5 (#45 drops abutting signal) | ✅ confirmed as written; **refined**: does NOT touch `windowed_beamformer` (F3) — doc didn't state this explicitly |
| `segmentation.md`: `trim` default — doc body says tests use 10; **production `DEFAULT_SEGMENT_ARGS` trim=20.0** | ⚠️ doc's prose ("trim=10") refers to the unit test; the real cached cache uses **trim=20.0** (`:42`). Minor doc ambiguity. |
| `segmentation.md`: windowed_beamformer per session `(1, n_windows, nthetas)`, f16 | ✅ confirmed (`:504-532,:632`) |
| `v5spfdataset.md`: render_session at `:1354`; output contract table | ✅ line numbers match current source; contract table accurate |
| `v5spfdataset.md`: `windowed_beamformer` batch shape `(1,S,≤windows_per_snapshot,nthetas)` | ✅ but for THIS config the `≤` is `==256` exactly (no subsample, F4) |
| `v5spfdataset.md`: SEGMENTATION_VERSION 3.7 (inventory "3.5" stale) | ✅ confirmed `spf/utils.py:14` |
| `v5spfdataset.md`: #11 mutable default harmless in v5spfdataset, live in v5inferencedataset | ✅ confirmed (F6) |
| `v5spfdataset.md`: #8 destructive recursion in get_segmentation | ✅ confirmed (F7) |
| `v5spfdataset.md` flip semantics list (`:1388-1455,:1500-1503`) | ✅ confirmed line-by-line; **adds** the per-receiver-independent-flip nuance (F2) not in the doc |
| `v5spfdataset.md`: `abs_signal_and_phase_diff` not flipped (`:1411`) | ✅ true but **unreachable here** (signal skipped) — doc says "training paths skip signal", consistent |
| `latest_models.md` | not present at `claude_docs/04_training_inference/latest_models.md` (path absent); could not cross-check — see note |

Note: `claude_docs/04_training_inference/latest_models.md` was not found at the given path during this
audit; if it exists elsewhere it was not cross-checked. The two dataset/segmentation reference docs are
otherwise accurate; the only substantive corrections are (a) trim=20 vs the test's 10, and (b) the
explicit statement that #45 does not reach `windowed_beamformer`.

---

## 4. Questions / assertions for the MODEL + TRAINING layers (cross-layer contract — to debate)

- **A1 (shape).** The model's `windows_stats_net` receives `windowed_beamformer` `(B,1,256,65)` and
  `all_windows_stats` `(B,1,3,256)` with **B=512 (=256×2 receivers)**, S=1, n_windows=256, θ on the
  LAST axis of windowed_beamformer (dim 3) and the LAST axis of all_windows_stats (dim 3 = windows, NOT
  θ). Assert the model indexes θ on windowed_beamformer dim −1 and treats all_windows_stats dim −1 as
  windows. If the model assumes `windows_per_snapshot`-variable counts it must handle exactly 256.
- **A2 (normalize flag).** `model.windows_stats_net.normalize_windowed_beamformer: false`. The batch
  delivers **raw f16-derived magnitudes upcast to f32** (per-window mean |beam|), unnormalized and
  un-masked. Assert the net's first layer is robust to absolute scale (amplitudes can be O(10²–10³));
  with `input_dropout:0.1` and `window_*` augmentations this is the only normalization. Debate whether
  the missing `downsampled_segmentation_mask` (NOT in batch) means the net must learn signal/noise
  window selection purely from `all_windows_stats` ch1 (stddev) — confirm the model does not silently
  expect a mask key.
- **A3 (flip coupling).** Flips are drawn **per (item,receiver)** independently (F2). For the single
  `beamformer` model this is fine (rows independent). **Assert no downstream code pairs r0/r1 of the
  same session expecting a shared flip** — if any paired/craft-fusion path is added it will see
  inconsistent `craft_y_rad` between the two rows of one session.
- **A4 (target frame).** Loss builds targets from `y_rad`/`y_rad_binned` (array-relative) for
  `single_loss`, and rotates predictions by `rx_theta_in_pis*π − craft_y_rad` etc. (`train…:769-783`).
  Assert the model's `single` output is in the **array-relative** θ frame (matching `y_rad`), since
  `single_loss` = `mse(output["single"], scatter(y_rad))` with no rotation (`:859-868`).
- **A5 (empirical is shape-only).** `empirical` `(B,1,65)` is in the batch but is **only** read for
  `.shape[:2]` by the target builders (F1). Assert the model does NOT consume `empirical` as a feature
  (consistent with `empirical_input:false`); if it does, that is an undocumented dependency.
- **A6 (dtype).** All float batch keys are **float32** (config `optim.dtype: torch.float32`, `amp:false`).
  `windowed_beamformer`/`all_windows_stats` upcast f16→f32 at collate; the cache precision (f16) is the
  true information content (F9). Assert the model does not re-cast to f16 expecting more precision than
  the cache holds.
- **A7 (val metric distribution).** `save_on: val/single_loss` is computed on a val split that may
  include BLADERF2 (train excludes it, F5). Assert checkpoint selection is acceptable under this
  device shift, or switch to a per-device alternate-val metric.
