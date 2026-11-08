# Pass-1 Surface Inventory — `spf/dataset/` + data-wrangling scripts

Read-only Pass-1 inventory per `claude_docs/DOCUMENTATION_GUIDE.md`. Line numbers verified
against source at inventory time. Status tags: ✅ Active · ⚠️ Deprecated/legacy/quirk · 🔴 Broken
(verified) · 🧪 Experimental/unused. `⚠️unverified` marks claims not fully confirmed in Pass-1.

Units/frames are noted where inferable. Angles are radians unless a name says `_in_pis` (units of
π) or `_mm`/`_m`/`_xy` (positions). `rx_wavelength_spacing` = d/λ (dimensionless).

---

## spf_dataset.py

Module-level helpers, then the two priority classes (`v5inferencedataset`, `v5spfdataset`).

### Module-level functions / classes / constants

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `yaml_as_dict` | 47 | function | Flatten a multi-doc YAML file into one dict | `my_file` (path str) | `dict` | file read (open) | ✅ Active (utility; ⚠️unverified callers) |
| `pos_to_rel` | 57 | function | Map absolute position to [-1,1] relative coord | `p` (mm/units), `width` | same type as `p` | none | ✅ Active |
| `rel_to_pos` | 61 | function | Inverse of `pos_to_rel` | `r` ([-1,1]), `width` | position | none | ✅ Active |
| `encode_vehicle_type` | 65 | function | Encode vehicle string to scalar (-0.5 wallarray / 0.5 rover) | `vehicle_type` (str) | float | raises via `assert 1==0` on unknown | ✅ Active (assert-on-unknown) |
| `dotdict` | 74 | class(dict) | Dot-access dict subclass | — | dict instance | none | ✅ Active (utility) |
| `v5_thetas_to_targets` | 84 (decor `@torch.jit.script` @83) | function | Gaussian soft-label over nthetas bins from target thetas | `target_thetas` (Tensor, rad), `nthetas` (int), `range_in_rad` (float), `sigma`=1 | Tensor (N,nthetas), rows sum to 1 | none (JIT-scripted) | ✅ Active |
| `v5_collate_keys_fast` | 113 | function | Collate selected keys across paired batch into TensorDict | `keys` (List[str]), `batch` (list of list of dict[str,Tensor]) | `TensorDict` | none | ✅ Active |
| `v5_collate_all_fast` | 131 | function | Collate ALL keys across paired batch (vstack) | `batch` | `dict[str,Tensor]` | none | ✅ Active (commented `@torch.jit.script` @130) |
| `v5_collate_beamsegnet` | 140 | function | Build training batch dict for beamsegnet (vstacks many keys; numpy for stats/beamformer) | `batch` (paired list) | `dict` (~10-12 keys) | none | ✅ Active |
| `get_idx_for_rand_session` | 197 | function | Random non-uniform increasing snapshot idxs | `max_offset` (int), `n` (int) | ndarray[int] (n,) | randomness (`np.random`) | ✅ Active |
| `v5_downsampled_segmentation_mask` | 206 | function | Build per-window binary mask from simple_segmentations | `session` (dict), `n_windows` (int) | Tensor (S,1,n_windows) | none (window_size hardcoded 2048) | ✅ Active ⚠️unverified callers |
| `v5_segmentation_mask` | 224 | function | Per-sample full-length binary mask from simple_segmentation | `session` (dict w/ `x`,`simple_segmentation`) | Tensor (1,1,1,samples) | none | ✅ Active (used in `v5_collate_beamsegnet`) |
| `get_empirical_dist` | 233 (decor `@cache` @232) | function | Look up empirical phase→angle dist for receiver | `v5ds`, `receiver_idx` (int) | ndarray/Tensor dist | `@cache` memoization keyed on (v5ds,idx) | ✅ Active (cache may retain ds refs) |
| `get_session_idxs` | 244 | function | Compute snapshot index array for a session (tiled/disjoint/random) | `session_idx`, `tiled_sessions` (bool), `snapshots_stride`, `snapshots_adjacent_stride`, `snapshots_per_session`, `random_adjacent_stride` (bool) | ndarray[int] | randomness if `random_adjacent_stride` | ✅ Active |
| `data_from_precomputed` | 278 | function | Slice precomputed yarr fields into per-session tensors (respecting skip_fields) | `v5ds`, `precomputed_data` (zarr grp), `segmentation` (list/dict), `snapshot_idxs` | `dict` of tensors | randomness via `subsample_tensor` (idx>windows_per_snapshot) | ✅ Active (TODO-flagged hacky subsample @284,300) |
| `v5spfdataset_manager` | 329 (decor `@contextmanager` @328) | function | Context manager wrapping `v5spfdataset`, with b2:// remote cache download | `*args`, `**kwds` (must incl `precompute_cache`) | yields `v5spfdataset` | network (b2 download) if `b2:` prefix; closes ds on exit; KeyError if no `precompute_cache` | ✅ Active |
| `subsample_tensor` | 371 | function | Random contiguous slice of a tensor along a dim | `x` (Tensor), `dim` (int), `new_size` (int) | Tensor | randomness (`torch.randint`); assert old≥new | ✅ Active |
| `SDRDEVICE` | 380 | class(Enum) | SDR device-type enum (UNKNOWN/SIMULATION/PLUTO/BLADERF2) | — | enum | none | ✅ Active |
| `uri_to_device_type` | 387 | function | Classify SDR URI string into `SDRDEVICE` | `uri` (str) | `SDRDEVICE` | none | ✅ Active |
| `ZarrWrapper` | 397 | class | Read-only zarr array wrapper allowing on-the-fly added keys | — | instance | none | ✅ Active (used in v4→v5) |
| `ZarrWrapper.__init__` | 398 | method | Init wrapper with underlying zarr + empty overlay dict | `self`, `zarr_array` | None | sets self attrs | ✅ Active |
| `ZarrWrapper.__contains__` | 402 | method | Key present in zarr or overlay | `self`, `item` | bool | none | ✅ Active |
| `ZarrWrapper.__getitem__` | 407 | method | Fetch from zarr then overlay | `self`, `key` | value | prints "NO KEY"; raises `KeyError` if missing | ✅ Active |
| `ZarrWrapper.__setitem__` | 415 | method | Write into overlay only (zarr is read-only) | `self`, `key`, `value` | None | raises `ValueError` if key already in zarr; mutates overlay | ✅ Active |
| `training_only_keys` | 421 | module constant | List of GT/label keys for training | — | list[str] | none | ✅ Active |
| `segmentation_based_keys` | 432 | module constant | Keys derived from segmentation (added to skip when skip_segmentation) | — | list[str] | none | ✅ Active |
| `v5_raw_keys` | 441 | module constant | f64 + 2xf64 + signal_matrix raw key list | — | list[str] | none | ✅ Active |
| `data_single_radio_to_raw` | 444 | function | Subset a per-radio dict to raw keys minus skip_fields | `d` (dict), `ds` | dict | none | ✅ Active ⚠️unverified callers |

### `v5inferencedataset` (class @448, `Dataset` subclass) — realtime streaming inference dataset

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__init__` | 449 | method | Build realtime inference dataset: parse yaml, steering vectors, empirical data; **start background reader thread** | many (`yaml_fn`, `nthetas`, `model_config_fn`, `paired`, `gpu`, `skip_fields`, `empirical_data_fn`, `empirical_individual_radio`, `empirical_symmetry`, `target_dtype`, `distance_normalization`, `target_ntheta`, `windows_per_snapshot`, `skip_detrend`, `skip_segmentation`=True, `vehicle_type`, `max_in_memory`, `realtime`=True, `v4`, `max_store_size`) | None | **mutates `skip_fields` in place** (@506 `+=`); creates mp Queue/Lock/Condition; reads yaml; opens empirical pickle; **spawns `threading.Thread`** (@606); asserts `paired`; assert receiver-config consistency | ⚠️ Active — **mutable default `skip_fields=[]` + in-place `+=` mutates caller's list**; thread not daemon |
| `_reader_loop` | 611 | method | Background thread: drain incoming queue into `self.store`, evict over `max_store_size` | `self` | None (loops) | **thread**; mutates `self.store`, `self.min_idx_stored`; prints "RET"/"RET2"; logs exceptions; reads mp Queue | ⚠️ Active — **bug: reuses loop var `idx` in eviction loop @631 shadowing queue idx**; prints debug |
| `__len__` | 647 | method | Return current serving index (not true length) | `self` | int (`serving_idx`) | none | ⚠️ Active — returns serving_idx, not item count |
| `__iter__` | 650 | method | Reset serving idx, return self | `self` | self | mutates `serving_idx` | ✅ Active |
| `__next__` | 654 | method | Block until next sample ready, advance idx | `self` | sample (list of 2 dicts) | mutates `serving_idx`; logs warning; **busy-waits, never raises StopIteration** | ⚠️ Active — unbounded iterator |
| `__getitem__` | 664 | method | Block (≤timeout) until both receivers' data at `idx` are in store, return | `self`, `idx`, `timeout`=30.0 | list[2] data dicts, or `None` on timeout | `time.sleep` busy-wait; prints debug; raises `ValueError` if idx < min_idx_stored; acquires lock | ⚠️ Active — assumes sequential access (comment @663); noisy prints |
| `write_to_idx` | 680 | method | Render a raw capture and enqueue it for the reader thread | `self`, `idx`, `ridx`, `raw` (dict) | None | heavy compute (render_session); puts to mp Queue | ✅ Active |
| `render_session` | 686 | method | Convert one raw radio capture into a full feature dict (segmentation, beamform, empirical, binning) | `self`, `ridx` (int), `data` (dict) | `data` dict (mutated + many added keys) | **mutates `data` in place**; GPU (segment_session gpu flag); calls `segment_session`; dtype casts; sets GT fields to ±inf (unknown) | ✅ Active — comment "does not respect flipping" @716; `rx_pos_mm`/`tx_pos_mm` zeroed @739 |
| `close` | 804 | method | Signal stop, join reader thread | `self` | None | sets stop_event; joins thread (≤1s); prints "CLOSE2/3" | ✅ Active |
| `__enter__` | 810 | method | Context enter | `self` | self | none | ✅ Active |
| `__exit__` | 813 | method | Context exit → close() | `self`, exc info | None | prints "CLOSE"; calls close() | ✅ Active |

### `v5spfdataset` (class @818, `Dataset` subclass) — primary on-disk dataset

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__init__` | 819 | method | Open zarr+yaml, configure sessions/steering, run refresh + segmentation + QC | ~35 params incl `prefix`, `nthetas`, `precompute_cache`, `phi_drift_max`=0.2, `min_mean_windows`=10, `ignore_qc`, `paired`, `gpu`, `snapshots_per_session`=1, `tiled_sessions`=True, `snapshots_stride`, `readahead`, `temp_file`, `temp_file_suffix`=".tmp", `skip_fields`=[], `n_parallel`=20, `empirical_data_fn`, `empirical_individual_radio`, `empirical_symmetry`, `target_dtype`, `snapshots_adjacent_stride`=1, `flip`, `double_flip`, `random_adjacent_stride`, `distance_normalization`=1000, `target_ntheta`, `segmentation_version`=`SEGMENTATION_VERSION`, `segment_if_not_exist`, `windows_per_snapshot`=256, `skip_detrend`, `vehicle_type`, `v4`, `realtime` | None | **zarr/LMDB read** (open, map_size 2^32); yaml read; opens empirical pickle; calls `mp_segment_zarr` (writes precompute cache, multiprocessing, GPU) via `get_segmentation`; **raises `ValueError`** on QC failure (phi_drift / windows / valid-seg); many `assert`s; logging | ✅ Active — central class; mutable default `skip_fields=[]` (passed through, not mutated here) |
| `refresh` | 1166 | method | Re-open zarr, load/cache per-receiver keys into tensors, recompute GT thetas/phis | `self` | bool (True if reloaded, False if no change) | **zarr read** (re-open store); mutates many `self` attrs (`cached_keys`, `valid_entries`, GT arrays); asserts rx_spacing mismatch <30 | ✅ Active |
| `get_emitter_type` | 1260 | method | Read emitter type from yaml or default "external" | `self` | str | none | ✅ Active |
| `close` | 1265 | method | Close zarr stores, drop refs to data/segmentation | `self` | None | **closes zarr LMDB stores**; nulls self attrs | ✅ Active |
| `estimate_phi` | 1276 | method | Seg-mask-weighted mean of window stats → phi estimate | `self`, `data` (dict w/ `all_windows_stats`,`downsampled_segmentation_mask`) | Tensor | none | ✅ Active ⚠️unverified callers |
| `__len__` | 1284 | method | n_sessions (paired) or n_sessions×n_receivers | `self` | int | none | ✅ Active |
| `v4_to_v5` | 1289 | method | Wrap receiver zarrs in `ZarrWrapper`, synthesize v5 keys from v4 (heading→rx_heading_in_pis, zero-fill missing) | `self` | None | mutates `self.receiver_data` (wraps); no-op if not v4 | ✅ Active |
| `reinit` | 1305 | method | Re-open zarr + segmentation in worker process (after fork) | `self` | None | zarr read (re-open); calls `get_segmentation` | ✅ Active (DataLoader-worker support) |
| `get_spacing_identifier` | 1324 | method | String id from spacing + rx_lo | `self` | str | none | ✅ Active ⚠️unverified callers |
| `get_collector_identifier` | 1328 | method | Vehicle/collector type from yaml/zarr filename | `self` | str ("rover"/"wallarray") | none | ✅ Active |
| `get_wavelength_identifier` | 1335 | method | String id incl wavelength spacing | `self` | str | none | ✅ Active ⚠️unverified callers |
| `get_values_at_key` | 1339 | method | Fetch values for key at idxs (zarr for signal_matrix, else cached) | `self`, `key`, `receiver_idx`, `idxs` | Tensor | zarr read for `signal_matrix` | ✅ Active |
| `get_session_idxs` | 1344 | method | Delegate to module `get_session_idxs` with self params | `self`, `session_idx` | ndarray[int] | randomness if random_adjacent_stride | ✅ Active |
| `render_session` | 1354 | method | Build full per-(receiver,session) feature dict: GT thetas/phis, flips, precomputed seg, empirical, binning | `self`, `receiver_idx`, `session_idx`, `double_flip`=False | `data` dict (~25 keys) | calls `reinit`; randomness (flip via `torch.rand`); may call `get_segmentation` for temp_file; dtype casts; asserts signal_matrix shape | ✅ Active — core read path; comment "does not respect flipping" @1411 |
| `get_ground_truth_phis` | 1508 | method | Compute GT phase diff from GT thetas via array geometry | `self` | Tensor (n_receivers, N), rad, π-normed | none | ✅ Active — long comment on sign ambiguity @1518-1521 |
| `get_all_phi_drifts` | 1529 | method | Per-receiver π-normed (mean_phase − GT phi) | `self` | list[Tensor] | none | ✅ Active |
| `get_craft_ground_truth_thetas` | 1545 | method | GT theta relative to craft heading (rx_theta_in_pis offset) | `self` | Tensor (N,), rad | logs warning if receiver disagreement >0.2 | ✅ Active |
| `get_ground_truth_thetas` | 1573 | method | Compute GT theta (array-relative) and absolute theta from tx/rx positions | `self` | (Tensor, Tensor) both (n_receivers,N), rad | asserts `n_receivers==2` | ✅ Active |
| `get_mean_phase` | 1598 | method | Load per-receiver mean_phase from precomputed zarr into `self.mean_phase` | `self` | None | zarr read; mutates `self.mean_phase` | ✅ Active |
| `__getitem__` | 1604 | method | Return paired list[2] (paired) or list[1] (unpaired) of rendered sessions; None if temp_file past valid | `self`, `idx` | list of data dicts, or None | randomness (double_flip); calls render_session; asserts idx bounds | ✅ Active |
| `get_estimated_thetas` | 1623 | method | Per-receiver theta estimate from mean_phase via `phase_diff_to_theta` | `self` | dict[str→ndarray] | none | ✅ Active ⚠️unverified callers |
| `results_fn` | 1636 | method | Build precompute `.pkl` cache path | `self` | str (path) | none | ✅ Active |
| `get_segmentation_version` | 1643 | method | Extract segmentation version from zarr/seg dict | `self`, `precomputed_zarr`, `segmentation` | float or None | none | ✅ Active |
| `get_segmentation` | 1652 | method | Load (or generate via `mp_segment_zarr`) segmentation + precomputed yarr; verify version; load mean_phase | `self`, `version` (float), `segment_if_not_exist` (bool), `precompute_to_idx`=-1 | `self.segmentation` (dict) | **heavy: mp_segment_zarr (writes cache, multiprocessing, GPU); pickle read; zarr open**; raises `ValueError` (missing/version-mismatch); on `UnpicklingError` **removes file** (@1755) and recurses; mutates self.segmentation/precomputed_zarr | ✅ Active — has docstring |

### `SessionsDatasetSimulated` (class @1780, `Dataset` subclass) — legacy simulated dataset

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__init__` | 1781 | method | Load simulated session args.pkl, list session files | `self`, `root_dir` (str), `snapshots_per_session`=5 | None | file read (lzma pickle `load`); `os.listdir`; prints "NOT LIVE"/warning; asserts time_steps | ⚠️ Legacy (simulated, not v5) |
| `idx_to_filename_and_start_idx` | 1804 | method | Map flat idx → (filename, start) | `self`, `idx` | (str, int) | asserts idx bounds | ⚠️ Legacy |
| `__len__` | 1811 | method | Total samples (live vs file count) | `self` | int | none | ⚠️ Legacy |
| `__getitem__` | 1817 | method | Generate (live) or load a session slice | `self`, `idx` | dict slices | randomness/compute (`generate_session`) or lzma load | ⚠️ Legacy |

**Notable (spf_dataset.py)**

- **Mutable-default + in-place mutation bug**: `v5inferencedataset.__init__` has `skip_fields: List[str]=[]`
  and does `self.skip_fields += segmentation_based_keys` (@506), an in-place `+=` that mutates the
  shared default list / caller's list. `v5spfdataset.__init__` also has `skip_fields=[]` default but
  stores it without in-place mutation. Both are surprising-for-name and warrant Pass-3.
- **`_reader_loop` variable shadowing (@611-633)**: the outer `idx` from `self.incoming_queue.get()` is
  reused as the loop variable in the eviction `for idx in sorted(...)[:-max_store_size]` (@631), and
  `self.min_idx_stored=min(self.store.keys())` (@633) runs inside the `with self.lock`. Flag for Pass-3.
- **Debug prints left in hot path**: `v5inferencedataset.__getitem__`/`render_session`/`close`/`_reader_loop`
  print "waitinf to get get", "RET", "RET2", "CLOSE", etc. (@616,645,669,675,805,814). Noisy, not errors.
- **`v5inferencedataset` busy-waits** (`time.sleep(0.02)` @673; `__next__` never raises `StopIteration`).
  Background reader thread is **not a daemon** (the `daemon=True` is commented out @608).
- **`get_segmentation` deletes the cache file** on `pickle.UnpicklingError` (@1755) then recurses — a
  destructive recovery path.
- **`@cache` on `get_empirical_dist`** (@232) keys on `(v5ds, receiver_idx)`, retaining dataset
  references for process lifetime (potential leak / stale-after-refresh). Flag for Pass-3.
- **`data_from_precomputed`** has two TODO-flagged "hacky and not right" random subsamples when
  windows exceed `windows_per_snapshot` (@284-288, @300-304), introducing randomness into reads.
- **QC raises** in `v5spfdataset.__init__` on high phi-drift, low average windows, or <20% valid
  segmentation (@1142-1153) unless `ignore_qc`.
- `v5_collate_all_fast` carries a commented-out `@torch.jit.script` decorator (@130).
- `SessionsDatasetSimulated` (@1780) is legacy/simulated, parallel to the v4/v5 path.

---

## segmentation.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `segment_single_session` | 50 | function | Thin wrapper to segment one session/receiver with defaults | `zarr_fn` (str), `steering_vectors_for_receiver` (ndarray), `session_idx` (int), `receiver_idx` (int), `gpu` (bool) | dict (from `segment_session`) | delegates (I/O in callee) | ⚠️unverified — no in-file callers |
| `mean_phase_from_simple_segmentation` | 77 | function | Signal-strength-weighted circular mean phase per session | `segmentation` (list of dicts w/ `simple_segmentation`) | ndarray (1D, NaN if no signal) | none (uses `torch.nan` placeholder) | ✅ Active |
| `mp_segment_zarr` | 107 | function | Core pipeline: segment all sessions, beamform, write yarr + pkl | `zarr_fn`, `results_fn` (str), `steering_vectors_for_all_receivers` (list), `precompute_to_idx`=-1, `gpu`=False, `n_parallel`=20, `skip_beamformer`=False, `temp_file`=False, `skip_detrend`=False | int (num sessions processed) | **heavy: zarr/LMDB read+write, pickle read+write, makedirs, multiprocessing Pool, tqdm, `assert len==2`, GPU via children, zarr_shrink** | ✅ Active — primary entry point |
| `segment_session_from_zarr_star` | 356 | function | Unpack arg dict → kwargs for `segment_session_from_zarr` (Pool helper) | `arg_dict` (dict) | dict | delegates | ✅ Active |
| `segment_session_from_zarr` | 361 | function | Load one session's signal_matrix from zarr, segment | `zarr_fn` (str), `receiver` (str e.g. "r0"), `session_idx` (int), `**kwrgs` | dict | zarr/LMDB read; complex64 cast | ✅ Active |
| `segment_session` | 375 | function | Per-session: detrend, beamform, segment, extract stats | `v` (complex ndarray (2,N)), `gpu`=True, `skip_beamformer`=False, `skip_detrend`=False, `skip_segmentation`=False, `fast_beamformer`=False, `**kwrgs` | dict keys: `windowed_beamformer`,`all_windows_stats`,`weighted_beamformer`,`weighted_windows_stats`,`simple_segmentation`,`downsampled_segmentation_mask` (singleton lead dim) | **mutates `v` (detrended); GPU via cupy**; asserts in callees; float16 casts | ✅ Active — central |
| `get_all_windows_stats` | 640 (decor `@njit`) | function | Numba-JIT per-window phase mean/stddev/median amplitude | `v` (complex (2,N)), `window_size`, `stride`, `trim` | tuple `(step_idxs, step_stats)` — stats (N_windows,3) | `assert v.ndim==2 & shape[0]==2`; JIT compile | ✅ Active |
| `simple_segment` | 668 | function | Full pipeline: window stats → combine → filter → recompute → mask | `v` (2,N), `window_size`, `stride`, `trim`, `max_stddev_threshold`, `drop_less_than_size`, `min_abs_signal` | dict: `simple_segmentation` (list), `downsampled_segmentation_mask` (bool ndarray), `all_windows_stats` (N_windows,3) | assert on shape; mutates window dicts via callees | ✅ Active |
| `drop_noise_windows` | 784 | function | Return only "signal"-type windows | `windows` (list of dicts) | list of dicts | none | 🧪 Unused — no in-file caller |
| `keep_signal_surrounded_by_noise` | 801 | function | Keep isolated signal windows (noise neighbors) | `windows` (list of dicts) | filtered list | none | ✅ Active (used in `simple_segment`) |
| `drop_windows_smaller_than` | 832 | function | Drop windows shorter than threshold | `windows` (list), `drop_less_than_size` (int) | filtered list | none | ✅ Active |
| `combine_windows` | 850 | function | Classify signal/noise, merge adjacent similar | `windows` (list), `max_stddev_threshold` (float), `min_abs_signal` (float) | list of dicts (+`type`) | **mutates input window dicts** | ✅ Active |
| `recompute_stats_for_windows` | 906 | function | Recompute mean/stddev/amplitude per window from raw signal | `windows` (list), `v` (2,N), `pd` (phase diff ndarray), `trim` (float) | windows (mutated) | **mutates input window dicts** | ✅ Active |
| `compute_downsampled_segmentation_mask` | 937 | function | Build bool per-window signal mask from segment indices | `simple_segmentation` (list), `n_windows` (int), `window_size` (int) | ndarray bool (n_windows,) | none | ✅ Active |
| `beamform_signal_cpu` | 1025 | function | (Intended) CPU beamforming — body is a stub | `v` (n_ant,n_samp), `steering_vectors`, `n_windows`=None | None (no return) | none | 🔴 Broken — body is `# ... existing code ...` placeholder |

**Notable (segmentation.py)**

- Only decorator: `@njit` on `get_all_windows_stats` (@640).
- 🔴 `beamform_signal_cpu` (@1025) is a stub (placeholder comment, returns None); last symbol; no callers.
- 🧪 `drop_noise_windows` (@784) defined but never called in-file.
- `cupy` import guarded by bare `try/except: pass` (@9-12) → `NameError` on `cp` if cupy absent and `gpu=True`.
- `mp_segment_zarr` hard-asserts exactly 2 receivers (@189); opens pickles without context managers (@152,344-350).
- `DEFAULT_SEGMENT_ARGS` module constant (@39); `mean_diff_threshold` commented out (@43).
- Two large near-duplicate ASCII beamforming explanation blocks (@423-493 and @966-1023).
- Commented-out experimental code + stray commented `breakpoint()` (@664).

---

## v4_data.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `v4rx_f64_keys` | 3 | module constant | Scalar f64 field names (9) | — | list[str] | none | ✅ Active |
| `v4rx_2xf64_keys` | 14 | module constant | 2-vector f64 field names (3) | — | list[str] | none | ✅ Active |
| `v4rx_keys` | 21 | function | Full v4 receiver key list | none | list[str] | none | ✅ Active |
| `v4rx_new_dataset` | 25 | function | Create a new v4 zarr dataset (wrapper) | `filename`, `timesteps`, `buffer_size`, `n_receivers`, `config`, `chunk_size`=1024, `compressor`=None | zarr group | zarr/file I/O (delegate) | ✅ Active |

**Notable:** No main/classes/decorators. Imports only `zarr_new_dataset`.

---

## v5_data.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `v5rx_f64_keys` | 10 | module constant | Scalar f64 field names (10; v5 adds tx/rx pos+heading) | — | list[str] | none | ✅ Active |
| `v5rx_2xf64_keys` | 24 | module constant | 2-vector f64 field names (3) | — | list[str] | none | ✅ Active |
| `v5rx_keys` | 31 | function | Full v5 receiver key list | none | list[str] | none | ✅ Active |
| `v5rx_new_dataset` | 35 | function | Create new v5 zarr dataset (adds skip_signal_matrix, remove_if_exists) | `filename`, `timesteps`, `buffer_size`, `n_receivers`, `config`, `chunk_size`=1024, `compressor`=None, `skip_signal_matrix`=False, `remove_if_exists`=True | zarr group | zarr/file I/O | ✅ Active |

**Notable:** Imports `compare_and_copy`, `zarr_open_from_lmdb_store`, `os`, `shutil` — all unused (⚠️ dead imports).

---

## fake_dataset.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `phi_to_signal_matrix` | 27 (decor `@torch.jit.script` @26) | function | Synthetic 2×buffer complex signal from phase diff w/ noise+drift | `phi` (Tensor), `buffer_size` (int), `noise` (float), `phi_drift` (float), `generator` (torch.Generator) | Tensor complex (2,buffer_size), ×200 | randomness (generator); GPU-capable | ✅ Active |
| `fake_yaml` | 61 | function | Produce fake receiver YAML config string for a version | `version` (int) | str (YAML) | none | ✅ Active |
| `fake_yaml_v5` | 120 | module constant | `fake_yaml(5)` | — | str | runs at import | ✅ Active |
| `fake_yaml_v4` | 121 | module constant | `fake_yaml(4)` | — | str | runs at import | ✅ Active |
| `create_empirical_dist_for_datasets` | 124 | function | Segment datasets, build empirical p-dist pickle | `datasets` (list[str]), `precompute_cache` (str), `nthetas` (int) | str (path to full.pkl) | file write (pkl); zarr I/O; argparse | ✅ Active |
| `segment_dataset` | 155 | function | Segment a single dataset via `v5spfdataset` | `ds_fn` (str), `nthetas` (int), `precompute_cache` (str) | None | zarr I/O; writes seg cache | ✅ Active |
| `create_fake_dataset` | 165 | function | Generate full synthetic v4/v5 zarr + yaml from config | `yaml_config_str` (str), `filename` (str), `orbits`=1, `n`=50, `noise`=0.01, `phi_drift`=0.0, `radius`=10000, `seed`=0 | None | seeds RNG (random/np/torch); deletes existing files; writes yaml+zarr; tqdm; randomness | ✅ Active |
| `theta_to_phi` | 220 | nested function | Incidence angle → phase diff | `theta`, `antenna_spacing_m`, `_lambda` | Tensor | none | ✅ Active |
| `phi_to_theta` | 223 | nested function | Phase diff → (theta, π−theta), optional clip | `phi`, `antenna_spacing_m`, `_lambda`, `limit`=False | (Tensor,Tensor) | none | 🧪 Unused (call commented @243) |
| `compare_and_copy_n` | 305 | function | Recursively copy first n records src→dst zarr | `prefix`, `src`, `dst`, `n` | None | **mutates dst (zarr writes)**; recursion | ✅ Active |
| `compare_and_copy_nth` | 318 | function | Recursively copy single n-th record src→dst | `prefix`, `src`, `dst`, `n` | None | **mutates dst (zarr writes)**; recursion | ✅ Active |
| `partial_dataset` | 330 | function | New v5 zarr with first n records of existing | `input_fn`, `output_fn`, `n` | None | opens lmdb zarr; copies yaml; writes new zarr; shrinks | ✅ Active |
| `PartialDatasetController` | 354 | class | Stateful per-record partial-dataset copier | — | instance | — | ✅ Active |
| `PartialDatasetController.__init__` | 355 | method | Open input zarr, create output zarr/yaml | `self`, `input_fn`, `output_fn` | None | opens lmdb zarr; copies yaml; creates output zarr | ✅ Active |
| `PartialDatasetController.copy_nth` | 374 | method | Copy n-th record input→output | `self`, `n` | None | **mutates output zarr** | ✅ Active |
| `get_parser` | 378 | function | Build argparse parser | none | ArgumentParser | none | ✅ Active |
| `__main__` block | 419 | module-main | Parse args, call `create_fake_dataset(fake_yaml_v5,...)` | argv | — | argparse; full dataset generation | ✅ Active |

**Notable:**
- `input_fn.replace(".zarr","")` at @331 and @356 is a **no-op** (return discarded) — ⚠️ likely bug.
- `phi_to_theta` (@223) defined but unused (🧪).
- Cleanup loop @183 targets `f"{filename}.zar"` (`.zar` not `.zarr`) — ⚠️ likely misses real path.
- `--orbits`/`--n` have string defaults but `type=int` (argparse applies type only to CLI values) — ⚠️ minor.

---

## spf_generate.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `c` | 8 | module constant | Speed of light 3e8 | — | float | none | ✅ Active |
| `_arctan2` | 11 | function | `np.arctan2(x,y)` wrapper (swapped semantics) | `x`, `y` | ndarray/float (rad) | none | ✅ Active |
| `BoundedPoint` | 15 | class | Point moving + bouncing off [0,width] walls | — | instance | — | ✅ Active |
| `BoundedPoint.__init__` | 16 | method | Init pos/vel/timestep/width | `self`, `pos`=ones(2)*0.5, `v`=zeros(2), `delta_time`=0.05, `width`=128 | None | **mutable default ndarray args** (⚠️ shared-mutable risk) | ✅ Active |
| `BoundedPoint.time_step` | 22 | method | Advance one step, reflect at walls | `self`, `eps`=1e-6 | (pos(2), heading, v(2)) | mutates `self.pos`, `self.v` | ✅ Active |
| `time_to_detector_offset` | 36 | function | Orbital (x,y) offset vs time | `t`, `orbital_width`, `orbital_height`, `orbital_frequency`=1/100, `phase_offset`=0 | ndarray (2,) | none | ⚠️ used only in disabled orbit branch |
| `generate_session` | 48 | function | Simulate one full RF session | `args_and_session_idx` ((args, idx)) | dict session (~16 keys) | seeds np RNG; **prints + `sys.exit(1)` on bad array_type/orbit**; randomness; compute | ✅ Active |
| `generate_session_and_dump` | 307 | function | Run session + dump to lzma pickle | `args_and_session_idx` | None | **file write** (`session_%08d.pkl`); randomness | ✅ Active |

**Notable:**
- 🔴 `generate_session` orbit branch (@221) prints error + `sys.exit(1)` — orbit path intentionally broken; only `"bounce"` works.
- `BoundedPoint.__init__` uses mutable numpy default args (⚠️).
- No `__main__`; library driven by external multiprocessing caller.

---

## spf_nn_dataset_wrapper.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `v5spfdataset_nn_wrapper` | 19 | class(Dataset) | Wrap v5 dataset, attach NN-inference outputs per entry | — | instance | — | ✅ Active |
| `__init__` | 20 | method | Load cached NN inference or realtime model+config | `self`, `ds`, `checkpoint_config_fn`, `checkpoint_fn`, `inference_cache`, `device`="cpu", `v4`=None, `absolute`=False | None | asserts `ds.paired`; NN inference (file/cache I/O, GPU); `torch.set_num_threads(4)`; model eval | ✅ Active |
| `to_absolute_north` | 84 | method | Rotate paired NN dist into absolute-north frame | `self`, `sample` | sample (mutated) | **`breakpoint()` @86**; mutates sample | 🔴 Broken (unconditional breakpoint) |
| `get_inference_for_idx` | 99 (decor `@lru_cache(4)` @98) | method | Per-receiver inference dicts for idx | `self`, `idx` (int) | list[dict] len 2 | lru_cache; realtime model run | ✅ Active |
| `get_and_annotate_entry_at_idx` | 108 (decor `@lru_cache(4)` @107) | method | Merge NN inference into `ds[idx]` samples | `self`, `idx` (int) | annotated sample | lru_cache; GPU/model; **`breakpoint()` @126** (realtime); mutation | ⚠️ realtime path has breakpoint (🔴 realtime) |
| `__iter__` | 131 | method | Reset iterator index | `self` | self | mutates `serving_idx` | ✅ Active |
| `__next__` | 135 | method | Next annotated sample | `self` | annotated sample | mutates `serving_idx`; inference; **never raises StopIteration** | ⚠️ Active (unbounded) |
| `__getitem__` | 141 (decor `@lru_cache(4)` @140) | method | Indexed annotated entry | `self`, `idx` | annotated sample | lru_cache; inference | ✅ Active |
| `__len__` | 144 | method | Delegate to ds | `self` | int | none | ✅ Active |
| `mean_phase` | 148 (`@property` @147) | property | Delegate `ds.mean_phase` | `self` | passthrough | none | ✅ Active |
| `ground_truth_phis` | 152 (`@property` @151) | property | Delegate | `self` | passthrough | none | ✅ Active |
| `craft_ground_truth_thetas` | 156 (`@property` @155) | property | Delegate | `self` | passthrough | none | ✅ Active |
| `absolute_thetas` | 160 (`@property` @159) | property | Delegate | `self` | passthrough | none | ✅ Active |

**Notable:**
- Two hardcoded `breakpoint()` calls (@86, @126) on realtime/absolute paths → 🔴 halt into pdb if reached.
- `@lru_cache(4)` on instance methods (@98,107,140) keys on `(self, idx)` — can leak instances / stale results.
- No module-level main.

---

## benchmark_dataset.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module script) | 1–91 | module-script | Benchmark DataLoader throughput over a v5 dataset (GPU) | CLI `sys.argv[1]`=ds_fn | exit code | reads argv; `sys.exit(1)` if argc≠2; builds `v5spfdataset` (GPU, zarr I/O); shuffles; DataLoader; tqdm; prints | ✅ Active (script) |

**Notable:** No defs. `if False:` dead block (@48-65). `random.shuffle(idxs)` on a torch tensor (@42) — unusual. Unused imports: `os`, `pickle`, `numpy`, `scipy`, `systematic_resample`, `v5_collate_beamsegnet`.

---

## benchmark_dataset_linear.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module script) | 1–97 | module-script | Linearly index first 3000 entries then `sys.exit(0)` | none (hardcoded ds_fn) | exit 0 | builds `v5spfdataset` (CPU); loops `ds[x]` 3000×; **`sys.exit(0)` @47** before DataLoader code; prints | ⚠️ Active but dead tail |

**Notable:** Hardcoded paths (@20; `/mnt/4tb_ssd/...`, `/home/mouse9911/precompute_cache...`). `sys.exit(0)` @47 makes the whole DataLoader benchmark unreachable (🔴 dead code). Near-duplicate of `benchmark_dataset.py`.

---

## compare_datasets.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module main) | 8–50 | module-main | Load two v5 datasets, report IDENTICAL/DIFFERENT | CLI `zarr_filenameA`, `zarr_filenameB`, `--precompute-cache` (req), `--nthetas`=65, `-s/--skip-signal-matrix`, `-r/--rendered` | exit 0/1 | argparse; builds two `v5spfdataset` (zarr I/O); prints; `sys.exit` | ✅ Active |

**Notable:** ✅ Active. _(Adversarial-pass correction: an earlier draft flagged `return` at module scope (@37,@43) as a SyntaxError — REFUTED. Those lines are `return_code = identical_datasets(...)` and `return_code = 0`, ordinary assignments. The module imports and runs.)_ No defs; uses `compare_and_check`, `identical_datasets`.

---

## open_partial_ds.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `open_partial_dataset_and_check_some` | 7 | function | Open v5 dataset partially, print length/valid, spot-check 10 random sessions' mean phase | `ds_fn`, `suffix`=".tmp", `n_parallel`=1, `skip_fields`=set() | None | creates temp dir; builds `v5spfdataset` (zarr/file I/O); prints; `random.shuffle`; asserts; **mutates `skip_fields` set** | ⚠️ mutable default arg |
| `__main__` block | 45 | module-main | CLI: parse args, build skip_fields, call checker | argv | none | argparse; calls function | ✅ Active CLI |

**Notable:**
- ⚠️ @8 mutable default `skip_fields=set()`; @14 `.add("signal_matrix")` mutates it (persists across calls).
- 🔴 @69 `__main__` builds `skip_fields` as a **list**, but @14 calls `.add()` → lists lack `.add` → `AttributeError` when invoked via CLI.
- @32 `min(ds.valid_entries)` — semantics depend on `v5spfdataset` internals (⚠️unverified).

---

## wall_array_v1_idxs.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `v1_column_names` | 6 | function | Ordered v1 column names: 5 fixed + nthetas beamformer cols | `nthetas`=65 | list[str] | none (np.linspace) | ✅ Active (pure) |
| `v1_time_idx` | 18 (`@cache`) | function | Index of "timestamp" | none | int | memoized | ✅ Active |
| `v1_tx_pos_idxs` | 23 (`@cache`) | function | Indices of tx_pos_x_mm, tx_pos_y_mm | none | list[int] (2) | memoized | ✅ Active |
| `v1_beamformer_start_idx` | 31 (`@cache`) | function | First column index containing "beamformer" | none | int or None | memoized | ✅ Active |

**Notable:** Not literal arrays — small `@cache` index helpers using default nthetas=65 layout.

---

## wall_array_v2_idxs.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `v2_column_names` | 6 | function | Ordered v2 column names: 13 fixed + nthetas beamformer cols | `nthetas`=65 | list[str] | none | ✅ Active |
| `v2_time_idx` | 26 (`@cache`) | function | Index of "timestamp" | none | int | memoized | ✅ Active |
| `v2_rx_pos_idxs` | 31 (`@cache`) | function | Indices of rx_pos_x_mm, rx_pos_y_mm | none | list[int] (2) | memoized | ✅ Active |
| `v2_rssi_idxs` | 39 (`@cache`) | function | Intended rssi0,rssi1 indices | none | list[int] (2) | memoized | 🔴 bug — both entries index "rssi0" (@41-42) |
| `v2_gain_idxs` | 47 (`@cache`) | function | Indices of gain0, gain1 | none | list[int] (2) | memoized | ✅ Active |
| `v2_avg_phase_diff_idxs` | 55 (`@cache`) | function | Indices of avg_phase_diff_1,_2 | none | list[int] (2) | memoized | ✅ Active |
| `v2_rx_theta_idx` | 63 (`@cache`) | function | Index of rx_theta (1-elem list) | none | list[int] (1) | memoized | ✅ Active |
| `v2_tx_pos_idxs` | 70 (`@cache`) | function | Indices of tx_pos_x_mm,_y_mm | none | list[int] (2) | memoized | ✅ Active |
| `v2_beamformer_start_idx` | 78 (`@cache`) | function | First "beamformer" column index | none | int or None | memoized | ✅ Active |

**Notable:** 🔴 `v2_rssi_idxs` (@39-43) returns `[rssi0_idx, rssi0_idx]` — second should be "rssi1". Genuine bug.

---

## precompute_3p3_to_3p31.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `process_yarr` | 13 | function | Migrate one precompute cache 3.3→3.31: recompute mean_phase from pkl, write into zarr, set version | `yarr_fn` (path) | None | **in-place zarr write** (mode rw, map_size 2^32); pkl read; torch/numpy | ⚠️ in-place migration; @32 TODO non-finite→0 hack |
| `nanstd` | 39 | function | Std-dev ignoring non-finite | `x` | scalar | none | 🧪 dead (only commented use @33) |
| `__main__` block | 43 | module-main | CLI: `--yarrs`, `--debug`, `--workers`; run serially or via ProcessPoolExecutor | argv | none | argparse; multiprocessing (16 workers); tqdm; process_yarr I/O | ✅ Active CLI |

**Notable:** @32 non-finite mean_phase→0 with explicit "TODO THIS SHOULD BE FIXED!!!". `nanstd` dead. Each worker opens zarr with 4 GiB map_size. Hardcoded r0/r1.

---

## zarr_utils.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `zarr_open_from_lmdb_store_cm` | 14 | function (contextmanager) | Context-managed LMDB zarr open w/ optional readahead | `filename`, `mode`="r", `readahead`=False | yields zarr group | zarr/LMDB I/O; `posix_fadvise`; closes on exit | ✅ Active |
| `new_yarr_dataset` | 30 | function | Create fresh yarr (segmentation precompute) zarr | `filename`, `n_receivers`, 5 shape args, `chunk_size`=1 | zarr group | **DESTRUCTIVE** (`zarr_remove_if_exists` deletes existing); writes version | ✅ Active |
| `zarr_open_from_lmdb_store` | 98 | function | Open LMDB zarr in r/rw/w with tuned params | `filename`, `mode`="r", `readahead`=False, `map_size`=2^37, `lock`=None | zarr group | zarr/LMDB I/O; raises NotImplementedError on bad mode | ✅ Active |
| `zarr_new_dataset` | 140 | function | Create fresh v5 RX capture zarr | `filename`, `timesteps`, `buffer_size`, `n_receivers`, `keys_f64`, `keys_2xf64`, `config`, `chunk_size`=512, `compressor`=None, `skip_signal_matrix`=False, `remove_if_exists`=True | zarr group | **DESTRUCTIVE**; writes config; raises NotImplementedError if compressor passed; asserts config round-trip | ✅ Active |
| `zarr_remove_if_exists` | 204 | function | Delete data.mdb/lock.mdb for a path | `zarr_fn` | none | **DESTRUCTIVE** (`os.remove`) | ✅ Active |
| `zarr_shrink` | 210 | function | Shrink LMDB map size to reclaim disk | `filename` | none | **in-place LMDB mutation** (`set_mapsize(1)`); prints; closes | ✅ Active |
| `compare_and_check` | 217 | function | Recursively verify dst zarr matches src | `prefix`, `src`, `dst`, `skip_signal_matrix`=False | True or raises | reads zarr; raises ValueError on mismatch | ✅ Active |
| `compare_and_copy` | 236 | function | Recursively copy src→dst (renames rx_heading→rx_heading_in_pis) | `prefix`, `src`, `dst`, `skip_signal_matrix`=False, `copy_up_to_idx`=None | none | **writes into dst zarr** | ✅ Active |
| `truncate_zarr` | 262 | function | Rebuild partial v4 zarr dropping trailing zero-timestamp entries | `zarr_fn`, `f64_keys`, `f64x2_keys` | none | **HIGHLY DESTRUCTIVE** — `shutil.move` to `.bkup`, new zarr at original path; logs; hardcodes 2 receivers | ✅ Active |

**Notable:** Three creators delete data.mdb/lock.mdb before writing (irreversible). `truncate_zarr` renames source to `.bkup` (aborts via assert if .bkup exists). `compare_and_copy` silently remaps `rx_heading`→`rx_heading_in_pis`. `zarr_new_dataset` raises if any non-None compressor (@162).

---

## zarr_rechunk.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `zarr_rechunk` | 18 | function | Re-chunk a v5 zarr into new file (or verify existing carbon copy) | `zarr_filename_in`, `zarr_filename_out`, `skip_signal_matrix`, `check_contents_if_exists`=True | int (0 ok, 1 mismatch) | reads input zarr+`.yaml`; **creates output zarr** + copies yaml; `zarr_shrink`; prints; mkdirs | ✅ Active |
| `__main__` CLI | 85 | cli-main | CLI wrapper | args: in, out, `-s/--skip-signal-matrix`, `--check-contents/--no-check-contents` | `sys.exit(code)` | parses args; exits | ✅ Active |

**Notable:** Creates output via `zarr_new_dataset` (deletes pre-existing at out path); if output exists only verifies. n_receivers hardcoded 2.

---

## zarr_shrink.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` CLI | 5 | cli-main | Thin CLI calling `zarr_shrink` on a filename | argv: zarr_filename | exit 1 if wrong argc | **in-place LMDB shrink**; prints | ✅ Active (trivial wrapper) |

**Notable:** No defs; delegates to `zarr_utils.zarr_shrink` (in-place mutation).

---

## zarr_fix_rx_spacing.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `zarr_fix` | 18 | function | Overwrite r0/r1 `rx_spacing` in-place to match yaml antenna-spacing-m | `zarr_filename_in` | None (implicit) | **DESTRUCTIVE in-place** (mode rw, writes rx_spacing[:] r0,r1); reads yaml; prints on mismatch | ✅ Active |
| `__main__` CLI | 41 | cli-main | CLI wrapper | arg: zarr_filename_in | `sys.exit(None)` | parses args; exits | ⚠️ Active — `sys.exit(None)` since `zarr_fix` returns None |

**Notable:** Modifies input zarr IN PLACE, no backup; reads only first receiver's antenna-spacing, applies to both.

---

## truncate_zarr.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` CLI | 7 | cli-main | CLI calling `truncate_zarr` with v4 key sets | arg: zarr_filename_in | `sys.exit(None)` | **HIGHLY DESTRUCTIVE** via `truncate_zarr` (moves original to `.bkup`, rebuilds in place) | ✅ Active (v4-only per comment) |

**Notable:** Delegates to `zarr_utils.truncate_zarr` with `v4rx_*` keys. Only works for version 4.

---

## segment_zarr.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `process_zarr` | 7 | function | Load a v5 dataset to trigger segmentation precompute; print phi drifts | `args` dict (input_zarr, precompute_cache, gpu, parallel) | none | **heavy I/O + GPU**: builds `v5spfdataset` w/ `segment_if_not_exist=True` (writes cache); prints | ✅ Active |
| `__main__` CLI | 23 | cli-main | Parallel driver over many zarrs | args: `-i/--input-zarrs`+, `-c/--precompute-cache`, `--gpu/--no-gpu`, `--debug/--no-debug`, `-p/--parallel`=12, `-w/--workers`=2 | none | ProcessPoolExecutor (or serial debug); GPU; segmentation writes | ✅ Active |

**Notable:** Writes precompute cache. Imports `cupy` at module top (@6) → import fails without CUDA/cupy.

---

## yarr_3p5_to_3p6.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `yarr_3p5_to_3p6` | 21 | function | Migrate a 3p5 yarr to 3p6 by creating new yarr + copying | `yarr_filename_in`, `output_folder` | int (0 ok, 1 if exists) | reads input yarr; **creates new yarr**; `compare_and_copy`; `zarr_shrink`; mkdirs; prints | ✅ Active |
| `__main__` CLI | 58 | cli-main | Multiprocess driver (Pool 30) over inputs | args: `-i/--inputs`+, `-o/--output-folder` | none | `multiprocessing.Pool(30)` + tqdm; per-task file creation | ✅ Active |

**Notable:** Skips (returns 1) rather than overwriting existing output (non-destructive to outputs). Hardcodes 2 receivers, chunk_size 1, pool 30. Unused imports: `compare_and_check`, `zarr_new_dataset`, `sys`.

---

## yarr_rechunk.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` CLI | 6 | cli-main | Copy a yarr's all_windows_stats / windowed_beamformer into freshly-chunked yarr | argv: in, out | exit 1 on bad argc | asserts output absent; reads input yarr; **creates new yarr** + bulk-copies 2 datasets r0/r1; closes | 🔴 likely broken |

**Notable:** 🔴 Calls `new_yarr_dataset` WITHOUT `weighted_beamformer_shape`/`downsampled_segmentation_mask_shape`/`mean_phase_shape` (no defaults in current signature @zarr_utils:30) → would raise `TypeError`. Stale call site. `assert not os.path.exists(output_fn)` guards overwrite.

---

## v4_tx_rx_to_v5.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `lat_lon_to_xy` | 19 | function | lat/lon → local x/y (km) via azimuthal-equidistant proj | `lat`, `lon`, `center_lat`, `center_lon` | np.array([x,y]) | none (pyproj) | ✅ Active |
| `smooth_out_timestamps_and_gps` | 24 | function | Fill zero gaps in times/gps by neighbor avg; trim trailing invalid | `timestamps_and_gps` (TimestampAndGPS) | mutated TimestampAndGPS | **mutates input in place**; asserts | ✅ Active |
| `TimestampAndGPS` | 43 | class(dataclass) | Holds times/gps_lats/gps_longs | `times`, `gps_lats`, `gps_longs` | instance | none | ✅ Active |
| `get_tx_xy_at_rx` | 51 | function | Interpolate TX gps at each RX timestamp → tx/rx xy (mm) | `rx_time_and_gps`, `tx_time_and_gps`, `gps_center_long_lat` | list of dicts | logs error if coords >300000mm | ✅ Active |
| `convert_list_dict_to_dict_lists` | 103 | function | Transpose list-of-dicts → dict-of-lists | `list_dict` | dict of lists | none | ✅ Active |
| `trim_valid_idxs_and_tx_rx_pos` | 113 | function | Align/trim r0 & r1 valid index lists | `valid_idxs_and_tx_rx_pos` | dict (adds idxs, transposes) | **mutates input dict**; asserts alignment ≤1 & set equality | ✅ Active |
| `compare_and_copy_with_idxs_and_aux_data` | 162 | function | Recursive copy of selected RX idxs into new v5 zarr, inject aux data, heading→pis | `prefix`, `src`, `dst`, `idxs`, `skip_signal_matrix`=False, `aux_data`={} | none | **writes into dst zarr**; catches per-row RuntimeError, logs failures; heading→pis | ✅ Active |
| `get_min_max_of_times` | 205 | function | Min(>0)/max of times array | `times` | (min,max) | none | 🧪 unused |
| `get_non_zero_mean` | 209 | function | Mean of non-~0 elements | `x` | scalar | none | ✅ Active |
| `merge_v4rx_v4tx_into_v5` | 213 | function | Merge v4 TX+RX pair into single v5 zarr w/ interpolated GPS | `tx_fn`, `rx_fn`, `zarr_out_fn`, `fix_config`, `collector_type`="unknown", `collector_version`="0.0", `receivers`=2, `dry_run`=False, `min_timesteps`=500 | int (timesteps) | reads tx/rx zarr+yaml; **creates output zarr** + writes yaml; copies; `zarr_shrink`; raises ValueError on bad GPS/boundaries; asserts; logs | ✅ Active |
| `__main__` CLI | 364 | cli-main | Driver: per (tx,rx) build v5 output; skip if exists | args: `--txs`+, `--rxs`+, `--output`, `--collector-type`=rover, `--collector-version`=3.1, `--dry-run`, `--fix-config/--no-fix-config` | none | logging; per-pair file creation; catches lmdb.CorruptedError/KeyError/ValueError; skips existing output dirs | ✅ Active |

**Notable:** `merge_v4rx_v4tx_into_v5` writes new v5 zarr + yaml; mutates loaded config; hardcodes 2-receiver assumptions. When injecting missing `collector` block (@326) hardcodes `{"type":"rover","version":"3.1"}` instead of using params — ⚠️ likely latent bug. `get_min_max_of_times` unused (🧪).

---

## ls_ds.py

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `ls_zarr` | 15 | function | Summarize a v5 dataset into cached `.ls.json`; reload/regenerate on version mismatch | `ds_fn`, `force`=False | dict ls_info or None | reads zarr (no segmentation); **writes `<ds_fn>.ls.json`**; prints on failure; recurses w/ force on mismatch | ✅ Active |
| `__main__` CLI | 71 | cli-main | Parallel driver; aggregate sample counts per config key; print CSV | args: `-i/--input-zarrs`+, `--debug/--no-debug`, `-w/--workers`=4 | prints CSV | ProcessPoolExecutor (or serial debug) + tqdm; reads/writes `.ls.json`; prints | ✅ Active |

**Notable:** Writes `<ds_fn>.ls.json` sidecar. `LS_VERSION=1.5` (@12) gates cache. Loads with `segment_if_not_exist=False`, `temp_file=True` (no zarr write to dataset). Unused imports: `multiprocessing`, `time`.

---

## Cluster summary

**Symbol counts by status** (approximate; module-level CLI/`__main__` blocks counted as one symbol each):

| Status | Count (approx) |
|---|---|
| ✅ Active | ~115 |
| ⚠️ Deprecated/legacy/quirk | ~14 |
| 🔴 Broken (verified) | ~7 |
| 🧪 Experimental/unused | ~6 |

🔴 Broken (verified):
- `segmentation.py::beamform_signal_cpu` (@1025) — stub body, returns None.
- ~~`compare_datasets.py` (@37,@43) — `return` at module scope → SyntaxError.~~ **REFUTED by adversarial pass** — those are `return_code = …` assignments; file is ✅ active.
- `wall_array_v2_idxs.py::v2_rssi_idxs` (@39) — both indices use "rssi0".
- `spf_nn_dataset_wrapper.py::to_absolute_north` (@86) and realtime `get_and_annotate_entry_at_idx` (@126) — unconditional `breakpoint()`.
- `spf_generate.py::generate_session` orbit branch (@221) — `sys.exit(1)`.
- `yarr_rechunk.py` (@6) — stale `new_yarr_dataset` call missing required shape args → TypeError.
- `open_partial_ds.py` CLI path (@14/@69) — `.add()` on a list → AttributeError.

**Top-5 deep-dive (Pass-2/3) candidates:**

1. **`v5spfdataset.__init__` + `refresh` + `get_segmentation` + `render_session`** (spf_dataset.py:819/1166/1652/1354) — the central read path; many side effects (zarr I/O, multiprocessing, GPU, destructive cache delete on UnpicklingError, QC raises), complex flip/double_flip and angle/frame logic (deg/π/rad mixing, sign-ambiguity comments). Highest fan-in, highest correctness risk.
2. **`segment_session` + `mp_segment_zarr`** (segmentation.py:375/107) — central segmentation pipeline; mutates input, GPU via cupy, multiprocessing, 2-receiver assert, float16 casts. Feeds all precomputed features.
3. **`v5inferencedataset`** (spf_dataset.py:448) — realtime threaded design with a mutable-default-list mutation bug, loop-variable shadowing in eviction, busy-wait/unbounded iterator, and debug prints; needs careful concurrency review.
4. **GT angle computation: `get_ground_truth_thetas` / `get_ground_truth_phis` / `get_craft_ground_truth_thetas`** (spf_dataset.py:1573/1508/1545) — the units/frame core (rad vs π, array-vs-craft-vs-absolute frames, sign ambiguity). #1 documented bug source class per the guide.
5. **`merge_v4rx_v4tx_into_v5`** (v4_tx_rx_to_v5.py:213) — v4→v5 conversion writing new zarr+yaml with GPS interpolation, heading→pis conversion, and a likely hardcoded-collector bug; destructive output creation.

**Cross-cutting themes:** (a) pervasive hardcoded `n_receivers=2` / `[0,1]` assumptions and asserts;
(b) many leftover `breakpoint()`/debug `print`/`sys.exit` and dead/commented blocks indicating
work-in-progress code; (c) several mutable-default-argument issues (`skip_fields=[]`/`set()`,
`BoundedPoint` ndarray defaults); (d) the zarr-wrangling scripts are heavily destructive (delete/move/in-place
overwrite) and a couple have stale call sites; (e) angle/frame conversions (rad/π/deg, array vs craft vs
absolute) are the main correctness-risk surface for Pass-3.
