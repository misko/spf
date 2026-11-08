# Inference, Inference-Cache, Empirical P(θ|φ) & Geometric Baseline

Deep-dive reference (Passes 1→3 + active bug hunt) for the non-training inference path:

- `spf/model_training_and_inference/models/single_point_networks_inference.py` — checkpoint load, run, and the npz inference cache.
- `spf/scripts/create_inference_cache.py` — CLI that populates the cache in parallel.
- `spf/scripts/create_empirical_p_dist.py` — the empirical P(θ|φ) heatmap table + symmetry rules.
- `spf/model_training_and_inference/baseline_algorithm.py` — the classical DOA → 2D-point baseline.

Units/frames: angles in **radians**, θ relative to detector/craft frame, φ = inter-antenna phase
difference in `[-π, π]`. See [`00_concepts/conventions.md`](../../00_concepts/conventions.md).

Status legend: ✅ Active · ⚠️ caveat · 🔴 broken (verified) · 🧪 experimental/unused ·
🔍 analysis (reasoned, not executed) · ⚠️ no-crash (silently-wrong, verify downstream before fixing).

Evidence pinned to working tree at branch `main` (commit `574c454`), verified at read time.

---

## A. `single_point_networks_inference.py`

### `load_model_and_config_from_config_fn_and_checkpoint(config_fn, checkpoint_fn, device=None)` · `:23` · ✅ Active

**Summary.** Builds the model from a YAML config and loads checkpoint weights. The single entry
point every consumer uses to materialize a runnable model from disk.

**Signature**
- Inputs: `config_fn` (path to `config.yml`); `checkpoint_fn` (path to `.pth`); `device` (str or None — if None, uses `config["optim"]["device"]`).
- Returns: `(model: nn.Module, config: dict)`.
- Side effects: reads config + checkpoint from disk (`torch.load` inside `load_checkpoint`), moves model to device (GPU), **mutates `config["optim"]["checkpoint"]` and `["device"]`** (`:27,:29`). Loads with `force_load=True` (`:37`).

**Called from** (Pass 2) — high fan-in:
- `run_nn_inference_on_ds` (`:267`).
- `particle_single_radio_nn_filter.py:59`, `particle_dual_radio_nn_filter.py`, `spf_nn_dataset_wrapper.py:64` (realtime path).
- Tests: `test_v5inference.py:175`, `test_new_models.py:42,84,131,161`.

**Contract.** Pre: config has `model`/`global`/`optim` sections; checkpoint matches model arch (else `load_checkpoint` raises). Post: returns model already `.to(device)`; **not** set to `.eval()` here — callers must call `model.eval()` themselves (they do, e.g. `:295`).

**Analysis.** `force_load=True` bypasses partial-load fallbacks in `load_checkpoint`; a state-dict mismatch raises. No issues found beyond the documented config mutation.

---

### `run_nn_inference_on_ds(...)` · `:255` · ✅ Active

**Summary.** Core compute path: load model+config, build an inference-mode dataloader over one
dataset, run the model over every batch, and reshape outputs to `(sessions, radios=2, snapshots, ntheta)`.

**Signature**
- Inputs: `ds_fn, config_fn, checkpoint_fn, device, batch_size, workers, precompute_cache, segmentation_version, v4=False`.
- Returns: `{"single": Tensor[sessions,2,snap,ntheta]}`, plus `"paired"` if the model emits it. Tensors are `.cpu()`.
- Side effects: `model.eval()`; dataloader file I/O; GPU; **re-raises** load errors (`:294`).

**Called from**: `get_nn_inference_on_ds_and_cache` (`:211, :238`) — both the no-cache and compute-then-save branches.

**Contract.** Reshape hardcodes **2 radios** (`radios = 2`, `sessions = sessions_times_radios // 2`, `:303-304`). Assumes the val dataloader yields radios interleaved in pairs. Output θ-dim may differ between `single` and `paired` (`single_ntheta` vs `paired_ntheta`, `:302,:311`).

**Analysis.**
- 🔍 **Empty-dataset → `outputs[0]` IndexError / `torch.vstack([])` crash** (`:301, :309`). If the dataloader yields zero batches, `torch.vstack([])` raises and `outputs[0]` (`:309`) would `IndexError`. Reached only for an empty/all-filtered dataset; the noexceptions wrapper would swallow it. Severity low (degenerate input). Confidence 🔍.
- `radios = 2` hardcode is correct for this 2-antenna rig but silently mis-reshapes any non-paired future dataset. ⚠️ no-crash if a 1-radio dataset ever flows in (would mis-fold the session axis without raising). Confidence 🔍.

---

### `get_nn_inference_on_ds_and_cache(...)` · `:190` · ⚠️ Active — **cache-key gaps**

**Summary.** Cache-aware wrapper. Hashes config + checkpoint (md5), builds a cache path keyed by
`dataset basename / segmentation_version / checkpoint_md5 / config_md5.npz`, returns cached npz if
present (local or B2), else computes and atomically saves.

**Signature**
- Inputs: same as `run_nn_inference_on_ds` plus `inference_cache` (base dir / b2 prefix or None), `crash_if_not_cached=True`.
- Returns: results dict of **numpy** arrays (note: numpy, not torch — `:249, :233`).
- Side effects: md5 of config+ckpt (each writes a `.md5` sidecar — see staleness below), `spf_exists` (B2 network), `b2_file_to_local_with_cache` (download), `os.makedirs`, `np.savez_compressed` to `.tmp` then `os.rename` (atomic-ish), GPU. `assert not crash_if_not_cached` if uncached (`:209, :235`).

**Cache key** (`:228`):
```
f"{inference_cache}/{ds_basename}/{segmentation_version:0.3f}/{checkpoint_checksum}/{config_checksum}.npz"
```

**Called from**: `get_inference_on_ds_noexceptions` (`:174`); `particle_single_radio_nn_filter.py:44`;
`spf_nn_dataset_wrapper.py:43`; `create_inference_cache.py` (`--debug` path); tests
`test_new_models.py:103`, `test_particle_filter.py:126,185`.

**Contract.** Pre: if `inference_cache is None` then `crash_if_not_cached` must be False (asserted `:209`).
Post: result dict keys are numpy arrays when going through the save path, but **torch tensors converted to numpy only on the save branch** — the no-cache branch (`:211`) returns torch tensors from `run_nn_inference_on_ds`. Callers wrap with `torch.as_tensor(...)` (`particle_single_radio_nn_filter.py:43`) so both work, but the **return dtype is inconsistent across branches** (numpy from cache hit/save, torch from no-cache). ⚠️ no-crash; confidence ✅ (read `:233` numpy vs `:211` torch).

**Analysis — NEW ISSUES (cache key does NOT capture everything that changes outputs):**
- 🔴/🔍 **`v4` is NOT in the cache key but changes outputs** (`:201, :228`). `v4` is threaded into `run_nn_inference_on_ds` → `convert_datasets_config_to_inference` → `load_dataloaders` → `load_single_dataset_using_configs(..., v4=...)` (`train_single_point.py:200,246`), which changes dataset loading/segmentation and thus model inputs. Two runs differing only in `v4` resolve to the **same** `.npz` path → stale/colliding cache. `spf_nn_dataset_wrapper.py:54` passes `v4=ds.v4` (varies), so this is reachable. Severity high (silently-wrong cached results), confidence 🔍 (path traced; not executed end-to-end). ⚠️ no-crash.
- 🔍 **`nthetas` / model `outputs` not in key.** The cache key captures `config_md5`, so if `nthetas` lives in the config the md5 covers it — **verify** `nthetas` is fully config-derived (it is read from `config["global"]["nthetas"]`). If any caller overrides nthetas outside the config, the key misses it. Confidence 🔍.
- ⚠️ **`precompute_cache` not in key.** Different precompute caches (different segmentation outputs) feeding the same `(ds, seg_version, ckpt, config)` would collide. In practice `segmentation_version` is the proxy for precompute content, so this is usually safe, but it is an implicit assumption. Confidence 🔍.
- ⚠️ **`get_md5_of_file` caches md5 in a `.md5` sidecar that is never invalidated** (`utils.py:88-103`, `@functools.cache`). If `config.yml` or `best.pth` is edited **in place** without deleting `<fn>.md5`, the stale sidecar md5 is returned and the cache key does not change → a changed checkpoint silently reuses old inference. Also `@functools.cache` memoizes per-process so an in-place file change mid-process is invisible. Severity medium, reachable if anyone overwrites a checkpoint/config without renaming. Confidence ✅ (read the sidecar read/write at `utils.py:89-102`). ⚠️ no-crash.
- ✅ Atomic save: writes `inference_cache_fn + ".tmp"` then `np.savez_compressed` appends `.npz` → file is actually `...npz.tmp.npz`, renamed to `...npz` (`:250-251`). Correct, but the `.tmp` intermediate name is `{path}.npz.tmp.npz` which is non-obvious; verified the rename target matches.

---

### `get_inference_on_ds_noexceptions(...)` · `:161` · 🔴 returns None on success (KNOWN #26 — CONFIRMED)

**Summary.** Exception-swallowing wrapper around `get_nn_inference_on_ds_and_cache`, used by the
parallel cache-population CLI so one bad dataset doesn't kill the pool.

**Analysis — CONFIRMS KNOWN_ISSUES #26.**
- 🔴 **Calls `get_nn_inference_on_ds_and_cache(...)` (`:174`) but never `return`s its value.** On success it returns `None`; on failure it logs and returns `None` (`:186-187`). Verified ✅ — there is no `return` statement anywhere in the function body.
- **Reachability / severity:** the **only** non-test caller is `create_inference_cache.py` (`:61`), which discards results (`results = list(... map(run_fn ...))`, `:81-95`) — it uses the function purely for its **cache-population side effect**, so the missing return is **benign there**. Severity is therefore **low in practice**: it would only bite a future caller who expects the dict back. No live caller currently relies on the return value (the filter/wrapper callers all call `get_nn_inference_on_ds_and_cache` directly, not the noexceptions variant). Confidence ✅.
- Note: signature lacks the `v4` param that `get_nn_inference_on_ds_and_cache` has (`:161-172` vs `:190-202`), so the noexceptions/CLI path can never set `v4` — it always uses the default `v4=False`. Minor; flagged ⚠️.

---

### `convert_datasets_config_to_inference(...)` · `:126` · ✅ Active

Produces an inference-mode `datasets_config`: stride 1, `flip/double_flip/shuffle=False`,
`train_on_val=True`, 1 snapshot/session, `skip_qc=True` (`:135-157`). Copies the dict (no arg
mutation). `precompute_cache` only overwritten if non-None (`:155`). `v4` written through (`:157`).
No issues found.

### `single_example_inference` · `:56` / `dataloader_inference` · `:92` · ✅ Active (test-only)

Eager per-example / per-dataloader-batch variants. Same `(sessions,2,snap,ntheta)` reshape. Called
only from tests (`test_new_models.py`). `dataloader_inference` reuses `load_dataloaders` and only
the **val** dataloader (`:96`).

### `single_example_realtime_inference` · `:42` · ⚠️ double inference per sample

**Analysis.** ⚠️ runs `model(single_example)` **twice** per sample — `yield model(...)` (`:51`)
then `outputs.append(model(...))` (`:52`) — and the `outputs` list is discarded (`return None`,
`:53`). Wasteful (2× compute) and the yielded vs appended outputs are independent forward passes
(could differ if the model had dropout in train mode, but callers `.eval()` first). Reached via
`test_v5inference.py:184` and realtime filter paths. Severity low (perf), confidence ✅.

---

## B. `create_inference_cache.py` · ✅ Active CLI

**Summary.** CLI that `partial`s an inference fn over a list of datasets and runs it through a
`multiprocessing.Pool` (or serial map in `--debug`). Populates the npz cache.

**Analysis.**
- Picks `get_inference_on_ds_noexceptions` normally (swallow per-dataset errors) vs
  `get_nn_inference_on_ds_and_cache` in `--debug` (`:60-64`). Results discarded either way (`:81-95`) — only cache files matter.
- Hardcodes `batch_size=64, workers=0` (`:69-70`) and `crash_if_not_cached=False` (`:72`).
- `.txt` dataset-list expansion only triggers when **exactly one** `--datasets` arg ends in `.txt` (`:76`). Mixed `.txt + prefix` lists are not expanded. ⚠️ minor.
- `--segmentation-version` is `required=True` (`:31-35`) → the `segmentation_version is None` default-fallback in `get_nn_inference_on_ds_and_cache` (`:203-207`) is unreachable from this CLI. Good.
- No issues that corrupt output; the `v4=False` default applies (param not exposed on CLI).

---

## C. `create_empirical_p_dist.py` — empirical P(θ|φ) table + symmetry

### `get_heatmap_for_radio(dss, radio_idx, bins)` · `:16` · ✅ Active

2D histogram of (ground-truth θ, mean phase φ) for one radio across datasets, masking non-finite φ
(`:19`). Returns `(heatmap[θ_bin][φ_bin], xedges, yedges)`. Note **rows = θ (ground truth), cols =
φ (observed)**. No issues.

### `create_heatmaps_and_plot(dss, bins, save_fig_to_prefix=None)` · `:32` · ✅ Active — **normalization is subtle**

**Summary.** Builds the conditional tables consumed at inference: `*_phi_norm` = P(θ|φ) and
`*_theta_norm` = P(φ|θ), for radios r0/r1 and their average `r`, with and without symmetry. Returns
`{"r0"/"r1"/"r" → {"sym"/"nosym"} → Tensor}`. The stored/consumed tensor is the **phi_norm**
(P(θ|φ)) one — `heatmaps[...]["sym"|"nosym"] = r*_phi_norm` (`:57-59`).

**The normalization (verified ✅):**
- Histogram `r` is `m[θ][φ]` (`:49`).
- `r_phi_norm = (r / r.sum(axis=0, keepdims=True)).T` (`:53-55`). Dividing by the **θ-axis sum**
  (axis 0) makes each **column** (fixed φ) sum to 1 → P(θ|φ); the `.T` then makes it `m[φ][θ]`.
  So the stored phi_norm tensor is indexed `[φ_bin][θ_bin]` and `tensor[φ].sum()==1`. ✅
- `get_empirical_dist` (`spf_dataset.py:239`) looks it up by `to_bin(mean_phase, n)` and indexes the
  **first** axis → φ → returns a θ-distribution. **Consistent.** ✅ verified the full producer→consumer
  chain (`create_heatmaps_and_plot` → pickle → `v5spfdataset.empirical_data` → `get_empirical_dist` →
  `to_bin(mean_phase_segmentation, empirical_dist.shape[0])` at `spf_dataset.py:782/1475`).

**Analysis — issues.**
- ⚠️ **suptitles are swapped relative to the figure content** (`:90-91`): `fig_phi_norm.suptitle("theta conditional on phi")` is on the figure that plots `r*_phi_norm` — that is actually correct (phi_norm IS P(θ|φ)). But `fig_theta_norm` (plots `r*_theta_norm` = P(φ|θ)) gets suptitle `"phi conditional on theta"` — also correct. **The inline comment at `:78` `# write maps in map[phi][theta] = pr(theta | phi)` is mis-pasted above the theta_norm block** (copy of the phi_norm comment). Cosmetic only, no output effect. Confidence ✅.
- 🔴/⚠️ **even `bins` produces a malformed table** — see `apply_symmetry_rules_to_heatmap` below. The CLI default `--nbins=50` (`:130`) is **even** → the `sym` entries are wrong-shaped/garbage. Production artifacts use odd bins (`nbins65`, `nbins7` per repo filenames), so reachable-but-usually-avoided. See next.

### `apply_symmetry_rules_to_heatmap(h)` · `:100` · 🔴 broken for even `bins` (NEW)

**Summary.** Enforces the front/back + ±θ reflection symmetry of a 2-antenna array on the histogram
`h[θ][φ]`: positive-θ rows are folded with flipped negative-θ rows, then a φ-reflection is added,
then the half is mirrored back to a full table.

**The symmetry math (Pass 3):**
- `half = h[:ceil(bins/2)] + flip(h[floor(bins//2):])` (`:105`) — folds θ and −θ (with full reversal,
  which also reverses φ): encodes "P(θ, φ) == P(−θ, −φ)". For **odd** bins, `ceil(bins/2)` and
  `bins - floor(bins//2)` both = `(bins+1)/2`, so the two slices are the same length and the center
  row is included in both (added to itself). For **even** bins the slices have **different lengths**
  → broadcasting/shape behavior is wrong.
- `half = half + flip(half, axis=0)` (`:107`) — adds the φ-reflection "P(θ,φ)==P(θ,−φ)" (i.e. ±90°
  ambiguity noted in the comment `:106`).
- `full = vstack([half[:-1], flip(half)])` (`:108`) — mirror back, dropping the duplicated center.

**Verified ✅ by execution** (read-only numeric check, not a source change):
| bins | output shape | verdict |
|---|---|---|
| 7  | (7,7)  | ✅ correct |
| 65 | (65,65)| ✅ correct |
| 8  | (7,8)  | 🔴 axis-0 collapsed to 7 — wrong |
| 50 | (49,50)| 🔴 axis-0 collapsed to 49 — wrong |

- 🔴 **For even `bins`, `apply_symmetry_rules_to_heatmap` returns a table whose θ axis is one short
  and whose fold is misaligned** (the `floor(bins//2)` vs `ceil(bins/2)` slices differ in length).
  Reached when `create_heatmaps_and_plot` is called with even bins — the CLI **default `--nbins=50`
  is even** (`:130`). The `sym` branch (`:44-47`) is the production lookup default
  (`empirical_symmetry=True`, `spf_dataset.py:463`), so a 50-bin table would feed garbage to every
  empirical consumer. **Mitigant:** all checked-in artifacts use odd bins (65/7). Severity high *if*
  default is ever used; confidence ✅ (executed). ⚠️ no-crash for the symmetry consumer (it would
  index a mis-shaped array, not necessarily raise).
- ⚠️ The returned table is **not re-normalized** after symmetrization (the `# / full.sum(...)`
  re-normalize is commented out at `:109`). Normalization to P(θ|φ) happens *before* symmetry in
  `create_heatmaps_and_plot` (`:53-55`) and symmetry is applied to the **raw histogram** `r0/r1/r`
  *before* the phi_norm division (`:44-47` run on `r0,r1,r`, then `:53-55` normalize) — so the final
  phi_norm IS normalized. **Verified the order is fold-then-normalize → OK.** ✅ (No bug; flagged
  because the commented-out line looks like a missing normalize but isn't on the live path.)

### `create_empirical_p_dist(args)` · `:151` · ✅ Active

Loads datasets (swallowing per-dataset load errors, `:168-169`), groups by
`{sdr_device_type}_{rx_spacing_str}` (`:206`), consistency-checks that rx wavelength spacing is
constant within a dataset (warns `:194`, asserts cross-radio equality `:201-203`), builds heatmaps
per group, pickles to `args.out`. The group key **exactly matches** the consumer key in
`get_empirical_dist` (`spf_dataset.py:239` `f"{v5ds.sdr_device_type}_{rx_spacing_str}"`). ✅ verified
key parity (this is the join that must not drift).

---

## D. `baseline_algorithm.py` — geometric DOA → 2D point

### `baseline_algorithm(session, width, steps=-1)` · `:27` · 🔴 `steps==-1` default broken (KNOWN #25 — CONFIRMED)

**Summary.** Legacy v1/Task2 baseline: for each timestep, take the top-N beamformer peaks → θ,
rotate by detector orientation → direction lines from the detector position, then vote line
intersections into N localization points via `get_top_n_points`.

**Analysis — CONFIRMS KNOWN_ISSUES #25.**
- 🔴 **`steps == -1` branch computes a shape but never assigns `steps`** (`:29-30`):
  ```python
  if steps == -1:
      session["beam_former_outputs_at_t"].shape[0]   # value discarded, steps NOT set
  for idx in np.arange(steps):                        # np.arange(-1) == [] → loop body never runs
  ```
  Verified ✅: `np.arange(-1)` is empty, so with the default `steps=-1` the function builds zero
  lines and returns `get_top_n_points([], ...)` → empty `final_points`, zero images. The `.shape[0]`
  on `:30` is evaluated and thrown away (missing `steps = ...`).
- **Reachability / severity:** the **only** caller is `plot.py:342`, which always passes an explicit
  `steps=idx` (`fp, imgs, pred_points = baseline_algorithm(session, width, steps=idx)`), so the
  broken default is **never hit in production** → severity **low**. Confidence ✅. The default is
  dead/broken, masked by the sole caller.
- Operates on **legacy v1 "session" dicts** (`beam_former_outputs_at_t`, `thetas_at_t`,
  `detector_orientation_at_t`, `detector_position_at_t`) — not the v5 dataset path. The whole module
  is legacy.

### `get_top_n_peaks(bf, n=2, peak_size=15)` · `:5` · ✅ Active (with asymmetric window)

Greedy top-N peak finder; copies `bf` (no arg mutation, `:6`), suppresses a window around each found
peak by setting it to `-inf` (`:11-12`), wrapping with `% bf.shape[0]`.
- ⚠️ **suppression window is asymmetric**: `np.arange(-peak_size//2, peak_size//2+1)` with
  `peak_size=15` → offsets `-8..+7` (because `-15//2 == -8` floor division), i.e. 16 bins, skewed
  one bin negative. Verified ✅. Minor — peaks are still suppressed, just slightly off-center.
- Wrap via modulo means a peak near index 0 suppresses the high end too — intended for a circular
  θ axis. ✅.

### `get_top_n_points(...)` · `:49` · ⚠️ dense geometry, int-dtype distance bug

Greedy assignment of lines to N intersection points via rasterized voting images
(`skimage.draw.line`). Returns `(final_points, imgs[n,W+1,W+1], points_per_line)`.
- ⚠️ **`line_to_point_distances` is `dtype=int`** (`:54`) but stores float distances
  (`distance_to_line`, `:81`) → distances are **truncated to int** on assignment, coarsening the
  "closest point" comparison (`:78, :106-107`) and the `>= threshold` test (`:61, :84`). Verified ✅
  the array is int and the value is float. Severity low (baseline is legacy/plot-only). ⚠️ no-crash.
- `boundary_point` (`:129`) can raise `AssertionError` on a zero direction vector (`:162`) or if
  neither candidate boundary point lies in the box (`:166`). Reachable for degenerate/horizontal/
  vertical lines. ⚠️.

### Dead helpers
- `frac_to_theta` (`:17`) — 🧪 no callers in repo.
- `lines_to_points` (`:170`) — 🧪 no callers (seeded-RNG intersection sampler, superseded by `get_top_n_points`).

---

## Cluster summary

| Function | File:line | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|---|
| `load_model_and_config_from_config_fn_and_checkpoint` | inference.py:23 | ✅ | high | ✅ test_new_models | mutates config optim; no `.eval()` |
| `run_nn_inference_on_ds` | inference.py:255 | ✅ | 2 | indirect | hardcoded radios=2; empty-ds crash 🔍 |
| `get_nn_inference_on_ds_and_cache` | inference.py:190 | ⚠️ | high | ✅ | **cache key omits `v4`**; md5 sidecar staleness |
| `get_inference_on_ds_noexceptions` | inference.py:161 | 🔴 | 1 (CLI) | ❌ | returns None (KNOWN #26) — benign at sole caller |
| `single_example_realtime_inference` | inference.py:42 | ⚠️ | realtime+test | ✅ | 2× forward per sample |
| `create_inference_cache __main__` | create_inference_cache.py | ✅ | CLI | ❌ | discards results; v4 not exposed |
| `create_heatmaps_and_plot` | empirical.py:32 | ✅ | 1 | ❌ | normalization correct; comment/suptitle mis-paste |
| `apply_symmetry_rules_to_heatmap` | empirical.py:100 | 🔴 | 1 | ❌ | **broken for even bins** (default 50!) |
| `create_empirical_p_dist` | empirical.py:151 | ✅ | CLI | ❌ | group key matches consumer key ✅ |
| `baseline_algorithm` | baseline.py:27 | 🔴 | 1 (plot) | ❌ | `steps==-1` default broken (KNOWN #25) — masked by caller |
| `get_top_n_peaks` | baseline.py:5 | ✅ | 2 | ❌ | asymmetric suppression window |
| `get_top_n_points` | baseline.py:49 | ⚠️ | 1 | ❌ | int-dtype distance truncation |

**Cross-cutting theme.** The inference *cache key* (`ds / seg_version / ckpt_md5 / config_md5`) is
the load-bearing correctness mechanism, and it has two real gaps: (1) the `v4` flag changes outputs
but is absent from the key, and (2) `get_md5_of_file` caches md5 in a never-invalidated `.md5`
sidecar + `@functools.cache`, so an **in-place** edit to a config/checkpoint silently reuses stale
inference. Both are silently-wrong (no crash). Separately, the two legacy/auxiliary paths (empirical
symmetry for even bins, baseline `steps==-1`) are broken-but-protected: even-bin symmetry is dodged
because production uses odd bins, and the baseline default is overridden by its sole caller. The
producer→consumer key/normalization join for the empirical table was verified end-to-end and is
correct on the odd-bin path.

**Open questions / TODO.**
- Confirm no production invocation builds the empirical table with even `--nbins` (default is 50; all
  checked-in artifacts appear to be 65/7). If any pipeline uses the default, its `sym` table is bad.
- Confirm whether any checkpoint/config files are ever overwritten **in place** in the cache-population
  pipeline (would trigger the `.md5` sidecar staleness). If checkpoints are always written to fresh
  paths, the staleness is unreachable.
- Confirm `v4` is held constant across all cache-populating callers for a given inference-cache dir;
  if not, the missing `v4` in the key causes collisions.
