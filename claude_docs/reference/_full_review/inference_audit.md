# Inference-Path Audit (batch cache · realtime/edge · filter consumers)

Read-only audit of **every inference code path** in SPF and its downstream consumers.
Evidence pinned to working tree at branch `main` (commit `574c454`), verified at read time.
Builds on (and does not repeat) `reference/spf/model_training_and_inference/inference_and_baseline.md`
and `reference/spf/dataset/v5inferencedataset.md`; known issues referenced by `KNOWN_ISSUES.md` number.

Severity = misbehavior × reachability (per `DOCUMENTATION_GUIDE.md`).
Confidence: ✅ verified (read the deciding lines / traced end-to-end) · 🔍 analysis (reasoned, not executed).
⚠️ no-crash = silently-wrong result; verify downstream before fixing.

---

## 0. Verdict summary

- **`.eval()` / `no_grad()` hygiene: clean on all production paths.** Every live inference site
  applies both. The only gap is `single_example_realtime_inference` (which delegates `.eval()` to
  its caller) whose sole caller — a test — forgets it (N5, test-only).
- **Live paths:** batch cache compute/consume (`run_nn_inference_on_ds` → npz cache → filters);
  `run_filters_on_data.py` NN filters (single + dual).
- **Dead paths:** realtime rover inference is a **no-op** — producers feed `v5inferencedataset`
  but the only consumer loop is commented out (`mavlink_radio_collection.py:331-336`); if
  re-enabled it immediately hits the #7 `breakpoint()`s. The B2 filter drivers are DOA (#3).
- **Broken-but-reachable (new):** the inference-cache **save** path cannot write to B2 even though
  the checked-in production config points the cache at `b2://…` (N1); a single-radio-NN cache miss
  aborts the entire filter run (N2).

---

## 1. `single_point_networks_inference.py` — branch map

| Function / branch | Lines | Status | Notes |
|---|---|---|---|
| `load_model_and_config_from_config_fn_and_checkpoint` | `:23-39` | ✅ | Mutates `config["optim"]["checkpoint"/"device"]` (`:27-29`); `force_load=True` (`:37`) bypasses the `load_single`/`load_paired` partial-load branches in `load_checkpoint` (`train_single_point.py:472-481`). Does **not** call `.eval()` — contract: caller must. All live callers do (see §6). |
| `single_example_realtime_inference` | `:42-53` | ⚠️ | `no_grad` ✅ (`:45`); **no `.eval()`** (caller's job → N5). Double forward per sample (`:51-52`, prior doc). `return None` after generator exhaust — dead `outputs` list. Only live caller: `test_v5inference.py:184`. |
| `single_example_inference` | `:56-89` | ✅ test-only | `eval` `:57`, `no_grad` `:71`. Same `//2` reshape as below. |
| `dataloader_inference` | `:92-123` | ✅ test-only | `eval` `:105`, `no_grad` `:107`. Uses only the **val** dataloader. |
| `convert_datasets_config_to_inference` | `:126-158` | ✅ | Copies dict; stride 1, flips/shuffle off, `train_on_val=True`, 1 snapshot/session (`:143-145` — the snapshots==1 invariant N4 depends on), `skip_qc`, `v4` written through (`:157`). |
| `get_inference_on_ds_noexceptions` | `:161-187` | 🔴 known #26 | Returns `None` always; swallows `Exception` (`:186-187`, not `KeyboardInterrupt`). **No `v4` param** → CLI cache population always computes with `v4=False` (concrete exploitation path of #41 — same key, different loader; see N11). |
| `get_nn_inference_on_ds_and_cache` — no-cache branch | `:208-221` | ✅ | `assert not crash_if_not_cached` (`:209`); returns **torch** tensors (dtype divergence vs numpy branches — prior doc). |
| — cache-hit branch | `:229-233` | ⚠️ | `spf_exists` supports B2 (`s3_utils.py:240-253`); `b2_file_to_local_with_cache` downloads with a per-file `FileLock` (`s3_utils.py:163-186`). Returns **numpy**. Key gaps = #41 (`v4`/basename), #47 (md5 sidecar). |
| — compute+save branch | `:234-252` | 🔴 NEW N1 | `assert not crash_if_not_cached` (`:235`) → N2. `os.makedirs`/`np.savez_compressed`/`os.rename` (`:236,250,251`) are **local-filesystem-only**: with a `b2://` cache base they write to a literal local `./b2:/…` directory and never upload. Rename is atomic locally, but the `.tmp.npz` name is shared across processes → N6 race. |
| `run_nn_inference_on_ds` | `:255-315` | ✅ | `eval` `:295` + `no_grad` `:297`. Float32 fixed (`:272`) regardless of config dtype. Re-raises loader errors (`:292-294`). |
| — the `(sessions·radios)→(sessions,2,…)` reshape | `:301-314` | ⚠️ | Hardcodes `radios=2` (`:304`). **Odd row counts crash loudly** (reshape numel mismatch → `RuntimeError`), they do not silently mis-fold. The r0/r1-adjacent interleave assumption is corroborated by the paired model itself, which splits the same batch as `[::2]`/`[1::2]` = radios (`single_point_networks.py:1041-1042,1047`). A hypothetical 1-radio dataset would crash, not corrupt. Empty dataset → `torch.vstack([])` raises at `:301` (prior doc 🔍). |
| — paired output | `:309-314` | ✅ | `paired` is computed once per pair and **duplicated to both radio rows** via `idxs = arange.repeat(1,2)` (`single_point_networks.py:1046-1052`) → consumers reading only radio 0's `paired` (dual filter) lose nothing. `paired_ntheta` may differ from `single_ntheta` (`:311`). |

## 2. `create_inference_cache.py`

| Branch | Lines | Notes |
|---|---|---|
| `--debug` → `get_nn_inference_on_ds_and_cache`, serial map | `:60-64, :80-86` | Exceptions propagate (debug intent). |
| default → `get_inference_on_ds_noexceptions` + `Pool(16)` | `:61, :89-95` | Per-dataset errors logged & swallowed; results discarded (cache side effect only). Fork + per-child CUDA init is safe here because the parent never touches CUDA. |
| `.txt` expansion | `:76-78` | Only when exactly one arg ends `.txt` (prior doc ⚠️). |
| hardcodes | `:69-72` | `batch_size=64, workers=0, crash_if_not_cached=False`; `v4` not exposed (→ N11); `--segmentation-version` required so the `None` fallback (`inference.py:203-207`) is unreachable from this CLI. |
| partial-cache states | — | Crash mid-`savez` leaves an orphan `{key}.npz.tmp.npz`; the real name only appears via `os.rename` → **no torn final file from a single writer** ✅. Two concurrent writers of the same key share the tmp name → N6 🔍. `b2://` cache base → N1. |

## 3. `spf/dataset/spf_nn_dataset_wrapper.py` — `v5spfdataset_nn_wrapper`

| Branch | Lines | Notes |
|---|---|---|
| `__init__` cached (`not ds.realtime`) | `:40-56` | `get_nn_inference_on_ds_and_cache(..., crash_if_not_cached=False, v4=ds.v4)` → computes on miss (contrast N2). `torch.as_tensor` bridges the numpy/torch dtype divergence ✅. |
| `__init__` cached + `absolute=True` | `:57-61` | Applies `cached_model_inference_to_absolute_north` to **every** key — including `"single"` → **N3** (wrong frame for `single`) and hardcoded-65 reshape (N4). |
| `__init__` realtime | `:62-82` | Loads model, `torch.set_num_threads(4)` (`:72` — process-global side effect), **`.eval()` at `:78`** ✅. Dead quantization/compile experiments commented out. `v4` ignored on this branch. |
| `to_absolute_north` (realtime) | `:84-96` | **`breakpoint()` `:86`** = known #7. Rotates **only `"paired"`** (inconsistent with the cached branch which rotates all keys — but this is actually the *more* correct behavior, see N3). Rotation math itself verified correct for `paired` (N4). The double `.reshape(paired_nn_inference.shape)` is a no-op (already flat) — benign. |
| `get_inference_for_idx` | `:98-105` | Non-realtime returns bare inference dicts; realtime returns the **full annotated sample** — shape asymmetry tolerated by the sole consumer (`observation` reads `[0]["paired"][0]`, present in both). |
| `get_and_annotate_entry_at_idx` realtime | `:108-129` | `no_grad` ✅ (`:121`); merges `v[ridx]` per radio (paired-collated batch of 2 rows — consistent with model output layout ✅). **`breakpoint()` `:126`** = known #7, unconditional on the realtime path. `absolute` → `to_absolute_north` (second #7 breakpoint). |
| `lru_cache(4)` ×3 | `:98,:107,:140` | N9 hygiene: bound-method lru_cache pins `self`+samples for interpreter life, is **shared across instances** (4 entries total per method), and `__getitem__`+`get_and_annotate…` double-cache the same values. |
| `__next__`/`__iter__` | `:131-138` | Unbounded iterator; relies on `ds` raising/blocking (v5inferencedataset never raises `StopIteration` — prior doc). |

## 4. Realtime path end-to-end (`mavlink_radio_collection.py`)

```
--realtime → v5inferencedataset (mavlink:272-283, max_store_size=3, realtime=True default @ spf_dataset.py:474)
           → v5spfdataset_nn_wrapper (mavlink:284-292: model loaded, .eval() @ wrapper:78, absolute=True, device=cpu)
producers  → DroneDataCollectorRaw(realtime_v5inf=v5inf) (mavlink:294-300)
           → write_to_record_matrix → v5inf.write_to_idx (data_collector.py:544-547)  [LIVE]
consumer   → `for x in nn_ds` loop **COMMENTED OUT** (mavlink:331-336)               [DEAD]
```

- **The realtime model never runs a forward pass in production** (N8 / extends #7): the only
  consumer is commented out; `nn_ds` is constructed and abandoned. The rover pays the model-load
  cost and the per-capture `render_session` cost (producer threads) for nothing; the store stays
  bounded at 3 via eviction, so it does not leak.
- If re-enabled: first `__next__` → wrapper `:126` `breakpoint()`, then (absolute=True) `:86`
  breakpoint → headless hang/`BdbQuit` (#7), plus v5inferencedataset races #32/#39.
- dtype/device: cpu + float32 throughout (`v5inferencedataset target_dtype=torch.float32` default,
  wrapper `device="cpu"`); window augmentation and flips are training-only (`self.training`-gated in
  `PrepareInput`, `single_point_networks.py:471-530`) and the wrapper calls `.eval()` → off ✅.
- CLI fragility (N7): `--realtime` without `--checkpoint`/`--checkpoint-config` crashes in
  `load_config_from_fn(None)` (`:264`) or the wrapper model load (`wrapper:64`); `--inference` is a
  dead flag — sets `yaml_config["inference"]=True` (`:69-70`), grep shows **no reader**, and the
  `if args.inference: pass` block (`:259-260`) is literal dead code.

## 5. Filter consumption

### `particle_single_radio_nn_filter.py` — `PFSingleThetaSingleRadioNN`
- Branch select on **`ds.temp_file`** (`:41`) (vs the wrapper's `ds.realtime`) — fine for its only
  callers (on-disk datasets via `run_filters_on_data.py:305`, tests).
- Cached branch (`:43-55`): keys the cache with `torch.as_tensor(...["single"])`; **omits
  `crash_if_not_cached`** → default `True` → **cache miss = AssertionError** → **N2**. Also omits
  `v4` → `False` (harmless here since this branch never computes — both miss paths assert).
- Realtime branch (`:57-75`): `.eval()` `:63` ✅, `no_grad` `:84` ✅.
- `observation` (`:87-88`): `[rx_idx, 0]` indexing consistent with `(2, snap, ntheta)` per-idx slice ✅.

### `particle_dual_radio_nn_filter.py` — `PFSingleThetaDualRadioNN`
- `observation` `:41` — live `breakpoint()` = known **#2** (P0, hangs the pool worker).
- `:42` reads `[0]["paired"][0]` — radio-0 only, **verified lossless**: the model duplicates the
  paired estimate onto both radio rows (`single_point_networks.py:1046-1052`) ✅.
- `cached_model_inference_to_absolute_north` (`:12-27`): the "unused helper" is actually **used** —
  by the wrapper's cached-absolute branch (`spf_nn_dataset_wrapper.py:8,59`). Math audit → N3/N4.
- Loads no model itself → no eval concern.

### `run_filters_on_data.py`
- B2 checkpoint fetch (`:83-99`): downloads `best.pth` + sibling `config.yml` into
  `checkpoints_cache_dir`; **sibling-path assumption verified correct** — the cache mirrors the B2
  key layout (`s3_utils.py:169` `local_fn = f"{tmpdir}/{fn[5:]}"`), so
  `dirname(local_ckpt)/config.yml` (`:302`, `:426`) resolves to the downloaded config ✅.
- `inference_cache` comes from the run YAML — the checked-in production config is
  **`b2://projectspf/md2/cache/inference`** (`ekf_and_pf_config.yml:55,71`) → N1/N2 apply; combined
  with #41/#47 staleness, a stale-or-missing key either silently recomputes per worker (dual NN)
  or aborts the whole run (single NN).
- `__main__` `run_filter_jobs(jobs)` missing `nparallel` = known **#12** (also drops
  `args.parallel/args.debug` and never wires `checkpoints_cache_dir`; the module-global at `:29`
  is dead). `eval()` on config strings = known **#13**.
- No per-job try/except in `run_jobs_with_one_dataset` (`:78-181`) → any filter exception
  propagates through `pool.imap_unordered` and kills the whole run (the mechanism behind N2's
  blast radius).

## 6. Model-eval hygiene sweep (every checkpoint-for-inference site)

| Site | `.eval()` | `no_grad` | Verdict |
|---|---|---|---|
| `run_nn_inference_on_ds` | `:295` ✅ | `:297` ✅ | clean |
| `single_example_inference` | `:57` ✅ | `:71` ✅ | clean (test-only) |
| `dataloader_inference` | `:105` ✅ | `:107` ✅ | clean (test-only) |
| `single_example_realtime_inference` | ❌ (delegated) | `:45` ✅ | caller must eval → see below |
| — caller `test_v5inference.py:175-190` | ❌ **forgot** | (inside gen) ✅ | **N5** — train-mode inference in test |
| `spf_nn_dataset_wrapper` realtime | `:78` ✅ | `:121` ✅ | clean |
| `PFSingleThetaSingleRadioNN` realtime | `:63` ✅ | `:84` ✅ | clean |
| `train_single_point.py` val loop | `:1218` ✅ (train resumes `:1297`) | (val ctx) | clean (training-owned) |
| `simple_train_filter.py:219-230` | n/a | n/a | training-resume load, not inference |
| `plot/save_load.py:11` | n/a | n/a | CPU-unpickler shim, no module state |

Dropout is real in these models (`nn.Dropout` at `single_point_networks.py:37-38,72,91,435` plus
`self.training`-gated input-dropout masks `:471-530`), so a missed `.eval()` is not hypothetical —
but every **production** load site has it. Verdict: **hygiene clean in production; one test-only gap (N5)**.

---

## 7. Issues (new this audit)

| # | Sev | Conf | Issue | Location / reachability |
|---|---|---|---|---|
| **N1** | 🟠 P1 · ⚠️ no-crash | ✅ | **Inference-cache save path cannot write to B2.** Cache-miss compute does `os.makedirs`/`np.savez_compressed`/`os.rename` on the raw cache string; with the checked-in `inference_cache: b2://projectspf/md2/cache/inference` these create a literal local `./b2:/…` directory and the npz is **never uploaded** — `spf_exists` stays False forever. Consequence: dual-NN filter jobs silently recompute full-dataset inference in **every** pool worker for every job; nothing in the repo can populate the remote cache (`create_inference_cache.py` shares the same save code). Read side (`spf_exists`/`b2_file_to_local_with_cache`) *does* speak B2, masking the asymmetry. Only escape: someone manually syncs a locally-populated cache to B2. | `single_point_networks_inference.py:236,250-251`; reachable via `ekf_and_pf_config.yml:55,71` → `run_filters_on_data.py` (the `_b2` driver itself is dead, #3, so exposure is via direct `run_filters_on_data` runs) |
| **N2** | 🟠 P1 (crash) | ✅ | **Single-radio NN filter aborts the whole filter run on any cache miss.** `PFSingleThetaSingleRadioNN` omits `crash_if_not_cached` → default `True` → `assert` at `inference.py:235` raises out of the constructor; `run_jobs_with_one_dataset` has no per-job try/except, so the exception propagates through `pool.imap_unordered` and kills the entire `run_filter_jobs` sweep. Inconsistent with the dual-NN path (wrapper passes `False` and computes). Trigger: any stale/missing key — guaranteed under N1 with the B2 cache, and made likelier by #41/#47. | `particle_single_radio_nn_filter.py:44-54` → `single_point_networks_inference.py:235`; `run_filters_on_data.py:643-649` |
| **N3** | 🟡 P2 · ⚠️ no-crash | ✅ | **Absolute-north rotation applied to the wrong frame for `"single"`.** The wrapper's cached `absolute=True` branch rotates **every** cached key by `rx_heading·π`, but `"single"` is radio-frame; correct absolutization needs `(rx_theta_in_pis + rx_heading_in_pis)·π` (frames: `spf_dataset.py:1583-1591`; the paired net itself rotates single→craft by `rx_theta_in_pis·π` at `single_point_networks.py:1024-1028`). Result: absolute-mode `"single"` sits in a frame that is neither radio, craft, nor absolute — off by the radio mount angle. **Latent**: the only consumer (`PFSingleThetaDualRadioNN.observation:42`) reads `"paired"` only. Also inconsistent with the realtime `to_absolute_north` which (more correctly) rotates only `"paired"` (`spf_nn_dataset_wrapper.py:84-96` vs `:57-61`). | `spf_nn_dataset_wrapper.py:57-61` |
| **N4** | 🟡 P2 | ✅ math / ✅ hardcode | **Paired absolute rotation verified correct; helper hardcodes 65 bins.** Sign/magnitude check out: `rotate_dist` shifts a distribution by +rot (`rf.py:1527-1556`), θ_abs = θ_craft + `rx_heading·π` (`spf_dataset.py:1545-1553,1583-1591`), and the 2π/n circular bin grid matches `theta_phi_to_bins`/`to_bin` (`filters.py:362-365`, `utils.py:250-255`). Interleave `[r0,r1,…]` matches the cache layout **only because** inference snapshots==1 (`convert_datasets_config_to_inference:144-145`). But `cached_model_inference_to_absolute_north` hardcodes `reshape(-1, 65)` — any checkpoint with `output_ntheta≠65` crashes (or, if numel divides, silently mis-folds); `rotate_dist`'s own 65-assert is commented out (`rf.py:1514`). | `particle_dual_radio_nn_filter.py:22` |
| **N5** | 🟡 P2 (test-only) | ✅ | **Train-mode inference in `test_v5inference.py`.** `single_example_realtime_inference` never calls `.eval()` (by contract), and its only caller `test_v5inference.py:175-190` also never does — the model stays in train mode, with live `nn.Dropout` and `self.training`-gated input dropout (`single_point_networks.py:37-38,471-530`). Benign today only because the test asserts nothing about outputs (it times the loop). No production caller exists. | `single_point_networks_inference.py:42-53`; `tests/test_v5inference.py:184` |
| **N6** | 🟡 P2 | 🔍 | **Concurrent cache-write race on a shared `.tmp` name.** Two pool workers computing the same missing key (two jobs on the same ds+checkpoint in `run_filters_on_data`, or a duplicated dataset in `create_inference_cache`) both write `{key}.npz.tmp.npz` then rename: writer B's `open('wb')` truncation can interleave with writer A's rename → a truncated npz published under the final name; next reader crashes in `np.load`. Narrow window; local caches only (B2 caches never save, N1). | `single_point_networks_inference.py:250-251` |
| **N7** | 🟡 P2 | ✅ | **Realtime CLI arg fragility + dead `--inference` flag.** `--realtime` without `--checkpoint-config` crashes (`load_config_from_fn(None)` at `:264` or wrapper model-load `wrapper:64`); `--inference` only sets `yaml_config["inference"]=True` (`:69-70`), which **nothing reads** (repo grep: single write site), and `if args.inference: pass` (`:259-260`) is literal dead code. | `mavlink_radio_collection.py:69-70,259-266` |
| **N8** | 🟠 P1 (capability gap) · confirms/extends #7 | ✅ | **Realtime NN inference is a production no-op.** Producer side live (`data_collector.py:544-547`), model loaded + eval'd (`mavlink:284-292` → `wrapper:78`), but the only consumer loop is commented out (`mavlink:331-336`) — no forward pass ever runs on the rover; `--realtime --checkpoint` buys model-load + per-capture `render_session` cost with zero output. Re-enabling immediately hits both #7 breakpoints (absolute=True path) and exposes #32/#39. | `mavlink_radio_collection.py:331-336` |
| **N9** | 🟢 P3 | ✅ | `lru_cache(4)` on three bound methods pins `self`+samples for interpreter life, is shared across all wrapper instances (thrash with >1 concurrent wrapper), and `__getitem__` double-caches `get_and_annotate_entry_at_idx`. | `spf_nn_dataset_wrapper.py:98,107,140` |
| **N10** | 🟢 P3 · cross-ref #41 | ✅ | **CLI-populated cache is always `v4=False`** (`get_inference_on_ds_noexceptions` has no `v4` param; `create_inference_cache` doesn't expose it) while the wrapper queries the **same key** with `v4=ds.v4` — the concrete producer/consumer pair through which the #41 key gap yields wrong cached results for v4 datasets. | `single_point_networks_inference.py:161-172`; `create_inference_cache.py:59-74`; `spf_nn_dataset_wrapper.py:37-38,54` |

### Verified-clean (checked, no issue)
- Reshape interleave assumption r0/r1-adjacent — corroborated by the model's own `[::2]/[1::2]`
  radio split (`single_point_networks.py:1041-1042`); odd counts crash loudly, never silently. ✅
- Dual filter reading only radio-0 `paired` — lossless (output duplicated per radio,
  `single_point_networks.py:1046-1052`). ✅
- B2 checkpoint→sibling-config path derivation in `run_filters_on_data` — cache layout mirrors B2
  keys (`s3_utils.py:169`), sibling holds. ✅
- `rotate_dist` bin-width convention vs `theta_phi_to_bins`/`to_bin` — consistent (2π/n circular). ✅
- Atomic single-writer cache save (`.tmp.npz` → rename) — correct locally (prior doc). ✅

### Known issues re-confirmed in passing (not re-derived)
#2 (dual-filter `breakpoint()` `:41`), #7 (wrapper breakpoints `:86,:126`), #12 (`run_filter_jobs(jobs)`
TypeError `:731`), #13 (`eval()` on config strings), #26 (noexceptions returns None), #41 (cache key
omits `v4`/dir), #47 (md5 sidecar staleness), #31/#32/#33/#39 (filter/realtime-dataset hazards).

---

## 8. Path liveness map

| Path | State | Why |
|---|---|---|
| Batch inference → npz cache (`create_inference_cache.py`, local cache dir) | ✅ live | Sole populated path; `v4` frozen to False (N10). |
| Cache consume → `PFSingleThetaSingleRadioNN` | ✅ live but brittle | Crash-on-miss aborts whole sweep (N2). |
| Cache consume → wrapper → `PFSingleThetaDualRadioNN` | 🔴 live-but-broken | #2 breakpoint fires on the first trajectory step; absolute mode adds N3/N4. |
| B2-backed inference cache | 🔴 broken (write half) | Reads work, saves land in a local `./b2:/` dir (N1). |
| B2 cloud filter drivers (`run_filters_on_data_b2.py`) | 💀 dead | #3 import error, ~14 months. |
| Realtime rover (`--realtime --checkpoint`) | 💀 no-op | Consumer commented out (N8); would hit #7 if revived. |
| `single_example_realtime_inference` | 🧪 test-only | And its test runs the model in train mode (N5). |
| Legacy baseline / empirical table | (prior doc) | #25/#48 unchanged. |
