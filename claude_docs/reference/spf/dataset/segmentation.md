# `spf/dataset/segmentation.py` — Segmentation / preprocessing pipeline

Reference doc per [`DOCUMENTATION_GUIDE.md`](../../../DOCUMENTATION_GUIDE.md). Read-only deep dive
(Passes 1→3 + active bug hunt). Evidence pinned to commit `574c4548` (`git rev-parse HEAD`).
Status tags: ✅ Active · ⚠️ quirk/legacy · 🔴 Broken (verified) · 🧪 Experimental/unused.
Confidence tiers: ✅ verified (read the deciding lines) · 🔍 analysis (reasoned, not executed).

Units/frames: phase differences are **radians**, wrapped to `[-π, π]` by `pi_norm`
(`spf/rf.py:746` `get_phase_diff`). `window_size`/`stride`/`*_idx` are **sample counts**.
`trim` is a **percentage** (0–100), not a fraction. See
[`00_concepts/conventions.md`](../../../00_concepts/conventions.md).

This module is the **hot path that builds the `.yarr` cache**: it reads the raw radio `signal_matrix`
out of the capture zarr, segments each session into signal/noise windows, beamforms across the
steering directions, computes per-window phase/amplitude statistics, and writes the compact
precomputed features (the `.yarr` + a `.pkl` of segment lists) consumed during training so the giant
raw signal zarr is never touched at train time.

---

## Call-graph overview (Pass 2)

```
v5spfdataset.get_segmentation        (spf_dataset.py:1652)   ── generate path
v5spfdataset.render_session          (spf_dataset.py:752)    ── realtime/temp path
segment_zarr.py::process_zarr        (segment_zarr.py:7, via v5spfdataset)
notebooks/benchmark_segmentation.py  (:44)
        │
        ▼
mp_segment_zarr (:107)  ── multiprocessing Pool over sessions, writes .yarr + .pkl
        │  (Pool.imap)
        ▼
segment_session_from_zarr_star (:356) → segment_session_from_zarr (:361)  [zarr read]
        │
        ▼
segment_session (:375)  ── detrend → beamform (GPU/CPU) → simple_segment → stats
        │                          │                          │
        │                          ▼                          ▼
        │         beamformer_given_steering_nomean_cp   simple_segment (:668)
        │         / _nomean / _nomean_fast (rf.py)              │
        │                                          ┌────────────┼───────────────┐
        │                                          ▼            ▼               ▼
        │                          get_all_windows_stats   combine_windows  recompute_stats_for_windows
        │                              (:640, @njit)        (:850)           (:906)
        │                                                  drop_windows_smaller_than (:832)
        │                                                  keep_signal_surrounded_by_noise (:801)
        │                                                  compute_downsampled_segmentation_mask (:937)
        ▼
mean_phase_from_simple_segmentation (:77)  ── written into .yarr `mean_phase`
```

`render_session` (`spf_dataset.py:752`) calls `segment_session` **directly** (no zarr/Pool), in-memory,
for realtime/temp-file use — a second live entry into `segment_session`.

---

### `mp_segment_zarr(...)` · `segmentation.py:107` · ✅ Active — PRIMARY ENTRY

**Summary.** The orchestrator. Opens the raw capture zarr, decides how many sessions to (re)process,
fans the per-session work across a `multiprocessing.Pool` (one pass per receiver, exactly 2
receivers), then stacks the per-session results into a freshly created (or reopened) `.yarr` zarr and
dumps the segment lists to a `.pkl`. Supports **incremental resume**: it counts how many sessions are
already computed and only processes the tail.

**Signature**
- Inputs: `zarr_fn` (str, raw capture zarr path); `results_fn` (str, output `.pkl` path — `.yarr`
  derived by `.replace(".pkl",".yarr")`); `steering_vectors_for_all_receivers` (list, per-receiver
  `[nthetas, n_antennas]` complex steering matrices); `precompute_to_idx` (int, −1 = all; else
  inclusive — `+=1` at `:182`); `gpu` (bool); `n_parallel` (int, 0 ⇒ serial `map`); `skip_beamformer`
  (bool); `temp_file` (bool, suppresses `zarr_shrink` and tqdm); `skip_detrend` (bool).
- Returns: int — number of sessions now computed (`already_computed` if nothing to do, else
  `precompute_to_idx`).
- Side effects: **zarr/LMDB read+write** (opens raw zarr `:139`; reopens existing `.yarr` `mode="rw"`
  `:155`; creates new `.yarr` `:257`; writes 6 datasets/receiver `:270-315`; closes store `:336`);
  **`pickle.load`/`pickle.dump`** via bare `open()` with **no context manager** (`:152`, `:344-350`);
  `os.makedirs` `:256`; `os.path.exists`; **spawns processes** (`Pool` `:216`); tqdm to stderr;
  `zarr_shrink` (in-place LMDB mutation) `:341`; **`assert len(z["receivers"]) == 2`** `:189`.

**Called from** (Pass 2)
- `v5spfdataset.get_segmentation` (`spf_dataset.py:1706`) → on generate/temp path; result discarded,
  then the `.pkl`/`.yarr` are reopened and read back (`:1719-1737`). Reachable from
  `v5spfdataset.__init__` and `segment_zarr.py` (CLI with `segment_if_not_exist=True`). ✅ live.
- `notebooks/benchmark_segmentation.py:44`. 🧪 notebook.
- Fan-in: 2 (one production, one notebook).

**Contract**
- *Preconditions:* `z["receivers"]` has **exactly 2** members (else `AssertionError` `:189`);
  steering list indexable `[0],[1]`; every session's `signal_matrix` length satisfies the windowing
  divisibility `(N − window_size) % stride == 0` (enforced downstream in `rf.py:661`); enough LMDB
  `map_size` (hard-coded `2**32` = 4 GiB at `:156`).
- *Postconditions:* `.yarr` datasets `r{0,1}/{all_windows_stats, windowed_beamformer,
  weighted_beamformer, weighted_windows_stats, downsampled_segmentation_mask, mean_phase}` filled for
  `[already_computed:precompute_to_idx]`; `.pkl` holds `{"version": SEGMENTATION_VERSION,
  "segmentation_by_receiver": {r0:[...], r1:[...]}}` of length `n_sessions` (prior + new + empty pad).
- *Invariants:* shapes of the new `.yarr` datasets are derived from **`results_by_receiver["r0"][0]`**
  (the first session of r0) `:241-252` — assumes all sessions/receivers share window-count shape.

**Tested by** — **none directly.** No test constructs `mp_segment_zarr`; only `simple_segment` is unit
tested (`tests/test_segmentation.py`). Exercised only transitively via `test_v5inference.py` /
`test_fake_data.py` end-to-end (⚠️ not confirmed to hit the resume branch).

**Analysis** (Pass 3)
- *Resume math (`already_computed`).* `:161-169` `already_computed = min over receivers of
  (sum over sessions where Σ all_windows_stats > 0)`. `inputs` spans
  `range(already_computed, precompute_to_idx)` `:211`; writes target `[already_computed:precompute_to_idx]`
  `:270-315`; the `.pkl` reuses `previous[...][:already_computed]` `:324`. **Self-consistent** ✅
  verified. **BUT** the "computed" test is `Σ stats > 0`: a session whose windows are all noise can
  still have negative `weighted_windows_stats` (placeholder `[-1,-1,-1]` `:613`) yet **positive**
  `all_windows_stats` (amplitudes/stddevs are ≥0), so this generally works; the hazard is a genuine
  all-zero session being reprocessed forever, or a receiver-skew where r0 has more done than r1 (the
  `min` masks it) → see Issue S2. 🔍 analysis.
- *2-receiver hard assert* `:189` — any non-2-receiver capture raises `AssertionError`. Matches the
  repo-wide hardcoded 2-RX assumption; reachable only with such a capture. ✅ verified, low sev.
- *Shape from first result* `:241-252` — if session 0 of r0 has a different window count than later
  sessions (variable-length captures), `np.vstack` `:272` raises. In practice `signal_matrix` length is
  fixed by `buffer_size`, so uniform. 🔍 analysis, low sev.
- *File handles leak* `:152`, `:344-350` open pickles without `with`; on exception the fd leaks. Cosmetic.
- *`weighted_beamformer_shape`* `:247-249` is derived from `windowed_beamformer.shape[2:]` (note `[2:]`,
  not `[1:]`) — `windowed_beamformer` per result is `(1, n_windows, nthetas)` (singleton prepended
  `:632`), so `[2:]` = `(nthetas,)`, giving `(n_sessions, nthetas)`. Matches the per-session
  `weighted_beamformer` shape `(nthetas,)`. ✅ verified correct (the `[2:]` vs `[1:]` asymmetry is
  intentional, not a bug).
- *Float16 storage* — `all_windows_stats`/`windowed_beamformer` are stored float16 in the `.yarr`
  (`zarr_utils.py:56,70`); the per-session arrays are cast to float16 in `segment_session` (`:532`,
  `:570`,`:628`). See Issue S1 for the precision concern.

---

### `segment_session(v, gpu=True, skip_beamformer=False, skip_detrend=False, skip_segmentation=False, fast_beamformer=False, **kwrgs)` · `segmentation.py:375` · ✅ Active — CENTRAL

**Summary.** The per-session workhorse. Detrends the raw 2×N complex signal, runs beamforming over the
steering vectors (GPU via cupy or CPU via numba), runs `simple_segment` to find signal windows,
computes window stats, derives a session-level "weighted" beamformer/stat by masking to signal
windows, and prepends a singleton leading axis to every ndarray so results stack cleanly per session.

**Signature**
- Inputs: `v` (complex ndarray `(2, N)`, two antenna elements, **not** wrapped — raw IQ); `gpu` (bool);
  `skip_beamformer`/`skip_detrend`/`skip_segmentation` (bool); `fast_beamformer` (bool, picks the numba
  CPU kernel); `**kwrgs` must contain `steering_vectors` (`[nthetas, 2]` complex), `window_size`,
  `stride`, `trim`, `max_stddev_threshold`, `drop_less_than_size`, `min_abs_signal`
  (`DEFAULT_SEGMENT_ARGS` `:39`).
- Returns: dict with leading-singleton ndarrays: `windowed_beamformer` `(1, n_windows, nthetas)`,
  `all_windows_stats` `(1, 3, n_windows)`, `weighted_beamformer` `(1, nthetas)`,
  `weighted_windows_stats` `(1, 3)`, `downsampled_segmentation_mask` `(1, n_windows)`,
  `simple_segmentation` (a **list**, left un-prefixed since not an ndarray `:633`).
- Side effects: **GPU** (cupy host↔device transfers `:504-512` when `gpu=True`); `assert` in callees;
  float16 casts; **does NOT mutate the caller's `v`** — `detrend_np` returns a *new* `np.vstack`
  (`detrend.py:551-554`), and the `v = detrend_np(v)` rebinding `:415` is local. (Refines the inventory's
  "mutates `v`" claim: when `skip_detrend=False` the local rebinds; when `skip_detrend=True` `v` is read
  but never written. ✅ verified — no in-place mutation of the caller's array.)

**Called from** (Pass 2)
- `segment_session_from_zarr` (`:372`) ← Pool worker ← `mp_segment_zarr`. ✅ live (gpu flag threaded from
  `v5spfdataset.gpu`).
- `v5spfdataset.render_session` (`spf_dataset.py:752`) — direct, in-memory, `fast_beamformer=True`,
  `skip_segmentation=self.skip_segmentation`. ✅ live (temp-file / realtime path).
- `segment_single_session` (`:50`) — thin wrapper, **no in-repo callers**. 🧪.
- Fan-in: 2 live + 1 dead wrapper.

**Contract**
- *Preconditions:* `v.shape == (2, N)` (asserted in `simple_segment`/`get_all_windows_stats`
  `:710`,`:648`); `(N − window_size) % stride == 0` (asserted in `rf.py:661`) — **N must be
  window-aligned**; `steering_vectors.shape[0] == nthetas`.
- *Postconditions:* row 0 of `all_windows_stats` = trimmed circular mean phase (rad), row 1 = trimmed
  circular stddev, row 2 = median abs amplitude. `downsampled_segmentation_mask` is bool length
  `n_windows`. With no signal windows, `weighted_windows_stats = [-1,-1,-1]` `:613`.
- *Invariants:* `windowed_beamformer` and the mask share `n_windows`; the weighted beamformer
  `:576-583` divides by `mask.sum() + 0.001` so it is finite even when the mask is empty.

**Tested by** — none directly (only `simple_segment` is unit-tested). 🔴 coverage gap on the
GPU/CPU beamformer branches, the weighted-stat path, and float16 casts.

**Analysis** (Pass 3)
- *GPU vs CPU selection* `:496-527`. **GPU branch** wraps `beamformer_given_steering_nomean_cp` (which,
  despite the `_cp` name, is written with plain `np.dot`/`np.absolute` in `rf.py:1240-1270` —
  cupy-ness comes from the `cp.asarray(...)`/`cp.asnumpy(...)` *around* the call at `:504-508`; numpy
  dispatches on the cupy array type). **CPU branch** uses `beamformer_given_steering_nomean` (plain
  numpy) or `_fast` (numba). Both then `.reshape(nthetas, -1, window_size).mean(axis=2).T` → per-window
  mean magnitude. Result cast float16 `:532`. ✅ verified the two branches compute the same quantity.
- *cupy import is bare-`except`* `:9-12`. If cupy is absent and `gpu=True`, `cp` is **undefined** →
  `NameError` at `:504` inside a Pool worker. Reachable iff someone sets `gpu=True` on a CUDA-less host.
  See Issue S3. ✅ verified (path), 🔍 (reachability config-dependent).
- *`skip_segmentation` branch* `:614-629` computes only `all_windows_stats` (no mask/weighted/simple
  keys). `mp_segment_zarr` always writes `weighted_*`/`downsampled_*`/`simple_segmentation` `:284-333`,
  so calling `mp_segment_zarr` with a per-session `skip_segmentation=True` would `KeyError`. **But**
  `mp_segment_zarr` never sets `skip_segmentation` (not in `DEFAULT_SEGMENT_ARGS`, not threaded), so this
  is unreachable from the yarr-build path; only `render_session` uses it (and `render_session` reads via
  `data_from_precomputed`, not the missing keys). 🔍 analysis, low sev. See Issue S4.
- *Empty-mask weighted stats* `:585-613`: guarded by `if mask.sum() > 0`, else placeholder. The
  `trim_mean(..., 0.1)` on a possibly tiny signal-window set could still warn/NaN if exactly the trimmed
  fraction empties it, but `trim_mean` of ≥1 element is defined. 🔍 low.

---

### `simple_segment(v, window_size, stride, trim, max_stddev_threshold, drop_less_than_size, min_abs_signal)` · `segmentation.py:668` · ✅ Active

**Summary.** The window-level segmentation algorithm. Computes per-window trimmed circular phase
stats, classifies+merges windows into signal/noise runs, drops tiny ones, keeps only signal runs
isolated by noise, recomputes stats on the survivors from the raw signal, and builds the downsampled
mask. The only unit-tested function in the module.

**Signature**
- Inputs as above; `trim` is a **percentage** (tests pass `trim=10` `:25`). `min_abs_signal` =
  amplitude threshold (default 40 `:46`). `max_stddev_threshold` = phase-stddev ceiling (default 0.5).
- Returns: `{"simple_segmentation": list[dict], "downsampled_segmentation_mask": bool ndarray
  (n_windows,), "all_windows_stats": ndarray (n_windows, 3)}`.
- Side effects: none on `v`; **mutates the per-window dicts** it builds (adds `type`, overwrites
  `mean`/`stddev`/`abs_signal_median` in `recompute_stats_for_windows`).

**Called from** (Pass 2)
- `segment_session` (`:552`). ✅ live.
- `tests/test_segmentation.py:22,67`; `tests/test_beamformer.py:4` (import). ✅ tested.
- Fan-in: 1 prod + 2 tests.

**Contract**
- *Preconditions:* `v.shape == (2, N)`, `(N − window_size) % stride == 0`.
- *Postconditions:* `simple_segmentation` entries carry `start_idx`/`end_idx` (sample units),
  `mean`/`stddev`/`abs_signal_median`, `type`. `n_windows = (N − window_size)//stride + 1`.
- *Invariants:* `downsampled_segmentation_mask` length equals `all_windows_stats.shape[0]`.

**Tested by** — `test_simple_segment` (`:5`) and `test_simple_segment_separate` (`:50`): inject 3
strong phase-coherent regions into a 600/400-sample random matrix (`window_size=100, stride=50,
trim=10`), assert recovered `start_idx`/`end_idx`/`mean`. Happy path only.

**Analysis** (Pass 3)
- *Window pipeline order* `:726-774`: stats → `combine_windows` → `drop_windows_smaller_than` →
  `keep_signal_surrounded_by_noise` → `recompute_stats_for_windows` → mask. ✅ verified consistent.
- *`combine_windows` merge keeps first window's stats* `:889` (only `end_idx` is extended; `mean`/`stddev`
  stay from the head window). This is **intentional** because `recompute_stats_for_windows` `:906`
  recomputes them from the raw signal afterward. ✅ verified — not a bug (the merge-time `mean` is
  scratch).
- *`keep_signal_surrounded_by_noise`* `:801-829` drops any signal window adjacent to another signal
  window — so two abutting genuine signal runs that `combine_windows` did **not** merge (because phase
  differed >0.2 or stddev >0.1, `:885-886`) are **both dropped**. The test `test_simple_segment`
  comments out the `end_idx` assertion for the second window (`:42`) — evidence the maintainers know the
  merge/keep interaction is touchy. 🔍 analysis: a real two-emitter or phase-stepping capture can lose
  legitimate windows. See Issue S5. **⚠️ silently-wrong-result** (fewer signal windows, no crash).
- *`compute_downsampled_segmentation_mask`* `:937-963` uses `start_idx//window_size :
  end_idx//window_size`. Because segments come from window boundaries (`start`/`end` are multiples of
  `stride`), but the **mask index uses `window_size`** not `stride`: if `stride != window_size` the
  integer-division mapping from sample index → window index is **wrong**. In the production config
  `stride == window_size == 2048` (`:40-41`) so it's correct; in the **tests** `stride=50 ≠
  window_size=100`, the mask is mis-indexed — but the tests only assert on `simple_segmentation`, not the
  mask, so they pass. See Issue S6. ✅ verified (math), 🔍 (reachability: only if a config sets
  `stride != window_size`; the checked-in default does not).
- *Off-by-one in windowing* — `n_windows = 1 + (N−window_size)//stride` (`rf.py:662`); `step*stride`
  start, `+window_size` end. The trailing `N − ((n_windows−1)*stride + window_size)` samples
  (`< stride`) are **silently dropped**. With `stride==window_size` and N a multiple, nothing is lost.
  ✅ verified, expected behavior.

---

### `get_all_windows_stats(v, window_size, stride, trim)` · `segmentation.py:640` · ✅ Active (`@njit`)

**Summary.** Numba-JIT wrapper: phase-diff the signal, then call the fast windowed circular
mean/stddev/median kernel. Returns `(step_idxs (n_windows,2), step_stats (n_windows,3))`.

- Side effects: `assert v.ndim==2 and v.shape[0]==2` `:648`; JIT compile on first call.
- Note `:661-664`: a second kernel (`_fast2`) and a `breakpoint()` are **commented out** — dead, no
  effect, but a leftover-debug smell.
- Called from `simple_segment` `:726` and `segment_session` `skip_segmentation` branch `:615`. ✅ live.
- Contract: requires `(N − window_size) % stride == 0` (asserted in the kernel, `rf.py:661`).

---

### `mean_phase_from_simple_segmentation(segmentation)` · `segmentation.py:77` · ✅ Active

**Summary.** Per-session signal-strength-weighted **circular** mean phase. Weight =
`segment_length × abs_signal_median / (stddev + 1e-6)`. Returns NaN (`torch.nan`) for sessions with no
signal windows. Written into the `.yarr` `mean_phase` field (`mp_segment_zarr:313`) and used by
`render_session` (`spf_dataset.py:775`).

- Side effects: none (builds numpy arrays; `mean_phase_mean` from `rf.py`).
- Output: 1-D ndarray, one value per session, units **radians** ([-π,π] from circular mean).
- *Analysis:* `np.hstack([torch.nan, ...])` `:97` mixes a torch scalar into numpy — works (torch.nan is
  a Python float) but stylistically odd. Empty-signal sessions correctly become NaN; downstream
  (`spf_dataset.py:784`) masks NaN means. ✅ verified.

---

### Leaf helpers (Pass 3, brief)

| Function | Line | Role | Note |
|---|---|---|---|
| `segment_single_session` | 50 | wrapper w/ `DEFAULT_SEGMENT_ARGS` | 🧪 no in-repo caller |
| `segment_session_from_zarr_star` | 356 | Pool arg-unpack helper | ✅ pickled & dispatched by `imap` |
| `segment_session_from_zarr` | 361 | zarr-read one session → `segment_session` | ✅ opens via context mgr `:367`, casts `complex64` |
| `drop_noise_windows` | 784 | keep only `type=="signal"` | 🧪 **unused** — no caller |
| `keep_signal_surrounded_by_noise` | 801 | drop signal adjacent to signal | ✅ used; see Issue S5 |
| `drop_windows_smaller_than` | 832 | size filter (sample units) | ✅ used |
| `combine_windows` | 850 | classify+merge runs; **mutates input dicts** | ✅ used; merge-mean is scratch (recomputed) |
| `recompute_stats_for_windows` | 906 | re-stat survivors from raw `v`; **mutates input dicts** | ✅ used |
| `compute_downsampled_segmentation_mask` | 937 | window-level bool mask | ✅ used; `//window_size` not `//stride` (Issue S6) |
| `beamform_signal_cpu` | 1025 | intended CPU beamformer | 🔴 **STUB** — body is `# ... existing code ...`, returns `None` (KNOWN_ISSUES #9) |

---

## NEW / refined issues (Pass 3 bug hunt)

Severity = misbehavior × reachability. None of these crash the *default* production config; most are
silently-wrong or config-gated.

| ID | Issue | Location | Verdict | Confidence |
|---|---|---|---|---|
| S1 | **float16 precision loss in cached window stats.** `all_windows_stats` (phase mean/stddev rad, amplitude) cast to **float16** before write (`:570`,`:628`; dtype `zarr_utils.py:56`). float16 has ~3 decimal digits / ~9.8e-4 rad resolution near 1.0 — fine for phase but **lossy for large amplitudes** (>2048 loses integer precision) which feed `min_abs_signal=40` thresholds only at read time. The amplitude median can saturate (`inf` above 65504). | `:570`, `:628`, `zarr_utils.py:56` | **P2** — degraded precision, no crash; default thresholds tolerate it. ⚠️ no-crash / verify downstream before any "fix" (downstream may rely on f16 binning). | ✅ verified |
| S2 | **Resume `already_computed` uses `Σ all_windows_stats > 0` and `min` across receivers.** A genuinely all-zero/NaN-stat session (e.g. dead capture) is treated as "not computed" and reprocessed every run; receiver skew is hidden by `min` so a partially-written `.yarr` can silently re-run only the tail. | `:161-169` | **P2** — wasteful/idempotency edge, not wrong output. | 🔍 analysis |
| S3 | **`gpu=True` without cupy ⇒ `NameError` in Pool worker.** `import cupy as cp` guarded by bare `except: pass` (`:9-12`); `cp.asarray` at `:504-507` is then undefined. Crash surfaces inside a child process (opaque traceback). | `:9-12`, `:504` | **P1 iff** a CUDA-less host runs with `gpu=True` (config-reachable: `v5spfdataset(gpu=True)` / `segment_zarr --gpu`); otherwise dead. | ✅ verified (path); 🔍 (reachability) |
| S4 | **`segment_session(skip_segmentation=True)` omits keys `mp_segment_zarr` later reads.** The skip branch (`:614-629`) does not set `weighted_beamformer`/`downsampled_segmentation_mask`/`simple_segmentation`; `mp_segment_zarr` unconditionally reads them (`:284-333`) ⇒ `KeyError`. **Unreachable today** because `mp_segment_zarr` never passes `skip_segmentation`. Latent landmine if the flag is ever threaded. | `:614-629` vs `:284` | **P2** — dead path. | ✅ verified (structure), 🔍 (unreached) |
| S5 | **`keep_signal_surrounded_by_noise` drops legitimate abutting signal runs** that `combine_windows` declined to merge (phase Δ>0.2 or stddev Δ>0.1). Two adjacent real signal segments both vanish from `simple_segmentation` → fewer/no signal windows → wrong `mean_phase`/empty mask, **no crash**. The test's commented-out `end_idx` assert (`test_segmentation.py:42`) is corroborating evidence. | `:801-829` (+`:882-902`) | **P2** (P1 for multi-emitter/phase-stepping captures) — silently-wrong segmentation. ⚠️ no-crash / verify downstream before any fix. | 🔍 analysis |
| S6 | **`compute_downsampled_segmentation_mask` indexes by `//window_size` but segments are strided.** Correct only when `stride == window_size` (the checked-in default 2048/2048). Any config with `stride != window_size` (e.g. overlapping windows, as in the unit tests) yields a mis-aligned mask; tests don't assert the mask so the bug is invisible there. | `:960-962` | **P2** — wrong mask, but only under a non-default `stride != window_size` config. ⚠️ no-crash. | ✅ verified (math); 🔍 (reachability) |
| S7 | **Pickles opened without context managers** (`:152`, `:344-350`); fd leak on exception; also `O(n²)` warning self-noted at `:151`. | `:152`, `:344` | **P2** — hygiene. | ✅ verified |
| S8 | **Leftover commented `breakpoint()` + dead second-kernel call** in `get_all_windows_stats` `:661-664`. No runtime effect (commented). | `:661-664` | **P2** — hygiene/WIP smell. | ✅ verified |

### KNOWN_ISSUES #9 — `beamform_signal_cpu` stub — **CONFIRMED** ✅ verified

`segmentation.py:1025-1040`: the function has a full docstring but its body is literally
`# ... existing code ...` (`:1040`) and nothing else — it returns `None`. **However its reachability is
nil:** a repo-wide grep finds **no callers** of `beamform_signal_cpu` anywhere (it is the last symbol in
the file and is never imported). The actual CPU beamforming used by `segment_session` is
`beamformer_given_steering_nomean` / `_nomean_fast` from `spf/rf.py` (`:515-527`), which are
fully implemented. So #9 is correct that the function is a stub, but it should be tagged **🔴 dead stub
(no callers)** — severity **P2**, not a live break. The "reported stub / KNOWN_ISSUES #9" framing in the
task is accurate; the CPU *fallback in practice* is `rf.py`'s `beamformer_given_steering_nomean`, which
is **not** broken.

---

## Module cluster summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `mp_segment_zarr` | ✅ Active | 1 prod + 1 nb | ❌ (indirect e2e only) | Primary `.yarr` writer; resume math OK but `Σ>0`/`min` idempotency edge (S2); 2-RX assert. |
| `segment_session` | ✅ Active | 2 prod | ❌ | GPU/CPU branches equivalent; `gpu=True` w/o cupy → NameError (S3); no `v` mutation (refined). |
| `simple_segment` | ✅ Active | 1 prod + 2 tests | ✅ happy path | `keep_signal_surrounded_by_noise` can drop real runs (S5); mask `//window_size` (S6). |
| `get_all_windows_stats` | ✅ Active (`@njit`) | 2 | ❌ | Commented `breakpoint()`/dead kernel (S8); requires window-aligned N. |
| `mean_phase_from_simple_segmentation` | ✅ Active | 2 (here + spf_dataset) | ❌ | NaN on no-signal; correct circular weighting. |
| `compute_downsampled_segmentation_mask` | ✅ Active | 1 | ❌ | `//window_size` mis-index when `stride≠window_size` (S6). |
| `combine_windows` / `recompute_stats_for_windows` | ✅ Active | 1 each | ❌ | Mutate input dicts; merge-time `mean` is scratch (recomputed) — not a bug. |
| `keep_signal_surrounded_by_noise` | ✅ Active | 1 | ❌ | Silently drops abutting signal runs (S5). |
| `drop_windows_smaller_than` | ✅ Active | 1 | ❌ | size filter (sample units). |
| `segment_session_from_zarr(_star)` | ✅ Active | Pool | ❌ | zarr read + complex64 cast; pickled across processes. |
| `segment_single_session` | 🧪 Unused | 0 | ❌ | dead wrapper. |
| `drop_noise_windows` | 🧪 Unused | 0 | ❌ | dead. |
| `beamform_signal_cpu` | 🔴 Stub (dead) | 0 | ❌ | KNOWN_ISSUES #9 confirmed; **no callers**, real CPU path is `rf.py`. |

**Cross-cutting themes.** (a) The whole module is **float16-lossy by design** for the cached stats/
beamformer (S1) — correctness rests on the assumption that f16 phase/amplitude resolution is "good
enough"; untested. (b) Several correctness properties **silently depend on `stride == window_size ==
2048`** (the mask indexing S6, the no-dropped-tail windowing) — the production config holds it, the
unit tests violate it but don't assert on the affected outputs, so the divergence is invisible. (c) The
segmentation *filtering* (`combine_windows` + `keep_signal_surrounded_by_noise`) is the real
silently-wrong-result surface (S5): it can throw away legitimate signal windows with no error, and a
maintainer already commented out a test assertion that this interaction breaks. (d) GPU/CPU are
selected per call and compute the same quantity, but `gpu=True` on a cupy-less host fails opaquely
inside a Pool child (S3). (e) Only `simple_segment` is unit-tested; the entire `.yarr`-writing path,
the beamformer branches, the resume logic, and the float16 casts have **no direct coverage**.
