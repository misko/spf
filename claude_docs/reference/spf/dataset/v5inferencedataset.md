# `v5inferencedataset` — realtime threaded inference dataset

Reference for the `v5inferencedataset` class in
[`spf/dataset/spf_dataset.py:448`](../../../../spf/dataset/spf_dataset.py).
Produced by a Pass 1→3 deep dive per
[`DOCUMENTATION_GUIDE.md`](../../../DOCUMENTATION_GUIDE.md). **Read-only audit — no code changed.**
Line numbers verified against source at documentation time.

Units/frames: angles are radians unless a name says `_in_pis` (units of π);
`rx_wavelength_spacing` = d/λ (dimensionless). See
[`conventions.md`](../../00_concepts/conventions.md).

---

## What it is

A `torch.utils.data.Dataset` subclass that serves **live, streaming** RF captures for realtime
inference (rover / drone MAVLink collection, rpi5 inference), as opposed to the on-disk
[`v5spfdataset`](../../../reference/_inventory/datasets.md). It is a **producer/consumer buffer**:

- **Producers** (radio read threads) call [`write_to_idx`](#write_to_idxself-idx-ridx-raw) →
  [`render_session`](#render_sessionself-ridx-data) (heavy per-capture feature extraction, run on
  the *calling* thread) → push `(idx, ridx, rendered_data)` onto a `multiprocessing.Queue`.
- A single **background reader thread** ([`_reader_loop`](#_reader_loopself)) drains the queue into
  `self.store`, a `{idx: {"count": int, "data": [r0, r1]}}` dict, evicting the oldest entries when
  `max_store_size` is exceeded.
- A **consumer** (the inference loop, via `__next__`/`__getitem__`) busy-waits until both receivers
  (`count == 2`) are present at the requested `idx`, then returns `[r0_data, r1_data]`.

`paired=True` is asserted (`spf_dataset.py:508`): each served item is always a 2-element list.

---

## Concurrency model

**One reader thread, N producer threads, one consumer thread; one queue, one dict, one lock.**

- **Producers → consumer handoff** is a `multiprocessing.Queue` (`incoming_queue`,
  `spf_dataset.py:493`). Despite the `multiprocessing` type, everything runs in **one process,
  multiple threads** — the reader is a `threading.Thread` (`spf_dataset.py:606`), and producers are
  the radio read threads driven by a `ThreadPoolExecutor(max_workers=2)` in
  [`DataCollector.run_inner_collector_thread`](../../../../spf/data_collector.py) (`data_collector.py:452-467`).
- **Shared mutable state:** `self.store` (dict), `self.min_idx_stored` (int). Both are written by
  the reader thread inside `with self.lock` (`spf_dataset.py:618-633`).
- **`self.lock`** is a `multiprocessing.Lock` (`spf_dataset.py:490`). It guards the reader's
  store-write/eviction block and the consumer's *final* `return self.store[idx]["data"]`
  (`spf_dataset.py:677-678`).
- **`self.condition`** is a `multiprocessing.Condition` (`spf_dataset.py:489`) that is **created but
  never used** — every `condition.wait()`/notify is commented out (`spf_dataset.py:668,671`). The
  consumer instead **busy-waits** with `time.sleep(0.02)` (`spf_dataset.py:673`).
- **Thread lifecycle:** the reader thread is started in `__init__` (`spf_dataset.py:609`) and is
  **not a daemon** — `daemon=True` is commented out (`spf_dataset.py:608`). It is stopped by
  `close()` setting `stop_event` and `join(timeout=1.0)` (`spf_dataset.py:806-808`). The reader loop
  polls the queue with `timeout=0.1` so it notices `stop_event` within ~100 ms.

**Lock discipline gaps (verified):**

| Access | Locked? | Evidence |
|---|---|---|
| reader writes `store`, runs eviction, sets `min_idx_stored` | ✅ yes | `spf_dataset.py:618-633` |
| consumer reads `store[idx]` / `store[idx]["count"]` in wait loop | ❌ **no** | `spf_dataset.py:670` |
| consumer final read `store[idx]["data"]` | ✅ yes | `spf_dataset.py:677-678` |
| consumer reads `min_idx_stored` for bounds check | ❌ no | `spf_dataset.py:665` |

The unlocked reads in the wait loop (`spf_dataset.py:670`) are the central hazard: they race against
the reader's `store.pop(idx)` eviction (`spf_dataset.py:632`). See [Issue R1](#race-r1).

---

## Methods

### `__init__(self, yaml_fn, nthetas, ...)` · `spf_dataset.py:449` · ⚠️ Active

**Summary.** Parses the receiver YAML, validates that all receivers share spacing/wavelength/
bandwidth/device-type, precomputes per-receiver steering vectors, optionally loads an empirical
phase→angle distribution pickle, builds the queue/lock/store, and **starts the background reader
thread**. After construction the object is immediately live and accepting `write_to_idx` calls.

**Signature (selected inputs)**
- `yaml_fn` (str) — receiver config path; parsed via `load_config` (`spf_dataset.py:516`).
- `nthetas` (int) — beamforming theta-discretization; also steering-vector `spacing`
  (`spf_dataset.py:580`).
- `paired` (bool) — **asserted True** (`spf_dataset.py:508`).
- `gpu` (bool) — passed to `segment_session` in `render_session`.
- `skip_fields` (List[str], **mutable default `[]`**) — fields to drop. Stored by reference and
  **mutated in place** (see Issue I1).
- `empirical_data_fn` (str | None) — pickle of empirical dists; loaded eagerly with
  `pickle.load(open(...))` (file handle not closed) (`spf_dataset.py:596`).
- `skip_segmentation` (bool, default **True**) — when True, appends `segmentation_based_keys` to
  `skip_fields` (`spf_dataset.py:505-506`).
- `max_store_size` (default `None`) — eviction threshold for the reader; `None` ⇒ **no eviction**
  (unbounded store).
- `max_in_memory` (default 10), `min_idx` (set 0), `realtime`, `v4`, `target_dtype`,
  `distance_normalization`, `windows_per_snapshot`, `skip_detrend`, `vehicle_type`, `target_ntheta`,
  `model_config_fn`, `empirical_individual_radio`, `empirical_symmetry`.
- Returns: None (constructor).
- **Side effects:** mutates caller's `skip_fields` list (`+=`, `spf_dataset.py:506`); reads YAML;
  opens + reads empirical pickle (leaked FD); creates `multiprocessing` Queue/Lock/Condition;
  **spawns a non-daemon `threading.Thread`** (`spf_dataset.py:606-609`); many `assert`s on receiver
  consistency (`spf_dataset.py:540-551,567-569`).

**Notable vestigial state:** `self.condition` (`:489`), `self.min_idx`/`max_in_memory` (`:487-488`),
`self.precomputed_entries`/`precomputed_zarr` (`:511-514`) are set but never read on any active
inference path (grep-confirmed; `precomputed_*` are only used by `v5spfdataset`, not this class).

---

### `_reader_loop(self)` · `spf_dataset.py:611` · ⚠️ Active

**Summary.** The background-thread body. Loops until `stop_event` is set: pulls
`(idx, ridx, rendered_data)` off `incoming_queue` (100 ms timeout), and under `self.lock` inserts it
into `self.store`, bumps `count`, runs eviction if over `max_store_size`, and recomputes
`min_idx_stored = min(self.store.keys())`.

**Signature**
- Inputs: `self`. Returns: None (runs until stop).
- **Side effects:** mutates `self.store`, `self.min_idx_stored`; consumes the queue; prints `"RET"`
  / `"RET2"` (`spf_dataset.py:616,645`); `logging.exception` on unexpected errors
  (`spf_dataset.py:642`).

**Control flow / exit paths (Pass 3):**
- Normal: `stop_event` set → loop condition false → falls through to `print("RET2")` (`:645`).
- `stop_event` observed right after a successful `get` → `print("RET"); return` (`:615-617`).
- `queue.Empty` (timeout) → `pass`, keep looping (`:635-636`).
- `OSError`/`EOFError` (queue closed at process exit) → `break` silently (`:637-639`).
- Any other `Exception` → log (unless stopping) and `break` (`:640-644`). **A single malformed
  payload or a `render_session` artifact that breaks insertion kills the reader permanently** — there
  is no restart; the consumer then busy-waits to its 30 s timeout forever after.

**Issues found:**

- <a id="bug-shadow"></a>**I2 — loop-variable shadowing in the locked eviction region**
  (`spf_dataset.py:619-633`, severity 🟠 P1, confidence high). The payload index is bound as `idx`
  at `:614`. The eviction loop rebinds the **same name**: `for idx in sorted(self.store.keys())[:
  -self.max_store_size]: self.store.pop(idx)` (`:631-632`). After eviction runs, `idx` no longer
  refers to the just-inserted item. Today this is *latent* (nothing after `:632` reads `idx` — the
  next statement is `min_idx_stored = min(self.store.keys())` at `:633`), so it does not currently
  misbehave, but it is a trap for any future edit that references `idx` after the eviction loop.
- <a id="bug-evict"></a>**I3 — eviction can drop a half-filled or actively-requested idx**
  (`spf_dataset.py:627-632`, severity 🟠 P1). Eviction is purely by `len(self.store)` and key order;
  it does **not** consider whether `count == 2` or whether the consumer is currently waiting on the
  evicted `idx`. With `max_store_size=3` (the production setting,
  `mavlink_radio_collection.py:282`), if producers run ahead of the consumer, the oldest idx is
  popped even if the consumer is mid-wait on it → that `__getitem__` either times out (returns None)
  or, if `min_idx_stored` advanced past it, the *next* call raises `ValueError` (`:665-666`).
- **I4 — slice `[:-max_store_size]`** keeps exactly the newest `max_store_size` keys
  (`sorted(...)[:-k]` are the oldest `len-k`). Correct for "evict oldest," verified.

---

### `__getitem__(self, idx, timeout=30.0)` · `spf_dataset.py:664` · ⚠️ Active

**Summary.** Blocks (busy-wait) until both receivers' rendered data for `idx` are in the store
(`count == 2`), then returns `[r0_data, r1_data]`. Returns **`None`** on timeout. Raises
`ValueError` if `idx` has already been evicted past (`idx < min_idx_stored`). The leading comment
(`spf_dataset.py:663`) states the hard assumption: **everything is requested in sequence.**

**Signature**
- Inputs: `idx` (int, monotonically increasing); `timeout` (float seconds, default 30.0).
- Returns: `list[2]` of per-receiver data dicts, **or `None`** on timeout (`spf_dataset.py:676`).
- **Side effects:** `time.sleep(0.02)` busy-wait (`:673`); prints `"waitinf to get get"` /
  `"ret waitinf to get get"` (`:669,675`); acquires `self.lock` only for the final read (`:677`);
  raises `ValueError` (`:666`).

**When it returns `None`:** only on the timeout path (`(time.time()-start) > timeout`,
`:674-676`). It never raises `StopIteration` and never returns `None` for a "valid but empty"
reason — `None` strictly means "data did not arrive within `timeout`."

**Issues found:**

- <a id="race-r1"></a>**R1 — unlocked store read races eviction** (`spf_dataset.py:670`, severity 🟠
  P1, confidence high). The wait condition `idx not in self.store or self.store[idx]["count"] != 2`
  reads the dict **without the lock**, concurrently with the reader's `self.store.pop(idx)` and
  inserts under lock (`:618-632`). Two concrete failure modes:
  1. **TOCTOU between check and locked return.** The loop can observe `count == 2` and exit
     (`:670`), then the reader evicts `idx` before the consumer acquires the lock and does
     `self.store[idx]["data"]` (`:677-678`) → **`KeyError`** (unhandled, propagates out of
     `__getitem__`). Narrow window but real under `max_store_size=3` + fast producers.
  2. **Dict mutated during membership test.** CPython dict reads are individually atomic under the
     GIL, so `idx in self.store` won't corrupt, but the compound read-then-read across two
     statements is not atomic; (1) is the materialization of that.
- **R2 — busy-wait, no backpressure** (`:673`). Polls every 20 ms; the `multiprocessing.Condition`
  that would replace this is dead code (`:489,668,671`). CPU-cheap but adds up-to-20 ms latency per
  sample and the producer→reader→store path has no signalling, so a slow consumer cannot slow
  producers — it just loses data to eviction (see I3).
- **R3 — `min_idx_stored` read unlocked** (`:665`) against the reader's write (`:633`). Benign on
  CPython (atomic int read) but means the bounds check can use a slightly stale value.

---

### `__next__(self)` / `__iter__(self)` / `__len__(self)` · `spf_dataset.py:654/650/647` · ⚠️ Active

- `__iter__` resets `self.serving_idx = 0`, returns `self` (`:650-652`).
- `__next__` loops `sample = self[self.serving_idx]` until non-None, logging a warning each empty
  spin (`:654-661`), then advances `serving_idx`. **Never raises `StopIteration`** — the iterator is
  unbounded; it relies on the consumer breaking out externally (e.g. the test breaks at `len(ds)`,
  `test_v5inference.py:188-189`). Because `__getitem__` returns `None` only on a 30 s timeout, an
  empty stream means `__next__` blocks ~30 s, logs, and retries forever.
- `__len__` returns `self.serving_idx`, i.e. the **current serving position, not an item count**
  (`:647-648`) — a `DataLoader` using `len()` would be wrong; this class is meant to be iterated,
  not indexed by a sampler.

---

### `write_to_idx(self, idx, ridx, raw)` · `spf_dataset.py:680` · ✅ Active

**Summary.** Producer entry point. Renders one raw radio capture into a full feature dict on the
**calling thread** (heavy compute), then enqueues `(idx, ridx, rendered_data)`.

- Inputs: `idx` (int session/record index), `ridx` (int receiver index 0/1), `raw` (dict of raw
  radio fields).
- Returns: None.
- **Side effects:** calls `render_session` (GPU-capable, segmentation, beamforming); `put` onto the
  `multiprocessing.Queue` (`:683`). The expensive work runs on the producer thread by design (so the
  reader stays light); confirmed by the comment at `:681`.

**Called from:** `DroneDataCollectorRaw.write_to_record_matrix` (`data_collector.py:547`) inside a
`ThreadPoolExecutor(max_workers=2)` (`data_collector.py:452`); tests (`test_v5inference.py:66,166`;
`test_particle_filter.py:270`); scratch scripts (`debug.py`, `debug3.py`,
`notebooks/benchmark_inference.py`).

---

### `render_session(self, ridx, data)` · `spf_dataset.py:686` · ✅ Active

**Summary.** Converts one raw radio capture into the full per-receiver feature dict the model
expects: tensorizes scalars, computes `abs_signal_and_phase_diff`, runs `segment_session`
(detrend + beamform + segment), merges precomputed features via `data_from_precomputed`, optionally
computes empirical distributions, bins targets, and casts floats to `target_dtype`.

- Inputs: `ridx` (int), `data` (dict, raw capture).
- Returns: the same `data` dict, **mutated in place** with ~25 added keys.
- **Side effects:** mutates `data`; GPU if `self.gpu` (`segment_session`, `:752-763`); sets all GT
  fields (`ground_truth_theta`/`phi`, `craft_*`, `y_rad`) to `±inf` since ground truth is unknown at
  inference (`:699-707`); **zeroes `rx_pos_mm`/`tx_pos_mm`** (`:739`).
- **Documented caveats (in source):** `"WARNING this does not respect flipping!"` for
  `abs_signal_and_phase_diff` (`:716`); the `rx_pos_mm`/`tx_pos_mm` vstack is commented out and
  replaced with zeros (`:725-739`) — positions are not propagated to inference.

This is the largest method and is the per-capture correctness surface, but it is **not the
concurrency target**; it runs entirely on the producer thread before anything is enqueued.

---

### `close(self)` / `__enter__` / `__exit__` · `spf_dataset.py:804/810/813` · ✅ Active

- `close()` sets `stop_event`, then `reader_thread.join(timeout=1.0)` (`:806-808`). Prints
  `"CLOSE2"`/`"CLOSE3"`.
- `__exit__` prints `"CLOSE"` and calls `close()` (`:813-815`). `__enter__` returns `self`.

**Issues found:**

- <a id="bug-close"></a>**L1 — bounded join can leak a non-daemon thread** (`:608,808`, severity 🟠
  P1, confidence high). The reader is **not a daemon** (`daemon=True` commented out, `:608`), yet
  `join` is bounded to 1.0 s. The reader's queue poll uses a 0.1 s timeout, so 1 s is normally
  enough — but `close()` does **not** drain or close `incoming_queue`. A `multiprocessing.Queue` has
  a feeder thread; if items are still buffered, the underlying pipe/feeder can keep the **interpreter
  from exiting** even after `join` returns (the classic "process hangs on exit with a non-empty mp
  Queue"). Combined with the non-daemon reader, an un-joined reader (join timed out) becomes a
  lingering thread that blocks interpreter shutdown.
- **L2 — `close()` does not `cancel_join_thread()` / `close()` the queue**, nor join producers
  (producers are owned by `DataCollector`). If a producer is mid-`render_session` when `close()`
  runs, its later `put` lands on a queue whose reader has exited → item silently dropped (acceptable
  at shutdown).

---

## Producer/consumer contract (Pass 2)

- **Producers must call `write_to_idx(idx, ridx, raw)` with `idx` monotonically non-decreasing and
  each `idx` written for both `ridx ∈ {0,1}`** before the consumer requests it. The consumer
  (`__getitem__`) blocks on `count == 2`.
- **Consumer must request `idx` in increasing order** (comment `:663`) and must tolerate `None`
  (timeout) and `ValueError` (idx evicted). The production consumer is
  `single_example_realtime_inference(..., realtime_ds=v5inf)`
  (`mavlink_radio_collection.py` path; `test_v5inference.py:184`).
- **`max_store_size` couples producer/consumer rate.** With no backpressure, a consumer slower than
  producers by more than `max_store_size` records will have idxs evicted out from under it
  (`mavlink_radio_collection.py:282` comment: *"needs to process fast enough otherwise delayed"*).

---

## Tests (Pass 3)

- `tests/test_v5inference.py::test_preprocessing_equal` (`:37-71`) — feeds 20 idxs from an on-disk
  `v5spfdataset`, then asserts `v5inf[idx]` matches `ds[idx]` field-by-field. **Single-threaded
  producer** (a plain loop), so it exercises render/store/serve but **not** concurrent
  producer/consumer, **not** eviction (`max_store_size` defaults to None), **not** timeout/`None`,
  **not** `ValueError`.
- `test_v5inference.py::test_v5inference_with_nn` (`:134-191`) — writes all idxs, then iterates via
  `single_example_realtime_inference`. Still writes-then-reads (no overlap), no eviction.
- `test_particle_filter.py:252-270` — similar producer pattern.
- **Coverage gaps:** no test for (a) concurrent producers + consumer (the real
  `ThreadPoolExecutor(max_workers=2)` path), (b) eviction under `max_store_size`, (c) the
  `__getitem__` timeout→`None` path, (d) the `min_idx_stored` `ValueError` path, (e) the R1
  check-then-evict `KeyError` race, (f) `close()` thread-leak behavior, (g) reader-death on a bad
  payload. The hazardous concurrency paths are entirely untested.

---

## Cluster summary

| Method | Status | Concurrency role | Top finding |
|---|---|---|---|
| `__init__` | ⚠️ Active | spawns reader thread | I1 mutable-default `skip_fields += ` mutates caller's list (`:506`); non-daemon thread (`:608`); leaked pickle FD (`:596`) |
| `_reader_loop` | ⚠️ Active | single consumer of queue | I2 loop-var `idx` shadowing in locked region (`:631`); I3 eviction ignores `count`/in-flight requests; reader dies permanently on any unexpected exception (`:640-644`) |
| `__getitem__` | ⚠️ Active | consumer | **R1** unlocked store read races eviction → possible `KeyError` (`:670` vs `:632`); busy-wait (`:673`); returns `None` only on 30 s timeout |
| `__next__`/`__iter__`/`__len__` | ⚠️ Active | consumer driver | unbounded iterator, never `StopIteration`; `__len__` returns serving position, not count |
| `write_to_idx` | ✅ Active | producer entry | heavy render runs on producer thread (by design) |
| `render_session` | ✅ Active | producer (pre-enqueue) | mutates `data`; GT set to ±inf; positions zeroed; "does not respect flipping" |
| `close`/`__enter__`/`__exit__` | ✅ Active | shutdown | **L1** non-daemon reader + bounded join + undrained mp Queue → potential lingering thread / hang on exit |

**Cross-cutting theme.** The class is a hand-rolled single-reader/multi-producer buffer whose
**locking is incomplete**: the reader writes the store under `self.lock`, but the consumer's hot
wait loop reads it unlocked, so the only real synchronization for the consumer is the final 2-line
locked return — which is exactly where the check-then-evict `KeyError` race lives (R1). The
intended proper-signalling design (the `multiprocessing.Condition`) was abandoned and left as dead
code, replaced by a 20 ms busy-wait. Eviction is rate-coupling without backpressure, so a slow
consumer silently loses data. Several fields (`condition`, `min_idx`, `max_in_memory`,
`precomputed_*`) are vestigial. Debug `print`s remain in every hot method.

### Top hazards (ranked)

1. **R1 (race / crash):** `__getitem__` reads `self.store` unlocked in its wait loop (`:670`); the
   reader can `pop(idx)` (`:632`) between the `count == 2` check and the locked
   `self.store[idx]["data"]` return (`:677-678`) → unhandled **`KeyError`**. Triggered by the
   production config `max_store_size=3` with producers ahead of the consumer.
2. **L1 (lifecycle / leak):** non-daemon reader thread (`:608`) + `join(timeout=1.0)` (`:808`) +
   `incoming_queue` never closed/drained → a timed-out join leaves a live thread and a buffered mp
   Queue feeder that can **block interpreter exit**.
3. **I3 + I2 (eviction correctness):** eviction by `len(store)` alone (`:627-632`) drops oldest idxs
   regardless of `count == 2` or active waiters, and reuses the loop variable `idx` (`:631`)
   shadowing the just-enqueued index — a latent bug that will bite the next edit to that block.

### Cross-references

- KNOWN_ISSUES #11 (`skip_fields` mutable default `+=`, `:506`) — **confirmed** (Issue I1 above).
  Recommend the issue be broadened: this file adds the larger, untested-and-more-dangerous R1
  (KeyError race) and L1 (shutdown leak) hazards that #11 does not capture.
- KNOWN_ISSUES #7 (`spf_nn_dataset_wrapper` unconditional `breakpoint()` on the realtime annotate
  path) sits **downstream** of this dataset — the production consumer wraps `v5inferencedataset` in
  `v5spfdataset_nn_wrapper` (`mavlink_radio_collection.py:284`), so that breakpoint can halt the same
  realtime path this dataset feeds.
