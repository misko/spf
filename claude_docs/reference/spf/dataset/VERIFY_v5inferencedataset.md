# Adversarial verification — `v5inferencedataset` concurrency doc

Target doc: `claude_docs/reference/spf/dataset/v5inferencedataset.md`
Source: `spf/dataset/spf_dataset.py` (class at `:448`), producer in `spf/data_collector.py`.
Method: read exact `with self.lock:` scopes + thread lifecycle. Read-only.

## (a) Priority claims table

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| R1 | `__getitem__` wait loop reads `self.store[idx]` UNLOCKED (`:670`); reader can `pop(idx)` (`:632`) between the `count==2` check and the locked `return self.store[idx]["data"]` (`:677-678`) → unhandled `KeyError` | **CONFIRMED** | Wait loop `:670` `while idx not in self.store or self.store[idx]["count"] != 2:` is OUTSIDE any lock — preceding `with self.condition:` is commented (`:668`); the *only* `with self.lock:` in `__getitem__` is `:677`, wrapping `:678` `return self.store[idx]["data"]` alone. Reader's `pop`/insert are under `with self.lock:` at `:618`, covering `:619-633` incl. eviction `:631-632`. The two consumer statements (loop exit at `:670`, locked read at `:677-678`) are not jointly atomic → check-then-evict KeyError race is real. |
| L1 | Reader non-daemon (`daemon=True` commented ~:608); `join(timeout=1.0)` (~:808); mp Queue never closed/drained | **CONFIRMED (all 3)** | `:606-608` `threading.Thread(target=self._reader_loop)  # , daemon=True)` — daemon commented out, so default non-daemon. `:808` `self.reader_thread.join(timeout=1.0)`. `close()` `:804-808` sets `stop_event` + joins only; no `incoming_queue.close()`, `cancel_join_thread()`, or drain anywhere in `close`. |
| 3 | Eviction by `len(store)` only (`:627-632`), ignores half-filled (count<2)/awaited idxs; shadows enqueued loop var `idx` (`:631`) | **CONFIRMED** | `:627-630` condition is purely `len(self.store) > self.max_store_size`; no `count`/waiter check. `:631` `for idx in sorted(self.store.keys())[: -self.max_store_size]:` rebinds `idx` (bound at `:614`). Latent: nothing after `:632` reads `idx`; next stmt `:633` uses `min(self.store.keys())`. |
| 4 | `multiprocessing.Condition` is dead code | **CONFIRMED** | Created `:489`. Every use commented: `:668` `# with self.condition:`, `:671` `# self.condition.wait(0.01)`. No `notify`/`wait` anywhere active (grep clean). |
| 5a | Consumer busy-waits via `time.sleep(0.02)` on `count==2` | **CONFIRMED** | `:670-673` loop body sole active stmt is `:673` `time.sleep(0.02)`. |
| 5b | KNOWN_ISSUES #11: mutable-default `skip_fields=[]` mutated via `+=` (~:506) | **CONFIRMED** | Default `:456-458` `skip_fields: List[str] = []`. Stored by reference `:503` `self.skip_fields = skip_fields`. In-place mutation `:506` `self.skip_fields += segmentation_based_keys` under `if self.skip_segmentation:` (default True, `:23` of sig). `+=` on a list is `list.__iadd__` → mutates the caller's shared default object. |

## (b) Hit rate

Priority claims: **6/6 confirmed (100%)**, all with exact line evidence matching the doc.

Spot-check of ~10 ordinary doc claims:
- `:490` lock is `multiprocessing.Lock` — ✅
- `:493` `incoming_queue = multiprocessing.Queue()` — ✅
- `:508` `assert self.paired` — ✅ (note: default `paired=False` at sig `:6`; assert still fires unless caller passes True)
- `:609` reader started in `__init__` — ✅
- `:614` queue `get(timeout=0.1)` — ✅
- `:633` `min_idx_stored = min(self.store.keys())` under lock — ✅
- `:637-639` OSError/EOFError → break — ✅
- `:640-644` other Exception → log unless stopping, then break (reader dies permanently) — ✅
- `:647-648` `__len__` returns `self.serving_idx` — ✅
- `:665-666` `ValueError` when `min_idx_stored > idx` — ✅ (doc says `idx < min_idx_stored`; equivalent)
- `:596` `pickle.load(open(...))` leaked FD — ✅
- `:683` `incoming_queue.put(...)` in `write_to_idx` — ✅
- `data_collector.py:452-462` `ThreadPoolExecutor(max_workers=2)` driving `write_to_record_matrix` → `write_to_idx` (`:547`) — ✅ (it's a `ThreadPoolExecutorWithQueueSizeLimit`, a ThreadPool subclass, threads not processes — consistent with doc's "one process, multiple threads").

Ordinary spot-check hit rate: **~13/13 accurate.** Combined doc accuracy excellent.

## (c) Overstated / missed

**Overstated:** None material. The lock-discipline table (doc `:58-66`) is exactly right — I confirmed the consumer's final read IS locked (`:677-678`) and the wait-loop read is NOT (`:670`); the doc does not falsely claim a lock covers the wait-loop read. R1's stated window ("narrow but real") is honest, not inflated.

Minor nit: doc line refs `:618-633` / `:677-678` are spot-on; `_reader_loop` def is at `:611` (doc says `:611` ✅). One pedantic point — R1 mode (2) note that "CPython dict reads are individually atomic under the GIL" is correct framing; the real bug is the cross-statement TOCTOU, which the doc correctly identifies as the materialization.

**Missed (additions, not corrections):**
- The `:665` bounds check `self.min_idx_stored > idx` reads `min_idx_stored`, which is **only initialized inside the locked reader block at `:633`**. If `__getitem__` is called before the reader has ever inserted (store empty, `min_idx_stored` unset), this raises `AttributeError`, not the documented `ValueError`/`None`. The doc lists `min_idx_stored` as read-unlocked (R3) but does not flag the **uninitialized-attribute** hazard on a cold consumer. Minor, but a real additional crash path.
- The eviction at `:631-632` recomputes `min(self.store.keys())` at `:633` only inside the `with self.lock` block but unconditionally (not just when eviction ran) — fine, but means an empty store (impossible here since insert precedes it) would throw; not reachable. No correction needed.

## (d) Grade

**A**

The doc's concurrency analysis is precise and verifiable. All six priority hazards confirmed against exact lock scopes — critically, R1 (the headline KeyError race) holds up under careful reading: the wait-loop read is genuinely unlocked and the locked final read is a separate non-atomic statement, exactly as claimed. No hazard was overstated; the one lock that does cover a read (the final return) is correctly attributed. Lifecycle claims (L1 non-daemon + bounded join + undrained mp Queue) all verified line-by-line. Only additions are a cold-start `AttributeError` path the doc didn't enumerate.
