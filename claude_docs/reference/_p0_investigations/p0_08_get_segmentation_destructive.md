# P0 ISSUE #8 — `v5spfdataset.get_segmentation` destructive cache deletion on corrupt `.pkl`

File: `spf/dataset/spf_dataset.py`, method `get_segmentation` (def @ line 1652).

## 1. CONFIRM — read of the body

**(a) `os.remove` of the `.pkl` on `UnpicklingError`** — CONFIRMED. Lines 1753-1755:
```python
        except pickle.UnpicklingError:
            # Handle corrupted segmentation file by removing and regenerating
            os.remove(results_fn)
```
`results_fn` is the `.pkl` path (`results_fn()` @ 1636-1641 returns `..._segmentation_nthetas{N}.pkl`).

**(b) It then recurses** — CONFIRMED. Lines 1756-1760:
```python
            return self.get_segmentation(
                version=version,
                segment_if_not_exist=segment_if_not_exist,
                precompute_to_idx=precompute_to_idx,
            )
```

**(c) Recursion is bounded** — CONFIRMED. On re-entry the `.pkl` no longer exists, so the path branches deterministically:
- If `segment_if_not_exist=True`: lines 1691 `if self.temp_file or not os.path.exists(results_fn):` is now True → `mp_segment_zarr(...)` regenerates the file (lines 1706-1716), then the `try` at 1719 succeeds. Recovers. No further recursion.
- If `segment_if_not_exist=False` (and not `temp_file`): lines 1683-1688 fire first:
```python
        if (
            not segment_if_not_exist
            and not self.temp_file
            and not os.path.exists(results_fn)
        ):
            raise ValueError(f"Segmentation file does not exist for {results_fn}")
```
→ raises `ValueError` immediately. Recursion terminates. **No infinite loop in either mode** — the `.pkl` is gone, so the corrupt-pkl `try` block at 1722 cannot be re-reached a second time. Bound = at most one extra frame.

**(d) Corrupt `.yarr` at ~1732 is uncaught** — CONFIRMED. The companion zarr/LMDB store is opened at lines 1732-1737:
```python
            precomputed_zarr = zarr_open_from_lmdb_store(
                results_fn.replace(".pkl", ".yarr"),
                mode="r",
                map_size=2**32,
                readahead=False,
            )
```
This is inside the same `try` (1719) but the **only** handler is `except pickle.UnpicklingError` (1753). A corrupt/locked/truncated `.yarr` raises an LMDB/zarr error (e.g. `lmdb.Error`, `KeyError`, zarr exception), which is **not** caught → propagates uncaught. So `.yarr` corruption is neither recovered nor cleaned up. Also note `os.remove` deletes **only the `.pkl`**, never the `.yarr` — after a `.pkl`-triggered recovery the old `.yarr` is left orphaned (re-used or overwritten by `mp_segment_zarr`).

## 2. REACHABILITY — how often is the destructive path hit?

The destructive branch requires `pickle.load` (line 1722) to raise `pickle.UnpicklingError` specifically. Realistic causes:
- **Interrupted/partial write**: process killed (OOM, SIGKILL, node preemption) mid-`mp_segment_zarr` save → truncated `.pkl`. Plausible on a busy GPU box, but the `.pkl` is small (the bulky data lives in the `.yarr`), so the write window is short.
- **Concurrent writers**: two processes segmenting the same prefix into the same `precompute_cache` simultaneously could interleave writes. No lock file is used. Possible in parallel test/training runs sharing a cache dir.
- **Version mismatch**: does NOT raise `UnpicklingError` — a wrong-version pickle unpickles fine and is caught later by the version check at 1764 (`ValueError`), not by this handler.
- **Concurrent read while a writer truncates/replaces the file**: a reader catching the file mid-replacement could see a short read → `UnpicklingError` on otherwise-recoverable data → false positive that deletes a file another process is mid-writing.

Net: **rare** under normal single-process operation; the `.pkl` corruption window is small because the heavy artifact is the `.yarr`. The more concerning trigger is **concurrent access to a shared cache** (no locking), which can manufacture a transient `UnpicklingError`.

## 3. TRIGGER & DAMAGE — which caller mode dominates?

`grep segment_if_not_exist` across callers:
- **Training (`spf/scripts/train_single_point.py`): `segment_if_not_exist=False`** at lines 68, 199, 242, 266 — the destructive-then-refuse mode.
- **Inference / readonly listing (`spf/scripts/ls_ds.py`): `=False`** (line 25).
- Internal dataset construction passes `self.segment_if_not_exist`, whose default is **`False`** (constructor default @ 854, stored @ 879-880; used @ 1082, 1318). So library/dataloader default is `False`.
- `=True` callers: notebooks (`simple_train.py`, `simple_train_filter.py`), `fake_dataset.py`, segmentation scripts (`segment_zarr.py`), and essentially **all tests** (`tests/conftest.py`, `test_fake_data.py`, `test_v5inference.py`, `test_paired_rotate_dist.py`, `test_zarr_tools.py`) — these intentionally generate caches, so deletion+regen is the desired behavior there.

**The production read paths (training dataloaders, inference, ls) use `False`** — i.e. the dangerous "delete then raise `ValueError`, refusing to rebuild" mode. The `True` mode is mostly the cache-creation/test path that recovers cleanly.

## 4. BLAST RADIUS

The deleted artifact is only the small `.pkl` (`simple_segmentations` dict + version). The expensive, hours-of-compute beamforming output is the **`.yarr`**, which is **NOT** deleted by `os.remove`. So:
- Data loss from the `os.remove` itself is **the cheap `.pkl` only** — the `.yarr` survives.
- In `False` mode, after deleting the `.pkl` the method raises, so the dataset fails to load until someone re-runs segmentation (or the next `=True` run regenerates). The `.yarr` is orphaned but intact; a regen with `=True` rebuilds the `.pkl` (and overwrites the `.yarr`).
- No backup of the `.pkl` is taken before `os.remove`. No lock guards the cache.
- **False-positive risk is real**: a transient/concurrent short read can raise `UnpicklingError` on a genuinely-good (mid-write or just-being-read) `.pkl` → it gets nuked. Combined with `=False` callers, that converts a transient race into a hard failure requiring manual re-segmentation. Severity of the destruction is moderated by the fact that the truly precious `.yarr` is untouched.

## 5. REAL-WORLD — tests & history

- **No test exercises corrupt-`.pkl` recovery.** `grep -rn "UnpicklingError\|corrupt\|os.remove" tests/` → no hits. The recovery branch is entirely untested.
- Git history of the handler:
  - `cfc2be6` "add support to handle multiple segmentation versions" introduced `os.remove(results_fn)` + recursive `get_segmentation(precompute_to_idx=...)`.
  - `0fc9701` added forwarding `version=` then `segment_if_not_exist=` into the recursive call (so the `False` mode now propagates — which is exactly what makes the re-entry raise instead of regen).
  - `5e2c10e` only added the comment.
  The destructive line has existed for several commits; the `segment_if_not_exist` forwarding (which created the "delete-then-refuse" behavior in `False` mode) was added in `0fc9701`.

## 6. SEVERITY VERDICT — **DOWNGRADE (P0 → P2/P3)**

Justification:
- The auto-regen case (`=True`) recovers correctly and is bounded.
- In the `=False` production case the method does delete-then-raise — but it **only deletes the small `.pkl`, never the expensive `.yarr`**. The "hours of segmentation compute" claim does not hold: the heavy artifact survives `os.remove`. Recovery cost is one cheap re-segmentation of the `.pkl` (or a single `=True` run that also reuses/overwrites the `.yarr`).
- Recursion is bounded; no infinite loop.
- The genuine defects are (i) a too-broad/false-positive-prone delete triggered by concurrent or transient short reads with no locking, (ii) the `=False` mode destroying then refusing to rebuild (operationally annoying, not catastrophic since `.yarr` persists and regen is available), and (iii) an **uncaught corrupt-`.yarr`** at line 1732 that crashes loudly with no cleanup. These are correctness/robustness bugs, not catastrophic data loss. Per the prompt's own rubric ("data-loss-on-rare-corruption with auto-regen is lower"), and given the precious artifact is not deleted, this is **not a P0**. Recommend **P2** (concurrency false-positive can nuke a cache + uncaught `.yarr` path) — would be P3 if the cache were never shared across processes.

## 7. FIX SKETCH (describe only — do NOT apply)

1. **Make deletion non-destructive / verify before nuking.** Before `os.remove`, rename the suspect `.pkl` to `.pkl.corrupt-<pid>-<ts>` instead of deleting, so a false positive is recoverable. Only after a successful regen, drop the backup.
2. **Guard against false positives from concurrency.** Wrap the corrupt-recovery in a retry-with-backoff (re-read once before deciding it is corrupt) and/or take a file lock around read+regen on `results_fn` so concurrent writers cannot manufacture a transient `UnpicklingError`.
3. **Do not refuse-to-rebuild after deleting.** In the recursive re-entry, after the method itself deleted a corrupt cache, force regeneration even when `segment_if_not_exist=False` (the caller asked to load an existing cache; the method just proved it was unusable and is the party that removed it). E.g. pass an internal `force_regen=True` on the recursive call so the 1683-1688 guard does not fire. This makes `False` callers self-heal instead of hard-failing.
4. **Catch `.yarr` corruption too.** Broaden the `except` at 1753 (or add a sibling `except`) to also handle zarr/LMDB open failures at 1732, and clean up / quarantine the matching `.yarr` (currently never removed) so `.yarr` corruption recovers symmetrically with `.pkl` corruption.
5. **Bound recursion explicitly.** Add a `_retry` depth guard to the recursive call as defense-in-depth against any future path that could re-enter the corrupt branch.
