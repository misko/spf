# VERIFY — segmentation.py new consequential claims (S3, S5, S6)

Adversarial verification. Read-only. Source: `spf/dataset/segmentation.py` + callers.
Doc under test: `claude_docs/reference/spf/dataset/segmentation.md`.
Evidence gathered on branch `main` (HEAD `574c454`). cupy is **NOT installed** in this
environment (`python -c "import cupy"` → `ModuleNotFoundError`), which directly exercises S3.

## Verdict table

| ID | Claim (doc) | Verdict | Final severity | Confidence |
|---|---|---|---|---|
| S3 | `gpu=True` without cupy ⇒ `NameError` in Pool worker (`cp` undefined) | **CONFIRMED** (failure) / **PARTIAL** (reachability nuance) | **P2** (P1 only on the `v5spfdataset(gpu=True)` direct path on a CUDA-less host; the `segment_zarr.py` CLI path is blocked earlier by its own hard `import cupy`) | High |
| S5 | `keep_signal_surrounded_by_noise` silently drops abutting signal runs combine didn't merge | **CONFIRMED** | **P2** (P1 for multi-emitter / phase-stepping captures) — silent wrong segmentation, no crash | High (logic) / Medium (real-data trigger rate) |
| S6 | mask indexes by `//window_size` but segments are strided → correct only when `stride==window_size` | **CONFIRMED** (math) / **REFUTED as a live bug** | **P3** — no production/live caller sets `stride != window_size`; default 2048/2048 is correct | High |

---

## S3 — `gpu=True` without cupy ⇒ `NameError` inside Pool worker

**The failure path is REAL.** ✅

- Guarded import, `segmentation.py:9-12`:
  ```python
  try:
      import cupy as cp
  except:
      pass
  ```
  If cupy is absent, `cp` is **never bound** (bare `except: pass` swallows `ModuleNotFoundError`).
- Call site, `segmentation.py:504-508` (inside `segment_session`, `if gpu:` branch):
  ```python
  segmentation_results["windowed_beamformer"] = cp.asnumpy(
      beamformer_given_steering_nomean_cp(
          steering_vectors=cp.asarray(kwrgs["steering_vectors"]),
          signal_matrix=cp.asarray(v.astype(np.complex64)),
      ) ...
  ```
- Reproduced the exact error in this (cupy-less) env:
  ```
  try: import cupy as cp
  except: pass
  cp.asarray([1,2,3])   →  NameError: name 'cp' is not defined
  ```
- **Refinement of the doc**: the doc says it crashes "at `:504`". Precisely, the **first** `cp.*`
  evaluated is `cp.asarray(...)` (the argument expression at `:506`/`:507`), so the `NameError`
  fires there, before `beamformer_given_steering_nomean_cp` runs. Note that
  `beamformer_given_steering_nomean_cp` (`rf.py:1240-1270`) is itself written in **plain numpy**
  (`np.dot`/`np.absolute`) — the cupy dependency lives entirely in the `cp.asarray/asnumpy`
  wrappers at the call site. Doc already states this (`:174-179`); confirmed accurate.
- It executes inside a `multiprocessing.Pool` worker (`mp_segment_zarr:216-219` → `imap` →
  `segment_session_from_zarr_star` → `segment_session`), so the traceback surfaces from a child
  process. ✅ matches doc.

**Reachability — PARTIAL / nuanced (this is the part that matters):**

Two distinct live entries thread `gpu` into `segment_session`, and they differ:

1. `v5spfdataset(..., gpu=True)` → `get_segmentation` → `mp_segment_zarr(gpu=self.gpu)`
   (`spf_dataset.py:1711`) **and** `render_session` → `segment_session(gpu=self.gpu)`
   (`spf_dataset.py:752-754`). `v5spfdataset.__init__` default is `gpu=False`
   (`spf_dataset.py:455`, `:828`), so the danger requires a caller to explicitly pass `gpu=True`.
   This path imports `spf.dataset.spf_dataset`, which does **not** hard-import cupy, so it loads
   fine on a CUDA-less host and only fails *later* inside the worker. → **genuinely reachable
   NameError** if a user constructs `v5spfdataset(gpu=True)` on a host without cupy.

2. `spf/scripts/segment_zarr.py` CLI — `--gpu` **defaults to `True`** (`segment_zarr.py:31,42`),
   which superficially looks like the dangerous default the doc implies. **BUT** `segment_zarr.py`
   has a hard top-level `import cupy` (`segment_zarr.py:5`). On a CUDA-less host this script
   **fails at import time with `ModuleNotFoundError`**, long before `mp_segment_zarr` runs — so the
   in-worker `NameError` is *not* the symptom you'd see from the CLI; you'd get an import error
   instead. The CLI is effectively self-gating to cupy-present hosts.

**Net:** the `NameError`-in-worker is confirmed and reachable via the `v5spfdataset(gpu=True)`
object/`render_session` API on a cupy-less host, but it is **not** the default (`gpu=False` default
everywhere except the CLI flag, and the CLI is blocked earlier by its own import). Severity **P2**
overall; rises to **P1** only for the specific `v5spfdataset(gpu=True)` programmatic path on a
CUDA-less host. The doc's "P1 iff CUDA-less host runs with gpu=True" is accurate; I'd add the
caveat that the CLI's `--gpu` default never reaches the NameError because of the earlier hard import.

---

## S5 — `keep_signal_surrounded_by_noise` drops abutting signal runs

**CONFIRMED.** ✅ The logic, `segmentation.py:817-828`:
```python
for window_idx, window in enumerate(windows):
    if window["type"] == "signal":
        if window_idx > 0 and windows[window_idx - 1]["type"] == "signal":
            continue            # prev is signal → drop THIS window
        if (window_idx + 1 < len(windows)
                and windows[window_idx + 1]["type"] == "signal"):
            continue            # next is signal → drop THIS window
        valid_windows.append(window)
```
For two **adjacent signal runs** A,B (A before B): processing A, "next is signal" (B) → A dropped;
processing B, "prev is signal" (A) → B dropped. **Both are silently removed.** Confirmed by reading.

It is real signal that can be dropped (not noise): adjacent signal runs are only kept *separate*
(unmerged) when `combine_windows` (`:882-902`) refused to merge them, i.e. phase
`pi_norm(|Δmean|) >= 0.2` OR `|Δstddev| >= 0.1` (`:885-886`). That is exactly the situation of a
two-emitter capture or a source whose phase steps between adjacent windows — both legitimate.
So `keep_signal_surrounded_by_noise` can discard genuine signal windows.

Consequences (no crash): fewer/zero entries in `simple_segmentation` →
`compute_downsampled_segmentation_mask` gets fewer/no signal windows →
`downsampled_segmentation_mask` shrinks/empties → `mean_phase_from_simple_segmentation` and the
weighted stats are computed over fewer windows (or fall to the `[-1,-1,-1]` / NaN no-signal path,
`:611-613`, `:90-91`). This is a **silently-wrong result**.

Corroborating evidence the doc cites is real: `tests/test_segmentation.py:42` has a **commented-out
assertion** on the second window's `end_idx`:
```python
assert segmented_windows[1]["start_idx"] == ground_truth_windows[1]["start_idx"]
# assert segmented_windows[1]["end_idx"] == ground_truth_windows[2]["end_idx"]
```
This is in `test_simple_segment`, whose injected regions at samples 300–400 and 500–600 are close in
sample space relative to window_size=100/stride=50; the maintainer disabled the `end_idx` check
rather than fix the merge/keep interaction. Confirms the interaction is known-fragile.

**Severity: P2** in general (silent, depends on capture content), **P1** for multi-emitter or
phase-stepping captures where adjacent unmerged signal runs are common. Reachability: this filter is
called on **every** session via `simple_segment` (`segmentation.py:762`), which is on the live
`.yarr`-build path (`mp_segment_zarr`) and the realtime `render_session` path. So the *code* is
always reached; whether it *drops real signal* depends on data. ⚠️ no-crash — verify downstream
tolerance before any fix (the whole pipeline assumes this filtering).

---

## S6 — mask `//window_size` vs strided segments

**CONFIRMED (math) but REFUTED as a live/production bug.** `compute_downsampled_segmentation_mask`,
`segmentation.py:955-963`:
```python
seg_mask = np.zeros(n_windows).astype(bool)
for window in simple_segmentation:
    seg_mask[
        window["start_idx"] // window_size : window["end_idx"] // window_size,
    ] = True
```
`start_idx`/`end_idx` are sample indices that originate from window boundaries produced by the
windowing kernel, whose step is **`stride`** (`get_all_windows_stats` → `rf.py` windowing). The mask
maps sample→window-index by dividing by **`window_size`**. These agree only when
`stride == window_size`; otherwise the sample→window-index mapping is wrong (e.g. with overlapping
windows the divisor should reflect `stride`, not `window_size`).

**Reachability — no live path triggers it:**
- `DEFAULT_SEGMENT_ARGS` is the *only* source of `window_size`/`stride` reaching segmentation, and it
  hard-codes `window_size=2048, stride=2048` (`segmentation.py:40-41`). With `stride==window_size`,
  `//window_size` is correct. ✅
- Confirmed by grep: every production caller (`mp_segment_zarr` `:209`, `segment_single_session`
  `:72`, `render_session` `:761`, `get_segmentation` path) passes `**DEFAULT_SEGMENT_ARGS`
  unchanged. No code in `spf/` (outside tests and unrelated nn.Conv strides) passes a different
  `stride`/`window_size` into segmentation. There is no CLI/config knob to override them.
- The **unit tests** (`test_segmentation.py:23-30`, `:67-76`) do use `window_size=100, stride=50`
  (`stride != window_size`), so the mask IS mis-indexed there — but both tests assert only on
  `["simple_segmentation"]`, never on `downsampled_segmentation_mask`, so the mis-index is invisible
  and the tests pass. This confirms the doc's framing exactly.

**Severity: P3.** Latent correctness bug, but unreachable in any production/live config because the
default (and only) production stride equals window_size. Would become a real bug only if someone
introduced an overlapping-window config. Doc's "✅ verified (math); 🔍 (reachability)" is accurate;
I downgrade the practical severity since no live config reaches it.

---

## Summary

- **S3 — CONFIRMED (PARTIAL on reachability).** The `NameError` (`cp` undefined) is real and
  reproduced in this cupy-less env; it fires at the `cp.asarray` argument (`:506-507`) inside a Pool
  worker. Reachable via `v5spfdataset(gpu=True)`/`render_session` on a CUDA-less host. The
  `segment_zarr.py` CLI (whose `--gpu` defaults True) does **not** hit it — it fails earlier on its
  own top-level `import cupy`. Defaults are `gpu=False`. **Final: P2** (P1 only on the explicit
  `gpu=True` programmatic path without cupy).
- **S5 — CONFIRMED.** Two abutting signal runs that `combine_windows` left unmerged are **both**
  dropped by `keep_signal_surrounded_by_noise` → silently fewer/zero signal windows, wrong
  `mean_phase`/empty mask, no crash. A test assert (`test_segmentation.py:42`) was commented out
  over this exact interaction. **Final: P2** (P1 for multi-emitter/phase-stepping captures).
- **S6 — CONFIRMED (math) / REFUTED (live).** `//window_size` is wrong when `stride != window_size`,
  but the only production config is 2048/2048, and no live caller overrides it; only the unit tests
  use overlapping windows and they don't assert the mask. **Final: P3.**

**The one that matters most: S5.** It is the only one reachable on the **default production
configuration with default `gpu=False`** that can silently corrupt output (drop legitimate signal,
skew `mean_phase`) with no error. S3 needs a non-default `gpu=True` + missing cupy; S6 needs a
non-default `stride != window_size`. S5 fires on the default path whenever the data contains
adjacent-but-distinct signal runs.
