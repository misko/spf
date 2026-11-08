# P0 #7 — Unconditional `breakpoint()` in spf_nn_dataset_wrapper.py

File: `/home/mouse9911/gits/spf/spf/dataset/spf_nn_dataset_wrapper.py`

## 1. CONFIRM — both breakpoints are LIVE (unconditional, not commented)

`to_absolute_north` (:84-96):
```python
84  def to_absolute_north(self, sample):
85      for ridx in range(2):
86          breakpoint()          # <-- LIVE, first statement of loop body
87          ntheta = sample[ridx]["paired"].shape[-1]
```

`get_and_annotate_entry_at_idx`, realtime branch (:116-129):
```python
121          with torch.no_grad():
122              nn_inference = self.model(single_example)
124          for ridx in range(2):
125              sample[ridx].update({k: v[ridx] for k, v in nn_inference.items()})
126          breakpoint()          # <-- LIVE, after inference, before absolute conversion
127          if self.absolute:
128              sample = self.to_absolute_north(sample)
129          return sample
```

Both are bare, uncommented `breakpoint()` calls (the surrounding quantization/jit lines ARE commented, these are not). No `PYTHONBREAKPOINT=0` set anywhere in the repo (grep clean), so `breakpoint()` resolves to `pdb.set_trace()`.

## 2. REACHABILITY

Class `v5spfdataset_nn_wrapper` is imported by:
- `spf/mavlink_radio_collection.py:14` (realtime edge inference entry point)
- `spf/filters/run_filters_on_data.py:17`
- `tests/test_v5inference.py:13`, `tests/test_particle_filter.py:11`
- debug scripts (debug.py / debug2.py / debug3.py — untracked, ignore)

The two breakpoints live ONLY on the **realtime** branch (`self.ds.realtime == True`):
- `:126` is inside `else` of `if not self.ds.realtime` → realtime only.
- `:86` is in `to_absolute_north`, called from `:128` which is only reached after `:126`.

Realtime reachability requires the wrapped ds to be a `v5inferencedataset` (class at `spf_dataset.py:448`, default `realtime: bool = True`). Two callers do this:

a) **mavlink_radio_collection.py** (`--realtime`):
`main()` → `if args.realtime:` (:271) builds `v5inferencedataset(...)` then `v5spfdataset_nn_wrapper(..., absolute=True)` (:284-292). NOTE: in the *current* code the `nn_ds` is built but the consumption loop is **commented out** (:332-334), so on the live rover path the breakpoint is constructed-but-not-iterated right now. The realtime inference loop is disabled — consistent with commit `41f8556 "fixes for inference, still missing realtime pf"`. So the live mission path constructs the wrapper but does not currently call `get_and_annotate_entry_at_idx`.

b) **tests/test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper** (:228) — DOES exercise it:
`v5inferencedataset(... realtime defaults True)` (:252) → `v5spfdataset_nn_wrapper(..., absolute=True)` (:271-279) → `PFSingleThetaDualRadioNN(nn_ds=nn_ds)` → `pf.trajectory(...)`. The PF's `observation()` (`particle_dual_radio_nn_filter.py:42`) calls `self.ds.get_inference_for_idx(idx)` → realtime → `get_and_annotate_entry_at_idx` → **hits `:126`**, then `absolute=True` → `to_absolute_north` → **hits `:86`**. No skip marker on the test.

Verdict on "actually used feature": realtime/absolute-north inference is **experimental / in-progress**, not a shipped/working mission feature (loop commented out, commit message says realtime PF still missing). But the path is **reachable today via the test suite**.

## 3. TRIGGER

- `get_and_annotate_entry_at_idx` :126 — wrap a realtime ds (`v5inferencedataset`) and call `[]`/`next()`/`get_inference_for_idx`. CLI: `mavlink_radio_collection --realtime --checkpoint ... --checkpoint-config ...` *if* the inference loop (:332) were uncommented; or running `pytest tests/test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper`.
- `to_absolute_north` :86 — same as above plus `absolute=True` (both mavlink and the PF test pass `absolute=True`).
- A **normal (non-realtime) training/inference run avoids both entirely**: when `ds.realtime==False`, code takes the cached branch (`:100-104`, `:110-115`) and `to_absolute_north`/the realtime branch are never entered. `cached_model_inference_to_absolute_north` (a different function in `particle_dual_radio_nn_filter.py`) handles absolute conversion for the cached path. test_v5inference.py uses plain `v5spfdataset` (realtime=False) so it does NOT hit the breakpoints.

## 4. BLAST RADIUS

`breakpoint()` → `pdb.set_trace()`. On a headless rover/RPi5 with no TTY on stdin, pdb blocks reading from stdin. Effect:
- If stdin is a closed/EOF pipe, pdb raises `BdbQuit`/`EOFError` and unwinds — the inference call crashes the collection process.
- If stdin is an open-but-idle fd (e.g. interactive launch, or ssh session left attached), the process **hangs indefinitely** at the inference call, silently stalling realtime DOA inference mid-mission. The drone keeps flying under the planner (`data_collector` thread), but produces no NN bearings.
- In CI / `pytest` (no `-s`, stdin captured), it would hang or error, blocking the test run.

Currently mitigated on the live mission only because the consuming loop at mavlink :332 is commented out, so the wrapper is built but never iterated.

## 5. REAL-WORLD COVERAGE

- `tests/test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper` exercises the realtime+absolute path and WOULD land on both breakpoints (no skip). This means the test is currently either failing/hanging or is not being run in CI.
- `tests/test_v5inference.py` exercises only the non-realtime branch (its "realtime" sub-case at :123 still passes a non-realtime `v5spfdataset`), so it does not hit them.

## 6. SEVERITY VERDICT — CONFIRMED P0 (for the realtime path), DOWNGRADE for live-mission-today

Justification: Both `breakpoint()` calls are live and unconditional on the only code path realtime inference can take. They are real production-blocking landmines for the realtime/absolute-north feature and they sit directly inside a checked-in test that would hang/break CI. That makes them CONFIRMED P0 for anyone enabling realtime inference. HOWEVER, the realtime mission path is currently experimental and partially disabled (mavlink consumption loop commented out, commit `41f8556` notes realtime PF is "still missing"), so they do not break a presently-shipping live mission. Net: keep as P0 to fix before realtime is re-enabled; the live-rover blast radius today is latent, not active. Trivial cost to remove, no reason to leave them.

## 7. FIX SKETCH (do NOT apply)

Delete the two bare `breakpoint()` statements at `spf_nn_dataset_wrapper.py:86` and `:126`. Two-line removal, no behavioral change to either real path. (Belt-and-suspenders: also set `PYTHONBREAKPOINT=0` in the rover launch env so any stray debug breakpoints become no-ops on embedded targets.) After removal, run `pytest tests/test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper` to confirm the realtime+absolute path completes.
