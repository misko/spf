# P0 Issue #2 — Live `breakpoint()` in PFSingleThetaDualRadioNN.observation()

**File:** `spf/filters/particle_dual_radio_nn_filter.py:41`
**Verdict: CONFIRMED P0** (severity confirmed; see caveat — it only hits the dual-radio-NN code path, but that path IS selected by the active run config and by 3 tests).

---

## 1. CONFIRM — is the breakpoint live?

YES, live (not commented). `spf/filters/particle_dual_radio_nn_filter.py:40-42`:

```python
40    def observation(self, idx):
41        breakpoint()
42        return self.ds.get_inference_for_idx(idx)[0]["paired"][0]
```

It is the first statement of `observation()`, so it fires before any work is done.
Git blame: introduced in commit `b0cdfb6` ("updates", 2026-02-15) — the current HEAD's parent line of work, not ancient.

---

## 2. REACHABILITY — who reaches it?

`observation()` is called by the base-class `ParticleFilter.trajectory()` loop at
`spf/filters/filters.py:333` (`z = self.observation(idx)`) on **every** iteration,
starting with `idx=0` — i.e. the first trajectory step. So any successful construction +
`pf.trajectory(...)` of `PFSingleThetaDualRadioNN` hits the breakpoint immediately.

Call chain to that class:
- `run_PF_single_theta_dual_radio_NN(...)` (`run_filters_on_data.py:416`) builds
  `PFSingleThetaDualRadioNN(nn_ds=nn_ds)` (line 437) and calls `pf.trajectory(...)` (line 438).
- It is wired into the dispatch table `fn_key_to_fn` at `run_filters_on_data.py:510`
  (`"run_PF_single_theta_dual_radio_NN": run_PF_single_theta_dual_radio_NN`), so it is a
  fully selectable job, not dead code.

**Is it in the active run set?** YES.
- `spf/model_training_and_inference/models/ekf_and_pf_config.yml:59` — `run_PF_single_theta_dual_radio_NN:`
  is an **uncommented, active** run block (checkpoint `march16/...x2/best.pth`, seg 3.5,
  full N/theta sweep, `absolute: [True, False]`). This is the main EKF/PF config.
- `spf/scripts/test.yml:9` — `run_PF_single_theta_dual_radio_NN:` also active.
- Only the `_debug.yml` variant has it commented out
  (`ekf_and_pf_config_debug.yml:37` is prefixed `#g-`).

**Referenced in tests?** YES — `tests/test_particle_filter.py`:
- import at line 18; class import at line 12.
- `run_PF_single_theta_dual_radio_NN(**args)` called at line 165 and line 224
  (tests `test_single_theta_dual_radio_NN`-style, both `temp_file` params).
- `PFSingleThetaDualRadioNN(nn_ds=nn_ds)` constructed directly at line 280
  (`test_single_theta_dual_radioNN_nnwrapper`) followed by `pf.trajectory(...)`.

---

## 3. TRIGGER — exact config/CLI that selects it

CLI:
```
python spf/filters/run_filters_on_data.py \
    -d <datasets> --empirical-pkl-fn <pkl> --work-dir <dir> \
    --config spf/model_training_and_inference/models/ekf_and_pf_config.yml
```
The `run_PF_single_theta_dual_radio_NN:` block (yml lines 59-75) generates jobs via
`config_to_jobs` → `fn_key_to_fn["run_PF_single_theta_dual_radio_NN"]`. Also triggered by
`--config spf/scripts/test.yml`. And by pytest on `tests/test_particle_filter.py`
(the two `run_PF_single_theta_dual_radio_NN` tests + the nnwrapper test).

---

## 4. BLAST RADIUS

Hard-block / hang. `breakpoint()` drops into pdb on stdin. In the production path the jobs
run under a `multiprocessing.Pool` (`run_filter_jobs`, `run_filters_on_data.py:643`) with no
interactive TTY → the pdb prompt hangs forever (or errors on EOF/`BdbQuit` depending on
environment), stalling that worker indefinitely. It does NOT cleanly raise/skip; it blocks
on the very first step before returning any metrics.

Sibling comparison: the **single-radio-NN** filter
(`spf/filters/particle_single_radio_nn_filter.py`, `observation()` at line 87) has **no**
breakpoint and is otherwise functional — so `run_PF_single_theta_single_radio_NN` still works.
Other PF siblings are clean too (the XY filter has only a *commented* `# breakpoint()` at
`particle_dualradioXY_filter.py:99`). So only the **dual-radio-NN** variant is dead; the rest
of the filter suite is unaffected.

---

## 5. REAL-WORLD

- Tests DO exercise `PFSingleThetaDualRadioNN`: `tests/test_particle_filter.py` lines 165,
  224, 280. These tests will hang/fail under the current breakpoint (they call
  `trajectory`, which hits line 41 immediately). This means the test suite for this filter is
  currently broken by the breakpoint — a strong signal it was committed accidentally during
  debugging.
- `type: "PF_single_theta_dual_radio_NN"` is the result label this runner would emit
  (`run_filters_on_data.py:450`); any committed DynamoDB/pickle reports with that type would
  pre-date commit `b0cdfb6`.

---

## 6. SEVERITY VERDICT — CONFIRMED P0

Justification: unlike a breakpoint in a never-selected variant, this filter IS selected by
the primary active config (`ekf_and_pf_config.yml:59`), by `test.yml`, and by 3 tests. It
fires on the first trajectory step under a non-interactive multiprocessing Pool, so it hangs
the worker permanently and makes the entire dual-radio-NN PF path non-runnable. The blast
radius is contained to this one filter variant (siblings work), which is why it's "this one
path is dead" rather than "everything is dead" — but the path is real, exercised, and on the
default config. Keep P0.

---

## 7. FIX SKETCH (do NOT apply)

Delete line 41 (`breakpoint()`) so `observation()` is just:
```python
def observation(self, idx):
    return self.ds.get_inference_for_idx(idx)[0]["paired"][0]
```
Minimal one-line removal. After removal, re-run `tests/test_particle_filter.py` (the
`run_PF_single_theta_dual_radio_NN` and `_nnwrapper` tests) to confirm the path completes.
Optionally add a CI/pre-commit grep guard for live `breakpoint()` calls under `spf/` to
prevent recurrence (the same commit added a commented one in the XY filter).
