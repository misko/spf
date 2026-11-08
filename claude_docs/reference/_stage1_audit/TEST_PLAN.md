# Stage-1 test plan — zero production-code changes

Goal: add the [coverage-matrix](TEST_COVERAGE_MATRIX.md) tests **without modifying any file under
`spf/`**. Everything lands in `tests/` (and `tests/model_configs/`). This is achievable because the
production functions are already testable as-is — we add a thin test-only scaffold and call prod
code unmodified.

## Principle

**Testability comes from the test side, never from a production seam.** Where a test needs control
(force a random flip, supply a config flag, build a batch), we use **fixtures, seeds, monkeypatch,
and direct imports** — not a new `force_flip=`/`testing=` parameter in production. Three enablers
(all verified to already exist) make this work:

- `tests/conftest.py` fixtures already `create_fake_dataset(...)` → `v5spfdataset(...)` (which
  segments the fake data, producing `windowed_beamformer`/`all_windows_stats`) and yield
  `(tmpdir, empirical_pkl, fn)`. The e2e tests ride these.
- All targets are module-level importable: `train_utils.global_config_to_keys_used`,
  `train_single_point.{load_optimizer,load_model,target_from_scatter,mse_loss}`,
  `rf.{circular_mean_single_fast,circular_stddev_fast,fast_percentile_1d}`,
  `single_point_networks.{AllWindowsStatsNet,SinglePointWithBeamformer}`.
- `conftest.update_config(in, updates, out)` + `tests/model_configs/*.yaml` already parametrize a
  training run against fake data.

## Shared scaffold (build once, reuse everywhere) — all in `tests/`

1. **Reuse the existing v5 fake-data fixture** (the session-scoped `create_fake_dataset` → segmented
   `v5spfdataset` one). No new fixture needed for the data itself.
2. **`tests/stage1_helpers.py`** (new, test-only):
   - `STAGE1_GLOBAL` / `STAGE1_MODEL`: the `global:`/`model:` dicts copied from
     `jul8_single_3p7_thin_noblade_jun26_2026.yaml` (so tests assert against the *real* config).
   - `make_stage1_batch(fn, tmpdir, batch=4)`: builds a `v5spfdataset` with the stage-1 keys and
     returns one collated batch (via the real `v5_collate_keys_fast`). Reused by SEG-7/8.
   - `fake_target_batch(B, S=1, n=65)`: `{"empirical": torch.zeros(B,S,n)}` for SEG-6.
3. **`tests/model_configs/test_windowed_single_net.yaml`** (new, test-only): mirrors the prod
   `windows_stats_net` structure but tiny (`hidden:32`, `n_layers:1`, few records) and is pointed at
   the fake dataset/tempdir at runtime via `update_config`. Drives the SEG-11 keystone smoke.

That's the entire footprint: **2 new test files + 1 test config + assertions added to existing test
modules.** Nothing under `spf/`.

## Per-segment plan (every row: production change = NONE)

| # | Test (file) | Calls (prod, unmodified) | Asserts | How we avoid a prod change |
|---|---|---|---|---|
| 11 | `test_train_smoke.py::test_windowed_beamformer_smoke` ⭐keystone | `train_single_point(args)` via the fake-data fixture + `update_config` on the new test config, `--steps 2` | a checkpoint is written; final loss is finite; runs the real 68→12→18→65 path | new **test config** points at fake data — no prod seam |
| 4 | `test_flip.py::test_flip_left_right_consistency` | `ds.render_session(...)` | with flip forced ON: `windowed_beamformer == unflipped.flip(dims=3)`, `all_windows_stats` ch0 negated (ch1/2 equal), `y_rad/y_phi/craft_y_rad` negated | **monkeypatch `torch.rand`** → 0.6 (flip) / 0.4 (no-flip); no `force_flip=` param |
| 6 | `test_target.py::test_single_fold_symmetry` + `::test_wrap_fold` | `target_from_scatter(batch, y_rad, …, single=True)` | sum over bins ≈ 1; mass(θ) ≈ mass(`sign(θ)·π−θ`); mass placed near ±π folds to the correct bins | build `{"empirical": zeros}` test-side; pure call |
| 2 | `test_rf.py::test_circular_mean_single_fast_wraparound`, `::test_circular_stddev_fast`, `::test_fast_percentile_1d` | the live `rf.py` `@njit` fns | match a numpy reference within 1e-5 on ±π-spanning inputs; `fast_percentile_1d` matches `np.percentile(...,'higher')` (ceil rank) | direct import; **rename the duplicate `test_circular_mean`** (test-side only) |
| 1 | `test_windowed_beamformer.py::test_windowed_beamformer_argmax` | `v5spfdataset[...]` via fixture | per-window `argmax(axis=-1)` lands at the bin implied by the injected `phi`; shape `(n_windows,65)` | rides the fake-data fixture; reads the real cached array |
| 7 | `test_models.py::test_allwindowsstatsnet_forward` | `AllWindowsStatsNet(**STAGE1_MODEL["windows_stats_net"])` | output `(B,1,12)`; window-aug is a **no-op under `.eval()`** (eval==eval deterministic) | instantiate with the real config dict; fake tensors |
| 8 | `test_models.py::test_singlepoint_forward_l1` | `load_model(STAGE1_MODEL, STAGE1_GLOBAL)` + `make_stage1_batch` | output `{"single": (B,1,65)}`, all ≥0, `sum(dim=2)≈1` | shared `make_stage1_batch` helper |
| 5 | `test_collate.py::test_keys_used` + `::test_collate_dtype` | `global_config_to_keys_used(STAGE1_GLOBAL)`, `v5_collate_keys_fast` | exact key set (catches `windowed`↔`weighted` swap); f16→f32 upcast; B doubles for paired | pure call + 2 fake items |
| 3 | `test_angles.py::test_ground_truth_frames_with_heading` | `v5spfdataset` GT-angle props | array/craft/absolute relations hold with non-zero mount **and** heading | a fake config string sets `rx_heading_in_pis` — test-side |
| 9 | `test_loss.py::test_mse_loss_handcomputed` | `mse_loss` | equals a hand-computed value on a 1×1×3 input | pure call |
| 10 | `test_optimizer.py::test_step_builds_stepLR` + `::test_cosine_raises` | `load_optimizer` | step → `StepLR(6,0.5)`; cosine → `pytest.raises(UnboundLocalError)` (regression-pin for #1) | pure call; pin documents the bug without fixing prod |

## Sequencing (highest leverage first)

1. **Scaffold** (`stage1_helpers.py` + `test_windowed_single_net.yaml`) — one small PR, unlocks the rest.
2. **SEG-11 keystone smoke** — the single test that transitively exercises SEG-1/5/6/7/8/9/10/11 on
   the *real* production path (today's e2e fixture runs a different model variant).
3. **SEG-4 flip** + **SEG-6 target fold** — your two correctness pillars, currently zero coverage.
4. **SEG-2 live circular stats** — the most load-bearing untested numerics (KNOWN_ISSUES #37).
5. The remaining quick units (SEG-1/3/5/7/8/9/10).

## Traps avoided (where a prod change would be tempting)

| Temptation (would touch `spf/`) | Test-side alternative |
|---|---|
| Add `force_flip=` to `render_session` | `monkeypatch.setattr(torch, "rand", lambda *a,**k: tensor([0.6]))` |
| Extract the windowed `reshape→mean→.T` into a prod fn to unit-test it | test it through the real `v5spfdataset` path (SEG-1) |
| Add config defaults so a partial config won't `KeyError` (#36) | tests always pass the **full** `STAGE1_GLOBAL` |
| Make the GPU `_cp` beamformer importable without cupy | `@pytest.mark.skipif(cupy is None)` — out of stage-1 scope |
| Fix the cosine `pass` (#1) to make SEG-10 pass | assert it **raises** (pin); fixing is a separate, explicit decision |

## The only existing-code edits (both test-side, not `spf/`)

- Rename one of the duplicate `test_circular_mean` in `tests/test_rf.py` so both run (the shadowed
  one currently never executes).
- (Optional) if `create_fake_dataset`'s default yaml pins `rx_heading_in_pis=0`, SEG-3 supplies a
  heading via a fake yaml string — still test-side.

## Runtime / CI

Fake datasets use small `n` and tiny nets, so the suite stays fast and CPU-only; the keystone smoke
runs `--steps 2`. GPU-only paths (`_cp`) are skipped. All randomness is seeded (`create_fake_dataset`
seeds numpy/torch/random; the flip test pins `torch.rand`) → deterministic.

**Net result: full stage-1 coverage, `git diff spf/` is empty.**
