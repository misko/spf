# Stage-1 pipeline — correctness + test-coverage matrix

One row per pipeline segment, in execution order. **Correctness** = verified read-only against
source (the reflection and segmentation math were checked with extra scrutiny). **No code/tests
were written** — the "Proposed" column is a plan. Detail per block in
[`coverage_segmentation.md`](coverage_segmentation.md), [`coverage_datapath.md`](coverage_datapath.md),
[`coverage_model_loop.md`](coverage_model_loop.md).

Coverage key: ✅ direct test · ⚠️ indirect/partial · ❌ none.

| # | Segment (where) | Correctness | Current test coverage | Proposed (planned) coverage | Recommendation |
|---|---|---|---|---|---|
| 1 | **Windowed beamformer** — per-window delay-and-sum that fills `windowed_beamformer` (`segmentation.py:504-532`) | ✅ correct (assumes `stride==window_size`, true here) | ❌ none direct — `test_rf.py::test_beamformer_functions` only checks numpy↔numba per-sample parity, not the windowed `reshape→mean→.T` writer | `test_windowed_beamformer_aggregation`: inject a known dominant angle per window → assert shape `(n_windows,65)` and per-window `argmax==injected bin` | **P1** — guards the model's main input against a silent axis/reshape permutation |
| 2 | **Per-window circular stats** — trimmed circ-mean/stddev(φ) + median\|sig\| → `all_windows_stats` (`rf.py` `circular_mean_single_fast`/`circular_stddev_fast`/`fast_median_abs`/`fast_percentile_1d`) | ✅ math correct; ⚠️ **live engine untested** (KNOWN_ISSUES #37) | ⚠️ primitives only (`test_circular_diff_to_mean_single`, `test_fast_median_abs` @1e-6). Live trimmed mean/stddev + ceil-rank `fast_percentile_1d` ❌. `test_circular_mean` is duplicate-shadowed **and** tests the wrong (non-live) fn | `test_circular_mean_single_fast_wraparound` (vs numpy ref, ±π-spanning, 1e-5); `test_circular_stddev_fast`; `test_fast_percentile_1d` (ceil convention vs `np.percentile`); rename the dup `test_circular_mean` | **P0** — feeds the model's φ channels; highest untested fan-in in the pipeline |
| 3 | **Ground-truth angles** — array/craft/absolute θ, φ↔θ (`v5spfdataset.get_ground_truth_*`) | ✅ correct (−sin sign acknowledged, empirically validated) | ✅ `test_angles::test_angles` + `test_fake_data::test_fake_data_array_orientation` — but only **zero-heading** | `test_ground_truth_frames_with_heading`: non-zero mount **and** `rx_heading_in_pis`, assert all three frame relations | **P2** — partially covered; add the heading decomposition case |
| 4 | **Reflection / flip aug** — `flip_left_right` θ→−θ across inputs+labels (`spf_dataset.py:1354-1500`) | ✅ **correct (proven)** — `windowed_beamformer.flip(dims=3)` ≡ θ→−θ (grid symmetric about 0); `all_windows_stats` ch0 negated, ch1/2 kept; all labels negated | ❌ **NONE** — 0 grep hits for `flip` in tests | `test_flip_left_right_consistency`: force a flip, assert every input/label field flips in lockstep with θ→−θ | **P0** — your top concern; zero coverage; a wrong axis or missed ch0 negation would mislabel silently |
| 5 | **Collation** — `v5_collate_keys_fast`, key-set from `global_config_to_keys_used`, paired flatten (B=512), f16→f32 (`spf_dataset.py`) | ✅ correct | ❌ none direct (only indirect via integration) | `test_global_config_to_keys_used_stage1`: assert the exact key set; assert f16→f32 upcast on `windowed_beamformer`/`all_windows_stats` | **P1** — catches a `windowed`↔`weighted` key swap |
| 6 | **Target construction** — analytic Gaussian + ±π wrap-fold + `single=True` front/back fold (`train_single_point.py:539-575`) | ✅ **correct (fold proven)** — `sin(sign(θ)·π−θ)=sin(θ)`; unit-sum preserved; composes with SEG-4, no double-fold; `scatter_k` inert | ❌ NONE direct (`test_rotate_dist` is a different fn) | `test_target_from_scatter_single_fold_symmetry`: sum==1 and mass(θ)==mass(sign(θ)·π−θ); `test_target_wrap_fold`: mass near ±π folds to the correct bins | **P0** — your top concern (reflection); the subtlest pieces have zero assertions |
| 7 | **Windowed-beamformer net** — `AllWindowsStatsNet` 68→…→12, window aug (`single_point_networks.py:738+`) | ✅ correct (aug train-only, no eval leak; `/500` + `elif` exclusivity noted) | ❌ none | `test_allwindowsstatsnet_forward_contract`: out `(B,1,12)`; assert window aug is a **no-op at `.eval()`** | **P1** |
| 8 | **Input assembly + FFNN** — `PrepareInput` (18-dim) + `FFNN` → `.abs()`→L1-norm → `{single:(B,1,65)}` | ✅ correct (18 = 6 scalars + 12; L1 over dim 2) | ❌ none direct (existing consistency test uses the `phase_input` variant, asserts no shapes) | `test_singlepoint_forward_shape_and_l1norm`: `(B,1,65)`, all ≥0, sum over dim2 ≈ 1 | **P1** |
| 9 | **Loss** — `mse_loss` SSE over 65 bins (`train_single_point.py:608`) | ✅ correct | ❌ none | `test_mse_loss_handcomputed`: equals a hand-computed value on a tiny input | **P2** — cheap, fast |
| 10 | **Optimizer / scheduler** — `load_optimizer` AdamW + StepLR(6, 0.5) (`train_single_point.py:389`) | ✅ correct for `step`; cosine branch is KNOWN_ISSUES #1 (not hit here) | ❌ none | `test_load_optimizer_step_builds_stepLR`; `test_load_optimizer_cosine_raises` (regression pin for #1) | **P1** — pins the #1 crash |
| 11 | **Train loop / checkpoint / val** — `train_single_point` (single_loss backprop, save_on→best.pth, resume) | ✅ correct mechanics (diagnostics `no_grad`; StepLR per-epoch; `train()` restored after val) | ⚠️ 1 e2e smoke (`conftest::single_net_checkpoint` → `test_single_checkpoints_exist`) **but trains a different model variant** (phase, no `windows_stats_net`) | `test_train_smoke_windowed_beamformer_config`: 2-step smoke on a **prod-shaped** config (transitively exercises SEG-1,5,6,7,8,9,10,11); add resume + best-only asserts | **P0** — keystone test; the current smoke never touches the production path |

## Meta-recommendation (priority order)

1. **The keystone gap is the test fixture itself** — `conftest.py`'s training fixture uses
   `tests/model_configs/test_single_net.yaml` (`phase_input:True`, **no** `windows_stats_net`),
   so the entire production windowed-beamformer path (SEG-1,5,6,7,8) is never run by CI. A single
   **prod-shaped smoke test** (SEG-11) is the highest-leverage addition — it exercises the real
   forward/loss/optimizer/checkpoint contract end-to-end and would catch any 68→12 / 18→65 / key-set
   regression in one shot.
2. **Then the two correctness pillars you called out**, both at **zero** coverage:
   `test_flip_left_right_consistency` (SEG-4) and `test_target_from_scatter_single_fold_symmetry`
   (SEG-6).
3. **Then the live circular-stats engine** (SEG-2) — the most load-bearing untested numeric code
   feeding the model, and the one KNOWN_ISSUES #37 already flagged.
4. The remaining P1/P2 unit tests (SEG-1,5,7,8,9,10) are quick, high-signal, and independent.

All segments are **correct as written**; these are coverage gaps, not bug fixes. The only behavior
that is both untested *and* known-broken is the cosine scheduler (#1, SEG-10) — a regression-pin
test there would document it until it's fixed.
