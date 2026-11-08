# Stage-1 data-path audit — SEG-3 … SEG-6 (correctness + test coverage)

READ-ONLY audit. No source/test code changed. Scope = the exact stage-1 config
`model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`
(`model: beamformer`, `flip: true`, `double_flip: false`, `scatter: continuous`, `sigma: 0.1`,
`windowed_beamformer_input: true`, `beamformer_input: false`, `phase_input: false`,
`signal_matrix_input: false`, `empirical_input: false`, `nthetas: 65`, `n_radios: 2`).

Every claim below was re-verified against source (`spf/dataset/spf_dataset.py`,
`spf/scripts/train_single_point.py`, `spf/scripts/train_utils.py`, `spf/rf.py`, `spf/utils.py`).
The map docs (`SUMMARY.md`, `v5spfdataset.md`, `train_single_point.md`) were used only as a guide.

Config-derived facts that anchor the whole audit:
- The model's beamformer feature is **`windowed_beamformer`** (`windowed_beamformer_input: true`),
  **not** `weighted_beamformer` (`beamformer_input: false`). `global_config_to_keys_used`
  (`train_utils.py:93-97`) therefore puts `windowed_beamformer` in the batch and **omits**
  `weighted_beamformer`. ⇒ for the flip proof the load-bearing flipped input is
  `windowed_beamformer.flip(dims=(3,))` at `spf_dataset.py:1502`.
- `all_windows_stats` is in the batch (always, `train_utils.py:69`); ch0 (mean φ) is negated on flip.
- `signal_matrix` / `abs_signal_and_phase_diff` are **not** in the batch → the one
  known-inconsistent flip field (`:1411` "does not respect flipping") is **unreachable** for this run.
- `empirical` is force-appended to the keys (`train_single_point.py:324`) but only used for
  `.shape[:2]`/`.device` in `target_from_scatter`; `empirical_input:false` so it is not a model feature.

---

## SEG-3 — Ground-truth angles (`get_ground_truth_thetas` / `get_ground_truth_phis` / `get_craft_ground_truth_thetas`)

### Correctness — VERDICT: ✅ CORRECT (frames consistent; φ↔θ relation verified)

`get_ground_truth_thetas` (`spf_dataset.py:1573-1596`):
- `d = tx_pos_mm − rx_pos_mm`, `rx_to_tx_theta = arctan2(d[0], d[1])` (`:1581-1583`) → **x-from-y**
  convention (`arctan2(Δx, Δy)`, 0 rad = +y/boresight). `absolute_theta = pi_norm(arctan2(dx,dy))`
  (`:1586`).
- `ground_truth_theta = pi_norm(arctan2(dx,dy) − (rx_theta_in_pis + rx_heading_in_pis)·π)` (`:1587-1590`)
  → **array-relative** θ. Subtracting both the mount offset and heading rotates world→array frame. ✅
- `assert n_receivers == 2` (`:1595`) — hard 2-rx assumption, holds (`n_radios:2`).
- The "pick the candidate in front of the craft" comment (`:1592-1594`) is **aspirational** — there is
  no candidate-resolution code; GT θ is taken directly from the known TX position. Front/back ambiguity
  is therefore NOT resolved here (it is handled later, in the SEG-6 target fold). Consistent with intent.

`get_craft_ground_truth_thetas` (`:1545-1571`):
- `craft = pi_norm(ground_truth_theta + rx_theta_in_pis·π)` (`:1546-1553`). Heading is **deliberately
  not** re-added (commented `:1550`) — adding it would give the absolute/world frame, not craft.
  So `craft = pi_norm(absolute − rx_heading_in_pis·π)` = world rotated by craft heading. ✅ frame-correct.
- Returns **receiver-0's** value only; receiver-1 disagreement >0.2 mean-abs only `logging.warning`s
  (`:1566-1569`), never raises. Low risk on synthetic/aligned data; a genuinely misaligned r1 would be
  silently ignored.

`get_ground_truth_phis` (`:1508-1527`):
- `φ = pi_norm(−sin(θ_array) · rx_wavelength_spacing · 2π)` (`:1512-1525`). θ is the **array-relative**
  θ (`:1514`, comment confirms). The relation is the standard ULA phase-difference model
  `Δφ = 2π·(d/λ)·sin(θ)`, here with a leading **−** sign. The `−` sign is acknowledged-ambiguous in the
  in-code comment (`:1518-1521`: "up to negative sign, which way do we spin… a −1 here is the same as
  −rx_spacing"). It is validated **empirically**, not derived: `test_fake_data_array_orientation`
  asserts `ground_truth_phis ≈ mean_phase` (atol 1e-3), so the sign/scale match the data generator and
  the rest of the pipeline. Physically unexplained but consistent. ✅
- φ uses array-relative θ (no heading/mount), so the φ↔θ relation is intrinsic to the array — exactly
  what the model's phase observable depends on. This is the key fact the SEG-4/SEG-6 reflections must
  preserve.

### Existing tests
- `test_angles.py::test_angles` — pure-math check of `pi_norm(arctan2(x,y) − rx_theta_in_pis·π)` for 4
  cardinal TX positions and 2 mount offsets. Validates the **array-relative** θ formula (mount offset
  only; **no heading**, no φ).
- `test_fake_data.py::test_fake_data_array_orientation` (`:52-96`) — the strongest validator:
  asserts `ground_truth_phis[r] ≈ mean_phase[r]` (φ correctness incl. the −sin sign/scale, both rx),
  and `arctan2(dx,dy) == craft_ground_truth_theta` (craft-frame correctness). Uses `n_parallel=0`,
  no flip.
- `test_fake_data.py::test_dataset_load` / `::test_dataset_load_drift` — assert `phi_drifts` ≈ injected
  drift (exercises `get_all_phi_drifts` → `get_ground_truth_phis` indirectly).
- `test_paired_rotate_dist.py` — uses `craft_ground_truth_theta` to mask paired beamformer output;
  indirectly checks craft frame vs. beamformer θ grid.

### Gaps
- **Top gap:** no test isolates `get_ground_truth_thetas` with a **non-zero `rx_heading_in_pis`**.
  `test_angles` exercises only the mount-offset term; `test_fake_data` uses the synthetic generator
  whose heading may be 0/constant. The world→array→craft frame decomposition (heading subtracted in
  array θ, NOT re-added in craft θ) is the most error-prone line and is untested in isolation.
- No test covers the r0/r1 craft-disagreement warning path.

### Proposed tests
1. `test_angles.py::test_ground_truth_frames_with_heading` — construct `tx_pos`, `rx_pos`,
   `rx_theta_in_pis`, `rx_heading_in_pis` all non-zero; assert
   `array_theta == pi_norm(arctan2(dx,dy) − (mount+heading)·π)` AND
   `craft_theta == pi_norm(arctan2(dx,dy) − heading·π)` AND
   `craft_theta == pi_norm(array_theta + mount·π)`. Catches: any future swap of "add vs subtract
   heading" or accidental re-adding heading to the craft frame (would silently shift every craft label).
2. `test_angles.py::test_phi_from_theta_sign` — assert
   `y_phi == pi_norm(−sin(y_rad)·rx_wavelength_spacing·2π)` element-wise on a known θ sweep. Catches:
   a flipped −sin sign or a missing 2π/spacing factor (would invert/scale every phase target and break
   model↔label agreement without crashing).

---

## SEG-4 — Reflection / flip augmentation (`render_session`, `flip_left_right`)

### Correctness — VERDICT: ✅ CORRECT (every model-consumed field flips consistently; the one
inconsistent field is unreachable for this config)

`flip_left_right = self.flip and (rand()>0.5)` (`:1357`). On flip (and since `double_flip:false`, the
`double_flip` branch at `:1390-1395,1467-1469` never fires), the following are applied:

| Field | Action on flip | Line | In batch for this config? |
|---|---|---|---|
| `ground_truth_phi` (`y_phi`) | `−φ` | `:1388` | yes (`y_phi`) |
| `ground_truth_theta` (`y_rad`) | `−θ` | `:1389` | yes (`y_rad`) |
| `craft_ground_truth_theta` | recomputed from flipped θ + original `craft_offset` | `:1398-1399` | yes (`craft_y_rad`) |
| `all_windows_stats[:,:,0]` (mean φ) | negated | `:1454` | yes |
| `mean_phase_segmentation` | negated | `:1455` | (not a model input; feeds empirical, off) |
| `weighted_beamformer` | `flip(dims=(2,))` | `:1500` | **no** (beamformer_input:false) |
| `windowed_beamformer` | `flip(dims=(3,))` | `:1502-1503` | **yes** ← load-bearing |
| `y_rad_binned`, `craft_y_rad_binned` | recomputed via `to_bin` from flipped θ | `:1480-1484` | yes |

**(a) Negation applied consistently to target AND inputs — PROVEN.**
- Target side: `y_rad`/`craft_y_rad` are negated *before* the binned targets are computed (`to_bin`
  runs at `:1480-1484`, after the negation at `:1389` and the craft recompute at `:1398`), so
  `y_rad_binned`/`craft_y_rad_binned` are derived from the already-flipped θ — internally consistent.
- Input side: the beamformer power map `windowed_beamformer` has shape `(1, S, n_windows, nthetas)`
  (dim 3 = θ axis; `data_from_precomputed:280-282`). The θ grid is
  `np.linspace(−π, π, nthetas)` with steering `sin(θ)` (`rf.py:1180-1181,
  precompute_steering_vectors`). The grid is symmetric about 0 and monotonic, so reversing the axis
  maps bin `i → nthetas−1−i`, i.e. `θ → −θ`. Thus `windowed_beamformer.flip(dims=(3,))` is exactly the
  `θ → −θ` reflection — the **same** negation applied to `y_rad`. ✅
- `all_windows_stats` ch0 is the per-window mean **phase-difference**; phase diff scales like
  `−sin(θ)·(d/λ)·2π` (SEG-3 relation), which is odd in θ, so `θ → −θ` ⇒ phase → −phase. Negating ch0
  (`:1454`) is the correct image of the same reflection. ch1 (stddev) and ch2 (abs median) are even
  under θ→−θ and correctly left unchanged. ✅
- Net: for this config every model input (`windowed_beamformer`, `all_windows_stats`) and every label
  (`y_rad`, `y_phi`, `craft_y_rad`, the binned variants) transform under one and the same left/right
  reflection `θ→−θ`. A model that learned the true map stays consistent across the flip ⇒ the
  augmentation is a valid invariance-imposing transform, not label noise.

**The one inconsistent field** is `abs_signal_and_phase_diff` (`:1411` "WARNING this does not respect
flipping!") — computed from the raw signal before any flip and never flipped. It is built only when
`signal_matrix` is kept (`:1410`), and this config has `signal_matrix_input: false` →
`abs_signal_and_phase_diff` is neither built nor collated. **Unreachable here.** ✅ (benign for stage 1).

**(c) Flip (L/R) vs target fold (front/back) are DIFFERENT reflections — confirmed (see SEG-6 for the
proof they compose).** The L/R flip is `θ → −θ` (negation, applied to inputs+labels in `render_session`).
The front/back fold is `θ → sign(θ)·π − θ` (applied only to the *target distribution* in
`target_from_scatter`, SEG-6). They are distinct group elements and act at different stages (input
augmentation vs. loss target), so they compose rather than cancel.

### Existing tests
- **NONE.** Grep of all five test files (and the whole `tests/` tree) for
  `flip` / `flip_left_right` / `double_flip` returns **zero matches**. No test ever sets
  `flip=True` on a `v5spfdataset`, and `test_fake_data`/`test_paired_rotate_dist` construct datasets
  without flip. The flip augmentation is entirely untested.

### Gaps
- **Top gap (highest concern):** the L/R flip consistency — that `windowed_beamformer.flip(dims=3)`,
  `all_windows_stats[:,:,0]` negation, and `y_rad`/`y_phi`/`craft_y_rad` negation all correspond to the
  *same* θ→−θ reflection. A regression (e.g. flipping the wrong axis of `windowed_beamformer`, or
  forgetting the `all_windows_stats` ch0 negation) would silently train the model on mislabeled data
  with no crash and no test failure.

### Proposed tests
1. `test_fake_data.py::test_flip_left_right_consistency` — build one fake dataset twice with identical
   seed, `flip=False` vs a forced-flip path (monkeypatch `torch.rand` to force `>0.5`, or add a
   `force_flip` kwarg in a test-only subclass). For the same `(receiver, session)` assert:
   `y_rad_flip ≈ −y_rad`, `y_phi_flip ≈ −y_phi`,
   `windowed_beamformer_flip ≈ windowed_beamformer.flip(dims=(3,))`,
   `all_windows_stats_flip[:,:,0] ≈ −all_windows_stats[:,:,0]` and
   `all_windows_stats_flip[:,:,1:] ≈ all_windows_stats[:,:,1:]`,
   `y_rad_binned_flip == to_bin(−y_rad, ntheta)`.
   **Bug it catches:** any field that doesn't flip in lockstep (wrong axis, missing negation, label/input
   desync) — the exact correctness property the user cares about.
2. `test_invariance.py::test_model_flip_invariance` (integration) — run the `beamformer` model forward
   on a batch and its flipped twin; assert `output_flip ≈ output.flip(dim=θ)` within tolerance. Catches
   a flip that is mathematically inconsistent end-to-end even if per-field shapes look right.

---

## SEG-5 — Collation (`v5_collate_keys_fast`)

### Correctness — VERDICT: ✅ CORRECT

`v5_collate_keys_fast(keys, batch)` (`spf_dataset.py:113-127`):
- `batch` is `List[List[Dict]]` — outer = batch items, inner = receivers (2, since `paired = n_radios>1`).
  For each requested key it `torch.vstack`s across **all (item, receiver)** dicts
  (`[x[key] for paired_sample in batch for x in paired_sample]`, `:116-117`). With `batch_size:256` and
  2 receivers this yields **B = 512 rows**, matching the SUMMARY's B=512. ✅
- **float16→float32 upcast** is applied to exactly `windowed_beamformer` and `all_windows_stats`
  (`:119-120`). For this config those are the two model beamformer/stats inputs; both stored as
  `target_dtype` (float16 by default in `render_session:1488-1494`) and upcast here. ✅ (Note: other
  fields like `y_rad` were already cast to `target_dtype` in `render_session`; the model's first op on
  scalar inputs handles dtype — not a correctness issue for the documented run since `dtype:
  torch.float32` and `amp:false`.)
- `batch_size = d["y_rad"].shape` (`:127`) — **silent hard dependency** on `y_rad` being requested. For
  this config `y_rad` is always in `global_config_to_keys_used` (`train_utils.py:75`), so safe. A config
  that dropped `y_rad` would `KeyError` here.
- **Key selection** is via `global_config_to_keys_used(global_config)` + `"empirical"` appended
  (`train_single_point.py:323-324`). For this config the produced key set =
  base(14) + `windowed_beamformer` + `gains` + `vehicle_type` + `sdr_device_type` + `empirical`.
  Verified against the flag block (`train_utils.py:88-105`): `beamformer_input:false` ⇒ no
  `weighted_beamformer`; `phase_input:false` ⇒ no `weighted_windows_stats`; `signal_matrix_input:false`
  ⇒ no `abs_signal_and_phase_diff`; `empirical_input:false` but empirical force-appended at the call
  site anyway. ✅
- **KeyError contract:** every requested key must exist in every dict. Verified all selected keys are
  produced by `render_session`/`data_from_precomputed` for this config's `skip_fields`
  (signal-related keys are skipped consistently on both sides). No mismatch.

### Existing tests
- **No dedicated unit test** of `v5_collate_keys_fast`. Exercised only indirectly through training /
  inference integration (`test_v5inference.py`, `test_particle_filter.py`, `tests/conftest.py` fixtures
  that call `train_single_point`). `global_config_to_keys_used` has no direct test either.

### Gaps
- **Top gap:** no test pins the key set ↔ config-flag mapping. A config-flag rename or a wrong
  `if global_config["..."]` (these are read **without `.get`**, `train_utils.py:88-105`) would either
  `KeyError` at runtime or silently drop/add a model input. Also untested: the float16→float32 upcast
  (a regression to fp16 model input under `amp:false` would change numerics silently).

### Proposed tests
1. `tests/test_collate.py::test_global_config_to_keys_used_stage1` — pass the stage-1 `global` dict and
   assert the returned key list **exactly equals** the expected set (incl. `windowed_beamformer`,
   excl. `weighted_beamformer`/`weighted_windows_stats`/`abs_signal_and_phase_diff`).
   **Bug it catches:** a flag-wiring regression that swaps `windowed_beamformer` for
   `weighted_beamformer` (the model would get the wrong, un-flipped-by-default feature).
2. `tests/test_collate.py::test_v5_collate_keys_fast_shapes_and_dtype` — build a 2-item paired fake
   batch, collate with the stage-1 keys; assert B==4 (2 items×2 rx), `batch_size==y_rad.shape`,
   and `windowed_beamformer.dtype==float32` / `all_windows_stats.dtype==float32` even when the input
   dicts are float16. **Bug it catches:** dropped upcast (fp16 leaking into an fp32/amp-off model) or a
   wrong vstack ordering that desyncs rows across keys.

---

## SEG-6 — Target construction (`target_from_scatter`, continuous; `single=True` front/back fold)

### Correctness — VERDICT: ✅ CORRECT (analytic Gaussian + wrap-fold sound; the `sign(θ)·π−θ` fold is
the correct front/back mirror and preserves the phase observable; composes with SEG-4, no double-fold)

`target_from_scatter(batch, y_rad, ..., sigma, k, target_ntheta, single)` (`train_single_point.py:539-575`):
- `assert −π ≤ y_rad ≤ π` (`:542`); `assert target_ntheta odd` (`:548`) — for nthetas=65 (odd) ✅.
- `:543-545` allocates `torch.zeros(...)` and **discards** it (dead line; uses only `batch["empirical"]`
  for shape/device). Harmless. `n = target_ntheta` (`:547`).
- **Padded grid:** `padding = n`, `effective_n = 3n` (`:549-551`). Grid
  `m = (arange(3n) − 3n//2) · (2/n) · π` (`:553-556`). Step = `2π/n`; the central `n` bins span exactly
  `[−π, π)` and the two pad blocks of width `n` extend ±2π around it. ✅
- **Analytic Gaussian** `exp(−((m − y_rad)/σ)²/2)`, then **L1-normalized over the full 3n axis**
  (`:557-558`). This is the continuous, analytic placement — the `scatter_k=21` blur kernel is **never
  used** in this path (it is only in `target_from_scatter_binned`). ✅ (config's `scatter_k:21` is inert,
  matches SUMMARY caveat.)
- **(d) Wrap-fold across ±π — PROVEN correct:** `rescaled.reshape(B,S,3,n).sum(axis=2)` (`:561`) sums the
  three length-`n` blocks element-wise. Block −1 covers `[−3π,−π)`, block 0 covers `[−π,π)`, block +1
  covers `[π,3π)`. Bin `j` of the output = mass at angles `{m_j − 2π, m_j, m_j + 2π}`, i.e. all aliases
  of `m_j` mod 2π. So a Gaussian tail that spills past +π is folded back to the correct bin near −π (and
  vice-versa). Because the normalize at `:558` was over the **full 3n** axis *before* folding, the folded
  result still sums to 1 (the fold only regroups existing normalized mass). ✅ Correct circular handling;
  no mass lost or double-counted, even for `y_rad` near ±π where the tail straddles the seam.

`single=True` front/back fold (`:563-574`):
- `y_rad_flipped = sign(y_rad)·π − y_rad` (`:564`); recurse with `single=False` to get the reflected
  target; `target = (target + target_flipped)/2` (`:574`).
- **(b) The fold preserves the physical observable — PROVEN.** The array's phase observable depends on
  `sin(θ)` (SEG-3: `φ ∝ −sin(θ)·(d/λ)·2π`). For the fold `θ' = sign(θ)·π − θ`:
  - `θ ≥ 0`: `θ' = π − θ`, and `sin(π − θ) = sin(θ)`.
  - `θ < 0`: `θ' = −π − θ`, and `sin(−π − θ) = −sin(π + θ) = −(−sin θ) = sin(θ)`.
  So `sin(θ') = sin(θ)` for all θ ∈ [−π,π] ⇒ identical phase difference ⇒ the two angles are
  **physically indistinguishable to a 2-element array** (the classic front/back ambiguity). Averaging
  the target with its `θ'` image makes the supervised distribution **bimodal exactly where the hardware
  cannot disambiguate**, which is the correct, honest label. ✅
  - Also note `θ' ∈ [−π,π]` (for θ∈[0,π], θ'∈[0,π]; for θ∈[−π,0), θ'∈(−π,0]), so the recursive call's
    `assert −π≤·≤π` (`:542`) holds. ✅
- **Mass / normalization:** each half is an L1-normalized distribution (sum 1); the mean of two
  distributions is again a distribution (sum 1). ✅ So the `single=True` target sums to 1.
- **(c) Composition with SEG-4 — confirmed NOT a double-fold.** SEG-4's L/R flip is `θ→−θ` applied to
  *both inputs and labels* in `render_session` (data augmentation). SEG-6's front/back fold is
  `θ→sign(θ)·π−θ` applied *only to the target distribution* inside the loss. They are **different
  reflections** (negation vs π-fold; one preserves `cos`/changes `sin`-sign asymmetry of the array
  geometry, the other preserves `sin`) acting at **different stages**. Composing them spans the full
  4-fold ambiguity set `{θ, −θ, π−θ, −(π−θ)}` (= `sign·π−θ` and its negation) — the genuine ambiguity of
  a 2-element ULA. There is no path where the same reflection is applied twice and cancels:
  `render_session` applies negation to data; `target_from_scatter(single=True)` applies the π-fold to the
  target it builds *from* `batch["y_rad"]`. With `double_flip:false`, the `render_session` π-fold branch
  (`:1390-1395`) is dead, so the π-fold lives **only** in the target. ✅

The driver call (`train_single_point.py:859-868`) passes `y_rad=batch["y_rad"][...,None]`,
`target_ntheta=output["single"].shape[-1]` (=65), `single=True`, then `mse_loss(output["single"],
target)` (`:868`). Both sides are L1-normalized distributions over 65 bins (`mse_loss` is SSE over the
bin axis), so the comparison is well-posed. ✅

### Existing tests
- `target_from_scatter` (continuous, `single=True`): **NO direct test**. Exercised only indirectly via
  the `single_net_checkpoint` conftest fixture (which uses `scatter: continuous`,
  `target_ntheta=7`), i.e. it runs but nothing asserts the wrap-fold or the symmetry-average properties.
- `test_rotate_dist.py` — tests `rotate_dist` (distribution rotation by an angle), a *different*
  function used in the no-grad aux metrics, not `target_from_scatter`. Not coverage for SEG-6.
- No test references `target_from_scatter` by name (grep = 0 matches).

### Gaps
- **Top gap:** the `single=True` front/back fold (`sign(θ)·π−θ`) and the ±π wrap-fold are the two most
  subtle, correctness-critical pieces in the whole loss and have **zero** direct assertions. A regression
  (e.g. `sign(θ)·π − θ` → `π − θ` without the sign, or summing the wrong reshape axis) would silently
  produce a wrong/unimodal target with no crash.

### Proposed tests
1. `tests/test_target.py::test_target_from_scatter_single_fold_symmetry` — for a sweep of
   `y_rad ∈ {−3, −1, 0.3, 1, 3}` (rad), σ=0.1, target_ntheta=65, `single=True`:
   assert the target **sums to 1** per row, and that target mass at the bin of `θ` equals the mass at the
   bin of `sign(θ)·π − θ` (within tolerance), i.e. the distribution is invariant under the front/back
   reflection. **Bug it catches:** a broken/asymmetric fold (dropped `sign`, wrong average) — exactly the
   physical-symmetry property the user flagged as top concern.
2. `tests/test_target.py::test_target_from_scatter_wrap_fold_seam` — set `y_rad = π − 0.02` (tail spills
   past +π) with `single=False`; assert (a) row sums to 1, (b) bins just inside −π carry non-negligible
   mass (tail wrapped), and (c) the result equals an independent reference computed by binning a wrapped
   Gaussian over `[−π,π)` mod 2π. **Bug it catches:** a wrap-fold that loses tail mass or folds to the
   wrong bin (would bias every near-seam target).
3. `tests/test_target.py::test_target_normalizes_to_one_all_paths` — parametrize `single ∈ {False,True}`,
   several σ; assert `target.sum(-1) ≈ 1`. **Bug it catches:** any future change that breaks the
   normalize-before-fold ordering (`:558` then `:561`) so the distribution no longer sums to 1.

---

## Cross-segment correctness summary

| Seg | Item | Verdict |
|---|---|---|
| 3 | array/craft/absolute frames + φ↔θ | ✅ correct (heading subtracted in array θ, not re-added in craft; φ=−sin(θ)·(d/λ)·2π, sign empirically validated) |
| 4 | L/R flip `θ→−θ` on inputs+labels | ✅ consistent — `windowed_beamformer.flip(dim3)` ≡ θ→−θ on the symmetric `linspace(−π,π)` grid; `all_windows_stats` ch0 negated; only-inconsistent field (`abs_signal_and_phase_diff`) is unreachable for this config |
| 5 | collation B=512, key set, fp16→fp32 | ✅ correct; key set matches flags exactly; `y_rad` present so `batch_size` safe |
| 6 | continuous target + `single` front/back fold | ✅ correct — `sin(sign(θ)·π−θ)=sin(θ)` preserves the phase observable; wrap-fold handles ±π seam; composes with SEG-4 (different reflections), no double-fold |

**Reflection proof bottom line (user's top concern):**
- (a) ✅ The L/R negation is applied to **both** the target (`y_rad`/`y_phi`/`craft_y_rad` and their
  binned forms, all derived after the negation) and the inputs (`windowed_beamformer` θ-axis flip =
  θ→−θ; `all_windows_stats` ch0 phase negated). No model-consumed field flips inconsistently for this
  config.
- (b) ✅ `sign(θ)·π − θ` is the correct front/back mirror: `sin(sign(θ)·π−θ) = sin(θ)` ⇒ same phase
  difference ⇒ physically indistinguishable to the 2-element array.
- (c) ✅ L/R flip (negation, in `render_session`, on data+labels) and the front/back fold (π-fold, in
  `target_from_scatter`, on the target only) are **different** reflections at **different** stages; with
  `double_flip:false` the π-fold exists only in the target → they compose to span the array's true
  4-way ambiguity, they do not double-fold/cancel.
- (d) ✅ The 3-block reshape-sum wrap-fold correctly folds Gaussian tails across ±π (each output bin
  collects all 2π-aliases) and preserves the unit-sum normalization.

**Test-coverage bottom line:** the four correctness pillars the user cares about —
**flip consistency (SEG-4), the `single` front/back fold and the wrap-fold (SEG-6)** — have **ZERO
direct test coverage** (grep for `flip` and `target_from_scatter` across all tests = 0 hits). SEG-3 is
partially covered (`test_angles`, `test_fake_data`) but the heading term and φ-sign are not isolated.
SEG-5 has only indirect integration coverage. Highest-value additions: the flip-consistency test
(SEG-4 #1) and the `single`-fold symmetry test (SEG-6 #1).
