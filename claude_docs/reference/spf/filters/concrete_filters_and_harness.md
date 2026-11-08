# Reference: concrete filter classes + `run_filters_on_data.py` harness

The **8 concrete SPF filter classes** (3 EKF, 5 PF) and the **batch run harness**. These plug into
the shared substrate documented in [`filters.md`](filters.md) (the `SPFFilter`/`ParticleFilter`
bases, the analytic `h()`/Jacobian observation models, the torch PF kernels). **Read `filters.md`
first** — this doc does *not* re-derive the observation math or the base-class lifecycle; it
documents each concrete class's state layout + overrides, and the harness's job dispatch + result
schema.

Passes 1-3 complete (read-only, no source changed). Verified against source at commit
`574c4548e23c18d95ff3c2b72d3b63f0b1f42251` on 2026-06-21. Units/frames per
[`conventions.md`](../../00_concepts/conventions.md); observation relation
`φ = 2π·(d/λ)·sin(θ−offset)` per [`beamforming_primer.md`](../../00_concepts/beamforming_primer.md).

> **Sign convention (load-bearing, repeated everywhere).** Every EKF/PF that uses the analytic or
> empirical model negates the antenna spacing on construction
> (`antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]`,
> e.g. `ekf_single_radio_filter.py:31`, `ekf_dualradio_filter.py:33`,
> `ekf_dualradioXY_filter.py:56`). The effective `d/λ` fed to `h()` is therefore **negative**. This
> must match the dataset's `mean_phase` sign — it is not asserted anywhere.

---

## How the concrete classes plug into the base

- **EKFs (1-3):** `class X(ExtendedKalmanFilter, SPFFilter)`. They use filterpy's
  `ExtendedKalmanFilter.update(z, HJacobian, Hx, residual=…, R=…)` for the Bayesian step, passing the
  analytic `h`/`hjacobian` from `filters.py` as `partial` callbacks plus `residual` (the φ-wrap). They
  **do not** use the base `SPFFilter.trajectory`; each defines its own `trajectory` loop
  (predict → maybe-update → record). `SPFFilter` contributes only `self.ds` and the metric helpers.
- **PFs (4-8):** `class X(ParticleFilter)`. Their `trajectory` calls
  `super().trajectory(**kwargs)` (the live PF engine in `filters.py:297`) and then post-processes the
  per-step dicts (unpack `mu[0]`→`theta`/`craft_theta`, `var[0]`→`P_theta`). They differ only in
  `predict`, `update` (the likelihood), `observation`, and `fix_particles`. **Exception:** `PFXYDualRadio`
  has **no** `trajectory` override, so its trajectory dicts contain only `{"mu","var"}` (no
  `theta`/`P_theta` keys) — its `metrics` and plot helper read `mu[0]`/`mu[[1,2]]` directly.

**State-vector layouts** (first element is always θ or derived-θ):

| Layout | Dim | Filters |
|---|---|---|
| `[θ, dθ/dt]` | 2 | EKF single (1), EKF dual (2), PF single (4), PF dual (5), PF single-NN (7), PF dual-NN (8) |
| `[tx_x, tx_y, dtx_x, dtx_y, rx_x, rx_y]` | 6 | EKF XY (3) |
| `[θ, x, y, dx/dt, dy/dt]` | 5 | PF XY (6) |

---

## 1. `SPFKalmanFilter` · `ekf_single_radio_filter.py:19` · ✅ Active

**Summary.** Single-radio EKF tracking `[θ, dθ/dt]` (dim_x=2, dim_z=1). θ is *folded to the front
half* `[-π/2, π/2]` after every predict/update via `fix_x`, resolving the 2-element-array front/back
arcsin ambiguity. Observation is one radio's phase difference φ.

**State.** `x = [θ, dθ/dt]`, np `(2,1)`. `setup` inits `[[0],[0]]` (line 107).

**Overrides.**
- `R_at_x()` (43) — dynamic measurement-noise scale `2.5·exp(-(|pi_norm(θ)|-π/2)²)`, reads `self.x[0,0]`.
- `fix_x()` (46) — `x[0] = reduce_theta_to_positive_y(pi_norm(x[0]))`; **asserts θ ∈ [-π/2,π/2]** (48).
- `predict(dt, noise_std)` (55) — CV: `F=F_cached(dt)`, `Q=Q_discrete_white_noise_cached(dim=2,dt,var=noise_std)`,
  `x←F·x`, `fix_x()`, `P←F·P·Fᵀ+Q`.
- `update(observation)` (68) — filterpy EKF update with `single_h`/`single_hjac` (asserts θ∈[-π/2,π/2]),
  `residual=residual` (φ-wrap), and `R = self.R if dynamic_R==0.0 else dynamic_R·(R_at_x()²)·5` (83); then `fix_x()`.
- `observation(idx)` (92) — cached `all_observations[idx].reshape(1,1)` from `mean_phase[f"r{rx_idx}"]`,
  else live `ds[idx][rx_idx]["mean_phase_segmentation"]`. **(2 branches: cached vs `temp_file`.)**
- `metrics(trajectory)` (101) — `single_radio_mse_theta_metrics` vs `ds.ground_truth_thetas[rx_idx]`.
- `trajectory(...)` (116) — own loop; **skips `update` when `observation` is non-finite** (151).
  `debug=True` adds `jacobian/hx/P_theta/observation` keys and calls `observation.item()` (so debug
  requires a scalar-convertible obs).

**Side effects:** mutates `self.x/P/F/Q`; asserts (shape in `single_h`, range in `fix_x`); reads `ds`.

**Tested by.** `tests/test_ekf.py::test_single_radio_ekf` (both `temp_file` branches), asserts
`mse_single_radio_theta < 0.03` for both radios; also runs `run_and_plot_single_radio_EKF`.

**Analysis.**
- *Correctness:* on the live path `fix_x` runs before the `single_h` assert can fire, so the redundant
  `θ∈[-π/2,π/2]` assert in `single_h` does not crash (✅ verified — `predict` calls `fix_x` at 62,
  `update` at 86). The dynamic-R formula differs from the dual filters (it multiplies by `R_at_x()²·5`
  rather than the diagonal `r·dynamic_R`).
- *Issue (43,83):* with `dynamic_R != 0`, R is built from `R_at_x()` which reads `self.x` — fine.
- *Test coverage:* MSE-threshold only; the dynamic-R branch is **not** exercised by `test_ekf.py`
  (test uses default `dynamic_R=0.0`). The harness config *does* exercise it (`phi_std:[0.0]` +
  `dynamic_R:[1.0,0.1]`, `ekf_and_pf_config.yml:15-20`).

---

## 2. `SPFPairedKalmanFilter` · `ekf_dualradio_filter.py:21` · ✅ Active

**Summary.** Dual-radio EKF over a **single shared craft θ** (dim_x=2, dim_z=2). Both radios observe
the same θ, distinguished by per-radio `radio_array_angle_offsets` (`theta-in-pis·π`). θ uses the
**full circle** `(-π,π]` (`fix_x` is just `pi_norm`, no front-half fold).

**State.** `x = [craft_θ, dθ/dt]`. `setup` → `[[0],[0]]` (135).

**Overrides.**
- `fix_x()` (55) — `x[0] = pi_norm(x[0])` (full circle, **no** range assert).
- `predict` (63) — identical structure to filter #1.
- `update(observation)` (76) — builds a **2×2 diagonal R** from `R_at_x(pi_norm(θ−offsetᵢ))` per
  radio; filterpy update with `paired_h`/`paired_hjac` + `residual`;
  `R = self.R if dynamic_R==0 else r·dynamic_R` (106).
- `observation(idx)` (114) — cached `all_observations[idx].reshape(2,1)` (vstack of `mean_phase["r0"]`,
  `["r1"]`, transposed), else live vstack of the two `mean_phase_segmentation`.
- `metrics` (128) — `dual_radio_mse_theta_metrics` vs `ds.craft_ground_truth_thetas`.
- `trajectory` (144) — own loop, records `craft_theta=x[0,0]`; skips update on non-finite obs.

**Tested by.** `test_ekf.py::test_paired_radio_ekf` (both `temp_file`), `mse_craft_theta < 0.005`;
passes `dynamic_R=False` → static-R branch (`False == 0` is True). Runs `run_and_plot_dualradio_EKF`.

**Analysis.** Near-line-for-line duplicate of #1 minus the front-fold (duplication candidate, not a
bug). *Issue considered:* `dynamic_R=False` arithmetic — `False == 0` → True → static branch, so the
test's `dynamic_R=False` works (✅ verified). The config exercises `dynamic_R:[1.0,0.1]` (line 31),
which takes the `r·dynamic_R` branch.

---

## 3. `SPFPairedXYKalmanFilter` · `ekf_dualradioXY_filter.py:43` · ✅ Active (⚠️ several subtleties)

**Summary.** The most complex EKF: dim_x=6 state `[tx_x, tx_y, dtx_x/dt, dtx_y/dt, rx_x, rx_y]`,
dim_z=2. Tracks the **transmitter XY position**; θ is *derived* inside `h()` from the relative TX-RX
geometry. The RX position (indices 4,5) is treated as **known** and re-injected from the dataset each
step. Uses 6×6 `Fxy_cached`/`Qxy_discrete_white_noise_cached` (`ekf_dualradioXY_filter.py:18-40`); the
4×4 white-noise block sits in the top-left (tx pos+vel), velocities for rx are zero.

**Overrides.**
- `fix_x()` (73) — **no-op** (the velocity-clip is commented out, line 74). ⚠️
- `predict` (82) — 6-D CV; only tx_x,tx_y integrate velocity (Fxy rows 2-5 are zero).
- `update(observation)` (96) — derives `target_theta = arctan2(rel_x, rel_y)` from current `x`,
  builds the 2×2 R from `R_at_x(target_theta−offsetᵢ)`, then filterpy update with `pairedXY_h`/`pairedXY_hjac`.
- `observation(idx)` (138) — **live `ds[idx]` only; NO cached branch** (unlike filters 1,2). ⚠️
  So on a non-`temp_file` dataset it still hits `ds[idx][…]["mean_phase_segmentation"]`.
- `metrics` (158) — `mse_craft_theta` (pred θ = `mu[0]`… see issue below) + `mse_xy`.
- `trajectory` (189) — see RX-reset below.

**The RX-reset-each-step mechanism (the subtlety to verify).** Per step (`trajectory`, 204-232):
1. `self.P[4:, 4:] = 0` (205) — zero the RX-position covariance block **before** predict.
2. `predict` (206).
3. `observation` (211).
4. `self.x[4], self.x[5] = true rx_pos` (213-214) — overwrite RX with the dataset's known RX position
   **before** the update.
5. `update` (227) — filterpy update; because `P[4:,4:]=0` and the Jacobian columns 4,5 are 0
   (see `filters.md` `pairedXY_hjacobian`: RX columns intentionally left zero), the update cannot move
   RX.
6. `self.x[4], self.x[5] = true rx_pos` **again** (229-230) — re-clamp RX after update.
7. `target_theta = pi_norm(arctan2(rel_x, rel_y))` recomputed (231-232), stored as `craft_theta`.

**Analysis (correctness).**
- *RX-reset is consistent and intentional:* the zeroed `P[4:,4:]`, the zeroed Jacobian columns 4,5,
  and the double re-injection together guarantee RX stays pinned to ground truth and contributes no
  covariance. This is *why* the missing RX Jacobian entries (documented as a non-bug in `filters.md`)
  never matter. ✅ verified by reading all three mechanisms.
- 🔍 **NEW — `mse_xy` is in mm² but `mse_craft_theta` mixes a θ MSE with no obvious bug; the
  `metrics` reads `x["mu"][0]` as θ** (159) — but `mu[0]` is `tx_x`, **not θ**. Wait: line 159
  `pred_theta = hstack([x["mu"][0] …])` takes element 0 of the 6-vec = **tx_x**, then compares it to
  `craft_ground_truth_thetas` via `torch_pi_norm_pi`. **This is comparing a tx_x position (mm) to an
  angle (rad).** The trajectory dict's `craft_theta` key (line 237) holds the *correct* derived θ, but
  `metrics` ignores it and reads `mu[0]` instead. So **`mse_craft_theta` from this filter's `.metrics`
  is computed against the wrong quantity** (tx_x position, not θ). Severity: **medium** —
  *silently wrong metric* ⚠️ no-crash. Reachability: `run_EKF_xy_dual_radio` (harness) and
  `test_pairedXY_radio_ekf` both call `.metrics`; the test only asserts `mse_craft_theta < 5` (a very
  loose bound the author flagged "we arent very good here"), so the wrong value still passes. Confidence:
  🔍 analysis — I read that `mu[0]`=tx_x and that `pred_theta` is built from `mu[0]`, but did not run it;
  **needs maintainer confirmation** whether comparing tx_x(mm) to θ(rad) is the intent (almost certainly
  not). The XY filter (#3) is also **excluded from the production config** (`run_EKF_xy_dual_radio` is
  commented out in `ekf_and_pf_config.yml:83-92`), lowering live impact.
- *`d²→0` division* in `pairedXY_hjacobian` (covered in `filters.md`): unguarded but TX rarely
  coincides with RX. Low.
- *`fix_x` no-op:* the velocity clip is dead (commented). The 6-D filter has no θ-wrap in state
  because θ is derived, so a no-op `fix_x` is fine here.

**Tested by.** `test_ekf.py::test_pairedXY_radio_ekf` (both `temp_file`), `dynamic_R=True`,
`noise_std=10`, asserts only `mse_craft_theta < 5`. ⚠️ Because `observation` has no cached branch, the
`temp_file=False` parametrization still reads live `ds[idx]` (works, just no speedup).

---

## 4. `PFSingleThetaSingleRadio` · `particle_single_radio_filter.py:18` · ✅ Active

**Summary.** Single-radio PF over `[θ, dθ/dt]` with **empirical-distribution likelihood**
(`theta_phi_to_p_vec` over a precomputed `get_empirical_dist(ds, rx_idx)` table). θ folded to front
half via `fix_particles_single` (fold + velocity-sign flip).

**Overrides.**
- `__init__` (23) — seeds `torch.Generator(0)`; caches `all_observations` (non-`temp_file`) and
  `cached_empirical_dist = get_empirical_dist(ds, rx_idx).T` (**transposed** so axis-0 = θ-bins, which
  `theta_phi_to_p_vec` requires — ✅ verified against `filters.md`).
- `observation(idx)` (32) — cached `all_observations[idx]` or live `mean_phase_segmentation`.
- `fix_particles` (37) — `fix_particles_single` (fold to [-π/2,π/2], flip dθ/dt).
- `predict` (41) — `θ += dt·dθ/dt`, then `add_noise` (default `[[0.1,0.001]]`). RNG.
- `update(z)` (47) — `weights *= theta_phi_to_p_vec(θ, z, emp_dist)`; `+= 1e-30`; normalize.
- `metrics` (56) — single-radio θ MSE.
- `trajectory(**kwargs)` (61) — `super().trajectory`, then per step `x["theta"]=mu[0]`, `x["P_theta"]=var[0]`.

**Tested by.** `test_particle_filter.py::test_single_theta_single_radio` (both `temp_file`),
`N=4096`, `mse_single_radio_theta < 0.05`. Also `plot_single_theta_single_radio`.

**Analysis.** Weight normalization is sound (`+1e-30` floor then `/sum`). ✅. The empirical table
transpose matches the `theta_phi_to_p_vec` contract.

---

## 5. `PFSingleThetaDualRadio` · `particle_dualradio_filter.py:30` · ✅ Active

**Summary.** Dual-radio PF over a shared `[craft_θ, dθ/dt]`. Likelihood combines both radios in a
single `@torch.jit.script` kernel `pf_single_theta_dual_radio_update` (`particle_dualradio_filter.py:14`),
which evaluates `theta_phi_to_p_vec` twice (per-radio offset) and multiplies. Full-circle θ via
`torch_pi_norm_pi`.

**Overrides.**
- `__init__` (31) — offsets from `theta-in-pis·π`; stacks the two empirical dists
  `vstack([get_empirical_dist(ds,0).T.unsqueeze(0), …(ds,1)…])` → `cached_empirical_dist[radio]`.
- `observation(idx)` (53) — cached `all_observations[idx]` (2-vec) or live concat of the two
  segmentations.
- `fix_particles` (64) — `torch_pi_norm_pi`.
- `predict` (67) — CV + `add_noise`.
- `update(z)` (73) — delegates to the jit kernel; **`self.weights = pf_single_theta_dual_radio_update(...)`**
  (returns normalized weights; floor `+1e-30` and `/sum` are inside the kernel, lines 26-27). ✅.
- `trajectory` (87) — `super()` + unpack `craft_theta=mu[0]`.

**Tested by.** `test_single_theta_dual_radio` (both `temp_file`), `N=4096`, `mse_craft_theta < 0.15`.

**Analysis.** Normalization is in the kernel and correct. ✅ The jit kernel takes `offsets` as a
tensor and indexes `offsets[0]/[1]` — fine.

---

## 6. `PFXYDualRadio` · `particle_dualradioXY_filter.py:12` · ✅ Active (⚠️ predict ordering smell)

**Summary.** PF tracking transmitter XY with a 5-D state `[θ, x, y, dx/dt, dy/dt]`. θ is **derived**
each step from the RX-TX geometry (`arctan2`), not propagated as an independent dim. Unique
**speed prior** `N(1.5, 3)` on `‖velocity‖` (`scipy.stats.norm(1.5,3).pdf`, line 20,88).

**Overrides.**
- `our_state(idx)` (29) — RX position `[rx_x, rx_y]` at idx from `cached_keys` (the only PF that
  overrides `our_state`; the base passes `our_state` into `predict`).
- `cached_empirical_dist(rx_idx)` (59) — `@cache`d method (⚠️ method-level `@cache` holds `self`
  alive in the cache — minor leak, works).
- `fix_particles` (63) — `torch_pi_norm_pi` on θ (cosmetic; θ is overwritten in predict anyway).
- `predict(dt, our_state, noise_std)` (66) — **(a)** `add_noise(particles, …)` over ALL dims,
  **(b)** `xy += dt·velocity`, **(c)** recompute `θ = arctan2(tx−rx)`. Signature order differs from
  siblings (`dt, our_state, noise_std`) but the base calls it by keyword (`filters.py:326-329`), so
  ✅ no mismatch.
- `update(z)` (76) — two `theta_phi_to_p_vec` (per-radio offset) × `speed_dist.pdf(‖vel‖)`; floor + normalize.
- `metrics` (94) — reads `mu[0]`=θ, `mu[[1,2]]`=xy directly (no `trajectory` override).
- **No `trajectory` override** → trajectory dicts are bare `{"mu","var"}`.

**Tested by.** `test_XY_dual_radio` (both `temp_file`), `N=4096`, `pos_err=50`, `vel_err=0.1`,
`mse_craft_theta < 0.25`. Also `plot_xy_dual_radio` (with `noise_std` zeroing θ-noise).

**Analysis.**
- 🔍 **NEW (low) — `add_noise` precedes θ-overwrite (predict 67-74).** `add_noise` perturbs *all 5*
  dims including θ (dim 0), but θ is immediately overwritten by the `arctan2` recompute (74), so the
  θ-noise is wasted. The run-harness/plot callers pass `noise_std[0]=0` anyway
  (`run_filters_on_data.py:477`, plot line 115), so it is harmless; but if a caller passed nonzero
  θ-noise it would silently have no effect. Not a correctness bug given current callers. Confidence:
  🔍 analysis. Severity: low.
- `speed_dist.pdf` returns a numpy array; `self.weights` is a torch tensor — `weights *= np_array`
  relies on torch's `__imul__` accepting a numpy array (works via implicit conversion). ✅ runs (test
  passes), though it materializes a numpy↔torch bridge each step.
- The XY PF is **commented out of the production config** (`run_PF_xy_dual_radio`,
  `ekf_and_pf_config.yml:77-79`), so it runs only via the test.

---

## 7. `PFSingleThetaSingleRadioNN` · `particle_single_radio_nn_filter.py:22` · ✅ Active

**Summary.** Single-radio PF identical to #4 except the likelihood is an **NN posterior** over θ-bins
rather than an empirical table. The model output is already a per-θ probability vector; `update` bins
each particle's θ (`theta_phi_to_bins`) and gathers the model probability.

**Overrides.**
- `__init__` (27) — two branches: **cached** (non-`temp_file`) runs `get_nn_inference_on_ds_and_cache`
  and stores `cached_model_inference["single"]`; **live** (`temp_file`) loads the model + config,
  `.eval()`, builds `model_datasets_config`/`model_keys_to_get` for on-the-fly inference. Heavy I/O
  (B2 checkpoint download via callees). The empirical-fn equality assert is **commented out** (36-39).
- `model_inference_at_observation_idx(idx)` (77) — cached slice, or live `model(collate(...))["single"].cpu()`
  under `no_grad`.
- `observation(idx)` (87) — `model_inference_at_observation_idx(idx)[rx_idx, 0]`.
- `fix_particles` (90) — `fix_particles_single` (front fold).
- `update(z)` (100) — `theta_bin = theta_phi_to_bins(θ, nbins=z.shape[0])`; `weights *= take(z, theta_bin)`;
  floor + normalize. (`z` is the model posterior, so `nbins = len(z)`.)
- `trajectory` (114) — `super()` + unpack θ/P_theta.

**Tested by.** No dedicated single-radio-NN test in `test_particle_filter.py` (the two
`test_single_theta_dual_radioNN` functions and the nnwrapper test exercise the **dual**-NN path).
⚠️ The single-NN class is exercised only via the harness (`run_PF_single_theta_single_radio_NN`,
config lines 43-58) — **no checked-in unit test**.

**Analysis.** Update math mirrors #4 with the NN posterior substituted; normalization sound. ✅.

---

## 8. `PFSingleThetaDualRadioNN` · `particle_dual_radio_nn_filter.py:30` · 🔴 Broken (live `breakpoint()`)

**Summary.** Dual-radio NN PF over `[craft_θ, dθ/dt]`. Wraps a `v5spfdataset_nn_wrapper` (`nn_ds`)
that serves paired NN inference; likelihood is the binned NN posterior. **`observation` contains a live
`breakpoint()`** that halts execution into pdb on the first step.

**Overrides.**
- `__init__` (31) — stores `nn_ds`, `self.absolute = nn_ds.absolute`, seeds RNG.
- `observation(idx)` (40) — 🔴 **`breakpoint()` at line 41**, then `return self.ds.get_inference_for_idx(idx)[0]["paired"][0]`.
- `fix_particles` (44) — `torch_pi_norm_pi`.
- `predict` (47) — CV + `add_noise`.
- `update(z)` (53) — bin + gather NN posterior; floor + normalize (same as #7).
- `metrics` (62) — `dual_radio_mse_theta_metrics` vs `craft_ground_truth_thetas` (relative) or
  `absolute_thetas.mean(axis=0)` (if `self.absolute`).
- `trajectory` (72) — `super()` + unpack `craft_theta`.

Module helper `cached_model_inference_to_absolute_north(ds, cached_model_inference)` (12) rotates each
radio's NN dist by RX heading into the absolute-north frame (`rotate_dist`); **hardcodes 65 bins**
(`reshape(-1, 65)`, line 22) and assumes 2 radios. Not called by the class itself (used by the
nn-wrapper plumbing).

**Reachability of the breakpoint (verified).** `ParticleFilter.trajectory` calls `self.observation(idx)`
**every step** (`filters.py:333`), and again if `debug=True` (line 348). So any live use halts. Two
reachers:
1. `run_PF_single_theta_dual_radio_NN` (`run_filters_on_data.py:416`) → builds the filter and calls
   `pf.trajectory` → first step hits the breakpoint. This function **is in the production config**
   (`ekf_and_pf_config.yml:59-75`, `absolute:[True,False]`).
2. `test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper` (line 228) constructs
   `PFSingleThetaDualRadioNN` directly and calls `pf.trajectory(... steps=n_steps)` (285) → also hits
   the breakpoint. **This test would hang in pdb** (under non-interactive pytest, `breakpoint()` raises
   `bdb.BdbQuit` or blocks on stdin) — confirming the filter is non-runnable as checked in.

**Analysis.** 🔴 **CONFIRMED P0** — live `breakpoint()` makes the entire dual-radio-NN path
non-runnable in any automated/batch context. Everything else in the class is sound. (Read-only — not
fixed.)

---

## Harness: `run_filters_on_data.py`

Batch orchestration: expand a YAML config into a Cartesian product of `(filter_fn, params)` jobs across
datasets, run them (serially in debug or via a multiprocessing `Pool`), and write per-run metrics to
**DynamoDB** (table `filter_metrics`). The companion `run_filters_on_data_b2.py` is the
production/cloud driver (see [`_inventory/filters.md`](../../_inventory/filters.md) — note: both
`*_b2.py` files import from a non-existent `spf.scripts.run_filters_on_data*` and are flagged
un-importable there).

### `run_filter_jobs(jobs, nparallel, debug=False, checkpoints_cache_dir=None, already_processed=[])` · `:626` · ✅ Active

**Summary.** The top-level executor. Partials `run_jobs_with_one_dataset` with the cache dir + skip
list, then either maps serially (`debug`) or runs a `Pool(nparallel)` of `imap_unordered`, with a tqdm
bar. Sets `torch.set_num_threads(1)` before the pool (line 642) to avoid oversubscription.

**Signature.** `jobs` (list of per-dataset job dicts from `generate_configs_to_run`); `nparallel`
(int, **required, no default**); `debug` (serial); `checkpoints_cache_dir`; `already_processed`
(list of `result_fn_without_workdir` to skip).

**Called from.**
- `__main__` (line 731) — 🔴 `run_filter_jobs(jobs)` **omits the required `nparallel`** → `TypeError`.
- `run_filters_on_data_b2.py` `main` — passes `nparallel=args.parallel` correctly (but that module is
  itself un-importable; see inventory).
- Tests: none call `run_filter_jobs` (tests call the `run_*` functions directly).

**Analysis.** 🔴 **#12 CONFIRMED** (`:731`): the direct-script entrypoint is broken — `nparallel` has
no default (`:627`) and `__main__` supplies only `jobs`. Severity P1; reachability = only the direct
`python -m spf.filters.run_filters_on_data` invocation (the working path is the B2 driver, which passes
it). Confidence ✅ verified (read both the signature and the call).

### `run_jobs_with_one_dataset(kwargs, checkpoints_cache_dir, already_processed=[])` · `:71` · ✅ Active

**Summary.** The per-dataset worker (one Pool task). Opens the dataset via `v5spfdataset_manager`,
iterates `kwargs["jobs"]`, downloads B2 checkpoints (for NN jobs), **skips** any job whose
`result_fn_without_workdir` is in `already_processed`, runs the filter fn, decorates each result with
`ds_fn`/`segmentation_version`/`precompute_cache`, and **writes to DynamoDB** via `table.put_item`.

**Side effects (heavy):** `boto3` DynamoDB resource + `put_item`; B2 downloads
(`b2_file_to_local_with_cache`); opens dataset (readahead); `float_to_decimal` (mutates result dicts
in place for DynamoDB). The file-system path (`use_file_system`) is **hardcoded `False`** (line 144),
so only the DynamoDB branch runs.

**Result schema (per result dict, written under DynamoDB item `{bucket: workdir, full_name:
result_fn_without_workdir, results: [<decimal-ized result dicts>]}`):** each `run_*` returns a list of
dicts containing at minimum `type`, `frequency`, `rx_wavelength_spacing`, the filter's hyperparams
(`phi_std`/`p`/`noise_std`/`dynamic_R` for EKF; `theta_err`/`theta_dot_err`/`N` for PF),
`metrics` (`{mse_*: float, runtime: float}`), and `routine`. NN runs add `checkpoint_fn`/`config_fn`
(and `absolute` for dual-NN). `run_jobs_with_one_dataset` then injects `ds_fn`,
`segmentation_version`, `precompute_cache`, and rewrites `checkpoint_fn`/`config_fn` back to the
original B2 paths (168-170).

### Job dispatch table `fn_key_to_fn` · `:502`

Maps config keys → `run_*` functions: `run_EKF_single_theta_single_radio`,
`run_EKF_single_theta_dual_radio`, `run_EKF_xy_dual_radio`, `run_PF_single_theta_single_radio`,
`run_PF_single_theta_dual_radio`, `run_PF_xy_dual_radio`, `run_PF_single_theta_single_radio_NN`,
`run_PF_single_theta_dual_radio_NN`. Each `run_*` constructs its filter, calls `.trajectory()` +
`.metrics()`, times it, and returns the result list. (One per concrete class; the XY PF/EKF runners
exist but are commented out of the live config.)

### `config_to_job_params(config)` · `:522` · ⚠️ `eval()` of config strings

**Summary.** Cartesian product of the config's value-lists into a list of param dicts. For each value
that is a `str`, it **`eval()`s it** (line 530, comment "TODO this might be dangerous"), catching only
`SyntaxError`. This is how `N: [128, 128 * 4, …]` (`ekf_and_pf_config.yml:34`) becomes integers —
the YAML stores `"128 * 4"` as a string and `eval` computes `512`.

**Analysis.** ⚠️ **#13 CONFIRMED** (`:530`): arbitrary `eval` of config strings. Reachability:
every `run_*` whose config has arithmetic-string values (all the `N:` lists) flows through here
(`config_to_jobs:547` → `generate_configs_to_run:578`). Severity P1 (code-exec via config file).
Confidence ✅ verified. Note: only `SyntaxError` is caught — a `NameError` from a non-numeric string
would propagate (but in practice the config strings are all arithmetic). The check at the end
(`len(jobs)==1 and len(jobs[0])==0 → []`) returns empty for an empty config.

### Other harness symbols (brief)
- `generate_configs_to_run` (569) — `os.makedirs(work_dir)`; loads YAML; `config_to_jobs`;
  `add_precompute_cache_to_job`; stamps `workdir`; seeds + shuffles jobs and datasets; expands a
  single `.txt` dataset-list into its lines. Returns the per-dataset job list.
- `add_precompute_cache_to_job` (551) — attaches the precompute cache by `segmentation_version` or by
  the `checkpoint_fn_and_segmentation_version` block (unpacking it into `checkpoint_fn` +
  `segmentation_version`); **raises `ValueError`** if neither key is present.
- `config_to_jobs` (542) — expands `list_config["runs"]` into `(fn, params)` jobs via the dispatch table.
- `args_to_str` (53) — canonical arg string; md5-hashes `*_fn` file args; **skips** `inference_cache`
  and `ds`. Feeds the DynamoDB `full_name` key (the de-dup identity).
- `float_to_decimal` (32) — recursive float→Decimal for DynamoDB; **mutates dicts/lists in place**.

---

## Pass 2 — which filters are actually selected (config + tests)

**By `ekf_and_pf_config.yml` (production):** EKF single (×2 blocks: static-R + dynamic-R), EKF dual
(×2 blocks), PF single (empirical), PF dual (empirical), PF single-NN, **PF dual-NN** (which hits the
breakpoint). **Commented out / NOT run:** `run_EKF_xy_dual_radio` (filter #3) and
`run_PF_xy_dual_radio` (filter #6). So the two XY filters are config-dead and only reachable via tests.

**By tests:**
- `test_ekf.py`: filters 1, 2, 3 (both `temp_file` branches) with loose MSE bounds.
- `test_particle_filter.py`: filters 4, 5, 6 (both `temp_file`), plus the dual-NN filter 8 via
  `run_PF_single_theta_dual_radio_NN` (two identically-named `test_single_theta_dual_radioNN` — the
  second shadows the first) **and** directly via `test_single_theta_dual_radioNN_nnwrapper`. **Filter 7
  (single-radio NN) has no dedicated test.**
- ⚠️ Tests 110-225 are **two functions with the same name** `test_single_theta_dual_radioNN`; Python
  keeps only the second definition, so the first is silently dropped (both bodies are near-identical,
  so no coverage loss, but the duplication is a smell). Both reach filter 8's breakpoint.

---

## NEW issues found (this pass) — read-only, not fixed

| # | Location | Severity | Conf | Description |
|---|---|---|---|---|
| N1 | `ekf_dualradioXY_filter.py:159` | 🟠 medium ⚠️ silently-wrong | 🔍 | `metrics.mse_craft_theta` builds `pred_theta` from `mu[0]` = **tx_x (mm)**, not the derived θ — compares a position to an angle. The correct θ lives in the `craft_theta` trajectory key but `metrics` ignores it. Reached by `run_EKF_xy_dual_radio` + `test_pairedXY_radio_ekf` (passes due to loose `<5` bound). XY EKF is config-dead. **No-crash; verify intent before any fix.** |
| N2 | `particle_dualradioXY_filter.py:67-74` | 🟢 low | 🔍 | `add_noise` perturbs θ (dim 0) but θ is overwritten by `arctan2` recompute → θ-noise wasted. Harmless since all callers pass `noise_std[0]=0`, but a nonzero θ-noise would silently no-op. |
| N3 | `tests/test_particle_filter.py:110 & 170` | 🟢 low (test smell) | ✅ | Two `def test_single_theta_dual_radioNN(...)` with identical names; the first is shadowed/never collected by pytest. |

(Previously-documented items reconfirmed: the XY `d²→0` unguarded division (`filters.py:179`); the
`.copy()`-on-torch-tensor bug in `ParticleFilter.trajectory:346` for `return_particles=True`; the
dual-radio `dynamic_R=True` bool-as-1 arithmetic — works but smells. See `filters.md`.)

## KNOWN_ISSUES cross-check
- **#2 (P0)** — 🔴 **CONFIRMED.** Live `breakpoint()` at `particle_dual_radio_nn_filter.py:41` in
  `observation()`; reached every step by `ParticleFilter.trajectory:333` via both
  `run_PF_single_theta_dual_radio_NN` (in production config) **and** the direct
  `test_single_theta_dual_radioNN_nnwrapper` test. ✅ verified.
- **#12 (P1)** — 🔴 **CONFIRMED.** `run_filters_on_data.py:731` calls `run_filter_jobs(jobs)` without
  the required `nparallel` (no default at `:627`) → `TypeError`. Direct-script entry broken; B2 driver
  passes it. ✅ verified.
- **#13 (P1)** — ⚠️ **CONFIRMED.** `config_to_job_params` `eval()`s str config values
  (`run_filters_on_data.py:530`, self-flagged dangerous, catches only `SyntaxError`). Reached by every
  arithmetic-string config value. ✅ verified.
- **#31** — 🔴 **CONFIRMED (refers to `filters.py:346`, base engine).** `ParticleFilter.trajectory`
  does `self.particles.copy()` on a **torch tensor** (no `.copy()` method → `AttributeError`) when
  `return_particles=True`. No concrete filter sets `return_particles=True` (all `run_*`/plot/test
  callers pass `False`), so it is **dead on the live path**; would crash only on the debug/inspection
  path. ✅ verified. (Lives in the base file, documented in `filters.md`; re-confirmed here because the
  concrete PFs all route through this engine.)

---

## Cluster summary — the 9 documented units

| # | Unit | Status | Selected by | Direct test | Top finding |
|---|---|---|---|---|---|
| 1 | `SPFKalmanFilter` (EKF single) | ✅ | config + test | ✅ MSE<0.03 | front-fold `fix_x`; dynamic-R formula differs from dual; live path safe. |
| 2 | `SPFPairedKalmanFilter` (EKF dual) | ✅ | config + test | ✅ MSE<0.005 | shared-θ, full circle; `dynamic_R=False`→static R works. |
| 3 | `SPFPairedXYKalmanFilter` (EKF XY) | ✅ (⚠️) | **test only** (config-dead) | ✅ MSE<5 (loose) | **N1: `mse_craft_theta` reads `mu[0]`=tx_x not θ** ⚠️ silently-wrong; RX re-injected each step (consistent). |
| 4 | `PFSingleThetaSingleRadio` | ✅ | config + test | ✅ MSE<0.05 | empirical likelihood; table transposed correctly. |
| 5 | `PFSingleThetaDualRadio` | ✅ | config + test | ✅ MSE<0.15 | jit dual-radio update kernel; normalize inside kernel. |
| 6 | `PFXYDualRadio` | ✅ (⚠️) | **test only** (config-dead) | ✅ MSE<0.25 | speed prior `N(1.5,3)`; N2 θ-noise wasted; numpy↔torch weight mul. |
| 7 | `PFSingleThetaSingleRadioNN` | ✅ | config only | ❌ no test | NN posterior likelihood; **no dedicated unit test**. |
| 8 | `PFSingleThetaDualRadioNN` | 🔴 | config + test | (hits breakpoint) | **#2 live `breakpoint()` @ line 41** — non-runnable. |
| H | `run_filters_on_data.py` harness | ✅ (⚠️) | n/a | indirect | **#12** `__main__` missing `nparallel`; **#13** config `eval()`; DynamoDB result schema. |

**Cross-cutting themes.**
1. **The two XY filters (3, 6) are config-dead** — commented out of `ekf_and_pf_config.yml` and reached
   only by tests with deliberately loose MSE bounds; this is exactly where the silently-wrong metric
   (N1) hides without surfacing as a test failure.
2. **Sign/frame conventions stay implicit and load-bearing** — every filter negates the antenna
   spacing on construction (never asserted vs `mean_phase` sign), single-radio folds to [-π/2,π/2]
   while dual/craft uses full circle, and the XY filters derive θ from `arctan2(rel_x, rel_y)` (+y
   frame). These are the documented #1 bug class.
3. **The breakpoint (#2) is reached by *both* a production config path and a checked-in test**, making
   the dual-radio-NN filter the single highest-priority blocker; the harness `__main__` (#12) and
   `eval()` (#13) are real but lower-impact because the canonical entry is the (separately broken) B2
   driver.
