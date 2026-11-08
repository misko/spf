# Reference: `spf/filters/filters.py`

Shared filter framework for SPF: the abstract `SPFFilter` base, the `ParticleFilter` base + its
torch kernels, the analytic EKF **observation models + Jacobians**, the motion-model helpers, and
the empirical-likelihood lookup path. The 8 concrete filter classes live in sibling files
(`ekf_*_filter.py`, `particle_*_filter.py`) and plug into these bases; this doc documents the
shared substrate and gives a plug-in matrix, not full per-class deep dives.

Passes 1-3 complete (read-only). Line numbers verified against source on 2026-06-21.
Units/frames per [`conventions.md`](../../00_concepts/conventions.md); observation relation per
[`beamforming_primer.md`](../../00_concepts/beamforming_primer.md).

---

## State & observation models

**The observation relation** (the "whole game", per
[`beamforming_primer.md:15`](../../00_concepts/beamforming_primer.md)):

```
φ = 2π · (d/λ) · sin(θ − offset)
```

implemented in `h_phi_observation_from_theta_state` (`filters.py:49`). `d/λ` =
`antenna_spacing_in_wavelengths`; `offset` = `radio_array_angle_offset` (a per-radio array
orientation, `theta-in-pis · π`, default 0). θ is the angle-of-arrival in radians.

> ⚠️ **Sign convention.** Every concrete filter negates the spacing on construction:
> `antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]`
> (e.g. `ekf_single_radio_filter.py:31`, `ekf_dualradio_filter.py:33`). So the effective
> `d/λ` passed to `h()` is **negative**. This flips the sign of φ vs the bare primer formula; it
> must match the dataset's `mean_phase` sign convention. Units/frame: φ in radians (wrapped to
> (-π,π] via `pi_norm`); θ in radians.

**State vector layouts** (the first element is always θ or derived-θ; see the plug-in matrix):

| Layout | Dim | Used by `h()` | Filters |
|---|---|---|---|
| `[θ, dθ/dt]` | 2 | `single_h…`, `paired_h…` | EKF single, EKF dual, all `[θ,dθ/dt]` PFs |
| `[tx_x, tx_y, dtx_x, dtx_y, rx_x, rx_y]` | 6 | `pairedXY_h…` (derives θ from rel-x/y) | EKF XY |
| `[θ, x, y, dx/dt, dy/dt]` | 5 | none (PF derives θ from geometry, empirical likelihood) | PF XY |

For the XY observation model, θ is reconstructed from relative TX-RX geometry
`target_theta = pi_norm(arctan2(rel_x, rel_y))` where `rel_x = x[0]-x[4]`, `rel_y = x[1]-x[5]`
(`filters.py:139-140`), then fed through the same `h()`.

**Motion model.** Constant-velocity (CV): `F = [[1, dt],[0,1]]` (`F_cached`, line 18), process
noise `Q = Q_discrete_white_noise(dim=2, dt, var=noise_std)` (`Q_discrete_white_noise_cached`,
line 13). The same `F`/`Q` describe both the EKF predict (`x ← F x`, `P ← F P Fᵀ + Q`) and the
PF predict (`particles[:,0] += dt·particles[:,1]` then Gaussian `add_noise`). XY filters use a 6×6
`F`/`Q` defined in `ekf_dualradioXY_filter.py` (`Fxy_cached`/`Qxy_discrete_white_noise_cached`),
not these 2×2 helpers.

---

## Observation models & Jacobians

### `h_phi_observation_from_theta_state(x, antenna_spacing_in_wavelengths, radio_array_angle_offset=0)` · `filters.py:49` · ✅ Active

**Summary.** Core analytic observation model: maps a `[θ, dθ/dt]` state to the expected phase
difference φ for one radio array via `φ = sin(θ − offset)·2π·(d/λ)`. The single building block all
EKF `h()` variants compose.

**Signature**
- Inputs: `x` (np `(2,1)`, `x[0,0]`=θ rad) ; `antenna_spacing_in_wavelengths` (d/λ, dimensionless,
  **negative** in practice — see sign note) ; `radio_array_angle_offset` (rad, default 0).
- Returns: np `(1,)` — φ in radians (un-wrapped; can exceed ±π for large d/λ).
- Side effects: `assert x.ndim==2 and shape (2,1)` (raises `AssertionError` on bad shape).

**Called from** (Pass 2)
- `single_h_phi_observation_from_theta_state` (line 88), `paired_h…` (lines 103,108),
  `pairedXY_h…` (lines 144,149) — all compose it. Indirectly the EKF `update()`/`trajectory()`
  call sites (via filterpy `ExtendedKalmanFilter.update`'s `Hx` callback).
- Fan-in: 3 direct wrappers + the EKF debug paths (`ekf_single_radio_filter.py:139`).

**Contract.** Pre: `x` is `(2,1)`; only `x[0,0]` is read. Post: returns a length-1 array. Pure
aside from the assert. Invariant: paired with `hjacobian_…` which is its exact analytic derivative.

**Analysis.** *Correctness:* derivative `d/dθ [sin(θ−off)·c] = cos(θ−off)·c` matches
`hjacobian_…` (line 76) — verified. No wrap inside; callers wrap the *residual* in φ-space via
`residual` (line 22). *Issues:* none in the function. Output is unwrapped φ — fine because the EKF
uses `residual = pi_norm(a−b)` for the innovation (line 79,105).

### `hjacobian_phi_observation_from_theta_state(x, …, offset=0)` · `filters.py:69` · ✅ Active
Jacobian `∂φ/∂x = [cos(θ−off)·2π·(d/λ), 0]`, shape np `(1,2)`. The `dθ/dt` column is 0 (velocity
does not affect the instantaneous φ). Asserts `(2,1)` shape. Exact derivative of `h_phi_…` —
verified by hand. Pure.

### `single_h_phi_observation_from_theta_state(x, antenna_spacing_in_wavelengths)` · `filters.py:86` · ✅ Active
Single-radio wrapper around `h_phi_…` (offset fixed at 0). **Asserts `θ ∈ [-π/2, π/2]`**
(line 87) — front-half constraint resolving the arcsin front/back ambiguity (primer §28-30).
**Called from** `SPFKalmanFilter.update` (via `partial`, `ekf_single_radio_filter.py:76`) and its
debug path (line 139). ⚠️ The assert can raise **mid-trajectory** if the EKF state drifts outside
[-π/2,π/2] before `fix_x` reduces it — but the single-radio `predict`/`update` both call `fix_x`
(which `reduce_theta_to_positive_y` + asserts the same range, `ekf_single_radio_filter.py:46-48`),
so on the live path θ is reduced first. Confidence: medium that no live crash; the double assert is
redundant.

### `single_hjacobian_phi_observation_from_theta_state(x, …)` · `filters.py:91` · ✅ Active
Single-radio Jacobian wrapper; same `θ∈[-π/2,π/2]` assert; delegates to `hjacobian_phi_…`. Used as
the `HJacobian` callback in `SPFKalmanFilter.update` (`ekf_single_radio_filter.py:72`).

### `paired_h_phi_observation_from_theta_state(x, …, radio_array_angle_offsets)` · `filters.py:98` · ✅ Active
Dual-radio `h()`: `vstack` of two `h_phi_…` calls, one per `radio_array_angle_offsets[0/1]`,
producing np `(2,1)` (one φ per radio) from a **single shared θ**. No range assert (dual filter
allows full (-π,π]). Used by `SPFPairedKalmanFilter.update` (`ekf_dualradio_filter.py:101`).
Pure.

### `paired_hjacobian_phi_observation_from_theta_state(x, …, offsets)` · `filters.py:117` · ✅ Active
Dual-radio Jacobian: `vstack` of two `hjacobian_phi_…` → np `(2,2)`. Row *i* = `[cos(θ−offᵢ)·c, 0]`.
Exact derivative of `paired_h_…`. Used as `HJacobian` in `SPFPairedKalmanFilter.update`
(`ekf_dualradio_filter.py:96`). Pure.

### `pairedXY_h_phi_observation_from_theta_state(x, …, offsets)` · `filters.py:136` · ✅ Active
**XY-state `h()`** (dim_x=6). Derives `θ = pi_norm(arctan2(rel_x, rel_y))` from relative TX-RX
position (`rel_x = x[0]-x[4]`, `rel_y = x[1]-x[5]`), builds a synthetic `[θ,0]` 2-state, then runs
the two-radio `h_phi_…` → np `(2,1)`. Used by `SPFPairedXYKalmanFilter.update`
(`ekf_dualradioXY_filter.py`). Pure. *Frame note:* `arctan2(rel_x, rel_y)` (x first) → bearing
measured from +y axis (the SPF craft/array frame), consistent with the primer's θ-from-+y
convention.

### `pairedXY_hjacobian_phi_observation_from_theta_state(x, …, offsets)` · `filters.py:158` · ✅ Active
**XY Jacobian via chain rule** → np `(2,6)`. Computes `∂φ/∂θ` (from the two-radio Jacobian rows)
times `∂θ/∂rel_x = rel_y/d²` and `∂θ/∂rel_y = -rel_x/d²` (`d² = rel_x²+rel_y²`, line 179), filling
only columns 0,1 (tx_x, tx_y); columns 2-5 (velocities, rx_x, rx_y) are zero.

**Analysis (correctness).** *Verified:* `∂/∂x arctan2(x,y) = y/(x²+y²)`, `∂/∂y arctan2(x,y) =
-x/(x²+y²)` — the standard derivative of `arctan2(x,y)` w.r.t. first/second arg; matches lines
180-181. The chain `∂φ/∂tx_x = (∂φ/∂θ)(∂θ/∂rel_x)(∂rel_x/∂tx_x=1)` — correct (line 185).
*Issues:* **(a)** `d²→0` division when TX≈RX (line 179) → `inf`/`nan` Jacobian, no guard. Severity:
low (TX rarely coincides with RX). **(b)** columns for rx_x (4), rx_y (5) are left 0 even though θ
depends on them (`∂θ/∂rx_x = -∂θ/∂rel_x`); this is **intentional** because the XY filter treats RX
position as *known* and re-injects it each step (`ekf_dualradioXY_filter.py:213-214`), zeroing its
covariance — so the missing RX Jacobian entries never matter. Confidence: high. Pure.

---

## Motion-model & helper functions

### `F_cached(dt)` · `filters.py:18` · ✅ Active
`@cache`d 2×2 CV transition `[[1,dt],[0,1]]`. Inputs `dt` (float). Returns np `(2,2)`. Side effect:
unbounded functools cache keyed on `dt`. Used by every 2-state EKF `predict`
(`ekf_single_radio_filter.py:56`, `ekf_dualradio_filter.py:64`). ⚠️ Returned array is **shared** —
callers must not mutate it in place (they don't; they `np.dot` into fresh arrays).

### `Q_discrete_white_noise_cached(**kwargs)` · `filters.py:13` · ✅ Active
`@cache`d wrapper over filterpy `Q_discrete_white_noise`. Kwargs must be hashable (they are:
`dim=2, dt=float, var=float`). Returns the process-noise Q matrix. Same shared-array caveat as
`F_cached`. Used by both 2-state EKF predicts.

### `residual(a, b)` · `filters.py:22` · ✅ Active
φ-space residual `pi_norm(a-b)` — wrap-aware innovation so the EKF takes the shorter arc around the
±π discontinuity. Passed as the `residual=` callback to filterpy `ExtendedKalmanFilter.update`
(`ekf_single_radio_filter.py:79`, `ekf_dualradio_filter.py:105`). Pure. **This is the only wrap
handling in the EKF innovation** — critical for correctness around ±π.

### `single_radio_mse_theta_metrics(trajectory, ground_truth_thetas)` · `filters.py:259` · ✅ Active
MSE of predicted θ vs ground truth **reduced to +y** (`reduce_theta_to_positive_y`) and wrapped
(`torch_pi_norm_pi`). Returns `{"mse_single_radio_theta": float}` or `{}` if trajectory empty.
Asserts `gt.shape[0] >= pred.shape[0]` then truncates GT to pred length, and `ndim==1`. Reads
`x["theta"]` from each trajectory entry. Used by `.metrics()` of the single-radio EKF/PF/NN filters.

### `dual_radio_mse_theta_metrics(trajectory, craft_ground_truth_thetas)` · `filters.py:278` · ✅ Active
MSE of predicted `craft_theta` vs craft GT, wrapped. Returns `{"mse_craft_theta": float}`. Same
shape asserts. ⚠️ **No empty-trajectory guard** (unlike the single-radio version) — `np.hstack([])`
on an empty list raises. Severity low (callers always produce a non-empty trajectory). Reads
`x["craft_theta"]`. Used by the dual-radio filters' `.metrics()`.

---

## `SPFFilter` base class · `filters.py:193` · ✅ Active (base)

**Summary.** Abstract template defining the filter lifecycle: `setup → (predict → update →
posterior_to_mu_var) per dataset index → trajectory`. Stores `self.ds`. All concrete methods are
no-op stubs **meant to be overridden**; the only base method with real logic is `trajectory`
(line 242), and **even that is overridden by every concrete class** (each EKF/PF defines its own
`trajectory`), so the base `SPFFilter.trajectory` is effectively **a documentation skeleton, never
executed on a live path**. 🧪 (base loop is dead in practice).

**Methods** (all `filters.py`):
- `__init__(ds)` (194) — sets `self.ds`. ✅
- `our_state(idx)` (201) — known RX state; base returns `None`. Overridden by PF XY (`our_state` →
  RX position). ✅ stub
- `predict(our_state, dt, noise_std)` (208) — base no-op. Every filter overrides. ✅ stub
- `update()` (215) — base no-op. ⚠️ base signature `update(self)` takes **no args**, but
  `SPFFilter.trajectory` (line 252) calls `self.update(prior=…, observation=…)` and every concrete
  override has a different signature (`update(self, observation)` / `update(self, z)`). The base
  `trajectory`+`update` pair is internally inconsistent — further evidence the base `trajectory` is
  never run. 🧪
- `observation(idx)` (222) — base no-op. Every filter overrides. ✅ stub
- `metrics(trajectory)` (229) — base no-op. ✅ stub
- `setup(initial_conditions)` (232) — base no-op. ✅ stub
- `posterior_to_mu_var(posterior)` (235) — returns `{"var": None, "mu": None}`. ✅ stub
- `trajectory(initial_conditions={}, dt=1.0, noise_std=0.01)` (242) — generic
  setup→predict→update→posterior loop over `range(len(self.ds))`. **Overridden by all concrete
  classes**; not reached in production. ⚠️ mutable default `initial_conditions={}` (shared dict)
  — harmless since never mutated. 🧪

**Called from / inheritance** (Pass 2): `SPFFilter` is a mixin base for all 3 EKFs (via
multiple inheritance `class X(ExtendedKalmanFilter, SPFFilter)`,
e.g. `ekf_single_radio_filter.py:19`) and the parent of `ParticleFilter`. The EKFs use it only for
`self.ds` + the metric-helper conventions; they override every lifecycle method. The run harness
(`run_filters_on_data.py:184-472`) calls each concrete filter's `.trajectory()` + `.metrics()`.

---

## `ParticleFilter` base class · `filters.py:292` · ✅ Active (base)

**Summary.** The **live** PF engine. Subclass of `SPFFilter`. Provides the seeded Gaussian
init + per-step `predict → fix_particles → observation → (update + resample) → estimate` loop in
`trajectory` (line 297). Subclasses supply `predict`/`update`/`observation`/`fix_particles`/
`our_state`. This `trajectory` *is* executed (every PF's own `trajectory` calls
`super().trajectory(**kwargs)`).

### `ParticleFilter.fix_particles()` · `filters.py:294`
Base hook returning `self.particles` unchanged; overridden by every PF (theta folding/wrapping).

### `ParticleFilter.trajectory(mean, std, N=128, noise_std=None, return_particles=False, debug=False, seed=0, steps=None, progress_bar=False)` · `filters.py:297` · ✅ Active

**Signature**
- Inputs: `mean` (torch `(1,D)`), `std` (torch `(1,D)`) — Gaussian init for particles; `N` particle
  count; `noise_std` (torch `(1,D)` or None → filter default); `seed` (int, RNG); `steps`
  (None → `len(ds)`); `return_particles`/`debug`/`progress_bar` flags.
- Returns: `list[dict]` with `{"var", "mu"}` per step (+`"particles"` if `return_particles`,
  +`"observation"` if `debug`). `mu`,`var` are torch `(D,)`.
- Side effects: creates `self.generator` (seeded `torch.Generator`), `self.particles`,
  `self.weights`; mutates them each step; consumes seeded RNG; optional tqdm bar.

**Loop** (lines 324-349): per idx — `predict(dt=1.0, …)`; `fix_particles()`; `z = observation(idx)`;
**if `z.isfinite().all()`**: `update(z)`, and if `neff(weights) < N/2` systematic-resample;
`mu,var = estimate(particles, weights)`; append.

**Contract.** Pre: subclass defines `predict`/`update`/`observation`/`fix_particles`; `observation`
returns a torch tensor exposing `.isfinite()`. Post: returns one dict per step (length = `steps` or
`len(ds)`). Resampling resets weights uniform (`resample_from_index`). Invariant: `weights` sums to
1 after each `update` (subclass responsibility) and after resample.

**Tested by.** `tests/test_particle_filter.py` — `test_single_theta_single_radio` (asserts
`mse_single_radio_theta < 0.05`), `test_single_theta_dual_radio` (`mse_craft_theta < 0.15`),
`test_XY_dual_radio` (`< 0.25`), `test_single_theta_dual_radioNN_nnwrapper` (direct
`pf.trajectory(... steps=n_steps)`). All driven through `run_PF_*` wrappers, both `temp_file`
branches.

**Analysis.**
- *Correctness:* seeded → deterministic. `dt=1.0` is **hard-coded** in the predict call (line 327)
  — the `dt` argument is *not* a parameter of PF `trajectory`, so PFs always step at dt=1 regardless
  of real time spacing. Severity: low/by-design (datasets are unit-step). Confidence: high.
- *Issue (line 346):* `trajectory[-1]["particles"] = self.particles.copy()` — `self.particles` is a
  **torch tensor**, which has no `.copy()` method (torch uses `.clone()`). So `return_particles=True`
  raises `AttributeError`. Severity: medium (only on the debug/inspection path; no test sets
  `return_particles=True` — all pass `return_particles=False`). Confidence: high. (Read-only —
  not fixed.)
- *Issue (line 325):* `range(len(self.ds) if steps is None else steps)` re-resolves `steps` even
  though `steps` was already defaulted to `len(self.ds)` at line 321-322 — redundant but harmless.
- *Issue (line 348):* `debug=True` calls `self.observation(idx)` a **second** time (already called
  at 333) — for the NN/live filters this re-runs inference; wasteful, and for the
  🔴 `PFSingleThetaDualRadioNN` it would hit the `breakpoint()` again. Severity: low.

---

## Torch PF kernels (`@torch.jit.script` unless noted)

### `create_gaussian_particles_xy(mean, std, N, generator)` · `filters.py:353` · ✅ Active
(jit decorator commented out — runs eager.) Samples `mean + randn(N,D)*std` using the seeded
generator. Asserts `mean`,`std` are `(1,D)`. RNG side effect. Returns torch `(N,D)`. Called once
per `ParticleFilter.trajectory` (line 311).

### `theta_phi_to_bins(theta_phi, nbins)` · `filters.py:362` · ✅ Active (jit)
Maps an angle in (-π,π] to a histogram bin `(nbins·(θ+π)/2π) mod nbins`. The `isinstance(...,float)`
branch (line 363) is **dead under jit** (jit sees a Tensor type), but the tensor branch is correct.
Used by `theta_phi_to_p_vec` and the NN PFs' bin lookups.

### `theta_phi_to_p_vec(thetas, phis, full_p)` · `filters.py:369` · ✅ Active (jit) — **empirical-likelihood core**
**Summary.** Vectorized empirical-likelihood lookup: bins `thetas` (rows) and `phis` (cols) of a
2-D probability table `full_p` `(nthetabins, nphibins)` and gathers `p(θ,φ)` per particle.
Implements `p = full_p[theta_bin, phi_bin]` via `torch.take(full_p[:, phi_bin], theta_bin)`.
- Inputs: `thetas` torch `(N,)` (particle θ, rad); `phis` scalar/tensor (observed φ, rad);
  `full_p` torch `(Tbins, Pbins)`. Returns torch `(N,)` likelihoods. Pure.
- **Called from:** `PFSingleThetaSingleRadio.update` (`particle_single_radio_filter.py:48`) and
  `pf_single_theta_dual_radio_update` (`particle_dualradio_filter.py:16,21`). The empirical-dist PFs
  multiply `weights *= theta_phi_to_p_vec(...)`. Fan-in: 2 (×2 radios in dual).
- *Analysis:* `torch.take(full_p[:, phi_bin], theta_bin)` — `full_p[:, phi_bin]` selects the phi
  column (length Tbins) then `take(theta_bin)` indexes by flat θ-bin. Correct because the sliced
  column is 1-D. Verified. Note `full_p` is passed **transposed** by callers
  (`get_empirical_dist(ds, rx).T`), so the table's axis-0 is θ-bins as required.

### `theta_phi_to_p(theta, phi, full_p)` · `filters.py:409` · 🧪 Dead
Scalar variant of `_p_vec`. **Zero callers** anywhere in the repo (grep confirms `theta_phi_to_p`
appears only at its definition; all live uses are `theta_phi_to_p_vec`). Dead code. ✅ math is
correct but unreferenced.

### `add_noise(particles, noise_std, generator)` · `filters.py:376` · ✅ Active
(jit commented out.) **In-place** `particles[:] += randn(...)*noise_std`. Mutates `particles`; RNG.
Returns `None`. Called by every PF `predict` (e.g. `particle_single_radio_filter.py:45`).

### `resample_from_index(particles, weights, indexes)` · `filters.py:383` · ✅ Active (jit)
**In-place** systematic resample: `particles[:] = particles[indexes]`, weights reset uniform then
re-normalized. Mutates both args. Called from `ParticleFilter.trajectory:340` when `neff < N/2`,
fed indices from filterpy `systematic_resample`.

### `neff(weights)` · `filters.py:390` · ✅ Active (jit)
Effective sample size `1/Σw²`. Used at line 338 as the resample trigger. Pure.

### `weighted_mean(inputs, weights)` · `filters.py:395` · ✅ Active (jit)
`Σ(w·x)/Σw` along dim 0. Helper for `estimate`. Pure.

### `estimate(particles, weights)` · `filters.py:400` · ✅ Active (jit)
Returns `(mean, var)` = weighted mean and weighted variance of particles (weights reshaped to
`(N,1)`). Called once per step (`trajectory:342`). Pure. Var is the *weighted second central
moment* (not bias-corrected) — fine as a spread estimate.

### `fix_particles_single(particles)` · `filters.py:418` · ✅ Active (jit)
**Folds θ into [-π/2, π/2] and flips dθ/dt sign** on reflected particles (the single-radio
front/back fold). In-place mutation; `while`-loop until all `|θ|≤π/2`. Used by the single-radio
PFs' `fix_particles` (`particle_single_radio_filter.py:38`,
`particle_single_radio_nn_filter.py`). *Analysis:* the reflection `θ ← sign(θ)·π − θ` correctly
maps e.g. `θ=2` to `π−2`; the velocity sign flip is the documented "required" correction so a
particle bouncing off the ±π/2 boundary reverses θ-velocity. The `while` loop terminates because
each fold halves the excess (verified for `|θ|<3π/2`; for pathological huge θ it still converges).
Pure-ish (mutates input).

---

## The 8 concrete filter classes — plug-in matrix

How each concrete class (sibling files) overrides the base hooks. Columns are the base methods;
cells cite the override's behavior.

| # | Class · file | base | `predict` | `update` (likelihood) | `observation` | `fix_*` | `trajectory` | metrics |
|---|---|---|---|---|---|---|---|---|
| 1 | `SPFKalmanFilter` `ekf_single_radio_filter.py:19` | EKF+SPFFilter | CV `F_cached`/`Q…`, `fix_x` | filterpy EKF w/ `single_h`/`single_hjac`, `residual`, dynamic R | `mean_phase[r{idx}]` or live `mean_phase_segmentation` | `fix_x`: reduce→+y, assert ∈[-π/2,π/2] | own (skip update on non-finite) | `single_radio_mse_theta` |
| 2 | `SPFPairedKalmanFilter` `ekf_dualradio_filter.py:21` | EKF+SPFFilter | CV, `fix_x`=`pi_norm` | filterpy EKF w/ `paired_h`/`paired_hjac`, per-radio R matrix | vstack r0,r1 `mean_phase` or live | `fix_x`: `pi_norm` only | own | `dual_radio_mse_theta` |
| 3 | `SPFPairedXYKalmanFilter` `ekf_dualradioXY_filter.py:43` | EKF+SPFFilter | 6-D CV `Fxy`/`Qxy` | filterpy EKF w/ `pairedXY_h`/`pairedXY_hjac` | live `ds[idx]` only (no cached branch ⚠️) | `fix_x` no-op (commented) | own: re-inject known RX pos, zero RX covar each step | craft-θ MSE + XY MSE |
| 4 | `PFSingleThetaSingleRadio` `particle_single_radio_filter.py:18` | ParticleFilter | CV + `add_noise` | `weights *= theta_phi_to_p_vec(θ, z, emp_dist)` | `mean_phase` or live | `fix_particles_single` (fold+flip) | `super()` + unpack mu[0]→θ | `single_radio_mse_theta` |
| 5 | `PFSingleThetaDualRadio` `particle_dualradio_filter.py:30` | ParticleFilter | CV + `add_noise` | jit `pf_single_theta_dual_radio_update`: ×2 `theta_phi_to_p_vec` w/ offsets | vstack r0,r1 or live | `torch_pi_norm_pi` | `super()` + mu[0]→craft_θ | `dual_radio_mse_theta` |
| 6 | `PFXYDualRadio` `particle_dualradioXY_filter.py:12` | ParticleFilter | integrate xy by velocity, recompute θ from RX/TX geometry + noise | ×2 empirical likelihood × **speed prior `N(1.5,3)`** | 2-vec `mean_phase` or live | `torch_pi_norm_pi` | base (no override) → returns raw mu/var | craft-θ + XY MSE |
| 7 | `PFSingleThetaSingleRadioNN` `particle_single_radio_nn_filter.py:22` | ParticleFilter | CV + `add_noise` | `weights *= NN_posterior[theta_bin]` | NN model inference (cached or live `model(z)`) | `fix_particles_single` | `super()` + unpack θ | `single_radio_mse_theta` |
| 8 | `PFSingleThetaDualRadioNN` `particle_dual_radio_nn_filter.py:30` 🔴 | ParticleFilter | CV + `add_noise` | `weights *= NN_posterior[bin]` | **`breakpoint()` @ line 41** then `get_inference_for_idx` | `torch_pi_norm_pi` | `super()` + craft_θ | craft/absolute-θ MSE |

**Plug-in mechanics.** EKFs (1-3) multiply-inherit `(ExtendedKalmanFilter, SPFFilter)` and delegate
the Bayesian update to filterpy, passing the analytic `h`/`hjacobian` from this module as `partial`
callbacks plus `residual` for φ-wrap. PFs (4-8) subclass `ParticleFilter` and rely on its
`super().trajectory()` engine; they differ only in the **likelihood** plugged into `update`
(empirical table lookup `theta_phi_to_p_vec`, vs. NN-posterior binning) and the θ-domain handling
in `fix_particles` (full-circle `pi_norm` for dual/craft θ vs. front-half fold `fix_particles_single`
for single radio).

---

## Cross-checks vs KNOWN_ISSUES.md

- **#2 (P0)** — 🔴 **CONFIRMED.** Live `breakpoint()` at `particle_dual_radio_nn_filter.py:41`
  inside `observation()`; fires on the first PF trajectory step → entire dual-radio-NN path
  non-runnable. Note `ParticleFilter.trajectory` (this file, line 333) calls `observation(idx)`
  every step, so it triggers immediately; `debug=True` would call it twice (line 348).
- **#12 (P1)** — 🔴 **CONFIRMED** (in sibling `run_filters_on_data.py:731`, not this file).
  `__main__` calls `run_filter_jobs(jobs)` without required `nparallel` → `TypeError`. The B2 driver
  passes it correctly; direct-script entry is broken.
- **#13 (P1)** — ⚠️ **CONFIRMED.** `config_to_job_params` `eval()`s config strings
  (`run_filters_on_data.py:~530`, self-flagged dangerous). Sibling file.

---

## Cluster summary

| Symbol | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `h_phi_observation_from_theta_state` | ✅ | 3 wrappers + EKF | ❌ (only via EKF tests) | Core φ=2π(d/λ)sin(θ−off); spacing negated by callers. |
| `hjacobian_phi_…` | ✅ | mirrors above | ❌ | Exact analytic derivative; verified by hand. |
| `single_h_phi_…` / `single_hjac_…` | ✅ | EKF single | ❌ (indirect) | Asserts θ∈[-π/2,π/2]; redundant w/ `fix_x`, could crash on drift. |
| `paired_h_…` / `paired_hjac_…` | ✅ | EKF dual | ❌ (indirect) | Shared-θ, two-radio stack; no range assert. |
| `pairedXY_h_…` / `pairedXY_hjac_…` | ✅ | EKF XY | ❌ (indirect) | Chain-rule Jac; RX columns left 0 (RX treated known); `d²→0` unguarded. |
| `F_cached` / `Q_discrete_white_noise_cached` | ✅ | both 2-state EKFs | ❌ | Shared cached arrays (don't mutate). |
| `residual` | ✅ | both EKFs | ❌ | Only φ-wrap in EKF innovation — load-bearing. |
| `SPFFilter` | ✅ base | inherited ×4 | ❌ | Base `trajectory`/`update` inconsistent & **never run** (all override). |
| `SPFFilter.trajectory` | 🧪 dead | 0 (overridden) | ❌ | Skeleton; calls `update(prior=,observation=)` mismatching every override. |
| `ParticleFilter.trajectory` | ✅ | 5 PFs (super) | ✅ (4 PF tests) | Live PF engine; `dt=1.0` hard-coded; `.copy()` on torch tensor bug (return_particles). |
| `theta_phi_to_p_vec` | ✅ | 2 (×2 radios) | ✅ (indirect) | Empirical-likelihood core; table passed transposed. |
| `theta_phi_to_p` | 🧪 dead | 0 | ❌ | Scalar duplicate, zero callers. |
| `theta_phi_to_bins` | ✅ | _p_vec + NN PFs | ✅ (indirect) | `isinstance float` branch dead under jit. |
| `add_noise` / `resample_from_index` / `neff` / `estimate` / `weighted_mean` | ✅ | PF engine | ✅ (indirect) | In-place mutation kernels; correct. |
| `fix_particles_single` | ✅ | single-radio PFs | ✅ (indirect) | Fold+velocity-flip; `while` terminates. |
| `single_/dual_radio_mse_theta_metrics` | ✅ | all `.metrics()` | ✅ (asserts MSE bounds) | dual lacks empty-trajectory guard. |

**Cross-cutting themes.**
1. **Two dead siblings** (`theta_phi_to_p`, and the entire `SPFFilter.trajectory`/no-arg `update`
   base path) coexist with the live engine — the base class reads as a design sketch that the EKF
   subclasses bypass entirely and the PFs only partially use.
2. **Direct unit-test coverage of the observation models/Jacobians is zero** — they are exercised
   only end-to-end through `test_ekf.py`'s MSE-threshold tests, so a sign/derivative regression
   would surface only as a degraded MSE, not a targeted failure. The φ-wrap (`residual`) and the
   sign-flipped spacing are the highest-leverage untested correctness points.
3. **Frame/sign conventions are implicit and load-bearing**: spacing is negated by every caller, θ
   is measured from +y (`arctan2(rel_x, rel_y)`), and the single-radio path folds to [-π/2,π/2]
   while dual/craft uses full (-π,π] — these are the documented #1 bug class and are not asserted
   centrally.
