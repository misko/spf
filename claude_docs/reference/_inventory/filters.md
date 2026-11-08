# Inventory: `spf/filters/`

Pass-1 surface inventory of all 13 `.py` files in `/home/mouse9911/gits/spf/spf/filters/`. Line numbers verified against source. Status legend: ✅ working/sound · ⚠️ caveat/smell · 🔴 broken · 🧪 test/debug/plot helper.

Cross-cutting note: every filter consumes a `v5spfdataset`-like `ds` and reads `mean_phase_segmentation` / `mean_phase` observations plus `get_empirical_dist` likelihood tables. The `temp_file` branch (live/streaming) vs cached branch (`mean_phase`/`cached_model_inference`) recurs throughout.

---

## filters.py

Shared base classes, EKF observation/Jacobian functions, and torch-jit particle-filter kernels.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `Q_discrete_white_noise_cached` | 13 | func (`@cache`) | Memoized wrapper over filterpy `Q_discrete_white_noise` | `**kwargs` | `np.ndarray` (process-noise Q) | functools cache (unbounded; kwargs must be hashable) | ✅ |
| `F_cached` | 18 | func (`@cache`) | Memoized 2×2 constant-velocity state-transition matrix | `dt: float` | `np.ndarray (2,2)` | functools cache | ✅ |
| `residual` | 22 | func | phi-space residual = `pi_norm(a-b)` (wrap-aware) for EKF update | `a, b` (np arrays) | same shape | none | ✅ |
| `h_phi_observation_from_theta_state` | 49 | func | Observation model: theta-state → phi `sin(theta-offset)*2π*spacing` | `x (2,1)`, `antenna_spacing_in_wavelengths`, `radio_array_angle_offset=0` | `np.ndarray (1,)` | asserts x shape | ✅ |
| `hjacobian_phi_observation_from_theta_state` | 69 | func | Jacobian dphi/dstate of the above | `x (2,1)`, spacing, offset=0 | `np.ndarray (1,2)` | asserts x shape | ✅ |
| `single_h_phi_observation_from_theta_state` | 86 | func | Single-radio h(); asserts theta∈[-π/2,π/2] | `x`, spacing | `np.ndarray (1,)` | assert (raises if theta out of range) | ✅ |
| `single_hjacobian_phi_observation_from_theta_state` | 91 | func | Single-radio H(); asserts theta∈[-π/2,π/2] | `x`, spacing | `np.ndarray (1,2)` | assert | ✅ |
| `paired_h_phi_observation_from_theta_state` | 98 | func | Dual-radio h(): stacks two h() with per-radio offsets | `x`, spacing, `radio_array_angle_offsets[2]` | `np.ndarray (2,1)` | none | ✅ |
| `paired_hjacobian_phi_observation_from_theta_state` | 117 | func | Dual-radio H(): stacks two Jacobians | `x`, spacing, offsets[2] | `np.ndarray (2,2)` | none | ✅ |
| `pairedXY_h_phi_observation_from_theta_state` | 136 | func | XY-state h(): derive theta from `arctan2(rel_x,rel_y)`, then two-radio phi | `x (6,1)`, spacing, offsets[2] | `np.ndarray (2,1)` | none | ✅ |
| `pairedXY_hjacobian_phi_observation_from_theta_state` | 158 | func | XY-state H(): chain-rule dphi/dtheta·dtheta/d(x,y) into 2×6 Jacobian | `x (6,1)`, spacing, offsets[2] | `np.ndarray (2,6)` | none | ✅ |
| `SPFFilter` | 193 | class | Abstract base filter: `our_state/predict/update/observation/metrics/setup/trajectory` | `ds` | — | sets `self.ds` | ✅ (base) |
| `SPFFilter.our_state` | 201 | method | Known RX state for idx (base returns None) | `idx` | `None` | none | ✅ |
| `SPFFilter.predict` | 208 | method | Base no-op predict | `our_state, dt, noise_std` | `None` | none | ✅ stub |
| `SPFFilter.update` | 215 | method | Base no-op update | — | `None` | none | ✅ stub |
| `SPFFilter.observation` | 222 | method | Base no-op observation | `idx` | `None` | none | ✅ stub |
| `SPFFilter.metrics` | 229 | method | Base no-op metrics | `trajectory` | `None` | none | ✅ stub |
| `SPFFilter.setup` | 232 | method | Base no-op setup | `initial_conditions` | `None` | none | ✅ stub |
| `SPFFilter.posterior_to_mu_var` | 235 | method | Base posterior→{mu,var} (returns None,None) | `posterior` | dict | none | ✅ stub |
| `SPFFilter.trajectory` | 242 | method | Generic predict→update→posterior loop over ds | `initial_conditions={}, dt=1.0, noise_std=0.01` | `{"trajectory":[...]}` | calls setup/predict/update | ✅ |
| `single_radio_mse_theta_metrics` | 259 | func | MSE of predicted vs reduced-to-+y ground-truth theta | `trajectory`, `ground_truth_thetas` | `dict{"mse_single_radio_theta"}` | asserts shapes; returns `{}` if empty | ✅ |
| `dual_radio_mse_theta_metrics` | 278 | func | MSE of predicted craft_theta vs craft GT | `trajectory`, `craft_ground_truth_thetas` | `dict{"mse_craft_theta"}` | asserts shapes | ✅ |
| `ParticleFilter` | 292 | class | Base PF: seeded init, predict/update loop, systematic resample on low neff | (subclass of SPFFilter) | — | — | ✅ (base) |
| `ParticleFilter.fix_particles` | 294 | method | Base hook (identity) — overridden by subclasses | — | `self.particles` | none | ✅ |
| `ParticleFilter.trajectory` | 297 | method | Seeded Gaussian init, per-step predict/fix/observe/update + resample, estimate mu/var | `mean, std, N=128, noise_std=None, return_particles=False, debug=False, seed=0, steps=None, progress_bar=False` | `list[dict{var,mu[,particles,observation]}]` | mutates `self.particles/weights/generator`; RNG (seeded) | ✅ |
| `create_gaussian_particles_xy` | 353 | func | Sample N Gaussian particles (jit commented out) | `mean (1,D), std (1,D), N, generator` | `Tensor (N,D)` | RNG (uses generator); asserts | ✅ |
| `theta_phi_to_bins` | 362 | func (`@torch.jit.script`) | Map angle → histogram bin index | `theta_phi: Tensor, nbins: int` | `Tensor long` (or int) | none | ✅ |
| `theta_phi_to_p_vec` | 369 | func (jit) | Vectorized likelihood lookup p(theta,phi) from 2-D table | `thetas, phis, full_p` | `Tensor` | none | ✅ |
| `add_noise` | 376 | func | In-place add Gaussian noise to particles (jit commented out) | `particles, noise_std, generator` | `None` | **mutates `particles` in place**; RNG | ✅ |
| `resample_from_index` | 383 | func (jit) | In-place resample particles + reset weights uniform | `particles, weights, indexes` | `None` | **mutates particles & weights in place** | ✅ |
| `neff` | 390 | func (jit) | Effective sample size `1/Σw²` | `weights` | scalar Tensor | none | ✅ |
| `weighted_mean` | 395 | func (jit) | Weighted mean along dim 0 | `inputs, weights` | `Tensor` | none | ✅ |
| `estimate` | 400 | func (jit) | Weighted mean & variance of particles | `particles, weights` | `(mean, var)` | none | ✅ |
| `theta_phi_to_p` | 409 | func (jit) | Scalar likelihood lookup | `theta, phi, full_p` | scalar | none | ⚠️ appears unused by filters (scalar variant of `_p_vec`) |
| `fix_particles_single` | 418 | func (jit) | Fold theta into [-π/2,π/2] and flip dtheta sign (reflection) | `particles` | `particles` | **mutates in place**; `while` loop | ✅ |

**Notable**
- Docstring-as-comment style: many functions documented by free-standing string literals *above* the `def` (lines 29-46, 63-66, etc.), not real docstrings.
- `theta_phi_to_p` (409) is a scalar duplicate of `theta_phi_to_p_vec`; no caller within filters/ — ⚠️ likely dead.
- `ParticleFilter.trajectory` ignores `steps` correctly but has redundant `len(self.ds) if steps is None else steps` inside the loop range after already resolving `steps` above (line 325 vs 321-322) — harmless.
- Several jit decorators commented out (`create_gaussian_particles_xy` 352, `add_noise` 375) — per commit history jit was removed; runs in eager mode.

---

## ekf_single_radio_filter.py

Single-radio Extended Kalman Filter on theta state.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `SPFKalmanFilter` | 19 | class | Single-radio EKF; multiple-inherits `ExtendedKalmanFilter, SPFFilter` | `ds, rx_idx, phi_std=0.5, p=5, dynamic_R=0.0, **kwargs` | — | asserts antenna spacing/wavelength equal across radios; sets `all_observations` from `mean_phase` | ✅ |
| `.__init__` | 20 | method | Configure R,P, antenna spacing in wavelengths, rx_idx | as above | — | reads `ds.yaml_config`, `ds.wavelengths`, `ds.mean_phase` | ✅ |
| `.R_at_x` | 43 | method | Dynamic measurement-noise scale as fn of current theta | self | float | reads `self.x` | ✅ |
| `.fix_x` | 46 | method | Wrap + reduce theta to +y; assert in [-π/2,π/2] | self | — | mutates `self.x`; assert | ⚠️ assert can raise mid-run |
| `.predict` | 55 | method | CV predict of x and P | `dt, noise_std` | — | mutates `self.F,Q,x,P` | ✅ |
| `.update` | 68 | method | EKF update via filterpy with single h/H, dynamic R option | `observation` | — | mutates `self.x,P`; calls `fix_x` | ✅ |
| `.observation` | 92 | method | phi observation for idx (cached or live segmentation) | `idx` | `np.ndarray (1,1)` | reads ds | ✅ |
| `.metrics` | 101 | method | single-radio theta MSE | `trajectory` | dict | reads `ds.ground_truth_thetas` | ✅ |
| `.setup` | 106 | method | Init state x=[0,0] | `initial_conditions={}` | — | mutates `self.x` | ✅ |
| `.posterior_to_mu_var` | 109 | method | returns {None,None} | `posterior` | dict | none | ✅ stub |
| `.trajectory` | 116 | method | Loop predict→(maybe)update; collects mu/var/theta + debug | `initial_conditions={}, dt=1.0, noise_std=0.01, max_iterations=None, debug=False` | `list[dict]` | skips update on non-finite obs | ✅ |
| `run_and_plot_single_radio_EKF` | 174 | func | Build 3×2 matplotlib figure of phi/theta/zscores for both radios | `ds, trajectories=None` | `matplotlib Figure` | constructs filter, runs trajectory, plots | 🧪 plot helper |

**Notable**
- State vector: `x = [theta, dtheta/dt]` (dim_x=2, dim_z=1).
- `fix_x` (46) asserts theta in [-π/2,π/2] after every predict/update — can crash on bad data.
- Hardcoded `phi_std=5.0, dynamic_R=1.0` inside the plot helper (182-184), differs from class default.

---

## ekf_dualradio_filter.py

Dual-radio (paired) EKF on a single shared theta state.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `SPFPairedKalmanFilter` | 21 | class | Dual-radio EKF (dim_x=2, dim_z=2); inherits EKF + SPFFilter | `ds, phi_std=0.5, p=5, dynamic_R=0.0, **kwargs` | — | asserts spacing/wavelength equality; builds `all_observations` (vstack r0,r1) | ✅ |
| `.__init__` | 22 | method | Setup R,P, per-radio angle offsets from `theta-in-pis`, spacing | as above | — | reads ds.yaml_config/wavelengths/mean_phase | ✅ |
| `.R_at_x` | 52 | method | Dynamic R scale for a given angle | `angle` | float | none | ✅ |
| `.fix_x` | 55 | method | Wrap theta to (-π,π] | self | — | mutates `self.x` | ✅ |
| `.predict` | 63 | method | CV predict | `dt, noise_std` | — | mutates F,Q,x,P | ✅ |
| `.update` | 76 | method | EKF update with paired h/H; per-radio dynamic R matrix | `observation` | — | mutates x,P | ✅ |
| `.observation` | 114 | method | 2×1 phi obs (cached or live) | `idx` | `np.ndarray (2,1)` | reads ds | ✅ |
| `.metrics` | 128 | method | dual-radio craft-theta MSE | `trajectory` | dict | reads `ds.craft_ground_truth_thetas` | ✅ |
| `.setup` | 133 | method | Init x=[0,0] | `initial_conditions={}` | — | mutates x | ✅ |
| `.posterior_to_mu_var` | 137 | method | {None,None} | `posterior` | dict | none | ✅ stub |
| `.trajectory` | 144 | method | predict→(maybe)update loop; collect craft_theta + debug | `initial_conditions={}, dt=1.0, noise_std=0.01, max_iterations=None, debug=False` | `list[dict]` | skip update on non-finite obs | ✅ |
| `run_and_plot_dualradio_EKF` | 203 | func | 3×1 figure of phi/craft-theta/zscores | `ds, trajectory=None` | Figure | constructs filter, runs, plots | 🧪 plot helper |

**Notable**
- State vector: `x = [craft_theta, dtheta/dt]` shared across both radios; offsets distinguish radios.
- Near-duplicate of single-radio EKF structure (predict/setup/trajectory bodies are line-for-line similar) — duplication candidate.

---

## ekf_dualradioXY_filter.py

Dual-radio EKF tracking transmitter XY position (6-D state).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `Qxy_discrete_white_noise_cached` | 19 | func (`@cache`) | 6×6 process-noise with 4×4 white-noise block in top-left | `**kwargs` | `np.ndarray (6,6)` | functools cache | ✅ |
| `Fxy_cached` | 26 | func (`@cache`) | 6×6 CV transition (only tx_x,tx_y integrate velocity) | `dt` | `np.ndarray (6,6)` | functools cache | ✅ |
| `SPFPairedXYKalmanFilter` | 43 | class | XY-tracking EKF (dim_x=6, dim_z=2) | `ds, phi_std=0.5, p=5, dynamic_R=0.0, **kwargs` | — | asserts spacing/wavelength equality | ✅ |
| `.__init__` | 44 | method | Setup; state = `[tx_x,tx_y,dtx_x,dtx_y,rx_x,rx_y]` | as above | — | reads ds.yaml_config/wavelengths | ✅ |
| `.R_at_x` | 70 | method | Dynamic R for angle | `angle` | float | none | ✅ |
| `.fix_x` | 73 | method | No-op (clip code commented out) | self | — | none | ⚠️ effectively a no-op |
| `.predict` | 82 | method | CV predict of XY state & P | `dt, noise_std` | — | mutates F,Q,x,P | ✅ |
| `.update` | 96 | method | EKF update using pairedXY h/H; derives theta from rel x/y | `observation` | — | mutates x,P | ✅ |
| `.observation` | 138 | method | 2×1 live phi obs (no cached branch) | `idx` | `np.ndarray (2,1)` | reads ds[idx] | ⚠️ always live (no `temp_file`/cached path) |
| `._ground_truth_xy` | 146 | method | GT tx XY from cached_keys | self | `Tensor (T,2)` | reads ds.cached_keys | ✅ |
| `.metrics` | 158 | method | craft-theta MSE + XY MSE | `trajectory` | dict | reads ds GT | ✅ |
| `.setup` | 170 | method | Init x from ds[0] tx/rx positions | `initial_conditions={}` | — | mutates x; reads ds[0] | ✅ |
| `.posterior_to_mu_var` | 182 | method | {None,None} | `posterior` | dict | none | ✅ stub |
| `.trajectory` | 189 | method | Per-step: zero RX covar, predict, **clamp RX pos from ds (known), update**, recompute theta | `initial_conditions={}, dt=1.0, noise_std=0.01, max_iterations=None, debug=False` | `list[dict]` | mutates x[4:6] to true RX pos each step (lines 213-214, 229-230); P[4:,4:]=0 | ✅ |
| `run_and_plot_dualradioXY_EKF` | 255 | func | 3×1 figure: phi, theta, XY tracks | `ds, trajectory=None` | Figure | constructs filter (`dynamic_R=True`), runs, plots | 🧪 plot helper |

**Notable**
- State vector: `x = [tx_x, tx_y, dtx_x/dt, dtx_y/dt, rx_x, rx_y]`; RX position (indices 4,5) is treated as *known* and re-injected from `ds` each step rather than estimated.
- `fix_x` is a no-op — the velocity-clip is commented out.
- `dynamic_R=True` passed in plot helper (261) — truthy, not the numeric 0.0/scale the class arithmetic expects (`r * self.dynamic_R` → `r * True`); works by Python bool-as-1 but smells. ⚠️

---

## particle_single_radio_filter.py

Particle filter, single radio, empirical-distribution likelihood.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `PFSingleThetaSingleRadio` | 18 | class | PF over `[theta, dtheta/dt]`, empirical-dist likelihood | `ds, rx_idx` | — | seeds generator; caches empirical dist + observations | ✅ |
| `.__init__` | 23 | method | Seed RNG, cache empirical dist & observations | `ds, rx_idx` | — | `get_empirical_dist`; reads ds.mean_phase | ✅ |
| `.observation` | 32 | method | phi obs (cached or live) | `idx` | scalar/Tensor | reads ds | ✅ |
| `.fix_particles` | 37 | method | Fold theta to [-π/2,π/2], flip dtheta | self | — | mutates particles | ✅ |
| `.predict` | 41 | method | CV step + Gaussian noise (default [0.1,0.001]) | `our_state, dt, noise_std` | — | mutates particles; RNG | ✅ |
| `.update` | 47 | method | Multiply weights by empirical likelihood; normalize | `z` | — | mutates self.weights | ✅ |
| `.metrics` | 56 | method | single-radio theta MSE | `trajectory` | dict | reads ds GT | ✅ |
| `.trajectory` | 61 | method | super().trajectory then unpack mu[0]→theta, var[0]→P_theta | `**kwargs` | `list[dict]` | calls base trajectory | ✅ |
| `plot_single_theta_single_radio` | 69 | func | 2×2 figure per radio | `ds` | Figure | constructs PF, runs, plots | 🧪 plot helper |

**Notable**
- State vector: `[theta, dtheta/dt]`. Likelihood = `theta_phi_to_p_vec` over `get_empirical_dist(ds, rx_idx)`.

---

## particle_dualradio_filter.py

Particle filter, dual radio, shared theta, jit update kernel.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `pf_single_theta_dual_radio_update` | 14 | func (`@torch.jit.script`) | Combine two radios' empirical likelihoods into new weights | `weights, particles, z, empirical_dist, offsets` | `Tensor` (weights) | none (returns new) | ✅ |
| `PFSingleThetaDualRadio` | 30 | class | PF over `[theta, dtheta/dt]` with two-radio likelihood | `ds` | — | seeds RNG; caches per-radio empirical dists & observations | ✅ |
| `.__init__` | 31 | method | Setup offsets, empirical dists, observations | `ds` | — | `get_empirical_dist`×2; reads ds | ✅ |
| `.observation` | 53 | method | 2-vector phi obs (cached or live) | `idx` | `Tensor (2,)` | reads ds | ✅ |
| `.fix_particles` | 64 | method | Wrap theta to (-π,π] | self | — | mutates particles | ✅ |
| `.predict` | 67 | method | CV + noise (default [0.1,0.001]) | `our_state, dt, noise_std` | — | mutates particles; RNG | ✅ |
| `.update` | 73 | method | Apply jit dual-radio update | `z` | — | sets self.weights | ✅ |
| `.metrics` | 82 | method | dual-radio craft-theta MSE | `trajectory` | dict | reads ds GT | ✅ |
| `.trajectory` | 87 | method | super then unpack mu[0]→craft_theta | `**kwargs` | `list[dict]` | base trajectory | ✅ |
| `plot_single_theta_dual_radio` | 95 | func | 2×1 figure | `ds` | Figure | constructs PF, runs, plots | 🧪 plot helper |

**Notable**
- State vector: `[theta, dtheta/dt]`; offsets from `theta-in-pis` separate the two radios. Same state as single-radio PF.

---

## particle_dualradioXY_filter.py

Particle filter tracking transmitter XY (5-D state), dual radio.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `PFXYDualRadio` | 12 | class | PF over `[theta, x, y, dx/dt, dy/dt]` with speed prior | `ds` | — | seeds RNG; speed prior `N(1.5,3)`; caches observations | ✅ |
| `.__init__` | 13 | method | Setup offsets, speed dist, observations | `ds` | — | reads ds | ✅ |
| `.our_state` | 29 | method | RX position at idx | `idx` | `Tensor (2,)` | reads ds.cached_keys | ✅ |
| `.ground_truth_xy` | 37 | method | GT tx XY | self | `Tensor (T,2)` | reads ds.cached_keys | ✅ |
| `.ground_truth_thetas` | 45 | method | GT craft thetas | self | Tensor | reads ds | ✅ |
| `.observation` | 48 | method | 2-vector phi obs (cached or live) | `idx` | `Tensor (2,)` | reads ds | ✅ |
| `.cached_empirical_dist` | 59 | method (`@cache`) | Per-radio empirical dist (memoized) | `rx_idx` | Tensor | `get_empirical_dist` | ⚠️ `@cache` on a method keeps `self` alive in cache (minor leak); works |
| `.fix_particles` | 63 | method | Wrap theta | self | — | mutates particles | ✅ |
| `.predict` | 66 | method | Noise, integrate xy by velocity, recompute theta from RX/TX geometry | `dt, our_state, noise_std` | — | mutates particles; RNG | ✅ |
| `.update` | 76 | method | Two-radio empirical likelihood × velocity prior; normalize | `z` | — | mutates weights | ✅ |
| `.metrics` | 94 | method | craft-theta MSE + XY MSE | `trajectory` | dict | reads GT; **stale `# breakpoint()` commented (line 99)** | ✅ |
| `plot_xy_dual_radio` | 108 | func | 3×1 figure: phi, theta, XY tracks | `ds` | Figure | constructs PF (N=128*16), runs, plots | 🧪 plot helper |

**Notable**
- State vector: `[theta, x, y, dx/dt, dy/dt]`; theta is *derived* from x,y,RX geometry in `predict` (not an independent dim).
- Has a velocity (speed) prior via `scipy.stats.norm(1.5,3).pdf` — unique among the PFs.
- `breakpoint()` at line 99 is **commented out** (dead), not live — contrast with the dual-radio-NN filter.
- No `trajectory` override → returns base mu/var dicts (no `craft_theta`/`P_theta` keys); `metrics` reads `mu[0]`/`mu[[1,2]]` directly so it's fine, but plot helper at 143 also reads `mu[0]` directly.

---

## particle_single_radio_nn_filter.py

Particle filter, single radio, neural-network likelihood (model inference replaces empirical dist).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `PFSingleThetaSingleRadioNN` | 22 | class | PF `[theta,dtheta/dt]` using NN model output as likelihood | `ds, rx_idx, checkpoint_fn, config_fn, inference_cache=None, device="cpu"` | — | loads model/config; runs+caches NN inference | ✅ |
| `.__init__` | 27 | method | Load config; cached-inference branch vs live model branch | as above | — | `get_nn_inference_on_ds_and_cache` or `load_model...`; reads B2/checkpoint via callees; disk cache | ⚠️ heavy I/O; commented-out empirical-fn assert (36-39) |
| `.model_inference_at_observation_idx` | 77 | method | NN output for idx (cached or live `model(z)`) | `idx` | Tensor | live branch runs model fwd, `.cpu()`; `no_grad` | ✅ |
| `.observation` | 87 | method | Per-radio NN posterior slice | `idx` | Tensor | calls inference | ✅ |
| `.fix_particles` | 90 | method | Fold theta to [-π/2,π/2] | self | — | mutates particles | ✅ |
| `.predict` | 94 | method | CV + noise | `our_state, dt, noise_std` | — | mutates particles; RNG | ✅ |
| `.update` | 100 | method | weights ×= NN prob at theta bin; normalize | `z` | — | mutates weights | ✅ |
| `.metrics` | 109 | method | single-radio theta MSE | `trajectory` | dict | reads ds GT | ✅ |
| `.trajectory` | 114 | method | super then unpack theta/P_theta | `**kwargs` | `list[dict]` | base trajectory | ✅ |

**Notable**
- State vector: `[theta, dtheta/dt]`. Likelihood comes from `single_point_networks_inference` model output, binned via `theta_phi_to_bins`.
- No plot helper. The `update` uses `z.shape[0]` as nbins (model output already a per-theta posterior vector).

---

## particle_dual_radio_nn_filter.py  🔴

Particle filter, dual radio, NN likelihood — **contains a live `breakpoint()`**.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `cached_model_inference_to_absolute_north` | 12 | func | Rotate per-radio NN dist by RX heading into absolute-north frame | `ds, cached_model_inference` | Tensor (same shape) | reads ds.cached_keys; calls `rotate_dist` | ✅ |
| `PFSingleThetaDualRadioNN` | 30 | class | PF `[theta,dtheta/dt]` over NN-wrapped dual-radio dataset | `nn_ds` | — | seeds RNG | ✅ (except observation) |
| `.__init__` | 31 | method | Store nn_ds, absolute flag, RNG | `nn_ds` | — | reads nn_ds.absolute | ✅ |
| `.observation` | 40 | method | Get paired NN inference for idx | `idx` | Tensor | **`breakpoint()` at line 41 — drops into pdb every call** + reads `ds.get_inference_for_idx` | 🔴 |
| `.fix_particles` | 44 | method | Wrap theta | self | — | mutates particles | ✅ |
| `.predict` | 47 | method | CV + noise | `our_state, dt, noise_std` | — | mutates particles; RNG | ✅ |
| `.update` | 53 | method | weights ×= NN prob at bin; normalize | `z` | — | mutates weights | ✅ |
| `.metrics` | 62 | method | dual-radio MSE vs craft or absolute thetas | `trajectory` | dict | reads ds GT (`craft_ground_truth_thetas` or `absolute_thetas.mean`) | ✅ |
| `.trajectory` | 72 | method | super then unpack craft_theta/P_theta | `**kwargs` | `list[dict]` | base trajectory | ✅ |

**Notable**
- 🔴 **CONFIRMED: live `breakpoint()` at line 41** inside `observation()`. Every `trajectory()` step calls `observation(idx)` → this filter halts into pdb on first use, making it non-runnable in any automated/batch path (`run_PF_single_theta_dual_radio_NN` in `run_filters_on_data.py:416` invokes it).
- State vector: `[theta, dtheta/dt]`.
- `cached_model_inference_to_absolute_north` (12) hardcodes 65 bins (`reshape(-1,65)`) and assumes 2 radios.

---

## run_filters_on_data.py

Orchestration: build filter jobs from a YAML config, run them across datasets, write metrics to DynamoDB (table `filter_metrics`).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `float_to_decimal` | 32 | func | Recursively convert floats→Decimal for DynamoDB | `obj` | converted obj | **mutates dict/list in place** | ✅ |
| `args_to_str` | 53 | func | Canonical arg string; hashes `*_fn` files via md5 | `args (dict)` | str | reads files (md5) | ✅ |
| `fake_runner` | 66 | func | No-op dataset sweep (perf/debug) | `ds, **kwargs` | None | iterates ds | 🧪 debug helper |
| `run_jobs_with_one_dataset` | 71 | func | Open one ds, run its jobs, skip already-done, write to DynamoDB | `kwargs, checkpoints_cache_dir, already_processed=[]` | `result_fns` | **B2 download, DynamoDB put_item, file I/O (disabled branch), opens ds** | ⚠️ heavy I/O; `use_file_system=False` hardcoded |
| `run_EKF_single_theta_single_radio` | 184 | func | Run single-radio EKF both rx, collect metrics | `ds, phi_std, p, noise_std, dynamic_R` | `list[dict]` | constructs/runs filter; timing | ✅ |
| `run_EKF_single_theta_dual_radio` | 222 | func | Run dual-radio EKF | `ds, phi_std, p, noise_std, dynamic_R` | `list[dict]` | runs filter | ✅ |
| `run_EKF_xy_dual_radio` | 257 | func | Run XY EKF | `ds, phi_std, p, noise_std, dynamic_R` | `list[dict]` | runs filter | ✅ |
| `run_PF_single_theta_single_radio_NN` | 292 | func | Run single-radio NN PF both rx | `ds, checkpoint_fn, inference_cache, theta_err=0.1, theta_dot_err=0.001, N=128` | `list[dict]` | model load/inference; B2 | ✅ |
| `run_PF_single_theta_single_radio` | 344 | func | Run single-radio empirical PF both rx | `ds, theta_err, theta_dot_err, N=128` | `list[dict]` | runs filter | ✅ |
| `run_PF_single_theta_dual_radio` | 385 | func | Run dual-radio empirical PF | `ds, theta_err, theta_dot_err, N=128, progress_bar=False` | `list[dict]` | runs filter | ✅ |
| `run_PF_single_theta_dual_radio_NN` | 416 | func | Wrap ds with NN wrapper, run dual-radio NN PF | `ds, checkpoint_fn, inference_cache=None, theta_err, theta_dot_err, N=128, absolute=False, steps=None` | `list[dict]` | builds `v5spfdataset_nn_wrapper`; **invokes 🔴 breakpoint filter** | 🔴 (transitively blocked) |
| `run_PF_xy_dual_radio` | 467 | func | Run XY PF | `ds, pos_err=15, vel_err=0.5, N=128*16` | `list[dict]` | runs filter | ✅ |
| `runner` | 497 | func | Unpack `(fn,args)` and call | `x` | result | calls fn | ✅ |
| `fn_key_to_fn` | 502 | dict | Map config keys → run_* functions | — | dict | none | ✅ |
| `config_to_job_params` | 522 | func | Cartesian product of config value lists; `eval()`s str values | `config` | `list[dict]` | **`eval()` on config strings (flagged dangerous in code)** | ⚠️ eval of config |
| `config_to_jobs` | 542 | func | Expand list_config["runs"] into `(fn,params)` jobs | `list_config` | `list[(fn,dict)]` | none | ✅ |
| `add_precompute_cache_to_job` | 551 | func | Attach precompute cache by segmentation_version / checkpoint | `job, precompute_caches` | `(fn,args)` | raises ValueError if neither key | ✅ |
| `generate_configs_to_run` | 569 | func | Build full job list across datasets, shuffle, assign workdir | `yaml_config_fn, work_dir, dataset_fns, seed, empirical_pkl_fn` | `list[dict]` | `os.makedirs`, reads YAML, reads .txt dataset list, RNG seed | ✅ |
| `run_filter_jobs` | 626 | func | Run jobs serially (debug) or via multiprocessing Pool | `jobs, nparallel, debug=False, checkpoints_cache_dir=None, already_processed=[]` | None | **multiprocessing Pool; sets torch threads=1** | ✅ |
| `get_parser` (in `__main__`) | 654 | func | CLI arg parser | — | parser | none | ✅ |
| `__main__` block | 652 | script | Parse args, build & run jobs | — | — | runs everything | ⚠️ calls `run_filter_jobs(jobs)` without `nparallel` (731) → relies on default? No default → **TypeError** | 🔴 |

**Notable**
- 🔴 **Bug at line 731**: `run_filter_jobs(jobs)` is called with only one positional arg, but `nparallel` is a required positional (no default at line 627). The `__main__` entrypoint would raise `TypeError`. (The B2 driver `run_filters_on_data_b2.py` calls `run_filter_jobs` correctly with `nparallel=args.parallel`, so the working path is the B2 one.)
- `config_to_job_params` uses `eval()` on string config values (line 530, self-flagged "might be dangerous").
- DynamoDB table name `filter_metrics` hardcoded; partition key = workdir ("bucket"), sort key = `full_name`.

---

## run_filters_on_data_b2.py

B2/AWS-Batch driver: select dataset chunk, download from B2, run filters, skip already-processed from DynamoDB.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_parser` | 28 | func | CLI parser (config/seed/empirical-pkl/parallel/work-dir/manifest) | — | parser | none | ✅ |
| `get_all_items_by_bucket` | 72 | func | Paginated DynamoDB query of all items for a workdir | `prefix, projection_expression=None` | `list[Item]` | **DynamoDB query (paginated); prints progress** | ✅ |
| `get_all_items_by_bucket_scan_full_name` | 116 | func | Query items projecting only `full_name` | `prefix` | list | DynamoDB | ✅ |
| `get_chunk_index` | 160 | func | Deterministic md5-based chunk assignment for a filename | `filename, total_chunks` | int | none | ✅ |
| `main` | 175 | func | Full B2 batch pipeline: manifest→list→download→run→cleanup | — | None | **load_dotenv, B2 list/download, temp dirs, DynamoDB, rmtree, env vars (AWS_BATCH_*, SPF_*)** | ⚠️ heavy side effects; broad `except` swallows errors (318) | ✅ |
| `__main__` | 330 | script | call `main()` | — | — | runs pipeline | ✅ |

**Notable**
- 🔴 **Un-importable** (adversarial-pass correction): imports `generate_configs_to_run, run_filter_jobs` from `spf.scripts.run_filters_on_data` (line 25) — but **`spf/scripts/run_filters_on_data*.py` do not exist** (and `spf/scripts/` has no `__init__.py`). There is no mirrored copy; only the `spf.filters.*` versions exist. This import raises `ImportError`, so the whole module (and the B2 filter-run driver) cannot load as written. Status corrected ✅→🔴.
- Reads chunk index from `AWS_BATCH_JOB_ARRAY_INDEX`/`SIZE`; supports `SPF_SUBSET_FILES`, `SPF_WORKDIR`.
- Bug-ish: line 192 `env_manifest = os.environ.get("SPF_WORKDIR", "")` reads SPF_WORKDIR (not an SPF_MANIFEST var) for the manifest override — ⚠️ likely copy-paste error.
- Broad bare-ish `except Exception` at 318 prints and continues; `except:` at 323 on rmtree.

---

## run_filters_report.py

Aggregate per-workdir result pkls into CSV summaries (file-system based). **NOT truncated — complete file (156 lines).**

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `read_pkl` | 11 | func | Load a pickle file | `pkl_fn` | obj | reads file | ✅ |
| `cannonical` | 15 | func | Build canonical run-name string (typo'd name) | `_ty, params` | str | none | ⚠️ misspelled `cannonical`; defined but no caller in this file (likely dead) |
| `result_to_tuple` | 30 | func | Project a result dict to a tuple over header keys | `result, header` | tuple | none | ✅ |
| `merge_results` | 40 | func | Group metrics by run key; infer movement from ds_fn substring | `results, header` | `dict[key→list[metrics]]` | **mutates result copies (pop)**; raises if movement unknown | ✅ |
| `get_type` | 65 | func | Classify run as XY/single_radio/dual_radio | `key` | str | raises ValueError on unknown | ⚠️ defined, no caller in file (likely dead) |
| `write_csv` | 76 | func | Average metrics per run key, write CSV | `merged, output_csv_fn, header` | None | **writes CSV file** | ✅ |
| `report_workdir_to_csv` | 104 | func | Walk workdir, read pkls per algorithm, write one CSV each | `workdir, output_csv_fn` | None | os.walk, file reads/writes | ✅ |
| `__main__` block | 136 | script | Parse args, run report | — | — | reads/writes files | ✅ |

**Notable**
- **TRUNCATION VERDICT: NOT truncated.** File is syntactically complete: ends with a well-formed `if __name__ == "__main__"` block (136-155) containing `get_parser`, `parse_args`, and the `report_workdir_to_csv` call.
- `cannonical` (typo) and `get_type` are defined but unused here — ⚠️ dead/leftover.
- `merge_results` movement inference only handles circle/bounce/calibrate; raises ValueError otherwise (stricter than the B2 variant).

---

## run_filters_report_b2.py

DynamoDB-based variant of the report: pulls items from DynamoDB into a pkl, then writes CSVs. Richer movement inference (rover/2dwallarray + routine field).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `read_pkl` | 13 | func | Load pickle | `pkl_fn` | obj | reads file | ✅ |
| `cannonical` | 17 | func | Canonical run-name string | `_ty, params` | str | none | ⚠️ no caller (dead, same as non-b2) |
| `result_to_tuple` | 32 | func | Project result to tuple | `result, header` | tuple | none | ✅ |
| `merge_results` | 42 | func | Group metrics; richer movement inference (routine field, rover/2dwallarray prefixes) | `results, header` | dict | mutates copies | ✅ |
| `get_type` | 83 | func | XY/single/dual classifier | `key` | str | raises on unknown | ⚠️ no caller (dead) |
| `write_csv` | 94 | func | Average metrics, write CSV incl `num_samples` column | `merged, output_csv_fn, header` | None | writes CSV | ✅ |
| `report_workdir_to_csv` | 124 | func | File-system walk variant (parallel to non-b2) | `workdir, output_csv_fn` | None | os.walk, file I/O | ⚠️ defined but `__main__` uses DynamoDB path instead, not this fn |
| `results_with_prefix` | 156 | func | Filter DynamoDB items by algorithm prefix | `results, prefix` | list | none | ✅ |
| `__main__` block | 164 | script | Pull from DynamoDB (cache to pkl), group by algorithm, write CSVs | — | — | **DynamoDB query, pickle cache, CSV writes, makedirs** | ✅ |

**Notable**
- 🔴 **Un-importable** (adversarial-pass correction): imports `get_all_items_by_bucket` from `spf.scripts.run_filters_on_data_b2` (line 10) — that module **does not exist** (`spf/scripts/` has no such file and no `__init__.py`). Raises `ImportError`; the B2 report driver cannot load as written. Status corrected ✅→🔴.
- `__main__` (164-215) uses the DynamoDB path (`get_all_items_by_bucket` + `results_with_prefix`), so the file-system `report_workdir_to_csv` (124) is unused in the main path — ⚠️ dead in this entrypoint.
- `write_csv` adds a `num_samples` column vs the non-b2 version.

---

## Cluster summary — the 8 filter classes

| # | Class | File | Algorithm | State vector | Radios | Likelihood / obs model | Status |
|---|---|---|---|---|---|---|---|
| 1 | `SPFKalmanFilter` | ekf_single_radio_filter.py:19 | EKF | `[theta, dθ/dt]` (2) | single | analytic `sin(theta)` h() + dynamic R | ✅ |
| 2 | `SPFPairedKalmanFilter` | ekf_dualradio_filter.py:21 | EKF | `[theta, dθ/dt]` (2) | dual (shared theta) | analytic paired h() | ✅ |
| 3 | `SPFPairedXYKalmanFilter` | ekf_dualradioXY_filter.py:43 | EKF | `[tx_x,tx_y,dx,dy,rx_x,rx_y]` (6) | dual | analytic XY→theta→phi h(); RX known | ✅ (⚠️ no-op fix_x, RX re-injected) |
| 4 | `PFSingleThetaSingleRadio` | particle_single_radio_filter.py:18 | Particle | `[theta, dθ/dt]` (2) | single | empirical dist (`get_empirical_dist`) | ✅ |
| 5 | `PFSingleThetaDualRadio` | particle_dualradio_filter.py:30 | Particle | `[theta, dθ/dt]` (2) | dual (shared theta) | empirical dist ×2 (jit kernel) | ✅ |
| 6 | `PFXYDualRadio` | particle_dualradioXY_filter.py:12 | Particle | `[theta, x, y, dx/dt, dy/dt]` (5) | dual | empirical dist ×2 + speed prior `N(1.5,3)` | ✅ |
| 7 | `PFSingleThetaSingleRadioNN` | particle_single_radio_nn_filter.py:22 | Particle | `[theta, dθ/dt]` (2) | single | NN model posterior (binned) | ✅ |
| 8 | `PFSingleThetaDualRadioNN` | particle_dual_radio_nn_filter.py:30 | Particle | `[theta, dθ/dt]` (2) | dual | NN model posterior (binned, abs-north rotate) | 🔴 live `breakpoint()` (line 41) |

### Counts
- **Filter classes:** 8 (3 EKF, 5 PF) · working ✅ 7, broken 🔴 1.
- **By algorithm × radios:** EKF{single:1, dual-shared-theta:1, dual-XY:1}; PF{single:2 (1 empirical,1 NN), dual-shared-theta:2 (1 empirical,1 NN), dual-XY:1 empirical}.
- **By likelihood:** analytic h() 3 · empirical dist 4 · NN posterior 2 (NN single is OK; NN dual is broken).
- **Total symbols inventoried across the 13 files:** ~120 (filters.py 34; ekf_single 12; ekf_dual 12; ekf_XY 16; pf_single 9; pf_dual 10; pf_XY 12; pf_single_NN 10; pf_dual_NN 9; run_on_data 22; run_on_data_b2 6; report 8; report_b2 9). Statuses: predominantly ✅, ~12 ⚠️ caveats, 2 🔴 (NN-dual breakpoint; `run_filters_on_data.py` `__main__` `run_filter_jobs(jobs)` missing `nparallel`), ~14 🧪 plot/debug helpers.

### Top 5 deep-dive candidates
1. **`PFSingleThetaDualRadioNN` (particle_dual_radio_nn_filter.py:40-41)** — 🔴 live `breakpoint()` in `observation()` blocks the entire dual-radio-NN PF path (incl. `run_PF_single_theta_dual_radio_NN`). Highest-priority fix.
2. **`run_filters_on_data.py` `__main__` (line 731)** — `run_filter_jobs(jobs)` omits the required `nparallel` arg → `TypeError` if run directly. Confirms the canonical entrypoint is the B2 driver; the direct script path is broken.
3. **`SPFPairedXYKalmanFilter` (ekf_dualradioXY_filter.py:43)** — most complex EKF: 6-D state, no-op `fix_x`, RX position re-injected each step, `dynamic_R=True` (bool) arithmetic, no cached-observation branch. Worth a correctness deep-dive.
4. 🔴 **Both `*_b2.py` files are un-importable** (adversarial-pass correction) — they import runner functions from `spf.scripts.run_filters_on_data*`, which **do not exist** (no such files, no `spf/scripts/__init__.py`). Not a "which copy is canonical" question: there is no other copy, so the cloud (B2/DynamoDB) filter-run and report drivers raise `ImportError` as written.
5. **`config_to_job_params` `eval()` (run_filters_on_data.py:530)** — config strings are `eval`'d (self-flagged dangerous); plus `float_to_decimal` mutates in place and DynamoDB schema coupling. Security/robustness deep-dive.
