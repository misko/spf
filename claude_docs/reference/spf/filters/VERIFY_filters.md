# Adversarial Verification: `filters.md`

Verified read-only against `spf/filters/filters.py` and sibling concrete filters on 2026-06-21.

## (a) Priority claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1a | `ParticleFilter.trajectory` does `self.particles.copy()` ~line 346 | CONFIRMED | `filters.py:346` `trajectory[-1]["particles"] = self.particles.copy()` |
| 1b | `particles` is a torch tensor (no `.copy()`) → AttributeError | CONFIRMED | `create_gaussian_particles_xy` (`:358`) returns `mean + (torch.randn(...)*std)`, a torch tensor. torch tensors have `.clone()`, not `.copy()`. So `return_particles=True` raises AttributeError. |
| 1c | `dt=1.0` hard-coded in PF predict | CONFIRMED | `filters.py:327` `dt=1.0,` inside the predict call; `dt` is not a `trajectory` parameter. |
| 1d | `debug=True` double-calls observation | CONFIRMED | `observation(idx)` called at `:333`, then again at `:348` `trajectory[-1]["observation"] = self.observation(idx)`. |
| 2a | base `SPFFilter.trajectory` (~:242) is dead — every concrete class overrides | CONFIRMED | base def at `:242`; all 3 EKFs define own `trajectory`, all 5 PFs route through `ParticleFilter.trajectory` (`:297`) which fully overrides the `SPFFilter` one. |
| 2b | base `trajectory` self-inconsistent: calls `update(prior=,observation=)` vs no-arg base `update` | CONFIRMED | base `update(self)` at `:215` (no args); `:252` calls `self.update(prior=prior, observation=self.observation(idx))`. Signature mismatch. |
| 2c | `theta_phi_to_p` (~:409) has zero callers | CONFIRMED | def at `:409`; repo-wide grep finds the symbol ONLY at its definition. Dead. |
| 3a | observation relation `φ = 2π·(d/λ)·sin(θ − offset)` at ~:49 | CONFIRMED | `:55-58` `np.sin(x[0,0] - radio_array_angle_offset) * 2 * np.pi * antenna_spacing_in_wavelengths`. |
| 3b | concrete filters negate spacing (`-...antenna-spacing-m`), flipping φ sign | CONFIRMED | `ekf_single_radio_filter.py:31`, `ekf_dualradio_filter.py:33`, `ekf_dualradioXY_filter.py:56`: all `antenna_spacing = -ds.yaml_config["receivers"][0]["antenna-spacing-m"]`. |
| 4a | `[θ, dθ/dt]` 2-D (EKF single/dual) | CONFIRMED | `ekf_single_radio_filter.py:21` `dim_x=2`; `ekf_dualradio_filter.py:23` `dim_x=2`. Module docstring `:30` `x = [ theta dtheta/dt ]`. |
| 4b | `[tx_x,tx_y,dtx_x,dtx_y,rx_x,rx_y]` 6-D XY-EKF | CONFIRMED | `ekf_dualradioXY_filter.py:46` `dim_x=6`; `pairedXY_h` uses `rel_x = x[0]-x[4]`, `rel_y = x[1]-x[5]` (`filters.py:139`), and RX re-injected at `x[4]/x[5]` (`:213-214`) — consistent with tx at 0,1 and rx at 4,5. |
| 4c | `[θ, x, y, dx/dt, dy/dt]` 5-D XY-PF | CONFIRMED | `particle_dualradioXY_filter.py:15` comment `# state [ theta, x , y , dx/dt, dy/dt]`; mean/std are 5-element (`:112-113`). |
| 5a | pairedXY Jacobian leaves RX columns (4,5) zero | CONFIRMED | `filters.py:184` `jacobian = np.zeros((2,6))`; only `[0,0],[0,1],[1,0],[1,1]` set (`:185-188`). Cols 2-5 remain 0. |
| 5b | unguarded `d²→0` division | CONFIRMED | `:179` `d2 = rel_x**2 + rel_y**2`; `:180-181` divide by `d2` with no guard. |

## Ordinary-claim spot-checks (10)

| Doc claim | Verdict | Evidence |
|---|---|---|
| `F_cached` line 18, `[[1,dt],[0,1]]` | CONFIRMED | `:18-19` `np.eye(2) + np.array([[0,1],[0,0]])*dt`. |
| `Q_discrete_white_noise_cached` line 13 | CONFIRMED | `:12-14`. |
| `residual` = `pi_norm(a-b)` line 22 | CONFIRMED | `:22-26`. |
| `single_h` asserts θ∈[-π/2,π/2] line 87 | CONFIRMED | `:87` `assert x[0,0] >= -np.pi/2 and x[0,0] <= np.pi/2`. |
| `hjacobian` `[cos(θ−off)·2π(d/λ), 0]` line 69-83 | CONFIRMED | `:76-80`, velocity col = `0`. Exact derivative of `h`. |
| `single_radio_mse_theta_metrics` line 259, empty guard | CONFIRMED | def `:259`; `:260-261` `if len(trajectory)==0: return {}`. |
| `dual_radio_mse_theta_metrics` line 278, NO empty guard | CONFIRMED | def `:278`; no length-0 guard before `np.hstack` `:279`. |
| `create_gaussian_particles_xy` line 353, asserts `(1,D)` | CONFIRMED | `:353-358`, asserts `mean.shape[0]==1`, `std.shape[0]==1`. |
| `fix_particles_single` fold + velocity flip, line 418 | CONFIRMED | `:418-426`: `θ ← sign(θ)·π − θ`, `particles[mask,1] *= -1`. |
| `theta_phi_to_p_vec` uses `torch.take(full_p[:,phi_bin], theta_bin)` line 369 | CONFIRMED | `:372`. Callers pass `.T` (`particle_dualradioXY_filter.py:61`). |
| `breakpoint()` at dual_radio_nn line 41 | CONFIRMED | `particle_dual_radio_nn_filter.py:41` inside `observation` (`:40`). |
| `neff` = `1/Σw²` line 390 | CONFIRMED | `:390-391`. |

## (b) Hit rate

Priority claims: 13/13 CONFIRMED.
Ordinary spot-checks: 12/12 CONFIRMED.
Overall: 25/25 = 100%. No REFUTED, no PARTIAL.

Math checks independently verified:
- `h` derivative `d/dθ[sin(θ−off)·c] = cos(θ−off)·c` matches `hjacobian` — correct.
- `∂arctan2(x,y)/∂x = y/(x²+y²)`, `∂/∂y = -x/(x²+y²)` matches `:180-181` — correct.
- Chain rule cols 0,1 only filled — correct given tx at indices 0,1.

## (c) Missed / misclassified

None material. Minor nitpicks (do not affect grade):
1. Doc note on line 325 ("re-resolves `steps`... already defaulted at line 321-322") is accurate but the redundancy is the `len(self.ds) if steps is None else steps` ternary at `:325` AFTER `steps` was set at `:321-322`. Doc states this correctly. Not a miss.
2. Doc frames `single_h` mid-trajectory assert crash as "medium confidence no live crash" — fair hedge; `fix_x` (`ekf_single_radio_filter.py:46-48`) reduces θ first on the live path. Accurate.
3. Matrix row 6 (`PFXYDualRadio`) "speed prior N(1.5,3)" — CONFIRMED at `particle_dualradioXY_filter.py:20` `scipy.stats.norm(1.5, 3)`. Not missed.

## (d) Grade

**A**

Every priority claim and every spot-check confirmed against source with exact line evidence. The doc's bug findings (torch `.copy()`, dead `theta_phi_to_p`, dead/inconsistent base `trajectory`, sign-flipped spacing, unguarded `d²→0`, RX-zero Jacobian columns, `breakpoint()`) are all real and correctly located. Math derivations independently re-verified. No fabricated symbols, no wrong line numbers, no misclassified active/dead status.
