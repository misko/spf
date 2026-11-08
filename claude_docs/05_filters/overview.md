# 05 · Filters / State Estimation — overview

The last pipeline stage: fuse the noisy per-snapshot angle estimates over **time** into smooth
emitter tracks. Two filter families (Kalman and particle), each in single-radio / dual-radio /
XY flavors, with empirical-table or NN likelihoods. Code: `spf/filters/`.

## The taxonomy

Each concrete filter is a point in a 3-axis space:

- **Algorithm**: EKF (Gaussian, linear motion + nonlinear observation) vs PF (particle cloud,
  resampling, non-parametric).
- **Estimand**: single-θ `[θ, dθ/dt]` vs XY `[tx_x, tx_y, ẋ, ẏ, (rx_x, rx_y)]`.
- **Radios / likelihood**: single-radio vs dual-radio fused; likelihood from the **empirical
  `P(θ|φ)` table** or from a **trained NN** (`*_nn_*`).

| Class | File | Algo | State | Radios | Likelihood | Status |
|---|---|---|---|---|---|---|
| `SPFKalmanFilter` | `ekf_single_radio_filter.py` | EKF | `[θ,θ̇]` | 1 | empirical | ✅ |
| `SPFPairedKalmanFilter` | `ekf_dualradio_filter.py` | EKF | `[θ,θ̇]` | 2 | empirical | ✅ |
| `SPFPairedXYKalmanFilter` | `ekf_dualradioXY_filter.py` | EKF | 6-d xy | 2 | empirical | ✅ |
| `PFSingleThetaSingleRadio` | `particle_single_radio_filter.py` | PF | `[θ,θ̇]` | 1 | empirical | ✅ |
| `PFSingleThetaDualRadio` | `particle_dualradio_filter.py` | PF | `[θ,θ̇]` | 2 | empirical | ✅ |
| `PFXYDualRadio` | `particle_dualradioXY_filter.py` | PF | xy | 2 | empirical | ✅ |
| `PFSingleThetaSingleRadioNN` | `particle_single_radio_nn_filter.py` | PF | `[θ,θ̇]` | 1 | **NN** | ✅ |
| `PFSingleThetaDualRadioNN` | `particle_dual_radio_nn_filter.py` | PF | `[θ,θ̇]` | 2 | **NN** | 🔴 broken |

## Common API (`spf/filters/filters.py`)

- **`SPFFilter`** (`:193`) — base interface: `setup()`, `predict(state, dt, noise_std)`,
  `update(observation)`, `observation(idx)` (fetch measurement at timestep), `trajectory(...)`
  (run over the whole dataset), `metrics(trajectory)` (error vs ground truth),
  `posterior_to_mu_var()`.
- **`ParticleFilter`** (`:292`) — adds particles `(N, D)` + weights + resampling. Its
  `trajectory(mean, std, N, noise_std, ...)` initializes a Gaussian particle cloud then loops
  `predict → observe → update(weight) → resample`, yielding `{"mu", "var"}` per timestep.

### Models inside the filters

- **Motion**: constant-velocity. Single-θ uses `F(dt) = [[1, dt],[0, 1]]`; XY uses a 6-d
  constant-velocity-for-tx / static-rx transition.
- **Observation** (the bridge to the physics): `φ = 2π·(d/λ)·sin(θ)` — same relation as the
  [beamforming primer](../00_concepts/beamforming_primer.md). Dual-radio stacks the two receivers'
  φ with their array-orientation offsets. EKF needs the Jacobians (provided in `filters.py`).
- **Empirical likelihood**: PF weights particles by `P(θ|φ)` looked up in the empirical table
  (`theta_phi_to_p_vec`). **NN likelihood**: the "observation" is already a distribution over θ
  bins (from the model / inference cache), so `update()` just indexes it by each particle's θ-bin.

## Run harness

- **`run_filters_on_data.py`** — orchestrates filter experiments. `run_filter_jobs(jobs, nparallel)`
  fans out over a `multiprocessing.Pool`; per dataset it opens `v5spfdataset_manager()`, runs the
  `run_{EKF,PF}_*` job functions, and records result dicts (type, frequency, `d/λ`, hyperparams,
  `metrics`, routine). Hyperparameter grids (e.g. `phi_std`, `p`, `noise_std`, `dynamic_R`) come
  from a config yaml and expand as a Cartesian product.
- **`run_filters_on_data_b2.py`** — cloud variant: pulls datasets from **B2** (S3-compatible),
  chunks work via `AWS_BATCH_JOB_ARRAY_INDEX`, writes results to **DynamoDB**.
- **`run_filters_report.py` / `_b2.py`** — aggregate results (group by config, average metrics)
  into CSV.

This is the **04→05 handoff**: NN filters consume the inference cache from
[`04`](../04_training_inference/overview.md); all filters consume the dataset + empirical table
from [`03`](../03_datasets/overview.md).

## Known issues / tech debt (spot-verified)

- 🔴 **`PFSingleThetaDualRadioNN` is broken** — a live `breakpoint()` at
  `particle_dual_radio_nn_filter.py:41` halts execution; the dual-radio NN path cannot run as-is.
  Its `cached_model_inference_to_absolute_north()` helper is also never called.
- **`run_filters_report.py` looks truncated** — the agent review reported `write_csv()` ending
  mid-function; verify before relying on local (non-b2) reporting.
- **`run_filters.sh` doesn't actually parallelize** — it loops datasets serially despite a
  `--parallel 30` flag.
- **Redundant `trajectory()` overrides** — nearly every subclass re-wraps `{"mu","var"}` into
  `{"theta"/"craft_theta", "P_theta"}`; ripe for consolidation.
- **XY filter RX-position handling** — RX position is reset each timestep then immediately
  EKF-updated; intent (`P[4:,4:]=0`) vs behavior is subtle — tread carefully.

## Known issues

Filter-relevant entries from [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) (canonical, verified;
severities current after per-issue P0 re-triage):

- 🔴 **P0 #2** — Live `breakpoint()` in `PFSingleThetaDualRadioNN.observation()`
  (`spf/filters/particle_dual_radio_nn_filter.py:41`). **CONFIRMED active**: filter is wired in
  (`ekf_and_pf_config.yml:59` + 3 tests). Under the prod `multiprocessing.Pool` the pdb prompt has
  no TTY → the **worker hangs forever** on step 0. Fix: delete line 41.
- 🔴 **P0 #3** — Both cloud filter drivers import from `spf.scripts.run_filters_on_data*`, which
  **do not exist** (`run_filters_on_data_b2.py:25`, `run_filters_report_b2.py:10`). **CONFIRMED but
  DORMANT**: commit `2276ed1` moved the files to `spf/filters/` without updating imports →
  `ModuleNotFoundError`; the whole B2/DynamoDB pipeline is dead-on-arrival, no CI/Batch runs it.
  Fix: `spf.scripts`→`spf.filters` in the 2 imports (+ stale Dockerfile path).
- 🟠 **P1 #12** — `run_filters_on_data.py` `__main__` calls `run_filter_jobs(jobs)` without the
  required `nparallel` → `TypeError` if run directly (`spf/filters/run_filters_on_data.py:731`).
  The B2 driver passes it correctly.
- 🟠 **P1 #13** — `config_to_job_params` `eval()`s config strings (self-flagged "might be
  dangerous") (`spf/filters/run_filters_on_data.py:530`).
- 🟠 **P1 #31** (✅ verified) — `ParticleFilter.trajectory` does `self.particles.copy()` but
  `particles` is a **torch tensor** (no `.copy()`; torch uses `.clone()`) → `AttributeError` when
  `return_particles=True`. Also `dt=1.0` is hard-coded in `predict`, and `debug=True` double-calls
  `observation(idx)` (`spf/filters/filters.py:346`).
- 🟡 **P2 #38** (✅ verified) — `SPFFilter.trajectory` (base) is dead and self-inconsistent (its
  `update(prior=…, observation=…)` call mismatches the no-arg base `update` and every override);
  `theta_phi_to_p` has zero callers (`spf/filters/filters.py:242, 409`).

## Leaf pages

- `filter_api.md` — ✅ **DONE** → [`../reference/spf/filters/filters.md`](../reference/spf/filters/filters.md):
  `SPFFilter`/`ParticleFilter` in depth + the motion/observation math & Jacobians + the 8-class
  plug-in matrix.
- `nn_filters.md` — _unwritten_; current source of truth is the inventory
  [`../reference/_inventory/filters.md`](../reference/_inventory/filters.md) (`particle_*_nn_filter.py`
  rows) — how NN outputs / the inference cache plug in as likelihood (and the breakpoint fix).
- `run_harness.md` — _unwritten_; current source of truth is the inventory
  [`../reference/_inventory/filters.md`](../reference/_inventory/filters.md) (`run_filters_on_data(+b2)`,
  `run_filters_report(+b2)` rows) — config grids, reports, DynamoDB/B2 wiring.
