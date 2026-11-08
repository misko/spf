# Reference: `spf/rf.py` — angle/phase core

Function-level reference produced with the 3-pass method in
[`../../DOCUMENTATION_GUIDE.md`](../../DOCUMENTATION_GUIDE.md). **Read-only review — no code was
changed.** Scope of this first pilot: the foundational angle/phase utilities
`pi_norm`, `circular_mean`, `phase_diff_to_theta`. Physics background:
[`../../00_concepts/beamforming_primer.md`](../../00_concepts/beamforming_primer.md); unit/frame
conventions: [`../../00_concepts/conventions.md`](../../00_concepts/conventions.md).

Clusters documented so far:
- **A — angle/phase utilities:** `pi_norm`, `circular_mean`, `phase_diff_to_theta`.
- **B — beamforming & steering vectors:** `precompute_steering_vectors`, `beamformer`, the
  `beamformer_given_steering*` family, `beamformer_thetas`, `thetas_from_nthetas`.

Evidence current as of commit `574c454` (line numbers may drift).

---

## Cluster A — angle/phase utilities

### `pi_norm(x)`  ·  `spf/rf.py:48`  ·  ✅ Active

**Summary.** Wraps an angle (or array of angles) into the principal interval `[-π, π)`. The
single most-used primitive in the codebase — every place that subtracts, averages, or compares
angles funnels through it to kill the ±π wraparound. Numba-compiled (`@njit`), elementwise on
numpy arrays and scalars.

**Signature**
- Inputs: `x` — angle(s) in **radians**, scalar or `np.ndarray` (any shape). No range assumption.
- Returns: same shape as `x`, each value in **`[-π, π)`** radians. Formula:
  `((x + π) % (2π)) − π`.
- Side effects: **none** (pure; returns a new value, does not mutate `x`).

**Called from** (Pass 2 — fan-in ≈ 20+ sites, the highest in the module)
- Filters: `filters.py:26,140,162`, `ekf_single_radio_filter.py:44,47`,
  `ekf_dualradio_filter.py:53,56,81,88,228`, `ekf_dualradioXY_filter.py:71,104,111,232,280` —
  normalizing state angles, measurement residuals, and `arctan2` target bearings.
- SDR: `sdr_controller.py:1011,1128` — phase-difference of the two antenna channels.
- Dataset/seg: `segmentation.py:885` (window-merge phase gate), `spf_dataset.py` (commented use).
- Internal `rf.py`: `circular_mean*` (`:283,296,322,341,369,377,421`), `circular_mean_simple_fast`
  (`:267`), `phase_diff_to_theta` (`:727,736`), `get_phase_diff` (`:746`), `get_avg_phase_fast2`
  (`:779`).
- Tests: `tests/test_angles.py:17,18`, `tests/test_gps.py:28`.

**Contract**
- *Preconditions:* `x` is real-valued radians; numpy/scalar (njit-compatible). No NaN expected
  (NaN propagates). Torch callers use the separate `torch_pi_norm` (`:77`) — do **not** pass
  tensors here.
- *Postconditions:* output ∈ `[-π, π)`; shape-preserving; `pi_norm` is idempotent on its own range.
- *Invariants:* `sin/cos(pi_norm(x)) == sin/cos(x)` (angle identity preserved); exact endpoint
  `x = π` maps to `−π` (half-open interval — relevant when callers test boundary equality).

**Tested by** (Pass 3)
- **No dedicated test.** Exercised only **indirectly**: `tests/test_gps.py:28`
  (`assert np.isclose(pi_norm(a-b), 0, atol=0.01)`) and `tests/test_angles.py:17-18` use it as a
  helper while asserting other geometry.

**Analysis**
- *Correctness:* verified. Python/numpy `%` returns a result with the sign of the (positive)
  divisor, so negative inputs wrap correctly — the well-known failure of C-style `fmod` does not
  apply here. Endpoint behavior (`π → −π`) is the standard half-open convention.
- *Issues found or considered:* none in the implementation. One **consumer-side caution**, not a
  bug here: a commented note at `rf.py:739-741` ("Do not use pi_norm here!! this can flip the sign
  on observations near the edge") warns that wrapping is *unwanted* in some edge contexts — see
  `phase_diff_to_theta` below. `pi_norm` itself is correct; the caution is about *where* to apply it.
- *Test coverage:* indirect only.
- *Missing test coverage:* no direct assertion of the core contract — range `[-π, π)`, the
  `π → −π` endpoint, idempotency, and shape preservation on arrays are all unverified by tests
  despite ~20 dependents. **Highest-value test gap in this cluster** given the fan-in.

**Open questions / TODO.** None — behavior is unambiguous; the gap is purely test coverage.

---

### `circular_mean(angles, trim, weights=None)`  ·  `spf/rf.py:273`  ·  ⚠️ Active but marked `# TODO remove`

**Summary.** Computes the circular (directional) mean of each **row** of a 2-D angle array, plus a
**trimmed** circular mean that discards the outlier fraction farthest from the untrimmed mean.
Returns both. This is how SPF turns many noisy per-sample phase differences into one robust phase
estimate. The line directly above it (`rf.py:272`) reads `# TODO remove`, yet it still has live
production callers — see status.

**Signature**
- Inputs:
  - `angles` (radians, `np.ndarray`, **`ndim == 2`** — asserted at `:274`; shape `(rows, n)`).
  - `trim` (float, **percent** in `[0, 100)`): fraction of farthest points to drop. `0.0` ⇒ no
    trimming.
  - `weights` (optional, broadcastable to `angles`): per-sample weights applied to the sin/cos
    components before summation.
- Returns: tuple `(mean, trimmed_mean)`, each `np.ndarray` shape `(rows,)`, values in `[-π, π)`.
  When `trim == 0.0`, both elements are identical.
- Side effects: **none** (pure).

**Called from** (Pass 2 — fan-in: 4 prod + 1 internal + tests)
- `sdr_controller.py:1013, 1027, 1043` — antenna phase **calibration** (per-channel mean phase).
- `sdr_controller.py:1131` — `circular_mean(diff.reshape(1, -1), 20)`; return flows to a runtime
  phase estimate.
- `rf.py:757` — inside `get_avg_phase(signal_matrix, trim=0.0)`, input shaped `(...)[None]` → 2-D.
- Tests: `tests/test_rf.py:25, 42, 161`.
- Every caller observed passes 2-D input and consumes the **trimmed** (second) return except the
  `trim=0.0` paths, where both are equal.

**Contract**
- *Preconditions:* `angles` is 2-D radians (enforced); `trim ∈ [0,100)`; if `weights` given it
  must broadcast to `angles` and be non-negative (negative/zero-sum weights make `arctan2(0,0)`
  ill-defined). Per-row sample count ≥ 1.
- *Postconditions:* both outputs shape `(rows,)`, range `[-π, π)`; `trim==0.0` ⇒ `mean == trimmed`.
- *Invariants:* result is rotation-equivariant (shifting all angles by δ shifts the mean by δ,
  mod 2π); independent per row.

**Tested by** (Pass 3)
- `tests/test_rf.py:151` `test_circular_mean` (**the one that actually runs** — see issue):
  asserts `circular_mean(angles, trim=0.0)` matches `circular_mean_simple_fast` to `1e-6` over
  three random shapes. Covers only the **untrimmed, unweighted** path.
- `tests/test_rf.py:18` `test_circular_mean` (**shadowed — does NOT run**): would assert the
  **trimmed** mean (`trim=20`) recovers a known ground-truth mean, the **weighted** path, and
  `torch_circular_mean` equivalence.

**Analysis**
- *Correctness:* untrimmed path verified (sin/cos accumulation + `arctan2`, then `pi_norm`).
  Trimmed path: `cm` is computed mod `2π`, distances via `circular_diff_to_mean`, mask keeps
  `dist ≤ percentile(100−trim)` per row, then re-averages the masked sin/cos. Logic is sound; note
  the mask uses **unweighted** distances while the re-average uses **weighted** sums — intended for
  the "weights as a 0/1 inclusion mask" use case (see the shadowed test) but subtle for fractional
  weights.
- *Issues found or considered:*
  1. 🟠 **Duplicate `test_circular_mean` definition** (`tests/test_rf.py:18` and `:151`). Python
     binds the **second** definition, so the first never executes under pytest. Result: the
     trimmed-mean accuracy, the weighted path, and the `torch_circular_mean` cross-check are
     **silently untested**. (Read-only finding; fix = rename one test.)
  2. ⚠️ **`# TODO remove` (`rf.py:272`) vs live callers.** The author flagged this for deletion,
     but `sdr_controller.py` calls it in 4 places. Removing it would break calibration. Status is
     genuinely ambiguous — treat as load-bearing until the SDR path migrates to a `*_fast` variant.
  3. Minor: `circular_mean` (np.percentile) and the njit `circular_mean_single_fast`
     (`fast_percentile_1d`, ceil-rank) use **different percentile conventions**; results can differ
     by one sample at small `n`. Not wrong, but the variants are not bit-identical.
- *Test coverage:* untrimmed/unweighted equivalence only.
- *Missing test coverage:* the **trimmed** path, the **weighted** path, and torch parity — all
  intended by the shadowed test and currently dead. Also: single-sample rows, all-equal angles,
  zero/negative weights.

**Open questions / TODO.** Is the intended replacement `circular_mean_single_fast` /
`circular_mean_simple_fast`? If so the SDR callers need migration before `circular_mean` can go.

---

### `phase_diff_to_theta(phase_diff, wavelength, distance_between_receivers, large_phase_goes_right)`  ·  `spf/rf.py:721`  ·  🧪 Defined; effectively unreached

**Summary.** Inverts the two-element array geometry: given a measured phase difference φ, solve
`sin θ = φ·λ / (2π·d)` for the angle of arrival θ. Because a 2-element array is front/back
ambiguous, it returns **three** candidates: the arcsin solution, its mirror `π − x`, and a derived
"averaged" value. `large_phase_goes_right` selects the sign convention.

**Signature**
- Inputs:
  - `phase_diff` (radians, **`np.ndarray`** — `.copy()`/`*=` at `:725-726` require a numpy array,
    not a scalar/torch tensor).
  - `wavelength` (length units, e.g. metres = `c / f_carrier`).
  - `distance_between_receivers` (same length units as `wavelength`; the ratio is what matters →
    `d/λ`).
  - `large_phase_goes_right` (bool): if `False`, negate `phase_diff` first (flip handedness).
- Returns: 3-tuple of `np.ndarray` (radians):
  `(x, π − x, (pi_norm(x) + pi_norm(π − x))/2 − π)` where `x = arcsin(clip(φλ/(2πd), ±(1−1e-8)))`.
- Side effects: **none** on the caller's array — the `large_phase_goes_right=False` branch copies
  before negating (`:725`), and the `True` branch only reassigns the local via `pi_norm` (`:727`).
  ✅ verified non-mutating on both paths.

**Called from** (Pass 2 — fan-in: 1, and that caller is itself uncalled)
- `spf/dataset/spf_dataset.py:1628`, inside `get_estimated_thetas()`:
  `phase_diff_to_theta(self.mean_phase[f"r{ridx}"].numpy(), speed_of_light/carrier_freq,
  antenna_spacing, large_phase_goes_right=False)`.
- 🔎 **`get_estimated_thetas` has no static callers** anywhere in the repo (verified by grep) —
  so this function is effectively unreached in current code paths.

**Contract**
- *Preconditions:* `phase_diff` a real numpy array in radians (ideally pre-wrapped, though the
  function re-wraps at `:727`); `wavelength` and `distance_between_receivers` same units, `d > 0`;
  `large_phase_goes_right` a bool.
- *Postconditions:* three radian arrays of `phase_diff`'s shape; `x ∈ [-π/2, π/2]` (arcsin range),
  `π − x ∈ [π/2, 3π/2]`. The `sin` argument is clipped to `±(1−1e-8)`, so out-of-physical-range φ
  saturates rather than raising.
- *Invariants:* `x` and `π − x` are the two geometry-consistent solutions for the same φ
  (front/back ambiguity); resolving between them requires a second receiver (see
  [filters](../../05_filters/overview.md)).

**Tested by** (Pass 3)
- **None.** No test references `phase_diff_to_theta` (grep-verified).

**Analysis**
- *Correctness:* the core relation `sin θ = φλ/(2πd)` matches the
  [primer](../../00_concepts/beamforming_primer.md) and the call-site units (both metres). Clipping
  guards the arcsin domain. The **third return value**, `(pi_norm(x) + pi_norm(π−x))/2 − π`, is not
  self-evidently meaningful and is the suspicious part of this function.
- *Issues found or considered:*
  1. ⚠️ **Third return value + the adjacent warning.** A commented note immediately below
     (`rf.py:739-741`): *"Do not use pi_norm here!! this can flip the sign on observations near the
     edge."* The third element applies `pi_norm` to `x` and `π−x` before averaging — exactly the
     pattern the note warns can flip signs near ±π/2. Whether this value is correct/used anywhere is
     unclear; it is **untested**. Flagged for a maintainer, not fixed.
  2. 🧪 **Dead path.** Both the function and its sole caller appear unreached; risk of silent rot.
- *Test coverage:* none.
- *Missing test coverage:* everything — sign-convention flip, arcsin clipping/saturation at the
  ±90° boundary, the two-candidate symmetry, and the questionable third return value.

**Open questions / TODO.** (a) Is `get_estimated_thetas` intended to be live (debug/inference
helper) or removable? (b) What consumes the third tuple element, and is the `pi_norm` warning at
`:739-741` about exactly this computation?

---

### Cluster A summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `pi_norm` | ✅ Active | ~20+ | ❌ (indirect only) | Highest-fan-in primitive with no dedicated test. |
| `circular_mean` | ⚠️ `TODO remove`, still live | 4 prod | ⚠️ partial | Duplicate test name hides the trim/weights/torch assertions. |
| `phase_diff_to_theta` | 🧪 effectively unreached | 1 (dead) | ❌ | Untested; questionable 3rd return + adjacent pi_norm warning. |

**Theme:** the most fundamental angle helpers have the **weakest** direct test coverage, and one
existing test is silently disabled by a name collision. A focused `test_rf.py` cleanup (rename the
duplicate; add direct `pi_norm` range/endpoint asserts) would close most of the gap.

---

## Cluster B — beamforming & steering vectors

The delay-and-sum beamforming core: build per-angle steering vectors, apply them to the IQ signal,
and read off power-vs-angle. Physics in [`beamforming_primer.md`](../../00_concepts/beamforming_primer.md).
A recurring theme here is **variant sprawl** — five near-identical "apply steering" functions plus
two full pipelines that re-implement the same steering math.

### `precompute_steering_vectors(receiver_positions, carrier_frequency, spacing=65, calibration=None, offset=0.0)`  ·  `spf/rf.py:1173`  ·  ✅ Active

**Summary.** Builds the steering-vector matrix once so it can be reused across many snapshots:
for each of `spacing` angles on `[-π, π]`, computes the complex phase pattern
`e^{-j·2π·(receiver·direction)/λ}` a signal from that angle would produce across the receivers.
This is the reusable half of `beamformer` (below), pulled out for the streaming/segmentation paths.

**Signature**
- Inputs: `receiver_positions` (`np.ndarray (n_receivers, 2)`, metres, X/Y); `carrier_frequency`
  (Hz); `spacing` (int = `nthetas`, number of angle bins, default 65); `calibration` (optional
  complex `(n_receivers,)` per-receiver correction); `offset` (radians, rotates the angle grid).
- Returns: `steering_vectors` `np.ndarray (spacing, n_receivers)`, complex. **dtype follows the
  inputs** (commonly complex128 unless positions/freq are float32).
- Side effects: none (pure).

**Called from** (fan-in: 5 prod)
- `data_collector.py:187` (live capture), `sdr_controller.py:1064, 1275`, `spf_dataset.py:577, 1037`
  (dataset/segmentation setup). Return flows into the `beamformer_given_steering*` appliers.

**Contract**
- *Preconditions:* `receiver_positions` shape `(n, 2)` in metres; `carrier_frequency > 0`;
  `calibration`, if given, broadcasts to `(n_receivers,)`. The downstream **numba** applier
  requires complex64/128 — caller is responsible for the dtype (see issues).
- *Postconditions:* shape `(spacing, n_receivers)`, unit-magnitude entries (before calibration);
  angle grid is `linspace(-π, π, spacing)` shifted by `offset`.
- *Invariants:* matches the steering math inlined in `beamformer` (`:1394-1423`) — the two must
  stay in sync (they are currently duplicated, not shared).

**Tested by**
- **No direct test.** Indirectly validated because `beamformer` (end-to-end tested) reproduces the
  same computation.

**Analysis**
- *Correctness:* verified against the primer relation `φ = 2π·d·sinθ/λ`; `thetas` uses `float32`
  while `beamformer`'s inline copy omits the dtype — a latent precision mismatch between the two
  code paths, not a bug on its own.
- *Issues found or considered:*
  1. 🟠 **Duplicated steering math.** `beamformer` (`:1394-1423`) re-implements this function inline
     instead of calling it. Two copies of the core formula can drift (one already differs in
     `thetas` dtype). Read-only finding.
  2. ⚠️ **dtype contract is implicit.** Output dtype depends on input dtypes, but the numba applier
     `beamformer_given_steering_nomean_fast` hard-requires complex64/128 and raises otherwise. No
     enforcement here.
- *Test coverage:* indirect only.
- *Missing test coverage:* direct shape/dtype/grid assertions; `calibration` and `offset` effects;
  parity with `beamformer`'s inline computation.

### `beamformer(receiver_positions, signal_matrix, carrier_frequency, calibration=None, spacing=65, offset=0.0)`  ·  `spf/rf.py:1352`  ·  ✅ Active

**Summary.** The complete one-call pipeline: generate the angle grid, build steering vectors
(inline), apply them to the signal, take magnitude, and average over time. Returns the angle grid,
the time-averaged power spectrum, and the steering vectors. Used by the simulator/tests and the SDR
calibration path. (Note: a **different** `beamformer` also exists at `sdr.py:128` — don't confuse them.)

**Signature**
- Inputs: `receiver_positions` `(n_receivers, 2)` m; `signal_matrix` `(n_receivers, n_samples)`
  complex; `carrier_frequency` Hz; `calibration` optional complex `(n_receivers,)`; `spacing` int
  (=nthetas, default 65); `offset` rad.
- Returns: `(thetas (spacing,), steer_dot_signal (spacing,), steering_vectors (spacing, n_receivers))`.
  `steer_dot_signal = |steering · signal|.mean(axis=1)` — magnitude **then** time-average.
- Side effects: none (pure).

**Called from** (fan-in: 2 tests + 3 prod/scripts)
- `tests/test_beamformer.py:81`, `tests/test_invariance.py:63`; `sdr_controller.py:1081`; standalone
  scripts `02_wifi_direction.py:88`, `01_phase_cal.py:102`.

**Contract**
- *Preconditions:* `receiver_positions` rows align with `signal_matrix` rows (same receiver order);
  metres + Hz units; `n_samples ≥ 1`.
- *Postconditions:* `thetas = linspace(-π, π, spacing)`; spectrum is real, non-negative, length
  `spacing`; the **peak index corresponds to the dominant arrival direction** (this is exactly what
  the test asserts).
- *Invariants:* steering vectors returned are the same ones applied internally.

**Tested by**
- `tests/test_beamformer.py:25` `test_beamformer` — simulates `ULADetector`/`UCADetector` +
  `IQSource` at a known angle, runs `beamformer(spacing=1025)`, and asserts the spectrum's max is at
  the bin closest to ground truth (within an interpolation tolerance). Strong end-to-end check.
- `tests/test_invariance.py:63` — exercises `beamformer` under rotations (invariance property).

**Analysis**
- *Correctness:* verified end-to-end by the peak-location test across ULA and UCA geometries. The
  docstring mentions `a(θ)ᴴ` (conjugate) but the code bakes the conjugate into `exp(-1j·args)` and
  uses a plain `matmul` — consistent, not a bug.
- *Issues found or considered:* 🟠 **inlines `precompute_steering_vectors`** rather than calling it
  (the duplication noted above) — the only divergence between this cluster's two steering
  implementations.
- *Test coverage:* good (peak location + rotation invariance).
- *Missing test coverage:* `calibration` path; `offset` path; non-default `spacing` correctness.

### `beamformer_given_steering*` family  ·  `spf/rf.py:1204–1333`

Four functions that apply already-computed steering vectors to a signal. They differ only in
backend and whether they average over time:

| Function | Line | Backend | Output | Prod caller |
|---|---|---|---|---|
| `beamformer_given_steering_nomean` | 1273 | numpy `np.dot` + `abs` | `(nthetas, n_samples)` | internal (`beamformer_given_steering`) |
| `beamformer_given_steering_nomean_fast` | 1204 | numba `@njit(parallel=True)` core | `(nthetas, n_samples)` | **none — test-only** 🧪 |
| `beamformer_given_steering_nomean_cp` | 1240 | **byte-identical to numpy version** | `(nthetas, n_samples)` | `segmentation.py:505` (GPU hot path) |
| `beamformer_given_steering` | 1309 | wraps `_nomean` + `.mean(axis=1)` | `(nthetas,)` | `data_collector.py:155`, `sdr_controller.py:1090,1285` |

**Shared contract.** Inputs: `steering_vectors (nthetas, n_antennas)`, `signal_matrix
(n_antennas, n_samples)`, complex, antenna axes must match. Output: real non-negative magnitudes.
All four are pure. The `_fast` variant additionally **requires complex64/128** and raises
`ValueError` otherwise (`:1210`).

**Tested by**
- `tests/test_rf.py:70-72` `test_beamformer_functions` — asserts `_nomean` (numpy) and `_nomean_fast`
  (numba) agree to `rtol=1e-4` on complex64 inputs. Covers the numpy↔numba parity only.

**Analysis**
- *Correctness:* numpy↔numba parity verified by test. `_nomean_cp` is **literally identical source**
  to `_nomean` — it references `np`, not `cp`. It runs on GPU only because its sole caller
  (`segmentation.py:505`) passes **cupy arrays** (`cp.asarray(...)`) and relies on NumPy's NEP-18
  `__array_function__` dispatch, then `cp.asnumpy()` on the result. Verified — it works, but the
  GPU-ness lives in the caller, not the function.
- *Issues found or considered:*
  1. 🟠 **`_cp` is a misnomer / dead-weight duplicate.** Identical to `_nomean`; the name implies a
     cupy implementation that isn't there. Confusing and fragile (depends on NEP-18 + caller dtype).
  2. 🧪 **`_fast` (numba) has no production caller.** The fastest CPU path is tested but unused;
     segmentation uses `_nomean` (CPU) or `_nomean_cp` (GPU), never `_fast`.
  3. **No runtime dispatch** across the three appliers — caller hard-codes which to use (echoes the
     module-wide theme from [`02_rf_signal_processing/overview.md`](../../02_rf_signal_processing/overview.md)).
- *Test coverage:* numpy↔numba parity.
- *Missing test coverage:* `_cp`/GPU path (untested — needs a GPU); `beamformer_given_steering`
  (the time-averaging wrapper, 3 prod callers) has no direct test.

### `beamformer_thetas(spacing)` / `thetas_from_nthetas(nthetas)`  ·  `spf/rf.py:1336 / 1200`

**Summary.** Both return `np.linspace(-π, π, spacing)` — the angle grid. `beamformer_thetas` is
used by `beamformer` (`:1394`) and `sdr_controller.py:1089`; **`thetas_from_nthetas` has zero
callers** (🧪 dead). A third inline copy of the same `linspace` lives in
`precompute_steering_vectors` (with `dtype=float32`).

**Analysis**
- *Issues found or considered:* 🟠 **Three definitions of the same angle grid**
  (`beamformer_thetas`, `thetas_from_nthetas`, and the inline `linspace` in
  `precompute_steering_vectors`/`beamformer`), one of which is dead and one of which differs in
  dtype. Prime consolidation target. *No code changed.*
- *Tested by:* none directly (grid correctness rides on the `beamformer` end-to-end test).

### Cluster B summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `precompute_steering_vectors` | ✅ Active | 5 prod | ❌ (indirect) | Steering math duplicated inline in `beamformer`; implicit dtype contract. |
| `beamformer` | ✅ Active | 2 test + 3 prod | ✅ (peak-location + invariance) | Re-inlines steering instead of reusing `precompute_*`. |
| `beamformer_given_steering_nomean` | ✅ Active | 1 internal + test | ✅ (vs numba) | Reference applier. |
| `beamformer_given_steering_nomean_fast` | 🧪 test-only | 0 prod | ✅ (vs numpy) | Fastest path is unused in production. |
| `beamformer_given_steering_nomean_cp` | ✅ Active (GPU hot path) | 1 prod | ❌ | Byte-identical to numpy version; "cp" name is misleading. |
| `beamformer_given_steering` | ✅ Active | 3 prod | ❌ | Time-averaging wrapper, untested directly. |
| `beamformer_thetas` | ✅ Active | 2 | ❌ (indirect) | One of three copies of the angle grid. |
| `thetas_from_nthetas` | 🧪 dead | 0 | ❌ | No callers anywhere. |

**Theme — variant sprawl:** the cluster carries five "apply steering" functions (one dead-weight
duplicate, one prod-unused), two full pipelines that re-implement the same steering math, and three
copies of the angle grid. The numerics are correct and the main `beamformer` path is well tested,
but the GPU production path (`_cp`) is untested and the proliferation invites drift. A consolidation
pass (single steering builder, single applier with a backend arg, single theta-grid helper) would
shrink the surface substantially — flagged for later, **no code modified here.**

---

## Cluster C — windowed circular statistics

The segmentation engine slices the per-sample phase-difference stream into overlapping windows and,
for each window, computes **three numbers**: the *trimmed circular mean* of the phase difference
(its direction estimate), the *trimmed circular standard deviation* (its stability / how "signal-y"
the window is), and the *median absolute signal amplitude* (its power). `segmentation.py` thresholds
on the stddev and amplitude to decide signal-vs-noise windows
([`02_rf_signal_processing/overview.md`](../../02_rf_signal_processing/overview.md)).

The same triple is implemented **four times over** — a numba "fast" path (the live one), a plain
serial numba path, a torch path, and a broken `prange` "fast2" path — backed by a small zoo of
primitives (`circular_diff_to_mean*` ×4, `fast_percentile*` ×2, `fast_median_abs`/`fast_abs_median`
which are byte-identical). The trim convention is **percent of farthest points to drop**, applied via
a **one-sided** `dist ≤ percentile(100−trim)` mask (not a symmetric two-tail trim). The numpy and
torch backends disagree on the percentile rule (see the cross-cutting note). All angles are
**radians**; see [`conventions.md`](../../00_concepts/conventions.md).

**Live path (the only one that runs in production):**
`get_all_windows_stats` (`segmentation.py:641`) → `windowed_trimmed_circular_mean_and_stddev_fast`
(`rf.py:657`) → `get_stats_for_signal_window` (`:686`) → {`circular_mean_single_fast` (`:300`),
`circular_stddev_fast` (`:94`), `fast_median_abs` (`:472`)}, each leaning on `fast_percentile_1d`
(`:348`) and `circular_diff_to_mean_single_fast` (`:234`). `recompute_stats_for_windows`
(`segmentation.py:906`) uses the non-windowed `get_stats_for_signal` (`:490`) on the same primitives.

### `windowed_trimmed_circular_mean_and_stddev_fast(v, pd, window_size, stride, trim=50.0)`  ·  `spf/rf.py:657`  ·  ✅ Active (canonical)

**Summary.** **The production windowing routine.** Walks `pd` in fixed windows (`window_size`,
hop `stride`) and fills an `(n_steps, 3)` stats table `[trimmed_circular_mean, trimmed_circular_stddev,
abs_signal_median]` plus an `(n_steps, 2)` `[start_idx, end_idx]` table. `@njit`; the per-step work is
delegated to `get_stats_for_signal_window`. Despite the `prange` at `:666` it runs **serially** —
`@njit` (not `@njit(parallel=True)`) does not parallelize `prange`.

**Signature**
- Inputs: `v` (complex `np.ndarray (2, N)`, the two-antenna signal matrix); `pd` (radians,
  `np.ndarray (N,)` float32 — the phase difference, produced by `get_phase_diff(v).astype(np.float32)`
  at the call site `segmentation.py:652`); `window_size` (int samples); `stride` (int samples);
  `trim` (float **percent** in `[0,100)`, default 50.0).
- Returns: `(step_idxs (n_steps,2) int32, step_stats (n_steps,3) float32)`, where
  `n_steps = 1 + (N − window_size)//stride`.
- Side effects: none (pure); `@njit`.

**Called from** (Pass 2 — fan-in: 1 prod, transitively load-bearing)
- `segmentation.py:658` inside `get_all_windows_stats`, itself called from `simple_segment`
  (`segmentation.py:726`) — the core signal-segmentation entry point exercised by
  `tests/test_segmentation.py` and the whole dataset pipeline. **This is the hot path.**

**Contract**
- *Preconditions:* `(N − window_size) % stride == 0` — **asserted** at `:661` (raises/aborts otherwise);
  `v.shape[1] == N == pd.shape[0]`; `v` has 2 rows. `pd` already wrapped to `[-π,π)` by the caller.
- *Postconditions:* row `i` summarizes window `[i·stride, i·stride+window_size)`; mean ∈ `[-π,π)`,
  stddev ≥ 0, median ≥ 0.
- *Invariants:* `step_idxs[:,1] − step_idxs[:,0] == window_size` for every row.

**Tested by** (Pass 3)
- **No direct unit test.** Covered transitively by `tests/test_segmentation.py::test_simple_segment*`
  (which call `simple_segment`). No test pins the exact stats values.

**Analysis**
- *Correctness:* window arithmetic verified; the divisibility assert makes ragged tails a caller error
  rather than silent truncation. Delegates correctly to `get_stats_for_signal_window`.
- *Issues found or considered:* 🟡 `prange` under plain `@njit` is **serial** — the "fast/parallel"
  intent is not realized (no correctness impact). The 3-column output omits the per-channel amplitude
  split that the broken `_fast2` attempted (4 columns).
- *Missing test coverage:* exact stat values, the trim path, single-window and boundary
  (`N==window_size`) cases.

### `windowed_trimmed_circular_mean_and_stddev(v, pd, window_size, stride, trim=50.0)`  ·  `spf/rf.py:698`  ·  ✅ Defined; ⚠️ call site commented

**Summary.** The plain (non-`_fast`) serial twin of the canonical routine. Same I/O and same
`(n_steps,3)`/`(n_steps,2)` outputs, but it pre-builds `step_idxs` vectorized and per-step calls
`get_stats_for_signal` (`:490`) instead of `get_stats_for_signal_window`. Functionally equivalent to
`_fast`; differs only in how it slices (`_pd = pd[start:end]` real slices vs. index-passing).

**Called from** — imported by `segmentation.py:27`, but the only call site is **commented out**
(`segmentation.py:723-725`). Effectively unused; `_fast` superseded it. Fan-in: 0 live.

**Analysis** — *Issues:* 🟡 dead-but-imported duplicate of `_fast`. Same divisibility assert (`:699`).
No direct test. `# @njit`/`prange` note from `_fast` does not apply (this one uses a plain `range`).

### `windowed_trimmed_circular_mean_and_stddev_fast2(v, pd, window_size, stride, trim)`  ·  `spf/rf.py:557`  ·  🔴 Broken (KNOWN_ISSUES #21)

**Summary.** A re-implementation that *inlines* everything (no helper calls for the mean/stddev) and
emits a **4-column** stats table `[circ_mean, circ_std, median|ch0|, median|ch1|]` — i.e. it splits
the amplitude median per antenna channel, unlike the 3-column canonical. It also reinterprets `trim`:
`trim<1` ⇒ fraction-per-tail (symmetric two-tail), `trim≥1` ⇒ absolute count per side — a **different
trim convention** from every other function in this cluster (which treat `trim` as a one-sided percent).

**Status: broken, and dead.**
- 🔴 **Type bug.** At `:612` and `:625` it calls `circular_diff_to_mean_single_fast(pd[k], mean_angle)`
  with a **scalar** `pd[k]`, but that function (`:234`) immediately does `angles.shape[0]` — scalars
  have no `.shape`, so under numba/numpy this raises. The intended scalar helper
  `circular_diff_to_mean_single_scalar` (`:548`) exists and is used correctly at `:585` but **not** at
  `:612`/`:625`. Confirms KNOWN_ISSUES #21.
- 🟡 **Decorator disabled.** `# @njit(parallel=True)` is commented out (`:556`) while the body uses
  `prange` (`:568`) and `math.*`; without njit `prange` is just `range`, so it would run as plain
  (slow) Python *if it ran at all*.
- 🧪 **Dead.** Imported at `segmentation.py:29` but its sole call site (`segmentation.py:661-663`) is
  **commented out**. Never executed.

**Analysis** — *Issues:* the function cannot currently produce output (scalar bug fires before return).
Because the only call site is commented, this is latent, not a live crash. *No test.* Flagged for a
maintainer: if the per-channel 4-column output is wanted, fix the scalar calls to use
`circular_diff_to_mean_single_scalar` (or pass a length-1 array) and restore the njit decorator. **No
code changed here.**

### `get_stats_for_signal_window(v, pd, start_idx, end_idx, trim)`  ·  `spf/rf.py:686`  ·  ✅ Active

**Summary.** Per-window worker for the canonical `_fast`: slices `pd[start:end]` and `v[:,start:end]`,
then returns `np.array([trimmed_cm, trimmed_stddev, abs_signal_median], float32)`. Builds the trimmed
mean from `circular_mean_single_fast(...)[1]`, feeds that mean into `circular_stddev_fast(...)[1]`, and
takes `fast_median_abs(v.flatten())`.

**Called from** — `windowed_trimmed_circular_mean_and_stddev_fast` (`:671`) only. `@njit`, pure.
Fan-in 1 (internal). No direct test (covered via segmentation).

**Analysis** — *Correctness:* note the **mean→stddev coupling**: stddev is measured about the *trimmed*
mean, and both the mean and stddev re-trim independently at the same `trim` percentile. Median uses
the **flattened** complex-magnitude-... actually `fast_median_abs` takes `|x|` of whatever it's given;
`v` here is complex, so `abs()` is the complex magnitude (numba supports `abs` on complex). Consistent
with `get_stats_for_signal`.

### `get_stats_for_signal(v, pd, trim)`  ·  `spf/rf.py:490`  ·  ✅ Active

**Summary.** The non-windowed sibling of `get_stats_for_signal_window` — identical computation but on a
whole `(v, pd)` rather than a slice. Returns the `(trimmed_cm, trimmed_stddev, abs_signal_median)`
3-tuple. `@njit`, pure.

**Called from** (fan-in: 2 prod)
- `segmentation.py:930` inside `recompute_stats_for_windows` — re-derives stats for merged windows
  from the original signal (return flows into `window["mean"/"stddev"/"abs_signal_median"]`).
- `windowed_trimmed_circular_mean_and_stddev` (`:714`) — the (dead) serial windower.
- No direct test; exercised via `recompute_stats_for_windows` in the segmentation tests.

**Analysis** — *Correctness:* same mean→stddev coupling as the window worker. `fast_median_abs(v.flatten())`
on complex `v` → magnitude median. Verified consistent with the `_window` variant.

### `circular_mean_single_fast(angles, trim, weights=None)`  ·  `spf/rf.py:300`  ·  ✅ Active

**Summary.** 1-D `@njit` circular mean with optional trimming and weights — the per-window mean engine.
Accumulates `Σsin`, `Σcos` (optionally weight-scaled), takes `arctan2` mod 2π for the untrimmed mean;
then computes per-sample circular distance to that mean (`circular_diff_to_mean_single_fast`), keeps
samples with `dist ≤ fast_percentile_1d(dists, 100−trim)`, and re-averages the kept sin/cos. Returns
`(pi_norm(cm), pi_norm(trimmed_cm))`.

**Signature**
- Inputs: `angles` (radians, 1-D); `trim` (percent `[0,100)`); `weights` (optional, broadcastable).
- Returns: `(mean, trimmed_mean)`, both ∈ `[-π,π)`. `trim==0.0` ⇒ both equal.
- Side effects: none; `@njit`.

**Called from** — `get_stats_for_signal_window` (`:690`), `get_stats_for_signal` (`:491`). Fan-in 2
internal. **No direct test** (the only `circular_mean*` tests cover `circular_mean`,
`circular_mean_simple_fast`, `torch_circular_mean`).

**Analysis** — *Correctness:* mirrors `circular_mean_single` (`:359`) but with the **ceil-rank**
`fast_percentile_1d` instead of `np.percentile`. The mask uses **unweighted** distances while the
re-average uses weighted sums — same subtlety flagged on `circular_mean` (Cluster A). Empty input
returns `(0.0, 0.0)` (`:305`).

### `circular_mean_single(angles, trim, weights=None)`  ·  `spf/rf.py:359`  ·  ✅ Defined; 🧪 no callers

**Summary.** numpy-percentile 1-D circular mean + trim + weights. Same math as
`circular_mean_single_fast` but uses `np.percentile(dists, 100−trim)` (interpolating) for the trim
threshold and asserts `ndim==1`. `@njit`.

**Called from** — **none in repo** (grep-verified). 🧪 dead. No test.

**Analysis** — *Issue:* 🟡 dead duplicate of `circular_mean_single_fast`, differing only in the
percentile convention (interpolating `np.percentile` vs. ceil-rank `fast_percentile_1d`) → results can
differ by one sample at small `n`.

### `circular_stddev_fast(v, mean, trim=50.0)`  ·  `spf/rf.py:94`  ·  ✅ Active (canonical stddev)

**Summary.** Circular standard deviation about a *given* `mean`, plus a trimmed version. Computes
per-sample circular distance `diffs = circular_diff_to_mean_single_fast(v, mean)`, then
`stddev = sqrt(Σdiffs² / (n−1))`; the trimmed stddev keeps `diffs ≤ fast_percentile_1d(diffs, 100−trim)`
and recomputes with `(trimmed_count − 1)` in the denominator. `@njit`.

**Signature**
- Inputs: `v` (radians 1-D); `mean` (radians scalar — typically the *trimmed* mean from
  `circular_mean_single_fast`); `trim` (percent, default 50.0).
- Returns: `(stddev, trimmed_stddev)`, both ≥ 0 (units: radians). `n≤1` ⇒ `(0.0,0.0)`;
  `trim==0.0` ⇒ both equal; `trimmed_count≤1` ⇒ trimmed=0.0.
- Side effects: none; `@njit`.

**Called from** — `get_stats_for_signal_window` (`:691`), `get_stats_for_signal` (`:494`). Fan-in 2
internal. **No direct test.**

**Analysis** — *Correctness:* sample stddev with `n−1` (Bessel). Because `mean` is the trimmed mean but
`diffs` are taken over **all** samples before re-trimming, the trimmed-stddev percentile is computed on
the full-window distances — consistent with how `circular_mean_single_fast` trims. Hand-checked sound.
*Issue:* none functional; it is a near-duplicate of `circular_stddev` (below) differing in percentile
backend.

### `circular_stddev(v, u, trim=50.0)`  ·  `spf/rf.py:134`  ·  ✅ Defined; 🧪 no callers

**Summary.** Vectorized twin of `circular_stddev_fast`: same circular-distance-to-`u` computation but
uses `np.percentile(diff, 100−trim)` for the trim threshold and numpy masking instead of a manual loop.
`@njit`. Returns `(stddev, trimmed_stddev)`.

**Called from** — **none** (grep-verified). 🧪 dead. No test.

**Analysis** — *Issue:* 🟡 dead duplicate; differs from `_fast` only in the percentile rule
(interpolating `np.percentile` vs. ceil-rank). Same one-sided `≤` mask convention.

### `circular_diff_to_mean_single_fast(angles, mean)`  ·  `spf/rf.py:234`  ·  ✅ Active (canonical distance)

**Summary.** Minimal **circular** distance from each angle to a scalar `mean`:
`min(|mean−a| mod 2π, 2π − that)` ∈ `[0, π]`. Explicit `@njit` loop, returns **float32**. The
wraparound-correct distance every trimmed stat depends on.

**Signature** — Inputs: `angles` (radians 1-D), `mean` (radians scalar). Returns: float32 `(n,)` ≥ 0,
≤ π. Side effects: none; `@njit`.

**Called from** — `circular_stddev_fast` (`:100`), `circular_stddev` (`:135`), `circular_mean_single_fast`
(`:325`), `circular_mean_single` (`:372`), **and (incorrectly, with a scalar)** the broken `_fast2`
(`:612,625`). Fan-in 4 valid + 1 misuse. Tested directly: `tests/test_rf.py:118`
(`test_circular_diff_to_mean_single`) asserts it matches `circular_diff_to_mean_single` to `1e-6` over
1000 random angles at several means. ✅ the one well-tested primitive here.

**Analysis** — *Correctness:* verified by test; wraparound handled via the `min(a, 2π−a)` trick. Note it
requires an **array** input (`angles.shape[0]`); passing a scalar (as `_fast2` does) is the #21 bug.

### `circular_diff_to_mean_single(angles, mean)`  ·  `spf/rf.py:224` · `circular_diff_to_mean(angles, means)`  ·  `spf/rf.py:247` · `circular_diff_to_mean_single_scalar(angle, mean)`  ·  `spf/rf.py:548`

The other three members of the circular-distance family:
- `circular_diff_to_mean_single` (`:224`, `@njit`): float64 vectorized version of `_single_fast`,
  asserts `ndim==1`. **Reference/oracle** — `circular_diff_to_mean_single_fast` is tested *against* it
  (`tests/test_rf.py:117-118`). Used only as the test oracle in prod-grep terms.
- `circular_diff_to_mean` (`:247`, `# @njit` disabled): distance of `angles` to **many** `means` (1-D),
  returns `(rows, n)`. Called by `circular_mean` (`:286`, Cluster A). Asserts `means.ndim==1`.
- `circular_diff_to_mean_single_scalar` (`:548`, `@njit`): scalar→scalar circular distance using a
  `diff>π` fold (signed-then-`abs`). Used **only** by `_fast2` (`:585`). Effectively dead with its host.

All four agree mathematically (modulo float32/64 and the signed-vs-min formulation, which coincide on
`[0,π]`). Duplication target.

### `fast_percentile_1d(x, percentile)`  ·  `spf/rf.py:348`  ·  ✅ Active (canonical percentile) · vs `fast_percentile(x, percentile)`  ·  `spf/rf.py:82`  ·  🧪 test-only

**Summary.** `fast_percentile_1d` is the `@njit` percentile used by every live trimmed stat: rank
`k = ceil(percentile/100 · n) − 1`, clamped to `[0,n−1]`, returned via `np.partition(x,k)[k]` (O(n), no
full sort). `fast_percentile` (`:82`, njit **commented out**) is a near-twin using **floor** rank
`k = floor(percentile/100 · (n−1))` — numpy's `method='lower'`.

**Two different conventions.**
- `fast_percentile` (floor / `(n−1)` scaling) — matches `np.percentile(..., interpolation='lower')`,
  which is exactly what its test asserts (`tests/test_rf.py:96-104`, `test_fast_percentile`). ✅ tested,
  but **only test-imported** (`tests/test_rf.py:12`, `tests/test_beamformer.py:18`) — no production caller.
- `fast_percentile_1d` (ceil / `n` scaling) — the **live** one (called by `circular_*_fast`/`stddev_fast`),
  but it is **untested directly** and uses a different rounding rule. At small `n` the two can pick a
  different sample (off-by-one rank).

**Analysis** — *Issue:* 🟠 the **tested** percentile (`fast_percentile`) is not the one in production;
the **production** percentile (`fast_percentile_1d`) is untested and rounds differently. Behavior of the
real trim threshold is therefore unverified. Empty input → `0.0` in both. Flagged; not fixed.

### `fast_median_abs(x)`  ·  `spf/rf.py:472`  ·  ✅ Active · vs `fast_abs_median(x)`  ·  `spf/rf.py:528`  ·  🧪 dead duplicate

**Summary.** Median of `|x|` via `np.partition` (handles even `n` by averaging the two middle order
statistics; empty → `0.0`). `@njit`. `fast_median_abs` and `fast_abs_median` are **byte-identical**
apart from a comment.

**Called from** — `fast_median_abs`: `get_stats_for_signal` (`:495`), `get_stats_for_signal_window`
(`:692`). `fast_abs_median`: **only** `_fast2` (`:643,644`) → dead with its host. Tested:
`tests/test_rf.py:139` (`test_fast_median_abs`) covers `fast_median_abs` over empty/odd/even/large
arrays vs `np.median(np.abs(...))`. ✅ well tested. `fast_abs_median` untested.

**Analysis** — *Issue:* 🟡 exact duplicate (`fast_abs_median`); only the dead `_fast2` uses it. *Correctness
of `fast_median_abs`:* verified by test (even-`n` two-middle average matches numpy).

### Torch variants (Cluster C)

None of these have any caller outside `rf.py` (grep-verified) — they form a parallel torch
implementation that the current pipeline does not invoke. They use **`torch.quantile`** (interpolating)
for trimming, a *third* percentile convention distinct from both numpy paths.

- **`torch_circular_stddev(v, u, trim)`** (`:157`) — torch twin of `circular_stddev`; trim threshold via
  `torch.quantile(diff, 1−trim/100)`. Called only by `torch_get_stats_for_signal`. No external caller.
- **`torch_get_stats_for_signal(v, pd, trim)`** (`:460`) — torch twin of `get_stats_for_signal`; returns
  `hstack(trimmed_cm, trimmed_stddev, abs_signal_median)`. Called only by
  `torch_windowed_trimmed_circular_mean_and_stddev`. No external caller.
- **`torch_windowed_trimmed_circular_mean_and_stddev(v, pd, window_size, stride, trim)`** (`:500`) — torch
  twin of the canonical windower; same divisibility assert (`:503`), `(n_steps,2)`/`(n_steps,3)` outputs.
  **No caller anywhere.** 🧪 entirely dead.
- **`torch_circular_diff_to_mean(angles, means)`** (`:59`) — torch twin of `circular_diff_to_mean`;
  asserts `means.ndim==1`. Used by `torch_circular_stddev`/`torch_circular_mean*` (the torch mean family,
  Cluster D-adjacent). Live *within* the torch mean path used by `torch_get_avg_phase*`.

**Analysis (torch block):** 🧪 the windowed-stats torch path is dead end-to-end; `torch_circular_stddev`
and `torch_get_stats_for_signal` are reachable only through it. They are untested. Their trim uses a
third (interpolating `torch.quantile`) convention, so numpy/torch parity for trimmed stats is **not**
guaranteed and **not** tested. (The torch *mean* family is partially live — see Cluster D.)

### Cluster C summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `windowed_trimmed_circular_mean_and_stddev_fast` | ✅ Active (canonical) | 1 prod (hot) | ❌ (via segmentation) | The live windower; `prange` runs serial under plain `@njit`. |
| `windowed_trimmed_circular_mean_and_stddev` | ⚠️ call site commented | 0 live | ❌ | Dead serial duplicate of `_fast`. |
| `windowed_…_fast2` | 🔴 Broken (#21) | 0 (commented) | ❌ | Scalar passed to array fn (`:612,625`); njit disabled; 4-col output. |
| `get_stats_for_signal_window` | ✅ Active | 1 internal | ❌ | Per-window worker; mean→stddev coupling. |
| `get_stats_for_signal` | ✅ Active | 2 prod | ❌ | Non-windowed twin; complex-magnitude median. |
| `circular_mean_single_fast` | ✅ Active | 2 internal | ❌ | Live trimmed mean; ceil-rank percentile. |
| `circular_mean_single` | 🧪 dead | 0 | ❌ | numpy-percentile duplicate of `_fast`. |
| `circular_stddev_fast` | ✅ Active (canonical) | 2 internal | ❌ | Live trimmed stddev. |
| `circular_stddev` | 🧪 dead | 0 | ❌ | numpy-percentile duplicate. |
| `circular_diff_to_mean_single_fast` | ✅ Active (canonical) | 4 + 1 misuse | ✅ | Only well-tested primitive; requires array input. |
| `circular_diff_to_mean_single` | ✅ test oracle | 0 prod | ✅ (as oracle) | Reference for `_fast`. |
| `circular_diff_to_mean` | ✅ Active | 1 (`circular_mean`) | ❌ | Many-means version. |
| `circular_diff_to_mean_single_scalar` | 🧪 dead | 1 (`_fast2`) | ❌ | Scalar dist; dies with `_fast2`. |
| `fast_percentile_1d` | ✅ Active (canonical) | many | ❌ | **Live percentile is untested**; ceil rank. |
| `fast_percentile` | 🧪 test-only | 0 prod | ✅ | Tested percentile is **not** the production one; floor rank. |
| `fast_median_abs` | ✅ Active | 2 prod | ✅ | Well tested. |
| `fast_abs_median` | 🧪 dead duplicate | 1 (`_fast2`) | ❌ | Byte-identical to `fast_median_abs`. |
| `torch_circular_stddev` / `torch_get_stats_for_signal` / `torch_windowed_…` | 🧪 dead | 0 ext | ❌ | Entire torch windowed-stats path unreached. |
| `torch_circular_diff_to_mean` | ✅ Active | torch-mean path | ❌ | Torch distance primitive. |

**Theme — three percentile conventions, and the wrong one is tested.** The live trim threshold
(`fast_percentile_1d`, ceil rank) is **untested**, while the *tested* `fast_percentile` (floor rank,
`method='lower'`) has no production caller; the torch path uses a third (`torch.quantile`,
interpolating). So the exact set of samples kept by trimming differs across backends and is unverified
on the path that actually runs. Layered on top: a dead `np.percentile` duplicate of every "fast"
function, two byte-identical median helpers, a four-way circular-distance family, a fully-dead torch
windowing stack, and one genuinely broken routine (`_fast2`, #21) kept alive only by a commented call
site. Only `circular_diff_to_mean_single_fast` and `fast_median_abs` have real unit tests. **No code
changed.**

---

## Cluster D — phase extraction

The front door of the whole RF stack: turn a raw 2-antenna complex signal matrix into a single
**average phase difference** between the antennas. That scalar (per snapshot) is the measurement the
filters and networks ultimately consume. The chain is always the same two steps — **(1)** per-sample
phase difference `pi_norm(∠s0 − ∠s1)` via `get_phase_diff`, then **(2)** a circular mean over the
samples — but it is forked into a numpy family (`get_avg_phase*`) and a torch family
(`torch_get_avg_phase*`), each with trim/no-trim and "fast" spellings. Angles are **radians**, phase
difference ∈ `[-π,π)`; `signal_matrix[0]` is antenna 0, `[1]` antenna 1 (sign convention matters for
downstream θ — see [`conventions.md`](../../00_concepts/conventions.md)).

`phi_to_signal_matrix` / `theta_to_phi` are **not present** in `rf.py` (grep-confirmed); the only
phase↔angle inversion is `phase_diff_to_theta`, documented in Cluster A.

### `get_phase_diff(signal_matrix)`  ·  `spf/rf.py:744`  ·  ✅ Active (canonical)

**Summary.** Per-sample phase difference between the two antennas: `pi_norm(∠s0 − ∠s1)`. The single
primitive feeding both the segmentation windowing (Cluster C) and the average-phase reducers below.
`@njit`.

**Signature**
- Inputs: `signal_matrix` (complex `np.ndarray (2, N)`); row 0 = antenna 0, row 1 = antenna 1.
- Returns: radians `(N,)`, each ∈ `[-π,π)` (wrapped by `pi_norm`).
- Side effects: none; `@njit`.

**Called from** (Pass 2 — fan-in: 4 prod)
- `segmentation.py:652, 714` — `pd = get_phase_diff(v).astype(np.float32)` feeding the windowed stats.
- `rf.py:757, 764, 770` — inside `get_avg_phase`, `get_avg_phase_fast`, `get_avg_phase_fast2`.
- No direct test; exercised through segmentation and the avg-phase tests' callers.

**Contract** — *Pre:* `signal_matrix.shape[0] == 2`, complex. *Post:* length-N radians in `[-π,π)`.
*Invariant:* `get_phase_diff(swap rows)` negates the result (antenna ordering sets the sign).

**Analysis** — *Correctness:* `np.angle` then subtract then `pi_norm` — wraparound-correct. The numba
twin of `torch_get_phase_diff`. No direct unit test (a gap, given it sets the measurement sign).

### `torch_get_phase_diff(signal_matrix)`  ·  `spf/rf.py:750`  ·  ✅ Active

**Summary.** Torch twin of `get_phase_diff`, but indexed for a **batched/leading-dim** layout:
`torch_pi_norm_pi(signal_matrix[:,0].angle() − signal_matrix[:,1].angle())` — note the antenna axis is
**dim 1** here (`[:,0]`/`[:,1]`), not dim 0 as in the numpy version.

**Signature** — Inputs: `signal_matrix` (complex tensor, antennas on **dim 1**, e.g. `(..., 2, N)`).
Returns: radians tensor ∈ `[-π,π)`. Side effects: none.

**Called from** (fan-in: 2 prod + torch avg-phase)
- `spf/dataset/spf_dataset.py:720, 1414` — `pd = torch_get_phase_diff(data["signal_matrix"][0]).to(float32)`.
  The dataset tensor is `signal_matrix ~ (1, 1, 2, N)` (comment at `spf_dataset.py:717`); after `[0]` it
  is `(1, 2, N)`, so `[:,0]`/`[:,1]` correctly select **antenna 0/1 on dim 1** (the leading size-1 dim is
  preserved). The `[:,...]` indexing is right *for a 3-D `(batch, 2, N)` input*, which is what callers pass.
- `torch_get_avg_phase_notrim`/`torch_get_avg_phase` (`:788, 797`) — there `[None]` adds the batch dim so
  the same `(batch, 2, N)` layout holds.

**Analysis** — *Correctness:* the antenna axis is **dim 1**, opposite to `get_phase_diff`'s **dim 0** —
intentional, because torch callers always supply a leading batch dim (`(batch, 2, N)`), verified against
the `(1,1,2,N)` dataset shape at `spf_dataset.py:717-720`. Not a bug, but the row-vs-batched indexing
mismatch between the numpy and torch twins is a footgun if `torch_get_phase_diff` is ever handed a bare
`(2, N)` matrix (it would diff two time samples). No direct test.

### `get_avg_phase(signal_matrix, trim=0.0)`  ·  `spf/rf.py:755`  ·  ✅ Active (canonical numpy)

**Summary.** The standard average-phase reducer: `get_phase_diff` → `circular_mean(pd[None], trim)` →
flatten to a length-2 array. Returns **two equal values** (the `(mean, trimmed_mean)` pair of
`circular_mean`, equal because the default `trim=0.0`). Used by the live data collector.

**Signature**
- Inputs: `signal_matrix` (complex `(2, N)`); `trim` (percent, default 0.0).
- Returns: `np.ndarray (2,)` radians (both entries equal when `trim==0.0`).
- Side effects: none. (`# @njit` commented — runs as plain numpy, because it calls the non-njit
  `circular_mean`.)

**Called from** — `data_collector.py:159` (`avg_phase_diff = get_avg_phase(signal_matrix)`), via
`data_to_snapshot` on the legacy V2/V3 collection path. Fan-in 1 prod. No direct test.

**Analysis** — *Correctness:* delegates to the Cluster-A `circular_mean` (untrimmed path verified). The
2-vector shape is a quirk: it just exposes `circular_mean`'s `(mean, trimmed)` tuple as an array. *Issue:*
inherits `circular_mean`'s `# TODO remove` status (Cluster A) — if `circular_mean` is migrated, this
caller moves too.

### `get_avg_phase_fast(signal_matrix)`  ·  `spf/rf.py:762`  ·  🧪 Defined; no callers · `get_avg_phase_fast2(signal_matrix)`  ·  `spf/rf.py:769`  ·  ✅ Active

**Summary.** Two "fast" numpy reducers:
- `get_avg_phase_fast` (`:762`): `get_phase_diff` → `circular_mean_simple_fast(pd[None])` → flatten.
  No trim. `# @njit` commented. **No caller anywhere** (grep) — 🧪 dead.
- `get_avg_phase_fast2` (`:769`): fully inlined `@njit` — computes `Σcos`/`Σsin` over `pd` and returns
  `np.array([pi_norm(arctan2), same], float32)`. **This is the live realtime reducer.**

**Called from** — `get_avg_phase_fast2`: `data_collector.py:291`
(`avg_phase_diff = get_avg_phase_fast2(signal_matrix)`) on the V4/V5 raw collection path (`ThreadedRXRaw`).
Fan-in 1 prod. `get_avg_phase_fast`: 0. Neither has a direct test.

**Analysis** — *Correctness:* `_fast2` is an untrimmed circular mean, equivalent to `_fast` and to
`get_avg_phase(trim=0)` up to float32 and the percentile-free path. Returns a duplicated 2-vector to
match the `(mean, trimmed)` shape contract even though it never trims. *Issue:* 🟡 `get_avg_phase_fast`
is dead; `_fast2`'s 2-vector duplication is cosmetic. No test on the live realtime reducer.

### `torch_get_avg_phase_notrim(signal_matrix)`  ·  `spf/rf.py:785`  ·  ✅ Active · `torch_get_avg_phase(signal_matrix, trim)`  ·  `spf/rf.py:794`  ·  🧪 no callers

**Summary.** Torch average-phase reducers:
- `torch_get_avg_phase_notrim` (`:785`): `torch_get_phase_diff` → `torch_circular_mean_notrim(pd[None])`
  → `hstack` to a length-2 tensor. **Live** on the fake-dataset path.
- `torch_get_avg_phase` (`:794`): trimmed version via `torch_circular_mean(pd[None], trim)`, wrapped in
  `torch.tensor(...)`. **No caller** (grep) — 🧪 dead; the `torch.tensor(tuple_of_tensors)` wrap is also
  suspect (constructing a tensor from a tuple of tensors).

**Called from** — `torch_get_avg_phase_notrim`: `spf/dataset/fake_dataset.py:289`
(`"avg_phase_diff": torch_get_avg_phase_notrim(signal_matrix)`). Fan-in 1 prod. `torch_get_avg_phase`: 0.
No direct test.

**Analysis** — *Correctness (notrim):* delegates to `torch_circular_mean_notrim` (untrimmed sin/cos mean,
straightforward). *Issue:* ⚠️ `torch_get_avg_phase` (trim) wraps a tuple of tensors in
`torch.tensor(...)` (`:795`), which is fragile/likely-erroring — but it is **dead**, so latent. The torch
trimmed mean family (`torch_circular_mean`, `:432`) is itself only tested via the shadowed Cluster-A test.

### Torch circular-mean family (Cluster D support)  ·  `rf.py:381–456`

Backing the torch reducers above:
- `torch_circular_mean_notrim(angles)` (`:381`) — per-row untrimmed circular mean; **live** (via
  `torch_get_avg_phase_notrim`). No direct test.
- `torch_circular_mean(angles, trim, weights=None)` (`:432`) — full torch mean+trim+weights; tested
  **only** by the **shadowed** `tests/test_rf.py:18` test (which never runs — see Cluster A finding), so
  effectively untested. Reachable via the dead `torch_get_avg_phase`.
- `torch_circular_mean_noweight(angles, trim)` (`:392`) — trim, no weights; **no callers** 🧪.
- `mean_phase_mean(angles, weights)` (`:416`) — numpy weighted circular mean (scalar out); **live** at
  `segmentation.py:96` (`mean_phase_from_simple_segmentation`). Asserts `weights` finite.
- `torch_mean_phase_mean(angles, weights)` (`:424`) — 🔴 **broken + dead** (KNOWN_ISSUES #20): uses
  `np.sin`/`np.cos`/`np.arctan2` on its (torch) inputs and then calls `torch_pi_norm` — mixing numpy ops
  on tensors. **No callers** (grep-confirmed), so latent. Confirms #20 precisely: both *broken* (numpy
  ops on torch input) and *dead* (zero call sites).

### Cluster D summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `get_phase_diff` | ✅ Active (canonical) | 4 prod | ❌ | Sets the measurement sign; untested. |
| `torch_get_phase_diff` | ✅ Active | 2 prod | ❌ | Antenna axis indexed on **dim 1** (`[:,0]`) vs numpy dim 0 — verify shape. |
| `get_avg_phase` | ✅ Active | 1 prod | ❌ | Canonical numpy reducer; rides `circular_mean`'s `TODO remove`. |
| `get_avg_phase_fast` | 🧪 dead | 0 | ❌ | No callers. |
| `get_avg_phase_fast2` | ✅ Active | 1 prod | ❌ | Live realtime reducer (V4/V5); untested. |
| `torch_get_avg_phase_notrim` | ✅ Active | 1 prod | ❌ | Live on fake-dataset path. |
| `torch_get_avg_phase` | 🧪 dead | 0 | ❌ | Suspect `torch.tensor(tuple)` wrap; dead. |
| `torch_circular_mean_notrim` | ✅ Active | 1 (torch reducer) | ❌ | — |
| `torch_circular_mean` | ⚠️ effectively untested | 1 (dead path) | ⚠️ (shadowed) | Only test is the shadowed one (Cluster A). |
| `torch_circular_mean_noweight` | 🧪 dead | 0 | ❌ | No callers. |
| `mean_phase_mean` | ✅ Active | 1 prod | ❌ | Live weighted mean in segmentation. |
| `torch_mean_phase_mean` | 🔴 broken + dead (#20) | 0 | ❌ | numpy ops on torch input; no callers. |

**Theme — the live reducers are the untested ones.** Every average-phase value that actually flows into
the pipeline comes from `get_phase_diff` + one of three live reducers (`get_avg_phase`,
`get_avg_phase_fast2`, `torch_get_avg_phase_notrim`) — **none of which has a direct test**. The tested
torch trimmed-mean lives behind a shadowed test that never runs, and the broken member
(`torch_mean_phase_mean`, #20) is safely dead. The most useful additions would be a direct
`get_phase_diff` sign/wrap test and a numpy/torch parity test on the live reducers — plus resolving the
`torch_get_phase_diff` antenna-axis question. **No code changed.**
