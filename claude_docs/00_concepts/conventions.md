# Conventions: units, angles, coordinate frames

These conventions are assumed silently across the codebase. Getting one wrong (degrees vs
radians, mm vs m, which frame θ is in) is the most common source of bugs.

## Units

| Quantity | Unit | Notes |
|---|---|---|
| Angles (θ, φ) | **radians** | Normalized to `[-π, π]` via `pi_norm()` / `torch_pi_norm()`. Plain `%` does **not** handle negatives correctly. |
| Orientation fields (`*_in_pis`) | **multiples of π** | e.g. `rx_theta_in_pis = 0.5` means π/2 rad. Stored this way in zarr; convert before trig. |
| Positions in zarr (`*_pos_x_mm`) | **millimeters** | Raw storage. |
| Positions in dataset tensors (`*_pos_xy`) | **meters** | The dataset divides mm by 1000 on the way out. |
| Antenna spacing (`rx_spacing`) | **meters** | Physical distance between the 2 elements. |
| Spacing in wavelengths (`rx_wavelength_spacing`, `d/λ`) | dimensionless | `rx_spacing / wavelength`; drives φ↔θ. |
| Frequencies (`rx_lo`, `rx_bandwidth`, `f-sampling`) | **Hz** | |
| Gains | **dB** | |
| Timestamps | **Unix seconds** (float) | `system_timestamp`; v4 also carries `gps_timestamp`. |

## Angle frame conventions

There are **several angle frames** in play; the dataset exposes them under different keys.
Confusing them is a classic bug, so they are spelled out here:

- **Array-relative θ (`y_rad` / `ground_truth_theta`)** — angle of the emitter relative to the
  receiver's antenna-array boresight. This is what a single receiver can physically estimate.
- **Craft-relative θ (`craft_y_rad` / `craft_ground_truth_theta`)** — angle relative to the
  vehicle's heading (combines both receivers' geometry).
- **Absolute θ (`absolute_theta`)** — angle in a world frame (e.g. rotated to north using
  `rx_heading_in_pis`).
- **φ targets (`y_phi`)** — the corresponding phase-difference target.

The `_in_pis` orientation fields (`rx_theta_in_pis`, `rx_heading_in_pis`) are what rotate
between these frames.

### The front/back ambiguity

A **2-element** array cannot distinguish a signal arriving from `+θ` vs `π−θ` (front vs back of
the array) — `arcsin` is two-to-one. The code handles this explicitly (returns multiple angle
candidates; uses symmetry when building empirical distributions). Single-receiver estimates are
typically constrained to `θ ∈ [-π/2, π/2]`. Fusing the two receivers of a craft is what resolves
the ambiguity. See [beamforming_primer.md](beamforming_primer.md).

## Tensor shape conventions

| Object | Shape | dtype |
|---|---|---|
| `signal_matrix` (one snapshot) | `(2, buffer_size)` | `complex64` |
| Steering vectors | `(nthetas, n_receiver_elements)` | `complex64/128` |
| Beamformer spectrum | `(nthetas,)` per snapshot | `float32` |
| Windowed beamformer | `(n_windows, nthetas)` | `float16` (cached) |
| Per-window stats | `(3, n_windows)` → mean φ, stddev φ, median abs signal | `float16` (cached) |

## Speed of light / wavelength

`c = 299_792_458 m/s`; `wavelength = c / f_carrier`. Used to turn `rx_spacing` (m) into `d/λ`.

## Naming gotchas (read once, save hours)

- **φ = phase difference, θ = angle of arrival.** Some files use `phi`/`theta` loosely; when in
  doubt, φ is the raw observable, θ is the estimated direction.
- **`rotate_dist()`** rotates a *probability distribution over angles*, not a scalar angle.
- **`paired`** = both radios of one craft, not ML pairs.
- **`nthetas` (global) vs `output_ntheta` (model)** can differ — two separate discretizations.
