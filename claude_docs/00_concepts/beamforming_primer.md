# Beamforming primer

The physics that turns raw antenna signals into an angle estimate. This is the conceptual
core of SPF; the implementation lives in `spf/rf.py` (documented in
[`02_rf_signal_processing/`](../02_rf_signal_processing/overview.md)).

## The geometry

A plane wave arriving from angle θ hits the two antenna elements (separated by distance `d`)
at slightly different times. That time offset shows up as a **phase difference** φ between the
two received signals:

```
path difference  Δd = d · sin(θ)
phase difference φ  = 2π · Δd / λ = 2π · (d/λ) · sin(θ)
```

So if you can measure φ, you can solve for θ:

```
sin(θ) = φ / (2π · d/λ)        ⇒   θ = arcsin( clip( φ / (2π·d/λ) ) )
```

This is the whole game. `d/λ` (`rx_wavelength_spacing`) is the conversion constant.

### Why it's ambiguous

`arcsin` is two-to-one: `+θ` and `π−θ` produce the same φ. With only 2 elements you cannot tell
front from back. SPF resolves this by (a) constraining a single receiver to `[-π/2, π/2]`,
(b) fusing the **two receivers** of a craft (different array orientations), and (c) applying
symmetry when building empirical `P(θ|φ)` tables. See
[conventions.md → front/back ambiguity](conventions.md#the-frontback-ambiguity).

## Two ways SPF extracts θ

SPF computes the angle estimate two complementary ways, and both feed downstream models:

### 1. Direct phase-difference estimate

Measure φ between the elements directly (circular mean of per-sample phase differences, with
trimming to reject outliers), then apply the `arcsin` relation. Robust statistics matter here
because φ wraps at ±π — naive averaging fails near the boundary, so SPF uses **circular**
mean/stddev throughout.

### 2. Delay-and-sum beamforming

Instead of solving for θ, *scan* it:

1. Pick a grid of candidate angles `θ_i`, `i = 1..nthetas` (65 typical).
2. For each θ_i, build a **steering vector** `s(θ_i) = e^{-j φ(θ_i)}` — the phase pattern a signal
   from θ_i would produce across the elements.
3. Apply it to the signal and measure output power `P(θ_i) = | s(θ_i)ᴴ · x |`.
4. The result is a **power-vs-angle spectrum** of length `nthetas`. Peaks ⇒ likely arrival angles.

Beamforming naturally produces a *distribution* over angle (good for fusing multiple receivers
and for feeding a network), whereas the direct estimate produces a single φ. SPF uses both.

## How this connects to the rest of the system

```
 raw IQ (2 × N)
      │
      ├─▶ circular-mean φ  ──────────────▶  φ  (and arcsin θ)        ┐
      │                                                              ├─▶ features for NN / filters
      └─▶ steering vectors ─▶ delay-and-sum ─▶ P(θ) spectrum (×65)   ┘
```

- **Segmentation** ([`03_datasets`](../03_datasets/overview.md)) runs this per window and caches
  the windowed beamformer + window phase stats.
- **NN training** ([`04_training_inference`](../04_training_inference/overview.md)) consumes the
  beamformer spectrum and/or phase stats and learns a calibrated `P(θ)` per snapshot.
- **Filters** ([`05_filters`](../05_filters/overview.md)) fuse the per-snapshot θ (from empirical
  table or NN) over time via the same `φ = 2π·(d/λ)·sin(θ)` observation model.

## Key terms recap

- **Steering vector** — the per-angle phase template, `(nthetas, n_elements)`.
- **`d/λ`** — antenna spacing in wavelengths, the φ↔θ constant.
- **Circular statistics** — mean/stddev that respect the ±π wraparound; used everywhere φ is averaged.
- **nthetas** — number of angle bins in the scan / network output (65 typical).
