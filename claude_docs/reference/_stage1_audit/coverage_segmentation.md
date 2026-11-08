# Stage-1 SEGMENTATION coverage audit — SEG-1 & SEG-2

READ-ONLY correctness + test-coverage audit. **No source/test code modified.** Config traced:
`model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`, model `beamformer`.
Evidence pinned to current `main` (commit `574c454`). Confidence: ✅ verified (read deciding
lines / ran a grep) · 🔍 analysis (reasoned, not executed).

Verified independently against source. The pre-existing audits
(`layer1_data_segmentation.md`, `segmentation.md`, `rf.md`) were used as a map and are accurate;
this report re-derives every claim from the source lines and adds concrete test proposals.

---

## SEG-1 — Windowed beamformer production (`windowed_beamformer`, the model's main input)

### Correctness verdict: ✅ CORRECT for this config (delay-and-sum + windowing + f16 verified)

**Live path.** `segment_session` (`segmentation.py:375`) → beamformer branch (`:496-540`):
- GPU: `beamformer_given_steering_nomean_cp(cp.asarray(steering), cp.asarray(v))` (`:504-512`).
- CPU: `beamformer_given_steering_nomean` (default) or `_fast` if `fast_beamformer=True` (`:515-527`).
  `render_session` passes `fast_beamformer=True` (`spf_dataset.py:752` path); the `.yarr` build
  (`mp_segment_zarr`) uses GPU or plain CPU. `fast_beamformer` flows to `_fast` (numba).

**Delay-and-sum math (verified).**
- `beamformer_given_steering_nomean_cp/_nomean` (`rf.py:1266,1302`): `np.absolute(np.dot(steering, signal))`
  — for each θ, `|Σ_ant a(θ)·x_ant|`. No conjugate operator; the conjugate is baked into the precomputed
  steering vectors (`exp(-j…)`), consistent with `beamformer` (the end-to-end tested pipeline). ✅
- `_fast_core` (`rf.py:1222-1236`): manual complex multiply, `sqrt(real²+imag²)` — "NO conjugate"
  comment matches the numpy path. `tests/test_rf.py:53 test_beamformer_functions` asserts
  `_nomean` vs `_nomean_fast` agree to `rtol=1e-4` over 10 random complex64 trials. ✅ This is the
  ONLY direct numerics test that touches a function on the SEG-1 path, and it does NOT cover `_cp`
  (the GPU `.yarr`-build path) — though `_cp` is byte-identical numpy source to `_nomean` (`rf.md`
  Cluster B), so the parity test transitively covers its math.

**Windowing / stride (2048/2048) (verified).**
`...reshape(nthetas, -1, window_size).mean(axis=2).T` (`:509-511`, `:524-526`).
- Beamformer output is `(nthetas, n_samples)` with `n_samples = n_windows·window_size`.
  `reshape(nthetas, -1, window_size)` groups **consecutive** samples into windows (C-order, last
  axis fastest) → window `w` = samples `[w·2048:(w+1)·2048)`. ✅ correct grouping.
- `.mean(axis=2)` → per-window mean **magnitude** (mean of |beam|, NOT power |·|²; the magnitude is
  taken per-sample before averaging). `.T` → `(n_windows, nthetas)`. Then `[None]` prepends a
  singleton at `:632` → `(1, n_windows, nthetas)`. Matches the contract θ-on-last-axis (dim 3 after
  collate). ✅
- ⚠️ **NON-OBVIOUS but correct:** this windowing has **no `stride` parameter** — it is a pure
  reshape, so it implicitly assumes **stride == window_size** (non-overlapping, contiguous, no
  dropped tail). `DEFAULT_SEGMENT_ARGS` sets `window_size == stride == 2048` (`:40-41`) and the
  capture buffer (524288) is an exact multiple (256 windows), so this holds. If `stride != window_size`
  were ever set, the windowed_beamformer windowing would **silently diverge** from the segmentation
  windowing in `windowed_trimmed_circular_mean_and_stddev_fast` (which DOES use stride). Not reachable
  for this config; flagged as a latent coupling.
- `skip_beamformer=True` fills with NaN (`:536-540`) — not used on this config's build path.

**float16 precision (verified).**
`segmentation_results["windowed_beamformer"].astype(np.float16)` (`:530-532`); cache dtype f16
(`zarr_utils.py`), upcast to f32 in collate. Per-window mean magnitudes are O(10²–10³) for these
captures — well inside f16 range (max 65504) and ~3-digit relative precision. The amplitude
**saturates to inf above 65504** and loses integer precision above 2048 (same boundary as F9/S1),
but for the nosig rover/wall captures the per-window beam magnitude stays small. P2/P3 precision
boundary, no crash. The f16 round-trip is **untested**.

**KNOWN_ISSUES #37 applicability to SEG-1:** N/A for the beamformer writer itself — #37 is about
`fast_percentile_1d` (a stats-path function); the windowed-beamformer path does not call it. The
`test_circular_mean` shadowing also does not touch SEG-1.

### Existing test coverage (SEG-1)
- **`tests/test_rf.py:53 test_beamformer_functions`** — asserts `beamformer_given_steering_nomean`
  (numpy) == `beamformer_given_steering_nomean_fast` (numba) to `rtol=1e-4`, complex64, 10 trials.
  Covers the **per-sample delay-and-sum magnitude only**. Does NOT cover: the `_cp` GPU variant
  directly, the `reshape→mean(2)→.T` per-window aggregation, the f16 cast, or the writer that fills
  `windowed_beamformer` in `segment_session`.
- **`tests/test_beamformer.py:25 test_beamformer`** — end-to-end peak-location on a simulated source
  (asserts spectrum argmax ≈ ground-truth angle). Validates steering-vector + dot-product physics but
  uses the `beamformer()` pipeline, **not** `segment_session`'s windowed writer.
- **NO direct test** of `segment_session`'s `windowed_beamformer` output (the windowing/aggregation/
  f16 chain). `grep windowed_beamformer tests/` → 0 hits. ✅ verified.

### Gaps (SEG-1)
1. **The windowed aggregation `reshape(nthetas,-1,window_size).mean(2).T` is untested.** A transpose-
   or reshape-order bug (e.g. window/θ axes swapped, or Fortran vs C grouping) would pass every
   existing test and silently feed the model a permuted main input.
2. **f16 cast untested** — no assertion that the f16 round-trip preserves the beam shape/argmax.
3. **GPU `_cp` path untested** (no GPU in CI); relies on NEP-18 dispatch identity with `_nomean`.
4. **The implicit `stride==window_size` assumption in the reshape is unguarded and unasserted.**

### Proposed tests (SEG-1)
- **`test_windowed_beamformer_aggregation` (tests/test_segmentation.py)** — *highest value.*
  Build a synthetic `(2, n_windows·2048)` signal with a **known per-window dominant angle** (inject a
  different steering-consistent phase ramp into each window), run `segment_session(v, gpu=False,
  steering_vectors=precompute_steering_vectors(...), window_size=2048, stride=2048, ...)`, and assert
  `windowed_beamformer[0]` has shape `(n_windows, nthetas)` AND `argmax(axis=1)` per window equals the
  injected angle bin. Would catch: reshape/transpose axis bugs, window↔θ swap, wrong window grouping,
  and an empty/NaN writer. (No existing test exercises this writer at all.)
- **`test_windowed_beamformer_matches_manual_loop` (tests/test_segmentation.py)** — assert the
  `reshape(...).mean(2).T` result equals a plain Python loop computing
  `mean_w |steering·signal[:, w*2048:(w+1)*2048]|` per window, to `1e-4` in float32 BEFORE the f16
  cast. Would catch the same axis/order bugs with a reference oracle and pin the "mean-of-magnitude
  (not power)" semantics.
- **`test_windowed_beamformer_float16_roundtrip` (tests/test_segmentation.py)** — assert the f16
  cache cast preserves per-window argmax and stays finite for representative amplitudes; document the
  65504 saturation boundary. Would catch a future amplitude-scale regression (F9/S1).

---

## SEG-2 — Per-window circular stats (`all_windows_stats` = trimmed circ-mean φ, trimmed circ-stddev φ, median|signal|)

### Correctness verdict: ⚠️ CORRECT math, but the production path is effectively UNTESTED and the trim percentile convention is unverified on the live function

**Live path (verified, `segmentation.py:658` → `rf.py`):**
`get_all_windows_stats` (`segmentation.py:641`, `@njit`) phase-diffs via
`get_phase_diff(v).astype(float32)` (`:652`) then calls
`windowed_trimmed_circular_mean_and_stddev_fast` (`rf.py:657`) →
`get_stats_for_signal_window` (`rf.py:686`) → {`circular_mean_single_fast` (`:300`),
`circular_stddev_fast` (`:94`), `fast_median_abs` (`:472`)}, all leaning on
`fast_percentile_1d` (`:348`) and `circular_diff_to_mean_single_fast` (`:234`).
`all_windows_stats` is then `.astype(float16).T` (`:569-570`) → rows `[circ_mean, circ_stddev,
median|abs|]`.

**Wraparound at ±π (verified ✅).**
- `get_phase_diff` (`rf.py:746`): `pi_norm(angle(x0) − angle(x1))` → φ ∈ `[-π, π)`. ✅
- `pi_norm` (`rf.py:49`): `((x+π) % 2π) − π` — Python `%` has divisor sign, so negatives wrap
  correctly; endpoint `π → −π` (half-open). ✅
- `circular_mean_single_fast` (`:300`): accumulates `Σsin/Σcos`, `arctan2 % 2π`, returns
  `pi_norm(cm), pi_norm(trimmed_cm)`. Wraparound-correct (sin/cos avoids the ±π discontinuity). ✅
- `circular_diff_to_mean_single_fast` (`:234`): `d = |mean−a| % 2π; min(d, 2π−d)` ∈ `[0,π]` — the
  wraparound-correct circular distance. **This is the one well-tested primitive** (see below). ✅
- `circular_stddev_fast` (`:94`): distances about the **trimmed** mean, `sqrt(Σd²/(n−1))` (Bessel);
  trimmed version re-thresholds with `(trimmed_count−1)` denom. Math sound; `n≤1 → (0,0)`. ✅

**Trim percentile convention (verified ⚠️).**
- `trim` is a **percent of farthest points to drop** (production `trim=20.0`, `DEFAULT_SEGMENT_ARGS:42`).
- Applied as a **one-sided** keep-mask: `dist ≤ fast_percentile_1d(dists, 100−trim)`
  (`circular_mean_single_fast:328,335`; `circular_stddev_fast:114,120`). NOT a symmetric two-tail trim
  — correct for a distance-from-mean array (all distances ≥0, so "farthest" = largest).
- `fast_percentile_1d` (`:348`) uses **ceil rank**: `k = ceil(p/100·n) − 1`, clamped, `np.partition(x,k)[k]`.
  This DIFFERS from `fast_percentile` (`:82`, **floor** rank `floor(p/100·(n−1))`, `method='lower'`)
  and from `np.percentile`/`torch.quantile` (interpolating). At small `n` they pick a different sample
  (off-by-one rank). For the production windows `n = window_size = 2048`, the rank difference is at most
  one sample out of 2048 → negligible effect on the kept set, but the exact threshold is **convention-
  dependent and unverified**.

**KNOWN_ISSUES #37 — CONFIRMED ✅ (applies to this live path).**
The production trim threshold is `fast_percentile_1d` (`rf.py:348`), which has **NO direct test**
(`grep fast_percentile_1d tests/` → 0 hits). The *tested* percentile is `fast_percentile`
(`tests/test_rf.py:77 test_fast_percentile`, imported at `test_rf.py:12`), which is **floor-rank and
has no production caller**. So the divergent-convention concern is real: the rule that actually runs is
unverified, and the verified rule is not the one that runs. ✅ refutes nothing; confirms #37 as written.

**`test_circular_mean` duplicate-name shadowing — CONFIRMED ✅ (relevant, but indirect to SEG-2).**
`tests/test_rf.py:18` and `:151` both define `test_circular_mean`; Python binds the **second**
(`:151`), so the first never runs under pytest. ✅ verified (two `def test_circular_mean` lines). The
shadowed `:18` test would have covered the **trimmed** mean (`trim=20`), the **weighted** path, and
`torch_circular_mean` parity. **Caveat for SEG-2:** even if un-shadowed, that test targets
`circular_mean` (`rf.py:273`, the np.percentile 2-D variant) and `torch_circular_mean` — **NOT** the
live `circular_mean_single_fast` (`rf.py:300`). So fixing the shadow would still leave the production
trimmed-mean engine untested; the shadow is corroborating evidence of weak trim coverage, but the live
function needs its own test regardless.

### Existing test coverage (SEG-2)
- **`tests/test_rf.py:108 test_circular_diff_to_mean_single`** — asserts
  `circular_diff_to_mean_single_fast` == `circular_diff_to_mean_single` to `1e-6` over 1000 random
  angles at means `{0, π/2, π, 3π/2, 2π}`. ✅ The one solid wraparound test on the SEG-2 path (covers
  the distance primitive used by both mean and stddev).
- **`tests/test_rf.py:126 test_fast_median_abs`** — asserts `fast_median_abs` == `np.median(np.abs(·))`
  over empty/single/two/odd/even/large arrays to `1e-6`. ✅ Covers channel-2 (median|abs|) directly.
- **`tests/test_rf.py:77 test_fast_percentile`** — covers `fast_percentile` (floor rank), **NOT** the
  live `fast_percentile_1d` (ceil rank). ❌ wrong function for production.
- **`tests/test_rf.py:151 test_circular_mean`** — covers `circular_mean` (untrimmed) vs
  `circular_mean_simple_fast`. ❌ not the live `circular_mean_single_fast`; untrimmed only.
- **`tests/test_segmentation.py:5,50 test_simple_segment*`** — call `simple_segment` (which calls the
  whole live stats stack transitively) but assert ONLY recovered segment `start_idx`/`end_idx`/`mean`
  on a happy-path 3-region synthetic with `window_size=100, stride=50, trim=10, min_abs_signal=10`.
  They do **not** pin any `all_windows_stats` value, do not test the stddev or median channels, and use
  `stride≠window_size` (which the production config never does). Coarse, indirect coverage only.

**Functions on the live SEG-2 path with NO direct test:** `circular_mean_single_fast`,
`circular_stddev_fast`, `fast_percentile_1d`, `windowed_trimmed_circular_mean_and_stddev_fast`,
`get_stats_for_signal_window`, `get_stats_for_signal`, `get_all_windows_stats`. (✅ grep-verified: none
appear in `tests/`.)

### Gaps (SEG-2)
1. **The production trimmed circular mean (`circular_mean_single_fast`) is untested** — no assertion it
   recovers a known mean, handles wraparound-spanning inputs (φ near ±π), or trims correctly.
2. **`circular_stddev_fast` is untested** — the mean→stddev coupling and the Bessel/`trimmed_count−1`
   denominator are unverified.
3. **`fast_percentile_1d` (the live trim threshold) is untested** (#37) — and rounds differently from
   the tested `fast_percentile`.
4. **The windowed writer `windowed_trimmed_circular_mean_and_stddev_fast` is untested** — window
   indexing (`step_idxs[:,1]−step_idxs[:,0]==window_size`), the divisibility assert, and the
   per-window stat values are unverified.
5. **f16 cast of `all_windows_stats` untested** — phase resolution ~9.8e-4 rad is fine, but the
   amplitude channel saturates >65504 / loses precision >2048 (F9/S1), unverified.

### Proposed tests (SEG-2)
- **`test_circular_mean_single_fast_wraparound` (tests/test_rf.py)** — *highest value.*
  For inputs clustered near ±π (e.g. angles drawn from `pi_norm(N(π, 0.2))` so they straddle the
  branch cut) and several known ground-truth means across `[-π,π)`, assert
  `circular_mean_single_fast(angles, trim=20)[1]` matches a **reference numpy circular mean**
  (`arctan2(mean(sin), mean(cos))` with an independent percentile-trim) within `1e-5` for the
  untrimmed case and `~1e-2` for the trimmed case. Include a single-element and all-equal-angle row.
  **Would have caught:** any wraparound regression in the live trimmed mean (the model's φ channel),
  and is the direct test that #37 says is missing. This is the single most load-bearing untested
  function feeding the model.
- **`test_fast_percentile_1d_convention` (tests/test_rf.py)** — assert `fast_percentile_1d` returns
  the **ceil-rank** order statistic on `[1..n]` arrays for `p ∈ {0,25,50,75,100}` and small `n` (3,5),
  documenting where it differs from `fast_percentile`/`np.percentile`. **Would have caught** a silent
  change to the trim rounding rule (#37 — the live trim threshold is currently unverified).
- **`test_circular_stddev_fast_reference` (tests/test_rf.py)** — vs a numpy reference circular stddev
  about a given mean (`sqrt(Σ circ_dist² /(n−1))`), untrimmed and trimmed; assert `n≤1 → (0,0)`.
  **Would have caught** a Bessel/denominator or mean-coupling regression in the model's stddev channel.
- **`test_windowed_stats_fast_values` (tests/test_segmentation.py or test_rf.py)** — on a synthetic
  `(2, 3·2048)` signal with a **known per-window phase** and amplitude, call
  `windowed_trimmed_circular_mean_and_stddev_fast(v, pd, 2048, 2048, trim=20)` and assert (a)
  `step_idxs` row spans == 2048, (b) row-0 circ-mean ≈ injected phase, (c) row-2 median ≈ injected
  amplitude. **Would have caught** a window-indexing or stat-channel-ordering bug in the writer that
  fills the model's `all_windows_stats`.

---

## Cross-segment summary

| | SEG-1 windowed_beamformer | SEG-2 all_windows_stats |
|---|---|---|
| Correctness | ✅ correct (math + windowing + f16) for stride==window_size==2048 | ⚠️ math correct; live trim percentile (#37) & live funcs unverified |
| Direct test of the production writer | **none** | **none** (only the distance primitive + median are directly tested) |
| Nearest existing test | `test_beamformer_functions` (per-sample delay-sum parity, not the windowed writer) | `test_circular_diff_to_mean_single` + `test_fast_median_abs` (primitives only); `test_simple_segment*` (indirect, segment-bounds only) |
| #37 applies | no | **yes — confirmed** (`fast_percentile_1d` untested; tested `fast_percentile` is floor-rank & unused) |
| `test_circular_mean` shadow applies | no | corroborating (line 18 shadowed by 151); but even un-shadowed it targets `circular_mean`, not the live `circular_mean_single_fast` |
| Top gap | windowed aggregation `reshape→mean(2)→.T` + f16 untested | live trimmed circular mean/stddev + ceil-rank percentile untested |
| Highest-value proposed test | `test_windowed_beamformer_aggregation` (per-window argmax == injected angle bin) | `test_circular_mean_single_fast_wraparound` (vs numpy circular mean within 1e-5 on ±π-spanning inputs) |
