# Theoretical versus empirical phase/bearing likelihoods — all-key audit

**Date:** 2026-08-25
**Scope:** all 48 keys and all 288 stored variants in
`empirical_dists/full_20260809_v1.pkl`, plus the 44-key `full.pkl` baseline.
**Method:** read-only analysis of both PKLs and all 2,445 provenance-pinned source
datasets. The existing PKLs and source datasets were not modified.

## Executive answer

Yes: there is a clean theoretical representation. For a two-element array with
spacing ratio \(\rho=d/\lambda\), the repository's actual coordinate convention is

\[
\boxed{\mu_\rho(\theta)=
\operatorname{wrap}\left[-2\pi\rho\sin\theta\right]}.
\]

Adding circular phase noise gives a likelihood \(P(\phi\mid\theta)\). With a uniform
bearing prior, Bayes' rule converts it to the same orientation as the pickle,
\(P(\theta\mid\phi)\). This model matches the major empirical ridges, their wrapping,
and their required front/back and spatial aliases. It is not, by itself, a complete
model of the tables.

The strongest conclusions are:

1. **The theoretical sign and geometry are correct.** The negative sign above beats the
   opposite sign for **48/48 keys**. On the production `r/sym` tables, nominal geometry
   plus one fitted circular-noise width has mean row TV **0.247** and correlation
   **0.727**. The opposite sign is essentially a negative control, not a plausible
   alternative.
2. **Simple calibration explains a substantial fraction of the remaining structure.**
   Fitting effective spacing gain \(g\), mount shift \(\delta\), constant phase offset
   \(c\), and phase-noise width lowers production-table TV to **0.175** and raises
   correlation to **0.843**. It improves 47/48 production keys, despite being fitted to
   the nonsymmetric table rather than directly to production `sym`.
3. **The empirical tables are much broader and more corpus-specific than ideal array
   theory.** The pooled nonsymmetric comparison improves from uniform TV **0.593**, to
   nominal theory **0.411**, to calibrated theory **0.275**. The residual is structured,
   not just white phase noise.
4. **Multimodality is expected.** Twenty-seven of 48 keys have \(d/\lambda>0.5\) and are
   spatially aliased. Depending on spacing and phase, the ideal inverse contains 2–8
   bearing solutions. Multiple ridges above \(0.5\) are not evidence of a broken fit.
5. **Sparse support and capture/device state explain the worst keys better than wavelength
   alone.** `PLUTO_0.56296` has only one source dataset and is worst; `PLUTO_0.90397` has
   six datasets and only 36.9% valid phase yield. Source count has Spearman correlation
   **−0.595** with production calibrated TV.
6. **The producer has two confirmed coordinate/discretization defects.** Auto-inferred
   histogram edges are discarded, although consumers assume `[-pi,pi]`; this is a large
   effect for `PLUTO_0.91964/r0` but not a global explanation. Separately, the 65-bin
   integer symmetry fold changes an already symmetric ideal model by TV **0.032–0.200**
   over the observed spacing range.
7. **The pickle is a posterior, not a sensor likelihood.** It stores
   \(P(\theta\mid\phi,\text{detected},\text{corpus})\), while particle filters multiply
   it as though it were \(P(\phi\mid\theta)\). Corpus bearing and detection priors are
   therefore injected into filter updates.

The practical reading is: **the wavelength/spacing theory is decisively present in the
data, but the shipped tables combine it with calibration offsets, capture priors,
selection effects, radio/corpus differences, multipath, sparse support, and producer
coordinate transforms.** A calibrated theoretical likelihood is a credible fallback or
regularizer, but these in-sample fits are not yet a replacement calibration.

![Fleet-wide fit summary](figures/2026_08_25_empirical_dist_analysis/fleet_fit_summary.png)

## 1. Inputs and completeness

| Input | Keys | Role | SHA-256 |
|---|---:|---|---|
| `empirical_dists/full_20260809_v1.pkl` | 48 | current table | `96b604799022d5139f5a1c7568cad0840dbcb90c8c5e3e4f4d03350108c01f84` |
| `empirical_dists/full.pkl` | 44 | previous baseline | `14bfe8026cdf266543679c2994571c3c687fc5887cef488ad65700971a26ae72` |

The current artifact contains 36 Pluto keys and 12 bladeRF keys, covering 45 unique
spacing ratios. It stores six variants per key:

```
r0/nosym  r0/sym  r1/nosym  r1/sym  r/nosym  r/sym
```

The report's primary physical target is pooled `r/nosym`; the production default is
pooled `r/sym`. The supplementary variant CSV evaluates all 288 matrices with the same
per-key pooled physical fit so radio disagreement remains visible rather than being
re-fitted away.

The embedded provenance requests 2,499 source stores and records 2,445 loaded. A second
read-only source pass successfully reopened all 2,445 recorded contributors and matched
all 48 per-key source counts. The existing rebuild report's statement that 43 sources lack
segmentation is inaccurate: embedded provenance records **53 missing segmentations plus
one spacing assertion failure**.

## 2. The theoretical representation

### 2.1 Forward likelihood

For measured phase \(\phi\), bearing \(\theta\), and circular concentration \(\kappa\),
the nominal model is

\[
P(\phi\mid\theta,\rho,\kappa)=
\frac{\exp\{\kappa\cos[\phi-mu_\rho(\theta)]\}}
     {2\pi I_0(\kappa)},\qquad
\mu_\rho(\theta)=\operatorname{wrap}[-2\pi\rho\sin\theta].
\]

The negative sign is implemented in
[`spf_dataset.py`](../spf/dataset/spf_dataset.py#L1525). Raw phase is receiver channel
0 minus channel 1. The positive sign written in the previous rebuild report is a plotting
interpretation error: its images used Matplotlib's default upper origin, visually
reversing phase.

The calibrated descriptive model is

\[
\mu(\theta)=\operatorname{wrap}
\left[c-2\pi g\rho\sin(\theta-\delta)\right],
\]

where \(g\) is effective spacing/phase gain, \(\delta\) is an angular shift, and \(c\) is
a phase offset. This is the same low-dimensional systematic model already used in
[`dataset_quality_scan.py`](../spf/scripts/dataset_quality_scan.py#L9).

### 2.2 Converting likelihood to the stored posterior

The current builder stores matrix rows as

\[
Q_{j,i}=P(\theta\in i\mid\phi\in j),
\]

by normalizing the joint histogram over theta and transposing it in
[`create_empirical_p_dist.py`](../spf/scripts/create_empirical_p_dist.py#L123). A
theoretical matrix requires a bearing/detection prior \(\pi_i\):

\[
Q^{\mathrm{theory}}_{j,i}=
\frac{\pi_i P(\phi\in j\mid\theta\in i)}
     {\sum_\ell \pi_\ell P(\phi\in j\mid\theta\in \ell)}.
\]

The main comparison uses a uniform \(\pi\), deliberately testing geometry rather than
reproducing the corpus trajectory. The raw-source audit separately tests the actual valid
frame prior.

### 2.3 Aliasing is part of the expected answer

![Theoretical geometry and alias progression](figures/2026_08_25_empirical_dist_analysis/theoretical_geometry.png)

| Spacing class | Current keys | Feasible spatial orders | Maximum bearing modes |
|---|---:|---:|---:|
| \(\rho\le0.5\) | 21 | 1 | 2 front/back |
| \(0.5<\rho<1\) | 22 | 1–2 | 2–4 |
| \(1\le\rho<1.5\) | 4 | 2–3 | 4–6 |
| \(\rho=1.54880\) | 1 | 3–4 | 6–8 |

The inverse branches follow from

\[
s_k=\frac{c-\phi-2\pi k}{2\pi g\rho},\quad |s_k|\le1,
\]

followed by the two front/back solutions of \(\sin(\theta-\delta)=s_k\). Thus a broad or
multimodal posterior at high spacing is physically necessary; the question is whether its
mass lies on the correct branches.

## 3. Comparison method

The report uses 65 fixed phase and bearing bins over `[-pi,pi]`, matching runtime consumer
coordinates. The midpoint matrix approximation changes mean TV by only **0.0006** versus
explicit sub-bin integration in a full all-key check, so it is not responsible for the
observed residuals. The symmetry-only diagnostic uses 9×9 sub-bin integration because
binning error is the quantity being measured there.

For each key:

1. Fit nominal \(\kappa\) to pooled `r/nosym`.
2. Fit \((g,\delta,c,\kappa)\) to the same matrix using deterministic differential
   evolution and mean row Jensen–Shannon divergence.
3. Evaluate total variation (primary), Jensen–Shannon divergence, flattened correlation,
   MAP-bin agreement, and model parameters.
4. Apply the repository's exact joint-histogram symmetry transform to the same fitted
   theory and compare it with `r/sym`; do not fit the production transform separately.
5. Carry the same per-key fit into `r0`, `r1`, `nosym`, and `sym` for all-variant
   diagnostics.

Mean row TV is

\[
\overline{TV}=\frac1{|J|}\sum_{j\in J}
\frac12\sum_i|Q_{j,i}-T_{j,i}|,
\]

where \(J\) is the set of nonempty empirical phase rows. Zero is identical and one is
disjoint. Every phase row is weighted equally because the PKL discards phase-bin counts;
operational phase-frequency weighting requires raw joint counts.

All fits are descriptive and in-sample. At aliased spacings or narrow bearing coverage,
\(g\), \(c\), and \(\delta\) are not uniquely identifiable. A fitted \(g\) is an effective
response, not proof the physical antenna spacing is wrong.

## 4. Aggregate results

| Target and model | Mean TV | Median TV | Mean corr | Interpretation |
|---|---:|---:|---:|---|
| `r/nosym`, uniform | 0.593 | 0.610 | — | no geometric information |
| `r/nosym`, nominal | 0.411 | 0.398 | 0.545 | ideal spacing/sign + fitted noise |
| `r/nosym`, calibrated | **0.275** | **0.249** | **0.751** | fit \(g,\delta,c,\kappa\) |
| production `r/sym`, nominal | 0.247 | 0.266 | 0.727 | nominal passed through repo fold |
| production `r/sym`, calibrated | **0.175** | **0.155** | **0.843** | same physical fit passed through fold |

Calibration reduces mean nonsymmetric TV by 33.1% relative to nominal and production TV
by 29.3%. It improves all 48 nonsymmetric targets and 47/48 production targets. Relative
to a uniform posterior, the calibrated nonsymmetric model has mean TV skill **0.517** and
median skill **0.599**.

The fitted circular phase spread is broad: median **36.9°**, interquartile range
**25.6–49.2°**, and range **21.9–85.3°**. bladeRF has a much tighter fitted median
(25.1°) than Pluto (38.9°), but this comparison is confounded by corpus and environment.

The raw-source residual check independently reaches the same conclusion without fitting a
posterior matrix. For
\(r=\operatorname{wrap}[\phi+2\pi\rho\sin\theta]\), per-key circular σ has mean
**0.880 rad**, median **0.877 rad (50.3°)**, and range **0.444–1.500 rad**. The absolute
r0/r1 circular-bias difference has median **0.355 rad (20.3°)** and maximum
**1.849 rad (106°)**; large cases include `0.56296`, `0.56318`, `0.57606`, and
`0.21367`. This directly supports broad phase noise plus receiver/calibration-state
mixture. The production symmetry fold hides much of that asymmetry by construction.

![Fitted effective parameters](figures/2026_08_25_empirical_dist_analysis/fitted_parameters.png)

Low-spacing groups often need \(g>1\): the largest are Pluto 0.12208 (2.709), 0.13124
(2.327), 0.20114 (2.173), and bladeRF 0.20548 (1.842). Above approximately 0.5, fitted
\(g\) usually clusters near one. This is consistent with known wall-array effective-spacing
systematics, but low-spacing fits are also the least identifiable, so the values should be
treated as hypotheses for raw held-out calibration, not literal tape-measure corrections.

The full 48-row table is in [`ALL_KEYS_TABLE.md`](data/2026_08_25_empirical_dist_analysis/ALL_KEYS_TABLE.md), with the machine-
readable version in [`metrics_all_keys.csv`](data/2026_08_25_empirical_dist_analysis/metrics_all_keys.csv).

## 5. Key-level results

### 5.1 The four new 2026 rover spacings

| Key | Sources | nominal production TV | calibrated production TV | production corr | calibrated nonsym TV | nonsym corr | fitted phase σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PLUTO_0.68181` | 3 | 0.194 | **0.163** | 0.950 | 0.271 | 0.855 | 22.3° |
| `PLUTO_0.83765` | 5 | 0.230 | **0.197** | 0.886 | 0.339 | 0.725 | 33.8° |
| `PLUTO_0.90397` | 6 | 0.260 | **0.260** | 0.725 | 0.443 | 0.493 | 54.2° |
| `PLUTO_0.91557` | 3 | 0.285 | **0.285** | 0.742 | 0.393 | 0.606 | 47.8° |

`0.68181` and `0.83765` visibly follow the theoretical branches. The simple calibration
barely helps `0.90397` and `0.91557`; their mismatch is not primarily a constant phase or
spacing error. `0.90397` is the clearer warning because it also has the poorest valid-frame
yield (36.9%), pronounced r0/r1 imbalance, and two empty rows in `r1/nosym`.

### 5.2 Worst and best production matches

| Key | Sources | calibrated production TV | corr | Main interpretation |
|---|---:|---:|---:|---|
| `PLUTO_0.56296` | 1 | **0.437** | 0.456 | too little support; worst overall |
| `PLUTO_0.91964` | 7 | **0.376** | 0.603 | sparse plus confirmed coordinate truncation |
| `PLUTO_0.91557` | 3 | 0.285 | 0.742 | new, sparse, broad response |
| `PLUTO_0.99302` | 8 | 0.273 | 0.680 | sparse, structured residual |
| `PLUTO_0.57606` | 101 | 0.264 | 0.697 | not sparse; likely pooled state/multipath |
| `PLUTO_0.90397` | 6 | 0.260 | 0.725 | sparse/low-yield RO4 |
| `PLUTO_0.41762` | 31 | **0.107** | 0.950 | best production TV |
| `PLUTO_0.28159` | 150 | **0.109** | 0.964 | best production correlation |

The 101-source `0.57606` exception is important: support is a strong fleet-level predictor,
not a complete explanation. Its table retains large coherent deviations after calibration,
which points toward mixed capture/radio states or structured propagation rather than
sampling noise alone.

![Representative empirical and theoretical heatmaps](figures/2026_08_25_empirical_dist_analysis/representative_heatmaps.png)

The complete visual comparison for every key is an eight-page PDF:
[`all_48_keys_atlas.pdf`](figures/2026_08_25_empirical_dist_analysis/all_48_keys_atlas.pdf). Its first page is also available
as [`all_keys_atlas_page1.png`](figures/2026_08_25_empirical_dist_analysis/all_keys_atlas_page1.png).

A second eight-page atlas covers the actual production `r/sym` target, showing empirical
production, the calibrated theory passed through the as-built symmetry fold, the same
physical theory before that fold, and the residual:
[`all_48_keys_production_sym_atlas.pdf`](figures/2026_08_25_empirical_dist_analysis/all_48_keys_production_sym_atlas.pdf).

## 6. All stored variants

Parameters below are always fitted to pooled `r/nosym` and then transferred. These are
therefore a diagnostic of radio/pooling differences, not six independently optimized fits.

| Variant | nominal mean TV | calibrated mean TV | calibrated median TV | calibrated mean corr |
|---|---:|---:|---:|---:|
| `r/nosym` | 0.411 | **0.275** | 0.249 | 0.751 |
| `r/sym` | 0.247 | **0.175** | 0.155 | 0.843 |
| `r0/nosym` | 0.504 | **0.385** | 0.349 | 0.620 |
| `r0/sym` | 0.307 | **0.257** | 0.238 | 0.705 |
| `r1/nosym` | 0.506 | **0.404** | 0.370 | 0.627 |
| `r1/sym` | 0.316 | **0.252** | 0.241 | 0.785 |

The pooled matrix matches one low-dimensional curve materially better than either radio
does under transferred parameters. Pooling and symmetry smooth away receiver-specific
offsets, coverage holes, and noise, but the pooled `r` histogram also adds r0/r1 matrices
by index even when their inferred coordinate grids differ. It should not be mistaken for a
single clean sensor likelihood.

Every variant/key row is in [`metrics_all_variants.csv`](data/2026_08_25_empirical_dist_analysis/metrics_all_variants.csv).

## 7. Strong negative controls

### 7.1 Phase sign

The repo-sign nominal model wins by Jensen–Shannon divergence for **48/48 keys**. An
independent direct comparison found mean production TV approximately 0.251 for the negative
sign versus 0.473 for the positive sign, and mean correlation 0.729 versus −0.006. This
rules out a global sign flip as the reason for poor individual fits.

### 7.2 Same spacing, different SDR device

Ideal theory depends on \(d/\lambda\), not device. The current table has three exact
Pluto/bladeRF spacing pairs:

| d/λ | production Pluto-vs-bladeRF TV | corr | nonsym TV |
|---:|---:|---:|---:|
| 0.48083 | 0.286 | 0.757 | 0.591 |
| 0.48400 | 0.297 | 0.765 | 0.515 |
| 0.48684 | 0.284 | 0.574 | 0.480 |

![Exact-spacing cross-device controls](figures/2026_08_25_empirical_dist_analysis/cross_device_same_spacing.png)

The theory matrices are identical at each pair, yet the empirical posteriors differ
substantially. This is strong evidence that device/capture/calibration/prior state matters.
It does **not** isolate a causal SDR-device effect because the source corpora and
environments are not controlled.

### 7.3 Nearly identical Pluto spacings

| Pair | empirical production TV | corr | ideal-theory TV |
|---|---:|---:|---:|
| 0.41762 vs 0.41764 | 0.176 | 0.905 | 0.000027 |
| 0.56318 vs 0.56319 | 0.095 | 0.960 | 0.000016 |
| 1.26578 vs 1.26599 | 0.147 | 0.919 | 0.000360 |

Wavelength differences this small cannot explain the empirical differences. These pairs
are especially direct evidence for different underlying capture, calibration, environment,
or selection distributions being pooled into a key that encodes only device class and
spacing ratio.

## 8. Producer-coordinate and symmetry effects

### 8.1 Auto-inferred histogram axes

The builder calls `np.histogram2d(..., bins=65)` without a range and discards the returned
edges. Runtime consumers index both dimensions as fixed `[-pi,pi]`. r0 and r1 infer their
own grids and are then added by integer cell.

The exhaustive raw-source comparison shows this is **confirmed but usually small**:

- For 47/48 keys, switching per-radio nonsymmetric theory from fixed consumer axes to the
  actual producer axes changes TV by less than approximately 0.001.
- `PLUTO_0.91964/r0` is the exception. Its theta range is only
  `[-2.92409, 2.71901]`, displacing the endpoint by 4.32–4.37 consumer bins. Using producer
  axes improves its per-radio theory TV from 0.643 to 0.584; adding the actual valid-frame
  bearing prior improves it to 0.510.
- `PLUTO_0.90397/r1` has the next-largest phase-axis displacement, about 0.232 bins; its
  pooled fixed-grid reconstruction changes production TV by only 0.0307.

So auto-ranging is a real contract defect and a major contributor to `0.91964`, but it does
not explain the fleet-wide mismatch.

### 8.2 Integer symmetry bias

The ideal model already obeys sign and front/back symmetries. The current implementation
folds integer row indices as though 65 histogram bins were 65 endpoint samples. Continuous
symmetry instead splits source-bin intervals across destination bins.

At the largest spacing, the maximum ridge-center displacement is

\[
2\pi\rho\sin(\pi/65)=0.47\text{ rad},
\]

or roughly 4.9 phase bins. In a controlled bin-integrated ideal model with circular phase
σ = 0.2 rad, the repository transform changes the posterior by:

| d/λ | symmetry-transform TV |
|---:|---:|
| 0.12208 | 0.032 |
| 0.48083 | 0.100 |
| 0.68181 | 0.134 |
| 0.90397 | 0.149 |
| 1.25103 | 0.182 |
| 1.54880 | 0.200 |

![Symmetry operator bias](figures/2026_08_25_empirical_dist_analysis/symmetry_operator_bias.png)

This is a construction artifact, not radio physics. It also explains why the lower TV of
`r/sym` cannot be interpreted simply as the hardware becoming more theoretical: both data
and theory have been passed through a strong, approximate regularizer.

## 9. Posterior prior and detection selection

The theoretical sensor model is \(P(\phi\mid\theta)\), but the pickle stores

\[
P(\theta\mid\phi,\text{finite segmented phase},\text{source corpus}).
\]

The finite-phase filter is not random: segmentation only emits phase when it detects
stable, sufficiently strong signal windows. Angle, range, gain, multipath, and platform
therefore affect which frames enter the table. Even before hardware systematics, a
nonuniform valid-frame bearing prior changes the posterior.

Across the raw per-radio nonsymmetric reconstructions, uniform-prior theory has mean TV
**0.5023** and median **0.4938**. Supplying the actual valid-frame bearing prior lowers
those to **0.4908** and **0.4858**, improving 38/48 keys. The mean improvement is 0.0114
and median is 0.0045, but it is material for `0.99302` (0.110), `0.68484` (0.091), and
`0.91964` (0.074).

On the six 2026 rover spacing groups specifically, removing the empirical bearing prior
changes mean row TV by **0.010–0.048** and changes the conditional MAP bin in
**9.2–32.3%** of phase rows; `0.90397` is largest. This is a real secondary global
explanation and a material local one, not the dominant fleet-wide cause. The PKL itself
cannot undo it because it omits raw joint counts and \(P(\phi)\).

This semantic mismatch matters downstream. Particle filters multiply the stored posterior
as a likelihood, effectively updating with

\[
w_{t-1}(\theta)P(\phi\mid\theta)P_{\text{corpus}}(\theta\mid\text{detected}),
\]

which double-counts the corpus prior unless it is uniform.

## 10. Ranked hypotheses for the residual mismatch

| Rank | Hypothesis | Evidence | Confidence / scope |
|---:|---|---|---|
| 1 | Correct geometry plus hardware/capture calibration | repo sign 48/48; four-parameter fit cuts nonsym TV 0.411→0.275; bladeRF phase offsets cluster around 33–55° | **High**, fleet-wide |
| 2 | Sparse and selected valid frames | worst key has n=1; support-vs-production-TV Spearman −0.595; `0.90397` yield 36.9% | **High**, strongest on sparse keys |
| 3 | Mixed device/radio/corpus/environment state omitted from key | exact-spacing device pairs differ TV 0.284–0.297; near-identical Pluto pairs differ far beyond theory | **High**, causal component not isolated |
| 4 | Structured multipath or heteroscedastic phase noise | coherent residual branches remain; `0.57606` is poor despite 101 sources; one von Mises cannot represent multiple paths | **Medium-high** |
| 5 | Corpus-bearing and detection prior | raw rover reweighting changes TV 0.010–0.048 and MAP bins up to 32.3% | **Certain but moderate** on tested rover keys |
| 6 | Auto-range coordinate mismatch | exact reconstruction; large 4.3-bin shift and material improvement for `0.91964/r0` | **Certain, key-specific**, not fleet-wide |
| 7 | Integer symmetry transform | controlled theoretical TV 0.032–0.200 | **Certain** for `sym`; construction, not RF |
| 8 | Aliasing causes multiple modes | 27/48 keys above 0.5, analytic 2–8 bearing solutions | **Certain**, but modes are expected rather than an error |

A global sign error, a simple transpose error, midpoint bin approximation, or universal
auto-range distortion are rejected as primary explanations.

## 11. Old versus current table

![Old versus current table and theory](figures/2026_08_25_empirical_dist_analysis/old_vs_current_theory.png)

Of 44 shared keys, three moved materially:

| Key | old→current production TV | old corr | Movement versus current calibrated theory |
|---|---:|---:|---:|
| `BLADERF2_0.65752` | 0.149 | 0.899 | **away** by TV +0.0178; 41 source segmentations were lost |
| `PLUTO_0.82703` | 0.107 | 0.937 | **toward** by TV −0.0324 after adding 19 rover sources |
| `PLUTO_0.67317` | 0.066 | 0.989 | **toward** by TV −0.0255 after adding 12 rover sources |

The remaining 41 keys are nearly unchanged. Details are in
[`metrics_old_vs_current.csv`](data/2026_08_25_empirical_dist_analysis/metrics_old_vs_current.csv).

## 12. What should change next

1. **Rebuild on explicit shared axes.** Pass `range=[[-pi,pi],[-pi,pi]]`; use identical
   edges for r0/r1; store the edges and validate them at read time.
2. **Store joint counts and both conditionals.** Persist raw `N(theta,phi)`, valid counts,
   `P(phi|theta)`, and `P(theta|phi)`. Particle filters should consume the likelihood,
   not the corpus posterior.
3. **Replace integer symmetry with bin-integrated coordinate symmetry.** Test it against
   the analytic sine response for odd, even, and unequal bin counts.
4. **Condition on more than `(device,d/lambda)`.** At minimum preserve receiver, radio or
   calibration state, phase-correction identity, support, and capture provenance. Consider
   a hierarchical model that partially pools nearby spacings rather than exact five-decimal
   keys.
5. **Fit on raw observations and validate by capture.** Compare nested models: ideal +
   noise; then \(g,c,\delta\); then uniform contamination; then multipath components. Split
   and bootstrap by physical capture, never by adjacent frame.
6. **Use theory as a regularizer/fallback.** For sparse keys, shrink empirical counts toward
   calibrated theory and require a minimum effective support. `PLUTO_0.56296` should not be
   treated as equally reliable to 100-source tables.
7. **Match phase correction and table identity.** Record table SHA/provenance in every
   result and cache key; do not apply corrected phase to an uncorrected table.

## 13. Limitations

- The principal table fits are in-sample descriptions, so their scores are optimistic and
  have no capture-bootstrap confidence interval.
- Matrix TV weights every nonempty phase row equally. The pickle discards phase counts, so
  an operational frequency-weighted score cannot be reproduced from it alone.
- `r/nosym` is pooled from r0/r1 count matrices built on separately inferred coordinates;
  fitted physical parameters should ultimately come from per-radio raw observations.
- \(g\), \(c\), and \(\delta\) are non-identifiable under narrow coverage and increasingly
  ambiguous under phase wrapping. They describe the table but do not uniquely diagnose a
  component.
- The von Mises model is unimodal in phase at a fixed bearing. It cannot represent coherent
  reflections, time-varying calibration, angle-dependent noise, or gain-state mixtures.
- Exact same-spacing device comparisons are strong negative controls but not controlled
  device experiments.
- The raw-source audit verifies all current provenance-loaded records, but the 54 rejected
  build inputs cannot contribute until their segmentation/spacing issues are resolved.

## 14. Reproduction and outputs

Run from the repository root:

```bash
uv run python \
  reports/data/2026_08_25_empirical_dist_analysis/analysis/compare_theory.py
```

The script uses deterministic per-key optimizer seeds and writes only within the canonical
analysis directory linked above.

| Output | Contents |
|---|---|
| [`ALL_KEYS_TABLE.md`](data/2026_08_25_empirical_dist_analysis/ALL_KEYS_TABLE.md) | full 48-row human-readable table |
| [`metrics_all_keys.csv`](data/2026_08_25_empirical_dist_analysis/metrics_all_keys.csv) | all key metrics and fitted parameters |
| [`metrics_all_variants.csv`](data/2026_08_25_empirical_dist_analysis/metrics_all_variants.csv) | 288 stored variant comparisons |
| [`metrics_cross_device.csv`](data/2026_08_25_empirical_dist_analysis/metrics_cross_device.csv) | exact-spacing device controls |
| [`metrics_old_vs_current.csv`](data/2026_08_25_empirical_dist_analysis/metrics_old_vs_current.csv) | 44 shared-key baseline analysis |
| [`analysis_metadata.json`](data/2026_08_25_empirical_dist_analysis/analysis_metadata.json) | input hashes, model, bounds, metric definitions |
| [`all_48_keys_atlas.pdf`](figures/2026_08_25_empirical_dist_analysis/all_48_keys_atlas.pdf) | every empirical/calibrated/residual heatmap |
| [`all_48_keys_production_sym_atlas.pdf`](figures/2026_08_25_empirical_dist_analysis/all_48_keys_production_sym_atlas.pdf) | every production/as-built/physical/residual heatmap |
| [`compare_theory.py`](data/2026_08_25_empirical_dist_analysis/analysis/compare_theory.py) | reproducible analysis and figure generator |

No empirical PKL, source dataset, precompute cache, or existing report was written or
modified by this analysis.
