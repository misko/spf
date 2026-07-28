# Per-frequency additive gain LUT per radio

## Decision summary

The best parsimonious model for the measured calibration grid is a separate
additive RX1/RX2 gain lookup table for every physical radio and every measured
frequency:

```text
predicted_phase(radio, frequency, gain_rx1, gain_rx2)
    = frequency_baseline(radio, frequency)
    + rx1_gain_effect(radio, frequency, gain_rx1)
    + rx2_gain_effect(radio, frequency, gain_rx2)
```

All additions and residuals are circular. The phase convention is RX1 minus
RX2.

Across four radios and 40,510 quality-valid observations, leave-one-epoch-out
performance was:

| MAE | RMSE | P95 | Maximum | Coverage |
|---:|---:|---:|---:|---:|
| **0.903°** | 1.348° | 3.069° | 9.005° | 100% |

The full RX1-by-RX2 cell LUT was slightly worse at 0.949° MAE while using
almost nine times as many parameters. The data therefore do not justify a
separate interaction value for every ordered gain pair.

## Formulation

For physical radio $r$, measured frequency $f$, RX1 gain $g_1$, and RX2
gain $g_2$, the model is:

```math
\widehat{\phi}_r(f,g_1,g_2)
=
\operatorname{wrap}\left(
C_r(f) + A_r(f,g_1) + B_r(f,g_2)
\right)
```

where:

- $\widehat{\phi}$ is the predicted systematic RX1-minus-RX2 phase;
- $C_r(f)$ is the radio- and frequency-specific baseline;
- $A_r(f,g_1)$ is the RX1 gain contribution;
- $B_r(f,g_2)$ is the RX2 gain contribution; and
- `wrap` maps phase onto the circular principal interval.

The calibration uses 26 dB as the reference gain:

```math
A_r(f,26)=0,\qquad B_r(f,26)=0.
```

Consequently:

```math
C_r(f)=\widehat{\phi}_r(f,26,26).
```

For example:

```text
predicted_phase(r, 2.412 GHz, 41 dB, 15 dB)
    = wrap(
        C[r, 2.412 GHz]
        + A[r, 2.412 GHz, 41 dB]
        + B[r, 2.412 GHz, 15 dB]
      )
```

A measured phase is corrected by subtracting the predicted calibration bias:

```math
\phi_{\mathrm{corrected}}
=
\operatorname{wrap}\left(
\phi_{\mathrm{measured}}-\widehat{\phi}_r(f,g_1,g_2)
\right).
```

## Meaning of “additive”

At a fixed radio and frequency, the model assumes that the effect of changing
RX1 gain does not depend materially on the selected RX2 gain, and vice versa:

```math
\widehat{\phi}=C+A(g_1)+B(g_2).
```

It deliberately omits an interaction term $I(f,g_1,g_2)$. The dense
Cartesian survey tested this assumption rather than merely imposing it: a
full cell LUT containing the interaction freedom did not improve held-out
prediction.

“Per-frequency” is equally important. Each frequency has independent
baseline and gain-effect parameters. This model does not interpolate to an
unmeasured frequency, and its 0.903° result must not be interpreted as
unseen-frequency performance.

“Per-radio” means the complete parameter set is fitted independently for each
physical Pluto. There is no parameter sharing between boards in this model.

## Parameter count

The survey contains 17 gain settings and 12 frequencies. At each frequency,
one radio has:

```text
1 baseline
+ 16 non-reference RX1 gain effects
+ 16 non-reference RX2 gain effects
= 33 parameters
```

The reference-gain effects are fixed to zero, so they are not fitted
parameters. Therefore:

| Scope | Parameters |
|---|---:|
| One frequency on one radio | 33 |
| All 12 frequencies on one radio | 396 |
| All 12 frequencies on four radios | 1,584 |

For comparison, a full cell LUT uses $17\times17=289$ values per frequency,
or 3,468 per radio and 13,872 across four radios.

## How the parameters are fitted

The committed implementation builds a sparse design matrix with one active
baseline column, at most one RX1 gain column, and at most one RX2 gain column
for every observation. It solves least squares, repeatedly moves every target
onto the nearest $2\pi$ branch around the current prediction, and refits.
The final prediction and errors are wrapped circularly.

Only quality-valid observations enter the fit. The reported accuracy uses
leave-one-epoch-out evaluation: two randomized sweep repetitions train the
table and the third repetition is unseen.

The direct, cross-sweep interpretation of the parameters is:

```math
\begin{aligned}
C_r(f) &\approx \operatorname{circmean}\phi_r(f,26,26),\\
A_r(f,g) &\approx
  \operatorname{wrap}\left(
  \operatorname{circmean}\phi_r(f,g,26)-C_r(f)
  \right),\\
B_r(f,g) &\approx
  \operatorname{wrap}\left(
  \operatorname{circmean}\phi_r(f,26,g)-C_r(f)
  \right).
\end{aligned}
```

The current dense fit is statistically stronger because all $17\times17$
gain pairs help estimate the additive terms. The equations above show why a
much smaller cross-shaped survey is sufficient to identify the same model.

## Accuracy by physical radio

| Pluto serial | MAE | RMSE | P95 | Maximum |
|---|---:|---:|---:|---:|
| `104000707f0700120f001a0095f2dbee49` | 0.830° | 1.265° | 2.937° | 8.410° |
| `104000f6ad020002fdff3a00bba2f096a1` | 0.917° | 1.392° | 3.166° | 8.389° |
| `104000b299050013f4ff0700255e35222f` | 0.901° | 1.310° | 2.923° | 9.005° |
| `104473b80a16000de6ff2000f8a6beca79` | 0.965° | 1.422° | 3.224° | 8.080° |

The narrow 0.830–0.965° MAE range says the formulation works similarly well
on all four boards after each board receives its own parameters.

## How much the parameters vary between radios

The following comparison uses the complete-data fit for each radio. Because
all gain effects are constrained to zero at the same 26 dB reference, their
between-radio differences are directly comparable.

For every parameter coordinate, all six pairs of the four radios were
compared using absolute circular differences:

| Parameter family | Pairwise mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Frequency baseline $C_r(f)$ | **18.525°** | 9.041° | 71.679° | 91.214° |
| RX1 gain effect $A_r(f,g)$ | 2.499° | 0.746° | 11.402° | 19.381° |
| RX2 gain effect $B_r(f,g)$ | 2.777° | 0.827° | 12.379° | 21.802° |
| Complete predicted gain surface | 18.537° | 9.106° | 70.998° | 98.771° |

The main board-to-board difference is therefore the frequency baseline, not
the normalized shape of the gain response. The typical gain-effect
coefficient differs by less than one degree at the median. Larger gain-effect
differences are concentrated near some high-band gain-table transitions.

### Pairwise summary

Serials are shortened to their final eight characters for readability.

| Radio pair | Baseline MAE | RX1-effect MAE | RX2-effect MAE | Full-surface MAE | Full-surface P95 |
|---|---:|---:|---:|---:|---:|
| `f2dbee49` / `a2f096a1` | 9.354° | 1.129° | 1.493° | 10.619° | 32.923° |
| `f2dbee49` / `5e35222f` | 16.478° | 1.092° | 2.452° | 16.472° | 43.843° |
| `f2dbee49` / `a6beca79` | 19.521° | 4.436° | 2.749° | 19.105° | 47.597° |
| `a2f096a1` / `5e35222f` | 9.860° | 1.028° | 3.090° | 9.005° | 29.928° |
| `a2f096a1` / `a6beca79` | 23.993° | 3.780° | 3.609° | 24.454° | 77.325° |
| `5e35222f` / `a6beca79` | 31.944° | 3.532° | 3.270° | 31.569° | 84.095° |

### Frequency dependence of the variation

The baseline column reports the maximum separation between any two radios at
that frequency. The gain columns report the mean pairwise parameter
difference across the 16 non-reference gain entries.

| Frequency | Maximum baseline separation | RX1-effect MAE | RX2-effect MAE | Full-surface MAE |
|---:|---:|---:|---:|---:|
| 868 MHz | 4.703° | 0.916° | 1.791° | 3.313° |
| 915 MHz | 8.164° | 1.212° | 2.568° | 4.840° |
| 1,280 MHz | 8.132° | 0.623° | 1.291° | 4.953° |
| 1,320 MHz | 15.356° | 1.147° | 2.175° | 7.953° |
| 2,412 MHz | 16.545° | 1.694° | 1.068° | 9.800° |
| 2,467 MHz | 19.809° | 0.705° | 1.366° | 11.793° |
| 3,990 MHz | 9.198° | 3.055° | 4.110° | 7.739° |
| 4,010 MHz | 9.337° | 4.185° | 5.217° | 7.612° |
| 5,766 MHz | 66.572° | 3.442° | 2.057° | 33.292° |
| 5,804 MHz | 77.920° | 4.107° | 2.654° | 39.444° |
| 5,838 MHz | 84.063° | 4.451° | 3.974° | 43.379° |
| 5,866 MHz | 91.214° | 4.456° | 5.053° | 48.326° |

This explains why a universal gain-shape LUT plus one target-radio baseline
at the exact operating frequency works much better than one board-wide
constant. It also explains why an anchor measured in one band must not be
silently reused in another band.

These are descriptive differences between fitted coefficients, not parameter
uncertainty intervals. The two independent repeat surveys changed common
cell means by only 0.532° and 0.705° MAE on their respective radios, which is
far smaller than the cross-radio baseline variation.

## Fitting a radio-specific model with much less testing

There are two useful meanings of “fit per radio,” with very different costs.

### Option A: fit the complete radio-specific additive LUT

At each required frequency, collect only the union of:

1. all 17 RX1 gains with RX2 fixed at 26 dB; and
2. all 17 RX2 gains with RX1 fixed at 26 dB.

The shared 26/26 cell is collected once, giving 33 unique cells rather than
289. Randomize their order within each epoch and retain three repetitions.

| Survey | Stored frames per frequency | Stored frames over 12 frequencies | Reduction from dense |
|---|---:|---:|---:|
| Dense 17×17 Cartesian, three repeats | 867 | 10,404 | baseline |
| Additive cross, three repeats | **99** | **1,188** | **8.76× fewer** |

This design identifies all 33 parameters per frequency. It should be the
default when a genuinely board-specific gain LUT is required, but its
prospective error has not yet been measured. It uses less redundant evidence
than the dense survey and cannot independently discover arbitrary RX1-by-RX2
interactions.

For an inexpensive interaction check, add a small predeclared set of
off-cross sentinel cells, for example:

```text
(0,0), (16,16), (41,41), (62,62), (16,41), (41,16) dB
```

With three repetitions, the cross plus six sentinels uses 117 stored frames
per frequency or 1,404 over all 12 frequencies, still 7.41× fewer than dense.
Score the sentinels without fitting them. If their circular residuals exceed a
predeclared acceptance limit, do not trust the additive table for that radio;
run a denser survey or store an interaction correction.

### Option B: reuse the universal gain shape and fit only a radio baseline

The four-radio results show that baseline parameters vary much more than
reference-normalized gain effects. The lowest-cost production candidate is
therefore:

1. ship the universal per-frequency additive gain LUT;
2. tune the target radio to the exact operating frequency;
3. set RX1=RX2=26 dB;
4. inject the common calibration tone;
5. collect three frames; and
6. store their circular-mean residual as one radio/frequency baseline value.

The adapted model is:

```math
\widehat{\phi}_r(f,g_1,g_2)
=
\operatorname{wrap}\left(
C_U(f)+A_U(f,g_1)+B_U(f,g_2)+\Delta_r(f)
\right),
```

where $\Delta_r(f)$ is the one measured target-radio value.

| Strategy | Frames at one operating frequency | Frames for all 12 frequencies | Four-radio leave-one-radio-out MAE |
|---|---:|---:|---:|
| Universal LUT, no target calibration | 0 | 0 | 14.171° |
| Universal LUT + one 26/26 value | **3** | **36** | **3.385°** |
| Universal LUT + fixed second gain value | 6 | 72 | 3.419° |

The fixed second gain anchor did not improve aggregate accuracy, so it is not
recommended. An exploratory 41/-1 dB second anchor reached 3.019° MAE, but
that pair was selected and scored on the same four radios and requires a
fifth unseen-radio validation before deployment.

### Recommended staged procedure

1. For normal field onboarding, start with Option B at every frequency the
   rover will actually use. Fail closed at frequencies without an anchor.
2. Save the radio serial, firmware fingerprint, frequency, gain reference,
   anchor frames, circular mean, circular dispersion, and LUT version in V7.
3. Validate the corrected phase on several predeclared gain-pair sentinels.
4. If the required accuracy is approximately 3–5°, stop.
5. If the target is approximately 1°, run Option A for that radio and only
   the required operating frequencies.
6. Run the dense Cartesian survey only when the cross-sweep sentinels reject
   additivity, when characterizing new hardware, or when updating the
   universal LUT.

The frame counts above describe stored calibration measurements. Exact wall
time will not scale perfectly with frame count because frequency settling,
gain writes, discarded post-change buffers, retries, and Python/USB overhead
also contribute. Benchmark the reduced runner directly before assigning a
field-test ETA.

## Limitations

- The gains are the tested dB gain settings from the calibration grid.
- The correction is supported only at measured frequencies and gains.
- Phase is circular; ordinary arithmetic means and unwrapped residuals are
  inappropriate at the $-180°/180°$ boundary.
- The model corrects systematic phase bias. It cannot make a buffer
  phase-safe if gain changes occurred during that buffer.
- A board-specific fit should be invalidated or rechecked after hardware-path,
  firmware, gain-table, RF-band, or calibration-fixture changes.
- The 0.903° result measures held-out repetitions on known cells. The
  proposed 33-cell cross sweep needs its own prospective dense-grid
  validation before replacing the current characterization procedure.

## Sources and reproducibility

- [Four-radio executive report](README.md)
- [Complete 13-model comparison](MODEL_MATRIX_REPORT.md)
- [Low-cost transfer analysis](LOW_COST_CALIBRATION_REPORT.md)
- [Machine-readable fitted coefficients and metrics](model_matrix.json)
- [Machine-readable low-cost calibration results](low_cost_calibration.json)
- [Model implementation](../../model_matrix.py)

The machine-readable model matrix records the exact dataset paths, radio
serials, input SHA-256 hashes, gain grid, frequency grid, fitted coefficients,
and held-out metrics used by this report.
