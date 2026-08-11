# E-GSC6 results — the equal-gain diagonal, measured

**Run 2026-08-11**, both radios, RC17, 8,784 frames, **100.00% quality-valid**.
Full report and data: [`reports/equal_gain_diagonal_20260811_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/equal_gain_diagonal_20260811_v1/REPORT.md)

## Answer

**Separability holds. The tandem premise is harness-health dependent, and tandem does not
null the differential phase.**

| | R18 (untouched control) | R17 (connector-damaged) |
|---|---:|---:|
| `D(g,g)` MAE — the tandem quantity | **1.52°** | 8.79° |
| `C(g,g)` MAE — the falsifier | **0.705°** | 0.751° |

`C(g,g)` is ~0.5° per cell on **both** units (n=480 each) with no LNA-state structure, at the
scale of the 0.355–0.368° noise floor. **There is no material interaction term** — the
separable per-arm model predicts the diagonal to about half a degree.

## The measured acceptance threshold (from R18)

This is what the experiment was built to produce — measurement in place of extrapolation:

| Band | `D(g,g)` | vs 6.65° baseline | plan projected |
|---|---:|---:|---:|
| low | 0.925° | **≥7.2×** (bound) | 6.0× |
| middle | 0.775° | **8.6×** | 5.3× |
| high | 2.846° | **2.3×** | 3.4× |

Better than projected below 4 GHz, **worse above it** — the band that matters most.

**The low-band row is a bound, not a value.** The anchor-drift gate passed, but it showed
R18's low-band `D(g,g)` (0.925°) sits *below* its own median anchor drift (1.08°), so that
band is not resolved above the run's anchor stability. Middle and high are resolved with
5.6× and 6.2× margin. Combined with the independent low-band tee-coupling confound (~1°),
read low-band as "at least 7.2×".

## Three things to carry into the firmware plan

1. **`D(g,g)` is not zero and is LNA-state structured**, so tandem leaves `D(g,g) = A(g)`.
   The residual model stays **required**, not a fallback. But it is predictable: `C ≈ 0.5°`,
   so a per-arm term indexed by LNA state suffices.
2. **Harness health dominates the benefit.** R17 is 5–7× worse than R18 in every band while
   their `C` values match; its high band is 0.3×, worse than doing nothing.
3. **H2 holds, restated.** `|D(g,g)|` is flat within an audited LNA state and steps between
   states. The original "transition cells vs frozen-word controls" phrasing was untestable —
   a frozen-word gain sits inside whichever plateau its neighbours occupy. The index-clamp
   recommendation and Campaign C's index selection stand.

## Why the decision rule was re-based during the plan review

Graded on **published** `A`, R17's high band (20.4°, 3× the 6.65° baseline) would have
declared **H0, "the interaction term dominates"** — which this same run refutes, since `C`
there is 0.923°. Grading on the **same-session** residual gives the correct reading. That
change was made before any data existed.
