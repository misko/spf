# E-GSC6 — the equal-gain diagonal, measured

**Run:** 2026-08-11 · **Radios:** R17 `104000bac495…`, R18 `1040007c4a94…`
**Capture:** 8,784 frames (4,392/radio), **100.00% quality-valid, zero rejections**
21 gains → 41 training-cross pairs + **20 held-out diagonal cells**, 24 LOs (8/band), 3 epochs
**Config:** `configs/e_gsc6_equal_gain_diagonal.yaml` (signature `a5ce0356b5432ad3`)
**Data:** `diagonal.json` · **Design:** `experiments/e_gsc6_equal_gain_diagonal/`

## Firmware provenance, and the one caveat it imposes

RC17, RAM-loaded through the `automate` flow so `firmware_verified` is a **genuine
attestation** rather than an assertion: `release_tag …-gain-series-v4-rc17`,
`device_fw …-rc16-7-g1f3fe`, `boot_mode ram`, `image_sha256 88a606f1…`. RC17's provenance
was checked independently against the release's own `SHA256SUMS` before use — DFU
`88a606f1…`, rootfs `be9ac420…`, XSA `a8b0dc57…`, all matching its hardware-gate report.

**Published `A` was measured on `rc12-9-g867e1`, so every comparison against published `A`
below is cross-firmware.** The **same-session `A`** — which §7's decision rule actually
grades on — is unaffected, because it comes from this run's own training cross.

## What the two emitted residuals are

The design rests on a property of the committed analysis, so no new code was written. At a
diagonal held-out cell the per-frequency fit anchors `rx*_effect[ref] = 0` and folds the
anchor into `intercept`, so:

| Emitted field | At `g1 == g2` reduces to | Is |
|---|---|---|
| `held_out_shared_gain_curve_metrics` | `phase − intercept` | **`D(g,g)`** — the tandem quantity |
| `held_out_independent_rx_metrics` | `phase − intercept − rx1_effect[g] − rx2_effect[g]` | **`C(g,g)`** — the interaction term, `D − A_session` |

## Headline: separability holds; the tandem premise is harness-health dependent

| | R18 (untouched control) | R17 (documented connector damage) |
|---|---:|---:|
| `D(g,g)` overall MAE | **1.52°** | 8.79° |
| `C(g,g)` overall MAE | **0.705°** | 0.751° |

**`C(g,g)` — the falsifier — is small on both units and nearly identical between them**
(per-cell mean 0.504° on R18, 0.568° on R17, n = 480 each), with no LNA-state structure. The
separable per-arm model predicts the diagonal to about half a degree, i.e. at the scale of
the 0.355–0.368° measured frame-level noise floor. **Separability holds. There is no
material interaction term.**

`D(g,g)` differs 5–7× between the two units while `C` does not. R17 is the unit whose
high-band mean `|A|` was previously driven 3.49° → 29.41° by connector work and never
recovered; R18 is the untouched control. **Absolute numbers should be taken from R18.**

## Per band

| Band | radio | `D(g,g)` MAE | p95 | `C(g,g)` MAE | published `A` |
|---|---|---:|---:|---:|---:|
| low ≤1300 | R18 | **0.925°** | 3.91° | 0.756° | 0.73° |
| middle 1301–4000 | R18 | **0.775°** | 2.97° | 0.428° | 1.24° |
| high >4000 | R18 | **2.846°** | 22.75° | 0.932° | 3.72° |
| low | R17 | 3.618° | 56.53° | 0.756° | 0.73° |
| middle | R17 | 2.346° | 15.05° | 0.575° | 1.24° |
| high | R17 | 20.419° | 94.40° | 0.923° | 3.72° |

On **R18, H1 is SUPPORTED**: `|D(g,g)|` is comparable to or **below** published `A` in every
band, and far below the 6.65° anchored unequal-gain baseline. On R17 it is 2–5.5× published
`A` and three times the baseline in the high band.

## The measured acceptance threshold — what this experiment was built to produce

From R18, replacing the plan's extrapolated projection:

| Band | measured `D(g,g)` | improvement vs 6.65° | plan projected |
|---|---:|---:|---:|
| low | 0.925° | **≥7.2×** (bound — see limitations) | 6.0× |
| middle | 0.775° | **8.6×** | 5.3× |
| high | 2.846° | **2.3×** | 3.4× |

Better than projected below 4 GHz, **worse than projected above it** — the band where the
threshold matters most. For contrast, R17's high band is 0.3×, i.e. *worse* than doing
nothing.

## What tandem AGC will not do

`D(g,g)` is **not zero** on either unit, and it is **LNA-state structured**. So holding both
arms at one common index does not remove the gain-dependent differential phase — it leaves
`D(g,g) = A(g)`. **The residual model stays required rather than becoming a fallback.**

The good news is that `A(g)` is predictable: since `C(g,g) ≈ 0.5°`, a per-arm term indexed
by LNA state captures the diagonal to about the noise floor.

## H2 — restated, because the original phrasing was untestable

H2 asked whether `|D(g,g)|` is larger at LNA-transition indices than at frozen-word
controls. **As written that comparison is not meaningful**, and the reason is the finding:

`|D(g,g)|` is **flat within an audited LNA state and steps between states**, with boundaries
exactly on the audited transitions. A "frozen-word control" gain therefore sits inside
whichever plateau its neighbours occupy, so comparing it to a transition cell compares
plateau *membership*, not RF-word behaviour. R17's high band makes the trap obvious: the
45 dB control lands in the 48° plateau, which would have scored the control as *noisier*
than the transitions.

Grouped correctly, by LNA word:

| Band | radio | LNA 0 | LNA 1 | LNA 2 | LNA 3 | between-state spread |
|---|---|---:|---:|---:|---:|---:|
| low | R18 | 0.42° | 1.18° | 0.94° | 0.93° | 0.76° |
| middle | R18 | 0.34° | 0.75° | 0.77° | 1.16° | 0.82° |
| high | R18 | 6.80° | 2.72° | 0.60° | 2.54° | 6.20° |
| low | R17 | 0.24° | 2.61° | 3.57° | 18.77° | 18.53° |
| middle | R17 | 0.49° | 1.56° | 1.93° | 6.96° | 6.46° |
| high | R17 | 14.12° | 3.96° | 0.48° | 48.56° | 48.08° |

**H2's underlying claim — that the diagonal phase tracks the audited RF word — is
SUPPORTED on both radios**, and the frozen-word controls behave like their state-mates
rather than standing out, which is the correct control behaviour. The index-clamp
recommendation and Campaign C's index selection stand.

`C(g,g)` shows **no** such structure on either radio, which is the same result stated
differently: the LNA-state dependence of the diagonal is entirely absorbed by the separable
per-arm terms.

## Decision-rule outcome (§7, judged per band, never pooled)

Grading on `|C(g,g)|` against the same-session `A`, as §7 specifies:

- **`C` is within a factor of ~1.5 of the 0.355–0.368° noise floor in every band on both
  radios** → the first row fires: *"separability holds; H1 → adopt the measured values as
  the campaign threshold."*
- Had the rule been graded on **published** `A` instead, R17's high band (20.4°, three times
  the 6.65° baseline) would have declared **H0, "the interaction term dominates"** — which
  the same run shows to be false, since `C` there is 0.923°. Re-basing §7 on the
  same-session residual during the plan review is what prevented that misreading.

## Limitations

- **Cross-firmware** against published `A` (RC17 vs `rc12-9-g867e1`). Same-session
  comparisons are unaffected.
- **R17's harness is known-disturbed**; its absolute numbers characterise that assembly, not
  the part. Two units is not a distribution.
- **The bare-tee harness** bounds cross-arm coupling at ≤1.25 dB (E-HCP1, frequency-flat),
  and in the **low band** that bound (~1°) and `D(g,g)` (0.925°) are the same size — so the
  low-band number should not be over-read. The high band, where the threshold matters, is
  comfortably clear of it.
- **Anchor drift: gate PASSED** (`anchor_drift.json`). The three epochs agree at nearly
  every LO with no step; worst single drift 3.99° (R17, 4000 MHz) and 3.12° (R18, 550 MHz),
  medians 0.52° and 0.43°. Crucially it **cannot explain R17's high band** — 0.63° median
  drift against 20.4° of `D(g,g)`, an order of magnitude short.
- **But the low-band threshold is a bound, not a value.** R18's low-band `D(g,g)` of 0.925°
  sits *below* its own median anchor drift of 1.08°, so that band is not resolved above the
  run's own anchor stability. Middle and high are resolved with 5.6× and 6.2× margin and
  stand as values. Combined with the independent low-band tee-coupling confound (~1°), there
  are **two reasons the low-band 7.2× should be read as "at least"** rather than as measured.
- `D(g,g)` p95 values are much larger than the MAEs (R18 high: 22.75° vs 2.846°), so the
  diagonal has a heavy tail. The per-LNA-state table shows why — the tail is plateau
  structure, not noise.
