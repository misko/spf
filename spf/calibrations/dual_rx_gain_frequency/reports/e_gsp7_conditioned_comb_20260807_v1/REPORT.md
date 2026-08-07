# E-GSP7 — a ten-LO comb calibrates, but only chosen *and* frozen

**Session:** `e_gsp7_20260807` · captured 2026-08-07 · SPF `70d84b8`, clean checkout
**Pre-registration:** [`experiments/e_gsp7_conditioned_comb/`](../../../../../experiments/e_gsp7_conditioned_comb/experiment_readme.md)
— combs, decision rule and limits committed **before the first frame**
**Raw capture:** `artifacts/dual_rx_gain_frequency/e_gsp7_20260807/` (gitignored)

---

## 1. Result

The E-GSC ledger's open row was *"Is a sparse protocol now recommendable? Not yet —
it needs a prospective sparse capture with the comb chosen by conditioning."* This
is that capture: 111 LOs, 3,330 frames, one unchanged-harness session.

**A ten-LO comb does calibrate — under a conjunction of two conditions, and it
fails badly if either is dropped.**

| Arm | N | cond | Free-delay MAE | Frozen-delay MAE | Baseline |
|---|---|---|---|---|---|
| **chosen-10** | 10 | 1.09 | 8.352 ✗ | **4.950 ✓ (1.51×)** | 7.476 |
| **ecal3-10** *(control)* | 10 | 17.92 | 9.369 ✗ | **23.503 ✗✗ (0.31×)** | 7.280 |
| linspace-10 *(control)* | 10 | 21.78 | 46.116 ✗ | 17.557 ✗ | 7.344 |
| **chosen-16** | 16 | 1.05 | **5.573 ✓ (1.38×)** | 5.285 ✓ (1.46×) | 7.717 |
| committed coefficients, **no refit** | — | — | **3.863 (1.93×)** | — | 7.451 |

Held-out MAE in degrees, on the 95–101 LOs each arm did not train on. Every arm ran
at **100% coverage**, so nothing here is a coverage artifact.

**The headline comparison is `chosen-10 frozen = 4.950` against
`ecal3-10 frozen = 23.503`** — a **4.7× gap** between two ten-LO combs with
identical hardware-state coverage, in the same session, on the same harness, in
the same hours. The only difference is where the LOs sit in frequency. That is the
confound E-CAL3 could not exclude, and it is excluded here by construction.

## 2. Four predictions, tested prospectively

Everything below was predicted by prior work and is confirmed here on fresh data.

**2.1 E-CAL3's failure reproduces, to the ratio.** The real E-CAL3 comb, refit
free-delay, scores **9.369° against a 7.280° baseline — 1.287× worse than no
model**. The original E-CAL3 scored 11.61° against 9.06°, i.e. **1.281× worse**.
Different session, different absolute scale, same failure at the same ratio.

**2.2 E-GSC's 73.4% recovery figure.** E-GSC predicted that freezing the delays
recovers 73.4% of the dense improvement at N = 10, from retrospective subsampling.
Measured prospectively here, against the committed-transfer improvement:
**70.4%**.

**2.3 `N* = 16` with free delays.** No ten-LO comb beat the baseline free-delay —
not even the well-conditioned one (8.352 vs 7.476). Sixteen did (5.573 vs 7.717).

**2.4 "~1.9× on transfer".** The committed coefficients, scored with **no refit**
on this fresh capture, improve 7.451° → **3.863°, a ratio of 1.93×**. That is the
convention-invariant headline E-GSC recommended and that `docs/learnings.md` L10
was corrected to state earlier the same day, now replicated on an independent
session and harness.

## 3. The delays explain all of it

E-GSC said the nonlinear ripple delays fail first while the linear terms stay
fine, and that below N ≈ 32 the two ripple slots are frequently exchanged. The
recovered delays say exactly that, against the fleet values (2.56, 0.92) ns:

| Arm | Recovered τ (ns), free-delay | |
|---|---|---|
| **chosen-16** | **(2.50, 0.98)** | ✓ both slots, within 2.3% and 6.5% |
| chosen-10 | (3.18, 5.67) | ✗ both wrong |
| ecal3-10 | (4.15, 0.16) | ✗ both wrong |
| linspace-10 | (2.44, 4.90) | ✗ τ₁ close, τ₂ wrong by 5× |

Sixteen well-placed LOs identify the ripple delays. Ten do not, however well
placed — which is precisely why ten LOs only work with the delays **supplied**
rather than fitted.

## 4. Freezing the delays on a badly-conditioned comb is actively dangerous

This was not predicted, and it is the sharpest practical warning in the result.

Freezing the delays *helps* the well-conditioned comb (8.352 → **4.950**) and
*catastrophically hurts* the aliased ones (9.369 → **23.503**; 46.116 → 17.557 is
still 2.4× worse than baseline).

The mechanism is the one `ripple_conditioning`'s own docstring describes: when a
comb aliases the two delays, the four ripple columns become near-collinear, so the
fitted amplitudes go "large and arbitrary". With the delays **free**, the optimiser
can escape — it moves τ to somewhere less collinear on that particular comb, as the
(4.15, 0.16) and (2.44, 4.90) fits show. That is not physics, it is the fit
defending itself. **Freezing the delays removes that escape route.** Pinning τ at
the correct fleet values while the comb cannot separate them is the worst of both
worlds, and it produces the single worst number in this report.

**Operationally: never freeze the delays without first checking the comb's
conditioning.** The two choices are not independent knobs.

## 5. Decision rule

Pre-registered, and it splits by regime — as the pre-registration anticipated when
it required free-delay and frozen-delay fits to be reported separately and never
merged into one headline.

| Regime | Branch reached | Verdict |
|---|---|---|
| **Frozen-delay, N = 10** | *"chosen-10 beats the baseline and the control does not"* | **Comb conditioning is the actionable lever.** |
| **Free-delay, N = 10** | *"neither beats the baseline → escalate to N = 16"* | Ten LOs is too few regardless of placement. |
| **Free-delay, N = 16** | *"chosen-16 succeeds"* | `N* = 16` is the floor when delays are fitted. |

**Combined recommendation, and it is a conjunction:** a ten-LO calibration is
viable **only** with (a) the comb chosen by conditioning **and** (b) the ripple
delays frozen at fleet values. Drop either and it is worse than applying no model
at all. If the delays must be fitted, the floor is sixteen well-placed LOs.

Even then, ten-LO frozen (4.950°) does not reach the committed coefficients'
transfer performance (3.863°). **If fleet delays are trusted enough to freeze,
the committed coefficients are trusted enough to use** — so the sparse refit earns
its keep only where a genuinely local fit is needed, not as a default.

## 6. Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| Completeness | 1,665 frames per radio | **pass** — 3,330/3,330 |
| Frame quality | — | **pass** — 1,665/1,665 quality-valid on **both** radios, no reason codes |
| Cells | across-epoch stability ≤ 5° | **R17 pass** 555/555; **R18 fail** 550/555 |
| Gain tables | pre/post audits identical | **pass** — all 6 tables byte-identical, high = `90d34d61…` |
| Coverage | states seen in training | **pass** — 100% on every arm |
| Provenance | git SHA + clean flag | **pass** — `70d84b8`, dirty = False |

**R18 does not pass the strict gate**, reported rather than absorbed. Five of 555
cells exceed the 5° across-epoch circular-std threshold, all in 5100–5350 MHz:

| LO (MHz) | gains | circstd |
|---|---|---|
| 5100 | (45, 26) | 5.14° |
| 5200 | (26, 26) | 6.11° |
| 5200 | (26, 5) | 12.53° |
| 5200 | (26, 45) | 8.43° |
| 5350 | (5, 26) | 5.11° |

Every frame is quality-valid; these fail on epoch-to-epoch spread at n = 3, where
a circular std is poorly determined. The cluster sits in the >4 GHz region whose
degradation is already documented (arm asymmetry 0.73° → 3.72°, cross-radio ρ
0.99 → 0.45), so it is consistent with a known effect rather than a new one.

**One of these deserves explicit attention: 5200 MHz is a member of the chosen-10
comb**, and its failing cells include the equal-gain anchor cell (26, 26) at 6.11°.
Anchoring is per-epoch, so a per-epoch anchor is a single frame and the
across-epoch spread is absorbed rather than propagated — but the chosen-10 arm is
nonetheless trained on an LO where R18's anchor is among the least stable in the
capture. **That works against chosen-10, not for it**, so it does not threaten the
positive result; it does mean chosen-10's 4.950° is, if anything, pessimistic.

## 7. Limits

- **Evaluation is within-session**, exactly as pre-registered. Training and
  held-out LOs share session, harness and thermal state, so this establishes
  **identifiability and conditioning**, not session-to-session robustness. Session
  drift is real and separately documented (a 1,356-parameter LUT degrades
  0.62° → 2.74° across a 12-hour boundary). The committed-coefficient row (§2.4) is
  the only genuinely cross-session number here.
- **A sparse *protocol* is still not fully demonstrated.** What is demonstrated is
  that a conditioning-chosen ten-LO comb is *sufficient to fit from*, given frozen
  delays. A deployable protocol also needs the cross-session repeat.
- **Two radios, one harness topology.** Unchanged since E-CAL1 arm 1.
- The objective was `ripple_conditioning` — the two-delay ripple basis only, not
  the full design matrix. It proved sufficient to separate success from failure,
  but a comb optimal for the ripple is not proven optimal overall.

## 8. Reproduce

```bash
python experiments/e_gsp7_conditioned_comb/select_comb.py   # regenerates the combs
python experiments/e_gsp7_conditioned_comb/analyze.py \
  artifacts/dual_rx_gain_frequency/e_gsp7_20260807 results.json
```

The model, design and scoring code is the committed E-GSC/E-GSP machinery imported
unmodified; only the V7 loader is new, because the existing loader reads `.npz`
extracts of the old campaign whose raw stores E-GSC1 verified are absent from this
machine.
