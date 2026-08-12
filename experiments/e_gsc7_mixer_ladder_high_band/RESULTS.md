# E-GSC7 — results

**Status: NOT RUN.** Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | each 1 dB step 52→62 is resolvable (>3× the 0.355–0.368° floor) | _pending_ |
| H2 | the nine steps sum to 5.420° ± 1° at 5766 MHz | _pending_ |
| H3 | step size roughly uniform across mixer 5→15 | _pending_ |
| H4 | refit raises rover coverage 76.0% → 100% | _pending_ |
| H5 | `h_mixer(6…14)` transfers across high-band LOs | _pending_ |

## Baseline to beat

| | |
|---|---|
| no correction | 14.2–14.8° MAE (L10) |
| equal-gain anchor alone | 6.65° MAE, 8.31° on unequal-gain cells |
| L26 on the rover corpus | **refuses 100% of frames** — mixer 5–15 unfitted |
| E-GSC6 refit (no new capture) | projected **76.0%** coverage, accuracy unmeasured |

## The consistency check that must be reported first

E-GSC6, 5766 MHz, shared-effect steps: `51→52 = +0.330°`, `52→62 = +5.420°`.
The nine 1 dB steps measured here must sum to 5.420° if the effect is additive
across the mixer ladder. **Report that sum before attempting any refit.**
