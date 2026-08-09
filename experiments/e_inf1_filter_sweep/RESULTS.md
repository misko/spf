# E-INF1 — results

**Status: NOT RUN.** Design and decision rules are pre-registered in
[`experiment_readme.md`](experiment_readme.md); this file exists so the
hypotheses cannot be edited after seeing the data.

| Hypothesis | Prediction | Outcome |
|---|---|---|
| H1 | NN dual-radio PF beats empirical on the rover corpus | _pending_ |
| H2 | best rover MSE ≥ 2× best frozen-val MSE | _pending_ |
| H3 | median `std(z)` > 1.5 on every corpus (filters overconfident) | _pending_ |
| H4 | MSE worse at d/λ = 0.904 than at 0.673 | _pending_ |

## Blocking item

3 of the 48 merged rover stores are RO4 at **d/λ = 0.90397**, which has no entry
in `empirical_dists/full.pkl`. `get_empirical_dist` is an exact-key lookup, so
every empirical (non-NN) filter raises `KeyError` on those. Resolve before stage
2 — either run `create_empirical_p_dist.py` for that spacing, or restrict the
empirical families to the 0.673 and 0.827 stores and say so here.

## Pilot observations (single dataset, 2026-08-07)

Not results — these motivated the experiment and are recorded so the stage-2
numbers can be sanity-checked against something.

On `rover_2026_08_01_19_31_21…RO3` (539 timesteps, d/λ = 0.827), one seed each:

| filter | MSE (rad²) | frame |
|---|---|---|
| PF dual, NN, `absolute=True` | 0.32–0.76 across 8 seeds | absolute_north |
| PF dual, NN, `absolute=False` | 1.86 | craft_relative |
| PF dual, empirical | 1.63–2.44 across 8 seeds | craft_relative |
| PF single, empirical (r0/r1) | 0.32 / 0.58 | radio_folded |
| EKF dual | 2.57 | craft_relative |

±1σ coverage for the NN dual PF was **25.6%** against 68.3% nominal.

⚠️ The two NN rows are **not** comparable — different ground truth. And the
per-seed spread (42% empirical, 106% NN) is why stage 2 runs 5 seeds.
