# The March 2025 deck's grids, reproduced on the 2026 rover corpus

**133,760 runs · 50,160 configurations · 16 captures · 5 seeds · 2026-08-11**

Config: [`rover2026_deck2025.yaml`](../../configs/rover2026_deck2025.yaml).
Every hyperparameter value visible in the "Droning on 2 (SPF)" deck, **none
omitted**, applied to all seven of our filter families rather than only the three
it showed.

Checking the deck's axes against every grid we had run found four gaps:

| axis | never tried before this run |
|---|---|
| `theta_err` | **0.1** |
| `theta_dot_err` | 0.075, 0.09, 0.12, 0.15 |
| `phi_std` | **8, 12, 14, 16, 18** |
| `noise_std` | 2e-6, 5e-6, 2e-5, 5e-5, 2e-4, 5e-4 |

The `phi_std` gap looked most promising: the deck's best EKF cell sits at
`phi_std` = 12, and our grids had gone `…, 5, 10, 20` — bracketing it without
ever landing on it.

---

## Result: none of it helps

Matched to E-INF1 on the same 16 captures and 5 seeds, paired per store:

| family | E-INF1 | deck grid | Δ | deck better on | Wilcoxon |
|---|---|---|---|---|---|
| `PF single radio NN` | 0.301 | 0.304 | +1.0% | 9/16 | 0.900 |
| `PF dual radio NN` [abs] | 0.534 | 0.518 | −2.9% | 8/16 | 0.860 |
| `PF single radio` | 0.541 | 0.540 | −0.1% | 10/16 | 0.528 |
| `PF dual radio NN` [craft] | 0.667 | 0.667 | −0.1% | 6/16 | 0.252 |
| `PF dual radio` | 0.838 | 0.837 | −0.1% | 6/16 | 0.900 |
| `EKF single radio` | 1.022 | 1.022 | +0.0% | 0/16 | — |
| `EKF dual radio` | 2.583 | 2.605 | +0.9% | 6/16 | 0.562 |

Everything is within ±3%. The one candidate — `PF dual radio NN` [absolute] at
−2.9%, using `theta_err` = 0.1 which had **never** been in any grid — is better
on **8 of 16** captures at **p = 0.860**. A coin flip. Not counted, by the same
paired standard that caught the false frame ranking in E-INF1.

`EKF single radio` is an exact reproduction: the deck grid's best configuration
*is* E-INF1's (`phi_std` 1.0, `p` 1.0, `noise_std` 1e-4), so every per-store
difference is identically zero and Wilcoxon is undefined.

### The `phi_std` region we had never sampled is worse

Best achievable MSE at each `phi_std`, over the deck's full `noise_std` × `p` grid:

| `phi_std` | 1 | 2.5 | 5 | **8** | 10 | **12** | **14** | **16** | **18** | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| EKF single [folded] | **1.022** | 1.071 | 1.346 | 1.245 | 1.377 | 1.266 | 1.256 | 1.261 | 1.244 | 1.249 |
| EKF dual [craft] | 2.625 | 2.746 | **2.605** | 2.714 | 2.746 | 2.701 | 2.683 | 2.836 | 2.815 | 2.880 |

The five values in bold are the ones our grids never contained. **Every one is
worse than `phi_std` = 1** for the single-radio EKF. The deck's `phi_std` = 12
optimum does not transfer.

---

## What *did* transfer: the shape, and only in one frame

This is the substantive finding, and it corrects something recorded earlier.

**`absolute_north` reproduces the deck's surface.** Our NN dual-radio absolute
panel optimises at `theta_dot_err` = 0.001–0.002 and degrades toward large
values — the same shape as the deck's `absoluteTrue` panel, whose optimum sits at
0.001–0.005. Two different corpora, different hardware, different routine, same
surface.

**`craft_relative` is inverted.** Our empirical dual-radio panel optimises at
`theta_dot_err` = 0.09 and is *worst* at 0.0001 (2.44). The deck's empirical
panel is the mirror image: best at 0.001–0.005, worst at 0.1–0.2 (1.43–1.97).

That is a real routine effect with a mechanism. `rover_diamond` is straight legs
with occasional sharp corners, so the receiver's heading is mostly constant and
the craft-relative bearing drifts slowly — favouring small process noise. Our
`rover_bounce` changes direction constantly, so craft-relative bearing moves fast
and needs large process noise. The **absolute** frame is insensitive to this
because the emitter's world-frame bearing drifts slowly whatever the receiver
does.

⚠️ **This corrects an earlier note.** When only the panels were visible I recorded
"`theta_dot_err` 0.1 is worst on rover_diamond and best on our rover_bounce" as a
flat routine dependence. It is narrower than that: the inversion is confined to
`craft_relative`. In `absolute_north` the two corpora agree. A frame and a routine
were being conflated.

It also independently reproduces **H3** on the deck's own axis resolution:
craft-relative optimises at `theta_dot_err` 0.09–0.12, absolute-north at
0.001–0.002 — a **~50×** ratio here, against the 20× measured on our own grid.

---

## Not comparable to the deck in absolute terms

The deck's filename footer records hardware absent from our corpus:

| | deck | ours |
|---|---|---|
| carrier | **5866 MHz** | 5766, 5840 |
| d/λ | **0.84138** | 0.67317 … 0.91557 |
| routine | rover_diamond / rover_center | bounce only |

The empirical table is keyed `{SDRDEVICE}_{d/λ:.5f}` with exact lookup, so the
deck used a measurement model that is not in `full_20260809_v1.pkl`. **Only the
shape of each surface is comparable, never the absolute MSE.**

---

## Every surface

Both views per family: the deck's is a **slice** with `theta_err` pinned (its
panels were made that way), ours minimises over the remaining axes. The EKFs have
no `theta_err`, so they appear once.

### PF single radio NN [radio_folded]

![PF single radio NN radio_folded__theta_err0.075](figures/heatmap_PF_single_theta_single_radio_NN__radio_folded__theta_err0.075.png)

*`theta_err` = 0.075.* Best **0.304 rad²** at `N` = 512, `theta_dot_err` = 0.12; worst cell 1.017; 48/104 cells within 5% of the best.

![PF single radio NN radio_folded__theta_err0.1](figures/heatmap_PF_single_theta_single_radio_NN__radio_folded__theta_err0.1.png)

*`theta_err` = 0.1.* Best **0.305 rad²** at `N` = 512, `theta_dot_err` = 0.1; worst cell 0.802; 53/104 cells within 5% of the best.

![PF single radio NN radio_folded](figures/heatmap_PF_single_theta_single_radio_NN__radio_folded.png)

*all axes minimised.* Best **0.304 rad²** at `N` = 512, `theta_dot_err` = 0.12; worst cell 0.802; 50/104 cells within 5% of the best.


### PF dual radio NN [absolute_north]

![PF dual radio NN absolute_north__theta_err0.075](figures/heatmap_PF_single_theta_dual_radio_NN__absolute_north__theta_err0.075.png)

*`theta_err` = 0.075.* Best **0.523 rad²** at `N` = 16384, `theta_dot_err` = 0.002; worst cell 1.579; 5/104 cells within 5% of the best.

![PF dual radio NN absolute_north__theta_err0.1](figures/heatmap_PF_single_theta_dual_radio_NN__absolute_north__theta_err0.1.png)

*`theta_err` = 0.1.* Best **0.518 rad²** at `N` = 32768, `theta_dot_err` = 0.001; worst cell 1.189; 4/104 cells within 5% of the best.

![PF dual radio NN absolute_north](figures/heatmap_PF_single_theta_dual_radio_NN__absolute_north.png)

*all axes minimised.* Best **0.518 rad²** at `N` = 32768, `theta_dot_err` = 0.001; worst cell 1.189; 6/104 cells within 5% of the best.


### PF single radio [radio_folded]

![PF single radio radio_folded__theta_err0.075](figures/heatmap_PF_single_theta_single_radio__radio_folded__theta_err0.075.png)

*`theta_err` = 0.075.* Best **0.540 rad²** at `N` = 4096, `theta_dot_err` = 0.075; worst cell 1.000; 53/104 cells within 5% of the best.

![PF single radio radio_folded__theta_err0.1](figures/heatmap_PF_single_theta_single_radio__radio_folded__theta_err0.1.png)

*`theta_err` = 0.1.* Best **0.542 rad²** at `N` = 512, `theta_dot_err` = 0.075; worst cell 0.878; 54/104 cells within 5% of the best.

![PF single radio radio_folded](figures/heatmap_PF_single_theta_single_radio__radio_folded.png)

*all axes minimised.* Best **0.540 rad²** at `N` = 4096, `theta_dot_err` = 0.075; worst cell 0.878; 54/104 cells within 5% of the best.


### PF dual radio NN [craft_relative]

![PF dual radio NN craft_relative__theta_err0.075](figures/heatmap_PF_single_theta_dual_radio_NN__craft_relative__theta_err0.075.png)

*`theta_err` = 0.075.* Best **0.667 rad²** at `N` = 16384, `theta_dot_err` = 0.12; worst cell 2.519; 41/104 cells within 5% of the best.

![PF dual radio NN craft_relative__theta_err0.1](figures/heatmap_PF_single_theta_dual_radio_NN__craft_relative__theta_err0.1.png)

*`theta_err` = 0.1.* Best **0.670 rad²** at `N` = 32768, `theta_dot_err` = 0.1; worst cell 2.055; 41/104 cells within 5% of the best.

![PF dual radio NN craft_relative](figures/heatmap_PF_single_theta_dual_radio_NN__craft_relative.png)

*all axes minimised.* Best **0.667 rad²** at `N` = 16384, `theta_dot_err` = 0.12; worst cell 2.055; 43/104 cells within 5% of the best.


### PF dual radio [craft_relative]

![PF dual radio craft_relative__theta_err0.075](figures/heatmap_PF_single_theta_dual_radio__craft_relative__theta_err0.075.png)

*`theta_err` = 0.075.* Best **0.837 rad²** at `N` = 4096, `theta_dot_err` = 0.09; worst cell 2.774; 18/104 cells within 5% of the best.

![PF dual radio craft_relative__theta_err0.1](figures/heatmap_PF_single_theta_dual_radio__craft_relative__theta_err0.1.png)

*`theta_err` = 0.1.* Best **0.846 rad²** at `N` = 4096, `theta_dot_err` = 0.09; worst cell 2.439; 24/104 cells within 5% of the best.

![PF dual radio craft_relative](figures/heatmap_PF_single_theta_dual_radio__craft_relative.png)

*all axes minimised.* Best **0.837 rad²** at `N` = 4096, `theta_dot_err` = 0.09; worst cell 2.439; 21/104 cells within 5% of the best.


### EKF single radio [radio_folded]

![EKF single radio radio_folded](figures/heatmap_EKF_single_theta_single_radio__radio_folded.png)

*all axes minimised.* Best **1.022 rad²** at `noise_std` = 1e-06, `phi_std` = 1; worst cell 1.377; 24/120 cells within 5% of the best.


### EKF dual radio [craft_relative]

![EKF dual radio craft_relative](figures/heatmap_EKF_single_theta_dual_radio__craft_relative.png)

*all axes minimised.* Best **2.605 rad²** at `noise_std` = 0.1, `phi_std` = 5; worst cell 3.318; 5/120 cells within 5% of the best.


---

## Reproducing

```bash
python spf/filters/run_filters_on_data.py \
  -d $(cat experiments/e_inf1_filter_sweep/stage2_rover_sample_n16.txt) \
  --empirical-pkl-fn empirical_dists/full_20260809_v1.pkl \
  --work-dir /mnt/qnap01/mouse9911/rovers_2026/filter_runs/deck2025 \
  --config spf/filters/configs/rover2026_deck2025.yaml \
  --results-backend local --parallel 4
```

```bash
python spf/filters/plot_hyperparam_heatmap.py \
  --results spf/filters/reports/e_inf2_deck2025_20260811_v1/results.json \
  --output-dir spf/filters/reports/e_inf2_deck2025_20260811_v1/figures \
  --slice theta_err=0.075,0.1
```

⚠️ **Use `--parallel 4`, not more.** This grid reaches `N` = 32768, where a worker
peaks at **11.3 GB RSS**. Two earlier attempts at `--parallel 22` and `12` were
killed by system-wide OOM — 22 workers in the high-N tail would want ~250 GB on a
188 GB box. At 4 workers the peak was 45 GB against 135 GB free. Results are
checkpointed per job and `--resume` is the default, so an interrupted run loses
only the in-flight jobs.
