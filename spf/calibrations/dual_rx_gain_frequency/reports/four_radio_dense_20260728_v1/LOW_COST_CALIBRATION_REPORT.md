# Four-radio low-cost calibration transfer

## Question

Can a gain LUT learned from other radios be transferred to a new radio after measuring only one or two scalar phase-calibration values, instead of collecting the full 10,404-frame dense calibration?

The evaluation is leave-one-physical-radio-out. The target radio never contributes to its universal gain LUT. Its only allowed adaptation data are the declared reference-gain anchor values; anchor frames are excluded from scoring.

- Phase convention: `RX1 minus RX2`.
- Reference gain: 26 dB on RX1 and RX2.
- One calibration value is the circular mean phase residual from the reference-gain cell at one frequency. Two values are measured at two frequencies and linearly interpolate/extrapolate the target-radio residual versus frequency.
- Fixed strategies are predeclared. Exploratory best anchors were selected on these same four radios and require validation on a fifth unseen radio.

## Input radios

| Serial | Completed | Quality-valid | Scalar-input SHA-256 |
|---|---:|---:|---|
| `104000707f0700120f001a0095f2dbee49` | 10404 | 10209 | `c4f6ba758f104104382ddbb98f737893e378014ab4b42f606e272b619424f6dd` |
| `104000f6ad020002fdff3a00bba2f096a1` | 10404 | 10200 | `3cdff911d608a6a1e9c30f1a2756e56f68e453aab5a950ab4d808edc6aaed492` |
| `104000b299050013f4ff0700255e35222f` | 10404 | 10184 | `90c4162a8428392d9e41162e491e0678707a1c7bcc6dd6572d3656597aff654e` |
| `104473b80a16000de6ff2000f8a6beca79` | 10404 | 9917 | `b648131701ecaf5d1910040427fbfb610244eca2f95f7fdf1c78eaff37bfd916` |

## Universal LUT plus target-radio anchors

| Universal base | Target values | Anchors | MAE ° | RMSE ° | P95 ° | Coverage |
|---|---:|---|---:|---:|---:|---:|
| Strict universal frequency + gain LUT (none) | 0 | none | 15.173 | 22.887 | 54.015 | 100.00% |
| Strict universal frequency + gain LUT (fixed) | 1 | 2412 MHz | 14.997 | 20.620 | 44.522 | 100.00% |
| Strict universal frequency + gain LUT (fixed) | 2 | 868 MHz, 5866 MHz | 12.055 | 17.418 | 38.663 | 100.00% |
| Strict universal frequency + gain LUT (exploratory best) | 1 | 2412 MHz | 14.997 | 20.620 | 44.522 | 100.00% |
| Strict universal frequency + gain LUT (exploratory best) | 2 | 1280 MHz, 5766 MHz | 9.959 | 14.203 | 33.266 | 100.00% |
| Strict universal per-frequency additive LUT (none) | 0 | none | 14.171 | 22.302 | 51.411 | 100.00% |
| Strict universal per-frequency additive LUT (fixed) | 1 | 2412 MHz | 14.008 | 19.973 | 42.885 | 100.00% |
| Strict universal per-frequency additive LUT (fixed) | 2 | 868 MHz, 5866 MHz | 10.996 | 16.918 | 37.430 | 100.00% |
| Strict universal per-frequency additive LUT (exploratory best) | 1 | 2412 MHz | 14.008 | 19.973 | 42.885 | 100.00% |
| Strict universal per-frequency additive LUT (exploratory best) | 2 | 2412 MHz, 5766 MHz | 8.514 | 13.150 | 31.752 | 100.00% |
| Strict universal full cell LUT (none) | 0 | none | 14.201 | 22.363 | 51.574 | 99.99% |
| Strict universal full cell LUT (fixed) | 1 | 2412 MHz | 14.037 | 20.065 | 43.353 | 100.00% |
| Strict universal full cell LUT (fixed) | 2 | 868 MHz, 5866 MHz | 10.871 | 16.943 | 37.192 | 100.00% |
| Strict universal full cell LUT (exploratory best) | 1 | 2412 MHz | 14.037 | 20.065 | 43.353 | 100.00% |
| Strict universal full cell LUT (exploratory best) | 2 | 868 MHz, 5766 MHz | 8.555 | 13.365 | 33.794 | 100.00% |

![Low-cost strategy comparison](low_cost_strategy_comparison.png)

## Calibration at each operating frequency

If a deployment uses one RF channel, the following strategies require only one or two values for that channel. Evaluating all 12 frequencies uses 12 or 24 values, still far below a dense gain sweep.

| Universal base | Values per operating frequency | Second gain pair | MAE ° | RMSE ° | P95 ° |
|---|---:|---|---:|---:|---:|
| Strict universal frequency + gain LUT | 1 | none | 5.392 | 7.040 | 14.640 |
| Strict universal frequency + gain LUT | 2 | RX1 62 / RX2 26 dB (fixed) | 5.834 | 7.457 | 15.060 |
| Strict universal frequency + gain LUT | 2 | RX1 41 / RX2 3 dB (exploratory best) | 5.340 | 6.791 | 13.554 |
| Strict universal per-frequency additive LUT | 1 | none | 3.385 | 5.054 | 11.274 |
| Strict universal per-frequency additive LUT | 2 | RX1 62 / RX2 26 dB (fixed) | 3.419 | 5.022 | 11.553 |
| Strict universal per-frequency additive LUT | 2 | RX1 41 / RX2 -1 dB (exploratory best) | 3.019 | 4.366 | 9.762 |
| Strict universal full cell LUT | 1 | none | 3.439 | 5.370 | 11.379 |
| Strict universal full cell LUT | 2 | RX1 62 / RX2 26 dB (fixed) | 3.378 | 5.179 | 10.962 |
| Strict universal full cell LUT | 2 | RX1 41 / RX2 -1 dB (exploratory best) | 3.096 | 4.741 | 9.656 |

## Per-radio result for the recommended base

| Serial | No calibration MAE ° | One global value MAE ° | Two global values MAE ° | One value at each frequency MAE ° |
|---|---:|---:|---:|---:|
| `104000707f0700120f001a0095f2dbee49` | 6.881 | 8.927 | 5.441 | 3.293 |
| `104000f6ad020002fdff3a00bba2f096a1` | 8.423 | 8.426 | 11.798 | 2.522 |
| `104000b299050013f4ff0700255e35222f` | 17.762 | 17.277 | 10.677 | 3.232 |
| `104473b80a16000de6ff2000f8a6beca79` | 23.901 | 21.624 | 16.218 | 4.524 |

## Independent dense-run repeatability

| Serial | Common quality-valid cells | Cell-mean drift MAE ° | RMSE ° | P95 ° |
|---|---:|---:|---:|---:|
| `104000b299050013f4ff0700255e35222f` | 3399 | 0.532 | 0.783 | 1.754 |
| `104473b80a16000de6ff2000f8a6beca79` | 3308 | 0.705 | 0.976 | 1.919 |

## Interpretation

The full model ladder is reported separately in `MODEL_MATRIX_REPORT.md`. This report focuses on deployment cost. A one-value strategy can only remove a board-wide phase offset. A two-value strategy additionally removes one frequency-linear differential-delay term. Neither can represent arbitrary band-specific retune offsets.

Use the fixed-anchor results for engineering decisions. The exploratory best-anchor rows are an upper bound and a proposal for the next-board test, not an unbiased estimate for future hardware.

A robust field calibration should acquire several frames at each anchor and store their circular mean as one scalar value. With three frames per anchor, one- and two-value calibration require 3 or 6 frames instead of 10,404, reductions of 3,468× and 1,734× respectively.

For multi-frequency operation, one value at each of the 12 measured frequencies uses 36 robust calibration frames (289× fewer than dense); two values per frequency use 72 frames (144.5× fewer).

## Reproduction

The exact command, dataset list, scalar hashes, JSON results, CSV table, and plot are stored beside this report.
