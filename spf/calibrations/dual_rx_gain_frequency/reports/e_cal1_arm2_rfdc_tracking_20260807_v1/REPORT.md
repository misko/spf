# E-CAL1 arm 2 — the RF-DC tracking loop is not the source either

**Session:** `e_cal1_arm2_20260807` · captured 2026-08-07 · SPF `25df125`, clean checkout
**Design:** [`experiments/e_cal1_rfdc_discriminator/`](../../../../../experiments/e_cal1_rfdc_discriminator/experiment_readme.md) (arm 2)
**Arm 1:** [`e_cal1_rfdc_20260807_v1/`](../e_cal1_rfdc_20260807_v1/REPORT.md)
**Raw capture:** `artifacts/dual_rx_gain_frequency/e_cal1_arm2_20260807/` (gitignored)

---

## 1. Result

Arm 1 measured the *total* RF-DC contribution at row 23 and found it negligible. Arm 2
repeats that measurement with the **continuous RF-DC tracking loop pinned off**
(`rf_dc_offset_tracking_en = 0`), leaving the one-shot initialization calibration exactly
as arm 1 ran it. It partitions arm 1's null.

**The tracking loop is not the source. Nothing changes when it is disabled.**

| | Arm 1 — tracking **ON** | Arm 2 — tracking **OFF** |
|---|---|---|
| RF-DC excess `H(9) − mid[H(8), H(10)]` | +0.069° ± 0.077 | **+0.019° ± 0.082** |
| Cluster-robust 95% CI | [−0.168°, +0.392°] | **[−0.258°, +0.250°]** |
| n (epoch-level) | 125 | 146 |
| Frames | 1050/1050 | 1050/1050 |

**Arm 1 − arm 2 = +0.050° ± 0.113 (t = 0.44)** — no detectable difference. Both arms are
individually consistent with zero, and arm 2's interval is slightly *tighter* because no
cluster had to be dropped.

This was the predicted outcome and it was pre-registered as such: arm 2 partitions a
quantity arm 1 had already measured as indistinguishable from zero, so it could
essentially only return "also zero". It does that cleanly. See
[§5](#5-what-this-does-not-settle) for the limit that still stands.

## 2. The pin is verified, not assumed

Arm 2's entire claim rests on the tracking loop actually being off. The AD9361 driver can
accept an attribute write without applying it, so the capture writes the attribute, reads
it back, and aborts on mismatch. Both radios recorded:

```
rf_dc_calibration_policy            = before_each_frequency_block   (unchanged from arm 1)
rf_dc_offset_tracking_en_requested  = 0
rf_dc_offset_tracking_en_observed   = {"voltage0": "0", "voltage1": "0"}
```

Request and observation are separate V7 fields precisely so a silently-ignored write
cannot masquerade as a null result. The pin was additionally re-asserted after every
`calib_mode` write and every LO retune, and hardware tests on these two radios confirmed
the write reaches silicon on this firmware.

## 3. Comparability with arm 1

The harness was **not touched between the two arms** — same session, same fixture, no
connector operation, USB enumeration unchanged (addresses 61/67 throughout). The arm-2
config copies arm 1's schedule byte-for-byte; the only fields that differ are `notes` and
`rf-dc-offset-tracking-en`, verified programmatically. Both runs used seed `2026080801`,
so the epoch and frequency orderings are identical.

That matters: an A→D connector re-mate moved the >4 GHz band by 12–34° in the source
campaign, which would dwarf the effect being partitioned here.

## 4. Per-cell detail and the one gate failure

RF-DC excess by (radio, LO), degrees:

| Radio | LO (MHz) | n | signed mean | sem | median \|·\| |
|---|---|---|---|---|---|
| R17 `0a003a` | 4001 | 25 | +0.083 | 0.076 | 0.124 |
| R17 `0a003a` | 5100 | 25 | +0.009 | 0.037 | 0.114 |
| R17 `0a003a` | 5766 | 25 | −0.575 | 0.155 | 0.822 |
| R18 `843ef2` | 4001 | 25 | −0.020 | 0.075 | 0.133 |
| R18 `843ef2` | 5100 | 21 | +0.209 | 0.386 | 1.119 |
| R18 `843ef2` | 5766 | 25 | +0.435 | 0.271 | 0.987 |

**R18 fails its validation gate**, and this is reported rather than absorbed: one cell of
21 — LO 5100 MHz, `g = (10, 5)` — has an across-epoch circular std of **5.15°** against
the `max-across-repeat-phase-std-deg: 5` threshold. It retained 23 of 25 epochs, so it
fails on phase stability, not on epoch count. R17 passed cleanly at 525/525.

Two things keep this from undermining the result:

- It is **the same cell that was weakest in arm 1** (R18 @ 5100 MHz), and the same cell
  the arm-1 pilot flagged before either capture. It is the low-SNR corner of this matrix,
  not a new failure.
- It is marginal (5.15° against 5.00°) and it is the *only* failing cell. Excluding that
  whole cluster changes the pooled arm-2 estimate from +0.019° to within its own sem.

Nonetheless: **arm 2's dataset does not pass the strict gate on R18**, and any downstream
use should carry that. Arm 1 passed both radios.

## 5. What this does not settle

**The central limit is unchanged, and arm 2 was never able to address it.** Both arms are
null. Neither can distinguish

> "the RF-DC machinery is quiet"

from

> "this harness cannot see RF-DC effects at all".

Only a **positive control** closes that: inject a perturbation of known magnitude and
confirm the pipeline recovers it. Until that runs, the honest statement is "no RF-DC
effect was detected by a measurement whose sensitivity to such an effect is inferred from
its noise floor, not demonstrated". The noise floor is well characterised here — the
LPF-only step measures 0.275° ± 0.071 in this session — which is *evidence* of
sensitivity, but not a demonstration.

Also still open: the row-11 (−3 dB) edge was not resampled, and the one-shot
initialization calibration (`calib_mode = rf_dc_offs`) fired at every frequency block in
both arms, so it is common-mode and cancels in the second difference rather than being
tested. A `never`-policy arm is now possible in code but was not run.

**Do not quote `mann_whitney_rfdc_vs_lpf` (p = 0.0073).** As in arm 1 it compares a 1 dB
step against a *halved* 2 dB step, so it detects a difference in spread, not location. It
is an artifact of the per-dB normalization in both arms.

## 6. Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| Completeness | 525/525 frames per radio | **pass** — 1050/1050 |
| Quality | ≥20 of 25 epochs per cell, circstd ≤ 5° | **R17 pass** (21/21); **R18 fail** (20/21, one cell at 5.15°) |
| Gain tables | pre/post audits identical | **pass** — all 6 tables byte-identical, high = `90d34d61…` |
| Tracking pin | requested state read back from the chip | **pass** — `0` observed on both channels, both radios |
| Harness | untouched since arm 1 | **pass** |
| Provenance | firmware, git SHA + clean flag in V7 | **pass** — `25df125`, dirty = False |

## 7. Reproduce

```bash
python ../e_cal1_rfdc_20260807_v1/analyze.py \
  artifacts/dual_rx_gain_frequency/e_cal1_arm2_20260807 results.json
```

The analysis code is arm 1's, unmodified — the two arms are scored by the identical
estimator, which is why the difference between them is interpretable.
