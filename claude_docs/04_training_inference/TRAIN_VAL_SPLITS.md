# Train / val splits and split experiments

**The authoritative description of how training and validation data are partitioned, and of every
experiment run to test whether a different partition trains a better model.**

- What data exists: **[`../03_datasets/DATA_OVERVIEW.md`](../03_datasets/DATA_OVERVIEW.md)**
- What the quality tags mean: **[`../03_datasets/QC_METRICS.md`](../03_datasets/QC_METRICS.md)**
- Generator: `spf/scripts/make_v2_splits.py` · Manifest:
  `data_quality_reports/scan_2026_07_12_v2/v2scan_splits_MANIFEST.json`

---

## 1. Current partitions

### 1.1 Training

| Regime | File | Lines | Rover | Wall | Rule |
|---|---|---|---|---|---|
| **base** | `/mnt/md2/splits/apr17_train_nosig_noroverbounce_noblade.txt` | **1,691** | 108 | 1,583 | the historical default |
| **r1** (label-clean) | `/mnt/md2/splits/v2scan/train_r1_labelclean.txt` | **1,630** | 83 | 1,547 | base − duplicates − 56 dropped |
| **r2** (no-degraded) | `/mnt/md2/splits/v2scan/train_r2_nodegraded.txt` | **1,217** | 47 | 1,170 | r1 − `nan20` − `noisy` |

Arithmetic (verified): 1,691 lines − 5 duplicates = 1,686 unique − 56 drops = **1,630** (r1);
1,630 − 413 = **1,217** (r2), where 413 = 204 `nan20` + 411 `noisy` − 202 in both.

### 1.2 Validation — the frozen set and its named strata

| Set | File | Lines | Rover | Rule |
|---|---|---|---|---|
| **frozen val** | `/mnt/md2/splits/apr17_val_nosig_noroverbounce.txt` | **565** | 31 | never edited, never filtered |
| `val_clean` | `v2scan/val_clean_v2.txt` | **149** | 9 | `status == OK` |
| `val_degraded` | `v2scan/val_degraded_v2.txt` | **133** | 17 | `status == QUARANTINE` ∪ noisy |
| `val_band915` | `v2scan/val_915_v2.txt` | **108** | 0 | sub-GHz band |

The three strata are **subsets of the frozen 565**, evaluated additively via `val_subset_groups` in
the config — the frozen list itself is never modified. wandb metric keys are
`val_clean/single_loss`, `val_degraded/single_loss`, **`val_band915/single_loss`**
(`train_single_point.py:1325`). *There is no `val_915/single_loss` key* — that spelling is prose
shorthand. Val batch counts: clean 1,991 · degraded 1,699 · band915 1,495.

### 1.3 The non-destruction contract

1. The frozen val list is **never edited**; new views are named subsets that must be subsets of it.
2. `make_v2_splits.py` writes with an **assert-no-overwrite guard** (`:175`, `:183`) — split
   artifacts are append-only by construction.
3. Decision metric is `val_clean/single_loss`; `val/single_loss` is the historical-continuity
   metric; `val_degraded` is reported but never optimised toward.
4. Eval-only runs **must** pass a scratch `--output` — `--val-and-exit` executes the save-best path
   and would otherwise overwrite `best.pth`.

---

## 2. How the splits are generated

`make_v2_splits.py` takes the base train list, the frozen val list, and `metrics.csv`, and emits
five `.txt` files plus a manifest recording every input/output sha256.

### 2.1 The r1 drop rules (`:121-141`) — an **exclusive, order-dependent** `elif` chain

| Reason | Condition | Count |
|---|---|---|
| `error` | `status == "ERROR"` in the scan | **15** |
| `ghost_not_in_scan` | name has no row in `metrics.csv` | **0** |
| `train_val_leak` | name also in the frozen val | **0** *(see below)* |
| `frozen_tail` | `frozen_tail` column truthy (issue #42) | **16** |
| `mount_anomaly` | `abs(mount_diff) >= 0.5` rad | **2** |
| `rover_apr5_bounce` | `name.startswith("rover_2025_04_05") and "bounce" in name` | **23** |

Three subtleties worth knowing:

- **`ghost_not_in_scan` fires zero times only because of the input file choice.** The pre-`_noblade`
  list contains **190** datasets with no scan row (all BladeRF-era, 2025-03-16 onward); `_noblade`
  is exactly that list minus those 190, so the ghost rule is effectively pre-applied.
- **`train_val_leak = 0` should be read as "0 after error attribution."** There *is* exactly one
  dataset in both manifests — `wallarrayv3_2025_02_05_06_38_25_nRX2_bounce_spacing0p07` — but it is
  also an ERROR row, and `error` precedes `train_val_leak` in the chain. Net effect nil (it is
  dropped either way; `r1 ∩ val = 0` and `r2 ∩ val = 0` are both verified).
- **`mount_anomaly` can never fire on a wall dataset** — `mount_diff` is populated for rover rows
  only (`dataset_quality_scan.py:211-213`).

### 2.2 The r2 rule (`:144`)

```python
r2 = [(line, name) for line, name in r1 if name not in nan20 and name not in noisy]
#   nan20 = tagged("QUAR:nan>20%")            noisy = tagged("FLAG:r0_noisy") | tagged("FLAG:r1_noisy")
```

---

## 3. The r2 quarantine leak

> **Open bug.** `train_r2_nodegraded` — the "no-degraded" regime — contains **8 QUARANTINE rover
> datasets**, 17 % of its rover content.

`QUAR:nan>20%` is emitted **only in the wall branch** of the gate. Rover quarantines are tagged
`QUAR:no_signal` (see [`QC_METRICS.md` §5.2](../03_datasets/QC_METRICS.md#52-platform-threshold-differences-at-a-glance)).
So `name not in nan20` is *structurally incapable* of removing a rover quarantine — the only thing
that removes any rover from r2 is the incidental `noisy` clause.

Of the 15 `QUAR:no_signal` datasets: 3 are in the frozen val (never in train), 4 are excluded from
r2 only **by accident** because they also carry noisy flags, and **8 survive**:

| Dataset | r0 NaN | r1 NaN |
|---|---|---|
| `dec28_mission1_rover1` | 0.70 | 0.90 |
| `dec28_mission2_rover1` | 0.48 | 0.93 |
| `rover_2025_02_22_14_40_12…RO3` | 0.93 | 0.86 |
| `rover_2025_02_22_15_10_50…RO3` | 0.91 | 0.90 |
| `rover_2025_02_22_15_44_37…RO3` | 0.92 | 0.91 |
| `rover_2025_02_22_20_36_13…RO3` | 0.89 | 0.92 |
| `rover_2025_02_22_21_13_06…RO3` | 0.92 | 0.91 |
| `rover_2025_03_02_22_20_28…RO3` | 0.91 | 0.90 |

**The train and val sides use different definitions of "degraded."** `val_degraded_v2` is built from
`status == "QUARANTINE"` (platform-agnostic, correct); `train_r2_nodegraded` is built from a
platform-specific *tag string*. The same rover dataset is therefore "degraded" in val and "not
degraded" in train.

**Fix:** replace `name not in nan20` with `name not in quarantine`, symmetric with the val rule.
That yields **1,209** instead of 1,217 — a delta of exactly 8, with no other dataset reclassified.
**All published E-DATA1 numbers used the 1,217 version.**

---

## 4. Split experiments

### 4.1 E-DATA1 — the staged data-quality ladder (2026-07-12 → 07-15)

The only campaign with per-stratum val metrics. wandb entity **`projectspf`**, project
**`2024_nov22_single_paired_multi`**. One run per arm, resumed across stages.

| Arm | Train split | wandb |
|---|---|---|
| **base** | `apr17_train_nosig_noroverbounce_noblade.txt` (1,691) | [`qc7g2ou4`](https://wandb.ai/projectspf/2024_nov22_single_paired_multi/runs/qc7g2ou4) |
| **r1** label-clean | `train_r1_labelclean.txt` (1,630) | [`yz0qq836`](https://wandb.ai/projectspf/2024_nov22_single_paired_multi/runs/yz0qq836) |
| **r2** no-degraded | `train_r2_nodegraded.txt` (1,217) | [`i6v9t0xk`](https://wandb.ai/projectspf/2024_nov22_single_paired_multi/runs/i6v9t0xk) |

**Controlled comparison — verified.** `diff` across the three configs returns **exactly two hunks**:
the train path and the output dir. Everything else is byte-identical (batch 256, seed 10,
`precompute_cache_3p7`, beamformer depth 4 / hidden 512, lr 1e-4, `val_every 25000`). All three
share an identical step-0 val row, confirming the same init.

**Protocol:** 250k steps per stage, kill if >3 % behind on `val_clean/single_loss` at 250k,
>1.5 % at 500k; survivors resume +250k toward 1M.

#### Results (Δ vs base)

| Stage | Arm | `val` | **`val_clean` (decision)** | `val_degraded` | `val_band915` |
|---|---|---|---|---|---|
| **250k** | base | 0.099119 | **0.102108** | 0.111688 | 0.112775 |
| | r1 | 0.099925 (+0.81 %) | **0.103300 (+1.17 %)** | 0.112390 (+0.63 %) | 0.113089 (+0.28 %) |
| | r2 | 0.103632 (+4.55 %) | **0.101250 (−0.84 %)** ← *led* | 0.130576 (+16.91 %) | 0.133649 (+18.51 %) |
| **500k** | **base** | **0.097112** | **0.100472** ← *best* | **0.109395** | **0.110412** |
| | r1 | 0.097740 (+0.65 %) | 0.101484 (+1.01 %) | 0.109841 (+0.41 %) | 0.110827 (+0.38 %) |
| | r2 | 0.102881 (+5.94 %) | 0.100712 (+0.24 %) | 0.132177 (+20.83 %) | 0.135414 (+22.64 %) |

> ⚠️ **"250k" numbers are the validation at step 225,000 and "500k" at step 475,000** — `val` fires
> on `step % 25000 == 0` and the run returns after the increment, so there is no val exactly at
> 250k/500k.

#### Conclusion

**Filtering the training set by quality did not help.** At 250k, r2 (no-degraded) led on the
decision metric by −0.84 % — the result recorded in `learnings.md`. **By 500k that advantage had
evaporated (+0.24 %, i.e. marginally worse than base) while its robustness penalty grew**:
+16.9 → **+20.8 %** on degraded data and +18.5 → **+22.6 %** on the 915 MHz band. r1 (label-clean)
is uniformly ~0.4–1.0 % behind base and never wins anything.

Neither arm trips the 1.5 % kill rule, but on the full four-set table **base dominates**. The
plausible mechanism for r2: it cycles a 26 %-smaller train set 5.95 epochs vs base's 4.37 in the same
steps, and it never sees the degraded conditions it is then evaluated on — a robustness loss, not a
label-quality gain. Note also that r2's training set still contains 8 quarantined rover captures
(§3), so it is not the clean regime it claims to be.

**Status: stage 3 (750k) was never launched.** The ladder stopped on 2026-07-15; no run exists after
it in either wandb store, and no commit ever recorded the stage-2 numbers.

#### Incident: the run that died

r1's first stage-1 attempt (`run-20260713_024703`) stopped at step **124,899 / 250,000** — a silent
ENOSPC freeze when `/` filled. Its wandb dir is the last one written to `~/gits/spf/wandb`; every
subsequent run uses `WANDB_DIR=/mnt/md1/spf_wandb`. r1 was resumed from
`checkpoint_e2_s125000.pth` and completed. Cost: ~6 h of GPU time. The runner script gained a
"refuse to launch if `/` has <10 G free" guard afterward.

### 4.2 Earlier campaigns (2025)

These predate `val_subset_groups`, so the **only** metric is `val/single_loss`, comparable **only
within a fixed val set**.

| Campaign | Date | Arms | Result |
|---|---|---|---|
| **Rover up-weighting** | 2025-03-12 | 1× / 5× / 50× rover multiplier | 5× ≈ baseline (0.09201 vs 0.09201, `x1zxisg6` vs `gsquklx6`); 50× ran only 50k steps — undecidable |
| **Train-set size** | 2025-03-23 | full 1,696 / half 848 / quarter 423 | monotone degradation: **0.0930 / 0.0956 / 0.0996** (`vcr7vg88`, `ydoeqoe6`, `vc0z93ym`) — but step counts differ 2×, so a trend not a controlled test |
| **BladeRF exclusion ("noblade")** | 2025-07-13 | 1,881 vs 1,691 files | full 0.09623 (`vx2iwayo`) vs noblade 0.09754 (`qa5y7csl`) — configs differ only in the split, but 1.55M vs 905k steps, so not decision-grade. **`_noblade` nevertheless became the permanent default** and is what E-DATA1's base arm trains on |
| **Rover-bounce exclusion** | 2025-02 → 03 | before/after | **Not comparable** — the val set changed at the same time as the train set. Record as a regime transition, not an experiment |
| **Dec-2024 band split** | 2024-12 | 2.4 GHz-only vs all | designed (`dec7_train_24.txt` / `dec7_train_no24.txt`), results not recoverable — no local wandb dir predates 2025-02-11 |

Up-weighting uses an optional integer second column in the manifest
(`train_single_point.py:96-109`), used only in the `march11_*_{5x,50x}rover.txt` files. The apr17
manifests contain **zero** weighted lines.

### 4.3 What the experiment record supports

1. **More data beats cleaner data.** Every campaign that removed data — quarter/half, noblade, r1,
   r2 — landed at or behind the larger baseline on the metric it was judged by.
2. **Quality filtering trades robustness for a transient gain.** r2's advantage existed only at
   250k and only on clean val, while costing 20 %+ on degraded and sub-GHz data throughout.
3. **Rover up-weighting is neutral** at 5×; 50× untested.
4. **Do not promote a regime on `val_clean` alone** — judge the full four-set table.

---

## 5. Split-file inventory

**Current (production):** the six files in §1, plus `/mnt/md2/splits/bounce_rover.txt` (17 rover
missions excluded since 2025-03-12; fully honoured — none appear in either apr17 manifest).

**Legacy eras** in `/mnt/md2/splits/` (77 files) — kept for provenance, not for use:

| Era | Prefix | Note |
|---|---|---|
| A | `dec7_*` | Dec 2024, wall-only, the original band-split design |
| B | `dec26_* / dec28_* / dec31_*` | rovers enter the corpus |
| C | `jan14_*` | Jan 2025 |
| D | `feb4_* → feb16_*` | incremental val growth |
| E | `march11_* → march30_*` | rover-bounce exclusion + up-weighting variants |
| F | `apr5_* / apr17_*` | **`apr17_val_nosig_noroverbounce.txt` becomes the frozen val** |

## 6. Regenerating

```bash
python spf/scripts/make_v2_splits.py   # all defaults; refuses to overwrite existing artifacts
```

Inputs default to the base train list, the frozen val list, and
`data_quality_reports/scan_2026_07_12_v2/metrics.csv`. Delete the target directory first if you
intend to regenerate — the no-overwrite assert is deliberate.
