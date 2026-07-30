# Data overview — everything we have captured

**The authoritative description of the SPF dataset corpus:** what exists, where it lives, how it is
organised, and how it scores on quality.

- Metric definitions, formulas and gate thresholds: **[`QC_METRICS.md`](./QC_METRICS.md)**
- What each data version's fields *mean*, per version: **[`formats/`](./formats/README.md)**
- Train/val partitions and split experiments: **[`../04_training_inference/TRAIN_VAL_SPLITS.md`](../04_training_inference/TRAIN_VAL_SPLITS.md)**
- Design rationale and root-cause investigations: `data_quality_plan.md` *(design record; its scan
  counts are stale — read numbers here)*

All numbers derived from `data_quality_reports/scan_2026_07_12_v2/metrics.csv` (2,250 datasets) and
from read-only inspection of the stores, on 2026-07-26.

---

## 1. Headline

| | |
|---|---|
| Datasets scanned | **2,250** (2,088 wall · 139 rover · 23 ERROR) |
| Status | **OK 727 · FLAG 1,176 · QUARANTINE 324 · ERROR 23** |
| Capture window | 2024-06 → 2025-04 |
| Raw capture footprint | **≈ 65 TB** across 20 directories on 3 arrays |
| Trained-on footprint | **5.5 GB** (`nosig_data`, signal-matrix stripped) |
| Precompute caches | ≈ 5.9 TB (`precompute_cache_3p4` … `3p7`) |
| Bands | 2.4 GHz (1,294) · 5.8 GHz (470) · sub-GHz 868/915 MHz (463) |
| Devices | PlutoSDR 2,179 · BladeRF2 48 |

**The single most important quality fact:** **no sub-GHz dataset in the entire fleet is OK** —
286 of 463 are QUARANTINE, 177 FLAG, 0 OK. Root cause in `learnings.md` L4 (IF parked at ~0 Hz;
crystal wander ≈ the IF offset, so the tone wandered through DC).

---

## 2. The two platforms

| | Wall array | Rover |
|---|---|---|
| Datasets | 2,088 | 139 |
| Motion | GRBL gantry | ArduPilot ground vehicle, GPS truth |
| Data version | **v5** | **v4** |
| Orchestrator | `grbl_radio_collection.py` | `mavlink_radio_collection.py` |
| tx↔rx range | 0.06–2.0 m (med **1.02 m**) | 0.01–58.9 m (med **29.0 m**) |
| Snapshot interval | med 0.995 s | med 0.489 s |
| Receiver speed (p99) | ~0.004 m/s | med **1.95 m/s** |
| Typical NaN | med ~0.3 % | med **~60 %** (bursty beacon — normal, see L2) |
| Status split | OK 694 · FLAG 1,085 · QUAR 309 | OK 33 · FLAG 91 · QUAR 15 |

`range_med_m` is the cleanest discriminator between them — the ranges do not overlap in practice.

---

## 3. Where the data physically lives

### 3.1 Raw wall captures — `2d_wallarray_v2_data` (≈ 62 TB)

| Directory | .zarr | Size | Date range | Band | In v2 scan |
|---|---|---|---|---|---|
| `/mnt/md0/…/jan` | 184 | 4.9 T | 2025-01-14 → 02-05 | 61 @2.4, 122 @900M | 184 |
| `/mnt/md0/…/oct_batch2` | 229 | 6.2 T | 2024-10-20 → 11-19 | 900 MHz | 229 |
| `/mnt/md1/…/june_fix` | 439 | 6.2 T | 2024-06-03 → 08-05 | 2.4 GHz | 439 |
| `/mnt/md1/…/aug` | 168 | 2.8 T | 2024-08-05 → 09-16 | 2.4 GHz | 168 |
| `/mnt/md1/…/sept` | 83 | 1.4 T | 2024-09-17 → 09-27 | 2.4 GHz | 83 |
| `/mnt/md1/…/oct` | 108 | 1.7 T | 2024-09-28 → 10-13 | 2.4 GHz | 108 |
| `/mnt/md1/…/nov` | 17 | 496 G | 2024-11-20 → 11-22 | 900 MHz | 17 |
| `/mnt/md1/…/nov_batch2` | 120 | 3.1 T | 2024-11-22 → 12-08 | 60 @2.4, 57 @900M | 120 |
| `/mnt/md2/…/dec` | 311 | 7.8 T | 2024-12-08 → 2025-01-14 | 273 @2.4, 38 @900M | 311 |
| `/mnt/md2/…/feb` | 246 | 12 T | 2025-02-05 → 02-28 | 40 @2.4, 198 @5.8 | 246 |
| `/mnt/md2/…/march` | 156 | 5.0 T | 2025-02-28 → 03-15 | 5.8 GHz | 156 |
| `/mnt/md2/…/march_nuand` | 82 | 3.8 T | 2025-03-16 → 03-30 | BladeRF | **15 of 82** |
| `/mnt/md2/…/april_nuand` | 121 | 6.6 T | 2025-04-03 → 04-16 | BladeRF | **33 of 121** |
| `/mnt/md2/…/april2_nuand` | 8 | 446 G | 2025-04-17 | BladeRF | **0** |
| `/mnt/md2/2d_wallarray_v2_data_missing_nosig` | 9 | 26 M | — | — | 0 (stubs) |

### 3.2 Raw rover captures — `/mnt/md2/rovers` (3.2 TB)

| Directory | .zarr | Size | Date range | In scan |
|---|---|---|---|---|
| `rover1` | 122 | 623 G | 2024-06-16 → 2025-04-05 | 46 |
| `rover2` | 114 | 645 G | 2024-11-13 → 2025-04-05 | 48 |
| `rover3` | 115 | 711 G | 2024-06-16 → 2025-04-05 | 49 |
| `merged` | 157 | 1.2 T | 2025-02-22 → 2025-04-05 | 140 |
| `merged_old` | 8 | 32 G | undated | 4 |

Tag ↔ directory is 1:1 (`rover1` ⇄ `_tag_RO1`). **`merged` holds the tx/rx-paired v5-style datasets**
that the scan and splits actually consume — the per-rover directories are the raw halves.

> **340 raw datasets are not referenced by the v2 scan at all** — mostly BladeRF (`april_nuand` 88,
> `march_nuand` 67, `april2_nuand` 8) and rover halves (rover1 76, rover2 66, rover3 66). They exist
> on disk but are outside every split and every quality number in this document.

### 3.3 Derived stores

| Path | Size | What |
|---|---|---|
| `/mnt/md2/cache/nosig_data` | **5.5 G** | signal-matrix-stripped copies — **everything trains and scans on these** |
| `/mnt/md2/cache/precompute_cache_3p7` | 1.6 T | **the cache the v2 scan used** (2,406 `.pkl` + 2,406 `.yarr`) |
| `…/precompute_cache_3p4 / 3p5 / 3p5_chunk1 / 3p6` | 0.5–1.6 T each | older segmentation versions |
| `/mnt/md0/checkpoints` | 461 G | 10 training eras incl. `jul12_2026` |
| `/mnt/md1/spf_wandb/wandb` | — | wandb run store |

The 4-order-of-magnitude size drop is the raw IQ: a 10,000-snapshot wall capture is
`signal_matrix (10000, 2, 524288) complex64` ≈ **42 GB raw → 2.8 MB nosig → 799 MB precompute**.

> ⚠️ All arrays are 96–98 % full. Free space on `/` has already caused one silent trainer freeze
> (ENOSPC mid-checkpoint) — see `learnings.md` E-DATA1 ops note.

---

## 4. Data format versions

The discriminator is `data-version:` in the per-dataset YAML sidecar.

| Version | Count | Platform | Position keys |
|---|---|---|---|
| **v4** | 140 | rover only | `gps_lat, gps_long, gps_timestamp, heading` |
| **v5** | 2,104 | wall only | `tx_pos_x_mm, tx_pos_y_mm, rx_pos_x_mm, rx_pos_y_mm, rx_heading_in_pis` |

Name prefixes are a 100 %-clean proxy: `wallarrayv3_*` → v5; `rover_*` (paired dotted name) and
`<mon><day>_missionN_roverM` → v4. Versions 2–3 wrote `.npy`; 4/5/6 write `.zarr`. **No v6 data
exists yet.**

---

## 5. Breakdown tables

### 5.1 Band × status *(corrected — see note)*

| Band | FLAG | OK | QUAR | Total |
|---|---|---|---|---|
| 2.4 GHz | 682 | 593 | 19 | **1,294** |
| 5.8 GHz | 317 | 134 | 19 | **470** |
| **sub-GHz (868/915)** | 177 | **0** | 286 | **463** |

> **Correction applied:** the CSV records `rx_lo` from *snapshot 0 of receiver 0 only*, and 59
> datasets never wrote that frame, so they read `rx_lo = 0`. Re-reading the median non-zero LO from
> each zarr recovers a real band for all 59. The table above is corrected; raw CSV counts are
> 1,278 / 431 / 459 with 59 unassigned.

### 5.2 Capture era × status — the sub-GHz campaign is visible

| Era | OK | FLAG | QUAR | ERR | Total | Dominant band |
|---|---|---|---|---|---|---|
| 2024-06 | 50 | 213 | 1 | 0 | 264 | 2.4 GHz |
| 2024-07 | 121 | 12 | 0 | 0 | 133 | 2.4 GHz |
| 2024-08 | 8 | 118 | 1 | 0 | 127 | 2.4 GHz |
| 2024-09 | 91 | 77 | 2 | 0 | 170 | 2.4 GHz |
| **2024-10** | **0** | 95 | 86 | 0 | 181 | 868 MHz |
| **2024-11** | **0** | 20 | 190 | 0 | 210 | 915 MHz |
| 2024-12 | 146 | 135 | 9 | 0 | 290 | 2.4 GHz |
| 2025-01 | 64 | 164 | 5 | 1 | 234 | 915.1 MHz |
| 2025-02 | 128 | 164 | 17 | 16 | 325 | 5.8 GHz |
| 2025-03 | 75 | 128 | 5 | 4 | 212 | 5.8 GHz |
| 2025-04 | 31 | 30 | 1 | 2 | 64 | mixed |
| undated (merged rover) | 13 | 20 | 7 | 0 | 40 | — |

**2024-10 and 2024-11 have zero OK datasets** — they are the sub-GHz campaign, not a rig failure.
Per `learnings.md` L8, era and band were confounded: October 2024 concurrently produced 96 *normal*
2.4 GHz datasets from the same rig.

### 5.3 Routine × status

| Routine | Platform | OK | FLAG | QUAR | ERR | Total |
|---|---|---|---|---|---|---|
| `bounce` | wall | 480 | 766 | 216 | 20 | 1,482 |
| `rx_random_circle` | wall | 144 | 144 | 21 | 2 | 311 |
| `rx_circle` *(retired)* | wall | 70 | 175 | 54 | 0 | 299 |
| `v4_calibrate` | wall | **0** | 0 | **18** | 0 | 18 |
| `bounce\|circle` | rover | 0 | 29 | 0 | 0 | 29 |
| `center\|circle` | rover | 19 | 9 | 8 | 1 | 37 |
| `diamond\|circle` | rover | 1 | 33 | 0 | 0 | 34 |
| merged (no schema) | rover | 13 | 20 | 7 | 0 | 40 |

`rx_circle` is the old name, retired after 2024-11 in favour of `rx_random_circle`.
`v4_calibrate` is 100 % QUARANTINE by design — it is a calibration routine, not training data.
Rover names are **paired** (`<rover-half>|<emitter-half>`): `diamond`/`center`/`bounce` only ever
appear as the rover half, `circle` only as the emitter half.

### 5.4 Rover d/λ groups (the 139)

| d/λ | n | OK | FLAG | QUAR | med circstd corr r0 | med g r0 | med heading |
|---|---|---|---|---|---|---|---|
| 0.418 | 32 | 12 | 13 | 7 | 0.48–0.54 | 0.96–1.04 | −0.15 … −0.20 |
| 0.673 | 21 | **0** | 21 | 0 | 1.008 | 0.90 | +0.07 |
| 0.685 | 19 | 1 | 11 | 7 | 0.571 | 1.04 | 0.00 |
| 0.827 | 24 | 6 | 18 | 0 | 0.977 | 0.93 | −0.06 |
| 0.841 | 28 | 13 | 14 | 1 | 0.370 | 0.96 | +0.09 |
| 0.920 | 7 | **0** | 7 | 0 | 0.569 | 0.90 | +0.11 |
| 0.993 | 8 | 1 | 7 | 0 | 0.773 | 0.91 | −0.16 |

The heading sign flip between the 0.418 groups (−0.15/−0.20) and the later groups (+0.07…+0.11) is
the **compass-calibration era boundary** — a skipped magcal is the traced cause of the Dec–Feb
heading bias (`ROVER_RUNBOOK.md` §3.6).

Wall d/λ spans 33 distinct values across 37 (device × spacing) groups — see
`scan_2026_07_12_v2/report_wall.md` for the full per-group table.

---

## 6. The 23 ERROR datasets

All 23 are marked `(persistent)` — they failed both the parallel pass and the serial re-check.
They are **excluded from every split**.

| Class | n | Signature |
|---|---|---|
| **A** — aborted capture, receiver frames never written | 9 wall | `AssertionError: 0.000000 -0.250000` |
| **B** — second receiver stream entirely empty | 2 | `AssertionError: Too many mismatches in rx_spacing` |
| **C** — nosig copy absent from disk | 6 wall | `No such file or directory` |
| **D** — segmentation missing from cache 3p7 only | 6 | `ValueError: Segmentation file does not exist` |

Classes C and D are **recoverable** — the raw capture still exists; only the derived artifact is
missing. Classes A and B are genuine capture failures.

---

## 7. Facts that repeatedly catch people out

1. **`FLAG:r*_gain` is not receiver gain.** It is the fitted effective/configured spacing ratio, and
   it flags *config families*, not captures — 1,268 wall datasets carry it because small-spacing rigs
   all inherit the same physical coupling floor (`learnings.md` L5). Do not drop data on it.
2. **Rover NaN of 46–70 % is normal**, not damage — the beacon is bursty. Only `QUAR:no_signal`
   (NaN > 90 % or < 100 valid) and heading bias are true rover failures (L2).
3. **Wall and rover use different quarantine tags** (`QUAR:nan>20%` vs `QUAR:no_signal`). Filtering
   on a tag string instead of `status == "QUARANTINE"` silently misses one platform — this is a live
   bug in the split generator (see `TRAIN_VAL_SPLITS.md`).
4. **`rx_lo` in the CSV is unreliable on 59 rows** (§5.1).
5. **Sub-GHz per-dataset phase is unusable** and recovery is closed (L4, L9). Amplitude/RSSI tasks
   and per-config medians remain valid.
6. **Raw data is immutable.** Never write to the stores in §3.1–3.2; all corrections go to new
   locations. `zarr_fix_rx_spacing` overwrites in place **with no backup** (KI#19).

## 8. Regenerating these numbers

```bash
python spf/scripts/dataset_quality_scan.py --splits <split1.txt> [...] \
  --precompute-cache /mnt/md2/cache/precompute_cache_3p7 \
  --output-dir data_quality_reports/scan_<YYYY_MM_DD> --parallel 12
```

Then read `report_rover.md` / `report_wall.md`, or query `metrics.csv` directly.
Metric meanings: [`QC_METRICS.md`](./QC_METRICS.md).
