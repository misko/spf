# Rover 2026 Q3 model-training setup

**Date:** 2026-08-26

**Status:** Prepared and loader-validated; training has not started

This document records the minimal setup for retraining the existing single-radio model family and then the paired model family with the 2026 Q3 rover data. It supplements the [model lineage report](../2026_08_25_model_training_lineage_and_data_state.md) and the [MD2 migration report](2026_08_26_rover_2026q3_v1_md2_migration.md).

## Intended comparison

The new run preserves the previous April train and validation manifests as ordered prefixes and appends rover data in new files. The original files are unchanged.

| Role | Historical prefix | Rover addition | New manifest |
| --- | --- | --- | --- |
| Train | `/mnt/md2/splits/apr17_train_nosig_noroverbounce_noblade.txt` (1,691 rows) | First 37 chronological stores | `/mnt/md2/splits/rover_2026q3_v1/train_apr17_plus_rover2026_n37.txt` (1,728 rows) |
| Evaluation | `/mnt/md2/splits/apr17_val_nosig_noroverbounce.txt` (565 rows) | Final 11 chronological stores | `/mnt/md2/splits/rover_2026q3_v1/eval_apr17_plus_rover2026_n11.txt` (576 rows) |

The rover additions partition the committed 48-store E-INF1 manifest. They have no rover-path overlap. The five duplicate train rows and one train/evaluation overlap inherited from the historical manifests are deliberately preserved for comparison with the previous setup.

The time split is a forward-development split, not a pristine test: these 48 stores were already used in E-INF1/E-INF2 evaluation. A later, post-freeze collection is still required for a final untouched test.

### Manifest identities

| File | SHA-256 |
| --- | --- |
| Historical train source | `6f8063db4e939f800e1d2b440104c59ba90882dec8fff66c9c9cc41f1b4ccdb5` |
| Historical validation source | `15008d63c89724bb743ce5d0fce254303d75c0593c1592ca1a8bb714dec89149` |
| Augmented train | `cde168555f86890b2a907d57142675b63c10971e8df1208261e4082f29a90e2b` |
| Augmented evaluation | `423f42e35de2f2a1b4fe09631bbcb02622efe4085987eba873f02f16a671918f` |

## Data and segmentation locations

| Artifact | Path |
| --- | --- |
| Promoted 2026 merged data | `/mnt/md2/rovers/2026q3_v1/merged` |
| Historical plus 2026 segmentation 3.7 | `/mnt/md2/cache/precompute_cache_3p7` |
| Isolated verified 2026 segmentation fallback | `/mnt/md2/cache/precompute_cache_3p7_rover_2026q3_v1` |
| Empirical table required by the current loader | `/home/mouse9911/gits/spf/empirical_dists/full_20260809_v1.pkl` |

The integrated cache has 2,454 matched PKL/YARR pairs: 2,406 historical pairs plus 48 rover pairs. The rover basenames did not collide with historical artifacts. The loader uses segmentation version 3.7, 65 beamformer angles, and `segment_if_not_exist=False`; it will not generate missing features during training.

`empirical_input` remains false in both models. The current training loader nevertheless opens, collates, and copies an empirical table, so the configurations use `full_20260809_v1.pkl`, which covers the new carrier/spacing cells. Its values are not model inputs under this configuration.

## Configurations

The new files retain the previous June architecture, augmentation, optimizer, seed, and validation cadence. Only dataset/artifact paths, output paths, and the paired warm-start checkpoint were changed.

| Stage | Configuration | Output |
| --- | --- | --- |
| Single | `model_configs/aug26_2026_single_3p7_apr17_plus_rover2026q3_v1.yaml` | `/mnt/md0/checkpoints/rover_2026q3_v1/single_seed10` |
| Paired | `model_configs/aug26_2026_paired_3p7_apr17_plus_rover2026q3_v1.yaml` | `/mnt/md0/checkpoints/rover_2026q3_v1/paired_seed10` |

The committed configuration SHA-256 values are:

- Single: `cbb195868ad2756113192a19ecf4650e0f64481c1bc736b46b66501d61ebe414`.
- Paired: `b4877749f344a8a4e6aa2ae556cf308d40f1080accc93473a510b7e831812826`.

The paired run loads and freezes the selected single model from:

```text
/mnt/md0/checkpoints/rover_2026q3_v1/single_seed10/best.pth
```

Therefore the stages must run sequentially. Do not launch paired training until that exact file exists.

## Completed preflight

Both YAML files were parsed with the production configuration loader and both models constructed successfully on CPU:

- Single model: 2,976,973 parameters.
- Paired model: 5,193,998 parameters.

The production dataloader was then constructed read-only from the single configuration:

- Train: 1,708 of 1,728 manifest rows loaded.
- Evaluation: 567 of 576 manifest rows loaded.
- Train examples: 14,710,647.
- Evaluation subsample examples: 1,782,287.
- Train batches at batch size 256: 57,464.
- Evaluation batches at batch size 256: 6,963.
- Every appended rover dataset loaded successfully.

The skipped rows are inherited historical missing/cache/metadata failures. This training code logs and skips individual failures. The counts above are therefore the expected startup signature; lower counts should stop the run for investigation.

## Exact launch sequence

Run from the repository root. To remain closest to the prior invocation, do not add `--steps`, `--resume`, `--output`, or name overrides.

### 1. Single model

```bash
cd /home/mouse9911/gits/spf
CUDA_VISIBLE_DEVICES=0 \
WANDB_DIR=/mnt/md1/spf_wandb \
/home/mouse9911/virtual-envs/spf/bin/python \
  spf/scripts/train_single_point.py \
  -c model_configs/aug26_2026_single_3p7_apr17_plus_rover2026q3_v1.yaml
```

At startup, confirm the expected loader signature above. After training, require:

```bash
test -f /mnt/md0/checkpoints/rover_2026q3_v1/single_seed10/best.pth
```

### 2. Paired model

```bash
cd /home/mouse9911/gits/spf
CUDA_VISIBLE_DEVICES=0 \
WANDB_DIR=/mnt/md1/spf_wandb \
/home/mouse9911/virtual-envs/spf/bin/python \
  spf/scripts/train_single_point.py \
  -c model_configs/aug26_2026_paired_3p7_apr17_plus_rover2026q3_v1.yaml
```

The `optim.checkpoint` entry is the stage-one warm start. It is not a request to resume an interrupted paired run.

## Operational rules

- Start from a nonexistent output leaf. The trainer refuses an existing output directory unless explicitly resuming.
- Use `--resume` only for a genuine interrupted run with a valid numeric checkpoint; do not use it after a failed zero-dataset launch.
- Record the Git commit, manifest hashes, config hashes, loaded counts, W&B run IDs, and selected checkpoint hashes with the results.
- Compare the new checkpoint on the combined evaluation manifest and, when reporting historical continuity, also evaluate it on the unchanged April validation manifest alone.
- Do not begin the paired stage until the single checkpoint has been selected and frozen.

## Current boundary

The manifests, segmentation, empirical table, and model configurations are ready. No GPU training was launched while preparing this setup.
