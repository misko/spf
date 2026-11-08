# Additive validation-set expansion — plan (non-destructive)

> **ADOPTED DESIGN (2026-07):** `val_subset_groups` — named val groups that must be **subsets of
> the frozen val set** (pure regrouping at `load_dataloaders:278`, no new data loading; hard
> assert on the subset property). The frozen val list is never edited; train regimes are new
> txt manifests; baseline single + regime experiments re-run with all groups. Details below
> supersede the earlier `additional_val_sets` sketch where they differ (no new-data val sets
> in-run; the temporal holdout remains an offline `--val-and-exit` evaluation).

Goal: keep the current val set **frozen forever** (so `val/single_loss` and `save_on` stay
byte-comparable across all history), while adding **named additional val sets** for the
data-quality experiments (regimes R0–R3, `data_quality_plan.md`) — without modifying any
existing split file, config, or behavior of existing configs.

## Why the obvious approach is a trap (verified against source)

`datasets.val_paths` does **not** support multiple manifests: the `.txt`-manifest branch in
`load_dataloaders` fires only when there is *exactly one* txt in both train and val
(`train_single_point.py:134-148`); appending a second txt makes the loader silently treat the
txt files themselves as dataset paths. **Never touch `val_paths` semantics.** Instead we use the
mechanism the trainer already has: `alternate_val_ds` — named validation groups evaluated every
val pass and logged as `val_<key>/...` (this is where the per-device/spacing wandb keys come
from today).

## Non-destruction guarantees (the contract)

- **P1 — splits are immutable artifacts.** Existing files in `/mnt/md2/splits/` are never
  edited. New manifests go to `/mnt/md2/splits/v2scan/`, with a checksum manifest committed to
  the repo (splits live outside git).
- **P2 — configs are append-only.** Existing `model_configs/*.yaml` untouched; new configs are
  copies with provenance-carrying names.
- **P3 — the primary metric is sacred.** The main `val_ds` construction path is not modified;
  `val/single_loss` / `val/paired_loss` and `save_on` behave identically. Additional sets are
  evaluation-only and can never influence checkpoint selection.
- **P4 — additive code only.** One new optional config key with a default; absent key ⇒
  bit-identical behavior (verified by re-running an old config).
- **P5 — leakage guard.** At load time, assert every additional-val dataset ∉ train set.

## Phase 0 — retroactive baselines (no code, no retrain)

The trainer already logs per-dataset val metrics for every run. Compute the new strata
aggregates from the **existing jun26/dec15 wandb summaries**:
`val_clean` (the 149 v2-OK val datasets), `val_degraded` (92 QUAR + 33 noisy), `val_915` (108),
`val_blade` (48), per-spacing groups. Output: a baseline table so every future run compares
against measured numbers — the expectations table from the regimes discussion gets filled in
before any retraining.

## Phase 1 — new split artifacts (`/mnt/md2/splits/v2scan/`)

| File | Contents | Source |
|---|---|---|
| `val_clean_v2.txt` | 149 OK-status members **of the frozen val** | v2 metrics.csv |
| `val_degraded_v2.txt` | 125 QUAR/noisy members of the frozen val | v2 metrics.csv |
| `val_915_v2.txt` / `val_no915_v2.txt` | band strata of the frozen val | v2 metrics.csv |
| `val_heldout_mar_apr2025.txt` | **new** never-trained-on data: the ~190 excluded Mar/Apr-2025 wall datasets, quality-screened (scanner run on them first — they were never scanned) | excluded-set scan |
| `train_r0_hygiene.txt` | train minus errors/ghosts, deduped, leak removed (1,671) | accounting results |
| `train_r1_labelclean.txt` | R0 minus mount-anomaly + rover-bounce/interference; frozen-tails listed with truncation indices in a sidecar | v2 metrics.csv |

Plus `v2scan/MANIFEST.json` (file → sha256 + generation provenance) committed to the repo, and
the generator script (`spf/scripts/make_v2_splits.py`) so every file is reproducible.

## Phase 2 — the one additive trainer feature

New optional key:

```yaml
datasets:
  additional_val_sets:
    clean:    /mnt/md2/splits/v2scan/val_clean_v2.txt
    degraded: /mnt/md2/splits/v2scan/val_degraded_v2.txt
    heldout_apr25: /mnt/md2/splits/v2scan/val_heldout_mar_apr2025.txt
```

Implementation sketch (small, in `load_dataloaders` + `train_utils.load_defaults`):
1. Default `additional_val_sets: {}` in `load_defaults` (avoids the #36 KeyError pattern).
2. For each named manifest: datasets already in the loaded val set are **regrouped** (no
   reload — zero extra memory/time for strata of the frozen val); genuinely new datasets are
   loaded and registered **only** into `alternate_val_ds` under the given name.
3. Metrics appear automatically as `val_<name>/single_loss` etc. through the existing
   alternate-val logging path.
4. `save_on` resolution is untouched (and an assert rejects `save_on` pointing at an
   additional set, to keep P3 honest).
5. Leakage assertion per P5.

## Phase 3 — new configs (append-only)

Naming: `<base>_<trainregime>_valplus_<date>.yaml`. First three:

| Config | train_paths | additional_val_sets | Purpose |
|---|---|---|---|
| `..._baseline_valplus_<date>` | unchanged (apr17) | all new sets | re-validate: same training, new eyes |
| `..._r0_valplus_<date>` | `train_r0_hygiene.txt` | all | null-test (expect no movement) |
| `..._r1_valplus_<date>` | `train_r1_labelclean.txt` | all | first real regime |

Also a no-retrain harness: `--val-and-exit` with the `baseline_valplus` config against the
**existing** jun26 checkpoints populates the strata columns for current production models.

## Phase 4 — verification checklist (before any experiment counts)

1. **Bit-compat proof**: old jun26 config, patched code, `--val-and-exit` on the jun26
   checkpoint → `val/single_loss` must reproduce **exactly** (≈0.0984). Proves P3/P4.
2. **Strata cross-check**: `val_clean` etc. from the new loader must match the Phase-0
   retro-computed baselines. Proves the plumbing.
3. **Negative test**: a manifest containing a train dataset must trip the P5 assertion.
4. Commit order: trainer patch (with the already-drafted #53/#54 fixes) → split generator +
   MANIFEST → new configs. Each independently revertible.

## Standing rules going forward

- The frozen val list is never edited; new insight = new *named* set, never a mutation.
- Decision metric for quality experiments: `val_clean/single_loss`; `val/single_loss` remains
  the historical-continuity metric; `val_degraded` is reported but never optimized toward
  (the era-correlation trap).
- Every new manifest gets a scan verdict + checksum before first use.

## RESULTS — E0/E1 baselines (2026-07-12, commit 7842a9e)

Bit-compat gate PASSED: patched vs HEAD `--val-and-exit` on jun26 single best.pth
(e12/s725k) produce identical `val/single_loss = 0.09735` and identical all ~22 group
metrics. (Training-time 0.0984 differs only by RNG/sampling state — val_subsample 0.5 +
window shuffle; 0.09735 is the canonical val-and-exit baseline.)

E1 baseline strata for the jun26 single checkpoint (all future runs compare here):

| metric | loss |
|---|---|
| val/single_loss (frozen, historical continuity) | 0.09735 |
| val_clean/single_loss (149 OK) — DECISION METRIC | 0.1014 |
| val_degraded/single_loss (133 QUAR+noisy) | 0.10902 |
| val_band915/single_loss (108 sub-GHz) | 0.11003 |

Note: val_clean > val overall is expected — scan-"clean" means trustworthy labels/inputs,
not easy: the clean stratum has proportionally more hard-but-trustworthy configs (wide
d/λ, 5.8 GHz), while overall val contains large easy 2.4 GHz populations. Ordering
clean < degraded < band915 matches the quality analysis. Eval-only runs MUST pass a
scratch `--output`: val-and-exit executes the save-best path (best.pth would be
rewritten in the config's output dir).
