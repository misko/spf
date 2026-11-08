# Non-default-branch audit — `train_single_point.py` + `train_utils.py`

Read-only audit of every branch the production config (`jul8_single_3p7_thin_noblade_jun26_2026.yaml`)
does **not** take, plus the operational machinery (resume / CLI / logging / plotting) that it does.
Complements `claude_docs/reference/_stage1_audit/SUMMARY.md` (prod forward/loss path — not re-audited here).
Severity = misbehavior × reachability per `DOCUMENTATION_GUIDE.md`; every finding carries
`✅ verified` (read/executed the deciding lines) or `🔍 analysis` (reasoned, not executed).

Prod operational values (from the config, for reachability calls): `log_every 100`, `val_every 25000`,
`plot_every 300000`, `checkpoint_every 5000`, `save_on: val/single_loss`, `epochs 100` (no `steps`),
`workers 24`, `amp false`, `dtype float32`, `device cuda`, `scatter continuous`, `loss mse`,
`head_start 0`, `direct_loss false`, `train_on_val false`, explicit `val_paths` (.txt manifests).
`--resume` **is** used operationally (wandb run `huaf6wjg` resumed 3×).

Line numbers: `tsp` = `spf/scripts/train_single_point.py`, `tu` = `spf/scripts/train_utils.py`,
`spn` = `spf/model_training_and_inference/models/single_point_networks.py`,
`utils` = `spf/utils.py`, `ds` = `spf/dataset/spf_dataset.py`.

---

## Branch table

| # | Branch | Taken by prod? | Verdict | Evidence | Severity |
|---|--------|----------------|---------|----------|----------|
| 1a | `--resume` auto-find newest checkpoint | **YES** (ops) | WORKS (edge crashes) | tsp:1045-1054 | P3 edges |
| 1b | Resume: `best.pth` tracking reset | **YES** (ops) | **RISKY** | tsp:1191, 1275-1292 | **P1** ✅ |
| 1c | Resume: unfreezes `load_single`/`load_paired` backbone | stage-2 resumes | **BROKEN** | tsp:1141-1161, 472-487 | **P1** (stage-2) ✅ |
| 1d | Resume: dataloader position restore | **YES** (ops) | WORKS | tsp:1195-1197; utils:184-199 | — ✅ |
| 1e | Resume: epoch-boundary off-by-one | **YES** (ops, rare alignment) | RISKY | tsp:1346-1356, 1194-1196 | P2 🔍 |
| 1f | Resume: RNG / scaler / best-loss state not restored | **YES** (ops) | RISKY | tsp:45-47, 1075/1135/1163, 1169 | P2 ✅ |
| 1g | Resume: wandb run-id reattach | **YES** (ops) | WORKS (step overlap) | tsp:1055-1061, 697-712 | P2 🔍 |
| 1h | `--resume-from` | occasional | WORKS (as designed) | tsp:1046, 1068, 1141-1150 | — ✅ |
| 1i | `optim/checkpoint` in config (warm start) | stage-2 | WORKS | tsp:1151-1161 | — ✅ |
| 1j | `optim/resume_step` config key | in prod config! | **DEAD** | prod yaml; no reader in tsp/tu | P3 ✅ |
| 2a | `--val-and-exit` | no | WORKS (side effect) | tsp:1212-1216, 1293-1294 | P2 side effect ✅ |
| 2b | `--no-val` | no | WORKS | tsp:1213, 1416-1419 | — ✅ |
| 2c | `--debug` | no | WORKS | tsp:1089-1094, 1324-1325 | — ✅ |
| 2d | `--debug-model` | no | **DEAD** | tsp:1406-1409; only used in `simple_train_filter.py:114` | P3 ✅ |
| 2e | `--steps` override | no | WORKS (quirks) | tsp:1064-1065, 1358-1359 | P3 ✅ |
| 2f | `--name` | no | WORKS (wandb only) | tsp:686-688 | — ✅ |
| 2g | `--save-prefix`, `--beamnet-latent` | no | **DEAD** | tsp:1395-1399, 1430 | P3 ✅ |
| 3a | `scatter: onehot` (`target_from_scatter_binned`) | no | **BROKEN** (3 defects) | tsp:578-597 | P2 ✅ |
| 3b | `loss: discrete` | no | WORKS (degenerate objective) | tsp:612-615, 1178-1179 | P3 ✅ |
| 3c | `direct_loss: true` | no | WORKS | tsp:961-973; spn:1174, 1306 | — ✅ |
| 3d | Paired target `single=False` (no fold) | no | WORKS (by design) | tsp:902-909, 540 default | — ✅ |
| 3e | `compute_loss:901` precedence quirk | no | RISKY (waste, no crash) | tsp:901-909; spn:1048-1053 | P3 ✅ |
| 3f | `scheduler: cosine` | no | **BROKEN** (UnboundLocalError) | tsp:389-397 | P2 ✅ |
| 4a | `random_snapshot_size: true` | no | WORKS | tsp:1086, 1200-1210; ds:127 | — 🔍 |
| 4b | `snapshots_per_session > 1` (multi path) | no | WORKS (diag losses wrong) | tsp:1083-1084, 769-845 | P3 🔍 |
| 4c | Alternate val dataloaders | **YES** | WORKS (no leak; ungated plots) | tsp:273-306, 357-367, 1255-1273 | P3 ✅ |
| 4d | `val_holdout_fraction` vs `val_paths` | val_paths wins (prod) | WORKS / brittle combos | tsp:141-164 | P3 ✅ |
| 4e | `train_on_val: true` | no | WORKS (ignored w/ .txt manifests) | tsp:141-152, 283-309 | P3 ✅ |
| 4f | `workers: 0` | no (`--debug` only) | WORKS | tsp:332, 1090 | — ✅ |
| 4g | `dtype float16` + `amp true` | no | RISKY | tsp:1111-1118, 1169-1172, 1301, 1323-1327 | P3 🔍 |
| 4h | `device: cpu` | no | WORKS (amp=false) | tsp:85, 1108-1109, 1169-1172 | P3 🔍 |
| 5a | `logger: simple` | no (`--debug` forces it) | WORKS | tsp:658-681, 1093-1094 | — ✅ |
| 5b | Unknown logger name | no | **BROKEN** (NameError) | tsp:1093-1096 | P3 ✅ |
| 5c | `plot_every` train-plot path | **YES** (step 0 + every 300k) | WORKS | tsp:1312, 750-899 | — ✅ |
| 5d | Val-plot plumbing (`val_plot_pred`) | **YES** (after 300k steps) | RISKY (latent assert; misfiled) | tsp:1227-1246, 1343 | P3 (prod-safe) ✅ |
| 5e | `checkpoint_every` / `val_every` interplay | **YES** | WORKS | tsp:1212-1216, 1346-1356 | — ✅ |
| 5f | `save_on` missing key | no (prod key valid) | **BROKEN** (KeyError, fail-fast) | tsp:1275-1276 | P3 ✅ |
| 5g | Output-dir `sys.exit` guard | **YES** | WORKS | tsp:1067-1073 | — ✅ |
| 6a | `epochs` vs `steps` termination | epochs (prod) | WORKS (quirk) | tsp:1194, 1358-1359 | P3 ✅ |
| 6b | `head_start > 0` | no (0) | WORKS | tsp:1315, 1016; see 3e | P3 ✅ |
| 6c | `load_paired: true` | no | WORKS (traj/multipaired only) | tsp:480-487; spn:1071, 1182 | — ✅ |
| 6d | `load_single` on multipaired model | no | **BROKEN** (AttributeError) | tsp:475; spn:1067-1110 (no attr) | P3 ✅ |
| 6e | `datasets/random_adjacent_stride` config key | no | **DEAD** (in this script) | tu:44; tsp:184-247 never passes it | P3 ✅ |
| 6f | Dead helpers: `FrozenModule`, `uniform_preds`, `weighted_beamformer`, `preload_dataset` | no | DEAD | tsp:400-413, 528-536, 600-605, 57-72 | P3 ✅ |

**Tally: 24 WORKS · 7 BROKEN · 6 DEAD · 6 RISKY** (some rows carry both a verdict and a caveat).

---

## 1. RESUME machinery (fires on the production path)

### 1a. `--resume` auto-find — WORKS, with edge crashes ✅
tsp:1048-1054: globs `{output}/*.pth`, excludes `best.pth`, parses step as
`int(basename_no_ext.split("_s")[-1])`, sorts by `(step, path)`, takes max. Filenames are
`checkpoint_e{epoch}_s{step}.pth` (tsp:429) so the parse is correct — **no off-by-one in parsing**;
`step` was already incremented before save (tsp:1345→1346), so the stored step = completed steps,
and `load_checkpoint` returns it verbatim (tsp:504-510).
Edges (P3, ✅): a dir containing **only `best.pth`** → `sorted([])[-1]` IndexError; any stray
`.pth` without `_s<int>` → ValueError. Both crash before training, no corruption.

### 1b. `best.pth` clobbered on every resume — **RISKY, P1** ✅
`best_val_loss_so_far = None` (tsp:1191) is a plain local, **never checkpointed or restored**.
After `--resume`, the first validation (skipped only for the very first batch via
`just_loaded_checkpoint`, tsp:1212-1216, then re-armed at the next `step % 25000 == 0`) hits
`best_val_loss_so_far == None` → unconditional `save_model(..., checkpoint_fn="best.pth")`
(tsp:1275-1292). If the run had already passed its best (e.g. overfitting regime), the historical
best model is **silently overwritten with a worse one — up to 3× for `huaf6wjg`**.
Reachability: prod (resume is routine + `save_on: val/single_loss` set) × misbehavior: silent loss
of the best artifact ⇒ **P1**.

### 1c. Resume unfreezes frozen backbones — **BROKEN for stage-2 resumes, P1** ✅
Stage-2 configs warm-start via `optim/checkpoint` + `model/load_single` (or `load_paired`), which
loads the sub-net and sets `requires_grad=False` (tsp:473-487). On `--resume`, the code takes the
`args.resume_from` branch with `force_load=True` (tsp:1141-1150), which does a **full**
`load_state_dict` and **never re-applies the freeze**; the `elif "checkpoint" in config["optim"]`
branch (tsp:1151) is skipped entirely. `requires_grad` is not part of a state dict, and
`single_loss` backprops into `single_radio_net` through the non-detached `output["single"]`
(spn:1021, 1049), so **the frozen backbone silently starts training after any stage-2 resume**.
Not the stage-1 prod config (no `load_*` there) ⇒ P1 scoped to stage-2 resumes.

### 1d. Dataloader position restore — WORKS ✅
tsp:1195-1197: each epoch sets `set_epoch_and_start_iteration(epoch, step % len(train_dataloader))`.
`StatefulDistributedSampler` reshuffles deterministically from `seed+epoch` and skips
`start_iter*batch_size` samples (utils:184-199 — the `i == start_iter*batch_size - 1` skip is
exact). Since `step` increments once per batch from 0 and every epoch has `len(dataloader)`
batches, `step % len` is the correct intra-epoch batch index; after a mid-epoch resume the
alignment self-heals at the next epoch boundary (`(S + len - i) % len == 0`).
Caveats: correctness assumes `len(train_dataloader)` is **identical** across runs — datasets that
fail to load are silently filtered (tsp:248-259), and wildcard configs depend on unsorted
`glob.glob` order (tsp:50-54) → position/holdout drift on resume (P2 🔍; prod uses fixed .txt
manifests, so low risk there).

### 1e. Epoch-boundary off-by-one — RISKY, P2 🔍
A checkpoint written at the exact end of epoch *e* (i.e. `step % len == 0`, possible whenever
`checkpoint_every` aligns with epoch length) stores `epoch=e` (tsp:1346-1356). Resume restarts
`range(e, epochs)` with `start_iteration=0` → **the whole epoch e is replayed** (data seen twice;
the restored `StepLR` hadn't stepped yet, so LR is consistent but the schedule sees one extra epoch
of data). Mid-epoch checkpoints resume exactly (no off-by-one). Reachability: needs
`checkpoint_every ≡ 0 (mod len)` — unlikely with prod's 5000 vs a large dataset, hence P2.

### 1f. What else silently diverges on resume — RISKY, P2 ✅
- **RNG**: `load_seed` only reseeds `torch` + `random` from scratch (tsp:45-47, called at
  1075/1135/1163); the stream *position* is not restored, so flip-augmentation / dropout /
  `random_snapshot_size` sequences differ from an uninterrupted run. `numpy` is never seeded
  (already flagged in stage-1 SUMMARY).
- **GradScaler** state is not saved/restored (tsp:1169-1172 vs save_model tsp:431-441) — amp-only,
  no-op in prod (P3).
- **LR schedule**: `scheduler_state_dict` **is** restored (tsp:501-502) and stepping is per-epoch
  (tsp:1360) ⇒ no LR divergence beyond 1e.
- **Config**: resume re-reads the *current* yaml, not the checkpoint's — silent hyperparameter
  changes possible; `{output}/config.yml` is overwritten (tsp:1104) after a timestamped backup
  (tsp:1056-1060, made whenever the old config had a `run_id`, which it always does per tsp:1101).

### 1g. wandb reattach — WORKS (this is the `huaf6wjg` ×3 mechanism), step overlap 🔍
`run_id` is persisted into `{output}/config.yml` on every run (tsp:1101-1105), read back on
`--resume` (tsp:1055-1061), and passed to `wandb.init(resume="allow", id=...)` (tsp:701-710) —
so all resumes reattach to the same run ✅. Divergence: training resumes from the newest
*checkpoint* step, but wandb history may already contain later steps from before the crash
(checkpoint every 5000, logs every 100) — `wandb.log(step=...)` to steps ≤ history is silently
dropped under `resume="allow"` 🔍, so up to ~4900 steps of post-resume metrics vanish per resume
(P2, cosmetic-metrics only).

### 1h/1i. `--resume-from` and `optim/checkpoint` — WORK as designed ✅
They are **mutually exclusive** with `--resume` (`assert args.resume_from is None`, tsp:1046), and
`--resume-from` requires a *fresh* output dir (`os.makedirs(..., exist_ok=args.resume)`,
tsp:1068 → `sys.exit(1)`). It restores model+optim+scheduler+epoch+step with `force_load=True`
but does **not** carry the wandb `run_id` → new run. `optim/checkpoint` (tsp:1151-1161)
deliberately discards the restored epoch/step (commented out, tsp:1159-1161) — warm-start
semantics, and honors `load_single`/`load_paired` freezing. Correct.

### 1j. `optim/resume_step` — **DEAD** ✅ (and sitting in the prod config)
The prod yaml carries `resume_step: 0`, but **nothing in `train_single_point.py`/`train_utils.py`
reads it** — the only `resume_step` in the tree is `spf/notebooks/simple_train_filter.py:275`.
Misleading config key; P3.

## 2. CLI flags

- **`--val-and-exit`** — WORKS (tsp:1212-1216 `or args.val_and_exit` bypasses the
  `just_loaded_checkpoint` guard; `return` at 1293-1294 after val). Two caveats ✅:
  (a) it runs the `save_on` block first, so **`--resume --val-and-exit` overwrites `best.pth`**
  with the newest checkpoint's weights (same mechanism as 1b) — P2; (b) the full train dataloader
  (24 workers) is constructed and its first batch fetched before val runs — overhead only.
- **`--no-val`** — WORKS: skips the whole val block; side effect: `best.pth` is never written
  (only periodic `checkpoint_e*_s*.pth`). ✅
- **`--debug`** — WORKS: `workers=0`, path lists truncated to 10 (tsp:166-168, 1089-1091),
  **forces SimpleLogger even if the config says wandb** (tsp:1093), asserts loss finiteness after
  backward (tsp:1324-1325 — note this makes `--debug`+amp trip on the first legitimate overflow). ✅
- **`--debug-model`, `--save-prefix`, `--beamnet-latent`** — **DEAD** in this script (parsed
  tsp:1395-1409, 1430; never read; copies of `simple_train_filter.py`'s parser). ✅
- **`--steps`** — WORKS: `if args.steps:` (tsp:1064-1065) overrides `optim/steps` (`--steps 0`
  silently ignored); termination `step >= steps` (tsp:1358-1359) runs *after* the
  `checkpoint_every` save, so a final checkpoint exists only if `steps % checkpoint_every == 0`;
  otherwise up to `checkpoint_every-1` steps of training are discarded on exit. P3 ✅
- **`--name`** — WORKS: wandb run name only (tsp:686-688); ignored by SimpleLogger. ✅

## 3. Target / loss alternates

### 3a. `scatter: onehot` → `target_from_scatter_binned` — **BROKEN if enabled, P2** ✅
Three independent defects (tsp:578-597):
1. **`single=` is silently ignored** — the signature accepts it but the body never folds the
   front/back reflection that `target_from_scatter` applies for `single=True` (tsp:563-574). Flip
   `scatter: onehot` on the prod model and the training objective silently loses the deliberate
   front/back symmetry described in the stage-1 SUMMARY.
2. **No wrap at ±π** — `GaussianBlur` uses reflect padding; **empirically verified** (spike at
   bin 0 blurs to `[0.403, 0.244, 0.054]` with zero mass at the far end): mass near ±π reflects
   instead of wrapping, unlike the continuous path's explicit 3-fold pad-and-sum (tsp:549-561).
   Targets within ~kσ of ±π are wrong.
3. **`y_rad` is ignored** (only `y_rad_binned` is used) — the three rotated diagnostic losses
   (`single_target_aligned_loss`, `single_craft_loss`, `single_plus_rand_loss`, tsp:760-845) pass
   a *modified* `y_rad` but the *unmodified* `y_rad_binned`, so under `onehot` all three
   diagnostics compare against the wrong (un-rotated) target. no_grad/metrics only.

**The blur axis itself is CORRECT** ✅: torchvision's `kernel_size[0]` is the *last*-dim (width)
kernel (`_get_gaussian_kernel2d`: `kernel1d_x = ...kernel_size[0]`), and an empirical probe
confirmed `(k,1)` blurs along θ (dim 2) with zero leakage across the session dim.
Also requires odd `k` (`scatter_k: 21` ✓) and `y_rad_binned < output_ntheta` (holds when
`output_ntheta == nthetas`, the `tu:57-61` default).

### 3b. `loss: discrete` — WORKS, degenerate objective ✅
tsp:612-614: `-(target·output).sum(axis=2).mean()` — mechanically sound (output is L1-normalized
non-negative, spn:987-993), bounded in [-1, 0], `best.pth` comparisons stay consistent. It rewards
putting all mass on the target's argmax bin (the sqrt/log variants that fix this are commented
out) — a modeling choice, not a bug. P3.

### 3c. `direct_loss: true` — WORKS ✅
Guarded inside `if paired and "multipaired" in output` (tsp:937, 961), and both multipaired models
always emit `multipaired_direct` (spn:1174, 1306) — no KeyError. No-op for `beamformer`/
`pairedbeamformer`. The `/30` balance constant (tsp:971-973) is a hardcoded TODO.

### 3d/3e. Paired symmetry & the `:901` precedence quirk ✅
Paired targets use `single=False` (default, tsp:540) — correct: the craft frame disambiguates
front/back, so no fold is wanted. tsp:901 parses as `(paired and "paired" in output) or
("multipaired" in output)`: with `head_start > epoch` on a multipaired model, `paired_target` is
computed and then discarded (both consumers re-check `paired and ...`, tsp:910/937). No crash —
multipaired models always include `"paired"` (spn:1048-1053 via 1114/1242) so
`output["paired"].shape` at tsp:908 is safe. Wasted compute only; P3.
Cosmetic: for multipaired models the plot writes `axs[2,*]` twice — the aligned-single diagnostic
(tsp:786-797) is overwritten by the multipaired panel (tsp:975-989). P3 ✅.
For S>1 sessions the aligned/craft/rand diagnostics rotate only `output["single"][:, 0]` and
broadcast against an (B,S,n) target (tsp:769-782) — diagnostics misleading for multipaired runs;
no_grad only. P3 🔍.

### 3f. `scheduler: cosine` — **BROKEN** ✅ (known; re-confirmed)
tsp:389-397: the branch is `pass`, then `return optimizer, train_scheduler` → UnboundLocalError at
startup. Fail-fast, config-gated (prod default `"step"` via tu:33). P2.

## 4. Data branches

### 4a. `random_snapshot_size: true` — WORKS 🔍
Only reachable for (traj)multipaired models (`assert` tsp:1086 blocks it elsewhere).
`batch_data[:, :effective]` slices a `TensorDict` whose `batch_size = y_rad.shape = (B, S)`
(ds:127), so all keys are sliced consistently; `math.ceil(rand*S) ∈ [1, S]` never produces 0
(tsp:1201-1207). `normalized_time` in the models tolerates S=1 (`+1e-3` guards, spn:1121/1126).
Not executed here, hence 🔍.

### 4b. `snapshots_per_session > 1` — WORKS (with 3e/3d caveats) ✅ asserts
Enforced `> 1` for multipaired models and `random_snapshot_size == False` otherwise
(tsp:1079-1086). `data_seen` bookkeeping uses `train_snapshots_per_session` (tsp:1338-1342) —
overstates data seen when `random_snapshot_size` truncates; metrics only.

### 4c. Alternate (per-device/spacing/routine) val dataloaders — WORKS; **no worker leak** ✅
Built by grouping val datasets on `sdr_device:collector:emitter:wavelength:rf:routine`
(tsp:273-281), per-group `val_subsample_fraction` subsample (tsp:296-306), one DataLoader each
with the full `workers: 24` and `persistent_workers=False` (tsp:316-335, 357-367). Every val pass
iterates each loader to exhaustion (tsp:1002 full for-loop) → workers spawn and are torn down per
loader per val cycle: expensive (24 procs × N groups every 25k steps) but not leaked.
**Prod-path caveat** ✅: each alternate loader passes `plot=True` unconditionally (tsp:1267) — NOT
gated by `plot_every` — so every val cycle builds and logs one 10×30-inch figure per group. Works
(all models emit `"single"`, and the fig is closed by the logger, tsp:729-731), just overhead. P3.

### 4d/4e. `val_holdout_fraction` / `val_paths` / `train_on_val` interplay ✅
Priority order (tsp:141-164): (1) both train & val are a **single .txt** → manifest loading with
per-line sampling; (2) `train_on_val` → val=train; (3) explicit `val_paths`; (4) holdout of the
**last** `max(1, frac·N)` shuffled train files. Consequences:
- Prod: branch (1); `val_holdout_fraction: 0.5` in the config is inert. ✅
- **`train_on_val: true` is silently ignored when both paths are single .txt manifests** (branch 1
  wins) — P3 ✅.
- Train = .txt but no `val_paths` → branch (4) treats the .txt file itself as a dataset; with one
  train file, `train_paths` becomes empty → `assert len(train_paths) > 0` (tsp:261) or load-fail →
  crash. Fail-fast config error; P3 ✅.
- `train_on_val` correctly disables alternates and subsampling (tsp:283, 307-309). ✅
- Holdout split determinism depends on seeded `random.shuffle` over **unsorted `glob.glob`** output
  (tsp:50-54, 139-140) — resume of a wildcard+holdout config can re-split train/val if directory
  enumeration order changes → train/val leakage. P2 🔍 (not prod).

### 4f. `workers: 0` vs 24 — WORKS ✅
`prefetch_factor = 1 if workers > 0 else None` (tsp:332) is the correct API dance;
`worker_init_fn` (tsp:36-42, expects `Subset→ConcatDataset→v5spfdataset`, which all three loader
shapes satisfy: tsp:293/308/312) is simply unused at 0 workers — main-process zarr handles work.

### 4g. `dtype: float16` + `amp: true` — RISKY 🔍, P3
`dtype` only sets the *dataset* `target_dtype` (tsp:90); the model stays fp32. The collate
force-casts the two big inputs to fp32 regardless (ds:119-120). Under autocast(cuda) the remaining
fp16/fp32 mixes are handled; the scaler sequence `scale→backward→step→update` is correct
(tsp:1323-1327). But **fp16 with `amp: false`** feeds fp16 tensors to fp32 Linear layers → dtype
RuntimeError 🔍; and scaler state isn't checkpointed (see 1f). `--debug`+amp asserts on the first
overflow-inf loss (tsp:1324-1325). Config-gated; P3.

### 4h. `device: cpu` — WORKS for `amp: false` 🔍
Dataset `gpu` flag keys off `== "cuda"` (tsp:85) ✓; `torch.GradScaler("cpu", enabled=False)` and
disabled autocast are no-ops (tsp:1169-1172, 1301). `amp: true` on cpu depends on torch's cpu
GradScaler support 🔍. P3.

## 5. Logger / plotting / checkpoint cadence

### 5a/5b. `wandb` vs `simple` ✅
Both loggers mean non-plot lists, close figures, and return the reported dict the `save_on` block
needs (tsp:665-680, 714-732). SimpleLogger ignores `--name` and mints a uuid `run_id`. **Any other
`logger.name` without `--debug` → `logger` is unbound → NameError** at first use (tsp:1093-1096) —
fail-fast; P3.

### 5c. Train-plot path (`plot_every`, **fires in prod**: step 0 + every 300k) — WORKS ✅
`plot = step == 0 or (step+1) % plot_every == 0` (tsp:1312). Verified against the prod model:
`output["single"]` always exists (spn:947-994), the 3×2 axes usage is consistent for
single-output models (tsp:750-899), the fig is asserted-unique, stored, logged, and closed
(tsp:1319-1321, 1343, 729-731). Since `log_every (100)` divides into the cadence, every plot is
flushed the same step it's created — the `assert "train_plot_pred" not in losses` (tsp:1320)
cannot fire in prod. **Prod trains past step 0, which exercises this exact path — no crash.**

### 5d. Val-plot plumbing — RISKY (latent), prod-safe ✅
The val figure is stored in the **train** loss accumulator (`losses["val_plot_pred"] = fig`,
tsp:1245-1246) and flushed by the next train log under the prefix **`train/val_plot_pred`**
(tsp:1343) — misfiled metric name, cosmetic. The unconditional
`assert "val_plot_pred" not in losses` (tsp:1233) crashes the run iff a val fig survives until the
next val, i.e. **iff `log_every > val_every`** once step > `plot_every` (300k). Prod: 100 < 25000
→ safe; a config with sparse logging + frequent val would die mid-run after 300k steps. Severity:
P1 misbehavior × unlikely config = P3.

### 5e. `checkpoint_every` / `val_every` interplay — WORKS ✅
Independent cadences: val runs *before* a step at `step % val_every == 0` (tsp:1213-1214),
checkpoints *after* increment at `step % checkpoint_every == 0` (tsp:1345-1346), `best.pth` only
during val. First val at step 0 always writes `best.pth` (fresh runs — fine; resumes — see 1b).

### 5f. `save_on` missing key — BROKEN (fail-fast) ✅
`reported_losses[config["optim"]["save_on"]]` (tsp:1275-1276) → KeyError on the first val (step 0)
for a typo'd or model-inapplicable key (e.g. `val/paired_loss` on a single model, whose list is
empty and thus filtered at tsp:667-670). Default `""` skips the block (tu:32). Prod's
`val/single_loss` always exists. P3.

### 5g. Output-dir guard — WORKS ✅
tsp:1067-1073: `os.makedirs(exist_ok=args.resume)` → logged error + `sys.exit(1)`. With `--resume`
on a *nonexistent* dir, the glob at tsp:1048 IndexErrors first (ugly but safe).

## 6. Other branches

- **`epochs` vs `steps` termination** ✅: `range(start_epoch, epochs)` (tsp:1194) vs the optional
  `steps` early-return (tsp:1358-1359) — see 2e for the missing-final-checkpoint quirk. `StepLR`
  steps per epoch (tsp:1360), so step-terminated runs never decay LR mid-epoch. Prod has no
  `steps` key (only the dead `resume_step`).
- **`head_start > 0`** ✅: gates paired/multipaired losses in train (tsp:1315) and val (tsp:1016)
  identically; with AdamW the gradient-less heads simply don't update during head start. Combined
  with 3e: wasted `paired_target` compute each step for multipaired models. Works.
- **`load_paired`** ✅: loads+freezes `model.multi_radio_net` (tsp:480-487), which exists on both
  multipaired models (spn:1071, 1182) and is a `PairedSinglePointWithBeamformer` — checkpoint must
  be from a `pairedbeamformer` run. WORKS. Conversely **`load_single` on a multipaired model** →
  AttributeError (`multi*` models have no `single_radio_net` attribute; it lives at
  `multi_radio_net.single_radio_net`) — config-error crash, P3 ✅.
- **`--beamnet-latent`** — dead here (2g); the model's `latent` comes from
  `model_config["latent"]` (spn:1193), not the arg.
- **Dead helpers** ✅: `FrozenModule` (tsp:400-413, its only use is commented at 478),
  `uniform_preds` (tsp:528-536), `weighted_beamformer` (tsp:600-605), `preload_dataset`
  (tsp:57-72; its Pool caller is commented at tsp:206-219).
- **`datasets/random_adjacent_stride`** — tu:44 defaults it, `v5spfdataset` supports it
  (ds:848, 899, 1351), but `load_single_dataset_using_configs` and both loader closures
  (tsp:75-93, 184-201, 229-247) never pass it → **setting it in a config is silently ignored** by
  this trainer. DEAD/misleading, P3 ✅.
- **compile × resume** 🔍: `torch.compile` wraps keys with `_orig_mod.`; resuming with a flipped
  `optim/compile` flag → state-dict key mismatch crash. Fail-fast, P3.

---

## Severity roll-up

| Sev | Finding | Conf |
|-----|---------|------|
| **P1** | 1b: `--resume` resets best-loss tracking → `best.pth` silently overwritten on every resume (prod ops path, resumed 3× already) | ✅ |
| **P1** (stage-2 scope) | 1c: `--resume` drops `load_single`/`load_paired` freezing → frozen backbone silently trains after resume | ✅ |
| P2 | 3a: `scatter: onehot` ignores `single=` (loses symmetry fold) + no ±π wrap + wrong diagnostics | ✅ |
| P2 | 3f: `scheduler: cosine` → UnboundLocalError | ✅ |
| P2 | 1e: epoch-boundary checkpoint replays a full epoch on resume | 🔍 |
| P2 | 1f/1g: RNG-stream/scaler state not restored; post-crash wandb steps silently dropped | ✅/🔍 |
| P2 | 4d: wildcard+holdout split not stable across resume (glob order) | 🔍 |
| P2 | 2a: `--resume --val-and-exit` clobbers `best.pth` | ✅ |
| P3 | 1j/6e: dead config keys (`resume_step` — present in prod yaml, `random_adjacent_stride`); 2d/2g dead flags; 5d latent val-plot assert (needs `log_every > val_every`); 5f `save_on` KeyError; 5b unknown-logger NameError; 6d `load_single`-on-multipaired AttributeError; 4c ungated alternate-val plots each val cycle (prod, overhead only); 2e steps-termination may skip final checkpoint; 4g fp16-without-amp dtype error | ✅ (except noted 🔍) |

**Prod-path verdicts confirmed safe:** train plotting (5c — exercised at step 0 of every prod
run), alternate-val loaders (4c — heavy but leak-free), checkpoint/val cadence (5e), resume
position restore (1d), wandb run-id reattach (1g), output-dir guard (5g).
