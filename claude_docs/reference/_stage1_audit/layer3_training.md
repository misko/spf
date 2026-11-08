# Layer 3 audit — Training loop + PyTorch optimization (stage-1)

Adversarial, READ-ONLY audit of `spf/scripts/train_single_point.py` (+ `train_utils.py`) tracing
the EXACT config `model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`.
Stage-1 launch: `python spf/scripts/train_single_point.py -c <that config>`. Venv torch 2.9.1.

Verdict legend: ✅ verified-correct · ⚠️ smell/caveat · 🔴 bug. "Affects-this-run?" is the key column.

---

## Config facts that drive this audit (from the YAML)

- `model.name: beamformer` → `SinglePointWithBeamformer` (`load_model`, `train_single_point.py:516-517`).
- `model.windows_stats_net` present, but does **NOT** set `output_phi` → defaults `False`
  (`single_point_networks.py:744`). So the model output dict has **only `"single"`** — no
  `output_phi`, no `paired`, no `multipaired`.
- `global.nthetas: 65`, `model.output_ntheta` unset → `load_defaults` sets `output_ntheta = nthetas = 65`
  (`train_utils.py:57-61`). Output shape `(B, S=1, 65)`.
- `datasets.scatter: continuous` → `scatter_fn = target_from_scatter` (`:1184-1185`).
- `datasets.sigma: 0.1`, `datasets.scatter_k: 21`. `optim.loss: mse` → `loss_fn = mse_loss` (`:1176-1177`).
- `optim.head_start: 0`, `optim.direct_loss: false`, `optim.dtype: torch.float32`, `optim.amp: false`.
- `optim.scheduler_step: 6`, **no `scheduler:` key** → default `"step"` (`train_utils.py:33`).
- `datasets.flip: true` → the `assert model.name=="beamformer"` (`:1078`) passes.
- `global.beamformer_input: false`, `global.phase_input: false` — both **present** (relevant to #36).
- `optim.compile` unset → default `False` (`train_utils.py:48`); `torch.compile` path skipped.

---

## TASK 1 — What is actually backpropagated for THIS config

**Single backward-driving term: `single_loss` (weight 1.0).** ✅

`compute_loss` (`:735`) accumulates `loss = 0` (`:748`) and the only branch that fires for this
config is `if single and "single" in output:` (`:757`). Inside:

- The trained target is built once with `single=True`:
  `target = scatter_fn(batch, y_rad=batch["y_rad"][...,None], y_rad_binned=batch["y_rad_binned"][...,None],
  sigma=0.1, k=21, target_ntheta=65, single=True)` (`:859-867`).
- `loss_d["single_loss"] = loss_fn(output["single"], target)` (`:868`), then `loss += loss_d["single_loss"]`
  (`:871`). This is the **only** add into `loss` that runs.

Auxiliary terms that are **OFF** for this config (verified by reading the branch guards):

| Term | Guard | Status this run |
|---|---|---|
| `single_all_windows_phi_loss` (×0.01) | `if "output_phi" in output` (`:873`) | OFF — model emits no `output_phi` |
| `paired_loss` (×1.0) | `paired and "paired" in output` (`:910`) | OFF — no `paired` key |
| `multipaired_loss` (×1.0) | `paired and "multipaired" in output` (`:937`) | OFF — no `multipaired` key |
| `multipaired_tx_pos_loss` (×0.1) | `"multipaired_tx_pos" in output` (`:952`) | OFF |
| `multipaired_direct_loss` (×1/30) | `direct_loss` arg (`:961`), `=optim.direct_loss=False` | OFF |

Note: `paired = epoch >= head_start = 0`, i.e. `paired=True` is passed (`:1315`), but the
`"paired"`/`"multipaired"` keys are absent from output, so the paired block at `:901` evaluates
`paired and "paired" in output or "multipaired" in output` → `(True and False) or False` → `False`,
so the `paired_target` build at `:902` does **not** run either. ✅ No KeyError on `output["paired"]`.

**`no_grad` diagnostics do NOT enter backward.** ✅ The four diagnostic targets/losses —
`single_target_aligned_loss` (`:759-783`), `single_craft_loss` (`:799-822`), `single_plus_rand_loss`
(`:824-845`), `single_loss_old` (`:847-857`) — are each wrapped in `with torch.no_grad():` and are
never added into `loss`. Same for `uniform_loss_single` (`:879`). They are logged only.

**Backward call.** `scaler.scale(loss_d["loss"]).backward()` (`:1323`); with `amp=false` the
`GradScaler` is a no-op pass-through (scale=1), `scaler.step`/`scaler.update` reduce to a plain
`optimizer.step()`. ✅ consistent with dtype float32.

**`mse_loss` semantics** (`:608-609`): `((output-target)**2).sum(axis=2).mean()` — SSE over the 65
bins, mean over `B×S`. Both `output["single"]` and `target` are L1-normalized distributions over
dim=2, so the shapes/axes line up `(B,1,65)`.

**Target normalization & shape** ✅:
- `output["single"]` is `normalize(...abs(), p=1, dim=2)` (`single_point_networks.py:987-993`) → sums to 1 over the 65 bins.
- `target` from `target_from_scatter(single=True)` is the mean of two L1-normalized distributions
  → still sums to 1. Both `(B,1,65)`. (Numerically: each `rescaled` is `normalize(p=1,dim=2)` at `:558`,
  the 3-block fold `:561` preserves total mass=1, and the `single=True` average `:574` of two unit-mass
  vectors is unit mass.)

---

## TASK 2 — `target_from_scatter` correctness (continuous path)

`target_from_scatter` (`:539-575`). For THIS config `single=True`, `sigma=0.1`, `target_ntheta=65`.

- `assert y_rad ∈ [−π, π]` (`:542`). ✅ Edge handled by the padded grid below, not by clamping.
- **No torchvision GaussianBlur here.** The continuous variant builds the Gaussian **analytically** in
  radians: `diff = ((m − y_rad)/sigma)**2`, `rescaled = normalize(exp(−diff/2), p=1, dim=2)`
  (`:557-558`). The `k`/`scatter_k=21` and `y_rad_binned` args are **accepted but unused** in this
  variant (the kernel size only matters for `target_from_scatter_binned`). ⚠️ doc-note: any claim that
  sigma+kernel-21 GaussianBlur builds the active target is **wrong for this config** — that is the
  onehot path. Active target = analytic Gaussian, sigma=0.1, no discrete kernel.
- **Grid + wrap-fold (numerically verified):** `padding=n=65`, `effective_n=3n=195`. The grid
  `m = (arange(195) − 97) * (2/65) * π` spans `[−9.376, +9.376]` rad with spacing `2π/65 ≈ 0.09666`.
  Reshaping `(B,S,3,65).sum(axis=2)` (`:561`) folds the two outer blocks (`±[3.19, 9.38]` rad) back
  onto the central block which spans `[−3.093, +3.093]` rad. This correctly returns Gaussian tails
  that cross the ±π seam. ✅ No off-by-one: the central 65 bins are bin centers at
  `(j−32)*2π/65` for `j∈[0,64]`, matching `rotate_dist`'s convention (`rf.py:1518`, `dtheta=2π/65`).
- **Symmetry fold (single=True):** `y_rad_flipped = sign(y_rad)*π − y_rad` (`:564`) is the up/down
  (front/back) reflection; `target = (target + target_flipped)/2` (`:574`). This is the "flip moved to
  the loss" — the target is symmetrized so the network is not penalized for the array's inherent
  front/back ambiguity. ✅ Confirmed it is applied **in the target build**, recursing once with
  `single=False`.
- ⚠️ Harmless dead line: `torch.zeros(...)` at `:543` allocated and discarded.

**Off-by-one / wraparound verdict:** ✅ none found. Bin spacing `2π/65`, fold is exact, central block
symmetric about 0.

---

## TASK 3 — Cosine crash (#1) and #36 verdicts

**#1 (cosine `UnboundLocalError`) — DOES NOT APPLY to this config.** ✅
`load_optimizer` (`:377-397`): the config has no `scheduler:` key, so `load_defaults` injects
`optim.scheduler = "step"` (`train_utils.py:33`). The `"step"` branch runs (`:383-388`), binding
`train_scheduler = StepLR(optimizer, step_size=6, gamma=0.5)`. `return optimizer, train_scheduler`
(`:397`) is valid. The `"cosine"` branch (`:389-393`, body `pass` → `train_scheduler` unbound) is
never reached. The bug is real but **opt-in** and not hit here.

**#36 (`global_config_to_keys_used` KeyError) — DOES NOT APPLY to this config.** ✅ / refuted-for-this-run.
The risky reads at `train_utils.py:88-105` are `beamformer_input`, `gains_input`, `vehicle_type_input`,
`phase_input`, `sdr_device_type_input`, `signal_matrix_input`, `empirical_input`,
`windowed_beamformer_input`. THIS config sets **all** of them explicitly: `beamformer_input: false`
(line 28), `gains_input: True` (30), `vehicle_type_input: true` (29), `phase_input: false` (35),
`sdr_device_type_input: true` (39), `signal_matrix_input: false` (38), `empirical_input: false` (32),
`windowed_beamformer_input: true` (31). No KeyError. The latent risk (#36) only bites configs that
omit `beamformer_input`/`phase_input`; this one is safe.

Resulting collate key list for this run (base 14 keys + flags): adds `windowed_beamformer` (31),
`gains` (30), `vehicle_type` (29), `sdr_device_type` (39); `empirical` force-appended (`:324-325`).
Does **not** add `weighted_beamformer` (beamformer_input false) or `weighted_windows_stats`
(phase_input false). ✅

---

## TASK 4 — Train/val/checkpoint mechanics

- **`save_on: val/single_loss` → best.pth.** ✅ At val time, `logger.log(val_losses, prefix="val/")`
  returns `reported_losses` keyed `"val/<name>"` (`WNBLogger.log` `:714-732`). Then
  `this_loss = reported_losses[config["optim"]["save_on"]]` (`:1276`) reads `"val/single_loss"`, which
  exists because `compute_loss` always populates `single_loss` for this model and `run_val_on_dataloader`
  accumulates it (`:1026`). When `this_loss < best_val_loss_so_far`, `save_model(..., checkpoint_fn="best.pth")`
  (`:1282-1292`) writes `best.pth`. ✅ The key resolves; no KeyError.
  - ⚠️ Caveat: `reported_losses` is built only from list-valued entries with `len(value)>0`
    (`:715-719`); `single_loss` is always non-empty in val, so fine. If `save_on` ever named an
    all-empty metric it would KeyError — not the case here.
- **scheduler_step vs val_every/epochs.** ⚠️ but benign. `StepLR(step_size=6, gamma=0.5)` is stepped
  **once per epoch** (`scheduler.step()` at `:1360`), so LR halves every 6 **epochs**, independent of
  `val_every=25000` (which is in **steps**). `get_last_lr()` is read every step (`:1333`) for logging.
  With `epochs=100`, LR decays `1e-4 → 1e-4*0.5^16 ≈ 1.5e-9` by the end — aggressive but intended
  (comment "started at 0.0002"). No crash; just a design choice. ✅ mechanics sound.
- **Step counting / resume.** This is a fresh run (`optim.resume_step: 0`, no `--resume`,
  no `optim.checkpoint` key). `step` starts at 0 (`:1137`). `set_epoch_and_start_iteration(epoch,
  start_iteration=step % len(train_dataloader))` (`:1195-1197`) → 0 at epoch start. `checkpoint_every:
  5000` writes rolling `checkpoint_e{e}_s{s}.pth` (`:1346-1356`); `val_every: 25000` triggers
  val+best (`:1212-1216`). `just_loaded_checkpoint` guard (`:1215`) is irrelevant on a fresh run.
  No step-counting bug for this run. ✅
  - ⚠️ Latent (not this run): the `StatefulDistributedSampler.__iter__` resume-skip (`utils.py:191-194`)
    iterates the whole shuffled sequence to skip `start_iter*batch_size−1` items and only matters when
    resuming mid-epoch; with `resume_step 0` it's a no-op. Resume path also resets nothing here.
- **dtype float32 + amp false consistency.** ✅ `dtype=torch.float32` (`:1112-1113`),
  `GradScaler(enabled=False)` (`:1169-1172`), `torch.autocast(enabled=False)` (`:1301`). All three
  agree — pure fp32, no scaling, no autocast. `torch.set_float32_matmul_precision("high")` (`:1121`)
  enables TF32 matmuls (speed, slight precision change) but does not affect correctness.

---

## TASK 5 — New training-layer bugs affecting THIS run

- ✅ **No new blocking bug found** in the train-loop layer for this config.
- ⚠️ **Output-dir guard.** `os.makedirs(args.output, exist_ok=args.resume)` then `sys.exit(1)` on
  `FileExistsError` (`:1068-1073`). `optim.output =
  checkpoints/jun26_2026/single_3p7_thin_noblade`. If that dir already exists from a prior attempt and
  `--resume` is not passed, the run **exits immediately**. Operational, not a code bug — but it will
  block a re-launch into a populated dir. Affects-this-run: only if dir pre-exists.
- ⚠️ **`numpy` not seeded.** `load_seed` seeds `torch` + `random` only (`:45-47`); `np.random` is
  untouched. `seed=10` therefore does **not** make numpy-driven randomness deterministic. For this
  config the training-loop RNG consumers are torch/random (shuffle via DistributedSampler `seed+epoch`,
  the `single_plus_rand_loss` rotation uses `torch.rand`), so determinism of the *backward* path is
  fine, but any numpy randomness inside the dataset/augmentation (e.g. flip) would not be seeded.
  Affects-this-run: partial determinism only; not a crash.
- ✅ **Dataloader workers.** `workers: 24`, `worker_init_fn` re-inits nested `v5spfdataset`s per worker
  (`:36-42`, opens zarr per process) — required and present. `persistent_workers: False`,
  `prefetch_factor: 1`. No silent no-op.
- ✅ **StatefulBatchsampler.** `shuffle=true`, `seed=10`, `batch_size=256`. Wraps a
  `StatefulDistributedSampler(num_replicas=1, rank=0)` (`utils.py:202-222`). `set_epoch` reseeds the
  shuffle each epoch (`seed+epoch`), giving different-but-deterministic order per epoch. Correct for
  single-GPU.
- ⚠️ **`compute_loss(training=...)` dead param** (`:745`) — accepted, never read. No effect.
- ⚠️ **Aux `no_grad` metrics roughly double `compute_loss` forward cost** (four extra `scatter_fn` +
  three `rotate_dist` calls). Performance only; correct.

---

## TASK 6 — BOUNDARY-IN: shapes/keys the loss expects (to check vs Layers 1 & 2)

**From the MODEL output (Layer 2 must provide):**
- `output["single"]`: float tensor `(B, S=1, 65)`, **L1-normalized over dim=2** (must sum to 1; the
  model already does `normalize(.abs(), p=1, dim=2)`). The loss does not re-normalize the model output.
- For this config the output dict should contain ONLY `"single"` (no `output_phi`/`paired`/`multipaired`).
  If Layer 2 ever adds `output_phi`, an extra ×0.01 term silently enters the backward.

**From the BATCH (Layer 1/data must provide), for the active path:**
- `batch["empirical"]`: used only for `.shape[:2]` and `.device` inside `target_from_scatter`
  (`:544-545`) — must exist with batch/session dims `(B, S, …)`. Force-collated (`:324-325`).
- `batch["y_rad"]`: `(B, S)`, radians **in [−π, π]** (asserted `:542` after `[...,None]` → `(B,S,1)`).
  Drives the active `single_loss` target.
- `batch["y_rad_binned"]`: `(B, S)` long index in `[0,65)` — passed into `scatter_fn` (`:861`) but
  **UNUSED** in the continuous variant. Still collated. (Used only if scatter were `onehot`.)
- `batch["craft_y_rad"]`, `batch["rx_theta_in_pis"]`, `batch["rx_heading_in_pis"]`, `batch["y_phi"]`,
  `batch["rx_pos_xy"]`, `batch["tx_pos_xy"]`, `batch["craft_y_rad_binned"]` — consumed only by the
  `no_grad` diagnostics / inactive paired branch. They must exist (they're in the collate list) but do
  NOT affect the gradient.
- Model-input keys for this config: `all_windows_stats`, `windowed_beamformer`, `gains`,
  `vehicle_type`, `sdr_device_type`, `rx_wavelength_spacing`, `rx_lo` (per `global_config_to_keys_used`
  + global flags). These are Layer 1↔2 contract, not loss contract.
- The batch object must support `.to(device)` (`:1298`) and dict-style `batch[key]` access.

---

## TASK 7 — Cross-check of existing docs

`claude_docs/reference/spf/scripts/train_single_point.md`: **accurate** on all points relevant here —
loss-assembly table, the cosine #1 break (correctly flagged opt-in), the #36 KeyError risk, the
discarded `torch.zeros`, the dead `training` arg, the `single=True` symmetry fold, the StepLR
gamma=0.5 once-per-epoch. One clarification to add (not an error): for the **continuous** scatter
path the Gaussian is **analytic in radians**, and `scatter_k`/`y_rad_binned` are **unused** — the
GaussianBlur((k,1)) only applies to the `onehot`/`target_from_scatter_binned` path. The doc states
this correctly under each function but a casual reader of the config (`sigma:0.1, scatter_k:21`) might
assume kernel-21 builds the active target; it does not.

`claude_docs/04_training_inference/latest_models.md`: consistent — lists the jul8/3p7 "thin" single
config as `beamformer → SinglePointWithBeamformer`, stage-1 producing `best.pth` consumed by the
paired stage via `optim.checkpoint` + `detach: true`. Matches this audit. (This config is the
`jun26_2026` sibling of the documented `dec15` one; same architecture/loss path.)

`KNOWN_ISSUES.md`: #1 and #36 entries match the source. Both confirmed NOT triggered by this config.

---

## Questions / assertions for the data + model layers

1. **(Model, Layer 2)** Assert `output["single"]` is exactly `(B,1,65)` and already L1-normalized over
   dim=2. The loss does NOT renormalize; if normalization were removed upstream, `single_loss` and the
   `uniform_loss_single` baseline become meaningless. Confirm `output_ntheta` resolves to 65 here.
2. **(Model, Layer 2)** Confirm the output dict contains ONLY `"single"` for the `beamformer` model
   with this `windows_stats_net` (no `output_phi` key). If `output_phi` ever appears, a ×0.01 phi term
   enters the backward unannounced.
3. **(Data, Layer 1)** Assert `batch["y_rad"] ∈ [−π, π]` strictly (hard `assert` at
   `train_single_point.py:542` will crash the run otherwise). Confirm `y_rad` is in the **radio**
   reference frame (the active `single_loss` uses raw `y_rad`, not `craft_y_rad`).
4. **(Data, Layer 1)** Confirm `batch["empirical"]` is always present and shaped `(B, S, …)` even
   though `empirical_input: false` — it is force-collated (`:324`) and `target_from_scatter` reads its
   `.shape[:2]`/`.device`. A missing `empirical` key crashes target build.
5. **(Data, Layer 1)** `y_rad_binned` is collated but UNUSED in the continuous loss path — confirm no
   other layer relies on it for stage-1, and that its `[0,65)` range (binning correctness) is irrelevant
   here (it is, for this config).
6. **(Data, Layer 1)** With `flip: true`, confirm the up/down flip augmentation in the dataset is
   **consistent** with the target-side symmetrization (`single=True` fold). The comment "flip moved to
   the loss" implies the loss now symmetrizes; verify the dataset isn't ALSO double-folding in a way
   that biases the target.
7. **(Data, Layer 1)** numpy RNG is not seeded by `load_seed`. If any dataset augmentation (incl. flip)
   uses `np.random`, stage-1 is not fully reproducible under `seed=10`. Confirm whether data-layer
   randomness uses torch/`random` (seeded) or numpy (not seeded).
