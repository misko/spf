# Stage-1 MODEL + OPTIMIZATION coverage audit (SEG-7 … SEG-11)

Read-only correctness + test-coverage audit of the MODEL/OPTIMIZATION segments for
`python spf/scripts/train_single_point.py -c model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`
(`model.name: beamformer` → `SinglePointWithBeamformer`).

Files owned:
- `spf/model_training_and_inference/models/single_point_networks.py`
- `spf/scripts/train_single_point.py`

**Config facts that pin the active code path** (verified against
`model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`):
`beamformer_input:false`, `empirical_input:false`, `phase_input:false`,
`gains_input:true`, `rx_spacing_input:true`, `frequency_input:true`,
`vehicle_type_input:true`, `sdr_device_type_input:true`,
`windowed_beamformer_input:true`, `nthetas:65`.
Model: `block:true bn:true norm:layer depth:4 hidden:512 dropout:0.0 input_dropout:0.1`.
`windows_stats_net: {windowed_beamformer:true, normalize_windowed_beamformer:false,
hidden_channels:64, outputs:12, n_layers:8, norm:false, act:leaky,
window_shrink:0.5, window_shuffle:0.15, window_dropout:0.25, window_fraction:0.5}`.
Optim: `loss:mse, scheduler (default)→"step", scheduler_step:6, weight_decay:0.001,
learning_rate:1e-4, amp:false, dtype:float32, checkpoint_every:5000, val_every:25000,
save_on:val/single_loss`.
There is **no `conv_type` key under `windows_stats_net`**, so `AllWindowsStatsNet`
defaults `conv_type="1d"` (`single_point_networks.py:753`). The 2d/cross/transformer/1d2d
branches are dead for this run.

---

## SEG-7 — windowed-beamformer net `AllWindowsStatsNet`

### Correctness — VERDICT: CORRECT (for this config), with caveats

- **68-channel contract.** `input_channels = 3` then `+= self.nthetas` (=65) ⇒ 68
  (`:786-788`). `create_1dconv` first layer is `Conv1d(68→64,k=11,pad=5)`
  (`:604-617`, hidden=64 from config). Stack: 1 entry block + `(n_layers-1)=7`
  same-res SkipConnect blocks + 1 stride-2 + `n_layers=8` same-res + 1 stride-2 +
  final `Conv1d(64→outputs=12)` (`:618-690`). Final `.mean(axis=2)` over the
  time/window axis (`:902`) ⇒ per-row 12-vector. **12-output contract holds.**
- **Channel assembly (`forward`, :848-872).** `all_windows_stats` (B,1,3,256) is
  normalized per-channel by its own max, with channels 0,1 forced to `π`
  (`:851-852`), giving 3 channels. `windowed_beamformer` (B,1,256,65) is
  `.transpose(2,3)` → (B,1,65,256) and, because `normalize_windowed_beamformer:false`,
  divided by the **hardcoded `/500`** (`:871`) — 65 channels. Concat dim=2 ⇒
  (B,1,68,256), reshaped to (B,68,256) (`:872-875`). **Order = [3 stats, 65 θ], 68 ✓.**
- **Reshape correctness.** `reshape(-1, channels, windows)` collapses (size_batch,
  size_snapshots) → rows; output reshaped back to `(size_batch, size_snapshots,
  outputs)` (`:903-906`). Correct for S=1.
- **Window augmentation is train-only.** All of `window_shrink`/`window_dropout`/
  `window_shuffle` are inside `if self.training:` (`:877-894`), so `model.eval()`
  disables them — **no eval-time leak**. Logic:
  - `window_shrink` (0.5): with prob 0.5 crop a contiguous slice
    `[start, start+windows*(1-window_fraction)]`; `start` ∈ `[0, window_fraction*windows)`
    (`:878-883`). With `window_fraction:0.5` keeps ~50% contiguous windows.
  - `window_dropout` (0.25): **`elif`** — only runs when `window_shrink` did NOT fire
    (`:886-887`). Drops each window iid with p≈0.25 via boolean mask
    `torch.rand(windows) > window_dropout`. **The `elif` means shrink and dropout are
    mutually exclusive per step — by design, not a bug, but worth noting.**
  - `window_shuffle` (0.15): independent `if`, permutes the window axis (`:891-894`).
- **Assert guard.** `assert window_dropout==0.0 or not norm` (`:783-785`). Config has
  `window_dropout:0.25` and `norm:false` ⇒ guard passes. (`norm:true` + dropout would
  abort — because variable-length LayerNorm size would break; correct guard.)

### Caveats / latent issues (non-blocking for this config)
1. **`torch.rand(1) < self.window_shrink` and `< self.window_shuffle`** use the *probability*
   as a threshold against U(0,1) — so `window_shrink:0.5` means "apply shrink 50% of
   steps", NOT "shrink by 50%". The *amount* of shrink is governed separately by
   `window_fraction`. Easy to misread; behavior is internally consistent.
2. **`torch.rand(windows)` / `torch.randperm(...)` for dropout/shrink are drawn on the
   default device (CPU) implicitly for `torch.rand(1)` and shrink's `torch.rand(1)`**;
   the dropout mask `torch.rand(windows)` is also CPU. Indexing a CUDA tensor with a CPU
   bool mask works in torch but forces a host-device sync. Not incorrect, minor perf.
3. **`window_dropout` changes the time length per step**, so the final `.mean(axis=2)`
   averages over a variable number of windows. Fine (mean is length-invariant), but
   means batch rows in the *same* batch all share the *same* dropped indices (mask is
   not per-row). Intended (augmentation), but reduces diversity.
4. **`/500` hardcoded scale** (`:871`) with `normalize_windowed_beamformer:false`: the 65
   beamformer channels are NOT normalized while the 3 stat channels are. Scale-sensitive
   surface (noted in SUMMARY.md). Not a correctness bug.

### Existing tests — **NONE**
No test references `AllWindowsStatsNet`, `windowed_beamformer`, `window_shrink`,
`window_dropout`, or `window_shuffle` (grep over `tests/*.py` is empty). The only e2e
fixture (`single_net_checkpoint`, conftest) runs `model_configs/test_single_net.yaml`,
which has **`phase_input:True`, `beamformer_input:False`, and NO `windows_stats_net`
block** — so `AllWindowsStatsNet` is **never instantiated by any test**. The entire 68→12
windowed path is untested.

### Proposed tests (gaps)
- **`test_allwindowsstatsnet_forward_contract`** (`tests/test_models.py`): build
  `AllWindowsStatsNet(nthetas=65, windowed_beamformer=True,
  normalize_windowed_beamformer=False, hidden_channels=64, outputs=12, n_layers=8,
  norm=False, act="leaky", conv_type="1d")`; feed a fake batch with
  `all_windows_stats:(B,1,3,256)` and `windowed_beamformer:(B,1,256,65)`; assert
  `out["all_windows_embedding"].shape == (B,1,12)` and `out` finite. **Catches** any
  break in the 68-channel concat / conv stack / final reshape.
- **`test_allwindows_eval_is_deterministic_no_aug`**: with all window-aug > 0, call
  `.eval()` and assert two forward passes on identical input are bit-identical
  (augmentation disabled); then `.train()` and assert outputs differ across passes
  (aug active). **Catches** an aug-leak-at-eval regression and a no-op-aug regression.
- **`test_allwindows_dropout_norm_guard`**: assert `AllWindowsStatsNet(window_dropout=0.25,
  norm=True, ...)` raises `AssertionError`. **Catches** removal of the `:783-785` guard.
- **`test_window_shrink_dropout_mutually_exclusive`** (optional/white-box): seed RNG so
  shrink fires, assert window count after equals the shrink count (not further reduced
  by dropout). **Catches** an `elif`→`if` regression that would double-shrink.

---

## SEG-8 — input assembly `PrepareInput.prepare_input` + `FFNN`/`BasicBlock`

### Correctness — VERDICT: CORRECT

- **18-dim input.** `PrepareInput.__init__` accumulates `self.inputs`
  (`:437-454`). For this config the active terms are: `gains(+2)`,
  `rx_spacing(+1)`, `frequency(+1)`, `vehicle_type(+1)`, `sdr_device_type(+1)` ⇒
  `self.inputs = 6`. `beamformer_input/empirical_input/phase_input` are false ⇒ 0.
  `SinglePointWithBeamformer.__init__` adds `additional_inputs += windows_stats_net.outputs
  = 12` (`:928-932`), so `FFNN(inputs = 6 + 12 = 18, ...)` (`:934-935`). **18 ✓ → Linear(18,512).**
- **prepare_input scaling** (`:493-536`): gains `/120` (2 ch), rx_spacing `*50`,
  frequency `log10(rx_lo+1)/20`, vehicle_type raw `[...,None]`, sdr_device_type `/10`.
  The window embedding 12-vec is appended via `additional_inputs` (`:538-541`,
  `forward :957/989`). Concat dim=2. Order: [12 window, gains2, spacing1, freq1, veh1,
  sdr1] — note **window features come first** because `additional_inputs` is the 2nd
  arg but is concatenated as `inputs + additional_inputs`... **correction:** code is
  `torch.concatenate(inputs + additional_inputs, dim=2)` (`:538-540`), so the *scalar*
  inputs come first and the 12-window embedding last. Either order is fine for an FFNN
  (Linear is permutation-agnostic given fixed mapping) as long as it's consistent
  between train & inference — and it is (same code path). ✓
- **input_dropout (0.1).** Train-only structured dropout: a `(8, B, S)` bool mask
  (`:476-481`) zeros each *input group* independently (mask row 0=beamformer,
  1=empirical, 2=phase, 3=rx_spacing, 4=frequency, 5=gains, 6=vehicle_type;
  sdr_device_type **deliberately never dropped**, `:534-536`). The `assert` at
  `:455-468` requires >1 active input group when `input_dropout>0` — config has 5 active
  groups ⇒ passes. **No eval leak** (guarded by `if self.training`).
  - ⚠️ **KNOWN_ISSUES #34 (in-place view mutation).** `v = batch["vehicle_type"][...,None]`
    then `v[dropout_mask[6]] = 0` (`:528-530`) mutates a **view** of `batch["vehicle_type"]`
    in place each training step. Benign here (single consumer; train loop never re-reads
    `batch["vehicle_type"]` after forward), but real. `gains`/`rx_spacing`/etc. create new
    tensors via arithmetic (`/120`, `*50`) so are NOT aliased; only `vehicle_type` (pure
    indexing) aliases.
- **FFNN construction** (`:55-112`). With `bn:true` AND `norm:"layer"`, control takes the
  **`elif norm=="layer"` branch** (`:88-106`): `Linear(18,512) → Dropout(0.0) → LeakyReLU
  → LayerNorm(512)`, then `depth=4` × `BasicBlock(512, LeakyReLU, bn=True, norm="layer",
  dropout=0.0)`, then `Linear(512, output_ntheta=65)`. **BasicBlock** (`:30-52`) is two
  `Linear→Dropout→LayerNorm→act` with a residual `out + identity` then act
  (`:48-52`). Shapes consistent (512→512). ✓
  - Note: model `dropout:0.0` ⇒ all `nn.Dropout(0.0)` are no-ops; the 0.1 dropout is
    `input_dropout` (structured, in `prepare_input`), a *different* mechanism. Correct.
- **abs → L1-norm.** `SinglePointWithBeamformer.forward` (`:987-993`):
  `F.normalize(ffnn(prepare_input(...)).abs(), dim=2, p=1)` ⇒ `{"single": (B,1,65)}`,
  non-negative, **sums to 1 over dim=2** (the 65-bin axis). ✓ `.abs()` guarantees
  non-negativity before L1; correct way to produce a distribution.

### Existing tests — **NONE direct**
No grep hit for `prepare_input`, `PrepareInput`, `FFNN`, `BasicBlock`, or `.abs()`/
`normalize` in `tests/`. **Indirect only:** `test_new_models.py::test_inference_single_checkpoint`
builds a `beamformer` model from the *test* config and runs forward, but with the
**phase_input** variant (`self.inputs` differs, no 12-window embedding) and only asserts
dataloader-vs-single-example *consistency*, **not** shape, not L1-norm, not the 18-dim
contract. So the 18-dim assembly used in production is untested.

### Proposed tests (gaps)
- **`test_singlepoint_forward_shape_and_l1norm`** (`tests/test_models.py`): instantiate
  `SinglePointWithBeamformer` with a global_config matching the production flags
  (gains/spacing/freq/veh/sdr on, windowed_beamformer on, nthetas=65) and a model_config
  with the `windows_stats_net` block; feed a fake batch; assert
  `out["single"].shape == (B,1,65)`, `out["single"].min() >= 0`, and
  `out["single"].sum(dim=2)` ≈ 1 (atol 1e-5). **Catches** an `.abs()` removal, a wrong
  `dim`/`p` in `F.normalize`, or a broken 18→65 head.
- **`test_prepare_input_dim_contract`**: assert `PrepareInput(model_config,
  global_config).inputs == 6` for this config, and that
  `prepare_input(batch, [embed12]).shape[-1] == 18`. **Catches** an off-by-one in the
  input accumulation (`:437-454`) or a dropped scalar.
- **`test_input_dropout_eval_noop`**: with `input_dropout:0.1`, `.eval()` then assert two
  forward passes are identical; `.train()` then assert the dropout mask path is reached
  (e.g. monkeypatch `torch.rand` or check variance over passes). **Catches** a
  train-guard regression that would leak dropout into inference.
- **`test_vehicle_type_not_mutated`** (regression for KNOWN_ISSUES #34): clone
  `batch["vehicle_type"]`, run a training-mode forward, assert the original batch tensor
  is unchanged. **Catches** the in-place view mutation if a future consumer depends on it
  (documents the issue even if currently benign).

---

## SEG-9 — loss `mse_loss`

### Correctness — VERDICT: CORRECT

`mse_loss(output, target) = ((output - target)**2).sum(axis=2).mean()`
(`train_single_point.py:608-609`). This is **SSE over the 65-bin axis (dim=2)** then mean
over (batch, snapshots). Both `output["single"]` and `target` are L1-normalized over
dim=2 (model side `:987-993`; target side `target_from_scatter` `F.normalize(...,p=1,
dim=2)` `:558` plus the single-mode average `:574` which preserves sum=1 since it's the
mean of two unit-sum vectors). So this is a proper per-row sum-of-squared-error between
two distributions, averaged. ✓ Reduction order (sum over bins, mean over rows) is the
standard choice and matches `uniform_loss_single`'s comparison baseline.

### Existing tests — **NONE**
No grep hit for `mse_loss`. Exercised only inside the e2e smoke fixture (no value
assertion — only that training runs and `best.pth` is produced).

### Proposed tests (gaps)
- **`test_mse_loss_handcomputed`** (`tests/test_models.py` or a new
  `tests/test_loss.py`): `output = tensor([[[0.25,0.25,0.25,0.25]]])`,
  `target = tensor([[[1.,0.,0.,0.]]])`; expected SSE =
  (0.75² + 0.25²·3) = 0.5625 + 0.1875 = 0.75; assert
  `mse_loss(output, target) == 0.75`. **Catches** a wrong axis (e.g. summing over the
  wrong dim) or a mean/sum swap.
- **`test_mse_loss_zero_when_equal`**: assert `mse_loss(x, x) == 0`.
- **`test_mse_loss_reduction_shape`**: 2-row batch with different per-row errors; assert
  result equals the mean of the two hand-computed per-row SSEs. **Catches** a reduction
  that sums instead of means over the batch.

---

## SEG-10 — optimizer/scheduler `load_optimizer`

### Correctness — VERDICT: CORRECT for this config; **KNOWN cosine bug confirmed**

- `AdamW(params, lr=1e-4, weight_decay=0.001)` (`:378-382`) — matches config. ✓
- `scheduler=="step"` (the resolved value: config has no `scheduler` key, and
  `train_utils.load_defaults` sets `optim/scheduler → "step"` and `optim/scheduler_step →
  1` default; the config overrides `scheduler_step:6`) ⇒
  `StepLR(optimizer, step_size=6, gamma=0.5)` (`:383-388`). **gamma is hardcoded 0.5**
  (not configurable). With `epochs:100` the LR halves every 6 *epochs* (scheduler stepped
  once per epoch at `:1360`). ✓
- **CONFIRMED BUG (KNOWN_ISSUES #1) — does NOT apply to this config but is real:** the
  `elif scheduler=="cosine"` branch body is just `pass` (`:389-393`), leaving
  `train_scheduler` **unbound**, so `return optimizer, train_scheduler` (`:397`) raises
  `UnboundLocalError`. Any config with `scheduler: cosine` crashes at startup. Not hit
  here (resolved scheduler is "step").

### Existing tests — **NONE**
No grep hit for `load_optimizer`, `StepLR`, or `scheduler` in `tests/`.

### Proposed tests (gaps)
- **`test_load_optimizer_step`** (`tests/test_models.py`): build a tiny `nn.Linear`,
  call `load_optimizer({"learning_rate":1e-4,"weight_decay":1e-3,"scheduler":"step",
  "scheduler_step":6}, params)`; assert returned optimizer is `AdamW` with the given lr/wd,
  and `scheduler.step_size==6`, `scheduler.gamma==0.5`. **Catches** a regression in the
  StepLR construction.
- **`test_load_optimizer_cosine_raises`** (regression for KNOWN_ISSUES #1): assert
  `load_optimizer({...,"scheduler":"cosine"}, params)` raises `UnboundLocalError` (or
  `NameError`). This **pins the known bug** so a future "fix" (binding `train_scheduler`)
  is forced to update the test — and documents that cosine is currently non-functional.
- **`test_load_optimizer_invalid_scheduler_raises`**: assert `scheduler:"foo"` raises
  `ValueError` (`:394-395`). **Catches** removal of the validation branch.

---

## SEG-11 — train loop / checkpoint / val `train_single_point`

### Correctness — VERDICT: CORRECT (mechanics sound), several operational caveats

- **Step/epoch loop** (`:1194-1361`). Per epoch: `set_epoch_and_start_iteration` for the
  stateful sampler resume (`:1195-1197`); per batch: optional val, then `m.train()`,
  `optimizer.zero_grad()`, autocast-wrapped forward (`amp:false` ⇒ autocast no-op),
  `compute_loss`, `scaler.scale(loss).backward()`, `scaler.step`, `scaler.update`,
  `optimizer.zero_grad(set_to_none=True)` (`:1297-1328`). With `amp:false` the GradScaler
  is disabled (`:1169-1172`) so `scaler.scale/step/update` are pass-throughs → plain
  fp32 SGD-via-AdamW. ✓
- **Backprop term.** Only `loss_d["loss"]` is backpropped; for the `beamformer` model
  `compute_loss` accumulates exactly `loss_d["single_loss"] = mse_loss(output["single"],
  target)` (`:868-871`); all `single_*_aligned/craft/rand/old` are `no_grad`
  diagnostics (`:759-857`), `output_phi` term is absent (no phi head in this config),
  paired/multipaired blocks are skipped (model emits only `"single"`). ✓
- **Scheduler stepping.** `scheduler.step()` at end of *epoch* (`:1360`), not per batch —
  correct for StepLR(step_size in epochs). `losses["learning_rate"]` logs
  `scheduler.get_last_lr()[0]` (`:1333`). ✓
- **Checkpointing.** `checkpoint_every:5000` → `save_model(...)` every 5000 steps
  (`:1346-1356`) writing `checkpoint_e{e}_s{s}.pth` + `config.yml` +
  `empirical_dist.pkl`. `save_model` stores model/optimizer/scheduler state + epoch/step
  (`:431-441`). ✓
- **Validation + best.pth.** `val_every:25000`; `args.val` default True. Val runs under
  `m.eval()` + `no_grad` (`:1217-1219`), computes losses on `val_dataloader` and the
  per-device-type `alternate_val_dataloaders`. `save_on:"val/single_loss"` ⇒
  `this_loss = reported_losses["val/single_loss"]`; saves `best.pth` when it improves
  (`:1275-1292`). The `val/` prefix matches `logger.log(..., prefix="val/")` (`:1248`).
  ✓ **`m.train()` is restored at `:1297`** after the val block, so eval mode doesn't leak
  into training. ✓
- **`just_loaded_checkpoint`** guard (`:1140,1215,1295`) skips the val at the very first
  step after a resume to avoid re-validating/re-saving immediately. ✓
- **Resume.** `--resume` picks the highest-step `*.pth` (excluding `best.pth`) by parsing
  `_s{step}` (`:1048-1054`), and reuses the prior `run_id`. `load_checkpoint(force_load=True)`
  loads model+optimizer+scheduler+epoch+step (`:498-510`). ✓

### Caveats / latent issues (non-blocking)
1. **Output-dir guard aborts** (`sys.exit(1)`) if `args.output` exists without `--resume`
   (`:1068-1073`). Operational, by design.
2. **`best.pth` selection uses the *main* val split only** — `val/single_loss`, not the
   per-device alternates. Intended (single scalar), but worth knowing.
3. **`val_every` and `checkpoint_every` are in *steps*** but val triggers on
   `step % val_every == 0` evaluated at the *top* of the batch loop, while checkpoint
   triggers *after* `step += 1` — so they will rarely coincide on the same step index.
   Not a bug; just asymmetric placement.
4. **Reproducibility (SUMMARY.md):** `load_seed` seeds torch + python `random` only, not
   numpy. Any numpy randomness in the data path is unpinned. Out of this segment's scope
   but affects determinism of any proposed train-loop test.

### Existing tests — **ONE indirect (smoke only)**
`tests/conftest.py::single_net_checkpoint` runs the *full* `train_single_point(args)` with
`--debug` for `steps:11`, and `test_new_models.py::test_single_checkpoints_exist` asserts
`best.pth` and `checkpoint_e1_s10.pth` exist. This **exercises the loop/checkpoint/val/save
mechanics end-to-end** — but with the phase_input test config (NOT the windowed-beamformer
production model), no loss-value assertion, no resume test, and `val_every:3 /
checkpoint_every:10` (test values, not production). No test for: resume-from-checkpoint
correctness, scheduler stepping, best.pth-improves-only logic, or eval-mode restoration.

### Proposed tests (gaps)
- **`test_resume_restores_step_epoch_and_lr`** (`tests/test_v5inference.py` or new
  `tests/test_train_loop.py`): run `train_single_point` for N steps, then run again with
  `--resume`; assert the resumed run starts at the saved `step`/`epoch` and that
  `scheduler.get_last_lr()` matches the pre-checkpoint value. **Catches** a resume
  regression in `load_checkpoint`/`set_epoch_and_start_iteration`.
- **`test_best_pth_only_saved_on_improvement`**: drive `train_single_point` (or just the
  `save_on` branch via a small harness) with a monotonically *increasing* val loss after
  the first val; assert `best.pth` mtime does not change after the first save. **Catches**
  a flipped comparison at `:1277-1281`.
- **`test_eval_mode_restored_after_val`** (white-box): after a val call inside the loop,
  assert `m.training is True` before the optimizer step. **Catches** removal of `m.train()`
  at `:1297` (which would silently disable dropout/aug for the rest of training).
- **`test_train_smoke_windowed_beamformer_config`**: add a *small* test model config that
  mirrors production (`windows_stats_net` block, `windowed_beamformer:true`,
  `beamformer_input:false`, all 5 scalar inputs) and run `train_single_point` for a few
  steps on the fake circle dataset; assert it completes and `best.pth` exists. **Catches**
  any break in the production model path that the existing phase_input smoke test cannot
  see (this is the single highest-value gap-filler — it would have caught a broken 68→12
  or 18→65 contract).

---

## Summary table

| SEG | Correctness | Existing test | Top gap | Highest-value proposed test |
|---|---|---|---|---|
| 7 AllWindowsStatsNet | CORRECT (68→12, aug train-only, guard ok); caveats: hardcoded `/500`, shrink/dropout `elif`-exclusive | **none** (test config has no `windows_stats_net`) | whole 68→12 windowed path untested | `test_allwindowsstatsnet_forward_contract` (assert `(B,1,12)`) |
| 8 PrepareInput + FFNN | CORRECT (18-dim, abs→L1 over dim2, train-only input_dropout); KNOWN_ISSUES #34 vehicle_type in-place | **none direct** (inference-consistency test uses phase_input variant) | 18-dim assembly + L1-norm contract untested in prod shape | `test_singlepoint_forward_shape_and_l1norm` (`(B,1,65)`, sum=1) |
| 9 mse_loss | CORRECT (SSE over dim2, mean over rows) | **none** | no value-level test | `test_mse_loss_handcomputed` (== 0.75 on tiny input) |
| 10 load_optimizer | CORRECT for step; **cosine branch raises UnboundLocalError (KNOWN_ISSUES #1, not hit here)** | **none** | cosine crash unpinned; StepLR construction untested | `test_load_optimizer_cosine_raises` (regression for #1) |
| 11 train loop | CORRECT mechanics (single_loss backprop, epoch-stepped StepLR, eval/train restore, resume) | **1 e2e smoke** (`single_net_checkpoint` → checkpoints_exist), no value/resume asserts, wrong model variant | no resume / best-only / prod-model-path test | `test_train_smoke_windowed_beamformer_config` (prod-shaped smoke) |
