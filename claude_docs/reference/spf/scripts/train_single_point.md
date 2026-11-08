# `spf/scripts/train_single_point.py` — Training loop, loss & target machinery

Reference doc produced via the Pass 1→3 method in
[`DOCUMENTATION_GUIDE.md`](../../../DOCUMENTATION_GUIDE.md). Read-only audit; no source changed.
Every non-trivial claim cites `file:line`. Units/frames link
[`conventions.md`](../../../00_concepts/conventions.md).

This is the **main offline training entrypoint** for the SPF single-point / paired / multi-paired
beamformer models. It also hosts the **model registry** (`load_model`), the **optimizer factory**
(`load_optimizer`), the **soft-label target builders** (`target_from_scatter*`), the **loss
functions** (`mse_loss`, `discrete_loss`), and the **composite loss assembler** (`compute_loss`).
The companion module `train_utils.py` holds config defaulting + the canonical dataset-key list
(`global_config_to_keys_used`).

Status legend: ✅ Active · ⚠️ caveat/smell · 🔴 broken (verified) · 🧪 experimental/unused.

---

## Loss assembly (the heart of the trainer)

`compute_loss` (`train_single_point.py:735`) builds a Python scalar `loss` (`:748`) by adding a
small number of *weighted* terms, while also computing many **auxiliary `no_grad` metrics** that
are logged but never backpropagated. Only the terms in the table below feed `loss_d["loss"]`
(`:991`), which is what `train_single_point` backwards through (`scaler.scale(loss_d["loss"]).backward()`,
`:1323`).

| Loss term (key) | Weight | Source line | When active | Target builder | Backprop? |
|---|---|---|---|---|---|
| `single_loss` | **1.0** | `:868`, added `:871` | `single=True and "single" in output` (`:757`) — every model that emits `"single"` (all but pure multipaired? see note) | `scatter_fn(..., single=True)` on `y_rad` (`:859`) | ✅ yes |
| `single_all_windows_phi_loss` | **0.01** | `:874`, added `:877` | only if `"output_phi" in output` (`:873`) — i.e. `AllWindowsStatsNet` with `output_phi` head | n/a — MSE of wrapped `output_phi − y_phi` | ✅ yes |
| `paired_loss` | **1.0** | `:911`, added `:912` | `paired and "paired" in output` (`:910`) — `pairedbeamformer` (also emitted by multipaired models) | `scatter_fn` on `craft_y_rad` / `craft_y_rad_binned` (`:902`) | ✅ yes |
| `multipaired_loss` | **1.0** | `:938`, added `:939` | `paired and "multipaired" in output` (`:937`) — `multipairedbeamformer`, `trajmultipairedbeamformer` | same `paired_target` (`:902`) | ✅ yes |
| `multipaired_tx_pos_loss` | **0.1** | `:956`, added `:959` | `"multipaired_tx_pos" in output` (`:952`) — `trajmultipairedbeamformer` with `pred_xy: True` | MSE vs `tx_pos_xy − rx_pos_xy` (relative position) | ✅ yes |
| `multipaired_direct_loss` | **1/30** | `:962`, added `:971` | `direct_loss=True` arg (`:961`), passed from `optim.direct_loss` | MSE of wrapped `multipaired_direct − craft_y_rad` (rad) | ✅ yes |

**`/30` magic constant** (`:972`): comment says `# TODO constant to keep losses balanced` — it
re-scales the radian-MSE direct term down to the magnitude of the distribution losses. ⚠️
Hand-tuned, untested, and *currently dead in every checked-in config* (both `multipaired*`
configs set `direct_loss: False`, see "Configs" below). Likewise `tx_pos 0.1` is only reachable
via the single `trajmultipairedbeamformer` `pred_xy: True` config. The `phi 0.01` weight is only
active when an `output_phi` head exists.

**Auxiliary `no_grad` metrics (logged, NOT in `loss`):** `single_target_aligned_loss` (`:783`,
rotate pred into craft frame then zero-align), `single_craft_loss` (`:822`), `single_plus_rand_loss`
(`:845`, random-rotation consistency), `single_loss_old` (`:857`, un-`single` target),
`uniform_loss_{single,paired,multipaired}` (`:879/:914/:941`, a uniform-distribution baseline to
compare the model against). These exist purely for wandb diagnostics. ⚠️ They roughly *double* the
forward compute of `compute_loss` via repeated `scatter_fn` + `rotate_dist` calls.

`loss_fn` is selected once in `train_single_point` from `optim.loss`: `"mse"`→`mse_loss`,
`"discrete"`→`discrete_loss` (`:1176-1181`). `scatter_fn` from `datasets.scatter`:
`"continuous"`→`target_from_scatter`, `"onehot"`→`target_from_scatter_binned` (`:1184-1189`).

---

## How configs & datasets flow in (Pass 2 data handoff)

1. `train_single_point(args)` loads `config = load_config_from_fn(args.config)` (`:1035`), which
   YAML-loads then applies `load_defaults` (`train_utils.py:6-9`). Config has four sub-dicts:
   `global`, `datasets`, `model`, `optim` (+ `logger`, `args`).
2. `load_model(config["model"], config["global"])` (`:1123`) dispatches by `model["name"]`.
3. `load_dataloaders(config["datasets"], config["optim"], config["global"], config["model"])`
   (`:1165`) builds train/val/alternate-val dataloaders. The **collate key list** is
   `global_config_to_keys_used(global_config)` (`:323`), the single source of truth for which
   dataset fields are materialized into each batch. `"empirical"` is force-appended (`:324`).
4. Each batch is a custom object with `.to(device)` (`:1298`) and dict-like access; `compute_loss`
   reads keys such as `y_rad`, `craft_y_rad`, `y_rad_binned`, `craft_y_rad_binned`, `y_phi`,
   `rx_theta_in_pis`, `rx_pos_xy`, `tx_pos_xy`, `empirical` — all of which must be present per
   `global_config_to_keys_used`.

**Required-but-not-defaulted config keys** (KeyError if absent — `load_defaults` does *not* inject
them): `optim.head_start`, `optim.direct_loss`, `optim.loss`, `optim.amp`, `optim.epochs`,
`optim.device`, `optim.dtype`, `global.phase_input`, `global.beamformer_input`, `global.rx_spacing_input`,
`datasets.sigma`, `datasets.scatter_k`, `datasets.scatter`, `datasets.shuffle`,
`datasets.batch_size`, `datasets.snapshots_adjacent_stride`. (Verified against `load_defaults`
`train_utils.py:24-64` — these names are absent there.)

---

## Pass 1 side-effect summary (I/O / GPU / randomness)

- **File I/O / checkpoints:** `save_model` writes `<prefix>/<ckpt>.pth` + `config.yml` and **copies
  `empirical_data_fn` → `empirical_dist.pkl`** (`:431-446`). `train_single_point` writes
  `config.yml` up front (`:1104`), `os.makedirs(... exist_ok=args.resume)` and **`sys.exit(1)` if
  the output dir already exists** (`:1068-1073`). `load_checkpoint` `torch.load(map_location=cpu)`
  (`:453`).
- **wandb (network):** `WNBLogger.__init__` calls `wandb.init` (`:701`); `.log` calls `wandb.log`
  (`:725`). Chosen when `logger.name == "wandb"` and not `--debug` (`:1093-1096`).
- **GPU:** model + batches moved to `config.optim.device`; AMP via `torch.autocast` +
  `torch.GradScaler` (`:1169`, `:1301`). `torch.set_float32_matmul_precision("high")` (`:1121`).
- **Randomness:** `load_seed` seeds `torch` + `random` (`:45-47`), called **three times**
  (`:1075,:1135,:1163`). Dataset shuffle, `random_snapshot_size` truncation (`:1200`), and the
  `single_plus_rand_loss` random rotation (`:824`) all consume RNG. `numpy` seed is *not* set by
  `load_seed`. ⚠️
- **matplotlib:** `compute_loss` builds a 3×2 figure when `plot=True` (`:752`); loggers
  `plt.close` it after logging (`:678`,`:731`).

---

## Function reference

### `train_single_point(args)` · `train_single_point.py:1033` · ✅ Active

**Summary.** The full training driver: load+default config, optional resume, build model /
optimizer / dataloaders, then an epoch loop that periodically validates+checkpoints and otherwise
does an AMP train step. Returns `None`; everything is side effects (files, wandb, GPU, RNG).

**Signature**
- Inputs: `args` (argparse Namespace from `get_parser_filter`) — `.config` (path, required),
  `.output`, `.resume`/`.resume_from`, `.steps`, `.debug`, `.val`, `.val_and_exit`, `.name`.
- Returns: `None`. Early-returns on `--val-and-exit` (`:1294`) and when `step >= optim.steps`
  (`:1358`).
- Side effects: see Pass-1 summary above (files, wandb, GPU, RNG, `sys.exit`).

**Control flow worth knowing**
- Resume (`--resume`): globs `<output>/*.pth`, picks highest `_s<step>` (excluding `best.pth`),
  backs up `config.yml`, reuses prior `run_id` (`:1045-1061`). `--resume-from` is mutually
  exclusive (`assert args.resume_from is None`, `:1046`).
- Assertions (`:1076-1086`): `flip` ⇒ `model.name == "beamformer"`; `multipaired*` ⇒
  `train/val_snapshots_per_session > 1`; otherwise `random_snapshot_size is False`.
- Val cadence: `step % optim.val_every == 0` (`:1214`) unless just resumed; runs main val + each
  alternate-val dataloader; saves `best.pth` keyed on `optim.save_on` (`:1275-1292`).
- Train step: `optimizer.zero_grad` → autocast forward → `compute_loss` → `scaler.scale(loss).backward`
  → `scaler.step` → `scaler.update` (`:1300-1328`). `--debug` asserts `loss.isfinite()` (`:1325`).
- `scheduler.step()` is called **once per epoch** (`:1360`), and `scheduler.get_last_lr()` is read
  every step (`:1333`) — so a broken scheduler crashes immediately (see `load_optimizer`).

**Tested by.** End-to-end through `tests/conftest.py` fixtures `single_net_checkpoint`
(`tests/conftest.py:255`) and `paired_net_checkpoint_using_single_checkpoint` (`:286`), each calling
`train_single_point(args)` with `--debug`. These fixtures are consumed by `tests/test_new_models.py`,
`tests/test_v5inference.py`, `tests/test_partial.py`, `tests/test_particle_filter.py`. So the
`beamformer` + `pairedbeamformer` happy paths (mse loss, continuous scatter, step scheduler) are
exercised; `multipaired*`, `cosine`, `discrete`, `onehot`, and `wandb` paths are **not**.

**Analysis.**
- *Correctness:* the step/val/checkpoint flow is internally consistent; `just_loaded_checkpoint`
  guards re-validating immediately after resume (`:1215`).
- *Issues:* `losses["epoch"]` is appended as `step / len(train_dataloader)` (`:1337`) while
  `val_losses["epoch"]` is the real epoch int — minor inconsistency, log-only. The `--debug`
  finite-check is only on the debug path; a non-finite loss in a real run is silently scaled
  (relies on `GradScaler` to skip). ⚠️ unverified impact.
- *Test gap:* no test covers resume, alternate-val, multipaired, or the optimizer/scheduler save
  round-trip with a non-step scheduler.

---

### `load_model(model_config, global_config)` · `train_single_point.py:513` · ✅ Active

**Summary.** The model-name registry. Dispatches `model_config["name"]` to one of five classes;
raises bare `ValueError` (`:525`) on anything else.

| `name` | Class | Line |
|---|---|---|
| `pass_through` | `SinglePointPassThrough` | `:514` |
| `beamformer` | `SinglePointWithBeamformer` | `:516` |
| `pairedbeamformer` | `PairedSinglePointWithBeamformer` | `:518` |
| `multipairedbeamformer` | `PairedMultiPointWithBeamformer` | `:520` |
| `trajmultipairedbeamformer` | `TrajPairedMultiPointWithBeamformer` | `:522` |

**Contract.** Pre: `model_config["name"]` present and one of the five strings. Post: returns an
unmoved `nn.Module` (caller does `.to(device)` at `:1123`). Other model classes
(`AllWindowsStatsNet` conv variants, `SignalMatrixNet`) are **not** registered here — they are only
reachable as sub-modules. (Confirms inventory note `training.md:67`.)

**Tested by.** Indirectly via the fixtures (`beamformer`, `pairedbeamformer` only).
**Issue:** bare `raise ValueError` with no message (`:525`) — unhelpful on a typo'd name; low severity.

---

### `load_optimizer(optim_config, params)` · `train_single_point.py:377` · 🔴 Broken (cosine)

**Summary.** Builds an `AdamW` (`:378`) and a scheduler. Only the `"step"` branch
(`StepLR`, gamma=0.5, `:383-388`) actually assigns `train_scheduler`.

**🔴 Verified bug (KNOWN_ISSUES #1).** When `optim_config["scheduler"] == "cosine"` the branch body
is just `pass` (`:389-393`), so `train_scheduler` is never bound and `return optimizer,
train_scheduler` (`:397`) raises `UnboundLocalError`. **Reachable via a checked-in config:**
`model_configs/dec3_single_config_sigma0p05_rerun_3p4_windows_beamformerwindow_nophase_nobeam_noempirical_nospacing_normalized_fix_bigger_4x.yaml:75` sets `scheduler: cosine` (verified). The
default is `"step"` (`train_utils.py:33`), so the bug only triggers when a config explicitly opts
into cosine. The `else` branch raises `ValueError` for any other string (`:394-395`).
Because `train_single_point` reads `scheduler.get_last_lr()` every step (`:1333`) and `scheduler.step()`
each epoch (`:1360`), the crash is at optimizer-construction time — training never starts with that
config.

**Tested by.** Step branch only, via fixtures (test configs omit `scheduler:` ⇒ default `"step"`).
The cosine branch has **no test**.

---

### `target_from_scatter(batch, y_rad, y_rad_binned, sigma, k, target_ntheta, single=False)` · `train_single_point.py:539` · ✅ Active ("continuous")

**Summary.** Builds a **soft-label distribution** over `target_ntheta` angle bins by placing a
Gaussian (in radians, width `sigma`) centered at `y_rad`, with **wrap-around padding** so the
distribution's tails fold back across the ±π seam. Optionally (`single=True`) averages in the
up/down-flipped target to bake in the array's front/back symmetry.

**Signature**
- Inputs: `y_rad` (rad, **must be in [−π, π]** — asserted `:542`), shape `(B, S, 1)`; `sigma`
  (rad, Gaussian width); `target_ntheta` (**must be odd** — asserted `:548`); `k`, `y_rad_binned`
  accepted but **unused** in this variant. `single` toggles the symmetry fold.
- Returns: target Tensor `(B, S, target_ntheta)`, **L1-normalized over the bin axis** (`:558` does
  `normalize(p=1, dim=2)` before the fold; after folding+averaging it remains a normalized
  distribution since both halves are normalized).
- Side effects: none (pure); recurses once when `single=True` (`:565`).

**Internals (verified `:543-575`).**
- `:543` allocates a `torch.zeros(...)` but **discards it** (dead line — result unused). ⚠️ harmless.
- Padding `effective_n = 3*n` (`padding = n`, `:549-551`); builds a symmetric bin-angle grid `m`
  spanning `[−π·(3n/2)/n·… ]` scaled so the central `n` bins map to `[−π, π]` (`:553-556`).
- Gaussian `exp(−((m−y_rad)/sigma)²/2)`, L1-normalized over the padded axis (`:557-558`), then the
  `3n` padded bins are **reshaped to `(B,S,3,n)` and summed over the pad dimension** (`:561`) —
  this is the wrap-fold that returns the tails into range.
- `single=True`: `y_rad_flipped = sign(y_rad)·π − y_rad` (`:564`) is the up/down reflection;
  target ← mean of original and flipped (`:574`).

**Contract.** Pre: `y_rad ∈ [−π,π]`, `target_ntheta` odd. Post: each `(B,S,·)` row sums to ~1
(distribution). Invariant: with `single=True`, the target is symmetric under the up/down flip.

**Tested by.** Indirectly via the `single_net_checkpoint` fixture (which uses `scatter:
continuous`, `target_ntheta=7` odd). No direct unit test of the wrap-fold or the symmetry average.
**Issues:** the discarded `torch.zeros` (`:543`); the odd-`n` assert means even `target_ntheta`
configs crash here (by design). The `k`/`y_rad_binned` params are vestigial (signature parity with
the binned variant).

---

### `target_from_scatter_binned(batch, y_rad, y_rad_binned, sigma, k, target_ntheta, single=False)` · `train_single_point.py:578` · ✅ Active ("onehot")

**Summary.** Hard one-hot target: `scatter` a 1 into the `y_rad_binned` bin (`:585-591`), then
optionally smooth with a 1-D `GaussianBlur((k,1), sigma)` and renormalize (`:592-596`). Uses
`y_rad_binned` (the precomputed bin index) and **ignores `y_rad`/`single`**.

**Signature.** Inputs: `y_rad_binned` (int index Tensor `(B,S,1)`), `sigma`, `k` (blur kernel size).
Returns `(B,S,target_ntheta)` distribution (normalized only if `sigma>0`, else raw one-hot, sum=1).
Side effects: none.

**Contract.** Pre: `y_rad_binned` valid index in `[0, target_ntheta)`. Post: rows sum to 1.
**Issue:** when `sigma == 0` no renorm runs — but a one-hot already sums to 1, so fine. ⚠️ No
out-of-range guard on `y_rad_binned` (relies on dataset). **No test** (checked-in test config uses
`continuous`).

---

### `mse_loss(output, target)` · `train_single_point.py:608` · ✅ Active

`((output − target)**2).sum(axis=2).mean()` — sum-of-squared-error over the bin/θ axis, mean over
batch×session. Pre: `output`/`target` same shape, axis-2 = distribution bins. Post: non-negative
scalar. No wraparound handling (operates on already-binned distributions, so OK). Exercised by the
fixtures (test configs set `loss: mse`).

### `discrete_loss(output, target)` · `train_single_point.py:612` · ✅ Active (untested)

`-(target * output).sum(axis=2).mean()` — negative dot product (a soft cross-entropy-ish score;
minimized when `output` mass aligns with `target` mass). Commented-out variants (`sqrt`, `log`)
above it (`:613,615`). Pre: both normalized distributions for the sign/scale to be meaningful.
**No checked-in config sets `loss: discrete`** and no test covers it. ⚠️ unverified at runtime.

### `compute_loss(...)` · `train_single_point.py:735` · ⚠️ Active (large)

Documented above under **Loss assembly**. Returns `(loss_d: dict, fig)`. The `loss_d["loss"]`
entry (`:991`) is the only backward-driving value. `training` arg is accepted but **never read**
inside the body (⚠️ dead parameter — verified by reading `:735-994`). `plot=True` adds a matplotlib
figure side effect. Many `no_grad` aux metrics (see above). Tested only for `single`/`paired`
(beamformer/pairedbeamformer); the multipaired/direct/tx_pos branches are **untested**.

---

### `load_dataloaders(...)` · `train_single_point.py:112` · ⚠️ Active (branchy)

**Summary.** Builds `(train_dl, val_dl, alternate_val_dls: dict)`. Resolves train/val file lists by
glob, then picks among four split modes (`:141-164`): (a) both single `.txt` manifests →
`load_dataset_manifest_with_sampling`; (b) `train_on_val` → val=train; (c) explicit `val_paths`;
(d) holdout last `val_holdout_fraction`. Builds `v5spfdataset`s (best-effort, failures filtered out,
`:222-258`), `ConcatDataset`s, optional `Subset` subsampling, and per-`(device:emitter:wavelength:…)`
**alternate val** dataloaders (`:273-281`). Collate keys come from `global_config_to_keys_used`
(`:323`) with `"empirical"` force-added (`:324-325`). Uses `StatefulBatchsampler` for resumable
shuffling (`:317`).

**Side effects:** heavy dataset/zarr I/O, logging, RNG (shuffle), `assert len>0` on paths
(`:261-262`). **Issues:** none in the split logic. _(Correction from the verification wave: an
earlier draft claimed the split branches had no final `else` → `UnboundLocalError`; this was
**REFUTED** — there **is** a final `else` at `:158-164` that binds both `train_paths`/`val_paths`.
No bug.)_

### `worker_init_fn`, `load_seed`, `load_single_dataset_using_configs`, `load_dataset_manifest_with_sampling` · ✅ Active helpers

- `worker_init_fn` (`:36`) re-inits nested `v5spfdataset`s per DataLoader worker (opens zarr per
  worker) — required for multi-worker correctness.
- `load_seed` (`:45`) seeds torch+random only (⚠️ not numpy).
- `load_single_dataset_using_configs` (`:75`) the single `v5spfdataset` factory threading the four
  config dicts + kwargs.
- `load_dataset_manifest_with_sampling` (`:96`) reads a manifest, duplicating each path by its
  optional integer sampling count; raises on malformed lines.

### `save_model` / `load_checkpoint` · `train_single_point.py:416 / 449` · ✅ / ⚠️

`save_model` writes `.pth` + `config.yml` + copies `empirical_dist.pkl` (`:431-446`).
`load_checkpoint` supports **partial loads**: `model.load_single` freezes & loads only
`single_radio_net` and **returns epoch/step 0** (`:473-479`); `model.load_paired` does the same for
`multi_radio_net` (`:480-487`); otherwise full state-dict load incl. optimizer+scheduler
(`:498-510`). ⚠️ partial-load resets epoch/step to 0 even mid-training (by design for transfer).
Exercised by the paired-from-single fixture (`tests/conftest.py:277`).

### Loggers `SimpleLogger` / `WNBLogger` · `train_single_point.py:658 / 683` · ✅

Both `.log(data, step, prefix)` mean-reduce list-valued losses (`:667/:715`), skip `"plot"` keys
for the scalar dict but pass figures through (wandb only) and `plt.close` them. `WNBLogger.__init__`
opens a wandb run (network). `SimpleLogger` is console-only and is what `--debug`/tests use.

### `new_log`, `nn_checksum`, `model_checksum`, `run_val_on_dataloader` · ✅

`new_log` (`:634`) the fresh per-window loss-accumulator dict (the canonical metric-name set).
`run_val_on_dataloader` (`:997`) runs `compute_loss(training=False)` over a dataloader, accumulates
`.item()`s, plots only the first batch. `nn_checksum`/`model_checksum` are debug aids
(mean-abs-param logging).

### Unused / experimental in this module · 🧪

`preload_dataset` (`:57`, only in commented-out preload), `FrozenModule` (`:400`, defined but
`load_checkpoint` freezes inline instead), `uniform_preds` (`:528`), `weighted_beamformer` (`:600`).
(Confirms inventory `training.md:166`.)

---

## `train_utils.py` reference

### `global_config_to_keys_used(global_config)` · `train_utils.py:67` · ✅ Active (the data handoff)

**Summary.** The single source of truth for which dataset/batch keys must be collated for a given
config. Returns a base list of ~14 always-needed keys (`:68-87`: `all_windows_stats`, `rx_lo`,
`rx_wavelength_spacing`, `y_rad`, `y_phi`, `craft_y_rad`, `system_timestamp`, `y_rad_binned`,
`craft_y_rad_binned`, `rx_pos_xy`, `tx_pos_xy`, `rx_theta_in_pis`, `rx_heading_in_pis`), then
conditionally appends per global flag (`:88-105`): `abs_signal_and_phase_diff`
(`signal_matrix_input`), `empirical`, `windowed_beamformer`, `weighted_beamformer`
(`beamformer_input`), `gains`, `vehicle_type`, `sdr_device_type`, `weighted_windows_stats`
(`phase_input`). Accepts `None` (returns base list only).

**Contract / caveat.** ⚠️ The flag block reads `global_config["beamformer_input"]`,
`["gains_input"]`, `["vehicle_type_input"]`, `["phase_input"]` **without** `.get` — these must
exist. `load_defaults` injects `gains_input`/`vehicle_type_input`/`sdr_device_type_input`
(`train_utils.py:26,27,37`) but **not** `beamformer_input` or `phase_input`, so a config missing
those raises `KeyError` here. (Verified: absent from `load_defaults`.) Mirrored/imported by the
inference module — keep in sync. **No direct test**; exercised via `load_dataloaders`.

### `load_config_from_fn` / `load_defaults` / `get_key_or_set_default` · `train_utils.py:6 / 24 / 12` · ✅

`load_config_from_fn` YAML-loads then applies defaults. `get_key_or_set_default` recursively
walks an `a/b/c` path and inserts a default **mutating `d` in place** (`:20-21`). `load_defaults`
(`:24-64`) sets ~22 defaults incl. `optim/scheduler="step"` (`:33` — the reason the cosine bug is
opt-in), `model/output_ntheta = global/nthetas` (`:57-61`), and
`val_snapshots_adjacent_stride = snapshots_adjacent_stride` (`:52-56`). Note `load_config_from_fn`
has unreachable `return None` after the `with` (`:9`) — harmless.

---

## Cluster summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `train_single_point` | ✅ | entrypoint (CLI + conftest) | ✅ via fixtures (beamformer/paired only) | Multipaired/cosine/discrete/wandb paths untested |
| `load_model` | ✅ | 1 (in driver) | indirect (2 of 5 names) | Bare `ValueError`; 3 model names untested |
| `load_optimizer` | 🔴 | 1 | step only | **cosine → UnboundLocalError**, reachable via `dec3_…_bigger_4x.yaml:75` |
| `target_from_scatter` | ✅ | 1 (compute_loss) | indirect | Discarded `torch.zeros` (`:543`); wrap-fold logic untested directly |
| `target_from_scatter_binned` | ✅ | 1 | ❌ none | No checked-in config uses `onehot` |
| `mse_loss` | ✅ | many (compute_loss) | indirect | Fine |
| `discrete_loss` | ✅ | 1 | ❌ none | No config sets `loss: discrete`; runtime-unverified |
| `compute_loss` | ⚠️ | 2 (train + val) | indirect (single/paired) | Many no-grad aux metrics double compute; `training` arg dead; direct/`/30` & tx_pos `0.1` untested |
| `load_dataloaders` | ✅ | 1 | indirect | branchy; split has a final `else` (`:158-164`) — earlier "no-else" claim refuted |
| `global_config_to_keys_used` | ✅ | 2+ (train + inference) | ❌ none | Reads non-defaulted flags (`beamformer_input`, `phase_input`) → KeyError risk |

**Cross-cutting theme.** The trainer's *backward-driving* loss is small and clean (mostly
`single_loss`/`paired_loss`/`multipaired_loss` at weight 1.0, plus three hand-tuned auxiliary
weights — `phi 0.01`, `tx_pos 0.1`, `direct 1/30`), but it is wrapped in a large body of
`no_grad` diagnostic metrics that roughly double `compute_loss` cost. **The three magic weights
and the cosine-scheduler path are all effectively untested**, and two of the three weights
(`direct 1/30`, partly `tx_pos 0.1`) are *dead in every checked-in config* because both
`multipaired*` configs set `direct_loss: False`. The one verified break — `load_optimizer` cosine
— is reachable from a real config but cannot be hit by the test suite (test configs default to
`scheduler: step`).

---

## Open questions / TODO

- _(Resolved by verification wave: the previously-suspected `load_dataloaders` split-mode gap is
  not real — a final `else` at `:158-164` always binds both paths.)_
- The `/30` direct-loss constant (`:972`) is self-flagged `TODO` — unverified that it actually
  balances the terms; no config currently exercises it.
- `compute_loss(training=...)` is passed but unused (`:745`, `:1019`, `:1317`) — confirm whether a
  training-vs-eval branch was intended and dropped.
