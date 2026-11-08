# 04 · Training & Inference — overview

How SPF learns the map from RF features (beamformer spectrum + phase stats) to a calibrated
angle-of-arrival distribution, and how trained checkpoints are run. Depends on
[`03_datasets`](../03_datasets/overview.md) for inputs.

## Entry point

```bash
python spf/scripts/train_single_point.py -c <model_config.yaml> \
    [--output DIR] [--resume] [--resume-from CKPT] [--debug] [--val-and-exit] \
    [--steps N] [--name WANDB_NAME]
```

`train_single_point()` (`spf/scripts/train_single_point.py:1033`) loads the YAML config, builds
train/val dataloaders over many `.zarr` files (`ConcatDataset` + the collation from
[`03`](../03_datasets/overview.md)), constructs the model, and runs the train/val loop with
checkpointing and (optional) wandb logging. `train_utils.py` holds config loading and
`global_config_to_keys_used()` (which dataset keys a given config requires).

## Model architectures (`spf/model_training_and_inference/models/single_point_networks.py`)

Selected by `model.name` in the config; registry in `load_model()` (`train_single_point.py:513`):

| `model.name` | Class | Idea | Output keys |
|---|---|---|---|
| `beamformer` | `SinglePointWithBeamformer` | One receiver → `P(θ)` via an FFNN over selected features. | `single` |
| `pairedbeamformer` | `PairedSinglePointWithBeamformer` | Fuse **both receivers** of a craft (rotate one into craft frame, then FFNN). | `single`, `paired` |
| `multipairedbeamformer` | `PairedMultiPointWithBeamformer` | Add a **transformer** over multiple snapshots (temporal fusion) + direct-angle head. | +`multipaired`, `multipaired_direct` |
| `trajmultipairedbeamformer` | `TrajPairedMultiPointWithBeamformer` | As above + predict emitter **xy trajectory**. | +trajectory/xy |

Shared building block: `PrepareInput` concatenates the enabled input features (beamformer,
empirical, phase stats, rx spacing, frequency, gains, vehicle-type & SDR-device embeddings) →
an `FFNN` of residual `BasicBlock`s. The network output is an L1-normalized **distribution over
`output_ntheta` angle bins**, not a scalar angle.

## Config schema (the important fields)

```yaml
global:                # discretization + which inputs are on
  nthetas: 65
  beamformer_input: true        # + empirical_input / phase_input / rx_spacing_input /
  frequency_input: true         #   gains_input / vehicle_type_input / sdr_device_type_input ...
datasets:
  train_paths: [glob...]        # val_paths or val_holdout_fraction
  batch_size: 256
  precompute_cache: <yarr dir>
  segmentation_version: 3.7
  scatter: continuous|onehot    # target distribution shape
  sigma: 0.1                    # Gaussian blur σ for continuous targets
  train_snapshots_per_session: 1   # >1 for multi/transformer models
model:
  name: beamformer|pairedbeamformer|multipairedbeamformer|trajmultipairedbeamformer
  hidden: 1024
  depth: 4
  norm: layer|batch
  dropout: 0.2
  output_ntheta: 65
  transformer: { d_model, n_heads, d_hid, n_layers, skip_connection }   # multi/traj only
optim:
  learning_rate: 1.0e-4
  loss: mse|discrete
  save_on: val/single_loss      # metric that selects the best checkpoint
logger:
  name: wandb|simple
  project: <name>
```

Representative configs live in `model_configs/` (200+; e.g. `apr17_single_3p6_thinner.yaml`).
**Note** `global.nthetas` and `model.output_ntheta` are two separate discretizations.

## Targets & losses

- **Targets** (`target_from_scatter*`): ground-truth θ → a distribution over bins, either
  Gaussian-blurred (`scatter: continuous`, width `sigma`/`scatter_k`) or one-hot smoothed.
  Optional symmetry handling for the front/back ambiguity.
- **Losses**: `mse_loss` (sum-sq over bins) or `discrete_loss` (negative dot product). Training
  optimizes a **weighted sum of auxiliary losses**: `single`, target-aligned/craft/rand-rotation
  variants, `paired` (after a `head_start` epoch), `multipaired`, `multipaired_direct` (weight
  ~1/30), `multipaired_tx_pos` (if predicting xy), plus a small phase regularizer.

## Inference

`single_point_networks_inference.py::load_model_and_config_from_config_fn_and_checkpoint()`
loads config + checkpoint, sets `eval()`, runs the model on a collated batch, and reshapes
`(sessions·radios, snapshots, ntheta)` → `(sessions, radios, snapshots, ntheta)`. Checkpoints
store `model_state_dict`, optimizer/scheduler state, `step`/`epoch`, and the `config`.

**Inference cache** (`create_inference_cache.py`) precomputes model outputs per dataset to a
`.npz`, keyed by `dataset / seg_version / checkpoint_md5 / config_md5`. This is what the **NN
filter variants** in [`05`](../05_filters/overview.md) consume instead of re-running the model.

## Baselines & evaluation

- **Empirical distribution** (`create_empirical_p_dist.py`): a data-built lookup `P(θ|φ)` (with
  symmetry rules). Used as a model input feature **and** as the likelihood in the non-NN filters.
- **Geometric baseline** (`baseline_algorithm.py`): peak-pick the beamformer, turn peaks into
  bearing lines, intersect across receivers to localize — a non-learned comparison point.
- **Metrics** (`segmentation_metrics.py`): phase error std, valid-prediction ratio; validation
  runs every `val_every` steps and checkpoints best-on-`save_on`.

## Known issues / tech debt (spot-verified)

- 🧪 **Unused model classes** — `CrossAttentionNet`, `Separate1D2DNet`, `TransformerTimeModel`,
  `SinglePointPassThrough` are defined but not in the active `load_model` paths.
- **Config redundancy** — `nthetas` vs `output_ntheta`; several overlapping snapshot-stride knobs
  (`snapshots_stride`, `snapshots_adjacent_stride`, `val_*`, `random_*`).
- **Magic loss weights** — e.g. `multipaired_direct` weighted 1/30, phase reg 0.01; rationale
  not documented in code.
- **Commented-out preload pool** in `train_single_point.py` — dead code around the dataloader.

## Known issues

Verified, training-relevant entries from the central [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md)
(read-only audit; nothing fixed). Severities are the **current** triaged values:

- **#1 🟡 P2** — `load_optimizer` cosine branch is a bare `pass`, so `train_scheduler` is never
  bound and `return optimizer, train_scheduler` raises `UnboundLocalError`. Downgraded from P0:
  **only 1 of 164 configs** sets `scheduler: cosine` (`dec3_…_bigger_4x.yaml:75`, abandoned, body
  commented); default is `step`. Fail-fast at optimizer build, no data loss. (`train_single_point.py:389`)
- **#25 🟡 P2** — `baseline_algorithm` `steps==-1` default → `np.arange(-1)` empty loop (sole caller
  always passes `steps`, masking it). (`baseline_algorithm.py:27`)
- **#26 🟡 P2** — `get_inference_on_ds_noexceptions` never returns the inner result (benign — the
  caller discards it; used only for the cache side-effect). (`single_point_networks_inference.py:161`)
- **#27 🟡 P2** — `check_phi_error.py` hardcodes `segmentation_version=3.5`, ignoring its
  `--segmentation-version` CLI arg. (`check_phi_error.py:42`)
- **#34 🟡 P2** — `PrepareInput.prepare_input` zeroes batch tensors **in place** during training
  dropout via aliased views — `weighted_windows_stats` (`:508-511`) **and** `vehicle_type`
  (`:528-530`, an unsqueeze view). Latent today (batch fresh per step) but fragile.
  (`single_point_networks.py:508-511, 528-530`)
- **#36 🟡 P2** — `global_config_to_keys_used` reads non-defaulted flags (`beamformer_input`,
  `phase_input`) → `KeyError` if a config omits them (`load_defaults` only sets
  `windowed_beamformer_input`). Latent — real configs set them. (`train_utils.py`)

## Planned leaf pages

- `training.md` — entry point, dataloader construction, the loop, checkpointing/resume.
  **DONE** → [`reference/spf/scripts/train_single_point.py`](../reference/spf/scripts/train_single_point.md)
  (training driver, `load_model`/`load_optimizer`, dataloader split modes, checkpoint
  save/partial-load, `global_config_to_keys_used`).
- `model_architectures.md` — each network's inputs/outputs/layers in detail. **DONE** →
  [`reference/spf/.../single_point_networks.py`](../reference/spf/model_training_and_inference/single_point_networks.md)
  (the 4 registered networks' forward contracts, `PrepareInput`, `FFNN`/`BasicBlock`, unused classes).
- `config_schema.md` — every field, with a fully annotated example. _(unwritten; current source of
  truth: [`reference/_inventory/training.md`](../reference/_inventory/training.md))_
- `losses_and_targets.md` — scatter targets + the auxiliary-loss stack. **DONE** → see the
  **Loss assembly** + `target_from_scatter*` sections of
  [`reference/spf/scripts/train_single_point.py`](../reference/spf/scripts/train_single_point.md).
- `inference_and_cache.md` — checkpoint loading + inference-cache layout/keys. _(unwritten; current
  source of truth: [`reference/_inventory/training.md`](../reference/_inventory/training.md))_
- `baseline_and_empirical.md` — empirical `P(θ|φ)`, geometric baseline, metrics. _(unwritten; current
  source of truth: [`reference/_inventory/training.md`](../reference/_inventory/training.md))_
