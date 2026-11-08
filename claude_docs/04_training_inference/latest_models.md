# Latest trained models (jul8 / 3p7 "thin" lineage)

What the current production models are, what each does, and exactly how they handle **signals**,
**symmetry**, and **preprocessing**. Derived from the actual configs + checkpoints (read-only),
cross-referenced with the verified contract docs in [`../reference/spf/`](../reference/spf/).

Evidence: `model_configs/jul8_*_3p7_thin*_dec15.yaml`, `checkpoints/jul8*/…`,
`spf/dataset/spf_dataset.py` (`render_session`), `spf/model_training_and_inference/models/single_point_networks.py`.

## Which models are "latest"

The most recently trained checkpoints (Dec 2025) are a **two-stage lineage**, not the
transformer/trajectory nets (those exist in code but weren't retrained):

| Stage | Config | `model.name` → class | Output dir | Newest ckpt |
|---|---|---|---|---|
| 1 — single radio | `jul8_single_3p7_thin_noblade_dec15.yaml` | `beamformer` → `SinglePointWithBeamformer` | `checkpoints/jul8_dec15/single_3p7_thin_noblade/` | `e28_s1630000` (Dec 17) |
| 2 — paired (both radios) | `jul8_paired_3p7_thin_dec15.yaml` | `pairedbeamformer` → `PairedSinglePointWithBeamformer` | `checkpoints/jul8/paired_3p7_thin_noblade_dec15/` | (loads stage-1 `best.pth`) |

The paired model **starts from the single model**: `model.load_single: true` +
`optim.checkpoint: …/single_3p7_thin_noblade/best.pth`, with the single sub-model `detach: true`
(frozen-ish — gradients don't flow back into it during paired training).

Name decoding: **3p7** = segmentation v3.7 · **thin** = small net (hidden 512, depth 4) · **noblade**
= training split excludes BladeRF data · **nosig** (in the split filename) = no raw IQ loaded ·
**noroverbounce** = excludes the rover "bounce" routine.

## What each model does

- **Stage 1 — `SinglePointWithBeamformer`**: takes **one** receiver's per-window beamformer power
  maps (+ metadata) and outputs a length-`nthetas=65` **distribution over angle-of-arrival θ**
  (array-relative). It cannot resolve the 2-element front/back ambiguity on its own.
- **Stage 2 — `PairedSinglePointWithBeamformer`**: runs stage-1 on **both** receivers of the craft,
  rotates radio-0's estimate into the craft frame, concatenates the two, and an FFNN fuses them
  into a **craft-relative θ** distribution. Fusing the two differently-oriented arrays is what
  resolves the per-radio front/back ambiguity. Saved on `val/paired_loss`.

Both heads are L1-normalized distributions over 65 angle bins; loss is MSE against a smoothed
target distribution (below). Neither predicts XY position (that's the unused `traj*` net).

## Signals — what the network actually consumes

The key fact: **these models do not see raw IQ, phase, or the full-session beamformer.** They see
the **windowed beamformer** plus scalar metadata. From the `global:` block (both configs identical):

| Input flag | Value | Meaning |
|---|---|---|
| `windowed_beamformer_input` | **true** | per-window power-vs-angle maps `(snapshots, n_windows, 65)` — the only signal input |
| `beamformer_input` | false | session-averaged beamformer **not** used |
| `phase_input` | false | weighted window phase stats **not** used |
| `signal_matrix_input` | false | raw complex IQ **not** loaded (hence `nosig` splits) |
| `empirical_input` | false | empirical `P(θ\|φ)` **not** fed to the net (see Symmetry) |
| `rx_spacing_input` | true | antenna spacing d/λ |
| `frequency_input` | true | carrier `rx_lo` |
| `gains_input` | true | per-channel RX gains |
| `vehicle_type_input` | true | wall-array vs rover embedding |
| `sdr_device_type_input` | true | SDR model embedding |

The windowed beamformer is processed by **`AllWindowsStatsNet`** (`windows_stats_net` block), an
8-layer conv/MLP over the window axis (`hidden_channels: 64`, `outputs: 12`, `act: leaky`,
`normalize_windowed_beamformer: false`) with heavy **window-axis augmentation**:
`window_fraction: 0.5`, `window_dropout: 0.25`, `window_shuffle: 0.15`, `window_shrink: 0.5`.
Its 12 outputs are concatenated with the metadata scalars by `PrepareInput` and fed to the core
FFNN (residual `BasicBlock`s, depth 4, hidden 512, `norm: layer`, `input_dropout: 0.1`/`0.2`).

> Provenance of the signal: the windowed beamformer is computed during **segmentation v3.7** from
> the raw IQ (delay-and-sum per 2048-sample window). So the "signal" the model learns from is a
> compressed, pre-beamformed representation, not the antenna samples. See
> [`../reference/spf/dataset/segmentation.md`](../reference/spf/dataset/segmentation.md).

## Symmetry — how the front/back & left/right ambiguities are handled

A 2-element array is ambiguous; the models exploit that with **three** symmetry mechanisms:

1. **Left/right flip augmentation (`flip: true`, stage-1)** — in `render_session`
   (`spf_dataset.py:1357,1387-1389`), 50% of samples negate `ground_truth_theta` and
   `ground_truth_phi` (mirror about boresight) and flip the windowed beamformer / mean-phase
   accordingly. Teaches invariance to the left/right sign.
2. **Front/back double-flip (`double_flip: true`, stage-2)** — `render_session:1390-1395` maps
   `θ → sign(θ)·π − θ` (front angle → its mirrored back angle), φ unchanged. This is the paired
   model's augmentation (it uses `double_flip` instead of `flip`), directly training the front/back
   fold that pairing is meant to resolve.
3. **Loss-side up/down symmetry** — a `flip_up_down` augmentation was **moved into the loss**
   (`render_session:1358` comment); `target_from_scatter` builds the target distribution with an
   optional symmetry fold (see [`../reference/spf/scripts/train_single_point.md`](../reference/spf/scripts/train_single_point.md)).

`empirical_symmetry: true` selects the **symmetrized** empirical `P(θ|φ)` table (`get_empirical_dist`
picks the `"sym"` variant, `spf_dataset.py:241`). **But** because `empirical_input: false`, the
empirical distribution is **loaded but not fed to the network** — it's used as a baseline/diagnostic
comparison, not a model input, for these models. (`empirical_individual_radio: false` → the pooled
`"r"` table.)

> Note: `empirical_data_fn` points at `empirical_dists/full.pkl`. The even-bins empirical off-by-one
> (KNOWN_ISSUES #48) does **not** bite here — `nthetas: 65` is odd.

## Preprocessing

- **Segmentation v3.7** (`segmentation_version: 3.7`, cache `/mnt/md2/cache/precompute_cache_3p7`):
  detrend → windowed delay-and-sum beamform → trimmed circular phase stats → signal/noise mask →
  cached `.yarr`. The windowed beamformer the model consumes comes from here.
- **Target smoothing** (`scatter: continuous`, `sigma: 0.1`): ground-truth θ → a 65-bin
  distribution built **analytically** in radians (`exp(-((m−y)/σ)²/2)`, L1-normed, with ±π-seam
  wrap-fold), the MSE regression target — not a one-hot. ⚠️ `scatter_k: 21` is **a no-op here** —
  the kernel-blur only applies to the unused `scatter: onehot`/binned path
  (`train_single_point.py:592-595`). Verified in the stage-1 audit.
- **No normalization of the windowed beamformer** (`normalize_windowed_beamformer: false`).
- **Snapshots**: `train_snapshots_per_session: 1` (single-snapshot; no temporal/transformer fusion).
- **Data filtering**: trained on `apr17_train_nosig_noroverbounce_noblade.txt` — excludes raw IQ,
  rover-bounce routines, and BladeRF data. `skip_qc: true` (segmentation QC gating off).
- **Subsampling/val**: `val_holdout_fraction: 0.5`, `val_subsample_fraction: 0.5`.

## Training setup

| | Stage 1 (single) | Stage 2 (paired) |
|---|---|---|
| init | from scratch | loads stage-1 `best.pth` (`load_single`, single `detach: true`) |
| loss / save_on | MSE / `val/single_loss` | MSE / `val/paired_loss` |
| lr (→ scheduler) | 1e-4 (from 2e-4), `scheduler_step: 6` | 2e-4, `scheduler_step: 12` |
| weight_decay | 0.001 | 0.0 |
| input_dropout | 0.1 | 0.2 |
| epochs / batch | 100 / 256 | 60 / 256 |
| precision | fp32, no AMP | fp32, no AMP |

`direct_loss: false`, `head_start: 0`, 24 dataloader workers, wandb project
`2024_nov22_single_paired_multi`.

## Caveats worth knowing (from KNOWN_ISSUES)

- **#45 (segmentation drops abutting signal)** touches these models directly — they consume the
  windowed beamformer that segmentation produces; multi-emitter / phase-stepping captures can be
  silently under-segmented.
- **#34 (PrepareInput in-place mutation)** of `weighted_windows_stats` / `vehicle_type` fires only
  with `input_dropout > 0` in training (both configs use it) — latent (batch is fresh per step).
- **#1 cosine scheduler crash** does **not** affect these — they use the default `step` scheduler.
- The raw-IQ flip warning (`render_session:1411` "does not respect flipping") is **moot** here:
  `signal_matrix_input: false`, so `abs_signal_and_phase_diff` is never built.
