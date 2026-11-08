# `single_point_networks.py` — Neural Network Architectures

Reference for `spf/model_training_and_inference/models/single_point_networks.py`.

Documents the **four active networks** in the `load_model` registry
(`spf/scripts/train_single_point.py:513`), the shared `PrepareInput`, the
`FFNN`/`BasicBlock` building blocks, the `SinglePointPassThrough` baseline, and the
**config-only / unused** classes.

Status legend: ✅ Active · ⚠️ caveat/smell · 🔴 verified bug · 🧪 experimental/unused.

Produced by a Pass 1→3 deep dive (DOCUMENTATION_GUIDE.md). Read-only; no source changes.
Cites `file:line`; unconfirmed claims marked ⚠️.

---

## Conventions used throughout

- **B** = batch size (number of (session × radio) rows in a batch).
- **S** = `snapshots_per_session` (sequence length of time snapshots per example).
- **nthetas** = number of discrete angle bins for the beamformer/empirical input
  (commonly 65). `output_ntheta` defaults to `global_config["nthetas"]` via
  `check_and_load_ntheta` (`:555-558`).
- All **discrete distributions are over angles in [-π, π]** (radians), L1-normalized
  over the last dim. See `00_concepts/conventions.md`.
- **Frames:** per-radio single-point estimates are in the **radio frame**; pairing rotates
  them into the **craft frame** by `rx_theta_in_pis · π` (`:1024`). See
  `00_concepts/conventions.md`.
- The model registry (`load_model`, train_single_point.py:513) maps config names →
  classes: `pass_through`→`SinglePointPassThrough`, `beamformer`→`SinglePointWithBeamformer`,
  `pairedbeamformer`→`PairedSinglePointWithBeamformer`,
  `multipairedbeamformer`→`PairedMultiPointWithBeamformer`,
  `trajmultipairedbeamformer`→`TrajPairedMultiPointWithBeamformer`. Any other name → `ValueError`
  (`:525`).

The batch dict keys consumed by these models are materialized by
`global_config_to_keys_used` (`spf/scripts/train_utils.py:67-106`) and collated to tensors of
shape `(B, S, ...)`.

---

## Building blocks

### `BasicBlock(...)`  ·  `single_point_networks.py:30`  ·  ✅ Active

**Summary.** Two-layer residual MLP block: `Linear→Dropout→Norm→Act` twice, then add the
residual and apply activation a final time. Used inside `FFNN` when `block=True`.

**Signature**
- Inputs (ctor): `hidden` (int, feature width), `act` (activation *class*, e.g. `nn.LeakyReLU`),
  `bn=True` (apply norm), `norm="batch"` (`"batch"`→`BatchNorm1d`, `"layer"`→`LayerNorm`,
  else `Identity`), `dropout=0.0`.
- `forward(x)`: `x` shape `(..., hidden)` → returns `(..., hidden)`.
- Side effects: none.

**Analysis.** `:50-52` — `out = act(n1(d1(linear1(x))))`, `out = act(n2(d2(linear2(out))))`,
returns `act(out + identity)`. The final activation-after-add is a smell (post-activation on a
residual sum is unusual) but not a bug. ⚠️ If `norm="batch"` and `x` is 3-D `(B, S, hidden)`,
`BatchNorm1d` expects `(N, C)` or `(N, C, L)` — verify callers feed 2-D when `bn=True, norm="batch"`
(FFNN reshapes? see below). In practice the active configs use `bn=False` so norms are `Identity`.

---

### `FFNN(...)`  ·  `single_point_networks.py:55`  ·  ✅ Active (core building block)

**Summary.** Configurable MLP: input `Linear`, then `depth` stacked layers (each a `BasicBlock`
when `block=True`, else `Linear+Act+Norm`), then output `Linear`. Norm is batch or layer; raises
`ValueError` on any other `norm` (`:108`).

**Signature**
- Inputs (ctor): `inputs`, `depth`, `hidden`, `outputs` (ints); `block` (bool); `bn` (bool, whether to
  use any norm); `norm` (`"batch"`|`"layer"`); `act` (activation class); `dropout=0.0`.
- `forward(x)`: applies `self.ffnn_internal_net` (an `nn.Sequential`). Input `(..., inputs)` →
  output `(..., outputs)`. (No reshape inside; trailing-dim transform.)
- Side effects: none (ctor raises `ValueError` on bad `norm`).

**Called from.** Constructed by `SinglePointWithBeamformer` (`:934`),
`PairedSinglePointWithBeamformer` (`:1007`), `TrajPairedMultiPointWithBeamformer.traj_net`
(`:1229`). Fan-in: 3 active call sites.

**Analysis.** Two ctor branches: `bn==False or norm=="batch"` uses `BatchNorm1d`; `norm=="layer"`
uses `LayerNorm`. ⚠️ When `bn=True, norm="batch"` and input is 3-D `(B,S,hidden)`, `BatchNorm1d`
would error — but `Sequential` applies modules to the whole tensor without reshape. The active
single/paired configs pass `bn=False` (`model_config["bn"]`, see `:944`,`:1016`), so the norm slots
are `Identity` and the 3-D `(B,S,·)` input flows through fine. `traj_net` uses `bn=True, norm="layer"`
(`:1235-1236`) → `LayerNorm(hidden)` which is shape-agnostic over the last dim. ✅

---

## Shared input assembly

### `PrepareInput`  ·  `single_point_networks.py:414`  ·  ✅ Active  ⚠️ in-place mutation

**Summary.** Assembles and normalizes the per-snapshot feature vector fed to the FFNN from a batch
dict. The ctor computes `self.inputs` — the integer width of the assembled feature vector — by
summing the dims of each enabled input source. `prepare_input(batch, additional_inputs)` concatenates
the enabled, scaled feature tensors (plus any `additional_inputs`) along `dim=2`.

**Ctor config (which keys / widths)** — each flag comes from `global_config`, dropout from
`model_config`:
| flag (global_config) | batch key consumed | scaling (in `prepare_input`) | width added to `self.inputs` |
|---|---|---|---|
| `beamformer_input` | `weighted_beamformer` | `v/(2·(mean+0.01))` (`:494-495`) | `nthetas` (`:438`) |
| `beamformer_mag_input` (requires beamformer) | (appends mean) | concat `v_mean` (`:497`) | `+1` (`:440`) |
| `empirical_input` | `empirical` | `(v − 1/nbins)·10` (`:503`) | `nthetas` (`:442`) |
| `phase_input` | `weighted_windows_stats` | passthrough (`:508`) | `+3` (`:444`) |
| `gains_input` | `gains` | `v/120` (`:523`) | `+2` (`:446`) |
| `rx_spacing_input` | `rx_wavelength_spacing` | `v[...,None]·50` (`:513`) | `+1` (`:448`) |
| `frequency_input` | `rx_lo` | `log10(v+1)/20` (`:518`) | `+1` (`:450`) |
| `vehicle_type_input` | `vehicle_type` | `v[...,None]` (`:528`) | `+1` (`:452`) |
| `sdr_device_type_input` | `sdr_device_type` | `v[...,None]/10` (`:533`) | `+1` (`:454`) |

- Returns: `torch.concatenate(inputs + additional_inputs, dim=2)` →
  **`(B, S, self.inputs + Σ additional_inputs_widths)`** (`:538-541`).
- `beamformer_dropout` (`model_config["beamformer_dropout"]`) wraps the beamformer feature in
  `nn.Dropout` (or `Identity`) (`:432-435`,`:498`).

**🔴/⚠️ Side effect — in-place zeroing of batch-tensor views (VERIFIED).** During training
(`self.training`) the method builds `dropout_mask` of shape `(8, B, S)` (`:476-481`) and applies
`v[dropout_mask[i]] = 0` to several feature tensors:
- `weighted_beamformer`-derived `v` (`:500`),
- `empirical`-derived `v` (`:505`),
- `weighted_windows_stats` (`:510`),
- `rx_wavelength_spacing` (`:515`), `rx_lo` (`:521`), `gains` (`:525`), `vehicle_type` (`:530`).

These `v` are derived tensors. For `empirical` (`:503`) and `phase_input` (`:508` —
`v = batch["weighted_windows_stats"]` directly, **no copy**), the assignment can mutate a **view of
the batch tensor**. The `phase_input` case is the clearest hazard: `v` is the batch tensor itself,
so `v[mask]=0` zeroes the batch in place. **Two branches are unsafe (verification correction):**
`weighted_windows_stats` (`phase_input`, `:508`) **and** `vehicle_type` (`:528` — `v =
batch["vehicle_type"][...,None]`, an unsqueeze **view**, mutated at `:530`). The arithmetic branches
(beamformer/empirical/rx_spacing/frequency/gains, `v = batch[...] * c`) genuinely build fresh
tensors and are safe. _(An earlier draft said "only phase_input"; the `vehicle_type` view-mutation
was missed.)_ `sdr_device_type` is **never** dropout-masked (commented out, `:534-535`).
- *Trigger:* training mode with `input_dropout > 0` AND `phase_input` enabled.
- *Severity:* medium (corrupts the batch dict for any downstream reader of
  `weighted_windows_stats`; if the same batch object is reused this contaminates later reads).
- *Confidence:* high (assignment by reference at `:508`,`:510`).

The ctor `assert` (`:455-468`) enforces that `input_dropout` may only be nonzero when **more than
one** input source is enabled (otherwise dropping the sole input is degenerate).

**Tested by.** No direct unit test of `prepare_input`; exercised indirectly through model inference
tests (`tests/test_new_models.py`) which run with `model.eval()` (so the `self.training` dropout path
is **not** covered). The in-place-mutation path has **no test**.

**Open questions.** Does any training config set `input_dropout>0` together with `phase_input`? If so
the in-place mutation is live. ⚠️ needs config audit.

---

## Active network 1 — `SinglePointWithBeamformer`

### `SinglePointWithBeamformer`  ·  `single_point_networks.py:913`  ·  ✅ Active (`"beamformer"`)

**Summary.** Per-radio single-snapshot direction estimator. Runs `PrepareInput` (+ optional
`AllWindowsStatsNet` / `SignalMatrixNet` sub-encoders whose embeddings are appended as
`additional_inputs`), pushes through an `FFNN`, and L1-normalizes the absolute value of the output
into a discrete distribution over `output_ntheta` angle bins.

**Ctor**
- `check_and_load_config` (defaults `output_ntheta`←`nthetas`, mutates `model_config`) (`:916`).
- `self.prepare_input = PrepareInput(model_config, global_config)` (`:917`).
- Optional `self.signal_matrix_net = SignalMatrixNet(**model_config["signal_matrix_net"])` only if
  `"signal_matrix_net" in model_config` (`:923-927`); adds `.outputs` to `additional_inputs` width.
- Optional `self.windows_stats_net = AllWindowsStatsNet(nthetas=…, **model_config["windows_stats_net"])`
  only if `"windows_stats_net" in model_config` (`:928-932`); adds `.outputs`.
- `self.single_point_with_beamformer_ffnn = FFNN(inputs=prepare_input.inputs + additional_inputs,
  depth, hidden, outputs=output_ntheta, block, norm, act=LeakyReLU, dropout, bn)` (`:934-945`).

**forward(batch) — I/O**
- **INPUT** `batch` dict. Keys depend on enabled flags (see `PrepareInput`). Tensors are `(B, S, ...)`.
  If `windows_stats_net`: also `all_windows_stats` (and optionally `windowed_beamformer`). If
  `signal_matrix_net`: also `abs_signal_and_phase_diff` `(B,S,4?,L)`, `rx_wavelength_spacing`, `rx_lo`.
- **OUTPUT** dict:
  - `"single"`: **`(B, S, output_ntheta)`**, L1-normalized over `dim=2` (`:987-993`).
  - `"output_phi"`: present **iff** `windows_stats_net` has `output_phi=True` (`:954-955`); shape
    `(B, S, 1)` (from `AllWindowsStatsNet`).
- **Side effects:** mutates `model_config` (ctor); **in-place normalize math on
  `abs_signal_and_phase_diff`** when `signal_matrix_net` active — `normalize_by[:, :, 2] = torch.pi`
  then `input = batch[...] / normalize_by` (`:962-967`); `normalize_by` is a fresh max-reduction so
  the in-place write is into that fresh tensor, not the batch (✅ safe). PrepareInput's training
  dropout mutation applies (see above).

**Analysis.** The output is forced non-negative via `.abs()` before L1-norm (`:990`), so `"single"`
is always a valid distribution summing to 1. The two sub-encoders are **only constructed if their
config block is present**; the default registered `"beamformer"` configs typically include
`windows_stats_net` (phase encoder) and not `signal_matrix_net`. ⚠️ The line `:969-971`
(`batch["rx_wavelength_spacing"]...expand(...)`) computes an expansion and discards it (dead) — the
real use is at `:976-979`.

**Tested by.** `tests/test_new_models.py::test_inference_single_checkpoint` (and
`_against_ds_inference`) run a trained `"beamformer"` checkpoint and assert dataloader vs
per-example inference agree on `"single"` (`:70`,`:114`). Eval-mode only; no shape/edge unit test.

---

## Active network 2 — `PairedSinglePointWithBeamformer`

### `PairedSinglePointWithBeamformer`  ·  `single_point_networks.py:997`  ·  ✅ Active (`"pairedbeamformer"`)

**Summary.** Fuses two radios viewing the same target. Runs the per-radio
`SinglePointWithBeamformer`, **rotates** each single-radio distribution from radio frame into the
**craft frame** (by `rx_theta_in_pis·π`), concatenates with invariant per-radio features, interleaves
the even/odd radio rows side-by-side, and runs a second `FFNN` to produce a fused paired distribution.

**Ctor**
- `self.single_radio_net = SinglePointWithBeamformer(model_config["single"], global_config)` (`:1001`).
- `self.prepare_input = PrepareInput(model_config["single"], global_config)` (`:1004`) — a *separate*
  PrepareInput producing invariant features.
- `self.detach = model_config.get("detach", True)` (`:1006`) — whether to detach the joined input
  from the single-net's graph.
- `self.paired_single_point_with_beamformer_ffnn = FFNN(inputs=(single.output_ntheta +
  prepare_input.inputs)·2, …, outputs=output_ntheta, …)` (`:1007-1018`). The `·2` is for the two
  stacked radios.

**forward(batch) — I/O**
- **INPUT** `batch` dict. Same keys as `SinglePointWithBeamformer`, **plus** `rx_theta_in_pis`
  `(B, S)` for the craft-frame rotation (`:1024`). **Row ordering matters:** even rows (`[::2]`) and
  odd rows (`[1::2]`) are the two radios of a pair (`:1041`) — B must be even / radio-interleaved.
- **OUTPUT** dict:
  - `"single"`: `(B, S, output_ntheta)` — raw per-radio estimates from the single net (`:1021`,`:1049`).
  - `"paired"`: **`(B, S, output_ntheta)`**, `normalize(abs(x[idxs]), p=1, dim=2)` (`:1052`). `x` is
    computed for `B/2` pairs then `x[idxs]` duplicates each pair's result back to both radio rows
    (`idxs` = `[0,0,1,1,2,2,…]`, `:1046`), so output B matches input B.
- **Side effects:** `detach_or_not` on the joined input (config-gated graph cut, `:1037`); inherits
  PrepareInput / single-net side effects.

**Fusion mechanics (verified).**
1. `single_radio_estimates = single_radio_net(batch)["single"]` (`:1021`), `(B,S,ntheta)`.
2. `rotation_offets = rx_theta_in_pis · π` (`:1024`).
3. `rotate_dist(single_radio_estimates[:,0], torch_pi_norm(rotation_offets)).unsqueeze(1)` (`:1025-1028`)
   — **⚠️ NOTE: only snapshot index 0 is rotated** (`[:, 0]`), then `unsqueeze(1)` restores a length-1
   snapshot axis. So the rotated input has S=1 effectively; `rotate_dist` (`spf/rf.py:1493`) rotates a
   discrete dist over [-π,π] bins by interpolation. The paired path therefore consumes only the first
   snapshot's rotated single estimate. ⚠️ This looks intentional for single-snapshot pairing but is a
   contract worth flagging.
4. `additional_inputs = prepare_input.prepare_input(batch)` (`:1031`) — invariant features `(B,S,inputs)`.
   ⚠️ shape mismatch hazard: rotated estimate is `(B,1,ntheta)` but additional_inputs is `(B,S,inputs)`;
   the `concatenate(..., dim=2)` at `:1033` requires matching dims 0,1 — implies **S==1** at the paired
   stage (consistent with `[:, 0]` above). Confirm callers run paired with S=1. ⚠️ unverified for all
   configs.
5. `joined = concatenate([rotated, additional_inputs], dim=2)`; detached if `self.detach`;
   `concatenate([joined[::2], joined[1::2]], dim=2)` stacks the two radios feature-wise (`:1039-1044`);
   FFNN → `(B/2, S, output_ntheta)`; `x[idxs]` re-expands to B rows.

**Tested by.** `tests/test_new_models.py::test_inference_paired_checkpoint` (`:117-204`): asserts
dataloader vs per-example agreement on both `"single"` and `"paired"`, and that the paired net's
`"single"` matches the standalone single net (weight-sharing/load correctness). Eval-mode only.

---

## Active network 3 — `PairedMultiPointWithBeamformer`

### `PairedMultiPointWithBeamformer`  ·  `single_point_networks.py:1067`  ·  ✅ Active (`"multipairedbeamformer"`)

**Summary.** Adds a **transformer over the snapshot/time sequence** on top of the paired net.
Takes the paired per-snapshot distributions, appends a normalized timestamp (and optionally xy
position), projects to `d_model`, runs a `TransformerEncoder` over the S snapshots, and produces a
refined distribution (`multipaired`, optionally a skip connection on the paired input) plus a scalar
direct angle (`multipaired_direct`).

**Ctor**
- `self.multi_radio_net = PairedSinglePointWithBeamformer(model_config["paired"], global_config)` (`:1071`).
- `transformer_config` → `d_model`, `n_heads`, `d_hid`, `n_layers`, `dropout`, `skip_connection`
  (default `True`, `:1077`). `use_xy` default `False` (`:1078`).
- `TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_hid, dropout, activation="gelu",
  batch_first=True)` × `n_layers`, with final `LayerNorm(d_model)` (`:1081-1093`).
- `input_net = Linear(input_size, d_model)` where `input_size = paired.output_ntheta + 1` (+2 if
  `use_xy`) (`:1094-1101`).
- `output_net = Linear(d_model, output_ntheta + 1)` — discrete bins + 1 direct scalar (`:1106-1108`).
- `input_dropout = Dropout1d(model_config["input_dropout"])` (`:1110`).

**forward(batch) — I/O**
- **INPUT** `batch` dict. All `PairedSinglePointWithBeamformer` keys, **plus**:
  - `system_timestamp` `(B, S)` — for normalized time (`:1118-1126`).
  - if `use_xy`: `rx_pos_xy` `(B, S, 2)` (`:1130`).
  - Requires **S > 1** (multipaired models assert `snapshots_per_session > 1` in the train loop, per
    training.md notes).
- **OUTPUT** dict (extends the paired net's output):
  - `"single"`, `"paired"` (from `multi_radio_net`).
  - `"multipaired"`: **`(B, S, output_ntheta)`**, L1-normalized (`NormP1Dim2`). With skip:
    `norm(output_paired + discrete_output)` (`:1166-1168`); without: `norm(discrete_output)` (`:1170-1172`).
  - `"multipaired_direct"`: **`(B, S, 1)`** — raw scalar (the `[-1]` channel), NOT normalized (`:1163`,`:1174`).
- **Side effects:** training-time **random time-flip augmentation** — with prob 0.5 (when training) the
  time axis is reversed by `system_timestamp[:, [-1]] - system_timestamp` (`:1116-1121`) — randomness;
  `input_dropout`; `detach_or_not` on paired output (`:1140`).

**Fusion mechanics (verified).** `output_paired` `(B,S,ntheta)` → optionally detached → concat with
`normalized_time.unsqueeze(2)` (`(B,S,1)`) and optionally `normalized_positions` (`(B,S,2)`) → `input_net`
→ `(B,S,d_model)` → `transformer_encoder` (attends across the S snapshots, `batch_first=True`) →
`output_net` → `(B,S,output_ntheta+1)`. Split into `discrete_output = [...:output_ntheta]` and
`direct_output = [...,[-1]]` (`:1159-1163`). The skip connection adds the transformer's discrete
refinement back onto the paired distribution before re-normalizing (`:1165-1172`).

**Tested by.** No dedicated test found in `tests/test_models.py` / `tests/test_new_models.py` (those
cover only single + paired checkpoints). **Coverage gap:** `multipaired`/`multipaired_direct` shapes,
the time-flip augmentation, and skip-connection branch are untested. ⚠️

---

## Active network 4 — `TrajPairedMultiPointWithBeamformer`

### `TrajPairedMultiPointWithBeamformer`  ·  `single_point_networks.py:1178`  ·  ✅ Active (`"trajmultipairedbeamformer"`) 🧪 newer

**Summary.** Variant of the multipoint net that compresses the whole sequence into a **single latent
trajectory vector** (read from transformer output position 0), then re-expands it across all snapshots
and runs a per-snapshot `traj_net` (FFNN) conditioned on time (+ optional xy). Predicts a discrete
distribution, a direct scalar, and (optionally) a **tx position** (2-D). Intended for trajectory-aware
localization.

**Ctor** (differs from `PairedMultiPointWithBeamformer`)
- `multi_radio_net = PairedSinglePointWithBeamformer(model_config["paired"], …)` (`:1182`).
- `skip_connection` default **`False`** here (`:1188`); `pred_xy = model_config.get("pred_xy", False)` (`:1190`).
- `self.latent = model_config["latent"]` (e.g. 8) (`:1193`).
- Transformer as before; `output_net = Linear(d_model, latent)` — produces the latent, **not** the
  distribution (`:1220-1222`).
- `traj_net = FFNN(inputs=latent + 1 (+2 if use_xy), depth=traj_layers, hidden=traj_hidden,
  outputs=output_ntheta + 1 + 2, block=True, bn=True, norm="layer", act=LeakyReLU)` (`:1226-1239`).
  The `+1+2` = direct scalar + 2-D tx-position prediction.

**forward(batch) — I/O**
- **INPUT** `batch` dict. Same as `PairedMultiPointWithBeamformer`: paired keys + `system_timestamp`
  `(B,S)` (+ `rx_pos_xy` `(B,S,2)` if `use_xy`). `batch_size, snapshots_per_example` read from
  `system_timestamp.shape` (`:1244`).
- **OUTPUT** dict:
  - `"single"`, `"paired"` (from `multi_radio_net`).
  - `"multipaired"`: **`(B, S, output_ntheta)`**, L1-normalized, **no** skip connection (`:1304`).
  - `"multipaired_direct"`: **`(B, S, 1)`** raw scalar (`:1302`,`:1306`).
  - `"multipaired_tx_pos"`: **`(B, S, 2)`** — present **iff** `pred_xy=True` (`:1307-1308`).
- **Side effects:** none beyond submodules (notably **no** training-time time-flip here, unlike
  network 3). `detach_or_not` on paired output (`:1261`).

**Fusion mechanics (verified).** `output_paired` + time(+xy) → `input_net` → `transformer_encoder` →
`output_net` → `(B,S,latent)` (`:1280-1282`). `trajectories = transformer_output[:, [0]].expand(B, S,
latent)` (`:1284-1286`) — **only snapshot 0's latent is used**, broadcast to all S snapshots (the
"trajectory" summary). Concat with time(+xy) → `traj_net` → `(B,S,output_ntheta+3)`. Split into
`discrete_output`, `tx_preds` (`[output_ntheta:output_ntheta+2]`), `direct_output` (`[-1]`)
(`:1300-1302`).

**Tested by.** None found. **Coverage gap:** entire network untested. 🧪 newer/experimental but
**is** registered and reachable via `load_model` name `"trajmultipairedbeamformer"`.

---

## Baseline — `SinglePointPassThrough`

### `SinglePointPassThrough`  ·  `single_point_networks.py:1312`  ·  ✅ Active (`"pass_through"`)

**Summary.** Non-learning baseline: returns the precomputed empirical distribution directly as the
prediction. Has a single dummy parameter `self.w = Parameter(zeros(1))` so the module has parameters
for the optimizer, but it contributes nothing (`+ self.w * 0.0`).

**Signature**
- Ctor: `(model_config, global_config)` — only creates `self.w` (`:1315`). Ignores both configs.
- `forward(batch)`: **INPUT** requires `batch["empirical"]` `(B, S, nthetas)`. **OUTPUT**
  `{"single": batch["empirical"] + w·0.0}` → `(B, S, nthetas)` (`:1319`).
- Side effects: none (the `+ w*0` is a no-op keeping `w` in the graph).

**Analysis.** Pure passthrough of the empirical input; useful as a floor for comparison. No
normalization applied (assumes `empirical` is already a distribution). Untested directly.

---

## Config-only / unused networks (🧪)

These four are **not** in the `load_model` registry. They are reachable only indirectly, and several
carry hardcoded dims (68 input, 12 output) and "chatGPT" provenance comments. None were found in any
checked-in config in this scan.

### `TransformerTimeModel`  ·  `:154`  ·  🧪 config-only
- **Reachable only** via `AllWindowsStatsNet(conv_type="transformer")` (`:827-835`). Not registered.
- forward: `x (B, 68, 256)` → `(B, 12)`. Transposes to `(B,256,68)`, projects 68→`d_model`, optional
  positional encoding, transformer encoder, mean-pool over seq, `Linear→12`.
- ⚠️ **Hardcoded**: input dim `68` (`self.input_fc = Linear(68, d_model)`, `:183`) and output `12`
  (`output_channels=12` hardcoded at the construction site `:833`, ignoring `AllWindowsStatsNet.outputs`).
  Likely mismatches a configured `outputs` ≠ 12 and an input channel count ≠ 68.

### `CrossAttentionNet`  ·  `:234`  ·  🧪 config-only
- **Reachable only** via `AllWindowsStatsNet(conv_type="cross")` (`:819-826`). Not registered.
- forward: `x (B, 68, 256)` → `(B, output_channels)`. First 3 channels → 1D conv path → query;
  remaining 65 channels → 2D conv path → key/value; cross-attention; `Linear→output_channels`.
- "chatGPT" comment (`:233`). Hardcoded 3/65 channel split.

### `Separate1D2DNet`  ·  `:327`  ·  🧪 config-only
- **Reachable only** via `AllWindowsStatsNet(conv_type="1d2d")` (`:811-818`). Not registered.
- forward: `x (B, 68, 256)` → `(B, output_channels)`. Splits into first-3-channels (1D conv) and
  last-65 (2D conv over a `(1,65,256)` image), global-pools each, concatenates, `Linear→output_channels`.

### `SignalMatrixNet`  ·  `:565`  ·  🧪 config-only
- **Reachable only** when `"signal_matrix_net" in model_config` inside `SinglePointWithBeamformer`
  (`:923-927`), which also needs `signal_matrix_input` global (so `abs_signal_and_phase_diff` is
  collated). Not in the registry; no checked-in config found.
- forward: `x (B, 4, L)` → 7× strided `Conv1d` (k=11) → final conv to `outputs` → `mean over time` →
  `(B, outputs)` (`:588-589`).

(Also unused in this module: `cdf`/`normal_correction_for_bounded_range`/`sigmoid_dist` helpers —
`normal_correction_for_bounded_range` IS tested by `tests/test_models.py::test_normal_correction`,
but is not called by any network here.)

---

## Cluster summary

| Symbol | Status | Registry name | forward INPUT (keys) | forward OUTPUT (shapes) | Direct test | Top finding |
|---|---|---|---|---|---|---|
| `BasicBlock` | ✅ | — | `x (…,hidden)` | `(…,hidden)` | ❌ | post-add activation smell; batchnorm 3-D hazard if `bn=True,norm=batch` |
| `FFNN` | ✅ | — | `x (…,inputs)` | `(…,outputs)` | ❌ (indirect) | active configs use `bn=False`→Identity norms |
| `PrepareInput` | ✅ ⚠️ | — | batch dict (flag-gated) | `(B,S,inputs+extra)` | ❌ | **in-place zeroes `weighted_windows_stats` batch tensor in training** (`:508-510`) |
| `SinglePointWithBeamformer` | ✅ | `beamformer` | beamformer/empirical/phase/… (`B,S,…`) | `single (B,S,ntheta)` [+`output_phi (B,S,1)`] | ✅ (eval) | sub-encoders only if config block present; dead expand at `:969` |
| `PairedSinglePointWithBeamformer` | ✅ | `pairedbeamformer` | above + `rx_theta_in_pis (B,S)`; radio-interleaved rows | `single (B,S,ntheta)`, `paired (B,S,ntheta)` | ✅ (eval) | rotates only snapshot 0 (`[:,0]`); implies S=1 at paired stage |
| `PairedMultiPointWithBeamformer` | ✅ | `multipairedbeamformer` | above + `system_timestamp (B,S)` [+`rx_pos_xy (B,S,2)`] | +`multipaired (B,S,ntheta)`, `multipaired_direct (B,S,1)` | ❌ | transformer over S snapshots; training time-flip aug (randomness) |
| `TrajPairedMultiPointWithBeamformer` | ✅ 🧪 | `trajmultipairedbeamformer` | above; `system_timestamp (B,S)` [+`rx_pos_xy`] | +`multipaired (B,S,ntheta)`, `multipaired_direct (B,S,1)` [+`multipaired_tx_pos (B,S,2)`] | ❌ | latent trajectory from snapshot-0 only, broadcast to all S; predicts tx-xy |
| `SinglePointPassThrough` | ✅ | `pass_through` | `empirical (B,S,ntheta)` | `single (B,S,ntheta)` | ❌ | pure passthrough; dummy `w` param |
| `TransformerTimeModel` | 🧪 | — (conv_type="transformer") | `x (B,68,256)` | `(B,12)` | ❌ | hardcoded 68→12 dims |
| `CrossAttentionNet` | 🧪 | — (conv_type="cross") | `x (B,68,256)` | `(B,out)` | ❌ | hardcoded 3/65 split |
| `Separate1D2DNet` | 🧪 | — (conv_type="1d2d") | `x (B,68,256)` | `(B,out)` | ❌ | hardcoded 3/65 split |
| `SignalMatrixNet` | 🧪 | — (`signal_matrix_net` block) | `x (B,4,L)` | `(B,outputs)` | ❌ | reachable only via optional config block |

### Cross-cutting themes
1. **All active distribution outputs (`single`/`paired`/`multipaired`) are `abs`-then-L1-normalized
   over `dim=2`** (the angle axis), so they are always valid distributions over `output_ntheta` bins in
   [-π, π]. The `*_direct` scalars and `tx_pos` are raw (un-normalized).
2. **Frame handling lives entirely in pairing:** the only craft-frame rotation is `rotate_dist(single,
   rx_theta_in_pis·π)` at `:1025-1028`, and it operates on **snapshot 0 only** — a strong, undocumented
   contract that the paired/multi stack runs effectively with S=1 at the pair step. ⚠️ worth confirming
   against the train loop's snapshot handling.
3. **Test coverage is thin and eval-only.** Only `beamformer` and `pairedbeamformer` are exercised
   (inference-equivalence tests, `model.eval()`). The two transformer/trajectory nets, the training-time
   augmentations (PrepareInput dropout in-place mutation, time-flip), and the `pass_through` baseline
   have **no direct tests**.

### Top findings (deep-dive)
1. **`PrepareInput.prepare_input` in-place mutation** (`:508-510`): `v = batch["weighted_windows_stats"]`
   then `v[dropout_mask[2]] = 0` zeroes a batch-tensor view in place during training — a real side
   effect that can contaminate any downstream reader if the batch is reused. Other branches build fresh
   tensors and are safe. (Confidence: high.)
2. **Paired rotation uses snapshot 0 only** (`:1025` `single_radio_estimates[:, 0]`): the paired and
   multipoint stacks effectively pair single-snapshot estimates; the `concatenate(dim=2)` at `:1033`
   imposes a matching-S constraint, implying S=1 at the pair stage. ⚠️ confirm intended.
3. **Registered-but-untested + config-only nets**: `multipairedbeamformer` and
   `trajmultipairedbeamformer` are live registry entries with **no tests**, and four classes
   (`TransformerTimeModel`/`CrossAttentionNet`/`Separate1D2DNet`/`SignalMatrixNet`) are reachable only
   via `AllWindowsStatsNet.conv_type` strings or an optional `signal_matrix_net` block — several with
   hardcoded 68→12 dims that would mismatch configured shapes.
