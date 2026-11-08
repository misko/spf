# Layer 2 (Model) Audit — `beamformer` / `SinglePointWithBeamformer`

READ-ONLY audit tracing config `model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`
(`model.name: beamformer`). Venv torch 2.9.1. Source:
`spf/model_training_and_inference/models/single_point_networks.py` (henceforth `spn.py`),
`spf/scripts/train_single_point.py` (`tsp.py`).

`load_model` (`tsp.py:513-517`): `name == "beamformer"` → `SinglePointWithBeamformer(model_config, global_config)`. ✅

---

## Config-resolved switches (from the YAML)

global: `beamformer_input=false`, `empirical_input=false`, `phase_input=false`,
`gains_input=True`, `rx_spacing_input=true`, `frequency_input=true`,
`vehicle_type_input=true`, `sdr_device_type_input=true`, `signal_matrix_input=false`,
`windowed_beamformer_input=true`, `nthetas=65`, `n_radios=2`.
(`beamformer_mag_input` key absent → `False` via `.get`, `spn.py:418`.)

model: `block=true, bn=true, depth=4, hidden=512, norm=layer, dropout=0.0,
input_dropout=0.1, detach=true`, plus a `windows_stats_net` block (no `signal_matrix_net` block).
optim: `dtype: torch.float32`, `amp: false`.

So in `SinglePointWithBeamformer.__init__`: `"windows_stats_net" in model_config` → True
(constructs `AllWindowsStatsNet`, `+outputs` to additional inputs); `"signal_matrix_net" in
model_config` → **False** (signal-matrix path is dead for this run; `self.signal_matrix_net=None`).
`output_ntheta` absent in model_config → `check_and_load_ntheta` defaults it to `nthetas=65`
(`spn.py:555-558`).

---

## TASK 1 — EXACT input-vector dimension vs FFNN first Linear (HARD CONTRACT) ✅

`PrepareInput.__init__` (`spn.py:437-454`) accumulates `self.inputs`:

| flag | enabled? | key | width added |
|---|---|---|---|
| beamformer_input | no | — | 0 (`:438`) |
| beamformer_mag_input | no | — | 0 (`:440`) |
| empirical_input | no | — | 0 (`:442`) |
| phase_input | no | — | 0 (`:444`) |
| gains_input | **yes** | `gains` | **+2** (`:446`) |
| rx_spacing_input | **yes** | `rx_wavelength_spacing` | **+1** (`:448`) |
| frequency_input | **yes** | `rx_lo` | **+1** (`:450`) |
| vehicle_type_input | **yes** | `vehicle_type` | **+1** (`:452`) |
| sdr_device_type_input | **yes** | `sdr_device_type` | **+1** (`:454`) |

`self.prepare_input.inputs = 2 + 1 + 1 + 1 + 1 = 6`.

`additional_inputs` (`spn.py:919-932`): signal_matrix_net absent → +0;
`windows_stats_net` present → `+ self.windows_stats_net.outputs = +12` (YAML `outputs: 12`).
So `additional_inputs = 12`.

**FFNN in_features = prepare_input.inputs + additional_inputs = 6 + 12 = 18.** (`spn.py:934-935`)

Runtime concat order (`prepare_input`, `spn.py:492-541`): only the enabled scalar branches
fire — `gains`/120 (2), `rx_wavelength_spacing[...,None]*50` (1), `(rx_lo[...,None]+1).log10()/20`
(1), `vehicle_type[...,None]` (1), `sdr_device_type[...,None]/10` (1) = **6**, then
`additional_inputs` = `[all_windows_embedding (B,S,12)]` appended → total last-dim **18**, concat on
`dim=2`. Matches the constructed `FFNN(inputs=18)` first `nn.Linear(18, 512)` (`spn.py:71`,
layer-norm branch `:88-93`). **CONTRACT HOLDS: 18 == 18.** ✅ High confidence.

(Note the runtime concat order is gains, rx_spacing, frequency, vehicle, sdr, then the 12-d
windows embedding — order differs from the ctor accumulation order but width is identical, which
is all the Linear cares about.)

### AllWindowsStatsNet output width = 12 (verify the `outputs` plumb-through)
`AllWindowsStatsNet(nthetas=65, **windows_stats_net)` → `outputs=12`. `windowed_beamformer=true`
sets `input_channels = 3 + nthetas = 3 + 65 = 68` (`spn.py:786-788`). `conv_type` defaults `"1d"`
→ `create_1dconv(...)` whose final layer is `nn.Conv1d(hidden_channels=64, outputs=12, ...)`
(`spn.py:689`). `forward` does `self.conv_net(input).mean(axis=2)` (`spn.py:902`) → `(N,12)` →
reshape `(B,S,12)` (`spn.py:903-906`). `output_phi` not set → default False → no `phi_network`,
no `output_phi` key. ✅ The 68 input channels are an *internal* contract of the windows net, NOT the
FFNN input; only its `outputs=12` reaches the FFNN.

---

## TASK 2 — Forward output contract ✅

`SinglePointWithBeamformer.forward` (`spn.py:947-994`):
- `windows_stats_net` runs → appends `all_windows_embedding (B,S,12)` to `additional_inputs`.
  `output_phi` NOT present (output_phi False) → `return_dict` has **no** `output_phi`.
- `signal_matrix_net is None` → that whole block skipped.
- `return_dict["single"] = F.normalize( ffnn(prepare_input(batch, additional_inputs)).abs(),
  dim=2, p=1 )` (`spn.py:987-993`).

FFNN output is `(B, S, output_ntheta=65)`. `.abs()` → nonnegative. `F.normalize(p=1, dim=2)` →
L1-normalize over **dim=2 (the 65-bin angle axis)**. So **OUTPUT = `{"single": (B, S, 65)}`,
each row a valid distribution summing to 1 over the angle axis.** ✅ Confirmed:
abs-then-L1-normalize over dim=2 (the nthetas axis). No `output_phi`, no signal-matrix outputs for
this config.

---

## TASK 3 — AllWindowsStatsNet window-axis augmentation ✅ (with a scale caveat)

Config: `window_shrink=0.5, window_shuffle=0.15, window_dropout=0.25, window_fraction=0.5`,
`norm=false`, `normalize_windowed_beamformer=false`.

Ctor assert `window_dropout==0.0 or not norm` (`spn.py:783-785`): `norm=false` → assert passes. ✅

Augmentation is gated on `if self.training:` (`spn.py:877-894`) — **train-only, not applied at
eval**. ✅ Logic:
- shrink: with prob `window_shrink` (0.5), crops a contiguous window slice
  `[start:end]` where `start = rand*window_fraction*windows`, `end = start + windows*(1-window_fraction)`
  (`spn.py:878-883`). Pure slice on the window (time) axis (dim 2 of the reshaped `(N,C,windows)`),
  applied **after reshaping to (-1, channels, windows)** so it crops all (batch×snapshot) rows
  identically — a fresh view, no in-place write.
- **elif** dropout: `input = input[:, :, torch.rand(windows) > window_dropout]` (`spn.py:886-887`) —
  boolean-mask the window axis, ~25% dropped. Note the `elif` means **shrink and dropout are mutually
  exclusive per call** (if shrink fires, dropout is skipped that step). Fresh tensor (advanced index).
- shuffle: independent `if` with prob 0.15 — `index_select(2, randperm(...))` permutes the window
  axis (`spn.py:891-894`). Fresh tensor.

**Leakage check:** all three operate on the local `input` tensor (a reshape/transpose of
`batch["all_windows_stats"]` and `batch["windowed_beamformer"]`); the slicing/masking/index_select
produce **new tensors**, none write back into `batch`. So no batch-dict mutation, no train/eval
leak. ✅ The crop+pool design (`mean(axis=2)` over the conv output) makes variable post-aug window
length harmless. Confidence: high.

**`normalize_windowed_beamformer=false` scale concern (REAL, low-severity).** In `forward`
(`spn.py:862-871`): with normalize false, the branch is
`inputs.append(batch["windowed_beamformer"].transpose(2,3) / 500)` — i.e. **raw power maps divided
by a hardcoded constant 500**, NOT per-row L1-normalized. The `all_windows_stats` branch IS
normalized (`/(max+1e-5)`, first 2 channels forced to π, `spn.py:851-860`), but the 65 windowed-
beamformer channels rely on the magic `/500` to land in a usable range. If real power magnitudes
drift far from the ~500 scale that constant assumed, these 65 channels enter the conv un-normalized
and could dominate or vanish. **Affects this run:** YES (windowed_beamformer=true, normalize=false).
Severity: low (it's a learned conv with LeakyReLU and no internal norm — `norm=false` — so a global
scale is partially absorbable, but it is an un-audited magic constant). Confidence: medium.

**dtype of the raw power maps.** Per Layer-1, `windowed_beamformer`/`all_windows_stats` may be
stored float16, but `V5Inference.__getitem__` casts every float tensor to `target_dtype`
(`spf_dataset.py:794-801`) and this run sets `dtype: torch.float32`. So they arrive float32 — the
`/500` and conv run in float32. No float16 range/precision hazard for this run. ✅

---

## TASK 4 — KNOWN_ISSUES #34 (in-place mutation via aliased views) — DOES IT BITE THIS RUN?

#34 (`claude_docs/KNOWN_ISSUES.md:91`) flags two unsafe in-place dropout zeroings in
`PrepareInput.prepare_input` during training (`input_dropout>0`):
1. `weighted_windows_stats` (phase_input branch, `spn.py:508-511`): `v = batch["weighted_windows_stats"]`
   (no copy) then `v[dropout_mask[2]] = 0` → mutates the batch tensor.
2. `vehicle_type` (`spn.py:528-531`): `v = batch["vehicle_type"][..., None]` (an **unsqueeze view**,
   shares storage) then `v[dropout_mask[6]] = 0` → mutates the underlying batch `vehicle_type`.

For THIS config (`input_dropout=0.1 > 0`, training):

- **phase_input is FALSE** → the `if self.phase_input:` block (`spn.py:507-511`) **never executes**.
  The `weighted_windows_stats` mutation **DOES NOT FIRE** for this run. ✅ (Also: `weighted_windows_stats`
  is not even produced — phase_input off — so the alias hazard is moot here.)
- **vehicle_type_input is TRUE** → the `if self.vehicle_type_input:` block (`spn.py:527-531`)
  **DOES execute**, and `v = batch["vehicle_type"][..., None]; v[dropout_mask[6]] = 0` **DOES fire
  in training**. This **mutates `batch["vehicle_type"]` in place** (the `[...,None]` view shares
  storage with the batch tensor; the masked-assignment writes through). **#34 bites this run via the
  vehicle_type branch.** Severity: medium-but-latent. Confidence: high.

  *Why latent (not currently corrupting results):* the train loop builds a fresh `batch_data` each
  step (`tsp.py:1298 batch_data = batch_data.to(device)` from the dataloader) and the model is the
  only consumer; `vehicle_type` is read once, within this same `prepare_input` call, *before* it
  matters again. The zeroing also semantically *is* the intended input-dropout (randomly drop the
  vehicle-type feature). The hazard is fragility: any future code that reads `batch["vehicle_type"]`
  after the forward pass (e.g. logging, a second model, a reused/cached batch) would see corrupted
  (partially-zeroed) values. No evidence of such a second reader in stage-1 single training today.

- `sdr_device_type` (`spn.py:532-536`): dropout mask **commented out** ("always pass in") → never
  mutated. ✅
- The other enabled branches (`gains` `/120`, `rx_spacing` `*50`, `frequency` `log10/20`) build
  fresh tensors via arithmetic (`v = batch[...] * c`) → safe even though they ARE dropout-masked
  (`spn.py:512-526`). The mask zeroing hits the fresh `v`, not the batch. ✅

**Verdict:** #34's `weighted_windows_stats` half does NOT fire (phase_input off); the
`vehicle_type` half DOES fire and mutates the batch in place every training step. Net effect on this
run is benign-but-fragile (intended dropout semantics, single consumer), so **no correctness bug for
the loss this run**, but the KNOWN_ISSUES entry is accurate and live via `vehicle_type`.

---

## TASK 5 — NEW model-layer bugs / hazards for this run

1. **BN+LayerNorm interaction — NO bug here, but the doc's old caveat is moot for this config.**
   YAML has `bn=true, norm=layer`. In `FFNN.__init__` the branch `bn==False or norm=="batch"` is
   FALSE (bn true, norm "layer") → takes the **`elif norm=="layer"`** branch (`spn.py:88-106`):
   `LayerNorm(hidden=512)` for the input stem and `BasicBlock(bn=True, norm="layer")` →
   `LayerNorm(512)` inside each block (`spn.py:43-45`). LayerNorm normalizes over the **last dim**,
   so it is shape-agnostic to the `(B, S, 512)` 3-D input — **no BatchNorm1d is ever instantiated**
   for this config, so the "BatchNorm1d errors on 3-D (B,S,·)" hazard noted in the reference doc
   **does not apply** here. ✅ (The reference doc's general warning assumed `bn=False`; this config
   uses `bn=True, norm=layer`, which is the *other* safe combination.) Confidence: high.

2. **dtype float16-vs-float32 mismatch — NOT present.** `optim.dtype = torch.float32` →
   `target_dtype=float32` → all batch float tensors cast to float32 (`spf_dataset.py:794-801`);
   model `.to(device)` keeps float32 params; `amp=false` (`torch.autocast(enabled=False)`,
   `tsp.py:1301`). No half/float32 mix. ✅

3. **Dead config-only nets accidentally engaged — NO.** `conv_type` not set in `windows_stats_net`
   → defaults `"1d"` (`spn.py:751`), so `create_1dconv` is used; the `2d/1d2d/cross/transformer`
   branches (and their hardcoded-68→12 chatGPT nets) are NOT constructed. `signal_matrix_net` block
   absent → `SignalMatrixNet` not built. ✅ No dead net is engaged.

4. **`detach: true` in a single model — INERT.** `model.detach` is read only by the *paired/multi*
   wrappers (`PairedSinglePointWithBeamformer.__init__:1006`, etc.). `SinglePointWithBeamformer`
   never reads `model_config["detach"]`. For `name=beamformer` the `detach: true` YAML key is simply
   ignored — no graph cut, no effect. ✅ Not a bug, just an inert config key.

5. **`dropout: 0.0` (FFNN internal dropout) — inert** (BasicBlock `d1/d2` are `Dropout(0.0)` =
   no-op). The only active dropout is `input_dropout=0.1` in `PrepareInput` (see #34). The FFNN
   `nn.Dropout(0.0)` stem (`spn.py:90`) is also a no-op. ✅

6. **`input_dropout` ctor assert passes** (`spn.py:455-468`): `input_dropout=0.1 != 0` requires the
   count of enabled inputs > 1; here 5 flags enabled (gains, rx_spacing, frequency, vehicle, sdr) →
   sum > 1 (booleans sum to 5). Assert satisfied. ✅ (Note `gains` counts as 1 in the assert sum
   though it contributes width 2 — the assert is about *number of sources*, not width.)

7. **(minor) input-dropout never drops the windows-embedding or sdr.** `dropout_mask` has 8 rows
   (`spn.py:476-481`) keyed by source index, but `additional_inputs` (the 12-d windows embedding) is
   appended in `forward`, *outside* `prepare_input`'s masking, and `sdr` masking is commented out.
   So input-dropout only ever zeroes gains/rx_spacing/frequency/vehicle features. Not a bug; just a
   scope note for the data/training layer. Confidence: high.

No NEW correctness-breaking bug found for this run. The only live KNOWN issue is #34 via
`vehicle_type` (latent/fragile, not corrupting this single-consumer run).

---

## TASK 6 — BOUNDARY-IN / BOUNDARY-OUT contract

### BOUNDARY-IN — batch keys + shapes the model REQUIRES (to check vs Layer 1)
All tensors `(B, S, ...)` where B = sessions×radios in the batch, S =
`train_snapshots_per_session = 1` for this config.

Consumed by `PrepareInput.prepare_input` (only enabled branches):
| key | required shape | used at | scaling |
|---|---|---|---|
| `y_rad` | `(B, S)` | `:476-481` | only for `dropout_mask` shape `(8,B,S)` in training |
| `gains` | `(B, S, 2)` | `:523` | `/120` → 2 dims |
| `rx_wavelength_spacing` | `(B, S)` | `:513` | `[...,None]*50` → 1 dim |
| `rx_lo` | `(B, S)` | `:518` | `(.+1).log10()/20` → 1 dim |
| `vehicle_type` | `(B, S)` | `:528` | `[...,None]` → 1 dim (⚠ mutated in place, #34) |
| `sdr_device_type` | `(B, S)` | `:533` | `[...,None]/10` → 1 dim |

Consumed by `AllWindowsStatsNet.forward`:
| key | required shape | used at |
|---|---|---|
| `all_windows_stats` | `(B, S, 3, W)` (W = windows/time, e.g. 256) | `:851-860` (normalize: max over dim 3; chans 0:2 set to π) |
| `windowed_beamformer` | `(B, S, W, nthetas=65)` → transposed to `(B,S,65,W)` | `:862-871` (`/500`) |

After concat (`spn.py:872`): `all_windows_stats` (3 chans) + `windowed_beamformer` (65 chans) =
**68 channels** along dim 2 — must match `input_channels = 3 + nthetas = 68` (`spn.py:786-788`).
**This is a second hard contract:** Layer-1 must deliver `all_windows_stats` with exactly 3 stat
channels and `windowed_beamformer` with exactly 65 (=nthetas) theta bins, else the first
`Conv1d(68, 64)` mismatches. ✅ (verify against Layer-1's collation at
`spf_dataset.py:182-183`).

NOT required (absent / disabled for this run, must NOT be assumed present):
`weighted_beamformer`, `empirical`, `weighted_windows_stats`, `abs_signal_and_phase_diff`,
`signal_matrix`, `rx_theta_in_pis`, `system_timestamp`, `rx_pos_xy` (those belong to
paired/multi/signal-matrix paths). `train_single_point.load_dataloaders` adds `windowed_beamformer`
to `skip_fields` only if `windowed_beamformer_input` is False (`tsp.py:124-125`) — here it's True,
so `windowed_beamformer` IS produced. ✅

### BOUNDARY-OUT — dict fed to the loss
`{"single": (B, S, 65)}`, L1-normalized over dim=2, nonnegative (a distribution). No `output_phi`,
no `paired`/`multipaired`/`*_direct`/`tx_pos` keys for `name=beamformer`. The training loop computes
`single` loss against the scattered binned target (`save_on: val/single_loss`, `loss: mse`).
Confirm Layer-3 reads only `output["single"]` for this model.

---

## TASK 7 — Cross-check of existing reference docs

`claude_docs/reference/spf/model_training_and_inference/single_point_networks.md`:
- Input-assembly table (`:101-111`), the FFNN bn/norm branch analysis (`:81-86`), the forward
  contract for `beamformer` (`single (B,S,ntheta)` L1-normalized over dim=2, `:176`,`:421`), the
  `signal_matrix_net`-only and `output_phi`-only gating, and the #34 in-place-mutation write-up
  (`:118-138`, correctly identifying BOTH `weighted_windows_stats` and `vehicle_type` as unsafe)
  are all **accurate** and match source. ✅
- **Inaccuracy / staleness flag (minor):** the doc's recurring BatchNorm-3D caveat
  (`:58-60`,`:82-85`,`:407`) is framed as "active configs use `bn=False` → Identity norms." This
  specific config uses `bn=True, norm=layer`, which takes the LayerNorm branch (also safe) — so the
  blanket "active configs use bn=False" generalization is **not true for this config**. The
  conclusion (no BatchNorm1d error) still holds, but for a different reason. Worth a note.
- The doc's "Open question" (`:147-148`) — "Does any training config set `input_dropout>0` together
  with `phase_input`?" — **ANSWERED for this config: NO.** `input_dropout=0.1` but `phase_input=false`,
  so the `weighted_windows_stats` mutation does not fire here; the `vehicle_type` mutation does.

`VERIFY_single_point_networks.md` and `latest_models.md`: `latest_models.md` was not found under
`claude_docs/reference/spf/model_training_and_inference/` (only `single_point_networks.md`,
`VERIFY_single_point_networks.md`, `inference_and_baseline.md` present). If `latest_models.md` lives
elsewhere it was outside this scan; flagging as **not located** rather than verified.

---

## Questions / assertions for the DATA + TRAINING layers

For **Layer 1 (data)** — assert the produced batch satisfies:
1. `all_windows_stats` is `(B, S, 3, W)` with exactly **3** channels and `windowed_beamformer` is
   `(B, S, W, 65)` with exactly **65** theta bins, so concat → 68 channels == `Conv1d(68, …)`.
   (Hard contract; mismatch is an immediate runtime error.)
2. `gains` is `(B, S, 2)` (width-2 feature). If gains is ever `(B,S,1)` or `(B,S)`, the FFNN input
   width drops from 18 and the `Linear(18,512)` mismatches.
3. `rx_wavelength_spacing`, `rx_lo`, `vehicle_type`, `sdr_device_type` are each `(B, S)` (the model
   does `[...,None]` to make them width-1). `y_rad` is `(B, S)` (needed for the training dropout-mask
   shape `(8, B, S)`).
4. All of the above arrive **float32** (target_dtype), given `optim.dtype=torch.float32` — confirm
   Layer-1's `__getitem__` dtype cast actually ran (it casts only float16/32/64 tensors;
   integer-coded `vehicle_type`/`sdr_device_type` stay integer/float as stored — verify they're
   float so `/10` and concat with float features don't dtype-promote oddly).
5. `weighted_windows_stats`, `weighted_beamformer`, `empirical`, `signal_matrix`,
   `abs_signal_and_phase_diff`, `rx_theta_in_pis`, `system_timestamp` are NOT required and need not
   be produced (phase/beamformer/empirical/signal-matrix/paired all OFF).

For **Layer 3 (training/loss)** — assert:
6. The loss reads **only** `output["single"]` `(B,S,65)` for `name=beamformer`; no `paired`/
   `multipaired`/`output_phi` consumed.
7. The batch dict is **not reused/re-read after the forward pass** within a step — because
   `prepare_input` mutates `batch["vehicle_type"]` in place during training (#34 via vehicle_type).
   If any post-forward logging/metric or a cached batch reads `vehicle_type`, it sees partially
   zeroed values. (Latent today; assert no second reader.)
8. `input_dropout=0.1` only drops gains/rx_spacing/frequency/vehicle features (NOT the 12-d windows
   embedding, NOT sdr) — confirm this is the intended dropout scope for the training objective.
9. With `S=1` (`train_snapshots_per_session: 1`), the model is fine (everything is dim-2 trailing
   transforms), but confirm the loss/scatter target is also `(B, 1, 65)` so the MSE aligns.

---

## Summary of findings (severity · confidence · affects-this-run?)

| # | finding | file:line | severity | conf | affects run? |
|---|---|---|---|---|---|
| F1 | FFNN input width = **18** (= 6 scalar + 12 windows embedding) == `Linear(18,512)`; contract holds | `spn.py:437-454, 919-935` | n/a (verified OK) | high | YES (verified) |
| F2 | Output `{"single": (B,S,65)}`, abs→L1-norm over dim=2; no `output_phi` | `spn.py:987-993` | n/a (verified OK) | high | YES |
| F3 | #34 `weighted_windows_stats` in-place mutation does **NOT** fire (phase_input off) | `spn.py:507-511` | none here | high | NO |
| F4 | #34 `vehicle_type` in-place mutation **DOES** fire each train step (unsqueeze view) | `spn.py:527-531` | medium (latent/fragile) | high | YES (benign, single consumer) |
| F5 | `normalize_windowed_beamformer=false` → raw power maps `/500` magic constant, un-normalized 65 chans | `spn.py:870-871` | low | medium | YES |
| F6 | Window aug (shrink/shuffle/dropout/fraction) train-only, no batch leak; shrink/dropout mutually exclusive (`elif`) | `spn.py:877-894` | none (verified OK) | high | YES |
| F7 | `bn=true,norm=layer` → LayerNorm everywhere; no BatchNorm1d 3-D hazard | `spn.py:88-106, 43-45` | none | high | YES |
| F8 | No dtype mismatch (float32 throughout; amp off) | `tsp.py:1111-1118,1301`; `spf_dataset.py:794-801` | none | high | YES |
| F9 | `detach: true` is inert for `beamformer` (only paired/multi read it) | `spn.py:1006` | none | high | YES (no effect) |
| F10 | signal_matrix / 2d/1d2d/cross/transformer nets NOT engaged | `spn.py:803-835, 921-927` | none | high | NO (dead) |
| F11 | reference-doc BatchNorm caveat assumes `bn=False`; this run uses `bn=True,norm=layer` (still safe, different reason) | doc `:58-60,82-85` | doc-staleness | high | doc only |
