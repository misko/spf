# VERIFY — `single_point_networks.md`

Adversarial verification of
`claude_docs/reference/spf/model_training_and_inference/single_point_networks.md`
against
`spf/model_training_and_inference/models/single_point_networks.py` and the
`load_model` registry at `spf/scripts/train_single_point.py:513-525`.

Method: read-only. Every claim cross-checked against quoted source lines.

---

## (a) Priority claims table

### Claim 1 — The 4 forward output contracts

| Net | Doc claims output keys | Source | Verdict |
|---|---|---|---|
| `SinglePointWithBeamformer` | `{single}` (+`output_phi` iff windows_stats_net.output_phi) | `:948` `return_dict={}`; `:987` `return_dict["single"]=...`; `:954-955` conditional `output_phi`; `:994` `return return_dict` | **CONFIRMED** |
| `PairedSinglePointWithBeamformer` | `{single, paired}` | `:1048-1053` `return {"single":..., "paired":...}` | **CONFIRMED** |
| `PairedMultiPointWithBeamformer` | `{single, paired, multipaired, multipaired_direct}` | `:1114` `output=self.multi_radio_net(batch)` (carries single+paired); `:1166/1170` `output["multipaired"]`; `:1174` `output["multipaired_direct"]` | **CONFIRMED** |
| `TrajPairedMultiPointWithBeamformer` | above + `multipaired_tx_pos` iff `pred_xy` | `:1242` inherits single+paired; `:1304` multipaired; `:1306` multipaired_direct; `:1307-1308` `if self.pred_xy: output["multipaired_tx_pos"]=tx_preds` | **CONFIRMED** |

**Shape spot-checks (2 of them):**
- `"single"` `(B,S,ntheta)`: `:987-993` `normalize(ffnn(prepare_input(...)).abs(), dim=2, p=1)`.
  FFNN outputs `output_ntheta` on the trailing dim of a `(B,S,·)` input → `(B,S,output_ntheta)`,
  normalized over dim=2. **CONFIRMED.**
- `"multipaired_direct"` `(B,S,1)`: `:1163` `direct_output = full_output[..., [-1]]` — fancy-index
  `[-1]` keeps a length-1 trailing dim → `(B,S,1)`, and it is NOT passed through `self.norm`
  (`:1174` assigns `direct_output` raw). **CONFIRMED** (shape and "not normalized").

**Verdict: CONFIRMED (all 4 contracts + both shape spot-checks).**

---

### Claim 2 — In-place mutation; by-reference vs fresh-tensor distinction

Doc claims: only the `phase_input` branch aliases the batch tensor by reference
(`v = batch["weighted_windows_stats"]` then `v[mask]=0`), while the other branches build
fresh tensors via arithmetic and are therefore safe.

Source check, each training-time `v[dropout_mask[i]]=0` branch:
- beamformer `:495` `v = batch["weighted_beamformer"] / (2*v_mean)` → fresh (division). `:498`
  `v = self.beamformer_dropout(v)` (Identity or Dropout). `:500` mutates fresh `v`. **fresh.**
- empirical `:503` `v = (batch["empirical"] - 1.0/...) * 10` → fresh (sub+mul). `:505` mutates. **fresh.**
- **phase `:508` `v = batch["weighted_windows_stats"]` → ALIAS (no op). `:510` `v[dropout_mask[2]]=0`
  mutates the batch tensor IN PLACE.** This is the unique by-reference case.
- rx_spacing `:513` `v = batch["rx_wavelength_spacing"][..., None] * 50`. NOTE: `[...,None]*50`
  is a fresh tensor; basic indexing with `None` then `*50` does not alias. **fresh.**
- frequency `:518` `v = (batch["rx_lo"][..., None] + 1).log10()/20` → fresh. **fresh.**
- gains `:523` `v = batch["gains"] / 120` → fresh. **fresh.**
- vehicle_type `:528` `v = batch["vehicle_type"][..., None]` → `[...,None]` (basic indexing /
  `unsqueeze`) RETURNS A VIEW, but `:530` `v[dropout_mask[6]]=0` would write through the view to
  `batch["vehicle_type"]`. ⚠️ This is a SECOND by-reference branch the doc does NOT flag.
- sdr_device_type `:532-536` dropout commented out (`:534-535`) → never masked. Doc CONFIRMED.

So the doc's headline (phase_input mutates by reference) is **CONFIRMED**, but its claim that
"only phase_input is by-ref / all other branches build fresh tensors and are safe" is **not
strictly true**: `vehicle_type` (`:528`) uses `[...,None]` which yields a view of the batch
tensor, and `:530` writes into it in training — a real second in-place write. The doc explicitly
lists `vehicle_type` among the masked branches (`:124`) but then asserts "Most other branches
build a fresh tensor via arithmetic (`v = batch[...] * c`), so those are safe" (`:129-130`),
which mis-states the vehicle_type case (no arithmetic, pure view).

**Verdict: PARTIAL.** Primary phase_input by-ref mutation CONFIRMED; the "only phase_input is
by-ref" distinction is REFUTED — `vehicle_type` (`:528`/`:530`) is also a view-write.
(Severity is lower for vehicle_type since `[...,None]` adds a trailing dim, but
`v[bool_mask]=0` still scatters zeros back into the underlying storage.)

---

### Claim 3 — Paired stage operates on snapshot 0 only → effectively S=1

Source `:1021-1035`:
- `:1021` `single_radio_estimates = self.single_radio_net(batch)["single"]` → `(B,S,ntheta)`.
- `:1025-1028` `rotate_dist(single_radio_estimates[:, 0], torch_pi_norm(rotation_offets)).unsqueeze(1)`
  — indexes `[:, 0]` (drops S→ `(B,ntheta)`), rotates, `.unsqueeze(1)` → `(B,1,ntheta)`.
- `:1031` `additional_inputs = self.prepare_input.prepare_input(batch)` → `(B,S,inputs)` (full S).
- `:1033` `torch.concatenate([single_radio_estimates_rotated, additional_inputs], dim=2)` — concat
  on dim=2 requires dims 0 and 1 to MATCH. `single_radio_estimates_rotated` has dim1=1;
  `additional_inputs` has dim1=S.

Verdict on "collapses to S=1": **CONFIRMED with nuance.** The rotated branch is hard-coded to
snapshot 0 (`[:, 0]`), giving dim1=1. The `concatenate(dim=2)` at `:1033` will only succeed if
`additional_inputs` also has dim1=1, i.e. **S must equal 1 for the paired forward to run without a
shape error** (PyTorch broadcasts dims being concatenated only on the concat axis, not on the
others — dims 0,1 must be equal). So either (a) callers feed S=1 to the paired net, or (b) the only
non-error case is S=1. The doc's firm reading — "the paired/multi stack runs effectively with S=1
at the pair step" — is **correct as a hard constraint**: with S>1 line `:1033` raises a runtime
shape error, NOT a silent collapse. (One caveat: the doc says "only snapshot 0 is rotated" which is
literally true, but the cleaner statement is "paired forward requires S=1 or it throws".)
**Verdict: CONFIRMED.**

---

### Claim 4 — Config-only nets never in `load_model`

- `load_model` body fully read (`train_single_point.py:513-525`): only branches are
  `pass_through`, `beamformer`, `pairedbeamformer`, `multipairedbeamformer`,
  `trajmultipairedbeamformer`, else `raise ValueError` (`:525`).
- `grep -n "TransformerTimeModel|CrossAttentionNet|Separate1D2DNet|SignalMatrixNet"
  spf/scripts/train_single_point.py` → **no matches**. None imported or referenced there.
- Reachability in source module:
  - `Separate1D2DNet` only via `AllWindowsStatsNet(conv_type=="1d2d")` `:811-818`. CONFIRMED.
  - `CrossAttentionNet` only via `conv_type=="cross"` `:819-826`. CONFIRMED.
  - `TransformerTimeModel` only via `conv_type=="transformer"` `:827-835`. CONFIRMED.
  - `SignalMatrixNet` only via `"signal_matrix_net" in model_config` inside
    `SinglePointWithBeamformer` `:923-927`. CONFIRMED.

**Verdict: CONFIRMED.**

---

## (b) Ordinary line-number spot-check (hit rate)

| # | Doc claim | Source | Hit |
|---|---|---|---|
| 1 | `load_model` registry / ValueError `:525` | `:525` bare `raise ValueError` | ✓ |
| 2 | `output_ntheta` defaults via `check_and_load_ntheta` `:555-558` | `:555-558` exactly | ✓ |
| 3 | `BasicBlock` at `:30`; final `act(out+identity)` `:50-52` | `:30`, `:48-52` | ✓ |
| 4 | `FFNN` `:55`; ValueError bad norm `:108` | `:55`, `:108` `raise ValueError(f"Norm not implemented {norm}")` | ✓ |
| 5 | `PrepareInput` `:414`; concat dim=2 `:538-541` | `:414`, `:538-541` | ✓ |
| 6 | phase width `+3` `:444` | `:443-444` `if self.phase_input: self.inputs += 3` | ✓ |
| 7 | `SinglePointWithBeamformer` `:913`; FFNN ctor `:934-945` | `:913`, `:934-945` | ✓ |
| 8 | dead expand at `:969` (discarded), real use `:976-979` | `:969-971` expansion not assigned; `:973-982` used | ✓ |
| 9 | `signal_matrix_net` in-place `normalize_by[:,:,2]=pi` then `/normalize_by`, fresh `:962-967` | `:962-967` (normalize_by from max-reduction = fresh) | ✓ |
| 10 | Paired ctor FFNN `(output_ntheta+inputs)*2` `:1007-1018` | `:1007-1018` `inputs=(...["output_ntheta"]+...inputs)*2` | ✓ |
| 11 | `idxs=[0,0,1,1,...]` re-expand `:1046` | `:1046` `arange.reshape(-1,1).repeat(1,2).reshape(-1)` | ✓ |
| 12 | `PairedMultiPoint` `:1067`; time-flip aug prob 0.5 `:1116` | `:1067`, `:1116` `torch.rand(1)>0.5` | ✓ |
| 13 | `output_net = Linear(d_model, output_ntheta+1)` `:1106-1108` | `:1106-1108` | ✓ |
| 14 | `TrajPaired` `:1178`; skip default False `:1188`; latent `:1193` | `:1178`, `:1188`, `:1193` | ✓ |
| 15 | traj_net outputs `output_ntheta+1+2` `:1226-1239`; tx split `[ntheta:ntheta+2]` `:1301` | `:1233`, `:1300-1302` | ✓ |
| 16 | trajectory from snapshot 0 broadcast `:1284-1286` | `:1284-1286` `transformer_output[:, [0]].expand(...)` | ✓ |
| 17 | `SinglePointPassThrough` `:1312`; `{"single": empirical + w*0}` `:1319` | `:1312`, `:1319` | ✓ |

**Hit rate: 17/17 ordinary line-number claims correct (100%).** Plus the 4 priority-claim line
anchors all matched.

Minor wording nits (not counted as misses, line numbers correct):
- Doc `:283` says `multipaired_direct` is "the `[-1]` channel" — accurate; source `:1163`.
- Doc `:284-286` time-flip: says reversed by `system_timestamp[:,[-1]] - system_timestamp`
  `:1116-1121` — matches `:1118-1121`.

---

## (c) Missed / misclassified

1. **`vehicle_type` view-write missed (Claim 2).** Doc asserts only `phase_input` aliases the
   batch by reference and "most other branches build a fresh tensor … so those are safe"
   (`:129-130`). But `vehicle_type` `:528` `v = batch["vehicle_type"][..., None]` is a VIEW
   (unsqueeze), and `:530` `v[mask]=0` writes through to the batch tensor in training. This is a
   second in-place batch mutation the doc does not flag. (`rx_spacing`/`frequency`/`gains`/
   `beamformer`/`empirical` are genuinely fresh — those parts are right.)

2. **Claim 3 framing slightly soft.** The doc repeatedly hedges ("implies S==1", "Confirm callers
   run paired with S=1", "⚠️ unverified for all configs"). The stronger, correct statement: with
   S>1 the `concatenate(dim=2)` at `:1033` RAISES a shape error (dims 0,1 must match) — paired
   forward is a hard S=1 contract, not a silent collapse. The doc gets the conclusion right but
   under-commits.

No missed classes: all classes in the source module
(`BasicBlock, FFNN, PositionalEncoding, TransformerTimeModel, CrossAttentionNet, Separate1D2DNet,
PrepareInput, SignalMatrixNet, SkipConnect, AllWindowsStatsNet, NormP1Dim2,
SinglePointWithBeamformer, PairedSinglePointWithBeamformer, PairedMultiPointWithBeamformer,
TrajPairedMultiPointWithBeamformer, SinglePointPassThrough`) are accounted for. `PositionalEncoding`
(`:118`), `SkipConnect` (`:592`), `create_1dconv`/`create_2dconv`, and `AllWindowsStatsNet` (`:738`)
are not given their own entries, but `AllWindowsStatsNet` is referenced as the dispatcher and the
helpers are plumbing — acceptable omissions, not factual errors.

---

## (d) Grade: **A−**

All four priority forward-output contracts CONFIRMED including both shape spot-checks; config-only
reachability CONFIRMED; 17/17 ordinary line anchors correct. The single substantive correction is
Claim 2's "only phase_input is by-reference" distinction, which is REFUTED by `vehicle_type`
(`:528`/`:530`) being a view-write too — a real (if lower-severity) second in-place batch mutation
the doc explicitly claimed could not happen in the other branches. Claim 3 is correct but
under-committed (it is a hard S=1 error contract, not a silent collapse). Otherwise the document is
accurate and well-anchored.
