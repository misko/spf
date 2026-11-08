# Paired (stage-2) training path audit — `jul8_paired_3p7_thin_noblade_jun26_2026.yaml`

Read-only audit of the exact production stage-2 run
`python spf/scripts/train_single_point.py -c model_configs/jul8_paired_3p7_thin_noblade_jun26_2026.yaml`
(`model.name: pairedbeamformer` = `PairedSinglePointWithBeamformer`, `load_single: true`,
`detach: true`, `flip: false`, `double_flip: true`, `head_start: 0`, `save_on: val/paired_loss`,
lr 2e-4, wd 0.0, `scheduler_step: 12`, continuous scatter, S=1, B=256 sessions ⇒ 512 radio-rows).
Complements `claude_docs/reference/_stage1_audit/SUMMARY.md` (stage-1 findings not repeated;
stage-1 contracts reused where marked). Severity = misbehavior × reachability; confidence =
`✅ verified` (deciding lines read) vs `🔍 analysis` (reasoned, not executed) per
`claude_docs/DOCUMENTATION_GUIDE.md`.

**Production status (verified live):** the run is a **fresh** (non-resumed) process — parent PID
2524847 started Wed Jul 1 06:49, cmdline `python spf/scripts/train_single_point.py -c
model_configs/jul8_paired_3p7_thin_noblade_jun26_2026.yaml` with **no `--resume`** (read from
`/proc/2524847/cmdline`). This matters for issue P1 below.

## Verdict

**The paired path as configured and currently running is internally consistent and correct.**
All five audited areas check out for *this* config. The real hazards found are (a) a silent
semantic change **if the run is ever restarted with `--resume`** (P1), and (b) two config-inert /
cross-module interactions that make the run slightly different from what the config reads as
(P2, P3). Nothing found invalidates the checkpoints produced so far.

---

## Branch 1 — Checkpoint loading & freezing · ✅ correct for this run

**Flow (✅ verified):** `optimizer = AdamW(m.parameters(), lr=2e-4, wd=0.0)` is built **before**
checkpoint load (`train_single_point.py:1133`). `args.resume_from is None` and
`"checkpoint" in config["optim"]` ⇒ `load_checkpoint(..., force_load=False)` (`:1151-1158`).
With `load_single: true` (`:472-479`):

- `model.single_radio_net.load_state_dict(checkpoint["model_state_dict"])` (`:475`) — the stage-1
  `best.pth` `model_state_dict` is a full `SinglePointWithBeamformer` state dict; `single_radio_net`
  is exactly that class built from `model.single`, which is **field-for-field identical** to the
  stage-1 `model:` section (compared `jul8_single_3p7_thin_noblade_jun26_2026.yaml` vs `model.single`:
  depth 4 / hidden 512 / norm layer / bn true / identical `windows_stats_net`). ✅ strict load succeeds.
- **Mismatch behavior:** `load_state_dict` default `strict=True` ⇒ any key/shape mismatch raises
  `RuntimeError` at startup (fail-fast, good). No partial/lenient path exists here.
- **Freeze:** `param.requires_grad = False` on every `single_radio_net` param (`:476-477`).
  Returned epoch/step are `0,0` (`:479`) and land in `_start_epoch,_step` (unused, `:1152`) —
  training starts at epoch 0 with a fresh optimizer/scheduler (stage-1 optimizer state is *not*
  loaded — correct).
- **Do frozen params sit in the optimizer?** Yes — the optimizer was built from all params before
  freezing. **They still never move:** no autograd graph reaches them (see Branch 3/4), so
  `p.grad` stays `None` forever, and AdamW skips grad-`None` params entirely (no state, no weight
  decay applied — and wd=0.0 anyway). `optimizer.zero_grad(set_to_none=True)` (`:1328`) preserves
  this. 🔍 analysis (standard AdamW semantics; freeze lines ✅ verified).
- **Buffer drift?** None possible: `bn: true` + `norm: "layer"` selects the **LayerNorm** branch of
  `FFNN` (`single_point_networks.py:88-106`) and `BasicBlock` (`:43-45`); `windows_stats_net`
  has `norm: false`. **No BatchNorm ⇒ no running-stats buffers** to drift while the frozen net runs
  in train mode. ✅ verified.
- **Best-checkpoint contents:** `save_model` stores the **full** `model.state_dict()` (`:434`) —
  single weights included — plus optimizer/scheduler/config and copies `empirical_dist.pkl`
  (`:442-446`). `best.pth` written when `reported_losses["val/paired_loss"]` improves
  (`:1275-1292`); `--val` defaults True (`:1415-1419`) so this is active. ✅ verified.

**🔴 P1 (latent, resume-only): `--resume` silently unfreezes the single net.**
`load_checkpoint(force_load=True)` (`:1141-1149` → `:472` `if not force_load:` skips the freeze
branch) does a full `load_state_dict` and **never re-applies `requires_grad=False`**. A freshly
constructed model has all params trainable; `output["single"]` is the *undetached* single-net
output (`single_point_networks.py:1049`), and `single_loss` is in the backward (`train_single_point.py:871`).
So a resumed paired run **starts jointly training the previously frozen single net at lr 2e-4 with
front/back-folded single targets** — a silent regime change vs the fresh run. Reachability: only via
`--resume` / `--resume-from` (the current production process did **not** resume — ✅ verified via
`/proc`). Severity: high misbehavior × config/CLI-gated. Confidence: ✅ verified code path (not executed).
Note the household rule of restarting experiments via monitoring agents makes accidental `--resume`
plausible — if this run must be restarted, prefer re-running fresh from the stage-1 checkpoint, or
re-freezing manually.

Also noted: `FrozenModule` (`:400-413`) is dead — the `load_checkpoint` freeze is inline and the
`FrozenModule` wrap is commented out (`:478`). Consequence: the frozen net stays in **train mode**
during training (see P4). `elif config["model"]["load_paired"]` (`:480`) is not evaluated here
(`load_single` wins); `load_paired` defaults False via `load_defaults` (`train_utils.py:50`), so no
KeyError exposure for this config.

## Branch 2 — `double_flip` augmentation · ✅ correct & paired-coherent

**One draw per session, both receivers (✅ verified):** `v5spfdataset.__getitem__` paired branch
draws `double_flip = self.double_flip and (torch.rand(1) > 0.5).item()` **once** (`spf_dataset.py:1611`)
and passes it to `render_session` for **both** receivers (`:1612-1615`). Contrast `flip`, which is
drawn *inside* `render_session` per receiver (`:1357`) and is therefore guarded by
`assert config["model"]["name"] == "beamformer"` when `flip: true` (`train_single_point.py:1076-1078`);
this config has `flip: false`, so `flip_left_right` is always False. Val datasets never get either
flag (`load_val_dataset` doesn't pass them, `train_single_point.py:184-201`; ctor defaults False,
`spf_dataset.py:846-847`) — validation is unaugmented. ✅

**The math (✅ verified against source):** with `double_flip=True`, `render_session` applies, in order:

1. `craft_offset = craft_gt − gt_theta` captured **before** flipping (`:1384-1386`) — equals
   `rx_theta_in_pis·π` (pi-normed), per `get_craft_ground_truth_thetas` (`:1545-1553`; heading
   deliberately excluded — the inline comment at `:1386` mentioning `rx_head_in_pis` is stale).
2. LR-mirror step (`flip_left_right or double_flip`, `:1387-1389`): `φ → −φ`, `θ → −θ`.
3. Front/back step (`double_flip` only, `:1390-1395`): `θ → sign(θ)·π − θ`, **φ unchanged** —
   correct, since `φ = −sin(θ)·spacing·2π` (`:1512-1525`) and `sin(sign(θ)π−θ) = sin(θ)`.
4. Net effect: `θ → θ − sign(θ)·π` ≡ **θ ± π**, `φ → −φ` — a 180° rotation of the world about the
   array. Craft recomputed as `torch_pi_norm(θ' + craft_offset)` (`:1398-1400`) ⇒ **craft → craft + π**.
   The commented-out extra `craft += π` block (`:1401-1404`) would have *double-applied* the shift —
   correctly disabled. ✅
5. **Cross-receiver coherence:** each receiver's Δθ = −sign(θ_r)·π; −π ≡ +π (mod 2π) so both
   receivers' craft targets shift to the *same* `torch_pi_norm(craft+π)` regardless of their θ signs. ✅
   (Edge case: θ_r == 0.0 exactly ⇒ `sign(0)=0` ⇒ that receiver's craft doesn't shift while the
   other's does — a π-inconsistent pair. Measure-zero on float data. 🔍 P7.)

**Every consumed batch key transformed consistently (✅ verified, keyed to
`global_config_to_keys_used`, `train_utils.py:67-106`):**

| key | transform | line | consistent? |
|---|---|---|---|
| `y_rad`,`y_phi` | θ±π, −φ | `:1387-1395,1406-1407` | ✅ |
| `craft_y_rad` | +π | `:1398-1400` | ✅ |
| `y_rad_binned`,`craft_y_rad_binned` | recomputed **after** flip | `:1480-1485` | ✅ |
| `all_windows_stats` ch0 (mean phase) | negated (ch1/2 rotation-invariant) | `:1454` | ✅ |
| `mean_phase_segmentation` | negated | `:1455` | ✅ |
| `windowed_beamformer` | `.flip(dims=(3,))` | `:1501-1504` | ✅ θ-grid is `linspace(−π,π,65)` (`rf.py:1201`), symmetric under bin k↔64−k; front/back component needs no transform since the 2-elt beamformer is a function of sin θ |
| `weighted_beamformer` | flipped `:1500` | ✅ (not collated here — `beamformer_input: false`) |
| `empirical` | rebuilt from already-negated mean phase | `:1471-1478` (order ✅) | ✅ (and inert: `empirical_input: false`, used only for shape/device in `target_from_scatter`) |
| `rx_theta_in_pis`,`rx_heading_in_pis`,`gains`,`rx_lo`,`rx_wavelength_spacing`,`vehicle_type`,`sdr_device_type` | unchanged | — | ✅ correct: mount angle/hardware are invariant under rotating the *world* about the craft |
| `rx_pos_xy`,`tx_pos_xy` | negated | `:1467-1469` | ✅ for relative angle (d = tx−rx → −d); not model inputs on this path (only multipaired `tx_pos` loss uses them) |
| `abs_signal_and_phase_diff` | **not** flipped (`:1410-1417` warning) | — | unreachable: `signal_matrix` in `skip_fields` (`signal_matrix_input: false`) ✅ |

**In-place safety (✅ verified):** the negations at `:1389,:1454-1455,:1468-1469` operate on fresh
tensors — `get_session_idxs` returns an `np.arange` array (`:273-275`) so tensor advanced-indexing
copies (`:1371,:1441,:1457-1465`), and zarr reads materialize new arrays in `data_from_precomputed`
(`:278-325`). No flip-flopping of cached data across epochs.

**Why this is coherent end-to-end (🔍 analysis, confirmed by construction):** for the frozen single
net, double_flip is exactly an LR-mirror of its inputs+targets (the front/back step is invisible to
a 2-element array) — the stage-1 net was trained flip-augmented, so its estimate mirrors accordingly.
In craft frame the mirrored single distribution has peaks at `{craft+π, 2·rxθ−craft}`; the true peak
`craft+π` coincides across both radios (matching the shifted `craft_y_rad` target) while the ghost
peaks differ per radio — precisely the ambiguity the paired head is meant to resolve.
**Test coverage: none** — the only double_flip references in `tests/` are `double_flip: False` in the
two test configs; `test_paired_rotate_dist.py` covers the frame rotation (Branch 3) but not double_flip.

## Branch 3 — Paired forward · ✅ correct

`PairedSinglePointWithBeamformer` (`single_point_networks.py:997-1053`), all ✅ verified:

- `single_radio_estimates = self.single_radio_net(batch)["single"]` (`:1021`) — (B,1,65),
  L1-normalized, run under `m.train()` during training (see P4).
- **Rotation into craft frame:** `rotate_dist(single_radio_estimates[:,0],
  torch_pi_norm(batch["rx_theta_in_pis"]·π)).unsqueeze(1)` (`:1024-1028`). Per-row `rx_theta_in_pis`
  ⇒ each radio rotated by its **own** mount angle. `rotate_dist` (`rf.py:1493-1556`) is a circular
  shift by `+r` with linear interpolation and re-normalization: mass at θ moves to θ+r, bin width
  2π/65 — matching the target grid of `target_from_scatter` (bin j ↔ (j−32)·2π/65, spacing 2π/65,
  `train_single_point.py:553-561`) and `to_bin` (`utils.py:250-255`). Since 65·(2π/65)=2π exactly,
  the modulo wrap is exact; the grid's absolute offset cancels (only differences enter). So
  `p_craft(θ) = p_radio(θ − rxθ)` ✅ — the same rotation the trainer's own `single_craft_loss`
  diagnostic uses (`train_single_point.py:809-822`), and **empirically validated** by
  `tests/test_paired_rotate_dist.py` (real windowed beamformers rotated by `rx_theta_in_pis·π`
  multiplied across radios peak at craft ground truth). Sign/radio assignment correct.
- **Invariant inputs:** a second `PrepareInput` built from `model_config["single"]` (`:1004`)
  contributes 6 scalars (gains 2 + spacing + freq + vehicle + sdr); FFNN input =
  (65+6)·2 = **142** matching `(output_ntheta + prepare_input.inputs)*2` (`:1007-1009`). ✅
- **Detach branch:** `joined_input_detached_or_not = detach_or_not(joined, self.detach)` (`:1037`,
  `detach: true`) cuts the graph **before** the paired FFNN ⇒ only
  `paired_single_point_with_beamformer_ffnn` receives gradients. Combined with the requires_grad
  freeze this is belt-and-braces for the fresh run (and the *only* protection on a resumed run — see P1).
- **Pairing/concat:** `[joined[::2], joined[1::2]]` concat on dim 2 (`:1039-1044`) — row order is
  guaranteed by `v5_collate_keys_fast` (`spf_dataset.py:113-127`): `[x[key] for paired_sample in
  batch for x in paired_sample]` with `__getitem__` returning `[r0, r1]` per session (`:1612-1615`),
  so rows 2i/2i+1 are the same session's two radios. ✅
- **Output keys/shapes:** `"single"` (B,1,65) undetached; `"paired"` =
  `normalize(|x[idxs]|, p=1, dim=2)` where `idxs` duplicates each fused row for both radio rows
  (`:1046-1052`) ⇒ (B,1,65), each session's paired estimate appears twice (matching per-row
  `craft_y_rad` targets, which agree across the pair up to the dataset's <0.2 rad receiver
  consistency warning, `spf_dataset.py:1563-1569`). ✅
- S=1 assumption (`[:,0]` at `:1026`): holds (`train_snapshots_per_session: 1`); S>1 would crash at
  the `:1033` concat (shape mismatch), not silently corrupt. 🔍 latent, config-gated.

## Branch 4 — Paired loss, target & gating · ✅ correct

`compute_loss` (`train_single_point.py:735-994`), called with `single=True`,
`paired = epoch >= head_start` = `epoch >= 0` = **always True** (train `:1315`, val `:1016`):

- **Paired target (✅ verified):** `scatter_fn(batch, y_rad=batch["craft_y_rad"][...,None],
  y_rad_binned=batch["craft_y_rad_binned"][...,None], ...)` (`:902-909`) **without** `single=` ⇒
  default `single=False` ⇒ **no front/back fold** — correct: two radios at different mount angles
  disambiguate front/back, so the craft target should be unimodal. Continuous scatter ignores
  `y_rad_binned` (`scatter_k: 21` inert, stage-1 caveat), but the key exists in the batch
  (`train_utils.py:81`; computed at `spf_dataset.py:1483-1485`) so no KeyError. `paired_loss = mse_loss`
  (SSE over 65 bins, both sides L1-normed) added at weight 1.0 (`:911-912`).
- **`single_loss` during paired training (✅ verified):** still computed (`:859-868`, folded
  `single=True` target on per-radio `y_rad`) and **added to the backpropped loss** (`:871`) — but it
  is a **constant**: `output["single"]` comes exclusively from frozen (`requires_grad=False`) params
  and non-grad inputs, so it carries no graph. Zero gradient contribution; cost is one no-op term
  plus its target construction. It *does* inflate the logged `train/loss` and `val/loss` (save_on
  uses `val/paired_loss` specifically, so best-checkpoint selection is unaffected). ✅
- The four `no_grad` diagnostics (`single_target_aligned_loss` `:759-783`, `single_craft_loss`
  `:799-822`, `single_plus_rand_loss` `:824-845`, `single_loss_old` `:847-857`) and
  `uniform_loss_{single,paired}` (`:879-886,:914-921`) are log-only. Their rotation matches the
  paired forward's (craft frame, heading excluded). ✅
- **head_start footgun (latent, 🔍):** with `head_start > 0` *and* a frozen single net, epochs before
  `head_start` would make `loss` a constant tensor ⇒ `loss.backward()` raises
  `RuntimeError: element 0 ... does not require grad`. Not reachable here (`head_start: 0`).
- **Precedence nit (latent, 🔍):** `if paired and "paired" in output or "multipaired" in output:`
  (`:901`) parses as `(paired and "paired" in output) or ("multipaired" in output)` — with
  `paired=False` and a multipaired model the target is built (referencing `output["paired"]`, which
  those models do emit) and then unused. Wasted compute only; dead on this path.

## Branch 5 — Val metrics, splits, keys, dead/inconsistent branches

- **`val/paired_loss` construction (✅ verified):** `run_val_on_dataloader` accumulates
  `loss_d["paired_loss"].item()` per batch (`:997-1030`); `WNBLogger.log` means the list and
  prefixes → `reported_losses["val/paired_loss"]` (`:714-732`); `save_on` lookup at `:1275-1281`.
  Present on every val pass (paired gate always true, `"paired"` always in output). `--val`
  defaults True. ✅
- **Per-dataset val splits (✅ verified):** alternate val loaders keyed
  `{sdr_device_type}:{collector}:{emitter}:{wavelength}:rf_{bw}:{routine}` (`:274-281`), subsampled
  by `val_subsample_fraction: 0.5` (`:296-306`), logged under `val_{key}/...` (`:1271-1273`) — never
  used for checkpoint selection. Val set from `apr17_val_nosig_noroverbounce.txt` with
  `val_holdout_fraction` ignored (explicit `val_paths` branch `:154-156`; stage-1 F5 resolution
  applies unchanged). Val datasets have `double_flip=False` ⇒ metrics on unaugmented data. ✅
- **Key existence:** `empirical` force-appended to collate keys (`:324-325`) and produced (dataset
  has `empirical_data_fn`; `spf_dataset.py:1471-1478`); `craft_y_rad_binned` produced `:1483-1485`
  and collated (`train_utils.py:81`). ✅
- **Dead-but-present on this path (✅ verified):** `FrozenModule` (`:400-413`); commented-out
  model-rebuild block in `load_checkpoint` (`:455-469`); `model.single.detach: true` (read by
  nothing — `SinglePointWithBeamformer` never reads `detach`); `scatter_k: 21` (continuous path);
  `torch.zeros(...)` discarded allocation in `target_from_scatter` (`:543-545`, stage-1);
  `optim.checkpoint_every` checkpoints are additionally written every 5000 steps (`:1346-1356`) —
  functioning, not dead. GradScaler/autocast no-ops (`amp: false`).

---

## Issues table

| # | Issue | Where | Severity (misbehavior × reachability) | Confidence | Affects current run? |
|---|---|---|---|---|---|
| P1 | `--resume` does not re-freeze `single_radio_net`: resumed paired run silently trains the single net via `single_loss` (undetached `output["single"]`, all-params optimizer, fresh `requires_grad=True`) | `train_single_point.py:472` (freeze skipped when `force_load=True`), `:1141-1149`; `single_point_networks.py:1049` | 🔴 high × **resume-only** (CLI-gated; auto-restart tooling makes it plausible) | ✅ verified path (not executed) | **No** (run is fresh, verified via `/proc/2524847/cmdline`) — but do NOT restart with `--resume` |
| P2 | `model.input_dropout: 0.2` is **dead** for `pairedbeamformer` — class never reads it; the paired `PrepareInput` is built from `model_config["single"]` ⇒ effective input_dropout 0.1 | `single_point_networks.py:997-1018` (reads only detach/depth/hidden/block/norm/bn/dropout), `:1004` | 🟡 config-inert / expectation mismatch × always | ✅ verified | Yes — run has 0.1, not the configured 0.2 |
| P3 | `vehicle_type` double-dropout: single net's `PrepareInput` **mutates `batch["vehicle_type"]` in place** (view + masked assign), then the paired `PrepareInput` re-reads and re-drops ⇒ ~19% correlated zeroing of the paired head's vehicle_type feature (train only; val clean). The paired path realizes the second-consumer hazard stage-1 flagged (KNOWN_ISSUES #34) | `single_point_networks.py:527-531` (only in-place field — others allocate via arithmetic), order `:1021` → `:1031` | 🟡 mild (one scalar feature, train-time noise) × every train step | ✅ verified mechanism; 🔍 magnitude | Yes — benign-leaning but real |
| P4 | Frozen single net runs in **train mode** during paired training (`FrozenModule`+`eval()` deliberately abandoned, `:478`): window_shrink 0.5 / window_dropout 0.25 / window_shuffle 0.15 / input_dropout 0.1 stay active inside the frozen net ⇒ paired head trains on stochastic single estimates, evals on deterministic ones | `train_single_point.py:400-413,:478,:1297`; `single_point_networks.py:877-894,:471-531` | 🟡 design choice (dropout-style augmentation), undocumented × every train step | ✅ verified | Yes — looks intended; no BatchNorm so no buffer corruption |
| P5 | `head_start > 0` + frozen single ⇒ pre-head_start epochs have constant loss ⇒ `backward()` RuntimeError at first step | `train_single_point.py:871,:1315,:1323` | 🟢 crash × config-gated (head_start 0 here) | 🔍 analysis | No |
| P6 | Operator precedence `paired and "paired" in output or "multipaired" in output` builds an unused target when `paired=False` for multipaired models | `train_single_point.py:901-909` | 🟢 wasted compute × not this path | ✅ verified parse | No |
| P7 | double_flip with θ_r == 0.0 exactly: `sign(0)=0` ⇒ that receiver skips the ±π shift ⇒ cross-receiver craft targets differ by π for that sample | `spf_dataset.py:1390-1395` | 🟢 measure-zero on float data | 🔍 analysis | Effectively no |
| P8 | **No test** exercises `double_flip=True` (test configs pin it False); the paired-coherence contract of Branch 2 is verified only by this reading | `tests/model_configs/*.yaml:26-27` | 🟡 coverage gap | ✅ verified | Risk carrier, not a bug |
| P9 | Stale comment: `craft_offset` annotated "(rx_theta_in_pis + rx_head_in_pis)·π" but heading is excluded from craft frame | `spf_dataset.py:1384-1386` vs `:1545-1553` | 🟢 doc/comment nit | ✅ verified | No |

## Doc cross-check

- `reference/spf/model_training_and_inference/single_point_networks.md:81-86` — **inaccurate**:
  claims "the active single/paired configs pass `bn=False` … so the norm slots are `Identity`".
  The production jul8 configs pass `bn: true, norm: layer` ⇒ **LayerNorm active** in every FFNN/
  BasicBlock slot. The doc's shape-safety conclusion happens to survive (LayerNorm is last-dim,
  3-D-safe) but the stated premise is wrong. Its `:241` "confirm callers run paired with S=1 ⚠️"
  is now confirmed for production (S=1).
- `reference/spf/scripts/train_single_point.md` — loss table (`:29-34`), `load_single` freeze +
  epoch/step-0 description (`:298-302`), dead `FrozenModule` (`:319-320`) all **accurate**. Gap: does
  not mention the P1 resume/unfreeze asymmetry, nor that `single_loss` is a zero-gradient constant
  in the frozen-paired regime.
- `reference/spf/dataset/v5spfdataset.md` — `render_session`/`__getitem__` double_flip description
  (`:177-182,:204-215`) and "flip/double_flip untested" (`:198,:427`) **accurate**; consistent with
  this audit.
- `_stage1_audit/SUMMARY.md` — the "vehicle_type mutation benign (single consumer)" caveat is
  **superseded on the paired path** by P3 (second consumer exists inside the same forward).
