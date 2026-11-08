# Adversarial Verification — training.md inventory

Verifier disposition: skeptical. READ-ONLY. Each 🔴/🧪 claim opened against source.

## (a) 🔴 broken + key 🧪 claims → verdict

| Claim (inventory) | Cited loc | Verdict | Evidence (quoted source) |
|---|---|---|---|
| `load_optimizer` cosine branch is `pass` → `train_scheduler` unbound → `UnboundLocalError` at return | train_single_point.py:377/389 | **CONFIRMED** | L389 `elif optim_config["scheduler"] == "cosine":` … L393 `pass`; L397 `return optimizer, train_scheduler`. In the cosine path `train_scheduler` is never assigned (only assigned inside the `step` branch L384). Real `UnboundLocalError`. |
| …and a config actually uses `cosine` | configs | **CONFIRMED + STRENGTHENED** | `model_configs/dec3_single_config_..._bigger_4x.yaml:75: scheduler: cosine`. So the broken path is reachable, not merely theoretical. Default is `step` (train_utils.py:33 `get_key_or_set_default(config,"optim/scheduler","step")`), and only this 1 of the model_configs uses cosine. |
| `get_inference_on_ds_noexceptions` calls inner fn but never returns its result (returns None on success) | inference.py:161 | **CONFIRMED** | L173-185 `try: get_nn_inference_on_ds_and_cache(...)` with no assignment and no `return`; except logs. Function implicitly returns None. Sole external caller `create_inference_cache.py:61` wraps it in `partial` and runs via `Pool.imap`, discarding return — pure cache side-effect, so the missing return is harmless in practice (inventory says exactly this). |
| `baseline_algorithm` `steps==-1` default: shape computed but discarded → `np.arange(-1)` empty loop → empty result | baseline_algorithm.py:27 | **CONFIRMED** | L29-30 `if steps == -1: session["beam_former_outputs_at_t"].shape[0]` — expression evaluated, result discarded, `steps` not reassigned. L31 `for idx in np.arange(steps)` → `np.arange(-1)` is empty. Bug is real; masked because the lone caller passes explicit `steps`. |
| `TransformerTimeModel` never in load_model registry; reachable only via `conv_type="transformer"` | single_point_networks.py:154 | **CONFIRMED (config-only, not strictly dead)** | Class def L154; instantiated at L828 inside `AllWindowsStatsNet` (`self.conv_net = TransformerTimeModel(`). Not in `load_model`. Reachable via conv_type string only. |
| `CrossAttentionNet` same | :234 | **CONFIRMED (config-only)** | Def L234; instantiated L820 inside AllWindowsStatsNet. |
| `Separate1D2DNet` same | :327 | **CONFIRMED (config-only)** | Def L327; instantiated L812 inside AllWindowsStatsNet. |
| `check_phi_error.py` hardcodes `segmentation_version=3.5`, ignoring CLI `--segmentation-version` | check_phi_error.py:42 | **CONFIRMED** | Parser adds `--segmentation-version` (L10-14, required); but L42 `segmentation_version=3.5,` is a literal. `args.segmentation_version` is never used. `if True:` wraps load (L32). |
| `segmentation_metrics.py` hardcodes `Pool(8)`, `--parallel` parsed but unused | seg_metrics.py:80 | **CONFIRMED** | L80 `with Pool(8) as p:`. Parser defines `-p/--parallel` (L28, default 8). `args.parallel` never referenced in `__main__`. |
| `frac_to_theta` unused (no callers) | baseline:17 | **CONFIRMED** | Repo grep: only the def. |
| `lines_to_points` unused | baseline:170 | **CONFIRMED** | Only the def. |
| `uniform_preds` unused | train:528 | **CONFIRMED** | Only the def. |
| `FrozenModule` defined but unused | train:400 | **CONFIRMED** | Only other hit is commented L478 `# model.single_radio_net = FrozenModule(...)`. |
| `preload_dataset` only in commented preload code | train:57 | **CONFIRMED** | Only other hit is commented L209. |
| `weighted_beamformer` not called in this file (likely dataset) | train:600 | **CONFIRMED** | The train-script function (L600) has no caller in that file. The many other matches are a distinct dataset-level field/zarr key + a `batch["weighted_beamformer"]` read in PrepareInput — different thing. Inventory's hedge is accurate. |
| `PrepareInput.prepare_input` in-place `v[mask]=0` on batch views during training | single_point_networks.py:~470 | **CONFIRMED** | `prepare_input` at L470; training-only dropout builds `dropout_mask` (L478) and zeroes in place at L500/505/510/515/520/525/530 `v[dropout_mask[i]] = 0`. |
| `SignalMatrixNet` reachable only via optional `signal_matrix_net` config; 4 in-ch, mean over time | :565 | **CONFIRMED** | Ctor `nn.Conv1d(4, ...)` L571; `forward` returns `self.conv_net(x).mean(axis=2)` L589 → `(B, outputs)`. |
| `cdf`/`normal_correction_for_bounded_range`/`sigmoid_dist` unused in file | :16/:21/:544 | **PLAUSIBLE (not exhaustively re-grepped)** | Defs present at cited lines; commented `@torch.jit.script`. Low-risk classification. |

No 🔴 claim was refuted. One is **harmless-in-practice** (`get_inference_on_ds_noexceptions` missing return — caller discards anyway), which the inventory already states.

## (b) Line-number hit rate

Spot-checked symbol definitions at cited lines (grep `^def`/`^class`):

baseline_algorithm.py: get_top_n_peaks(5), frac_to_theta(17), line_to_mx(21), baseline_algorithm(27), get_top_n_points(49), boundary_point(129), lines_to_points(170) — 7/7.

single_point_networks.py: BasicBlock(30), FFNN(55), PositionalEncoding(118), TransformerTimeModel(154), CrossAttentionNet(234), Separate1D2DNet(327), PrepareInput(414/470 method), SignalMatrixNet(565), SkipConnect(592), create_1dconv(604), create_2dconv(695), AllWindowsStatsNet(738), SinglePointWithBeamformer(913), PairedSinglePointWithBeamformer(997), NormP1Dim2(1056), PairedMultiPointWithBeamformer(1067), TrajPairedMultiPointWithBeamformer(1178), SinglePointPassThrough(1312), warn_ntheta(551), check_and_load_ntheta(555), sigmoid_dist(544) — all exact.

inference.py: all 8 funcs (23/42/56/92/126/161/190/255) — exact.
train_single_point.py: 14 sampled (36/45/377/400/416/513/528/539/608/612/735/997/1033/1364) — exact.

**Hit rate: 100%** (≈50 symbols checked across the priority + spot-check sample; every cited line matched the definition).

### Forward shape spot-checks (3)
- `SignalMatrixNet.forward` → `.mean(axis=2)` over conv of 4-in-ch ⇒ `(B, outputs)`. **MATCHES**.
- `AllWindowsStatsNet.forward` → returns `{"all_windows_embedding": output.reshape(...)}` and conditionally `r["output_phi"] = ...` (L~904-911). **MATCHES** the `{"all_windows_embedding"[, "output_phi"]}` claim.
- `TransformerTimeModel` docstring confirms `(batch, 68, 256)` in, `output_dim=12` out (L156-166). **MATCHES** the hardcoded-68/12 claim.

## (c) Missed / misclassified

- None material. nn.Module classes in single_point_networks.py are all enumerated. The "config-only vs dead" distinction for `TransformerTimeModel`/`CrossAttentionNet`/`Separate1D2DNet` is handled correctly: the Notable section (L68) and the cluster summary (L265-270) explicitly call them "reachable only via AllWindowsStatsNet.conv_type", i.e. config-only, not strictly dead. The 🧪 marks are appropriate.
- The cluster-summary 🔴 count of "4" lumps a 4th item that is really a mutation-risk umbrella (`target_from_scatter`/PrepareInput) rather than a hard crash. Minor: only 3 are genuine bugs; the 4th is a flagged smell. Not an error, just looser labeling.
- `get_inference_on_ds_noexceptions` 🔴 is technically correct but operationally benign (caller discards return) — inventory already notes this, so not a miss.

## (d) Overall grade: **A**

Every 🔴 and every checked 🧪 claim verified against source with exact line evidence; line-number hit rate 100%; forward-shape claims accurate; config-only-vs-dead nuance handled correctly. The cosine-scheduler bug is even understated — a real config triggers it. Only nit is the slightly loose "4 bugs" tally in the summary.
