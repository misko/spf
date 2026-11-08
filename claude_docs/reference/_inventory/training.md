# PASS-1 Surface Inventory — Training & Inference Cluster

Read-only inventory of the SPF model-training, model-definition, simulation, and
script files. Line numbers verified against source at inventory time.

Status legend: ✅ working/normal · ⚠️ caveat/smell · 🔴 likely bug/broken ·
🧪 experimental/unused · ⚠️unverified runtime-unconfirmed claim.

---

## baseline_algorithm.py
`spf/model_training_and_inference/baseline_algorithm.py` — classical (non-NN)
direction-of-arrival → 2D point-localization baseline operating on simulation
"session" dicts. Consumed by `spf/plot/plot.py`.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_top_n_peaks` | 5 | func | Find indices of top-N peaks in a 1D beamformer array, suppressing a window around each found peak | `bf: np.ndarray (B,)`, `n=2`, `peak_size=15` | `np.ndarray (n,) int` peak indices | Copies `bf` (no mutation of arg); none | ✅ |
| `frac_to_theta` | 17 | func | Map fraction [0,1] → angle [-π,π] | `fracs: np.ndarray/float` | same shape, radians | none | 🧪 unused (no callers found in repo) |
| `line_to_mx` | 21 | func | Convert a (point, direction) line to (intercept, slope, point, dir-vec) | `line: ((x,y),((m_x,m_y),))` | tuple `(b, m, (x,y), (m_x,m_y))` | none; `1e-9` guards div-by-zero | ✅ |
| `baseline_algorithm` | 27 | func | Top-level baseline: per-timestep peak θ → lines → top-N points | `session: dict`, `width: int`, `steps=-1` | `(final_points, imgs, points_per_line)` from `get_top_n_points` | none | 🔴 `steps==-1` branch computes shape but never assigns `steps`; `np.arange(-1)` → empty loop → returns empty. Caller in plot.py always passes `steps=idx`, masking bug |
| `get_top_n_points` | 49 | func | Greedy assignment of lines to N intersection points via rasterized voting images | `line_representations: list`, `n`, `width`, `threshold=3`, `mn=4`, `lines_per_step=2` | `(final_points: np.ndarray, imgs: (n,W+1,W+1), points_per_line: (L/2,2))` | none; uses `skimage.draw.line` | ⚠️ dense index/geometry logic; `line_to_point_distances` dtype int truncates float distances |
| `boundary_point` | 129 | func | Compute where a line exits the [0,width]² box given slope sign | `b,m,point,width,mvec` | `(x,y)` boundary point | raises `AssertionError` on invalid/zero direction | ⚠️ many branches; asserts can fire |
| `lines_to_points` | 170 | func | Randomly sample line-pair intersections with validity filters | `lines: list`, `t: int` | `list[list[(x,y)]]` per line | RNG seeded per-line (`12345+idx*1337`) deterministic | 🧪 unused (no callers found) |

Notable:
- 🔴 `baseline_algorithm` `steps==-1` default is dead/broken (line 29–30: `.shape[0]` evaluated but discarded). Only works because the sole caller passes explicit `steps`.
- Module operates on legacy v1/Task2 "session" dicts (`beam_former_outputs_at_t`, `thetas_at_t`, `detector_*_at_t`), not the v5 dataset path.
- `frac_to_theta` and `lines_to_points` appear dead.

---

## models/single_point_networks.py
`spf/model_training_and_inference/models/single_point_networks.py` — all NN
module definitions for single-point / paired / multi-point beamformer models plus
input-prep and helper conv builders. Registry consumers in `train_single_point.load_model`.

| Symbol | Line | Kind | Purpose | Inputs (forward) | Outputs (forward) | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `cdf` | 16 | func | Normal CDF (erf) | `mean, sigma: Tensor`, `value: float` | Tensor | none | 🧪 unused in this file (commented jit); ⚠️unverified external use |
| `normal_correction_for_bounded_range` | 21 | func | Renorm factor for truncated normal on [-max_y,max_y] | `mean, sigma, max_y` | Tensor | `assert max_y>0` | 🧪 unused here |
| `BasicBlock` | 30 | nn.Module | 2-layer residual MLP block w/ optional BN/LN + dropout | ctor: `hidden, act, bn=True, norm="batch", dropout=0.0`; fwd `x:(...,hidden)` | `(...,hidden)` | none | ✅ |
| `FFNN` | 55 | nn.Module | Configurable MLP (optionally stacks `BasicBlock`s); batch or layer norm | ctor: `inputs,depth,hidden,outputs,block,bn,norm,act,dropout`; fwd `x` | `(...,outputs)` | raises `ValueError` on bad norm | ✅ core building block |
| `PositionalEncoding` | 118 | nn.Module | Standard sinusoidal PE | ctor `d_model,max_len=10000`; fwd `x:(B,L,d)` | `(B,L,d)` | registers `pe` buffer | ✅ (only used by `TransformerTimeModel`) |
| `TransformerTimeModel` | 154 | nn.Module | Transformer encoder over time; **hardcoded input dim 68, output 12** | fwd `x:(B,68,256)` | `(B,12)` | none | 🧪⚠️ reachable only via `AllWindowsStatsNet conv_type="transformer"`; hardcoded 68/12 likely mismatches configured `outputs` |
| `CrossAttentionNet` | 234 | nn.Module | 1D path (first 3 ch) as query, 2D path (rest) as KV, cross-attn | fwd `x:(B,68,256)` | `(B,output_channels)` | none | 🧪 reachable only via `conv_type="cross"`; comment "chatGPT" |
| `Separate1D2DNet` | 327 | nn.Module | Split 3-ch (1D conv) + 65-ch (2D conv), concat, project | fwd `x:(B,68,256)` | `(B,output_channels)` | none | 🧪 reachable only via `conv_type="1d2d"` |
| `PrepareInput` | 414 | nn.Module | Assemble & normalize the FFNN input feature vector from a batch dict; computes `self.inputs` count | ctor `model_config, global_config`; `prepare_input(batch, additional_inputs=[])` | concatenated Tensor `(B,S,inputs+extra)` | **mutates batch tensors in-place via dropout mask assignment** (`v[mask]=0`) during training; randomness (training dropout) | ⚠️ in-place writes into input `v` (views of batch tensors); training-only dropout |
| `sigmoid_dist` | 544 | func | Sigmoid then L1-normalize | `x` | normalized Tensor | none | 🧪 unused in file |
| `warn_ntheta` | 551 | func (lru_cache 1) | One-time warning about missing `output_ntheta` | none | None | logging.warning | ✅ |
| `check_and_load_ntheta` | 555 | func | Default `output_ntheta` from global config | `model_config, global_config` | None | **mutates `model_config`** | ✅ |
| `check_and_load_config` | 561 | func | Wrapper calling `check_and_load_ntheta` | same | None | mutates `model_config` | ✅ |
| `SignalMatrixNet` | 565 | nn.Module | 7-layer strided 1D conv over raw signal+phase (4 in-ch) | ctor `k=11,hidden_channels=32,outputs=8`; fwd `x:(B,4,L)` | `(B,outputs)` (mean over time) | none | 🧪 reachable only if `signal_matrix_net` in model_config (needs `signal_matrix_input`) |
| `SkipConnect` | 592 | nn.Module | Residual wrapper (adds input if shapes match) | ctor `internal_module`; fwd `x` | module(x) [+x] | none | ✅ |
| `create_1dconv` | 604 | func | Build a `Sequential` of SkipConnect 1D conv blocks w/ 2 stride-2 downsamples | many conv params | `nn.Sequential` | none | ✅ |
| `create_2dconv` | 695 | func | Build a `Sequential` of SkipConnect 2D conv blocks | `hidden_channels,padding,k,n_layers,act` | `nn.Sequential` | none | ✅ (used by `conv_type="2d"`) |
| `AllWindowsStatsNet` | 738 | nn.Module | Phase/window-stats encoder (1d/2d/1d2d/cross/transformer); optional φ head; window aug (shrink/shuffle/dropout) | ctor (many, see Notable); fwd `batch: dict` | `{"all_windows_embedding": (B,S,outputs)[, "output_phi": (B,S,1)]}` | training-time random window aug (randomness); logging.info on init | ⚠️ large; many conv_type branches; `output_channels=12` hardcoded in transformer branch |
| `SinglePointWithBeamformer` | 913 | nn.Module | Per-radio single-point estimator: PrepareInput + optional sub-nets → FFNN → L1-normalized dist | ctor `model_config, global_config`; fwd `batch` | `{"single": (B,S,output_ntheta)[, "output_phi"]}` | mutates `model_config` (check_and_load); in-place batch normalize math | ✅ primary "beamformer" model |
| `PairedSinglePointWithBeamformer` | 997 | nn.Module | Pair two radios (rotate single dist to craft frame) → FFNN → paired dist | ctor; fwd `batch` | `{"single", "paired": (B,S,output_ntheta)}` | `detach` of joined input (config); even/odd radio interleave | ✅ "pairedbeamformer" |
| `NormP1Dim2` | 1056 | nn.Module | L1-normalize abs over dim=2 | fwd `x` | normalized | none | ✅ |
| `detach_or_not` | 1061 | func | Conditionally detach a tensor | `x, detach` | Tensor | none | ✅ |
| `PairedMultiPointWithBeamformer` | 1067 | nn.Module | Transformer over time sequence of paired dists + skip; outputs multipaired dist + direct angle | ctor; fwd `batch` | adds `{"multipaired", "multipaired_direct"}` | training-time random time-flip (randomness); input dropout | ✅ "multipairedbeamformer" |
| `TrajPairedMultiPointWithBeamformer` | 1178 | nn.Module | Transformer → latent trajectory → FFNN traj_net; predicts dist + direct + tx-xy | ctor (adds `latent, traj_layers, traj_hidden, pred_xy`); fwd `batch` | adds `{"multipaired","multipaired_direct"[,"multipaired_tx_pos"]}` | none beyond submodules | 🧪 "trajmultipairedbeamformer" (registered; newer/experimental) |
| `SinglePointPassThrough` | 1312 | nn.Module | Pass through `batch["empirical"]` as the prediction (single dummy param) | ctor; fwd `batch` | `{"single": empirical + w*0}` | none | ✅ "pass_through" baseline |

Notable:
- **Model registry (`load_model`)** maps configs → 5 classes: `pass_through`→`SinglePointPassThrough`, `beamformer`→`SinglePointWithBeamformer`, `pairedbeamformer`→`PairedSinglePointWithBeamformer`, `multipairedbeamformer`→`PairedMultiPointWithBeamformer`, `trajmultipairedbeamformer`→`TrajPairedMultiPointWithBeamformer`.
- **Never directly registered / only conv_type-selected inside `AllWindowsStatsNet`:** `TransformerTimeModel` (`"transformer"`), `CrossAttentionNet` (`"cross"`), `Separate1D2DNet` (`"1d2d"`), plus `create_2dconv` (`"2d"`). These carry hardcoded dims (68 input, 12 output) and "chatGPT" provenance comments → flagged 🧪.
- `SignalMatrixNet` reachable only when `signal_matrix_net` present in model_config (and `signal_matrix_input` global).
- `cdf`, `normal_correction_for_bounded_range`, `sigmoid_dist` appear unused within this module (commented-out `@torch.jit.script`).
- `PrepareInput.prepare_input` does **in-place zeroing** of (views into) batch tensors during training dropout — a mutation side effect worth a deep dive.
- `AllWindowsStatsNet` ctor key args: `k,hidden_channels,outputs,output_phi,norm,norm_size,n_layers,windowed_beamformer,normalize_windowed_beamformer,nthetas,act,network_inputs,conv_type,hidden_channels_2d,n_layers_2d,window_{shrink,shuffle,dropout,fraction}`.

---

## models/single_point_networks_inference.py
`spf/model_training_and_inference/models/single_point_networks_inference.py` —
inference helpers: load model+config from checkpoint, run on dataset, cache to
npz (local/B2). Consumed by `create_inference_cache.py` and plotting/eval.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `load_model_and_config_from_config_fn_and_checkpoint` | 23 | func | Build model from config yaml + load checkpoint weights | `config_fn, checkpoint_fn, device=None` | `(model, config)` | reads files (torch.load), GPU move, mutates `config["optim"]` | ✅ |
| `single_example_realtime_inference` | 42 | generator | Stream model outputs over a realtime dataset | `model, global_config, optim_config, realtime_ds` | yields model output dicts; returns None | GPU; `model(...)` called twice/sample (perf) | ⚠️ double inference per sample (line 51 yield + 52 append); `outputs` list discarded |
| `single_example_inference` | 56 | func | Run model per-example over a v5 dataset, reshape to (sessions,radios,snap,θ) | `model, global_config, datasets_config, optim_config` | `{"single"[, "paired"]}` reshaped | `model.eval()`; builds dataset (file I/O); GPU | ✅ |
| `dataloader_inference` | 92 | func | Run model over val dataloader; reshape results | `model, global_config, datasets_config, optim_config, model_config` | `{"single"[, "paired"]}` | `model.eval()`; dataloader I/O; GPU | ✅ |
| `convert_datasets_config_to_inference` | 126 | func | Produce an inference-mode datasets_config (stride 1, no flip/shuffle, train_on_val) | `datasets_config, ds_fn, precompute_cache, segmentation_version, batch_size=1, workers=1, v4=False` | new datasets_config dict | copies dict (no mutation of arg) | ✅ |
| `get_inference_on_ds_noexceptions` | 161 | func | Exception-swallowing wrapper around caching inference | (see caching fn) | None | logs errors; **returns None even on success** (calls inner but discards result) | 🔴 calls `get_nn_inference_on_ds_and_cache` but never returns its value — used as cache-populate side-effect only; if treated as result-returning it silently yields None |
| `get_nn_inference_on_ds_and_cache` | 190 | func | Cache-aware inference: hash config+ckpt, load npz from B2/local or compute+save | `ds_fn,config_fn,checkpoint_fn,inference_cache,device,batch_size,workers,precompute_cache,crash_if_not_cached,segmentation_version,v4` | results dict (np arrays) | file I/O (npz read/write, rename), B2 network fetch, GPU; `assert not crash_if_not_cached` | ⚠️ atomic-rename via `.tmp`+`os.rename`; asserts when uncached & crash flag set |
| `run_nn_inference_on_ds` | 255 | func | Core: load model, build inference dataloader, run, reshape to (sessions,radios,snap,θ) | `ds_fn,config_fn,checkpoint_fn,device,batch_size,workers,precompute_cache,segmentation_version,v4=False` | `{"single"[, "paired"]}` (cpu tensors) | `model.eval()`; dataloader I/O; GPU; re-raises load errors | ✅ |

Notable:
- Reshape convention everywhere: `sessions = sessions_times_radios // 2`, `radios = 2` (hardcoded 2-radio assumption).
- 🔴 `get_inference_on_ds_noexceptions` has no `return` of the inner result — purely populates cache; mismatch if a caller expects data back.
- ⚠️ `single_example_realtime_inference` runs `model(single_example)` twice per sample (yields one, appends the other) — wasteful and slightly inconsistent.
- Cache key path encodes `segmentation_version`, checkpoint md5, config md5 → `.npz`.

---

## simulate/01_generate_data.py
`spf/model_training_and_inference/simulate/01_generate_data.py` — CLI to generate
synthetic v1 sessions (argparse → `generate_session{,_and_dump}` via joblib).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` block | 11 | script | Parse ~30 args, mkdir output, dump args.pkl, generate sessions (serial profile / parallel joblib) | CLI args | files on disk | `os.mkdir(args.output)` (fails if exists), `dump` pickle, parallel session generation, file writes | ⚠️ no functions defined; `--numba`/`--random-*` use `type=bool` (any non-empty string → True, classic argparse bool pitfall) |

Notable:
- Entire file is an `if __name__=="__main__"` script; no importable symbols.
- Several `type=bool` args (`--numba`, `--random-silence`, `--random-emitter-timing`, `--reference`, `--live`, `--profile`) — `bool("False")==True` foot-gun.
- Legacy v1 simulation path (`spf.dataset.spf_generate`).

---

## simulate/90_session_plotter.py
`spf/model_training_and_inference/simulate/90_session_plotter.py` — CLI to plot a
single legacy session and render a GIF.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` block | 6 | script | Load `SessionsDatasetTask2`, plot full session, build GIF | CLI `--dataset,--session-idx,--snapshots-per-session,--output_prefix` | `<prefix>.gif` + frame files | reads dataset, writes image/gif files | 🧪 legacy Task2 plotting path |

Notable:
- No importable functions; thin wrapper over `plot_full_session` + `filenames_to_gif`.

---

## scripts/train_single_point.py
`spf/scripts/train_single_point.py` — main training entrypoint + dataloader
construction, checkpointing, loss computation, the **model registry**, and the
train loop.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `worker_init_fn` | 36 | func | Reinit nested v5 datasets per DataLoader worker | `worker_id` | None | calls `_dataset.reinit()` (opens zarr per worker) | ✅ |
| `load_seed` | 45 | func | Seed torch + random | `global_config` | None | global RNG state mutation | ✅ |
| `expand_wildcards_and_join` | 50 | func | Glob-expand a list of path patterns | `paths` | list[str] | filesystem glob | ✅ |
| `preload_dataset` | 57 | func | Best-effort construct a v5 dataset (warm cache) | `prefix,nthetas,precompute_cache,segmentation_version` | None | dataset open (file I/O); logs errors | 🧪 only referenced in commented-out preload code |
| `load_single_dataset_using_configs` | 75 | func | Build one `v5spfdataset` from the 4 config dicts | `prefix, datasets/global/model/optim configs, **kwargs` | `v5spfdataset` | file I/O; GPU flag | ✅ |
| `load_dataset_manifest_with_sampling` | 96 | func | Read a manifest file w/ optional per-line sampling count | `dataset_manifest_fn` | list[str] paths (duplicated by sampling) | file read; raises on malformed line | ✅ |
| `load_dataloaders` | 112 | func | Build train/val/alternate-val dataloaders (split logic, subsetting, StatefulBatchsampler) | `datasets_config,optim_config,global_config,model_config,step,epoch,no_tqdm` | `(train_dl, val_dl, alternate_dls: dict)` | heavy dataset I/O; logging; randomness (shuffle); asserts non-empty | ⚠️ long/branchy; multiple split modes (.txt manifest / train_on_val / holdout) |
| `load_optimizer` | 377 | func | AdamW + StepLR (cosine branch is a no-op `pass`) | `optim_config, params` | `(optimizer, scheduler)` | raises on bad scheduler | 🔴 `scheduler=="cosine"` → `pass` leaves `train_scheduler` undefined → `UnboundLocalError` at return |
| `FrozenModule` | 400 | nn.Module | Wrap a net frozen+eval (overrides `.train()` to no-op) | ctor `net`; fwd `x` | `net(x)` | sets requires_grad=False, eval | 🧪 defined but not used (load_checkpoint freezes inline instead) |
| `save_model` | 416 | func | Save checkpoint .pth + config.yml + copy empirical_dist.pkl | `prefix,model,optimizer,scheduler,epoch,step,config,running_config,checkpoint_fn=None` | None | torch.save, yaml dump, file copy; logs checksum | ✅ |
| `load_checkpoint` | 449 | func | Load full or partial (single/paired) state dict; optionally freeze | `checkpoint_fn,config,model,optimizer,scheduler,force_load=False` | `(model,optimizer,scheduler,epoch,step)` | torch.load (cpu map); mutates model/opt/sched; freezes params | ⚠️ partial-load paths freeze sub-net & return epoch/step 0 |
| `load_model` | 513 | func | **Model registry**: name → class instance | `model_config, global_config` | nn.Module | raises `ValueError` on unknown name | ✅ central dispatch (5 names) |
| `uniform_preds` | 528 | func | Uniform distribution target baseline | `batch, target_ntheta` | Tensor `(B,S,ntheta)` | none | 🧪 unused (not called in this file) |
| `target_from_scatter` | 539 | func | Build soft label dist via gaussian-in-radians scatter w/ wrap padding; optional up/down-flip symmetry | `batch,y_rad,y_rad_binned,sigma,k,target_ntheta,single=False` | target Tensor `(B,S,target_ntheta)` | recursion for `single=True`; asserts y_rad range & odd n | ✅ ("continuous" scatter) |
| `target_from_scatter_binned` | 578 | func | One-hot scatter + optional GaussianBlur smoothing | same signature | target Tensor | none | ✅ ("onehot" scatter) |
| `weighted_beamformer` | 600 | func | Segmentation-mask-weighted mean of windowed beamformer | `batch` | Tensor | none | 🧪 not called in this file (likely used in dataset) |
| `mse_loss` | 608 | func | Sum-over-θ MSE then mean | `output,target` | scalar Tensor | none | ✅ |
| `discrete_loss` | 612 | func | Negative dot-product (cross-entropy-like) loss | `output,target` | scalar Tensor | none | ✅ |
| `nn_checksum` | 618 | func | Mean abs of all params | `net` | float | none | ✅ debug aid |
| `model_checksum` | 622 | func | Log sub-net + model checksums | `prefix,m` | None | logging | ✅ |
| `new_log` | 634 | func | Fresh dict of empty loss lists | none | dict | none | ✅ |
| `SimpleLogger` | 658 | class | Console logger (mean-reduce losses, close figs) | ctor `args,logger_config,full_config`; `.log(data,step,prefix)` | losses dict | logging; `plt.close` figs | ✅ |
| `WNBLogger` | 683 | class | Weights & Biases logger | ctor (wandb.init); `.log(...)` | losses dict | **network (wandb), wandb.log**, plt.close | ✅ |
| `compute_loss` | 735 | func | Compute all single/paired/multipaired/direct/tx-pos losses (+ optional pred/label plots) | `output,batch_data,datasets_config,loss_fn,scatter_fn,single,paired,plot,direct_loss,training` | `(loss_d: dict, fig)` | matplotlib figure creation; uses `rotate_dist` | ⚠️ very large; many `no_grad` aux-metric branches; figure side effect |
| `run_val_on_dataloader` | 997 | func | Run validation over a dataloader, accumulate losses | `dataloader,config,loss_fn,scatter_fn,epoch,m,plot=False` | `(val_losses dict, fig)` | model fwd (GPU); only first batch plotted | ✅ |
| `train_single_point` | 1033 | func | Full training driver: config load, resume, optim, AMP, train/val loop, checkpointing | `args` | None | extensive: file I/O, wandb, GPU, RNG, `sys.exit` on existing output dir | ⚠️ central long driver |
| `get_parser_filter` | 1364 | func | Build argparse parser for the CLI | none | `ArgumentParser` | none | ✅ |
| `__main__` | 1434 | script | logging config + parse + run | — | — | runs training | ✅ |

Notable:
- 🔴 `load_optimizer` `cosine` branch is `pass`, so `train_scheduler` is never assigned and the `return optimizer, train_scheduler` raises `UnboundLocalError`. The `step` scheduler path is the only working one.
- 🧪 `FrozenModule`, `uniform_preds`, `preload_dataset`, `weighted_beamformer` are defined but unused within this file (preload only in commented code).
- `compute_loss` computes many *auxiliary, no-grad* losses (aligned/craft/rand/old) for logging; only `single_loss` (+optional phi, paired, multipaired, direct, tx_pos) feed the backward `loss`.
- Multipaired models require `train/val_snapshots_per_session > 1` (asserted); `flip` only allowed with `beamformer`.
- Loss weighting constants are hardcoded: phi `0.01`, tx_pos `0.1`, direct `/30`.

---

## scripts/train_utils.py
`spf/scripts/train_utils.py` — config loading/defaulting + the canonical list of
batch keys a model needs. Imported widely (train + inference).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `load_config_from_fn` | 6 | func | YAML-load a config and apply defaults | `fn` | config dict | file read | ✅ |
| `get_key_or_set_default` | 12 | func | Recursive `a/b/c` keyed default insert | `d, key, default` | None | **mutates `d`** | ✅ |
| `load_defaults` | 24 | func | Inject all default config keys | `config` | config dict | mutates config | ✅ |
| `global_config_to_keys_used` | 67 | func | Map global_config flags → list of batch keys to collate | `global_config` (or None) | list[str] keys | none | ✅ central key list |

Notable:
- `global_config_to_keys_used` is the single source of truth for which dataset keys are materialized; mirrored in inference module import.
- Defaults include `signal_matrix_input`, `empirical_input`, `sdr_device_type_input`, `segmentation_version`, etc.

---

## scripts/create_inference_cache.py
`spf/scripts/create_inference_cache.py` — CLI driver to populate the NN inference
cache over many datasets in parallel (or debug-serial).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_parser` (nested) | 14 | func | Build argparse (config/ckpt/cache/seg-version/datasets/...) | none | parser | none | ✅ |
| `__main__` block | 12 | script | Resolve dataset list (.txt expansion), `partial` an inference fn, run via `multiprocessing.Pool` (or serial in debug) | CLI args | cache files written (side-effect) | multiprocessing, GPU, file/network I/O | ✅ |

Notable:
- Chooses `get_inference_on_ds_noexceptions` (swallow errors) normally vs `get_nn_inference_on_ds_and_cache` in `--debug`.
- Hardcodes `batch_size=64, workers=0` in the partial.

---

## scripts/create_empirical_p_dist.py
`spf/scripts/create_empirical_p_dist.py` — build empirical P(θ|phase-diff) heatmaps
per device-type+spacing from datasets; pickles the result and optionally plots.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_heatmap_for_radio` | 16 | func | 2D histogram (gt θ vs mean phase) for one radio across datasets | `dss, radio_idx, bins` | `(heatmap, xedges, yedges)` | none | ✅ |
| `get_heatmap` | 25 | func | Average r0/r1 heatmaps | `dss, bins=50` | np.ndarray | none | 🧪 not called within file (helper) |
| `create_heatmaps_and_plot` | 32 | func | Build phi-norm & theta-norm conditional maps (±symmetry), plot, return tensors | `dss, bins, save_fig_to_prefix=None` | dict `{"r0","r1","r"}→{"sym","nosym"}→Tensor` | matplotlib figs + `savefig` if prefix; `plt.close` | ✅ |
| `apply_symmetry_rules_to_heatmap` | 100 | func | Enforce θ/phi reflection symmetry on a heatmap | `h` | np.ndarray | none | ✅ |
| `get_empirical_p_dist_parser` | 112 | func | Build argparse | none | parser | none | ✅ |
| `create_empirical_p_dist` | 151 | func | Load datasets, group by device+spacing (consistency-check rx spacing), build heatmaps, pickle | `args` | None | dataset I/O, GPU, prints, `os.makedirs`, `pickle.dump`, savefig | ⚠️ long; warns/asserts on rx-spacing consistency |
| `__main__` block | 245 | script | Parse + run | — | — | as above | ✅ |

Notable:
- Two normalizations: `*_phi_norm` (P(θ|phase)) and `*_theta_norm` (P(phase|θ)); suptitles appear swapped relative to var names (⚠️ cosmetic).
- `get_heatmap` is an unused convenience helper.

---

## scripts/segmentation_metrics.py
`spf/scripts/segmentation_metrics.py` — compute per-frequency/routine phase-error
std & finite-fraction metrics over many zarrs in parallel.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_parser` | 13 | func | Build argparse (input-zarrs/cache/seg-version/parallel) | none | parser | none | ✅ |
| `ds_to_metrics` | 33 | func | Open one dataset, compute phase-diff std / finite-fraction / count | `args: dict` | `(routine, carrier_freq, Tensor[std,frac,n])` or `(None,None,None)` | dataset I/O (context-managed); logs errors on failure | ✅ |
| `__main__` block | 66 | script | Expand .txt, Pool-map metrics, aggregate weighted std/notnan per freq+routine, print table | CLI | stdout table | multiprocessing, file I/O | ⚠️ `Pool(8)` hardcoded (ignores `--parallel`) |

Notable:
- ⚠️ `Pool(8)` is hardcoded at line 80; `--parallel` arg is parsed but unused.
- Aggregation weights std/notnan by sample count `metrics[:,2]`.

---

## scripts/check_phi_error.py
`spf/scripts/check_phi_error.py` — quick CLI to print per-radio phase error stats
for a single dataset.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_parser` (nested) | 8 | func | Build argparse (seg-version/cache/dataset) | none | parser | none | ✅ |
| `__main__` block | 6 | script | Load `v5spfdataset`, print per-radio nanmean/var phase error + freq + antenna spacing | CLI | stdout | dataset I/O, GPU | ⚠️ hardcodes `segmentation_version=3.5` (line 42) despite parsing `--segmentation-version` |

Notable:
- ⚠️ The parsed `--segmentation-version` arg is **ignored**; `v5spfdataset(... segmentation_version=3.5 ...)` is hardcoded.
- `if True:` wraps the dataset load (commented-out try/except).

---

## Cluster summary

Counts by status (symbols/tables above; script `__main__` blocks counted):

- ✅ working/normal: ~46
- ⚠️ caveat/smell: ~14
- 🔴 likely bug: 4 — `baseline_algorithm` (steps==-1 dead default), `load_optimizer` (cosine → UnboundLocalError), `get_inference_on_ds_noexceptions` (returns None on success), and the `target_from_scatter`/in-place class of mutation risks (see PrepareInput).
- 🧪 experimental/unused: ~17 (e.g. `TransformerTimeModel`, `CrossAttentionNet`, `Separate1D2DNet`, `SignalMatrixNet`, `FrozenModule`, `uniform_preds`, `preload_dataset`, `weighted_beamformer`, `frac_to_theta`, `lines_to_points`, `cdf`, `normal_correction_for_bounded_range`, `sigmoid_dist`, `get_heatmap`, `TrajPairedMultiPointWithBeamformer`).

Never-instantiated / config-only model classes (cross-checked vs `load_model`
and grep): `TransformerTimeModel`, `CrossAttentionNet`, `Separate1D2DNet` are
reachable **only** via `AllWindowsStatsNet.conv_type` strings ("transformer"/
"cross"/"1d2d") and carry hardcoded 68→12 dims + "chatGPT" comments. `SignalMatrixNet`
reachable only via optional `signal_matrix_net` model_config block. None appear in
any checked-in config under this scan.

Top 5 deep-dive candidates:
1. **`load_optimizer` cosine branch (train_single_point.py:389)** — `pass` ⇒ `UnboundLocalError`; verify no config uses `scheduler: cosine`.
2. **`get_inference_on_ds_noexceptions` (single_point_networks_inference.py:161)** — missing `return`; confirm callers only want the cache side-effect, not the dict.
3. **`PrepareInput.prepare_input` in-place batch mutation (single_point_networks.py:470)** — `v[mask]=0` on views of batch tensors during training; check for cross-sample contamination / autograd correctness.
4. **`compute_loss` (train_single_point.py:735)** — large; many aux no-grad losses + hardcoded weighting constants; confirm which terms actually drive `loss` per model.
5. **`baseline_algorithm` / `get_top_n_points` (baseline_algorithm.py:27/49)** — broken default + int-dtype distance arrays + dense geometry; verify still-correct vs only caller in `plot.py`.
