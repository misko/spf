# SPF Processing Pipeline — Review Report (verified content skeleton)

Source of truth: `claude_docs/` (verified docs), spot-checked against source, configs, and wandb
run summaries at commit `574c454`. Severity/issue #s = `claude_docs/KNOWN_ISSUES.md` (current,
re-triaged). Test coverage = `claude_docs/reference/_stage1_audit/TEST_COVERAGE_MATRIX.md`
(✅ direct · ⚠️ indirect/partial · ❌ none). Anything not fully verified is marked ⚠️.

---

## Stage 1 — Data collection (raw IQ → `.zarr`)

**(a) Purpose.** Move radios through space while recording raw IQ + ground-truth positions. Two
platform families: **wall array** (CNC gantry, GRBL steppers, XY in mm → v5 format,
`spf/grbl_radio_collection.py`) and **rover** (ArduPilot via MAVLink, GPS → v4 format,
`spf/mavlink_radio_collection.py`). Both drive a motion *routine* (bounce/circle/center/diamond)
and record from two 2-element receivers plus an emitter (ESP32 WiFi / LoRa / 5.8 GHz VTX / SDR-TX).

**(b) Key transform.** None (capture). Per-receiver `ThreadedRX` threads read the SDR forever,
tag position+timestamp, and queue snapshots (drop-stale, single-buffer);
`DroneDataCollectorRaw` / `GrblDataCollectorRaw` (`spf/data_collector.py`) write the zarr.
AD9361 RX1↔RX2 **phase-inversion register calibration** in `setup_rx_config()` is load-bearing
for phase coherence.

**(c) Input → output.** In: SDR buffers + GRBL/MAVLink position state. Out: `{prefix}.zarr`
(LMDB-backed) + `.yaml` (config copy) + `.log`. One **snapshot** = one receiver capture:
`signal_matrix (2, 524288) complex64` (2 antenna elements × buffer) + per-snapshot metadata.
Per receiver `r{0,1}`: `signal_matrix (snapshots, 2, buffer_size) complex64`;
`system_timestamp f64`; v5: `tx/rx_pos_x_mm/y_mm`, `rx_heading_in_pis`, `rx_theta_in_pis` (all f64);
v4 instead: `gps_timestamp, gps_lat, gps_long, heading` (degrees; `rx_heading_in_pis` deliberately
unwritten in v4 — see refuted issue #16); shared: `rx_spacing` (m), `rx_lo`, `rx_bandwidth` (Hz),
`avg_phase_diff, rssis, gains (snapshots, 2)`.

**(d) Production parameters** (`rover_receiver_config_pi_3mhz.yaml`, verified):
`buffer-size: 524288` (2¹⁹) · `f-carrier: 5.866e9` (fleet spans 868/915 MHz, 2.412–2.467 GHz,
5.766–5.866 GHz) · `f-sampling: 30e6` · `bandwidth: 3e6` · `antenna-spacing-m: 0.05075`
(fleet 0.025–0.080 m) · `n-records-per-receiver: 600000` · `theta-in-pis: 0 / 0.5` (the two arrays
mounted 90° apart) · `data-version: 4` (rover) / 5 (wall).

**(e) Verified issues.** #42 🟠P1 ⚠️silently-wrong — GRBL `PointOutOfBoundsException` kills the
motion thread; collection keeps stamping the frozen position on every later record. #40 🟠P1
safety — rover EKF arm gate ORs `EKF_POS_HORIZ_REL` twice, omits `HORIZ_ABS` → can arm on a
relative-only estimate. #43/#44 🟠P1 safety — unmapped flight mode `KeyError` strands the planner;
`move_to_point`/RTL has no effective timeout. #18 🟠P1 — `sudo shutdown` from the MAVLink handler
thread. #17 🟠P1 — `yaml_defaults` reads a module-global. #46 🟡P2 — unlocked cross-thread motion
state. #5/#29/#30 🟡P2 — CirclePlanner overshoot (does not manifest), dead `wait_while_moving`,
duplicated device-mapping parser.

**(f) Test coverage.** Not in the stage-1 matrix (it starts at segmentation). Planners/GRBL/Drone
have fake-hardware tests, but the real serial/MAVLink hot paths (incl. all of (e)) are untested;
`GRBLController.update_status` recursion (#50) is only reachable on real serial.

---

## Stage 2 — Segmentation / preprocessing (v3.7: `.zarr` → `.yarr`)

**(a) Purpose.** Decide which 2048-sample windows of each snapshot contain signal vs noise and
precompute per-window phase stats + the windowed beamformer, so training never touches the raw
IQ. Entry: `mp_segment_zarr()` (`spf/dataset/segmentation.py:107`), multiprocessing Pool over
sessions, exactly 2 receivers (hard assert).

**(b) Key math.** Per session: detrend (`detrend_np`, removes DC/linear drift) → per-window
**delay-and-sum beamform** against precomputed steering vectors
`s(θ)=e^{−j2π(d/λ)sinθ}`, `reshape(nthetas,−1,window).mean.T` → per-window **trimmed circular**
mean/stddev of φ (`get_phase_diff = pi_norm(∠row0 − ∠row1)`) + median |signal| →
threshold/merge/drop (`simple_segment`: combine_windows, drop_windows_smaller_than,
keep_signal_surrounded_by_noise, recompute_stats) → boolean mask → signal-weighted session stats.

**(c) Input → output.** In: `signal_matrix (2, 524288) complex64` per snapshot. Out (`.yarr`,
per receiver): `all_windows_stats (snapshots, 3, 256) f16` (ch0 = trimmed circ-mean φ rad,
ch1 = circ-stddev, ch2 = median |sig|); `windowed_beamformer (snapshots, 256, 65) f16`;
`weighted_beamformer (snapshots, 65) f32`; `weighted_windows_stats (snapshots, 3) f32`;
`downsampled_segmentation_mask (snapshots, 256) bool`; `mean_phase (snapshots,) f32` (NaN if no
signal windows); plus a `.pkl` of segment lists, versioned `SEGMENTATION_VERSION = 3.7`
(`spf/utils.py:14`).

**(d) Production parameters** (`DEFAULT_SEGMENT_ARGS`): `window_size = stride = 2048` (⇒
256 windows/snapshot, several correctness properties silently require stride==window_size) ·
`trim = 20` (%) · `max_stddev_threshold = 0.5` · `drop_less_than_size = 3000` (samples) ·
`min_abs_signal = 40` · `gpu = False` default (cupy path optional).

**(e) Verified issues.** #45 🟠P1(multi-emitter)/🟡P2 ⚠️silently-wrong —
`keep_signal_surrounded_by_noise` drops *both* of two abutting signal runs that combine_windows
declined to merge (phase Δ>0.2); fires on the default config; corroborated by a commented-out
test assert. #9 🟡 (re-triaged) — `beamform_signal_cpu` is a dead stub with **no callers**; the
real CPU path (`rf.py beamformer_given_steering_nomean(_fast)`) works. #49 🟡P2 — `gpu=True`
without cupy → `NameError` inside a Pool worker. #37 🟡P2 — trimmed-percentile backend divergence
(live ceil-rank `fast_percentile_1d` untested; tested floor-rank variant has no prod callers).
#8 🟡P2 — `get_segmentation` pkl recovery can delete-then-refuse-to-rebuild. S1 — f16 caching is
lossy by design (amplitudes >2048; saturation at 65504). S6 — mask indexes `//window_size` not
`//stride` (correct only for the production 2048/2048).

**(f) Test coverage.** Windowed beamformer writer ❌ (only numpy↔numba parity is tested);
live circular-stats engine ❌ (**matrix P0 gap** — primitives ⚠️ only, `test_circular_mean` is
duplicate-shadowed and tests the wrong fn); `simple_segment` ✅ happy path only;
`mp_segment_zarr` ❌ direct (e2e transitively via `test_fake_data`/`test_v5inference`).

---

## Stage 3 — Dataset layer (`v5spfdataset`)

**(a) Purpose.** Read `{prefix}.zarr + .yarr + .yaml`, compute ground-truth angles from
positions, and serve torch tensors to training/inference/filters
(`spf/dataset/spf_dataset.py:818`). `v5spfdataset` reads v4 on the fly via `v4_to_v5`.
`v5inferencedataset` (`:448`) is the threaded realtime variant.

**(b) Key transform.** `__getitem__(idx)`: with `paired=True` (n_radios 2) returns
`[render_session(r0, idx), render_session(r1, idx)]`; index space `[0, n_sessions)` paired,
`× n_receivers` unpaired. θ frames: `y_rad`/`ground_truth_theta` = **array-relative**;
`craft_y_rad` = **craft-heading-relative**; `absolute_theta` = world `arctan2(dx,dy)`;
`y_phi` = phase target (−sin sign convention, empirically validated). Augmentation in
`render_session`: `flip` (50%: negate θ/φ, flip windowed_beamformer θ-axis + stats ch0 —
consistent across all collated keys); `double_flip` (paired: θ → sign(θ)·π−θ, φ negated).
`v5_collate_keys_fast` stacks paired samples flat ⇒ **B = batch_size × 2 = 512 rows**, upcasts
f16→f32.

**(c) Input → output (stage-1 config keys).** `windowed_beamformer (B,1,256,65) f32` ·
`all_windows_stats (B,1,3,256) f32` · `gains (B,1,2)` · `rx_wavelength_spacing, rx_lo,
vehicle_type, sdr_device_type (B,1)` · `y_rad, y_phi, craft_y_rad (B,1)` · `empirical (B,1,65)`
(force-appended; used only for shape/device in target building — not a model input here).
Positions come out in **meters** (`*_pos_xy`, mm/1000).

**(d) Key parameters.** `nthetas: 65` · `snapshots_per_session: 1` · `snapshots_stride: 1` ·
`paired = n_radios>1` (True) · `flip: true / double_flip: false` (stage 1); inverted for stage 2 ·
`precompute_cache: /mnt/md2/cache/precompute_cache_3p7` · `segmentation_version: 3.7` ·
`skip_qc: true` · `empirical_data_fn: empirical_dists/full.pkl` (symmetrized, pooled "r" table).

**(e) Verified issues.** #7 🔴P0 — two live `breakpoint()` in `spf_nn_dataset_wrapper.py:86,126`
on the only realtime/absolute-north path. #32/#39/#33 🟠P1 — `v5inferencedataset` unlocked store
read vs eviction (`KeyError` under prod `max_store_size=3`), `min_idx_stored` init race
(`AttributeError`), non-daemon reader thread blocking exit. #11 🟠P1 — mutable-default
`skip_fields=[]` mutated in place. #19 🟠P1 — destructive zarr migration scripts. Raw-IQ field
`abs_signal_and_phase_diff` does **not** respect flip (moot for prod: `signal_matrix_input: false`).

**(f) Test coverage.** Ground-truth angles ✅ (`test_angles`, `test_fake_data`) but zero-heading
only; **flip/double_flip aug ❌ NONE** (matrix P0 gap); collate + key-set ❌ direct;
`v5inferencedataset` concurrency paths ❌.

---

## Stage 4 — Targets & symmetry

**(a) Purpose.** Convert scalar ground-truth θ into the 65-bin distribution the model regresses
against, encoding exactly the ambiguities a 2-element array physically has.

**(b) Key math** (`target_from_scatter*`, `train_single_point.py:539-575`, audited): analytic
Gaussian in radians `exp(−((m−y)/σ)²/2)` over the 65-bin θ grid, **±π wrap-fold** (mass beyond
the seam folded to the correct bins), L1-normalized to sum 1. With `single=True` (stage 1) the
target is averaged with its **front/back reflection** θ → sign(θ)·π−θ (valid because
`sin(sign(θ)π−θ)=sinθ` ⇒ same φ) — the supervised distribution is deliberately bimodal.

**(c) Input → output.** In: `y_rad (B,1)` rad (+ `empirical` for shape/device only). Out: target
`(B,1,65)` f32, sums to 1 per row.

**(d) Parameters.** `scatter: continuous` · `sigma: 0.1` (rad) · `scatter_k: 21` — **inert** on
this path (kernel blur only applies to the unused `scatter: onehot`/binned path,
`train_single_point.py:592-595`; audit-verified) · `single=True` for the `beamformer` model.

**(e) Symmetry composition (verified, no double-fold).** Left/right handled as *input
augmentation* (`flip`: negate θ/φ + flip features); front/back handled in the *target* (the
π-fold above); they are different reflections at different stages, so they compose. Stage 2
(paired) swaps to `double_flip` data augmentation — directly training the fold that fusing two
differently-oriented arrays is meant to resolve. A former `flip_up_down` aug was moved into the
loss/target side. `empirical_symmetry: true` selects the symmetrized P(θ|φ) table, but
`empirical_input: false` ⇒ it is loaded, not fed to the net.

**(f) Test coverage.** ❌ NONE direct (matrix P0 gap SEG-6): no assertion on unit-sum, wrap-fold,
or fold symmetry; `test_rotate_dist` covers a different function.

---

## Stage 5 — Model (`SinglePointWithBeamformer` → `PairedSinglePointWithBeamformer`)

**(a) Purpose.** Learn a calibrated per-snapshot P(θ) from the windowed beamformer + scalar
metadata. Stage 1: one receiver → array-relative θ (ambiguous by design). Stage 2: fuse both
receivers → craft-relative θ (resolves front/back).

**(b) Key transform** (audit-verified shape contracts):
`AllWindowsStatsNet` concatenates `all_windows_stats` (3) + `windowed_beamformer` (65) →
**(B, 68, 256)**, Conv1d(68→64)×8 (leaky, no norm) → **12** window features
(`single_point_networks.py:786-788`). Beamformer values scaled by a hardcoded `/500`
(`normalize_windowed_beamformer: false`). Train-only window augmentation: `window_fraction 0.5,
window_dropout 0.25, window_shuffle 0.15, window_shrink 0.5` (no-op at eval).
`PrepareInput`: 12 + 6 scalars (gains 2, spacing 1, freq 1, vehicle 1, sdr 1) = **18-dim** →
`FFNN Linear(18→512)`, depth-4 residual `BasicBlock`s, layer norm, `input_dropout 0.1` →
`.abs()` → **L1-norm over dim 2** → `{"single": (B,1,65)}`.
Paired model: runs the (frozen-ish, `detach: true`) single net on both radios, rotates radio-0's
distribution into the craft frame (`rotate_dist`), concatenates, fusion FFNN → `{"paired"}`;
initialized via `model.load_single: true` from stage-1 `best.pth`.

**(c) Input → output.** In: the stage-3 batch (B=512 rows, S=1, f32). Out: `single (B,1,65)`
(+ `paired (B,1,65)` stage 2) — non-negative, each row sums to 1. Raw IQ, phase stats, and the
session-averaged beamformer are **not** inputs (`signal_matrix/phase/beamformer_input: false`).

**(d) Parameters.** `hidden: 512, depth: 4, norm: layer` ("thin") · windows_stats_net
`hidden_channels: 64, outputs: 12, n_layers: 8` · `output_ntheta = nthetas = 65` ·
`input_dropout 0.1` (single) / `0.2` (paired) · embeddings for vehicle + SDR device type.

**(e) Verified issues.** #34 🟡P2 — `PrepareInput` mutates `vehicle_type` batch views in place
during dropout (fires every training step; benign single-consumer today, fragile). Unused config
nets (2d/cross/transformer/1d2d) are not engaged (`conv_type` defaults "1d"). Unused model
classes (`CrossAttentionNet`, `TransformerTimeModel`, traj nets) exist but weren't retrained.
`/500` scale + no normalization is an acknowledged scale-sensitivity surface.

**(f) Test coverage.** AllWindowsStatsNet forward/aug-no-op ❌; PrepareInput 18-dim + L1-norm
contract ❌ direct; the CI training fixture uses a **different variant** (phase_input, no
windows_stats_net) so the production forward path is never exercised (matrix P0 keystone gap).

---

## Stage 6 — Training

**(a) Purpose.** `spf/scripts/train_single_point.py -c <config>` — build ConcatDataset over
`.zarr` globs, collate, train/val loop with checkpointing + wandb.

**(b) What is backpropagated (audit-verified).** Exactly **one term**: `single_loss =
mse_loss(output["single"], target)` (SSE over 65 bins, both L1-normed) for stage 1;
`paired_loss` for stage 2. All other logged losses (`single_craft`, `plus_rand`,
`target_aligned`, `uniform`) are `no_grad` diagnostics. fp32, `amp: false` (scaler/autocast
no-ops).

**(c) Input → output.** In: stage-3 batches. Out: checkpoints (`model_state_dict`, optimizer/
scheduler state, step/epoch, config); `best.pth` selected on `save_on`.

**(d) Parameters (jun26 configs, verified).** Optimizer **AdamW**; stage 1: `lr 1e-4`
(restarted from 2e-4), `weight_decay 1e-3`, `StepLR(step=6, γ=0.5)` per epoch, `epochs 100`,
`batch 256` (→512 rows), `save_on val/single_loss`; stage 2: `lr 2e-4, wd 0.0,
scheduler_step 12, epochs 60, save_on val/paired_loss, head_start 0`. 24 workers,
`val_every 25000`, `val_holdout_fraction 0.5`, `val_subsample_fraction 0.5`; train split
`apr17_train_nosig_noroverbounce_noblade.txt`; seed 10 (⚠️ numpy RNG not seeded — only torch +
python random).

**(e) Results (verified from wandb summaries).**
- jun26 single (`run-…ygj4glp5`): **val/single_loss 0.0984** @ step 1,875,000 (final) — ties
  dec15 single (0.0984 @ 1.64M).
- jun26 paired (`run-…sj12xn0k`): **val/paired_loss 0.1517 @ step 1,054,099** (last logged;
  ⚠️ the quoted "0.1512 @ 1M" is presumably the exact-1.0M val point — not directly re-read
  here, summary shows 0.1517 @ 1.05M) vs dec15 paired **0.1615 @ 2.75M** — better with ~2.6×
  fewer steps.
- **BladeRF OOD gap: +36.2%** — mean per-dataset val single_loss 0.1323 (22 BladeRF sets) vs
  0.0971 (62 non-BladeRF), for the noblade-trained single model (verified by recomputation).

**(f) Issues + coverage.** #1 🟡P2 — cosine scheduler branch → `UnboundLocalError` (only 1
abandoned config uses it; default `step` unaffected). #36 🟡P2 — `global_config_to_keys_used`
KeyError on omitted flags (real configs set them). Loss ❌ tests; optimizer/scheduler ❌;
train loop ⚠️ one e2e smoke on a non-production model variant.

---

## Stage 7 — Inference & empirical baseline

**(a) Purpose.** Run checkpoints over datasets (populating an `.npz` cache the NN filters
consume), and maintain the non-learned **empirical P(θ|φ)** lookup used as baseline + PF
likelihood.

**(b) Key transforms.** `load_model_and_config_from_config_fn_and_checkpoint` (builds model,
`force_load=True`, does **not** call `.eval()` — callers must). `run_nn_inference_on_ds`
reshapes `(sessions·radios, S, ntheta) → (sessions, 2, S, ntheta)` (radios=2 hardcoded).
Empirical table: 2-D histogram of (ground-truth θ, observed `mean_phase` φ) per
`{sdr_device_type}_{rx_spacing}` group, symmetrized (fold θ/−θ and φ-reflection), then
column-normalized → **P(θ|φ)** stored `[φ_bin][θ_bin]`, each φ-row sums to 1; consumer indexes
by `to_bin(mean_phase)` (producer→consumer join verified end-to-end).

**(c) Input → output.** In: dataset prefix + config.yml + best.pth. Out: cached
`{inference_cache}/{ds_basename}/{seg_version:0.3f}/{checkpoint_md5}/{config_md5}.npz` with
`single` (and `paired`) arrays `(sessions, 2, S, 65)`; ⚠️ return dtype differs by branch
(numpy from cache, torch from no-cache). Empirical: `empirical_dists/full.pkl` →
`{group → {"sym"/"nosym"} → (nbins_φ, nbins_θ)}` tensors.

**(d) Parameters.** `crash_if_not_cached=True` default; CLI `create_inference_cache.py`
hardcodes `batch_size 64, workers 0, v4=False`; empirical `--nbins` default **50 (even —
broken)**, checked-in artifacts use 65/7 (odd — correct).

**(e) Verified issues.** #41 🟠P1 ⚠️silently-wrong — cache key **omits `v4`** (which changes
outputs via `v4_to_v5`) and keys on basename only → v4/v5 or same-name datasets collide on the
same `.npz` and return stale/wrong inference. #47 🟡P2 — `get_md5_of_file` caches md5 in a
never-invalidated `.md5` sidecar + `functools.cache` → in-place checkpoint/config edits silently
reuse stale cache. #48 🟡P2 — `apply_symmetry_rules_to_heatmap` broken for even bins (default
`--nbins=50`!), mitigated by odd-bin artifacts. #26 🟡P2 — `get_inference_on_ds_noexceptions`
returns None (benign at sole caller). #25 🟡P2 — legacy geometric baseline `steps=-1` default
broken (masked by sole caller).

**(f) Test coverage.** Model load + inference ⚠️ (`test_new_models`, `test_v5inference`);
cache-key correctness ❌; empirical builder ❌ (the even-bin break was found by executing the
math, not by any test).

---

## Stage 8 — Filters (EKF / PF)

**(a) Purpose.** Fuse per-snapshot angle evidence over time into smooth θ(t) (and, in unused
variants, XY) emitter tracks. `spf/filters/`; harness `run_filters_on_data.py` fans jobs over a
`multiprocessing.Pool` with Cartesian hyperparameter grids.

**(b) Key math.** Observation `φ = 2π·(d/λ)·sin(θ − offset)` per radio, `offset =
theta-in-pis·π`; **every concrete filter negates the spacing on construction**
(`antenna_spacing = −yaml["antenna-spacing-m"]`) so the effective d/λ is negative — this matches
the dataset's `mean_phase` (−sin) sign convention and is load-bearing. Motion: constant-velocity
`F = [[1,dt],[0,1]]` (+6-d XY variant). EKF uses the analytic Jacobian
`∂φ/∂θ = cos(θ−off)·2π·(d/λ)`. PF likelihood: empirical `P(θ|φ)` lookup (`theta_phi_to_p_vec`)
or, for `*NN` filters, the cached NN distribution indexed by each particle's θ-bin.

**(c) Input → output.** In: `v5spfdataset` (`mean_phase`/observations, ground truth for metrics)
+ empirical table or inference-cache `.npz`. Out: per-timestep `{"mu","var"}` →
`{theta/craft_theta, P_theta}` trajectories + metric dicts (MSE vs
`(craft_)ground_truth_thetas`) → CSV reports.

**(d) Taxonomy & status (8 classes).** Active in the production grid config
(`ekf_and_pf_config.yml`): `SPFKalmanFilter` (EKF θ, 1 radio) ✅ · `SPFPairedKalmanFilter`
(EKF θ, 2) ✅ · `PFSingleThetaSingleRadio` ✅ · `PFSingleThetaDualRadio` ✅ ·
`PFSingleThetaSingleRadioNN` ✅ · `PFSingleThetaDualRadioNN` 🔴 **broken** (wired in + tested,
but see #2). Excluded from the prod config: `SPFPairedXYKalmanFilter` (EKF XY) and
`PFXYDualRadio` (commented out). Typical grid knobs: `phi_std, p, noise_std, dynamic_R, N`
(particles, e.g. 128–4096).

**(e) Verified issues.** #2 🔴P0 **confirmed active** — live `breakpoint()` at
`particle_dual_radio_nn_filter.py:41`; under the prod Pool the pdb prompt has no TTY → worker
hangs forever on step 0. #3 🔴P0 dormant — both B2/DynamoDB cloud drivers import
`spf.scripts.run_filters_*` which no longer exist (moved in `2276ed1`) → the cloud pipeline is
dead-on-arrival (~14 months). #31 🟠P1 — `ParticleFilter.trajectory` `.copy()` on a torch tensor
→ `AttributeError` with `return_particles=True`; `dt=1.0` hardcoded in predict. #12/#13 🟠P1 —
harness `__main__` TypeError; `eval()` of config strings. #51 🟡P2 — XY-EKF metrics compare
tx_x (mm) against radians (hidden: config-dead + loose test bound). #38 🟡P2 — dead base
`trajectory`. #52 🟡P2 — duplicate test name shadows the first NN-dual test.

**(f) Test coverage.** ⚠️ moderate: `test_ekf.py` / `test_particle_filter.py` cover the
single/dual θ paths on fake data with loose bounds (`mse_craft_theta < 0.15–0.25`; XY `<5`);
the dual-NN tests exist but hit #2/#7; base-class edge paths (`return_particles`, debug) ❌.

---

## Cross-cutting — conventions

- **Angles**: radians on **[−π, π]** everywhere, normalized via `pi_norm`/`torch_pi_norm`
  (plain `%` mishandles negatives). φ = raw inter-element phase difference (the observable);
  θ = angle of arrival (the estimand); linked by `φ = 2π(d/λ)sin(θ)` — with a **−sin/negated-
  spacing sign convention** shared by the dataset (`y_phi`) and every filter.
- **`*_in_pis` fields** (`rx_theta_in_pis`, `rx_heading_in_pis`): orientations stored as
  multiples of π (0.5 ⇒ π/2). v4 stores rover `heading` in **degrees** instead; the v4→v5
  converter bridges via `heading/360·2` (do not "fix" — load-bearing, refuted issue #16).
- **Positions**: zarr stores **mm** (`*_pos_x_mm`); dataset tensors emit **meters**
  (`*_pos_xy` = mm/1000). `rx_spacing` in meters; `rx_wavelength_spacing` = d/λ dimensionless.
- **Frames**: array-relative (`y_rad`) vs craft-relative (`craft_y_rad`) vs world
  (`absolute_theta`); a 2-element array cannot separate θ from π−θ (front/back) — resolved by
  pairing the two 90°-offset receivers.
- **Vocabulary**: snapshot (one capture, one receiver) ⊃ window (2048 samples); session =
  `snapshots_per_session` snapshots (=1 in prod); "paired" = both radios of one craft, not ML
  pairs. `c = 299,792,458 m/s`.

## Cross-cutting — top-10 issues by severity × reachability

| Rank | # | Sev | One line |
|---|---|---|---|
| 1 | #2 | 🔴P0 active | Live `breakpoint()` in the wired-in dual-radio-NN PF → prod Pool worker hangs forever on step 0. |
| 2 | #7 | 🔴P0 active | Two unconditional `breakpoint()`s on the only realtime/absolute-north inference path → headless rover hang. |
| 3 | #41 | 🟠P1 | Inference cache key omits `v4` + keys on basename → silently returns wrong/stale model outputs on collision. |
| 4 | #45 | 🟠P1/🟡 | Segmentation silently drops abutting signal runs on the **default** config (multi-emitter/phase-stepping data). |
| 5 | #42 | 🟠P1 | GRBL out-of-bounds point kills the motion thread; every later record stamped with the frozen stale position. |
| 6 | #40 | 🟠P1 safety | Rover EKF arm gate duplicates `HORIZ_REL`, omits `HORIZ_ABS` → can launch absolute waypoints on a relative-only fix. |
| 7 | #43+#44 | 🟠P1 safety | Unmapped flight mode `KeyError` strands the planner; `move_to_point`/RTL can hang with no effective timeout. |
| 8 | #3 | 🔴P0 dormant | Entire B2/DynamoDB cloud filter pipeline dead (imports moved modules); 2-line import fix + Dockerfile path. |
| 9 | #32+#39 | 🟠P1 | `v5inferencedataset` unlocked store read vs eviction (`KeyError`) + `min_idx_stored` init race (`AttributeError`) — realtime prod settings. |
| 10 | #47 | 🟡P2 | Never-invalidated `.md5` sidecars: in-place checkpoint/config edits silently reuse stale inference caches (compounds #41). |

(Honorable mentions: #31 PF `return_particles` crash; #48 even-bin empirical table — dodged by
odd-bin artifacts; #1 cosine scheduler — 1 abandoned config.)

## Key numbers (master table)

| Quantity | Value | Source/verified |
|---|---|---|
| Carrier `f-carrier` (`rx_lo`) | production mix: 868/915 MHz · 2.412/2.464/2.467 GHz · 5.766–5.866 GHz | capture configs + wandb val-dataset keys |
| Wavelength λ | 0.328 m @915 MHz · 0.124 m @2.412 GHz · 0.0511 m @5.866 GHz | c/f |
| Antenna spacing d | 0.025–0.080 m (rover prod 50.75 mm) | configs + wandb keys |
| d/λ (`rx_wavelength_spacing`) | **0.122 – 1.549** across production datasets (>0.5 ⇒ aliased/ambiguous φ for many sets) | wandb val keys, recomputed |
| Sample rate `f-sampling` | 30 MHz (rover prod; tests 16 MHz) | `rover_receiver_config_pi_3mhz.yaml` |
| RF bandwidth | 3 MHz (prod; tests 300 kHz) | same |
| Buffer / snapshot | 524,288 samples (2¹⁹) × 2 elements, complex64 (~4 MiB/snapshot) | config + audit |
| Window size / stride | 2048 / 2048 samples | `DEFAULT_SEGMENT_ARGS` |
| Windows per snapshot | 256 | 524288/2048 |
| nthetas (= output_ntheta) | 65 | configs |
| Segmentation version | 3.7 (`SEGMENTATION_VERSION`, `spf/utils.py:14`) | source |
| Seg thresholds | trim 20% · stddev ≤0.5 · min segment 3000 · min \|sig\| 40 | segmentation.py defaults |
| Records per receiver (collection) | 600,000 | rover config |
| Batch size | 256 (→ 512 collated rows, ×2 radios) | config + audit |
| Snapshots per session | 1 | config |
| Optimizer | AdamW, lr 1e-4 (single) / 2e-4 (paired), wd 1e-3 / 0.0 | configs + audit |
| Scheduler | StepLR step 6 (single) / 12 (paired), γ 0.5, per epoch | audit |
| Epochs | 100 (single) / 60 (paired); fp32, no AMP | configs |
| Target | continuous Gaussian σ=0.1 rad, ±π wrap-fold, single=True π-fold; `scatter_k 21` inert | audit |
| Model | AllWindowsStatsNet 68ch→12 · PrepareInput 18-dim · FFNN d4×h512 → L1-normed (B,1,65) | audit-verified contracts |
| jun26 single final | val/single_loss **0.0984** @ 1.875M steps (dec15: 0.0984 @ 1.64M) | wandb summary |
| jun26 paired | val/paired_loss **0.1517 @ 1.054M** (⚠️ "0.1512 @ 1.0M" not directly re-read) vs dec15 **0.1615 @ 2.75M** | wandb summaries |
| BladeRF OOD gap | **+36.2%** (0.1323 vs 0.0971 mean per-dataset val single_loss) | recomputed from wandb summary |
| Filter particles N | 128–4096 (grid) | harness config/tests |
| Uniform-prediction floor | val/uniform_loss_single ≈ 0.129 (context for 0.0984) | wandb summary |
