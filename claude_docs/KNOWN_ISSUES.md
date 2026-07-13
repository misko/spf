# Known Issues (verified)

Source-of-truth bug list distilled from the Pass-1 inventories in
[`reference/_inventory/`](reference/_inventory/) and **confirmed by an independent adversarial
verification pass** (each `VERIFY_*.md`). Every item below was checked against the actual source;
line numbers are cited. **Read-only audit — nothing here has been fixed.** Fixing is a separate,
explicit decision.

Severity:
- 🔴 **P0** — reachable crash on an active path, silently-wrong results, or data loss.
- 🟠 **P1** — broken off the hot path, correctness quirk, or data-integrity risk.
- 🟡 **P2** — dead code, ignored CLI args, duplication, debug cruft.

> Provenance note: the adversarial pass also **refuted** one drafted claim — `compare_datasets.py`
> was wrongly flagged as a SyntaxError; it is fine. Not listed below.

---

## 🔴 P0 — high impact

> **Re-triaged after per-issue investigation** (full reports in
> [`reference/_p0_investigations/`](reference/_p0_investigations/)). Each P0 was checked for
> reachability + blast radius. **Result: 3 confirmed active P0, 5 downgraded** — a clean
> demonstration that severity = misbehavior × reachability. All verdicts are read-only; nothing fixed.

### Confirmed active P0

| # | Issue | Location | Investigated verdict |
|---|---|---|---|
| 2 | Live `breakpoint()` in `PFSingleThetaDualRadioNN.observation()` — fires on first trajectory step. | `spf/filters/particle_dual_radio_nn_filter.py:41` | **CONFIRMED P0.** Filter is *active*: uncommented run block in `ekf_and_pf_config.yml:59` + 3 tests (`test_particle_filter.py:165,224,280`). Under the prod `multiprocessing.Pool` the pdb prompt has no TTY → worker **hangs forever** on step 0. Sibling single-radio-NN is fine. **Fix: delete line 41.** |
| 7 | Two unconditional `breakpoint()` in the NN dataset wrapper. | `spf/dataset/spf_nn_dataset_wrapper.py:86,126` | **CONFIRMED P0 (realtime path).** Both live, on the only path realtime/absolute-north inference takes. Realtime itself is experimental (consumer loop commented at `mavlink_radio_collection.py:332`), **but** a checked-in non-skipped test (`test_particle_filter.py::test_single_theta_dual_radioNN_nnwrapper`) hits both. On a headless rover → hang/`BdbQuit`. **Fix: delete both; set `PYTHONBREAKPOINT=0` in rover env.** |
| 3 | Both cloud filter drivers import from `spf.scripts.run_filters_on_data*`, which **do not exist**. | `spf/filters/run_filters_on_data_b2.py:25`, `run_filters_report_b2.py:10` | **CONFIRMED P0 but DORMANT.** Root cause: commit `2276ed1` moved `spf/scripts/run_filters_*` → `spf/filters/` without updating the in-file imports → `ModuleNotFoundError`. The whole B2/DynamoDB pipeline is dead-on-arrival (~14 months); `docker/repo/Dockerfile:38` CMD path is *also* stale. No CI/Batch wiring currently runs it. **Fix: `spf.scripts`→`spf.filters` in 2 imports (+ Dockerfile path).** |

### Downgraded after investigation (canonical severity now P2)

| # | Was | Now | Issue + verdict | Location |
|---|---|---|---|---|
| 1 | 🔴 | 🟡 P2 | `load_optimizer` cosine → `UnboundLocalError`. **Only 1 of 164 configs** sets `scheduler: cosine` (`dec3_…_bigger_4x.yaml:75`, abandoned, body commented); default is `step`. Fail-fast crash at optimizer build, no data loss, no test/CI hits it. | `spf/scripts/train_single_point.py:389` |
| 4 | 🔴 | 🟡 P2 (dead branch) | `swap_lat_long` 2-D-array branch is a no-op. **But no caller passes a 2-D `(N,2)` array** — all live callers pass 1-D/tuple/list, which take the *correct* swapping branch. Live rover GPS math is correct today; the buggy branch is simply never reached. Fix is defensive only — re-check for out-of-tree notebook callers first. | `spf/gps/gps_utils.py:8-10` |
| 5 | 🔴 | 🟡 P2 | `CirclePlanner` radian-increment vs `< 360`. Overshoot is real (~57×) **but does not manifest**: `angle_to_pos` uses `sin/cos` (periodic) so every lap is the correct circle, and collection is ended by `n-records-per-receiver`, not generator exhaustion. Recorded XY is correct. Fix: loop `while self.running` (don't cap at one lap — could strand long runs). | `spf/motion_planners/planner.py:205` |
| 6 | 🔴 | 🟡 P2 | `setup_rxtx_and_phase_calibration` → `AttributeError`. **Not on the live collection path** — its only caller is `sdr_controller.py:1256` under the standalone `--mode rxcal` CLI; normal collection uses `setup_rxtx`/`setup_rx` which call `setup_rx_config()` correctly. Crashes only if someone runs `--mode rxcal` on real hardware. **Fix: `setup_rx()`→`setup_rx_config()` (both lines).** | `spf/sdrpluto/sdr_controller.py:971,982` |
| 8 | 🔴 | 🟡 P2 | `get_segmentation` destructive recovery. The expensive `.yarr` is **not** deleted (only the cheap `.pkl`), so recovery is one re-segmentation, not "hours lost". Real risk is narrower: the common caller mode (`segment_if_not_exist=False`: training/inference/`ls_ds`) **deletes-then-refuses-to-rebuild**, and a concurrent short-read can manufacture a false `UnpicklingError` that nukes a good cache. **Fix: force regen on the post-delete call; quarantine not `os.remove`; also catch `.yarr` corruption.** | `spf/dataset/spf_dataset.py:1753-1760` |

_(Issues #1, #4, #5, #6, #8 are now P2; they remain listed here for the audit trail. The cheapest real fixes are the three confirmed P0s — all are one-or-two-line deletions/renames.)_

## 🟠 P1 — medium

| # | Issue | Location |
|---|---|---|
| 9 | `beamform_signal_cpu` is a stub returning `None` (the CPU beamforming fallback). | `spf/dataset/segmentation.py:~1025` |
| 10 | `v2_rssi_idxs` sets **both** RSSI indices to `"rssi0"`. | `spf/dataset/wall_array_v2_idxs.py:~40` |
| 11 | `v5inferencedataset` mutable-default `skip_fields=[]` mutated in place via `+=`. | `spf/dataset/spf_dataset.py:506` |
| 12 | `run_filters_on_data.py` `__main__` calls `run_filter_jobs(jobs)` without the required `nparallel` → `TypeError` if run directly. | `spf/filters/run_filters_on_data.py:731` |
| 13 | `config_to_job_params` `eval()`s config strings (self-flagged "might be dangerous"). | `spf/filters/run_filters_on_data.py:530` |
| 14 | `FakePPlus.get_rssi_and_gain` reads undefined `self.dev` (only `self.sdr` is set). | `spf/sdrpluto/sdr_controller.py:543` |
| 15 | `sdr.py` constructs detectors and **creates matplotlib figures at import time**. | `spf/sdrpluto/sdr.py:212-230` |
| ~~16~~ | ❌ **RE-VERIFIED BENIGN (not a bug)** — the V4 path is internally consistent: `data.heading` is set dynamically in **degrees** and `v4rx_f64_keys` uses `"heading"`, so `rx_heading_in_pis` is simply never written in V4; the v4→v5 converter bridges via `heading/360*2`. **Do NOT "fix" the dataclass** — it would break the converter + the integration test. | `spf/data_collector.py:540` |
| 17 | `yaml_defaults` reads a module-global `args` (not a parameter) → `NameError` if imported as a library. | `spf/mavlink_radio_collection.py:32-71` |
| 18 | `Drone.handle_RC_CHANNELS` runs `sudo shutdown` / `sys.exit` from the MAVLink message thread. | `spf/mavlink/mavlink_controller.py:897-909` |
| 19 | Destructive zarr scripts: `zarr_fix_rx_spacing` overwrites in place with no backup; `precompute_3p3_to_3p31` does in-place migration with a `TODO THIS SHOULD BE FIXED!!!` non-finite→0 hack. | `spf/scripts/zarr_fix_rx_spacing.py`, `spf/dataset/precompute_3p3_to_3p31.py` |

## 🟡 P2 — low (dead code / hygiene)

| # | Issue | Location |
|---|---|---|
| 20 | `torch_mean_phase_mean` does `np.*` ops on torch input (broken) — also dead (no callers). | `spf/rf.py:424` |
| 21 | `windowed_…_fast2` passes scalars to an array fn (crashes); only call site is commented out. | `spf/rf.py:557` (call site `segmentation.py:661`) |
| 22 | `QAMSource.demod_signal` / `MixedSource.signal` call non-callable instances (legacy `sdr.py`). | `spf/sdrpluto/sdr.py:72,49` |
| 23 | `yarr_rechunk.py` calls `new_yarr_dataset` missing 3 required args → `TypeError`. | `spf/scripts/yarr_rechunk.py` |
| 24 | `open_partial_ds.py` CLI calls `.add()` on a list. | `spf/dataset/open_partial_ds.py` |
| 25 | `baseline_algorithm` `steps==-1` default → `np.arange(-1)` empty loop (sole caller always passes `steps`). | `spf/model_training_and_inference/baseline_algorithm.py:27` |
| 26 | `get_inference_on_ds_noexceptions` never returns the inner result (benign — caller discards). | `spf/model_training_and_inference/models/single_point_networks_inference.py:161` |
| 27 | `check_phi_error.py` hardcodes `segmentation_version=3.5`, ignoring its `--segmentation-version` CLI arg. | `spf/scripts/check_phi_error.py:42` |
| 28 | `segmentation_metrics.py` hardcodes `Pool(8)`, ignoring `--parallel`. | `spf/scripts/segmentation_metrics.py:80` |
| 29 | `wait_while_moving` indexes `a/b_motor_steps` keys that `update_status` never sets → `KeyError`; also dead (no callers). | `spf/grbl/grbl_interactive.py:363` |
| 30 | Device-mapping parsing duplicated verbatim across the two collection orchestrators. | `spf/grbl_radio_collection.py:94-102`, `spf/mavlink_radio_collection.py:37-45` |

---

## Added by Phase-2 deep dives

Found while writing the `reference/spf/*` contract docs. **Confidence** column distinguishes
✅ source-verified (read the lines) from 🔍 analysis-level (a reasoned race/path, not yet
reproduced). The 🔍 items still need a confirming pass before they're treated as certain.

| # | Sev | Confidence | Issue | Location |
|---|---|---|---|---|
| 31 | 🟠 P1 | ✅ verified | `ParticleFilter.trajectory` does `self.particles.copy()` but `particles` is a **torch tensor** (no `.copy()`; torch uses `.clone()`) → `AttributeError` when `return_particles=True`. Also `dt=1.0` is hard-coded in `predict`, and `debug=True` double-calls `observation(idx)`. | `spf/filters/filters.py:346` |
| 32 | 🟠 P1 | ✅ verified | `v5inferencedataset.__getitem__` reads `self.store` **unlocked** at `:670` (the only `with self.lock` is the final return at `:677`); the reader thread can `pop(idx)` (eviction, under the lock at `:618`) in between → unhandled `KeyError`. Reachable under the production `max_store_size=3` when producers outrun the consumer. _(Confirmed by deep-dive verification wave.)_ | `spf/dataset/spf_dataset.py:670` vs `618/632` |
| 33 | 🟠 P1 | ✅ verified | `v5inferencedataset` reader thread is **non-daemon** (`daemon=True` commented out) with `join(timeout=1.0)`, and its `multiprocessing.Queue` is never closed/drained → a timed-out join can leave a live thread + queue feeder blocking interpreter exit. | `spf/dataset/spf_dataset.py:608/808` |
| 34 | 🟡 P2 | ✅ verified | `PrepareInput.prepare_input` zeroes batch tensors **in place** during training dropout via aliased views — `weighted_windows_stats` (`:508-511`) **and** `vehicle_type` (`:528-530`, an unsqueeze view). _(Verification correction: the deep dive wrongly said "only phase_input"; the arithmetic branches — beamformer/empirical/rx_spacing/frequency/gains — are genuinely fresh, but `vehicle_type` is not.)_ Latent today (batch fresh per step) but fragile. | `spf/model_training_and_inference/models/single_point_networks.py:508-511, 528-530` |
| ~~35~~ | — | ❌ **REFUTED** | ~~`load_dataloaders` 4-way path-split has no final `else` → unbound vars.~~ **Fabricated** — there **is** a final `else` at `train_single_point.py:158-164` binding both paths. Not a bug. | — |
| 36 | 🟡 P2 | ✅ verified | `global_config_to_keys_used` reads non-defaulted flags (`beamformer_input`, `phase_input`) → `KeyError` if a config omits them (`load_defaults` only sets `windowed_beamformer_input`). Latent — real configs set them. _(Confirmed by verification wave.)_ | `spf/scripts/train_utils.py` |
| 37 | 🟡 P2 | ✅ verified | rf.py trimmed-stat **backend divergence**: the live `fast_percentile_1d` (ceil-rank) is untested; the *tested* `fast_percentile` (floor-rank) has zero production callers; the torch path uses a third convention (`torch.quantile`). Trimmed stats differ by backend and are unverified on the real path. | `spf/rf.py:348` (live) vs `82` (tested) |
| 38 | 🟡 P2 | ✅ verified | `SPFFilter.trajectory` (base) is dead and self-inconsistent (its `update(prior=…, observation=…)` call mismatches the no-arg base `update` and every override); `theta_phi_to_p` has zero callers. | `spf/filters/filters.py:242, 409` |
| 39 | 🟠 P1 | ✅ verified | `v5inferencedataset`: `min_idx_stored` is initialized **only inside the locked reader block** (`:633`), so a consumer calling `__getitem__` before the first insert hits `AttributeError` at `:665` (not the intended `None`/wait). _(Found by the deep-dive verification wave — missed by the deep dive itself.)_ | `spf/dataset/spf_dataset.py:633, 665` |

## Added by Phase-3 deep investigation (all adversarially verified)

A second deep round (6 module deep-dives + a focused verify wave) on the modules that were only
inventoried before — segmentation, the `Drone`/MAVLink stack, GRBL + planners, the concrete
filters + harness, the SDR capture path, and inference/cache/baseline. **Every issue below was
confirmed by an independent skeptic against source.** Reference docs in `reference/spf/*`;
verify reports in the `VERIFY_*_new.md` files. The standout theme is **silently-wrong-result**
bugs (no crash) and **rover safety**.

### 🟠 P1 — silently-wrong or safety, reachable

| # | Sev | Issue | Location |
|---|---|---|---|
| 40 | 🟠 P1 · safety · ⚠️ silently-wrong | **EKF arm/launch gate too lax.** `healthy_ekf_flag` ORs `EKF_POS_HORIZ_REL` **twice** and **omits `EKF_POS_HORIZ_ABS`** (the GPS-absolute bit waypoint nav depends on). Net: the rover can arm and execute absolute lat/long waypoints with only a *relative* horizontal estimate. Partially mitigated by the separate `gps_check`, which does not close the window. The duplicate OR strongly implies an intended-but-missing `HORIZ_ABS`. | `spf/mavlink/mavlink_controller.py:264-268` |
| 41 | 🟠 P1 · ⚠️ silently-wrong | **Inference cache returns wrong results.** The cache key `{basename}/{seg_version}/{checkpoint_md5}/{config_md5}.npz` **omits `v4`**, which flows into `v4_to_v5()` and changes outputs; two runs differing only in `v4` (or v4-vs-v5 datasets sharing a basename) **collide on the same `.npz`** and silently return stale/wrong inference. Aggravated by keying on `basename` only (same-name datasets in different dirs also collide). | `spf/model_training_and_inference/models/single_point_networks_inference.py:228` |
| 42 | 🟠 P1 · ⚠️ silently-wrong | **GRBL exception freezes position mid-collection.** `to_steps` raises `PointOutOfBoundsException` at `move_to:376` (outside the method's only `try`); it propagates to `run_planner`'s bare `except` (`:493-494`) which only logs → the daemon motion thread **dies**, but collection keeps recording, reading the now-frozen `controller.position["xy"]` cache → every later record is stamped with the same **stale position**, no operator signal. Most exposed via `CirclePlanner` (no bounds clamp). | `spf/grbl/grbl_interactive.py:376, 493` |
| 43 | 🟠 P1 · safety | **Unmapped flight mode strands the planner.** `handle_HEARTBEAT` does `custom_mode_mapping[msg.custom_mode]` unguarded; an unmapped Rover mode (2, 8, 9, 13, 14, ≥17) raises `KeyError` on the message-loop thread (no try/except in the dispatcher) → kills the daemon thread, freezes all state, strands the planner. | `spf/mavlink/mavlink_controller.py:838` |
| 44 | 🟠 P1 · safety | **RTL / move can hang forever.** `move_to_point` loops on distance with `sleep(0.1)` and **no timeout/abort**; `move_to_home` starts its `max_wait` clock *after* the blocking `move_to_point` already returned, so `max_wait` bounds nothing. An unreachable target hangs the planner thread / RTL indefinitely. | `spf/mavlink/mavlink_controller.py:436, 558` |
| 45 | 🟠 P1 (multi-emitter) / 🟡 P2 (typical) · ⚠️ silently-wrong | **Segmentation drops abutting signal.** `keep_signal_surrounded_by_noise` drops *both* of any two adjacent signal runs that `combine_windows` refused to merge (legitimate multi-emitter / phase-stepping data, phase Δ≥0.2) → silently fewer/zero signal windows, wrong `mean_phase`/empty mask. Fires on the **default production config** (`gpu=False`, default args). Corroborated by a commented-out assert at `test_segmentation.py:42`. | `spf/dataset/segmentation.py:817-828` |

### 🟡 P2 — verified, narrower reach

| # | Issue | Location |
|---|---|---|
| 46 | Cross-thread motion state (`mav_mode`/`armed`/`gps`) read in `run_planner` with **no lock** while the handler thread writes them → torn multi-field decisions (GIL bounds it to logical, not struct, tearing). | `spf/mavlink/mavlink_controller.py:466,803,835` |
| 47 | `get_md5_of_file` caches md5 in a **never-invalidated `.md5` sidecar** (+ `functools.cache`) → an **in-place** edit of a config/checkpoint silently reuses a stale inference cache (no mtime check). | `spf/utils.py:88-103` |
| 48 | `apply_symmetry_rules_to_heatmap` produces a one-short θ axis for **even `bins`** (CLI default `--nbins=50` is even) → misbinned empirical dist. Mitigated: all checked-in artifacts + the test harness use **odd** bins (65/7). | `spf/scripts/create_empirical_p_dist.py:100,130` |
| 49 | `gpu=True` with cupy absent → `NameError` (bare `except:pass` import) inside a Pool worker. Reachable only via the **programmatic** `v5spfdataset(gpu=True)` path; the `segment_zarr.py --gpu` CLI dies earlier at its hard `import cupy`. Default `gpu=False`. | `spf/dataset/segmentation.py:9-12, 504` |
| 50 | `GRBLController.update_status` recurses on every `ok`/blank/unparseable serial line with **no depth cap** → `RecursionError` on chatty/malformed real serial (fake-GRBL can't trigger it). | `spf/grbl/grbl_interactive.py:331-334` |
| 51 | `SPFPairedXYKalmanFilter.metrics` builds `pred_theta` from `x["mu"][0]` = **tx_x (mm)**, comparing position to `craft_ground_truth_thetas` (radians); the correct `craft_theta` key is ignored. Hidden because XY-EKF is config-dead and the test bound is loose (`test_ekf.py:84`, `<5`). | `spf/filters/ekf_dualradioXY_filter.py:159` |
| 52 | Duplicate `def test_single_theta_dual_radioNN` — pytest collects only the second; the first never runs (same failure mode as the `test_circular_mean` collision). | `tests/test_particle_filter.py:110, 170` |

### 🟢 P3 — verified latent/hygiene (this round)

`radios_to_online` calls `close_rx()` with no None-guard → `AttributeError` if an SDR *emitter* fails verification (`data_collector.py:388`, hardware-only) · `ThreadedRX.get_rx` dead `sys.exit(1)` (off-by-one) → returns `None` → `TypeError` downstream, but `read_forever` aborts the thread cleanly anyway (`data_collector.py:261,288`) · `from_steps` positive-y-only branch (real but **dead** — workspace is all y>0) (`grbl_interactive.py:165`) · `compute_downsampled_segmentation_mask` assumes `stride==window_size` (true for the only production args) (`segmentation.py:960`) · `setup_rxtx_and_phase_calibration` never increments `retries` (infinite loop, but gated behind the #6 `AttributeError`) (`sdr_controller.py:1001`) · baseline `line_to_point_distances` is `dtype=int` (truncates) + asymmetric peak-suppression window (`baseline_algorithm.py:54,11`).

## Added by Phase-4 full review (paired training · training branches · inference)

Read-only branch-enumeration audit of the paired (stage-2) path, all non-default training
branches, and the complete inference surface (reports in `reference/_full_review/`). The two
resume P1s were **verified directly** (source read) before promotion; branch counts: 24 works ·
7 broken · 6 dead · 6 risky.

### 🟠 P1

| # | Confidence | Issue | Location |
|---|---|---|---|
| 53 | ✅ verified | **`--resume` silently clobbers `best.pth`.** `best_val_loss_so_far = None` (`:1191`) is never restored from the checkpoint, and the save gate is `== None or <` (`:1275-1292`) → the **first val after any resume overwrites `best.pth` unconditionally**, even with a worse score. **Historical impact CONFIRMED by reading the artifact:** dec15 paired's `best.pth` contains `epoch 37, step 2,150,000`, written Dec 29 15:00 — early segment 3, right after the second resume (val ≈ 0.159–0.160), vs the true best ≈ **0.1548 @ ~739k** (end of segment 1) — the on-disk "best" is ~3–4% worse than the real best. Recovery without retraining: the true-best-region numbered checkpoints survive (`checkpoint_e12_s725000/730000/735000.pth`). Blast radius: no repo config consumes the dec15 paired `best.pth` (only its own training config's `output:` references the dir), and the jun26 lineage is clean (jun26 single was a single-segment run, never resumed). | `spf/scripts/train_single_point.py:1191, 1275-1292`; artifact `checkpoints/jul8/paired_3p7_thin_noblade_dec15/best.pth` |
| 54 | ✅ verified | **`--resume` unfreezes frozen backbones on stage-2.** Resume paths use `force_load=True` (`:1141-1150`); the `requires_grad=False` freeze applied by `load_single`/`load_paired` sits inside `if not force_load:` (`:471-487`) and is never re-applied → a resumed paired run silently trains the single sub-net at lr 2e-4 via the non-detached `single_loss`. **Ops rule: never `--resume` a stage-2 run** without re-freezing. (Current jun26 paired run verified fresh/non-resumed → unaffected.) | `spf/scripts/train_single_point.py:1141-1150, 471-487` |
| 55 | ✅ verified | **Inference-cache save can't write to B2.** The save path is local-fs-only, but the prod filter config points the cache at `b2://projectspf/...` → saves land in a literal local `./b2:/` directory; the remote cache is never populated → silent full recompute per worker. | `single_point_networks_inference.py:236-251`; `ekf_and_pf_config.yml:55,71` |
| 56 | ✅ verified | **Single-NN particle filter omits `crash_if_not_cached`** (defaults True) → any cache miss raises through the `multiprocessing.Pool` and **aborts the entire filter sweep** (the dual-NN path computes on miss instead — inconsistent). | `spf/filters/particle_single_radio_nn_filter.py:44-54` |

### 🟡 P2 (verified unless marked 🔍)

| # | Issue | Location |
|---|---|---|
| 57 | Paired config's `model.input_dropout: 0.2` is **dead** for `pairedbeamformer` — effective input dropout is the single block's 0.1. Config reads differently than it trains. | `single_point_networks.py:997-1018` |
| 58 | `scatter: onehot` is triply broken if enabled: ignores `single=` (front/back fold vanishes), no ±π wrap (reflect-pad), ignores `y_rad` (corrupts rotated diagnostics). Kernel axis itself is correct. | `train_single_point.py:578-597` |
| 59 | ⚠️ no-crash — absolute-mode NN wrapper rotates the `"single"` output by `rx_heading` only; radio-frame output needs `(rx_theta+rx_heading)` → wrong frame. Latent (only `"paired"` is consumed today). | `spf_nn_dataset_wrapper.py:57-61` |
| 60 | Escalates #34: in paired training `vehicle_type` is mutated in place by the single net's forward, then re-read + re-dropped by the paired `PrepareInput` → ~19% correlated double-dropout on that feature (train only). | `single_point_networks.py:527-531` |
| 61 | Frozen single net runs in **train mode** during stage-2 → window shrink/shuffle/dropout + input-dropout stay active on the frozen backbone (stochastic single estimates in training, deterministic at eval). Design note. | `train_single_point.py:478` |
| 62 | 🔍 concurrent workers computing the same missing cache key share one `.tmp.npz` name → truncated-npz publish race. | `single_point_networks_inference.py:250-251` |
| 63 | `--realtime` without `--checkpoint-config` crashes at startup; the `--inference` flag is written but never read (dead). Realtime consumer loop itself is commented out → realtime inference is currently a **no-op** end-to-end. | `mavlink_radio_collection.py:69-70, 259-266, 331-336` |
| 64 | Dual-NN filter hardcodes `reshape(-1, 65)` — breaks any non-65-ntheta checkpoint. | `particle_dual_radio_nn_filter.py:22` |

P3/hygiene: `optim.resume_step` sits in prod configs but nothing reads it; `random_adjacent_stride`
defaulted but never passed; `head_start>0` + frozen single would crash backward (🔍, no config uses
it); `tests/test_v5inference.py:184` runs realtime inference without `.eval()` (test-only).

**All-clears from this round:** paired checkpoint loading, double-flip paired coherence (drawn
once per session, all keys consistent), the craft-frame rotation (`rotate_dist` sign/grid verified
+ tested), paired loss/target (`craft_y_rad`, correctly no front/back fold), `.eval()`+`no_grad`
hygiene on every live inference site, the prod plotting path, wandb resume reattach, and the
radios=2 reshape interleave.

## Notes for triage

**Cheapest confirmed-active fixes (do first):** the three real P0s — delete `breakpoint()`
(#2, #7), and the `spf.scripts`→`spf.filters` import rename (#3) — plus the one-liners #1/#6.

**The most dangerous class is silently-wrong-result (no crash), now the bulk of the P1s:**
- **#40** — rover can arm/launch on a relative-only EKF estimate (safety gate logic bug).
- **#41** — inference cache silently returns wrong results across v4/v5 / same-basename datasets.
- **#42** — a GRBL out-of-bounds point freezes position and stamps every later record stale.
- **#45** — segmentation silently drops abutting multi-emitter signal on the **default** config.
- **#47** — in-place edits of a config/checkpoint reuse a stale inference cache.
These look plausible but are wrong; **verify downstream compensation before any fix** (e.g. #16
turned out to be load-bearing-by-design — do NOT "fix" it).

**Rover safety cluster** (#40, #43, #44, #46, plus the existing #18): the MAVLink message-handler
thread does dangerous/unbounded things (shutdown from a handler, `KeyError`-strands-planner,
no-timeout RTL, unlocked state). Worth a dedicated safety pass before the next field mission.

**Reachability did most of the triage work:** 5 of the original 8 "P0"s downgraded once we checked
whether anything reaches them — and several genuinely-broken functions are simply dead. Severity =
misbehavior × reachability; the per-issue reports in `reference/_p0_investigations/` and the
`VERIFY_*_new.md` files carry the evidence.

## #55 — live breakpoint() calls in realtime NN inference path (FOUR sites)
`spf/dataset/spf_nn_dataset_wrapper.py` has two `breakpoint()` calls left in
(`to_absolute_north` and the realtime branch of `get_and_annotate_entry_at_idx`,
absolute=True path). `mavlink_radio_collection.py --realtime` constructs exactly this
configuration (absolute=True, realtime ds), so the first consumed sample would drop the
field process into pdb. Currently unreachable only because the consumption loop in
mavlink_radio_collection.py:332-335 is COMMENTED OUT (realtime inference wired but
dormant — "still missing realtime pf"). Remove the breakpoints before enabling. UPDATE: review found two more —
`spf/filters/particle_dual_radio_nn_filter.py:41` (PFSingleThetaDualRadioNN.observation,
kills the realtime PF) and `spf/scripts/test.py:13` (module level).

## #56 — rpi5 inference config uses slow_attack while fleet capture used fast_attack
`data_collection/rpi5_inference/inference_config.yaml` sets `rx-gain-mode: slow_attack`
for both receivers; all wall v5 capture configs (training data) used `fast_attack`.
Train/inference AGC-behavior mismatch — gains_input distribution shifts at deployment.

## #57 — realtime inference heading always 0 (absolute bearings are craft-relative)
`data_collector.py:540` sets `data.heading` dynamically; `DataSnapshotV4` has no such
field so `asdict()` drops it → `rx_heading_in_pis=0.0` in every realtime sample. Disk
path unaffected (getattr). With absolute=True the output is silently craft-relative.
Fix: set `data.rx_heading_in_pis = heading/180` before asdict. See
claude_docs/reference/_realtime_inference_review.md.

## #58 — realtime consumer lifecycle: guaranteed crash-or-hang
v5inferencedataset iterator starts at 0; max_store_size=3 evicts → ValueError for any
late/slow consumer; getitem timeout returns None → TypeError in collate; render
exceptions swallowed by executor (last 4 futures unchecked) → 30s/idx infinite hang.

## #59 — v4→v5 heading conversion divides by 720 instead of 180 (4× small)
spf_dataset.py:1298-1300. Arithmetic confirmed; IMPACT UNVERIFIED for merged rover
corpus (separate merge path; audit Δθ evidence suggests rover labels unaffected).
Verify which data flows through this branch before fixing.

## #60 — realtime pushes ~8MB signal_matrix through mp.Queue per snapshot/radio
Plus asdict deep-copy + astype copy + 3 stacked lru_cache(4) retention in the wrapper.
Probable Pi5 throughput blocker at 0.1s/sample.
