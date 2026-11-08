# Pass-1 Surface Inventory — RF / SDR cluster

Read-only surface inventory (Pass 1 per `DOCUMENTATION_GUIDE.md`). Line numbers verified against
source as of this pass. Inputs/outputs/side-effects taken from signature + a single read of the
body; deeper contracts (Pass 2/3) are out of scope here.

Status legend: ✅ active · ⚠️ deprecated/TODO-remove · 🔴 broken (verified) · 🧪 experimental/unused.

---

## `spf/rf.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `rf_linspace` | 35 | func | Cached float64 linspace | `s,e` (floats), `i` (int n) | np.float64 array | `@functools.lru_cache(1024)` (cache state) | ✅ active |
| `pi_norm` | 49 | func | Wrap angle to [-π,π) | `x` (rad) | rad | none; `@njit` | ✅ deep-documented; see rf.md |
| `pi_norm_halfpi` | 54 | func | Wrap angle to [-π/2,π/2) | `x` (rad) | rad | none; `@njit` | 🧪 no callers seen in this file; verify Pass 2 |
| `torch_circular_diff_to_mean` | 59 | func | Min circular dist of angles to each mean | `angles` (tensor rad), `means` (1-D tensor rad) | tensor | asserts `means.ndim==1`; jit commented out | ✅ active |
| `torch_pi_norm_pi` | 72 | func | Torch wrap to [-π,π) | `x` (tensor rad) | tensor | none; jit commented | ✅ active |
| `torch_pi_norm` | 77 | func | Torch wrap to [-max,max) | `x` (tensor), `max_angle` (float=π) | tensor | none | ✅ active |
| `fast_percentile` | 82 | func | k-th lower percentile via partition | `x` (1-D arr), `percentile` (0-100) | scalar | none; njit commented out | 🧪 likely superseded by `fast_percentile_1d`; verify Pass 2 |
| `circular_stddev_fast` | 94 | func | Circular stddev + trimmed, fast loop | `v` (rad arr), `mean` (rad), `trim`=50.0 | (stddev, trimmed_stddev) | none; `@njit` | ✅ active |
| `circular_stddev` | 134 | func | Circular stddev + trimmed (vectorized) | `v` (rad arr), `u` (mean rad), `trim`=50.0 | (stddev, trimmed) | none; `@njit` | ✅ active |
| `torch_circular_stddev` | 157 | func | Torch circular stddev + trimmed | `v` (tensor), `u` (mean), `trim` (float) | (tensor, tensor) | none; jit commented | ✅ active |
| `torch_reduce_theta_to_positive_y` | 181 | func | Fold θ to upper half-plane (torch) | `ground_truth_thetas` (tensor rad) | tensor rad | clones arg (no mutation of input) | ✅ active |
| `reduce_theta_to_positive_y` | 195 | func | Fold θ to upper half-plane (np or torch) | `ground_truth_thetas` (arr/tensor rad) | arr/tensor | copies/clones arg; ⚠️ commented-out np block lines 206-209 (dead) | ✅ active |
| `circular_diff_to_mean_single` | 224 | func | Circular dist of 1-D angles to scalar mean | `angles` (1-D rad), `mean` (float) | arr | asserts ndim==1; `@njit` | ✅ active |
| `circular_diff_to_mean_single_fast` | 234 | func | Same, explicit loop float32 | `angles` (1-D rad), `mean` (float) | float32 arr | none; `@njit` | ✅ active |
| `circular_diff_to_mean` | 247 | func | Circular dist of angles to each of many means | `angles`, `means` (1-D) | arr | asserts; njit commented out | ✅ active |
| `circular_mean_simple_fast` | 255 | func | Per-row circular mean (no trim) | `angles` (2-D rad) | (result, result) float32 | none; `@njit` | ✅ active |
| `circular_mean` | 273 | func | Vectorized circular mean + trimmed | `angles` (2-D rad), `trim`, `weights=None` | (cm, trimmed_cm) rad | asserts ndim==2; marked `# TODO remove` (l.272) | ✅ deep-documented; see rf.md (⚠️ TODO-remove tag) |
| `circular_mean_single_fast` | 300 | func | 1-D circular mean + trimmed, fast | `angles` (1-D), `trim`, `weights=None` | (cm, trimmed) | none; `@njit` | ✅ active |
| `fast_percentile_1d` | 348 | func | k-th percentile via partition (ceil rule) | `x` (1-D), `percentile` | scalar | none; `@njit` | ✅ active |
| `circular_mean_single` | 359 | func | 1-D circular mean + trimmed (np.percentile) | `angles` (1-D), `trim`, `weights=None` | (cm, trimmed) | asserts ndim==1; `@njit` | ✅ active |
| `torch_circular_mean_notrim` | 381 | func | Torch per-row circular mean, no trim | `angles` (2-D tensor) | (r, r) | asserts; jit commented | ✅ active |
| `torch_circular_mean_noweight` | 392 | func | Torch circular mean + trim, no weights | `angles` (2-D), `trim` | (cm, trimmed) | builds `_cm` via py loop; jit commented | ✅ active |
| `mean_phase_mean` | 416 | func | Weighted circular mean (np) | `angles` (rad), `weights` | rad scalar | asserts weights finite | ✅ active |
| `torch_mean_phase_mean` | 424 | func | "Torch" weighted circular mean | `angles`, `weights` | scalar | ⚠️ uses `np.sin`/`np.arctan2` on torch input then `torch_pi_norm` — likely broken for tensors | 🔴 suspect mixed np/torch; verify Pass 3 |
| `torch_circular_mean` | 432 | func | Torch circular mean + trim + weights | `angles` (2-D), `trim`, `weights=None` | (cm, trimmed) | py loop over rows | ✅ active |
| `torch_get_stats_for_signal` | 460 | func | (trimmed_cm, stddev, abs median) torch | `v` (tensor), `pd` (phase diff), `trim` | hstack 3-vec | none; jit commented | ✅ active |
| `fast_median_abs` | 472 | func | Median of abs via partition | `x` (1-D) | scalar | none; `@njit` | ✅ active |
| `get_stats_for_signal` | 490 | func | (trimmed_cm, stddev, abs median) njit | `v`, `pd`, `trim` | 3-tuple | none; `@njit` | ✅ active |
| `torch_windowed_trimmed_circular_mean_and_stddev` | 500 | func | Windowed stats over phase diff (torch) | `v`,`pd`,`window_size`,`stride`,`trim` | (step_idxs, step_stats) | asserts divisibility; jit commented | ✅ active |
| `fast_abs_median` | 528 | func | Median of abs (duplicate of `fast_median_abs`) | `x` (1-D) | scalar | none; `@njit` | ⚠️ near-exact duplicate of l.472 |
| `circular_diff_to_mean_single_scalar` | 548 | func | Circular dist scalar-to-scalar | `angle`, `mean` (rad) | scalar | none; `@njit` | ✅ active |
| `windowed_trimmed_circular_mean_and_stddev_fast2` | 557 | func | Windowed stats, prange, 4-col output | `v`,`pd`,`window_size`,`stride`,`trim` | (idxs n, stats nx4) | njit commented out (decorator on l.556 disabled); uses `prange` w/o njit | ⚠️ `@njit(parallel)` commented; `prange` runs serial; also calls `circular_diff_to_mean_single_fast` with scalar args (l.612,625) → likely 🔴 |
| `windowed_trimmed_circular_mean_and_stddev_fast` | 657 | func | Windowed stats, prange, 3-col output | `v`,`pd`,`window_size`,`stride`,`trim`=50.0 | (idxs nx2, stats nx3) | asserts divisibility; `@njit` | ✅ active |
| `get_stats_for_signal_window` | 686 | func | Per-window stats helper | `v`,`pd`,`start_idx`,`end_idx`,`trim` | 3-vec float32 | none; `@njit` | ✅ active |
| `windowed_trimmed_circular_mean_and_stddev` | 698 | func | Windowed stats (non-fast, serial) | `v`,`pd`,`window_size`,`stride`,`trim`=50.0 | (idxs nx2, stats nx3) | asserts; `@njit` | ✅ active |
| `phase_diff_to_theta` | 721 | func | Phase diff → θ candidates (arcsin) | `phase_diff`,`wavelength`,`distance_between_receivers`,`large_phase_goes_right` | (x, π-x, mid) rad | copies arg if flip; `@njit` | ✅ deep-documented; see rf.md |
| `get_phase_diff` | 745 | func | RX0-RX1 phase diff, wrapped | `signal_matrix` (2×N complex) | rad arr | `@njit` | ✅ active |
| `torch_get_phase_diff` | 750 | func | Torch phase diff | `signal_matrix` (...,2,N tensor) | tensor rad | jit commented | ✅ active |
| `get_avg_phase` | 755 | func | Avg phase via `circular_mean` | `signal_matrix`, `trim`=0.0 | 2-vec | njit commented out | ✅ active (used in data_collector) |
| `get_avg_phase_fast` | 762 | func | Avg phase via simple_fast | `signal_matrix` | 2-vec | njit commented out | ✅ active |
| `get_avg_phase_fast2` | 769 | func | Avg phase, inline loop | `signal_matrix` | float32 2-vec | `@njit` | ✅ active (used in data_collector) |
| `torch_get_avg_phase_notrim` | 785 | func | Torch avg phase no trim | `signal_matrix` (tensor) | tensor | jit commented | ✅ active |
| `torch_get_avg_phase` | 794 | func | Torch avg phase + trim | `signal_matrix`, `trim` | tensor | wraps in `torch.tensor(...)` | ✅ active |
| `rotation_matrix` | 803 | func | 2×2 rotation matrix (cached) | `orientation` (rad) | 2×2 float64 | `@functools.lru_cache(1024)` | ✅ active |
| `Source` | 813 | class | Base RF point source (unit cos+jsin) | `pos` (N×2) | — | asserts pos.shape[1]==2 | ✅ active (sim) |
| `Source.signal` | 818 | method | Unit-freq IQ signal | `sampling_times` | complex arr | none | ✅ active |
| `Source.demod_signal` | 823 | method | Identity demod | `signal`, `demod_times` | signal | none | ✅ active |
| `IQSource` | 827 | class | IQ source at frequency/phase | `pos,frequency,phase=0,amplitude=1` | — | — | ✅ active |
| `IQSource.signal` | 834 | method | cos+jsin at freq/phase | `sampling_times` | complex arr | none | ✅ active |
| `SinSource` | 841 | class | Real sin source | `pos,frequency,phase=0,amplitude=1` | — | — | ✅ active |
| `SinSource.signal` | 848 | method | amplitude·sin | `sampling_times` | real arr | none | ✅ active |
| `MixedSource` | 854 | class | Product of two sources | `pos,source_a,source_b,h=None` | — | — | 🧪 `h` unused; product calls sources as callables | ⚠️ unverified usage |
| `MixedSource.signal` | 861 | method | a(t)·b(t) | `sampling_times` | arr | none; ⚠️ calls `self.source_a(...)` (source not callable?) | ⚠️ unverified |
| `NoiseWrapper` | 865 | class | Add complex Gaussian noise to source | `internal_source,sigma=1` | — | — | ✅ active |
| `NoiseWrapper.signal` | 871 | method | signal + N(0,σ) complex | `sampling_times` (2-D) | complex arr | asserts ndim==2; **randomness** (`np.random.randn`) | ✅ active |
| `Detector` | 883 | class | Antenna array sim w/ sources | `sampling_frequency,orientation=0,sigma=0` | — | — | ✅ active |
| `Detector.add_source` | 893 | method | Append source + stack pos | `source` | — | mutates `self.sources`, `self.source_positions` | ✅ active |
| `Detector.distance_receiver_to_source` | 905 | method | Pairwise rx↔src distances | — | arr | none | ✅ active |
| `Detector.rm_sources` | 911 | method | Clear sources | — | — | mutates self | ✅ active |
| `Detector.set_receiver_positions` | 915 | method | Set rx positions | `receiver_positions` | — | mutates self | ✅ active |
| `Detector.add_receiver` | 918 | method | Append rx position | `receiver_position` | — | mutates self | ✅ active |
| `Detector.n_receivers` | 931 | method | Count rx | — | int | none | ✅ active |
| `Detector.all_receiver_pos` | 935 | method | Rotated (+offset) rx positions | `with_offset=True` | n×2 float32 | none (uses cached `rotation_matrix`) | ✅ active |
| `Detector.receiver_pos` | 946 | method | Single rx position rotated | `receiver_idx`,`with_offset=True` | float32 | none | ✅ active |
| `Detector.get_signal_matrix` | 961 | method | Simulate received IQ matrix | `start_time`,`duration`,`rx_lo=0` | n_rx×n_samples complex | **randomness** if sigma>0; heavy compute | ✅ active |
| `linear_receiver_positions` | 1020 | func | Zero-centered ULA positions (cached) | `n_elements`,`spacing` | n×2 float32 | `@functools.lru_cache` | ✅ active |
| `ULADetector` | 1026 | class | Uniform linear array detector | `sampling_frequency,n_elements,spacing,sigma=0,orientation=0` | — | sets rx positions | ✅ active (used by sdr_controller) |
| `circular_receiver_positions` | 1035 | func | UCA positions (cached) | `n_elements`,`radius` | n×2 | `@functools.lru_cache` | ✅ active |
| `UCADetector` | 1041 | class | Uniform circular array detector | `sampling_frequency,n_elements,radius,sigma=0,orientation=0` | — | sets rx positions | ✅ active |
| `get_thetas` | 1049 | func | θ grid + unit vectors (cached) | `spacing` (n) | (thetas, n×2 vecs) | `@functools.lru_cache` | ✅ active |
| `dbfs` | 1103 | func | IQ → dBFS FFT magnitude | `raw_data` | dB arr | none | ✅ active (Jon Kraft) |
| `precompute_steering_vectors` | 1173 | func | Steering vectors for θ grid | `receiver_positions`,`carrier_frequency`,`spacing=65`,`calibration=None`,`offset=0` | n_thetas×n_ant complex | none | ✅ deep-documented; see rf.md |
| `thetas_from_nthetas` | 1200 | func | θ grid -π..π | `nthetas` | arr | none | ✅ deep-documented; see rf.md |
| `beamformer_given_steering_nomean_fast` | 1204 | func | Per-sample beam power (njit core) | `steering_vectors`,`signal_matrix` | n_thetas×n_samples | raises ValueError on bad dtype | ✅ deep-documented; see rf.md |
| `beamformer_given_steering_nomean_fast_core` | 1222 | func | Inner triple-loop beam power | `steering_vectors`,`signal_matrix`,`output` | — | **mutates `output`**; `@njit(parallel=True)` (prange) | ✅ deep-documented; see rf.md |
| `beamformer_given_steering_nomean_cp` | 1240 | func | GPU/np beam power (no mean) | `steering_vectors`,`signal_matrix` | abs(dot) | none; jit commented (cupy) | ✅ deep-documented; see rf.md |
| `beamformer_given_steering_nomean` | 1273 | func | CPU beam power (no mean) | `steering_vectors`,`signal_matrix` | abs(dot) | none | ✅ deep-documented; see rf.md |
| `beamformer_given_steering` | 1309 | func | Time-averaged beam power spectrum | `steering_vectors`,`signal_matrix` | n_thetas | none | ✅ deep-documented; see rf.md |
| `beamformer_thetas` | 1336 | func | θ grid -π..π | `spacing` | arr | none | ✅ deep-documented; see rf.md |
| `beamformer` | 1352 | func | Full beamform (build SV + power) | `receiver_positions`,`signal_matrix`,`carrier_frequency`,`calibration=None`,`spacing=65`,`offset=0` | (thetas, sds, SV) | none | ✅ deep-documented; see rf.md |
| `get_peaks_for_2rx` | 1436 | func | Geometric peak indices for 2-RX beam | `beam_former_output` (1-D) | (i,i,i) ints | none | ⚠️ test-only (no prod caller; used in tests) |
| `get_top_3_peaks` | 1454 | func | Top-3 prominent peaks (scipy) | `beam_former_output`,`distance=1`,`prominence=0.1` | list ≤3 idx | none (chatgpt-authored) | 🧪 experimental |
| `rotate_dist` | 1493 | func | Rotate discrete angular dist (torch) | `input_dist` (B×n), `rotations` (B rad) | B×n normalized | none (chatgpt-authored) | ✅ Active — ~6 import sites (training, particle filters, single_point_networks, dataset wrapper) [adversarial-pass correction] |

**Notable (rf.py):**
- Massive duplication of the circular-mean / circular-stddev / windowed-stats family across 4
  flavors (np-vectorized, njit-loop "fast", torch, scalar). E.g. `fast_median_abs` (472) and
  `fast_abs_median` (528) are near-identical; `fast_percentile` (82, njit-disabled) vs
  `fast_percentile_1d` (348). Heavy Pass-2 dedup target.
- `torch_mean_phase_mean` (424) mixes `np.sin`/`np.arctan2` on its inputs then calls
  `torch_pi_norm` — almost certainly broken for torch tensors. Flag 🔴 for Pass 3.
- `windowed_trimmed_circular_mean_and_stddev_fast2` (557) has its `@njit(parallel=True)`
  decorator commented out yet uses `prange`, and calls `circular_diff_to_mean_single_fast`
  (an array function) with scalar element args at l.612/625 — suspect dead/broken path.
- Many `# @torch.jit.script` / `# @njit` decorators are commented out (jit globally disabled per
  l.22 `# numba = False`); the "fast" variants therefore may not actually be JIT-compiled.
- A large block of `beamformer_numba*` code (l.1054-1099) is fully commented out (dead).
- `pi_norm_halfpi` (54), `fast_percentile` (82), `get_top_3_peaks` look unused within this
  module — confirm fan-in in Pass 2. _(Adversarial-pass correction: `rotate_dist` is NOT unused —
  ~6 import sites repo-wide, it's ✅ active core. `fast_percentile` and `get_peaks_for_2rx` are
  test-only, not fully dead.)_

---

## `spf/data_collector.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `ThreadPoolExecutorWithQueueSizeLimit` | 35 | class | ThreadPoolExecutor with bounded queue | `maxsize=50,*a,**kw` | — | replaces `_work_queue` | ✅ active |
| `ProcessPoolExecutorWithQueueSizeLimit` | 41 | class | ProcessPool wrapper w/ semaphore cap | `max_workers=None,maxsize=0` | — | spawns processes; semaphore | 🧪 not obviously used here; verify Pass 2 |
| `ProcessPoolExecutorWithQueueSizeLimit.submit` | 49 | method | Submit w/ semaphore acquire | `fn,*a,**kw` | future | acquires semaphore; callback releases | 🧪 |
| `ProcessPoolExecutorWithQueueSizeLimit.shutdown` | 57 | method | Shut down pool | `wait=True` | — | shuts down executor | 🧪 |
| `ProcessPoolExecutorWithQueueSizeLimit.__enter__/__exit__` | 61/64 | method | Context manager | — | self/— | calls shutdown on exit | 🧪 |
| `DataSnapshotRaw` | 69 | class (dataclass) | Raw per-sample record | signal_matrix, timestamps, rssis, gains, rx geom, avg_phase_diff, heading | — | none | ✅ active |
| `DataSnapshotV4` | 83 | class (dataclass) | V4 record (+gps) | adds gps_timestamp/lat/long | — | none | ✅ active |
| `DataSnapshotV5` | 90 | class (dataclass) | V5 record (+tx/rx mm positions) | adds tx/rx pos mm | — | none | ✅ active |
| `DataSnapshot` | 98 | class (dataclass) | Legacy V2/V3 record (beam_sds) | timestamp, rx geom, beam_sds, signal_matrix, rssis, gains | — | none | ⚠️ legacy (V2/V3) |
| `prepare_record_entry_v3` | 110 | func | Pack V3 row (gps time split to 2 f32) | `ds:DataSnapshot`, `current_pos_heading_and_time` | hstack arr | struct pack/unpack | ⚠️ legacy V3 |
| `prepare_record_entry_v2` | 134 | func | Pack V2 row | `ds`, `rx_pos`, `tx_pos` | hstack arr | none | ⚠️ legacy V2 (used by GrblDataCollector) |
| `data_to_snapshot` | 151 | func | Build legacy DataSnapshot w/ beamform | `current_time,signal_matrix,steering_vectors,rssis,gains,rx_config` | DataSnapshot | calls beamformer + get_avg_phase | ⚠️ legacy (used by ThreadedRX) |
| `ThreadedRX` | 173 | class | Threaded SDR reader → DataSnapshot | `pplus,time_offset,nthetas,seconds_per_sample=0` | — | asserts rx_pos set | ⚠️ legacy V2/V3 path |
| `ThreadedRX.read_forever` | 185 | method | Loop: read, queue, pace, recalibrate | — | — | **thread loop**; queue I/O; logging; `soft_reset_radio` (hardware); sleeps; mutates self | ✅ active |
| `ThreadedRX.join` | 232 | method | Join read thread | — | — | blocks | ✅ active |
| `ThreadedRX.start_read_thread` | 235 | method | Spawn thread or process | `thread=True` | — | **spawns thread/process**; sets self.run | ✅ active |
| `ThreadedRX.get_rx` | 245 | method | Read raw SDR + rssi/gain w/ retries | `max_retries=15` | dict or None | **SDR hardware I/O**; logging; `sys.exit(1)` on give-up; sleeps | ✅ active |
| `ThreadedRX.get_data` | 264 | method | get_rx → DataSnapshot | — | DataSnapshot | raises ValueError on None; time read | ✅ active |
| `ThreadedRXRaw` | 282 | class | Raw reader → snapshot_class (V4/V5) | — | — | — | ✅ active |
| `ThreadedRXRaw.get_data` | 283 | method | get_rx → raw snapshot + avg phase | — | snapshot | SDR I/O (via get_rx); asserts spacing>0.001; **randomness/time** | ✅ active |
| `ThreadedRXRawV4` | 306 | class | Raw reader bound to DataSnapshotV4 | `**kwargs` | — | sets snapshot_class | ✅ active |
| `ThreadedRXRawV5` | 314 | class | Raw reader bound to DataSnapshotV5 | `**kwargs` | — | sets snapshot_class | ✅ active |
| `DataCollector` | 322 | class | Base orchestrator (radios→threads→zarr) | `yaml_config,data_filename,position_controller,thread_class,tag=""` | — | calls `setup_record_matrix` | ✅ active |
| `DataCollector.radios_to_online` | 340 | method | Bring radios online + setup emitter/rx | — | — | **hardware setup (SDR tx/rx)**; logging; `sys.exit(1)` on fail; sleeps; mutates self | ✅ active |
| `DataCollector.prepare_threads` | 413 | method | Build + start reader threads | — | — | **spawns reader threads**; builds collector thread | ✅ active |
| `DataCollector.start` | 435 | method | Start collector thread | — | — | starts thread | ✅ active |
| `DataCollector.done` | 438 | method | Join collector | — | — | blocks | ✅ active |
| `DataCollector.is_collecting` | 441 | method | Collecting flag | — | bool | none | ✅ active |
| `DataCollector.setup_record_matrix` | 444 | method | Abstract | — | — | raises NotImplementedError | ✅ active (abstract) |
| `DataCollector.write_to_record_matrix` | 447 | method | Abstract | — | — | raises NotImplementedError | ✅ active (abstract) |
| `DataCollector.run_inner_collector_thread` | 450 | method | Pull from queues, submit writes | — | — | thread-pool I/O; logging tracebacks | ✅ active |
| `DataCollector.run_collector_thread` | 482 | method | Main collector loop + teardown | — | — | logging; calls close; joins threads; mutates self.run flags | ✅ active |
| `DataCollector.close` | 501 | method | No-op base close | — | — | none | ✅ active |
| `DroneDataCollectorRaw` | 506 | class | V4 drone collector (zarr + realtime inf) | `realtime_v5inf,*a,**kw` | — | — | ✅ active |
| `DroneDataCollectorRaw.setup_record_matrix` | 515 | method | Create V4 zarr dataset | — | — | **file/zarr creation (LMDB)**; asserts buffer sizes | ✅ active |
| `DroneDataCollectorRaw.write_to_record_matrix` | 536 | method | Write row + GPS + realtime inf | `thread_idx,record_idx,data` | — | mutates `data`; **zarr write**; realtime inf write | ✅ active |
| `DroneDataCollectorRaw.close` | 555 | method | Close + shrink zarr | — | — | **file close + zarr_shrink**; logging | ✅ active |
| `GrblDataCollectorRaw` | 564 | class | V5 grbl collector | `*a,**kw` | — | — | ✅ active |
| `GrblDataCollectorRaw.setup_record_matrix` | 572 | method | Create V5 zarr dataset | — | — | **zarr creation**; honors dry-run; asserts | ✅ active |
| `GrblDataCollectorRaw.write_to_record_matrix` | 593 | method | Write V5 row + tx/rx mm | `thread_idx,record_idx,data` | — | reads controller position; mutates data; **zarr write**; asserts rx_lo>1 | ✅ active |
| `GrblDataCollectorRaw.close` | 613 | method | Close + shrink V5 zarr | — | — | **file close + shrink**; logging | ✅ active |
| `GrblDataCollector` | 624 | class | V2 grbl collector (memmap) | `*a,**kw` | — | — | ⚠️ legacy V2 |
| `GrblDataCollector.setup_record_matrix` | 632 | method | Create float32 memmap | — | — | **creates np.memmap file** | ⚠️ legacy V2 |
| `GrblDataCollector.write_to_record_matrix` | 645 | method | Write V2 row to memmap | `thread_idx,record_idx,data` | — | reads controller pos; **memmap write** | ⚠️ legacy V2 |

**Notable (data_collector.py):**
- Three parallel data-format paths: V2 (memmap, `GrblDataCollector`/`ThreadedRX`/`DataSnapshot`),
  V4 (zarr, drone+GPS), V5 (zarr, grbl xy). V2 path is legacy; V4/V5 are current.
- `ThreadedRX.get_rx` calls `sys.exit(1)` (l.261) on repeated SDR failure — a library-level hard
  exit, surprising side effect worth flagging for Pass 3.
- `ProcessPoolExecutorWithQueueSizeLimit` appears defined but unused in this file (the collector
  uses `ThreadPoolExecutorWithQueueSizeLimit`); verify fan-in.
- `DroneDataCollectorRaw.write_to_record_matrix` sets `data.heading` (l.540) but `DataSnapshotV4`
  stores `rx_heading_in_pis`, not `heading` — possible attribute mismatch; flag Pass 3.

---

## `spf/sdrpluto/sdr.py`

> ⚠️ Module-level side effects: imports `matplotlib.pyplot`; at import time it **constructs
> `ula_d` (l.212), `uca_d` (l.228) and creates two matplotlib figures + calls `plot_space`**.
> This is an older/standalone sim variant; classes here are duplicates of those in `rf.py`.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `Source` | 16 | class | Base point source (sim) | `pos` | — | none | 🧪 duplicate of rf.py Source |
| `Source.signal` | 20 | method | Unit IQ signal | `sampling_times` | complex arr | none | 🧪 |
| `Source.demod_signal` | 25 | method | Identity demod | `signal,demod_times` | signal | none | 🧪 |
| `SinSource` | 29 | class | Sin/IQ source | `pos,frequency,phase` | — | none | 🧪 |
| `SinSource.signal` | 35 | method | cos+jsin | `sampling_times` | complex arr | none | 🧪 |
| `MixedSource` | 42 | class | Product source | `pos,source_a,source_b` | — | none | 🧪 |
| `MixedSource.signal` | 48 | method | a(t)·b(t) | `sampling_times` | arr | none | 🧪 |
| `QAMSource` | 52 | class | QAM modulated source | `pos,carrier_frequency,signal_frequency,sigma=0` | — | none | 🧪 unused experimental |
| `QAMSource.signal` | 60 | method | QAM signal + noise | `sampling_times` | arr | **randomness** | 🧪 |
| `QAMSource.demod_signal` | 70 | method | QAM demod | `signal,demod_times` | arr | ⚠️ calls SinSource instances as callables (likely broken) | 🔴 suspect |
| `NoiseWrapper` | 76 | class | Add Gaussian noise | `internal_source,sigma=1` | — | none | 🧪 |
| `NoiseWrapper.signal` | 82 | method | signal + noise | `sampling_times` | complex arr | **randomness** | 🧪 |
| `Receiver` | 88 | class | Holds rx position | `pos` | — | none | 🧪 |
| `Detector` | 93 | class | Sim detector (list-based) | `sampling_frequency` | — | none | 🧪 duplicate of rf.py |
| `Detector.add_source` | 99 | method | Append source | `source` | — | mutates self | 🧪 |
| `Detector.rm_sources` | 102 | method | Clear sources | — | — | mutates self | 🧪 |
| `Detector.add_receiver` | 105 | method | Append receiver | `receiver` | — | mutates self | 🧪 |
| `Detector.get_signal_matrix` | 108 | method | Simulate rx matrix | `start_time,duration,rx_lo=0` | n_rx×n complex | none (sigma not applied) | 🧪 |
| `beamformer` | 128 | func | Slow per-θ beamformer (sim) | `detector,signal_matrix,carrier_frequency,calibration=None,spacing=65` | (thetas, sds, SV) | none | 🧪 duplicate/legacy of rf.beamformer |
| `plot_space` | 162 | func | Plot array + source geometry | `ax,d,wavelength=1` | — | **matplotlib drawing** | 🧪 |
| `ULADetector` | 205 | class | ULA (list-based) | `sampling_frequency,n_elements,spacing` | — | adds receivers | 🧪 duplicate of rf.py |
| `UCADetector` | 217 | class | UCA (list-based) | `sampling_frequency,n_elements,radius` | — | adds receivers | 🧪 duplicate of rf.py |

**Notable (sdr.py):**
- This entire module looks like a **legacy / standalone duplicate** of the simulation classes in
  `rf.py`. It has **import-time side effects** (creates detectors + matplotlib figures at
  l.212-230) which is dangerous if imported as a library. Strong 🧪/⚠️ candidate; confirm whether
  anything imports it (Pass 2). Note: this is NOT the production Pluto driver — that lives in
  `sdr_controller.py`.
- `QAMSource.demod_signal` and `MixedSource.signal` call `SinSource`/source *instances* as if
  callable; instances are not callable → would raise `TypeError`. 🔴 suspect.

---

## `spf/sdrpluto/sdr_controller.py`

> Module-level: `pplus_online = {}` (global radio registry, l.58), `run_radios = True` (global
> shutdown flag, l.60), `c = 3e8` (l.56). This IS the production Pluto/BladeRF driver layer.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `bladerf_serial_to_info` | 33 | func | Map bladeRF serial→URI | — | dict | **enumerates USB devices**; prints on error | ✅ active |
| `Config` | 63 | class | Base config w/ repr | — | — | none | ✅ active |
| `Config.__repr__` | 64 | method | Pretty repr of attrs | — | str | none | ✅ active |
| `ReceiverConfig` | 74 | class | RX config + computes rx_pos via ULA | many (lo, bw, sr, uri, gains, spacing, theta...) | — | logging; builds `ULADetector` → rx_pos/rx_pos_rotated | ✅ active |
| `EmitterConfig` | 130 | class | TX config | lo, bw, sr, uri, gains, cyclic... | — | none | ✅ active |
| `rx_config_from_receiver_yaml` | 156 | func | Build ReceiverConfig from yaml dict | `receiver_yaml` | ReceiverConfig | none | ✅ active |
| `args_to_rx_config` | 182 | func | Build ReceiverConfig from argparse | `args` | ReceiverConfig | none | ✅ active (CLI) |
| `args_to_tx_config` | 199 | func | Build EmitterConfig from argparse | `args` | EmitterConfig | none | ✅ active (CLI) |
| `shutdown_radios` | 212 | func | Set global `run_radios=False` | — | — | **mutates global** | ✅ active |
| `get_uri` | 217 | func | Resolve URI from configs | `rx_config,tx_config,uri` | str | asserts consistency | ✅ active |
| `get_pplus` | 231 | func | Get/create PPlus/BladeRF/Fake by URI | `rx_config,tx_config,uri` | PPlus-like | **mutates global `pplus_online`**; logging; ⚠️ "not thread safe" (l.230); opens hardware | ✅ active |
| `FakeSdr` | 253 | class | Fake SDR (random data) | — | — | none | 🧪 test/fake |
| `FakeSdr.set_buffer_size` | 254 | method | Set buffer size | `buffer_size` | — | mutates self | 🧪 |
| `FakeSdr.rx` | 257 | method | Random 2×N signal | — | 2×N arr | sleeps 0.3s; **randomness** | 🧪 |
| `FakeSdr.tx` | 261 | method | No-op tx | `_` | — | none | 🧪 |
| `BladeRFSdr` | 265 | class | BladeRF 2.0 driver | `uri,rx_config,tx_config,phase_calibration=0` | — | **opens bladeRF hardware**; logging; asserts | ✅ active |
| `BladeRFSdr.soft_reset_radio` | 309 | method | No-op | — | — | none | ✅ active |
| `BladeRFSdr.rssis` | 312 | method | Mean RSSI per channel | — | arr | **reads hardware RSSI** | ✅ active |
| `BladeRFSdr.gains` | 323 | method | Returns [0,0] (stubbed) | — | arr | none | ⚠️ stubbed (gains hardcoded 0) |
| `BladeRFSdr.set_config` | 333 | method | Store rx/tx config | `rx_config,tx_config` | — | asserts (tx must be None); mutates self | ✅ active |
| `BladeRFSdr.close` | 349 | method | No-op | — | — | none | ✅ active |
| `BladeRFSdr.setup_rx_config` | 352 | method | Configure AD9361 regs + sync | — | — | **writes RFIC registers (hardware)**; prints; asserts; sync_config | ✅ active |
| `BladeRFSdr.setup_tx_config` | 479 | method | No-op | — | — | none | 🧪 stub |
| `BladeRFSdr.close_rx/close_tx` | 482/485 | method | No-op | — | — | none | ✅ active (stub) |
| `BladeRFSdr.check_for_freq_peak` | 488 | method | Always True | — | True | none | ⚠️ stub (no real check) |
| `BladeRFSdr.rx` | 491 | method | Sync RX → 2×N int16 complex | — | 2×N arr | **reads hardware samples** | ✅ active |
| `FakePPlus` | 502 | class | Fake PPlus wrapper | `uri,rx_config,tx_config,phase_calibration=0` | — | none | 🧪 test/fake |
| `FakePPlus.rx` | 518 | method | Random data via FakeSdr | `num_samples=None` | arr | randomness (via FakeSdr) | 🧪 |
| `FakePPlus.soft_reset_radio` | 521 | method | Sleep 0.1 | — | — | sleeps | 🧪 |
| `FakePPlus.rssis/gains` | 524/527 | method | Random 2-vec | — | arr | **randomness** | 🧪 |
| `FakePPlus.get_rssi_and_gain` | 530 | method | Placeholder rssi/gain | — | arr | ⚠️ reads `self.dev.*` (undefined attr) → would error | 🔴 suspect dead |
| `FakePPlus.set_config` | 561 | method | Store config + new FakeSdr | `rx_config,tx_config` | — | mutates self; asserts | 🧪 |
| `FakePPlus.close / setup_* / close_* / check_for_freq_peak` | 576-592 | method | Stubs | — | — | mostly none | 🧪 |
| `PPlus` | 595 | class | Production PlutoSDR driver | `uri,rx_config,tx_config,phase_calibration=0` | — | **opens `adi.ad9361` hardware**; logging; sleeps; destroys buffers | ✅ active |
| `PPlus.rx` | 630 | method | Read samples | `num_samples=None` | arr | **hardware RX** | ✅ active |
| `PPlus.soft_reset_radio` | 633 | method | Disabled (early return); LO toggle dead | — | — | none (body after return is dead) | ⚠️ effectively no-op |
| `PPlus.get_rssi_and_gain` | 641 | method | Read RSSI+gain via iio | — | 4-vec | **reads hardware attrs**; has leftover `# breakpoint()` (l.644) | ✅ active |
| `PPlus.rssis` | 656 | method | Read RSSI per channel | — | 2-vec | **hardware read** | ✅ active |
| `PPlus.gains` | 664 | method | Read hardwaregain per channel | — | 2-vec | **hardware read** | ✅ active |
| `PPlus.set_config` | 672 | method | Store rx/tx config | `rx_config,tx_config` | — | asserts; mutates self | ✅ active |
| `PPlus.close` | 686 | method | Close TX | — | — | logging; close_tx | ✅ active |
| `PPlus.__del__` | 691 | method | Destructor closes TX | — | — | **hardware tx off**; mutates sdr | ✅ active |
| `PPlus.setup_rx_config` | 703 | method | Full Pluto RX setup (regs, lo, gain) | — | — | **extensive hardware config**; asserts; destroys buffers | ✅ active |
| `PPlus.setup_tx_config` | 798 | method | Full Pluto TX setup | — | — | **hardware TX config**; logging; sleeps; asserts | ✅ active |
| `PPlus.close_tx` | 837 | method | Disable + destroy TX buffer | — | — | **hardware tx off**; swallows TypeError | ✅ active |
| `PPlus.close_rx` | 849 | method | Clear rx_config | — | — | mutates self | ✅ active |
| `PPlus.check_for_freq_peak` | 856 | method | Verify IF peak present in FFT | — | bool | **hardware RX** + FFT | ✅ active |
| `make_tone` | 870 | func | Build cyclic TX tone buffer | `tx_config:EmitterConfig` | complex arr | none | ✅ active |
| `setup_rx` | 885 | func | Bring an RX online (config + warmup) | `rx_config,provided_pplus_rx=None` | PPlus | logging; **hardware setup + warmup rx**; sleeps | ✅ active |
| `setup_rxtx` | 905 | func | Bring RX+TX online, verify peak, retry | `rx_config,tx_config,leave_tx_on=False,provided_pplus_rx=None` | (rx,tx) or (None,None) | **hardware tx/rx**; logging; sleeps; honors global run_radios | ✅ active |
| `setup_rxtx_and_phase_calibration` | 959 | func | Inter-antenna phase calibration | `rx_config,tx_config,tolerance=0.01,n_calibration_frames=800,...` | (rx,tx) or (None,None) | **hardware rx/tx**; logging; ⚠️ calls `pplus_rx.setup_rx()` (l.971,982) — method not defined on PPlus | 🔴 suspect (calls missing `setup_rx`) |
| `plot_recv_signal` | 1054 | func | Live plot rx signal + beamform | `pplus_rx,nthetas,fig,axs,frames=-1,title,signal_matrixs` | — | **matplotlib live drawing**; hardware rx; honors run_radios | ✅ active (debug tool) |
| `__main__` block | 1153 | script | CLI: rx / rxcal / tx modes | argparse | — | logging to file; **hardware**; `sys.exit`; `time.sleep(1800)` | ✅ active (CLI) |

**Notable (sdr_controller.py):**
- `setup_rxtx_and_phase_calibration` (959) calls `pplus_rx.setup_rx()` at l.971 and l.982, but
  `PPlus` defines `setup_rx_config`, not `setup_rx` — this would raise `AttributeError`. Strong
  🔴 candidate (the function may be dead/unused, or relies on a different object). Flag Pass 2/3.
- Global mutable state: `pplus_online` registry + `run_radios` flag, plus `get_pplus` explicitly
  marked "not thread safe" (l.230). Concurrency hazard with the threaded readers in
  `data_collector.py`.
- `BladeRFSdr.gains` returns hardcoded `[0,0]`; `BladeRFSdr.check_for_freq_peak` always returns
  True; `PPlus.soft_reset_radio` is an early-return no-op with dead code after. Several behaviors
  are stubbed.
- `FakePPlus.get_rssi_and_gain` references `self.dev` which is never set → dead/broken path.
- Leftover `# breakpoint()` at l.644.

---

## `spf/sdrpluto/detrend.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `detrend_1d` | 10 | func | Per-window linear detrend (torch) | `x` (1-D tensor), `remove,window_size=1024,merge_similar,thresholds,pad_mode,pad_value` | (x_detrended, slopes, intercepts) | raises ValueError on bad input; uses `torch.linalg.lstsq` | 🧪 likely unused (np path is used) |
| `detrend_1d.segment_1d` (nested) | 69 | func | Window + pad helper (torch) | `signal,wsize` | (x_wins,pad_len) | raises on bad pad | 🧪 nested |
| `detrend_1d.fit_slopes_intercepts` (nested) | 97 | func | Batch lstsq fit (torch) | `win` | (slopes,intercepts,trend) | none | 🧪 nested |
| `detrend_1d.can_merge/merge_two` (nested) | 131/136 | func | Merge similar windows | seg dicts | bool / merged seg | mutates segA | 🧪 nested |
| `merge_dynamic_windows` | 205 | func | Detrend complex tensor (real+imag) | `x` (complex tensor), remove flags, thresholds... | (x_det, 4× slope/intercept) | raises ValueError on non-complex/non-1D | 🧪 likely unused |
| `detrend_1d_np` | 301 | func | Per-window linear detrend (numpy) | `x` (1-D arr), remove, window_size, merge_similar... | (x_det, slopes, intercepts) | raises ValueError; loops lstsq | 🧪 likely unused (fastwindow path used) |
| `detrend_1d_np.segment_1d` (nested) | 357 | func | Window + pad (np) | `data,wsize` | (x_wins,pad_len) | raises on bad pad | 🧪 nested |
| `detrend_1d_np.fit_slopes_intercepts_np` (nested) | 389 | func | Row-by-row lstsq (np) | `wins` | (slopes,intercepts,trends) | none | 🧪 nested |
| `detrend_1d_np.can_merge/merge_two` (nested) | 433/438 | func | Merge helpers (np) | seg dicts | bool/seg | mutates segA | 🧪 nested |
| `merge_dynamic_windows_np` | 512 | func | Detrend complex array (real+imag, fast) | `x` (1-D complex arr) | x_detrended (complex) | none; `# @njit` commented out | ✅ active (imported by rf.py) |
| `detrend_np` | 551 | func | Detrend 2-channel signal matrix | `v` (2×N complex) | 2×N complex | none | ✅ active (imported by rf.py) |
| `detrend_1d_np_fastwindow` | 558 | func | Vectorized closed-form window detrend | `x` (1-D arr), `window_size=1024` | x_detrended | asserts N % window_size == 0 | ✅ active |
| `detrend_1d_np_fastwindow2` | 616 | func | njit loop window detrend | `x` (1-D arr), `window_size=1024` | x_detrended | asserts divisibility; `@njit` | 🧪 alt impl; verify fan-in |

**Notable (detrend.py):**
- Five+ detrend implementations: full torch (`detrend_1d`/`merge_dynamic_windows`), full numpy
  (`detrend_1d_np`), and two "fast" numpy variants (`detrend_1d_np_fastwindow`,
  `..._fastwindow2`). Only the fast-window numpy path (`detrend_np` → `merge_dynamic_windows_np`
  → `detrend_1d_np_fastwindow`) is imported by `rf.py`; the torch and full-np versions look
  unused.
- `merge_dynamic_windows_np` (512) calls `detrend_1d_np_fastwindow` (NOT the full `detrend_1d_np`),
  so the merge/threshold machinery in `detrend_1d_np` is never reached via that path. The big
  docstring on `merge_dynamic_windows_np` describing `remove_real`/`remove_imag` is stale (those
  params don't exist).
- `@njit` decorators commented out on `merge_dynamic_windows_np` (l.511).

---

## `spf/sdrpluto/benchmark.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `get_sdr` | 14 | func | Open + configure AD9361 for benchmark | `uri,rx_buffer_size=32,rx_buffers=None` | adi.ad9361 | **opens + configures hardware** | ✅ active (bench tool) |
| `benchmark` | 33 | func | Stream samples, write zarr, measure rate | `uri,rx_buffer_size,total_samples=2**24,rx_buffers,filename,compress,chunk_size,store` | dict of metrics | **hardware rx**; **zarr/file write**; spawns write thread; asserts; sleeps | ✅ active (bench tool) |
| `benchmark.write_to_disk` (nested) | 83 | func | Drain queue → zarr | — | — | **zarr write**; thread target | ✅ active (nested) |
| `__main__` block | 118 | script | CLI sweep over buffer/chunk/store/compress | argparse | prints metrics | hardware + file I/O | ✅ active (CLI) |

**Notable (benchmark.py):** Standalone throughput/compression benchmark CLI; not imported by the
pipeline. `total_bits = total_samples * 2 * 128` (l.103) — the `128` looks like an assumed
bits-per-complex-sample constant; worth a sanity check in Pass 3.

---

## `spf/sdrpluto/close_sdr.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `__main__` block | 5 | script | Open SDR, disable TX, destroy buffer | `--ip` | — | **opens hardware; turns TX off; destroys buffer** | ✅ active (utility CLI) |

**Notable (close_sdr.py):** Tiny utility script, no functions/classes — just a `__main__` that
force-quiets a Pluto's TX. No reusable symbols.

---

## `spf/sdrpluto/test_throughput.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `SDRBenchmark` | 8 | class | Pluto RX config + sample-rate benchmark | `uri,rx_config` (dict) | — | **opens adi.ad9361 hardware** | ✅ active (bench tool) |
| `SDRBenchmark.setup_rx_config` | 13 | method | Configure RX (regs, lo, gain, buffers) | — | — | **extensive hardware config**; asserts | ✅ active |
| `SDRBenchmark.benchmark_sample_rate` | 72 | method | Measure effective sample rate | `num_buffers=100` | float (Sps) | **hardware rx**; time | ✅ active |
| `main` | 86 | func | CLI: configure + benchmark sample rate | argparse | — | prints; hardware; constructs SDRBenchmark | ✅ active (CLI) |
| `__main__` block | 177 | script | Entry → main() | — | — | — | ✅ active |

**Notable (test_throughput.py):** Despite the `test_` name this is **not a pytest module** — it's
a standalone CLI benchmark (an `adi.ad9361` driver duplicate of `PPlus.setup_rx_config`). Naming
is misleading. `rx_config` here is a plain dict, not `ReceiverConfig`.

---

## Cluster summary

**Counts (by symbol, including methods & nested funcs; ~180 symbols inventoried):**

| File | Symbols | Notes |
|---|---|---|
| rf.py | 79 | 11 deep-documented (rf.md); heavy duplicate families |
| data_collector.py | 41 | 3 data-format paths (V2 legacy / V4 / V5) |
| sdr.py | 22 | entire module legacy/duplicate, import-time side effects |
| sdr_controller.py | 49 | production driver; global state; some stubs |
| detrend.py | ~14 (+nested) | 5 detrend impls, 2 actually used |
| benchmark.py | 4 | standalone CLI |
| close_sdr.py | 1 (script) | utility CLI |
| test_throughput.py | 5 | misnamed CLI (not a test) |

**Approximate status distribution** (Pass-1, provisional — confirm fan-in in Pass 2):
- ✅ active: ~115
- ⚠️ deprecated / legacy / TODO-remove / stub: ~30 (V2 path, `circular_mean` TODO-remove,
  `DataSnapshot`, stubbed bladeRF methods, no-op `soft_reset_radio`)
- 🔴 broken (suspected, verify Pass 3): ~6
  - `torch_mean_phase_mean` (rf.py:424) — np ops on torch input
  - `windowed_trimmed_circular_mean_and_stddev_fast2` (rf.py:557) — disabled njit + scalar misuse
  - `QAMSource.demod_signal` / `MixedSource.signal` (sdr.py) — calling instances as functions
  - `FakePPlus.get_rssi_and_gain` (sdr_controller.py:530) — undefined `self.dev`
  - `setup_rxtx_and_phase_calibration` (sdr_controller.py:959) — calls missing `setup_rx`
- 🧪 experimental / unused: ~30 (entire sdr.py sim module, unused detrend torch/np-full paths,
  `get_top_3_peaks`, `rotate_dist`, `pi_norm_halfpi`, `fast_percentile`)

**Top 5 things worth a deep dive (Pass 2/3):**
1. **The circular-statistics duplicate explosion in `rf.py`** — 4 parallel implementations
   (np-vectorized / njit-"fast" / torch / scalar) of mean/stddev/windowed stats, several
   near-identical (`fast_median_abs` vs `fast_abs_median`, `fast_percentile` vs `_1d`). Which is
   actually on the hot path? Do they agree numerically (trim semantics differ: percentile vs
   quantile, ≤ vs <)?
2. **`setup_rxtx_and_phase_calibration` calling `pplus_rx.setup_rx()`** (sdr_controller.py:971/982)
   — method does not exist on `PPlus`; determine if dead code or a real latent crash.
3. **Global radio state + thread safety** — `pplus_online`/`run_radios` globals and the
   "not thread safe" `get_pplus` interacting with `ThreadedRX`/`DataCollector`'s reader threads.
4. **`sdr.py` is a legacy duplicate with import-time matplotlib side effects** — confirm nothing
   imports it; if so it's a deletion candidate (and an import-time hazard meanwhile).
5. **Detrend path selection** — `detrend_np`→`merge_dynamic_windows_np`→`detrend_1d_np_fastwindow`
   is the live path; the torch and full-np detrend implementations (with merge/threshold logic)
   appear orphaned, and `merge_dynamic_windows_np`'s docstring is stale. Confirm and prune.

**Cross-cutting theme:** This cluster is a working production SDR pipeline encrusted with multiple
generations of duplicated/abandoned implementations (sim classes, detrend variants, circular-stats
variants, V2 data path) and JIT decorators that are mostly commented out. The biggest Pass-2/3
risks are (a) deciding which duplicate is canonical, and (b) the handful of methods that reference
undefined attributes/methods and would crash if reached.
