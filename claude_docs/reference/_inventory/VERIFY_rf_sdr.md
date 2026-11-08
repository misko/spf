# Adversarial Verification — rf_sdr.md inventory

Verifier role: refute Pass-1 inventory against source. Read-only. Source as of 2026-06-21.

## (a) Every 🔴 broken / 🧪 dead claim → verdict + evidence

| # | Claim (inventory) | Cited loc | Verdict | Evidence |
|---|---|---|---|---|
| 1 | `setup_rxtx_and_phase_calibration` calls nonexistent `pplus_rx.setup_rx()` | sdr_ctrl 959/971/982 | **CONFIRMED** | 971 & 982 call `pplus_rx.setup_rx()`. `grep "def setup_rx"` shows only `setup_rx_config` (352/579/703) as a method; `setup_rx` (885) is a **module-level function**, not a PPlus method → `AttributeError`. NOT dead: real caller at 1256 (rxcal CLI path) → latent crash, not unreachable. Inventory said "may be dead/unused" — refined: it is REACHABLE. |
| 2 | `torch_mean_phase_mean` misuses np on torch input | rf.py 424 | **CONFIRMED (+ also dead)** | 426–428 use `np.sin/np.cos/np.arctan2` then `torch_pi_norm(cm)`. Broken for tensors. Stronger: `grep` shows **zero callers** anywhere → dead AND broken. |
| 3 | `windowed_..._fast2` broken (njit commented; scalar misuse) | rf.py 557 | **CONFIRMED (+ effectively dead)** | 556 `# @njit(parallel=True)` commented; 568 uses `prange` (runs serial / NameError risk if numba absent). 612 & 625 call `circular_diff_to_mean_single_fast(pd[k], mean_angle)` with scalar `pd[k]`; that fn (234) does `angles.shape[0]` → crashes on scalar. Only call site (segmentation.py:661) is **commented out**; import at :29 is unused → effectively dead. |
| 4 | `FakePPlus.get_rssi_and_gain` references undefined `self.dev` | sdr_ctrl 530 | **CONFIRMED** | 543–545 read `self.dev.lna_gain/.vga1/.vga2`. FakePPlus `__init__` (503) sets `self.sdr`, never `self.dev` → `AttributeError` if called. (Real PPlus uses `self.sdr._ctrl`; this fake path is broken.) |
| 5 | `QAMSource.demod_signal` calls SinSource instances as callables | sdr.py 70 | **CONFIRMED** | 72: `self.lo_in_phase(demod_times)` — Source/SinSource define no `__call__` → `TypeError`. |
| 5b | `MixedSource.signal` calls source instances as callables | sdr.py 48 | **CONFIRMED** | 49: `self.source_a(sampling_times) * self.source_b(...)` — instances not callable → `TypeError`. |
| 5c | (implied) `QAMSource.signal` also suspect | sdr.py 60 | **REFUTED** | 60–68 correctly call `.signal(...)`. Inventory did NOT flag this one — correctly. |
| 6 | sdr.py creates matplotlib figures at import time | sdr.py ~212–230 | **CONFIRMED** | 212 `ula_d=ULADetector(...)`, 213 `plt.subplots`, 214 `plot_space`; 228–230 repeat for `uca_d`. `import matplotlib.pyplot as plt` at line 1. Import-time side effects real. |
| 7 | `fast_median_abs` vs `fast_abs_median` duplication | rf.py 472 / 528 | **CONFIRMED** | Bodies at 472–486 and 528–544 are near-identical (partition-of-abs median). Both LIVE: `fast_median_abs` used at 495/692 + tests; `fast_abs_median` used at 643/644. Duplication real; neither is dead. |
| 8 | `pi_norm_halfpi` unused | rf.py 54 | **CONFIRMED dead** | grep: zero callers repo-wide. |
| 9 | `fast_percentile` superseded/unused | rf.py 82 | **PARTIALLY-CORRECT** | No production callers, but USED by tests (test_rf.py:96, test_beamformer.py:18). Not purely dead. |
| 10 | `get_top_3_peaks` experimental/unused | rf.py 1454 | **CONFIRMED dead** | Only reference is commented-out line 1488. |
| 11 | `get_peaks_for_2rx` experimental, verify callers | rf.py 1436 | **PARTIALLY-CORRECT** | Used by test_dataset.py:92 only (no production). Test-only, not fully dead. |
| 12 | `rotate_dist` experimental/unused | rf.py 1493 | **REFUTED** | Heavy production use: particle_dual_radio_nn_filter, single_point_networks (1025), train_single_point (769/809/841), spf_nn_dataset_wrapper (89). NOT experimental. Misclassification. |
| 13 | `ProcessPoolExecutorWithQueueSizeLimit` unused | data_coll 41 | **CONFIRMED dead** | grep: no callers outside its own class def. |
| 14 | torch/full-np detrend paths unused | detrend 10/205/301/616 | **CONFIRMED dead** | `merge_dynamic_windows` (torch) only def, no external caller; `detrend_1d` only called internally by `merge_dynamic_windows` (orphan chain); `detrend_1d_np` referenced only in a docstring; `detrend_1d_np_fastwindow2` zero callers. Live path is `detrend_np`→`merge_dynamic_windows_np`→`detrend_1d_np_fastwindow`. |
| 15 | DroneDataCollectorRaw sets `data.heading` but V4 stores `rx_heading_in_pis` | data_coll 540 | **CONFIRMED** | 540 sets `data.heading`. Base `DataSnapshotRaw` (69) field is `rx_heading_in_pis` (79), NOT `heading`. zarr write (551) iterates v4rx keys (= `rx_heading_in_pis`), so `data.heading` is a stray attr never persisted. Real latent bug. (Minor: inventory's DataSnapshotRaw row text says field "heading" — imprecise, real name is `rx_heading_in_pis`.) |

**Broken-claim score (the 🔴 set, claims 1–6 + the get_rssi/setup_rx specials): 6 of 6 CONFIRMED.** No 🔴 refuted. Dead/unused 🧪 claims: mostly confirmed; one clear REFUTE (`rotate_dist`) and two downgraded to test-only (`fast_percentile`, `get_peaks_for_2rx`).

## (b) Line-number spot-check hit rate

20 ordinary rows sampled across all 7 files (rf.py ×8, data_collector ×5, sdr_controller ×3, detrend ×2, benchmark ×1, test_throughput ×1). Every cited line landed exactly on the named `def`/`class`. **Hit rate: 20/20 = 100%.** Line-number accuracy is excellent. The few extra precise loc claims checked in the 🔴 table (424, 471/528, 543, 556, 612/625, 971/982, 212–230, 540) also all matched.

## (c) Missed symbols / misclassifications

- **Misclassification (severity: medium):** `rotate_dist` marked 🧪 experimental — it is a core production helper with ~6 import sites incl. training and particle filters. Should be ✅ active.
- **Under-stated severity:** `setup_rxtx_and_phase_calibration` and `torch_mean_phase_mean` were hedged as "suspect / verify Pass 2/3". Both are firmly broken; #1 is reachable (caller at 1256), #2 is dead. The inventory's caution was warranted but the verdicts are now definitive.
- **Minor field imprecision:** `DataSnapshotRaw` row lists field "heading"; actual field is `rx_heading_in_pis`. This is the very mismatch the Notable section flags, so internally the doc is right; only the table cell is loose.
- **No missed top-level symbols found** in rf.py: `grep "^def |^class "` = 69 top-level defs/classes, consistent with the inventory's enumeration (79 incl. methods). Commented-out `beamformer_numba*` block (1054–1099, inside a `"""..."""` string after `if numba:`) correctly noted as dead.
- **No wrong "pure" classifications found:** side-effect tags spot-checked (e.g. `Detector.add_source` mutates, `get_signal_matrix` randomness, zarr writes) are accurate.

## (d) Overall accuracy grade: **A−**

Rationale: 100% line-number accuracy; 6/6 broken claims confirmed against source with no false alarms; dead-code claims overwhelmingly correct. Deductions only for one genuine misclassification (`rotate_dist` is production, not experimental) and a couple of "dead" labels that are actually test-only rather than fully dead. The inventory is trustworthy and its highest-stakes 🔴 flags are all real bugs.
