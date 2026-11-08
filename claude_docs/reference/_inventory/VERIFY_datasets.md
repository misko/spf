# Adversarial Verification — `datasets.md` inventory

Read-only refutation pass against source. Default-skeptic.

## (a) Every 🔴 / 🧪 claim → verdict + evidence

| # | Claim (file:line) | Verdict | Evidence |
|---|---|---|---|
| 1 | `compare_datasets.py` has `return` at module scope @37,@43 → SyntaxError | **REFUTED** | @37 `return_code = identical_datasets(...)`, @43 `return_code = 0` — these are **variable assignments**, not `return` statements. No SyntaxError exists; the module imports and runs fine. The inventory misread `return_code = ...` as `return`. **Major false claim.** |
| 2 | `segmentation.py::beamform_signal_cpu` is a stub (~1025) | **CONFIRMED** | @1025 `def beamform_signal_cpu(...)`, body is docstring then @1040 `# ... existing code ...`. Returns None. No callers (grep: only the definition). |
| 3 | `wall_array_v2_idxs::v2_rssi_idxs` both indices = "rssi0" (~39) | **CONFIRMED** | @40-42 `return [v2_column_names().index("rssi0"), v2_column_names().index("rssi0")]`. Second should be "rssi1". Genuine bug. |
| 4 | Two unconditional `breakpoint()` in spf_nn_dataset_wrapper.py (~86,~126) | **CONFIRMED** | @86 `breakpoint()` inside `to_absolute_north` for-loop; @126 `breakpoint()` in `get_and_annotate_entry_at_idx` realtime branch (after sample.update, before `if self.absolute`). Both unconditional. |
| 5 | `yarr_rechunk.py` calls `new_yarr_dataset` with missing args → TypeError | **CONFIRMED** | yarr_rechunk @18-23 passes only `all_windows_stats_shape`, `windowed_beamformer_shape`, `n_receivers`. zarr_utils `new_yarr_dataset` @30-40 requires positional `weighted_beamformer_shape`, `downsampled_segmentation_mask_shape`, `mean_phase_shape` (no defaults). Missing 3 → TypeError. |
| 6 | `open_partial_ds.py` calls `.add()` on a list (CLI path @14/@69) | **CONFIRMED** | @69 `skip_fields = []` (list) → passed to function → @14 `skip_fields.add("signal_matrix")`. Lists lack `.add` → AttributeError on CLI invocation. (Note: only the CLI path is broken; calling the function with default `set()` works.) |
| 7 | `v5inferencedataset` mutable-default `skip_fields=[]` mutated via `+=` (~506) | **CONFIRMED** | @456-458 `skip_fields: List[str] = []` default; @503 `self.skip_fields = skip_fields` (no copy); @506 `self.skip_fields += segmentation_based_keys` in-place mutation. @506 is inside v5inferencedataset (class 448-816). |
| 8 | `get_segmentation` deletes `.pkl` on UnpicklingError then recurses (~1755) | **CONFIRMED** | @1753 `except pickle.UnpicklingError:`, @1755 `os.remove(results_fn)`, @1756 `return self.get_segmentation(...)`. Destructive recovery, as described. |
| 9 | `spf_generate.py::generate_session` orbit branch `sys.exit(1)` (@221) | **CONFIRMED** | @221 `if args.detector_trajectory == "orbit":` → @222 print → @223 `sys.exit(1)` (inventory cited @221 for the branch; exit is @223). |
| 10 | `benchmark_dataset_linear.py` `sys.exit(0)` @47 makes tail dead | **CONFIRMED** | @47 `sys.exit(0)` immediately after the `for x in idxs: ds[x]` loop and before `random.shuffle`/DataLoader code. Tail unreachable. |
| 11 | 🧪 `drop_noise_windows` (@784) unused | **CONFIRMED** | grep whole repo: only the definition at segmentation.py:784. No callers. |
| 12 | 🧪 `phi_to_theta` (fake_dataset @223) unused, call commented @243 | **CONFIRMED** | Only reference is commented `# _thetas = phi_to_theta(...)` @243. |
| 13 | 🧪 `nanstd` (precompute_3p3 @39) dead, only commented use @33 | **CONFIRMED** | Only reference is commented `# diff_std = nanstd(...)` @33. |
| 14 | 🧪 `get_min_max_of_times` (v4_tx_rx_to_v5 @205) unused | **CONFIRMED** | grep: only the definition at @205. No callers. |

## (b) Line-number hit rate

Spot-checked ~46 line citations across spf_dataset.py (module funcs + every v5inferencedataset/v5spfdataset/SessionsDatasetSimulated method), segmentation.py, v4_data.py, plus the broken/dead targets. **All matched exactly: 46/46 = 100%.** Notable secondary citations also confirmed: mp_segment_zarr 2-receiver assert @189 (`assert len(z["receivers"]) == 2`); cupy bare try/except guard @9-12.

## (c) Missed / misclassified

- **Completeness: excellent.** Every method of `v5spfdataset` (24 methods) and `v5inferencedataset` (12 methods) is present at the correct line; no missed methods. `ZarrWrapper` and `SessionsDatasetSimulated` methods also complete.
- **Side-effect classifications: accurate.** Destructive zarr writers (`new_yarr_dataset`, `zarr_new_dataset`, `zarr_remove_if_exists`, `truncate_zarr`, `zarr_fix`) correctly flagged DESTRUCTIVE. `get_segmentation`'s `os.remove` recovery and `ls_zarr`'s `.ls.json` sidecar write correctly noted. No "claimed pure but writes/deletes" mismatches found.
- **One classification nuance (claim #1):** `compare_datasets.py` is tagged 🔴 Broken (SyntaxError) but is NOT broken — it is a working CLI script. Should be ✅ Active. This is the single substantive error and it inflates the 🔴 count by one.
- Minor: inventory attributes the `+=` to "shared default list / caller's list" — correct, but worth noting the v5spfdataset default `skip_fields=[]` (@835) is genuinely NOT mutated in-place (only assigned @887), exactly as the inventory states.

## (d) Overall accuracy grade

**A−**

Of 14 🔴/🧪 claims, **13 CONFIRMED, 1 REFUTED** (compare_datasets SyntaxError — a real misread, since `return_code =` is an assignment). Line-number hit rate 100% on a large sample. Completeness and side-effect classification are thorough and correct. The single false-broken claim is the only material defect; it would have been caught by attempting an import (the inventory's own guidance noted a module-scope `return` would prevent import). Everything else holds up under adversarial scrutiny.
