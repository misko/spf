# Realtime/PF-NN inference — consolidated RED/GREEN test matrix (2026-07-12)

Sources: 3 independent review passes + manual verification of top claims. Companion:
_realtime_inference_review.md, KNOWN_ISSUES #55-60. RED = test fails today because of
the issue; GREEN = passes after the paired fix. Mark new tests strict-xfail until fixed.

| ID | Location | Issue | Sev | RED test | GREEN criterion |
|----|----------|-------|-----|----------|-----------------|
| RT1 | data_collector.py:540,545 | heading set as dynamic attr; asdict() drops it → realtime rx_heading_in_pis always 0 (KI#57) | BLOCK | unit: DataSnapshotV4, set heading=90, asdict → expect rx_heading_in_pis=0.5 | field carried; realtime==zarr heading |
| RT2 | wrapper:86,126; pf_nn:41; scripts/test.py:13 | 4 live breakpoint() (KI#55) | BLOCK | static lint (no breakpoint/pdb in spf/) + sys.breakpointhook raise in wrapper test | lint clean; realtime branch runs |
| RT3 | spf_dataset.py:650,664-678 | consumer starts at idx 0 vs max_store_size=3 eviction → ValueError; timeout None → TypeError in collate; render exceptions swallowed (KI#58) | BLOCK | write 10 idxs (store=3): iterate → today ValueError; getitem timeout → wrapper raises TypeError | start-at-latest or clean skip; descriptive error on None; render errors surfaced |
| RT4 | spf_dataset.py:1298-1300 | v4 heading (deg/360)/2 — typo of merge script's *2 (KI#59) | MAJ | fake v4 ds heading=180 → expect rx_heading_in_pis=1.0 (today 0.25) | /180 both paths; ==v4_tx_rx_to_v5 result |
| RT5 | spf_dataset.py:680-683; collector:545 | 8MB signal_matrix pickled per snapshot through mp.Queue + asdict deep-copy + 3×lru_cache retention (KI#60) | MAJ | payload-schema test: queue dict must exclude signal_matrix after feature extraction (or size cap) | features extracted writer-side or shared-mem |
| RT6 | wrapper init / mavlink:277 | no startup check: checkpoint needing empirical/phase/etc. inputs KeyErrors per-sample in bg thread | MAJ | construct realtime wrapper with empirical_input=true ckpt → expect informative ValueError at init (today passes init) | keys_to_get validated vs producible keys at load |
| RT7 | pf_nn:12-27; wrapper:84-96 | absolute paths disagree: cached rotates ALL heads (incl 'single', wrong angle) + hardcoded reshape(-1,65); realtime rotates only 'paired' | MAJ | one-hot dist, known heading, nbins∈{7,65}: argmax shifts exactly; cached==realtime output | single needs mount+heading or not rotated; nbins generic; paths equivalent |
| RT8 | spf_dataset.py:602,647 | __len__ = serving_idx = −1 | MIN | after n writes len(ds)>=0 and meaningful | len = readable count |
| RT9 | wrapper:98,107,140 | 3 stacked lru_cache(4): staleness on idx reuse, self+8MB retention | MIN | rewrite idx 0 with new data → wrapper returns fresh (today stale) | version-keyed or no caching of realtime |
| RT10 | spf_dataset.py:604-645 | reader thread non-daemon; first exception permanently kills ingestion silently | MIN | inject exception in render → expect surfaced error/restart (today silent hang) | error propagates to consumer |
| RT11 | spf_dataset.py:456,506 | mutable default skip_fields mutated across instances | MIN | two constructions w/ defaults → second unaffected (today grows) | default=None pattern |
| RT12 | sp_networks_inference.py:42-53 | realtime generator: double forward per sample, unbounded outputs list, no model.eval() | MIN | call-counter wrapper: exactly 1 forward/sample; memory flat | single forward; eval() enforced |
| RT13 | mavlink:262-292 | --realtime without --checkpoint crashes late (load_config(None)) | MIN | argparse-level: expect early clear error | validated at parse time |
| RT14 | rpi5 yamls; mavlink:33-53 | port→radio maps differ between the two rpi5 configs; no physical cross-check; AGC slow_attack vs fleet fast_attack (KI#56) | MIN | config cross-check test over data_collection/rpi5_inference/*.yaml | consistent maps + AGC policy assert |
| PF1 | filters.py:346 | return_particles: Tensor.copy() AttributeError | MED | trajectory(return_particles=True) → today AttributeError | .clone() |
| PF2 | filters.py:399-405 | estimate(): arithmetic mean of wrapped angles → bearing flips at ±π seam | MED | particles straddling ±π → mu≈π expected (today ≈0) | circular mean (atan2 of phasor sum) |
| PF3 | pf_nn:62-70 | absolute GT: non-circular mean of two radios' bearings; metric still named mse_craft_theta | MED | absolute_thetas straddling ±π, craft_theta=π → mse≈0 expected (today ≈π²) | circular avg + correct metric key |
| PF4 | tests/test_particle_filter.py:111,170 | duplicate test def — first shadowed (F811 not in flake8 select) | MED | add F811(+T100) to flake8 select — fails today | deduped; gate stays |
| PF5 | tests/test_particle_filter.py:165,224,285 | NN-PF tests assert nothing (smoke only) | MED | add mse_craft_theta<0.3 assertions (currently can't reach: RT2) | asserting tests pass post-fixes |
| PF6 | filters.py:326-330 | dt hardcoded 1.0; timestamps ignored — wrong motion model at variable realtime rate | LOW | duplicate-every-snapshot ds → theta_dot must halve (today identical) | dt from system_timestamp |
| PF7 | pf_nn:42 | observation uses radio 0 only ([0]['paired'][0]) — benign now (heads duplicated), silent info-loss for per-radio models | LOW | stub with differing paired heads → weights must reflect both | both radios fused or assert heads equal |
| PF8 | run_filters_on_data.py:530 | eval() on YAML config values (own TODO) | LOW | config with '__import__("os")' string → must not execute | safe parse (ast.literal_eval) |
| TS1 | tests/test_v5inference.py | GOLDEN parity gap: input-only, loose rtol, round-trip (not collector asdict schema), skip_segmentation=False only (production=True untested) | P0 | replay RAW zarr via write_to_idx w/ exact collector schema; compare OUTPUTS vs get_nn_inference_on_ds_and_cache; parametrize skip_segmentation | ≤1e-4 output parity both modes |
| TS2 | — | no trainer-vs-inference checkpoint-load parity | P1 | fixed batch (tests/test_batch.pkl): both load paths → identical outputs, model in eval() | exact match |
| TS3 | — | NaN/no-signal snapshot behavior unpinned realtime | P1 | zeroed signal_matrix → write_to_idx → finite outputs | finite, uniform-ish dist |
| TS4 | — | zero coverage of mavlink --realtime | P1 | --fake-drone --realtime --checkpoint tiny -n 20: exit 0 + inference produced | e2e green on self-hosted CI |

## Verified NON-issues (do not chase)
- Bin/rotation conventions consistent end-to-end (targets, theta_phi_to_bins, to_bin,
  rotate_dist, ±rotation signs) — numerically checked; divergent v5_thetas_to_targets
  is notebook-only.
- No GT leakage into inference (y_rad used for shape only; realtime fills inf/zeros).
- PF weight numerics fine (float64, renorm each step, systematic resample).
- Cached-inference row order matches ds order (shuffle=False forced).
- Core realtime feature parity (windows stats/beamformer/detrend/fp16) holds for the
  deployed windowed_beamformer-only config.

## Suggested fix order (each unlocks the next test tier)
1. RT2 breakpoints + flake8 T100/F811 gate (unblocks all realtime tests)
2. RT1 + RT4 heading fixes (+ their unit reds)  3. RT3 lifecycle  4. RT7/PF2/PF3
rotation+circular-stats  5. TS1 golden  6. rest.
