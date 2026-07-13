# Realtime/Pi inference stack — review findings + test plan (2026-07-12)

Two independent read-only reviews (bug-hunt + test-surface), top claims re-verified
against source. Area: spf_nn_dataset_wrapper.py, single_point_networks_inference.py,
v5inferencedataset, DroneDataCollectorRaw realtime hookup, mavlink_radio_collection
--realtime, particle_dual_radio_nn_filter.

## Verified blockers
- **B1 (KI#57): realtime heading is always 0 — "absolute" bearings are craft-relative.**
  data_collector.py:540 sets `data.heading` as a DYNAMIC attr; DataSnapshotV4 has no
  heading field, so `asdict(data)` (line 545) drops it and rx_heading_in_pis stays 0.0.
  Disk path uses getattr → zarr fine; realtime dict wrong. One-line fix: set
  `data.rx_heading_in_pis = heading/180` before asdict.
- **B2 (KI#55 extended): third live breakpoint()** — particle_dual_radio_nn_filter.py:41
  in PFSingleThetaDualRadioNN.observation (plus the two in spf_nn_dataset_wrapper).
  Also spf/scripts/test.py:13 module-level. Realtime PF cannot run at all.
- **B3 (KI#58): consumer lifecycle broken by design.** v5inferencedataset iterator
  starts at idx 0 but max_store_size=3 evicts → guaranteed ValueError for any consumer
  starting after ~4 records; getitem timeout returns None → TypeError deep in collate;
  render exceptions vanish in the executor (last 4 futures never checked) → consumer
  hangs 30s/idx forever. Likely why the consumption loop is commented out.

## Verified majors
- **M1 (KI#59): v4 heading conversion off 4×** — spf_dataset.py:1298-1300:
  (deg/360)/2 = deg/720; should be deg/180 for "in_pis". IMPACT UNVERIFIED for the
  merged-rover corpus (merges go through v4_tx_rx_to_v5.py, a different path; audit's
  tight per-era Δθ suggests rover labels are NOT 4×-corrupted) — verify which loaders
  hit this branch before fixing blind.
- **M2: ~8MB signal_matrix pickled through mp.Queue per snapshot/radio + retained by
  3 stacked lru_caches** — probable Pi 5 throughput killer (KI#60).
- **M3: model-input compatibility unchecked at startup** — checkpoints needing
  empirical/weighted/phase/signal inputs KeyError per-sample in a background thread
  (skip_segmentation=True path produces neither). Should assert at load.
- **M4: absolute-north inconsistency** — offline cached path rotates EVERY head incl.
  `single` (needs mount+heading, gets heading only); realtime rotates only `paired`.
  `cached_model_inference_to_absolute_north` hardcodes reshape(-1, 65).

## Minors (see agent transcripts)
lru_cache-on-methods staleness/self-retention; __len__ = serving_idx = −1;
non-daemon reader thread dies permanently on first exception; mutable default
skip_fields mutated; single_example_realtime_inference double-forward + unbounded
list + no eval(); --realtime without --checkpoint crashes late; port→USB map has no
physical cross-check (r0/r1 swap risk); two configs disagree on port→radio mapping;
duplicate test_single_theta_dual_radioNN (first shadowed, F811 not in flake8 select).

## GOOD NEWS (verified): core feature parity holds for the deployed config
all_windows_stats + windowed_beamformer are computed by the same functions/args in
both paths (same detrend, same 256 windows, fp16, beamformer nomean == nomean_fast);
gains/rx_lo/spacing/vehicle/device keys match. The parity failures are heading (B1)
and the conditional inputs (M3) only.

## Test plan
P0 (would catch today's bugs; CPU/CI-safe):
 1. no-debugger static guard (breakpoint/pdb lint) + add T100,F811 to flake8 — fails
    on 4 files today, that is the point.
 2. GOLDEN: realtime/offline OUTPUT parity — replay RAW zarr records via write_to_idx
    exactly as data_collector builds them (asdict schema), compare model outputs vs
    get_nn_inference_on_ds_and_cache; parametrize skip_segmentation True/False
    (production=True is currently untested). Fixtures exist (fake_dataset + tiny
    CPU-trained checkpoints in conftest).
 3. nn-wrapper realtime-branch == cached-branch, absolute in {False,True} (currently
    hits breakpoint; also pins to_absolute_north vs cached equivalence).
 4. absolute-north unit: one-hot dist + known heading → argmax shifts by exactly
    heading bins; parametrize nbins {7,65} (65-hardcode fails today).
P1: store eviction/late-reader/timeout semantics; clean error on None from getitem;
    NaN/no-signal snapshot → finite outputs; checkpoint-load parity vs trainer path
    (test_batch.pkl exists); mavlink --fake-drone --realtime e2e (would have caught
    everything at once).
P2: single-forward-per-sample counter; config cross-checks (AGC mode, port maps,
    carrier/spacing consistency both radios); no print() in hot loops.

Existing coverage: test_v5inference.py has input-parity (loose rtol, non-production
settings, round-trip not collector-schema) + a no-assert smoke; the ONLY test of the
wrapper realtime branch (test_single_theta_dual_radioNN_nnwrapper) cannot pass due to
breakpoints. mavlink --realtime has zero coverage. CI runs full pytest on self-hosted.
