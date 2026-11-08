# Adversarial Verification — Filters & Inference/Cache (NEW claims)

Mode: READ-ONLY. Default skepticism. Each claim assessed for: real? trigger? reachable? + verdict + final severity + confidence.

Sources inspected:
- `spf/filters/ekf_dualradioXY_filter.py`
- `tests/test_particle_filter.py`, `tests/test_ekf.py`
- `spf/filters/ekf_dualradio_filter.py`, `spf/filters/filters.py`
- `spf/model_training_and_inference/models/single_point_networks_inference.py`
- `spf/utils.py`
- `spf/scripts/create_empirical_p_dist.py`
- `spf/dataset/spf_nn_dataset_wrapper.py`, `spf/dataset/spf_dataset.py`
- `spf/model_training_and_inference/models/ekf_and_pf_config.yml`
- `spf/dataset/fake_dataset.py`

---

## A) CONCRETE FILTERS

### N1 — XY-EKF `metrics` compares tx_x (mm) to θ (radians) — VERDICT: CONFIRMED (with a correction to the claim)

The metric bug is real and is exactly as described, but the claim's file attribution for the test is **wrong**: the loose-bound test that hides it lives in `tests/test_ekf.py`, NOT `tests/test_particle_filter.py`. (The `test_XY_dual_radio` in test_particle_filter.py exercises the *particle* filter `run_PF_xy_dual_radio`, a different code path, with bound `< 0.25`.)

Evidence — `ekf_dualradioXY_filter.py:158-168`:
```
158    def metrics(self, trajectory):
159        pred_theta = torch.tensor(np.hstack([x["mu"][0] for x in trajectory]))
160        pred_xy = torch.tensor(np.hstack([x["mu"][[0, 1]] for x in trajectory]).T)
161        return {
162            "mse_craft_theta": (
163                torch_pi_norm_pi(self.ds.craft_ground_truth_thetas - pred_theta) ** 2
```

- **Index / units confirmed.** State is documented at line 45: `# state x = [ tx_x, tx_y, dtx_x/dt, dtx_y/dt, rx_x, rx_y]`. `setup()` (line 173) initializes `x[0] = self.ds[0][0]["tx_pos_x_mm"].item()`. So `x["mu"][0]` is **tx_x in millimetres**, an absolute position, NOT an angle.
- `self.ds.craft_ground_truth_thetas` is `torch_pi_norm(...)` (radians in [-π, π]) — `spf_dataset.py:1545-1571`. So line 163 subtracts radians from millimetres, then `torch_pi_norm_pi` wraps the nonsensical mm value into [-π, π]. Pure garbage metric.
- **The correct θ is computed and ignored.** Line 232 computes `target_theta = pi_norm(np.arctan2(rel_x, rel_y))` and line 237 stores it as `current_instance["craft_theta"]`. `metrics` never reads `craft_theta`.
- **Contrast proves intent.** The sibling dual-radio EKF (`ekf_dualradio_filter.py:128-131`) delegates to `dual_radio_mse_theta_metrics` (`filters.py:278-289`), which correctly uses `np.hstack([x["craft_theta"] for x in trajectory])`. The XY filter diverges from this correct pattern.
- **Loose bound that hides it — `tests/test_ekf.py:81-84`:**
```
81        kf = SPFPairedXYKalmanFilter(ds=ds, phi_std=5.0, p=0.1, dynamic_R=True)
82        pairedXY_trajectory = kf.trajectory(debug=True, dt=1.0, noise_std=10)
83        pairedXY_metrics = kf.metrics(pairedXY_trajectory)
84        assert pairedXY_metrics["mse_craft_theta"] < 5  # we arent very good here
```
  The `< 5` bound (with the self-aware comment "we arent very good here") passes trivially because the metric is always wrapped into [-π,π] (max possible squared error ≈ π² ≈ 9.87, and the wrapped garbage typically lands < 5). The test cannot detect that the number is meaningless.
- **Config-dead confirmed.** In `ekf_and_pf_config.yml` the `run_EKF_xy_dual_radio` block is fully commented out (lines 83-92), as is `run_PF_xy_dual_radio` (77-79). So this filter never runs in the production sweep — the bug is exercised only by the loose unit test.

Real? YES. Trigger? Any call to `SPFPairedXYKalmanFilter.metrics` yields a meaningless `mse_craft_theta`. Reachable? Only via `tests/test_ekf.py::test_pairedXY_radio_ekf` (config path is dead). 

**Final severity: LOW (silently-wrong).** The metric is definitively broken and self-misleading, but the filter is config-dead and the only consumer is a loose, self-deprecating unit test. Impact is confined to a non-production diagnostic. Note correction: claim cited the wrong test file (test_particle_filter.py); actual loose-bound test is test_ekf.py:84. Confidence: HIGH.

### N3 — duplicate `def test_single_theta_dual_radioNN` (shadowing) — VERDICT: CONFIRMED

Both defs exist in `tests/test_particle_filter.py`:
- First: line 111 `def test_single_theta_dual_radioNN(` (body lines 111-166)
- Second: line 170 `def test_single_theta_dual_radioNN(` (body lines 170-225)

The two bodies are essentially identical (same args N=4096, theta_err=0.1, theta_dot_err=0.001). Per Python module semantics the second binding overwrites the first at class/module load, so pytest collects and runs only the second. The first is dead. (A distinct third test, `test_single_theta_dual_radioNN_nnwrapper` at line 228, is unaffected.)

Real? YES. Trigger? Module load — second def shadows first. Reachable? Always.

**Final severity: LOW.** No correctness impact (bodies are duplicates), but it is dead test code / lost coverage intent and a maintenance hazard. Confidence: HIGH.

---

## B) INFERENCE / CACHE

### Claim 3 — inference cache key OMITS `v4` → cross-`v4` collisions — VERDICT: CONFIRMED  ***[HIGH, silently-wrong]***

**This is the headline finding.**

Cache key — `single_point_networks_inference.py:227-228`:
```
227    ds_basename = os.path.basename(ds_fn)
228    inference_cache_fn = f"{inference_cache}/{ds_basename}/{segmentation_version:0.3f}/{checkpoint_checksum}/{config_checksum}.npz"
```
The key components are: dataset basename, segmentation_version, checkpoint md5, config md5. **`v4` is absent** from the key. Meanwhile `v4` is an explicit parameter of `get_nn_inference_on_ds_and_cache` (line 201) and flows through to `run_nn_inference_on_ds(..., v4=v4)` (line 247) → `convert_datasets_config_to_inference(..., v4=v4)` (line 280) → `datasets_config["v4"]` (line 157) → `load_dataloaders`.

- **v4 actually changes outputs.** `datasets_config["v4"]` reaches the dataset (`spf_dataset.py:475/495` `self.v4 = v4`), and `v4=True` triggers `v4_to_v5()` (`spf_dataset.py:1289-1303`), which **rewrites receiver data**: recomputes `rx_heading_in_pis` from raw `heading/360/2` and fills missing v5 keys via `ZarrWrapper`. These feed segmentation and model inputs, so inference outputs differ between `v4=True` and `v4=False` on the same zarr. This is not a cosmetic flag.
- **Reachable.** `spf_nn_dataset_wrapper.py:37-54` passes `v4=v4` (defaulting to `self.ds.v4`, line 38) into `get_nn_inference_on_ds_and_cache`. So a v4 dataset wrapped for NN inference reaches the cache with `v4=True`, and a v5 dataset with the same basename + same checkpoint/config/seg-version reaches it with `v4=False`. **They collide on the same `.npz`** and the second run silently reads the first's stale results.
- Aggravating: the key uses only `os.path.basename(ds_fn)`, so two datasets in different directories sharing a basename also collide — but that is secondary to the v4 omission.

Real? YES. Trigger? Two inference runs differing only in `v4` (or v4-vs-v5 datasets with same basename/checkpoint/config/seg) → cache hit returns the wrong precomputed tensors. Reachable? YES, via the NN wrapper.

**Final severity: HIGH (silently-wrong).** A correctness-affecting input is omitted from a content-addressed cache key, producing silently-wrong cached inference with no error. Confidence: HIGH.

### Claim 4 — `get_md5_of_file` `.md5` sidecar never invalidated → stale inference cache on in-place edit — VERDICT: CONFIRMED

`utils.py:87-103`:
```
87  @functools.cache
88  def get_md5_of_file(fn, cache_md5=True):
89      if os.path.exists(fn + ".md5"):
90          try:
91              return open(fn + ".md5", "r").readlines()[0].strip()
92          except:
93              pass
...
100      if cache_md5:
101          with open(fn + ".md5", "w") as f:
102              f.write(md5)
```
- The sidecar is **read whenever it exists** (line 89) with **zero mtime / size / content comparison** against `fn`. A repo-wide search found only read (89/91) and write (101) of `*.md5` — no `os.remove`, no `getmtime`, no invalidation anywhere.
- Two-layer staleness: (a) the on-disk `.md5` sidecar persists across processes and is trusted blindly; (b) `@functools.cache` additionally pins the first-seen md5 for the process lifetime.
- Consequence: an **in-place edit** of a config or checkpoint (same path, contents changed) leaves the old `.md5` in place → `get_md5_of_file` returns the stale digest → the inference cache key (line 222-223 `config_checksum`/`checkpoint_checksum`) is unchanged → a stale `.npz` is reused for the edited file.

Real? YES. Trigger? Edit a `config.yml`/`best.pth` in place without renaming, after a prior run created its `.md5`. Reachable? YES — `get_md5_of_file` is the cache-key source for both checkpoint and config (lines 222-223).

**Final severity: MEDIUM (silently-wrong).** Correctness hazard, but requires an in-place mutation of an otherwise content-addressed artifact (less common than swapping files by path). Confidence: HIGH.

### Claim 5 — `apply_symmetry_rules_to_heatmap` produces a one-short θ axis for EVEN bins; CLI default 50 is even — VERDICT: CONFIRMED

`create_empirical_p_dist.py:100-109`:
```
100  def apply_symmetry_rules_to_heatmap(h):
101      bins = h.shape[0]
105      half = h[: math.ceil(bins / 2)] + np.flip(h[math.floor(bins // 2) :])
107      half = half + np.flip(half, axis=0)
108      full = np.vstack([half[:-1], np.flip(half)])
109      return full
```
- **Empirically verified** by executing the function on random heatmaps:
  - bins=7 (odd)  → output θ axis = 7  (correct)
  - bins=65 (odd) → output θ axis = 65 (correct)
  - bins=50 (even) → output θ axis = **49** (one short)
  - bins=64 (even) → output θ axis = **63** (one short)
- Algebra: even bins → `ceil(bins/2)=floor(bins//2)=bins/2`, so `half` has `bins/2` rows → `full = (bins/2 − 1) + (bins/2) = bins − 1`. Odd bins → `half` has `(bins+1)/2` rows → `full = ((bins+1)/2 − 1) + (bins+1)/2 = bins`. Off-by-one is exclusive to even bins.
- **CLI default is even.** `--nbins` default `= 50` (`create_empirical_p_dist.py:128-131`). A manual run at the default produces a 49-row θ axis — malformed (non-square, axis ≠ requested resolution).
- **Mitigation confirmed.** Checked-in/active artifacts use ODD bins: the only on-disk empirical-dist artifacts use `nbins65`, and the test harness `fake_dataset.py:139-141` sets `--nbins {nthetas}` = 65 (odd). So tests and shipped dists dodge the bug.
- Downstream this is silently-wrong rather than a crash: consumers derive `nbins` from `full_p.shape[0]` (`filters.py:370` `theta_phi_to_bins(thetas, nbins=full_p.shape[0])`), so a 49-row map just produces a misaligned θ binning instead of erroring.

Real? YES. Trigger? Run `create_empirical_p_dist` with any even `--nbins` (the default 50 included). Reachable? YES via CLI default, but NOT via the test path or checked-in artifacts (odd bins).

**Final severity: MEDIUM (silently-wrong), mitigated in practice.** Genuine off-by-one reachable through the default CLI value, but all in-repo usage uses odd bins so production artifacts are unaffected. Confidence: HIGH.

---

## Summary table

| # | Claim | Verdict | Final severity | Confidence |
|---|-------|---------|----------------|------------|
| N1 | XY-EKF `metrics` uses `mu[0]` (tx_x mm) vs θ radians; ignores `craft_theta`; loose `<5` test | CONFIRMED (claim mis-cited test file: it's `test_ekf.py:84`, not `test_particle_filter.py`) | LOW (silently-wrong; config-dead) | HIGH |
| N3 | Two `def test_single_theta_dual_radioNN` (lines 111 & 170); second shadows first | CONFIRMED | LOW | HIGH |
| 3 | Inference cache key OMITS `v4`; v4 affects outputs; reachable via wrapper | **CONFIRMED — HIGH, silently-wrong** | **HIGH** | HIGH |
| 4 | `.md5` sidecar + functools.cache never invalidated by mtime → stale cache on in-place edit | CONFIRMED | MEDIUM (silently-wrong) | HIGH |
| 5 | `apply_symmetry_rules_to_heatmap` off-by-one for EVEN bins; default `--nbins=50` even; artifacts use odd (65/7) | CONFIRMED | MEDIUM (silently-wrong, mitigated) | HIGH |

## Silently-wrong flags
- **Claim 3 (HIGH):** wrong cached inference returned with no error when only `v4` differs.
- Claim 4: stale cached inference on in-place config/checkpoint edits.
- Claim 5: malformed (one-short) θ axis silently misbinned for even `nbins`.
- N1: meaningless `mse_craft_theta` reported as if valid.

## Corrections to the claims as stated
- N1 attributed the loose-bound hiding test to `tests/test_particle_filter.py`; it is actually `tests/test_ekf.py:84` (`assert pairedXY_metrics["mse_craft_theta"] < 5`). The metric/units/index bug and the config-dead point are otherwise fully confirmed.
