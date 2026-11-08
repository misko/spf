# Evaluate the recent ROVER development cohorts

Use these frozen recording lists to compare aggregate scores and tune filter parameters through the local inference service. Every recording has existing preprocessing version 3.7. The August 2026 paired model and `empirical_dists/full_20260809_v1.pkl` use compatible artifacts.

The August model's saved `datasets.val_subset_groups` currently blocks its single-recording inference path. EKF and empirical-PF starter sweeps completed; the August neural sweep awaits review of a job-local configuration change clearing those training-report groups. No automatic configuration rewrite is enabled. Model weights, feature settings, and core functions remain unchanged.

## Select a cohort

| Cohort | Windows | Purpose | Antenna spacing coverage |
| --- | ---: | --- | --- |
| `rover_tune16_v1` | 16 | Tune filter settings on training-side development recordings | 35/43/47 mm counts 6/5/5 |
| `rover_devcheck11_v1` | 11 | Check settings selected on the tuning cohort | 35/43/47 mm counts 3/5/3 |
| `rover_all48_v1` | 48 | Report aggregate development results after selecting settings | All recent recordings |

The tuning sample comes from the 37 recent neural-training recordings. Existing `spf.filters.stratified_sample.stratified_sample` selects 16 by `d_lambda`, with seed 0. The sample covers July 31 and August 1, 2, 4, and 5 at 5766 MHz. Geometry stratification represents the rare 47 mm array better than joint geometry/day stratification for this sample size. Selection uses no performance scores.

The development-check cohort contains all 11 recent checkpoint-validation windows from August 7 at 5840 MHz. It changes both day and carrier relative to tuning. Both cohorts already influenced model development. All 48 also contributed to the empirical calibration table. These are development comparisons, not sealed held-out accuracy claims.

The neural train/check split has no shared raw RX or TX captures. Within each cohort, some merged windows share a raw source. The six repeated-RX groups contain disjoint positive timestamp ranges with no repeated timestamps. Keep their legitimate windows, but treat capture and seed spread as descriptive. Do not assume frames or merged windows are independent sessions.

`cohorts.json` records exact basenames, raw-source identities, split membership, selection rules, and source-manifest hashes. The older E-INF sample of 16 includes four checkpoint-validation windows, so it is not this tuning roster.

## Configure the service

Resolve the frozen basenames against local paths:

```bash
.venv/bin/python experiments/inference_eval_v1/build_catalog.py \
  --datasets-root /mnt/md2/rovers/2026q3_v1/merged \
  --precompute-cache /mnt/md2/cache/precompute_cache_3p7 \
  --empirical empirical_dists/full_20260809_v1.pkl \
  --checkpoint /models/august-paired/best.pth \
  --output /work/rover-evaluation-catalog.json
```

The builder checks every required capture, YAML, segmentation pickle, and feature store before publishing the catalog. Use `--base-catalog` to retain previously configured recordings and models. It writes only the requested catalog.

Start the service as described in [the service guide](../../spf/inference_service/README.md), using that catalog. On kalman, preserve the isolated compatible LMDB runtime. Select a named cohort in the browser rather than selecting its members manually.

## Run a sweep

`starter_sweeps.json` contains two-by-two requests for EKF, empirical PF, and neural PF. They provide a small end-to-end check across all 16 tuning recordings. `coarse_sweeps.json` contains wider three-by-three requests with five PF seeds. Inspect the expanded trial count before launching the wider grids.

Submit one named request from either file:

```bash
.venv/bin/python - <<'PY'
import json
import urllib.request
from pathlib import Path

spec = json.loads(Path("experiments/inference_eval_v1/starter_sweeps.json").read_text())["pf_dual_nn"]
request = urllib.request.Request(
    "http://127.0.0.1:8766/api/runs",
    data=json.dumps(spec).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
PY
```

EKF varies measurement noise and process noise through `phi_std` and `noise_std`. PF varies particle count `N` and angular-velocity noise `theta_dot_err`. Other settings remain fixed. Neural PF uses `absolute=false`, so its scores share the craft-relative frame used by EKF and empirical PF. Keep absolute-north experiments separate.

Each cell first averages per-recording MSE equally within each seed, then averages those cohort means equally across seeds. It reports seed spread separately from capture spread. RMSE is the square root of aggregate MSE. Longer recordings do not receive extra weight. A missing, invalid, duplicate, or mismatched-frame trial prevents publication of a successful heatmap.

The service reuses existing filtering, scoring, aggregation, and heatmap functions. Neural predictions are computed once per recording in a sweep and reused across its cells and seeds. Outputs include every trial, complete aggregated cells, the numeric matrix, and the heatmap image. A starter-grid minimum is a result within that grid, not a demonstrated global optimum.

## Check selected settings

Select the configuration using only `rover_tune16_v1`. Freeze its settings before submitting an evaluation on `rover_devcheck11_v1`. For PF, repeat the selected configuration over the agreed seeds when comparing stochastic performance. An initial seed-zero check is a functional check, not a seed-averaged confirmation.

Use `rover_all48_v1` for descriptive reporting after selection. Its overlap with tuning and neural training remains part of the result. Reserve a new independent collection for a final held-out claim.
