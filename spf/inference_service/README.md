# Run historical inference and inspect saved results

The local service runs dual-radio EKF, empirical particle filters, and neural particle filters through existing SPF functions. It saves tracks, scores, diagnostic figures, and provenance. The browser can play back those saved tracks without running inference again.

This first version uses the **current-code** execution profile. It does not claim to reproduce every PDF-era filter behavior. Live ingestion, feature replay, raw-IQ replay, and warm-model workers remain later work. Existing preprocessing and filtering code is unchanged.

## Configure recordings and models

Create a catalog outside the source data and preprocessing directories:

```json
{
  "datasets": {
    "recording_a": {
      "label": "Development recording A",
      "path": "/data/recording_a.zarr",
      "precompute_cache": "/data/precompute_3p5",
      "empirical": "/models/empirical.pkl",
      "segmentation_version": 3.5
    }
  },
  "models": {
    "paired_model": {
      "label": "Paired model",
      "checkpoint": "/models/paired/best.pth"
    }
  }
}
```

Every recording requires its matching `.yaml` and existing preprocessing cache. Model checkpoints require a sibling `config.yml`. The neural path uses the calibration path in that existing model configuration; the catalog's calibration applies to the outer filter dataset. Both are recorded in provenance. Verify those choices before comparing experiments.

The worker copies model and configuration bytes into the job directory and verifies their SHA-256 hashes against the originals. The existing checksum helper can then create its sidecars inside the job directory. Original model inputs remain read-only, and provenance records both source and execution-copy identities.

## Start the service

Use the repository's configured Python environment:

```bash
python -m spf.inference_service \
  --catalog /work/catalog.json \
  --output-dir /work/inference-runs \
  --port 8765
```

Open `http://127.0.0.1:8765`. Select recordings, an algorithm, parameters, and a seed. Submit the run and inspect progress, logs, scores, and plots. The timeline is **results playback**, not a test of the streaming pipeline. Scores use all frames; display rendering may sample points to fit the screen.

The service is local-only. It has no authentication or remote deployment configuration. It validates Host and Origin headers, accepts catalog identifiers rather than arbitrary paths, and serves only published result artifacts.

The current environment on kalman needs a compatible LMDB runtime for existing dataset code. LMDB 2.3.0 rejects repeated opens made by that code. An isolated LMDB 1.6.2 installation was used during verification. To keep the repository environment unchanged, install that version to a separate directory and prepend it to `PYTHONPATH` when starting the service:

```bash
uv pip install --python .venv/bin/python --target /tmp/spf-lmdb-runtime --no-deps lmdb==1.6.2
PYTHONPATH=/tmp/spf-lmdb-runtime .venv/bin/python -m spf.inference_service \
  --catalog /work/catalog.json --output-dir /work/inference-runs
```

Child workers inherit the selected runtime. Dependency versions are recorded. Execution is CPU-only with two Torch threads in this milestone. A compatible GPU environment and warm-model execution require separate parity checks.

## Use the API

```bash
curl http://127.0.0.1:8765/api/catalog
curl -H 'Content-Type: application/json' \
  -d '{"dataset_ids":["recording_a"],"algorithm":"ekf_dual","seed":0,"params":{"phi_std":12,"p":0.1,"noise_std":0.01,"dynamic_R":0}}' \
  http://127.0.0.1:8765/api/runs
```

`GET /api/runs` lists jobs. `GET /api/runs/{id}` returns status and progress. `POST /api/runs/{id}/cancel` with JSON `{}` cancels queued work or terminates the running worker before reporting cancellation. `GET /api/runs/{id}/results` exposes successful results. Artifact URLs use `/api/runs/{id}/artifacts/{filename}`.

Jobs transition from `queued` to `running`, then `succeeded`, `failed`, or `cancelled`. After an unclean stop, unfinished records become `interrupted` and require an explicit new run. The output directory has one owning server. On Linux, workers terminate if the server dies.

## Preserve experimental evidence

Each job has an immutable request, progress, worker log, provenance, and result artifacts. Successful jobs contain native-precision NPZ tracks, JSON playback arrays, and the existing diagnostic figure per recording. Metrics and trajectories come from one filter execution. Unavailable nonfinite display/metric values are represented as JSON `null`; native NPZ arrays preserve their original values.

The worker calls `open_dataset`, `run_filter`, and `summarize_run` from `spf.filters.plot_filter_run`. It uses `save_track` from `spf.filters.dump_tracks` without the latter script's optional float32 casting. Preprocessing regeneration is disabled. Original stores are read through existing read-only LMDB paths, and audit guards reject source writes and deletes.

Provenance records effective parameters, seeds, angular frame, source hashes, revision and local modification paths, dependency versions, and SHA-256 hashes of capture metadata, capture stores, preprocessing stores, calibration, and model artifacts. Input hashing happens in the background worker and can take time. Files changing during execution cause failure. Completed results open without scientific imports or inference.

The multi-recording aggregate gives each recording equal weight and is labeled `mean_dataset_mse`. Compare algorithms only on matched recordings and angular frames. A saved result is not proof that input data was held out from training or calibration.

Workers run sequentially in isolated processes. Each job uses a fresh neural prediction cache, so model startup and inference repeat for a new job. This favors provenance and isolation in the first milestone. Within a completed job, viewing and playback reuse immutable outputs.

## Select a named development cohort

The catalog can define named cohorts alongside datasets and models:

```json
{
  "cohorts": {
    "rover_tuning_v1": {
      "dataset_ids": ["recording_a", "recording_b"],
      "role": "development_tuning",
      "description": "Frozen development recordings for filter tuning",
      "overlap_notes": ["These recordings may overlap neural training or empirical calibration."]
    }
  }
}
```

The supported roles are `development_tuning`, `development_validation`, `development_reporting`, and `historical_reference`. Every member must be a distinct capture path already present in `datasets`. The browser shows membership, role, and overlap notes. Selecting a cohort disables manual membership edits; switching to manual selection allows a separate experiment.

Use `cohort_id` instead of `dataset_ids` for an aggregate evaluation. The server snapshots the complete roster and metadata at submission. Changing a catalog later does not change the saved request. A cohort role is a declaration of intended use, not proof that recordings were held out from model training, checkpoint selection, calibration, or earlier filter tuning. Record those overlaps explicitly. Sessions that share transmitters or acquisition conditions are correlated; capture and seed standard deviations are descriptive, not confidence intervals.

## Run a balanced hyperparameter sweep

Submit one algorithm and two numeric axes per sweep. Keep the remaining settings fixed:

```json
{
  "kind": "sweep",
  "cohort_id": "rover_tuning_v1",
  "algorithm": "pf_dual_nn",
  "model_id": "paired_model",
  "fixed_params": {"theta_err": 0.075, "absolute": false},
  "grid": {"N": [128, 512, 2048], "theta_dot_err": [0.001, 0.01, 0.1]},
  "seeds": [0, 1, 2]
}
```

POST this JSON to `/api/runs`. The browser has a **Hyperparameter sweep** mode with the same fields and a preview of the expanded capture-trial count. For EKF use `noise_std` and `phi_std` axes and `seeds: [0]`; it is deterministic. Fixed parameters cannot also appear in the grid. Requests are limited to 100 cells, 16 seeds, and 10,000 capture trials.

Each cell runs every selected capture under every selected seed. For a seed, the score is the equal-capture mean MSE. The cell score then averages those seed means. Long recordings do not receive extra weight. RMSE is the square root of aggregate MSE, not an average of capture RMSEs. Seed SD measures variation of cohort means across seeds. Capture SD measures variation of per-capture seed means. Neither statistic treats individual frames as independent trials.

The worker reads and hashes inputs once, opens each recording once, and reuses its fresh neural prediction cache across sweep cells and seeds. It invokes the existing `run_filter` and `summarize_run` for each trial. It does not generate hundreds of per-trial diagnostic figures. Balanced report rows use `spf.evaluation.aggregate.aggregate`; the heatmap uses the existing `family_configs` and `fig_family` functions. Only two parameters vary, so there is no hidden selection over additional axes.

Results include every capture trial in `trials.json`, aggregate cells and existing report rows in `sweep_report.json`, numeric axes and matrix values in `heatmap_values.json`, and a heatmap PNG. A missing, duplicate, nonfinite, truncated, or mismatched-frame trial prevents successful publication. Partial trial records remain on disk for diagnosing failed jobs; they are not shown as a valid heatmap.

For direct algorithm comparisons, use craft-relative bearings for EKF, empirical PF, and neural PF. Neural sweeps default to `absolute: false`. Absolute-north runs remain a separate angular frame. Existing single evaluations retain their original defaults.

Choose hyperparameters on the tuning cohort. Evaluate the selected settings on a separate development validation cohort without searching that cohort's heatmap for a new winner. An independent final holdout requires an additional lineage audit and may require newly collected recordings.
