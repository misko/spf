# Filter sweep configs

Grids for `spf/filters/run_filters_on_data.py`. All paths are **local** — no
`b2://`, no DynamoDB.

## One config per corpus, and why

`precompute_caches` is a single global `segmentation_version -> path` map, so a
config can only ever describe **one** corpus. The rover-2026 precompute lives on
qnap01; the historical wall/rover corpus lives on md2. Hence `rover2026_*.yaml`
and `historical_*.yaml` rather than one file with both.

## The two stages

| File | Datasets | Configs | Purpose |
|---|---|---|---|
| `*_smoke.yaml` | 1 | ~8 | prove the harness runs against a corpus before spending hours |
| `*_coarse.yaml` | ~16 stratified | ~500 | find the region of good hyperparameters, **5 seeds each** |

The shipped `spf/model_training_and_inference/models/ekf_and_pf_config.yml`
expands to **5,926 jobs per dataset**. At the measured 4.155 cpu-s per filter
timestep and a parallel ceiling of 8.8x on this box, that is ~12 h for the rover
corpus but **22 days** for the frozen val set and **87 days** for train. It was
designed for AWS Batch. Locally the grid has to be cut, which is what these
configs do.

## Seeds

Particle filters are stochastic. Since `spf/filters/resample.py` they are
reproducible from `seed`, but different seeds still give different answers — the
per-dataset spread was measured at 42% (empirical dual-radio) and 106% (NN
dual-radio) of mean MSE. Averaging over a 16-dataset sample cuts that by only
sqrt(16) = 4x, so the coarse stage runs **5 seeds per config**. On a full corpus
(565 datasets, sqrt = 24x) one seed is enough.
