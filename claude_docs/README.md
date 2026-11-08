# SPF — Signal Processing Fun

> RF emitter localization: move radios through space, capture raw IQ, beamform it into
> angle-vs-power, learn/estimate the angle-of-arrival, and fuse angles over time into
> emitter tracks.

This directory is the engineering documentation for the SPF codebase. It is organized to
mirror the **data pipeline**, the natural mental model for the whole system.

## The pipeline in one diagram

```
 ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌─────────────┐
 │ 01 DATA         │   │ 02 RF / SDR      │   │ 03 DATASETS  │   │ 04 TRAINING /    │   │ 05 FILTERS  │
 │ COLLECTION      │──▶│ SIGNAL CORE      │──▶│ (zarr v4/v5) │──▶│ INFERENCE (NN)   │──▶│ (EKF / PF)  │
 │ rovers, gantries│   │ rf.py, sdrpluto  │   │ + segmentation│   │ angle-of-arrival │   │ emitter     │
 │ GRBL / MAVLink  │   │ beamforming      │   │ caches        │   │ distributions    │   │ tracks      │
 └─────────────────┘   └──────────────────┘   └──────────────┘   └──────────────────┘   └─────────────┘
        raw IQ                spectra              tensors             per-snapshot θ          θ(t), xy(t)
```

Everything shares one set of **conventions** (angles in radians on `[-π, π]`, positions in
mm, the snapshot/session/window vocabulary, the `d/λ` phase math). Read `00_concepts/` first.

## Map of this documentation

| Section | What it covers | Status of docs |
|---|---|---|
| [`00_concepts/`](00_concepts/) | Glossary, unit/frame conventions, beamforming primer | ✅ written |
| [`01_data_collection/`](01_data_collection/overview.md) | Hardware platforms, motion control, planners, orchestrators, radio targets | 🟡 overview only |
| [`02_rf_signal_processing/`](02_rf_signal_processing/overview.md) | `rf.py` beamforming, SDR hardware control, threaded capture | 🟡 overview only |
| [`03_datasets/`](03_datasets/overview.md) | zarr formats, segmentation, dataset API, generation, migration; **per-version format specs in [`03_datasets/formats/`](03_datasets/formats/README.md)** | 🟡 overview only |
| [`04_training_inference/`](04_training_inference/overview.md) | Training entry, model architectures, config schema, losses, inference cache | 🟡 overview only |
| [`05_filters/`](05_filters/overview.md) | EKF / particle filters, NN-likelihood variants, run harness | 🟡 overview only |

`🟡 overview only` = the section's `overview.md` exists; deeper leaf pages are planned (listed
at the bottom of each overview) and will be filled in a second pass.

## How to read by goal

- **"I want to understand the system end-to-end"** → this page, then each `overview.md` in order 01→05.
- **"I'm collecting data in the field"** → [`01_data_collection/overview.md`](01_data_collection/overview.md).
- **"I'm training a model"** → [`04_training_inference/overview.md`](04_training_inference/overview.md), after [`03_datasets/overview.md`](03_datasets/overview.md).
- **"I'm debugging beamforming / phase"** → [`00_concepts/beamforming_primer.md`](00_concepts/beamforming_primer.md) then [`02_rf_signal_processing/overview.md`](02_rf_signal_processing/overview.md).
- **"What does this weird word mean?"** → [`00_concepts/glossary.md`](00_concepts/glossary.md).

## Status tags used throughout

Components are tagged so the docs distinguish *how it works today* from *what's lying around*:

- ✅ **Active** — current, used in production runs / current configs.
- ⚠️ **Deprecated / legacy** — superseded but still present.
- 🔴 **Broken** — known not to run as-is (verified).
- 🧪 **Experimental / unused** — defined but not wired into the main paths.

Each section overview ends with a **Known issues / tech debt** list. These were surfaced by a
code review and spot-verified against source; line references point at the evidence.

## Source-of-truth pointers (top-level)

- Core RF math: `spf/rf.py`
- SDR hardware: `spf/sdrpluto/sdr_controller.py`
- Capture orchestration: `spf/data_collector.py`, `spf/grbl_radio_collection.py`, `spf/mavlink_radio_collection.py`
- Datasets: `spf/dataset/spf_dataset.py` (`v5spfdataset`, `v5inferencedataset`)
- Training: `spf/scripts/train_single_point.py`; models in `spf/model_training_and_inference/models/`
- Filters: `spf/filters/` (`filters.py` base, `run_filters_on_data.py` harness)
- Global constants/versions: `spf/utils.py` (e.g. `SEGMENTATION_VERSION = 3.7`)
