# 03 · Datasets & Data Formats — overview

How raw captures become model-ready tensors: the versioned **zarr** storage format, the
**segmentation** cache, and the `v5spfdataset` API that training/inference/filters all consume.

## Storage layout

A dataset is a base `prefix` with two stores:

- **`{prefix}.zarr`** — raw capture (LMDB-backed zarr). Holds the IQ + metadata.
- **`{prefix}_segmentation_nthetas{N}.yarr`** — derived segmentation cache (per-window phase
  stats + windowed beamformer). Versioned by `SEGMENTATION_VERSION` (currently **3.7**,
  `spf/utils.py:14`).

### `.zarr` schema (per receiver `r0`, `r1`)

```
receivers/r{i}/
  signal_matrix        (snapshots, 2, buffer_size)  complex64   # raw IQ, 2 elements
  system_timestamp     (snapshots,)                 float64
  tx_pos_x_mm/y_mm     (snapshots,)                 float64     # v5
  rx_pos_x_mm/y_mm     (snapshots,)                 float64     # v5
  rx_theta_in_pis      (snapshots,)                 float64
  rx_heading_in_pis    (snapshots,)                 float64     # v5 only
  rx_spacing           (snapshots,)                 float64     # meters
  rx_lo, rx_bandwidth  (snapshots,)                 float64     # Hz
  avg_phase_diff       (snapshots, 2)               float64
  rssis, gains         (snapshots, 2)               float64
config                  # the run's YAML, as a string
```

### `.yarr` segmentation schema (per receiver)

```
r{i}/
  all_windows_stats            (snapshots, 3, n_windows)   float16  # mean φ, stddev φ, |signal|
  windowed_beamformer          (snapshots, n_windows, nthetas) float16
  weighted_beamformer          (snapshots, nthetas)        float32  # session-level
  weighted_windows_stats       (snapshots, 3)              float32
  downsampled_segmentation_mask(snapshots, n_windows)      bool     # signal vs noise
  mean_phase                   (snapshots,)                float32
version                          # SEGMENTATION_VERSION (3.7)
```

## Format versions

**Per-version detail lives in [`formats/`](formats/README.md)** — one document per data version,
each with an exhaustive field table, example configs, and traps. The rules those documents follow
are in [`formats/contracts.md`](formats/contracts.md). Summary:

| Version | Positioning | Notes | Status |
|---|---|---|---|
| [v1](formats/v1_data_format.md) | emitter XY only | memmap, beamformer-only, no rx position ⇒ unlabelable | ⚠️ abandoned |
| [v2](formats/v2_data_format.md) | motor XY mm | memmap, beamformer-only, no raw IQ | ⚠️ legacy |
| v3 | — | **never existed**; the `3.x` versions in this project are *segmentation* versions | — |
| [v4](formats/v4_data_format.md) | GPS (`gps_lat`, `gps_long`, `gps_timestamp`) | Rover format; raw signal matrix; most of the rover corpus | 🟡 legacy-readable |
| [v5](formats/v5_data_format.md) | motor XY mm + `rx_heading_in_pis` | Wall-array format; the training format | ✅ active |
| [v6](formats/v6_data_format.md) | GPS | direct-USB proto v1: gain *indices*, no RSSI, NaN level columns | ⚠️ transitional |
| [v7](formats/v7_data_format.md) | GPS | direct-USB proto v2: start/end gain **and** RSSI in dB; current rover production | ✅ active |

`v5spfdataset` can read v4 (and the v6/v7 supersets) by wrapping them on the fly (`v4_to_v5`).
Schema definitions: `spf/dataset/v{4,5,6,7}_data.py`; the transport/version contract is
`spf/capture_schema.py`. The segmentation sidecar has its own document:
[`formats/precompute_cache_format.md`](formats/precompute_cache_format.md).

## Segmentation: what and why

Segmentation (`spf/dataset/segmentation.py`, entry `mp_segment_zarr()`) decides which **windows**
of a snapshot contain real signal vs noise, and precomputes the per-window phase stats +
windowed beamformer so training doesn't redo it every epoch. Pipeline per window: detrend →
beamform → trimmed **circular** mean/stddev of φ → threshold into a signal/noise mask → merge →
drop tiny segments. Defaults: window 2048, 20% trim, stddev threshold 0.5, min segment 3000.
The result is cached in the `.yarr`; `segment_if_not_exist=True` generates it lazily.

## The dataset API

### `v5spfdataset` (`spf/dataset/spf_dataset.py:818`) — training/eval

Reads `{prefix}.zarr` + `.yarr` + `.yaml`, computes ground-truth angles from positions, and
returns torch tensors. Key constructor args: `prefix`, `nthetas`, `precompute_cache`,
`snapshots_per_session`, `snapshots_stride`, `tiled_sessions`, `paired`, `skip_fields`,
`empirical_data_fn`, `segment_if_not_exist`, `gpu`, `ignore_qc`. `len()` = number of sessions
(×2 if `paired`).

`__getitem__` returns a dict (or list of 2 dicts if `paired`) keyed by, among others:

- **Targets**: `y_rad` (array-relative θ), `craft_y_rad`, `absolute_theta`, `y_phi`, `y_rad_binned`.
- **Features**: `weighted_beamformer`, `windowed_beamformer`, `all_windows_stats`,
  `weighted_windows_stats`, `downsampled_segmentation_mask`, `empirical`.
- **Metadata**: `rx_pos_xy`/`tx_pos_xy` (meters), `rx_theta_in_pis`, `rx_heading_in_pis`,
  `rx_wavelength_spacing`, `rx_lo`, `gains`, `vehicle_type`, `sdr_device_type`, `system_timestamp`.
- **Raw** (unless in `skip_fields`): `signal_matrix`.

See [conventions.md](../00_concepts/conventions.md) for the frame each θ key is in.

### `v5inferencedataset` (`spf/dataset/spf_dataset.py:448`) — real-time

Async/threaded variant for live streams; ingests from a queue, returns `None` until a session is
ready, no precomputed segmentation required. Used by on-rover realtime inference.

### Collation (handoff to training)

`v5_collate_keys_fast(keys, batch)` selects only the keys a model needs, stacks paired samples
into one flat batch, and returns a `TensorDict`. `v5_collate_beamsegnet` is the beamformer+seg
variant. This is the **03→04 handoff**.

## Data generation (synthetic)

- **`fake_dataset.py`** (`create_fake_dataset`) — parametric: orbiting emitter → φ via geometry →
  2-channel complex signal + noise → written as v4/v5 zarr. Fast, used heavily in tests.
- **`spf_generate.py`** — higher-fidelity physics simulation using `rf.py`'s `ULADetector`
  (orbit/bounce trajectories, multi-source). Produces intermediate structures.

## Migration / wrangling scripts (`spf/scripts/`)

| Script | Transformation |
|---|---|
| `v4_tx_rx_to_v5.py` | v4 → v5: GPS → XY mm projection + tx/rx timestamp interpolation. |
| `yarr_3p5_to_3p6.py`, `precompute_3p3_to_3p31.py` | Segmentation version upgrades. |
| `zarr_utils.py` | Schema creation + helpers (`zarr_shrink`, `truncate_zarr`, `compare_and_copy`). |
| `zarr_rechunk.py`, `zarr_shrink.py`, `truncate_zarr.py`, `zarr_fix_rx_spacing.py` | Storage fixes. |
| `segment_zarr.py` | Run segmentation over a zarr. |

## Known issues / tech debt (spot-verified)

- **Version sprawl** — segmentation versions 3.3/3.31/3.5/3.6/3.7 with one-off upgrade scripts;
  current is **3.7** but documented upgrade paths only reach 3.6. Confirm a dataset's `version`
  before training.
- **"session" vs "snapshot" vs "paired"** overloaded — see [glossary](../00_concepts/glossary.md).
- 🧪 Likely-dead helpers: `open_partial_ds.py`, `segment_zarr copy.py`, `wall_array_v{1,2}_idxs.py`
  (no current references found — verify before deleting).
- **v4-on-the-fly wrapping** (`v4_to_v5`) is not exercised on every code path; treat mixed-format
  training runs with care.

## Known issues

Dataset-relevant entries from [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) (source-of-truth bug list,
verified + adversarially confirmed). Severities below are the **current** triaged values:

- **#7 🔴 P0** (confirmed, realtime path) — two unconditional `breakpoint()` in
  `spf/dataset/spf_nn_dataset_wrapper.py:86,126`; both on the only path realtime/absolute-north
  inference takes → hang/`BdbQuit` on a headless rover.
- **#8 🟡 P2** (downgraded) — `get_segmentation` destructive recovery: on `UnpicklingError` it
  `os.remove`s the `.pkl` then recurses. The expensive `.yarr` is **not** deleted, so the risk is
  narrower than first stated; but `segment_if_not_exist=False` callers delete-then-refuse-to-rebuild
  (`spf/dataset/spf_dataset.py:1753-1760`).
- **#9 🟠 P1** — `beamform_signal_cpu` is a stub returning `None` (CPU beamforming fallback)
  (`spf/dataset/segmentation.py:~1025`).
- **#10 🟠 P1** — `v2_rssi_idxs` sets **both** RSSI indices to `"rssi0"`
  (`spf/dataset/wall_array_v2_idxs.py:~40`).
- **#11 🟠 P1** — `v5inferencedataset` mutable-default `skip_fields=[]` mutated in place via `+=`
  (`spf/dataset/spf_dataset.py:506`). (`v5spfdataset` has the same default but does **not** mutate it.)
- **#19 🟠 P1** — destructive zarr scripts: `zarr_fix_rx_spacing` overwrites in place with no backup;
  `precompute_3p3_to_3p31` does in-place migration with a `TODO THIS SHOULD BE FIXED!!!` non-finite→0
  hack (`spf/scripts/zarr_fix_rx_spacing.py`, `spf/dataset/precompute_3p3_to_3p31.py`).
- **#32 🟠 P1** (✅ verified) — `v5inferencedataset.__getitem__` reads `self.store` **unlocked** at
  `:670`; the reader thread can `pop(idx)` (eviction) in between → unhandled `KeyError`. Reachable
  under the production `max_store_size=3` when producers outrun the consumer
  (`spf/dataset/spf_dataset.py:670` vs `618/632`).
- **#33 🟠 P1** (✅ verified) — `v5inferencedataset` reader thread is **non-daemon** with
  `join(timeout=1.0)`, and its `multiprocessing.Queue` is never closed/drained → a timed-out join can
  leave a live thread + queue feeder blocking interpreter exit (`spf/dataset/spf_dataset.py:608/808`).
- **#39 🟠 P1** (✅ verified) — `v5inferencedataset.min_idx_stored` is initialized **only** inside the
  locked reader block (`:633`), so a consumer calling `__getitem__` before the first insert hits
  `AttributeError` at `:665` instead of the intended `None`/wait (`spf/dataset/spf_dataset.py:633,665`).

## Planned leaf pages

- `dataset_api.md` — ✅ **DONE.** Split across two verified reference docs:
  - [`v5spfdataset.md`](../reference/spf/dataset/v5spfdataset.md) — the on-disk training/eval class:
    verified `__getitem__`/`render_session` output-dict contract (every key, shape, units) plus the
    four θ-frame keys (array / craft / absolute / φ) and the constructor args.
  - [`v5inferencedataset.md`](../reference/spf/dataset/v5inferencedataset.md) — the realtime threaded
    class: producer/consumer + concurrency/locking contract and the eviction/timeout behavior.
- `data_formats.md` — full v4/v5/yarr field reference + version history. *(Unwritten — current source
  of truth: the [`datasets.md` inventory](../reference/_inventory/datasets.md).)*
- `segmentation.md` — algorithm + caching internals. *(Unwritten — current source of truth: the
  [`datasets.md` inventory](../reference/_inventory/datasets.md), `segmentation.py` section.)*
- `generation.md` — fake vs physics generation. *(Unwritten — current source of truth: the
  [`datasets.md` inventory](../reference/_inventory/datasets.md), `fake_dataset.py`/`spf_generate.py`
  sections.)*
- `migration_scripts.md` — each conversion script in detail. *(Unwritten — current source of truth:
  the [`datasets.md` inventory](../reference/_inventory/datasets.md), wrangling-script sections.)*
