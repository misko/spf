# Precompute cache format (segmentation sidecar)

| | |
|---|---|
| **Status** | **live** — required by every training run; several coexisting caches (`precompute_cache_3p4` … `3p7`), footprints in [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md) |
| **Container** | a **pair** of files per dataset: `*.yarr` (Zarr over LMDB) + `*.pkl` (pickle) |
| **Written by** | `spf/dataset/segmentation.py` (`new_yarr_dataset`, `spf/scripts/zarr_utils.py:30`); batch driver `spf/scripts/segment_zarr.py` |
| **Read by** | `v5spfdataset` → `data_from_precomputed` (`spf/dataset/spf_dataset.py:278`) |
| **Applies to** | [v4](./v4_data_format.md), [v5](./v5_data_format.md), [v6](./v6_data_format.md), [v7](./v7_data_format.md) — any format carrying `signal_matrix` |
| **Current version** | `SEGMENTATION_VERSION = 3.7` (`spf/utils.py:14`) |

This is **derived, regenerable data** — not a capture format. It exists because segmenting a
524,288-sample buffer and beamforming every window is far too slow to do inside a training loop.
Section conventions follow [`contracts.md`](./contracts.md), adapted: a derived artifact has no
"collection types" or "example configs" section.

---

## 1. Motivation

The rover's emitter is a **bursty beacon**, so a large fraction of every buffer is noise (46–70 %
NaN is normal, `docs/learnings.md` L2). Training on the whole buffer trains mostly on silence.
Segmentation finds the windows that actually contain signal, and the cache stores:

- which windows were signal (`downsampled_segmentation_mask`, `simple_segmentations`),
- per-window statistics and beamformer responses, so the network sees windows rather than raw IQ,
- signal-weighted summaries, so a model can consume one vector per snapshot.

The cache is **versioned** because the segmentation algorithm keeps changing: `3.2` → `3.7` are
different answers over the same raw captures, which is why several caches coexist.
Version mismatch is a correctness issue, not a performance one — see §5.

## 2. Naming and pairing

From `v5spfdataset.results_fn` (`spf/dataset/spf_dataset.py:1653`):

```
<precompute_cache>/<basename(prefix) with "_nosig" removed>_segmentation_nthetas<N>.pkl
<precompute_cache>/<same stem>_segmentation_nthetas<N>.yarr
```

Two consequences worth internalizing:

- **`_nosig` is stripped**, so a signal-bearing capture and its IQ-stripped copy share one cache
  entry. That is intended — the derived data is identical.
- **`nthetas` is in the filename**, so caches for different `nthetas` coexist without clobbering.
  Unlike v2, `nthetas` here is explicit rather than implied by row width.

The `.yarr` holds the arrays; the `.pkl` holds the variable-length segmentation lists plus the
version. **Both are required**; a `.yarr` without its `.pkl` is unusable.

## 3. Container layout

```
<stem>.yarr/                       LMDB store, map_size 2**32
├── version                        (1,) float32 — SEGMENTATION_VERSION at write time
└── r{0,1}/                        RECEIVER COUNT IS HARDCODED TO 2 (segmentation.py:259)
    ├── all_windows_stats          (S, 3, W)   float16, chunks (1, -1, -1), uncompressed
    ├── weighted_windows_stats     (S, 3)      float32, chunks (1, -1), uncompressed
    ├── windowed_beamformer        (S, W, nthetas) float16, chunks (1, -1, -1), uncompressed
    ├── weighted_beamformer        (S, nthetas)    float32, chunks (1, -1), uncompressed
    ├── downsampled_segmentation_mask (S, W)  bool,   chunks (1, -1), uncompressed
    └── mean_phase                 (S,)        float32, chunks (-1), uncompressed
```

`S` = sessions (= capture timesteps), `W` = windows per snapshot, `3` = the statistic axis.
Shapes are derived from the first segmentation result rather than declared
(`spf/dataset/segmentation.py:241-256`), so `W` is a property of the algorithm version and the
buffer size, not a config field.

```
<stem>.pkl                         pickle.dump({...}, ...)   segmentation.py:344-350
├── "version"                      float — SEGMENTATION_VERSION
└── "segmentation_by_receiver"     {"r0": [{"simple_segmentation": [...]}, ...], "r1": [...]}
```

Notes on the store:

- **Nothing is compressed.** `compressor=None` on every array, with a comment recording that this
  was a "quiet change during 3p5" (`zarr_utils.py:71`) — as was `chunk_size` 16 → 1 (`:39`).
  Caches written before and after that change differ in layout while carrying compatible version
  numbers. ⚠ treat 3p5-era caches with suspicion.
- **float16 for the big arrays** (`all_windows_stats`, `windowed_beamformer`) is a deliberate
  size/precision trade; the weighted summaries stay float32.
- `zarr_shrink` runs on close for non-temp files (`segmentation.py:341`).

## 4. Fields

All fields here are **post-processed**; none is recorded. Producer for every row:
`segment_session` → `segment_session_from_zarr` (`spf/dataset/segmentation.py:361+`), consuming
`signal_matrix[session_idx]` from the capture.

| Field | Shape | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|---|
| `version` | `(1,)` | float32 | — | Segmentation version that wrote this cache | **Check before trusting anything else.** `get_segmentation_version` (`spf_dataset.py:1660`) treats a scalar-shaped `version` as 3.0 for backwards compatibility |
| `all_windows_stats` | `(S, 3, W)` | float16 | mixed | Per-window statistics — mean, stddev, median signal strength | The network's window-level input. float16 ⇒ do not use for precise thresholds |
| `weighted_windows_stats` | `(S, 3)` | float32 | mixed | Signal-weighted collapse of the above over windows | One vector per snapshot |
| `windowed_beamformer` | `(S, W, nthetas)` | float16 | linear power (arb.) | Beamformer response per window | Subsampled to `windows_per_snapshot` at load time if wider (`spf_dataset.py:285-288`) |
| `weighted_beamformer` | `(S, nthetas)` | float32 | linear power (arb.) | Signal-weighted beamformer per snapshot | The compact bearing evidence |
| `downsampled_segmentation_mask` | `(S, W)` | bool | — | Which windows were classified as signal | The "was there a beacon burst here" answer |
| `mean_phase` | `(S,)` | float32 | radians | Mean phase difference over segmented (signal-only) windows | Differs from the capture's `avg_phase_diff`, which averages the **whole** buffer including noise. On bursty rover data this is the more meaningful of the two |
| `segmentation_by_receiver` (pkl) | list per session | — | — | `simple_segmentation`: variable-length window descriptors | Variable length is why it is a pickle and not a zarr array |

Loader-side keys built from these: `simple_segmentations`, `mean_phase_segmentation`
(`segmentation_based_keys`, `spf_dataset.py:432-438`). Any of them can be suppressed via
`skip_fields`.

## 5. Behaviour a caller must know

- **Regeneration is on-demand and implicit.** `v5spfdataset` calls `get_segmentation(...)` on
  open (`spf_dataset.py:1333`) and, with `segment_if_not_exist=True`, will compute and write a
  missing cache — turning a "load a dataset" call into hours of CPU. In a training job that is
  usually a mistake; precompute in batch with `spf/scripts/segment_zarr.py` instead.
- **Version mismatch changes results silently** unless checked. A `precompute_cache_3p4`
  directory and a `3p7` directory hold structurally identical files with different contents.
  The frozen-val-set contract in `docs/learnings.md` exists partly because of this — read it
  before mixing caches across experiments.
- **Receiver count is hardcoded to 2** (`segmentation.py:259`, and the `for r_idx in [0, 1]`
  write loop). A single-radio capture (rover 2) does not fit this shape. ⚠ unverified how
  single-receiver rover captures are handled in practice — check before caching rover 2 data.
- **Upgrade path**: `spf/scripts/yarr_3p5_to_3p6.py` migrates one version pair; there is no
  general converter. `spf/dataset/precompute_3p3_to_3p31.py` handles another specific hop.
- **B2-backed caches**: `v5spfdataset` can pull a `.yarr` + `.pkl` pair from B2 into a local
  directory (`spf_dataset.py:330-355`). The *inference* cache's write path is separately broken
  for B2 (KI#55 / N1) — different artifact, related trap.

## 6. Verification

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
import pickle, sys
stem = sys.argv[1]   # .../<name>_segmentation_nthetas65
z = zarr_open_from_lmdb_store(stem + '.yarr', mode='r')
print('yarr version:', z['version'][0])
for k, a in z['r0'].arrays():
    print(f'  r0/{k}: {a.shape} {a.dtype}')
meta = pickle.load(open(stem + '.pkl', 'rb'))
print('pkl version:', meta['version'],
      '| receivers:', sorted(meta['segmentation_by_receiver'].keys()),
      '| sessions:', len(meta['segmentation_by_receiver']['r0']))
" /mnt/md2/cache/precompute_cache_3p7/<name>_segmentation_nthetas65
```

Invariants: `.yarr` and `.pkl` versions agree with each other **and** with the
`segmentation_version` the loader is configured for; `S` equals the capture's written timestep
count; `nthetas` in the filename equals `windowed_beamformer.shape[-1]`.

## 7. Changelog

- **2026-07-29** — created. Layout from `new_yarr_dataset` (`spf/scripts/zarr_utils.py:30-95`)
  and `spf/dataset/segmentation.py:241-350`; naming from `v5spfdataset.results_fn`. The
  uncompressed/chunk-1 "quiet change during 3p5" and the hardcoded 2-receiver assumption are
  recorded as hazards; single-radio handling left unverified.
