# v5 capture data format

| | |
|---|---|
| **Status** | **live** — the wall-array capture format and the canonical training format |
| **Container** | Zarr hierarchy over an LMDB store (`*.zarr` directory holding `data.mdb`) |
| **Written by** | `GrblDataCollectorRaw` (`spf/data_collector.py:932`), orchestrated by `spf/grbl_radio_collection.py` |
| **Read by** | `v5spfdataset` (`spf/dataset/spf_dataset.py:833`) — the class every training run uses |
| **Supersedes** | [v2](./v2_data_format.md) (flat memmap, beamformer-only) |
| **Related** | [v4](./v4_data_format.md) is the rover sibling; `v5spfdataset` upgrades v4 to this shape on the fly |
| **Defining module** | `spf/dataset/v5_data.py` |

Section order and field-table rules come from [`contracts.md`](./contracts.md).

---

## 1. Motivation

v2 stored *conclusions* — a precomputed beamformer response per snapshot — and threw the IQ away.
That made every downstream idea impossible to try retroactively: change the segmentation, change
the beamformer, change `nthetas`, and you had to re-capture. v5 stores **raw IQ** (`signal_matrix`)
plus the geometry needed to label it, and pushes all interpretation into the loader and the
precompute cache. Everything the project has learned about segmentation (versions 3.2 → 3.7) was
only possible because v5 kept the samples.

The other change is **self-description**: the entire capture YAML is serialized into the store
(`config` array), so a dataset is interpretable without the launcher that produced it.

There is no v3. The number was skipped; the `3.x` versions in this project are
**segmentation** versions (`SEGMENTATION_VERSION = 3.7`, `spf/utils.py:14`), which is an
unrelated axis. Do not read `precompute_cache_3p7` as a data version.

## 2. Collection types that produce it

| | |
|---|---|
| Platform | **2D wall array** — GRBL gantry moving a receiver and an emitter on rails |
| Orchestrator | `spf/grbl_radio_collection.py` (dispatches on `data-version == 5`, `:170`) |
| Position truth | **Motor coordinates** from the GRBL controller, in mm — sub-mm and effectively noiseless |
| Transport | `iio` (USB-IIO to PlutoSDR); BladeRF2 for the `*_nuand` captures |
| Routines | `bounce`, `circle`, `calibrate`, `center` |
| Corpus | the wall-array captures under `2d_wallarray_v2_data`, 2024-06 → 2025-04 — counts and footprints in [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md) |

The rover never wrote v5 natively. Rover data reaches v5 shape by one of two routes described in
§6: on-the-fly upgrade in the loader, or the offline `v4_tx_rx_to_v5.py` merge.

## 3. Example configs

Committed configs, wall array v2:

- `data_collection/2d_wall_array/2d_wall_array_v2/*.yaml`
- Simulator/bench variants: `data_collection/rover/rover_v3.1/capture_configs/rover_receiver_config_simulator.yaml`

The format-determining fields:

```yaml
data-version: 5

receivers:
  - receiver-port: 1
    theta-in-pis: 0            # array orientation, multiples of pi
    antenna-spacing-m: 0.05075
    nelements: 2
    array-type: linear
    buffer-size: 524288        # B — complex samples per element per record
    f-carrier: 2.4671e+9
    f-sampling: 30.0e+6
    bandwidth: 3.0e+6

n-thetas: 65                   # only affects derived beamforming, not the raw store
n-records-per-receiver: 10000  # T — preallocated timesteps
dry-run: False                 # True disables the zarr entirely
```

Coupling a reader should know:

- **`buffer-size` must be identical across receivers** or `setup_record_matrix` asserts
  (`spf/data_collector.py:940-948`).
- `n-records-per-receiver` is both the loop bound and the preallocated time dimension
  (`spf/data_collector.py:732`, `:953`).
- `dry-run: True` skips dataset creation altogether — a config-level way to get no output.
- v5 is **not** gated by `spf/capture_schema.py`; that module validates v6/v7 transport
  contracts only. v5 configs are accepted as-is.

## 4. Container layout

```
<name>.zarr/                     LMDB store (data.mdb, lock.mdb)
├── config                       (1,) str — yaml.dump of the entire capture config
└── receivers/
    ├── r0/
    │   ├── signal_matrix        (T, 2, B)  complex64, chunks (1, 2, B//2), Blosc zstd clevel=1 BITSHUFFLE
    │   ├── system_timestamp     (T,)   float64, chunks (T,), no compressor
    │   ├── ... 9 more f64 keys  (T,)   float64
    │   ├── avg_phase_diff       (T, 2) float64
    │   ├── rssis                (T, 2) float64
    │   └── gains                (T, 2) float64
    └── r1/  … same, one group per entry in the YAML receivers list
```

Created by `v5rx_new_dataset` → `zarr_new_dataset` (`spf/scripts/zarr_utils.py:161`).

Sizing behaviour worth knowing:

- The store is **preallocated to `T` timesteps** but chunks materialize only when written, so a
  short run does not cost the full footprint. Nothing trims the unwritten tail — there is no
  `resize` on completion.
- `zarr_shrink` (`spf/scripts/zarr_utils.py:231`) is called on close to collapse the LMDB
  map size.
- Only `signal_matrix` is compressed. Every scalar key is written with `compressor=None` and a
  single chunk spanning all of `T`, which makes whole-column reads cheap and partial writes
  expensive.
- `skip_signal_matrix=True` produces the **`nosig`** variant: same layout, IQ omitted. That is
  the 5.5 GB corpus training actually streams.

## 5. Recorded fields (exhaustive)

`T` = timesteps, `B` = buffer size. Key lists: `v5rx_f64_keys` / `v5rx_2xf64_keys`
(`spf/dataset/v5_data.py:10-29`). All writes go through
`GrblDataCollectorRaw.write_to_record_matrix` (`spf/data_collector.py:961-978`).

### Raw signal

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `signal_matrix` | `(T, 2, B)` | complex64 | ADC counts (12-bit scale) | `data_collector.py:977` | The two antenna elements' IQ for one snapshot | Everything downstream derives from this. Absent in `nosig` copies. Element axis order = Pluto RX1, RX2 |

### Platform geometry (GRBL motor truth)

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `tx_pos_x_mm` | `(T,)` | float64 | mm | `data_collector.py:969` | Emitter x in gantry frame | With `rx_pos_*`, gives the ground-truth bearing |
| `tx_pos_y_mm` | `(T,)` | float64 | mm | `data_collector.py:970` | Emitter y | |
| `rx_pos_x_mm` | `(T,)` | float64 | mm | `data_collector.py:971` | Receiver-array centre x | |
| `rx_pos_y_mm` | `(T,)` | float64 | mm | `data_collector.py:972` | Receiver-array centre y | |
| `rx_heading_in_pis` | `(T,)` | float64 | multiples of π | dataclass default | Craft heading | **Wall array has no heading — this is 0 throughout.** It exists so v4 rover data can be upgraded into the same schema |
| `rx_theta_in_pis` | `(T,)` | float64 | multiples of π | dataclass, from `theta-in-pis` | Array mount orientation | Rotates array-frame θ into the craft/world frame. Constant for a capture |

### Radio configuration and per-record radio state

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `system_timestamp` | `(T,)` | float64 | s (host epoch) | dataclass | Host clock when the buffer was received | The only clock on the wall array. Monotonicity is a QC gate (`ts_nonmonotonic`) |
| `rx_spacing` | `(T,)` | float64 | m | dataclass, from `antenna-spacing-m` | Element separation | **Configured, not measured.** `d/λ` sanity is the `F:gain` / `rX_gain` QC metric; see `docs/learnings.md` |
| `rx_lo` | `(T,)` | float64 | Hz | dataclass | RX local-oscillator frequency | Asserted `> 1` at write time (`:973`). With `rx_spacing` gives `d/λ` |
| `rx_bandwidth` | `(T,)` | float64 | Hz | dataclass | RX filter bandwidth | |
| `avg_phase_diff` | `(T, 2)` | float64 | radians, wrapped to (−π, π] | `data_collector.py:458` | Circular mean of the per-sample RX1−RX2 phase difference | Cheap bearing proxy; the QC `circstd_corr` metric is built on it. **Both entries are identical** — `get_avg_phase_fast2` duplicates one mean (`spf/rf.py:768-780`). The length-2 axis is vestigial |
| `rssis` | `(T, 2)` | float64 | dB (Pluto scale) | dataclass | Per-element RSSI as reported by the AD9361 | Coarse level only; not calibrated across devices |
| `gains` | `(T, 2)` | float64 | dB | dataclass | Per-element hardware RX gain | Under `slow_attack` AGC this moves during a capture. Needed to interpret `rssis`. **Read `docs/learnings.md` on what `F:gain` means before using it** |

### Store-level arrays and attributes

| Field | Shape | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|---|
| `config` | `(1,)` | str | — | `yaml.dump` of the whole capture config (`zarr_utils.py:191`) | The authoritative record of spacing, LO, routine, buffer size. **Trust this over the filename** |

Zarr **attributes** are also part of the format. `DataCollector._record_receiver_identities`
(`spf/data_collector.py:634-690`, reached from `radios_to_online()` which
`grbl_radio_collection.py:179` calls) writes `sdr_identity_version` on the store and, per receiver
group: `sdr_identity_version`, `sdr_family`, `iio_uri_at_capture`, `rx_transport`, plus
`sdr_serial`, `usb_vendor_id`, `usb_product_id`, `usb_bus_at_capture`, `usb_address_at_capture`,
`usb_port_path` when known. On this format `rx_transport` is `iio`, so the direct-USB and
firmware-provenance attributes are **absent** — `_capture_firmware_provenance` returns `None` for
non-direct-USB receivers (`:104-105`). Full inventory:
[v7 §5.5](./v7_data_format.md#55-store-level-arrays-and-attributes).

Older captures predate this block and simply have no attributes; treat a missing attribute as
"unknown", never as a claim about the hardware.

## 6. Post-processed fields

Nothing in §5 is derived. Everything below is computed after capture.

### 6.1 Loader-derived, in memory only

Computed by `v5spfdataset` on open/`__getitem__`; never written to the raw store.

| Field | Origin | Meaning |
|---|---|---|
| `rx_pos_mm`, `tx_pos_mm` | stacked from `*_pos_*_mm` (`spf_dataset.py:1238`) | Position as a 2-vector |
| `rx_pos_xy` | `rx_pos_mm / distance_normalization` (`:1474`) | Normalized position for the network |
| `ground_truth_theta` | geometry: tx−rx bearing rotated by `rx_theta_in_pis` | Array-frame bearing label |
| `craft_ground_truth_theta` | same, craft frame | Label for the fused two-array model |
| `ground_truth_phi` | θ→φ mapping via `d/λ` | Phase-domain label |
| `absolute_theta` | world frame | |
| `y_rad`, `y_phi`, `craft_y_rad`, `y_rad_binned` | training targets, `training_only_keys` (`spf_dataset.py:421-430`) | What the loss is computed against |

The −sin sign convention in the θ derivation is **empirically validated, not derived** — see
`claude_docs/reference/_stage1_audit/coverage_datapath.md` and
`tests/test_fake_data.py::test_fake_data_array_orientation`.

### 6.2 Segmentation / beamforming precompute cache

`windowed_beamformer`, `weighted_beamformer`, `all_windows_stats`, `weighted_windows_stats`,
`downsampled_segmentation_mask`, `simple_segmentations`, `mean_phase_segmentation` — a separate
sidecar keyed by segmentation version, regenerable from the IQ. See
[`precompute_cache_format.md`](./precompute_cache_format.md).

### 6.3 Offline derived copies

| Artifact | Producer | Note |
|---|---|---|
| `nosig` copies | `zarr_rechunk.py --skip-signal-matrix` | Same schema, IQ stripped; what training streams |
| Rechunked copies | `spf/scripts/zarr_rechunk.py` | Read-throughput tuning only |
| v4→v5 merges | `spf/scripts/v4_tx_rx_to_v5.py` | Joins a rover rx capture with the *emitter's* own capture on GPS time to synthesize `tx_pos_*_mm`. **New file, never in place** |

## 7. Reading it

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

with v5spfdataset_manager(
    "/mnt/md2/.../wallarrayv3_2024_07_18_02_00_47_nRX2_bounce_spacing0p05075",  # no .zarr
    nthetas=65,
    precompute_cache="/mnt/md2/cache/precompute_cache_3p7",
    paired=True,
    ignore_qc=True,
    skip_fields=["signal_matrix"],   # required for nosig copies
) as ds:
    sample = ds[0]
```

Notes:

- `prefix` omits the `.zarr` suffix.
- `skip_fields` must be a list; a bare string is wrapped defensively
  (`normalize_skip_fields`, `spf_dataset.py:448`) because `list("signal_matrix")` used to
  explode into 13 single-character entries.
- `v4=False` (the default) for genuine v5 files.
- Without `ignore_qc=True` the loader applies quality gates and may refuse the dataset.

## 8. Known issues and traps

- **`avg_phase_diff`'s second entry carries no information** (§5). Code that averages the two
  columns is averaging a value with itself.
- **`rx_heading_in_pis` is structurally present but always 0** on wall captures. Do not
  interpret it as a measurement.
- **`rx_spacing` is what the YAML claimed**, not what was built. Mislabeled-spacing captures
  exist; the in-place fixer `zarr_fix_rx_spacing` **overwrites with no backup** (KI#19) — copy
  first, and per root `CLAUDE.md` prefer writing a corrected copy to a new location.
- **Sub-GHz captures are untrustworthy at the physics level**, not the format level: 0 of 463
  are OK. Root cause in `docs/learnings.md` L4 (IF parked near 0 Hz).
- **A killed capture leaves an untrimmed tail** of `T − written` zero rows and a `.tmp`
  filename. Nothing resizes the store.
- `n-thetas` in the config does *not* constrain the raw store — it only seeds derived
  beamforming. Two datasets with different `n-thetas` are still schema-identical.

## 9. Verification

The key list a conforming file must expose:

```bash
python3 -c "from spf.dataset.v5_data import v5rx_keys; print(sorted(v5rx_keys()))"
```

Compare against a real file:

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.dataset.v5_data import v5rx_keys
import sys
z = zarr_open_from_lmdb_store(sys.argv[1], mode='r')
have, want = set(z['receivers/r0'].keys()), set(v5rx_keys())
print('missing:', sorted(want - have)); print('extra:', sorted(have - want))
" /path/to/dataset.zarr
```

Invariants: `rx_lo > 1` for every written row; `signal_matrix.shape == (T, 2, B)` with `B` equal
to the config's `buffer-size`; `system_timestamp` non-decreasing over written rows;
`yaml.safe_load(z['config'][0])['data-version'] == 5`.

Fleet-scale checking is `spf/scripts/dataset_quality_scan.py` (see
[`../QC_METRICS.md`](../QC_METRICS.md)); it validates content quality, not schema conformance.

## 10. Changelog

- **2026-07-29 (review pass)** — documented the zarr **attributes** (identity block; no
  firmware provenance on the `iio` path); corrected writer citations (`:970-973`→`:969-972`,
  signal_matrix `:985`→`:977`, `rx_lo` assert `:975`→`:973`, `v5spfdataset` `:838`→`:833`,
  `normalize_skip_fields` `:447`→`:448`).
- **2026-07-29** — created. Field tables generated from `spf/dataset/v5_data.py` and the
  `GrblDataCollectorRaw` writer; `avg_phase_diff` duplication traced to
  `get_avg_phase_fast2`.
