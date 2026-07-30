# Capture data formats

One document per capture data version. **Read [`contracts.md`](./contracts.md) before editing
or adding one** — it defines the required sections and the field-table rules.

## Version matrix

| Version | Status | Container | Platform | Written by | Defining module | Doc |
|---|---|---|---|---|---|---|
| **v1** | abandoned | `np.memmap` float32 | wall array v1 | — (historical) | `spf/dataset/wall_array_v1_idxs.py` | [v1](./v1_data_format.md) |
| **v2** | legacy | `np.memmap` float32 | wall array | `GrblDataCollector` | `spf/dataset/wall_array_v2_idxs.py` | [v2](./v2_data_format.md) |
| v3 | never existed | — | — | — | — | see [v2 §1](./v2_data_format.md#1-motivation) |
| **v4** | legacy-readable, **most of the rover corpus** | Zarr/LMDB | rover | `DroneDataCollectorRaw` | `spf/dataset/v4_data.py` | [v4](./v4_data_format.md) |
| **v5** | **live** (wall array), the training format | Zarr/LMDB | wall array | `GrblDataCollectorRaw` | `spf/dataset/v5_data.py` | [v5](./v5_data_format.md) |
| **v6** | transitional, superseded before production | Zarr/LMDB | rover (direct-USB proto v1) | `DroneDataCollectorRawV6` | `spf/dataset/v6_data.py` | [v6](./v6_data_format.md) |
| **v7** | **live** (rover), current production | Zarr/LMDB | rover (direct-USB proto v2) | `DroneDataCollectorRawV7` | `spf/dataset/v7_data.py` | [v7](./v7_data_format.md) |

Derived artifacts, shared across versions:

| Artifact | What it is | Doc |
|---|---|---|
| Precompute cache (`.yarr` + `.pkl`) | segmentation, windowed beamformer, window stats | [precompute_cache](./precompute_cache_format.md) |

## Which one am I holding?

Every snippet in this folder needs the project environment active (`spf` deps: numpy, zarr,
torch) — on the dev box that is one of `/home/mouse9911/virtual-envs/spf*`; a bare system
`python3` has no numpy.

The capture config is embedded in every Zarr-based capture, so:

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
import sys, yaml
z = zarr_open_from_lmdb_store(sys.argv[1], mode='r')
c = yaml.safe_load(z['config'][0])
print('data-version:', c.get('data-version'))
print('receiver keys:', sorted(z['receivers/r0'].keys()))
" /path/to/dataset.zarr
```

The receiver key list is the tiebreaker if `data-version` is absent or lying: v4 and v5 differ
in position vs GPS fields, v6/v7 add gain metadata, and v7 is the only version carrying
`rssi_db_start` and a `radio_metadata_schema_version` attribute.

## Related documentation

- Corpus inventory, footprints, quality splits — [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md)
- QC metric definitions and gate thresholds — [`../QC_METRICS.md`](../QC_METRICS.md)
- Synthetic/fake data generation — [`../synthetic_raw_data.md`](../synthetic_raw_data.md)
- Units and angle-frame conventions — [`../../00_concepts/conventions.md`](../../00_concepts/conventions.md)
- Known bugs by id — [`../../KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md)
