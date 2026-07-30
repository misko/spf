# v6 capture data format

| | |
|---|---|
| **Status** | **transitional** — implemented, superseded by v7 before it became the production format |
| **Container** | Zarr hierarchy over an LMDB store |
| **Written by** | `DroneDataCollectorRawV6` (`spf/data_collector.py:849`), via `spf/mavlink_radio_collection.py:320` |
| **Read by** | `v5spfdataset` with `v4=True` (v4 superset); the metadata arrays have no loader |
| **Supersedes / superseded by** | [v4](./v4_data_format.md) → v6 → [v7](./v7_data_format.md) |
| **Requires** | `rx-transport: direct_usb` + `direct-usb.protocol-version: 1` — and **v6 is the only data version protocol v1 permits** (`spf/capture_schema.py:110`) |
| **Defining module** | `spf/dataset/v6_data.py` |

Section order and field-table rules come from [`contracts.md`](./contracts.md).

---

## 1. Motivation

v6 was the first attempt at the problem v7 now owns: making the hardware gain **attributable to
the samples in a buffer**, by having custom Pluto firmware snapshot gain at the start and end of
each buffer and ship it in-band over a direct USB bulk transport instead of a post-hoc USB-IIO
read. See [v7 §1](./v7_data_format.md#1-motivation) for why that matters.

It was superseded on two counts:

1. **Gain is reported as an AD9361 gain-table index** (`uint8`), not dB. Converting an index to
   dB needs the device's active gain table, which is not recorded in the capture — so the numbers
   are not self-interpreting.
2. **No RSSI endpoints at all.** `rssis` and `gains` are NaN-filled on the protocol-v1 path
   (`spf/sdrpluto/sdr_controller.py:897-898`), so a v6 capture has *less* usable level
   information than v4 despite carrying more metadata.

Protocol v2 fixed both (dB values, RSSI endpoints) and got data version 7.

## 2. Collection types that produce it

Same platform as v4/v7 — rover v3.1, `spf/mavlink_radio_collection.py`, GPS + compass truth —
with `rx-transport: direct_usb` at `protocol-version: 1`.

**No committed capture config uses `data-version: 6`.** Every `data-version` in the repo's YAML
is 4, 5, or 7. v6 therefore exists as a code path and a schema, and — as far as this document's
evidence goes — not as a body of recorded data. ⚠ unverified: whether any v6 captures exist on
the storage arrays; the scanned corpus in [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md) reports
only v4 (rover) and v5 (wall).

Treat v6 as **read-support you may need for a bench capture from that window**, not as a target
for new work.

## 3. Example configs

No committed example exists. To construct one, take a v4 rover config and add, per receiver:

```yaml
data-version: 6

receivers:
  - receiver-port: 2
    rx-transport: direct_usb
    direct-usb:
      protocol-version: 1
      require-gain-metadata: true      # optional; enforces non-dummy gains
```

Unlike v7, **v6 gets no normalization pass** — `normalize_capture_config` returns early for any
version other than 7 (`spf/capture_schema.py:35`), so every transport field must be written out
explicitly. `validate_transport_schema` still applies: protocol v1 with any `data-version` other
than 6 is a hard error, and all receivers in one capture must share the transport.

Setting `require-gain-metadata: true` makes the collector raise on a buffer flagged
`DUMMY_GAINS` (`spf/sdrpluto/sdr_controller.py:880`) rather than recording synthetic gains.

## 4. Container layout

```
<name>.zarr/
├── config                       (1,) str
└── receivers/r{0,1}/
    ├── signal_matrix            (T, 2, B)  complex64, Blosc zstd
    ├── <9 v4 f64 keys>          (T,)       float64
    ├── avg_phase_diff|rssis|gains (T, 2)   float64
    ├── <7 scalar metadata keys>  (T,)      per-key dtype, uncompressed, one chunk
    └── <5 per-element metadata>  (T, 2)    per-key dtype, uncompressed, one chunk
```

Note: **no `radio_metadata_schema_version` attribute.** That attr is set only by v7
(`spf/dataset/v7_data.py:66`), so its absence plus the presence of `gain_index_start` is the
v6 signature.

## 5. Recorded fields (exhaustive)

Key lists: `v6rx_f64_keys`, `v6rx_2xf64_keys` (verbatim copies of the v4 lists),
`v6rx_scalar_keys`, `v6rx_2x_keys` (`spf/dataset/v6_data.py:9-30`). Metadata written by
`DroneDataCollectorRawV6.write_to_record_matrix` (`spf/data_collector.py:879-888`); values
originate in `_direct_v1_rx_buffer` (`spf/sdrpluto/sdr_controller.py:877-920`).

### 5.1 Inherited v4 fields

Identical to [v4 §5](./v4_data_format.md#5-recorded-fields-exhaustive): `signal_matrix`,
`system_timestamp`, `gps_timestamp`, `gps_lat`, `gps_long`, `heading`, `rx_theta_in_pis`,
`rx_spacing`, `rx_lo`, `rx_bandwidth`, `avg_phase_diff`, `rssis`, `gains`.

**With one severe difference:** on the protocol-v1 path `rssis` and `gains` are written as
`np.full(2, np.nan)` (`sdr_controller.py:897-898`). A v6 capture has **no dB-valued level data
at all** — not in the v4 columns, not in the metadata columns.

### 5.2 Per-record metadata — `(T,)`

| Field | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|
| `gain_metadata_valid` | bool | — | Firmware asserts the gain snapshots are trustworthy | Filter on this before touching any gain field |
| `gain_metadata_flags` | **uint16** | bitfield | Raw `MetadataFlags` for the buffer | ⚠ `MetadataFlags` now defines bits up to 18 (`direct_usb_protocol.py:52-68`); anything ≥ bit 16 **cannot be represented** in uint16. Protocol v1 does not set those bits (they are the v2 RSSI/dB flags), so this is a latent hazard rather than a live bug — but do not port v6 readers forward |
| `stream_id` | uint64 | — | Firmware stream identity | A change means the stream restarted |
| `buffer_sequence` | uint64 | count | Firmware buffer counter | Gap detection |
| `sample_sequence` | uint64 | samples | Sequence number of the first sample in the buffer | Meaningful only with `SAMPLE_SEQUENCE_VALID` |
| `gain_start_read_duration_ns` | uint32 | ns | Cost of the start-of-buffer gain read | Firmware health, not signal timing |
| `gain_end_read_duration_ns` | uint32 | ns | Cost of the end-of-buffer gain read | As above |

### 5.3 Per-element metadata — `(T, 2)`, element axis = RX1, RX2

| Field | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|
| `gain_index_start` | uint8 | **AD9361 gain-table index** | Gain at the buffer's first sample | **Not dB.** Needs the device's active gain table, which the capture does not record. Under protocol v1 these come from the firmware; `0xFF` is the sentinel the *other* paths use when indices are unavailable (legacy IIO `:137-138`, protocol v2 `:938-939`) |
| `gain_index_end` | uint8 | gain-table index | Gain at the last sample | Same caveat |
| `gain_endpoints_equal` | bool | — | Start and end index are identical | True ⇒ the record has one unambiguous gain state. The usable clean-subset selector |
| `first_gain_change_sample` | int32 | sample index | Where gain first moved inside the buffer; **−1 = never moved** | `0xFFFFFFFF` normalized to −1 (`sdr_controller.py:884-892`). −1 means "no change", never "unknown" |
| `iq_power_dbfs` | float32 | dBFS | Mean post-gain sample power vs 12-bit full scale | **The only absolute level measurement in a v6 capture.** Computed on the host from the IQ (`sdr_controller.py:121`), independent of the radio's reporting. `-inf` for an all-zero buffer |

### 5.4 Store-level arrays and attributes

`config` — `(1,)` str array, `yaml.dump` of the capture config. Authoritative for spacing/LO/routine.

Zarr **attributes**: v6 carries the same identity + direct-USB + firmware-provenance attribute set
as v7, written by the base collector (`spf/data_collector.py:634-690`) — see
[v7 §5.5](./v7_data_format.md#55-store-level-arrays-and-attributes) for the full inventory. Two v6
differences: `radio_metadata_schema_version` is **absent**, and `gain_metadata_protocol_version` is
**1**. v6 is also **not** subject to v7's `firmware_verified` / `hardware_fingerprint_v1` hard
gates, so a v6 capture carries weaker provenance guarantees even when the attributes are present.

### Field-set delta vs v7

| | v6 | v7 |
|---|---|---|
| Gain units | table index (uint8) | **dB** (float32) |
| RSSI endpoints | ✗ | ✓ `rssi_db_start` / `rssi_db_end` |
| `rssi_metadata_valid` | ✗ | ✓ |
| RSSI read-duration counters | ✗ | ✓ |
| `gain_metadata_flags` width | uint16 | uint32 |
| `radio_metadata_schema_version` attr | ✗ | `= 2` |
| `gains` / `rssis` (v4 columns) | **NaN** | populated from the dB endpoints |

## 6. Post-processed fields

No field in §5 is derived. Downstream processing is **identical to v4** — the loader sees the v4
subset, runs the `v4_to_v5` upgrade (synthesizing `rx_heading_in_pis`, zero-filling positions),
and the segmentation precompute cache applies unchanged. See
[v4 §6](./v4_data_format.md#6-post-processed-fields) and
[`precompute_cache_format.md`](./precompute_cache_format.md).

The v6 metadata arrays have **no consumer anywhere in the repo** outside the direct-USB
soak/smoke tools. Nothing in training or the filters reads them.

## 7. Reading it

Load as a v4 superset:

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

with v5spfdataset_manager(prefix, nthetas=65, precompute_cache=..., v4=True, paired=True) as ds:
    sample = ds[0]
```

Read the metadata directly — and expect the v4 level columns to be NaN:

```python
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
import numpy as np

r0 = zarr_open_from_lmdb_store("/path/to/v6.zarr", mode="r")["receivers/r0"]
print("gains all-NaN:", bool(np.all(np.isnan(r0["gains"]))))      # expected True
clean = np.asarray(r0["gain_metadata_valid"]) & np.all(r0["gain_endpoints_equal"], axis=1)
print("stable-gain records:", int(clean.sum()))
print("median iq_power_dbfs:", float(np.median(np.asarray(r0["iq_power_dbfs"])[clean])))
```

## 8. Known issues and traps

- **`gains` and `rssis` are NaN, by construction.** Code that averages them silently produces
  NaN; code that `nanmean`s them silently produces nothing. This is the single biggest v6 trap.
- **Gain indices are not dB and are not convertible from the capture alone.** Without the active
  AD9361 gain table, a v6 gain number supports only equality/inequality reasoning
  (`gain_endpoints_equal`), not magnitude.
- **`gain_metadata_flags` is uint16 while the flag enum has grown past bit 15.** Latent, not
  live, under protocol v1 — but any reader shared with v7 must widen.
- **`first_gain_change_sample == -1` means "no change"**, not "unknown".
- All v4 rover traps apply unchanged: zero-filler positions, `heading` in degrees, configured
  (not measured) `rx_spacing`, normal 46–70 % NaN, untrimmed tails on killed runs. See
  [v4 §8](./v4_data_format.md#8-known-issues-and-traps).
- **Do not start new work on v6.** Protocol v1 firmware is not what the boot path RAM-loads; the
  pinned firmware in the production configs is a protocol-v2 build.

## 9. Verification

**A committed v6 validator exists:** `spf/scripts/validate_direct_usb_gain_zarr.py`
("Validate and report throughput for an SPF v6 direct-USB capture"). It rejects frames carrying
`DUMMY_GAINS`, `GAIN_READ_FAILED`, `DEVICE_IIO_OVERFLOW` or `FPGA_EVENT_OVERFLOW` and reports
throughput. Note it has **no RSSI checks** — v6 has no RSSI to check — which is the cleanest way to
see the v6/v7 difference in one place (compare
[v7 §9](./v7_data_format.md#9-verification)).

```bash
python3 -m spf.scripts.validate_direct_usb_gain_zarr /path/to/v6.zarr
```

Schema-only key check:

```bash
python3 -c "from spf.dataset.v6_data import v6rx_keys; print(sorted(v6rx_keys()))"
```

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.dataset.v6_data import v6rx_keys
import sys, yaml, numpy as np
z = zarr_open_from_lmdb_store(sys.argv[1], mode='r')
print('data-version:', yaml.safe_load(z['config'][0]).get('data-version'))
print('has v7 attr (should be None):', z.attrs.get('radio_metadata_schema_version'))
have, want = set(z['receivers/r0'].keys()), set(v6rx_keys())
print('missing:', sorted(want - have)); print('extra:', sorted(have - want))
" /path/to/v6.zarr
```

Signature test: `gain_index_start` present **and** `radio_metadata_schema_version` absent ⇒ v6.
`gain_db_start` present ⇒ v7.

## 10. Changelog

- **2026-07-29 (review pass)** — added the committed `validate_direct_usb_gain_zarr.py` to §9 and
  the attribute inventory to §5.4; corrected the `0xFF` sentinel claim (it is the legacy-IIO and
  protocol-v2 paths that fill `0xFF`, not the v1 path documented here) and the NaN-fill citation
  (`:896-897`→`:897-898`).
- **2026-07-29** — created. Field tables from `spf/dataset/v6_data.py`; NaN level columns and
  index-vs-dB semantics traced to `_direct_v1_rx_buffer`; uint16 flag-width hazard noted against
  the current `MetadataFlags` enum. Existence of recorded v6 captures on the arrays left
  unverified.
