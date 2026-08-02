# v7 capture data format

| | |
|---|---|
| **Status** | **live** — current rover production format |
| **Container** | Zarr hierarchy over an LMDB store, plus store- and receiver-level attributes (§5.5) |
| **Written by** | `DroneDataCollectorRawV7` (`spf/data_collector.py:890`), orchestrated by `spf/mavlink_radio_collection.py:327` |
| **Read by** | `v5spfdataset` with `v4=True` (v4 field superset); the extra metadata arrays are **not consumed by training** |
| **Supersedes** | [v6](./v6_data_format.md) (direct-USB protocol v1, gain *indices*) |
| **Requires** | `rx-transport: direct_usb` + `direct-usb.protocol-version: 2`, enforced by `spf/capture_schema.py` |
| **Defining module** | `spf/dataset/v7_data.py` |

Section order and field-table rules come from [`contracts.md`](./contracts.md).

---

## 1. Motivation

v4 recorded `gains` and `rssis` as **one host-side reading per buffer, taken after the fact over
USB-IIO**. Under `slow_attack` AGC the hardware gain moves *during* a 524,288-sample buffer, so a
single post-hoc number cannot say what gain applied to the samples — which makes `rssis`
uninterpretable in absolute terms and pollutes any attempt to use level as a feature. That is the
root of the long-running `F:gain` confusion recorded in `docs/learnings.md`.

v7 exists to make the gain state **attributable to the samples**:

1. Custom Pluto firmware snapshots gain and RSSI **at the start and the end of every buffer**,
   in dB, and reports them in-band over a direct USB bulk protocol (v2) — no separate IIO
   round-trip, no ambiguity about ordering.
2. It reports **whether the gain moved inside the buffer**, and at which sample it first moved,
   so a record can be accepted, rejected, or split on that basis.
3. It carries **stream and sequence counters**, so dropped or reordered buffers become
   detectable rather than silently stitched.
4. It records **`iq_power_dbfs`**, a host-computed power measure that is independent of the
   radio's own reporting and therefore usable as a cross-check.

v7 is where v6's idea landed after one revision: v6 exposed AD9361 gain **table indices**
(`uint8`), which need a per-device table to convert to dB. v7 reports **dB directly** and adds
RSSI endpoints, which v6 lacked entirely.

## 2. Collection types that produce it

| | |
|---|---|
| Platform | **Rover v3.1** — RPi + ArduPilot ground vehicle, 1–2 PlutoPlus SDRs |
| Orchestrator | `spf/mavlink_radio_collection.py`, launched at boot by `drone_run.sh` |
| Position truth | **GPS** (`gps_lat`/`gps_long`) + compass `heading` — noisy, unlike the wall array's motor coordinates |
| Transport | **direct USB bulk, protocol v2** — vendor interface, not USB-IIO |
| Firmware | Pinned per config: `pluto-firmware.release-tag`, RAM-booted at boot with a checksum gate |
| Routines | `bounce` (rover 1, 3), `circle` (rover 2) |
| Status | This is what a rover writes today. Every `*_production_v7.yaml` is v7 |

The direct-USB firmware is **RAM-booted every boot** by `spf-pluto-direct-usb.service`; QSPI is
left alone. A rover that fails the firmware gate does not reach collection.

## 3. Example configs

Canonical, one per rover, selected at boot from `/home/pi/rover_id`
(`spf/scripts/rover_capture_config.py:18-22`):

- `data_collection/rover/rover_v3.1/capture_configs/rover1_production_v7.yaml` (2 radios, 35 mm)
- `data_collection/rover/rover_v3.1/capture_configs/rover2_production_v7.yaml` (1 radio, 50.75 mm)
- `data_collection/rover/rover_v3.1/capture_configs/rover3_production_v7.yaml` (2 radios, 43 mm)

```yaml
data-version: 7

pluto-firmware:                  # all seven keys required, production uses persistent QSPI
  release-tag: v0.38-plutoplus-spf-gain-rssi-fingerprint-v2
  image-sha256: 5f8220bc...
  boot-mode: qspi

receivers:
  - receiver-port: 2             # USB2 = Radio A
    theta-in-pis: 0              # array A: athwartships
    antenna-spacing-m: 0.043
    nelements: 2
    buffer-size: 524288
    f-carrier: 5.766e+9
    bandwidth: 3.0e+6
  - receiver-port: 1             # USB1 = Radio B
    theta-in-pis: 0.5            # array B: fore-aft, 90 deg from A

n-records-per-receiver: 3000     # T — about 25 min at seconds-per-sample 0.5
routine: bounce
seconds-per-sample: 0.5
```

**The transport fields are implicit and must not be repeated.** `normalize_capture_config`
(`spf/capture_schema.py:26`) — applied both by the boot resolver and by the collector itself
(`spf/mavlink_radio_collection.py:191`, with `validate_transport_schema` at `:194`) — materializes,
per receiver, `rx-transport: direct_usb`,
`direct-usb.protocol-version: 2`, `require-gain-metadata: true`, and
`frame-count-per-request: 1`. Explicitly *conflicting transport or metadata*
values are rejected, not overwritten — including `require-gain-metadata:
false`, which is a hard error under v7. The frame count is a transport tuning
value rather than a schema version: an explicitly configured positive value is
accepted when it does not exceed the gadget's advertised finite-frame limit.
Values above one use a bounded rolling queue (one production-sized USB transfer
per radio by default) and still return one timestampable frame at a time. Keep
the default at one until the rolling path passes the attached two-radio gate.

Cross-version rules enforced by `validate_transport_schema` (`:81`):

| Transport | Protocol | Allowed data-version |
|---|---|---|
| `direct_usb` | 1 | **6 only** |
| `direct_usb` | 2 | **4** (compatibility, metadata discarded) or **7** (full) |
| `iio` | — | 4, 5 |

All receivers in one capture must share a transport. Antenna labels for the physical build
(A0 starboard / A1 port / B0 bow / B1 stern) are in the
[rover README](../../../data_collection/rover/rover_v3.1/README.md).

## 4. Container layout

```
<name>.zarr/
├── (attrs) radio_metadata_schema_version = 2      # v7_data.py:66 — the cheapest version probe
├── (attrs) sdr_identity_version = 1               # data_collector.py:670
├── config                       (1,) str
└── receivers/r{0,1}/
    ├── (attrs) identity + direct-USB + firmware provenance + hardware fingerprint  # see 5.5
    ├── signal_matrix            (T, 2, B)  complex64, Blosc zstd, chunks (1, 2, B//2)
    ├── <9 v4 f64 keys>          (T,)       float64,  uncompressed, one chunk
    ├── avg_phase_diff|rssis|gains (T, 2)   float64,  uncompressed
    ├── <10 scalar metadata keys> (T,)      per-key dtype, uncompressed
    └── <7 per-element metadata> (T, 2)     per-key dtype, uncompressed
```

The base layout is byte-identical to [v4](./v4_data_format.md) — v7 reuses `v4rx_f64_keys` and
`v4rx_2xf64_keys` verbatim (`spf/dataset/v7_data.py:9-10`) and appends. That is deliberate: a v4
reader opens a v7 file and sees a valid v4 dataset.

## 5. Recorded fields (exhaustive)

Key lists: `v7rx_f64_keys`, `v7rx_2xf64_keys`, `v7rx_scalar_keys`, `v7rx_2x_keys`
(`spf/dataset/v7_data.py:9-34`). Writers: the v4 fields via
`DroneDataCollectorRaw.write_to_record_matrix` (`spf/data_collector.py:815`), the metadata via
`DroneDataCollectorRawV7.write_to_record_matrix` (`:920`). Values originate in
`_direct_v2_rx_buffer` (`spf/sdrpluto/sdr_controller.py:922-959`).

### 5.1 Inherited v4 fields

Identical in shape, dtype, units and meaning to [v4 §5](./v4_data_format.md#5-recorded-fields-exhaustive):
`signal_matrix`, `system_timestamp`, `gps_timestamp`, `gps_lat`, `gps_long`, `heading`,
`rx_theta_in_pis`, `rx_spacing`, `rx_lo`, `rx_bandwidth`, `avg_phase_diff`, `rssis`, `gains`.

Two differences in **provenance**, not schema — under protocol v2 these are no longer IIO reads:

| Field | v4 source | v7 source |
|---|---|---|
| `gains` | host IIO `gains()` after the buffer | `metadata.gain_db_end` — the firmware's end-of-buffer snapshot, in dB (`sdr_controller.py:937`) |
| `rssis` | host IIO `rssis()` after the buffer | `metadata.rssi_db_end` (`:936`) |

So on a v7 capture `gains` and `rssis` are duplicates of `gain_db_end` / `rssi_db_end` cast to
float64. Prefer the explicit `*_db_end` names in new code.

### 5.2 Per-record metadata — `(T,)`

| Field | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|
| `gain_metadata_valid` | bool | — | Derived, not a raw bit: both `START_VALID` and `END_VALID` set **and** `DUMMY_GAINS` clear (`direct_usb_protocol.py:395-400`) | **The first filter on any gain analysis.** False ⇒ ignore the gain columns for that record. Because it already folds in three flags, checking it is usually enough; go to `gain_metadata_flags` only to find out *why* |
| `rssi_metadata_valid` | bool | — | Same, for the RSSI snapshots | v7-only; v6 had no RSSI at all |
| `gain_metadata_flags` | uint32 | bitfield | Raw `MetadataFlags` for the buffer | The diagnostic field. Bit table below |
| `stream_id` | uint64 | — | Firmware stream identity | Changes ⇒ the stream restarted; sequence numbers are not comparable across values |
| `buffer_sequence` | uint64 | count | Firmware's buffer counter | **Gap detection.** Non-consecutive ⇒ dropped buffers between records |
| `sample_sequence` | uint64 | samples | Sequence number of the buffer's first sample | Sample-exact continuity; only meaningful with `SAMPLE_SEQUENCE_VALID` set |
| `gain_start_read_duration_ns` | uint32 | ns | Cost of the start-of-buffer gain read | Firmware-side health/latency, **not** signal timing |
| `gain_end_read_duration_ns` | uint32 | ns | Cost of the end-of-buffer gain read | As above |
| `rssi_start_read_duration_ns` | uint32 | ns | Cost of the start-of-buffer RSSI read | v7-only |
| `rssi_end_read_duration_ns` | uint32 | ns | Cost of the end-of-buffer RSSI read | v7-only |

### 5.3 Per-element metadata — `(T, 2)`, element axis = RX1, RX2

| Field | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|
| `gain_db_start` | float32 | dB | Hardware RX gain at the buffer's **first** sample | With `gain_db_end`, bounds the gain that applied to these samples. NaN under protocol v1 |
| `gain_db_end` | float32 | dB | Gain at the **last** sample | The value mirrored into `gains` |
| `rssi_db_start` | float32 | dB | RSSI at the first sample | v7-only. Pair with `gain_db_*` to reason about absolute input level |
| `rssi_db_end` | float32 | dB | RSSI at the last sample | Mirrored into `rssis` |
| `gain_endpoints_equal` | bool | — | Start and end gain are identical | **True ⇒ this record has one unambiguous gain.** The clean-subset selector for level-dependent analysis |
| `first_gain_change_sample` | int32 | sample index | Where the gain first moved inside the buffer; **−1 = never moved** | `0xFFFFFFFF` is normalized to −1 (`sdr_controller.py:924-932`). Lets a record be truncated at the change instead of discarded |
| `iq_power_dbfs` | float32 | dBFS | Mean post-gain complex-sample power vs 12-bit full scale | Computed **on the host from the IQ** (`sdr_controller.py:121`), so it is independent of firmware reporting — the cross-check that catches a lying radio. `full_scale_power = 2·2048²`; `-inf` for an all-zero buffer |

### 5.4 `gain_metadata_flags` bits

From `MetadataFlags` (`spf/sdrpluto/direct_usb_protocol.py:52-68`):

| Bit | Name | Reading |
|---|---|---|
| 0 | `START_VALID` | start-of-buffer gain snapshot is valid |
| 1 | `END_VALID` | end-of-buffer gain snapshot is valid |
| 2 / 3 | `RX1_ENDPOINT_CHANGED` / `RX2_ENDPOINT_CHANGED` | that element's endpoints differ |
| 4 | `SAMPLE_SEQUENCE_VALID` | `sample_sequence` is meaningful |
| 5 | `FPGA_EVENTS_VALID` | FPGA gain-event capture is meaningful |
| 6 / 7 | `RX1_CHANGED_IN_BUFFER` / `RX2_CHANGED_IN_BUFFER` | gain moved mid-buffer for that element |
| 8 / 9 | `RX1_LOCKED_AT_END` / `RX2_LOCKED_AT_END` | AGC settled by the buffer's end |
| 10 | `GAIN_FULL_TABLE_MODE` | AD9361 full gain-table mode |
| 11 | `DEVICE_IIO_OVERFLOW` | **device-side overflow — sample loss** |
| 12 | `GAIN_READ_FAILED` | gain read failed; snapshots unusable |
| 13 | `FPGA_EVENT_OVERFLOW` | more gain events than the FPGA buffer holds |
| 14 | `DUMMY_GAINS` | **synthetic gains — the firmware is not really measuring.** Protocol v2 rejects the frame at decode (`direct_usb_protocol.py:949-950`, "protocol v2 does not accept dummy gains"), so it cannot reach a v7 store. (The `require-gain-metadata` config guard at `sdr_controller.py:880-881` is the *protocol-v1/v6* path.) |
| 15 / 16 | `RSSI_START_VALID` / `RSSI_END_VALID` | per-endpoint RSSI validity |
| 17 | `RSSI_READ_FAILED` | RSSI read failed |
| 18 | `GAIN_DB_VALUES` | gain endpoints are in **dB** (v7) rather than table indices (v6) |

Bits outside `KNOWN_FLAGS` (`:102`) indicate a firmware newer than this checkout.

### 5.5 Store-level arrays and attributes

Zarr **attributes** are part of the format and carry the provenance that makes a v7 capture
auditable. They are written by `DataCollector._record_receiver_identities`
(`spf/data_collector.py:634-690`) via `_identity_zarr_attrs` (`:48-97`) and
`_capture_firmware_provenance` (`:100-167`). Because that lives in the **base** collector, v4/v5/v6
captures carry the identity subset too — only the v7 *gates* below are version-specific.

| Where | Attribute | Meaning |
|---|---|---|
| store | `radio_metadata_schema_version` = 2 | Set only by v7 (`v7_data.py:66`). Fastest version probe |
| store | `sdr_identity_version` = 1 | `SDR_IDENTITY_VERSION` (`data_collector.py:45,670`) |
| store | `config` *(array, `(1,)` str)* | `yaml.dump` of the **normalized** config — includes the materialized transport keys and the firmware pin |
| receiver | `sdr_identity_version`, `sdr_family`, `iio_uri_at_capture`, `rx_transport` | Always present. `rx_transport` is how you tell a direct-USB capture from an IIO one **without** parsing the config |
| receiver | `sdr_serial`, `usb_vendor_id`, `usb_product_id`, `usb_bus_at_capture`, `usb_address_at_capture`, `usb_port_path` | Present when known (omitted, not null, when unavailable). `usb_port_path` is the stable physical path — the one that survives re-enumeration |
| receiver *(direct-USB only)* | `direct_usb_serial`, `direct_usb_bus`, `direct_usb_port_path`, `direct_usb_interface`, `direct_usb_bulk_in_endpoint`, `direct_usb_bulk_out_endpoint` | Wire-level attachment. The `direct_usb_*` duplicates exist to preserve the original v6 attribute names |
| receiver *(direct-USB only)* | `gain_metadata_protocol_version`, `direct_usb_protocol_min`, `direct_usb_protocol_max`, `direct_usb_supported_features`, `gain_metadata_capability_flags` | Negotiated protocol window and feature/capability bitmaps (`MetadataFeatures`, `CapabilityFlags`, `direct_usb_protocol.py:74-87`). **`gain_metadata_protocol_version` must be 2 for v7** |
| receiver *(direct-USB only)* | `firmware_release_tag`, `firmware_image_sha256`, `firmware_git_sha`, `firmware_gadget_git_sha`, `firmware_boot_mode`, `firmware_verified`, `firmware_ready_manifest_version` | Firmware provenance copied from the config **and cross-checked** against the boot-time ready manifest (`/run/spf/direct_usb_ready.json`). `firmware_verified` is True only if every pinned field matched *and* the manifest itself verified |
| receiver *(direct-USB only)* | `hardware_fingerprint_v1`, `hardware_fingerprint_schema_version` | The passive post-firmware hardware fingerprint, attached only when the manifest's serial/bus/address/port-path all match this receiver **and** `firmware_verified` is True |

**Two hard gates make these load-bearing under v7** (`data_collector.py:679-690`): the collector
raises before capturing if `firmware_verified is not True` or if `hardware_fingerprint_v1` is not a
dict. So a v7 capture that exists on disk is, by construction, from boot-verified pinned firmware
on fingerprint-matched hardware. That is a stronger provenance claim than any other version makes,
and it is the reason to prefer v7 data for anything gain-related.

## 6. Post-processed fields

No field in §5 is derived — all are written at capture time.

Downstream processing is **identical to v4**, because that is the schema `v5spfdataset` sees:
the v4→v5 on-the-fly upgrade (`spf_dataset.py:1304`) synthesizes `rx_heading_in_pis` from
`heading/360*2` and zero-fills the wall-array position keys, then all v5 derived fields and the
segmentation precompute cache apply unchanged. See
[v4 §6](./v4_data_format.md#6-post-processed-fields) and
[`precompute_cache_format.md`](./precompute_cache_format.md).

**The v7 metadata arrays are not read by any training path** — the loader takes the v4 subset.
They exist today for capture-quality analysis (bench validation, soak tests, the
`dual_rx_gain_frequency` calibration work). Treat "does training use it?" as *no* until a
consumer is added.

## 7. Reading it

Training/loader path — v7 is opened as a v4 superset:

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

with v5spfdataset_manager(prefix, nthetas=65, precompute_cache=..., v4=True, paired=True) as ds:
    sample = ds[0]
```

Metadata path — read the arrays directly, since no loader surfaces them:

```python
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
import numpy as np

z = zarr_open_from_lmdb_store("/path/to/rover.zarr", mode="r")
r0 = z["receivers/r0"]
clean = np.asarray(r0["gain_metadata_valid"]) & np.all(r0["gain_endpoints_equal"], axis=1)
print("records with one unambiguous gain:", clean.sum(), "/", clean.size)
print("median gain (dB):", np.nanmedian(np.asarray(r0["gain_db_end"])[clean], axis=0))
```

Always open read-only. Raw captures are immutable (root `CLAUDE.md`).

## 8. Known issues and traps

- **`gains` / `rssis` are aliases** of `gain_db_end` / `rssi_db_end` on v7, not independent
  measurements. Do not treat agreement between them as corroboration.
- **A valid-looking gain column can still be meaningless** if `gain_metadata_valid` is False or
  `GAIN_READ_FAILED` is set. Filter on the validity flags first — they exist because the
  failure is silent otherwise.
- **`first_gain_change_sample == -1` means "no change", not "unknown"**; unknown is expressed
  through the validity flags. Reading −1 as a sample index poisons any arithmetic.
- **Rover NaN of 46–70 % is normal**, not corruption — the emitter is a bursty beacon. Do not
  quarantine on NaN alone (`ROVER_RUNBOOK.md` §data-quality; `docs/learnings.md` L2).
- **`heading` is degrees; `rx_heading_in_pis` is `deg/180`.** Two different scalings live one
  field apart, and the realtime path has a separate documented `/720` heading bug (KI#55–#60).
- **`rx_spacing` is configured, not measured** — the 47/43/35 mm data-surgery story is in
  `ROVER_RUNBOOK.md`. KI#19: the in-place fixer overwrites with no backup.
- **The metadata arrays are uncompressed with a single chunk spanning `T`**
  (`v7_data.py:69-82`), so each per-record write rewrites the whole column. Fine at 2 Hz;
  do not assume it scales.
- **Firmware/config coupling is strict.** A rover whose RAM-booted firmware does not match the
  config's `image-sha256` fails the boot gate before collecting. Do not hand-edit the pin to
  make a capture start.

## 9. Verification

**Use the committed validator — v7 is the one format with a real conformance checker:**

```bash
python3 -m spf.scripts.validate_direct_usb_v7_zarr /path/to/rover.zarr --expected-frames 3000 --expected-receivers 2
```

`spf/scripts/validate_direct_usb_v7_zarr.py` raises on the first violation and otherwise emits a
JSON summary (per-receiver serial, USB port path, gain/RSSI min-max in dB, unique stream count,
endpoint-changed frame count, median frame rate, interval p99). What it enforces:

| Check | Detail |
|---|---|
| Signal shape/dtype | exactly `(expected_frames, 2, 524288)` complex64 — **the 524288 buffer size is hardcoded in the validator**, so a capture with a different `buffer-size` fails here even though it is schema-valid |
| IQ sanity | every frame finite; neither channel all-zero |
| **Legacy-alias contract** | `gains == gain_db_end` and `rssis == rssi_db_end` — the aliasing in §5.1 is an enforced contract, not an accident |
| Unsafe flags | rejects any frame with `DUMMY_GAINS`, `GAIN_READ_FAILED`, `RSSI_READ_FAILED`, `DEVICE_IIO_OVERFLOW`, `FPGA_EVENT_OVERFLOW` |
| Flag consistency | `gain_endpoints_equal` must agree with the raw endpoint-changed flags |
| Continuity | `buffer_sequence` strictly +1 per frame; `sample_sequence` strictly +524288; `system_timestamp` strictly increasing |
| Provenance | schema version 2, `sdr_identity_version` 1, per-receiver `rx_transport == direct_usb`, negotiated protocol v2, `firmware_verified`, and a present/valid `hardware_fingerprint_v1` |

Defaults are `--expected-frames 100 --expected-receivers 2`, i.e. tuned for the boot preflight —
pass the real values for a production capture, and `--expected-receivers 1` for rover 2.

Ad-hoc key-list check, if you only want the schema:

```bash
python3 -c "from spf.dataset.v7_data import v7rx_keys; print(sorted(v7rx_keys()))"
```

Conformance check against a file:

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.dataset.v7_data import v7rx_keys
import sys, yaml, numpy as np
z = zarr_open_from_lmdb_store(sys.argv[1], mode='r')
cfg = yaml.safe_load(z['config'][0])
assert cfg['data-version'] == 7, cfg.get('data-version')
assert z.attrs.get('radio_metadata_schema_version') == 2
have, want = set(z['receivers/r0'].keys()), set(v7rx_keys())
print('missing:', sorted(want - have)); print('extra:', sorted(have - want))
r0 = z['receivers/r0']
print('gain_metadata_valid frac:', float(np.mean(r0['gain_metadata_valid'])))
print('rssi_metadata_valid frac:', float(np.mean(r0['rssi_metadata_valid'])))
print('endpoints_equal frac:', float(np.mean(r0['gain_endpoints_equal'])))
" /path/to/rover.zarr
```

Config-level validation is available without touching a capture:

```bash
python3 -m spf.scripts.rover_capture_config --rover-id 3 --format null
```

which runs `normalize_capture_config` + `validate_production_config` and fails loudly on any
v7 contract violation. Related checkers: `spf/sdrpluto/direct_usb_smoke.py` (single-frame
metadata dump), `direct_usb_soak.py` (sustained validity/endpoint-change rates),
`spf/scripts/pluto_ready_manifest.py` (boot-time firmware + mapping attestation).

## 10. Changelog

- **2026-07-29** — created. Field tables from `spf/dataset/v7_data.py`; semantics traced to
  `_direct_v2_rx_buffer` and `MetadataFlags`; `gains`/`rssis` aliasing confirmed at
  `sdr_controller.py:936-937`.
- **2026-07-29 (review pass)** — corrected the `DUMMY_GAINS` enforcement point (protocol-level
  reject at `direct_usb_protocol.py:949-950`, not the v1 config guard); documented
  `gain_metadata_valid` as a derived property; added §5.5 store/receiver **attributes** incl. the
  two v7 hard gates; replaced the hand-rolled §9 with the committed
  `spf/scripts/validate_direct_usb_v7_zarr.py`; fixed line citations (`v7_data.py:65`→66,
  `validate_transport_schema` `:82`→`:81`).
