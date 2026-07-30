# v4 capture data format

| | |
|---|---|
| **Status** | **legacy-readable** — no longer written, but holds essentially the whole historical rover corpus |
| **Container** | Zarr hierarchy over an LMDB store |
| **Written by** | `DroneDataCollectorRaw` (`spf/data_collector.py:785`), via `spf/mavlink_radio_collection.py:313` |
| **Read by** | `v5spfdataset` with `v4=True` — upgraded to v5 shape in memory on open |
| **Superseded by** | [v6](./v6_data_format.md) → [v7](./v7_data_format.md) |
| **Sibling** | [v5](./v5_data_format.md) — same era, wall array instead of rover |
| **Defining module** | `spf/dataset/v4_data.py` |

Section order and field-table rules come from [`contracts.md`](./contracts.md).

---

## 1. Motivation

v4 is **v5's idea applied to a moving vehicle**. Both keep raw IQ and embed the capture config;
they differ only in how position truth is expressed:

| | v5 (wall array) | v4 (rover) |
|---|---|---|
| Position | `tx_pos_*_mm` / `rx_pos_*_mm` — motor coordinates, mm | `gps_lat` / `gps_long` — GPS degrees |
| Orientation | fixed mount, no heading | `heading` from the compass, degrees |
| Emitter position | known, on the same gantry | **not recorded** — the emitter is a separate craft |
| Time | `system_timestamp` only | `system_timestamp` **and** `gps_timestamp` |

The last row is the important one. A rover capture is not self-labelling: it knows where the
*receiver* was, not where the *emitter* was. Producing a supervised label requires joining
against the emitter's own capture on GPS time — which is what `v4_tx_rx_to_v5.py` exists to do.
That asymmetry, not the field list, is what makes rover data harder than wall data.

## 2. Collection types that produce it

| | |
|---|---|
| Platform | **Rover v3.1** — RPi + ArduPilot, 1–2 PlutoPlus SDRs |
| Orchestrator | `spf/mavlink_radio_collection.py` |
| Position truth | GPS + compass heading (noisy, unlike gantry coordinates) |
| Transport | `iio` (USB-IIO) historically; also legal with `direct_usb` protocol v2 as a **compatibility mode** that discards the metadata (`spf/capture_schema.py:113`) |
| Routines | `bounce`, `circle`, `center`, `calibrate` |
| Corpus | the recorded rover captures under `/mnt/md2/rovers` (rover1/2/3), 2024-06 → 2025-04 — counts and footprints in [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md) |

Nothing writes v4 in production today — the boot path selects `*_production_v7.yaml`. v4 remains
reachable by hand-launching an older config, and it remains the format of the recorded corpus.

## 3. Example configs

Committed v4 rover configs (still present, no longer the boot default):

- `data_collection/rover/rover_v3.1/capture_configs/rover_receiver_config_pi_3mhz_35mm.yaml` (rover 1)
- `.../rover_receiver_config_pi_3mhz_43mm.yaml` (rover 3)
- `.../rover_single_receiver_config_pi_3mhz.yaml` (rover 2, one radio)
- `.../rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2_v4.yaml` (direct-USB compatibility mode)

```yaml
data-version: 4

receivers:
  - receiver-port: 2             # USB2 = Radio A
    theta-in-pis: 0
    antenna-spacing-m: 0.043
    nelements: 2
    buffer-size: 524288
    f-carrier: 5.766e+9
    bandwidth: 3.0e+6
  - receiver-port: 1             # USB1 = Radio B
    theta-in-pis: 0.5

n-records-per-receiver: 600000   # see the trap in §8
seconds-per-sample: 0.5
routine: null                    # supplied on the command line in this generation
drone-uri: serial
```

Note that `spf/mavlink_radio_collection.py` normalizes and validates **every** config it loads
(`:191`, `:194`), so a v4 config is still subject to `validate_transport_schema` — a v4 config
requesting direct-USB protocol v1 is rejected outright (that combination is v6-only).

Two things a reader will trip on:

- **`n-records-per-receiver: 600000` in these files is inert history.** The v4-era launcher always
  overrode it with `-n 3000`. A v4 config run *without* `-n` preallocates 600,000 timesteps
  (≈83 h at 0.5 s/sample) and never completes a capture. The v7 configs moved the real value
  into the YAML.
- **`routine: null`** — the v4-era launcher passed `-r bounce`. The v7 generation moved this into
  the config too.

## 4. Container layout

```
<name>.zarr/
├── config                       (1,) str — yaml.dump of the capture config
└── receivers/r{0,1}/
    ├── signal_matrix            (T, 2, B)  complex64, Blosc zstd clevel=1 BITSHUFFLE, chunks (1, 2, B//2)
    ├── system_timestamp         (T,)   float64, uncompressed, one chunk
    ├── gps_timestamp            (T,)   float64
    ├── gps_lat                  (T,)   float64
    ├── gps_long                 (T,)   float64
    ├── heading                  (T,)   float64
    ├── rx_theta_in_pis          (T,)   float64
    ├── rx_spacing               (T,)   float64
    ├── rx_lo                    (T,)   float64
    ├── rx_bandwidth             (T,)   float64
    ├── avg_phase_diff           (T, 2) float64
    ├── rssis                    (T, 2) float64
    └── gains                    (T, 2) float64
```

Created by `v4rx_new_dataset` (`spf/dataset/v4_data.py:25`) → `zarr_new_dataset`
(`spf/scripts/zarr_utils.py:161`). Same preallocation, `zarr_shrink`-on-close, and
compress-only-the-IQ behaviour as [v5 §4](./v5_data_format.md#4-container-layout).

## 5. Recorded fields (exhaustive)

Key lists: `v4rx_f64_keys`, `v4rx_2xf64_keys` (`spf/dataset/v4_data.py:3-18`). All writes in
`DroneDataCollectorRaw.write_to_record_matrix` (`spf/data_collector.py:815-840`).

### Raw signal

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `signal_matrix` | `(T, 2, B)` | complex64 | ADC counts (12-bit scale) | `data_collector.py:837` | Both elements' IQ for one snapshot | The only irreplaceable field. Element axis = RX1, RX2 |

### Platform state (GPS / compass, from the MAVLink position controller)

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `gps_lat` | `(T,)` | float64 | degrees | `data_collector.py:825` | Receiver latitude | Rover truth. Note the write order: `gps_long` is assigned from `[0]`, `gps_lat` from `[1]` |
| `gps_long` | `(T,)` | float64 | degrees | `:824` | Receiver longitude | |
| `gps_timestamp` | `(T,)` | float64 | s (GPS epoch) | `:826` | GPS-derived time | Independent of the host clock — this is the field the v4→v5 emitter/receiver join uses |
| `heading` | `(T,)` | float64 | **degrees** | `:819` | Craft heading from the compass | The only heading in v4. **Degrees, not multiples of π** |
| `system_timestamp` | `(T,)` | float64 | s (host epoch) | dataclass | Host clock at buffer receipt | Pi clocks are GPS-synced at boot and between captures; drift within a capture is possible |

### Array/radio configuration (constant per capture)

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `rx_theta_in_pis` | `(T,)` | float64 | multiples of π | dataclass, from `theta-in-pis` | Array mount orientation | `0` = array A (athwartships), `0.5` = array B (fore-aft) |
| `rx_spacing` | `(T,)` | float64 | m | dataclass, from `antenna-spacing-m` | Element separation | **Configured, not measured.** Mislabeled-spacing captures exist; see §8 |
| `rx_lo` | `(T,)` | float64 | Hz | dataclass | RX local oscillator | With `rx_spacing` gives `d/λ` |
| `rx_bandwidth` | `(T,)` | float64 | Hz | dataclass | RX filter bandwidth | |

### Per-record radio state

| Field | Shape | Dtype | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|---|
| `avg_phase_diff` | `(T, 2)` | float64 | radians, wrapped | `data_collector.py:458` | Circular mean of per-sample RX1−RX2 phase difference | **Both entries identical** — `get_avg_phase_fast2` duplicates one mean (`spf/rf.py:768-780`) |
| `rssis` | `(T, 2)` | float64 | dB (Pluto scale) | dataclass | Per-element RSSI, host IIO read **after** the buffer | The core v4 weakness: one post-hoc reading for a buffer during which AGC moved. This is what v7 fixed |
| `gains` | `(T, 2)` | float64 | dB | dataclass | Per-element RX gain, host IIO read after the buffer | Same caveat. **Read `docs/learnings.md` on `F:gain` before using it** |

### Store-level arrays and attributes

| Field | Shape | Dtype | Units | Meaning | Use / caveat |
|---|---|---|---|---|---|
| `config` | `(1,)` | str | — | `yaml.dump` of the capture config | Authoritative for spacing/LO/routine. Trust over the filename |

Zarr **attributes**: written by `DataCollector._record_receiver_identities`
(`spf/data_collector.py:634-690`, reached from `radios_to_online()` at
`mavlink_radio_collection.py:338`) — `sdr_identity_version` on the store, and per receiver the
identity block (`sdr_family`, `iio_uri_at_capture`, `rx_transport`, `sdr_serial`, `usb_*`). A v4
capture taken over `iio` has **no** firmware-provenance or fingerprint attributes; a v4 capture
taken in direct-USB compatibility mode does carry them. Full inventory:
[v7 §5.5](./v7_data_format.md#55-store-level-arrays-and-attributes). Most of the recorded v4 corpus
predates this block entirely — a missing attribute means "unknown", not a hardware claim.

**Not present in v4** (and the reason the loader has an upgrade step): `rx_heading_in_pis`,
`tx_pos_x_mm`, `tx_pos_y_mm`, `rx_pos_x_mm`, `rx_pos_y_mm`.

## 6. Post-processed fields

### 6.1 The v4→v5 in-memory upgrade

`v5spfdataset.v4_to_v5()` (`spf/dataset/spf_dataset.py:1304-1320`) runs on open when `v4=True`.
It wraps each read-only receiver group in a `ZarrWrapper` (`:397`) so synthetic keys can be
added without touching the file, then:

| Synthesized key | How | Watch out |
|---|---|---|
| `rx_heading_in_pis` | `(heading / 360) * 2` — i.e. `deg/180` | Must match `v4_tx_rx_to_v5.py:175`; the comment in the code exists because these two sites drifted once |
| every other missing `v5rx_f64_keys` entry | **zero filler** (`system_timestamp * 0`) | This is how `tx_pos_*_mm` and `rx_pos_*_mm` become zeros. **A zero position is a filler, not a measurement** |

Consequence: a v4 dataset loaded this way exposes the full v5 field set, and geometry-derived
labels built from the position keys are meaningless unless the positions were supplied by the
offline merge below. Nothing on disk changes.

### 6.2 Offline v4→v5 merge (the real labelling path)

`spf/scripts/v4_tx_rx_to_v5.py` — `merge_v4rx_v4tx_into_v5` (`:213`) takes the **receiver's**
v4 capture and the **emitter's** own v4 capture, smooths and aligns both GPS/time series
(`smooth_out_timestamps_and_gps`), selects mutually valid indices, and writes a **new v5 zarr**
with real `tx_pos_*_mm` / `rx_pos_*_mm`. Properties:

- Output is a new file — the inputs are opened `mode="r"`.
- `min_timesteps=500` rejects too-short overlaps.
- `skip_signal_matrix=True` produces the `nosig` variant in the same pass.
- `rx_heading_in_pis` is derived here too, from `heading/360*2` (`:175`).

### 6.3 Everything else

Loader-derived training fields and the segmentation precompute cache behave exactly as for v5 —
see [v5 §6](./v5_data_format.md#6-post-processed-fields) and
[`precompute_cache_format.md`](./precompute_cache_format.md). One v4-specific branch:
`if not self.temp_file and not self.v4:` (`spf_dataset.py:1071`) gates some cached-key
computation, so v4 and v5 do not follow identical loader paths.

## 7. Reading it

```python
from spf.dataset.spf_dataset import v5spfdataset_manager

with v5spfdataset_manager(
    prefix,                                   # no .zarr suffix
    nthetas=65,
    precompute_cache="/mnt/md2/cache/precompute_cache_3p7",
    v4=True,                                  # REQUIRED — without it the missing keys raise
    paired=True,
    ignore_qc=True,
) as ds:
    sample = ds[0]
```

Forgetting `v4=True` is the usual first failure: the v5 key lookup hits a key the file does not
have. Remember that any position field you read back on a raw v4 file is zero filler (§6.1).

## 8. Known issues and traps

- **Zero positions are filler.** `tx_pos_*_mm` / `rx_pos_*_mm` are synthesized zeros on raw v4.
  Any bearing computed from them is meaningless. Use the merged v5 output for labels.
- **`heading` is degrees**, while every `*_in_pis` field is multiples of π. The realtime path has
  a separate `/720` heading bug (KI#55–#60) — do not copy heading arithmetic from it.
- **`gains`/`rssis` are post-hoc host reads** and cannot be attributed to the samples in the
  buffer. This is the motivating defect for [v7](./v7_data_format.md). Treat v4 level data as
  coarse at best.
- **`n-records-per-receiver: 600000`** in the committed v4 configs is inert only because the
  launcher overrode it (§3). Hand-launching without `-n` gives a capture that never finishes,
  never renames off `.tmp`, and never returns the rover home.
- **`rx_spacing` mislabelling is a real, encountered failure** (the 47/43/35 mm story in
  `ROVER_RUNBOOK.md`). The scan flags it as `rx_spacing` ERRORs and `rX_gain`. KI#19: the
  in-place fixer overwrites with **no backup** — copy first, and prefer a corrected copy in a
  new location per root `CLAUDE.md`.
- **Rover NaN 46–70 % is normal** (bursty beacon). Quarantine on `no_signal` and heading-common
  bias, not on NaN (`docs/learnings.md` L2).
- **`gps_long` ← `[0]`, `gps_lat` ← `[1]`** in the writer. Correct as written, but easy to
  invert when reimplementing.
- **Killed captures leave an untrimmed zero tail** and a `.tmp` name.

## 9. Verification

```bash
python3 -c "from spf.dataset.v4_data import v4rx_keys; print(sorted(v4rx_keys()))"
```

```bash
python3 -c "
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.dataset.v4_data import v4rx_keys
import sys, yaml, numpy as np
z = zarr_open_from_lmdb_store(sys.argv[1], mode='r')
print('data-version:', yaml.safe_load(z['config'][0]).get('data-version'))
have, want = set(z['receivers/r0'].keys()), set(v4rx_keys())
print('missing:', sorted(want - have)); print('extra:', sorted(have - want))
ts = np.asarray(z['receivers/r0']['system_timestamp'])
written = int((ts > 0).sum())
print('written rows:', written, 'of', ts.size, '(tail is unwritten preallocation)')
" /path/to/rover.zarr
```

A v7 file passes this check too — v7 is a strict superset. To tell them apart use
`z.attrs.get('radio_metadata_schema_version')` (2 ⇒ v7) or look for `gain_db_start`.

Fleet-scale quality checks: `spf/scripts/dataset_quality_scan.py`
([`../QC_METRICS.md`](../QC_METRICS.md)).

## 10. Changelog

- **2026-07-29 (review pass)** — documented the zarr **attributes**; corrected writer citations
  (`heading` `:820`→`:819`, `gps_long` `:825`→`:824`, `gps_lat` `:826`→`:825`, `gps_timestamp`
  `:827`→`:826`, `ZarrWrapper` `:395`→`:397`).
- **2026-07-29** — created. Field tables from `spf/dataset/v4_data.py`; upgrade/filler behaviour
  traced to `v5spfdataset.v4_to_v5`; the `600000` history confirmed against `drone_run.sh` git
  history.
