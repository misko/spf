# v2 capture data format

| | |
|---|---|
| **Status** | **legacy** — not written by any current path; kept for historical captures and the `make_v2_splits` tooling |
| **Container** | flat `np.memmap`, float32 — **one file, no zarr, no compression, no self-description** |
| **Written by** | `GrblDataCollector` (`spf/data_collector.py:992`), via `spf/grbl_radio_collection.py:164` (`data-version == 2`) |
| **Read by** | index helpers in `spf/dataset/wall_array_v2_idxs.py`; **not** `v5spfdataset` |
| **Superseded by** | [v5](./v5_data_format.md) |
| **Defining module** | `spf/dataset/wall_array_v2_idxs.py` (`v2_column_names`) |

Section order and field-table rules come from [`contracts.md`](./contracts.md).

---

## 1. Motivation

v2 stores **the conclusion, not the evidence**: each record is a fixed-width row ending in a
65-bin beamformer response, with the IQ discarded. That was cheap — a full capture is a few MB
instead of tens of GB — and it works if you already know the right beamformer and the right
`nthetas` forever.

That assumption failed. Every subsequent advance (segmentation versions 3.2 → 3.7, per-window
statistics, learned models over raw phase) needed the samples that v2 threw away, and there is no
path from a v2 file back to them. [v5](./v5_data_format.md) exists precisely to keep raw IQ and
push interpretation downstream.

**There is no v3.** The number was skipped between v2 and v4. The `3.x` versions in this project
are *segmentation* versions (`SEGMENTATION_VERSION = 3.7`, `spf/utils.py:14`) and the `3p4`…`3p7`
in precompute-cache paths refers to those, not to a data format. v1 is documented separately at
[v1](./v1_data_format.md).

## 2. Collection types that produce it

| | |
|---|---|
| Platform | **2D wall array** — GRBL gantry, both emitter and receiver on motors |
| Orchestrator | `spf/grbl_radio_collection.py` |
| Position truth | motor coordinates, mm |
| Transport | `iio` (USB-IIO to PlutoSDR) |
| Status | Nothing writes v2 today. The GRBL path selects v5 |

## 3. Example configs

The distinguishing field is simply `data-version: 2`; the receiver blocks look like v5's. What
matters extra:

```yaml
data-version: 2
n-thetas: 65                     # DEFINES THE ROW WIDTH — see the trap in §8
n-records-per-receiver: 10000
```

`n-thetas` is a schema parameter in v2, not a derived-analysis knob: the row width is
`13 + n_thetas`. Two v2 files captured with different `n-thetas` are **different formats** with
no marker to tell them apart. In v5 the same field affects nothing on disk.

## 4. Container layout

A single memmap created in `GrblDataCollector.setup_record_matrix`
(`spf/data_collector.py:1000-1011`):

```python
np.memmap(
    filename,
    dtype="float32",
    mode="w+",
    shape=(
        2,                                        # receivers — HARDCODED (TODO in source)
        n_records_per_receiver,                   # T
        len(v2_column_names(nthetas=n_thetas)),   # 13 + n_thetas columns
    ),
)
```

Properties: no compression, no chunking, no embedded config, no header, no dtype record. **The
file is uninterpretable without knowing `n_thetas` and the column order** — which live only in
`v2_column_names` and in the capture's `.log`/`.yaml` sidecars, if those survived.

The receiver axis is hardcoded to 2 with a `# TODO should be nreceivers` in the source
(`:1007`).

## 5. Recorded fields (exhaustive)

Columns come from `v2_column_names(nthetas)` (`spf/dataset/wall_array_v2_idxs.py:6-22`); values
from `prepare_record_entry_v2` (`spf/data_collector.py:297-311`). Every value is float32 — there
is no per-field dtype.

Row index = column index. All shapes are scalar per `(receiver, record)`.

| # | Field | Units | Written by | Meaning | Use / caveat |
|---|---|---|---|---|---|
| 0 | `timestamp` | s (host epoch) | `data_collector.py:301` | Host clock at snapshot | Only clock present |
| 1 | `tx_pos_x_mm` | mm | `:302` | Emitter x, gantry frame | |
| 2 | `tx_pos_y_mm` | mm | `:302` | Emitter y | |
| 3 | `rx_pos_x_mm` | mm | `:303` | Receiver-array centre x | |
| 4 | `rx_pos_y_mm` | mm | `:303` | Receiver-array centre y | With tx pos, gives the ground-truth bearing |
| 5 | `rx_theta` | **radians** | `:304` | Array mount orientation, written as `rx_theta_in_pis * π` | **The only place in the project storing orientation in radians.** Every zarr format stores multiples of π under `rx_theta_in_pis` |
| 6 | `rx_spacing_m` | m | `:305` | Element separation | Configured, not measured |
| 7 | `avg_phase_diff_1` | radians, wrapped | `:306` | Circular mean of RX1−RX2 per-sample phase difference | |
| 8 | `avg_phase_diff_2` | radians, wrapped | `:306` | Trimmed circular mean | **Equal to column 7** — this path calls `get_avg_phase(signal_matrix)` with default `trim=0.0`, and `circular_mean` returns `(r, r)` when `trim == 0` (`spf/rf.py:272-283`) |
| 9 | `rssi0` | dB (Pluto scale) | `:307` | Element-0 RSSI, host IIO read | |
| 10 | `rssi1` | dB | `:307` | Element-1 RSSI | See the `v2_rssi_idxs` bug in §8 |
| 11 | `gain0` | dB | `:308` | Element-0 RX gain | Post-hoc host read; same attribution weakness as v4 |
| 12 | `gain1` | dB | `:308` | Element-1 RX gain | |
| 13 … 13+n−1 | `beamformer_angle_<θ>` | linear power (arb.) | `:309` | `beam_sds` — beamformer response at each of `n_thetas` angles from `np.linspace(-π, π, nthetas)` | **The whole point of v2 and its whole limitation.** Fixed at capture time; cannot be recomputed differently |

**Absent, versus v5:** `signal_matrix` (the decisive loss), `rx_lo`, `rx_bandwidth`,
`rx_heading_in_pis`, and any embedded config. Without `rx_lo` you cannot compute `d/λ` from the
file alone, so even the recorded beamformer cannot be re-mapped to phase.

Column-index helpers, all `@cache`d: `v2_time_idx`, `v2_rx_pos_idxs`, `v2_tx_pos_idxs`,
`v2_rssi_idxs`, `v2_gain_idxs`, `v2_avg_phase_diff_idxs`, `v2_rx_theta_idx`,
`v2_beamformer_start_idx`.

## 6. Post-processed fields

Nothing in the modern derived stack applies. `v5spfdataset` cannot open v2, so there are no
loader-derived training fields and **no precompute cache** — the segmentation pipeline needs IQ
that v2 does not have.

Post-processing that does exist:

| Artifact | Producer | Note |
|---|---|---|
| Train/val split lists | `spf/scripts/make_v2_splits.py` | Despite the name, its defaults point at the modern `nosig` split files (`/mnt/md2/splits/apr17_*`), not at v2 captures. ⚠ the "v2" in the filename refers to the split-generation scheme, not the data format |

Ground-truth bearing can still be derived offline from columns 1–5 by the same geometry v5 uses;
nothing in the repo does this for v2 today.

## 7. Reading it

There is no loader. Open it by hand, and you must supply `nthetas` yourself:

```python
import numpy as np
from spf.dataset.wall_array_v2_idxs import v2_column_names, v2_beamformer_start_idx

nthetas = 65                                   # MUST match capture time
cols = v2_column_names(nthetas=nthetas)
m = np.memmap(path, dtype="float32", mode="r").reshape(2, -1, len(cols))

r0 = m[0]
print("records:", r0.shape[0])
print("beamformer block:", r0[:, v2_beamformer_start_idx():].shape)
print("theta (radians!):", r0[0, cols.index("rx_theta")])
```

Sanity check that `nthetas` was right: `m.size % (2 * (13 + nthetas)) == 0`. A wrong guess
reshapes without error and silently shears every column — see §8.

## 8. Known issues and traps

- **A wrong `nthetas` corrupts silently.** The file has no header, so `reshape` succeeds with any
  divisor and every field lands in the wrong column. This is the defining hazard of the format.
- **`rx_theta` is in radians here and multiples of π everywhere else.** Copying angle code
  between v2 and v5 without rescaling introduces a factor of π.
- **`avg_phase_diff_2` duplicates `avg_phase_diff_1`** (`trim=0.0`). Averaging the pair averages
  a value with itself.
- **`v2_rssi_idxs()` returns `rssi0` twice** (`spf/dataset/wall_array_v2_idxs.py:39-43`) — it
  should return `rssi0, rssi1`, as the adjacent `v2_gain_idxs` correctly does. Any caller reads
  element 0's RSSI for both elements. ⚠ no current caller found in the repo, so this is latent;
  do not build on the helper without fixing it.
- **Receiver count is hardcoded to 2** (`data_collector.py:1006`). A 1-receiver v2 capture would
  still allocate two planes, the second all zeros.
- **The beamformer is frozen.** No re-segmentation, no different `nthetas`, no phase-domain
  modelling. If a question needs the IQ, a v2 capture cannot answer it — recapture is the only
  option.
- **No embedded config.** Spacing, LO and routine survive only in sidecar files and the filename.
  Do not trust a filename for `rx_spacing` (the rover spacing-mislabel story in
  `ROVER_RUNBOOK.md` is the cautionary case).

## 9. Verification

There is no schema to check — verification is arithmetic on the file size:

```bash
python3 -c "
import numpy as np, sys
from spf.dataset.wall_array_v2_idxs import v2_column_names
path, nthetas = sys.argv[1], int(sys.argv[2])
ncols = len(v2_column_names(nthetas=nthetas))
n = np.memmap(path, dtype='float32', mode='r').size
print('columns:', ncols, '| divides cleanly:', n % (2 * ncols) == 0,
      '| records/receiver:', n // (2 * ncols))
" /path/to/capture 65
```

If it does not divide cleanly, `nthetas` is wrong — try 65, then whatever the capture's sidecar
YAML says. Invariants worth asserting once reshaped: `timestamp` non-decreasing over written
rows; `rx_spacing_m` constant; `rx_theta` in [−π, π].

## 10. Changelog

- **2026-07-29** — created. Columns from `v2_column_names`; values traced to
  `prepare_record_entry_v2`; `trim=0.0` duplication confirmed in `spf/rf.py:272-283`;
  `v2_rssi_idxs` duplicate-index bug recorded as latent.
