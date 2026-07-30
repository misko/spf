# v1 capture data format

| | |
|---|---|
| **Status** | **abandoned** — no writer survives in the repo; documented so the column helpers are not mistaken for live code |
| **Container** | flat `np.memmap`, float32 (same family as [v2](./v2_data_format.md)) |
| **Written by** | ⚠ **no writer exists in this checkout.** No `data-version: 1` config, no collector branch |
| **Read by** | column-index helpers in `spf/dataset/wall_array_v1_idxs.py` only |
| **Superseded by** | [v2](./v2_data_format.md) |
| **Defining module** | `spf/dataset/wall_array_v1_idxs.py` (`v1_column_names`) |

Section order and field-table rules come from [`contracts.md`](./contracts.md). This document is
deliberately short: the format is dead, and the only reason to document it is that
`wall_array_v1_idxs.py` is still importable and looks current.

---

## 1. Motivation

v1 is the original wall-array record: a timestamp, the **emitter** position, an average phase
difference, and a beamformer response. It has **no receiver position and no array orientation**,
which means a v1 record cannot be turned into a bearing label — you know where the transmitter
was but not where the receiver was or which way its array pointed.

[v2](./v2_data_format.md) fixed exactly that by adding `rx_pos_x_mm`, `rx_pos_y_mm`, `rx_theta`,
`rx_spacing_m`, and the per-element RSSI/gain columns. Everything else about the container is the
same, which is why the two helper modules look nearly identical.

## 2. Collection types that produce it

Wall array v1 (`data_collection/2d_wall_array/2d_wall_array_v1/`), historical. **Nothing produces
v1 today** — `spf/grbl_radio_collection.py` dispatches on `data-version` 2 and 5 only
(`:164`, `:170`), and no committed YAML declares version 1.

⚠ unverified: whether any v1 files still exist on the storage arrays. The scanned corpus in
[`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md) contains only v4 and v5 datasets.

## 3. Example configs

None committed. Not reconstructable from this checkout — there is no collector branch to satisfy.

## 4. Container layout

By analogy with v2 (`shape=(2, T, len(v1_column_names(nthetas)))`, float32, no header, no
compression, no embedded config). ⚠ unverified — the writer that would confirm the receiver-axis
and row ordering is gone.

Row width is `5 + n_thetas`.

## 5. Recorded fields (exhaustive)

From `v1_column_names(nthetas)` (`spf/dataset/wall_array_v1_idxs.py:6-14`). All float32.

| # | Field | Units | Meaning | Use / caveat |
|---|---|---|---|---|
| 0 | `timestamp` | s (host epoch) | Host clock at snapshot | |
| 1 | `tx_pos_x_mm` | mm | Emitter x, gantry frame | |
| 2 | `tx_pos_y_mm` | mm | Emitter y | |
| 3 | `avg_phase_diff_1` | radians, wrapped | Circular mean of RX1−RX2 per-sample phase difference | |
| 4 | `avg_phase_diff_2` | radians, wrapped | Trimmed circular mean | Expected to equal column 3 under `trim=0.0`, as in v2 — ⚠ unverified for v1, no writer to check |
| 5 … 5+n−1 | `beamformer_angle_<θ>` | linear power (arb.) | Beamformer response over `np.linspace(-π, π, nthetas)` | Frozen at capture time |

**Absent, versus v2:** `rx_pos_x_mm`, `rx_pos_y_mm`, `rx_theta`, `rx_spacing_m`, `rssi0/1`,
`gain0/1`. **Absent, versus v5:** all of the above plus `signal_matrix`, `rx_lo`,
`rx_bandwidth`, and any embedded config.

Helpers: `v1_time_idx`, `v1_tx_pos_idxs` (both `@cache`d).

## 6. Post-processed fields

None. No loader, no precompute cache, no derived labels — and no way to construct a bearing label
at all, since receiver position and array orientation were never recorded.

## 7. Reading it

Only by hand, and only with `nthetas` known out of band:

```python
import numpy as np
from spf.dataset.wall_array_v1_idxs import v1_column_names

nthetas = 65                                   # must be supplied; nothing in the file says
cols = v1_column_names(nthetas=nthetas)
m = np.memmap(path, dtype="float32", mode="r").reshape(2, -1, len(cols))
```

⚠ the leading `2` is assumed from v2's writer. Verify against file size before trusting it.

## 8. Known issues and traps

- **No receiver position or orientation ⇒ no labels.** This is not a gap to work around; it is
  why the format was replaced.
- **No header, no embedded config**, so a wrong `nthetas` reshapes cleanly and shears every
  column — the same silent-corruption hazard as [v2 §8](./v2_data_format.md#8-known-issues-and-traps).
- **The helper module is live code for a dead format.** `wall_array_v1_idxs.py` imports fine and
  has no deprecation marker. Do not read its existence as evidence that v1 data is supported.

## 9. Verification

```bash
python3 -c "
import numpy as np, sys
from spf.dataset.wall_array_v1_idxs import v1_column_names
path, nthetas = sys.argv[1], int(sys.argv[2])
ncols = len(v1_column_names(nthetas=nthetas))
n = np.memmap(path, dtype='float32', mode='r').size
print('columns:', ncols, '| divides by 2*ncols:', n % (2 * ncols) == 0,
      '| divides by ncols:', n % ncols == 0)
" /path/to/capture 65
```

If it divides by `ncols` but not `2*ncols`, the file is single-plane and the assumed receiver axis
in §4 is wrong for v1.

## 10. Changelog

- **2026-07-29** — created. Columns from `v1_column_names`; absence of any writer or config
  confirmed by grep over `grbl_radio_collection.py` and the committed YAML. Container layout and
  the existence of surviving v1 files both left explicitly unverified.
