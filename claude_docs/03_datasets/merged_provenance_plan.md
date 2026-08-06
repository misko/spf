# Storing radio provenance in merged datasets

**Status:** proposed, 2026-08-06. Supersedes the "known gap" note in
`v7_tx_rx_merge.py`'s docstring and the 2026-08-01 report appendix.

## The problem

`v7_tx_rx_merge.py` drops **31 of 31 receiver attrs and 3 of 4 root attrs**.
Measured field-by-field in the 2026-08-05 report, Appendix B. Every *array*
survives — the merge adds five and drops none — so the loss is entirely
metadata:

- `sdr_serial` / `direct_usb_serial` — the only in-band identifier of the
  physical Pluto. **In no other field.**
- `capture_status`, `capture_records_written_by_receiver` — whether the source
  capture finalised cleanly.
- `firmware_verified`, `firmware_device_fw`, `hardware_fingerprint_v1` — the
  attested running firmware, as distinct from the configured intent.
- `usb_port_path`, `usb_address_at_capture` — physical attachment.

`config.receiver-port` does survive, so `r0 → port 2` is answerable.
`r0 → which serial` is not.

## Three findings that shape the plan

Established by inspecting the corpus, not assumed.

### 1. The precompute never reads attrs — so it does not need reprocessing

`segment_zarr.py`, `spf/dataset/spf_dataset.py` and `spf/dataset/v5_data.py`
contain **zero** `.attrs` references. Adding attrs to a merged store cannot
invalidate a precomputed tensor derived from that store.

This is the finding that makes the whole job cheap. The brief asked to
"reprocess all merged versions and their precomputed properties"; the
precomputed properties do not need to be touched at all.

### 2. Every merged dataset can resolve its sources

24 merged datasets across four directories; 48 source references; **48 resolve.**
The merged filename is `<rx_name>.<tx_name>.zarr` and no source basename is
duplicated anywhere in the two campaigns.

| Directory | merged stores | size |
|---|---:|---:|
| `rovers_july_2026/merged` | 4 | 31 G |
| `rovers_august_2026/merged` (aug 1) | 14 | 106 G |
| `rovers_august_2026/merged_aug2` | 4 | 22 G |
| `rovers_august_2026/merged_aug4` | 2 | 25 G |
| **total** | **24** | **184 G** |

Five RX sources resolve only via a `.zarr.tmp` suffix — see finding 3. All five
still carry their full attrs, so **backfill coverage is 100%**.

### 3. Five merged datasets were built from unfinalised captures — invisibly

The merged output name drops the `.tmp`, so a merged store built from
`x.zarr.tmp` is named `x.zarr.<tx>.zarr` and reads as finalised.

| Merged dataset (RX part) | source `capture_status` | source suffix |
|---|---|---|
| `2026_07_31_18_35_35 … 0p035` | `in_progress` | `.zarr.tmp` |
| `2026_07_31_18_49_02 … 0p035` | `incomplete` | `.zarr.tmp` |
| `2026_08_01_19_31_21 … 0p043` | `in_progress` | `.zarr.tmp` |
| `2026_08_01_22_10_01 … 0p043` | `incomplete` | `.zarr.tmp` |
| `2026_08_01_22_57_45 … 0p043` | `in_progress` | `.zarr.tmp` |

The other 19 are `complete` from finalised `.zarr`. **This is exactly the
question the dropped `capture_status` prevents anyone from asking**, and it is
already load-bearing: five of 24 datasets in the training corpus came from
captures that never finalised, and nothing in those datasets says so.

## Design

### There is already a place, and for RX it is empty

Zarr groups carry their own `.zattrs`. The merged store **already has every slot
the source uses** — the merge simply writes nothing into them:

| Slot | Source store | Merged store |
|---|---:|---:|
| root `.zattrs` | 4 keys | 1 key |
| `receivers/r0/.zattrs` | 31 keys | **0 — empty** |
| `receivers/r1/.zattrs` | 31 keys | **0 — empty** |

So restoring RX provenance needs **no new schema, no namespace, no nesting**. It
is a dict copy into the identical slot, next to `_copy_receiver` in
`v7_tx_rx_merge.py`:

```python
new_zarr.attrs.update(dict(rx_zarr.attrs))            # root: capture_status, ...
for r in range(receivers):
    new_zarr[f"receivers/r{r}"].attrs.update(
        dict(rx_zarr[f"receivers/r{r}"].attrs))       # 31 attrs incl. sdr_serial
```

An earlier draft of this plan proposed a nested `attrs["provenance"]` record
holding rx and tx side by side. That was over-designed: it invented a container
for something zarr already models, and it only looked necessary because the plan
assumed TX needed the same treatment as RX. It does not — see below.

### TX needs GPS, and the GPS is already there

The merged store has no TX receiver group and should not grow one: its
`receivers/` **are** the RX's receivers. What the TX contributes is position, and
that is already stored as `tx_pos_x_mm` / `tx_pos_y_mm` — the TX GPS track
interpolated onto every RX timestamp.

**That projection is exactly invertible from the merged store alone.** The store
also holds the RX's raw `gps_lat` / `gps_long` *and* its projected
`rx_pos_x_mm` / `rx_pos_y_mm`, which over-determines the aeqd projection centre.
Fitting it on one merged dataset:

```
recovered centre : 37.8349747, -122.4788380     residual RMS 0.0000 mm
TX GPS inverted  : lat 37.834841..37.835118   lon -122.478979..-122.478701
TX source actual : lat 37.834840..37.835120   lon -122.478979..-122.478701
```

Agreement is 0.04 m / 0.24 m at the bounding-box corners, and that residue is the
merge trimming TX to the overlap window — not projection error.

**Consequence: no legacy merged dataset needs anything added to recover TX GPS.**
It is already present and exact. What is missing is only the convenience of not
having to run a least-squares fit to get it, and the identity of which TX store
it came from.

### The whole change, then

| What | Where it goes | New schema? |
|---|---|---|
| RX radio attrs — `sdr_serial`, firmware, USB path (31 × 2) | `receivers/r{0,1}/.zattrs` — **exists, empty** | no |
| RX capture status — `capture_status`, record counts (4) | root `.zattrs` — exists | no |
| **`projection`** = `{proj: "aeqd", lat_0, lon_0, units: "m"}` | root `.zattrs` | **1 key** |
| **`tx_source`** = `{store, suffix, finalized, capture_status}` | root `.zattrs` | **1 key** |

**Two new root keys.** Everything else lands in slots zarr already provides.

`projection` because inverting `tx_pos_*_mm` should not require a fit, and
because it pins the frame the XY coordinates are expressed in — the merge derives
it from the mean TX GPS, so it varies per dataset.

`tx_source` because the TX store's name and finalisation state are the one thing
about the TX that genuinely is not recoverable from the merged arrays. Four
fields, not thirty-one: the TX's radio identity does not matter when only its
GPS is used.

`suffix` / `finalized` are recorded explicitly in `tx_source` — and, for RX, come
free with the copied root attrs — because the merged filename destroys the
`.tmp` distinction (finding 3).

### Legacy backfill: writing into empty slots

The RX slots in the 24 legacy stores are **empty**. Filling them adds keys where
none exist and changes no existing value, which is additive in the sense the
append-only convention is about — unlike editing a split file or a config. So
in-place is defensible here, and it keeps one read path instead of two.

The tool supports both, defaulting to the safer one:

- `--sidecar` (default) writes `<name>.provenance.json` mirroring the native
  layout: `{"root": {...}, "receivers": {"r0": {...}, "r1": {...}}, "projection": {...}, "tx_source": {...}}`.
- `--in-place` writes the same content into the store's own attrs, and refuses to
  run if any target slot is non-empty.

Start with `--sidecar`, review the JSON, then promote with `--in-place` once it
has been checked. `load_provenance()` prefers in-band and falls back to the
sidecar, so both eras read identically and the promotion is invisible to callers.

**Why not re-run the merge.** Re-merging rewrites 184 GB of arrays that are
already byte-correct to change ~50 KB of metadata, and every precompute keyed to
the current stores would follow. The arrays are not in question.

## Execution

Each phase is independently revertible and leaves the corpus usable.

**P0 — Freeze the inventory.** Emit and commit `merged_provenance_inventory.json`:
24 datasets, resolved source paths, suffix, `capture_status`, r0/r1 serials,
fitted projection centre. This is what later phases diff against, and it is worth
having on its own — it is how finding 3 surfaced.

**P1 — Writer.** The three-line attrs copy above, plus `projection` and
`tx_source`, in `v7_tx_rx_merge.py`. Tests: a round-tripped `sdr_serial`; a
`.zarr.tmp` source setting `finalized: false`; `projection` inverting
`tx_pos_*_mm` back to the TX source's GPS within a tolerance.

**P2 — Backfill tool.** `backfill_merged_provenance.py --dry-run` prints what it
would write; `--sidecar` then `--in-place`. Idempotent; refuses to overwrite a
non-empty slot.

**P3 — Accessor.** `spf/dataset/provenance.py::load_provenance(store)`, wired into
`merged_df_figure.py` and `df_metrics.py` so figures label results by physical
radio rather than `r0`.

**P4 — Verify.** All 24 answer: which serial was r0, was the source finalised,
and what frame are the XY coordinates in. Assert the five known-unfinalised are
flagged.

**P5 — Aug 5.** Merges run after P1 are born correct and need no backfill.

**P6 — Optional: precompute linkage.** Not required — the precompute reads no
attrs (finding 1) — but stamping the source serials into precompute outputs would
let a cached tensor be traced to the radio that produced it.

## Cost

| | Reads | Writes | Wall clock |
|---|---|---|---|
| Backfill 24 legacy | attrs only, no arrays | ~50 KB | minutes |
| Re-merge 24 | ~500 GB of sources | 184 GB + precompute | days |

## What it changes for open questions

The 2026-08-05 report's central unresolved question is whether the channel
deficits on RO1/RO3/RO4 are cross-array coupling (5.3) or real hardware faults.
That is a question about *physical radios*, and the merged datasets the DF
metrics are computed from cannot currently name one. Across the whole corpus only
two distinct r0 serials appear (`…192e5a` on the 35 mm arrays, `…3b1bd0` on the
43 mm arrays); with the attrs restored, per-serial error statistics become a
query rather than a re-derivation from filenames.
