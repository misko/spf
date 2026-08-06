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

### The record

One schema-versioned namespace, `attrs["provenance"]`:

```jsonc
{
  "schema_version": 1,
  "generator": {
    "tool": "v7_tx_rx_merge.py",
    "git_commit": "<sha of spf at merge time>",
    "written_utc": "2026-08-06T09:14:22Z",
    "min_timesteps": 500,
    "projection_center": {"lat": 37.83538, "lon": -122.4785}
  },
  "sources": {
    "rx": {
      "store": "rover_2026_08_05_23_31_21_…_tag_RO1.zarr",
      "suffix": ".zarr",              // ".zarr.tmp" records an unfinalised source
      "finalized": true,
      "attrs_sha256": "<hash of the serialised source attrs>",
      "root": { "capture_status": "complete", … },
      "receivers": { "r0": { …31 attrs… }, "r1": { …31 attrs… } }
    },
    "tx": { … same shape … }
  }
}
```

Design choices, and why:

- **One namespace, not 62 flattened keys.** Source attrs collide between rx and
  tx (both have `sdr_serial`), and a flat merge would need prefixes that then
  need parsing. Nesting keeps the source record byte-identical to what was read.
- **`attrs_sha256` over the serialised source attrs.** Lets a later run detect
  that a source was replaced or re-created since the record was written. Cheap,
  and the alternative is trusting a filename forever.
- **`suffix` and `finalized` recorded explicitly**, because the merged filename
  destroys that distinction (finding 3).
- **`projection_center` and `min_timesteps`** because the merge is not a pure
  function of its inputs — the XY frame origin is derived from the TX GPS mean,
  and a different `--min-timesteps` yields a different row count.

### The derived index

The canonical record is nested, but the hot question — *which physical radio was
r0?* — should not require walking it. The writer also emits a small flat index,
always regenerated from the record and never hand-edited:

```jsonc
attrs["radio_identity"] = {
  "rx": {"r0": "10400090fd95…192e5a", "r1": "104000d02597…846432"},
  "tx": {"r0": "…"}
}
attrs["source_status"] = {"rx": "complete", "tx": "complete", "rx_finalized": true}
```

Redundant by construction. The justification is that "recoverable" and
"answerable" are different properties, and the corpus has just demonstrated the
cost of the gap between them.

### Where it lives — three write paths, one read path

| Era | Location | Written by |
|---|---|---|
| New merges (Aug 5 onward) | in-band `attrs` | `v7_tx_rx_merge.py` |
| The 24 legacy merges | sidecar `<name>.provenance.json` | `backfill_merged_provenance.py` |
| Either | — | `load_provenance()` — prefers in-band, falls back to sidecar |

**Why a sidecar for legacy rather than writing attrs in place.** Repo convention
is that artifacts are append-only. A sidecar adds without mutating, is reversible
by deleting one file, and is reviewable as plain JSON *before* anything trusts
it. Merged directories already carry `.yaml` sidecars, so the pattern is
established there. Cost is ~50 KB across the corpus.

If in-band is preferred for legacy too, it is `--in-place` on the same tool: the
record is identical, only the destination changes. The tradeoff is mutating 24
existing artifacts to save one indirection in the reader.

**Why not re-run the merge.** Re-merging rewrites 184 GB of arrays that are
already byte-correct in order to change ~50 KB of metadata, and every precompute
keyed to the current stores would have to be regenerated behind it. The arrays
are not in question — only the metadata is.

## Execution

Each phase is independently revertible and leaves the corpus usable.

**P0 — Freeze the inventory.** Emit and commit
`merged_provenance_inventory.json`: 24 datasets, resolved source paths,
suffix, `capture_status`, r0/r1 serials. This is the artifact every later phase
diffs against, and it is worth having on its own — it is how finding 3 surfaced.

**P1 — Schema and writer.** Add the record to `v7_tx_rx_merge.py`. Tests:
round-trip through a merge; a `.zarr.tmp` source sets `finalized: false`; a
source whose attrs are absent yields an explicit `null`, never a silent omission.

**P2 — Backfill tool.** `backfill_merged_provenance.py`, `--dry-run` first,
printing the JSON it would write. Idempotent. Refuses to write a sidecar when
in-band provenance already exists, so the two paths cannot disagree.

**P3 — Accessor.** `spf/dataset/provenance.py::load_provenance(store)`, used by
`merged_df_figure.py` and `df_metrics.py` so figures can label results with the
physical radio rather than `r0`.

**P4 — Verify.** All 24 answer both questions: which serial was r0, and was the
source finalised. Assert the five known-unfinalised are flagged.

**P5 — Aug 5.** Merges run after P1 are born with in-band provenance and need no
backfill.

**P6 — Optional: precompute linkage.** Not required — the precompute reads no
attrs (finding 1) — but stamping `attrs_sha256` into precompute outputs would
let a cached tensor be traced to the radio that produced it.

## Cost

| | Reads | Writes | Wall clock |
|---|---|---|---|
| Backfill 24 legacy | attrs only, no arrays | ~50 KB of JSON | minutes |
| Re-merge 24 | ~500 GB of sources | 184 GB + precompute | days |

## What it changes for open questions

The 2026-08-05 report's central unresolved question is whether the channel
deficits on RO1/RO3/RO4 are cross-array coupling (5.3) or real hardware faults.
That is a question about *physical radios*, and the merged datasets — the ones
the DF metrics are computed from — currently cannot name one. Across the whole
corpus only two distinct r0 serials appear (`…192e5a` on the 35 mm arrays,
`…3b1bd0` on the 43 mm arrays); with provenance in place, per-serial error
statistics become a one-line query instead of a re-derivation from filenames.
