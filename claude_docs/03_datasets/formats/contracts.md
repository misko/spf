# Data-format documentation contract

**This file defines what a data-format document must contain.** Every capture data version in
SPF gets exactly one file in this folder, named `v<N>_data_format.md`. If a document is missing
a required section, it is incomplete — add the section with an explicit "not established" note
rather than silently dropping it.

- Index of the versions and their status: [`README.md`](./README.md)
- Corpus-level facts (how many datasets, where they live, quality): [`../DATA_OVERVIEW.md`](../DATA_OVERVIEW.md)
- Field conventions shared by all versions (units, angle frames): [`../../00_concepts/conventions.md`](../../00_concepts/conventions.md)

---

## 1. Ground rules

1. **Code is the source of truth; the document is a map.** Every factual claim cites the code
   that makes it true, as `path:line` (e.g. `spf/dataset/v7_data.py:12`). If code and doc
   disagree, the code is right and the doc is a bug.
2. **Exhaustive means exhaustive.** The field tables must list *every* array in the container
   **and every container/group attribute**, including ones nobody uses. A field that exists in
   the file but not in the doc is the exact failure this folder exists to prevent. If a version's
   field list is generated from a Python list (it usually is), name that list so a reader can
   diff it. Attributes are easy to miss because they are not in those lists — they are written
   separately (`DataCollector._record_receiver_identities`), so check them by opening a real file,
   not by reading the schema module.
3. **Separate "recorded" from "derived".** A field written by the collector at capture time and
   a field computed later by a loader or script are different kinds of thing with different
   trust levels. They never share a table.
4. **Mark what you did not verify.** Use `⚠ unverified` inline. A documented guess that looks
   like a fact is worse than an admitted gap.
5. **Raw data is immutable** (root `CLAUDE.md`). A format doc describes what is on disk; it
   never proposes rewriting existing captures. Corrections go to new locations.
6. **Never restate corpus counts here.** Dataset counts, sizes and quality splits live in
   `DATA_OVERVIEW.md` and go stale fast. Link, don't copy.

---

## 2. Required sections

Each `v<N>_data_format.md` must have these sections, in this order, with these headings.

### 0. Status header (unnumbered, at the top)

A short key/value block, so the reader knows within five seconds whether this version matters:

| Row | Content |
|---|---|
| **Status** | `live` / `legacy-readable` / `transitional` / `abandoned` |
| **Container** | e.g. "Zarr over an LMDB store", "flat `np.memmap`" |
| **Written by** | the collector class and orchestrator script |
| **Read by** | the loader class(es), and whether training consumes it directly |
| **Superseded by / supersedes** | version links |
| **Defining module** | the file that declares the key lists |

### 1. Motivation

Why this version exists **as distinct from its predecessor**. What could not be done before.
What the new fields were introduced to answer. If a version was skipped (there is no v3), say
so here rather than leaving a reader hunting.

### 2. Collection types that produce it

Which platform (wall array / rover / bench / synthetic), which orchestrator, which motion
routines, which transport (`iio` vs `direct_usb`), and which hardware. State explicitly whether
any *currently running* capture path writes this version, or whether it exists only in the
historical corpus.

### 3. Example configs

Point at **real committed configs**, by path, not invented YAML. Include a trimmed excerpt of
the fields that determine the format (`data-version`, `rx-transport`, `direct-usb`, receiver
block, `n-records-per-receiver`, `buffer-size`). Note any schema validation that gates the
version (`spf/capture_schema.py`) and any config→format coupling a reader would otherwise
have to discover by crashing.

### 4. Container layout

The physical shape of the artifact: directory/file naming, group hierarchy, array shapes,
dtypes, chunking, compressor, **attributes**, and where the capture config is embedded. Give the
store's sizing behaviour if it is surprising (preallocation, shrink-on-close). One tree diagram
beats three paragraphs.

### 5. Recorded fields (exhaustive)

The core table. One row per array, with these columns, in this order:

| Column | Rule |
|---|---|
| **Field** | exact key name as it appears in the container |
| **Shape** | in terms of `T` (timesteps) and `B` (buffer size), e.g. `(T, 2)` |
| **Dtype** | the on-disk dtype, not the in-memory one |
| **Units** | physical units, or `—` for dimensionless/enum. **Never blank.** "multiples of π", "mm", "dBFS", "ns", "Hz" |
| **Written by** | `file:line` of the assignment |
| **Meaning** | what the number physically is |
| **Use / caveat** | what it is good for, and what it is *not* good for |

Group rows by origin (radio-derived / platform-derived / protocol metadata) with subheadings
when the table exceeds ~15 rows. Close the section with a **"Store-level arrays and attributes"**
subsection covering the embedded config and every zarr attribute, and say which attributes are
*enforced* (a capture cannot exist without them) versus merely present. Any field that is a known trap (constant, vestigial, unit
mismatch, silently zero on some path) gets its caveat spelled out in the row **and** a mention
in section 8.

### 6. Post-processed fields

Everything not written at capture time: loader-derived values, ground-truth angles computed
from geometry, segmentation/beamforming caches, and offline conversions. For each, say **who
computes it**, **where it is stored** (in-memory only, sidecar cache, new zarr), **whether it
is versioned**, and **whether it can be regenerated** from the raw capture. Cross-link the
shared precompute-cache document rather than duplicating it.

### 7. Reading it

The minimum working example that opens this version, and the flags a reader must pass for it
to behave (e.g. `v4=True`). Note anything the loader silently synthesizes or fills so that a
reader does not mistake a filler for a measurement.

### 8. Known issues and traps

Version-specific hazards, each linked to its `KNOWN_ISSUES.md` id or `docs/learnings.md`
entry where one exists. This section is allowed to be blunt.

### 9. Verification

A concrete, copy-pasteable way to check that a file on disk actually conforms — which keys
must be present, which invariants must hold, and the script or command that checks it.
**Look for a committed checker before writing a snippet** (`ls spf/scripts/validate_*`): v6 and v7
each have one, and a hand-rolled snippet that duplicates it will drift. If no automated checker
exists, say so explicitly, and give the manual inspection snippet. Note any hardcoded expectations
in the checker (e.g. a fixed buffer size) that would fail a legitimate capture.

### 10. Changelog

Dated, append-only entries recording what changed in the **document**, with the commit or
evidence behind each. Newest first.

---

## 3. Shared conventions

These apply across every version; do not re-derive them per document.

| Thing | Convention | Notes |
|---|---|---|
| Orientation fields (`*_in_pis`) | multiples of π | `0.5` means π/2 rad. Convert before trig. |
| Raw platform heading | **degrees** | Only in v4's `heading`. The `_in_pis` twin is `deg/180`. |
| Positions (`*_pos_*_mm`) | millimetres | Wall array only; zero-filled on rover captures. |
| Antenna spacing (`rx_spacing`) | metres | Physical distance between the two elements. |
| Frequencies (`rx_lo`, `rx_bandwidth`) | Hz | |
| Durations (`*_duration_ns`) | nanoseconds | Host-side read cost, not signal timing. |
| Power (`iq_power_dbfs`) | dBFS vs 12-bit full scale | `full_scale_power = 2·2048²`, `spf/sdrpluto/sdr_controller.py:121` |
| Receiver index | `r0`, `r1` | Ordering follows the YAML `receivers:` list order, **not** `receiver-port`. |
| Element index within a receiver | `0`, `1` → Pluto RX1, RX2 | Verified at the wire level: `signal_matrix[0]` takes IQ channel 0, `[1]` channel 1 (`spf/sdrpluto/direct_usb_receiver.py:75-78`; the IIO path de-interleaves the same way, `sdr_controller.py:673`). Which *physical* antenna is on which channel is a cabling fact — labels in the [rover README](../../../data_collection/rover/rover_v3.1/README.md) |
| Timesteps | `T` = `n-records-per-receiver` | Preallocated; a killed run leaves the tail unwritten and untrimmed. |
| Buffer size | `B` = `buffer-size` | Complex samples per element per record. |

---

## 4. Adding a new data version

Checklist, in order:

1. Add the key lists in a new `spf/dataset/v<N>_data.py` and the contract in
   `spf/capture_schema.py`.
2. Copy the nearest existing format doc, **not** a blank file, so the section order survives.
3. Fill every section. Diff your field table against the module's key lists mechanically —
   `python -c "from spf.dataset.v<N>_data import v<N>rx_keys; print(sorted(v<N>rx_keys()))"` —
   and paste that command into section 9.
4. Add the row to [`README.md`](./README.md)'s version matrix.
5. If the new version changes a project-level conclusion (what is trustworthy, what to train
   on), record it in `docs/learnings.md` in the same change, per root `CLAUDE.md`.

---

## 5. Changelog

- **2026-07-29 (review pass)** — after a verification pass over the initial documents, three
  rules were tightened because the first draft violated them: attributes must be enumerated
  (§1.2, §2.4, §2.5), a committed validator must be preferred over a hand-rolled snippet (§2.9),
  and the element-order convention now carries its wire-level citation.
- **2026-07-29** — contract created alongside the initial `v1/v2/v4/v5/v6/v7` documents.
