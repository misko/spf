# Board revision checklist — MANDATORY, EVERY revision, no exceptions

Two placement errors shipped past ad-hoc review in one day (XT60 pads off-board;
USB-C rotated 180° with all 20 pads off-board). Both were caught by humans, both
had left machine-readable traces (copper_edge_clearance counts rising) that were
triaged by COUNT instead of CONTENT. Hence this checklist: every board revision
reviews ALL placements and ALL pads, mechanically first, then by eye.

## Gate 1 — automated audit (runs on the .kicad_pcb; gates GUI edits too)
```
/usr/bin/python3 kicad/audit_board.py     # must print AUDIT: PASS
```
Invariants: I1 every copper pad ≥0.3 mm inside the outline · I2 no footprint
parked off-board · I3 each edge connector's body overhangs its DECLARED mate
edge (registry in the script — update it when adding/moving connectors) ·
I4 nothing under mounting-hole screw heads · I5 unnetted pads only where
whitelisted · I6 bbox overlap listing · I7 DRC category counts must not exceed
the committed `drc_baseline.json`.

Rules of engagement:
- The MATE registry and FLOAT_OK whitelist are part of the design: moving a
  connector means updating the registry IN THE SAME CHANGE.
- A DRC category may only be re-baselined (`--update-baseline`) after reading
  the individual report items and recording WHY in the commit message. Counts
  are never dismissed by category name.
- The generator (`generate_board.py`) also enforces I1 at build time; the
  standalone audit exists because hand edits in pcbnew bypass the generator.

## Gate 2 — visual review (repo PDF/PNG rule applies to boards)
1. Export full-board render + per-edge crops:
   `kicad-cli pcb export svg ... ; rsvg-convert ...`
2. Check each board edge: every connector's mouth faces off-board; pads inboard;
   nothing floats past the outline except declared overhangs.
3. Fresh-eyes pass: hand the renders to a NEW agent (or human) to describe each
   edge connector's orientation back — mismatches with intent = stop.

## Gate 3 — deltas, not absolutes
- `git diff --stat` the .kicad_pcb + generator: every anchor change must map to
  a stated reason.
- Re-run the audit AFTER routing/pour changes too (fills move, pads don't —
  but routing reruns regenerate the board, so anchors may shift).

## Gate 4 — record
- Commit audit output (PASS line) in the commit message or REVIEW_FINDINGS.md.
- Open items (accepted warns, known grazes) listed explicitly with owners.

Checklist owner: whoever touches the board files in that revision.
