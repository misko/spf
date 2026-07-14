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

## Canonical routing pipeline (v4.2, 2026-07: KiCadRoutingTools)
KRT lives at ~/gits/KiCadRoutingTools (do NOT rely on /tmp clones). Order is
load-bearing — deviations reintroduce failures we already debugged:
1. Start from a TRACK-FREE, UNFILLED board (KRT mis-parses filled zones and
   will route straight through existing copper — 400+ crossings observed).
2. Mounting-hole keepout squares on User.2; route with `--keepout`.
3. `bga_fanout.py` U1/U2 (VQFN) BEFORE any routing — fanout on a routed board
   finds its escape lanes already taken. U3 (0.4mm QFN) cannot be fanned out
   or routed between pads at ANY legal geometry; its escape count is a hard
   cap, so its nets go in the hardest-first set.
4. Hardest-first: thin pass (0.15/0.13, 0.45/0.2 vias, fab-tier advanced) for
   the escape-bound nets, THEN the standard pass (0.2/0.3, 0.6/0.3, power
   nets 1.2mm), THEN thin reconciliation of leftovers.
5. Import into the pcbnew base (textual segment/via import), fill zones,
   DRC, audit_board.py. Judge clearance by CLASS (different-net <0.10mm =
   real; everything else = margin/triage), not raw counts.
6. `export_fab.py` regenerates gerbers/drill/BOM/CPL (DNP excluded).
Residual ~14 airlines (U3 escapes + U4/U1 pin joins) are the expected manual
push-and-shove hour; autorouters cannot close them (verified: more layers,
bare-board, smaller vias, spread grids all plateau the same).
