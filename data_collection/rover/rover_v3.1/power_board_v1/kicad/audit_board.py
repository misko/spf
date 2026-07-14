"""Placement/pad audit — MANDATORY for every board revision (see
../BOARD_REVISION_CHECKLIST.md). Runs on the .kicad_pcb directly, so it gates
GUI edits as well as generated boards.

    /usr/bin/python3 audit_board.py [--update-baseline]

Exit 0 = all invariants pass and DRC categories are at/below the committed
baseline. Anything else = fix or explicitly re-baseline with justification.
"""
import collections
import json
import re
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
PCB = HERE / "power_board_v1.kicad_pcb"
BASELINE = HERE / "drc_baseline.json"

# board frame (must match generate_board.py)
X0, Y0, W, H = 50.0, 50.0, 90.0, 65.0
EDGE_MARGIN_MM = 0.3
SCREW_HEAD_R_MM = 3.2  # M3 pan head + margin

# Declared mating directions for edge connectors: the connector BODY must
# overhang (or touch) this edge, and all copper must be inboard.
MATE = {"J1": "W", "J3": "N", "J12": "N", "J4": "E", "J5": "E", "J13": "E", "J7": "S"}
# Pads intentionally un-netted (LM74800 RTN must float; NC pins)
FLOAT_OK = {("U4", "13")}

fails = []
warns = []
board = pcbnew.LoadBoard(str(PCB))
fps = {f.GetReference(): f for f in board.GetFootprints()}
mm = pcbnew.ToMM

# --- I1: every copper pad inside the outline ------------------------------
for r, f in fps.items():
    for p in f.Pads():
        pos, hx, hy = p.GetPosition(), p.GetSizeX() / 2, p.GetSizeY() / 2
        if (mm(pos.x) - mm(int(hx)) < X0 + EDGE_MARGIN_MM or
                mm(pos.x) + mm(int(hx)) > X0 + W - EDGE_MARGIN_MM or
                mm(pos.y) - mm(int(hy)) < Y0 + EDGE_MARGIN_MM or
                mm(pos.y) + mm(int(hy)) > Y0 + H - EDGE_MARGIN_MM):
            fails.append(f"I1 pad-off-board: {r} pad {p.GetNumber()}")

# --- I2: footprint centers on the board (no parked parts) -----------------
for r, f in fps.items():
    c = f.GetPosition()
    if not (X0 <= mm(c.x) <= X0 + W and Y0 <= mm(c.y) <= Y0 + H):
        fails.append(f"I2 parked-off-board: {r} at ({mm(c.x):.1f},{mm(c.y):.1f})")

# --- I3: declared mate-direction overhang for edge connectors -------------
for r, side in MATE.items():
    f = fps.get(r)
    if f is None:
        continue
    bb = f.GetBoundingBox(False, False)
    over = {"W": X0 - mm(bb.GetLeft()), "E": mm(bb.GetRight()) - (X0 + W),
            "N": Y0 - mm(bb.GetTop()), "S": mm(bb.GetBottom()) - (Y0 + H)}[side]
    if over < -1.0:
        fails.append(f"I3 mate-direction: {r} declared {side} but body is "
                     f"{-over:.1f}mm short of that edge (rotated wrong?)")

# --- I4: mounting-hole screw-head keepouts --------------------------------
holes = [f.GetPosition() for ref, f in fps.items() if ref.startswith("H")]
for r, f in fps.items():
    if r.startswith("H"):
        continue
    for p in f.Pads():
        for hpos in holes:
            dx = abs(mm(p.GetPosition().x) - mm(hpos.x))
            dy = abs(mm(p.GetPosition().y) - mm(hpos.y))
            if max(dx, dy) < SCREW_HEAD_R_MM:
                fails.append(f"I4 screw-head: {r} pad {p.GetNumber()} under "
                             f"mounting-hole head")
    bb = f.GetBoundingBox(False, False)
    for hpos in holes:
        if (mm(bb.GetLeft()) < mm(hpos.x) < mm(bb.GetRight()) and
                mm(bb.GetTop()) < mm(hpos.y) < mm(bb.GetBottom())):
            warns.append(f"I4w body-over-hole: {r} covers a mounting hole")

# --- I5: unnetted pads only where whitelisted ------------------------------
for r, f in fps.items():
    if r.startswith("H"):
        continue
    for p in f.Pads():
        num = p.GetNumber()
        if num and p.GetNetCode() <= 0 and (r, num) not in FLOAT_OK:
            nm = p.GetPadName()
            if num not in ("MP", "S1", ""):  # shells handled by footprints
                warns.append(f"I5 unnetted: {r} pad {num}")

# --- I6: bbox overlaps ------------------------------------------------------
boxes = [(r, f.GetBoundingBox(False, False)) for r, f in fps.items()
         if not r.startswith("H")]
overlaps = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].Intersects(boxes[j][1]):
            overlaps.append((boxes[i][0], boxes[j][0]))
if overlaps:
    warns.append(f"I6 bbox-overlaps ({len(overlaps)}): {overlaps}")

# --- I7: DRC category counts vs committed baseline -------------------------
rpt = "/tmp/audit_drc.txt"
pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
counts = collections.Counter(
    re.findall(r"\[(\w+)\]", Path(rpt).read_text()))
counts = {k: v for k, v in counts.items()
          if k in ("clearance", "copper_edge_clearance", "hole_clearance",
                   "courtyards_overlap", "holes_co_located", "shorting_items",
                   "unconnected_items", "starved_thermal",
                   "solder_mask_bridge", "silk_overlap")}
if "--update-baseline" in sys.argv:
    BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True))
    print("baseline updated:", counts)
elif BASELINE.exists():
    base = json.loads(BASELINE.read_text())
    for k, v in counts.items():
        if v > base.get(k, 0):
            fails.append(f"I7 DRC-regression: {k} {base.get(k, 0)} -> {v} "
                         f"(read drc report items, don't hand-wave counts!)")
else:
    warns.append("I7 no drc_baseline.json — run --update-baseline once clean")

print(f"pads audited: {sum(len(list(f.Pads())) for f in fps.values())} "
      f"across {len(fps)} footprints")
for w in warns:
    print("WARN:", w)
for x in fails:
    print("FAIL:", x)
print("AUDIT:", "FAIL" if fails else "PASS",
      f"({len(fails)} fails, {len(warns)} warns)")
sys.exit(1 if fails else 0)
