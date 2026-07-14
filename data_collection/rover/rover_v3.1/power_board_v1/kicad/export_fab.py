"""Regenerate fab/ outputs (gerbers, drill, JLC BOM + CPL) from the board.

    /usr/bin/python3 export_fab.py

Run AFTER audit_board.py passes. LCSC part numbers are carried over from the
existing fab/bom_jlc.csv (keyed by Comment+Footprint) so order-time fills
survive regeneration. Parts whose value contains "DNP" are excluded from
BOM/CPL (they still appear in the gerbers).
"""
import csv
import re
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
FAB = HERE / "fab"
PCB = HERE / "power_board_v1.kicad_pcb"

board = pcbnew.LoadBoard(str(PCB))
FAB.mkdir(exist_ok=True)

# ------------------------------------------------------------------ gerbers
pc = pcbnew.PLOT_CONTROLLER(board)
po = pc.GetPlotOptions()
po.SetOutputDirectory(str(FAB))
po.SetPlotFrameRef(False)
po.SetAutoScale(False)
po.SetMirror(False)
po.SetUseGerberAttributes(True)
po.SetUseGerberProtelExtensions(True)
po.SetCreateGerberJobFile(True)
po.SetSubtractMaskFromSilk(True)
po.SetPlotViaOnMaskLayer(False)
LAYERS = [
    ("F_Cu", pcbnew.F_Cu), ("In1_Cu", pcbnew.In1_Cu),
    ("In2_Cu", pcbnew.In2_Cu), ("B_Cu", pcbnew.B_Cu),
    ("F_Silkscreen", pcbnew.F_SilkS), ("B_Silkscreen", pcbnew.B_SilkS),
    ("F_Mask", pcbnew.F_Mask), ("B_Mask", pcbnew.B_Mask),
    ("F_Paste", pcbnew.F_Paste), ("B_Paste", pcbnew.B_Paste),
    ("Edge_Cuts", pcbnew.Edge_Cuts),
]
for name, layer in LAYERS:
    pc.SetLayer(layer)
    pc.OpenPlotfile(name, pcbnew.PLOT_FORMAT_GERBER, name)
    pc.PlotLayer()
pc.ClosePlot()

# -------------------------------------------------------------------- drill
ew = pcbnew.EXCELLON_WRITER(board)
ew.SetOptions(False, False, board.GetDesignSettings().GetAuxOrigin(), False)
ew.SetFormat(True)
ew.CreateDrillandMapFilesSet(str(FAB), True, False)

# ---------------------------------------------------------------- BOM / CPL
old_lcsc = {}
bom_path = FAB / "bom_jlc.csv"
if bom_path.exists():
    with open(bom_path) as f:
        for row in csv.DictReader(f):
            if row.get("LCSC"):
                old_lcsc[(row["Comment"], row["Footprint"])] = row["LCSC"]

groups = {}
cpl_rows = []
for fp in board.GetFootprints():
    ref = fp.GetReference()
    if ref.startswith("H"):
        continue
    val = fp.GetValue()
    if "DNP" in val:
        continue
    footprint = str(fp.GetFPID().GetLibItemName())
    groups.setdefault((val, footprint), []).append(ref)
    pos = fp.GetPosition()
    cpl_rows.append([ref, val, footprint,
                     round(pcbnew.ToMM(pos.x), 3), round(-pcbnew.ToMM(pos.y), 3),
                     "top" if fp.GetLayer() == pcbnew.F_Cu else "bottom",
                     round(fp.GetOrientationDegrees(), 1)])

with open(bom_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Comment", "Designator", "Footprint", "LCSC"])
    for (val, footprint), refs in sorted(groups.items()):
        w.writerow([val, ",".join(sorted(refs)), footprint,
                    old_lcsc.get((val, footprint), "")])

with open(FAB / "cpl_jlc.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Designator", "Val", "Package", "Mid X", "Mid Y", "Layer",
                "Rotation"])
    for row in sorted(cpl_rows):
        w.writerow(row)

print(f"gerbers: {len(LAYERS)} layers + drill -> {FAB}")
print(f"BOM: {len(groups)} line items; CPL: {len(cpl_rows)} parts; "
      f"LCSC carried over: {sum(1 for k in groups if k in old_lcsc)}")
