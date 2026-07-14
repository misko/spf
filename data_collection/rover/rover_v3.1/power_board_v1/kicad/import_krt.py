"""Import KRT-routed segments/vias (KiCad-9 dialect) into the v7 board."""
import re, collections, sys
from pathlib import Path
import pcbnew

krt_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pb_krt2.kicad_pcb"
src = Path(krt_file).read_text()
netname = dict(re.findall(r'\(net (\d+) "([^"]+)"\)', src))
board = pcbnew.LoadBoard("power_board_v1.kicad_pcb")
existing_vias = [t.GetPosition() for t in board.GetTracks() if type(t).__name__ == "PCB_VIA"]
LAY = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu, "In1.Cu": pcbnew.In1_Cu, "In2.Cu": pcbnew.In2_Cu}
nseg = nvia = 0
for m in re.finditer(r'\(segment\s+\(start ([\d.-]+) ([\d.-]+)\)\s+\(end ([\d.-]+) ([\d.-]+)\)\s+\(width ([\d.]+)\)\s+\(layer "([^"]+)"\)\s+\(net (\d+)\)', src):
    x1, y1, x2, y2, w, lay, nn = m.groups()
    net = board.FindNet(netname.get(nn, ""))
    if net is None or net.GetNetCode() <= 0:
        continue
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pcbnew.VECTOR2I_MM(float(x1), float(y1)))
    t.SetEnd(pcbnew.VECTOR2I_MM(float(x2), float(y2)))
    t.SetWidth(pcbnew.FromMM(float(w)))
    t.SetLayer(LAY.get(lay, pcbnew.F_Cu))
    t.SetNet(net)
    board.Add(t)
    nseg += 1
for m in re.finditer(r'\(via\s+\(at ([\d.-]+) ([\d.-]+)\)\s+\(size ([\d.]+)\)\s+\(drill ([\d.]+)\)\s+\(layers "[^"]+" "[^"]+"\)[^)]*\(net (\d+)\)', src):
    x, y, s, dr, nn = m.groups()
    px, py = pcbnew.FromMM(float(x)), pcbnew.FromMM(float(y))
    if any(abs(v.x - px) < 100000 and abs(v.y - py) < 100000 for v in existing_vias):
        continue
    net = board.FindNet(netname.get(nn, ""))
    if net is None or net.GetNetCode() <= 0:
        continue
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pcbnew.VECTOR2I(px, py))
    v.SetWidth(pcbnew.FromMM(float(s)))
    v.SetDrill(pcbnew.FromMM(float(dr)))
    v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    v.SetNet(net)
    board.Add(v)
    existing_vias.append(v.GetPosition())
    nvia += 1
ds = board.GetDesignSettings()
ds.m_MinClearance = pcbnew.FromMM(0.10)   # JLC 4+ layer standard floor
ds.m_TrackMinWidth = pcbnew.FromMM(0.10)
pcbnew.ZONE_FILLER(board).Fill(board.Zones())
board.Save("power_board_v1.kicad_pcb")
print(f"imported {nseg} segments, {nvia} vias")
rpt = "/tmp/drc_final.txt"
pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
k = collections.Counter(re.findall(r"\[(\w+)\]", Path(rpt).read_text()))
print("DRC@4L-floor:", {x: k[x] for x in ("unconnected_items","clearance","shorting_items","hole_clearance","copper_edge_clearance") if x in k})
