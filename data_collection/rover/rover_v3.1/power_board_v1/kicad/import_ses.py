"""Rebuild all tracks/vias from /tmp/pb.ses (full-wiring session), refill, DRC."""
import collections
import re
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
PCB = HERE / "power_board_v1.kicad_pcb"
SES = Path("/tmp/pb.ses")

board = pcbnew.LoadBoard(str(PCB))

ses = SES.read_text()

LAY = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}
mm = lambda v: float(v) / 10000.0

added_t = added_v = 0
for nm in re.finditer(r'\(net (\S+)\n((?:.|\n)*?)(?=\n      \(net |\n    \)\n  \))', ses):
    name, body = nm.group(1), nm.group(2)
    net = board.FindNet(name)
    if net is None or net.GetNetCode() <= 0:
        continue
    for w in re.finditer(r'\(path (\S+) (\d+)((?:\s+-?\d+)+)\s*\)', body):
        layer, width, coords = w.group(1), int(w.group(2)), w.group(3).split()
        pts = [(mm(coords[i]), -mm(coords[i + 1])) for i in range(0, len(coords), 2)]
        for a, b in zip(pts, pts[1:]):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pcbnew.VECTOR2I_MM(*a))
            t.SetEnd(pcbnew.VECTOR2I_MM(*b))
            t.SetWidth(pcbnew.FromMM(width / 10000.0))
            t.SetLayer(LAY.get(layer, pcbnew.F_Cu))
            t.SetNet(net)
            board.Add(t)
            added_t += 1
    for vm in re.finditer(r'\(via "Via\[0-1\]_(\d+):(\d+)_um" (-?\d+) (-?\d+)', body):
        w_um, d_um, x, y = vm.groups()
        px, py = pcbnew.FromMM(mm(x)), pcbnew.FromMM(-mm(y))
        if any(abs(t.GetPosition().x - px) < 100000 and abs(t.GetPosition().y - py) < 100000
               for t in board.GetTracks() if type(t).__name__ == "PCB_VIA"):
            continue
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pcbnew.VECTOR2I_MM(mm(x), -mm(y)))
        v.SetWidth(pcbnew.FromMM(int(w_um) / 1000.0))
        v.SetDrill(pcbnew.FromMM(int(d_um) / 1000.0))
        v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        v.SetNet(net)
        board.Add(v)
        added_v += 1

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
board.Save(str(PCB))
print(f"rebuilt: {added_t} segments, {added_v} vias")
rpt = str(HERE / "drc_report.txt")
pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
kinds = collections.Counter(re.findall(r"\[(\w+)\]", Path(rpt).read_text()))
print("DRC top:", dict(kinds.most_common(10)))
print("unconnected:", kinds.get("unconnected_items", 0))
# which nets remain unrouted?
un = re.findall(r'\[unconnected_items\].*?\n.*?net "?([\w\-+.]+)', Path(rpt).read_text())
print("unrouted nets:", dict(collections.Counter(un).most_common(15)))
