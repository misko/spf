"""Generate power_board_v1.kicad_pcb — P4 starting point.

Run with the SYSTEM python (pcbnew API ships with the kicad apt package):
    kicad-cli sch export netlist power_board_v1.kicad_sch -o netlist.net
    /usr/bin/python3 generate_board.py

Produces: 90x65 mm outline @ (50,50)-(140,115), 4x M3 mounting holes, all
components placed in FOOTPRINTS.md floorplan zones (coarse grid inside each
zone — refine interactively), every pad bound to its schematic net so the
ratsnest is complete. Layout order per PRODUCTION_REVIEW.md checklist B:
power loops -> Kelvin/INA analog -> USB pairs -> logic.
"""
import re
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
NETLIST = HERE / "netlist.net"
OUT = HERE / "power_board_v1.kicad_pcb"

STD = "/usr/share/kicad/footprints"
PROJ = str(HERE / "power_board_v1.pretty")

# board frame
X0, Y0, W, H = 50.0, 50.0, 90.0, 65.0

# explicit anchors for the parts whose position drives the power loops
ANCHOR = {
    # input chain, top-left: XT60 (overhangs left edge) -> fuse (vertical) ->
    # shunt + INA Kelvin pair
    "J1": (62, 64, 180), "F1": (71, 68, 90), "R20": (80, 53, 0), "U5": (79, 58, 0),
    "J2": (81, 70, 0), "D1": (80, 64, 0),
    # front-end b2b + LM74800 (left, mid)
    "Q2": (68, 84, 0), "Q3": (68, 96, 0), "U4": (58, 92, 0),
    "C1": (76, 82, 0), "C2": (76, 86, 0), "C3": (76, 90, 0),
    # supervisor (left, lower); U9/D4 are resolver-movable
    "U3": (64, 104, 0), "U6": (56, 84, 0), "U9": (52, 91, 0),
    "J7": (60, 112, 0), "J11": (73, 111.5, 0), "J8": (81, 111.5, 0),
    # buck A (top center) — hot loop: CA5 at FET drains, SW node tight to LA1
    "U1": (86, 60, 0), "QA1": (98, 54, 0), "QA2": (98, 66, 0), "LA1": (108, 58, 0),
    "CA5": (92, 53, 90), "CA6": (108, 66, 0), "CA7": (108, 73, 90),
    # buck B (bottom center), mirror
    "U2": (86, 92, 0), "QB1": (98, 86, 0), "QB2": (98, 98, 0), "LB1": (108, 90, 0),
    "CB5": (92, 85, 90), "CB6": (118, 97, 0),
    # rail A output (top right)
    "L2": (121, 54, 0), "CA11": (128, 53, 0), "J3": (136, 60, 90),
    "R23": (122, 65, 0), "R24": (122, 69, 0),
    # right edge, top->bottom: J3 / XT30 fallback / USB-A x2 (ports overhang)
    "J12": (137, 77, 270),
    "J4": (136, 92, 0), "J5": (136, 108, 0),
    # rail B output + radio USB switches
    "L4": (118, 80, 0), "CB11": (119, 92, 0),
    "U7": (102, 103, 0), "U8": (102, 109, 0),
    "CB7": (108, 101, 90),
    "J9": (114, 102, 90), "J10": (114, 108, 90), "J6": (90, 112, 0),
}

# zone grids for everything else: (x_start, y_start, cols, dx, dy)
ZONE_OF = {
    # buck A small passives (below U1, above grid-free lane)
    "A": (76, 74, 5, 5.0, 4.0),
    # buck B small passives
    "B": (76, 103, 7, 4.0, 4.0),
    # supervisor + front-end ladder passives (left, bottom block)
    "S": (52, 97, 7, 4.0, 4.0),
    # usb/aux passives (between CA7 and the right-edge connectors)
    "U": (84, 64, 2, 6.0, 4.0),
}
GRID_REFS = {
    "A": ["RA1", "RA2", "RA3", "RA4", "RA5", "RA6", "CA1", "CA2", "CA3", "CA4",
          "CA8", "CA9", "CA10", "R4", "R5"],
    "B": ["RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "CB1", "CB2", "CB3", "CB4",
          "CB8", "CB9", "CB10"],
    "S": ["R10", "R11", "C4", "R25", "R26", "R27", "R28", "R29", "R30",
          "R18", "R19", "R21", "R22", "R1", "R2", "R3", "D4"],
    "U": ["R16", "R17", "D2", "D3"],
}


def parse_netlist(path):
    s = path.read_text()
    comps = {}
    for m in re.finditer(r'\(comp \(ref "([^"]+)"\)(.*?)(?=\(comp \(ref|\(libparts)', s, re.S):
        ref, body = m.group(1), m.group(2)
        fp = re.search(r'\(footprint "([^"]*)"\)', body)
        val = re.search(r'\(value "([^"]*)"\)', body)
        comps[ref] = (fp.group(1) if fp else "", val.group(1) if val else "")
    pad_net = {}
    nets = set()
    for m in re.finditer(r'\(net \(code "\d+"\) \(name "([^"]+)"\)(.*?)(?=\(net \(code|\Z)', s, re.S):
        name, body = m.group(1), m.group(2)
        nets.add(name)
        for r, p in re.findall(r'\(node \(ref "([^"]+)"\) \(pin "([^"]+)"\)', body):
            pad_net[(r, p)] = name
    return comps, pad_net, nets


def load_fp(fpid):
    lib, name = fpid.split(":")
    root = PROJ if lib == "power_board_v1" else f"{STD}/{lib}.pretty"
    fp = pcbnew.FootprintLoad(root, name)
    if fp is None:
        raise RuntimeError(f"footprint not found: {fpid}")
    return fp


def main():
    comps, pad_net, nets = parse_netlist(NETLIST)
    board = pcbnew.BOARD()

    netmap = {}
    for n in sorted(nets):
        ni = pcbnew.NETINFO_ITEM(board, n)
        board.Add(ni)
        netmap[n] = ni

    # outline
    for (xa, ya, xb, yb) in [
        (X0, Y0, X0 + W, Y0), (X0 + W, Y0, X0 + W, Y0 + H),
        (X0 + W, Y0 + H, X0, Y0 + H), (X0, Y0 + H, X0, Y0),
    ]:
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I_MM(xa, ya))
        seg.SetEnd(pcbnew.VECTOR2I_MM(xb, yb))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(pcbnew.FromMM(0.15))
        board.Add(seg)

    # M3 mounting holes, 5 mm in from corners
    for i, (hx, hy) in enumerate(
        [(X0 + 5, Y0 + 5), (X0 + W - 5, Y0 + 5), (X0 + 5, Y0 + H - 5),
         (X0 + W - 5, Y0 + H - 5)], 1
    ):
        mh = pcbnew.FootprintLoad(f"{STD}/MountingHole.pretty", "MountingHole_3.2mm_M3")
        mh.SetReference(f"H{i}")
        mh.SetPosition(pcbnew.VECTOR2I_MM(hx, hy))
        board.Add(mh)

    # grid positions for non-anchored refs
    gridpos = {}
    for zone, refs in GRID_REFS.items():
        gx, gy, cols, dx, dy = ZONE_OF[zone]
        for i, ref in enumerate(refs):
            gridpos[ref] = (gx + (i % cols) * dx, gy + (i // cols) * dy, 0)

    placed, missing_pads = 0, []
    for ref, (fpid, val) in sorted(comps.items()):
        if not fpid:
            print(f"SKIP {ref}: no footprint")
            continue
        fp = load_fp(fpid)
        fp.SetReference(ref)
        fp.SetValue(val)
        x, y, rot = ANCHOR.get(ref) or gridpos.get(ref) or (X0 + 2, Y0 - 6, 0)
        if ref not in ANCHOR and ref not in gridpos:
            print(f"UNPLACED (parked above board): {ref}")
        fp.SetPosition(pcbnew.VECTOR2I_MM(x, y))
        if rot:
            fp.SetOrientationDegrees(rot)
        for pad in fp.Pads():
            key = (ref, pad.GetNumber())
            if key in pad_net:
                pad.SetNet(netmap[pad_net[key]])
            elif pad.GetNumber():
                missing_pads.append(key)
        board.Add(fp)
        placed += 1

    # design rules (JLC 2-layer 2oz capable: 0.2 clearance, 0.2 min track)
    ds = board.GetDesignSettings()
    ds.m_MinClearance = pcbnew.FromMM(0.2)
    ds.m_TrackMinWidth = pcbnew.FromMM(0.2)
    ds.m_ViasMinSize = pcbnew.FromMM(0.6)
    ds.m_MinThroughDrill = pcbnew.FromMM(0.3)

    # GND pours, both layers (2-layer: B.Cu = reference plane, F.Cu = stitched fill)
    gnd = netmap.get("GND")
    for layer in (pcbnew.B_Cu, pcbnew.F_Cu):
        z = pcbnew.ZONE(board)
        z.SetLayer(layer)
        z.SetNet(gnd)
        z.SetLocalClearance(pcbnew.FromMM(0.25))
        z.SetMinThickness(pcbnew.FromMM(0.25))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        z.SetThermalReliefGap(pcbnew.FromMM(0.3))
        z.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.4))
        out = z.Outline()
        out.NewOutline()
        for x, y in [(X0 + 0.5, Y0 + 0.5), (X0 + W - 0.5, Y0 + 0.5),
                     (X0 + W - 0.5, Y0 + H - 0.5), (X0 + 0.5, Y0 + H - 0.5)]:
            out.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
        board.Add(z)

    board.Save(str(OUT))
    print(f"wrote {OUT.name}: {placed} footprints, {len(nets)} nets, "
          f"{len(board.Zones())} GND zones (unfilled)")

    # reload from disk (attaches a project — ZONE_FILLER/DRC segfault without one)
    board = pcbnew.LoadBoard(str(OUT))
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    board.Save(str(OUT))
    print("zones filled")
    unconnected = [k for k in missing_pads if not k[1].startswith("MP")]
    print(f"pads without nets (NC/shield/EP-float expected): {len(missing_pads)}",
          sorted(set(r for r, _ in unconnected)))

    # greedy overlap resolver: anchors (power loops/connectors) stay fixed; the
    # movable part of each colliding pair is nudged apart until bboxes clear
    fixed = set(ANCHOR) | {"H1", "H2", "H3", "H4"}
    fps = {f.GetReference(): f for f in board.GetFootprints()}

    def bb(f):
        return f.GetBoundingBox(False, False)

    for _ in range(200):
        moved = False
        refs = sorted(fps)
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                a, b = fps[refs[i]], fps[refs[j]]
                ba, bbx = bb(a), bb(b)
                if not ba.Intersects(bbx):
                    continue
                mv = b if refs[j] not in fixed else (a if refs[i] not in fixed else None)
                if mv is None:
                    continue
                other = a if mv is b else b
                bm, bo = bb(mv), bb(other)
                # push along the axis of least separation, plus margin
                dx = (bo.GetRight() - bm.GetLeft() + pcbnew.FromMM(0.4)
                      if bm.GetCenter().x >= bo.GetCenter().x
                      else -(bm.GetRight() - bo.GetLeft() + pcbnew.FromMM(0.4)))
                dy = (bo.GetBottom() - bm.GetTop() + pcbnew.FromMM(0.4)
                      if bm.GetCenter().y >= bo.GetCenter().y
                      else -(bm.GetBottom() - bo.GetTop() + pcbnew.FromMM(0.4)))
                p = mv.GetPosition()
                if abs(dx) <= abs(dy):
                    p.x += dx
                else:
                    p.y += dy
                # clamp inside the board frame
                p.x = max(pcbnew.FromMM(X0 + 3), min(pcbnew.FromMM(X0 + W - 3), p.x))
                p.y = max(pcbnew.FromMM(Y0 + 3), min(pcbnew.FromMM(Y0 + H - 3), p.y))
                mv.SetPosition(p)
                moved = True
        if not moved:
            break
    board.Save(str(OUT))

    boxes = [(r, bb(f)) for r, f in fps.items() if not r.startswith("H")]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].Intersects(boxes[j][1]):
                overlaps.append((boxes[i][0], boxes[j][0]))
    print(f"bbox overlaps after resolver ({len(overlaps)}):", overlaps[:20])

    # headless DRC (KiCad 7 python API)
    rpt = str(HERE / "drc_report.txt")
    pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
    txt = Path(rpt).read_text()
    import collections
    kinds = collections.Counter(re.findall(r"\[(\w+)\]", txt))
    print("DRC violation kinds:", dict(kinds))


if __name__ == "__main__":
    main()
