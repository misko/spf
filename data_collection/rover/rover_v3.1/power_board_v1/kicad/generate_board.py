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
    # input, top-left: J1 mate faces WEST off-edge (rot 270 = pads vertical,
    # verified pads on-board); battery then flows monotonically west->east
    "J1": (57, 72, 90), "F1": (64, 68, 90), "R20": (74, 62, 0), "U5": (74, 69, 0),
    "J2": (52, 109, 0), "D4": (52, 53, 0),
    # front-end at mid-left (review: no more bottom-left hairpin)
    "Q2": (62, 82, 0), "Q3": (62, 92, 0), "U4": (54, 90, 0),
    "C1": (70, 80, 0), "C2": (70, 84, 0), "C3": (70, 88, 0), "C16": (70, 92, 0),
    "CE1": (92, 74, 0),
    # supervisor
    "U3": (64, 101, 0), "U6": (78, 80, 0), "D5": (84, 64, 0), "U9": (78, 86, 0),
    "J7": (68, 113, 0), "J11": (80, 112, 0), "J8": (88, 112, 0),
    # buck A: Cin column between controller and FET stack; QA1 drain faces west
    "U1": (82, 59, 0), "QA1": (96, 55, 180), "QA2": (96, 63, 0), "LA1": (106, 59, 0),
    "CA51": (90, 53, 90), "CA52": (90, 58, 90), "CA53": (90, 63, 90),
    "CA61": (116, 53, 0), "CA62": (116, 58, 0), "CA63": (116, 63, 0), "CA64": (116, 68, 0),
    "CA7": (122, 72, 90),
    # buck B mirror
    "U2": (82, 92, 0), "QB1": (96, 87, 180), "QB2": (96, 95, 0), "LB1": (106, 92, 0),
    "CB51": (90, 85, 90), "CB52": (90, 90, 90), "CB53": (90, 95, 90),
    
    # rail A output: USB-C moved to TOP edge (rot 0 mate faces north; frees the
    # right edge and clears mounting hole H2)
    "J3": (127, 52, 0), "CA111": (121, 59, 0), "CA112": (121, 63, 0),
    # XT30 fallback mates EAST; pegs now on-board (review blocker fix)
    "J12": (128, 72, 270),
    "J4": (136, 85, 0), "J5": (136, 100, 0),
    # rail B output + USB switches; ESD arrays + RILIM moved next to their ports
    "L4": (112, 80, 0), "CB111": (119, 79, 0), "CB112": (119, 83, 0),
    "U7": (105, 84, 0), "U8": (120, 110, 0), "R16": (101, 82, 0),
    "CB61": (118, 86, 0), "CB62": (118, 91, 0), "CB63": (118, 96, 0), "CB64": (118, 101, 0),
    "CB7": (104, 102.3, 90),
    "J9": (113, 104, 0), "J10": (86, 101, 90), "J6": (99, 113, 0), "F2": (90, 106, 0),
}

# zone grids for everything else: (x_start, y_start, cols, dx, dy)
ZONE_OF = {
    "A": (76, 72, 5, 5.0, 4.0),
    "B": (76, 98, 5, 5.0, 4.0),
    "S": (52, 97, 6, 4.0, 4.0),
    "U": (98, 64, 4, 5.0, 4.0),
}
GRID_REFS = {
    "A": ["RA1", "RA2", "RA3", "RA4", "RA5", "RA6", "CA1", "CA2", "CA3", "CA4",
          "CA8", "CA9", "CA10", "R4", "R5"],
    "B": ["RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "CB1", "CB2", "CB3", "CB4",
          "CB8", "CB9", "CB10", "D2", "D3", "R17"],
    "S": ["R10", "R11", "C4", "R25", "R26", "R27", "R28", "R29", "R30",
          "R18", "R19", "R22", "R1", "R2", "R3", "C7", "C8", "C17"],
    "U": ["C5", "C6", "C9", "C10", "C13", "C14", "R21", "R31", "R32", "R33", "R34", "D1", "R23", "R24", "D6", "D7"],
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

    # power pads connect SOLID to pours (review: thermal-relief spokes starve 5-6A
    # return paths); thermal vias under buck-controller EPs and beside LS-FET sources
    POWER_REFS = {"QA1", "QA2", "QB1", "QB2", "Q2", "Q3", "LA1", "LB1", "L4",
                  "CE1", "CA7", "CB7", "R20", "F1", "J1"}
    for f in board.GetFootprints():
        r = f.GetReference()
        if r in POWER_REFS:
            for pad in f.Pads():
                pad.SetZoneConnection(pcbnew.ZONE_CONNECTION_FULL)
        if r in ("U1", "U2"):
            for pad in f.Pads():
                if pad.GetNumber() == "21":
                    pad.SetZoneConnection(pcbnew.ZONE_CONNECTION_FULL)
    gndcode = netmap["GND"]
    via_sites = []
    for r in ("U1", "U2"):
        f = board.FindFootprintByReference(r)
        c = f.GetPosition()
        for dx, dy in [(-0.6, -0.9), (0.6, -0.9), (-0.6, 0.9), (0.6, 0.9)]:
            via_sites.append((pcbnew.ToMM(c.x) + dx, pcbnew.ToMM(c.y) + dy))
    for r in ("QA2", "QB2", "Q2", "Q3"):
        f = board.FindFootprintByReference(r)
        c = f.GetPosition()
        for dy in (-2.2, 0.0, 2.2):
            via_sites.append((pcbnew.ToMM(c.x) - 3.2, pcbnew.ToMM(c.y) + dy))
    for vx, vy in via_sites:
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pcbnew.VECTOR2I_MM(vx, vy))
        v.SetDrill(pcbnew.FromMM(0.3))
        v.SetWidth(pcbnew.FromMM(0.6))
        v.SetNet(gndcode)
        v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        board.Add(v)

    board.Save(str(OUT))
    print(f"wrote {OUT.name}: {placed} footprints, {len(nets)} nets, "
          f"{len(board.Zones())} GND zones (unfilled), {len(via_sites)} thermal vias")

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
                dx = (bo.GetRight() - bm.GetLeft() + pcbnew.FromMM(0.7)
                      if bm.GetCenter().x >= bo.GetCenter().x
                      else -(bm.GetRight() - bo.GetLeft() + pcbnew.FromMM(0.7)))
                dy = (bo.GetBottom() - bm.GetTop() + pcbnew.FromMM(0.7)
                      if bm.GetCenter().y >= bo.GetCenter().y
                      else -(bm.GetBottom() - bo.GetTop() + pcbnew.FromMM(0.7)))
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
    # refill zones AFTER moves (review fix: stale fill overlapped nudged pads)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
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
