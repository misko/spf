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
    # input + front-end (left edge)
    "J1": (56, 60, 180), "F1": (56, 72, 0), "D1": (62, 79, 0), "R20": (66, 56, 0),
    "Q2": (62, 88, 0), "Q3": (62, 98, 0), "U4": (56, 108, 0),
    "C1": (66, 104, 0), "C2": (66, 108, 0), "C3": (66, 112, 0),
    "R1": (52, 92, 0), "R2": (52, 96, 0), "R3": (52, 100, 0), "J2": (52, 84, 0),
    # buck A (top center): VSW in from left, 5VA out right
    "U1": (88, 62, 0), "QA1": (97, 56, 0), "QA2": (97, 66, 0), "LA1": (106, 60, 0),
    "CA5": (80, 56, 0),  # input caps hug the FET pair
    "CA6": (112, 66, 0), "CA7": (112, 72, 90),
    # buck B (bottom center)
    "U2": (88, 94, 0), "QB1": (97, 88, 0), "QB2": (97, 98, 0), "LB1": (106, 92, 0),
    "CB5": (80, 88, 0), "CB6": (112, 98, 0), "CB7": (112, 104, 90),
    # outputs (right edge)
    "L2": (118, 56, 0), "CA11": (124, 60, 0), "J3": (134, 58, 90),
    "R23": (128, 64, 0), "R24": (128, 68, 0), "J12": (134, 70, 90),
    "L4": (118, 88, 0), "CB11": (124, 92, 0),
    "U7": (118, 98, 0), "U8": (118, 106, 0), "J4": (134, 96, 90), "J5": (134, 106, 90),
    "J6": (124, 112, 0), "J9": (128, 100, 0), "J10": (128, 108, 0),
    # supervisor strip (bottom left) — INA226 within Kelvin reach of R20? R20 is at
    # the input; keep U5 near the shunt instead:
    "U5": (72, 56, 0),
    "U3": (64, 66, 0), "U6": (72, 66, 0), "U9": (72, 74, 0),
    "J7": (56, 113, 0), "J11": (66, 113, 0), "J8": (74, 113, 0),
    "D4": (52, 113, 0),
}

# zone grids for everything else: (x_start, y_start, cols, dx, dy)
ZONE_OF = {
    # buck A small passives
    "A": (76, 70, 5, 5.0, 4.0),
    # buck B small passives
    "B": (76, 102, 5, 5.0, 4.0),
    # supervisor passives
    "S": (52, 104, 6, 4.0, 4.0),
    # usb/aux passives
    "U": (114, 78, 5, 5.0, 4.0),
}
GRID_REFS = {
    "A": ["RA1", "RA2", "RA3", "RA4", "RA5", "RA6", "CA1", "CA2", "CA3", "CA4",
          "CA8", "CA9", "CA10", "R4", "R5"],
    "B": ["RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "CB1", "CB2", "CB3", "CB4",
          "CB8", "CB9", "CB10"],
    "S": ["R10", "R11", "C4", "R25", "R26", "R27", "R28", "R29", "R30",
          "R18", "R19", "R21", "R22"],
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

    board.Save(str(OUT))
    print(f"wrote {OUT.name}: {placed} footprints, {len(nets)} nets")
    unconnected = [k for k in missing_pads if not k[1].startswith("MP")]
    print(f"pads without nets (NC/shield/EP-float expected): {len(missing_pads)}",
          sorted(set(r for r, _ in unconnected)))


if __name__ == "__main__":
    main()
