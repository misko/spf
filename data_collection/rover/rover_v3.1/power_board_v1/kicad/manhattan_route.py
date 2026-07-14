"""Deterministic Manhattan router for the 4-layer board.

Strategy: B.Cu is EMPTY (GND lives on In1, power on In2), so every unconnected
pair routes as F.Cu stub -> via -> B.Cu L-path -> via -> F.Cu stub, with pure
geometry clearance checks; short pairs go direct on F.Cu when the corridor is
clear. KiCad DRC is the final referee (run audit_board.py after).

    /usr/bin/python3 manhattan_route.py
"""
import collections
import math
import re
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
PCB = HERE / "power_board_v1.kicad_pcb"

CLR = 0.21          # clearance to foreign copper, mm (rule is 0.2)
SIG_W = 0.3         # signal width
PWR_W = 1.2         # power width
PWR_NETS = {"VSW", "VBATT_RAW", "VBATT_F", "VBATT_S", "FE_MID", "5V_A", "5V_B",
            "5VB_PRE", "AUX_5V", "VBUS1", "VBUS2", "VBATT_FD", "SW_A", "SW_B"}
VIA_W, VIA_DRILL = 0.6, 0.3
X0, Y0, W, H = 50.0, 50.0, 90.0, 65.0

board = pcbnew.LoadBoard(str(PCB))
mm = pcbnew.ToMM

# fresh-route mode: drop all tracks (keep vias: plane stitches + thermals),
# so long hauls route through an open board (greedy passes walled them off)
import os
if os.environ.get("WIPE_TRACKS") == "1":
    n = 0
    for t in list(board.GetTracks()):
        if type(t).__name__ == "PCB_TRACK":
            board.Remove(t)
            n += 1
    print(f"wiped {n} track segments (vias kept)")
    board.Save(str(PCB))
    board = pcbnew.LoadBoard(str(PCB))


def seg_pt_dist(ax, ay, bx, by, px, py):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def seg_seg_dist(a, b):
    (ax, ay, bx, by), (cx, cy, dx, dy) = a, b
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    if (ccw((ax, ay), (bx, by), (cx, cy)) * ccw((ax, ay), (bx, by), (dx, dy)) < 0 and
            ccw((cx, cy), (dx, dy), (ax, ay)) * ccw((cx, cy), (dx, dy), (bx, by)) < 0):
        return 0.0
    return min(seg_pt_dist(ax, ay, bx, by, cx, cy), seg_pt_dist(ax, ay, bx, by, dx, dy),
               seg_pt_dist(cx, cy, dx, dy, ax, ay), seg_pt_dist(cx, cy, dx, dy, bx, by))


# ---------------- obstacle model ------------------------------------------
# pads as exact rectangles: (x, y, hx, hy, net, blocks_bcu). Rotated pads use
# their bbox (conservative but axis-aligned, which our tracks are too).
pads = []
for f in board.GetFootprints():
    for p in f.Pads():
        bb = p.GetBoundingBox()
        cx, cy = mm(bb.GetCenter().x), mm(bb.GetCenter().y)
        hx, hy = mm(bb.GetWidth()) / 2, mm(bb.GetHeight()) / 2
        pth = p.GetAttribute() != pcbnew.PAD_ATTRIB_SMD
        pads.append((cx, cy, hx, hy, p.GetNetname(), pth))


def seg_rect_dist(x1, y1, x2, y2, cx, cy, hx, hy):
    """exact distance segment <-> axis-aligned rectangle"""
    if (min(x1, x2) <= cx + hx and max(x1, x2) >= cx - hx and
            min(y1, y2) <= cy + hy and max(y1, y2) >= cy - hy):
        # endpoint inside?
        for (px, py) in ((x1, y1), (x2, y2)):
            if abs(px - cx) <= hx and abs(py - cy) <= hy:
                return 0.0
    corners = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    d = min(seg_seg_dist((x1, y1, x2, y2), (e[0][0], e[0][1], e[1][0], e[1][1]))
            for e in edges)
    return d

# existing tracks/vias per layer: (x1,y1,x2,y2,halfwidth,net)
occ = {pcbnew.F_Cu: [], pcbnew.B_Cu: []}
via_pts = []  # (x, y, net)
for t in board.GetTracks():
    if type(t).__name__ == "PCB_VIA":
        p = t.GetPosition()
        via_pts.append((mm(p.x), mm(p.y), t.GetNetname()))
        for lay in occ:
            occ[lay].append((mm(p.x), mm(p.y), mm(p.x), mm(p.y),
                             mm(t.GetWidth()) / 2, t.GetNetname()))
    else:
        s, e = t.GetStart(), t.GetEnd()
        if t.GetLayer() in occ:
            occ[t.GetLayer()].append((mm(s.x), mm(s.y), mm(e.x), mm(e.y),
                                      mm(t.GetWidth()) / 2, t.GetNetname()))


def seg_ok(layer, x1, y1, x2, y2, halfw, net):
    lo_x, hi_x = X0 + 0.6, X0 + W - 0.6
    lo_y, hi_y = Y0 + 0.6, Y0 + H - 0.6
    for (x, y) in ((x1, y1), (x2, y2)):
        if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
            return False
    for (px, py, hx, hy, pnet, pth) in pads:
        if pnet == net:
            continue
        if layer == pcbnew.B_Cu and not pth:
            continue
        if seg_rect_dist(x1, y1, x2, y2, px, py, hx, hy) < halfw + CLR:
            return False
    for (ax, ay, bx, by, ohw, onet) in occ[layer]:
        if onet == net:
            continue
        if seg_seg_dist((x1, y1, x2, y2), (ax, ay, bx, by)) < ohw + halfw + CLR:
            return False
    return True


def via_ok(x, y, net):
    for (px, py, hx, hy, pnet, pth) in pads:
        if pnet != net and seg_rect_dist(x, y, x, y, px, py, hx, hy) < VIA_W / 2 + CLR:
            return False
    for (vx, vy, vnet) in via_pts:
        if math.hypot(x - vx, y - vy) < 0.58 and vnet != net:
            return False
        if math.hypot(x - vx, y - vy) < 0.5 and vnet == net:
            return False
    for lay in occ:
        for (ax, ay, bx, by, ohw, onet) in occ[lay]:
            if onet == net and seg_pt_dist(ax, ay, bx, by, x, y) < 0.2:
                continue
            if onet != net and seg_pt_dist(ax, ay, bx, by, x, y) < ohw + VIA_W / 2 + CLR:
                return False
    return True


def add_track(layer, x1, y1, x2, y2, w, net_obj):
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pcbnew.VECTOR2I_MM(x1, y1))
    t.SetEnd(pcbnew.VECTOR2I_MM(x2, y2))
    t.SetWidth(pcbnew.FromMM(w))
    t.SetLayer(layer)
    t.SetNet(net_obj)
    board.Add(t)
    occ[layer].append((x1, y1, x2, y2, w / 2, net_obj.GetNetname()))


def add_via(x, y, net_obj):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pcbnew.VECTOR2I_MM(x, y))
    v.SetWidth(pcbnew.FromMM(VIA_W))
    v.SetDrill(pcbnew.FromMM(VIA_DRILL))
    v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    v.SetNet(net_obj)
    board.Add(v)
    via_pts.append((x, y, net_obj.GetNetname()))
    for lay in occ:
        occ[lay].append((x, y, x, y, VIA_W / 2, net_obj.GetNetname()))


def l_paths(x1, y1, x2, y2):
    """candidate corner points for L-routes, with lane offsets"""
    for off in (0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0):
        yield (x2 + off if abs(x2 - x1) > 0.01 else x2, y1 + (off if abs(x2 - x1) <= 0.01 else 0))
        yield (x1 + (0 if abs(y2 - y1) > 0.01 else off), y2 + (off if abs(y2 - y1) <= 0.01 else 0))
        yield (x2, y1 + off)
        yield (x1, y2 + off)



# ---------------- A* maze router on B.Cu for long hauls --------------------
GRID = 0.25  # mm/cell


def build_grid(halfw, net):
    nx, ny = int(W / GRID) + 1, int(H / GRID) + 1
    blocked = bytearray(nx * ny)
    infl = halfw + CLR
    def block_rect(cx, cy, hx, hy):
        x0i = max(0, int((cx - hx - infl - X0) / GRID))
        x1i = min(nx - 1, int((cx + hx + infl - X0) / GRID) + 1)
        y0i = max(0, int((cy - hy - infl - Y0) / GRID))
        y1i = min(ny - 1, int((cy + hy + infl - Y0) / GRID) + 1)
        for yi in range(y0i, y1i + 1):
            base = yi * nx
            for xi in range(x0i, x1i + 1):
                blocked[base + xi] = 1
    for (px, py, hx, hy, pnet, pth) in pads:
        if pnet != net and pth:
            block_rect(px, py, hx, hy)
    for (sx, sy, ex, ey, ohw, onet) in occ[pcbnew.B_Cu]:
        if onet == net:
            continue
        steps = max(1, int(math.hypot(ex - sx, ey - sy) / GRID))
        for i in range(steps + 1):
            t = i / steps
            block_rect(sx + (ex - sx) * t, sy + (ey - sy) * t, ohw, ohw)
    # board margin
    m = int(0.8 / GRID)
    for yi in range(ny):
        for xi in list(range(m)) + list(range(nx - m, nx)):
            blocked[yi * nx + xi] = 1
    for xi in range(nx):
        for yi in list(range(m)) + list(range(ny - m, ny)):
            blocked[yi * nx + xi] = 1
    return blocked, nx, ny


import heapq


def astar(blocked, nx, ny, sx, sy, tx, ty):
    si = (int((sy - Y0) / GRID), int((sx - X0) / GRID))
    ti = (int((ty - Y0) / GRID), int((tx - X0) / GRID))
    # clear a landing pad around endpoints (they are our own via sites)
    for (cy, cx) in (si, ti):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < ny and 0 <= xx < nx:
                    blocked[yy * nx + xx] = 0
    openq = [(0, si, None)]
    came, cost = {}, {si: 0}
    while openq:
        _, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == ti:
            break
        cy, cx = cur
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = cy + dy, cx + dx
            if not (0 <= yy < ny and 0 <= xx < nx) or blocked[yy * nx + xx]:
                continue
            # bend penalty encourages straight runs
            bend = 0
            if parent is not None:
                pdy, pdx = cy - parent[0], cx - parent[1]
                if (pdy, pdx) != (dy, dx):
                    bend = 3
            g = cost[cur] + 1 + bend
            nxt = (yy, xx)
            if g < cost.get(nxt, 1 << 30):
                cost[nxt] = g
                h = abs(yy - ti[0]) + abs(xx - ti[1])
                heapq.heappush(openq, (g + h, nxt, cur))
    if ti not in came:
        return None
    path = [ti]
    while came[path[-1]] is not None:
        path.append(came[path[-1]])
    path.reverse()
    # collapse collinear
    pts = [path[0]]
    for i in range(1, len(path) - 1):
        (ay_, ax_), (by_, bx_), (cy_, cx_) = path[i - 1], path[i], path[i + 1]
        if (by_ - ay_, bx_ - ax_) != (cy_ - by_, cx_ - bx_):
            pts.append(path[i])
    pts.append(path[-1])
    return [(X0 + xi * GRID, Y0 + yi * GRID) for (yi, xi) in pts]


def astar3d(bF, bB, nx, ny, sx, sy, tx, ty, net, w):
    """A* over (layer, y, x); layer flip = via (checked with via_ok at commit)"""
    si = (1, int((sy - Y0) / GRID), int((sx - X0) / GRID))   # start on B
    ti = (1, int((ty - Y0) / GRID), int((tx - X0) / GRID))
    grids = {0: bF, 1: bB}
    for (cy, cx) in ((si[1], si[2]), (ti[1], ti[2])):
        for g in grids.values():
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        g[yy * nx + xx] = 0
    openq = [(0, si, None)]
    came, cost = {}, {si: 0}
    while openq:
        _, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur[1:] == ti[1:]:
            ti = cur
            break
        lay, cy, cx = cur
        for dl, dy, dx, step in ((0, 0, 1, 1), (0, 0, -1, 1), (0, 1, 0, 1), (0, -1, 0, 1),
                                 (1 - 2 * lay, 0, 0, 14)):
            nl, yy, xx = lay + dl, cy + dy, cx + dx
            if not (0 <= yy < ny and 0 <= xx < nx) or grids[nl][yy * nx + xx]:
                continue
            bend = 0
            if parent is not None and dl == 0:
                if (cy - parent[1], cx - parent[2]) != (dy, dx):
                    bend = 3
            g = cost[cur] + step + bend
            nxt = (nl, yy, xx)
            if g < cost.get(nxt, 1 << 30):
                cost[nxt] = g
                h = abs(yy - ti[1]) + abs(xx - ti[2])
                heapq.heappush(openq, (g + h, nxt, cur))
    if ti not in came:
        return None
    path = [ti]
    while came[path[-1]] is not None:
        path.append(came[path[-1]])
    path.reverse()
    # collapse collinear same-layer runs
    pts = [path[0]]
    for i in range(1, len(path) - 1):
        p0, p1, p2 = path[i - 1], path[i], path[i + 1]
        if p0[0] != p1[0] or p1[0] != p2[0] or            (p1[1] - p0[1], p1[2] - p0[2]) != (p2[1] - p1[1], p2[2] - p1[2]):
            pts.append(path[i])
    pts.append(path[-1])
    return [(l, X0 + xi * GRID, Y0 + yi * GRID) for (l, yi, xi) in pts]


def route_maze3d(net, net_obj, w, ax, ay, bx, by, stubs):
    LAYS = {0: pcbnew.F_Cu, 1: pcbnew.B_Cu}
    for (sax, say) in stubs[:14]:
        vax, vay = ax + sax, ay + say
        if not via_ok(vax, vay, net):
            continue
        if (sax or say) and not seg_ok(pcbnew.F_Cu, ax, ay, vax, vay, w / 2, net):
            continue
        for (sbx, sby) in stubs[:14]:
            vbx, vby = bx + sbx, by + sby
            if not via_ok(vbx, vby, net):
                continue
            if (sbx or sby) and not seg_ok(pcbnew.F_Cu, bx, by, vbx, vby, w / 2, net):
                continue
            bB, nx, ny = build_grid(w / 2, net)
            bF, _, _ = build_grid_layer(pcbnew.F_Cu, w / 2, net)
            path = astar3d(bF, bB, nx, ny, vax, vay, vbx, vby, net, w)
            if path is None:
                continue
            nodes = [(1, vax, vay)] + path[1:-1] + [(1, vbx, vby)]
            ok = True
            for i in range(len(nodes) - 1):
                (l1, x1, y1), (l2, x2, y2) = nodes[i], nodes[i + 1]
                if l1 == l2:
                    if not seg_ok(LAYS[l1], x1, y1, x2, y2, w / 2, net):
                        ok = False
                        break
                else:
                    if not via_ok(x1, y1, net):
                        ok = False
                        break
            if not ok:
                continue
            if sax or say:
                add_track(pcbnew.F_Cu, ax, ay, vax, vay, w, net_obj)
            add_via(vax, vay, net_obj)
            for i in range(len(nodes) - 1):
                (l1, x1, y1), (l2, x2, y2) = nodes[i], nodes[i + 1]
                if l1 == l2:
                    add_track(LAYS[l1], x1, y1, x2, y2, w, net_obj)
                else:
                    add_via(x1, y1, net_obj)
            add_via(vbx, vby, net_obj)
            if sbx or sby:
                add_track(pcbnew.F_Cu, vbx, vby, bx, by, w, net_obj)
            return "B-3d"
    return None


def build_grid_layer(layer, halfw, net):
    nx, ny = int(W / GRID) + 1, int(H / GRID) + 1
    blocked = bytearray(nx * ny)
    infl = halfw + CLR
    def block_rect(cx, cy, hx, hy):
        x0i = max(0, int((cx - hx - infl - X0) / GRID))
        x1i = min(nx - 1, int((cx + hx + infl - X0) / GRID) + 1)
        y0i = max(0, int((cy - hy - infl - Y0) / GRID))
        y1i = min(ny - 1, int((cy + hy + infl - Y0) / GRID) + 1)
        for yi in range(y0i, y1i + 1):
            base = yi * nx
            for xi in range(x0i, x1i + 1):
                blocked[base + xi] = 1
    for (px, py, hx, hy, pnet, pth) in pads:
        if pnet != net and (pth or layer == pcbnew.F_Cu):
            block_rect(px, py, hx, hy)
    for (sx, sy, ex, ey, ohw, onet) in occ[layer]:
        if onet == net:
            continue
        steps = max(1, int(math.hypot(ex - sx, ey - sy) / GRID))
        for i in range(steps + 1):
            t = i / steps
            block_rect(sx + (ex - sx) * t, sy + (ey - sy) * t, ohw, ohw)
    m = int(0.8 / GRID)
    for yi in range(ny):
        for xi in list(range(m)) + list(range(nx - m, nx)):
            blocked[yi * nx + xi] = 1
    for xi in range(nx):
        for yi in list(range(m)) + list(range(ny - m, ny)):
            blocked[yi * nx + xi] = 1
    return blocked, nx, ny


def route_maze(net, net_obj, w, ax, ay, bx, by, stubs):
    for (sax, say) in stubs[:20]:
        vax, vay = ax + sax, ay + say
        if not via_ok(vax, vay, net):
            continue
        if (sax or say) and not seg_ok(pcbnew.F_Cu, ax, ay, vax, vay, w / 2, net):
            continue
        for (sbx, sby) in stubs[:20]:
            vbx, vby = bx + sbx, by + sby
            if not via_ok(vbx, vby, net):
                continue
            if (sbx or sby) and not seg_ok(pcbnew.F_Cu, bx, by, vbx, vby, w / 2, net):
                continue
            blocked, nx, ny = build_grid(w / 2, net)
            path = astar(blocked, nx, ny, vax, vay, vbx, vby)
            if path is None:
                continue
            # verify each simplified segment with exact geometry (grid is coarse)
            segs = [(vax, vay)] + path[1:-1] + [(vbx, vby)]
            ok = all(seg_ok(pcbnew.B_Cu, segs[i][0], segs[i][1],
                            segs[i + 1][0], segs[i + 1][1], w / 2, net)
                     for i in range(len(segs) - 1))
            if not ok:
                continue
            if sax or say:
                add_track(pcbnew.F_Cu, ax, ay, vax, vay, w, net_obj)
            add_via(vax, vay, net_obj)
            for i in range(len(segs) - 1):
                add_track(pcbnew.B_Cu, segs[i][0], segs[i][1],
                          segs[i + 1][0], segs[i + 1][1], w, net_obj)
            add_via(vbx, vby, net_obj)
            if sbx or sby:
                add_track(pcbnew.F_Cu, vbx, vby, bx, by, w, net_obj)
            return "B-astar"
    return None


def route_pair(net, ax, ay, bx, by):
    net_obj = board.FindNet(net)
    w = PWR_W if net in PWR_NETS else SIG_W
    # micro-joins between same-net pads narrow to fit the pad field
    d = abs(ax - bx) + abs(ay - by)
    if d < 1.2:
        w = min(w, 0.25)
    elif d < 2.5:
        w = min(w, 0.35)
    # 1) direct F.Cu
    if seg_ok(pcbnew.F_Cu, ax, ay, bx, by, w / 2, net):
        add_track(pcbnew.F_Cu, ax, ay, bx, by, w, net_obj)
        return "F-direct"
    # 2) F.Cu L
    for (cx, cy) in ((bx, ay), (ax, by)):
        if (seg_ok(pcbnew.F_Cu, ax, ay, cx, cy, w / 2, net) and
                seg_ok(pcbnew.F_Cu, cx, cy, bx, by, w / 2, net)):
            add_track(pcbnew.F_Cu, ax, ay, cx, cy, w, net_obj)
            add_track(pcbnew.F_Cu, cx, cy, bx, by, w, net_obj)
            return "F-L"
    # 3) via -> B.Cu manhattan -> via, with escape-stub search around each end
    stubs = [(0, 0)] + [(r * dx, r * dy) for r in (0.8, 1.4, 2.0, 2.8, 3.6)
                        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                         (0.7, 0.7), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7))]
    for (sax, say) in stubs:
        vax, vay = ax + sax, ay + say
        if not via_ok(vax, vay, net):
            continue
        if (sax or say) and not seg_ok(pcbnew.F_Cu, ax, ay, vax, vay, w / 2, net):
            continue
        for (sbx, sby) in stubs:
            vbx, vby = bx + sbx, by + sby
            if abs(vax - vbx) < 0.3 and abs(vay - vby) < 0.3:
                continue
            if not via_ok(vbx, vby, net):
                continue
            if (sbx or sby) and not seg_ok(pcbnew.F_Cu, bx, by, vbx, vby, w / 2, net):
                continue
            candidates = list(l_paths(vax, vay, vbx, vby))
            mx, my = (vax + vbx) / 2, (vay + vby) / 2
            zpaths = []
            for off in (0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0):
                zpaths.append([(vax, my + off), (vbx, my + off)])
                zpaths.append([(mx + off, vay), (mx + off, vby)])
            for zp in zpaths:
                (z1x, z1y), (z2x, z2y) = zp
                if (seg_ok(pcbnew.B_Cu, vax, vay, z1x, z1y, w / 2, net) and
                        seg_ok(pcbnew.B_Cu, z1x, z1y, z2x, z2y, w / 2, net) and
                        seg_ok(pcbnew.B_Cu, z2x, z2y, vbx, vby, w / 2, net)):
                    if (sax or say):
                        add_track(pcbnew.F_Cu, ax, ay, vax, vay, w, net_obj)
                    add_via(vax, vay, net_obj)
                    add_track(pcbnew.B_Cu, vax, vay, z1x, z1y, w, net_obj)
                    add_track(pcbnew.B_Cu, z1x, z1y, z2x, z2y, w, net_obj)
                    add_track(pcbnew.B_Cu, z2x, z2y, vbx, vby, w, net_obj)
                    add_via(vbx, vby, net_obj)
                    if (sbx or sby):
                        add_track(pcbnew.F_Cu, vbx, vby, bx, by, w, net_obj)
                    return "B-Z"
            for (cx, cy) in candidates:
                if (seg_ok(pcbnew.B_Cu, vax, vay, cx, cy, w / 2, net) and
                        seg_ok(pcbnew.B_Cu, cx, cy, vbx, vby, w / 2, net)):
                    if sax or say:
                        add_track(pcbnew.F_Cu, ax, ay, vax, vay, w, net_obj)
                    add_via(vax, vay, net_obj)
                    add_track(pcbnew.B_Cu, vax, vay, cx, cy, w, net_obj)
                    add_track(pcbnew.B_Cu, cx, cy, vbx, vby, w, net_obj)
                    add_via(vbx, vby, net_obj)
                    if sbx or sby:
                        add_track(pcbnew.F_Cu, vbx, vby, bx, by, w, net_obj)
                    return "B-manhattan"
    # 4) full maze route on B.Cu
    stubs2 = [(0, 0)] + [(r * dx, r * dy) for r in (0.8, 1.4, 2.2, 3.0)
                         for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                          (0.7, 0.7), (-0.7, -0.7))]
    r = route_maze(net, net_obj, w, ax, ay, bx, by, stubs2)
    if r:
        return r
    return route_maze3d(net, net_obj, w, ax, ay, bx, by, stubs2)


# ---------------- unconnected pairs from DRC -------------------------------
rpt = "/tmp/mr_drc.txt"
pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
txt = Path(rpt).read_text()
pairs = []
for mviol in re.finditer(
        r'\[unconnected_items\][^\n]*\n[^\n]*\n\s+@\(([\d.]+) mm, ([\d.]+) mm\):[^\n]*\[(\w+)\][^\n]*\n\s+@\(([\d.]+) mm, ([\d.]+) mm\)', txt):
    x1, y1, net, x2, y2 = mviol.groups()
    pairs.append((net, float(x1), float(y1), float(x2), float(y2)))
pairs.sort(key=lambda p: -(abs(p[1] - p[3]) + abs(p[2] - p[4])))  # longest first
print(f"{len(pairs)} unconnected pairs to route")

done = collections.Counter()
failed = []
for (net, ax, ay, bx, by) in pairs:
    kind = route_pair(net, ax, ay, bx, by)
    if kind:
        done[kind] += 1
    else:
        failed.append((net, round(ax, 1), round(ay, 1), round(bx, 1), round(by, 1)))
print("routed:", dict(done), "| failed:", len(failed))
for f_ in failed[:12]:
    print("  FAILED", f_)

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
board.Save(str(PCB))
pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, False)
kinds = collections.Counter(re.findall(r"\[(\w+)\]", Path(rpt).read_text()))
print("post-route DRC:", {k: kinds[k] for k in
      ("unconnected_items", "clearance", "shorting_items", "hole_clearance",
       "copper_edge_clearance", "starved_thermal") if k in kinds})
