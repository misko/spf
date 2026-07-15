# JLCPCB submission — SPF rover power board v4.5 (2026-07-14)

## What to upload where
1. PCB order page  -> power_board_v1_gerbers.zip   (gerbers + drills, 13 files)
2. Assembly: BOM   -> bom_jlc.csv                  (93/96 lines with LCSC codes)
3. Assembly: CPL   -> cpl_jlc.csv                  (132 placements, all top side)

## Order options (MUST match)
- Layers: 4          (stackup auto-detected from gerbers; confirm 4L in form)
- Via/hole: board has 0.25/0.15 mm vias -> select the ADVANCED / small-via
  option (standard min is 0.45/0.2 — order fails or drifts without it)
- Board: 100 x 65 mm, 2 designs no, panel no (single)
- Assembly: top side only

## Preview checklist (before paying)
- Rotation check on polarized/oriented parts (top side only):
  D1,D6,D7 (SMBJ unidirectional TVS), D5 (B5819W), D4 (LED),
  CE1/CA7/CB7 (electrolytic/polymer), all ICs U1-U10, D2/D3/D8 (SOT-23-6),
  J3 (USB-C), JST latch orientation on J2/J7/J8/J11
- GH 2P clone (C225118, J2/J8): confirm land-pattern match in preview
- 1x4 headers (C705184, J9/J10/J14): confirm vertical THT in preview

## Not assembled by JLC (hand-solder after delivery)
- J4, J5, J13: USB-A THT (CNC Tech 1001-011-01101; DigiKey 3064739, $0.86)
- F1 cartridge: 15 A ATO blade fuse (holder IS assembled: C207061)

## Thin-stock lines (verified in stock today — recheck if ordering later)
- C614136 ATtiny1616-SN (U3): 15    | alt C145558 (313, 125C grade)
- C454289 CE1 hybrid: 25            | alt: LCSC consign / Mouser EEH-ZA1V101P
- C2838328 L4 1uH: 159              | C17700181 LA1/LB1 3.3uH: 318
- C485912 LM5145 x2: 584

## Post-delivery firmware note
U3 is ATtiny1616-SN (816-compatible superset): build with -mmcu=attiny1616.

Full decision audit: analysis/jlc_lcsc_decisions_2026-07-14.md (repo).

## Mounting pattern (v4.6 — changed!)
Asymmetric quad, M3 NPTH 3.2mm: H1 (55,55), H3 (55,110), H2 (146.5,74.75),
H4 (131.5,110) in board coords (outline x 50-150, y 50-115; i.e. from the
top-left corner: (5,5), (5,60), (96.5,24.75), (81.5,60)).
H2 takes a button/pan-head M3 ONLY (2.9mm to connector bodies — no washer).
Chassis/standoff plate must match this pattern.

## Operating mode decision (2026-07-14): power-only, no firmware initially
- U3 (ATtiny1616) IS populated but ships unprogrammed — all pins tri-state,
  board runs on hardware defaults. Firmware flash via J11 (SerialUPDI +
  pymcuprog, MCU=attiny1616) any time later; that adds smart low-battery
  cutoff, panel switch, graceful Pi shutdown, telemetry.
- USB ports are POWER-ONLY in this deployment: leave J9/J10/J14 headers
  unconnected (they're populated for future data passthrough — cheap).
- Hardware-default behavior without firmware:
  * board is ON whenever battery >= ~11.05V (front-end ladder UVLO;
    OV cutoff 14.9V) — J2 panel switch is INERT (it's an MCU input).
    Put a physical switch in the battery + lead if needed.
  * crude hardware UV cutoff only — do NOT leave a pack connected
    unattended for long periods; the smart 10.2V/11.7V timed cutoff
    needs firmware.
  * all 3 USB-A ports always on (1.7A/1.7A/3A hardware limits),
    USB-C advertises 5V/3A to the Pi, aux J6 polyfused 2A.
  * no graceful Pi shutdown: power drops at UVLO — shut the Pi down
    manually before killing power when possible.
