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
