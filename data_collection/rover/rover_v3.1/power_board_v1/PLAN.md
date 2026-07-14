# Power board v1 → production: staged plan

Each phase has a deliverable and a hard gate; nothing advances past a red gate.
"Me" = doable in Claude sessions here; "KiCad" = needs KiCad (install locally with
`sudo apt install kicad` to unlock in-session ERC/DRC/render iteration); "Vendor" = fab.

## P0 — freeze requirements (0.5 d) [decisions] — **CLOSED 2026-07-13**
1. Battery range: **DECIDED 3S-only** (fixed LPD divider, SMBJ16A TVS, 9-13V buck design point).
2. Motor-path scope: **DECIDED off-board** — board stays 2-layer/~10A; AUX_CTL gates an
   external motor-path FET contactor so motors share on/off + LPD decisions.
3. Connectors: **DECIDED** — XT60 in; Pi via on-board USB-C receptacle (TH shell, 10k Rp,
   strain-relief anchor) + SPECIFIED CABLE Silkland 0.5ft 240W C-to-C (ASIN B0CQ4SX256;
   bring-up validates >=4.9V at Pi under 5A) + XT30 fallback pads; JST-GH signal headers.
4. Assembler: **DECIDED JLCPCB assembly** for qty-5 prototypes (drives P2 part selection:
   basic lib preferred, extended OK, hand-rework possible).
GATE: DESIGN.md rev-locked. ✅

## P1 — electrical detail design (1-2 d) [me + TI Quickstart] — **substantially DONE 2026-07-13**
- ✅ Buck detail math from LM25145 datasheet eqs @ 9-13V→5.1V, 6A & 5A, 606 kHz →
  RT/SS/COMP/FET-class/L/Cin/Cout/RILIM in P1_DETAIL_DESIGN.md §2; transcribed into
  schematic v2 (103 items — black boxes killed). [QS] Quickstart cross-check at P2.
- ✅ Gate-drive trade study → **LM74800-Q1** front-end (P1_DETAIL_DESIGN.md §1)
- ✅ MCU pin map (§3, all 18 signals) + supervisor firmware (firmware/) with
  host-tested pure state machine (10 scenarios, `make test` green)
GATE: schematic v2 ✅, zero black boxes ✅, connectivity ✅ — KiCad 7.0.11 installed
2026-07-13 (generator now emits 7.0 dialect); kicad-cli netlist export resolves
73 nets with zero unexpected single-node nets; schematic PDF exported + PNG
visual pass done (fixed ref/value text collisions + ATtiny symbol width).
NOTE: CLI ERC needs KiCad 8 (`ppa:kicad/kicad-8.0-releases`) — run ERC in the
eeschema GUI or after upgrade; netlist audit covers connectivity meanwhile.

## P2 — BOM lockdown (0.5 d) [me: web-verified] — **DONE 2026-07-13**
BOM.md: every line MPN + LCSC C# + package + stock + alternate; ~$43-45/board
assembled @ qty 5. Availability-driven substitutions (all design-verified):
LM25145→LM5145RGYR, SQJQ140E→2×CSD18543Q3A, ATtiny816 SOIC→UQFN; RILIM 301/243 Ω.
GATE: no line without MPN + footprint + availability ✅ (order-time re-check list
in BOM.md §G).

## P3 — footprints & mechanical (0.5-1 d) [KiCad] — **schematic side DONE 2026-07-13**
✅ All 96 components carry footprint fields (netlist-verified); 2 custom footprints
authored from TI drawings + pcbnew-validated; USB-C symbol uses real GCT pad names;
physical pads throughout (schematic v3). FOOTPRINTS.md lists the 5 open
datasheet-vs-stdlib verifications for the P4 review.
REMAINING: connector orientation vs harness check at P4 close.
MOUNTING RESOLVED 2026-07-13: chassis adapter will be 3D-PRINTED to match the
board — the 4x M3 corner holes (5mm inset) are final. GATE: footprint checklist.

## P4 — layout (2-3 d) [KiCad; checklist B in PRODUCTION_REVIEW.md] — **STARTED 2026-07-13**
✅ Starting board generated (kicad/generate_board.py): 90x65 outline, 4x M3 holes,
all 96 footprints placed in floorplan zones, every pad net-bound (ratsnest live).
Board-gen audit caught + fixed 2 schematic bugs: USBLC6 ESD arrays were 2-pin
symbols (D− line + VBUS pin unprotected/floating → full 6-pin flow-through now);
ATtiny EP now grounded. Only U4's EP floats (required).
ROUTING STATUS 2026-07-13 (FINAL headless state): board is 4-LAYER (mid-review
fallback exercised: In1 = solid GND plane, In2 = power pours VSW/VBATT_S/5V_A/5V_B).
Custom Python maze router (manhattan_route.py: exact-rect clearances, A* +
two-layer 3D maze, micro-joins) + 93 plane-stitch vias: 362 segments + 327 vias
placed; ~110 pad-pair connections REMAIN for interactive pcbnew (~1-1.5 h,
ratsnest-guided; dense clusters where scripted via placement has no legal sites).
freerouting exhausted: no version both reads KiCad-7 4-layer DSN and writes SES
(1.x parser crash / 2.1 empty exporter / 2.2.x reader regression) — documented.
CRITICAL-NET POLISH LIST for the human pass regardless: buck SW nodes + hot
loops, USB D+/- pairs, R20-INA226 Kelvin, FET gate drives.
OLD-: freerouting pipeline (route_board.py -> import_ses.py,
audit_board.py gate) delivered 809 track segments + 89 vias + 68 GND stitching
vias + 3 F.Cu power pours (VSW/VBATT_S/5V_A) + B.Cu ground plane. 136/221
pad-pair connections routed; 85 remain (power tie-ins in carved-pour regions +
congested corridors) -> INTERACTIVE pcbnew session (~1-2 h, ratsnest-guided).
Then: DRC scrub to baseline, re-run audit_board.py (checklist gates 1-4).
FORMER REMAINING (interactive pcbnew): placement refinement → route power loops →
Kelvin/INA pair → USB pairs → logic → GND pour + AGND islands → thermal reliefs.
Order: power loops → Kelvin/INA analog → USB pairs → logic. 2-layer 2 oz; decide
4-layer fallback at mid-review. GATE: DRC clean + checklist B + fresh-eyes review
of gerber renders (agent-reviewable as images).

## P5 — pre-production validation (0.5 d) [me + KiCad + JLC DFM]
ERC/DRC artifacts committed; netlist↔schematic diff; BOM+CPL export; JLC DFM upload.
GATE: assembler DFM pass.

## P6 — prototype run (1-2 wk lead) [vendor; parallel: firmware bench harness (me)]
Qty 5 assembled. Parallel: UPDI flashing rig, I2C test scripts, Pi-side service
(low-batt shutdown daemon + MAVLink battery bridge). GATE: boards in hand.

## P7 — bring-up (1-2 bench days) [checklist D; results committed]
Current-limited first power → rails/Bode/load-step → LPD timing → thermal soak →
EMI vs GPS lock + SDR noise floor → Pi5 integration (usb_max_current_enable=1,
vcgencmd clean). GATE: all items green or dispositioned to rev B.

## P8 — field trial + rev B (1 wk)
One rover converted, multi-mission soak with INA226 telemetry logged into captures.
Rev B from findings; release = tagged fab package (gerbers+BOM+CPL+firmware+docs).

Critical path: P1 → P4 → P6 lead time. Engineering ~5-7 working days; calendar ~3-4
weeks dominated by fab. Parallel tracks: firmware (P1), BOM (P2), Pi-side software
(P6) never block the board.
