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
GATE: schematic v2 ✅, zero black boxes ✅, ERC — **pending KiCad install**
(`sudo apt install kicad`); deferred to P4 entry if not installed sooner.

## P2 — BOM lockdown (0.5 d) [me: web-verified] — **DONE 2026-07-13**
BOM.md: every line MPN + LCSC C# + package + stock + alternate; ~$43-45/board
assembled @ qty 5. Availability-driven substitutions (all design-verified):
LM25145→LM5145RGYR, SQJQ140E→2×CSD18543Q3A, ATtiny816 SOIC→UQFN; RILIM 301/243 Ω.
GATE: no line without MPN + footprint + availability ✅ (order-time re-check list
in BOM.md §G).

## P3 — footprints & mechanical (0.5-1 d) [KiCad]
Land patterns per datasheet (VQFN-20 wettable flanks!), board outline + mounting
from rover chassis, connector orientation vs harness. GATE: footprint checklist.

## P4 — layout (2-3 d) [KiCad; checklist B in PRODUCTION_REVIEW.md]
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
