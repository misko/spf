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

## P1 — electrical detail design (1-2 d) [me + TI Quickstart]
- Run TI Quickstart/WEBENCH for both bucks @ 10-32V→5.1V, 6A & 5A, 600 kHz →
  exact RT/SS/COMP/FET/L/Cin/Cout; transcribe into schematic (kill black boxes)
- Gate-drive trade study: LTC7004 vs LM74800 vs ideal-diode-ctrl integrated CP
- MCU pin map vs ATtiny816 alt functions; supervisor FIRMWARE written + reviewed
  (thresholds, 10 s qualifier, 60 s handshake, I2C regs, USB port-cycle cmds)
GATE: schematic v2, zero black boxes, ERC clean.

## P2 — BOM lockdown (0.5 d) [me: web-verified]
Every line: MPN, footprint, JLC basic/extended, stock, 1 alternate; cost @ qty 5/25.
GATE: no line without MPN + footprint + availability.

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
