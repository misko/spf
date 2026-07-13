# Adversarial review 2026-07-13 — findings & disposition

Three parallel adversarial reviewers (electrical / PCB-DFM / consistency). Full
reports in the session transcript. The pin-level netlist survived all datasheet
attacks (7 ICs verified); findings clustered in decoupling, cross-domain nets,
BOM quantity modeling, and placement.

## Fixed in schematic v4 (generate_schematic.py)
| Finding | Fix |
|---|---|
| BLOCKER: zero capacitance on MCU_3V3/3V3_SW/LDO pins (TPS7A16 + AP2112 datasheet "must") | C5-C10 added (LDO in/out, MCU VDD, INA226 VS) |
| BLOCKER: INA226 V_SCL abs-max = VS+0.3V violated whenever rail B off (pull-ups were on always-on MCU_3V3); also back-powered the dead Pi | R18/R19 (I2C) + R22 (PGOOD_B) moved to switched 3V3_SW |
| BLOCKER: bulk caps were single symbols with "3x10u" text values — BOM/CPL export would fab 1/3 of designed capacitance | CA51-53/CB51-53 (10u), CA61-64/CB61-64 (47u), CA111/112, CB111/112 (22u) — every physical cap is its own ref (127 components now) |
| MAJOR: my first pull-up fix created a bootstrap deadlock (PGOOD_A pulled to 3V3_SW which derives from buck B whose EN *is* PGOOD_A) | Per LM5145 fig 8-4: R21 20k to 5V_A (master out) + R33/R34 divider → EN_B ≈2.3V and 3.3V-safe PGOOD_A for MCU/Pi |
| MAJOR: rail A sensed upstream of pi-filter; Pi 4.75V budget failed (~4.71V @5A worst) | L2 deleted (pi-filter was for radios only); rail A senses 5V_A directly; setpoint 5.18V (RFB2 3k65). Rail B keeps filter (SDR ripple), 5.08V |
| MAJOR: rails glitch ON at pack plug-in / MCU reset (FE_EN released at boot) | C16 2u2 on FE_EN (~450ms divider delay) + firmware drives FE_EN low as FIRST reset action |
| MINOR: rail A OCP worst-case (IRDSON 180uA, RDS 9.9-11 mOhm) below 6A rating | RILIM A 348R (wc-min 6.3A), B 261R (wc-min 4.8A) |
| MINOR: TPS2553 ENs floated without MCU; missing IN caps | R31/R32 100k to 5V_B; C13/C14 100n |
| MINOR: no input bulk/damping; LM74800 A-pin cap missing | CE1 100u hybrid on VSW (behind soft-start = no hot-plug inrush); C15 100n at U4.A |
| MINOR: reverse battery exceeds TPS7A16 IN -0.3V (TVS conducts ~-1V until fuse clears) | D5 Schottky into U6.IN (net VBATT_FD) |
| MINOR: aux short drops both radios; DESIGN.md wants per-rail 5V TVS | F2 2A polyfuse → AUX_5V; D6/D7 SMBJ5.0A on 5V_A/5V_B |
| NIT: panel line unfiltered | C17 100n on SW_SENSE |

## Fixed in firmware (all 11 host tests green)
- FE_EN driven low as the first instruction after reset (boot-glitch fix)
- Cold turn-on floor: OFF now gates at 10.65V (WARN+100mV), not 11.7V — a
  mid-charge pack is switchable; RECONNECT 11.7V still governs CUT recovery
- HARD RULE documented in apply_outputs(): PB4 must never drive push-pull HIGH
  (3.3V on the EN tap back-feeds OV through R2 → power path turns OFF)

## Fixed in board (generate_board.py)
- J1 XT60 mate now faces WEST off-edge (was mating into the board interior);
  pads on copper; clear of mounting hole H1
- J12 XT30 pegs now ON-board (were drilled outside the outline); mates east
- J3 USB-C moved to the TOP edge — clears mounting hole H2
- Monotonic power corridor: J1→F1(vertical)→R20/U5 Kelvin→front-end mid-left→
  VSW east to both buck Cin columns (no more VBATT_S/VSW crossing diagonals)
- Buck hot loops: HS drain rotated to face the Cin column, LS stacked 8mm below,
  L pad adjacent to SW
- Zones REFILLED after the overlap resolver (stale fill = as-saved shorts)
- Solid (non-relief) pour connection on all power pads; 20 GND thermal vias
  (4 per controller EP, 3 per LS/front-end FET)
- ESD arrays (D2/D3), RILIM (R16/R17) near their ports; U9 next to its loads

## Accepted / open (do at routing or P5)
- 4 residual sub-mm bbox grazes between small movables (D2-J6, CB3-F2, D7-D1,
  R18-C7) — one-click GUI nudges
- LM5145 custom footprint intra-pad clearance 0.175mm vs 0.2 netclass: TI's own
  pattern; add DRC exclusion (JLC can fab it)
- lib_footprint_issues DRC noise: set FPID nicknames in board gen
- Mounting holes FINAL at corners (5mm inset) — chassis adapter is 3D-printed to fit (2026-07-13)
- USB-C VBUS hard-wired hot (no CC gating): fine for captive Pi cable; Rp=10k
  is a hard 3A ceiling — `usb_max_current_enable=1` must ship in the rover image
- Polymer 6.3V at 81% derating: swap to 10V part if JLC stocks one
- Docs errata (consistency review #6,7,9,10,14): P1 F-number citations, ~25uA
  (not 10uA) cut-state figure, TPS7A1650→1633 strings, DESIGN.md stale lines
  (chemistry jumper / LPD bypass / 10-32V sweep / SQJQ140E / 4-pin aux / EN-from-Pi),
  PRODUCTION_REVIEW = v1 snapshot stamp, BOM adds (C5-C17, CE1, D5-D7, F2,
  R31-R34, split caps; NTC + 22u C# locks)
- Behavioral doc syncs: AUX_CTL requires Pi I2C arm (reg 0x11); PGOOD semantics;
  bench NTC/REG_TEMP validity only when ON
