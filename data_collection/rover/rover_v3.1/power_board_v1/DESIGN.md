# Rover power board v1 — design spec

**REV-LOCKED 2026-07-13** — all four P0 decisions closed (3S-only; motor path
off-board; USB-C receptacle + Silkland cable + XT30 fallback; JLCPCB assembly).
Changes past this point go through PLAN.md phase gates.

One PCB replacing: main power switch + low-power-disconnect brick + loose 10-32V->5V
USB bucks. Adds what the Apr-2025 switch failures, the 0.1V-hysteresis LPD, and the
"no battery telemetry" ArduPilot review finding all asked for.

## Requirements
- **P0 DECISION (2026-07-13): 3S-only.** VIN 9-13 V operating (Li-ion 3S). Simplifies:
  fixed LPD divider (no chemistry jumper), TVS -> SMBJ16A class, buck compensation
  optimized for the narrow range. Silicon ratings kept >=42 V anyway (parts already
  chosen; free margin against transients).
- 5V outputs: Pi 5 (5.1 V / 6 A rail A) ; 2x PlutoPlus + 2 A aux (5.1 V / 5 A rail B)
- Power on/off control; low-power disconnect; battery telemetry to the Pi
- Very low ripple at radio ports; no brownout under Pi transients

## Architecture
BATT -> reverse-polarity FET -> fuse 15 A -> back-to-back NFET high-side switch
(soft-start ~10 ms; gate logic from panel switch at mA — replaces arc-prone
mechanical switching; DPDT relay footprint provided as alt-populate) ->
  -> Buck A: 5.1 V / 6 A  -> Pi 5 (XT30 or screw + USB-C pigtail)
  -> Buck B: 5.1 V / 5 A  -> 2x USB-A ports (per-port TPS2553 load switch,
     EN from Pi GPIO => software power-cycle of a hung Pluto) + aux screw terminal
  -> AUX_CTL output: gate signal for an external motor-path FET contactor so the
     Cytron/motor 30 A path shares the same on/off + LPD decision (motors stay OFF
     this board — keeps it 2-layer simple).
     **P0 DECISION (2026-07-13): motor path confirmed OFF-BOARD.**

## Low-power disconnect (fixes both field complaints)
- Defaults (3S, jumper-selectable divider): CUT 10.2 V, RECONNECT 11.7 V
  (1.5 V hysteresis vs the old 0.1 V), qualification delay ~10 s below threshold
  (motor-stall sag immunity)
- Comparator + TLV431 reference, or the MCU option below
- GRACEFUL SHUTDOWN: LOW_BATT GPIO asserted to Pi ~60 s before cut (Pi service
  closes the capture zarr + `shutdown -h`), then hard disconnect

## Telemetry (closes the "no BATT_* monitor" gap)
- INA226 (I2C to Pi): pack V, current, power -> logged alongside captures; the Pi
  can also forward to ArduPilot as MAVLink battery status
- Status LEDs + PGOOD_A/B GPIOs

## Regulators
- 2x wide-VIN sync buck, 42-60 V rated: LM5146-Q1/LM25145 controller class
  (or TPSM/LMZ power modules for the simplest assembly), fsw 600 kHz
  (LM25145 max is 1 MHz — the earlier 2.1 MHz spec was invalid; 600 kHz keeps
  fsw well above GPS/telemetry IF products and inductors small),
  shielded inductors, LC pi post-filter per rail
  (<10 mVpp at radio ports); 5.1 V setpoints (cable-drop compensation)
- Efficiency target >=90 % @ 12 V in / 46 W out (~5 W dissipation, copper-pour
  cooling, no fan dependence)

## Optional MCU variant (recommended): ATtiny816/STM32C011
Implements LPD thresholds+delays, soft on/off, shutdown handshake sequencing, and
I2C status; analog-only fallback circuit stays on the board un-populated.

## Protections
Reverse polarity (FET), 15 A input fuse, per-USB-port current limit (TPS2553),
TVS on VIN (SMBJ16A — 3S-only decision) and each 5 V rail, ESD arrays on USB.

## Connectors / pinout
XT60 in; USB-C receptacle w/ 10k Rp + specified 0.5ft 240W cable (Silkland B0CQ4SX256), XT30 fallback pads (rail A); 2x USB-A + 4-pin screw aux (rail B);
6-pin Pi header: 3V3ref, SDA, SCL, LOW_BATT, PGOOD, GND; 2-pin panel switch;
2-pin AUX_CTL (motor contactor gate); jumpers: chemistry select, LPD bypass.

## Bring-up plan
1. Bench PSU sweep 10-32 V, no load: rails at 5.10 +-1 %, PGOOD asserted
2. Load test: rail A 6 A + rail B 5 A electronic load, 30 min thermal soak
   (<60 C hotspots), ripple scope check (<10 mVpp radio ports, <30 mV Pi)
3. LPD: sweep VIN down w/ 10 s qualifier check, verify hysteresis + reconnect;
   verify LOW_BATT precedes cut by >=60 s at typical discharge slope
4. Inrush scope: <5 A peak at switch-on into full downstream capacitance
5. Pi 5 + 2 Pluto integration: stress (stress-ng + both radios streaming),
   confirm no vcgencmd undervolt flags; GPIO power-cycle of each radio
6. EMI sanity: GPS lock time next to board; SDR noise floor with bucks on/off

## KiCad next steps
kicad/ project skeleton: 2-layer, 2 oz copper, ~80x60 mm; JLC assembly-friendly
parts (basic lib where possible). Draft schematic sheets: input+switch+LPD,
buck A, buck B+USB, MCU+telemetry.
