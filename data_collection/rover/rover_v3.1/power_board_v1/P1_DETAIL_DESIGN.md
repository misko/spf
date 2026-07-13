# P1 — electrical detail design (working doc)

Companion to DESIGN.md (rev-locked spec) and PRODUCTION_REVIEW.md (findings F1-F9).
All equations cite the governing datasheet: LM25145 (TI SNVSAT9, June 2017) and
LM7480x-Q1 (TI SNOSD95C, Dec 2020). Values marked [QS] must be cross-checked with the
TI LM25145 Quickstart Calculator before P2 BOM lock — the math below follows the
datasheet design procedure exactly, but Quickstart also validates loss/thermal corner
cases we can't run here.

---

## 1. Gate-drive trade study — RESOLVED: LM74800-Q1 front-end

Candidates: (a) LTC7004 high-side NFET driver + separate reverse-polarity PFET,
(b) discrete charge-pump/zener b2b NFET switch (as in schematic v1) + PFET,
(c) LM74800-Q1 ideal-diode controller driving back-to-back NFETs.

**Winner: (c) LM74800-Q1.** One WSON-12 controller + one dual-NFET replaces THREE
blocks of schematic v1 (reverse PFET Q1, discrete b2b switch gate machinery, RC
soft-start) and closes review finding F5 (PFET orientation error class disappears —
reverse protection becomes the DGATE ideal diode):

| Parameter (datasheet §7.5) | Value | Why it matters here |
|---|---|---|
| Input range | 3-65 V op, -65 V reverse | 3S (9-13 V) with huge margin; survives reversed pack |
| Shutdown current, EN low | **2.87 µA typ (5 µA max)** | LPD cut-state battery drain ~nothing |
| Operating IQ | ~400 µA | irrelevant when rails are on |
| DGATE regulation | V(A)-V(C) = 10.5 mV | ideal diode, ~mW loss at 5 A vs ~2 W Schottky |
| Fast reverse blocking | -4.5 mV threshold, 0.5 µs | protects during pack hot-unplug |
| Gate drive (VS > 5 V) | 10-13 V above rail | standard-level FETs; **FET VGS rating must be ≥15 V abs** |
| HGATE source current | 55 µA typ | free soft-start via gate cap (below) |
| EN/UVLO | 1.231 V rise / 1.132 V fall; <0.67 V = 2.87 µA shutdown | MCU drives it; precision divider = analog LPD fallback |
| OV pin | 1.231 V threshold ladder | free overvoltage cutoff (set 15.0 V — charger fault) |

Wiring (typical app, SNOSD95C §10.2): BATT+ → Q1A source (pin A), common-drain node
(pin C), Q2A source → SW_BATT rail (pin OUT). FETs: **SQJQ140E dual 40 V b2b NFET**
kept from v1 (VGS ±20 V ✓, one package does both positions).

- Soft-start: C_HG = 47 nF from HGATE to OUT → dv/dt = 55 µA / 47 nF ≈ 1.2 V/ms →
  ~10 ms ramp to 12 V. Inrush into ~260 µF downstream ≈ 0.3 A. Meets the <5 A
  bring-up spec with 15x margin.
- OV ladder on SW/OV pins: cut above ~15 V (bad charger left attached). SW pin's
  internal ladder-disconnect kills divider leakage in shutdown.
- Analog LPD fallback (MCU unpopulated variant): EN/UVLO divider gives fixed 8 %
  hysteresis → cut 10.2 V / reconnect ~11.1 V (not the full 11.7 V — accepted for
  the fallback path only; the MCU implements the real 10.2/11.7 + 10 s qualifier).
- EN/UVLO node: MCU GPIO (open-drain-emulated: drive low / release) wire-ANDs with
  the divider, so MCU absent == divider rules, MCU present == MCU rules.

## 2. Buck A and Buck B — LM25145, complete component math

Operating point: VIN 9-13 V (3S), VOUT 5.10 V, fsw 600 kHz, diode emulation
(SYNCIN → AGND) for light-load efficiency; buck B EN sequenced from buck A PGOOD
(datasheet Fig 34 master-slave) so the Pi rail rises first.

Common values (both rails):
- **RT = 10⁴/600 = 16.67 kΩ → 16.5 kΩ E96** (fsw ≈ 606 kHz) [eq 3]
- **VREF = 0.8 V**; feedback: RFB1 = 20.0 kΩ, RFB2 = 20k/(5.10/0.8 - 1) = 3.72 kΩ
  → **3.74 kΩ E96** → VOUT = 5.08 V (within 5.1 V ±2 % spec incl. cable-drop intent)
- **Soft-start**: tSS = 4 ms → CSS = 12.5·4 = 50 nF → **47 nF** [eq 5]
- **EN/UVLO divider** (buck's own floor, well below the LPD): VIN(on) 8.5 V,
  VIN(off) 7.5 V → RUV1 = 1 V/10 µA = **100 kΩ**, RUV2 = 100k·1.2/(8.5-1.2) =
  **16.5 kΩ E96** [eq 1,2]
- Duty: D = 0.392 (13 V) / 0.425 (12 V) / 0.567 (9 V) — all mid-range, no min-on/off
  issues at 606 kHz (ton = 647 ns @ 13 V).
- VCC from internal LDO off VIN (9-13 V — dropout region of the DVCC trick not
  needed; gate-drive loss ≈ (12-7.5 V)·12 mA ≈ 55 mW, negligible).
- CVCC = 2.2 µF, CBST = 0.1 µF (datasheet fixed recommendations).

### Inductor [eq 7,8]
LF = (VOUT/VIN)·(VIN-VOUT)/(ΔIL·fsw), worst ripple at VIN = 13 V.
Target ΔIL ≈ 30 % of IOUT.

| | Rail A (6 A, Pi 5) | Rail B (5 A, radios+aux) |
|---|---|---|
| L chosen | **3.3 µH shielded** | **3.3 µH shielded** (same part) |
| ΔIL @ 13 V | 1.57 A (26 %) | 1.57 A (31 %) |
| IL(peak) | 6.8 A | 5.8 A |
| Isat requirement | ≥ 9 A (incl. OCP headroom) | ≥ 8 A |
Candidate class: 3.3 µH, ≥10 A Isat, ≤15 mΩ DCR, 7x7 shielded (e.g. XAL7070/WE-XHMI
class) — exact MPN at P2 vs JLC stock.

### Output capacitors [eq 9,10]
Ripple (eq 9, ESR≈0): Cout ≥ ΔIL/(8·fsw·ΔV) = 1.57/(8·606k·20 mV) ≈ 16 µF — trivial.
Load-off transient governs (eq 10), ΔIOUT = 3 A step, ΔVovershoot = 50 mV:
Cout ≥ L·ΔI²/((VOUT+ΔV)² - VOUT²) = 3.3µ·9/0.512 ≈ **58 µF effective minimum**.
Choose per rail: **4x 47 µF 10 V X7R 1210** (≈100-110 µF effective after DC derating
at 5.1 V) **+ 220 µF polymer** (≤25 mΩ ESR) bulk → ~5x the minimum; the polymer ESR
zero is used in compensation below. Plus LC pi post-filter per DESIGN.md
(1 µH + 2x22 µF) on the radio ports only.

### Input capacitors [eq 11-13]
ICIN,rms ≈ IOUT·√(D(1-D)) ≈ 3.0 A (rail A). With ΔVIN = 200 mV:
CIN ≥ D(1-D)·IOUT/(fsw·ΔVIN) = 0.244·6/(606k·0.15) ≈ 16 µF.
Choose per rail: **3x 10 µF 50 V X7R 1210 + 2.2 µF + 0.1 µF** at the FET loop, plus
one shared **100 µF hybrid electrolytic** at the board input (damps the input LC
against the pack lead inductance; rated ≥2 A rms each, ripple shared).

### Power MOSFETs (40 V NexFET class, VGS(th) < 2.5 V, VGS abs ±20 V)
- High-side Q1: ~10 nC Qg(7.5 V), ≤10 mΩ — conduction 0.42·36·0.009 ≈ 0.14 W,
  switching ≈ 0.1 W @ 606 kHz.
- Low-side Q2: RDS(on) is BOTH the sync rectifier loss AND the OCP sense element
  (RDS-ON mode). ≤8 mΩ target: conduction 0.58·36·0.008 ≈ 0.17 W.
- Exact MPNs at P2 against JLC basic/extended stock (CSD18543Q3A-class or paired
  30 V/40 V SON5x6). [QS] verify losses.

### OCP — RDS(on) valley sensing [eq 6]
RILIM = (Ilimit - ΔIL/2)·RDS(on)Q2,hot / 200 µA, with CILIM: R·C ≈ 6 ns.
RDS(on),hot ≈ 8 mΩ·1.35 ≈ 10.8 mΩ (the ILIM source has +4500 ppm/°C tracking, so use
25 °C RDS(on) with the datasheet's tracking assumption → use 10 mΩ effective):

| | valley limit target | RILIM | CILIM |
|---|---|---|---|
| Rail A | 7.8 A avg → 7.0 A valley | 7.0·0.010/200µ = **348 Ω E96** | **18 pF** |
| Rail B | 6.5 A avg → 5.7 A valley | 5.7·0.010/200µ = **287 Ω E96** | **22 pF** |

[QS] OCP setpoints move with the actual FET chosen at P2 — recompute then.

### Type-III compensation [Table 5, eq 15-16]
Plant: L = 3.3 µH, Cout_eff ≈ 200 µF (derated ceramics + polymer) →
fo = 1/(2π√(L·C)) ≈ 6.2 kHz. ESR zero (polymer 25 mΩ·220 µF) ≈ 29 kHz.
Target fc = fsw/10 ≈ 60 kHz; kFF = 15.
- Kmid = (fc/fo)/kFF = (60/6.2)/15 = 0.645
- **RC1 = Kmid·RFB1 = 12.9k → 13.0 kΩ**
- ωz1 = 0.5·ωo → **CC1 = 2/(ωz1·RC1) = 8.2 nF**
- ωp2 = ωsw/2 → **CC2 = 1/(ωp2·RC1) = 39 pF**
- ωz2 = ωo → **CC3 = 1/(ωz2·RFB1) = 1.2 nF**
- ωp1 = ωESR → **RC2 = 1/(ωp1·CC3) = 4.64 kΩ**
Same values both rails (same L, same Cout). Phase margin target 50-70°; [QS] verify
+ bench Bode at P7 (bring-up item 2).

## 3. Always-on supervisor domain

- **LDO**: TPS7A1650 (60 V, ~5 µA IQ) from BATT+ (before the b2b switch) → 3.3 V
  MCU rail. (HT7550 rejected earlier — Vin < 32 V. F-finding stands.)
- **MCU**: ATtiny816 (20-pin VQFN/SOIC). Runs LPD state machine + I2C telemetry.
- **INA226 moves to the SWITCHED side** (3.3 V derived from rail B via 100 mA LDO or
  divider-fed from Pi 3V3 header pin): its ~330 µA IQ would dominate cut-state drain
  on the always-on rail, and its consumer (the Pi) is dead when rails are off anyway.
  MCU's own ADC provides pack-V telemetry in all states.
- **Cut-state battery drain**: LM74800 2.87 µA + TPS7A1650 ~5 µA + MCU sleep ~2 µA
  ≈ **10 µA** → 0.007 Ah/month; a stored 3 Ah pack loses <1 %/yr to the board.

### ATtiny816 pin map (all 17 usable pins; UPDI kept for programming)
| Pin | Signal | Dir | Function |
|---|---|---|---|
| PA0 | UPDI | — | programming header (3-pin: UPDI/3V3/GND) |
| PA1 | USB_EN1 | out | TPS2553 port 1 enable (Pluto A power-cycle) |
| PA2 | USB_EN2 | out | TPS2553 port 2 enable (Pluto B power-cycle) |
| PA3 | LED_STAT | out | status LED (state machine heartbeat pattern) |
| PA4 | VIN_SENSE | ADC | pack V ÷ 7.8 divider (13 V → 1.67 V; 1.5 V ref w/ gain, see fw) |
| PA5 | FAULT_USB | in | TPS2553 FAULT̅ wired-OR, pull-up (F8 fix: NOT tied to LOW_BATT) |
| PA6 | V5A_SENSE | ADC | rail A ÷ 4 divider (brownout diagnostics) |
| PA7 | NTC | ADC | board NTC (thermal telemetry) |
| PB0 | SCL | I2C | slave to Pi (addr 0x36) |
| PB1 | SDA | I2C | slave to Pi |
| PB2 | SW_SENSE | in | panel switch (F7 fix: mA-level logic sense, debounced in fw) |
| PB3 | AUX_CTL | out | external motor-contactor gate command (P0 decision 2) |
| PB4 | FE_EN | out | LM74800 EN/UVLO (release-high via divider / drive low) |
| PB5 | BUCK_A_EN | out | buck A EN override (default release; B follows PGOOD_A) |
| PC0 | PGOOD_A | in | rail A power good |
| PC1 | PGOOD_B | in | rail B power good |
| PC2 | LOW_BATT | out | to Pi GPIO (60 s early warning, active low) |
| PC3 | SHDN_ACK | in | from Pi GPIO: "capture closed, halting now" (early cut OK) |

## 4. Supervisor firmware (firmware/)
State machine + thresholds implemented in `firmware/supervisor.c` (bare-metal
avr-gcc); register map in `firmware/README.md`. Contract:

- Thresholds: CUT 10.2 V / RECONNECT 11.7 V (1.5 V hysteresis), LOW_BATT warn at
  10.55 V; **10 s qualifier** below threshold before any state change (motor-stall
  sag immunity); **60 s LOW_BATT → cut handshake**, shortened if Pi asserts SHDN_ACK.
- States: OFF → PRECHARGE(soft-start settle) → ON → WARN(LOW_BATT asserted) →
  DYING(60 s countdown) → CUT(FE_EN low, 2.87 µA) → (V ≥ 11.7 V for 10 s) → ON.
  Plus SWOFF (graceful panel-switch-off handshake, 5 s budget) and FAULT
  (PGOODs never rose in PRECHARGE — latched off until the switch is toggled,
  preventing an on/off chatter loop).
- I2C regs (RO unless noted): 0x00 status/state, 0x01-02 VIN mV, 0x03-04 V5A mV,
  0x05 temp, 0x06 flags (PGOOD_A/B, FAULT_USB, SW, LOW_BATT, AUX), 0x10 (RW)
  USB port-cycle command, 0x11 (RW) AUX_CTL, 0x12 (RW) shutdown request, 0xF0 fw ver.
- WDT 1 s; brown-out detect 2.6 V; all thresholds compile-time constants in one
  header block; unit-testable pure-function core (threshold/qualifier logic takes
  (mv, now_ms, state) → state).

## 5. Schematic v2 delta list (to apply to kicad/generate_schematic.py)
1. Delete Q1 reverse-PFET block + discrete b2b gate machinery + RC soft-start →
   LM74800-Q1 (U_FE) + SQJQ140E + C_HG 47 nF + EN/UVLO divider + OV ladder (15.0 V).
2. Buck A/B black-box values → §2 exact values (RT/CSS/RFB/RUV/RILIM/CILIM/L/Cin/
   Cout/comp RC1,CC1,CC2,RC2,CC3; SYNCIN→AGND; PGOOD_A→EN_B sequencing resistor).
3. INA226 supply net: 3V3_SW (switched), not 3V3_AON.
4. MCU pin net names per §3 table (incl. SHDN_ACK new).
5. Panel switch → PB2 sense only (no power path). AUX_CTL → 2-pin JST-GH.
GATE for P1 close: schematic v2 regenerated, zero black boxes, ERC clean (needs
`sudo apt install kicad` locally, else ERC deferred to P4 entry).

## 6. Open items rolling to P2
- MPN lock vs JLC stock: FETs (buck + SQJQ140E availability), 3.3 µH inductors,
  polymer 220 µF, TPS7A1650 footprint variant, ATtiny816 VQFN vs SOIC.
- [QS] Quickstart cross-check of §2 (losses, OCP corners, comp Bode).
- Recompute RILIM/CILIM for the actual FET RDS(on).
