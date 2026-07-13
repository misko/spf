# Power board v1 — production-readiness review (2026-07-13)

HONEST STATUS: schematic capture is v1 (netlist-complete, buck stages still
datasheet-black-boxes); THERE IS NO PCB LAYOUT YET. This review covers the schematic,
pins every part to its datasheet, maps to reference designs, applies the catches found,
and defines the gates that stand between here and "production ready". A hardware design
is not production-ready until the gates below are green — no exceptions.

## Review findings (applied to the schematic in this commit)
| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| 1 | MCU LDO was HT7550 (Vin max 24-30 V) vs 32 V input spec | BLOCKER | → TPS7A1650 (3-60 V, EN+PG) |
| 2 | TPS2553 ILIM pins floating — current limit undefined | BLOCKER | RILIM 20 kΩ → ~1.2 A per port (DS eq.) |
| 3 | fsw spec said 2.1 MHz; LM25145 max is 1 MHz | MAJOR | fsw = 600 kHz; EMI handled by π-filters + bring-up scan |
| 4 | No I2C pullups | MAJOR | 4k7 ×2 to MCU_5V |
| 5 | PGOOD open-drain outputs had no pullups | MAJOR | 10 k ×2 |
| 6 | No USB ESD protection | MAJOR | USBLC6-2 per port (D+/D− placeholder; route both lines in detail pass) |
| 7 | No UPDI programming access | MINOR | J11 header |
| 8 | Gate-drive for back-to-back NFETs needs charge pump (noted on sheet) | OPEN | detail design: LTC7004/LM74800-class or controller w/ integrated CP |
| 9 | Pi 5 will limit USB to 600 mA without 5 A PD negotiation | NOTE | fixed-supply rail + `usb_max_current_enable=1` in config.txt; document in bring-up |

## BOM core with datasheets (verified 2026-07-13)
| Ref | Part | Key spec | Datasheet |
|---|---|---|---|
| U1,U2 | TI LM25145RGYR | sync buck ctrl, VIN 6-42 V, 0.8-40 V out, fsw ≤1 MHz, VQFN-20 | ti.com/lit/ds/symlink/lm25145.pdf |
| U3 | Microchip ATtiny816 | supervisor MCU, 1.8-5.5 V | microchip.com ATtiny816 DS |
| U5 | TI INA226 | I2C V/I monitor, bus 0-36 V | ti.com/product/INA226 |
| U6 | TI TPS7A1650DGN | LDO 3-60 V in, 100 mA, EN+PG | ti.com/product/TPS7A16 |
| U7,U8 | TI TPS2553DBV | USB switch 2.5-6.5 V, ILIM 75 mA-1.7 A adj (RILIM), 85 mΩ | ti.com/lit/ds/symlink/tps2553.pdf |
| Q1 | Vishay SQJ457EP | P-FET −40 V, rev-pol | vishay.com |
| Q2,Q3 | Vishay SQJQ140E | N-FET 40 V, 2 mΩ-class | vishay.com |
| D1 | SMBJ33A | TVS 33 V standoff | littelfuse.com |
| D2,D3 | USBLC6-2SC6 | USB ESD array | st.com |
| R20 | 2 mΩ 3 W shunt | Kelvin, e.g. WSLP2726 | vishay.com |
| CBL1 | Silkland 240W 0.5ft C-to-C | SPECIFIED Pi cable (ASIN B0CQ4SX256); validate >=4.9 V @ Pi @ 5 A in bring-up | amazon B0CQ4SX256 |
| L1,L3 | 4.7 µH shielded ≥10 A Isat | e.g. XAL7070-472 | coilcraft.com |
| FETs for bucks | per LM25145 Quickstart output | 40 V logic-level pair | — |

## Reference-design validation
- LM25145: datasheet §Application (5 V output design example) + TI Quickstart Design
  Tool for the exact RT/SS/COMP/FET/L/C values at (10-32 V in, 5.1 V, 6 A, 600 kHz).
  NOTE: LM5146-Q1-EVM12V exists but is a 15-85 V/12 V design — cite for LAYOUT
  practices (power loop, gate routing), not for component values.
- TPS2553: datasheet front-page application circuit (RILIM 20 kΩ, RFAULT 100 kΩ) —
  our port circuit matches it.
- INA226: TI INA226EVM topology; Kelvin-sense layout rules from DS layout section.
- Pi 5 power: Raspberry Pi documentation — 5 V/5 A USB-C, `usb_max_current_enable`,
  official supply is 5.1 V (our rail A setpoint matches).

## Checklists
### A. Schematic sign-off (current status in brackets)
- [x] Every pin of every part connected or explicitly NC (v1: FAULT pins NC by choice)
- [x] Reverse-polarity FET orientation verified (body diode direction)
- [x] Back-to-back main FETs common-source, gate pulldown + soft-start RC
- [x] All open-drain outputs pulled up (PGOOD, I2C) [added this review]
- [x] ILIM/config resistors on every adjustable part [added]
- [x] Regulator VIN ratings ≥ 42 V margin over 32 V max input (LM25145 ✓, TPS7A16 ✓)
- [ ] ERC clean in KiCad (BLOCKED: no KiCad on this machine — run on open)
- [ ] Buck black-boxes expanded to full LM25145 app circuit (Quickstart values)
- [ ] Gate-drive charge pump detail design (finding 8)
- [ ] MCU firmware pin assignment cross-checked against ATtiny816 alternate functions
### B. Layout (not started — gate for next phase)
- [ ] Power loop: Cin–FETs–L–Cout minimized per LM5146 EVM layout guide
- [ ] Kelvin connection at shunt; INA226 traces as differential pair
- [ ] Gate loops short; no gate trace under inductor
- [ ] π-filter caps at connector, not at buck
- [ ] Thermal: 2 oz copper, via-stitched pours under FETs/controller pads
- [ ] USB D+/D− 90 Ω differential, length-matched, ESD at connector
- [ ] Creepage: 32 V nets vs logic ≥0.5 mm; XT60 pads ≥2 mm from GND pour
- [ ] Mounting holes, connector edge placement vs rover harness
### C. DFM / assembly
- [ ] JLC/assembler part availability check (basic vs extended)
- [ ] Footprint review vs datasheet land patterns (VQFN-20 wettable flanks)
- [ ] Panelization, fiducials, test points on all rails + I2C
### D. Bring-up (extends DESIGN.md plan)
- [ ] Current-limited first power (0.5 A limit), rails unloaded
- [ ] LM25145 Quickstart values vs measured Bode/load-step (scope method)
- [ ] LPD thresholds + 10 s qualifier + LOW_BATT→cut sequence timing
- [ ] Full load thermal soak; EMI sanity vs GPS lock + SDR noise floor
- [ ] Pi5: verify usb_max_current_enable=1; vcgencmd throttled flags clean
### E. Production gates (ALL must be green before "production ready")
1. ERC clean → 2. WEBENCH/Quickstart-validated buck designs → 3. full layout + DRC
→ 4. design review vs checklist B → 5. prototype run (qty 3-5) → 6. bring-up plan
executed and logged → 7. field trial on one rover → 8. rev B with findings → release.
