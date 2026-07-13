# P3 — footprints & mechanical

**STATUS 2026-07-13: assignments DONE.** Every symbol carries a Footprint property
(96/96, netlist-verified); two custom footprints authored in
`kicad/power_board_v1.pretty/` from the TI mechanical drawings and validated via
the pcbnew API (21 + 13 pads load cleanly):
- `VQFN-20_3.5x4.5_P0.5_LM5145RGY` — SNVSAI4B p.65 land pattern (0.25×0.6 pads,
  0.5 pitch, EP 1.7×2.7)
- `WSON-12_3x3_P0.5_LM74800DRR` — SNOSD95C DRR0012E (0.62×0.25 pads, **0.5 pitch**
  — the 0.45 previously noted here was wrong; EP 1.3×2.5 = RTN, must stay
  unconnected, which the schematic enforces by having no pad-13 pin)

Land-pattern source of truth: the part datasheet's recommended pattern. Verify
stdlib picks against the LCSC part's actual datasheet at the P4 review (clone
parts sometimes differ from the genuine drawing!). Open verifications:
- ATtiny816 VQFN EP size vs stdlib QFN-20 EP1.65x1.65 (Microchip drawing)
- TPS7A16 DGN EP (1.57×1.89) vs stdlib MSOP-8 EP1.68x1.88 variant
- cjiang FXL0630 land vs stdlib L_Chilisin_BMRx00060630
- USB-A C2345 drawing vs USB_A_CNCTech_1001-011-01101
- **F1**: BOM part (Keystone 3557-10) vs assigned placeholder footprint
  (Littelfuse FLR_178.6165) — reconcile: either source the Littelfuse holder or
  author the Keystone footprint from its drawing

## Footprint map
| Refs | Part (BOM.md) | KiCad lib footprint / source | Watch-outs |
|---|---|---|---|
| U1, U2 | LM5145RGYR VQFN-20 3.5×4.5 | `Package_DFN_QFN:VQFN-20-1EP_3.5x4.5mm_P0.5mm` class — verify EP size vs SNVSAI4B | thermal vias 4×; pin-1 orientation both stages identical |
| U4 | LM74800 WSON-12 3×3 (DRR) | `Package_SON:WSON-12-1EP_3x3mm_P0.45mm` class — verify vs SNOSD95C | **RTN pad must float** (datasheet: do NOT tie EP to GND plane) |
| U3 | ATTINY816-MNR UQFN-20 3×3 | `Package_DFN_QFN:QFN-20-1EP_3x3mm_P0.4mm` | 0.4 mm pitch — JLC OK; UPDI pad accessible for probe |
| U5 | INA226 VSSOP-10 | `Package_SO:VSSOP-10_3x3mm_P0.5mm` | Kelvin routing to R20 (paired traces, no plane detour) |
| U6 | TPS7A1633DGNR MSOP-8 PwrPAD | `Package_SO:MSOP-8-1EP_3x3mm_P0.65mm` | EP to GND, 2 vias |
| U7, U8 | TPS2553 SOT-23-6 | `Package_TO_SOT_SMD:SOT-23-6` | — |
| U9 | AP2112K SOT-25 | `Package_TO_SOT_SMD:SOT-23-5` | — |
| Q* ×6 | CSD18543Q3A VSONP-8 3.3×3.3 | CUSTOM from TI SON 3.3×3.3 drawing (LSON-CLIP) — NOT the generic 5×6 DFN-8! | 3.3×3.3 body; big drain pad; verify against C840100 datasheet page |
| LA1, LB1 | MWSA1005S-3R3MT 11.5×10 | CUSTOM from Sunlord MWSA1005S drawing | keep SW-node pad side toward the FETs |
| L2, L4 | FXL0630 7×6.6 | CUSTOM from cjiang FXL0630 drawing | — |
| CA7, CB7 | Lelon 6.3×5.9 SMD polymer | `Capacitor_SMD:CP_Elec_6.3x5.9` | polarity silk |
| R20 | 2 mΩ 2512 | `Resistor_SMD:R_2512_6332Metric` ± Kelvin pads | check "-4" 4-terminal drawing; if 2-pad, route Kelvin sense from pad inner edges |
| D1 | SMB | `Diode_SMD:D_SMB` | — |
| D2, D3 | SOT-23-6 | `Package_TO_SOT_SMD:SOT-23-6` | — |
| J1 | XT60PW-M | CUSTOM from AMASS drawing (2 power pins + 2 mech pegs) | 6 mm² copper lands, wave/hand solder |
| J12 | XT30PW-M | CUSTOM from AMASS drawing | DNP |
| J3 | USB4105-GF-A | GCT footprint download (KiCad file provided by GCT) | TH shell tabs; SMD signal pins 0.5 mm |
| J4, J5 | USB-A TH r/a | `Connector_USB:USB_A_Stewart_SS-52100-001_Horizontal` class — verify vs C2345 drawing | port face off board edge |
| J7 | SM08B-GHS-TB | `Connector_JST:JST_GH_SM08B-GHS-TB_1x08-1MP_P1.25mm_Horizontal` | side-entry: cable exits parallel to board |
| J2, J8 | BM02B-GHS-TBT | `Connector_JST:JST_GH_BM02B-GHS-TBT_1x02-1MP_P1.25mm_Vertical` | — |
| J11 | BM03B-GHS-TBT | `Connector_JST:JST_GH_BM03B-GHS-TBT_1x03-1MP_P1.25mm_Vertical` | — |
| F1 | Keystone 3557-10 | CUSTOM from Keystone drawing (2 large TH slots) | fuse insertion clearance above |
| J9, J10 | 2.54 4-pin | `Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical` | — |

## Board outline & mechanical (proposal — see power_board_v1_floorplan.png)
- **90 × 65 mm**, 2-layer, 2 oz outer copper. (Grew from 80×60: MWSA1005S inductors
  are 11.5×10 and the TH connectors need edge real estate.)
- 4× M3 mounting holes, 5 mm from corners; matches rover plate grid (verify against
  chassis before P4 — measure the actual rover mounting pattern!).
- Connector edges: battery/input on LEFT edge; Pi power (USB-C + XT30 pads)
  TOP-RIGHT; radio USB-A ports RIGHT edge; signal harnesses (Pi JST-GH, AUX, panel
  switch, UPDI) BOTTOM edge. Fuse F1 top-left, reachable in the enclosure.
- Zones (left→right power flow): input protection + LM74800 front-end (left ~20 mm)
  → buck A top-center / buck B bottom-center (hot loops tight, SW nodes minimal,
  Cin within 5 mm of FET pairs) → pi filters + outputs (right ~20 mm).
  Supervisor strip (MCU/LDO/INA226) along the bottom-left; INA226 within 10 mm of
  R20 with Kelvin pair routing.
- Bottom layer = unbroken GND pour under the power stages; analog AGND island for
  RT/SS/COMP/FB components stitched at one point per controller (checklist B).
- LED D4 on the bottom edge next to the Pi harness (visible when mounted).
- Thermal: FET drain pads pour into ~2 cm² copper each side; inductors over pour;
  no components under the Pi mounting shadow (enclosure TBD).

## P3 remaining (needs KiCad)
- Transcribe footprints onto symbols (schematic v2 has logical pinouts; VQFN
  physical pin mapping happens here — MUST re-map LM5145 RGY 20-pin physical
  numbering onto the 16 logical pins in the generator).
- Board outline + mounting holes in pcbnew; import floorplan zones.
- GATE: footprint checklist (every part's pattern vs datasheet, pin-1 marks,
  courtyard overlaps, JLC DFM rules ≥0.2 mm clearance).
