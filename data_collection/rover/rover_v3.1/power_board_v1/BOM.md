# Power board v1 — BOM (P2 lockdown)

Status: **P2 LOCKED 2026-07-13** — every line has MPN + package + LCSC C# (or is a
commodity basic-lib value). Verified against LCSC/JLC by three parallel catalog
sweeps; stock/prices are point-in-time — re-check every C# in the JLCPCB BOM tool
at order placement (P5). Substitutions from the P1 design: LM25145→LM5145 (stock;
datasheet-verified equation-identical), SQJQ140E→2×CSD18543Q3A (stock/price),
ATtiny816 SOIC→UQFN (stock). RILIM recomputed for CSD18543Q3A (301/243 Ω).

Source of truth for values: P1_DETAIL_DESIGN.md §2 + kicad/generate_schematic.py v2.
Board qty assumption: 5 prototypes (JLCPCB assembly, P0 decision 4).

## A. Passives — resistors (all 0402 1% unless noted; JLC basic lib)
| Refs | Value | Qty/board | Note |
|---|---|---|---|
| R1 | 887 kΩ | 1 | LM74800 ladder top |
| R2 | 28.7 kΩ | 1 | ladder mid (EN tap) |
| R3 | 82.5 kΩ | 1 | ladder bottom (OV tap) |
| RA1, RB1 | 16.5 kΩ | 2 | RT (606 kHz) |
| R5 | 16.5 kΩ | 1 | buck A EN lo |
| R4, R10† | 100 kΩ / 680 kΩ† | 1+1 | EN hi; †VIN sense hi (680k) |
| R11 | 100 kΩ | 1 | VIN sense lo |
| RA2 | 301 Ω | 1 | RILIM A — recomputed for CSD18543Q3A 8.5 mΩ: 7.0 A valley × 0.0085 / 200 µA |
| RB2 | 243 Ω | 1 | RILIM B — 5.7 A valley × 0.0085 / 200 µA |
| RA3, RB3 | 20.0 kΩ | 2 | RFB1 |
| RA4, RB4 | 3.74 kΩ | 2 | RFB2 (5.08 V) |
| RA5, RB5 | 13.0 kΩ | 2 | RC1 comp |
| RA6, RB6 | 4.64 kΩ | 2 | RC2 comp |
| R16, R17 | 20.0 kΩ | 2 | TPS2553 RILIM (1.2 A) |
| R18, R19 | 4.7 kΩ | 2 | I2C pull-ups (MCU_3V3) |
| R21, R22, R27, R30 | 10.0 kΩ | 4 | PGOOD pu ×2, NTC top, FAULT pu |
| R23, R24 | 10.0 kΩ | 2 | USB-C Rp (3 A advertisement) |
| R25 | 30.0 kΩ | 1 | V5A sense hi |
| R26 | 10.0 kΩ | 1 | V5A sense lo |
| R29 | 1.0 kΩ | 1 | status LED |
| R28 | NTC 10 kΩ B3380 0603 | 1 | e.g. NCP18XH103F03RB-class (verify C#) |
| R20 | Milliohm HoSCG2512-3-R002-F-4, 2 mΩ 1 % 3 W 2512 | 1 | **C22450699** ($0.10, stock 2.9k); "-4" suffix suggests 4-terminal — verify land pattern from datasheet at P3; alt LR2512D-3W-2mR C500736 |

## B. Passives — capacitors (JLC basic lib)
| Refs | Value | Size | Qty/board | Note |
|---|---|---|---|---|
| C1, C2, CA3, CB3, C4 (+decoupling kit at P3) | 100 nF 50 V X7R | 0402 | 5+ | FE CAP/VS, BST ×2, VSENSE hold |
| C3 | 47 nF 50 V | 0402 | 1 | HGATE dv/dt (10 ms soft-start) |
| CA1, CB1 | 47 nF | 0402 | 2 | SS (4 ms) |
| CA2, CB2 | 2.2 µF 16 V X7R | 0603 | 2 | VCC LDO |
| CA4 | 18 pF C0G | 0402 | 1 | CILIM A (RC≈6 ns) |
| CB4 | 22 pF C0G | 0402 | 1 | CILIM B |
| CA5, CB5 | 10 µF 50 V X7R | 1210 | 6 | buck input (3 per rail) — FH 1210B106K500NT C116808 (JLC "Economic PCBA"); alts C3844168 / C77102 |
| CA6, CB6 | 47 µF 10 V X7R | 1210 | 8 | buck output ceramic (4 per rail) — Murata GRM32ER71A476KE15L C84494; alt TDK C76680 |
| CA11, CB11 | 22 µF 10 V X7R | 0805/1206 | 4 | pi-filter out (2 per rail) |
| CA8, CB8 | 8.2 nF X7R | 0402 | 2 | CC1 comp |
| CA9, CB9 | 39 pF C0G | 0402 | 2 | CC2 comp |
| CA10, CB10 | 1.2 nF X7R | 0402 | 2 | CC3 comp |
| CA7, CB7 | Lelon OCV221M0JTR-0606, 220 µF 6.3 V polymer | 6.3×5.9 SMD | 2 | **C250009** ($0.15, stock 4.8k). 5.1 V on 6.3 V = 81 % derating — acceptable for polymer (no MLCC-style DC bias loss) but prefer a 10 V part if one shows up in the JLC BOM tool; ESR ~20-25 mΩ (OCV series, unconfirmed — comp assumes 25 mΩ; bench Bode at P7 verifies) |

## C. Semiconductors — ICs VERIFIED vs LCSC 2026-07-13 (all Extended lib)
| Refs | Part | LCSC C# | Package | Stock | ~$ q10 | Notes |
|---|---|---|---|---|---|---|
| U1, U2 | **LM5145RGYR** | C485912 | VQFN-20 RGY | 400 | 2.18 | **substitution verified against LM5145 datasheet (SNVSAI4B)**: LM25145RGYR (C2876740) has only 7 pcs at LCSC. LM5145 (75 V) is equation-identical — RT=10⁴/Fsw, VREF 0.8 V, ISS 10 µA, kFF 15, ILIM 200 µA + 6 ns, same L/Cout eqs — so ALL computed values in P1_DETAIL_DESIGN.md §2 carry over unchanged |
| U4 | LM74800QDRRRQ1 | C3215600 | WSON-12 DRR | 1,935 | 1.58 | exact part; alt LM74801 (C2873366, 48 pcs) |
| U3 | **ATTINY816-MNR** | C2052778 | UQFN-20 3×3 | 3,023 | 0.70 | SOIC (-SNR, C2054204) is zero-stock; UQFN is cheaper anyway |
| U5 | INA226AIDGSR | C49851 | VSSOP-10 | 11,139 | 0.69 | exact part |
| U6 | TPS7A1633DGNR | C181239 | MSOP-8 PwrPAD | 1,842 | 1.43 | exact part (60 V, 5 µA IQ); alt TPS7B6933 (C108471) only 40 V — keep TPS7A16 |
| U7, U8 | TPS2553DBVR | C55266 | SOT-23-6 | 39,155 | 0.32 | AUTO-RETRY version ✓ (do NOT sub C111738 = latching -1) |
| U9 | AP2112K-3.3TRG1 | C51118 | SOT-25 | 30,780 | 0.16 | Extended; basic-lib fallback AMS1117-3.3 (C6186) has no EN + 5 mA IQ — fine since rail is switched |
| Q2, Q3 | **2× CSD18543Q3A** (common-drain b2b) | C840100 | VSONP-8 3.3×3.3 | 1,040 | 0.65 | **substitution**: SQJQ140E is unobtainable (5 pcs @ $8.74). Same part as buck FETs — one feeder for all 6 FETs/board. VGS ±20 V ✓ vs LM74800's 13 V drive |
| QA1,QB1 / QA2,QB2 | CSD18543Q3A (HS+LS both rails) | C840100 | VSONP-8 | 1,040 | 0.65 | 60 V, 8.1 mΩ @10 V (~8.5 @7.5 V), Qg 11.1 nC. RILIM recomputed → 301/243 Ω. 30 pcs for 5 boards vs 1,040 stock ✓. Budget alt: TDM3436 C380232 (verify ±VGS ≥16 V before use) |
| D1 | SMBJ16A (Littelfuse) | C151254 | SMB | 5,255 | 0.13 | cheap alt YFW C2898407 |
| D2, D3 | USBLC6-2SC6 | C7519 | SOT-23-6 | ok | 0.10 | |
| D4 | KT-0805G green LED | C2297 | 0805 | ok | 0.01 | **Basic lib confirmed** |

## D. Magnetics — VERIFIED vs LCSC/JLC 2026-07-13
| Refs | Part | LCSC C# | Specs | Qty | Notes |
|---|---|---|---|---|---|
| LA1, LB1 | Sunlord MWSA1005S-3R3MT | C17700181 | 3.3 µH, Isat 16 A, DCR 10 mΩ, 11.5×10 mm | 2 | chosen for margin (OCP peak ≈8.6 A) + one feeder both rails; price TBD in JLC BOM tool. Fallback (rail B only): cjiang FXL0630-3R3-M C167219 (9.5 A/22 mΩ, 7×6.6, $0.17) |
| L2, L4 | cjiang FXL0630-1R0-M | C167216 | 1 µH, Isat 15 A, DCR 7.4 mΩ, 7×6.6 | 2 | $0.17; alt MPCA-0630-1R0 C268399 |

## E. Connectors / electromechanical — VERIFIED vs LCSC 2026-07-13
All TH connectors need JLCPCB hand-solder/wave add-on; all lines Extended
(≈$3 loader fee each). JLC-library stock can differ from LCSC retail — recheck
C#s in the JLC BOM tool at order time.

| Refs | Part | LCSC C# | Mount | ~$ q10 | Qty | Notes |
|---|---|---|---|---|---|---|
| J1 | AMASS XT60PW-M | C98732 | TH r/a | 0.39 | 1 | genuine AMASS |
| J12 | AMASS XT30PW-M30.G.Y | C431092 | TH r/a | 0.23 | 1 (DNP) | stock 5.4k |
| J3 | GCT USB4105-GF-A | C3020560 | SMD pins + **TH shell tabs** | 0.55 | 1 | 16P, CC1/CC2 exposed; strain relief ✓ (alt: SHOU HAN C2765186 $0.07 but shell-tab style unverified) |
| J4, J5 | Jing Ext. 901-211A1021D10100 USB-A | C2345 | TH r/a | 0.04 | 2 | classic cheap TH |
| J7 | JST SM08B-GHS-TB (8-pin GH, **side-entry**) | C265111 | SMD | 0.44 | 1 | Pixhawk-standard horizontal; if top-entry wanted: BM08B-GHS-TBT (C# unverified) — decide at P3 vs harness routing |
| J2, J8 | JST BM02B-GHS-TBT (2-pin GH, top) | C161690 | SMD | 0.12 | 2 | genuine JST |
| J11 | JST BM03B-GHS-TBT (3-pin GH, top) | C161691 | SMD | 0.43 | 1 | stock 2.3k |
| J9, J10 | 2.54 4-pin header (Pi USB passthrough) | basic-lib generic | TH | 0.02 | 2 | or captive pigtail at P3 |
| F1 | Keystone 3557-10 ATO holder | C3205403 | TH | 1.70 | 1 | **stock only 383** — alt MPD BF353 C3206285 |
| F1 fuse | Littelfuse 0257015.PXPV ATO 15 A | C142692 | blade | 0.05 | 1 | NOT 0287015 (that's MINI blade) |
| CBL1 | Silkland 0.5 ft 240 W USB-C cable, ASIN B0CQ4SX256 | — (Amazon) | — | ~10 | 1/rover | P0 decision 3 |

## F. Cost estimate @ qty 5 (LCSC point-in-time prices)
Parts/board: ICs ~$9.6 (2×LM5145 4.36, LM74800 1.58, tiny816 0.70, INA226 0.69,
TPS7A16 1.43, 2×TPS2553 0.64, AP2112K 0.16) + 6×FET ~$3.9 + magnetics ~$1.4 +
polymer/shunt/TVS/ESD/LED ~$0.7 + connectors ~$4.1 + commodity passives ~$1.5
≈ **$21/board**. Overheads: ~20 Extended feeders ≈ $60/order (≈$12/board @5),
PCB 2-layer 2 oz ~$2, assembly + hand-solder TH ~$8-10 → **≈ $43-45/board
assembled @ qty 5** (scales to ~$28 @ qty 25). Within the ≤$50 informal budget.

## G. Order-time checklist (P5)
- [ ] Re-verify every C# stock in JLCPCB BOM tool (LCSC ≠ JLC assembly stock)
- [ ] Keystone 3557-10 fuse holder stock (383) — swap to BF353 C3206285 if dry
- [ ] TDM3436 VGS abs-max ≥ ±16 V check ONLY IF substituted for CSD18543Q3A
- [ ] Confirm USB4105-GF-A shell tabs vs footprint (strain-relief requirement)
- [ ] Shunt "-4" Kelvin land pattern vs LCSC 2-pin drawing (datasheet check at P3)
- [ ] Polymer cap: prefer 10 V 220 µF if JLC tool offers one in stock
- [ ] F1 footprint/part reconciliation: BOM says Keystone 3557-10 but the assigned
      footprint is the Littelfuse FLR_178.6165 stdlib placeholder — pick ONE
      (source Littelfuse holder, or author Keystone footprint from its drawing)
- [ ] L2/L4: verify cjiang FXL0630 land vs the assigned Chilisin 0630 footprint

## H. Review additions 2026-07-13 (schematic v4 — REVIEW_FINDINGS.md)
| Refs | Part | Package | Note |
|---|---|---|---|
| C6, C8, C10, C13, C14, C15, C17 | 100 nF 50 V X7R | 0402 | decoupling (basic lib) |
| C5, C9 | 1 µF 16 V X7R | 0603 | LDO in/out (basic lib) |
| C7, C16 | 2.2 µF 16 V X7R | 0603 | MCU rail / FE_EN delay (basic lib) |
| CE1 | 100 µF 25 V hybrid/polymer, 8×10 SMD | CP_Elec_8x10 | VSW bulk+damping — pick C# in JLC tool |
| D5 | B5819W Schottky 1A/40V | SOD-123 | LCSC C8598 class — verify at order |
| D6, D7 | SMBJ5.0A | SMB | 5 V rail TVS — C113996 class, verify |
| F2 | 2 A polyfuse 1812 | Fuse_1812 | e.g. BSMD1812-200-16V — verify |
| R31, R32 | 100 kΩ | 0402 | TPS2553 EN pull-ups |
| R21 | 20.0 kΩ (was 10k) | 0402 | PGOOD_A pull-up to 5V_A |
| R33 | 20.0 kΩ | 0402 | sequencing divider hi |
| R34 | 16.5 kΩ | 0402 | sequencing divider lo |
| RA2 | 348 Ω (was 301) | 0402 | RILIM A worst-case |
| RB2 | 261 Ω (was 243/287) | 0402 | RILIM B worst-case |
| RA4 | 3.65 kΩ (was 3.74k) | 0402 | rail A 5.18 V setpoint |
| — | L2 pi-filter inductor DELETED from rail A (qty for L4 only: 1) | | |
| CA51-53/CB51-53, CA61-64/CB61-64, CA111/112, CB111/112 | same 10µ/47µ/22µ parts as §B, now explicit per-instance refs | | quantities unchanged from §B intent |

## I. USB requirement change 2026-07-13 (3x USB-A + USB-C)
| Refs | Part | Note |
|---|---|---|
| J13 | USB-A TH r/a (same C2345) | third port, general-purpose |
| J14 | 2.54 4-pin header | Pi USB3 passthrough |
| F3 | 3 A-hold polyfuse 1812 | port-3 protection (no soft-switch) |
| D8 | USBLC6-2SC6 (C7519) | port-3 ESD |
| RB2 | now 348 Ω (rail B 5A→6A, wc-min 6.3 A) | CB4 now 18 pF |
| R16/R17 | 15 kΩ (TPS2553 max ≈1.7 A per radio port) | see constraint note |
CONSTRAINTS (flagged, decide before fab): (1) TPS2553 tops out at 1.7 A — true
3 A per-port WITH soft-cycling needs a TPS2595/TPS25945-class eFuse swap
(pinout verify pending); radio ports at 1.7 A exceed Pluto draw (~1 A) but not
the 3 A spec. (2) USB-C: rail delivers 6 A but non-PD USB-C can only ADVERTISE
3 A (10k Rp) — Pi 5 works at 5 A with usb_max_current_enable=1; true 5 A
advertisement requires adding a PD source controller (TPS25730-class) in rev B.
