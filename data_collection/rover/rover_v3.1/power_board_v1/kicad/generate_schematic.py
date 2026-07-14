"""Generate power_board_v1.kicad_sch (KiCad 7 dialect) from the DESIGN.md architecture.

v3 (P3): all symbol pin numbers are PHYSICAL PACKAGE PADS, verified against
manufacturer datasheets (SNVSAI4 LM5145 RGY; DS40001913A ATtiny816 VQFN;
SBOS547 INA226 DGS; SLVS841 TPS2553 DBV; SBVS171F TPS7A16 DGN; AP2112 DS39724
SOT-25; SLPS432 CSD18543Q3A SON; SNOSD95C LM74800 DRR). The netlist can now
drive pcbnew directly once footprints are assigned (FOOTPRINTS.md).

v2 (P1 detail design, see ../P1_DETAIL_DESIGN.md):
  - front-end = LM74800-Q1 + CSD18543Q3A back-to-back pair (SQJQ140E unobtainable)
  - both LM5145 buck stages fully expanded (LM25145 out of stock; eq-identical)
  - MCU rail 3.3 V; INA226 on switched 3V3_SW; full ATtiny816 pin map

Capture style: every symbol pin carries a global net label (no drawn wires), so
the netlist is complete and the layout is trivially editable in eeschema. All
symbols are embedded (no external libraries needed). Run:
    python generate_schematic.py
then open power_board_v1.kicad_pro in KiCad 7+.
"""

import uuid
from pathlib import Path

HERE = Path(__file__).parent
ROOT_UUID = str(uuid.uuid4())
PROJECT = "power_board_v1"


def u():
    return str(uuid.uuid4())


# ------------------------------------------------------------------ symbol library
# pins: (number, name, side 'L'|'R', slot_index_from_top)
def lib_symbol(name, w, h, pins, ref="U"):
    x0, y0 = -w / 2, -h / 2
    out = [f'    (symbol "pwr:{name}" (in_bom yes) (on_board yes)']
    out.append(f'      (property "Reference" "{ref}" (at 0 {h/2+1.27:.2f} 0) (effects (font (size 1.27 1.27))))')
    out.append(f'      (property "Value" "{name}" (at 0 {-h/2-1.27:.2f} 0) (effects (font (size 1.27 1.27))))')
    out.append(f'      (symbol "{name}_0_1"')
    out.append(f'        (rectangle (start {x0:.2f} {y0:.2f}) (end {w/2:.2f} {h/2:.2f})'
               f' (stroke (width 0.254) (type default)) (fill (type background)))')
    out.append("      )")
    out.append(f'      (symbol "{name}_1_1"')
    for num, pname, side, slot in pins:
        y = h / 2 - 2.54 * (slot + 1)
        if side == "L":
            px, ang = x0 - 2.54, 0
        else:
            px, ang = w / 2 + 2.54, 180
        out.append(
            f'        (pin passive line (at {px:.2f} {y:.2f} {ang}) (length 2.54)'
            f' (name "{pname}" (effects (font (size 1.27 1.27))))'
            f' (number "{num}" (effects (font (size 1.27 1.27)))))'
        )
    out.append("      )")
    out.append("    )")
    return "\n".join(out), {num: (side, h / 2 - 2.54 * (slot + 1), w) for num, _, side, slot in pins}


SYMBOLS = {}
PINMAPS = {}


def defsym(name, w, h, pins, ref="U"):
    s, pm = lib_symbol(name, w, h, pins, ref)
    SYMBOLS[name] = s
    PINMAPS[name] = (pm, w, h)


defsym("XT60", 5.08, 7.62, [("1", "+", "R", 0), ("2", "-", "R", 1)], ref="J")
defsym("SCREW2", 5.08, 7.62, [("1", "1", "L", 0), ("2", "2", "L", 1)], ref="J")
defsym("HDR3", 5.08, 10.16, [(str(i), f"P{i}", "L", i - 1) for i in range(1, 4)], ref="J")
defsym("HDR8", 5.08, 22.86, [(str(i), f"P{i}", "L", i - 1) for i in range(1, 9)], ref="J")
defsym("USB_A", 7.62, 15.24,
       [("1", "VBUS", "L", 0), ("2", "D-", "L", 1), ("3", "D+", "L", 2), ("4", "GND", "L", 3),
        ("5", "SHIELD", "L", 4)], ref="J")
defsym("HDR4", 5.08, 12.7, [(str(i), f"P{i}", "R", i - 1) for i in range(1, 5)], ref="J")
# pad names = GCT USB4105 footprint pads (16P power-only receptacle, source role)
defsym("USBC_PWR", 15.24, 33.02,
       [("A4", "VBUS", "L", 0), ("A9", "VBUS", "L", 1), ("B4", "VBUS", "L", 2), ("B9", "VBUS", "L", 3),
        ("A5", "CC1", "L", 5), ("B5", "CC2", "L", 6),
        ("A1", "GND", "L", 8), ("A12", "GND", "L", 9), ("B1", "GND", "L", 10), ("B12", "GND", "L", 11),
        ("A6", "D+", "R", 0), ("A7", "D-", "R", 1), ("B6", "D+", "R", 2), ("B7", "D-", "R", 3),
        ("A8", "SBU1", "R", 5), ("B8", "SBU2", "R", 6), ("S1", "SHIELD", "R", 9)], ref="J")
defsym("FUSE", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="F")
defsym("RES", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
defsym("CAP", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="C")
defsym("IND", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="L")
defsym("TVS", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="D")
defsym("LED", 7.62, 5.08, [("1", "A", "L", 0), ("2", "K", "R", 0)], ref="D")
defsym("SHUNT", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
# USBLC6-2SC6 SOT-23-6 (ST): 1 I/O1, 2 GND, 3 I/O2, 4 I/O2', 5 VBUS, 6 I/O1'
defsym("USBLC6", 10.16, 17.78,
       [("1", "I/O1", "L", 0), ("3", "I/O2", "L", 2), ("2", "GND", "L", 5),
        ("6", "I/O1'", "R", 0), ("4", "I/O2'", "R", 2), ("5", "VBUS", "R", 5)], ref="D")
# CSD18543Q3A SON 3.3x3.3 (SLPS432): 1-3 = S, 4 = G, 5-8 = D (thermal pad = D)
defsym("NFET_SON", 10.16, 17.78,
       [("4", "G", "L", 2), ("2", "S", "L", 4), ("3", "S", "L", 5),
        ("5", "D", "R", 0), ("6", "D", "R", 1), ("7", "D", "R", 2), ("8", "D", "R", 3),
        ("1", "S", "R", 5)], ref="Q")
# TPS7A16 DGN (SBVS171F): 1 OUT, 2 FB/DNC(float!), 3 PG, 4 GND, 5 EN, 6 NC, 7 DELAY, 8 IN, pad 9
defsym("TPS7A16", 12.7, 17.78,
       [("8", "IN", "L", 0), ("5", "EN", "L", 2), ("4", "GND", "L", 4), ("9", "PAD", "L", 5),
        ("1", "OUT", "R", 0), ("3", "PG", "R", 2), ("7", "DELAY", "R", 3),
        ("2", "FB/DNC", "R", 4), ("6", "NC", "R", 5)], ref="U")
# AP2112 SOT-25 (DS39724): 1 VIN, 2 GND, 3 EN, 4 NC, 5 VOUT
defsym("AP2112", 10.16, 12.7,
       [("1", "VIN", "L", 0), ("3", "EN", "L", 1), ("2", "GND", "L", 3),
        ("5", "VOUT", "R", 0), ("4", "NC", "R", 2)], ref="U")
# LM74800-Q1 WSON-12 (SNOSD95C table 6-1) — already physical
defsym("LM74800", 17.78, 33.02,
       [("1", "DGATE", "L", 0), ("2", "A", "L", 2), ("3", "VSNS", "L", 4), ("4", "SW", "L", 6),
        ("5", "OV", "L", 8), ("6", "EN_UVLO", "L", 10), ("7", "GND", "L", 11),
        ("8", "HGATE", "R", 0), ("9", "OUT", "R", 2), ("10", "VS", "R", 4),
        ("11", "CAP", "R", 6), ("12", "C", "R", 8)], ref="U")
# LM5145 VQFN-20 RGY (SNVSAI4 fig 6-1): physical pads; EP = pad 21 -> GND
defsym("LM5145", 17.78, 40.64,
       [("20", "VIN", "L", 0), ("1", "EN", "L", 2), ("2", "RT", "L", 4), ("3", "SS", "L", 6),
        ("11", "ILIM", "L", 8), ("8", "SYNCIN", "L", 10), ("6", "AGND", "L", 12),
        ("12", "PGND", "L", 13), ("21", "EP", "L", 14),
        ("14", "VCC", "R", 0), ("17", "BST", "R", 2), ("18", "HO", "R", 4), ("19", "SW", "R", 6),
        ("13", "LO", "R", 8), ("5", "FB", "R", 10), ("4", "COMP", "R", 12), ("10", "PGOOD", "R", 14),
        ("7", "SYNCOUT", "R", 1), ("9", "NC", "R", 3), ("15", "NC", "R", 5), ("16", "NC", "R", 7)],
       ref="U")
# TPS2557 DRB VSON-8 (SLVS931B): 1 GND, 2-3 IN, 4 EN(hi), 5 ILIM, 6-7 OUT, 8 FAULT, 9 PAD->GND
defsym("TPS2557", 12.7, 22.86,
       [("2", "IN", "L", 0), ("3", "IN", "L", 1), ("4", "EN", "L", 3),
        ("1", "GND", "L", 6), ("9", "PAD", "L", 7),
        ("6", "OUT", "R", 0), ("7", "OUT", "R", 1), ("8", "FAULT", "R", 3),
        ("5", "ILIM", "R", 5)], ref="U")
# TPS2553 DBV (SLVS841): 1 IN, 2 GND, 3 EN, 4 FAULT, 5 ILIM, 6 OUT
defsym("TPS2553", 12.7, 15.24,
       [("1", "IN", "L", 0), ("3", "EN", "L", 2), ("2", "GND", "L", 4),
        ("6", "OUT", "R", 0), ("4", "FAULT", "R", 2), ("5", "ILIM", "R", 4)], ref="U")
# INA226 VSSOP-10 (SBOS547): 1 A1, 2 A0, 3 ALERT, 4 SDA, 5 SCL, 6 VS, 7 GND, 8 VBUS, 9 IN-, 10 IN+
defsym("INA226", 15.24, 27.94,
       [("6", "VS", "L", 0), ("10", "IN+", "L", 2), ("9", "IN-", "L", 3), ("8", "VBUS", "L", 4),
        ("1", "A1", "L", 6), ("2", "A0", "L", 7), ("7", "GND", "L", 9),
        ("4", "SDA", "R", 0), ("5", "SCL", "R", 1), ("3", "ALERT", "R", 3)], ref="U")
# ATtiny816 SOIC-20 (DS40001913A table 5-1, SOIC column). Package changed from
# VQFN-20 0.4mm 2026-07-14: QFN escape saturation made the last 4-5 airlines
# unroutable at any legal geometry; 1.27mm pitch removes the problem. Same
# ports, so firmware is unchanged. Mapping: QFN k -> SOIC k-3 (k=4..20),
# QFN 1/2/3 -> SOIC 18/19/20, QFN EP(21) dropped (no EP on SOIC).
defsym("ATTINY816", 40.64, 53.34,
       [("1", "VDD", "L", 0), ("20", "GND", "L", 18),
        ("16", "PA0/UPDI", "L", 2), ("2", "PA4/VIN_SENSE", "L", 4), ("4", "PA6/V5A_SENSE", "L", 6),
        ("5", "PA7/NTC", "L", 8), ("3", "PA5/FAULT_USB", "L", 10), ("9", "PB2/SW_SENSE", "L", 12),
        ("15", "PC3/SHDN_ACK", "L", 14), ("12", "PC0/PGOOD_A", "L", 16),
        ("7", "PB4/FE_EN", "R", 0), ("6", "PB5/BUCKA_EN", "R", 2), ("14", "PC2/LOW_BATT", "R", 4),
        ("8", "PB3/AUX_CTL", "R", 6), ("10", "PB1/SDA", "R", 8), ("11", "PB0/SCL", "R", 10),
        ("17", "PA1/USB_EN1", "R", 12), ("18", "PA2/USB_EN2", "R", 14), ("19", "PA3/LED", "R", 16),
        ("13", "PC1/PGOOD_B", "R", 18)],
       ref="U")

# ------------------------------------------------------------------ footprints
# default by symbol type; per-ref overrides below (P3, see ../FOOTPRINTS.md + BOM.md)
SYM_FP = {
    "RES": "Resistor_SMD:R_0402_1005Metric",
    "CAP": "Capacitor_SMD:C_0402_1005Metric",
    "TVS": "Diode_SMD:D_SMB",
    "LED": "LED_SMD:LED_0805_2012Metric",
    "SHUNT": "Resistor_SMD:R_2512_6332Metric",
    "FUSE": "Fuse:FuseHolder_Blade_ATO_Littelfuse_FLR_178.6165",
    "XT60": "Connector_AMASS:AMASS_XT60PW-M_1x02_P7.20mm_Horizontal",
    "SCREW2": "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-2_1x02_P5.00mm_Horizontal",
    "HDR3": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
    "HDR4": "Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
    "HDR8": "Connector_JST:JST_GH_SM08B-GHS-TB_1x08-1MP_P1.25mm_Horizontal",
    "USB_A": "Connector_USB:USB_A_CNCTech_1001-011-01101_Horizontal",
    "USBC_PWR": "Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal",
    "NFET_SON": "Package_SON:VSON-8_3.3x3.3mm_P0.65mm_NexFET",
    "TPS7A16": "Package_SO:MSOP-8-1EP_3x3mm_P0.65mm_EP1.68x1.88mm",
    "AP2112": "Package_TO_SOT_SMD:SOT-23-5",
    "LM74800": "power_board_v1:WSON-12_3x3_P0.5_LM74800DRR",
    "LM5145": "power_board_v1:VQFN-20_3.5x4.5_P0.5_LM5145RGY",
    "TPS2553": "Package_TO_SOT_SMD:SOT-23-6",
    "TPS2557": "Package_SON:VSON-8-1EP_3x3mm_P0.65mm_EP1.65x2.4mm",
    "INA226": "Package_SO:VSSOP-10_3x3mm_P0.5mm",
    "ATTINY816": "Package_SO:SOIC-20W_7.5x12.8mm_P1.27mm",
}
REF_FP = {
    # capacitor size exceptions
    "CA2": "Capacitor_SMD:C_0603_1608Metric", "CB2": "Capacitor_SMD:C_0603_1608Metric",
    **{f"C{s}5{i}": "Capacitor_SMD:C_1210_3225Metric" for s in "AB" for i in "123"},
    **{f"C{s}6{i}": "Capacitor_SMD:C_1210_3225Metric" for s in "AB" for i in "1234"},
    **{f"C{s}11{i}": "Capacitor_SMD:C_1206_3216Metric" for s in "AB" for i in "12"},
    "CA7": "Capacitor_SMD:CP_Elec_6.3x5.9", "CB7": "Capacitor_SMD:CP_Elec_6.3x5.9",
    "CE1": "Capacitor_SMD:CP_Elec_8x10",
    "C5": "Capacitor_SMD:C_0603_1608Metric", "C7": "Capacitor_SMD:C_0603_1608Metric",
    "C9": "Capacitor_SMD:C_0603_1608Metric", "C16": "Capacitor_SMD:C_0603_1608Metric",
    "D5": "Diode_SMD:D_SOD-123",
    "F2": "Fuse:Fuse_1812_4532Metric",
    # magnetics
    "LA1": "Inductor_SMD:L_Sunlord_MWSA1005S", "LB1": "Inductor_SMD:L_Sunlord_MWSA1005S",
    "L2": "Inductor_SMD:L_Chilisin_BMRx00060630", "L4": "Inductor_SMD:L_Chilisin_BMRx00060630",
    # USBLC6-2 ESD arrays are SOT-23-6, not SMB
    "D2": "Package_TO_SOT_SMD:SOT-23-6", "D3": "Package_TO_SOT_SMD:SOT-23-6",
    # JST-GH per BOM (symbols are generic 2-pin)
    "J2": "Connector_JST:JST_GH_BM02B-GHS-TBT_1x02-1MP_P1.25mm_Vertical",
    "J8": "Connector_JST:JST_GH_BM02B-GHS-TBT_1x02-1MP_P1.25mm_Vertical",
    "J11": "Connector_JST:JST_GH_BM03B-GHS-TBT_1x03-1MP_P1.25mm_Vertical",
    # inboard DNP pigtail header per BOM §I (was XT30PW-M pre-3-port redesign)
    "J12": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
}

# ------------------------------------------------------------------ instances
BODY = []
LABELS = []


def place(sym, ref, value, x, y, nets):
    """nets: {pin_number: net_name or None}"""
    pm, w, h = PINMAPS[sym]
    # ref above / value below, clear of the body (fixed +-14 collided on small parts)
    ry, vy = y - h / 2 - 1.6, y + h / 2 + 1.6
    fp = REF_FP.get(ref, SYM_FP.get(sym, ""))
    BODY.append(
        f'  (symbol (lib_id "pwr:{sym}") (at {x:.2f} {y:.2f} 0) (unit 1)'
        f' (in_bom yes) (on_board yes) (dnp no) (uuid "{u()}")\n'
        f'    (property "Reference" "{ref}" (at {x:.2f} {ry:.2f} 0) (effects (font (size 1.27 1.27))))\n'
        f'    (property "Value" "{value}" (at {x:.2f} {vy:.2f} 0) (effects (font (size 1.27 1.27))))\n'
        f'    (property "Footprint" "{fp}" (at {x:.2f} {y:.2f} 0) (effects (font (size 1.27 1.27)) hide))\n'
        + "\n".join(f'    (pin "{n}" (uuid "{u()}"))' for n in pm)
        + f'\n    (instances (project "{PROJECT}" (path "/{ROOT_UUID}" (reference "{ref}") (unit 1))))\n  )'
    )
    for pin_num, net in nets.items():
        if net is None:
            continue
        side, py, _ = pm[pin_num]
        if side == "L":
            lx, ang, just = x - _ / 2 - 2.54, 180, "right"
        else:
            lx, ang, just = x + _ / 2 + 2.54, 0, "left"
        LABELS.append(
            f'  (global_label "{net}" (shape passive) (at {lx:.2f} {y - py:.2f} {ang})'
            f' (fields_autoplaced) (effects (font (size 1.27 1.27)) (justify {just})) (uuid "{u()}"))'
        )


def text(s, x, y, size=2.0):
    BODY.append(f'  (text "{s}" (at {x:.2f} {y:.2f} 0)'
                f' (effects (font (size {size} {size}) bold) (justify left)) (uuid "{u()}"))')


def fet(ref, value, x, y, g, d, s):
    """CSD18543Q3A: all 8 physical pins netted (1-3 S, 4 G, 5-8 D)."""
    place("NFET_SON", ref, value, x, y,
          {"4": g, "5": d, "6": d, "7": d, "8": d, "1": s, "2": s, "3": s})


def buck_stage(suffix, uref, x0, y0, rilim, cilim, en_net, vout, rfb2, pgood_net=None):
    """Fully-expanded LM5145 stage (values: P1_DETAIL_DESIGN.md section 2).
    Review fixes: bulk caps are individual physical instances (3x Cin, 4x Cout);
    RILIM sized for worst-case IRDSON 180uA x RDS(on)hot-max 9.9-11 mOhm."""
    S = suffix
    place("LM5145", uref, f"LM5145 rail {S}", x0, y0,
          {"20": "VSW", "1": en_net, "2": f"RT_{S}", "3": f"SS_{S}", "11": f"ILIM_{S}",
           "8": "GND", "6": "GND", "12": "GND", "21": "GND",
           "14": f"VCC_{S}", "17": f"BST_{S}", "18": f"HO_{S}", "19": f"SW_{S}",
           "13": f"LO_{S}", "5": f"FB_{S}", "4": f"COMP_{S}",
           "10": pgood_net or f"PGOOD_{S}",
           "7": None, "9": None, "15": None, "16": None})
    # frequency / soft-start / bias (10 mm pitch keeps ref/value text clear)
    place("RES", f"R{S}1", "16k5 RT (606kHz)", x0 - 45, y0 - 15, {"1": f"RT_{S}", "2": "GND"})
    place("CAP", f"C{S}1", "47n SS (4ms)", x0 - 45, y0 - 5, {"1": f"SS_{S}", "2": "GND"})
    place("CAP", f"C{S}2", "2u2 VCC", x0 - 45, y0 + 5, {"1": f"VCC_{S}", "2": "GND"})
    place("CAP", f"C{S}3", "100n BST", x0 - 45, y0 + 15, {"1": f"BST_{S}", "2": f"SW_{S}"})
    place("RES", f"R{S}2", rilim, x0 - 45, y0 + 25, {"1": f"ILIM_{S}", "2": f"SW_{S}"})
    place("CAP", f"C{S}4", cilim, x0 - 45, y0 + 35, {"1": f"ILIM_{S}", "2": "GND"})
    # power FETs (physical 8-pin SON)
    fet(f"Q{S}1", "CSD18543Q3A HS", x0 + 35, y0 - 20, f"HO_{S}", "VSW", f"SW_{S}")
    fet(f"Q{S}2", "CSD18543Q3A LS", x0 + 35, y0 + 4, f"LO_{S}", f"SW_{S}", "GND")
    # LC — every physical capacitor is its own instance (review BLOCKER fix)
    place("IND", f"L{S}1", "MWSA1005S-3R3 16A", x0 + 35, y0 + 22, {"1": f"SW_{S}", "2": vout})
    for i in range(3):
        place("CAP", f"C{S}5{i+1}", "10u 50V X7R", x0 - 20, y0 + 24 + 6 * i, {"1": "VSW", "2": "GND"})
    for i in range(4):
        place("CAP", f"C{S}6{i+1}", "47u 10V X7R", x0 + 35, y0 + 30 + 6 * i, {"1": vout, "2": "GND"})
    place("CAP", f"C{S}7", "220u poly 25mR", x0 + 35, y0 + 56, {"1": vout, "2": "GND"})
    # feedback + type-III compensation (sense point = the rail the load sees)
    place("RES", f"R{S}3", "20k RFB1", x0 + 65, y0 - 15, {"1": vout, "2": f"FB_{S}"})
    place("RES", f"R{S}4", rfb2, x0 + 65, y0 - 5, {"1": f"FB_{S}", "2": "GND"})
    place("RES", f"R{S}5", "13k RC1", x0 + 65, y0 + 5, {"1": f"COMP_{S}", "2": f"CX_{S}"})
    place("CAP", f"C{S}8", "8n2 CC1", x0 + 65, y0 + 15, {"1": f"CX_{S}", "2": f"FB_{S}"})
    place("CAP", f"C{S}9", "39p CC2", x0 + 65, y0 + 25, {"1": f"COMP_{S}", "2": f"FB_{S}"})
    place("CAP", f"C{S}10", "1n2 CC3", x0 + 65, y0 + 35, {"1": vout, "2": f"CY_{S}"})
    place("RES", f"R{S}6", "4k64 RC2", x0 + 65, y0 + 45, {"1": f"CY_{S}", "2": f"FB_{S}"})


# --- section 1: input & protection (battery -, chassis = GND) ---
text("1. INPUT + PROTECTION (3S: 9-13V)", 20, 20)
place("XT60", "J1", "XT60_BATT", 30, 40, {"1": "VBATT_RAW", "2": "GND"})
place("FUSE", "F1", "15A ATO", 60, 40, {"1": "VBATT_RAW", "2": "VBATT_F"})
place("TVS", "D1", "SMBJ16A", 60, 55, {"1": "VBATT_F", "2": "GND"})
place("SHUNT", "R20", "2m 3W shunt", 90, 40, {"1": "VBATT_F", "2": "VBATT_S"})
place("CAP", "C15", "100n at U4.A", 90, 55, {"1": "VBATT_S", "2": "GND"})
# input bulk goes on VSW (behind the soft-started switch: no hot-plug inrush)
place("CAP", "CE1", "100u hybrid 25V", 120, 40, {"1": "VSW", "2": "GND"})

# --- section 2: front-end — LM74800 ideal diode + load switch (P1 trade study) ---
text("2. FRONT-END: LM74800-Q1 + 2x CSD18543Q3A b2b (rev-pol + on/off + OV; 2.87uA off)", 20, 72)
place("LM74800", "U4", "LM74800-Q1", 60, 105,
      {"1": "DG_FE", "2": "VBATT_S", "3": "VBATT_S", "4": "FE_LAD", "5": "FE_OV",
       "6": "FE_EN", "7": "GND", "8": "HG_FE", "9": "VSW", "10": "FE_MID",
       "11": "FE_CAP", "12": "FE_MID"})
fet("Q2", "CSD18543Q3A diode", 105, 92, "DG_FE", "FE_MID", "VBATT_S")
fet("Q3", "CSD18543Q3A switch", 105, 116, "HG_FE", "FE_MID", "VSW")
place("CAP", "C1", "100n CAP-VS", 105, 132, {"1": "FE_CAP", "2": "FE_MID"})
place("CAP", "C2", "100n VS-GND", 105, 142, {"1": "FE_MID", "2": "GND"})
place("CAP", "C3", "47n HGATE dv/dt (10ms)", 140, 92, {"1": "HG_FE", "2": "VSW"})
# EN/OV ladder from SW pin: EN falls 1.132V @ 10.16V (analog LPD fallback),
# EN rises 1.231V @ 11.05V, OV trips @ 14.9V (charger fault) — MCU overrides via FE_EN
place("RES", "R1", "887k ladder-top", 20, 105, {"1": "FE_LAD", "2": "FE_EN"})
place("RES", "R2", "28k7 ladder-mid", 20, 117, {"1": "FE_EN", "2": "FE_OV"})
place("RES", "R3", "82k5 ladder-bot", 20, 129, {"1": "FE_OV", "2": "GND"})
# boot-glitch fix: delays divider self-enable ~450ms so the MCU takes control
# first (rails no longer pulse ON at pack plug-in / MCU reset)
place("CAP", "C16", "2u2 EN delay", 20, 141, {"1": "FE_EN", "2": "GND"})
place("SCREW2", "J2", "PANEL_SW", 20, 90, {"1": "SW_SENSE", "2": "GND"})
place("CAP", "C17", "100n sw debounce", 33, 90, {"1": "SW_SENSE", "2": "GND"})

# --- section 3: supervisor (MCU, always-on 3V3) + telemetry (switched) ---
text("3. SUPERVISOR (always-on ~25uA) + TELEMETRY (switched)", 20, 158)
# D5 protects the always-on LDO from reverse-battery (TPS7A16 IN abs max -0.3V;
# the SMBJ16A conducts ~-1V until the fuse clears)
place("TVS", "D5", "B5819W schottky", 30, 166, {"1": "VBATT_F", "2": "VBATT_FD"})
place("TPS7A16", "U6", "TPS7A1633 3V3 60V LDO", 30, 178,
      {"8": "VBATT_FD", "5": "VBATT_FD", "4": "GND", "9": "GND", "1": "MCU_3V3",
       "3": None, "7": None, "2": None, "6": None})
place("CAP", "C5", "1u LDO in", 48, 168, {"1": "VBATT_FD", "2": "GND"})
place("CAP", "C6", "100n LDO in", 48, 174, {"1": "VBATT_FD", "2": "GND"})
place("CAP", "C7", "2u2 LDO out", 48, 180, {"1": "MCU_3V3", "2": "GND"})
place("CAP", "C8", "100n MCU vdd", 48, 186, {"1": "MCU_3V3", "2": "GND"})
place("ATTINY816", "U3", "ATtiny816 (SOIC-20)", 88, 195,
      {"1": "MCU_3V3", "20": "GND", "16": "UPDI", "2": "VSENSE", "4": "V5A_SENSE",
       "5": "NTC", "3": "FAULT_USB", "9": "SW_SENSE", "15": "SHDN_ACK", "12": "PGOOD_A",
       "7": "FE_EN", "6": "EN_A", "14": "LOW_BATT", "8": "AUX_CTL", "10": "SDA",
       "11": "SCL", "17": "EN_USB1", "18": "EN_USB2", "19": "LED_STAT", "13": "PGOOD_B"})
# VIN sense: high-Z divider (16uA) + hold cap for the accumulating ADC
place("RES", "R10", "680k vsense-hi", 30, 195, {"1": "VBATT_S", "2": "VSENSE"})
place("RES", "R11", "100k vsense-lo", 30, 205, {"1": "VSENSE", "2": "GND"})
place("CAP", "C4", "100n vsense hold", 30, 215, {"1": "VSENSE", "2": "GND"})
place("RES", "R25", "30k v5a-hi", 30, 227, {"1": "5V_A", "2": "V5A_SENSE"})
place("RES", "R26", "10k v5a-lo", 30, 237, {"1": "V5A_SENSE", "2": "GND"})
place("RES", "R27", "10k NTC-top (3V3_SW)", 135, 162, {"1": "3V3_SW", "2": "NTC"})
place("RES", "R28", "10k NTC 3380K", 135, 173, {"1": "NTC", "2": "GND"})
place("RES", "R29", "1k LED", 135, 184, {"1": "LED_STAT", "2": "LED_K"})
place("LED", "D4", "green status", 135, 195, {"1": "LED_K", "2": "GND"})
place("RES", "R30", "10k FAULT pu", 135, 206, {"1": "MCU_3V3", "2": "FAULT_USB"})
# telemetry on the SWITCHED side (INA226 IQ ~330uA would dominate cut-state drain)
place("AP2112", "U9", "AP2112K-3.3 (switched)", 135, 222,
      {"1": "5V_B", "3": "5V_B", "2": "GND", "5": "3V3_SW", "4": None})
place("CAP", "C9", "1u LDO out", 155, 218, {"1": "3V3_SW", "2": "GND"})
place("CAP", "C10", "100n INA vs", 155, 224, {"1": "3V3_SW", "2": "GND"})
place("INA226", "U5", "INA226 (addr 0x40)", 178, 190,
      {"6": "3V3_SW", "10": "VBATT_F", "9": "VBATT_S", "8": "VBATT_S",
       "1": "GND", "2": "GND", "7": "GND", "4": "SDA", "5": "SCL", "3": None})
place("SCREW2", "J8", "AUX_CTL out (JST-GH)", 178, 215, {"1": "AUX_CTL", "2": "GND"})

# --- section 4: buck A (Pi rail, 6A) ---
# Review fixes: pi-filter DELETED on rail A (sense point was upstream of it and
# the Pi 4.75V budget didn't close); setpoint raised to 5.18V (RFB2 3k65) to
# cover cable drop; RILIM 348R = worst-case-min 6.3A valley
text("4. BUCK A  5.18V/6A (Pi 5)  [values: P1_DETAIL_DESIGN.md sec 2]", 230, 20)
place("RES", "R4", "100k EN-A hi (8.5V on)", 200, 30, {"1": "VSW", "2": "EN_A"})
place("RES", "R5", "16k5 EN-A lo (7.5V off)", 200, 42, {"1": "EN_A", "2": "GND"})
buck_stage("A", "U1", 280, 50, "348R RILIM (wc-min 6.3A)", "18p CILIM", "EN_A", "5V_A",
           "3k65 RFB2 (5.18V)", pgood_net="PGOODA_RAW")
# sequencing per LM5145 fig 8-4: PGOOD_A pulls to the MASTER output (5V_A) — a
# 3V3_SW pull-up would deadlock (3V3_SW derives from buck B). The divider gives
# EN_B ~2.3V (>1.231V) and a 3.3V-safe logic level for the MCU/Pi (net PGOOD_A).
place("RES", "R21", "20k PGOOD_A pu (5V_A)", 200, 54, {"1": "5V_A", "2": "PGOODA_RAW"})
place("RES", "R33", "20k seq div hi", 200, 66, {"1": "PGOODA_RAW", "2": "PGOOD_A"})
place("RES", "R34", "16k5 seq div lo", 200, 78, {"1": "PGOOD_A", "2": "GND"})
place("CAP", "CA111", "22u local", 385, 25, {"1": "5V_A", "2": "GND"})
place("CAP", "CA112", "22u local", 385, 32, {"1": "5V_A", "2": "GND"})
place("TVS", "D6", "SMBJ5.0A", 385, 39, {"1": "5V_A", "2": "GND"})
place("USBC_PWR", "J3", "USB4105-GF-A (TH shell)", 425, 42,
      {"A4": "5V_A", "A9": "5V_A", "B4": "5V_A", "B9": "5V_A",
       "A5": "CC1_A", "B5": "CC2_A",
       "A1": "GND", "A12": "GND", "B1": "GND", "B12": "GND", "S1": "GND",
       "A6": None, "A7": None, "B6": None, "B7": None, "A8": None, "B8": None})
place("RES", "R23", "10k Rp (3A adv)", 420, 70, {"1": "5V_A", "2": "CC1_A"})
place("RES", "R24", "10k Rp (3A adv)", 420, 81, {"1": "5V_A", "2": "CC2_A"})
place("SCREW2", "J12", "5V_A pigtail hdr DNP", 420, 95, {"1": "5V_A", "2": "GND"})

# --- section 5: buck B (radios + aux, 5A; EN sequenced from PGOOD_A) ---
# pi-filter KEPT on rail B (SDR ripple >> 37mV DCR drop); sense stays at 5VB_PRE
text("5. BUCK B  5.08V/6A (3xUSB-A+aux) — EN = PGOOD_A (Pi rail first)", 230, 115)
buck_stage("B", "U2", 280, 150, "348R RILIM (wc-min 6.3A)", "18p CILIM", "PGOOD_A", "5VB_PRE",
           "3k74 RFB2 (5.08V)")
place("IND", "L4", "pi-filter 1u", 385, 125, {"1": "5VB_PRE", "2": "5V_B"})
place("CAP", "CB111", "22u pi-out", 385, 132, {"1": "5V_B", "2": "GND"})
place("CAP", "CB112", "22u pi-out", 385, 139, {"1": "5V_B", "2": "GND"})
place("TVS", "D7", "SMBJ5.0A", 385, 146, {"1": "5V_B", "2": "GND"})

# --- section 6: USB power-switched ports + passthrough ---
text("6. RADIO USB (power inject + data passthrough)", 230, 218)
place("TPS2557", "U7", "TPS2557 ~3A", 260, 242,
      {"2": "5V_B", "3": "5V_B", "4": "EN_USB1", "1": "GND", "9": "GND",
       "6": "VBUS1", "7": "VBUS1", "8": "FAULT_USB", "5": "ILIM1"})
place("USB_A", "J4", "USB_A radio1", 295, 242,
      {"1": "VBUS1", "2": "D1_N", "3": "D1_P", "4": "GND", "5": "GND"})
place("HDR4", "J9", "from Pi USB1", 325, 242, {"1": None, "2": "D1_N", "3": "D1_P", "4": "GND"})
place("TPS2557", "U8", "TPS2557 ~3A", 260, 272,
      {"2": "5V_B", "3": "5V_B", "4": "EN_USB2", "1": "GND", "9": "GND",
       "6": "VBUS2", "7": "VBUS2", "8": "FAULT_USB", "5": "ILIM2"})
place("USB_A", "J5", "USB_A radio2", 295, 272,
      {"1": "VBUS2", "2": "D2_N", "3": "D2_P", "4": "GND", "5": "GND"})
place("HDR4", "J10", "from Pi USB2", 325, 272, {"1": None, "2": "D2_N", "3": "D2_P", "4": "GND"})
# aux output gets its own protection (a short must not drop both radios)
place("FUSE", "F2", "2A polyfuse", 345, 250, {"1": "5V_B", "2": "AUX_5V"})
place("SCREW2", "J6", "AUX 5V 2A", 355, 257, {"1": "AUX_5V", "2": "GND"})

place("RES", "R16", "20k RILIM1 (~3A)", 230, 242, {"1": "ILIM1", "2": "GND"})
place("RES", "R17", "20k RILIM2 (~3A)", 230, 272, {"1": "ILIM2", "2": "GND"})
# EN defaults ON without the MCU (fallback variant); local IN caps per SLVS841F
place("RES", "R31", "100k EN1 pu (5V_B)", 230, 252, {"1": "5V_B", "2": "EN_USB1"})
place("RES", "R32", "100k EN2 pu (5V_B)", 230, 282, {"1": "5V_B", "2": "EN_USB2"})
place("CAP", "C13", "100n U7 in", 245, 252, {"1": "5V_B", "2": "GND"})
place("CAP", "C14", "100n U8 in", 245, 282, {"1": "5V_B", "2": "GND"})
place("USBLC6", "D2", "USBLC6-2SC6", 310, 258,
      {"1": "D1_N", "6": "D1_N", "3": "D1_P", "4": "D1_P", "5": "VBUS1", "2": "GND"})
place("USBLC6", "D3", "USBLC6-2SC6", 310, 288,
      {"1": "D2_N", "6": "D2_N", "3": "D2_P", "4": "D2_P", "5": "VBUS2", "2": "GND"})

# third USB-A port (general purpose, 3A): polyfuse + ESD, no soft-switch
place("TPS2557", "U10", "TPS2557 ~3A (always on)", 345, 300,
      {"2": "5V_B", "3": "5V_B", "4": "EN_USB3", "1": "GND", "9": "GND",
       "6": "VBUS3", "7": "VBUS3", "8": "FAULT_USB", "5": "ILIM3"})
place("RES", "R35", "100k EN3 pu", 345, 316, {"1": "5V_B", "2": "EN_USB3"})
place("RES", "R36", "20k RILIM3 (~3A)", 345, 322, {"1": "ILIM3", "2": "GND"})
place("USB_A", "J13", "USB_A port3 (3A)", 295, 302,
      {"1": "VBUS3", "2": "D3_N", "3": "D3_P", "4": "GND", "5": "GND"})
place("HDR4", "J14", "from Pi USB3", 325, 302, {"1": None, "2": "D3_N", "3": "D3_P", "4": "GND"})
place("USBLC6", "D8", "USBLC6-2SC6", 310, 318,
      {"1": "D3_N", "6": "D3_N", "3": "D3_P", "4": "D3_P", "5": "VBUS3", "2": "GND"})

# --- section 7: Pi harness + programming ---
text("7. PI HARNESS (JST-GH) + UPDI", 20, 255)
place("HDR8", "J7", "PI GPIO harness", 40, 280,
      {"1": "SDA", "2": "SCL", "3": "LOW_BATT", "4": "SHDN_ACK",
       "5": "PGOOD_A", "6": "PGOOD_B", "7": "GND", "8": "GND"})
# pull-ups on the SWITCHED rail (review BLOCKER: INA226 V_SCL abs-max = VS+0.3V,
# and always-on pulls would back-power the dead Pi; also makes PGOOD read LOW
# when the bucks are unpowered, so PRECHARGE fault detection actually works)
place("RES", "R18", "4k7 I2C pu (3V3_SW)", 75, 270, {"1": "3V3_SW", "2": "SDA"})
place("RES", "R19", "4k7 I2C pu (3V3_SW)", 75, 282, {"1": "3V3_SW", "2": "SCL"})
place("RES", "R22", "10k PGOOD_B pu (3V3_SW)", 110, 282, {"1": "3V3_SW", "2": "PGOOD_B"})
place("HDR3", "J11", "UPDI prog (UPDI/3V3/GND)", 40, 300,
      {"1": "UPDI", "2": "MCU_3V3", "3": "GND"})

# ------------------------------------------------------------------ emit
sch = []
sch.append('(kicad_sch (version 20230121) (generator spf_generate_schematic)')
sch.append(f'  (uuid "{ROOT_UUID}")')
sch.append('  (paper "A2")')
sch.append('  (title_block (title "SPF rover power board v1") (date "2026-07-13") (rev "v4")')
sch.append('    (comment 1 "v4: adversarial-review fixes (decoupling, split bulk caps, 3V3_SW pulls, PGOOD seq divider, rail-A sense point, RILIM wc, aux polyfuse, 5V TVS)"))')
sch.append("  (lib_symbols")
sch.extend(SYMBOLS.values())
sch.append("  )")
sch.extend(LABELS)
sch.extend(BODY)
sch.append(f'  (sheet_instances (path "/" (page "1")))')
sch.append(")")
content = "\n".join(sch)
assert content.count("(") == content.count(")"), (
    content.count("("), content.count(")"))
(HERE / "power_board_v1.kicad_sch").write_text(content)

(HERE / "power_board_v1.kicad_pro").write_text(
    '{\n  "board": { "design_settings": {} },\n'
    '  "meta": { "filename": "power_board_v1.kicad_pro", "version": 1 },\n'
    '  "schematic": { "legacy_lib_dir": "", "legacy_lib_list": [] }\n}\n'
)
print("wrote power_board_v1.kicad_sch (+.kicad_pro) v3;",
      f"{len(BODY)} items, {len(LABELS)} net labels, parens balanced")
