"""Generate power_board_v1.kicad_sch (KiCad 8) from the DESIGN.md architecture.

v2 (P1 detail design, see ../P1_DETAIL_DESIGN.md):
  - front-end = LM74800-Q1 ideal-diode + load-switch controller driving the
    CSD18543Q3A back-to-back pair (SQJQ140E unobtainable at LCSC) (replaces the v0 reverse-PFET + discrete gate
    machinery; closes review finding F5)
  - both LM25145 buck stages fully expanded: RT/SS/ILIM/comp/FETs from the
    datasheet math (no black boxes) — values marked [QS] pending Quickstart check
  - MCU rail corrected to 3.3 V (Pi I2C is not 5 V tolerant); INA226 moved to the
    switched 3V3_SW rail (cut-state battery drain ~25 uA total)
  - full ATtiny816 pin map (18 signals) per P1_DETAIL_DESIGN.md section 3

Capture style: every symbol pin carries a global net label (no drawn wires), so
the netlist is complete and the layout is trivially editable in eeschema. All
symbols are embedded (no external libraries needed). Run:
    python generate_schematic.py
then open power_board_v1.kicad_pro in KiCad 8+.
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
    out = [f'    (symbol "pwr:{name}" (exclude_from_sim no) (in_bom yes) (on_board yes)']
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
    PINMAPS[name] = (pm, w)


defsym("XT60", 5.08, 7.62, [("1", "+", "R", 0), ("2", "-", "R", 1)], ref="J")
defsym("SCREW2", 5.08, 7.62, [("1", "1", "L", 0), ("2", "2", "L", 1)], ref="J")
defsym("HDR3", 5.08, 10.16, [(str(i), f"P{i}", "L", i - 1) for i in range(1, 4)], ref="J")
defsym("HDR8", 5.08, 22.86, [(str(i), f"P{i}", "L", i - 1) for i in range(1, 9)], ref="J")
defsym("USB_A", 7.62, 12.7,
       [("1", "VBUS", "L", 0), ("2", "D-", "L", 1), ("3", "D+", "L", 2), ("4", "GND", "L", 3)], ref="J")
defsym("HDR4", 5.08, 12.7, [(str(i), f"P{i}", "R", i - 1) for i in range(1, 5)], ref="J")
defsym("USBC_PWR", 7.62, 12.7,
       [("1", "VBUS", "L", 0), ("2", "CC1", "L", 1), ("3", "CC2", "L", 2), ("4", "GND", "L", 3)], ref="J")
defsym("FUSE", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="F")
defsym("RES", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
defsym("CAP", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="C")
defsym("IND", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="L")
defsym("TVS", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="D")
defsym("LED", 7.62, 5.08, [("1", "A", "L", 0), ("2", "K", "R", 0)], ref="D")
defsym("SHUNT", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
defsym("NFET", 7.62, 10.16, [("1", "G", "L", 1), ("2", "D", "R", 0), ("3", "S", "R", 2)], ref="Q")
defsym("LDO", 10.16, 10.16, [("1", "IN", "L", 0), ("2", "GND", "L", 2), ("3", "OUT", "R", 0)], ref="U")
# LM74800-Q1 WSON-12 (SNOSD95C table 6-1)
defsym("LM74800", 17.78, 33.02,
       [("1", "DGATE", "L", 0), ("2", "A", "L", 2), ("3", "VSNS", "L", 4), ("4", "SW", "L", 6),
        ("5", "OV", "L", 8), ("6", "EN_UVLO", "L", 10), ("7", "GND", "L", 11),
        ("8", "HGATE", "R", 0), ("9", "OUT", "R", 2), ("10", "VS", "R", 4),
        ("11", "CAP", "R", 6), ("12", "C", "R", 8)], ref="U")
# LM25145 (logical pinout; VQFN-20 physical mapping at P3 footprint stage)
defsym("LM5145", 17.78, 40.64,
       [("1", "VIN", "L", 0), ("2", "EN", "L", 2), ("3", "RT", "L", 4), ("4", "SS", "L", 6),
        ("5", "ILIM", "L", 8), ("6", "SYNCIN", "L", 10), ("7", "AGND", "L", 12), ("8", "PGND", "L", 14),
        ("9", "VCC", "R", 0), ("10", "BST", "R", 2), ("11", "HO", "R", 4), ("12", "SW", "R", 6),
        ("13", "LO", "R", 8), ("14", "FB", "R", 10), ("15", "COMP", "R", 12), ("16", "PGOOD", "R", 14)],
       ref="U")
defsym("TPS2553", 12.7, 15.24,
       [("1", "IN", "L", 0), ("2", "EN", "L", 2), ("3", "GND", "L", 4),
        ("4", "OUT", "R", 0), ("5", "FAULT", "R", 2), ("6", "ILIM", "R", 4)], ref="U")
defsym("INA226", 12.7, 17.78,
       [("1", "VS", "L", 0), ("2", "GND", "L", 5), ("3", "IN+", "L", 2), ("4", "IN-", "L", 3),
        ("5", "SDA", "R", 0), ("6", "SCL", "R", 1), ("7", "ALERT", "R", 3)], ref="U")
# ATtiny816 SOIC/VQFN-20: 18 signals + VCC/GND (P1_DETAIL_DESIGN.md section 3)
defsym("ATTINY816", 20.32, 50.8,
       [("1", "VCC", "L", 0), ("2", "GND", "L", 18),
        ("3", "PA0/UPDI", "L", 2), ("4", "PA4/VIN_SENSE", "L", 4), ("5", "PA6/V5A_SENSE", "L", 6),
        ("6", "PA7/NTC", "L", 8), ("7", "PA5/FAULT_USB", "L", 10), ("8", "PB2/SW_SENSE", "L", 12),
        ("9", "PC3/SHDN_ACK", "L", 14), ("10", "PC0/PGOOD_A", "L", 16),
        ("11", "PB4/FE_EN", "R", 0), ("12", "PB5/BUCKA_EN", "R", 2), ("13", "PC2/LOW_BATT", "R", 4),
        ("14", "PB3/AUX_CTL", "R", 6), ("15", "PB1/SDA", "R", 8), ("16", "PB0/SCL", "R", 10),
        ("17", "PA1/USB_EN1", "R", 12), ("18", "PA2/USB_EN2", "R", 14), ("19", "PA3/LED", "R", 16),
        ("20", "PC1/PGOOD_B", "R", 18)],
       ref="U")

# ------------------------------------------------------------------ instances
BODY = []
LABELS = []


def place(sym, ref, value, x, y, nets):
    """nets: {pin_number: net_name or None}"""
    pm, w = PINMAPS[sym]
    BODY.append(
        f'  (symbol (lib_id "pwr:{sym}") (at {x:.2f} {y:.2f} 0) (unit 1)'
        f' (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no) (uuid "{u()}")\n'
        f'    (property "Reference" "{ref}" (at {x:.2f} {y-14:.2f} 0) (effects (font (size 1.27 1.27))))\n'
        f'    (property "Value" "{value}" (at {x:.2f} {y+14:.2f} 0) (effects (font (size 1.27 1.27))))\n'
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
            f' (fields_autoplaced yes) (effects (font (size 1.27 1.27)) (justify {just})) (uuid "{u()}"))'
        )


def text(s, x, y, size=2.0):
    BODY.append(f'  (text "{s}" (exclude_from_sim no) (at {x:.2f} {y:.2f} 0)'
                f' (effects (font (size {size} {size}) bold) (justify left)) (uuid "{u()}"))')


def buck_stage(suffix, uref, x0, y0, rilim, cilim, en_net, vout_pre):
    """Fully-expanded LM25145 stage (values: P1_DETAIL_DESIGN.md section 2)."""
    S = suffix
    place("LM5145", uref, f"LM5145 5.1V rail {S}", x0, y0,
          {"1": "VSW", "2": en_net, "3": f"RT_{S}", "4": f"SS_{S}", "5": f"ILIM_{S}",
           "6": "GND", "7": "GND", "8": "GND", "9": f"VCC_{S}", "10": f"BST_{S}",
           "11": f"HO_{S}", "12": f"SW_{S}", "13": f"LO_{S}", "14": f"FB_{S}",
           "15": f"COMP_{S}", "16": f"PGOOD_{S}"})
    # frequency / soft-start / bias
    place("RES", f"R{S}1", "16k5 RT (606kHz)", x0 - 45, y0 - 15, {"1": f"RT_{S}", "2": "GND"})
    place("CAP", f"C{S}1", "47n SS (4ms)", x0 - 45, y0 - 5, {"1": f"SS_{S}", "2": "GND"})
    place("CAP", f"C{S}2", "2u2 VCC", x0 - 45, y0 + 5, {"1": f"VCC_{S}", "2": "GND"})
    place("CAP", f"C{S}3", "100n BST", x0 - 45, y0 + 15, {"1": f"BST_{S}", "2": f"SW_{S}"})
    # OCP (RDS-on valley sensing) [QS]
    place("RES", f"R{S}2", rilim, x0 - 45, y0 + 25, {"1": f"ILIM_{S}", "2": f"SW_{S}"})
    place("CAP", f"C{S}4", cilim, x0 - 45, y0 + 35, {"1": f"ILIM_{S}", "2": "GND"})
    # power FETs (40V NexFET class, MPN at P2)
    place("NFET", f"Q{S}1", "CSD18543Q3A HS", x0 + 35, y0 - 15,
          {"1": f"HO_{S}", "2": "VSW", "3": f"SW_{S}"})
    place("NFET", f"Q{S}2", "CSD18543Q3A LS", x0 + 35, y0, {"1": f"LO_{S}", "2": f"SW_{S}", "3": "GND"})
    # LC
    place("IND", f"L{S}1", "MWSA1005S-3R3 16A", x0 + 35, y0 + 12, {"1": f"SW_{S}", "2": vout_pre})
    place("CAP", f"C{S}5", "3x10u 50V in", x0 - 20, y0 + 28, {"1": "VSW", "2": "GND"})
    place("CAP", f"C{S}6", "4x47u 10V out", x0 + 35, y0 + 24, {"1": vout_pre, "2": "GND"})
    place("CAP", f"C{S}7", "220u poly 25mR", x0 + 35, y0 + 34, {"1": vout_pre, "2": "GND"})
    # feedback + type-III compensation [QS]
    place("RES", f"R{S}3", "20k RFB1", x0 + 65, y0 - 15, {"1": vout_pre, "2": f"FB_{S}"})
    place("RES", f"R{S}4", "3k74 RFB2", x0 + 65, y0 - 5, {"1": f"FB_{S}", "2": "GND"})
    place("RES", f"R{S}5", "13k RC1", x0 + 65, y0 + 5, {"1": f"COMP_{S}", "2": f"CX_{S}"})
    place("CAP", f"C{S}8", "8n2 CC1", x0 + 65, y0 + 15, {"1": f"CX_{S}", "2": f"FB_{S}"})
    place("CAP", f"C{S}9", "39p CC2", x0 + 65, y0 + 25, {"1": f"COMP_{S}", "2": f"FB_{S}"})
    place("CAP", f"C{S}10", "1n2 CC3", x0 + 65, y0 + 35, {"1": vout_pre, "2": f"CY_{S}"})
    place("RES", f"R{S}6", "4k64 RC2", x0 + 65, y0 + 45, {"1": f"CY_{S}", "2": f"FB_{S}"})


# --- section 1: input & protection (battery -, chassis = GND) ---
text("1. INPUT + PROTECTION (3S: 9-13V)", 20, 20)
place("XT60", "J1", "XT60_BATT", 30, 40, {"1": "VBATT_RAW", "2": "GND"})
place("FUSE", "F1", "15A ATO", 60, 40, {"1": "VBATT_RAW", "2": "VBATT_F"})
place("TVS", "D1", "SMBJ16A", 60, 55, {"1": "VBATT_F", "2": "GND"})
place("SHUNT", "R20", "2m 3W shunt", 90, 40, {"1": "VBATT_F", "2": "VBATT_S"})

# --- section 2: front-end — LM74800 ideal diode + load switch (P1 trade study) ---
text("2. FRONT-END: LM74800-Q1 + SQJQ140E b2b (rev-pol + on/off + OV; 2.87uA off)", 20, 72)
place("LM74800", "U4", "LM74800-Q1", 60, 105,
      {"1": "DG_FE", "2": "VBATT_S", "3": "VBATT_S", "4": "FE_LAD", "5": "FE_OV",
       "6": "FE_EN", "7": "GND", "8": "HG_FE", "9": "VSW", "10": "FE_MID",
       "11": "FE_CAP", "12": "FE_MID"})
place("NFET", "Q2", "CSD18543Q3A diode", 105, 90, {"1": "DG_FE", "2": "FE_MID", "3": "VBATT_S"})
place("NFET", "Q3", "CSD18543Q3A switch", 105, 105, {"1": "HG_FE", "2": "FE_MID", "3": "VSW"})
place("CAP", "C1", "100n CAP-VS", 105, 120, {"1": "FE_CAP", "2": "FE_MID"})
place("CAP", "C2", "100n VS-GND", 105, 130, {"1": "FE_MID", "2": "GND"})
place("CAP", "C3", "47n HGATE dv/dt (10ms)", 135, 90, {"1": "HG_FE", "2": "VSW"})
# EN/OV ladder from SW pin: EN falls 1.132V @ 10.16V (analog LPD fallback),
# EN rises 1.231V @ 11.05V, OV trips @ 14.9V (charger fault) — MCU overrides via FE_EN
place("RES", "R1", "887k ladder-top", 20, 105, {"1": "FE_LAD", "2": "FE_EN"})
place("RES", "R2", "28k7 ladder-mid", 20, 117, {"1": "FE_EN", "2": "FE_OV"})
place("RES", "R3", "82k5 ladder-bot", 20, 129, {"1": "FE_OV", "2": "GND"})
place("SCREW2", "J2", "PANEL_SW", 20, 90, {"1": "SW_SENSE", "2": "GND"})

# --- section 3: supervisor (MCU, always-on 3V3) + telemetry (switched) ---
text("3. SUPERVISOR (always-on ~25uA) + TELEMETRY (switched)", 20, 150)
place("LDO", "U6", "TPS7A16 3V3 60V LDO", 30, 170, {"1": "VBATT_F", "2": "GND", "3": "MCU_3V3"})
place("ATTINY816", "U3", "ATtiny816", 80, 190,
      {"1": "MCU_3V3", "2": "GND", "3": "UPDI", "4": "VSENSE", "5": "V5A_SENSE",
       "6": "NTC", "7": "FAULT_USB", "8": "SW_SENSE", "9": "SHDN_ACK", "10": "PGOOD_A",
       "11": "FE_EN", "12": "EN_A", "13": "LOW_BATT", "14": "AUX_CTL", "15": "SDA",
       "16": "SCL", "17": "EN_USB1", "18": "EN_USB2", "19": "LED_STAT", "20": "PGOOD_B"})
# VIN sense: high-Z divider (16uA) + hold cap for the accumulating ADC
place("RES", "R10", "680k vsense-hi", 30, 190, {"1": "VBATT_S", "2": "VSENSE"})
place("RES", "R11", "100k vsense-lo", 30, 200, {"1": "VSENSE", "2": "GND"})
place("CAP", "C4", "100n vsense hold", 30, 210, {"1": "VSENSE", "2": "GND"})
place("RES", "R25", "30k v5a-hi", 30, 222, {"1": "5V_A", "2": "V5A_SENSE"})
place("RES", "R26", "10k v5a-lo", 30, 232, {"1": "V5A_SENSE", "2": "GND"})
place("RES", "R27", "10k NTC-top (3V3_SW)", 120, 150, {"1": "3V3_SW", "2": "NTC"})
place("RES", "R28", "10k NTC 3380K", 120, 160, {"1": "NTC", "2": "GND"})
place("RES", "R29", "1k LED", 120, 172, {"1": "LED_STAT", "2": "LED_K"})
place("LED", "D4", "green status", 120, 182, {"1": "LED_K", "2": "GND"})
place("RES", "R30", "10k FAULT pu", 120, 194, {"1": "MCU_3V3", "2": "FAULT_USB"})
# telemetry on the SWITCHED side (INA226 IQ ~330uA would dominate cut-state drain)
place("LDO", "U9", "AP2112K-3.3 (switched)", 120, 210, {"1": "5V_B", "2": "GND", "3": "3V3_SW"})
place("INA226", "U5", "INA226 (addr 0x40)", 155, 190,
      {"1": "3V3_SW", "2": "GND", "3": "VBATT_F", "4": "VBATT_S", "5": "SDA", "6": "SCL", "7": None})
place("SCREW2", "J8", "AUX_CTL out (JST-GH)", 155, 215, {"1": "AUX_CTL", "2": "GND"})

# --- section 4: buck A (Pi rail, 6A) ---
text("4. BUCK A  5.1V/6A (Pi 5)  [values: P1_DETAIL_DESIGN.md sec 2]", 230, 20)
place("RES", "R4", "100k EN-A hi (8.5V on)", 200, 30, {"1": "VSW", "2": "EN_A"})
place("RES", "R5", "16k5 EN-A lo (7.5V off)", 200, 42, {"1": "EN_A", "2": "GND"})
buck_stage("A", "U1", 280, 45, "301R RILIM (7A valley)", "18p CILIM", "EN_A", "5VA_PRE")
place("IND", "L2", "pi-filter 1u", 380, 25, {"1": "5VA_PRE", "2": "5V_A"})
place("CAP", "CA11", "2x22u pi-out", 380, 37, {"1": "5V_A", "2": "GND"})
place("USBC_PWR", "J3", "USB-C PWR OUT (TH shell)", 415, 35,
      {"1": "5V_A", "2": "CC1_A", "3": "CC2_A", "4": "GND"})
place("RES", "R23", "10k Rp (3A adv)", 415, 55, {"1": "5V_A", "2": "CC1_A"})
place("RES", "R24", "10k Rp (3A adv)", 415, 67, {"1": "5V_A", "2": "CC2_A"})
place("SCREW2", "J12", "XT30 fallback pads", 415, 80, {"1": "5V_A", "2": "GND"})

# --- section 5: buck B (radios + aux, 5A; EN sequenced from PGOOD_A) ---
text("5. BUCK B  5.1V/5A (radios+aux) — EN = PGOOD_A (Pi rail first)", 230, 115)
buck_stage("B", "U2", 280, 145, "243R RILIM (5.7A valley)", "22p CILIM", "PGOOD_A", "5VB_PRE")
place("IND", "L4", "pi-filter 1u", 380, 125, {"1": "5VB_PRE", "2": "5V_B"})
place("CAP", "CB11", "2x22u pi-out", 380, 137, {"1": "5V_B", "2": "GND"})

# --- section 6: USB power-switched ports + passthrough ---
text("6. RADIO USB (power inject + data passthrough)", 230, 215)
place("TPS2553", "U7", "TPS2553 1.7A", 260, 240,
      {"1": "5V_B", "2": "EN_USB1", "3": "GND", "4": "VBUS1", "5": "FAULT_USB", "6": "ILIM1"})
place("USB_A", "J4", "USB_A radio1", 295, 240, {"1": "VBUS1", "2": "D1_N", "3": "D1_P", "4": "GND"})
place("HDR4", "J9", "from Pi USB1", 325, 240, {"1": None, "2": "D1_N", "3": "D1_P", "4": "GND"})
place("TPS2553", "U8", "TPS2553 1.7A", 260, 270,
      {"1": "5V_B", "2": "EN_USB2", "3": "GND", "4": "VBUS2", "5": "FAULT_USB", "6": "ILIM2"})
place("USB_A", "J5", "USB_A radio2", 295, 270, {"1": "VBUS2", "2": "D2_N", "3": "D2_P", "4": "GND"})
place("HDR4", "J10", "from Pi USB2", 325, 270, {"1": None, "2": "D2_N", "3": "D2_P", "4": "GND"})
place("SCREW2", "J6", "AUX 5V 2A", 355, 255, {"1": "5V_B", "2": "GND"})

place("RES", "R16", "20k RILIM1 (1.2A)", 230, 240, {"1": "ILIM1", "2": "GND"})
place("RES", "R17", "20k RILIM2 (1.2A)", 230, 270, {"1": "ILIM2", "2": "GND"})
place("TVS", "D2", "USBLC6-2 esd", 310, 255, {"1": "D1_P", "2": "GND"})
place("TVS", "D3", "USBLC6-2 esd", 310, 285, {"1": "D2_P", "2": "GND"})

# --- section 7: Pi harness + programming ---
text("7. PI HARNESS (JST-GH) + UPDI", 20, 250)
place("HDR8", "J7", "PI GPIO harness", 40, 275,
      {"1": "SDA", "2": "SCL", "3": "LOW_BATT", "4": "SHDN_ACK",
       "5": "PGOOD_A", "6": "PGOOD_B", "7": "GND", "8": "GND"})
place("RES", "R18", "4k7 I2C pu", 75, 265, {"1": "MCU_3V3", "2": "SDA"})
place("RES", "R19", "4k7 I2C pu", 75, 277, {"1": "MCU_3V3", "2": "SCL"})
place("RES", "R21", "10k PGOOD_A pu", 110, 265, {"1": "MCU_3V3", "2": "PGOOD_A"})
place("RES", "R22", "10k PGOOD_B pu", 110, 277, {"1": "MCU_3V3", "2": "PGOOD_B"})
place("HDR3", "J11", "UPDI prog (UPDI/3V3/GND)", 40, 297,
      {"1": "UPDI", "2": "MCU_3V3", "3": "GND"})

# ------------------------------------------------------------------ emit
sch = []
sch.append('(kicad_sch (version 20231120) (generator "spf_generate_schematic") (generator_version "8.0")')
sch.append(f'  (uuid "{ROOT_UUID}")')
sch.append('  (paper "A2")')
sch.append('  (title_block (title "SPF rover power board v1") (date "2026-07-13") (rev "v2")')
sch.append('    (comment 1 "P1 detail design: LM74800 front-end, expanded LM25145 stages; see P1_DETAIL_DESIGN.md"))')
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
print("wrote power_board_v1.kicad_sch (+.kicad_pro) v2;",
      f"{len(BODY)} items, {len(LABELS)} net labels, parens balanced")
