"""Generate power_board_v1.kicad_sch (KiCad 8) from the DESIGN.md architecture.

v0 capture style: every symbol pin carries a global net label (no drawn wires), so
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
defsym("HDR6", 5.08, 17.78, [(str(i), f"P{i}", "L", i - 1) for i in range(1, 7)], ref="J")
defsym("USB_A", 7.62, 12.7,
       [("1", "VBUS", "L", 0), ("2", "D-", "L", 1), ("3", "D+", "L", 2), ("4", "GND", "L", 3)], ref="J")
defsym("HDR4", 5.08, 12.7, [(str(i), f"P{i}", "R", i - 1) for i in range(1, 5)], ref="J")
defsym("FUSE", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="F")
defsym("RES", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
defsym("CAP", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="C")
defsym("IND", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="L")
defsym("TVS", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="D")
defsym("SHUNT", 7.62, 5.08, [("1", "1", "L", 0), ("2", "2", "R", 0)], ref="R")
defsym("NFET", 7.62, 10.16, [("1", "G", "L", 1), ("2", "D", "R", 0), ("3", "S", "R", 2)], ref="Q")
defsym("PFET", 7.62, 10.16, [("1", "G", "L", 1), ("2", "S", "R", 0), ("3", "D", "R", 2)], ref="Q")
defsym("LDO_5V", 10.16, 10.16, [("1", "IN", "L", 0), ("2", "GND", "L", 2), ("3", "OUT", "R", 0)], ref="U")
defsym("BUCK_5V1", 15.24, 17.78,
       [("1", "VIN", "L", 0), ("2", "EN", "L", 2), ("3", "GND", "L", 5),
        ("4", "SW", "R", 0), ("5", "FB", "R", 2), ("6", "PGOOD", "R", 4)], ref="U")
defsym("TPS2553", 12.7, 15.24,
       [("1", "IN", "L", 0), ("2", "EN", "L", 2), ("3", "GND", "L", 4),
        ("4", "OUT", "R", 0), ("5", "FAULT", "R", 2), ("6", "ILIM", "R", 4)], ref="U")
defsym("INA226", 12.7, 17.78,
       [("1", "VS", "L", 0), ("2", "GND", "L", 5), ("3", "IN+", "L", 2), ("4", "IN-", "L", 3),
        ("5", "SDA", "R", 0), ("6", "SCL", "R", 1), ("7", "ALERT", "R", 3)], ref="U")
defsym("ATTINY816", 17.78, 25.4,
       [("1", "VCC", "L", 0), ("2", "GND", "L", 8), ("3", "VSENSE", "L", 3), ("4", "UPDI", "L", 5),
        ("5", "EN_MAIN", "R", 0), ("6", "LOW_BATT", "R", 1), ("7", "AUX_CTL", "R", 2),
        ("8", "SDA", "R", 4), ("9", "SCL", "R", 5), ("10", "EN_USB1", "R", 7), ("11", "EN_USB2", "R", 8)],
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


# --- section 1: input & protection (battery -, chassis = GND) ---
text("1. INPUT + PROTECTION", 20, 20)
place("XT60", "J1", "XT60_BATT", 30, 40, {"1": "VBATT_RAW", "2": "GND"})
place("PFET", "Q1", "SQJ457EP rev-pol", 60, 40, {"1": "GND", "2": "VBATT_RAW", "3": "VBATT_P"})
place("FUSE", "F1", "15A ATO", 90, 40, {"1": "VBATT_P", "2": "VBATT_F"})
place("TVS", "D1", "SMBJ33A", 90, 55, {"1": "VBATT_F", "2": "GND"})
place("SHUNT", "R20", "2m 3W shunt", 120, 40, {"1": "VBATT_F", "2": "VBATT_S"})

# --- section 2: high-side switch + soft start ---
text("2. HIGH-SIDE SWITCH (soft-start)", 20, 80)
place("NFET", "Q2", "SQJQ140E", 60, 100, {"1": "GATE_MAIN", "2": "VBATT_S", "3": "SW_MID"})
place("NFET", "Q3", "SQJQ140E", 90, 100, {"1": "GATE_MAIN", "2": "VSW", "3": "SW_MID"})
place("RES", "R1", "100k gate pd", 60, 120, {"1": "GATE_MAIN", "2": "SW_MID"})
place("CAP", "C1", "100n soft-start", 90, 120, {"1": "GATE_MAIN", "2": "SW_MID"})
place("RES", "R2", "1M chg-pump fb", 120, 120, {"1": "GATE_DRV", "2": "GATE_MAIN"})
place("NFET", "Q4", "2N7002 en-drv", 120, 100, {"1": "EN_MAIN", "2": "GATE_DRV", "3": "GND"})
place("SCREW2", "J2", "PANEL_SW", 30, 100, {"1": "SW_PANEL", "2": "GND"})
text("note: gate supply via charge pump / LTC7004-class driver in detail design", 20, 135, 1.4)

# --- section 3: supervisor (MCU) + telemetry ---
text("3. SUPERVISOR + TELEMETRY", 20, 155)
place("LDO_5V", "U6", "HT7550 (VBATT->5V mcu)", 35, 175, {"1": "VBATT_F", "2": "GND", "3": "MCU_5V"})
place("ATTINY816", "U3", "ATtiny816", 75, 185,
      {"1": "MCU_5V", "2": "GND", "3": "VSENSE", "4": "UPDI", "5": "EN_MAIN", "6": "LOW_BATT",
       "7": "AUX_CTL", "8": "SDA", "9": "SCL", "10": "EN_USB1", "11": "EN_USB2"})
place("RES", "R10", "100k vsense-hi", 35, 200, {"1": "VBATT_S", "2": "VSENSE"})
place("RES", "R11", "10k vsense-lo", 35, 212, {"1": "VSENSE", "2": "GND"})
place("INA226", "U5", "INA226", 120, 185,
      {"1": "MCU_5V", "2": "GND", "3": "VBATT_F", "4": "VBATT_S", "5": "SDA", "6": "SCL", "7": "LOW_BATT"})
place("SCREW2", "J8", "AUX_CTL out", 120, 210, {"1": "AUX_CTL", "2": "GND"})

# --- section 4: buck A (Pi rail) ---
text("4. BUCK A  5.1V/6A (Pi 5)", 170, 20)
place("BUCK_5V1", "U1", "LM5146-Q1 stage", 190, 45,
      {"1": "VSW", "2": "VSW", "3": "GND", "4": "SW_A", "5": "FB_A", "6": "PGOOD_A"})
place("IND", "L1", "4u7 shielded", 225, 35, {"1": "SW_A", "2": "5VA_PRE"})
place("CAP", "C10", "2x47u in", 165, 70, {"1": "VSW", "2": "GND"})
place("CAP", "C11", "4x100u out", 225, 70, {"1": "5VA_PRE", "2": "GND"})
place("RES", "R12", "FB hi 51k1", 255, 45, {"1": "5VA_PRE", "2": "FB_A"})
place("RES", "R13", "FB lo 10k", 255, 60, {"1": "FB_A", "2": "GND"})
place("IND", "L2", "pi-filter 1u", 255, 25, {"1": "5VA_PRE", "2": "5V_A"})
place("SCREW2", "J3", "PI5 5.1V (USB-C pigtail)", 285, 35, {"1": "5V_A", "2": "GND"})

# --- section 5: buck B (radios + aux) ---
text("5. BUCK B  5.1V/5A (radios+aux)", 170, 95)
place("BUCK_5V1", "U2", "LM5146-Q1 stage", 190, 120,
      {"1": "VSW", "2": "VSW", "3": "GND", "4": "SW_B", "5": "FB_B", "6": "PGOOD_B"})
place("IND", "L3", "4u7 shielded", 225, 110, {"1": "SW_B", "2": "5VB_PRE"})
place("CAP", "C12", "2x47u in", 165, 145, {"1": "VSW", "2": "GND"})
place("CAP", "C13", "4x100u out", 225, 145, {"1": "5VB_PRE", "2": "GND"})
place("RES", "R14", "FB hi 51k1", 255, 120, {"1": "5VB_PRE", "2": "FB_B"})
place("RES", "R15", "FB lo 10k", 255, 135, {"1": "FB_B", "2": "GND"})
place("IND", "L4", "pi-filter 1u", 255, 100, {"1": "5VB_PRE", "2": "5V_B"})

# --- section 6: USB power-switched ports + passthrough ---
text("6. RADIO USB (power inject + data passthrough)", 170, 165)
place("TPS2553", "U7", "TPS2553 1.7A", 190, 190,
      {"1": "5V_B", "2": "EN_USB1", "3": "GND", "4": "VBUS1", "5": "LOW_BATT", "6": None})
place("USB_A", "J4", "USB_A radio1", 225, 190, {"1": "VBUS1", "2": "D1_N", "3": "D1_P", "4": "GND"})
place("HDR4", "J9", "from Pi USB1", 255, 190, {"1": None, "2": "D1_N", "3": "D1_P", "4": "GND"})
place("TPS2553", "U8", "TPS2553 1.7A", 190, 220,
      {"1": "5V_B", "2": "EN_USB2", "3": "GND", "4": "VBUS2", "5": None, "6": None})
place("USB_A", "J5", "USB_A radio2", 225, 220, {"1": "VBUS2", "2": "D2_N", "3": "D2_P", "4": "GND"})
place("HDR4", "J10", "from Pi USB2", 255, 220, {"1": None, "2": "D2_N", "3": "D2_P", "4": "GND"})
place("SCREW2", "J6", "AUX 5V 2A", 285, 205, {"1": "5V_B", "2": "GND"})

# --- section 7: Pi harness ---
text("7. PI HARNESS", 20, 230)
place("HDR6", "J7", "PI GPIO harness", 40, 250,
      {"1": "SDA", "2": "SCL", "3": "LOW_BATT", "4": "PGOOD_A", "5": "PGOOD_B", "6": "GND"})

# ------------------------------------------------------------------ emit
sch = []
sch.append('(kicad_sch (version 20231120) (generator "spf_generate_schematic") (generator_version "8.0")')
sch.append(f'  (uuid "{ROOT_UUID}")')
sch.append('  (paper "A2")')
sch.append('  (title_block (title "SPF rover power board v1") (date "2026-07-13") (rev "v0")')
sch.append('    (comment 1 "generated: see DESIGN.md; v0 net-label capture, rearrange freely"))')
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
print("wrote power_board_v1.kicad_sch (+.kicad_pro);",
      f"{len(BODY)} items, {len(LABELS)} net labels, parens balanced")
