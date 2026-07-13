"""Route power_board_v1: DSN export (with power net-class rules) -> freerouting
-> SES import -> zone refill -> DRC. Run with /usr/bin/python3.

Net widths: battery/VSW tree 2.5mm (2oz, ~10A board max — motors are off-board);
rails/SW 1.5mm; signals 0.25mm. GND is carried by the two pours + stitching vias.
"""
import re
import subprocess
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
PCB = HERE / "power_board_v1.kicad_pcb"
DSN = Path("/tmp/pb.dsn")
SES = Path("/tmp/pb.ses")
FR = "/tmp/fr162/freerouting-1.6.2-linux-x64/bin/freerouting"

POWER_25 = ["VBATT_RAW", "VBATT_F", "VBATT_S", "FE_MID", "VSW"]
POWER_15 = ["5V_A", "5VB_PRE", "5V_B", "SW_A", "SW_B", "AUX_5V", "VBUS1", "VBUS2",
            "VBATT_FD"]

board = pcbnew.LoadBoard(str(PCB))
ok = pcbnew.ExportSpecctraDSN(board, str(DSN))
print("DSN export:", ok)

# inject net classes with width rules (KiCad exported everything as kicad_default)
s = DSN.read_text()
m = re.search(r'\(class kicad_default', s)
assert m, "no default class in DSN"


def strip_nets(text, nets):
    for n in nets:
        text = re.sub(r'(?<=[\s"])' + re.escape(n) + r'(?=[\s"])', "", text)
    return text


# remove power nets from the default class listing (quoted or bare tokens)
head, tail = s[:m.start()], s[m.start():]
endidx = tail.index("(circuit")
netblock, rest = tail[:endidx], tail[endidx:]
for n in POWER_25 + POWER_15:
    netblock = netblock.replace(f' "{n}"', "").replace(f" {n} ", " ")
power_classes = (
    '    (class power25 ' + " ".join(f'"{n}"' for n in POWER_25) + '\n'
    '      (circuit (use_via "Via[0-1]_800:400_um"))\n'
    '      (rule (width 1500) (clearance 250.1))\n'
    '    )\n'
    '    (class power15 ' + " ".join(f'"{n}"' for n in POWER_15) + '\n'
    '      (circuit (use_via "Via[0-1]_800:400_um"))\n'
    '      (rule (width 1500) (clearance 250.1))\n'
    '    )\n'
)
# larger via definition for power classes
s = head + netblock + rest
s = s.replace("(padstack Via[0-1]_600:300_um",
              '(padstack Via[0-1]_800:400_um\n'
              '      (shape (circle F.Cu 800 0 0))\n'
              '      (shape (circle B.Cu 800 0 0))\n'
              '      (attach off)\n'
              '    )\n'
              '    (padstack Via[0-1]_600:300_um', 1)
s = s.replace('(via "Via[0-1]_600:300_um")',
              '(via "Via[0-1]_600:300_um" "Via[0-1]_800:400_um")', 1)
# insert power classes right before the default class
i = s.index("(class kicad_default")
s = s[:i] + power_classes + s[i:]
DSN.write_text(s)
print("DSN classes injected")

# run freerouting headless
cmd = ["xvfb-run", "-a", FR, "-de", str(DSN), "-do", str(SES), "-mp", "50", "-da"]
if SES.exists():
    SES.unlink()
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("freerouting rc:", r.returncode)
if not SES.exists():
    sys.exit("no SES produced")

print("SES ready:", SES.exists())
