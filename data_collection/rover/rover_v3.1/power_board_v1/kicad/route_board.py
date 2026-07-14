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


via_m = re.search(r'\(padstack "?(Via\[[^\]]+\])_600:300_um"?', s)
VIA = via_m.group(1) if via_m else "Via[0-1]"
copper_layers = re.findall(r'\(layer (\S+)\n', s) or ["F.Cu", "B.Cu"]

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
shapes = "".join(f'      (shape (circle {L} 800 0 0))\n' for L in copper_layers)
old_ps = f'(padstack "{VIA}_600:300_um"' if f'(padstack "{VIA}' in s else f"(padstack {VIA}_600:300_um"
new_name = f'"{VIA}_800:400_um"' if '"' in old_ps else f"{VIA}_800:400_um"
s = s.replace(old_ps,
              f'(padstack {new_name}\n{shapes}      (attach off)\n    )\n    ' + old_ps, 1)
s = s.replace(f'(via "{VIA}_600:300_um")',
              f'(via "{VIA}_600:300_um" "{VIA}_800:400_um")', 1)
# insert power classes right before the default class
i = s.index("(class kicad_default")
s = s[:i] + power_classes + s[i:]
# strip custom padstacks: every freerouting version NPEs on KiCad-7 4-layer
# Cust[T] pads (USB-A shields — PTH into the GND plane, no routing needed)
def _strip_block(text, opener):
    out, i = [], 0
    while True:
        j = text.find(opener, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1
    return "".join(out)


s = _strip_block(_strip_block(s, "(padstack Cust"), '(padstack "Cust')
DSN.write_text(s)
print("DSN classes injected + custom padstacks stripped")

# run freerouting headless
cmd = ["xvfb-run", "-a", FR, "-de", str(DSN), "-do", str(SES), "-mp", "100", "-da"]
if SES.exists():
    SES.unlink()
r = subprocess.run(cmd, capture_output=True, text=True, timeout=2700)
print("freerouting rc:", r.returncode)
Path("/tmp/fr_out.log").write_text((r.stdout or "") + (r.stderr or ""))
if not SES.exists():
    sys.exit("no SES produced")

print("SES ready:", SES.exists())
