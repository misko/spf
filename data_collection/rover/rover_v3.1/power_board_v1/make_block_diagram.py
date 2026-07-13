"""Block schematic for the rover power board v1 (see DESIGN.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

P, D, C = "#c0392b", "#2b6cb0", "#777"

def box(ax, x, y, w, h, t, lines=(), fc="#f8f8f8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=fc, edgecolor="#333", lw=1.1))
    ax.text(x+w/2, y+h-0.015, t, ha="center", va="top", fontsize=9, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x+w/2, y+h-0.042-i*0.026, ln, ha="center", va="top", fontsize=7.4)
    return (x, y, w, h)

def edge(ax, p0, p1, color=P, label=None, ly=0.015, style="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=1.7, linestyle=style, shrinkA=2, shrinkB=2))
    if label:
        ax.text((p0[0]+p1[0])/2, (p0[1]+p1[1])/2+ly, label, fontsize=7.2, color=color, ha="center")

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.0); ax.axis("off")
ax.set_title("Rover power board v1 — block schematic (red=power, blue=logic/data)", fontsize=12, pad=10)

batt = box(ax, 0.02, 0.62, 0.16, 0.22, "BATT XT60", ["10–32 V", "3S–7S Li-ion"], "#fdeaea")
prot = box(ax, 0.24, 0.62, 0.19, 0.22, "Input protection", ["reverse-pol FET", "15 A fuse", "SMBJ33A TVS"], "#fdeaea")
sw   = box(ax, 0.49, 0.62, 0.23, 0.22, "High-side switch", ["back-to-back NFETs", "soft-start ~10 ms", "(relay alt-footprint)", "panel switch = mA gate"], "#fdeaea")
lpd  = box(ax, 0.49, 0.30, 0.23, 0.24, "LPD + supervisor", ["cut 10.2 / rec 11.7 V", "10 s qualifier", "MCU or comparator", "LOW_BATT 60 s prewarn"], "#fff4e0")
bucka= box(ax, 0.82, 0.72, 0.22, 0.20, "Buck A  5.1 V / 6 A", ["LM5146 class, 2.1 MHz", "π-filter, <30 mVpp"], "#eaf0fa")
buckb= box(ax, 0.82, 0.46, 0.22, 0.20, "Buck B  5.1 V / 5 A", ["π-filter <10 mVpp", "radios + aux"], "#eaf0fa")
pi   = box(ax, 1.14, 0.72, 0.30, 0.20, "Raspberry Pi 5", ["USB-C 5.1 V / 5 A budget", "I2C + GPIO harness"], "#eafaf0")
usb  = box(ax, 1.14, 0.42, 0.30, 0.24, "2× USB-A (Plutos) + aux", ["TPS2553 per port", "EN ← Pi GPIO:", "software radio power-cycle", "aux 2 A screw terminal"], "#eafaf0")
ina  = box(ax, 0.82, 0.18, 0.22, 0.20, "INA226 telemetry", ["pack V / I / P", "I2C → Pi → MAVLink"], "#fff4e0")
aux  = box(ax, 0.24, 0.30, 0.19, 0.20, "AUX_CTL", ["gate for external", "motor-path contactor", "(Cytron 30 A stays off-board)"], "#f2f2f2")

edge(ax, (0.18, 0.73), (0.24, 0.73), label="pack")
edge(ax, (0.43, 0.73), (0.49, 0.73))
edge(ax, (0.72, 0.79), (0.82, 0.815), label="switched V")
edge(ax, (0.72, 0.70), (0.82, 0.56))
edge(ax, (1.04, 0.82), (1.14, 0.82), label="5.1 V")
edge(ax, (1.04, 0.56), (1.14, 0.545), label="5.1 V")
edge(ax, (0.605, 0.54), (0.605, 0.62), D, label="enable", ly=0.0)
edge(ax, (0.43, 0.40), (0.49, 0.40), D, style="--")
edge(ax, (0.72, 0.37), (0.82, 0.30), D, label="V sense", ly=-0.02)
edge(ax, (1.04, 0.26), (1.22, 0.42), D, label="I2C + LOW_BATT + PGOOD", ly=-0.025)
edge(ax, (1.29, 0.66), (1.29, 0.72), D, label="GPIO EN", ly=0.0)

fig.tight_layout()
out = __file__.replace("make_block_diagram.py", "power_board_v1_block.png")
fig.savefig(out, dpi=140); print("wrote", out)
